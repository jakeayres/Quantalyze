"""M12: open Fermi sheets from any band (`generators.open_sheets_from_dispersion`).

The sheets are traced line by line across them, over one period G. These tests check the
nodes (on ε = 0, velocities ∇ε/ħ, independent root-finding), the grouping into sheets, the
k_z slices, rotation to any direction of G, and that bad input fails clearly. The exact
transport of warped sheets is known result K20.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
A, B_, D = 3.6e-10, 7.7e-10, 1.35e-9  # chain spacing, interchain spacing, layer spacing (m)
TA, TB, TC = 0.25 * E, 0.025 * E, 0.002 * E  # along the chains (y), across them (x), between layers
MU = -2 * TA * np.cos(np.pi / 4)  # quarter filling of the chain band
G = 2 * np.pi / B_


def energy(kx, ky):
    """Chains along y: two sheets near k_y = ±π/4A that run along k_x."""
    return -2 * TA * np.cos(ky * A) - 2 * TB * np.cos(kx * B_) - MU


def gradient(kx, ky):
    return 2 * TB * B_ * np.sin(kx * B_), 2 * TA * A * np.sin(ky * A)


def energy3(kx, ky, kz):
    return energy(kx, ky) - 2 * TC * np.cos(kz * D) * np.cos(kx * B_)


def gradient3(kx, ky, kz):
    gx, gy = gradient(kx, ky)
    return (gx + 2 * TC * B_ * np.cos(kz * D) * np.sin(kx * B_), gy,
            2 * TC * D * np.sin(kz * D) * np.cos(kx * B_))


def sheets(n=256, **kwargs):
    kwargs = {"period": (G, 0.0), "across": (-np.pi / A, np.pi / A), "tau": 1e-13, **kwargs}
    return gen.open_sheets_from_dispersion(n, energy=energy, gradient=gradient, **kwargs)


def test_nodes_lie_on_the_fermi_surface_with_velocity_grad_eps():
    """Two sheets, one period each, ordered along n̂ = +k_y; ε = 0 to 1e-13 of t; v matches
    finite differences of ε to 1e-6; the period comes back as given."""
    frames, period = sheets()
    assert len(frames) == 2 and period == (G, 0.0)
    for side, df in zip((-1, 1), frames):
        assert list(df.columns) == ["kx", "ky", "vx", "vy", "tau"] and len(df) == 256
        assert np.all(np.sign(df.ky) == side)
        np.testing.assert_allclose(np.diff(df.kx), G / 256, rtol=1e-12)
        assert df.kx.iloc[0] == pytest.approx(-G / 2, rel=1e-15)
        on_surface = np.max(np.abs(energy(df.kx, df.ky))) / TA
        h = 1e-7 / A
        fd = ((energy(df.kx + h, df.ky) - energy(df.kx - h, df.ky)) / (2 * h * HBAR),
              (energy(df.kx, df.ky + h) - energy(df.kx, df.ky - h)) / (2 * h * HBAR))
        error = np.max(np.hypot(df.vx - fd[0], df.vy - fd[1]) / np.hypot(df.vx, df.vy))
        print(f"sheet {side:+d}: max |eps|/t = {on_surface:.1e}; velocity vs finite differences {error:.1e}")
        assert on_surface <= 1e-13 and error <= 1e-6


def test_crossings_match_an_independent_root_finder():
    """Each node is where scipy's brentq, run line by line, puts the crossing (1e-14)."""
    frames, _ = sheets()
    worst = 0.0
    for df in frames:
        for i in range(0, 256, 9):
            kx = df.kx.iloc[i]
            lo, hi = (0.0, np.pi / A) if df.ky.iloc[i] > 0 else (-np.pi / A, 0.0)
            root = brentq(lambda ky: energy(kx, ky), lo, hi, xtol=1e-15 * 2 * np.pi / A,
                          rtol=4 * np.finfo(float).eps)
            worst = max(worst, abs(df.ky.iloc[i] / root - 1))
    print(f"max relative difference from brentq: {worst:.1e}")
    assert worst <= 1e-14


def test_tau_as_a_function_of_k():
    frames, _ = sheets(tau=lambda kx, ky: 1e-13 * (1 + 0.5 * np.cos(kx * B_)))
    for df in frames:
        np.testing.assert_allclose(df.tau, 1e-13 * (1 + 0.5 * np.cos(df.kx * B_)), rtol=1e-15)


def test_kz_slices():
    """With n_kz: evenly spaced slices over 2π/d, v_z = (1/ħ)∂ε/∂k_z, and a band that does not
    depend on k_z gives the 2D conductivity (1e-12)."""
    frames, period = gen.open_sheets_from_dispersion(128, energy=energy3, gradient=gradient3, period=(G, 0.0),
                                                     across=(-np.pi / A, np.pi / A), tau=1e-13, n_kz=6,
                                                     layer_spacing=D)
    df = frames[1]
    assert list(df.columns) == ["kx", "ky", "kz", "vx", "vy", "vz", "tau"] and len(df) == 6 * 128
    np.testing.assert_allclose(np.unique(df.kz), -np.pi / D + 2 * np.pi * np.arange(6) / (6 * D), rtol=1e-14)
    np.testing.assert_allclose(df.vz, gradient3(df.kx, df.ky, df.kz)[2] / HBAR, rtol=1e-14, atol=0)
    assert np.max(np.abs(energy3(df.kx, df.ky, df.kz))) / TA <= 1e-13
    fields = np.array([0.0, 3.0, 30.0, -30.0])
    s3 = bz.conductivity(frames, fields, layer_spacing=D, kz="kz", period=period)
    assert np.all(np.isfinite(s3.iloc[:, 1:].to_numpy()))
    flat, _ = gen.open_sheets_from_dispersion(128, energy=lambda kx, ky, kz: energy(kx, ky),
                                              gradient=lambda kx, ky, kz: (*gradient(kx, ky), 0 * kx),
                                              period=(G, 0.0), across=(-np.pi / A, np.pi / A), tau=1e-13,
                                              n_kz=4, layer_spacing=D)
    two_d = bz.conductivity(sheets(128)[0], fields, layer_spacing=D, period=period)
    three_d = bz.conductivity(flat, fields, layer_spacing=D, kz="kz", period=period)
    plane = three_d[["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]].to_numpy()
    reference = two_d[["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]].to_numpy()
    assert np.max(np.abs(plane - reference)) <= 1e-12 * np.max(np.abs(reference))


def test_sheets_along_any_direction():
    """A triclinic cell has G off the axes. Rotating the band and G together by α rotates
    σ to R σ Rᵀ (1e-12)."""
    alpha = 0.4
    c, s = np.cos(alpha), np.sin(alpha)
    back = lambda kx, ky: (c * kx + s * ky, -s * kx + c * ky)  # noqa: E731  (rotate k by −α)

    def energy_r(kx, ky):
        return energy(*back(kx, ky))

    def gradient_r(kx, ky):
        gx, gy = gradient(*back(kx, ky))
        return c * gx - s * gy, s * gx + c * gy

    fields = np.array([0.0, 3.0, 30.0])
    base, period = sheets(256)
    turned, period_r = gen.open_sheets_from_dispersion(256, energy=energy_r, gradient=gradient_r,
                                                       period=(c * G, s * G), across=(-np.pi / A, np.pi / A),
                                                       tau=1e-13)
    sigma = sum(conductivity_tensor(df.kx, df.ky, df.vx, df.vy, df.tau, fields, layer_spacing=D, period=period)
                for df in base)
    sigma_r = sum(conductivity_tensor(df.kx, df.ky, df.vx, df.vy, df.tau, fields, layer_spacing=D,
                                      period=period_r) for df in turned)
    R = np.array([[c, -s], [s, c]])
    error = np.max(np.abs(sigma_r - R @ sigma @ R.T)) / np.max(np.abs(sigma))
    print(f"rotated band and G: |sigma' - R sigma R^T| / |sigma| = {error:.1e}")
    assert error <= 1e-12


def test_rejects_bad_input_clearly():
    with pytest.raises(ValueError, match="do not cross"):  # no sheet in the range
        sheets(across=(-0.1 * np.pi / A, 0.1 * np.pi / A))
    with pytest.raises(ValueError, match="do not cross"):  # the only sheet in the range is cut by it
        sheets(across=(0.0, 0.25 * np.pi / A))
    with pytest.raises(ValueError, match="changes along the period"):  # one sheet whole, the other cut
        sheets(across=(-0.5 * np.pi / A, 0.25 * np.pi / A))
    with pytest.raises(ValueError, match="period"):
        sheets(period=(0.0, 0.0))
    for wrong in (1.5 * G, 0.5 * G):  # not a period of the band (the band is even in k_x, so ±G/2 would not tell)
        with pytest.raises(ValueError, match="do not repeat after `period`"):
            sheets(period=(wrong, 0.0))
    assert len(sheets(period=(-2 * G, 0.0))[0]) == 2  # two periods at once, either sign: still a period
    with pytest.raises(ValueError, match="across"):
        sheets(across=(1.0, -1.0))
    with pytest.raises(ValueError, match="layer_spacing"):
        sheets(n_kz=4)
    with pytest.raises(ValueError, match="not finite"):
        gen.open_sheets_from_dispersion(64, energy=lambda kx, ky: np.where(ky > 0, np.nan, energy(kx, ky)),
                                        gradient=gradient, period=(G, 0.0), across=(-np.pi / A, np.pi / A),
                                        tau=1e-13)
    pocket = lambda kx, ky: kx**2 + ky**2 - (0.3 * G) ** 2  # noqa: E731  (a closed pocket, not sheets)
    with pytest.raises(ValueError, match="changes along the period|do not cross"):
        gen.open_sheets_from_dispersion(64, energy=pocket, gradient=lambda kx, ky: (2 * kx, 2 * ky),
                                        period=(G, 0.0), across=(-G, G), tau=1e-13)
