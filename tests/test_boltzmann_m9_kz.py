"""M9: k_z-warped Fermi surfaces (B ∥ c).

A warped surface is given as k_z slices; each carries its v_z along the in-plane orbit,
and σ (3×3) is the average over the slices. Checks the generator, the grid validation,
the 3×3 numba kernel against the NumPy one, and convergence to the reference.
"""
import dataclasses
import re

import numba
import numpy as np
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _kernel, _kernel_py
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

M = ELECTRON_MASS
D = 1e-9
TAU = 1e-13
K0, K4, KW = 7.0e9, 0.25e9, 0.3e9  # m⁻¹
C2 = HBAR**2 / (2 * M)


def k_fermi(phi, kz):
    return K0 - K4 * np.cos(4 * phi) + KW * np.cos(kz * D) * np.cos(2 * phi)


def dk_fermi_dphi(phi, kz):
    return 4 * K4 * np.sin(4 * phi) - 2 * KW * np.cos(kz * D) * np.sin(2 * phi)


def energy(kx, ky, kz):
    """ε = ħ²(k² − k_F(φ, k_z)²)/2m: a rounded square whose cos2φ distortion alternates with k_z."""
    return C2 * (kx**2 + ky**2 - k_fermi(np.arctan2(ky, kx), kz) ** 2)


def gradient(kx, ky, kz):
    phi = np.arctan2(ky, kx)
    r2 = kx**2 + ky**2
    f, f_phi = k_fermi(phi, kz), dk_fermi_dphi(phi, kz)
    f_kz = -KW * D * np.sin(kz * D) * np.cos(2 * phi)
    return (2 * C2 * (kx + f * f_phi * ky / r2), 2 * C2 * (ky - f * f_phi * kx / r2), -2 * C2 * f * f_kz)


def warped(n, n_kz=4):
    return bz.generators.from_dispersion_3d(n, n_kz, energy=energy, gradient=gradient, tau=TAU,
                                            max_radius=2 * K0, layer_spacing=D)


def x_per_tesla():
    return ELEMENTARY_CHARGE * TAU / M  # ω_cτ per tesla: ε = ħ²(k² − k_F²)/2m has cyclotron mass m


def tensor(sigma):
    return sigma[[c for c in sigma.columns if c.startswith("sigma_")]].to_numpy().reshape(len(sigma), 3, 3)


def test_generator_velocities_and_nodes():
    df = warped(128)
    kx, ky, kz = (df[c].to_numpy() for c in ("kx", "ky", "kz"))
    h = 1e-6 * K0
    fd = [(energy(kx + h, ky, kz) - energy(kx - h, ky, kz)) / (2 * h * HBAR),
          (energy(kx, ky + h, kz) - energy(kx, ky - h, kz)) / (2 * h * HBAR),
          (energy(kx, ky, kz + h) - energy(kx, ky, kz - h)) / (2 * h * HBAR)]
    v = np.stack([df.vx, df.vy, df.vz])
    error = np.max(np.linalg.norm(v - np.array(fd), axis=0) / np.linalg.norm(v, axis=0))
    on_surface = np.max(np.abs(energy(kx, ky, kz)) / (HBAR * np.linalg.norm(v, axis=0) * K0))
    print(f"max |v - grad eps/hbar| / |v| = {error:.1e}; max |eps| / (hbar |v| k) = {on_surface:.1e}")
    assert error < 1e-6 and on_surface < 1e-12
    np.testing.assert_allclose(np.unique(kz), -np.pi / D + 2 * np.pi * np.arange(4) / (4 * D), rtol=1e-15)


def test_kz_grid_must_cover_one_period_evenly():
    df = warped(64, 4)
    with pytest.raises(ValueError, match="evenly spaced"):
        bz.conductivity(df[df.kz != df.kz.unique()[1]], 1.0, layer_spacing=D, kz="kz")  # a slice missing
    with pytest.raises(ValueError, match="evenly spaced"):
        bz.conductivity(df, 1.0, layer_spacing=2 * D, kz="kz")  # the slices span half the period


def test_output_columns_and_components():
    s = bz.conductivity(warped(128), [0.0, 5.0], layer_spacing=D, kz="kz")
    assert list(s.columns) == ["field"] + [f"sigma_{i}{j}" for i in "xyz" for j in "xyz"]
    rho = bz.resistivity(s)
    product = rho.iloc[:, 1:].to_numpy().reshape(-1, 3, 3) @ tensor(s)
    np.testing.assert_allclose(product, np.broadcast_to(np.eye(3), product.shape), atol=1e-12)
    assert bz.magnetoresistance(s, "zz").iloc[0] == 0.0
    with pytest.raises(ValueError):
        bz.magnetoresistance(bz.conductivity(warped(64).query("kz == kz.min()"), [0.0, 1.0], layer_spacing=D),
                             "zz")


def test_onsager_in_three_dimensions():
    fields = np.array([0.5, 5.0, 50.0])
    s = tensor(bz.conductivity(warped(128), np.concatenate([fields, -fields]), layer_spacing=D, kz="kz"))
    assert np.array_equal(s[3:], np.transpose(s[:3], (0, 2, 1)))


def test_carrier_density_is_the_enclosed_volume():
    """For ε = ħ²k²/2m − 2t_z cos(k_z d) − E_F the slice areas πk_F(k_z)² average to 2πmE_F/ħ²,
    so n = g_s A/(4π²d) = g_s mE_F/(2πħ²d), whatever t_z."""
    ef, tz = 1.9e-19, 2e-21
    df = bz.generators.from_dispersion_3d(
        256, 6, tau=TAU, max_radius=2e10, layer_spacing=D,
        energy=lambda kx, ky, kz: C2 * (kx**2 + ky**2) - 2 * tz * np.cos(kz * D) - ef,
        gradient=lambda kx, ky, kz: (2 * C2 * kx, 2 * C2 * ky, 2 * tz * D * np.sin(kz * D) + 0 * kx))
    n = bz.carrier_density(df, layer_spacing=D, kz="kz")
    expected = 2 * M * ef / (2 * np.pi * HBAR**2 * D)
    print(f"n / expected - 1 = {n / expected - 1:.1e}")
    assert n == pytest.approx(expected, rel=1e-8)


def slice_arrays(df, kz):
    part = df[df.kz == kz]
    c = prepare_contour(part.kx, part.ky, part.vx, part.vy, part.tau, vz=part.vz)
    return c


def test_numba_3x3_kernel_matches_numpy():
    df = warped(256)
    fields = np.geomspace(1e-6, 1e6, 25) / x_per_tesla()
    worst = 0.0
    for kz in df.kz.unique():
        c = slice_arrays(df, kz)
        order = np.roll(np.arange(c.damping.size)[::-1], 1)
        both = _kernel.orbit_sums_both3(c.damping, c.lx, c.ly, c.lz, fields)
        forward = _kernel_py.orbit_sums(c.damping, c.lx, c.ly, fields, lz=c.lz)
        backward = _kernel_py.orbit_sums(c.damping[::-1], c.lx[order], c.ly[order], fields, lz=c.lz[order])
        for actual, expected in ((both[0], forward), (both[1], backward)):
            worst = max(worst, np.max(np.max(np.abs(actual - expected), axis=(1, 2))
                                      / np.max(np.abs(expected), axis=(1, 2))))
        np.testing.assert_array_equal(both[0], _kernel.orbit_sums3(c.damping, c.lx, c.ly, c.lz, fields))
    print(f"max per-field relative difference, 3x3 numba vs NumPy: {worst:.1e}")
    assert worst <= 1e-12


def test_numba_3x3_kernel_is_deterministic_without_fastmath_or_loop_allocations():
    df = warped(256)
    c = slice_arrays(df, df.kz.unique()[1])
    fields = np.geomspace(1e-3, 1e3, 40) / x_per_tesla()
    original = numba.get_num_threads()
    try:
        results = []
        for count in sorted({1, 3, numba.config.NUMBA_NUM_THREADS}):
            numba.set_num_threads(count)
            results.append(_kernel.orbit_sums_both3(c.damping, c.lx, c.ly, c.lz, fields))
    finally:
        numba.set_num_threads(original)
    assert all(np.array_equal(r, results[0]) for r in results)

    kernel = _kernel._orbit_sums_blocks3
    options = {k: v for k, v in kernel.targetoptions.items() if k not in ("cache", "nopython")}
    fresh = numba.njit(**options)(kernel.py_func)
    fresh(np.full(64, 1e-2), np.ones(64), np.ones(64), np.ones(64), np.array([1.0, 2.0]), 2, True)
    ir = "\n".join(fresh.inspect_llvm().values())
    assert not [flag for flag in (" fast ", " reassoc ", " contract ", " nnan ", " arcp ", " afn ") if flag in ir]
    bodies = [body for body in re.split(r"\ndefine ", ir) if "__numba_parfor_gufunc" in body.split("{", 1)[0]]
    assert bodies and not any("NRT_MemInfo_alloc" in body for body in bodies)


@pytest.mark.slow
@pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
def test_converges_to_the_reference_as_n_to_the_minus_two(x):
    """Same four k_z slices on both sides; in-plane N = 64…1024. The in-plane block and σ_zz
    (≈10³ times smaller, so checked on its own) must both converge with slope in [−2.1, −1.9]."""
    field = x / x_per_tesla()
    kzs = -np.pi / D + 2 * np.pi * np.arange(4) / (4 * D)
    exact = np.zeros((3, 3))
    for kz in kzs:
        contour = ref.ParametricContour.polar(k_fermi=lambda p, kz=kz: k_fermi(p, kz),
                                              dk_fermi=lambda p, kz=kz: dk_fermi_dphi(p, kz), mass=M, tau=TAU)
        vz = lambda p, kz=kz: HBAR / M * k_fermi(p, kz) * KW * D * np.sin(kz * D) * np.cos(2 * p)  # noqa: E731
        exact += ref.sigma(dataclasses.replace(contour, vz=vz), field, layer_spacing=D)[0] / kzs.size
    sizes = np.array([64, 128, 256, 512, 1024])
    plane, zz = [], []
    for n in sizes:
        actual = tensor(bz.conductivity(warped(n), field, layer_spacing=D, kz="kz"))[0]
        plane.append(np.max(np.abs(actual[:2, :2] - exact[:2, :2])) / np.max(np.abs(exact[:2, :2])))
        zz.append(abs(actual[2, 2] / exact[2, 2] - 1))
    slopes = [np.polyfit(np.log(sizes), np.log(e), 1)[0] for e in (plane, zz)]
    print(f"x = {x}: in-plane errors " + ", ".join(f"{e:.1e}" for e in plane) + f" (slope {slopes[0]:.3f}); "
          "sigma_zz errors " + ", ".join(f"{e:.1e}" for e in zz) + f" (slope {slopes[1]:.3f})")
    assert all(-2.1 <= slope <= -1.9 for slope in slopes)
