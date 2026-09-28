"""M6: FermiSurface rebuilt on the exact solver.

Gate: FermiSurface on a circle passes K1. Also documents how its numbers changed
from the old one-orbit implementation (the snapshot in tests/data).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
K_F = 7e9
TAU = 1e-13
C = 1e-9
X = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
SNAPSHOT = Path(__file__).parent / "data" / "boltzmann_legacy_snapshot.json"


def circle_surface(n=512, endpoint=True, **kwargs):
    theta = np.linspace(0, 2 * np.pi, n + 1 if endpoint else n, endpoint=endpoint)
    kwargs.setdefault("mass", M)
    kwargs.setdefault("tau", TAU)
    return bz.FermiSurface(theta, np.full(theta.size, K_F), kwargs["mass"], kwargs["tau"], C)


@pytest.mark.parametrize("endpoint", [True, False], ids=["closing-point", "open-grid"])
def test_k1_on_fermi_surface(endpoint):
    """K1 through FermiSurface: Drude σ (1e-4), no MR (|MR| < 1e-4), R_H = −1/(ne) (1e-4),
    with n = FermiSurface.carrier_density() = 2·πk_F²/(4π²c)."""
    fs = circle_surface(endpoint=endpoint)
    n = fs.carrier_density()
    assert n == pytest.approx(an.circle_density(K_F, C), rel=1e-12)
    fields = np.concatenate([[0.0], X * M / (E * TAU)])
    sxx, sxy, syx, syy = fs.calculate_conductivity(fields)
    sigma = np.stack([sxx, sxy, syx, syy], axis=1).reshape(-1, 2, 2)
    expected = an.drude_circle(fields[1:], density=n, mass=M, tau=TAU)
    sigma_error = np.max(np.abs(sigma[1:] - expected) / np.abs(expected))
    rho_xx = an.resistivity(sigma)[:, 0, 0]
    mr = np.max(np.abs(rho_xx / rho_xx[0] - 1))
    hall = np.max(np.abs(an.hall_coefficient(sigma[1:], fields[1:]) * (-n * E) - 1))
    print(f"sigma {sigma_error:.1e}, |MR| {mr:.1e}, R_H {hall:.1e}")
    assert sigma_error < 1e-4 and mr < 1e-4 and hall < 1e-4


def test_scalar_field_returns_floats_and_arrays_return_arrays():
    fs = circle_surface(128)
    single = fs.calculate_conductivity(3.0)
    assert len(single) == 4 and all(isinstance(value, float) for value in single)
    several = fs.calculate_conductivity([3.0, 30.0])
    assert all(value.shape == (2,) for value in several)
    np.testing.assert_allclose([v[0] for v in several], single, rtol=1e-15)


def test_old_integration_options_are_accepted_but_ignored():
    fs = circle_surface(128)
    with pytest.warns(DeprecationWarning):
        with_options = fs.calculate_conductivity(3.0, phi_points=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        without = fs.calculate_conductivity(3.0)
    assert with_options == without


def test_contour_velocities_are_normal_with_speed_hbar_kf_over_m():
    """On a rounded square the normal comes from dk_F/dθ (3-point, uneven grids allowed)."""
    theta = np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, 400))
    k = 7.35e9 - 0.25e9 * np.cos(4 * theta)
    fs = bz.FermiSurface(theta, k, 5 * M, TAU, C)
    df = fs.contour()
    np.testing.assert_allclose(np.hypot(df.vx, df.vy), HBAR * k / (5 * M), rtol=1e-14)
    exact_x = k * np.cos(theta) - 1e9 * np.sin(4 * theta) * -np.sin(theta)
    exact_y = k * np.sin(theta) - 1e9 * np.sin(4 * theta) * np.cos(theta)
    norm = np.hypot(exact_x, exact_y)
    error = np.max(np.hypot(df.vx / np.hypot(df.vx, df.vy) - exact_x / norm, df.vy / np.hypot(df.vx, df.vy) - exact_y / norm))
    print(f"max normal-direction error on an uneven 400-point grid: {error:.1e}")
    assert error < 1e-2


def test_change_from_the_old_implementation_is_documented():
    """The old code (snapshot) truncated after one orbit, an error of ~e^{−2π/ω_cτ}. On the
    snapshot's circle (99 nodes) the new FermiSurface stays within its O(N⁻²) error of
    Drude at every field (under 2e-3 with so few nodes: about 12/N²); the old one is badly
    off once ω_cτ ≳ 1."""
    data = json.loads(SNAPSHOT.read_text())["cases"]["circle"]
    fs = bz.FermiSurface(np.array(data["theta"]), np.array(data["fermi_wavevector"]),
                         np.array(data["effective_mass"]), np.array(data["relaxation_time"]), data["c_axis_length"])
    mass, tau = data["effective_mass"][0], data["relaxation_time"][0]
    n = fs.carrier_density()
    for field, old in zip(data["fields"], data["sigma_xx_xy_yx_yy"]):
        drude = an.drude_circle(field, density=n, mass=mass, tau=tau)[0]
        new = fs.calculate_conductivity(field)
        old_error = abs(old[0] / drude[0, 0] - 1)
        new_error = abs(new[0] / drude[0, 0] - 1)
        x = E * field * tau / mass
        print(f"omega_c tau = {x:.2f}: sigma_xx vs Drude, old {old_error:.1e}, new {new_error:.1e} "
              f"(the snapshot uses only {len(data['theta']) - 1} nodes)")
        assert new_error < 2e-3
        if x > 1:
            assert old_error > 0.1
