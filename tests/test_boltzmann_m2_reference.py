"""M2 reference: brute-force quadrature of the Chambers tube integral.

The reference must reproduce exact analytic results to 1e-8 before it is used to
check the fast kernels. Each test prints the expected value, the reference value and
the relative error.
"""
import numpy as np
import pytest

from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._reference import ParametricContour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

from quantalyze.beta.boltzmann import _analytic as an

E = ELEMENTARY_CHARGE
D = 1e-9  # m
K_F = 7.0e9  # m⁻¹
TAU = 1.0e-13  # s
X = np.array([0.01, 0.1, 1.0, 10.0, 100.0])  # ω_cτ


def relative_errors(actual, expected):
    """Per-component relative error, for tensors whose components are all non-zero."""
    return np.abs(actual - expected) / np.abs(expected)


@pytest.mark.parametrize("charge", [-E, +E])
def test_k1_isotropic_drude(charge):
    """K1: circle, m = m_e, constant τ. σ_xx = σ_yy = σ₀/(1+x²), σ_xy = −σ_yx = sσ₀x/(1+x²),
    σ₀ = ne²τ/m, x = eBτ/m (Drude; Ashcroft & Mermin ch. 1)."""
    fields = X * ELECTRON_MASS / (E * TAU)
    contour = ParametricContour.circle(k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU)
    actual = ref.sigma(contour, fields, layer_spacing=D, charge=charge)
    expected = an.drude_circle(fields, density=an.circle_density(K_F, D), mass=ELECTRON_MASS, tau=TAU,
                               charge=charge)
    error = relative_errors(actual, expected)
    for x, a, e, err in zip(X, actual, expected, error):
        print(f"x = {x:6g}: sigma_xx {a[0, 0]:.12e} vs {e[0, 0]:.12e}, sigma_xy {a[0, 1]:.12e} vs "
              f"{e[0, 1]:.12e}, max rel error {err.max():.1e}")
    assert np.max(error) < 1e-8


def test_k4_anisotropic_mass():
    """K4: ellipse, m_y = 4m_x, constant τ, x = eBτ/√(m_x m_y). σ_xx = ne²τ/m_x/(1+x²),
    σ_yy = ne²τ/m_y/(1+x²), σ_xy = −σ_yx = s ne²τx/(√(m_x m_y)(1+x²)); ρ_xx = m_x/(ne²τ),
    ρ_yy = m_y/(ne²τ) and R_H = −1/(ne) at every B (anisotropic-mass Drude)."""
    mx, my = ELECTRON_MASS, 4 * ELECTRON_MASS
    fields = X * np.sqrt(mx * my) / (E * TAU)
    n = an.circle_density(K_F, D)  # the ellipse has area πk_F²
    contour = ParametricContour.ellipse(k_fermi=K_F, mass_x=mx, mass_y=my, tau=TAU)
    actual = ref.sigma(contour, fields, layer_spacing=D)
    expected = an.drude_ellipse(fields, density=n, mass_x=mx, mass_y=my, tau=TAU)
    error = relative_errors(actual, expected)
    print("sigma max rel error per field:", ", ".join(f"{e:.1e}" for e in error.max(axis=(1, 2))))
    assert np.max(error) < 1e-8

    rho = an.resistivity(actual)
    rho_xx_error = np.abs(rho[:, 0, 0] / (mx / (n * E**2 * TAU)) - 1)
    rho_yy_error = np.abs(rho[:, 1, 1] / (my / (n * E**2 * TAU)) - 1)
    hall_error = np.abs(an.hall_coefficient(actual, fields) * (-n * E) - 1)
    print(f"no MR: rho_xx {rho_xx_error.max():.1e}, rho_yy {rho_yy_error.max():.1e}; "
          f"R_H = -1/(ne): {hall_error.max():.1e}")
    assert rho_xx_error.max() < 1e-8 and rho_yy_error.max() < 1e-8 and hall_error.max() < 1e-8


def test_k4_rotated_ellipse():
    """K4: rotating the ellipse by α = 30° gives σ → R(α) σ Rᵀ(α)."""
    mx, my = ELECTRON_MASS, 4 * ELECTRON_MASS
    fields = X[[0, 2, 4]] * np.sqrt(mx * my) / (E * TAU)
    alpha = np.pi / 6
    rotated = ref.sigma(ParametricContour.ellipse(k_fermi=K_F, mass_x=mx, mass_y=my, tau=TAU, rotation=alpha),
                        fields, layer_spacing=D)
    expected = an.rotate(an.drude_ellipse(fields, density=an.circle_density(K_F, D), mass_x=mx, mass_y=my,
                                          tau=TAU), alpha)
    error = np.max(np.abs(rotated - expected), axis=(1, 2)) / np.max(np.abs(expected), axis=(1, 2))
    print("rotated: max |d sigma| / max |sigma| per field:", ", ".join(f"{e:.1e}" for e in error))
    assert np.max(error) < 1e-8


def test_zero_field_branch():
    """σ(0) = σ₀ 𝟙 on the K1 circle, and σ(10⁻⁶ T) is the Drude tensor there: the field
    branch joins the zero-field branch continuously (σ_xy ≈ −σ₀ω_cτ ≈ −2e-8 σ₀ is real)."""
    contour = ParametricContour.circle(k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU)
    sigma_0 = an.circle_density(K_F, D) * E**2 * TAU / ELECTRON_MASS
    zero, tiny = ref.sigma(contour, [0.0, 1e-6], layer_spacing=D)
    drude = an.drude_circle(1e-6, density=an.circle_density(K_F, D), mass=ELECTRON_MASS, tau=TAU)[0]
    print(f"sigma(0)/sigma_0 - 1 = {zero[0, 0] / sigma_0 - 1:.1e}, "
          f"max |sigma(1e-6 T) - Drude| / sigma_0 = {np.max(np.abs(tiny - drude)) / sigma_0:.1e}")
    np.testing.assert_allclose(zero, sigma_0 * np.eye(2), rtol=1e-10, atol=1e-12 * sigma_0)
    np.testing.assert_allclose(tiny, drude, rtol=0, atol=1e-12 * sigma_0)


@pytest.mark.slow
@pytest.mark.parametrize("x", [0.1, 10.0, 100.0])
def test_result_unchanged_when_more_orbits_are_summed(x):
    """Once e^{−KZ} < 1e-16, adding orbits changes nothing (1e-12). Keeping only the
    current orbit, as the old code did, is badly wrong once ω_cτ ≳ 1."""
    contour = ParametricContour.polar(k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p),
                                      dk_fermi=lambda p: 1e9 * np.sin(4 * p), mass=5 * ELECTRON_MASS,
                                      tau=lambda p: TAU / (1 + 0.6 * np.cos(4 * p)))
    field = x * 5 * ELECTRON_MASS / (E * TAU)
    k = ref.orbit_count(contour, field)
    converged = ref.sigma(contour, field, layer_spacing=D)
    more = ref.sigma(contour, field, layer_spacing=D, orbits=3 * k)
    one = ref.sigma(contour, field, layer_spacing=D, orbits=1)
    change = np.max(np.abs(more - converged)) / np.max(np.abs(converged))
    truncated = np.max(np.abs(one - converged)) / np.max(np.abs(converged))
    print(f"x = {x}: K = {k}, change with 3K orbits = {change:.1e}, error with one orbit = {truncated:.1e}")
    assert change <= 1e-12
    if x >= 10:
        assert truncated > 0.1


@pytest.mark.slow
def test_parametrisation_does_not_matter():
    """The same circle, parametrised non-uniformly, backwards, and with dk/dφ from finite
    differences, gives the same σ: the reference's own geometry is consistent."""
    fields = np.array([0.1, 1.0, 10.0]) * ELECTRON_MASS / (E * TAU)
    tau = lambda phi: TAU / (1 + 0.6 * np.cos(4 * phi))  # noqa: E731  (of the polar angle)
    standard = ref.sigma(ParametricContour.circle(k_fermi=K_F, mass=ELECTRON_MASS, tau=tau), fields,
                         layer_spacing=D)
    v_f = HBAR * K_F / ELECTRON_MASS

    def warped(sign):
        angle = lambda p: sign * (p + 0.3 * np.sin(p))  # noqa: E731  (monotone, 2π-periodic offset)
        return ParametricContour(
            k=lambda p: (K_F * np.cos(angle(p)), K_F * np.sin(angle(p))),
            v=lambda p: (v_f * np.cos(angle(p)), v_f * np.sin(angle(p))),
            tau=lambda p: tau(angle(p)),
        )

    for label, contour in [("non-uniform, finite-difference dk", warped(+1)), ("backwards", warped(-1))]:
        actual = ref.sigma(contour, fields, layer_spacing=D)
        error = np.max(np.abs(actual - standard)) / np.max(np.abs(standard))
        print(f"{label}: max |d sigma| / max |sigma| = {error:.1e}")
        assert error < 1e-9


def test_k7_anisotropic_tau_on_a_circle():
    """K7: circle, 1/τ(φ) = (1/τ₀)(1 + 0.6 cos4φ), x̄ = ω_c⟨τ⟩.

    σ(0) = (ne²/m)⟨τ⟩𝟙. Ong's low-field Hall coefficient R_H(0) = (1/nq)⟨τ²⟩/⟨τ⟩² = 1.25/nq
    (N. P. Ong, PRB 43, 193 (1991)). At high field R_H → 1/nq and ρ_xx(∞)/ρ_xx(0) = ⟨1/τ⟩⟨τ⟩
    = 1.25. The exact solution approaches these as x̄² and 1/x̄², so each limit is
    Richardson-extrapolated, L ≈ (4R(x̄) − R(2x̄))/3, and must match to 1e-8. MR is
    positive and rises monotonically.
    """
    mean_tau = 1.25 * TAU  # ⟨τ⟩ = τ₀/√(1 − 0.6²)
    contour = ParametricContour.circle(k_fermi=K_F, mass=ELECTRON_MASS,
                                       tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6))
    n = an.circle_density(K_F, D)
    q = -E
    xbar = np.array([1e-4, 2e-4, 1e-3, 0.1, 1.0, 10.0, 1e3, 2e3])
    fields = xbar * ELECTRON_MASS / (E * mean_tau)
    sigma = ref.sigma(contour, np.concatenate([[0.0], fields]), layer_spacing=D, charge=q)
    hall = an.hall_coefficient(sigma[1:], fields) * n * q  # R_H in units of 1/nq
    rho_xx = an.resistivity(sigma)[:, 0, 0]
    ratio = rho_xx[1:] / rho_xx[0]
    r = dict(zip(xbar, zip(hall, ratio)))

    zero_error = np.max(np.abs(sigma[0] - n * E**2 * mean_tau / ELECTRON_MASS * np.eye(2))) / (
        n * E**2 * mean_tau / ELECTRON_MASS)
    low = (4 * r[1e-4][0] - r[2e-4][0]) / 3
    high = (4 * r[2e3][0] - r[1e3][0]) / 3
    saturated = (4 * r[2e3][1] - r[1e3][1]) / 3
    checks = [
        ("sigma(0) = (ne^2/m)<tau>", zero_error, 0.0),
        ("R_H(0) nq = 1.25", low / 1.25 - 1, None),
        ("R_H(inf) nq = 1", high - 1, None),
        ("rho_xx(inf)/rho_xx(0) = 1.25", saturated / 1.25 - 1, None),
    ]
    for label, error, _ in checks:
        print(f"{label}: relative error {error:+.1e}")
    for label, error, _ in checks:
        assert abs(error) < 1e-8, label

    # The extrapolation is justified: the corrections really scale as x̄² and 1/x̄²
    low_scaling = (r[2e-4][0] - 1.25) / (r[1e-4][0] - 1.25)
    high_scaling = (r[1e3][0] - 1) / (r[2e3][0] - 1)
    print(f"correction ratio between x and 2x: low field {low_scaling:.4f}, high field {high_scaling:.4f} (4 expected)")
    assert abs(low_scaling / 4 - 1) < 1e-2 and abs(high_scaling / 4 - 1) < 1e-2

    # At the §8 points themselves the limits hold to §8's 1e-3
    print(f"at xbar = 1e-3: R_H nq = {r[1e-3][0]:.8f}; at xbar = 1e3: R_H nq = {r[1e3][0]:.8f}, "
          f"rho ratio = {r[1e3][1]:.8f}")
    assert abs(r[1e-3][0] / 1.25 - 1) < 1e-3
    assert abs(r[1e3][0] - 1) < 1e-3 and abs(r[1e3][1] / 1.25 - 1) < 1e-3

    mr = ratio - 1
    print("MR at xbar =", ", ".join(f"{x:g}: {m:.3e}" for x, m in zip(xbar, mr)))
    assert np.all(mr > 0) and np.all(np.diff(mr) > 0)
