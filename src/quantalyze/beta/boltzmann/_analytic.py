"""Analytic results that the Boltzmann tests compare against (private; tests only).

The Drude and two-band results are derived from the Drude equation of motion
m_i dv_i/dt = q(E + v × B)_i − m_i v_i/τ with B = B ẑ, which is independent of
the Chambers tube integral used by quantalyze.beta.boltzmann. The low- and high-field
expansions of the Chambers σ for an arbitrary smooth pocket are evaluated spectrally
on a parametrised contour, sharing no code with the kernels. SI units throughout.
Tensors have index order [[xx, xy], [yx, yy]].
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ...core.constants import ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE


def circle_density(k_fermi, layer_spacing, spin_degeneracy=2):
    """n = g_s πk_F² / (4π² d) = g_s k_F² / (4π d), in m⁻³."""
    return spin_degeneracy * k_fermi**2 / (4 * np.pi * layer_spacing)


def rotate(sigma, angle):
    """R(α) σ Rᵀ(α) for a tensor of shape (..., 2, 2)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ sigma @ R.T


def drude_ellipse(field, *, density, mass_x, mass_y, tau, charge=-E, rotation=0.0):
    """Drude tensor of a band with principal masses m_x, m_y and constant τ (S/m).

    Steady state: v_x = μ_x(E_x + v_y B), v_y = μ_y(E_y − v_x B), μ_i = qτ/m_i, so
    σ = nq / (1 + μ_xμ_y B²) · [[μ_x, μ_xμ_y B], [−μ_xμ_y B, μ_y]].
    Rotating the band by α (about ẑ) gives R(α) σ Rᵀ(α).
    """
    B = np.atleast_1d(np.asarray(field, dtype=float))  # (nB,)
    mu_x = charge * tau / mass_x
    mu_y = charge * tau / mass_y
    denominator = 1 + mu_x * mu_y * B**2
    sigma = np.empty((B.size, 2, 2))
    sigma[:, 0, 0] = density * charge * mu_x / denominator
    sigma[:, 1, 1] = density * charge * mu_y / denominator
    sigma[:, 0, 1] = density * charge * mu_x * mu_y * B / denominator
    sigma[:, 1, 0] = -sigma[:, 0, 1]
    return rotate(sigma, rotation)


def drude_circle(field, *, density, mass, tau, charge=-E):
    """Isotropic Drude tensor: σ_xx = σ₀/(1+x²), σ_xy = sσ₀x/(1+x²), x = eBτ/m (S/m)."""
    return drude_ellipse(field, density=density, mass_x=mass, mass_y=mass, tau=tau, charge=charge)


def drude_mobility(field, *, density, mobility, charge=-E):
    """Isotropic Drude tensor from a density and a (positive) mobility μ = eτ/m (S/m)."""
    # Only the ratio τ/m enters, so take m = 1 kg.
    return drude_circle(field, density=density, mass=1.0, tau=mobility / E, charge=charge)


def two_band(field, *, n_e, mu_e, n_h, mu_h):
    """σ of an electron band plus a hole band: the Drude tensors summed (S/m)."""
    return (
        drude_mobility(field, density=n_e, mobility=mu_e, charge=-E)
        + drude_mobility(field, density=n_h, mobility=mu_h, charge=+E)
    )


def two_band_rho_xx(field, *, n_e, mu_e, n_h, mu_h):
    """Closed-form two-band ρ_xx (Ω·m).

    ρ_xx = [(n_eμ_e + n_hμ_h) + (n_eμ_h + n_hμ_e)μ_eμ_h B²]
           / (e[(n_eμ_e + n_hμ_h)² + (n_h − n_e)²μ_e²μ_h²B²])
    """
    B = np.asarray(field, dtype=float)
    numerator = (n_e * mu_e + n_h * mu_h) + (n_e * mu_h + n_h * mu_e) * mu_e * mu_h * B**2
    denominator = E * ((n_e * mu_e + n_h * mu_h) ** 2 + (n_h - n_e) ** 2 * mu_e**2 * mu_h**2 * B**2)
    return numerator / denominator


def two_band_hall_coefficient(field, *, n_e, mu_e, n_h, mu_h):
    """Closed-form two-band R_H (m³/C).

    R_H = [(n_hμ_h² − n_eμ_e²) + (n_h − n_e)μ_e²μ_h²B²]
          / (e[(n_eμ_e + n_hμ_h)² + (n_h − n_e)²μ_e²μ_h²B²])
    """
    B = np.asarray(field, dtype=float)
    numerator = (n_h * mu_h**2 - n_e * mu_e**2) + (n_h - n_e) * mu_e**2 * mu_h**2 * B**2
    denominator = E * ((n_e * mu_e + n_h * mu_h) ** 2 + (n_h - n_e) ** 2 * mu_e**2 * mu_h**2 * B**2)
    return numerator / denominator


def resistivity(sigma):
    """ρ = σ⁻¹ for a tensor of shape (nB, 2, 2) (Ω·m)."""
    return np.linalg.inv(sigma)


def hall_coefficient(sigma, field):
    """R_H = ½(ρ_yx − ρ_xy) / B (m³/C). NaN at B = 0."""
    rho = resistivity(sigma)
    B = np.atleast_1d(np.asarray(field, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(B == 0, np.nan, 0.5 * (rho[:, 1, 0] - rho[:, 0, 1]) / B)


def angular_average(f):
    """⟨f⟩ = (1/2π) ∫₀^{2π} f(φ) dφ by adaptive quadrature."""
    value, _ = quad(f, 0.0, 2 * np.pi, epsabs=0.0, epsrel=1e-13, limit=500)
    return value / (2 * np.pi)


# ---------------------------------------------------------------------------
# Low- and high-field expansions of the Chambers σ for any smooth closed pocket
#
# In the damping coordinate g = ∫ds/τ (T), with the mean free path ℓ = vτ (m), the
# history obeys B·Dw = ℓ − w, where D = d/dg along the carriers' motion and B > 0, and
# σ_ij = (g_s e³/4π²ħ²d) ∮ ℓ_i w_j dg. Expanding w = (1 + BD)⁻¹ℓ in powers of B gives the
# Jones–Zener series; expanding in 1/B gives the high-field series. Both are evaluated
# spectrally on the parametrised contour, independently of the kernels.
# ---------------------------------------------------------------------------

def _wavenumbers(m):
    k = np.fft.fftfreq(m, d=1.0 / m)  # integers 0, 1, ..., −1
    if m % 2 == 0:
        k[m // 2] = 0.0  # the Nyquist mode has no well-defined derivative
    return k


def _periodic_derivative(values):
    """d/dφ of smooth 2π-periodic samples at φ_j = 2πj/M (last axis)."""
    k = _wavenumbers(values.shape[-1])
    return np.fft.ifft(1j * k * np.fft.fft(values, axis=-1), axis=-1).real


def _periodic_antiderivative(values):
    """The zero-mean antiderivative in φ of smooth periodic samples whose mean is zero."""
    k = _wavenumbers(values.shape[-1])
    inverse = np.zeros(k.shape, dtype=complex)
    inverse[k != 0] = 1 / (1j * k[k != 0])
    return np.fft.ifft(inverse * np.fft.fft(values, axis=-1), axis=-1).real


def _orbit_on_grid(contour, points, charge):
    """dg/dφ (T), ℓ (m, shape (d, M)) and the direction of motion (+1 along +φ) for B > 0."""
    phi = 2 * np.pi * np.arange(points) / points
    kx, ky = contour.k(phi)
    if contour.dk is not None:
        dkx, dky = (np.broadcast_to(a, phi.shape) for a in contour.dk(phi))
    else:
        dkx, dky = _periodic_derivative(np.stack([kx, ky]))
    vx, vy = (np.broadcast_to(a, phi.shape).astype(float) for a in contour.v(phi))
    tau = np.broadcast_to(contour.tau(phi), phi.shape).astype(float)
    components = [vx, vy] if contour.vz is None else [vx, vy, np.broadcast_to(contour.vz(phi), phi.shape)]
    ell = np.stack(components) * tau  # (d, M)
    g_prime = HBAR * np.hypot(dkx, dky) / (E * np.hypot(vx, vy) * tau)  # (M,)
    # ħ dk/dt = q v × B: for B = +B ẑ the motion is along q (v_y, −v_x).
    along = np.sign(charge) * (dkx * vy - dky * vx)
    if not (np.all(along > 0) or np.all(along < 0)):
        raise ValueError("v must be normal to the contour and point to the same side all the way round")
    return g_prime, ell, (1.0 if along[0] > 0 else -1.0)


def _prefactor(layer_spacing, spin_degeneracy):
    return spin_degeneracy * E**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)


def jones_zener(contour, *, layer_spacing, charge=-E, spin_degeneracy=2, points=8192):
    """Low-field expansion σ(B) = σ⁽⁰⁾ + Bσ⁽¹⁾ + B²σ⁽²⁾ + O(B³) for B along +ẑ (S/m, S/m/T, S/m/T²).

    With D = d/dg along the motion, w = ℓ − BDℓ + B²D²ℓ − …, so
    σ⁽⁰⁾ = ∮ℓ_iℓ_j dg, σ⁽¹⁾ = −∮ℓ_i Dℓ_j dg (antisymmetric: Ong's ℓ-area) and
    σ⁽²⁾ = ∮ℓ_i D²ℓ_j dg = −∮Dℓ_i Dℓ_j dg, each times g_s e³/(4π²ħ²d). Spectrally accurate
    for smooth pockets resolved by `points`.

    Args:
        contour: A closed `ParametricContour` (from the reference module).
        layer_spacing: Interlayer spacing d (m).
        charge: Carrier charge q (C).
        spin_degeneracy: Spin degeneracy g_s.
        points: Number of equally spaced parameter values.

    Returns:
        (σ⁽⁰⁾, σ⁽¹⁾, σ⁽²⁾), each of shape (d, d).
    """
    g_prime, ell, direction = _orbit_on_grid(contour, points, charge)
    dell = _periodic_derivative(ell)  # dℓ/dφ, (d, M)
    step = 2 * np.pi / points
    zeroth = np.einsum("m,im,jm->ij", g_prime * step, ell, ell)
    first = -direction * np.einsum("m,im,jm->ij", np.full(points, step), ell, dell)
    second = -np.einsum("m,im,jm->ij", step / g_prime, dell, dell)
    prefactor = _prefactor(layer_spacing, spin_degeneracy)
    return prefactor * zeroth, prefactor * first, prefactor * second


def low_field_resistivity(zeroth, first, second):
    """ρ(B) = ρ⁽⁰⁾ + Bρ⁽¹⁾ + B²ρ⁽²⁾ + O(B³) from the σ expansion (Ω·m, Ω·m/T, Ω·m/T²).

    R_H(0) = ½(ρ⁽¹⁾_yx − ρ⁽¹⁾_xy), and the magnetoresistance is ρ⁽²⁾_ii/ρ⁽⁰⁾_ii · B² + O(B⁴).
    """
    r0 = np.linalg.inv(zeroth)
    return r0, -r0 @ first @ r0, r0 @ first @ r0 @ first @ r0 - r0 @ second @ r0


def high_field(contour, *, layer_spacing, charge=-E, spin_degeneracy=2, points=8192):
    """High-field expansion σ(B) = H⁽¹⁾/B + H⁽²⁾/B² + O(1/B³) of a closed pocket, B along +ẑ.

    With R = D⁻¹ℓ, the real-space orbit scaled by B (zero mean over g),
    w = R/B − D⁻¹R/B² + …, so H⁽¹⁾ = ∮ℓ_i R_j dg (antisymmetric) and H⁽²⁾ = ∮R_iR_j dg,
    each times g_s e³/(4π²ħ²d). ∮ℓ dg = 0 on a closed orbit, which makes R periodic.

    Returns:
        (H⁽¹⁾, H⁽²⁾) in S·T/m and S·T²/m, each of shape (d, d).
    """
    g_prime, ell, direction = _orbit_on_grid(contour, points, charge)
    flux = ell * g_prime  # dR/dφ up to the direction, (d, M)
    flux = flux - flux.mean(axis=1, keepdims=True)  # zero analytically; removes rounding
    orbit = direction * _periodic_antiderivative(flux)
    orbit -= np.sum(orbit * g_prime, axis=1, keepdims=True) / np.sum(g_prime)
    step = 2 * np.pi / points
    first = np.einsum("m,im,jm->ij", g_prime * step, ell, orbit)
    second = np.einsum("m,im,jm->ij", g_prime * step, orbit, orbit)
    prefactor = _prefactor(layer_spacing, spin_degeneracy)
    return prefactor * first, prefactor * second


def high_field_resistivity(first, second):
    """ρ(B) = B·H⁽¹⁾⁻¹ + ρ(∞) + O(1/B) with ρ(∞) = −H⁽¹⁾⁻¹H⁽²⁾H⁽¹⁾⁻¹ (2×2 only).

    Returns:
        (H⁽¹⁾⁻¹, ρ(∞)): the Hall slope (R_H(∞) = ½(H⁽¹⁾⁻¹_yx − H⁽¹⁾⁻¹_xy)) and the saturated ρ.
    """
    inverse = np.linalg.inv(first)
    return inverse, -inverse @ second @ inverse

def spectral_area(kx, ky):
    """Area enclosed by a smoothly, evenly parametrised closed contour (m⁻²). Tests only.

    Treats the nodes as samples of a smooth periodic curve k(θ), θ_j = 2πj/N,
    differentiates spectrally, and applies the trapezoid rule to
    A = ½∮(k_x dk_y − k_y dk_x). Both steps are spectrally accurate for smooth
    periodic curves, so a uniformly sampled circle gives πk_F² to rounding
    (a polygon/shoelace area would only be O(N⁻²)).
    """
    kx = np.asarray(kx, dtype=float) - np.mean(kx)
    ky = np.asarray(ky, dtype=float) - np.mean(ky)
    n = kx.size
    wavenumber = np.fft.fftfreq(n, d=1.0 / n)  # integers 0, 1, ..., −1
    if n % 2 == 0:
        wavenumber[n // 2] = 0.0  # Nyquist mode has no well-defined derivative
    dkx = np.fft.ifft(1j * wavenumber * np.fft.fft(kx)).real  # dk_x/dθ
    dky = np.fft.ifft(1j * wavenumber * np.fft.fft(ky)).real  # dk_y/dθ
    return abs(0.5 * np.sum(kx * dky - ky * dkx) * (2 * np.pi / n))
