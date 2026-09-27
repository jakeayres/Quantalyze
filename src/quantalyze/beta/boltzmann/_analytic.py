"""Analytic results that the Boltzmann tests compare against (private; tests only).

Everything here is derived from the Drude equation of motion
m_i dv_i/dt = q(E + v × B)_i − m_i v_i/τ with B = B ẑ, which is independent of
the Chambers tube integral used by quantalyze.beta.boltzmann. SI units throughout.
Tensors have shape (nB, 2, 2) with index order [[xx, xy], [yx, yy]].
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ...core.constants import ELEMENTARY_CHARGE

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
