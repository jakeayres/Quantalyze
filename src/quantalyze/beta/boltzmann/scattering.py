"""Scattering models: relaxation times τ(φ), and scattering kernels P(k, k′).

The relaxation-time models take the polar angle φ (rad) of the Fermi-surface points
about the pocket centre and return τ (s). Pass one to a generator through a lambda or
`functools.partial`, for example
`tau=lambda phi: bz.scattering.cos4phi(phi, 1e-13, anisotropy=0.6)`.

A scattering kernel goes further: it says where scattered carriers go, not only how
often they leave, so the conductivity includes the in-scattering (the vertex
correction). `spin_fluctuation_kernel` builds one for scattering by spin fluctuations
peaked at an ordering wavevector; pass it to `conductivity(..., scattering_kernel=...)`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def constant(phi, tau):
    """Isotropic relaxation time.

    Args:
        phi: Polar angle about the pocket centre (rad), array-like.
        tau: Relaxation time (s).

    Returns:
        ndarray of τ (s), the shape of `phi`.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> bz.scattering.constant(np.zeros(3), 1e-13)
        array([1.e-13, 1.e-13, 1.e-13])
    """
    return np.full(np.shape(phi), float(tau))


def cos4phi(phi, tau0, anisotropy):
    """Fourfold scattering rate, 1/τ(φ) = (1/τ₀)(1 + a cos4φ).

    For this model ⟨τ⟩ = τ₀/√(1 − a²), and ⟨τ²⟩/⟨τ⟩² = ⟨1/τ⟩⟨τ⟩ = 1/√(1 − a²).

    Args:
        phi: Polar angle about the pocket centre (rad), array-like.
        tau0: Relaxation time τ₀ set by the mean scattering rate (s).
        anisotropy: Amplitude a of the cos4φ modulation of 1/τ, with |a| < 1.

    Returns:
        ndarray of τ (s), the shape of `phi`.

    Raises:
        ValueError: If |a| ≥ 1, which would make 1/τ vanish or go negative.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> tau = bz.scattering.cos4phi(np.array([0.0, np.pi / 4]), 1e-13, anisotropy=0.6)
    """
    if not abs(anisotropy) < 1:
        raise ValueError(f"|anisotropy| must be below 1, not {anisotropy}")
    return tau0 / (1 + anisotropy * np.cos(4 * np.asarray(phi, dtype=float)))


def hot_spot(phi, tau0, strength, width, positions: Sequence[float] = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    """Isotropic scattering plus narrow hot spots.

    1/τ(φ) = (1/τ₀)[1 + h Σ_p exp((cos(φ − φ_p) − 1) / w²)]. Each bump is a periodic
    (von Mises) peak of height h and angular width ≈ w centred on φ_p, smooth all the
    way round the pocket.

    Args:
        phi: Polar angle about the pocket centre (rad), array-like.
        tau0: Relaxation time away from the hot spots (s).
        strength: Height h of each hot spot, so 1/τ = (1 + h)/τ₀ on a hot spot.
        width: Angular width w of each hot spot (rad).
        positions: Hot-spot angles φ_p (rad). The default puts one along each axis,
            the antinodal directions of a pocket about (π/a, π/a).

    Returns:
        ndarray of τ (s), the shape of `phi`.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> phi = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        >>> tau = bz.scattering.hot_spot(phi, 1e-13, strength=9.0, width=0.1)
    """
    phi = np.asarray(phi, dtype=float)
    bumps = np.zeros(phi.shape)
    for position in positions:
        bumps += np.exp((np.cos(phi - position) - 1) / width**2)
    return tau0 / (1 + strength * bumps)


def spin_fluctuation_kernel(strength, *, wavevector, correlation_length, lattice_constants, power=1):
    """A scattering kernel from spin fluctuations peaked at an ordering wavevector Q.

    P(k, k′) = strength · ½ Σ_± [1 + Σ_i ξ_i² δ_i²]^(−power), with δ = (k − k′) ∓ Q reduced
    component by component into the first Brillouin zone, [−π/a_i, π/a_i). This is the
    Ornstein–Zernike form of the spin susceptibility projected onto the Fermi surface.
    Carriers are scattered most strongly between points k and k′ ≈ k ± Q, so the
    scattering rate peaks at the "hot spots", where k + Q also lies on the Fermi surface,
    over a width of about 1/ξ. With power = 1 it is the quasi-static form (k_BT ≫ ħω_sf),
    P ∝ T χ(q); with power = 2 the low-temperature form, P ∝ T² χ(q)²/χ_Q. Both terms ±Q
    are kept, so P(k, k′) = P(k′, k), as detailed balance requires.

    The strength has units J·m³/s (a rate per unit density of states per spin, per
    volume), and carries the temperature dependence. Calibrate it with
    `bz.out_scattering_rate`, which gives the rate it produces at every node.

    Args:
        strength: The kernel at δ = 0 (J·m³/s).
        wavevector: Q = (Q_x, Q_y), or (Q_x, Q_y, Q_z) for a surface given as k_z slices (m⁻¹).
        correlation_length: ξ (m), one value or one per axis.
        lattice_constants: (a, b), or (a, b, c) (m): the reciprocal-lattice periods
            2π/a_i used to reduce δ. Only orthogonal axes are handled; for an oblique cell
            write your own kernel.
        power: 1 (quasi-static) or 2 (quantum regime).

    Returns:
        A function P(kx, ky, kx2, ky2), or P(kx, ky, kz, kx2, ky2, kz2) when Q has three
        components, for `conductivity(..., scattering_kernel=...)`.

    Raises:
        ValueError: If the arguments have the wrong shape or are not positive.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> a = 3.87e-10
        >>> kernel = bz.scattering.spin_fluctuation_kernel(
        ...     1e-33, wavevector=(np.pi / a, np.pi / a), correlation_length=5 * a, lattice_constants=(a, a))
    """
    q = np.asarray(wavevector, dtype=float)
    if q.ndim != 1 or q.size not in (2, 3) or not np.all(np.isfinite(q)):
        raise ValueError(f"wavevector must be (Q_x, Q_y) or (Q_x, Q_y, Q_z), not {wavevector!r}")
    dim = q.size
    xi = np.broadcast_to(np.asarray(correlation_length, dtype=float), (dim,))
    periods = 2 * np.pi / np.asarray(lattice_constants, dtype=float)
    if periods.shape != (dim,) or not np.all(np.isfinite(periods) & (periods > 0)):
        raise ValueError(f"lattice_constants must be {dim} positive lengths, one per component of Q, "
                         f"not {lattice_constants!r}")
    if not np.all(np.isfinite(xi) & (xi > 0)):
        raise ValueError(f"correlation_length must be positive, not {correlation_length!r}")
    if power not in (1, 2):
        raise ValueError(f"power must be 1 or 2, not {power!r}")
    strength = float(strength)
    if not (np.isfinite(strength) and strength >= 0):
        raise ValueError(f"strength must be finite and non-negative, not {strength!r}")

    def kernel(*k):
        if len(k) != 2 * dim:
            raise TypeError(f"this kernel takes {2 * dim} arguments (k and k′ with {dim} components each), "
                            f"not {len(k)}; the length of the wavevector sets it")
        total = 0.0
        for sign in (1.0, -1.0):
            squared = 0.0
            for i in range(dim):
                delta = np.asarray(k[i]) - np.asarray(k[dim + i]) - sign * q[i]
                delta = delta - periods[i] * np.round(delta / periods[i])  # into the first zone
                squared = squared + (xi[i] * delta) ** 2
            total = total + (1.0 + squared) ** (-power)
        return 0.5 * strength * total

    return kernel
