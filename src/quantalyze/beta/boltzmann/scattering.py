"""Models of the relaxation time τ(φ) around a Fermi-surface pocket.

Each model takes the polar angle φ (rad) of the Fermi-surface points about the
pocket centre and returns τ (s). Pass one to a generator through a lambda or
`functools.partial`, for example
`tau=lambda phi: bz.scattering.cos4phi(phi, 1e-13, anisotropy=0.6)`.
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
