"""Plain-NumPy O(N) Chambers kernel (private). The oracle for the numba kernel.

On a prepared contour, carriers take geometric time s_n (s·T) to cross segment n
at mean scattering rate γ_n, so at field B the real time is h_n = s_n/|B| and the
damping is z_n = γ_n h_n. The history w(t) = ∫_{−∞}^{t} v(t′) e^{−∫γ} dt′ obeys
dw/dt = v − γw. Taking v linear in time across each segment and γ constant, one
segment is solved exactly (an exponential integrator):

    w_{n+1} = e^{−z_n} w_n + h_n [ v_n φ₂(z_n) + v_{n+1} (φ₁(z_n) − φ₂(z_n)) ].

Round a closed orbit w must return to itself, so starting from w_0 = 0 gives
w_N = S, and the periodic solution is w_0 = S / (1 − e^{−Z}), Z = Σ z_n. That sums
the contribution of every earlier orbit exactly, with no truncation in ω_cτ.
"""
from __future__ import annotations

import numpy as np

# Below this z, φ₁ and φ₂ come from their Taylor series through z⁵: the closed forms
# cancel catastrophically there.
_SMALL_Z = 1e-2
# φ₂'s closed form still loses ~4e-14 just above 1e-2 (1 − (1+z)e^{−z} ≈ z²/2), so up to
# z = 1 it is summed from a longer series, which is accurate to rounding.
_MEDIUM_Z = 1.0

# φ₁(z) = Σ (−z)^j / (j+1)!,  φ₂(z) = Σ (−1)^j (j+1)/(j+2)! z^j
_PHI1_SMALL = np.array([1.0, -1 / 2, 1 / 6, -1 / 24, 1 / 120, -1 / 720])
_PHI2_SMALL = np.array([1 / 2, -1 / 3, 1 / 8, -1 / 30, 1 / 144, -1 / 840])


def _phi2_coefficients(terms: int) -> np.ndarray:
    coefficients = np.empty(terms)
    factorial = 2.0  # (j + 2)! for j = 0
    for j in range(terms):
        coefficients[j] = (-1) ** j * (j + 1) / factorial
        factorial *= j + 3
    return coefficients


_PHI2_MEDIUM = _phi2_coefficients(19)  # through z¹⁸; the next term is < 1e-17 at z = 1


def _horner(coefficients: np.ndarray, z: np.ndarray) -> np.ndarray:
    result = np.full(z.shape, coefficients[-1])
    for c in coefficients[-2::-1]:
        result = result * z + c
    return result


def phi1(z) -> np.ndarray:
    """φ₁(z) = (1 − e^{−z})/z, accurate to rounding for all z ≥ 0."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape)
    small = z < _SMALL_Z
    out[small] = _horner(_PHI1_SMALL, z[small])
    large = ~small
    out[large] = -np.expm1(-z[large]) / z[large]
    return out


def phi2(z) -> np.ndarray:
    """φ₂(z) = (1 − (1 + z)e^{−z})/z², accurate to rounding for all z ≥ 0."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape)
    small = z < _SMALL_Z
    medium = (z >= _SMALL_Z) & (z < _MEDIUM_Z)
    large = z >= _MEDIUM_Z
    out[small] = _horner(_PHI2_SMALL, z[small])
    out[medium] = _horner(_PHI2_MEDIUM, z[medium])
    zl = z[large]
    # e^{−z} underflows to 0 for large z, which is correct: never divide by it.
    out[large] = (-np.expm1(-zl) - zl * np.exp(-zl)) / zl / zl  # z² would overflow for z > 1e154
    return out


def orbit_sums(s, gamma, vx, vy, field, vz=None) -> np.ndarray:
    """Σ_n ½ s_n (v_{i,n} w_{j,n} + v_{i,n+1} w_{j,n+1}) for each field (s²·T·m²/s², i.e. σ
    without the prefactor g_s e³ / 4π²ħ²d).

    The nodes must be ordered along the carriers' motion at these fields.

    Args:
        s: Geometric time of each segment (s·T), shape (N,).
        gamma: Scattering rate on each segment (s⁻¹), shape (N,).
        vx: Node velocities v_x (m/s), shape (N,).
        vy: Node velocities v_y (m/s), shape (N,).
        field: Field magnitudes |B| > 0 (T), shape (nB,).
        vz: Node velocities v_z (m/s), shape (N,), for a k_z slice of a warped surface;
            the orbit stays in its k_z plane (B ∥ ẑ) and v_z is carried along it.

    Returns:
        ndarray of shape (nB, 2, 2), or (nB, 3, 3) with vz.
    """
    s = np.asarray(s, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    b = np.asarray(field, dtype=np.float64)
    n_nodes = s.size
    components = [vx, vy] if vz is None else [vx, vy, vz]
    v = np.stack(components, axis=1).astype(np.float64)  # (N, d), d = 2 or 3
    v_next = np.roll(v, -1, axis=0)  # (N, d)

    h = s[None, :] / b[:, None]  # real time per segment, (nB, N)
    z = gamma[None, :] * h  # (nB, N)
    decay = np.exp(-z)  # (nB, N)
    p1, p2 = phi1(z), phi2(z)
    weight_start = h * p2  # multiplies v_n, (nB, N)
    weight_end = h * (p1 - p2)  # multiplies v_{n+1}, (nB, N)
    total = np.sum(z, axis=1)  # Z, (nB,)

    # Pass 1: one orbit from w_0 = 0 gives S.
    w = np.zeros((b.size, v.shape[1]))  # (nB, d)
    for n in range(n_nodes):
        w = decay[:, n, None] * w + weight_start[:, n, None] * v[n] + weight_end[:, n, None] * v_next[n]

    # Closure: the periodic w_0 = S / (1 − e^{−Z}); expm1 keeps it accurate when Z is small.
    history = np.empty((b.size, n_nodes, v.shape[1]))  # w_n, (nB, N, d)
    history[:, 0] = w / (-np.expm1(-total))[:, None]

    # Pass 2: w_n at every node, forwards again (dividing by e^{−z} would overflow).
    for n in range(n_nodes - 1):
        history[:, n + 1] = (decay[:, n, None] * history[:, n] + weight_start[:, n, None] * v[n]
                             + weight_end[:, n, None] * v_next[n])
    history_next = np.roll(history, -1, axis=1)  # w_{n+1}, with w_N = w_0

    # Trapezoid over each segment of ds = |B| dt
    return 0.5 * (np.einsum("n,ni,bnj->bij", s, v, history) + np.einsum("n,ni,bnj->bij", s, v_next, history_next))
