"""Plain-NumPy O(N) Chambers kernel (private). The oracle for the numba kernel.

The kernel works in the damping coordinate g = ∫ds/τ (T), the damping a carrier
accumulates along its orbit, with the mean free path ℓ = vτ (m). There the history
w(g) = (1/|B|) ∫_{−∞}^{g} ℓ(g′) e^{−(g−g′)/|B|} dg′ obeys dw/dg = (ℓ − w)/|B|, and
σ_ij = (g_s e³/4π²ħ²d) ∮ ℓ_i w_j dg: the damping is uniform in g, and all the
anisotropy of τ is carried by ℓ and by the spacing of the nodes in g.

The contour is modelled as ℓ linear in g on each segment, and both the history and
the outer integral are evaluated exactly for that model. On a segment of damping Δg,
with z = Δg/|B|, ℓ = a at its start and b at its end (in the order of motion):

    w_b = e^{−z} w_a + z [p₁ a + (p₀ − p₁) b],
    ∫ ℓ_i w_j dg = Δg [(p₀ − p₁) a_i + p₁ b_i] w_{a,j}
                   + Δg z [M_aa (a_i a_j + b_i b_j) + M_ab a_i b_j + M_ba b_i a_j],

with the moments p_k(z) = ∫₀¹ r^k e^{−zr} dr and the overlaps M built from them. The
discrete σ is therefore exactly the Chambers conductivity of a continuous model, so it
keeps its physics: Onsager symmetry holds to rounding, the symmetric part is positive
definite, and it departs from σ(0) as B². (A trapezoid rule for the outer integral
would add a spurious term linear in |B| once the history decays within a segment.)

Round a closed orbit w must return to itself, so starting from w_0 = 0 gives w_N = S,
and the periodic solution is w_0 = S / (1 − e^{−Z}), Z = Σ z_n. That sums the
contribution of every earlier orbit exactly, with no truncation in ω_cτ.
"""
from __future__ import annotations

import math

import numpy as np

# p₃ comes from a positive series below SERIES_Z (short below SMALL_Z), and p₂, p₁, p₀ from
# the downward recurrence p_{k−1} = (z p_k + e^{−z})/k, whose terms are all positive. At
# and above SERIES_Z the upward recurrence p_k = (k p_{k−1} − e^{−z})/z is stable instead.
SMALL_Z = 0.1
SERIES_Z = 2.0

_THIRD = 1.0 / 3.0
_SIXTH = 1.0 / 6.0


def _p3_coefficients(terms: int) -> np.ndarray:
    """1/(j+4)! for j < terms, padded with zeros to a multiple of four (see `_series`)."""
    coefficients = np.zeros(-(-terms // 4) * 4)
    coefficients[:terms] = [1.0 / math.factorial(j + 4) for j in range(terms)]
    return coefficients


# p₃(z) = 6 e^{−z} Σ_j z^j/(j+4)!: the first omitted term is below 4e-19 of the sum.
_P3_SHORT = _p3_coefficients(9)  # z < 0.1
_P3_LONG = _p3_coefficients(21)  # z < 2


def _horner(coefficients: np.ndarray, z: np.ndarray) -> np.ndarray:
    result = np.full(z.shape, coefficients[-1])
    for c in coefficients[-2::-1]:
        result = result * z + c
    return result


def _series(coefficients: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Σ c_j z^j as four interleaved Horner chains in z⁴, whose latencies overlap in the
    compiled kernel. Every coefficient and z are positive, so nothing cancels."""
    z2 = z * z
    z4 = z2 * z2
    chains = [_horner(coefficients[r::4], z4) for r in range(4)]
    return (chains[0] + z * chains[1]) + z2 * (chains[2] + z * chains[3])


def moments(z):
    """e^{−z} and the moments p_k(z) = ∫₀¹ r^k e^{−zr} dr, k = 0…3, for z ≥ 0.

    p₀ = (1 − e^{−z})/z and p₁ = (1 − (1 + z)e^{−z})/z². Each is accurate to rounding for
    every z ≥ 0 (their closed forms cancel catastrophically for small z).

    Returns:
        (e^{−z}, p₀, p₁, p₂, p₃), arrays of the shape of `z`.
    """
    z = np.asarray(z, dtype=np.float64)
    decay = np.exp(-z)  # underflows to 0 for large z, which is correct: never divide by it
    p = np.empty((4,) + z.shape)
    series = z < SERIES_Z
    zs, es = z[series], decay[series]
    p3 = 6.0 * es * np.where(zs < SMALL_Z, _series(_P3_SHORT, zs), _series(_P3_LONG, zs))
    p2 = (zs * p3 + es) * _THIRD
    p1 = (zs * p2 + es) * 0.5
    p[0][series], p[1][series], p[2][series], p[3][series] = zs * p1 + es, p1, p2, p3
    zl, el = z[~series], decay[~series]
    inverse = 1.0 / zl
    p0 = (1.0 - el) * inverse
    p1 = (p0 - el) * inverse
    p2 = (2.0 * p1 - el) * inverse
    p[0][~series], p[1][~series], p[2][~series], p[3][~series] = p0, p1, p2, (3.0 * p2 - el) * inverse
    return decay, p[0], p[1], p[2], p[3]


def overlaps(p0, p1, p2, p3):
    """The double integrals ∫₀¹du f(u) ∫₀^u e^{−z(u−u′)} g(u′) du′ for linear f, g.

    With f and g each 1 − u (the start node's weight, a) or u (the end node's, b):
    M_aa = M_bb = p₀/3 − p₁/2 + p₃/6, M_ab (f = 1 − u, g = u′) = p₀/6 − p₁/2 + p₂/2 − p₃/6
    and M_ba (f = u, g = 1 − u′) = p₀/6 + p₁/2 − p₂/2 − p₃/6. At z = 0 they are 1/8, 1/24
    and 5/24; for large z, 1/(3z), 1/(6z) and 1/(6z).

    Returns:
        (M_aa, M_ab, M_ba).
    """
    return (p0 * _THIRD - p1 * 0.5 + p3 * _SIXTH,
            p0 * _SIXTH - p1 * 0.5 + p2 * 0.5 - p3 * _SIXTH,
            p0 * _SIXTH + p1 * 0.5 - p2 * 0.5 - p3 * _SIXTH)


def orbit_sums(damping, lx, ly, field, lz=None) -> np.ndarray:
    """Σ over segments of ∫ ℓ_i w_j dg for each field (T·m², i.e. σ without the prefactor
    g_s e³ / 4π²ħ²d).

    The nodes must be ordered along the carriers' motion at these fields.

    Args:
        damping: Damping Δg_n = ∫ds/τ of each segment (T), shape (N,).
        lx: Mean free paths ℓ_x = v_x τ at the nodes (m), shape (N,).
        ly: Mean free paths ℓ_y (m), shape (N,).
        field: Field magnitudes |B| > 0 (T), shape (nB,).
        lz: Mean free paths ℓ_z = v_z τ (m), shape (N,), for a k_z slice of a warped
            surface; the orbit stays in its k_z plane (B ∥ ẑ) and ℓ_z is carried along it.

    Returns:
        ndarray of shape (nB, 2, 2), or (nB, 3, 3) with lz.
    """
    g = np.asarray(damping, dtype=np.float64)
    b = np.asarray(field, dtype=np.float64)
    n_nodes = g.size
    components = [lx, ly] if lz is None else [lx, ly, lz]
    ell = np.stack(components, axis=1).astype(np.float64)  # a, the start of each segment, (N, d)
    ell_next = np.roll(ell, -1, axis=0)  # b, its end, (N, d)

    z = g[None, :] * (1.0 / b)[:, None]  # (nB, N)
    decay, p0, p1, p2, p3 = moments(z)
    aa, ab, ba = overlaps(p0, p1, p2, p3)
    step_start, step_end = z * p1, z * (p0 - p1)  # history weights on a and b, (nB, N)
    total = np.sum(z, axis=1)  # Z, (nB,)

    # Pass 1: one orbit from w_0 = 0 gives S.
    w = np.zeros((b.size, ell.shape[1]))  # (nB, d)
    for n in range(n_nodes):
        w = decay[:, n, None] * w + step_start[:, n, None] * ell[n] + step_end[:, n, None] * ell_next[n]

    # Closure: the periodic w_0 = S / (1 − e^{−Z}); expm1 keeps it accurate when Z is small.
    history = np.empty((b.size, n_nodes, ell.shape[1]))  # w at the start of each segment, (nB, N, d)
    history[:, 0] = w / (-np.expm1(-total))[:, None]

    # Pass 2: w at every node, forwards again (dividing by e^{−z} would overflow).
    for n in range(n_nodes - 1):
        history[:, n + 1] = (decay[:, n, None] * history[:, n] + step_start[:, n, None] * ell[n]
                             + step_end[:, n, None] * ell_next[n])

    # Each segment's ∫ ℓ_i w_j dg, exactly: the part carried in from w_a, then the part
    # built up along the segment itself.
    carried = (np.einsum("bn,ni,bnj->bij", g * (p0 - p1), ell, history)
               + np.einsum("bn,ni,bnj->bij", g * p1, ell_next, history))
    k = g * z  # Δg z, (nB, N)
    local = (np.einsum("bn,ni,nj->bij", k * aa, ell, ell) + np.einsum("bn,ni,nj->bij", k * aa, ell_next, ell_next)
             + np.einsum("bn,ni,nj->bij", k * ab, ell, ell_next) + np.einsum("bn,ni,nj->bij", k * ba, ell_next, ell))
    return carried + local
