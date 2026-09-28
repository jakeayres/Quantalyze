"""Numba O(N) Chambers kernel (private). The only module that imports numba for it.

The same model as the NumPy kernel: in the damping coordinate g = ∫ds/τ the mean free
path ℓ = vτ is linear on each segment, and both the history step
w_b = e^{−z} w_a + z [p₁ a + (p₀ − p₁) b] and each segment's ∫ ℓ_i w_j dg are exact.

Each segment's ∫ ℓ_i w_j dg is the part carried in from w_a plus a local part that
does not involve w. The local parts are summed once per field, while the segment
weights are computed, in terms of u = a + b and d = b − a:
Δg z [α u_i u_j + β d_i d_j + γ (u_i d_j − d_i u_j)] with α = (p₀ − p₁)/4,
β = (p₀/3 − p₁ + 2p₃/3)/4 and γ = (p₂ − p₁)/4. On the reversed orbit (what the opposite
field produces) each segment is crossed the other way, which transposes its local
part, so the reversed orbit reuses the sum's transpose.

The carried part runs as a single pass per orientation. Writing w_n = w_n⁽⁰⁾ + P_n w_0,
where w⁽⁰⁾ starts from zero and P_n = Π_{m<n} e^{−z_m}, it is Σ c_n·w_n⁽⁰⁾ + (Σ c_n P_n)·w_0.
The closure w_0 = S/(1 − e^{−Z}), with S = w_N⁽⁰⁾, is known at the end of the pass, so no
second pass is needed. Both orientations share each segment's weights (the reversed
orbit has the same z_n), which is what symmetrisation and its Onsager check need.

The helpers are inlined into the parallel kernels at numba's IR level
(`inline="always"`) rather than left to LLVM. Numba ≥ 0.62 (LLVM ≥ 20) otherwise keeps
them as separate calls, and that code shares a core poorly with its hyperthread twin:
about 25% slower with every hardware thread busy, although no slower on one thread.

Determinism: each field is computed start to finish by one thread (`prange` over
fields, serial loops over nodes), with no `fastmath`, so every sum is evaluated in
the same order whatever the thread count and results are bitwise reproducible.
Nothing is allocated inside the parallel loop: the fields are split into one
contiguous block per thread, and each block keeps its segment weights in its own
row of a scratch array allocated before the loop. How the fields are split affects
only scheduling, never the arithmetic done for a field.
"""
from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from ._kernel_py import _P3_LONG, _P3_SHORT, _THIRD, SERIES_Z, SMALL_Z

# Rows of the per-segment weights: e^{−z} and the history weights z p₁ and z(p₀ − p₁) on
# the start and end nodes. The same two, swapped and times |B| (Δg = |B| z), weight ℓ at
# the start and end nodes against the w carried in.
_ROWS = 3


@njit(cache=True, inline="always")
def _series(coefficients, z):
    """Σ c_j z^j as four interleaved Horner chains in z⁴ (their latencies overlap). The
    coefficients are padded to a multiple of four, and all of them and z are positive."""
    z2 = z * z
    z4 = z2 * z2
    k = coefficients.size - 4
    s0 = coefficients[k]
    s1 = coefficients[k + 1]
    s2 = coefficients[k + 2]
    s3 = coefficients[k + 3]
    for k in range(coefficients.size - 8, -1, -4):
        s0 = s0 * z4 + coefficients[k]
        s1 = s1 * z4 + coefficients[k + 1]
        s2 = s2 * z4 + coefficients[k + 2]
        s3 = s3 * z4 + coefficients[k + 3]
    return (s0 + z * s1) + z2 * (s2 + z * s3)


@njit(cache=True, inline="always")
def moments(z):
    """(e^{−z}, p₀, p₁, p₂, p₃) for one z ≥ 0, with p_k = ∫₀¹ r^k e^{−zr} dr."""
    decay = np.exp(-z)
    if z < SERIES_Z:  # positive series for p₃, then the (all-positive) downward recurrence
        series = _series(_P3_SHORT, z) if z < SMALL_Z else _series(_P3_LONG, z)
        p3 = 6.0 * decay * series
        p2 = (z * p3 + decay) * _THIRD
        p1 = (z * p2 + decay) * 0.5
        return decay, z * p1 + decay, p1, p2, p3
    inverse = 1.0 / z  # the upward recurrence is stable here; e^{−z} → 0 is fine
    p0 = (1.0 - decay) * inverse
    p1 = (p0 - decay) * inverse
    p2 = (2.0 * p1 - decay) * inverse
    return decay, p0, p1, p2, (3.0 * p2 - decay) * inverse


@njit(cache=True, inline="always")
def _segment(g, inverse_field, weights, n):
    """Fill segment n's weights; return z_n and the local-part weights Δg z (α, β, γ)."""
    z = g * inverse_field
    decay, p0, p1, p2, p3 = moments(z)
    weights[0, n] = decay
    weights[1, n] = z * p1
    weights[2, n] = z * (p0 - p1)
    k = 0.25 * g * z
    return z, k * (p0 - p1), k * (p0 * _THIRD - p1 + 2.0 * p3 * _THIRD), k * (p2 - p1)


@njit(cache=True, inline="always")
def _segment_weights(damping, lx, ly, b, weights):
    """Fill every segment's weights at field |B| = b. Returns Z = Σ z_n and the local parts
    Σ Δg z [M_aa (a_i a_j + b_i b_j) + M_ab a_i b_j + M_ba b_i a_j], in the forward order."""
    n_nodes = damping.size
    inverse_field = 1.0 / b
    total = 0.0
    xx = 0.0
    xy = 0.0
    yx = 0.0
    yy = 0.0
    for n in range(n_nodes):
        q = n + 1 if n + 1 < n_nodes else 0
        z, alpha, beta, gamma = _segment(damping[n], inverse_field, weights, n)
        ux, uy = lx[n] + lx[q], ly[n] + ly[q]
        dx, dy = lx[q] - lx[n], ly[q] - ly[n]
        symmetric = alpha * ux * uy + beta * dx * dy
        antisymmetric = gamma * (ux * dy - dx * uy)
        xx += alpha * ux * ux + beta * dx * dx
        xy += symmetric + antisymmetric
        yx += symmetric - antisymmetric
        yy += alpha * uy * uy + beta * dy * dy
        total += z
    return total, xx, xy, yx, yy


@njit(cache=True, inline="always")
def _sweep(weights, lx, ly, total, reverse):
    """The carried part of the orbit sums over one orientation, divided by |B|. Forwards,
    segment n runs from node n to n+1. Reversed, the orbit starts at node 0 and crosses
    segment j = N−1−m from node j+1 to node j at step m."""
    n_nodes = lx.size
    wx = 0.0  # w⁽⁰⁾, the history started from zero
    wy = 0.0
    decay = 1.0  # P_n
    xx = 0.0
    xy = 0.0
    yx = 0.0
    yy = 0.0
    cx = 0.0  # Σ c_n P_n, what w_0 multiplies
    cy = 0.0
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0  # start node
            q = j  # end node
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        ax, ay, bx, by = lx[p], ly[p], lx[q], ly[q]
        ox = weights[2, j] * ax + weights[1, j] * bx  # what w_a is weighted by in ∫ ℓ_i w_j dg, over |B|
        oy = weights[2, j] * ay + weights[1, j] * by
        xx += ox * wx
        xy += ox * wy
        yx += oy * wx
        yy += oy * wy
        cx += ox * decay
        cy += oy * decay
        wx = weights[0, j] * wx + weights[1, j] * ax + weights[2, j] * bx
        wy = weights[0, j] * wy + weights[1, j] * ay + weights[2, j] * by
        decay *= weights[0, j]
    # The closure w_0 = S/(1 − e^{−Z}) sums every earlier orbit (expm1 keeps it accurate
    # when Z is small).
    closure = -np.expm1(-total)
    w0x = wx / closure
    w0y = wy / closure
    return xx + cx * w0x, xy + cx * w0y, yx + cy * w0x, yy + cy * w0y


@njit(parallel=True, cache=True)
def _orbit_sums_blocks(damping, lx, ly, field, n_blocks, both):
    """The parallel kernel: fields in `n_blocks` contiguous blocks, one per thread.

    Returns an array of shape (2, nB, 2, 2): [0] forwards, [1] on the reversed orbit
    (left as zeros unless `both`).
    """
    n_nodes = damping.size
    n_fields = field.size
    result = np.zeros((2, n_fields, 2, 2))
    scratch = np.empty((n_blocks, _ROWS, n_nodes))  # each block's segment weights
    for block in prange(n_blocks):
        weights = scratch[block]
        for f in range(block * n_fields // n_blocks, (block + 1) * n_fields // n_blocks):
            b = field[f]
            total, lxx, lxy, lyx, lyy = _segment_weights(damping, lx, ly, b, weights)
            xx, xy, yx, yy = _sweep(weights, lx, ly, total, False)
            result[0, f, 0, 0] = b * xx + lxx
            result[0, f, 0, 1] = b * xy + lxy
            result[0, f, 1, 0] = b * yx + lyx
            result[0, f, 1, 1] = b * yy + lyy
            if both:  # the reversed orbit crosses every segment the other way: local part transposed
                xx, xy, yx, yy = _sweep(weights, lx, ly, total, True)
                result[1, f, 0, 0] = b * xx + lxx
                result[1, f, 0, 1] = b * xy + lyx
                result[1, f, 1, 0] = b * yx + lxy
                result[1, f, 1, 1] = b * yy + lyy
    return result


def _blocks(n_fields: int) -> int:
    return max(1, min(get_num_threads(), n_fields))


def orbit_sums_both(damping, lx, ly, field):
    """Orbit sums for the nodes' own orientation and for the reversed orbit.

    Args:
        damping: Damping Δg_n = ∫ds/τ of each segment (T), float64 of shape (N,).
        lx: Mean free paths ℓ_x = v_x τ at the nodes (m), float64 of shape (N,).
        ly: Mean free paths ℓ_y (m), float64 of shape (N,).
        field: Field magnitudes |B| > 0 (T), float64 of shape (nB,).

    Returns:
        ndarray of shape (2, nB, 2, 2): [0] with the nodes in the given order of motion,
        [1] on the reversed orbit (node order 0, N−1, …, 1). Each is Σ ∫ ℓ_i w_j dg over
        the segments, σ without g_s e³/4π²ħ²d.
    """
    return _orbit_sums_blocks(damping, lx, ly, field, _blocks(field.size), True)


def orbit_sums(damping, lx, ly, field):
    """Orbit sums with the nodes in the given order of motion: σ without g_s e³/4π²ħ²d.

    Args:
        damping: Damping Δg_n = ∫ds/τ of each segment (T), float64 of shape (N,).
        lx: Mean free paths ℓ_x = v_x τ at the nodes (m), float64 of shape (N,).
        ly: Mean free paths ℓ_y (m), float64 of shape (N,).
        field: Field magnitudes |B| > 0 (T), float64 of shape (nB,).

    Returns:
        ndarray of shape (nB, 2, 2).
    """
    return _orbit_sums_blocks(damping, lx, ly, field, _blocks(field.size), False)[0]


# k_z-warped surfaces: each slice is an in-plane orbit (B ∥ ẑ keeps k_z fixed) that
# carries ℓ_z along with it, so the same recursion runs on three components.

@njit(cache=True, inline="always")
def _segment_weights3(damping, lx, ly, lz, b, weights):
    """As `_segment_weights` with (ℓ_x, ℓ_y, ℓ_z): returns Z and the nine local sums, row by
    row. (They are kept in scalars, not an array, so they stay in registers.)"""
    n_nodes = damping.size
    inverse_field = 1.0 / b
    total = 0.0
    xx = 0.0
    xy = 0.0
    xz = 0.0
    yx = 0.0
    yy = 0.0
    yz = 0.0
    zx = 0.0
    zy = 0.0
    zz = 0.0
    for n in range(n_nodes):
        q = n + 1 if n + 1 < n_nodes else 0
        z, alpha, beta, gamma = _segment(damping[n], inverse_field, weights, n)
        u0, u1, u2 = lx[n] + lx[q], ly[n] + ly[q], lz[n] + lz[q]
        d0, d1, d2 = lx[q] - lx[n], ly[q] - ly[n], lz[q] - lz[n]
        xx += alpha * u0 * u0 + beta * d0 * d0
        yy += alpha * u1 * u1 + beta * d1 * d1
        zz += alpha * u2 * u2 + beta * d2 * d2
        s01, a01 = alpha * u0 * u1 + beta * d0 * d1, gamma * (u0 * d1 - d0 * u1)
        s02, a02 = alpha * u0 * u2 + beta * d0 * d2, gamma * (u0 * d2 - d0 * u2)
        s12, a12 = alpha * u1 * u2 + beta * d1 * d2, gamma * (u1 * d2 - d1 * u2)
        xy += s01 + a01
        yx += s01 - a01
        xz += s02 + a02
        zx += s02 - a02
        yz += s12 + a12
        zy += s12 - a12
        total += z
    return total, xx, xy, xz, yx, yy, yz, zx, zy, zz


@njit(cache=True, inline="always")
def _sweep3(weights, lx, ly, lz, total, reverse):
    """As `_sweep` with (ℓ_x, ℓ_y, ℓ_z): the nine carried sums (over |B|), row by row."""
    n_nodes = lx.size
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    decay = 1.0
    c0 = 0.0
    c1 = 0.0
    c2 = 0.0
    xx = 0.0
    xy = 0.0
    xz = 0.0
    yx = 0.0
    yy = 0.0
    yz = 0.0
    zx = 0.0
    zy = 0.0
    zz = 0.0
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0
            q = j
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        a0, a1, a2 = lx[p], ly[p], lz[p]
        b0, b1, b2 = lx[q], ly[q], lz[q]
        o0 = weights[2, j] * a0 + weights[1, j] * b0
        o1 = weights[2, j] * a1 + weights[1, j] * b1
        o2 = weights[2, j] * a2 + weights[1, j] * b2
        xx += o0 * w0
        xy += o0 * w1
        xz += o0 * w2
        yx += o1 * w0
        yy += o1 * w1
        yz += o1 * w2
        zx += o2 * w0
        zy += o2 * w1
        zz += o2 * w2
        c0 += o0 * decay
        c1 += o1 * decay
        c2 += o2 * decay
        w0 = weights[0, j] * w0 + weights[1, j] * a0 + weights[2, j] * b0
        w1 = weights[0, j] * w1 + weights[1, j] * a1 + weights[2, j] * b1
        w2 = weights[0, j] * w2 + weights[1, j] * a2 + weights[2, j] * b2
        decay *= weights[0, j]
    closure = -np.expm1(-total)
    first0 = w0 / closure
    first1 = w1 / closure
    first2 = w2 / closure
    return (xx + c0 * first0, xy + c0 * first1, xz + c0 * first2,
            yx + c1 * first0, yy + c1 * first1, yz + c1 * first2,
            zx + c2 * first0, zy + c2 * first1, zz + c2 * first2)


@njit(parallel=True, cache=True)
def _orbit_sums_blocks3(damping, lx, ly, lz, field, n_blocks, both):
    """The parallel 3×3 kernel, as `_orbit_sums_blocks`. Shape (2, nB, 3, 3)."""
    n_nodes = damping.size
    n_fields = field.size
    result = np.zeros((2, n_fields, 3, 3))
    scratch = np.empty((n_blocks, _ROWS, n_nodes))
    for block in prange(n_blocks):
        weights = scratch[block]
        for f in range(block * n_fields // n_blocks, (block + 1) * n_fields // n_blocks):
            b = field[f]
            total, lxx, lxy, lxz, lyx, lyy, lyz, lzx, lzy, lzz = _segment_weights3(damping, lx, ly, lz, b, weights)
            xx, xy, xz, yx, yy, yz, zx, zy, zz = _sweep3(weights, lx, ly, lz, total, False)
            out = result[0, f]
            out[0, 0] = b * xx + lxx
            out[0, 1] = b * xy + lxy
            out[0, 2] = b * xz + lxz
            out[1, 0] = b * yx + lyx
            out[1, 1] = b * yy + lyy
            out[1, 2] = b * yz + lyz
            out[2, 0] = b * zx + lzx
            out[2, 1] = b * zy + lzy
            out[2, 2] = b * zz + lzz
            if both:  # the reversed orbit crosses every segment the other way: local part transposed
                xx, xy, xz, yx, yy, yz, zx, zy, zz = _sweep3(weights, lx, ly, lz, total, True)
                out = result[1, f]
                out[0, 0] = b * xx + lxx
                out[0, 1] = b * xy + lyx
                out[0, 2] = b * xz + lzx
                out[1, 0] = b * yx + lxy
                out[1, 1] = b * yy + lyy
                out[1, 2] = b * yz + lzy
                out[2, 0] = b * zx + lxz
                out[2, 1] = b * zy + lyz
                out[2, 2] = b * zz + lzz
    return result


def orbit_sums_both3(damping, lx, ly, lz, field):
    """3×3 orbit sums on the orbit and on the reversed orbit, shape (2, nB, 3, 3)."""
    return _orbit_sums_blocks3(damping, lx, ly, lz, field, _blocks(field.size), True)


def orbit_sums3(damping, lx, ly, lz, field):
    """3×3 orbit sums with the nodes in the given order of motion, shape (nB, 3, 3)."""
    return _orbit_sums_blocks3(damping, lx, ly, lz, field, _blocks(field.size), False)[0]
