"""Numba O(N) Chambers kernel (private). The only module that imports numba for it.

The same recursion as the NumPy kernel: one exact exponential-integrator step per
segment, w_{n+1} = e^{−z_n} w_n + h_n [v_n φ₂(z_n) + v_{n+1}(φ₁(z_n) − φ₂(z_n))], a
first pass from w_0 = 0 to get S = w_N, the periodic closure w_0 = S/(1 − e^{−Z}),
then a second pass that accumulates σ.

The reversed orbit (what the opposite field produces) crosses the same segments in
the opposite direction, so it has the same z_n. `orbit_sums_both` therefore computes
each segment's weights once per field and runs both orientations from them, which
is what symmetrisation needs.

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

from ._kernel_py import _MEDIUM_Z, _PHI1_SMALL, _PHI2_MEDIUM, _PHI2_SMALL, _SMALL_Z


@njit(cache=True)
def _horner(coefficients, z):
    result = coefficients[-1]
    for k in range(coefficients.size - 2, -1, -1):
        result = result * z + coefficients[k]
    return result


@njit(cache=True)
def phi(z):
    """(φ₁(z), φ₂(z)) for one z ≥ 0: φ₁ = (1 − e^{−z})/z, φ₂ = (1 − (1+z)e^{−z})/z²."""
    if z < _SMALL_Z:  # Taylor through z⁵: the closed forms cancel catastrophically here
        return _horner(_PHI1_SMALL, z), _horner(_PHI2_SMALL, z)
    p1 = -np.expm1(-z) / z
    if z < _MEDIUM_Z:  # φ₂'s closed form still loses ~4e-14 here; sum its series instead
        return p1, _horner(_PHI2_MEDIUM, z)
    return p1, (-np.expm1(-z) - z * np.exp(-z)) / z / z  # e^{−z} → 0 is fine; never divide by it


@njit(cache=True)
def _segment_weights(s, gamma, b, decay, start, end):
    """Fill e^{−z_n}, h_n φ₂(z_n) and h_n(φ₁ − φ₂) for every segment at field |B| = b;
    return Z = Σ z_n. One expm1 per segment gives e^{−z}, φ₁ and (for z ≥ 1) φ₂."""
    total = 0.0
    for n in range(s.size):
        h = s[n] / b
        z = gamma[n] * h
        e = np.expm1(-z)  # e^{−z} − 1, accurate for small z
        if z < _SMALL_Z:
            p1 = _horner(_PHI1_SMALL, z)
            p2 = _horner(_PHI2_SMALL, z)
        else:
            p1 = -e / z
            if z < _MEDIUM_Z:
                p2 = _horner(_PHI2_MEDIUM, z)
            else:
                p2 = (-e - z * (1.0 + e)) / z / z
        decay[n] = 1.0 + e
        start[n] = h * p2
        end[n] = h * (p1 - p2)
        total += z
    return total


@njit(cache=True)
def _sweep(decay, start, end, s, vx, vy, total, reverse):
    """σ sums over one orbit, given the segment weights. Forwards, segment n runs from
    node n to n+1. Reversed, the orbit starts at node 0 and crosses segment j = N−1−m
    from node j+1 to node j at step m."""
    n_nodes = s.size

    # Pass 1: one orbit from w = 0 gives S; the closure w_0 = S/(1 − e^{−Z}) sums every
    # earlier orbit (expm1 keeps it accurate when Z is small).
    wx = 0.0
    wy = 0.0
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0  # start node
            q = j  # end node
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        wx = decay[j] * wx + start[j] * vx[p] + end[j] * vx[q]
        wy = decay[j] * wy + start[j] * vy[p] + end[j] * vy[q]
    closure = -np.expm1(-total)
    w0x = wx / closure
    w0y = wy / closure

    # Pass 2: w node by node, accumulating the trapezoid over each segment.
    wx = w0x
    wy = w0y
    xx = 0.0
    xy = 0.0
    yx = 0.0
    yy = 0.0
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0
            q = j
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        if m == n_nodes - 1:  # the orbit closes on w_0
            nx = w0x
            ny = w0y
        else:
            nx = decay[j] * wx + start[j] * vx[p] + end[j] * vx[q]
            ny = decay[j] * wy + start[j] * vy[p] + end[j] * vy[q]
        half = 0.5 * s[j]
        xx += half * (vx[p] * wx + vx[q] * nx)
        xy += half * (vx[p] * wy + vx[q] * ny)
        yx += half * (vy[p] * wx + vy[q] * nx)
        yy += half * (vy[p] * wy + vy[q] * ny)
        wx = nx
        wy = ny
    return xx, xy, yx, yy


@njit(parallel=True, cache=True)
def _orbit_sums_blocks(s, gamma, vx, vy, field, n_blocks, both):
    """The parallel kernel: fields in `n_blocks` contiguous blocks, one per thread.

    Returns an array of shape (2, nB, 2, 2): [0] forwards, [1] on the reversed orbit
    (left as zeros unless `both`).
    """
    n_nodes = s.size
    n_fields = field.size
    result = np.zeros((2, n_fields, 2, 2))
    scratch = np.empty((n_blocks, 3, n_nodes))  # each block's segment weights
    for block in prange(n_blocks):
        decay = scratch[block, 0]
        start = scratch[block, 1]
        end = scratch[block, 2]
        for f in range(block * n_fields // n_blocks, (block + 1) * n_fields // n_blocks):
            total = _segment_weights(s, gamma, field[f], decay, start, end)
            for orientation in range(2 if both else 1):
                xx, xy, yx, yy = _sweep(decay, start, end, s, vx, vy, total, orientation == 1)
                result[orientation, f, 0, 0] = xx
                result[orientation, f, 0, 1] = xy
                result[orientation, f, 1, 0] = yx
                result[orientation, f, 1, 1] = yy
    return result


def _blocks(n_fields: int) -> int:
    return max(1, min(get_num_threads(), n_fields))


def orbit_sums_both(s, gamma, vx, vy, field):
    """Orbit sums for the nodes' own orientation and for the reversed orbit.

    Args:
        s: Geometric time of each segment (s·T), float64 of shape (N,).
        gamma: Scattering rate on each segment (s⁻¹), float64 of shape (N,).
        vx: Node velocities v_x (m/s), float64 of shape (N,).
        vy: Node velocities v_y (m/s), float64 of shape (N,).
        field: Field magnitudes |B| > 0 (T), float64 of shape (nB,).

    Returns:
        ndarray of shape (2, nB, 2, 2): [0] with the nodes in the given order of motion,
        [1] on the reversed orbit (node order 0, N−1, …, 1). Each is
        Σ_n ½ s_n (v_{i,n} w_{j,n} + v_{i,n+1} w_{j,n+1}), σ without g_s e³/4π²ħ²d.
    """
    return _orbit_sums_blocks(s, gamma, vx, vy, field, _blocks(field.size), True)


def orbit_sums(s, gamma, vx, vy, field):
    """Orbit sums with the nodes in the given order of motion: σ without g_s e³/4π²ħ²d.

    Args:
        s: Geometric time of each segment (s·T), float64 of shape (N,).
        gamma: Scattering rate on each segment (s⁻¹), float64 of shape (N,).
        vx: Node velocities v_x (m/s), float64 of shape (N,).
        vy: Node velocities v_y (m/s), float64 of shape (N,).
        field: Field magnitudes |B| > 0 (T), float64 of shape (nB,).

    Returns:
        ndarray of shape (nB, 2, 2).
    """
    return _orbit_sums_blocks(s, gamma, vx, vy, field, _blocks(field.size), False)[0]


# k_z-warped surfaces: each slice is an in-plane orbit (B ∥ ẑ keeps k_z fixed) that
# carries v_z along with it, so the same recursion runs on three velocity components.

@njit(cache=True)
def _sweep3(decay, start, end, s, vx, vy, vz, total, reverse, out):
    """As `_sweep` with (v_x, v_y, v_z): writes the 3×3 orbit sums into `out`."""
    n_nodes = s.size
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0
            q = j
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        w0 = decay[j] * w0 + start[j] * vx[p] + end[j] * vx[q]
        w1 = decay[j] * w1 + start[j] * vy[p] + end[j] * vy[q]
        w2 = decay[j] * w2 + start[j] * vz[p] + end[j] * vz[q]
    closure = -np.expm1(-total)
    first0 = w0 / closure
    first1 = w1 / closure
    first2 = w2 / closure

    for i in range(3):
        for k in range(3):
            out[i, k] = 0.0
    w0 = first0
    w1 = first1
    w2 = first2
    for m in range(n_nodes):
        if reverse:
            j = n_nodes - 1 - m
            p = j + 1 if j + 1 < n_nodes else 0
            q = j
        else:
            j = m
            p = m
            q = m + 1 if m + 1 < n_nodes else 0
        if m == n_nodes - 1:  # the orbit closes on w_0
            n0 = first0
            n1 = first1
            n2 = first2
        else:
            n0 = decay[j] * w0 + start[j] * vx[p] + end[j] * vx[q]
            n1 = decay[j] * w1 + start[j] * vy[p] + end[j] * vy[q]
            n2 = decay[j] * w2 + start[j] * vz[p] + end[j] * vz[q]
        half = 0.5 * s[j]
        vp0, vp1, vp2 = vx[p], vy[p], vz[p]
        vq0, vq1, vq2 = vx[q], vy[q], vz[q]
        out[0, 0] += half * (vp0 * w0 + vq0 * n0)
        out[0, 1] += half * (vp0 * w1 + vq0 * n1)
        out[0, 2] += half * (vp0 * w2 + vq0 * n2)
        out[1, 0] += half * (vp1 * w0 + vq1 * n0)
        out[1, 1] += half * (vp1 * w1 + vq1 * n1)
        out[1, 2] += half * (vp1 * w2 + vq1 * n2)
        out[2, 0] += half * (vp2 * w0 + vq2 * n0)
        out[2, 1] += half * (vp2 * w1 + vq2 * n1)
        out[2, 2] += half * (vp2 * w2 + vq2 * n2)
        w0 = n0
        w1 = n1
        w2 = n2


@njit(parallel=True, cache=True)
def _orbit_sums_blocks3(s, gamma, vx, vy, vz, field, n_blocks, both):
    """The parallel 3×3 kernel, as `_orbit_sums_blocks`. Shape (2, nB, 3, 3)."""
    n_nodes = s.size
    n_fields = field.size
    result = np.zeros((2, n_fields, 3, 3))
    scratch = np.empty((n_blocks, 3, n_nodes))
    for block in prange(n_blocks):
        decay = scratch[block, 0]
        start = scratch[block, 1]
        end = scratch[block, 2]
        for f in range(block * n_fields // n_blocks, (block + 1) * n_fields // n_blocks):
            total = _segment_weights(s, gamma, field[f], decay, start, end)
            for orientation in range(2 if both else 1):
                _sweep3(decay, start, end, s, vx, vy, vz, total, orientation == 1, result[orientation, f])
    return result


def orbit_sums_both3(s, gamma, vx, vy, vz, field):
    """3×3 orbit sums on the orbit and on the reversed orbit, shape (2, nB, 3, 3)."""
    return _orbit_sums_blocks3(s, gamma, vx, vy, vz, field, _blocks(field.size), True)


def orbit_sums3(s, gamma, vx, vy, vz, field):
    """3×3 orbit sums with the nodes in the given order of motion, shape (nB, 3, 3)."""
    return _orbit_sums_blocks3(s, gamma, vx, vy, vz, field, _blocks(field.size), False)[0]
