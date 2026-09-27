"""Numba O(N) Chambers kernel (private). The only module that imports numba for it.

The same recursion as the NumPy kernel: one exact exponential-integrator step per
segment, w_{n+1} = e^{−z_n} w_n + h_n [v_n φ₂(z_n) + v_{n+1}(φ₁(z_n) − φ₂(z_n))], a
first pass from w_0 = 0 to get S = w_N, the periodic closure w_0 = S/(1 − e^{−Z}),
then a second pass that accumulates σ.

Determinism: each field is computed start to finish by one thread (`prange` over
fields, a serial loop over nodes), and no `fastmath`, so every sum is evaluated in
the same order whatever the thread count and results are bitwise reproducible.
Nothing is allocated inside the field loop: each pass keeps only the running w, and
the second pass recomputes each segment's weights rather than storing them.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

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


@njit(parallel=True, cache=True)
def orbit_sums(s, gamma, vx, vy, field):
    """Σ_n ½ s_n (v_{i,n} w_{j,n} + v_{i,n+1} w_{j,n+1}) for each field: σ without the
    prefactor g_s e³ / 4π²ħ²d.

    The nodes must be ordered along the carriers' motion at these fields.

    Args:
        s: Geometric time of each segment (s·T), float64 of shape (N,).
        gamma: Scattering rate on each segment (s⁻¹), float64 of shape (N,).
        vx: Node velocities v_x (m/s), float64 of shape (N,).
        vy: Node velocities v_y (m/s), float64 of shape (N,).
        field: Field magnitudes |B| > 0 (T), float64 of shape (nB,).

    Returns:
        ndarray of shape (nB, 2, 2).
    """
    n_nodes = s.size
    result = np.zeros((field.size, 2, 2))  # the only allocation, before the field loop
    for f in prange(field.size):
        b = field[f]

        # Pass 1: one orbit from w_0 = 0 gives S = w_N; Z = Σ z_n alongside.
        wx = 0.0
        wy = 0.0
        total = 0.0
        for n in range(n_nodes):
            m = n + 1 if n + 1 < n_nodes else 0
            h = s[n] / b
            z = gamma[n] * h
            p1, p2 = phi(z)
            decay = np.exp(-z)
            start = h * p2
            end = h * (p1 - p2)
            wx = decay * wx + start * vx[n] + end * vx[m]
            wy = decay * wy + start * vy[n] + end * vy[m]
            total += z

        # Closure: the periodic w_0 = S / (1 − e^{−Z}), with expm1 for small Z.
        closure = -np.expm1(-total)
        w0x = wx / closure
        w0y = wy / closure

        # Pass 2: w_n node by node, accumulating the trapezoid over each segment.
        wx = w0x
        wy = w0y
        xx = 0.0
        xy = 0.0
        yx = 0.0
        yy = 0.0
        for n in range(n_nodes):
            m = n + 1 if n + 1 < n_nodes else 0
            h = s[n] / b
            z = gamma[n] * h
            p1, p2 = phi(z)
            decay = np.exp(-z)
            start = h * p2
            end = h * (p1 - p2)
            if m == 0:  # the orbit closes on w_0
                nx = w0x
                ny = w0y
            else:
                nx = decay * wx + start * vx[n] + end * vx[m]
                ny = decay * wy + start * vy[n] + end * vy[m]
            half = 0.5 * s[n]
            xx += half * (vx[n] * wx + vx[m] * nx)
            xy += half * (vx[n] * wy + vx[m] * ny)
            yx += half * (vy[n] * wx + vy[m] * nx)
            yy += half * (vy[n] * wy + vy[m] * ny)
            wx = nx
            wy = ny

        result[f, 0, 0] = xx
        result[f, 0, 1] = xy
        result[f, 1, 0] = yx
        result[f, 1, 1] = yy
    return result
