"""Conductivity of a Fermi contour in a magnetic field (private; re-exported from bz).

`conductivity_tensor` is the array layer: it validates and orients the contour,
runs a kernel for each sign of B, symmetrises, and applies the prefactor
g_s e³ / (4π²ħ² d). Carriers on the reversed orbit are what a field of the
opposite sign produces, so σ(−B) is the same kernel on the reversed node order.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, HBAR
from . import _kernel, _kernel_py
from ._contour import prepare_contour

_BACKENDS = ("python", "numba")


def _orbit_sums(backend, s, gamma, vx, vy, field):
    if backend == "python":
        return _kernel_py.orbit_sums(s, gamma, vx, vy, field)
    # Contiguous float64 so the compiled kernel sees one array type (the reversed orbit is a view).
    arrays = (np.ascontiguousarray(a, dtype=np.float64) for a in (s, gamma, vx, vy, field))
    return _kernel.orbit_sums(*arrays)


def _zero_field_sums(s, gamma, vx, vy):
    """Σ_n v_n v_nᵀ ½(s_{n−1} + s_n)/γ_{n−1}: the recursion's exact limit as B → 0.

    As B → 0 the history at node n becomes v_n/γ_{n−1}, the rate on the segment the
    carrier has just crossed (nodes in order of motion). Using the same discrete sum as
    the field branch makes σ(B) join σ(0) with no jump. It is
    σ_ij(0) = (g_s e²/4π²ħd) ∮ |dk| τ v_i v_j/|v| discretised, since s = ħ|dk|/(e|v|).
    """
    weight = 0.5 * (np.roll(s, 1) + s) / np.roll(gamma, 1)  # (N,)
    v = np.stack([vx, vy], axis=1)  # (N, 2)
    return np.einsum("n,ni,nj->ij", weight, v, v)


def conductivity_tensor(
    kx,
    ky,
    vx,
    vy,
    tau,
    field,
    *,
    layer_spacing: float,
    charge: float = -ELEMENTARY_CHARGE,
    spin_degeneracy: float = 2,
    symmetrize: bool = True,
    remove_drift: bool = True,
    period: Optional[Sequence[float]] = None,
    backend: str = "numba",
) -> np.ndarray:
    """Conductivity tensor of one Fermi contour at each field (S/m).

    Args:
        kx: Node wavevectors k_x (m⁻¹), shape (N,).
        ky: Node wavevectors k_y (m⁻¹), shape (N,).
        vx: Group velocities v_x = (1/ħ) ∂ε/∂k_x (m/s), shape (N,).
        vy: Group velocities v_y (m/s), shape (N,).
        tau: Relaxation times (s), shape (N,) or a float.
        field: Magnetic field B along ẑ (T), a float or shape (nB,). Any sign, including 0.
        layer_spacing: Interlayer spacing d (m). Required: body-centred cells have
            d = c/2, not c.
        charge: Carrier charge q (C), ±e.
        spin_degeneracy: Spin degeneracy g_s.
        symmetrize: Replace σ(B) by ½[σ(B) + σ(−B)ᵀ], which removes the discretisation's
            small breaking of Onsager symmetry (and a spurious 1/B term in low-field R_H).
        remove_drift: Remove the discretisation drift of a closed orbit's ∮ v dt.
        period: Reciprocal-lattice vector for open orbits. Not supported yet.
        backend: "numba" (compiled, parallel over fields) or "python" (plain NumPy, the
            slower reference implementation the numba kernel is tested against).

    Returns:
        ndarray of shape (nB, 2, 2), σ with index order [[xx, xy], [yx, yy]] (S/m).

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta.boltzmann._response import conductivity_tensor
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> sigma = conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"],
        ...                             [0.0, 10.0], layer_spacing=1e-9)  # (2, 2, 2)
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, not {backend!r}")
    contour = prepare_contour(kx, ky, vx, vy, tau, charge=charge, remove_drift=remove_drift, period=period)
    fields = np.atleast_1d(np.asarray(field, dtype=np.float64))  # (nB,)
    if fields.ndim != 1:
        raise ValueError(f"field must be a float or 1-D, not of shape {fields.shape}")
    if not np.all(np.isfinite(fields)):
        raise ValueError("field must be finite")

    # The prepared order is the motion for B > 0; B < 0 runs the orbit backwards.
    forward = (contour.s, contour.gamma, contour.vx, contour.vy)
    order = np.roll(np.arange(contour.s.size)[::-1], 1)  # node 0, N−1, …, 1
    backward = (contour.s[::-1], contour.gamma[::-1], contour.vx[order], contour.vy[order])

    magnitude = np.abs(fields)
    nonzero = magnitude > 0
    unique = np.unique(magnitude[nonzero])  # (nU,)
    need_forward = symmetrize or np.any(fields > 0)
    need_backward = symmetrize or np.any(fields < 0)
    empty = np.empty((0, 2, 2))
    sums_forward = _orbit_sums(backend, *forward, unique) if unique.size and need_forward else empty
    sums_backward = _orbit_sums(backend, *backward, unique) if unique.size and need_backward else empty

    result = np.empty((fields.size, 2, 2))
    for index, b in enumerate(fields):
        if b == 0:
            along = _zero_field_sums(*forward)
            against = _zero_field_sums(*backward) if symmetrize else None
        else:
            u = np.searchsorted(unique, abs(b))
            along = sums_forward[u] if b > 0 else sums_backward[u]
            against = (sums_backward[u] if b > 0 else sums_forward[u]) if symmetrize else None
        result[index] = along if against is None else 0.5 * (along + against.T)

    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    return prefactor * result
