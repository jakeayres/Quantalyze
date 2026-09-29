"""Magnetotransport with a scattering kernel: the Boltzmann equation beyond the relaxation time (private).

A kernel P(k, k′) (J·m³/s) is the rate at which a carrier at k scatters into states
near k′, per unit density of states per spin, per unit volume. The vector mean free path
L then obeys, along each orbit (ħk̇ = q v × B),

    dL/dt + L/τ = v + ∫ dμ(k′) P(k, k′) L(k′),     1/τ = 1/τ₀ + ∫ dμ(k′) P(k, k′),

with dμ the density of states per spin per volume on the Fermi surface and τ₀ a
background relaxation time (the `tau` column), and σ_ij = g_s e² ∫ dμ v_i L_j. The first
term on the right is the relaxation-time source; the integral is the in-scattering
(the current vertex correction) that a relaxation time leaves out.

In the damping coordinate g = ∫ds/τ of the relaxation-time kernel, with the total τ and
ℓ = vτ, the equation reads |B| dw/dg + w = ℓ + τI[w]. The relaxation-time kernel is
exact for sources that are piecewise linear in g, so it defines, per contour, the
matrix G_nm = ⟨φ_n, R φ_m⟩ of the resolvent R = (|B| d/dg + 1)⁻¹ between hat functions
φ_n. Projecting the equation onto the same hat functions (a Galerkin method) gives

    c = ℓ + Ĵ G_Ω c,     σ = (g_s e³/4π²ħ²d) ℓᵀ G_Ω c,

where G_Ω is block-diagonal over contours (each weighted by ω, 1/N_z for a k_z slice)
and Ĵ_pq = κ τ̂_p P_pq τ̂_q + δ_pq Γ_p(τ_p − τ̂_p)/(ω_p m_p) is the in-scattering on the
nodes. Here κ = e/(4π²ħ²d), m_p = ½(Δg_{p−1} + Δg_p), τ̂_p = s̄_p/m_p and Γ_p = Σ_q P_pq μ_q
with μ_p = κ ω_p s̄_p the node's density of states. Because Ĵ is symmetric and
G(−B) = G(B)ᵀ, σ(−B) = σ(B)ᵀ exactly. The diagonal term makes the kernel conserve
particles exactly on the nodes: the constant is then a zero mode wherever there is no
background, and it is handled by a bordered solve. With P = 0 this is exactly the
relaxation-time result.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ...core.constants import ELEMENTARY_CHARGE, HBAR
from ._contour import MIN_NODES, ContourGeometry, PreparedContour, contour_geometry, finish_contour
from ._kernel_py import moments, overlaps

# Dense matrices of a component above this many nodes would need gigabytes.
MAX_COMPONENT_NODES = 8192
# A component's net drift below this fraction of its scale is discretisation.
_DRIFT_TOLERANCE = 1e-2
# The out-scattering rate from every other node must agree with the full one to this.
_RESOLUTION_TOLERANCE = 1e-2
# σ(−B) and σ(B)ᵀ agree to rounding; so do G on the reversed orbit and Gᵀ.
_ONSAGER_TOLERANCE = 1e-9
# P(k, k′) and P(k′, k) must agree to this fraction of max P.
_SYMMETRY_TOLERANCE = 1e-10
# Evaluate the kernel in blocks of at most this many elements.
_BLOCK_ELEMENTS = 1 << 22


def propagator(damping, field) -> np.ndarray:
    """The propagator matrix G_nm = ⟨φ_n, R φ_m⟩ of one contour at a field |B| > 0 (T).

    Row n is the outer node and column m the source node, so the relaxation-time orbit
    sum is ℓ_iᵀ G ℓ_j. The history at every node for a unit source at every node is built
    by the kernel's own two-pass recurrence, run for all sources at once: never from
    exponentials of differences of cumulative damping, which lose digits at low field.

    Args:
        damping: Damping Δg_n of each segment, in the order of motion (T), shape (N,).
        field: Field magnitude |B| > 0 (T).

    Returns:
        ndarray of shape (N, N) (T).
    """
    damping = np.asarray(damping, dtype=np.float64)
    n_nodes = damping.size
    z = damping / field  # (N,)
    decay, p0, p1, p2, p3 = moments(z)
    m_aa, m_ab, m_ba = overlaps(p0, p1, p2, p3)
    start, end = z * p1, z * (p0 - p1)  # history weights on a segment's start and end source values
    history = np.empty((n_nodes, n_nodes))  # rows: node; columns: unit source
    row = np.zeros(n_nodes)
    for n in range(n_nodes):  # one orbit from w_0 = 0 gives S
        row *= decay[n]
        row[n] += start[n]
        row[(n + 1) % n_nodes] += end[n]
    history[0] = row / (-np.expm1(-z.sum()))  # the periodic closure sums every earlier orbit
    for n in range(n_nodes - 1):  # forwards again: dividing by e^{−z} would overflow
        history[n + 1] = decay[n] * history[n]
        history[n + 1, n] += start[n]
        history[n + 1, n + 1] += end[n]
    # Segment n's outer integral: the part carried in from w at its start node, then the
    # part built up along the segment itself.
    result = (damping * (p0 - p1))[:, None] * history
    result += np.roll((damping * p1)[:, None] * history, 1, axis=0)
    local = damping * z
    first, second = np.arange(n_nodes), (np.arange(n_nodes) + 1) % n_nodes
    np.add.at(result, (first, first), local * m_aa)
    np.add.at(result, (second, second), local * m_aa)
    np.add.at(result, (first, second), local * m_ab)  # outer on the start node, source on the end
    np.add.at(result, (second, first), local * m_ba)
    return result


def mass_matrix(damping) -> np.ndarray:
    """G at B = 0: the consistent mass matrix ⟨φ_n, φ_m⟩ of the hat functions in g (T)."""
    damping = np.asarray(damping, dtype=np.float64)
    n_nodes = damping.size
    first, second = np.arange(n_nodes), (np.arange(n_nodes) + 1) % n_nodes
    result = np.zeros((n_nodes, n_nodes))
    np.add.at(result, (first, first), damping / 3)
    np.add.at(result, (second, second), damping / 3)
    np.add.at(result, (first, second), damping / 6)
    np.add.at(result, (second, first), damping / 6)
    return result


def _reversed_order(n_nodes: int) -> np.ndarray:
    """Node order of the reversed orbit: 0, N−1, …, 1 (as the relaxation-time kernels use)."""
    return np.roll(np.arange(n_nodes)[::-1], 1)


@dataclass(frozen=True)
class ContourInput:
    """One contour as given: arrays in the input order.

    Attributes:
        kx, ky, vx, vy: Nodes (m⁻¹) and group velocities (m/s), shape (N,).
        tau0: Background relaxation time (s), shape (N,); np.inf for none.
        period: (G_x, G_y) of an open orbit, or None.
        vz: v_z (m/s) of a k_z slice, or None.
        kz: The slice's k_z (m⁻¹), or None.
        weight: 1/N_z for one of N_z slices, else 1.
    """

    kx: np.ndarray
    ky: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    tau0: np.ndarray
    period: Optional[tuple] = None
    vz: Optional[np.ndarray] = None
    kz: Optional[float] = None
    weight: float = 1.0

    def every_other(self, n_distinct: int) -> "ContourInput":
        rows = slice(0, n_distinct, 2)
        return ContourInput(self.kx[rows], self.ky[rows], self.vx[rows], self.vy[rows], self.tau0[rows],
                            self.period, None if self.vz is None else self.vz[rows], self.kz, self.weight)


def _located(nodes, node: int) -> str:
    contour, row = nodes
    return f"node {row[node]} of contour {contour[node]}"


def _kernel_matrix(kernel: Callable, kx, ky, kz, where) -> np.ndarray:
    """P_pq = P(k_p, k_q) on all nodes, checked and made exactly symmetric (J·m³/s)."""
    size = kx.size
    result = np.empty((size, size))
    rows = max(1, _BLOCK_ELEMENTS // size)
    for start in range(0, size, rows):
        stop = min(size, start + rows)
        first = (kx[start:stop, None], ky[start:stop, None]) + (() if kz is None else (kz[start:stop, None],))
        second = (kx[None, :], ky[None, :]) + (() if kz is None else (kz[None, :],))
        value = np.asarray(kernel(*first, *second), dtype=np.float64)
        try:
            result[start:stop] = np.broadcast_to(value, (stop - start, size))
        except ValueError:
            raise ValueError("the scattering kernel must return an array that broadcasts against its arguments "
                             f"(one row of k against every k′); it returned shape {value.shape}") from None
    bad = np.argwhere(~np.isfinite(result))
    if bad.size:
        p, q = bad[0]
        raise ValueError(f"the scattering kernel is {result[p, q]} for k at {_located(where, p)} and k′ at "
                         f"{_located(where, q)}; it must be finite everywhere, including k = k′")
    bad = np.argwhere(result < 0)
    if bad.size:
        p, q = bad[0]
        raise ValueError(f"the scattering kernel is negative ({result[p, q]:.3e}) for k at {_located(where, p)} and "
                         f"k′ at {_located(where, q)}; scattering rates cannot be negative")
    scale = np.max(result)
    asymmetry = np.max(np.abs(result - result.T))
    if asymmetry > _SYMMETRY_TOLERANCE * scale:
        raise ValueError(
            f"the scattering kernel is not symmetric: P(k, k′) and P(k′, k) differ by up to {asymmetry / scale:.1e} "
            "of its maximum. Detailed balance needs P(k, k′) = P(k′, k); for a kernel peaked at a momentum "
            "transfer Q, include both +Q and −Q, for example F(k − k′ − Q) + F(k − k′ + Q)")
    result += result.T
    result *= 0.5
    return result


def _check_resolution(kernel, mu, gamma, edges, where) -> None:
    """Warn if the out-scattering rate changes when every other node of a contour is used."""
    worst, worst_at = 0.0, None
    active = gamma > 0
    for start, stop in zip(edges[:-1], edges[1:]):
        weights = mu[start:stop]
        half = np.zeros_like(weights)
        half[::2] = weights[::2] * (weights.sum() / weights[::2].sum())
        block = kernel[:, start:stop]
        change = np.zeros_like(gamma)
        change[active] = np.abs(block[active] @ (half - weights)) / gamma[active]
        node = int(np.argmax(change))
        if change[node] > worst:
            worst, worst_at = change[node], (node, start)
    if worst > _RESOLUTION_TOLERANCE:
        node, start = worst_at
        warnings.warn(
            f"the scattering kernel changes too fast between neighbouring nodes: the out-scattering rate at "
            f"{_located(where, node)} changes by {worst:.1e} when every other node of contour "
            f"{where[0][start]} is used. It is under-resolved (or, on an open sheet, not periodic along it), so the "
            "scattering rates and the in-scattering can be off by several per cent; use more nodes (or a smoother "
            "kernel) and compare with twice as many",
            UserWarning,
            stacklevel=4,
        )


@dataclass
class _Component:
    """A set of contours coupled by the kernel, with everything the solve needs."""

    contours: List[int]
    nodes: np.ndarray  # global node indices, (n,)
    ell: np.ndarray  # (n, d), drift-corrected where discretisation
    jmat: np.ndarray  # (n, n)
    border_u: np.ndarray  # (n,)
    border_v: np.ndarray  # (n,)
    drift: np.ndarray  # (d,), what is left after the drift rule
    background: float  # max of b on the component; 0 when pure-conserving


@dataclass
class CollisionSystem:
    """The prepared contours, the kernel on their nodes, and the components to solve."""

    prepared: List[PreparedContour]
    weights: np.ndarray  # (C,)
    edges: np.ndarray  # (C + 1,)
    gamma: np.ndarray  # (N_tot,), the kernel's out-scattering rate (s⁻¹)
    components: List[_Component]
    free: List[int]  # contours with no kernel: plain relaxation time
    dimension: int


def _rows(contours, geometries):
    contour = np.concatenate([np.full(g.kx.size, i) for i, g in enumerate(geometries)])
    row = np.concatenate([g.index for g in geometries])
    return contour, row


def density_weights(geometries, weights, layer_spacing) -> np.ndarray:
    """μ_p = κ ω_p s̄_p: the density of states per spin per volume at each node (J⁻¹ m⁻³)."""
    kappa = ELEMENTARY_CHARGE / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    return np.concatenate([kappa * w * 0.5 * (np.roll(g.s, 1) + g.s) for g, w in zip(geometries, weights)])


def geometries_of(contours, *, charge) -> List[ContourGeometry]:
    return [contour_geometry(c.kx, c.ky, c.vx, c.vy, charge=charge, period=c.period, vz=c.vz) for c in contours]


def kernel_on_nodes(kernel, contours, geometries):
    """P on the prepared nodes of every contour, and where each node came from."""
    kx = np.concatenate([g.kx for g in geometries])
    ky = np.concatenate([g.ky for g in geometries])
    three = any(c.kz is not None for c in contours)
    kz = np.concatenate([np.full(g.kx.size, c.kz) for c, g in zip(contours, geometries)]) if three else None
    where = _rows(contours, geometries)
    return _kernel_matrix(kernel, kx, ky, kz, where), where


def build_system(contours: List[ContourInput], kernel, *, layer_spacing, charge, remove_drift) -> CollisionSystem:
    """Everything that does not depend on the field."""
    geometries = geometries_of(contours, charge=charge)
    weights = np.array([c.weight for c in contours], dtype=np.float64)
    sizes = [g.kx.size for g in geometries]
    edges = np.cumsum([0] + sizes)
    omega = np.repeat(weights, sizes)
    kappa = ELEMENTARY_CHARGE / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    sbar = np.concatenate([0.5 * (np.roll(g.s, 1) + g.s) for g in geometries])
    mu = kappa * omega * sbar
    pmat, where = kernel_on_nodes(kernel, contours, geometries)
    gamma = pmat @ mu
    _check_resolution(pmat, mu, gamma, edges, where)

    tau0 = np.concatenate([np.broadcast_to(np.asarray(c.tau0, dtype=np.float64), (c.kx.size,))[g.index]
                           for c, g in zip(contours, geometries)])
    bad = np.flatnonzero(np.isnan(tau0) | (tau0 <= 0))
    if bad.size:
        raise ValueError(f"tau must be positive (or np.inf for no background scattering); it is {tau0[bad[0]]} at "
                         f"{_located(where, bad[0])}")
    rate = 1.0 / tau0 + gamma
    bad = np.flatnonzero(~(rate > 0) | ~np.isfinite(rate))
    if bad.size:
        raise ValueError(f"nothing scatters carriers at {_located(where, bad[0])}: tau is np.inf there and the "
                         "scattering kernel gives no out-scattering. Give a background tau, or a kernel that "
                         "scatters everywhere")
    # τ₀ itself where the kernel adds nothing (1/(1/τ₀) can differ in the last bit), so that
    # contours the kernel does not touch are exactly the relaxation-time ones.
    tau = np.where(gamma > 0, 1.0 / rate, tau0)
    prepared = [finish_contour(g, tau[a:b], remove_drift=remove_drift)
                for g, a, b in zip(geometries, edges[:-1], edges[1:])]
    dimension = 2 if prepared[0].lz is None else 3
    ell = np.concatenate([np.stack([p.lx, p.ly] + ([p.lz] if dimension == 3 else []), axis=1) for p in prepared])
    mass = np.concatenate([0.5 * (np.roll(p.damping, 1) + p.damping) for p in prepared])
    tau_hat = sbar / mass
    background = np.where(np.isinf(tau0), 0.0, tau / tau0)
    is_open = np.repeat([bool(np.any(p.period)) for p in prepared], sizes)

    # Contours coupled by nonzero blocks of P form components.
    blocks = np.add.reduceat(np.add.reduceat(pmat, edges[:-1], axis=0), edges[:-1], axis=1)  # (C, C)
    label = list(range(len(contours)))

    def root(i):
        while label[i] != i:
            label[i] = label[label[i]]
            i = label[i]
        return i

    for i, j in zip(*np.nonzero(blocks)):
        label[root(i)] = root(j)
    groups = {}
    for i in range(len(contours)):
        groups.setdefault(root(i), []).append(i)

    components, free = [], []
    for members in groups.values():
        if not any(blocks[i, j] > 0 for i in members for j in members):
            free.extend(members)
            continue
        nodes = np.concatenate([np.arange(edges[i], edges[i + 1]) for i in members])
        if nodes.size > MAX_COMPONENT_NODES:
            raise ValueError(
                f"{nodes.size} nodes are coupled by the scattering kernel, more than {MAX_COMPONENT_NODES}: the dense "
                f"matrices would need about {3 * 8 * nodes.size**2 / 1e9:.1f} GB. Use fewer nodes per contour or "
                "fewer k_z slices, and extrapolate=True to keep the accuracy")
        sub = np.ix_(nodes, nodes)
        jmat = kappa * tau_hat[nodes, None] * pmat[sub] * tau_hat[None, nodes]
        # The kernel's in-scattering of a constant equals the out-scattering it causes,
        # exactly: this conserves particles on the nodes (an O(N⁻²) correction).
        jmat[np.diag_indices(nodes.size)] += gamma[nodes] * (tau[nodes] - tau_hat[nodes]) / (omega[nodes] * mass[nodes])
        weighted = omega[nodes] * mass[nodes]
        local_ell = ell[nodes].copy()
        drift = weighted @ local_ell
        scale = weighted @ np.linalg.norm(local_ell, axis=1)
        pure = not np.any(background[nodes] > 0)
        if np.linalg.norm(drift) <= _DRIFT_TOLERANCE * scale:
            # A complete surface: the net drift is discretisation. Keeping it would add
            # (1 − b)D²/(bΣωm) to σ, which diverges as the background b vanishes.
            opened = is_open[nodes]
            subset = opened if np.any(opened) else np.ones(nodes.size, dtype=bool)
            local_ell[subset, :2] -= drift[:2] / weighted[subset].sum()
            if dimension == 3:
                local_ell[:, 2] -= drift[2] / weighted.sum()
            drift = weighted @ local_ell
        elif pure:
            raise ValueError(
                "the scattering kernel conserves particles on these contours and nothing else relaxes them, but "
                "together they carry a net current, so sigma would be infinite. Give the whole Fermi surface (for "
                "example both open sheets and every k_z slice), or a background tau")
        else:
            warnings.warn(
                "the contours coupled by the scattering kernel carry a net current, so the Fermi surface looks "
                "incomplete (one open sheet without its partner, say); the result depends on the background "
                "relaxation", UserWarning, stacklevel=3)
        largest = np.max(background[nodes])
        border_u = np.ones(nodes.size) if pure else background[nodes] / largest
        components.append(_Component(members, nodes, local_ell, jmat, border_u, weighted / weighted.max(), drift,
                                     0.0 if pure else largest))
    return CollisionSystem(prepared, weights, edges, gamma, components, free, dimension)


def _blocks(system: CollisionSystem, component: _Component, field: float, check: bool) -> list:
    """ω_o G_o for each contour of a component, and the propagator-level Onsager check."""
    result = []
    for i in component.contours:
        damping = system.prepared[i].damping
        g = mass_matrix(damping) if field == 0 else propagator(damping, field)
        if check and field != 0:
            order = _reversed_order(damping.size)
            reverse = propagator(damping[::-1], field)
            defect = np.max(np.abs(reverse - g[np.ix_(order, order)].T)) / np.max(np.abs(g))
            if not defect <= _ONSAGER_TOLERANCE:
                warnings.warn(
                    f"Onsager check failed: the propagator on the reversed orbit and the transpose differ by "
                    f"{defect:.1e} at |B| = {field:.6g} T. They agree to rounding, so this is a bug; please report it "
                    "with the contour that triggers it", RuntimeWarning, stacklevel=5)
        result.append(system.weights[i] * g)
    return result


def _solve_component(system: CollisionSystem, component: _Component, field: float, check: bool):
    """The bordered solve at one |B|: (σ without the prefactor, (d, d); the nodal solution X; α)."""
    blocks = _blocks(system, component, field, check)
    size = component.nodes.size
    starts = np.cumsum([0] + [b.shape[0] for b in blocks])
    matrix = np.zeros((size + 1, size + 1))
    matrix[:size, :size] = np.eye(size)
    for g, a, b in zip(blocks, starts[:-1], starts[1:]):
        matrix[:size, a:b] -= component.jmat[:, a:b] @ g
    matrix[:size, size] = component.border_u
    matrix[size, :size] = component.border_v
    rhs = np.vstack([component.ell, np.zeros((1, system.dimension))])
    solution = lu_solve(lu_factor(matrix, overwrite_a=True, check_finite=False), rhs, check_finite=False)
    x, alpha = solution[:size], solution[size]
    carried = np.empty_like(x)
    for g, a, b in zip(blocks, starts[:-1], starts[1:]):
        carried[a:b] = g @ x[a:b]
    sigma = component.ell.T @ carried
    if component.background > 0:
        # The constant on the component, relaxed only by the background, carries the drift.
        sigma += np.outer(component.drift, alpha) / component.background
    return sigma, x, alpha


def _kernel_tensor(system: CollisionSystem, fields: np.ndarray, symmetrize: bool) -> np.ndarray:
    """Σ over components of σ without the prefactor at each field, (nB, d, d)."""
    dim = system.dimension
    result = np.zeros((fields.size, dim, dim))
    if not system.components:
        return result
    magnitude = np.abs(fields)
    unique, inverse = np.unique(magnitude, return_inverse=True)
    check = symmetrize or bool(np.any(fields < 0))
    per_field = np.zeros((unique.size, dim, dim))
    for k, field in enumerate(unique):
        for component in system.components:
            per_field[k] += _solve_component(system, component, float(field), check)[0]
        if field == 0:
            per_field[k] = 0.5 * (per_field[k] + per_field[k].T)
    along = per_field[inverse]
    # σ(−B) = σ(B)ᵀ exactly: G(−B) = G(B)ᵀ and Ĵ is symmetric.
    return np.where((fields < 0)[:, None, None], np.transpose(along, (0, 2, 1)), along)


def conductivity(contours: List[ContourInput], fields, *, kernel, layer_spacing, charge, spin_degeneracy,
                 remove_drift, symmetrize, relaxation_time, extrapolate) -> np.ndarray:
    """σ (S/m) of contours coupled by a scattering kernel, shape (nB, d, d).

    `relaxation_time(prepared, fields)` gives σ without the prefactor for a contour the
    kernel does not touch (the relaxation-time path, unchanged).
    """
    fields = np.atleast_1d(np.asarray(fields, dtype=np.float64))
    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)

    def total(inputs):
        system = build_system(inputs, kernel, layer_spacing=layer_spacing, charge=charge, remove_drift=remove_drift)
        result = prefactor * _kernel_tensor(system, fields, symmetrize)
        # In contour order, with the prefactor and then the slice count applied per contour,
        # exactly as the relaxation-time path sums: contours the kernel does not touch then
        # give the same bits.
        for i in sorted(system.free):
            result += prefactor * relaxation_time(system.prepared[i], fields) / round(1.0 / system.weights[i])
        return result, system

    result, system = total(contours)
    if extrapolate:
        halves = []
        for contour, prepared in zip(contours, system.prepared):
            n_distinct = prepared.kx.size
            if n_distinct % 2 or n_distinct // 2 < MIN_NODES:
                raise ValueError(f"extrapolate needs an even number of distinct nodes, at least {2 * MIN_NODES}, "
                                 f"so that every other node is again a contour; this one has {n_distinct}")
            halves.append(contour.every_other(n_distinct))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            half, _ = total(halves)
        for warning in caught:
            warnings.warn(f"extrapolate: on every other node, {warning.message}", warning.category, stacklevel=3)
        # The discretisation error is c/N² + O(N⁻⁴) with the same c on every other node.
        result = (4.0 * result - half) / 3.0
    return result


def _to_input_order(values: np.ndarray, contour: ContourInput, prepared) -> np.ndarray:
    """Values on the prepared nodes, back in the input rows (a dropped closing row repeats node 0)."""
    out = np.empty((contour.kx.size,) + values.shape[1:])
    out[:] = values[0]
    out[prepared.index] = values
    return out


def out_scattering_rates(contours: List[ContourInput], kernel, *, layer_spacing, charge) -> List[np.ndarray]:
    """Γ at every input row of every contour (s⁻¹)."""
    geometries = geometries_of(contours, charge=charge)
    weights = [c.weight for c in contours]
    pmat, _ = kernel_on_nodes(kernel, contours, geometries)
    gamma = pmat @ density_weights(geometries, weights, layer_spacing)
    edges = np.cumsum([0] + [g.kx.size for g in geometries])
    return [_to_input_order(gamma[a:b], c, g) for c, g, a, b in zip(contours, geometries, edges[:-1], edges[1:])]


def density_of_states(contours: List[ContourInput], *, layer_spacing, charge) -> float:
    """Σ μ over every node of every contour (J⁻¹ m⁻³ per spin)."""
    geometries = geometries_of(contours, charge=charge)
    return float(np.sum(density_weights(geometries, [c.weight for c in contours], layer_spacing)))


def mean_free_paths(contours: List[ContourInput], kernel, *, layer_spacing, charge, remove_drift) -> List[np.ndarray]:
    """The vector mean free path L at B = 0 in the input rows of each contour, (N, d) each (m).

    Without a kernel on a contour it is ℓ = vτ. On a component with a kernel it is the
    solution c at B = 0: the true solution where the background relaxes the charge mode,
    and the one with zero Ω-weighted mean where nothing does (L is then defined only up to
    a constant, which changes no conductivity).
    """
    system = build_system(contours, kernel, layer_spacing=layer_spacing, charge=charge, remove_drift=remove_drift)
    values = [np.stack([p.lx, p.ly] + ([p.lz] if system.dimension == 3 else []), axis=1) for p in system.prepared]
    for component in system.components:
        _, x, alpha = _solve_component(system, component, 0.0, check=False)
        if component.background > 0:
            x = x + alpha[None, :] / component.background
        starts = np.cumsum([0] + [system.prepared[i].kx.size for i in component.contours])
        for i, a, b in zip(component.contours, starts[:-1], starts[1:]):
            values[i] = x[a:b]
    return [_to_input_order(v, c, p) for v, c, p in zip(values, contours, system.prepared)]
