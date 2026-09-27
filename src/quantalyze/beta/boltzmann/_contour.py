"""Validate a Fermi-surface contour and prepare it for the conductivity kernels.

`prepare_contour` is the one place the input contract is enforced. It takes
nodes k_n with group velocities v_n = ∇ε/ħ and relaxation times τ_n, and returns
them ordered along the carriers' motion, with the per-segment geometric time
s_n and scattering rate γ_n that the kernels integrate over.

Conventions: SI units, B = +B ẑ, and ħ dk/dt = q v × B, so a carrier moves
along dk ∝ q (v_y, −v_x). Segment n joins node n to node n+1 (indices mod N).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, HBAR

MIN_NODES = 16

# Nodes closer together than this fraction of the contour's size are the same point.
_SAME_POINT = 1e-12
# A closing segment (last node back to the first) longer than this many times every
# other segment means the nodes do not go all the way round: the contour is open.
_OPEN_GAP = 5.0
# Warn when a segment is further than this from perpendicular to the mean velocity
# direction at its ends (|cos| of the angle between them; 0.1 is about 5.7°).
_NORMAL_TOLERANCE = 0.1


@dataclass(frozen=True)
class PreparedContour:
    """A validated contour, ordered along the motion, with per-segment quantities.

    All arrays are float64, C-contiguous and read-only.

    Attributes:
        kx: Node wavevectors k_x (m⁻¹), in the order the carriers move. Shape (N,).
        ky: Node wavevectors k_y (m⁻¹). Shape (N,).
        vx: Group velocities v_x with the drift removed (m/s). Shape (N,).
        vy: Group velocities v_y with the drift removed (m/s). Shape (N,).
        tau: Relaxation times τ (s). Shape (N,).
        s: Geometric time of segment n → n+1, ħ|Δk_n|/(2e) · (1/|v_n| + 1/|v_{n+1}|),
            using the original velocities (s·T). The real time is s_n/|B|. Shape (N,).
        gamma: Mean scattering rate on segment n, ½(1/τ_n + 1/τ_{n+1}) (s⁻¹). Shape (N,).
        drift: The velocity v̄ subtracted from every node (m/s); zero if drift
            removal was off. Shape (2,).
        charge: Carrier charge q (C).
    """

    kx: np.ndarray
    ky: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    tau: np.ndarray
    s: np.ndarray
    gamma: np.ndarray
    drift: np.ndarray
    charge: float


def _as_1d(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, not of shape {array.shape}")
    return array


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array, dtype=np.float64)
    array.setflags(write=False)
    return array


def prepare_contour(
    kx,
    ky,
    vx,
    vy,
    tau,
    *,
    charge: float = -ELEMENTARY_CHARGE,
    remove_drift: bool = True,
    period: Optional[Sequence[float]] = None,
) -> PreparedContour:
    """Validate a closed Fermi-surface contour and prepare it for the kernels.

    The nodes may be given in either direction and from any starting node. The
    orientation is taken from the velocities, not the input order: the result runs
    along ħ dk/dt = q v × B for B = +B ẑ, and keeps the input's first node first.
    A final node that repeats the first is dropped.

    With drift removal on, the velocity v̄ = Σ ½ s_n (v_n + v_{n+1}) / Σ s_n is
    subtracted from every node. On a closed orbit ∮ v dt = 0 exactly, so v̄ is pure
    discretisation error; removing it with the same trapezoid weights the kernels use
    makes the discrete ∮ v dt vanish, which stops σ_xx from levelling off at high
    field instead of falling as 1/B².

    Args:
        kx: Node wavevectors k_x (m⁻¹), array-like of shape (N,).
        ky: Node wavevectors k_y (m⁻¹), array-like of shape (N,).
        vx: Group velocities v_x = (1/ħ) ∂ε/∂k_x (m/s), shape (N,). Not unit vectors.
        vy: Group velocities v_y = (1/ħ) ∂ε/∂k_y (m/s), shape (N,).
        tau: Relaxation times (s), shape (N,) or a single float; finite and positive.
        charge: Carrier charge q (C), ±e. The default −e is for band electrons; a
            hole-like pocket is described by its inward-pointing velocities, not by q.
        remove_drift: Subtract the discretisation drift v̄ from the velocities.
        period: Reciprocal-lattice vector (G_x, G_y) (m⁻¹) joining the last node of an
            open orbit to its first. Open orbits are not supported yet.

    Returns:
        PreparedContour with the ordered nodes, velocities, s_n, γ_n and drift.

    Raises:
        ValueError: If the arrays are not 1-D or differ in length; if any value is
            NaN or infinite; if τ ≤ 0 or v = 0 at a node; if there are fewer than 16
            distinct nodes; if two consecutive nodes coincide; if the contour is not
            closed; if the nodes are not ordered along the contour; or if |q| ≠ e.
        NotImplementedError: If `period` is given.

    Warns:
        UserWarning: If the velocities are not normal to the contour: a sign of
            unit-vector velocities, swapped components or mixed units, or of a contour
            sampled too coarsely where it curves sharply.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta.boltzmann._contour import prepare_contour
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> contour = prepare_contour(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"])
    """
    if period is not None:
        raise NotImplementedError("open orbits (period) are not supported yet")

    charge = float(charge)
    if not np.isfinite(charge) or charge == 0:
        raise ValueError(f"charge must be finite and non-zero, not {charge}")
    # The geometric time and the σ prefactor are written for carriers of charge ±e.
    if abs(abs(charge) / ELEMENTARY_CHARGE - 1) > 1e-12:
        raise ValueError(f"charge must be ±e ({ELEMENTARY_CHARGE} C), not {charge}")

    kx, ky, vx, vy = (_as_1d(n, a) for n, a in (("kx", kx), ("ky", ky), ("vx", vx), ("vy", vy)))
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 0:
        tau = np.full(kx.shape, float(tau))
    tau = _as_1d("tau", tau)
    lengths = {"kx": kx.size, "ky": ky.size, "vx": vx.size, "vy": vy.size, "tau": tau.size}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"kx, ky, vx, vy and tau must all have the same length, not {lengths}")
    for name, array in (("kx", kx), ("ky", ky), ("vx", vx), ("vy", vy), ("tau", tau)):
        bad = np.flatnonzero(~np.isfinite(array))
        if bad.size:
            raise ValueError(f"{name} must be finite; it is {array[bad[0]]} at node {bad[0]}")
    bad = np.flatnonzero(tau <= 0)
    if bad.size:
        raise ValueError(f"tau must be positive; it is {tau[bad[0]]} at node {bad[0]}")
    speed = np.hypot(vx, vy)  # (N,)
    bad = np.flatnonzero(speed == 0)
    if bad.size:
        raise ValueError(f"the velocity must be non-zero; it vanishes at node {bad[0]}")
    if kx.size < MIN_NODES:
        raise ValueError(f"a contour needs at least {MIN_NODES} nodes, not {kx.size}")

    # A closing point that repeats the first node, e.g. from θ = linspace(0, 2π, N).
    size = max(np.ptp(kx), np.ptp(ky))
    same_point = _SAME_POINT * size
    if np.hypot(kx[-1] - kx[0], ky[-1] - ky[0]) <= same_point:
        kx, ky, vx, vy, tau, speed = (a[:-1] for a in (kx, ky, vx, vy, tau, speed))
        if kx.size < MIN_NODES:
            raise ValueError(
                f"a contour needs at least {MIN_NODES} distinct nodes, not {kx.size} "
                "(the last point repeats the first)"
            )

    dkx = np.roll(kx, -1) - kx  # segment n: node n → n+1, (N,)
    dky = np.roll(ky, -1) - ky
    length = np.hypot(dkx, dky)
    bad = np.flatnonzero(length <= same_point)
    if bad.size:
        n = bad[0]
        raise ValueError(f"nodes {n} and {(n + 1) % kx.size} coincide (a zero-length segment)")
    if length[-1] > _OPEN_GAP * np.max(length[:-1]):
        raise ValueError(
            "the contour is not closed: the gap from the last node back to the first is "
            f"{length[-1] / np.max(length[:-1]):.1f} times the longest other segment. "
            "Sample the whole orbit, or pass period for an open orbit"
        )

    # Direction of motion on each segment: dk/dt ∝ q (v_y, −v_x), averaged over its ends.
    sign = np.sign(charge)
    tx = 0.5 * sign * (vy + np.roll(vy, -1))
    ty = -0.5 * sign * (vx + np.roll(vx, -1))
    along = dkx * tx + dky * ty  # (N,)
    if np.all(along < 0):
        order = np.roll(np.arange(kx.size)[::-1], 1)  # reverse, keeping node 0 first
        kx, ky, vx, vy, tau, speed = (a[order] for a in (kx, ky, vx, vy, tau, speed))
        dkx = np.roll(kx, -1) - kx
        dky = np.roll(ky, -1) - ky
        length = np.hypot(dkx, dky)
    elif not np.all(along > 0):
        majority_forward = np.sum(along > 0) >= np.sum(along < 0)
        against = np.flatnonzero(along <= 0 if majority_forward else along >= 0)
        raise ValueError(
            "the nodes are not in order along the contour: segment(s) starting at node(s) "
            f"{against[:5].tolist()} run against the motion set by the velocities"
        )

    # Velocities should be normal to the contour: compare each segment with the mean of
    # the unit velocities at its ends.
    ux, uy = vx / speed, vy / speed
    mx, my = ux + np.roll(ux, -1), uy + np.roll(uy, -1)
    mean_norm = np.hypot(mx, my)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(mean_norm > 0, np.abs(dkx * mx + dky * my) / (length * mean_norm), 1.0)
    worst = int(np.argmax(cosine))
    if cosine[worst] > _NORMAL_TOLERANCE:
        angle = np.degrees(np.arcsin(min(cosine[worst], 1.0)))
        warnings.warn(
            f"the velocities are not normal to the contour (off by {angle:.1f}° on segment {worst}). "
            "Either v is not the group velocity ∇ε/ħ in m/s (unit vectors, swapped components, "
            "k not in m⁻¹), or the contour is too coarsely sampled where it curves sharply",
            UserWarning,
            stacklevel=2,
        )

    s = HBAR * length / (2 * ELEMENTARY_CHARGE) * (1 / speed + 1 / np.roll(speed, -1))  # (N,), s·T
    gamma = 0.5 * (1 / tau + 1 / np.roll(tau, -1))  # (N,), 1/s

    drift = np.zeros(2)
    if remove_drift:
        drift[0] = np.sum(0.5 * s * (vx + np.roll(vx, -1))) / np.sum(s)
        drift[1] = np.sum(0.5 * s * (vy + np.roll(vy, -1))) / np.sum(s)
        vx = vx - drift[0]
        vy = vy - drift[1]

    return PreparedContour(
        kx=_frozen(kx),
        ky=_frozen(ky),
        vx=_frozen(vx),
        vy=_frozen(vy),
        tau=_frozen(tau),
        s=_frozen(s),
        gamma=_frozen(gamma),
        drift=_frozen(drift),
        charge=charge,
    )


# Gauss–Legendre on [0, 1]: three points integrate the degree-5 Green's-theorem
# integrand of a cubic segment exactly.
_GL3_U = 0.5 * (1 + np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]))
_GL3_W = 0.5 * np.array([5 / 9, 8 / 9, 5 / 9])


def enclosed_area(kx, ky, vx, vy) -> float:
    """Area enclosed by a closed Fermi contour, using the velocities for the curvature (m⁻²).

    The group velocity is normal to the Fermi contour, so each node also gives the
    contour's tangent direction. Each segment is replaced by the cubic Hermite curve
    through its two nodes with those tangent directions (scaled by the chord length),
    and the area follows exactly from Green's theorem, A = ½∮(k_x dk_y − k_y dk_x).
    The error is O(N⁻⁴), even for irregularly spaced nodes, where a polygon (shoelace)
    area is only O(N⁻²): about 2e-10 against 2.5e-5 on a circle with N = 512.

    Args:
        kx: Node wavevectors k_x (m⁻¹), in order around the contour (either direction).
        ky: Node wavevectors k_y (m⁻¹).
        vx: Group velocities v_x at the nodes (m/s); only their direction is used.
        vy: Group velocities v_y (m/s).

    Returns:
        The enclosed area (m⁻²), positive.
    """
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    kx, ky = kx - np.mean(kx), ky - np.mean(ky)  # centre first: less cancellation
    speed = np.hypot(vx, vy)
    tx, ty = -np.asarray(vy) / speed, np.asarray(vx) / speed  # unit tangents, up to sign
    x1, y1 = np.roll(kx, -1), np.roll(ky, -1)
    cx, cy = x1 - kx, y1 - ky  # chords, (N,)
    chord = np.hypot(cx, cy)
    tx1, ty1 = np.roll(tx, -1), np.roll(ty, -1)
    # Point each tangent along its segment, and scale it by the chord length.
    start = np.sign(tx * cx + ty * cy) * chord
    end = np.sign(tx1 * cx + ty1 * cy) * chord
    m0x, m0y, m1x, m1y = start * tx, start * ty, end * tx1, end * ty1
    total = 0.0
    for u, w in zip(_GL3_U, _GL3_W):
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        d00, d10, d01, d11 = 6 * u**2 - 6 * u, 3 * u**2 - 4 * u + 1, -6 * u**2 + 6 * u, 3 * u**2 - 2 * u
        x = h00 * kx + h10 * m0x + h01 * x1 + h11 * m1x
        y = h00 * ky + h10 * m0y + h01 * y1 + h11 * m1y
        dx = d00 * kx + d10 * m0x + d01 * x1 + d11 * m1x
        dy = d00 * ky + d10 * m0y + d01 * y1 + d11 * m1y
        total += w * np.sum(x * dy - y * dx)
    return abs(0.5 * total)
