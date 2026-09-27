"""Magnetotransport of Fermi contours (private; everything public is re-exported from bz).

`conductivity_tensor` is the array layer: it validates and orients the contour,
runs a kernel for each sign of B, symmetrises, and applies the prefactor
g_s e³ / (4π²ħ² d). Carriers on the reversed orbit are what a field of the
opposite sign produces, so σ(−B) is the same kernel on the reversed node order.

`conductivity` is the DataFrame layer on top. It sums σ over pockets, and
`resistivity`, `hall_coefficient` and `magnetoresistance` derive from its output.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...core.constants import ELEMENTARY_CHARGE, HBAR
from . import _kernel, _kernel_py
from ._contour import enclosed_area, prepare_contour

_BACKENDS = ("python", "numba")


def _orbit_sums_both(backend, forward, backward, field, both):
    """Orbit sums on the forward and (if `both`) the reversed orbit, shape (2, nB, 2, 2)."""
    if backend == "python":
        reverse = _kernel_py.orbit_sums(*backward, field) if both else np.full((field.size, 2, 2), np.nan)
        return np.stack([_kernel_py.orbit_sums(*forward, field), reverse])
    # One compiled call does both: the reversed orbit shares every segment's weights.
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (*forward, field)]
    if both:
        return _kernel.orbit_sums_both(*arrays)
    return np.stack([_kernel.orbit_sums(*arrays), np.full((field.size, 2, 2), np.nan)])


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
    remove_drift: Optional[bool] = None,
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
        remove_drift: Remove the discretisation drift of a closed orbit's ∮ v dt (default:
            closed contours only; never on open orbits).
        period: Reciprocal-lattice vector (G_x, G_y) (m⁻¹) of an open orbit, or None.
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

    # Each distinct |B| is computed once, on both orientations; B = 0 has its own branch.
    magnitude = np.abs(fields)
    unique, index = np.unique(magnitude, return_inverse=True)  # (nU,), (nB,)
    sums = np.empty((2, unique.size, 2, 2))  # [orientation, |B|]
    nonzero = unique > 0
    if np.any(nonzero):
        both = symmetrize or bool(np.any(fields < 0))  # the reversed orbit is only sometimes needed
        sums[:, nonzero] = _orbit_sums_both(backend, forward, backward, unique[nonzero], both)
    if not np.all(nonzero):  # unique is sorted, so B = 0 comes first
        sums[0, 0] = _zero_field_sums(*forward)
        sums[1, 0] = _zero_field_sums(*backward)

    along = np.where(fields[:, None, None] >= 0, sums[0, index], sums[1, index])  # σ(B), (nB, 2, 2)
    if symmetrize:
        against = np.where(fields[:, None, None] >= 0, sums[1, index], sums[0, index])  # σ(−B)
        result = 0.5 * (along + np.transpose(against, (0, 2, 1)))
    else:
        result = along

    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    return prefactor * result


_SIGMA_COLUMNS = ["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]


def _periods(period, n_frames: int) -> list:
    """One period per contour: None, a single (G_x, G_y) for all, or a list aligned with dfs."""
    if period is None:
        return [None] * n_frames
    items = list(period)
    if len(items) == 2 and all(np.isscalar(g) for g in items):
        return [period] * n_frames
    if len(items) != n_frames:
        raise ValueError(f"period must be one (G_x, G_y) pair or a list of {n_frames} (one per contour)")
    return items


def conductivity(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    field,
    *,
    layer_spacing: float,
    kx: str = "kx",
    ky: str = "ky",
    vx: str = "vx",
    vy: str = "vy",
    tau: Union[str, float] = "tau",
    charge: float = -ELEMENTARY_CHARGE,
    spin_degeneracy: float = 2,
    symmetrize: bool = True,
    remove_drift: Optional[bool] = None,
    period=None,
    backend: str = "numba",
) -> pd.DataFrame:
    """Magnetoconductivity tensor of one or more Fermi pockets (S/m).

    Solves the Boltzmann equation in the relaxation-time approximation with the
    Shockley–Chambers tube integral, for B along ẑ. It is exact in ω_cτ: every earlier
    orbit is included. Each DataFrame is one Fermi contour: nodes in order along it
    (either direction, any starting node), with the group velocity at each. A contour
    is either closed (a pocket) or, with `period`, one period of an open sheet.
    Several contours are combined by summing σ, which is the physically correct way
    (averaging ρ is not).

    Args:
        dfs: One contour DataFrame, or a list of them (one per pocket).
        field: Magnetic field B along ẑ (T), a float or array-like. Any sign, including 0.
        layer_spacing: Interlayer spacing d (m). Required: body-centred cells have
            d = c/2, not c.
        kx: Column of wavevectors k_x (m⁻¹).
        ky: Column of wavevectors k_y (m⁻¹).
        vx: Column of group velocities v_x = (1/ħ) ∂ε/∂k_x (m/s), not unit vectors.
        vy: Column of group velocities v_y (m/s).
        tau: Column of relaxation times (s), or one relaxation time for every node.
        charge: Carrier charge q (C), ±e. Keep the default −e for band electrons: a
            hole-like pocket is described by its inward-pointing velocities.
        spin_degeneracy: Spin degeneracy g_s.
        symmetrize: Enforce Onsager symmetry, σ(B) → ½[σ(B) + σ(−B)ᵀ]. This removes a
            small discretisation error, including a spurious 1/B term in low-field R_H.
        remove_drift: Remove the discretisation drift of each closed orbit's ∮ v dt, so
            σ_xx falls as 1/B² at high field instead of levelling off. The default (None)
            does this for closed contours only; on open orbits the drift is physical.
        period: For open orbits, the reciprocal-lattice vector (G_x, G_y) (m⁻¹) that
            takes the last node of a contour on to its first: one pair for every
            contour, or a list aligned with `dfs` (None for closed pockets).
        backend: "numba" (compiled, parallel over fields) or "python" (plain NumPy).

    Returns:
        DataFrame with columns field (T), sigma_xx, sigma_xy, sigma_yx, sigma_yy (S/m),
        one row per field in the order given.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> sigma = bz.conductivity(df, np.linspace(0, 30, 31), layer_spacing=1e-9)
        >>> list(sigma.columns)
        ['field', 'sigma_xx', 'sigma_xy', 'sigma_yx', 'sigma_yy']
    """
    frames = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
    if not frames:
        raise ValueError("dfs must contain at least one contour")
    fields = np.atleast_1d(np.asarray(field, dtype=np.float64))  # (nB,)
    periods = _periods(period, len(frames))
    total = np.zeros((fields.size, 2, 2))
    for frame, frame_period in zip(frames, periods):
        tau_values = frame[tau] if isinstance(tau, str) else float(tau)
        total += conductivity_tensor(
            frame[kx], frame[ky], frame[vx], frame[vy], tau_values, fields,
            layer_spacing=layer_spacing, charge=charge, spin_degeneracy=spin_degeneracy,
            symmetrize=symmetrize, remove_drift=remove_drift, period=frame_period, backend=backend,
        )
    result = pd.DataFrame(total.reshape(fields.size, 4), columns=_SIGMA_COLUMNS)
    result.insert(0, "field", fields)
    return result


def _sigma_components(sigma: pd.DataFrame):
    xx, xy, yx, yy = (sigma[c].to_numpy(dtype=np.float64) for c in _SIGMA_COLUMNS)
    return xx, xy, yx, yy, xx * yy - xy * yx  # components and det σ, each (nB,)


def resistivity(sigma: pd.DataFrame) -> pd.DataFrame:
    """Resistivity tensor ρ = σ⁻¹ at each field (Ω·m).

    Args:
        sigma: Output of `conductivity`, with columns field, sigma_xx, sigma_xy,
            sigma_yx, sigma_yy.

    Returns:
        DataFrame with columns field (T), rho_xx, rho_xy, rho_yx, rho_yy (Ω·m).

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> rho = bz.resistivity(bz.conductivity(df, [0.0, 10.0], layer_spacing=1e-9))
    """
    xx, xy, yx, yy, det = _sigma_components(sigma)
    return pd.DataFrame({
        "field": sigma["field"].to_numpy(dtype=np.float64),
        "rho_xx": yy / det,
        "rho_xy": -xy / det,
        "rho_yx": -yx / det,
        "rho_yy": xx / det,
    }, index=sigma.index)


def hall_coefficient(sigma: pd.DataFrame) -> pd.Series:
    """Hall coefficient R_H = ½(ρ_yx − ρ_xy)/B at each field (m³/C).

    Taking the antisymmetric part of ρ removes any even-in-B contamination. For a
    single closed pocket at high field R_H → 1/(nq): −1/(ne) for electron-like and
    +1/(ne) for hole-like pockets. R_H is NaN at B = 0, where it is undefined.

    Args:
        sigma: Output of `conductivity`.

    Returns:
        Series named "hall_coefficient" (m³/C), aligned with `sigma`.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> r_h = bz.hall_coefficient(bz.conductivity(df, [0.0, 10.0], layer_spacing=1e-9))
    """
    _, xy, yx, _, det = _sigma_components(sigma)
    b = sigma["field"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(b == 0, np.nan, (xy - yx) / (2 * det * b))
    return pd.Series(values, index=sigma.index, name="hall_coefficient")


def magnetoresistance(sigma: pd.DataFrame) -> pd.Series:
    """Magnetoresistance (ρ_xx(B) − ρ_xx(0)) / ρ_xx(0) at each field (dimensionless).

    Args:
        sigma: Output of `conductivity`, which must include a B = 0 row.

    Returns:
        Series named "magnetoresistance", aligned with `sigma`; exactly 0 at B = 0.

    Raises:
        ValueError: If `sigma` has no B = 0 row.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> mr = bz.magnetoresistance(bz.conductivity(df, np.linspace(0, 30, 7), layer_spacing=1e-9))
    """
    rho_xx = resistivity(sigma)["rho_xx"].to_numpy()
    zero = np.flatnonzero(sigma["field"].to_numpy() == 0)
    if zero.size == 0:
        raise ValueError("magnetoresistance needs a B = 0 row in sigma; include 0 in the fields")
    reference = rho_xx[zero[0]]
    return pd.Series((rho_xx - reference) / reference, index=sigma.index, name="magnetoresistance")


def carrier_density(
    df: pd.DataFrame,
    kx: str = "kx",
    ky: str = "ky",
    vx: str = "vx",
    vy: str = "vy",
    *,
    layer_spacing: float,
    spin_degeneracy: float = 2,
) -> float:
    """Carrier density of one closed pocket from the k-space area it encloses (m⁻³).

    n = g_s A / (4π² d): the Luttinger count per layer divided by the layer spacing.
    For a hole-like pocket this is the density of holes. The area uses the velocities
    for the contour's curvature, so it is accurate to O(N⁻⁴) even for irregular sampling.

    Args:
        df: One contour DataFrame (nodes in order around the pocket).
        kx: Column of wavevectors k_x (m⁻¹).
        ky: Column of wavevectors k_y (m⁻¹).
        vx: Column of group velocities v_x (m/s).
        vy: Column of group velocities v_y (m/s).
        layer_spacing: Interlayer spacing d (m).
        spin_degeneracy: Spin degeneracy g_s.

    Returns:
        The carrier density n (m⁻³), positive.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> n = bz.carrier_density(df, layer_spacing=1e-9)  # g_s k_F² / (4π d)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("carrier_density takes one contour DataFrame; call it once per pocket")
    contour = prepare_contour(df[kx], df[ky], df[vx], df[vy], 1.0)  # validates; τ plays no part
    area = enclosed_area(contour.kx, contour.ky, contour.vx + contour.drift[0], contour.vy + contour.drift[1])
    return spin_degeneracy * area / (4 * np.pi**2 * layer_spacing)
