"""Magnetotransport of Fermi contours (private; everything public is re-exported from bz).

`conductivity_tensor` is the array layer: it validates and orients the contour,
runs a kernel for each sign of B, symmetrises, and applies the prefactor
g_s e³ / (4π²ħ² d). Carriers on the reversed orbit are what a field of the
opposite sign produces, so σ(−B) is the same kernel on the reversed node order. The
kernels solve a continuous model exactly, so σ(−B) = σ(B)ᵀ holds to rounding; when
both orientations are computed, a disagreement beyond rounding is reported as a
warning (it would mean a bug).

`conductivity` is the DataFrame layer on top. It sums σ over pockets, and
`resistivity`, `hall_coefficient` and `magnetoresistance` derive from its output.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...core.constants import ELEMENTARY_CHARGE, HBAR
from . import _kernel, _kernel_py
from ._contour import enclosed_area, prepare_contour

_BACKENDS = ("python", "numba")
# σ(−B) and σ(B)ᵀ agree to ~1e-15 of max|σ| (rounding); a larger difference is a bug.
_ONSAGER_TOLERANCE = 1e-9


def _orbit_sums_both(backend, forward, backward, field, both):
    """Orbit sums on the forward and (if `both`) the reversed orbit, shape (2, nB, d, d).

    `forward` and `backward` are (Δg, ℓ_x, ℓ_y) or (Δg, ℓ_x, ℓ_y, ℓ_z), so d = 2 or 3.
    """
    dim = len(forward) - 1
    missing = np.full((field.size, dim, dim), np.nan)
    if backend == "python":
        def run(arrays):
            damping, lx, ly, *lz = arrays
            return _kernel_py.orbit_sums(damping, lx, ly, field, lz=lz[0] if lz else None)

        return np.stack([run(forward), run(backward) if both else missing])
    # One compiled call does both: the reversed orbit shares every segment's weights.
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in forward]
    b = np.ascontiguousarray(field, dtype=np.float64)
    if dim == 2:
        return _kernel.orbit_sums_both(*arrays, b) if both else np.stack([_kernel.orbit_sums(*arrays, b), missing])
    return _kernel.orbit_sums_both3(*arrays, b) if both else np.stack([_kernel.orbit_sums3(*arrays, b), missing])


def _zero_field_sums(damping, *paths):
    """∮ ℓ_i ℓ_j dg with ℓ linear in g on each segment: the kernels' exact limit as B → 0.

    As B → 0 the history w becomes ℓ itself, so each segment contributes
    Δg [⅓(a_i a_j + b_i b_j) + ⅙(a_i b_j + b_i a_j)], with a and b ℓ at its two ends. It
    is σ_ij(0) = (g_s e²/4π²ħd) ∮ |dk| τ v_i v_j/|v| discretised, since dg = ħ|dk|/(e|v|τ);
    it is the same for either orientation, and σ(B) departs from it as B².
    """
    ell = np.stack(paths, axis=1)  # (N, d)
    ell_next = np.roll(ell, -1, axis=0)
    third, sixth = damping / 3, damping / 6
    return (np.einsum("n,ni,nj->ij", third, ell, ell) + np.einsum("n,ni,nj->ij", third, ell_next, ell_next)
            + np.einsum("n,ni,nj->ij", sixth, ell, ell_next) + np.einsum("n,ni,nj->ij", sixth, ell_next, ell))


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
    vz=None,
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
        symmetrize: Replace σ(B) by ½[σ(B) + σ(−B)ᵀ]. The solver satisfies Onsager's
            relation σ(−B) = σ(B)ᵀ to rounding, so this changes σ only at that level; it
            also checks the relation on every run, and warns if it fails.
        remove_drift: Remove the discretisation drift of a closed orbit's in-plane
            ∮ v dt (default: closed contours only; never on open orbits).
        period: Reciprocal-lattice vector (G_x, G_y) (m⁻¹) of an open orbit, or None.
        vz: Velocities v_z (m/s), shape (N,), if the contour is one k_z slice of a warped
            surface. The result is then 3×3, as if the whole Fermi surface were this slice
            (σ of a warped surface is the average over its slices).
        backend: "numba" (compiled, parallel over fields) or "python" (plain NumPy, the
            slower reference implementation the numba kernel is tested against).

    Returns:
        ndarray of shape (nB, 2, 2), or (nB, 3, 3) with `vz`: σ with index order
        [[xx, xy, …], [yx, yy, …], …] (S/m).

    Warns:
        RuntimeWarning: If σ(−B) and σ(B)ᵀ, computed on the two orientations of the orbit,
            differ by more than 1e-9 of max|σ| at some field. They agree to rounding, so
            this signals a bug.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> sigma = bz.conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"],
        ...                                [0.0, 10.0], layer_spacing=1e-9)  # (2, 2, 2)
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, not {backend!r}")
    contour = prepare_contour(kx, ky, vx, vy, tau, charge=charge, remove_drift=remove_drift, period=period, vz=vz)
    fields = np.atleast_1d(np.asarray(field, dtype=np.float64))  # (nB,)
    if fields.ndim != 1:
        raise ValueError(f"field must be a float or 1-D, not of shape {fields.shape}")
    if not np.all(np.isfinite(fields)):
        raise ValueError("field must be finite")

    # The prepared order is the motion for B > 0; B < 0 runs the orbit backwards.
    paths = (contour.lx, contour.ly) if contour.lz is None else (contour.lx, contour.ly, contour.lz)
    dim = len(paths)
    forward = (contour.damping, *paths)
    order = np.roll(np.arange(contour.damping.size)[::-1], 1)  # node 0, N−1, …, 1
    backward = (contour.damping[::-1], *(p[order] for p in paths))

    # Each distinct |B| is computed once, on both orientations; B = 0 has its own branch.
    magnitude = np.abs(fields)
    unique, index = np.unique(magnitude, return_inverse=True)  # (nU,), (nB,)
    sums = np.empty((2, unique.size, dim, dim))  # [orientation, |B|]
    nonzero = unique > 0
    if np.any(nonzero):
        both = symmetrize or bool(np.any(fields < 0))  # the reversed orbit is only sometimes needed
        sums[:, nonzero] = _orbit_sums_both(backend, forward, backward, unique[nonzero], both)
        if both:
            _check_onsager(sums[0, nonzero], sums[1, nonzero], unique[nonzero])
    if not np.all(nonzero):  # unique is sorted, so B = 0 comes first
        sums[:, 0] = _zero_field_sums(*forward)

    along = np.where(fields[:, None, None] >= 0, sums[0, index], sums[1, index])  # σ(B), (nB, d, d)
    if symmetrize:
        against = np.where(fields[:, None, None] >= 0, sums[1, index], sums[0, index])  # σ(−B)
        result = 0.5 * (along + np.transpose(against, (0, 2, 1)))
    else:
        result = along

    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    return prefactor * result


def _check_onsager(along, against, magnitude):
    """Warn if σ(B) (`along`) and σ(−B)ᵀ (`against` transposed) differ beyond rounding."""
    difference = np.max(np.abs(along - np.transpose(against, (0, 2, 1))), axis=(1, 2))  # (nB,)
    scale = np.max(np.abs(along), axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        defect = np.where(scale > 0, difference / scale, difference)
    worst = int(np.argmax(defect))
    if not defect[worst] <= _ONSAGER_TOLERANCE:  # also catches NaN
        warnings.warn(
            f"Onsager check failed: sigma(-B) and sigma(B)^T differ by {defect[worst]:.1e} of max|sigma| at "
            f"|B| = {magnitude[worst]:.6g} T. The solver satisfies this to rounding, so this is a bug; "
            "please report it with the contour that triggers it",
            RuntimeWarning,
            stacklevel=3,
        )


_AXES = "xyz"


def _columns(prefix: str, dim: int) -> list:
    return [f"{prefix}_{_AXES[i]}{_AXES[j]}" for i in range(dim) for j in range(dim)]


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


def _kz_slices(frame: pd.DataFrame, kz: str, layer_spacing: float) -> list:
    """The rows of each k_z slice, checking the slices evenly cover one period 2π/d."""
    values = np.unique(frame[kz].to_numpy(dtype=np.float64))
    if values.size > 1:
        spacing = 2 * np.pi / (layer_spacing * values.size)
        if not np.allclose(np.diff(values), spacing, rtol=1e-6, atol=0):
            raise ValueError(
                f"the {values.size} k_z values must be evenly spaced over one period 2π/d, i.e. "
                f"{spacing:.6e} m⁻¹ apart; they are {np.diff(values).min():.6e} to "
                f"{np.diff(values).max():.6e} m⁻¹ apart"
            )
    column = frame[kz].to_numpy()
    return [frame[column == value] for value in values]


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
    kz: Optional[str] = None,
    vz: str = "vz",
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

    A Fermi surface warped along k_z is given as slices: pass `kz`, and each distinct
    k_z value is one contour (with B ∥ ẑ, k_z is constant on every orbit). The slices
    must be evenly spaced over one period 2π/d, and the result is the full 3×3 tensor.

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
        kz: Column of wavevectors k_z (m⁻¹) for a k_z-warped surface; None for a
            two-dimensional one.
        vz: Column of group velocities v_z (m/s), used with `kz`.
        charge: Carrier charge q (C), ±e. Keep the default −e for band electrons: a
            hole-like pocket is described by its inward-pointing velocities.
        spin_degeneracy: Spin degeneracy g_s.
        symmetrize: Return ½[σ(B) + σ(−B)ᵀ]. The solver satisfies Onsager's relation
            σ(−B) = σ(B)ᵀ to rounding, so this changes σ only at that level; it also
            computes both orientations of each orbit and warns if the relation fails.
        remove_drift: Remove the discretisation drift of each closed orbit's in-plane
            ∮ v dt, so σ_xx falls as 1/B² at high field instead of levelling off. The
            default (None) does this for closed contours only; on open orbits the drift
            is physical.
        period: For open orbits, the reciprocal-lattice vector (G_x, G_y) (m⁻¹) that
            takes the last node of a contour on to its first: one pair for every
            contour, or a list aligned with `dfs` (None for closed pockets).
        backend: "numba" (compiled, parallel over fields) or "python" (plain NumPy).

    Returns:
        DataFrame with columns field (T) and sigma_xx, sigma_xy, sigma_yx, sigma_yy (S/m);
        with `kz`, the nine components sigma_xx, sigma_xy, sigma_xz, sigma_yx, …, sigma_zz.
        One row per field, in the order given.

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
    dim = 2 if kz is None else 3
    options = dict(layer_spacing=layer_spacing, charge=charge, spin_degeneracy=spin_degeneracy,
                   symmetrize=symmetrize, remove_drift=remove_drift, backend=backend)
    total = np.zeros((fields.size, dim, dim))
    for frame, frame_period in zip(frames, periods):
        slices = [frame] if kz is None else _kz_slices(frame, kz, layer_spacing)
        for part in slices:
            tau_values = part[tau] if isinstance(tau, str) else float(tau)
            # Each slice stands for 1/N_z of the k_z period: average them (a periodic
            # trapezoid rule in k_z, exact to rounding for smooth warping).
            total += conductivity_tensor(
                part[kx], part[ky], part[vx], part[vy], tau_values, fields, period=frame_period,
                vz=None if kz is None else part[vz], **options,
            ) / len(slices)
    result = pd.DataFrame(total.reshape(fields.size, dim * dim), columns=_columns("sigma", dim))
    result.insert(0, "field", fields)
    return result


def _dimension(sigma: pd.DataFrame) -> int:
    return 3 if "sigma_zz" in sigma.columns else 2


def resistivity(sigma: pd.DataFrame) -> pd.DataFrame:
    """Resistivity tensor ρ = σ⁻¹ at each field (Ω·m).

    Args:
        sigma: Output of `conductivity`: columns field and the 2×2 (or, for a k_z-warped
            surface, 3×3) components of σ.

    Returns:
        DataFrame with columns field (T) and rho_xx, rho_xy, rho_yx, rho_yy (Ω·m), or the
        nine components rho_xx … rho_zz for a 3×3 σ.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> rho = bz.resistivity(bz.conductivity(df, [0.0, 10.0], layer_spacing=1e-9))
    """
    dim = _dimension(sigma)
    s = sigma[_columns("sigma", dim)].to_numpy(dtype=np.float64).reshape(-1, dim, dim)  # (nB, d, d)
    if dim == 2:
        det = s[:, 0, 0] * s[:, 1, 1] - s[:, 0, 1] * s[:, 1, 0]
        rho = np.stack([s[:, 1, 1], -s[:, 0, 1], -s[:, 1, 0], s[:, 0, 0]], axis=1) / det[:, None]
    else:
        rho = np.linalg.inv(s).reshape(-1, 9)
    result = pd.DataFrame(rho, columns=_columns("rho", dim), index=sigma.index)
    result.insert(0, "field", sigma["field"].to_numpy(dtype=np.float64))
    return result


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
    rho = resistivity(sigma)
    b = sigma["field"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(b == 0, np.nan, (rho["rho_yx"].to_numpy() - rho["rho_xy"].to_numpy()) / (2 * b))
    return pd.Series(values, index=sigma.index, name="hall_coefficient")


def magnetoresistance(sigma: pd.DataFrame, component: str = "xx") -> pd.Series:
    """Magnetoresistance (ρ_ii(B) − ρ_ii(0)) / ρ_ii(0) at each field (dimensionless).

    Args:
        sigma: Output of `conductivity`, which must include a B = 0 row.
        component: Which diagonal component: "xx", "yy", or (for a k_z-warped surface)
            "zz", the interlayer magnetoresistance.

    Returns:
        Series named "magnetoresistance", aligned with `sigma`; exactly 0 at B = 0.

    Raises:
        ValueError: If `sigma` has no B = 0 row, or `component` is not available.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> mr = bz.magnetoresistance(bz.conductivity(df, np.linspace(0, 30, 7), layer_spacing=1e-9))
    """
    rho = resistivity(sigma)
    column = f"rho_{component}"
    if component not in ("xx", "yy", "zz") or column not in rho.columns:
        raise ValueError(f"component must be 'xx', 'yy' or (for a k_z-warped surface) 'zz', not {component!r}")
    values = rho[column].to_numpy()
    zero = np.flatnonzero(sigma["field"].to_numpy() == 0)
    if zero.size == 0:
        raise ValueError("magnetoresistance needs a B = 0 row in sigma; include 0 in the fields")
    reference = values[zero[0]]
    return pd.Series((values - reference) / reference, index=sigma.index, name="magnetoresistance")


def carrier_density(
    df: pd.DataFrame,
    kx: str = "kx",
    ky: str = "ky",
    vx: str = "vx",
    vy: str = "vy",
    *,
    layer_spacing: float,
    spin_degeneracy: float = 2,
    kz: Optional[str] = None,
) -> float:
    """Carrier density of one closed pocket from the k-space area it encloses (m⁻³).

    n = g_s A / (4π² d): the Luttinger count per layer divided by the layer spacing, or
    for a k_z-warped pocket (`kz` given) its average over the k_z slices, which is the
    enclosed volume g_s V/(8π³). For a hole-like pocket this is the density of holes.
    The area uses the velocities for the contour's curvature, so it is accurate to
    O(N⁻⁴) even for irregular sampling.

    Args:
        df: One contour DataFrame (nodes in order around the pocket).
        kx: Column of wavevectors k_x (m⁻¹).
        ky: Column of wavevectors k_y (m⁻¹).
        vx: Column of group velocities v_x (m/s).
        vy: Column of group velocities v_y (m/s).
        layer_spacing: Interlayer spacing d (m).
        spin_degeneracy: Spin degeneracy g_s.
        kz: Column of wavevectors k_z (m⁻¹), for a k_z-warped pocket given as slices.

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
    slices = [df] if kz is None else _kz_slices(df, kz, layer_spacing)
    areas = []
    for part in slices:
        contour = prepare_contour(part[kx], part[ky], part[vx], part[vy], 1.0)  # validates; τ plays no part
        areas.append(enclosed_area(contour.kx, contour.ky, contour.vx, contour.vy))
    return spin_degeneracy * float(np.mean(areas)) / (4 * np.pi**2 * layer_spacing)
