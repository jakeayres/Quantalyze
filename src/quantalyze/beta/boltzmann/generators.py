"""Fermi-surface contours as DataFrames.

Two kinds of generator:

- **From your own band:** `from_dispersion` traces the pocket ε(k) = 0 of any
  dispersion you supply, `open_sheets_from_dispersion` traces open Fermi sheets
  (one period of each, ready for `period=`), and `polar` builds a pocket of any
  shape from k_F(φ).
- **Analytic test pockets:** `circle`, `ellipse`, `tight_binding` and
  `open_sheets`, whose exact answers are known.

Every generator returns the columns `kx`, `ky` (m⁻¹), `vx`, `vy` (m/s) and
`tau` (s), with the velocity the true group velocity v = ∇ε/ħ at each node.
Closed contours are returned as N distinct nodes (no repeated closing point),
in counter-clockwise order of the polar angle about the pocket centre.

`tau` may be a float, or a function of the polar angle φ (rad) of each node
about the pocket centre, such as the models in
`quantalyze.beta.boltzmann.scattering`. Open sheets have no centre, so
`open_sheets_from_dispersion` takes a function of k instead.

Examples:
    >>> from quantalyze.beta import boltzmann as bz
    >>> from quantalyze.core.constants import ELECTRON_MASS
    >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
    >>> list(df.columns)
    ['kx', 'ky', 'vx', 'vy', 'tau']
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ...core.constants import HBAR

Tau = Union[float, Callable[[np.ndarray], np.ndarray]]

# Samples along each ray that check it crosses the Fermi surface once and bracket the crossing.
_RAY_SAMPLES = 64
# Samples along each line across open sheets: finer, since one line crosses several sheets.
_LINE_SAMPLES = 256

_CARRIER_SIGN = {"electron": 1.0, "hole": -1.0}


def _carrier_sign(carrier: str) -> float:
    try:
        return _CARRIER_SIGN[carrier]
    except KeyError:
        raise ValueError(f"carrier must be 'electron' or 'hole', not {carrier!r}") from None


def _polar_angles(n_points: int) -> np.ndarray:
    return 2 * np.pi * np.arange(n_points) / n_points  # (N,)


def _tau_column(tau: Tau, phi: np.ndarray) -> np.ndarray:
    if callable(tau):
        values = np.asarray(tau(phi), dtype=np.float64)
    else:
        values = np.full(phi.shape, float(tau))
    return np.broadcast_to(values, phi.shape).astype(np.float64)  # (N,)


def _frame(kx, ky, vx, vy, tau) -> pd.DataFrame:
    return pd.DataFrame({
        "kx": np.asarray(kx, dtype=np.float64),
        "ky": np.asarray(ky, dtype=np.float64),
        "vx": np.asarray(vx, dtype=np.float64),
        "vy": np.asarray(vy, dtype=np.float64),
        "tau": np.asarray(tau, dtype=np.float64),
    })


def circle(n_points: int, *, k_fermi: float, mass: float, tau: Tau, carrier: str = "electron") -> pd.DataFrame:
    """Circular pocket of a parabolic band, ε = ±ħ²(k² − k_F²)/2m.

    For electrons the velocity v = ħk/m points outwards; for holes the band is
    inverted, so v = −ħk/m points inwards.

    Args:
        n_points: Number of nodes N, evenly spaced in angle.
        k_fermi: Fermi wavevector k_F (m⁻¹).
        mass: Band mass m (kg).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad).
        carrier: "electron" or "hole".

    Returns:
        DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13, carrier="hole")
    """
    sign = _carrier_sign(carrier)
    phi = _polar_angles(n_points)
    kx, ky = k_fermi * np.cos(phi), k_fermi * np.sin(phi)
    speed = sign * HBAR * k_fermi / mass
    return _frame(kx, ky, speed * np.cos(phi), speed * np.sin(phi), _tau_column(tau, phi))


def ellipse(
    n_points: int,
    *,
    k_fermi: float,
    mass_x: float,
    mass_y: float,
    tau: Tau,
    rotation: float = 0.0,
    carrier: str = "electron",
) -> pd.DataFrame:
    """Elliptical pocket of an anisotropic parabolic band.

    In its principal frame ε = ±[ħ²k_x²/2m_x + ħ²k_y²/2m_y − ε_F], with ε_F chosen so
    that the enclosed area is πk_F² (the same carrier density as a circle of radius
    k_F). The semi-axes are k_F(m_x/m_y)^¼ along x and k_F(m_y/m_x)^¼ along y, and
    v = ±ħ(k_x/m_x, k_y/m_y). The whole pocket (k and v) is then rotated by `rotation`
    about ẑ. Nodes are evenly spaced in the ellipse's parametric angle.

    Args:
        n_points: Number of nodes N.
        k_fermi: Geometric-mean Fermi wavevector √(k_a k_b) (m⁻¹).
        mass_x: Band mass along the principal x axis (kg).
        mass_y: Band mass along the principal y axis (kg).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad).
        rotation: Angle of the principal x axis from the lab x axis (rad).
        carrier: "electron" or "hole".

    Returns:
        DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.ellipse(512, k_fermi=7e9, mass_x=ELECTRON_MASS,
        ...                            mass_y=4 * ELECTRON_MASS, tau=1e-13)
    """
    sign = _carrier_sign(carrier)
    t = _polar_angles(n_points)
    ratio = (mass_x / mass_y) ** 0.25
    qx, qy = k_fermi * ratio * np.cos(t), k_fermi / ratio * np.sin(t)  # principal frame
    ux, uy = sign * HBAR * qx / mass_x, sign * HBAR * qy / mass_y
    c, s = np.cos(rotation), np.sin(rotation)
    kx, ky = c * qx - s * qy, s * qx + c * qy
    vx, vy = c * ux - s * uy, s * ux + c * uy
    return _frame(kx, ky, vx, vy, _tau_column(tau, np.arctan2(ky, kx)))


def _spectral_derivative(values: np.ndarray) -> np.ndarray:
    """d/dφ of a smooth periodic function sampled at φ_j = 2πj/N."""
    n = values.size
    wavenumber = np.fft.fftfreq(n, d=1.0 / n)  # integers 0, 1, ..., −1
    if n % 2 == 0:
        wavenumber[n // 2] = 0.0  # Nyquist mode has no well-defined derivative
    return np.fft.ifft(1j * wavenumber * np.fft.fft(values)).real


def polar(
    n_points: int,
    *,
    k_fermi: Callable[[np.ndarray], np.ndarray],
    mass: float,
    tau: Tau,
    dk_fermi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    carrier: str = "electron",
) -> pd.DataFrame:
    """Star-shaped pocket of any shape, given its Fermi wavevector k_F(φ).

    The dispersion is ε = ±ħ²(k² − k_F(φ)²)/2m, so on the contour
    v = ±(ħ/m)(k_F r̂ − k_F′(φ) φ̂), which is normal to the contour but not radial.
    Energy contours of this band enclose A(ε) = A_F + 2πmε/ħ², so m is exactly the
    cyclotron mass of the pocket whatever its shape.

    Args:
        n_points: Number of nodes N, evenly spaced in polar angle.
        k_fermi: Function returning k_F(φ) (m⁻¹) for an array of angles φ (rad).
            It must be positive and 2π-periodic.
        mass: Cyclotron mass m (kg).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad).
        dk_fermi: Function returning dk_F/dφ (m⁻¹ per rad). If omitted, it is found by
            spectral differentiation of the N samples of k_F, which is accurate to
            rounding when k_F(φ) is smooth and resolved by the N points.
        carrier: "electron" (v outwards) or "hole" (v inwards).

    Returns:
        DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Raises:
        ValueError: If k_F(φ) is not positive at every node.

    Examples:
        A rounded-square pocket, k_F(φ) = k₀ − k₄ cos4φ:

        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.polar(
        ...     512, k_fermi=lambda phi: 7.35e9 - 0.25e9 * np.cos(4 * phi),
        ...     mass=5 * ELECTRON_MASS, tau=1e-13, carrier="hole")
    """
    sign = _carrier_sign(carrier)
    phi = _polar_angles(n_points)
    kf = np.broadcast_to(np.asarray(k_fermi(phi), dtype=np.float64), phi.shape)  # (N,)
    if not np.all(kf > 0):
        raise ValueError("k_fermi(φ) must be positive at every angle")
    if dk_fermi is None:
        dkf = _spectral_derivative(kf)  # (N,)
    else:
        dkf = np.broadcast_to(np.asarray(dk_fermi(phi), dtype=np.float64), phi.shape)
    cos, sin = np.cos(phi), np.sin(phi)
    v_radial = sign * HBAR * kf / mass
    v_angular = -sign * HBAR * dkf / mass
    vx = v_radial * cos - v_angular * sin
    vy = v_radial * sin + v_angular * cos
    return _frame(kf * cos, kf * sin, vx, vy, _tau_column(tau, phi))


def from_dispersion(
    n_points: int,
    *,
    energy: Callable[[np.ndarray, np.ndarray], np.ndarray],
    gradient: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    tau: Tau,
    max_radius: Union[float, Callable[[float], float]],
    center: Sequence[float] = (0.0, 0.0),
) -> pd.DataFrame:
    """Closed pocket of any band ε(k), traced by root-finding along rays.

    The Fermi contour ε(k) = 0 about `center` is found along N rays evenly spaced in
    polar angle, so the pocket must be star-shaped about the centre (each ray crosses
    it exactly once) and lie within `max_radius` of it. All rays are solved together
    (Newton steps along each ray from the gradient, safeguarded by bisection), to the
    precision of the floating-point numbers. The velocity is v = ∇ε/ħ from the supplied
    gradient, so whether the pocket is electron- or hole-like follows from the band itself.

    Args:
        n_points: Number of nodes N, evenly spaced in polar angle about `center`.
        energy: Function ε(k_x, k_y) (J) measured from the Fermi level, vectorised
            over NumPy arrays of k_x, k_y (m⁻¹).
        gradient: Function returning (∂ε/∂k_x, ∂ε/∂k_y) (J·m) for arrays k_x, k_y (m⁻¹).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad)
            about `center`.
        max_radius: How far from the centre to search along each ray (m⁻¹), as a float
            or a function of φ (rad). The pocket must close within it.
        center: Pocket centre (k_x, k_y) (m⁻¹).

    Returns:
        DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Raises:
        ValueError: If the pocket is not closed within `max_radius`, or not star-shaped,
            about `center`, or the energy is not finite along a ray.

    Examples:
        A parabolic band with a fourfold quartic correction:

        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS, HBAR
        >>> c2, c4, ef = HBAR**2 / (2 * ELECTRON_MASS), 2e-59, 1.9e-19
        >>> df = bz.generators.from_dispersion(
        ...     512, tau=1e-13, max_radius=2e10,
        ...     energy=lambda kx, ky: c2 * (kx**2 + ky**2) + c4 * (kx**4 + ky**4) - ef,
        ...     gradient=lambda kx, ky: (2 * c2 * kx + 4 * c4 * kx**3, 2 * c2 * ky + 4 * c4 * ky**3))
    """
    cx, cy = (float(c) for c in center)
    energy_centre = float(energy(np.array(cx), np.array(cy)))
    if not np.isfinite(energy_centre):
        raise ValueError(f"energy is {energy_centre} at the pocket centre; it must be finite there")
    if energy_centre == 0:
        raise ValueError("the Fermi level passes through the pocket centre")

    phi = _polar_angles(n_points)
    if callable(max_radius):
        reach = np.array([float(max_radius(angle)) for angle in phi])  # (N,)
    else:
        reach = np.full(n_points, float(max_radius))
    radius = _ray_crossings(energy, gradient, phi, reach, (cx, cy), energy_centre)  # |k − centre|, (N,)

    kx = cx + radius * np.cos(phi)
    ky = cy + radius * np.sin(phi)
    grad_x, grad_y = gradient(kx, ky)
    return _frame(kx, ky, np.asarray(grad_x) / HBAR, np.asarray(grad_y) / HBAR, _tau_column(tau, phi))


def _ray_crossings(energy, gradient, phi, reach, centre, energy_centre) -> np.ndarray:
    """Distance from the centre to the Fermi crossing on each ray (m⁻¹), shape (N,).

    Each ray is sampled at `_RAY_SAMPLES` evenly spaced points out to its reach, which
    checks that it crosses ε = 0 exactly once and brackets the crossing. Then every ray
    is refined at once by Newton steps along the ray, dε/dr = ∇ε·r̂ from the supplied
    gradient, falling back to bisection whenever a step would leave the bracket or is not
    shrinking it fast enough, so it converges even if the gradient is poor. The tolerance
    is scipy's brentq's: 1e-15 of the reach plus 4 ulp of the radius.
    """
    cx, cy = centre
    n_rays = phi.size
    ux, uy = np.cos(phi), np.sin(phi)
    radii = reach[:, None] * np.linspace(0.0, 1.0, _RAY_SAMPLES + 1)  # column 0 is the centre, (N, S+1)
    values = np.empty(radii.shape)
    values[:, 0] = energy_centre
    along = energy((cx + radii[:, 1:] * ux[:, None]).ravel(), (cy + radii[:, 1:] * uy[:, None]).ravel())
    values[:, 1:] = np.asarray(along, dtype=np.float64).reshape(n_rays, _RAY_SAMPLES)
    bad = np.flatnonzero(~np.all(np.isfinite(values), axis=1))
    if bad.size:
        raise ValueError(f"energy is not finite along the ray at φ = {phi[bad[0]]:.3f} rad, within max_radius")

    # A sample exactly on the Fermi surface (ε = 0) is itself the crossing: give it the sign
    # of the sample before it, rather than count a change into and out of zero.
    signs = np.sign(values)
    last = np.maximum.accumulate(np.where(signs != 0, np.arange(_RAY_SAMPLES + 1), 0), axis=1)
    filled = np.take_along_axis(signs, last, axis=1)
    changes = filled[:, 1:] != filled[:, :-1]  # (N, S)
    crossings = np.count_nonzero(changes, axis=1)
    bad = np.flatnonzero(crossings != 1)
    if bad.size:
        angle = phi[bad[0]]
        if crossings[bad[0]] == 0:
            raise ValueError(
                f"no Fermi crossing within max_radius of the centre at φ = {angle:.3f} rad: "
                "the pocket is open, absent or larger than max_radius"
            )
        raise ValueError(f"the pocket is not star-shaped about the centre (φ = {angle:.3f} rad)")

    rays = np.arange(n_rays)
    after = np.argmax(changes, axis=1) + 1  # first sample past the crossing
    lower = last[rays, after - 1]  # last non-zero sample before it
    lo, hi = radii[rays, lower], radii[rays, after]
    f_lo = values[rays, lower]
    return _refine(energy, gradient, cx, cy, ux, uy, lo, hi, f_lo, 1e-15 * reach)


def _refine(energy, gradient, cx, cy, ux, uy, lo, hi, f_lo, tolerance) -> np.ndarray:
    """The distance r along k = c + r û at which ε = 0, given brackets [lo, hi] (m⁻¹).

    All crossings are refined at once, by safeguarded Newton steps ("rtsafe"): dε/dr = ∇ε·û
    from the supplied gradient, with a bisection whenever the Newton step leaves [lo, hi] or
    would shrink the bracket less than halving it did two steps ago. The tolerance is
    scipy's brentq's: `tolerance` plus 4 ulp of r.
    """
    step_before = hi - lo
    step = step_before.copy()
    r = 0.5 * (lo + hi)
    for _ in range(200):
        kx, ky = cx + r * ux, cy + r * uy
        f = np.asarray(energy(kx, ky), dtype=np.float64)
        gx, gy = gradient(kx, ky)
        slope = np.asarray(gx, dtype=np.float64) * ux + np.asarray(gy, dtype=np.float64) * uy
        beyond = np.sign(f) == np.sign(f_lo)  # the crossing is further out than r
        lo, f_lo = np.where(beyond, r, lo), np.where(beyond, f, f_lo)
        hi = np.where(beyond, hi, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = r - f / slope
            usable = (np.isfinite(newton) & (newton >= lo) & (newton <= hi)
                      & (np.abs(2 * f) <= np.abs(step_before * slope)))
        step_before = step
        step = np.where(usable, r - newton, r - 0.5 * (lo + hi))
        converged = (np.abs(step) <= tolerance + 4 * np.finfo(float).eps * np.abs(r)) | (f == 0)
        r = np.where(f == 0, r, r - step)
        if np.all(converged):
            return r
    raise RuntimeError("root-finding along the rays did not converge")  # bisection alone takes ~60 steps


def _sign_crossings(values):
    """Where the sign of each row changes, ignoring samples exactly on ε = 0: returns the
    change mask, shape (rows, S − 1), and the index of the last non-zero sample so far."""
    signs = np.sign(values)
    columns = np.arange(values.shape[1])
    first = np.argmax(signs != 0, axis=1)  # leading zeros take the first non-zero sign
    last = np.maximum.accumulate(np.where(signs != 0, columns, 0), axis=1)
    last = np.maximum(last, first[:, None])
    filled = np.take_along_axis(signs, last, axis=1)
    return filled[:, 1:] != filled[:, :-1], last


def _line_crossings(energy, gradient, starts_x, starts_y, nx, ny, across):
    """Every crossing of ε = 0 on each line k = start + r n̂, r ∈ `across` (m⁻¹).

    Returns the distances r, shape (lines, sheets), in increasing order along each line.
    Every line must cross the same number of times (at least once).
    """
    lo_r, hi_r = across
    n_lines = starts_x.size
    radii = np.linspace(lo_r, hi_r, _LINE_SAMPLES + 1)  # (S,)
    grid_x = starts_x[:, None] + radii[None, :] * nx
    grid_y = starts_y[:, None] + radii[None, :] * ny
    values = np.asarray(energy(grid_x.ravel(), grid_y.ravel()), dtype=np.float64).reshape(grid_x.shape)
    bad = np.flatnonzero(~np.all(np.isfinite(values), axis=1))
    if bad.size:
        raise ValueError(f"energy is not finite on the line through k = ({starts_x[bad[0]]:.4e}, "
                         f"{starts_y[bad[0]]:.4e}) m⁻¹, within `across`")
    if np.any(np.all(values == 0, axis=1)):
        raise ValueError("energy is zero all along a line: the Fermi level lies on a flat band")
    changes, last = _sign_crossings(values)
    counts = np.count_nonzero(changes, axis=1)
    if counts.min() == 0:
        raise ValueError("some lines across the sheets do not cross the Fermi surface within `across`: "
                         "widen `across`, or check the band and chemical potential")
    if counts.min() != counts.max():
        raise ValueError(
            f"the number of Fermi crossings changes along the period (from {counts.min()} to {counts.max()}): "
            "the surface is not a set of open sheets across this range (a closed pocket, sheets that "
            "touch, or `across` cutting through a sheet). Narrow `across` to the sheets you want"
        )
    n_sheets = counts[0]
    lines, samples = np.nonzero(changes)  # row-major: each line's crossings in order, (lines·sheets,)
    after = samples + 1
    lower = last[lines, samples]
    starts_x, starts_y = starts_x[lines], starts_y[lines]
    r = _refine(energy, gradient, starts_x, starts_y, nx, ny, radii[lower], radii[after],
                values[lines, lower], np.full(lines.size, 1e-15 * (hi_r - lo_r)))
    return r.reshape(n_lines, n_sheets)


def from_dispersion_3d(
    n_points: int,
    n_kz: int,
    *,
    energy: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    gradient: Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tau: Tau,
    max_radius: Union[float, Callable[[float], float]],
    layer_spacing: float,
    center: Sequence[float] = (0.0, 0.0),
) -> pd.DataFrame:
    """A k_z-warped Fermi surface as slices: `from_dispersion` at evenly spaced k_z.

    The slices sit at k_z = −π/d + 2πj/(N_z d), j = 0…N_z−1, one period of k_z, as
    `bz.conductivity(..., kz="kz")` expects. With B along ẑ each carrier stays in its
    slice, carrying the out-of-plane velocity v_z = (1/ħ) ∂ε/∂k_z with it.

    Args:
        n_points: Number of nodes N per slice, evenly spaced in polar angle about `center`.
        n_kz: Number of slices N_z. The k_z average is a periodic trapezoid rule, so a
            few points per period of the warping are usually enough.
        energy: Function ε(k_x, k_y, k_z) (J) measured from the Fermi level, vectorised
            over arrays of k_x, k_y (m⁻¹) at one k_z (m⁻¹).
        gradient: Function returning (∂ε/∂k_x, ∂ε/∂k_y, ∂ε/∂k_z) (J·m).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad)
            about `center`.
        max_radius: How far from the centre to search along each ray (m⁻¹), a float or
            a function of φ (rad).
        layer_spacing: Interlayer spacing d (m), which sets the k_z period 2π/d.
        center: Pocket centre (k_x, k_y) (m⁻¹), the same for every slice.

    Returns:
        DataFrame with columns kx, ky, kz (m⁻¹), vx, vy, vz (m/s) and tau (s).

    Raises:
        ValueError: If a slice's pocket is not closed, or not star-shaped, about `center`.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS, HBAR
        >>> c2, tz, d, ef = HBAR**2 / (2 * ELECTRON_MASS), 1e-21, 1e-9, 1.9e-19
        >>> df = bz.generators.from_dispersion_3d(
        ...     256, 8, tau=1e-13, max_radius=2e10, layer_spacing=d,
        ...     energy=lambda kx, ky, kz: c2 * (kx**2 + ky**2) - 2 * tz * np.cos(kz * d) - ef,
        ...     gradient=lambda kx, ky, kz: (2 * c2 * kx, 2 * c2 * ky, 2 * tz * d * np.sin(kz * d) + 0 * kx))
    """
    slices = []
    for j in range(n_kz):
        kz = -np.pi / layer_spacing + 2 * np.pi * j / (n_kz * layer_spacing)
        part = from_dispersion(
            n_points, tau=tau, max_radius=max_radius, center=center,
            energy=lambda kx, ky, kz=kz: energy(kx, ky, kz),
            gradient=lambda kx, ky, kz=kz: gradient(kx, ky, kz)[:2],
        )
        grad_z = np.broadcast_to(np.asarray(gradient(part["kx"].to_numpy(), part["ky"].to_numpy(), kz)[2]),
                                 (len(part),))
        part.insert(2, "kz", kz)
        part.insert(5, "vz", grad_z / HBAR)
        slices.append(part)
    return pd.concat(slices, ignore_index=True)


def open_sheets_from_dispersion(
    n_points: int,
    *,
    energy: Callable[..., np.ndarray],
    gradient: Callable[..., Tuple[np.ndarray, ...]],
    period: Sequence[float],
    across: Sequence[float],
    tau: Union[float, Callable[..., np.ndarray]],
    n_kz: Optional[int] = None,
    layer_spacing: Optional[float] = None,
) -> Tuple[list, Tuple[float, float]]:
    """Open Fermi sheets of any band ε(k): one period of each, ready for `conductivity`.

    The sheets run along the reciprocal-lattice vector G = `period`. The function steps
    along one period, k_∥ = −|G|/2 … |G|/2 in N even steps, and on each line across the
    sheets (along n̂, G turned by +90°) finds every crossing of ε = 0 with
    r = k·n̂ in `across`. The first crossing on every line belongs to sheet 1, the second
    to sheet 2, and so on (ordered along n̂). The velocity is v = ∇ε/ħ from the supplied
    gradient, and the crossings are found to the precision of the floating-point numbers.

    With `n_kz`, the same is done on N_z evenly spaced slices k_z = −π/d + 2πj/(N_z d) of
    a band warped along k_z, and each sheet gets `kz` and `vz` columns: pass
    `kz="kz"` to `conductivity`.

    Args:
        n_points: Number of nodes N per period along each sheet (per slice).
        energy: Function ε(k_x, k_y) (J) measured from the Fermi level, vectorised over
            NumPy arrays of k_x, k_y (m⁻¹); ε(k_x, k_y, k_z) with `n_kz`.
        gradient: Function returning (∂ε/∂k_x, ∂ε/∂k_y) (J·m); with `n_kz`, also ∂ε/∂k_z.
        period: Reciprocal-lattice vector (G_x, G_y) (m⁻¹) along which the sheets repeat,
            e.g. (2π/a, 0) for sheets running along k_x.
        across: Range (r_min, r_max) (m⁻¹) of k·n̂ to search across the sheets, n̂ being G
            turned by +90° (for G along +k_x, n̂ is +k_y). Every line must cross the Fermi
            surface the same number of times within it.
        tau: Relaxation time (s): a float, or a function τ(k_x, k_y) of arrays (m⁻¹),
            τ(k_x, k_y, k_z) with `n_kz`.
        n_kz: Number of k_z slices N_z, for a band warped along k_z; None for a 2D band.
        layer_spacing: Interlayer spacing d (m), which sets the k_z period 2π/d. Required
            with `n_kz`.

    Returns:
        (sheets, period): a list with one DataFrame per sheet (columns kx, ky, vx, vy, tau;
        kx, ky, kz, vx, vy, vz, tau with `n_kz`), and the period (G_x, G_y) to pass to
        `conductivity`.

    Raises:
        ValueError: If a line does not cross the Fermi surface, the number of crossings
            changes along the period or between slices, the sheets do not repeat after
            `period`, the energy is not finite within `across`, or the arguments are
            inconsistent.

    Examples:
        Quasi-one-dimensional chains along y: two sheets near k_y = ±π/2b that run along k_x.

        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> a = b = 4e-10
        >>> ta, tb = bz.units.ev_to_joule(0.02), bz.units.ev_to_joule(0.25)
        >>> sheets, period = bz.generators.open_sheets_from_dispersion(
        ...     256, period=(2 * np.pi / a, 0.0), across=(-np.pi / b, np.pi / b), tau=1e-13,
        ...     energy=lambda kx, ky: -2 * ta * np.cos(kx * a) - 2 * tb * np.cos(ky * b),
        ...     gradient=lambda kx, ky: (2 * ta * a * np.sin(kx * a), 2 * tb * b * np.sin(ky * b)))
        >>> sigma = bz.conductivity(sheets, [0.0, 10.0], layer_spacing=1e-9, period=period)
    """
    g = np.asarray(period, dtype=np.float64)
    if g.shape != (2,) or not np.all(np.isfinite(g)) or not np.any(g):
        raise ValueError(f"period must be a finite, non-zero pair (G_x, G_y), not {period!r}")
    lo_r, hi_r = (float(v) for v in across)
    if not (np.isfinite(lo_r) and np.isfinite(hi_r) and hi_r > lo_r):
        raise ValueError(f"across must be a finite range (r_min, r_max) with r_max > r_min, not {across!r}")
    if n_kz is not None and layer_spacing is None:
        raise ValueError("layer_spacing is required with n_kz: it sets the k_z period 2π/d")
    length = float(np.hypot(*g))
    ux, uy = g / length  # along the sheets
    nx, ny = -uy, ux  # across them
    along = length * (np.arange(n_points) / n_points - 0.5)  # (N,)
    starts_x, starts_y = along * ux, along * uy

    def build(energy2, gradient2, kz=None):
        r = _line_crossings(energy2, gradient2, starts_x, starts_y, nx, ny, (lo_r, hi_r))  # (N, sheets)
        # The line one period on must cross where line j does. j is generic: at the ends of the
        # period an inversion-symmetric band would match whatever G is.
        j = int(0.382 * n_points)
        try:
            shifted = _line_crossings(energy2, gradient2, starts_x[j:j + 1] + g[0], starts_y[j:j + 1] + g[1],
                                      nx, ny, (lo_r, hi_r))
        except ValueError:
            shifted = None
        if shifted is None or shifted.shape[1] != r.shape[1] or np.max(np.abs(shifted[0] - r[j])) > 1e-8 * (hi_r - lo_r):
            raise ValueError("the sheets do not repeat after `period`: pass the reciprocal-lattice vector G "
                             "with ε(k + G) = ε(k)")
        frames = []
        for sheet in range(r.shape[1]):
            kx, ky = starts_x + r[:, sheet] * nx, starts_y + r[:, sheet] * ny
            grad = gradient(kx, ky) if kz is None else gradient(kx, ky, kz)
            if callable(tau):
                values = tau(kx, ky) if kz is None else tau(kx, ky, np.full(kx.shape, kz))
            else:
                values = float(tau)
            frame = _frame(kx, ky, np.asarray(grad[0]) / HBAR, np.asarray(grad[1]) / HBAR,
                           np.broadcast_to(np.asarray(values, dtype=np.float64), kx.shape))
            if kz is not None:
                frame.insert(2, "kz", kz)
                frame.insert(5, "vz", np.broadcast_to(np.asarray(grad[2], dtype=np.float64), kx.shape) / HBAR)
            frames.append(frame)
        return frames

    if n_kz is None:
        return build(energy, gradient), (float(g[0]), float(g[1]))
    per_slice = []
    for j in range(n_kz):
        kz = -np.pi / layer_spacing + 2 * np.pi * j / (n_kz * layer_spacing)
        per_slice.append(build(lambda kx, ky, kz=kz: energy(kx, ky, kz),
                               lambda kx, ky, kz=kz: gradient(kx, ky, kz)[:2], kz))
    counts = {len(frames) for frames in per_slice}
    if len(counts) != 1:
        raise ValueError(f"the number of sheets changes between k_z slices ({sorted(counts)}): "
                         "the surface is not a set of open sheets at every k_z")
    sheets = [pd.concat([frames[i] for frames in per_slice], ignore_index=True) for i in range(len(per_slice[0]))]
    return sheets, (float(g[0]), float(g[1]))


def tight_binding(
    n_points: int,
    *,
    tau: Tau,
    lattice_constant: float,
    hopping: float,
    chemical_potential: float,
    next_hopping: float = 0.0,
    third_hopping: float = 0.0,
    center: Sequence[float] = (0.0, 0.0),
) -> pd.DataFrame:
    """Closed pocket of a square-lattice tight-binding band.

    ε(k) = −2t(cos k_xa + cos k_ya) − 4t′ cos k_xa cos k_ya − 2t″(cos 2k_xa + cos 2k_ya) − μ.

    A convenience wrapper around `from_dispersion`: the pocket must be star-shaped
    about `center` and closed inside the square of half-width π/a around it.

    Args:
        n_points: Number of nodes N, evenly spaced in polar angle about `center`.
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad)
            about `center`.
        lattice_constant: Lattice constant a (m).
        hopping: Nearest-neighbour hopping t (J).
        chemical_potential: Chemical potential μ (J).
        next_hopping: Next-nearest-neighbour hopping t′ (J).
        third_hopping: Third-neighbour hopping t″ (J).
        center: Pocket centre (k_x, k_y) (m⁻¹), e.g. (π/a, π/a) for a hole pocket
            about the zone corner.

    Returns:
        DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Raises:
        ValueError: If the pocket is not closed, or not star-shaped, about `center`.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> a = bz.units.angstrom_to_meter(3.87)
        >>> df = bz.generators.tight_binding(
        ...     512, tau=1e-13, lattice_constant=a,
        ...     hopping=bz.units.ev_to_joule(0.25), next_hopping=bz.units.ev_to_joule(-0.0625),
        ...     chemical_potential=0.0, center=(np.pi / a, np.pi / a))
    """
    a = lattice_constant
    t1, t2, t3, mu = hopping, next_hopping, third_hopping, chemical_potential

    def energy(kx, ky):
        x, y = kx * a, ky * a
        return (
            -2 * t1 * (np.cos(x) + np.cos(y))
            - 4 * t2 * np.cos(x) * np.cos(y)
            - 2 * t3 * (np.cos(2 * x) + np.cos(2 * y))
            - mu
        )

    def gradient(kx, ky):
        x, y = kx * a, ky * a
        return (
            a * (2 * t1 * np.sin(x) + 4 * t2 * np.sin(x) * np.cos(y) + 4 * t3 * np.sin(2 * x)),
            a * (2 * t1 * np.sin(y) + 4 * t2 * np.cos(x) * np.sin(y) + 4 * t3 * np.sin(2 * y)),
        )

    def half_cell(angle):  # distance to the edge of the square of half-width π/a
        return (np.pi / a) / max(abs(np.cos(angle)), abs(np.sin(angle)))

    return from_dispersion(n_points, energy=energy, gradient=gradient, tau=tau,
                           max_radius=half_cell, center=center)


def open_sheets(
    n_points: int,
    *,
    k0: float,
    velocity: float,
    tau: Tau,
    period: float,
    warping: float = 0.0,
) -> list:
    """A pair of open Fermi sheets near k_x = ±k0, periodic in k_y.

    Sheet s = ±1 is the zero of ε = ħv₀(s k_x − k0 − δ cos(2πk_y/G)), so it sits at
    k_x = s(k0 + δ cos(2πk_y/G)) with v = (s v₀, v₀ δ (2π/G) sin(2πk_y/G)). With δ = 0
    the sheets are flat and v = ±v₀ x̂. Each sheet has N nodes at k_y = −G/2 + jG/N,
    j = 0…N−1, so its last segment ends at the first node shifted by G ŷ.

    Args:
        n_points: Number of nodes N per sheet.
        k0: Mean distance of the sheets from k_x = 0 (m⁻¹).
        velocity: Speed v₀ normal to the flat sheets (m/s).
        tau: Relaxation time (s), as a float or a function of the polar angle φ (rad)
            of each node about the origin.
        period: Reciprocal-lattice period G along k_y (m⁻¹).
        warping: Warping amplitude δ (m⁻¹).

    Returns:
        List of two DataFrames, the sheet at +k0 then the sheet at −k0, each with
        columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s).

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> plus, minus = bz.generators.open_sheets(512, k0=5e9, velocity=2e5, tau=1e-13,
        ...                                          period=2 * np.pi / 3.87e-10)
    """
    b = 2 * np.pi / period
    ky = period * (np.arange(n_points) / n_points - 0.5)  # (N,)
    sheets = []
    for side in (1.0, -1.0):
        kx = side * (k0 + warping * np.cos(b * ky))
        vx = np.full(n_points, side * velocity)
        vy = velocity * warping * b * np.sin(b * ky)
        sheets.append(_frame(kx, ky, vx, vy, _tau_column(tau, np.arctan2(ky, kx))))
    return sheets
