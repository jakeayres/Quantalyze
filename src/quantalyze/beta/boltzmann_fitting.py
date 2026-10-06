"""Fit Boltzmann-transport models to measured magnetotransport.

`quantalyze.beta.boltzmann` computes ρ_xx(B) and the Hall resistivity from a Fermi surface
and a relaxation time. This module runs it backwards: from measured ρ_xx(B) and Hall
resistivity ρ_H(B), it finds the parameters of a scattering rate 1/τ(k) on a known Fermi
surface, the parameters of the Fermi surface itself, or both at once.

Models are plain Python functions, and their argument names say what they need:

- `rate` returns the scattering rate 1/τ (s⁻¹) at every point of the Fermi surface. An
  argument named after a column of the surface DataFrame (`kx`, `ky`, `vx`, `vy`, or one
  you add, such as `phi` from `polar_angle`) receives that column as an array. Every other
  argument is a parameter to fit: `def rate(gamma)` is an isotropic rate, and
  `def rate(phi, gamma0, a4)` a fourfold one.
- `surface` is either a fixed contour DataFrame (or a list of them, one per pocket), or a
  function of parameters to fit that builds one, for example with
  `bz.generators.tight_binding`. Without a `rate`, its own `tau` column is used.

An argument with a default value is held at that value unless `p0` gives it a starting
value. Arguments with the same name in `surface` and `rate` are the same parameter.

The Hall resistivity is ρ_H = R_H B = ½(ρ_yx − ρ_xy): the antisymmetric part of E_y/j_x
for B along +ẑ, as a Hall measurement V_y t/I_x gives it. It is negative for electrons.

Examples:
    >>> import numpy as np
    >>> from quantalyze.beta import boltzmann as bz
    >>> from quantalyze.beta import boltzmann_fitting as bzf
    >>> from quantalyze.core.constants import ELECTRON_MASS
    >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=2 * ELECTRON_MASS, tau=1.0)
    >>> surface["phi"] = bzf.polar_angle(surface)
    >>> def rate(phi, gamma0, a4):
    ...     return gamma0 * (1 + a4 * np.cos(4 * phi))
    >>> data = bzf.transport(surface, np.linspace(0, 30, 31), rate=rate,
    ...                      parameters={"gamma0": 5e12, "a4": 0.4}, layer_spacing=1e-9)
    >>> result = bzf.fit_transport(data, surface, rate=rate, p0={"gamma0": 1e12, "a4": 0.0},
    ...                            bounds={"a4": (-0.95, 0.95)}, layer_spacing=1e-9)
    >>> result["a4"]  # ≈ 0.4
"""
from __future__ import annotations

import inspect
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import boltzmann as bz

Surface = Union[pd.DataFrame, List[pd.DataFrame], Callable[..., Union[pd.DataFrame, List[pd.DataFrame]]]]
Rate = Union[Callable[..., np.ndarray], Sequence[Optional[Callable[..., np.ndarray]]], None]

_REQUIRED = inspect.Parameter.empty
# Two parameters more correlated than this are not determined separately by the data.
_CORRELATION_WARNING = 0.99
# Two starts whose χ² differ by less than this (in units where the best fit has χ²_ν = 1
# when no errors are given) fit the data equally well.
_EQUALLY_GOOD = 1.0
# ... and they are different solutions if some parameter differs by more than this many σ.
_DIFFERENT_SOLUTION = 3.0


def polar_angle(df: pd.DataFrame, center: Sequence[float] = (0.0, 0.0), *, kx: str = "kx", ky: str = "ky") -> pd.Series:
    """Polar angle φ of each Fermi-surface point about a pocket centre (rad), in [0, 2π).

    Add it as a column of the surface, and a `rate` with an argument named `phi` receives it.

    Args:
        df: A contour DataFrame.
        center: Pocket centre (k_x, k_y) (m⁻¹), e.g. (π/a, π/a) for a pocket about the zone corner.
        kx: Column of wavevectors k_x (m⁻¹).
        ky: Column of wavevectors k_y (m⁻¹).

    Returns:
        Series named "phi" (rad), aligned with `df`.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> df = bz.generators.circle(256, k_fermi=7e9, mass=ELECTRON_MASS, tau=1e-13)
        >>> df["phi"] = bzf.polar_angle(df)
    """
    angle = np.arctan2(df[ky].to_numpy(dtype=np.float64) - center[1], df[kx].to_numpy(dtype=np.float64) - center[0])
    return pd.Series(np.mod(angle, 2 * np.pi), index=df.index, name="phi")


def _arguments(function, what: str) -> Dict[str, object]:
    """The named arguments of `function` and their defaults (`_REQUIRED` if none), in order."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        raise TypeError(f"cannot read the arguments of {what}; pass a Python function with named arguments") from None
    arguments = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY):
            raise TypeError(f"{what} must take named arguments: '{parameter}' can't be matched to a column or "
                            "a parameter by name")
        arguments[parameter.name] = parameter.default
    return arguments


def _as_frames(result, what: str):
    """(list of DataFrames, whether it was a single DataFrame)."""
    if isinstance(result, pd.DataFrame):
        return [result], True
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
        raise TypeError(f"{what} returned a tuple, like (sheets, period) from open_sheets_from_dispersion: return "
                        "only the sheets, and pass period=... to the fit")
    if isinstance(result, (list, tuple)) and result and all(isinstance(f, pd.DataFrame) for f in result):
        return list(result), False
    raise TypeError(f"{what} must be a contour DataFrame or a list of them, not {type(result).__name__}")


class InvalidModelError(ValueError):
    """A model gave a scattering rate that is not finite and positive."""


class _Model:
    """A surface and a rate model, and how their arguments map onto columns and parameters.

    Which rate arguments are columns is decided once, on the surface at the starting
    values (`known`), so the parameter list is fixed for the whole fit.
    """

    def __init__(self, surface: Surface, rate: Rate, tau: str, known: Mapping[str, float]):
        self.surface = surface
        self.tau = tau
        self.surface_arguments = _arguments(surface, "surface") if callable(surface) else {}
        frames, self.single = _as_frames(self._build(self._surface_values(known)), "surface")

        rates = list(rate) if isinstance(rate, (list, tuple)) else [rate] * len(frames)
        if len(rates) != len(frames):
            raise ValueError(f"rate must be one function, or a list of {len(frames)} (one per contour), "
                             f"not a list of {len(rates)}")
        self.rates = rates

        # Each rate's arguments: columns of every contour it applies to, or parameters.
        self.columns, self.rate_parameters = {}, {}  # by id of the rate function
        rate_parameters = {}
        for function in {id(f): f for f in rates if f is not None}.values():
            used = [frame for frame, f in zip(frames, rates) if f is function]
            columns, own = [], []
            for name, default in _arguments(function, "rate").items():
                present = [name in frame.columns for frame in used]
                if all(present):
                    columns.append(name)
                elif any(present):
                    raise ValueError(f"rate argument {name!r} is a column of some contours but not others: add it "
                                     "to every contour, or rename the argument")
                else:
                    own.append(name)
                    rate_parameters.setdefault(name, []).append(default)
            self.columns[id(function)] = columns
            self.rate_parameters[id(function)] = own
        if all(f is None for f in rates) and tau not in frames[0].columns:
            raise ValueError(f"without a rate, the surface needs a {tau!r} column")

        # A parameter is optional only if every function that takes it gives a default.
        defaults = {name: [default] for name, default in self.surface_arguments.items()}
        for name, values in rate_parameters.items():
            defaults.setdefault(name, []).extend(values)
        self.parameters = {name: (_REQUIRED if any(v is _REQUIRED for v in values) else values[0])
                           for name, values in defaults.items()}

    def _surface_values(self, known: Mapping[str, float]) -> dict:
        values = {}
        for name, default in self.surface_arguments.items():
            if name in known:
                values[name] = known[name]
            elif default is not _REQUIRED:
                values[name] = default
            else:
                raise ValueError(f"surface needs a value for {name!r}: give it a starting value in p0, or a value "
                                 "in fixed")
        return values

    def _build(self, surface_values: Mapping[str, float]):
        return self.surface(**surface_values) if callable(self.surface) else self.surface

    def values(self, given: Mapping[str, float]) -> dict:
        """Every parameter's value: from `given`, or its default."""
        unknown = sorted(set(given) - set(self.parameters))
        if unknown:
            raise ValueError(f"unknown parameter(s) {unknown}; the model's parameters are {list(self.parameters)}")
        values = {}
        for name, default in self.parameters.items():
            if name in given:
                values[name] = given[name]
            elif default is not _REQUIRED:
                values[name] = default
            else:
                raise ValueError(f"no value for parameter {name!r}; the model's parameters are {list(self.parameters)}")
        return values

    def frames(self, values: Mapping[str, float]) -> list:
        """The contours at these parameter values, with τ = 1/rate where there is a rate."""
        frames, _ = _as_frames(self._build({n: values[n] for n in self.surface_arguments}), "surface")
        if len(frames) != len(self.rates):
            raise ValueError(f"surface returned {len(frames)} contours here, but {len(self.rates)} at the start")
        result = []
        for frame, function in zip(frames, self.rates):
            if function is None:
                result.append(frame)
                continue
            columns = self.columns[id(function)]
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise ValueError(f"surface no longer has the column(s) {missing} that rate takes")
            arguments = {c: frame[c].to_numpy() for c in columns}
            arguments.update({n: values[n] for n in self.rate_parameters[id(function)]})
            gamma = np.broadcast_to(np.asarray(function(**arguments), dtype=np.float64), (len(frame),))
            bad = np.flatnonzero(~(np.isfinite(gamma) & (gamma > 0)))
            if bad.size:
                raise InvalidModelError(
                    f"rate must be finite and positive at every point; it is {gamma[bad[0]]} at row {bad[0]} with "
                    f"parameters {dict(values)}. Use bounds to keep the parameters where the rate is positive")
            result.append(frame.assign(**{self.tau: 1.0 / gamma}))
        return result

    def transport(self, values: Mapping[str, float], field, layer_spacing: float, options: Mapping) -> pd.DataFrame:
        """ρ_xx and the Hall resistivity at each field, from the contours at these values."""
        options = {"symmetrize": False, **options}  # Onsager holds to rounding: no need for the reversed orbit
        sigma = bz.conductivity(self.frames(values), field, layer_spacing=layer_spacing, tau=self.tau, **options)
        rho = bz.resistivity(sigma)
        return pd.DataFrame({
            "field": sigma["field"].to_numpy(),
            "resistivity": rho["rho_xx"].to_numpy(),
            "hall_resistivity": 0.5 * (rho["rho_yx"].to_numpy() - rho["rho_xy"].to_numpy()),
        })


def build_surface(surface: Surface, parameters: Optional[Mapping[str, float]] = None, *, rate: Rate = None,
                  tau: str = "tau"):
    """The Fermi surface a model describes at given parameters, with τ = 1/rate.

    Args:
        surface: A contour DataFrame, a list of them, or a function of parameters returning one.
        parameters: Values of the model's parameters, by name (those with defaults may be left out).
        rate: Scattering rate 1/τ (s⁻¹): a function whose arguments are columns of the surface or
            parameters, or a list of them (one per contour; None keeps that contour's τ). None
            keeps the surface's own τ.
        tau: The column τ (s) is written to.

    Returns:
        The contour DataFrame (or list of them) with the `tau` column set.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=ELECTRON_MASS, tau=1.0)
        >>> surface["phi"] = bzf.polar_angle(surface)
        >>> df = bzf.build_surface(surface, {"gamma0": 1e13, "a4": 0.5},
        ...                        rate=lambda phi, gamma0, a4: gamma0 * (1 + a4 * np.cos(4 * phi)))
    """
    parameters = dict(parameters or {})
    model = _Model(surface, rate, tau, parameters)
    frames = model.frames(model.values(parameters))
    return frames[0] if model.single else frames


def transport(surface: Surface, field, *, layer_spacing: float, rate: Rate = None,
              parameters: Optional[Mapping[str, float]] = None, tau: str = "tau", **options) -> pd.DataFrame:
    """ρ_xx(B) and the Hall resistivity ρ_H(B) of a model at given parameters (Ω·m).

    The forward model that `fit_transport` fits: it builds the surface, sets τ = 1/rate, and
    computes σ with `bz.conductivity` (summed over contours), then ρ = σ⁻¹.

    Args:
        surface: A contour DataFrame, a list of them, or a function of parameters returning one.
        field: Magnetic field B along ẑ (T), a float or array-like.
        layer_spacing: Interlayer spacing d (m).
        rate: Scattering rate 1/τ (s⁻¹), as in `build_surface`.
        parameters: Values of the model's parameters, by name.
        tau: The column of τ (s).
        **options: Passed to `bz.conductivity`: e.g. `period` for open sheets, `kz` for a
            warped surface, `charge`, `extrapolate`, `scattering_kernel`. `symmetrize`
            defaults to False here, as the result is the same to rounding.

    Returns:
        DataFrame with columns field (T), resistivity (ρ_xx, Ω·m) and hall_resistivity
        (ρ_H = ½(ρ_yx − ρ_xy) = R_H B, Ω·m).

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=ELECTRON_MASS, tau=1.0)
        >>> model = bzf.transport(surface, np.linspace(0, 30, 31), rate=lambda gamma: gamma,
        ...                       parameters={"gamma": 1e13}, layer_spacing=1e-9)
    """
    parameters = dict(parameters or {})
    model = _Model(surface, rate, tau, parameters)
    fields = np.atleast_1d(np.asarray(field, dtype=np.float64))
    return model.transport(model.values(parameters), fields, layer_spacing, options)


class TransportFit:
    """The result of `fit_transport`: best-fit parameters, their uncertainties and diagnostics.

    Attributes:
        names: The fitted (free) parameters, in model order.
        parameters: Every parameter's value at the best fit, fixed ones included (dict).
        errors: One-standard-deviation error of each free parameter (dict). inf if the data
            do not determine it.
        covariance: Covariance of the free parameters (DataFrame).
        correlation: Correlation matrix of the free parameters (DataFrame).
        chi_squared: Σ residual² at the best fit, priors included.
        degrees_of_freedom: Number of residuals minus number of free parameters.
        absolute: True if the residuals were weighted by measured errors. If False, each
            channel was weighted by its RMS and the covariance scaled by the reduced χ², as
            `scipy.optimize.curve_fit` does without `sigma`.
        singular_values: Singular values of the Jacobian with each parameter's column
            normalised, divided by the largest (descending). A value near 0 marks a
            combination of parameters the data barely constrain, whatever their units.
        starts: One row per starting point: the parameters it converged to, its chi_squared
            and success (DataFrame).
        result: scipy's `OptimizeResult` for the best start.
    """

    def __init__(self, *, model, names, parameters, errors, covariance, chi_squared, degrees_of_freedom, absolute,
                 singular_values, starts, result, layer_spacing, options):
        self._model = model
        self._layer_spacing = layer_spacing
        self._options = options
        self.names = names
        self.parameters = parameters
        self.errors = errors
        self.covariance = pd.DataFrame(covariance, index=names, columns=names)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.sqrt(np.diag(covariance))
            self.correlation = pd.DataFrame(covariance / np.outer(sigma, sigma), index=names, columns=names)
        self.chi_squared = chi_squared
        self.degrees_of_freedom = degrees_of_freedom
        self.absolute = absolute
        self.singular_values = singular_values
        self.starts = starts
        self.result = result

    @property
    def reduced_chi_squared(self) -> float:
        """χ² per degree of freedom: about 1 for a good fit when the errors are measured ones."""
        return self.chi_squared / self.degrees_of_freedom

    def __getitem__(self, name: str) -> float:
        """A best-fit parameter by name, e.g. `result["gamma0"]`."""
        return self.parameters[name]

    def summary(self) -> pd.DataFrame:
        """Every parameter's value, its error (NaN if fixed) and whether it was fitted.

        Returns:
            DataFrame indexed by parameter name, with columns value, error and fitted.
        """
        return pd.DataFrame({
            "value": [self.parameters[n] for n in self.parameters],
            "error": [self.errors.get(n, np.nan) for n in self.parameters],
            "fitted": [n in self.errors for n in self.parameters],
        }, index=list(self.parameters))

    def evaluate(self, field) -> pd.DataFrame:
        """The best-fit model at any fields.

        Args:
            field: Magnetic field B (T), a float or array-like.

        Returns:
            DataFrame with columns field, resistivity and hall_resistivity (Ω·m), as `transport`.
        """
        fields = np.atleast_1d(np.asarray(field, dtype=np.float64))
        return self._model.transport(self.parameters, fields, self._layer_spacing, self._options)

    @property
    def surface(self):
        """The best-fit Fermi surface, with τ = 1/rate (a DataFrame, or a list of them)."""
        frames = self._model.frames(self.parameters)
        return frames[0] if self._model.single else frames

    def __repr__(self) -> str:
        return (f"TransportFit(χ² = {self.chi_squared:.4g}, ν = {self.degrees_of_freedom}, "
                f"χ²/ν = {self.reduced_chi_squared:.4g})\n{self.summary()}")


def _channel(data, column, error, name):
    """(values, errors or None) of one measured channel."""
    values = data[column].to_numpy(dtype=np.float64)
    if error is None:
        return values, None
    if isinstance(error, str):
        errors = data[error].to_numpy(dtype=np.float64)
    else:
        errors = np.full(values.shape, float(error))
    bad = np.isfinite(errors) & ~(errors > 0)
    if np.any(bad):
        raise ValueError(f"{name}_error must be positive; it is {errors[bad][0]}")
    return values, errors


def _scale(start, bound):
    """The natural size of a parameter, so the optimiser works with numbers of order 1."""
    if start != 0:
        return abs(start)
    lo, hi = bound
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return 0.5 * (hi - lo)
    return 1.0


def _covariance(jacobian, scale, chi_squared, degrees_of_freedom, absolute):
    """Covariance of the parameters (in their own units) and the relative singular values."""
    n = jacobian.shape[1]
    norms = np.linalg.norm(jacobian, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.linalg.svd(jacobian / np.where(norms > 0, norms, 1.0), compute_uv=False)
    relative = normalised / normalised[0] if normalised[0] > 0 else normalised
    _, s, vt = np.linalg.svd(jacobian, full_matrices=False)
    if s[-1] <= max(jacobian.shape) * np.finfo(float).eps * s[0]:
        return np.full((n, n), np.inf), relative
    covariance = (vt.T / s**2) @ vt
    if not absolute and degrees_of_freedom > 0:
        covariance *= chi_squared / degrees_of_freedom
    return covariance * np.outer(scale, scale), relative


def fit_transport(
    data: pd.DataFrame,
    surface: Surface,
    *,
    layer_spacing: float,
    p0: Union[Mapping[str, float], Sequence[Mapping[str, float]]],
    rate: Rate = None,
    fixed: Optional[Mapping[str, float]] = None,
    bounds: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
    priors: Optional[Mapping[str, Sequence[float]]] = None,
    field: str = "field",
    resistivity: Optional[str] = "resistivity",
    hall_resistivity: Optional[str] = "hall_resistivity",
    resistivity_error: Union[str, float, None] = None,
    hall_resistivity_error: Union[str, float, None] = None,
    tau: str = "tau",
    least_squares_options: Optional[Mapping] = None,
    **options,
) -> TransportFit:
    """Fit a scattering rate, a Fermi surface, or both, to measured ρ_xx(B) and Hall resistivity.

    Least squares (`scipy.optimize.least_squares`) on the residuals
    (model − data)/error of each channel, with the model computed exactly by
    `bz.conductivity`. Give `p0` a starting value for every parameter to fit; parameters
    with defaults, and those in `fixed`, are held.

    Two curves at one temperature constrain only a few numbers: σ(0), the weak-field
    Hall and magnetoresistance coefficients, and how they change up to the highest
    ω_cτ reached. Models with more parameters than that can fit the data equally well
    with very different values. So check `correlation` and `singular_values`, and pass
    a list of starting points to `p0`: the best is returned, all are listed in
    `starts`, and a warning is given if two starts reach different solutions that fit
    equally well. The data also fix only ℓ = vτ, not τ: an error in the Fermi velocity
    passes straight into τ.

    Prepare the data first: symmetrise ρ_xx and antisymmetrise the Hall resistivity in
    field (`qz.symmetrize`, `qz.antisymmetrize`), convert to resistivities in Ω·m, remove
    quantum oscillations, and bin to a few hundred points at most.

    Args:
        data: Measured data, one row per field.
        surface: The Fermi surface: a contour DataFrame (or list, one per pocket), or a
            function of parameters to fit returning one, e.g.
            `lambda mu: bz.generators.tight_binding(512, tau=1.0, chemical_potential=mu, ...)`.
        layer_spacing: Interlayer spacing d (m).
        p0: Starting values of the parameters to fit, by name, or a list of such dicts (one
            fit per start; each must name the same parameters). Values far from 0 set each
            parameter's scale, so give one of the right size (e.g. 1e12 for a rate in s⁻¹).
        rate: Scattering rate 1/τ (s⁻¹) as a function: arguments named after columns of the
            surface receive them, the rest are parameters. A list gives one per contour
            (None keeps that contour's τ). None uses the surface's own τ column.
        fixed: Values of parameters to hold, by name.
        bounds: (lower, upper) of parameters, by name; None for no bound on that side. Use
            them to keep the rate positive.
        priors: (mean, standard deviation) of parameters known independently (a Fermi
            velocity from ARPES, a carrier density...), by name: each adds the residual
            (value − mean)/std.
        field: Column of the field B (T).
        resistivity: Column of ρ_xx (Ω·m), or None to fit only the Hall resistivity.
        hall_resistivity: Column of the Hall resistivity ρ_H = R_H B (Ω·m), negative for
            electrons in B > 0, or None to fit only ρ_xx.
        resistivity_error: One-standard-deviation error of ρ_xx (Ω·m): a column or a number.
        hall_resistivity_error: The same for the Hall resistivity. Give errors for every
            fitted channel or none: without them, each channel is weighted by its RMS.
        tau: The surface column holding τ (s).
        least_squares_options: Passed to `scipy.optimize.least_squares` (e.g. `max_nfev`).
        **options: Passed to `bz.conductivity` (e.g. `period`, `kz`, `charge`,
            `extrapolate`, `scattering_kernel`).

    Returns:
        TransportFit.

    Raises:
        ValueError: If a parameter has no starting or fixed value, a name is unknown, a
            starting point is outside its bounds or gives an invalid model, or there are
            more free parameters than data.

    Warns:
        UserWarning: If two parameters are more than 99% correlated, the data do not
            determine every parameter, two starts reach different but equally good
            solutions, or the optimiser did not converge. Warnings from
            `bz.conductivity` (an under-resolved hot spot, say) are given once, at the
            best fit.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=2 * ELECTRON_MASS, tau=1.0)
        >>> surface["phi"] = bzf.polar_angle(surface)
        >>> def rate(phi, gamma0, a4):
        ...     return gamma0 * (1 + a4 * np.cos(4 * phi))
        >>> data = bzf.transport(surface, np.linspace(0, 30, 31), rate=rate,
        ...                      parameters={"gamma0": 5e12, "a4": 0.4}, layer_spacing=1e-9)
        >>> result = bzf.fit_transport(data, surface, rate=rate, p0={"gamma0": 1e12, "a4": 0.0},
        ...                            bounds={"a4": (-0.95, 0.95)}, layer_spacing=1e-9)
    """
    starts_given = [dict(p0)] if isinstance(p0, Mapping) else [dict(p) for p in p0]
    if not starts_given:
        raise ValueError("p0 must give at least one starting point")
    free = list(starts_given[0])
    if any(set(p) != set(free) for p in starts_given):
        raise ValueError("every starting point in p0 must name the same parameters")
    fixed = dict(fixed or {})
    both = sorted(set(free) & set(fixed))
    if both:
        raise ValueError(f"parameter(s) {both} are both fixed and given a starting value")

    model = _Model(surface, rate, tau, {**starts_given[0], **fixed})
    for name in [*fixed, *(bounds or {}), *(priors or {})]:
        if name not in model.parameters:
            raise ValueError(f"unknown parameter {name!r}; the model's parameters are {list(model.parameters)}")
    model.values({**starts_given[0], **fixed})  # every parameter has a value, and every name is known
    free = [name for name in model.parameters if name in free]  # model order
    if not free:
        raise ValueError(f"nothing to fit: give p0 a starting value for some of {list(model.parameters)}")
    held = {n: v for n, v in model.values({**starts_given[0], **fixed}).items() if n not in free}

    lower = np.array([_bound((bounds or {}).get(n), 0, -np.inf) for n in free])
    upper = np.array([_bound((bounds or {}).get(n), 1, np.inf) for n in free])
    if np.any(lower >= upper):
        raise ValueError(f"each lower bound must be below its upper bound: {dict(zip(free, zip(lower, upper)))}")

    # The measured channels: where each is finite, and how each residual is weighted.
    fields = data[field].to_numpy(dtype=np.float64)
    channels = []
    for name, column, error in (("resistivity", resistivity, resistivity_error),
                                ("hall_resistivity", hall_resistivity, hall_resistivity_error)):
        if column is not None:
            values, errors = _channel(data, column, error, name)
            channels.append([name, values, errors])
    if not channels:
        raise ValueError("fit at least one of resistivity and hall_resistivity")
    given = [errors is not None for _, _, errors in channels]
    if any(given) and not all(given):
        raise ValueError("give errors for every fitted channel or for none (then each is weighted by its RMS)")
    absolute = all(given)
    used = np.isfinite(fields)
    for channel in channels:
        name, values, errors = channel
        mask = np.isfinite(fields) & np.isfinite(values) & (np.isfinite(errors) if errors is not None else True)
        if not np.any(mask):
            raise ValueError(f"no finite {name} data to fit")
        if errors is None:
            rms = float(np.sqrt(np.mean(values[mask] ** 2)))
            if rms == 0:
                raise ValueError(f"{name} is zero everywhere: there is nothing to weight it by")
            errors = np.full(values.shape, rms)
        channel.extend([mask, errors])
    used = np.zeros(fields.shape, dtype=bool)
    for _, _, _, mask, _ in channels:
        used |= mask
    model_fields = fields[used]
    pieces = [(name, np.flatnonzero(mask[used]), values[mask], errors[mask])
              for name, values, _, mask, errors in channels]

    prior_items = []
    for name, (mean, std) in (priors or {}).items():
        if not (np.isfinite(std) and std > 0):
            raise ValueError(f"the prior on {name!r} needs a positive standard deviation, not {std}")
        prior_items.append((name, float(mean), float(std)))
    n_residuals = sum(p[1].size for p in pieces) + len(prior_items)
    dof = n_residuals - len(free)
    if dof <= 0:
        raise ValueError(f"{len(free)} free parameters but only {n_residuals} residuals")

    def evaluate(values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = model.transport(values, model_fields, layer_spacing, options)
        residuals = [(frame[name].to_numpy()[index] - measured) / errors for name, index, measured, errors in pieces]
        residuals.append(np.array([(values[name] - mean) / std for name, mean, std in prior_items]))
        return np.concatenate(residuals)

    best = None
    rows = []
    for number, start in enumerate(starts_given):
        p_start = np.array([float(start[n]) for n in free])
        outside = [(n, v) for n, v, lo, hi in zip(free, p_start, lower, upper) if not lo <= v <= hi]
        if outside:
            raise ValueError(f"starting point {number} is outside the bounds: {dict(outside)}")
        scale = np.array([_scale(v, (lo, hi)) for v, lo, hi in zip(p_start, lower, upper)])

        def values_at(u, scale=scale):
            return {**held, **dict(zip(free, u * scale))}

        try:
            first = evaluate(values_at(p_start / scale))
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"the model fails at starting point {number}: {error}") from error
        if not np.all(np.isfinite(first)):
            raise ValueError(f"the model is not finite at starting point {number}")
        penalty = np.full(n_residuals, 1e3 * (1.0 + np.max(np.abs(first))))

        def residuals(u, values_at=values_at, penalty=penalty):
            # The optimiser may step where the model is invalid (a negative rate, a surface that
            # no longer closes): a large residual turns it back.
            try:
                r = evaluate(values_at(u))
            except (ValueError, RuntimeError):
                return penalty
            return r if np.all(np.isfinite(r)) else penalty

        result = least_squares(residuals, p_start / scale, bounds=(lower / scale, upper / scale),
                               **dict(least_squares_options or {}))
        fitted = result.x * scale
        chi_squared = float(np.sum(result.fun**2))
        rows.append({**dict(zip(free, fitted)), "chi_squared": chi_squared, "success": bool(result.success)})
        if best is None or chi_squared < best[1]:
            best = (result, chi_squared, scale, fitted)

    result, chi_squared, scale, fitted = best
    values = {**held, **dict(zip(free, fitted))}
    parameters = {name: values[name] for name in model.parameters}
    covariance, singular = _covariance(result.jac, scale, chi_squared, dof, absolute)
    errors = dict(zip(free, np.sqrt(np.diag(covariance))))
    starts = pd.DataFrame(rows)

    _diagnose(free, covariance, errors, starts, chi_squared, dof, absolute, result)
    model.transport(parameters, model_fields, layer_spacing, options)  # its warnings, once, at the best fit

    return TransportFit(model=model, names=free, parameters=parameters, errors=errors, covariance=covariance,
                        chi_squared=chi_squared, degrees_of_freedom=dof, absolute=absolute, singular_values=singular,
                        starts=starts, result=result, layer_spacing=layer_spacing, options=options)


def _bound(pair, side, default):
    if pair is None:
        return default
    value = pair[side]
    return default if value is None else float(value)


def _diagnose(free, covariance, errors, starts, chi_squared, dof, absolute, result):
    """Warn when the fit does not determine its parameters."""
    if not result.success:
        warnings.warn(f"the fit did not converge: {result.message}", UserWarning, stacklevel=3)
    if not np.all(np.isfinite(covariance)):
        warnings.warn("the data do not determine every parameter: some combination of them does not change the "
                      "model. Fix or remove one", UserWarning, stacklevel=3)
        return
    sigma = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            if abs(correlation[i, j]) > _CORRELATION_WARNING:
                warnings.warn(
                    f"{free[i]} and {free[j]} are {100 * abs(correlation[i, j]):.2f}% correlated: the data hardly "
                    "determine them separately, and their errors are unreliable. Try several starting points, "
                    "or fix one", UserWarning, stacklevel=3)
    # Starts that fit as well as the best but land elsewhere.
    unit = 1.0 if absolute else chi_squared / dof
    for number, row in starts.iterrows():
        if unit > 0 and (row["chi_squared"] - chi_squared) / unit < _EQUALLY_GOOD:
            apart = [n for n in free if errors[n] > 0 and abs(row[n] - starts.loc[starts["chi_squared"].idxmin(), n])
                     > _DIFFERENT_SOLUTION * errors[n]]
            if apart:
                warnings.warn(
                    f"starting point {number} reached a different solution that fits as well "
                    f"(Δχ² = {(row['chi_squared'] - chi_squared) / unit:.2g}; {', '.join(apart)} differ by over "
                    f"{_DIFFERENT_SOLUTION:g}σ): the data do not single out one set of parameters",
                    UserWarning, stacklevel=3)
