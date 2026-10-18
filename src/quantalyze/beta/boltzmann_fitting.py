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

Several datasets, for example one sample at several temperatures, are fitted together with
`fit_joint`. Each `Dataset` has its own data, surface, rate and named constants (such as its
temperature, which a model argument of that name receives). Parameters with the same name
are shared by every dataset; those listed in `separate` get one value per dataset, or per
group of datasets with the same value of a constant.

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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import boltzmann as bz

Surface = Union[pd.DataFrame, List[pd.DataFrame], Callable[..., Union[pd.DataFrame, List[pd.DataFrame]]]]
Rate = Union[Callable[..., np.ndarray], Sequence[Optional[Callable[..., np.ndarray]]], None]
Separate = Union[Sequence[str], Mapping[str, str], None]

_REQUIRED = inspect.Parameter.empty
# What `separate` groups by when it names no constant: each dataset on its own.
_DATASET = "dataset"
_WEIGHTINGS = ("curve", "point")
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


class _FitResult:
    """Best-fit values, uncertainties and diagnostics: what `TransportFit` and `JointFit` share."""

    def __init__(self, *, names, parameters, errors, covariance, chi_squared, degrees_of_freedom, absolute,
                 singular_values, starts, result, fit_quality):
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
        self.fit_quality = fit_quality

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

    def _describe(self) -> str:
        return f"χ² = {self.chi_squared:.4g}, ν = {self.degrees_of_freedom}, χ²/ν = {self.reduced_chi_squared:.4g}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._describe()})\n{self.summary()}"


class TransportFit(_FitResult):
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
        fit_quality: How well each channel is fitted (DataFrame): one row per channel, and
            one for the priors, with columns channel, points, sigma (what each residual was
            divided by, Ω·m; NaN for measured errors), chi_squared, share (of the total χ²),
            rms_residual (the RMS of model − data, Ω·m) and relative_rms (rms_residual over
            the RMS of the data).
        singular_values: Singular values of the Jacobian with each parameter's column
            normalised, divided by the largest (descending). A value near 0 marks a
            combination of parameters the data barely constrain, whatever their units.
        starts: One row per starting point: the parameters it converged to, its chi_squared
            and success (DataFrame).
        result: scipy's `OptimizeResult` for the best start.
    """

    def __init__(self, *, model, layer_spacing, options, **fitted):
        super().__init__(**fitted)
        self._model = model
        self._layer_spacing = layer_spacing
        self._options = options

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


class Dataset:
    """One dataset of a joint fit: its measured data, Fermi surface, rate model and constants.

    Everything that can differ from one measurement to the next belongs to its dataset: the
    data and the channels it has, the Fermi surface (fixed, or a function of parameters), the
    rate model, the layer spacing, options such as `period` or `kz`, and named constants such
    as the temperature. An argument of the surface or rate function named after one of the
    dataset's constants receives its value. Every other argument that is not a column of the
    surface is a parameter, matched across datasets by name (see `fit_joint`).

    Args:
        data: Measured data, one row per field, with the columns named below; None for a
            dataset that is only evaluated (`joint_transport`), not fitted.
        surface: The Fermi surface, as in `fit_transport`: a contour DataFrame, a list of
            them (one per pocket), or a function returning one.
        layer_spacing: Interlayer spacing d (m).
        rate: Scattering rate 1/τ (s⁻¹), as in `fit_transport`: a function, a list of them
            (one per contour; None keeps that contour's τ), or None to keep the surface's τ.
            Defaults to None.
        constants: Known values for this dataset, by name, e.g. `{"T": 10.0}`. A constant can
            also label groups of datasets for `fit_joint(separate=...)`. Defaults to none.
        name: The dataset's name in results, and in the names of its own copies of separate
            parameters, e.g. "a4[10 K]". Defaults to its position in the list of datasets
            ("0", "1", ...).
        field: Column of the field B (T). Defaults to "field".
        resistivity: Column of ρ_xx (Ω·m), or None if the dataset has none. Defaults to
            "resistivity".
        hall_resistivity: Column of the Hall resistivity ρ_H = R_H B (Ω·m), negative for
            electrons in B > 0, or None. Defaults to "hall_resistivity".
        resistivity_error: One-standard-deviation error of ρ_xx (Ω·m): a column or a number.
            Defaults to None.
        hall_resistivity_error: The same for the Hall resistivity. Defaults to None.
        tau: The surface column holding τ (s). Defaults to "tau".
        weight: Relative weight of this dataset's curves when no errors are given: their
            contributions to χ² are multiplied by it. Defaults to 1.
        **options: Passed to `bz.conductivity` for this dataset: e.g. `period`, `kz`,
            `charge`, `extrapolate`.

    Raises:
        TypeError: If `data` is neither a DataFrame nor None, or `constants` is not a dict
            with string keys.
        ValueError: If `layer_spacing` or `weight` is not finite and positive, or a column
            named here is not in `data`.

    Examples:
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=2 * ELECTRON_MASS, tau=1.0)
        >>> data = bzf.transport(surface, [0.0, 10.0, 20.0], rate=lambda gamma: gamma,
        ...                      parameters={"gamma": 1.2e12}, layer_spacing=1e-9)
        >>> dataset = bzf.Dataset(data, surface, rate=lambda T, gamma0, alpha: gamma0 + alpha * T,
        ...                       constants={"T": 10.0}, name="10 K", layer_spacing=1e-9)
    """

    def __init__(self, data: Optional[pd.DataFrame], surface: Surface, *, layer_spacing: float, rate: Rate = None,
                 constants: Optional[Mapping[str, Any]] = None, name: Optional[str] = None, field: str = "field",
                 resistivity: Optional[str] = "resistivity", hall_resistivity: Optional[str] = "hall_resistivity",
                 resistivity_error: Union[str, float, None] = None,
                 hall_resistivity_error: Union[str, float, None] = None, tau: str = "tau", weight: float = 1.0,
                 **options):
        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a DataFrame or None, not {type(data).__name__}")
        layer_spacing = float(layer_spacing)
        if not (np.isfinite(layer_spacing) and layer_spacing > 0):
            raise ValueError(f"layer_spacing must be finite and positive, got {layer_spacing}")
        weight = float(weight)
        if not (np.isfinite(weight) and weight > 0):
            raise ValueError(f"weight must be finite and positive, got {weight}")
        constants = {} if constants is None else constants
        if not isinstance(constants, Mapping) or not all(isinstance(key, str) for key in constants):
            raise TypeError(f"constants must be a dict of values by name, e.g. {{'T': 10.0}}, not {constants!r}")
        if data is not None:
            for argument, column in (("field", field), ("resistivity", resistivity),
                                     ("hall_resistivity", hall_resistivity)):
                if column is not None and column not in data.columns:
                    raise ValueError(f"data has no column {column!r} (the {argument} argument)")
            for argument, column in (("resistivity_error", resistivity_error),
                                     ("hall_resistivity_error", hall_resistivity_error)):
                if isinstance(column, str) and column not in data.columns:
                    raise ValueError(f"data has no column {column!r} (the {argument} argument)")
        self.data = data
        self.surface = surface
        self.layer_spacing = layer_spacing
        self.rate = rate
        self.constants = dict(constants)
        self.name = None if name is None else str(name)
        self.field = field
        self.resistivity = resistivity
        self.hall_resistivity = hall_resistivity
        self.resistivity_error = resistivity_error
        self.hall_resistivity_error = hall_resistivity_error
        self.tau = tau
        self.weight = weight
        self.options = options

    def __repr__(self) -> str:
        rows = 0 if self.data is None else len(self.data)
        return f"Dataset(name={self.name!r}, rows={rows}, constants={self.constants})"


def _as_datasets(datasets) -> List[Dataset]:
    if isinstance(datasets, Dataset):
        return [datasets]
    if not isinstance(datasets, (list, tuple)) or not all(isinstance(d, Dataset) for d in datasets):
        raise TypeError("datasets must be a list of bzf.Dataset")
    if not datasets:
        raise ValueError("datasets must contain at least one Dataset")
    return list(datasets)


def _separate_rules(separate: Separate) -> Dict[str, str]:
    """{parameter: what its copies are grouped by}: "dataset", or the name of a constant."""
    if separate is None:
        return {}
    if isinstance(separate, str):
        raise TypeError(f"separate must be a list or dict of parameter names, not the string {separate!r}; "
                        f"write separate=[{separate!r}]")
    items = separate.items() if isinstance(separate, Mapping) else ((name, _DATASET) for name in separate)
    rules = {}
    for name, by in items:
        if not isinstance(name, str) or not isinstance(by, str):
            raise TypeError("separate must name parameters, and what they are grouped by, as strings")
        rules[name] = by
    return rules


def _lookup(mappings: Sequence[Mapping], joint: str, argument: str) -> Tuple[bool, Any]:
    """A value for a joint parameter from the first mapping that has it: by its own name, then (for a
    copy of a separate parameter) by the name of the parameter."""
    for key in (joint,) if joint == argument else (joint, argument):
        for mapping in mappings:
            if key in mapping:
                return True, mapping[key]
    return False, None


class _Member:
    """One dataset of a joint model: its own `_Model`, and the joint name of each of its parameters."""

    def __init__(self, dataset: Dataset, name: str, rules: Mapping[str, str], known: Sequence[Mapping]):
        self.dataset = dataset
        self.name = name
        self.constants = dict(dataset.constants)
        self._rules = rules
        surface_arguments = _arguments(dataset.surface, "surface") if callable(dataset.surface) else {}
        start = {}  # what the surface is first built with, to see which rate arguments are columns
        for argument in surface_arguments:
            if argument in self.constants:
                start[argument] = self.constants[argument]
                continue
            found, value = _lookup(known, self.joint_name(argument), argument)
            if found:
                start[argument] = value
        self.model = _Model(dataset.surface, dataset.rate, dataset.tau, start)
        for columns in self.model.columns.values():
            clash = sorted(set(columns) & set(self.constants))
            if clash:
                raise ValueError(f"{clash[0]!r} is both a column of the surface and a constant of dataset {name!r}: "
                                 "rename one of them")
        self.arguments = {}  # model argument → joint parameter, constants left out
        for argument in self.model.parameters:
            if argument in self.constants:
                if argument in rules:
                    raise ValueError(f"{argument!r} is in separate, but it is a constant of dataset {name!r}")
                continue
            self.arguments[argument] = self.joint_name(argument)

    def joint_name(self, argument: str) -> str:
        """The joint parameter a model argument stands for: its own name if shared, name[group] if separate."""
        by = self._rules.get(argument)
        if by is None:
            return argument
        if by == _DATASET:
            return f"{argument}[{self.name}]"
        if by not in self.constants:
            raise ValueError(f"separate groups {argument!r} by the constant {by!r}, but dataset {self.name!r} has no "
                             f"constant {by!r}")
        return f"{argument}[{self.constants[by]}]"

    def local(self, values: Mapping[str, float]) -> dict:
        """The values of this dataset's model arguments, from the joint parameter values."""
        return {**self.constants, **{argument: values[joint] for argument, joint in self.arguments.items()}}


class _Joint:
    """Datasets as one model: their members, and the joint parameters they share or keep separate.

    The surfaces are first built with values from `known` (mappings by joint or parameter
    name, the first that has a value wins), to see which rate arguments are columns.
    """

    def __init__(self, datasets, separate: Separate, known: Sequence[Mapping], check_separate: bool = True):
        self.datasets = _as_datasets(datasets)
        self.rules = _separate_rules(separate)
        names = [str(number) if d.name is None else d.name for number, d in enumerate(self.datasets)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"dataset names must be unique; {repeated[0]!r} is used more than once")
        self.members = [_Member(d, name, self.rules, known) for d, name in zip(self.datasets, names)]

        defaults: Dict[str, list] = {}
        self.base: Dict[str, Tuple[str, Optional[str]]] = {}  # joint name → (model argument, group or None)
        for member in self.members:
            for argument, joint in member.arguments.items():
                defaults.setdefault(joint, []).append(member.model.parameters[argument])
                self.base[joint] = (argument, None if joint == argument else joint[len(argument) + 1:-1])
        # A parameter is optional only if every model that takes it gives a default.
        self.parameters = {joint: (_REQUIRED if any(v is _REQUIRED for v in values) else values[0])
                           for joint, values in defaults.items()}
        self.copies: Dict[str, List[str]] = {}  # separate parameter → its copies
        for joint, (argument, group) in self.base.items():
            if group is not None:
                self.copies.setdefault(argument, []).append(joint)

        unknown = [name for name in self.rules if name not in self.copies]
        if unknown and check_separate:
            raise ValueError(f"separate names {unknown}, which no dataset's model takes as a parameter; the "
                             f"parameters are {list(self.parameters)}")
        for constant in sorted({c for member in self.members for c in member.constants}):
            users = [member.name for member in self.members if constant in member.arguments]
            if users:
                holders = [member.name for member in self.members if constant in member.constants]
                raise ValueError(f"{constant!r} is a constant of dataset(s) {holders} but a parameter of dataset(s) "
                                 f"{users}: give it in the constants of every dataset whose model takes it")

    def expand(self, mapping: Optional[Mapping], what: str) -> Dict[str, Tuple[Any, bool]]:
        """{joint name: (value, whether named on its own)} from a mapping by joint name, or by the name of a
        separate parameter, which sets every copy. A copy's own name takes precedence."""
        expanded = {}
        for key, value in (mapping or {}).items():
            if key in self.parameters:
                expanded[key] = (value, True)
            elif key in self.copies:
                for joint in self.copies[key]:
                    if not expanded.get(joint, (None, False))[1]:
                        expanded[joint] = (value, False)
            else:
                raise ValueError(f"unknown parameter {key!r} in {what}; the model's parameters are "
                                 f"{list(self.parameters)}")
        return expanded

    def values(self, given: Mapping[str, float]) -> dict:
        """Every joint parameter's value: from `given` (by joint name), or its default."""
        values = {}
        for name, default in self.parameters.items():
            if name in given:
                values[name] = given[name]
            elif default is not _REQUIRED:
                values[name] = default
            else:
                raise ValueError(f"no value for parameter {name!r}; the model's parameters are {list(self.parameters)}")
        return values


def _evaluate(joint: _Joint, values: Mapping[str, float], field) -> Dict[str, pd.DataFrame]:
    """Each dataset's model at the joint parameter values: at its data's fields, or at `field`."""
    curves = {}
    for member in joint.members:
        dataset = member.dataset
        if field is None:
            if dataset.data is None:
                raise ValueError(f"dataset {member.name!r} has no data to take the fields from: pass field=...")
            fields = dataset.data[dataset.field].to_numpy(dtype=np.float64)
            keep = np.isfinite(fields)
            frame = member.model.transport(member.local(values), fields[keep], dataset.layer_spacing, dataset.options)
            frame.index = dataset.data.index[keep]
        else:
            fields = np.atleast_1d(np.asarray(field, dtype=np.float64))
            frame = member.model.transport(member.local(values), fields, dataset.layer_spacing, dataset.options)
        curves[member.name] = frame
    return curves


class JointFit(_FitResult):
    """The result of `fit_joint`: best-fit parameters, their uncertainties and diagnostics.

    Parameters keep their names from the models; the copies of a separate parameter are
    named name[group], e.g. "a4[10 K]".

    Attributes:
        names: The fitted (free) parameters, in model order.
        parameters: Every parameter's value at the best fit, fixed ones included (dict).
        errors: One-standard-deviation error of each free parameter (dict). inf if the data
            do not determine it.
        covariance: Covariance of the free parameters (DataFrame).
        correlation: Correlation matrix of the free parameters (DataFrame).
        chi_squared: Σ residual² at the best fit, over every dataset, priors included.
        degrees_of_freedom: Number of residuals minus number of free parameters.
        absolute: True if the residuals were weighted by measured errors. If False, the
            covariance is scaled by the reduced χ², as `scipy.optimize.curve_fit` does
            without `sigma`.
        weighting: How the residuals were weighted: "errors" (measured errors), or "curve"
            or "point" (see `fit_joint`).
        datasets: The names of the datasets, in order.
        fit_quality: How well each curve (one channel of one dataset) is fitted (DataFrame):
            one row per curve, and one for the priors, with columns dataset, channel, points,
            sigma (what each residual was divided by, Ω·m; NaN for measured errors),
            chi_squared, share (of the total χ²), rms_residual (the RMS of model − data, Ω·m)
            and relative_rms (rms_residual over the RMS of the data).
        singular_values: Singular values of the Jacobian with each parameter's column
            normalised, divided by the largest (descending). A value near 0 marks a
            combination of parameters the data barely constrain, whatever their units.
        starts: One row per starting point: the parameters it converged to, its chi_squared
            and success (DataFrame).
        result: scipy's `OptimizeResult` for the best start.
    """

    def __init__(self, *, joint, weighting, **fitted):
        super().__init__(**fitted)
        self._joint = joint
        self.weighting = weighting
        self.datasets = [member.name for member in joint.members]

    def summary(self) -> pd.DataFrame:
        """Every parameter's value, its error (NaN if fixed), whether it was fitted, and what it is.

        Returns:
            DataFrame indexed by parameter name, with columns value, error, fitted, parameter
            (the model argument) and group (the dataset, or value of a constant, that a copy
            of a separate parameter belongs to; missing for a shared parameter).
        """
        table = super().summary()
        table["parameter"] = [self._joint.base[n][0] for n in table.index]
        table["group"] = [self._joint.base[n][1] for n in table.index]
        return table

    def evaluate(self, field=None, datasets: Optional[Sequence[Dataset]] = None) -> Dict[str, pd.DataFrame]:
        """The best-fit model of each dataset: at its measured fields, exactly what the fit compared with the data.

        Args:
            field: Magnetic field B (T), a float or array-like; None for each dataset's own
                measured fields. Defaults to None.
            datasets: Datasets to evaluate with the fitted parameters instead of the fitted
                ones (e.g. data held back from the fit), linked as in the fit. Defaults to None.

        Returns:
            A dict of DataFrames by dataset name, each with columns field, resistivity and
            hall_resistivity (Ω·m), as `transport`. At the measured fields, the rows are
            indexed as in the dataset's data, leaving out rows without a finite field.

        Raises:
            ValueError: If a dataset in `datasets` needs a parameter the fit did not have, such
                as a new copy of a separate parameter; give it with `joint_transport` instead.
        """
        if datasets is None:
            joint = self._joint
        else:
            joint = _Joint(datasets, self._joint.rules, [self.parameters], check_separate=False)
        values = joint.values({n: v for n, v in self.parameters.items() if n in joint.parameters})
        return _evaluate(joint, values, field)

    @property
    def surfaces(self) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
        """The best-fit Fermi surface of each dataset, with τ = 1/rate, by dataset name."""
        surfaces = {}
        for member in self._joint.members:
            frames = member.model.frames(member.local(self.parameters))
            surfaces[member.name] = frames[0] if member.model.single else frames
        return surfaces

    def _describe(self) -> str:
        return f"{super()._describe()}, weighting = {self.weighting!r}"


def joint_transport(datasets: Sequence[Dataset], parameters: Mapping[str, float], *, separate: Separate = None,
                    field=None) -> Dict[str, pd.DataFrame]:
    """ρ_xx(B) and the Hall resistivity ρ_H(B) of every dataset of a joint model (Ω·m).

    The forward model that `fit_joint` fits, at given parameters: each dataset's surface is
    built, τ = 1/rate is set with the dataset's constants, and ρ follows from
    `bz.conductivity` as in `transport`. Use it to make synthetic data, or to predict a
    dataset held back from a fit.

    Args:
        datasets: The datasets, a list of `Dataset`; their data are needed only for their
            fields, when `field` is None.
        parameters: Values of the joint parameters, by name; the name of a separate
            parameter sets every copy, and a copy's own name (e.g. "a4[10 K]") just that one.
            Parameters with defaults may be left out.
        separate: Which parameters differ between datasets, as in `fit_joint`. Defaults to
            None.
        field: Magnetic field B (T), a float or array-like, for every dataset; None for each
            dataset's own measured fields. Defaults to None.

    Returns:
        A dict of DataFrames by dataset name, each with columns field, resistivity and
        hall_resistivity (Ω·m), as `transport`. At the measured fields, the rows are indexed
        as in the dataset's data, leaving out rows without a finite field.

    Raises:
        ValueError: If a parameter has no value or is unknown, or `field` is None and a
            dataset has no data.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=2 * ELECTRON_MASS, tau=1.0)
        >>> datasets = [bzf.Dataset(None, surface, rate=lambda T, gamma0, alpha: gamma0 + alpha * T,
        ...                         constants={"T": T}, name=f"{T} K", layer_spacing=1e-9) for T in (10, 50)]
        >>> curves = bzf.joint_transport(datasets, {"gamma0": 1e12, "alpha": 2e10}, field=np.linspace(0, 30, 31))
        >>> curves["50 K"].head(2)
    """
    joint = _Joint(datasets, separate, [dict(parameters)])
    expanded = joint.expand(parameters, "parameters")
    return _evaluate(joint, joint.values({name: value for name, (value, _) in expanded.items()}), field)


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
    with defaults, and those in `fixed`, are held. To fit several datasets at once, with
    parameters shared between them, use `fit_joint`.

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
            column is not in `data`, a starting point is outside its bounds or gives an
            invalid model, or there are more free parameters than data.

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
    dataset = Dataset(data, surface, layer_spacing=layer_spacing, rate=rate, field=field, resistivity=resistivity,
                      hall_resistivity=hall_resistivity, resistivity_error=resistivity_error,
                      hall_resistivity_error=hall_resistivity_error, tau=tau)
    dataset.options = dict(options)  # everything else goes to bz.conductivity, as it always has
    fit = _fit([dataset], p0=p0, separate=None, fixed=fixed, bounds=bounds, priors=priors, weighting="point",
               least_squares_options=least_squares_options, label=False)
    return TransportFit(model=fit.joint.members[0].model, layer_spacing=dataset.layer_spacing,
                        options=dataset.options, **fit.results(dataset_column=False))


def fit_joint(
    datasets: Sequence[Dataset],
    *,
    p0: Union[Mapping[str, float], Sequence[Mapping[str, float]]],
    separate: Separate = None,
    fixed: Optional[Mapping[str, float]] = None,
    bounds: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
    priors: Optional[Mapping[str, Sequence[float]]] = None,
    weighting: str = "curve",
    least_squares_options: Optional[Mapping] = None,
) -> JointFit:
    """Fit several datasets at once, with parameters shared between them or kept separate.

    One least-squares problem over the residuals of every dataset, each dataset's model
    computed exactly by `bz.conductivity` as in `fit_transport`. Parameters are matched by
    name: a model argument with the same name in two datasets is one parameter, fitted to
    both. A parameter listed in `separate` gets one copy per dataset instead, named
    name[dataset name], or one per group of datasets sharing the value of a constant, named
    name[value]. A model argument named after a dataset's constant receives its value. So:

    - **Shared and separate terms:** `rate(phi, gamma_imp, gamma_in)` with
      `separate=["gamma_in"]` fits one gamma_imp to every dataset and one gamma_in to each.
    - **A law across datasets:** a rate `rate(phi, T, gamma0, alpha)` returning
      `gamma0 + alpha * T`, with T in each dataset's `constants`, fits gamma0 and alpha.
    - **Different surfaces:** give each dataset its own surface; the rate's parameters are
      still shared by name.
    - **Groups of datasets:** `separate={"gamma_imp": "sample"}` fits one gamma_imp per value
      of the constant "sample", shared by all the datasets with that value.

    In `p0`, `fixed`, `bounds` and `priors`, the name of a separate parameter sets every copy,
    and a copy's own name (e.g. "a4[10 K]") sets that copy only, taking precedence.

    **Weighting.** With measured errors for every curve (one channel of one dataset), each
    residual is (model − data)/error and χ² is absolute. Without them, the residuals of each
    curve are divided by one number, and the covariance is scaled by the reduced χ²:

    - `weighting="curve"` (the default): the curve's RMS × √(its points / the mean number
      of points per curve). Every curve counts equally, whatever the size of its values or
      its number of points.
    - `weighting="point"`: the curve's RMS, as `fit_transport` does. Every point counts
      equally relative to its curve, so curves with more points count for more.

    A dataset's `weight` multiplies its curves' contributions to χ². `fit_quality` in the
    result gives each curve's scale and share of χ². With one dataset, `fit_joint` with
    weighting="point" does exactly what `fit_transport` does; with "curve", the two agree
    whenever both channels have the same number of points.

    Args:
        datasets: The datasets, a list of `Dataset`.
        p0: Starting values of the parameters to fit, by name, or a list of such dicts (one
            fit per start; each must name the same parameters). Values far from 0 set each
            parameter's scale, so give one of the right size (e.g. 1e12 for a rate in s⁻¹).
        separate: Parameters that differ between datasets: a list of names (one copy per
            dataset), or a dict from name to "dataset" or to the name of a constant (one copy
            per value of it). Defaults to None: every parameter is shared.
        fixed: Values of parameters to hold, by name.
        bounds: (lower, upper) of parameters, by name; None for no bound on that side.
        priors: (mean, standard deviation) of parameters known independently, by name: each
            adds the residual (value − mean)/std, once per copy of a separate parameter.
        weighting: "curve" or "point": how curves without measured errors are weighted (see
            above). Defaults to "curve".
        least_squares_options: Passed to `scipy.optimize.least_squares` (e.g. `max_nfev`).

    Returns:
        JointFit.

    Raises:
        TypeError: If `datasets` is not a list of `Dataset`, or `separate` is not a list or
            dict of names.
        ValueError: If `weighting` is unknown; a parameter has no starting or fixed value, or
            a name is unknown; a separate parameter is grouped by a constant that a dataset
            lacks; a name is a constant of some datasets and a parameter of others; dataset
            names repeat; a dataset has no data; some curves have errors and others do not;
            `weight` is combined with measured errors; a starting point is outside its
            bounds or gives an invalid model; or there are more free parameters than data.

    Warns:
        UserWarning: As `fit_transport`: if two parameters are more than 99% correlated, the
            data do not determine every parameter, two starts reach different but equally
            good solutions, or the optimiser did not converge.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.beta import boltzmann_fitting as bzf
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> surface = bz.generators.circle(256, k_fermi=7e9, mass=2 * ELECTRON_MASS, tau=1.0)
        >>> def rate(T, gamma0, alpha):
        ...     return gamma0 + alpha * T
        >>> fields = np.linspace(0, 30, 31)
        >>> datasets = [
        ...     bzf.Dataset(bzf.transport(surface, fields, rate=rate, parameters={"gamma0": 1e12, "alpha": 2e10, "T": T},
        ...                               layer_spacing=1e-9),
        ...                 surface, rate=rate, constants={"T": T}, name=f"{T} K", layer_spacing=1e-9)
        ...     for T in (10, 50, 100)]
        >>> result = bzf.fit_joint(datasets, p0={"gamma0": 5e11, "alpha": 1e10})
        >>> result["alpha"]  # ≈ 2e10
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    fit = _fit(datasets, p0=p0, separate=separate, fixed=fixed, bounds=bounds, priors=priors, weighting=weighting,
               least_squares_options=least_squares_options, label=True)
    return JointFit(joint=fit.joint, weighting="errors" if fit.absolute else weighting, **fit.results())


@dataclass
class _Fitted:
    """What `_fit` found, before it is wrapped in a `TransportFit` or `JointFit`."""

    joint: _Joint
    names: List[str]
    parameters: Dict[str, float]
    errors: Dict[str, float]
    covariance: np.ndarray
    chi_squared: float
    degrees_of_freedom: int
    absolute: bool
    singular_values: np.ndarray
    starts: pd.DataFrame
    result: Any
    fit_quality: pd.DataFrame

    def results(self, dataset_column: bool = True) -> dict:
        quality = self.fit_quality if dataset_column else self.fit_quality.drop(columns="dataset")
        return dict(names=self.names, parameters=self.parameters, errors=self.errors, covariance=self.covariance,
                    chi_squared=self.chi_squared, degrees_of_freedom=self.degrees_of_freedom, absolute=self.absolute,
                    singular_values=self.singular_values, starts=self.starts, result=self.result,
                    fit_quality=quality)


class _Residuals:
    """The residuals the optimiser sees for one starting point, and their Jacobian.

    The residuals of each dataset at the last point are kept, so a difference step in one
    parameter recomputes only the datasets whose model takes it. The Jacobian is then the
    same as forward differences of every residual, at a fraction of the cost.
    """

    def __init__(self, joint, free, held, scale, fields, pieces, prior_items):
        self.members = joint.members
        self.free = free
        self.held = held
        self.scale = scale
        self.fields = fields  # each dataset's fields to evaluate at
        self.pieces = pieces  # each dataset's (channel, row index, measured, error) per curve
        self.prior_items = prior_items
        self.penalty = None  # what an invalid model gives, set once the starting point is known
        self.affects = [[number for number, member in enumerate(joint.members) if name in member.arguments.values()]
                        for name in free]
        self.offsets = np.cumsum([0] + [sum(piece[2].size for piece in own) for own in pieces])
        self._key = None
        self._parts = None

    def values(self, u):
        return {**self.held, **dict(zip(self.free, u * self.scale))}

    def part(self, number, values):
        """The residuals of one dataset."""
        member = self.members[number]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = member.model.transport(member.local(values), self.fields[number], member.dataset.layer_spacing,
                                           member.dataset.options)
        return np.concatenate([(frame[name].to_numpy()[index] - measured) / errors
                               for name, index, measured, errors in self.pieces[number]])

    def priors(self, values):
        return np.array([(values[name] - mean) / std for name, mean, std in self.prior_items])

    def parts(self, u):
        key = u.tobytes()
        if key != self._key:
            values = self.values(u)
            parts = [self.part(number, values) for number in range(len(self.members))]
            self._key, self._parts = key, parts
        return self._parts

    def evaluate(self, u):
        """The residuals at scaled parameters u; raises where the model is invalid."""
        return np.concatenate([*self.parts(u), self.priors(self.values(u))])

    def __call__(self, u):
        # The optimiser may step where the model is invalid (a negative rate, a surface that
        # no longer closes): a large residual turns it back.
        try:
            r = self.evaluate(u)
        except (ValueError, RuntimeError):
            return self.penalty
        return r if np.all(np.isfinite(r)) else self.penalty

    def jacobian(self, u, lower, upper):
        """Forward differences of the residuals at u, with steps as scipy's "2-point" takes them."""
        r0 = self(u)
        valid = r0 is not self.penalty
        prior0 = self.priors(self.values(u))
        start = self.offsets[-1]
        steps = _difference_steps(u, lower, upper)
        jacobian = np.empty((r0.size, u.size))
        for j, affected in enumerate(self.affects):
            shifted = u.copy()
            shifted[j] = u[j] + steps[j]
            dx = shifted[j] - u[j]
            if not valid:
                jacobian[:, j] = (self(shifted) - r0) / dx
                continue
            values = self.values(shifted)
            column = np.zeros(r0.size)
            try:
                for number in affected:
                    new = self.part(number, values)
                    if not np.all(np.isfinite(new)):
                        raise ValueError("the model is not finite")
                    column[self.offsets[number]:self.offsets[number + 1]] = new - self._parts[number]
            except (ValueError, RuntimeError):
                column = self.penalty - r0
            else:
                column[start:] = self.priors(values) - prior0
            jacobian[:, j] = column / dx
        return jacobian


def _difference_steps(u, lower, upper):
    """Forward-difference steps √ε·max(1, |u|), reversed where a step would cross a bound."""
    step = np.sqrt(np.finfo(float).eps) * np.where(u >= 0, 1.0, -1.0) * np.maximum(1.0, np.abs(u))
    step = np.where((u + step < lower) | (u + step > upper), -step, step)
    outside = (u + step < lower) | (u + step > upper)
    if np.any(outside):  # bounds on both sides closer than a step: go half way to the further one
        up, down = upper - u, u - lower
        step = np.where(outside, np.where(up >= down, 0.5 * up, -0.5 * down), step)
    return step


def _fit_quality(joint, pieces, residuals, absolute, n_priors) -> pd.DataFrame:
    """One row per curve (and one for the priors): its scale, χ² and misfit."""
    rows = []
    offset = 0
    for member, own in zip(joint.members, pieces):
        for name, _, measured, errors in own:
            r = residuals[offset:offset + measured.size]
            offset += measured.size
            rms_residual = float(np.sqrt(np.mean((r * errors) ** 2)))  # r × error = model − data
            rms_data = float(np.sqrt(np.mean(measured ** 2)))
            rows.append({"dataset": member.name, "channel": name, "points": int(measured.size),
                         "sigma": np.nan if absolute else float(errors[0]), "chi_squared": float(np.sum(r ** 2)),
                         "rms_residual": rms_residual,
                         "relative_rms": rms_residual / rms_data if rms_data > 0 else np.nan})
    if n_priors:
        rows.append({"dataset": None, "channel": "priors", "points": n_priors, "sigma": np.nan,
                     "chi_squared": float(np.sum(residuals[offset:] ** 2)), "rms_residual": np.nan,
                     "relative_rms": np.nan})
    table = pd.DataFrame(rows)
    total = table["chi_squared"].sum()
    table.insert(table.columns.get_loc("chi_squared") + 1, "share",
                 table["chi_squared"] / total if total > 0 else np.nan)
    return table


def _fit(datasets, *, p0, separate, fixed, bounds, priors, weighting, least_squares_options, label) -> _Fitted:
    """The least-squares fit behind `fit_transport` (one dataset) and `fit_joint`.

    `label` adds the dataset's name to error messages about its data.
    """
    starts_given = [dict(p0)] if isinstance(p0, Mapping) else [dict(p) for p in p0]
    if not starts_given:
        raise ValueError("p0 must give at least one starting point")
    if any(set(p) != set(starts_given[0]) for p in starts_given):
        raise ValueError("every starting point in p0 must name the same parameters")
    fixed = dict(fixed or {})
    both = sorted(set(starts_given[0]) & set(fixed))
    if both:
        raise ValueError(f"parameter(s) {both} are both fixed and given a starting value")

    joint = _Joint(datasets, separate, [fixed, starts_given[0]])
    fixed_given = joint.expand(fixed, "fixed")
    bounds_given = joint.expand(bounds, "bounds")
    priors_given = joint.expand(priors, "priors")
    starts = [joint.expand(start, "p0") for start in starts_given]
    # Between p0 and fixed, a copy's own name beats the shared one: fixing "a4[10 K]" while p0
    # gives "a4" fits every other copy.
    free = [n for n in joint.parameters if n in starts[0] and (n not in fixed_given or starts[0][n][1])]
    values = joint.values({**{n: v for n, (v, _) in fixed_given.items()}, **{n: starts[0][n][0] for n in free}})
    if not free:
        raise ValueError(f"nothing to fit: give p0 a starting value for some of {list(joint.parameters)}")
    held = {n: v for n, v in values.items() if n not in free}

    def pair(n):
        return bounds_given[n][0] if n in bounds_given else None

    lower = np.array([_bound(pair(n), 0, -np.inf) for n in free])
    upper = np.array([_bound(pair(n), 1, np.inf) for n in free])
    if np.any(lower >= upper):
        raise ValueError(f"each lower bound must be below its upper bound: {dict(zip(free, zip(lower, upper)))}")

    # The measured curves: where each is finite, and what each residual is divided by.
    def where(member):
        return f" in dataset {member.name!r}" if label else ""

    curves = []  # [dataset number, channel, values, errors or None, mask]
    for number, member in enumerate(joint.members):
        dataset = member.dataset
        if dataset.data is None:
            raise ValueError(f"dataset {member.name!r} has no data to fit")
        fields = dataset.data[dataset.field].to_numpy(dtype=np.float64)
        own = []
        for name, column, error in (("resistivity", dataset.resistivity, dataset.resistivity_error),
                                    ("hall_resistivity", dataset.hall_resistivity, dataset.hall_resistivity_error)):
            if column is not None:
                measured, errors = _channel(dataset.data, column, error, name)
                mask = np.isfinite(fields) & np.isfinite(measured) & (np.isfinite(errors) if errors is not None else True)
                if not np.any(mask):
                    raise ValueError(f"no finite {name} data to fit{where(member)}")
                own.append([number, name, measured, errors, mask])
        if not own:
            raise ValueError(f"fit at least one of resistivity and hall_resistivity{where(member)}")
        curves.extend(own)
    given = [curve[3] is not None for curve in curves]
    if any(given) and not all(given):
        raise ValueError(f"give errors for every fitted channel{' of every dataset' if label else ''} or for none "
                         "(then each is weighted by its RMS)")
    absolute = all(given)
    if absolute and any(member.dataset.weight != 1.0 for member in joint.members):
        raise ValueError("weight applies only without measured errors, which already weight every point")
    if not absolute:
        mean_points = np.mean([np.count_nonzero(curve[4]) for curve in curves])
        for curve in curves:
            number, name, measured, _, mask = curve
            rms = float(np.sqrt(np.mean(measured[mask] ** 2)))
            if rms == 0:
                raise ValueError(f"{name} is zero everywhere{where(joint.members[number])}: there is nothing to "
                                 "weight it by")
            sigma = rms
            if weighting == "curve":
                sigma = rms * np.sqrt(np.count_nonzero(mask) / mean_points)
            weight = joint.members[number].dataset.weight
            if weight != 1.0:
                sigma = sigma / np.sqrt(weight)
            curve[3] = np.full(measured.shape, sigma)
    member_fields, pieces = [], []
    for number, member in enumerate(joint.members):
        own = [curve for curve in curves if curve[0] == number]
        fields = member.dataset.data[member.dataset.field].to_numpy(dtype=np.float64)
        used = np.zeros(fields.shape, dtype=bool)
        for curve in own:
            used |= curve[4]
        member_fields.append(fields[used])
        pieces.append([(name, np.flatnonzero(mask[used]), measured[mask], errors[mask])
                       for _, name, measured, errors, mask in own])

    prior_items = []
    for name, ((mean, std), _) in priors_given.items():
        if not (np.isfinite(std) and std > 0):
            raise ValueError(f"the prior on {name!r} needs a positive standard deviation, not {std}")
        prior_items.append((name, float(mean), float(std)))
    n_residuals = sum(piece[2].size for own in pieces for piece in own) + len(prior_items)
    dof = n_residuals - len(free)
    if dof <= 0:
        raise ValueError(f"{len(free)} free parameters but only {n_residuals} residuals")

    best = None
    rows = []
    for number, start in enumerate(starts):
        p_start = np.array([float(start[n][0]) for n in free])
        outside = [(n, v) for n, v, lo, hi in zip(free, p_start, lower, upper) if not lo <= v <= hi]
        if outside:
            raise ValueError(f"starting point {number} is outside the bounds: {dict(outside)}")
        scale = np.array([_scale(v, (lo, hi)) for v, lo, hi in zip(p_start, lower, upper)])
        residuals = _Residuals(joint, free, held, scale, member_fields, pieces, prior_items)
        try:
            first = residuals.evaluate(p_start / scale)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"the model fails at starting point {number}: {error}") from error
        if not np.all(np.isfinite(first)):
            raise ValueError(f"the model is not finite at starting point {number}")
        residuals.penalty = np.full(n_residuals, 1e3 * (1.0 + np.max(np.abs(first))))

        # With several datasets, each difference step recomputes only the datasets it affects.
        if len(joint.members) > 1:
            def jacobian(u, residuals=residuals, lo=lower / scale, hi=upper / scale):
                return residuals.jacobian(u, lo, hi)
        else:
            jacobian = "2-point"
        options = {"jac": jacobian, **dict(least_squares_options or {})}
        result = least_squares(residuals, p_start / scale, bounds=(lower / scale, upper / scale), **options)
        fitted = result.x * scale
        chi_squared = float(np.sum(result.fun**2))
        rows.append({**dict(zip(free, fitted)), "chi_squared": chi_squared, "success": bool(result.success)})
        if best is None or chi_squared < best[1]:
            best = (result, chi_squared, scale, fitted)

    result, chi_squared, scale, fitted = best
    values = {**held, **dict(zip(free, fitted))}
    parameters = {name: values[name] for name in joint.parameters}
    covariance, singular = _covariance(result.jac, scale, chi_squared, dof, absolute)
    errors = dict(zip(free, np.sqrt(np.diag(covariance))))
    starts_table = pd.DataFrame(rows)

    _diagnose(free, covariance, errors, starts_table, chi_squared, dof, absolute, result)
    for member, fields in zip(joint.members, member_fields):  # their warnings, once, at the best fit
        member.model.transport(member.local(parameters), fields, member.dataset.layer_spacing, member.dataset.options)

    return _Fitted(joint=joint, names=free, parameters=parameters, errors=errors, covariance=covariance,
                   chi_squared=chi_squared, degrees_of_freedom=dof, absolute=absolute, singular_values=singular,
                   starts=starts_table, result=result,
                   fit_quality=_fit_quality(joint, pieces, result.fun, absolute, len(prior_items)))


def _bound(pair, side, default):
    if pair is None:
        return default
    value = pair[side]
    return default if value is None else float(value)


# Warnings point at the user's call: _diagnose ← _fit ← fit_transport or fit_joint ← user.
_STACKLEVEL = 4


def _diagnose(free, covariance, errors, starts, chi_squared, dof, absolute, result):
    """Warn when the fit does not determine its parameters."""
    if not result.success:
        warnings.warn(f"the fit did not converge: {result.message}", UserWarning, stacklevel=_STACKLEVEL)
    if not np.all(np.isfinite(covariance)):
        warnings.warn("the data do not determine every parameter: some combination of them does not change the "
                      "model. Fix or remove one", UserWarning, stacklevel=_STACKLEVEL)
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
                    "or fix one", UserWarning, stacklevel=_STACKLEVEL)
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
                    UserWarning, stacklevel=_STACKLEVEL)
