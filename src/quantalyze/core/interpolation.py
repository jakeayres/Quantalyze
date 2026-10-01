import pandas as pd
import numpy as np
from typing import Union, Sequence
from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator

# Methods handled by scipy.interpolate.interp1d, and the fewest points each needs.
_INTERP1D_METHODS = {
    'linear': 2, 'slinear': 2, 'nearest': 2, 'nearest-up': 2, 'previous': 2, 'next': 2, 'zero': 2,
    'quadratic': 3, 'cubic': 4,
}
# Methods that never overshoot the data, and the fewest points each needs.
_LOCAL_METHODS = {'pchip': 2, 'akima': 3}


def _points_needed(method) -> int:
    """The fewest data points `method` can interpolate, or raise if it is not a method."""
    if isinstance(method, (int, np.integer)) and not isinstance(method, bool) and method >= 0:
        return max(2, int(method) + 1)
    if isinstance(method, str) and method in _INTERP1D_METHODS:
        return _INTERP1D_METHODS[method]
    if isinstance(method, str) and method in _LOCAL_METHODS:
        return _LOCAL_METHODS[method]
    options = ", ".join(repr(m) for m in [*_INTERP1D_METHODS, *_LOCAL_METHODS])
    raise ValueError(f"Unknown method {method!r}. Use one of {options}, or a spline order as an int.")


def interpolate(df: pd.DataFrame, x_column: str, onto: Union[np.ndarray, Sequence], method: str = 'linear', extrapolate: bool = False) -> pd.DataFrame:
    """
    Resample every column of a DataFrame onto new x values.

    Use it to put irregularly logged data onto a regular grid, or to put two
    measurements onto the same x values so they can be compared row by row.

    Each column is interpolated using only the rows where both it and `x_column`
    are not NaN, so a column with gaps doesn't spoil the others. Rows that share
    an x value (e.g. a thermometer reading that didn't change) are averaged first.

    Warning:
        New x values outside the range of the data are set to NaN, because there is
        nothing to interpolate between. Pass `extrapolate=True` to extend the curve
        from the nearest points instead. For a column with gaps, the range is the
        range of its own non-NaN values.

    Args:
        df (pd.DataFrame): The data. Every column must be numeric. Rows do not need
            to be sorted.
        x_column (str): The column to interpolate along, e.g. `'temperature'`.
        onto (Union[np.ndarray, Sequence]): The new x values.
        method (str): How to interpolate between points:
            `'linear'` (default) joins them with straight lines.
            `'cubic'` and `'quadratic'` fit a smooth spline through every point, which
            follows a smoothly curving function well but can overshoot between points.
            `'pchip'` and `'akima'` are smooth but never overshoot, so suit sharp
            features such as a transition.
            `'nearest'`, `'previous'` and `'next'` copy a neighbouring point.
            Other `scipy.interpolate.interp1d(kind=...)` options also work.
        extrapolate (bool): If `True`, new x values outside the range of the data are
            extended from the nearest points (a line, or curve for higher-order
            methods) instead of being set to NaN. Defaults to `False`.

    Returns:
        pd.DataFrame: A new DataFrame with `x_column` equal to `onto` and every other
            column interpolated. It has a fresh 0, 1, 2, ... index.

    Raises:
        ValueError: If `x_column` is not a column of `df`, a column is not numeric,
            `method` is not recognised, or a column has too few non-NaN points for
            `method` (e.g. a cubic needs at least 4 distinct x values).

    Examples:
        >>> import numpy as np
        >>> import quantalyze as qz
        >>> regular = qz.interpolate(df, 'temperature', onto=np.arange(10, 300, 10))
        >>> smooth = qz.interpolate(df, 'temperature', onto=np.linspace(2, 300, 1000), method='pchip')
        >>> extended = qz.interpolate(df, 'temperature', onto=[0, 400], extrapolate=True)
    """
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame.")
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' is not numeric ({df[column].dtype}). "
                f"Select the numeric columns first, e.g. df[['{x_column}', ...]]."
            )
    needed = _points_needed(method)

    onto = np.asarray(onto)  # Ensure 'onto' is converted to a NumPy array
    interpolated_data = {x_column: onto}
    for column in df.columns:
        if column == x_column:
            continue

        # Drop this column's NaN rows, then average rows that share an x value (this also sorts by x).
        valid = df[[x_column, column]].dropna()
        averaged = valid.groupby(x_column, sort=True)[column].mean()
        x = averaged.index.to_numpy(dtype=float)
        y = averaged.to_numpy(dtype=float)
        if len(x) < needed:
            raise ValueError(
                f"Column '{column}' has {len(x)} distinct x value(s) with data, but method={method!r} "
                f"needs at least {needed}. Drop the column or use a lower-order method."
            )

        if method in _LOCAL_METHODS:
            interpolator_class = PchipInterpolator if method == 'pchip' else Akima1DInterpolator
            interpolated_data[column] = interpolator_class(x, y)(onto, extrapolate=extrapolate)
        else:
            fill_value = "extrapolate" if extrapolate else np.nan
            interpolator = interp1d(x, y, kind=method, bounds_error=False, fill_value=fill_value, assume_sorted=True)
            interpolated_data[column] = interpolator(onto)

    return pd.DataFrame(interpolated_data)
