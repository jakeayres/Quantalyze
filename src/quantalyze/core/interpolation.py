import pandas as pd
import numpy as np
from typing import Union, Sequence
from scipy.interpolate import interp1d

def interpolate(df: pd.DataFrame, x_column: str, onto: Union[np.ndarray, Sequence], method: str = 'linear') -> pd.DataFrame:
    """
    Resample every column of a DataFrame onto new x values.

    Use it to put irregularly logged data onto a regular grid, or to put two
    measurements onto the same x values so they can be compared row by row.

    Warning:
        New x values outside the range of the data are **extrapolated** (a line, or
        curve for higher-order methods, is extended from the nearest points), not
        set to NaN. Keep `onto` inside `df[x_column].min()` to `df[x_column].max()`.

    Args:
        df (pd.DataFrame): The data. Every column other than `x_column` must be numeric.
            Rows do not need to be sorted.
        x_column (str): The column to interpolate along, e.g. `'temperature'`.
        onto (Union[np.ndarray, Sequence]): The new x values.
        method (str): How to interpolate, passed to `scipy.interpolate.interp1d(kind=...)`:
            `'linear'` (default), `'nearest'`, `'quadratic'`, `'cubic'`, ...

    Returns:
        pd.DataFrame: A new DataFrame with `x_column` equal to `onto` and every other
            column interpolated. It has a fresh 0, 1, 2, ... index.

    Raises:
        ValueError: If `x_column` is not a column of `df`.

    Examples:
        >>> import numpy as np
        >>> import quantalyze as qz
        >>> regular = qz.interpolate(df, 'temperature', onto=np.arange(10, 300, 10))
        >>> smooth = qz.interpolate(df, 'temperature', onto=np.linspace(2, 300, 1000), method='cubic')
    """
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame.")

    onto = np.asarray(onto)  # Ensure 'onto' is converted to a NumPy array
    interpolated_data = {x_column: onto}
    for column in df.columns:
        if column != x_column:
            interpolator = interp1d(df[x_column], df[column], kind=method, bounds_error=False, fill_value="extrapolate")
            interpolated_data[column] = interpolator(onto)

    return pd.DataFrame(interpolated_data)
