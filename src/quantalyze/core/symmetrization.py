import numpy as np
import pandas as pd
from .smoothing import bin


def _mirror(dfs, x_column, y_column, minimum, maximum, step, sign) -> pd.DataFrame:
    """Bin onto a grid symmetric about zero and combine each point with its mirror image."""
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    dfs = [df[(df[x_column] >= minimum) & (df[x_column] <= maximum)] for df in dfs]
    combined = pd.concat(dfs)
    combined = combined.sort_values(by=x_column)
    # Snap the grid to whole steps so it is exactly symmetric about zero; otherwise
    # reversing it would pair each point with the wrong partner.
    limit = step * np.floor(maximum / step + 1e-9)
    combined = bin(combined, x_column, -limit, limit, step)
    values = combined[y_column].values
    return pd.DataFrame(data={
        x_column: combined[x_column].values,
        y_column: (values + sign * values[::-1]) / 2,
    })


def symmetrize(dfs, x_column, y_column, minimum, maximum, step) -> pd.DataFrame:
    """
    Keep only the part of a signal that is even in x: `(y(x) + y(-x)) / 2`.

    The standard use is removing the Hall (odd-in-field) contamination from a
    longitudinal resistance measured in both field directions. The data are combined,
    binned onto an evenly spaced grid that is symmetric about zero, and each point is
    averaged with its mirror image.

    Args:
        dfs (pandas.DataFrame or list of pandas.DataFrame): The data, e.g. `[up_sweep, down_sweep]`.
            Together they must cover both positive and negative x.
        x_column (str): The column to symmetrize about, e.g. `'field'`.
        y_column (str): The column to symmetrize, e.g. `'resistance'`.
        minimum (float): Input rows with x below this are discarded before binning. It
            does **not** set the start of the output grid (see `maximum`).
        maximum (float): The output grid always runs from `-maximum` to `+maximum`. Input
            rows with x above this are discarded.
        step (float): Grid spacing. If `maximum` is not a whole number of steps, the grid
            stops at the largest whole number of steps below it.

    Returns:
        pandas.DataFrame: Two columns, `x_column` (the grid) and `y_column` (the symmetric
            part). Other columns are dropped. Grid points with no data on either side are NaN.

    Examples:
        >>> import quantalyze as qz
        >>> sym = qz.symmetrize(
        ...     [up, down],
        ...     x_column='field',
        ...     y_column='resistance',
        ...     minimum=-9,
        ...     maximum=9,
        ...     step=0.1,
        ... )
    """
    return _mirror(dfs, x_column, y_column, minimum, maximum, step, sign=+1)


def antisymmetrize(dfs, x_column, y_column, minimum, maximum, step) -> pd.DataFrame:
    """
    Keep only the part of a signal that is odd in x: `(y(x) - y(-x)) / 2`.

    The standard use is extracting the Hall resistance from a transverse voltage that
    also picks up some longitudinal (even-in-field) signal from contact misalignment.
    The data are combined, binned onto an evenly spaced grid that is symmetric about
    zero, and each point is differenced with its mirror image.

    Args:
        dfs (pandas.DataFrame or list of pandas.DataFrame): The data, e.g. `[up_sweep, down_sweep]`.
            Together they must cover both positive and negative x.
        x_column (str): The column to antisymmetrize about, e.g. `'field'`.
        y_column (str): The column to antisymmetrize, e.g. `'hall_resistance'`.
        minimum (float): Input rows with x below this are discarded before binning. It
            does **not** set the start of the output grid (see `maximum`).
        maximum (float): The output grid always runs from `-maximum` to `+maximum`. Input
            rows with x above this are discarded.
        step (float): Grid spacing. If `maximum` is not a whole number of steps, the grid
            stops at the largest whole number of steps below it.

    Returns:
        pandas.DataFrame: Two columns, `x_column` (the grid) and `y_column` (the
            antisymmetric part, exactly zero at x = 0). Other columns are dropped. Grid
            points with no data on either side are NaN.

    Examples:
        >>> import quantalyze as qz
        >>> hall = qz.antisymmetrize(
        ...     [up, down],
        ...     x_column='field',
        ...     y_column='hall_resistance',
        ...     minimum=-9,
        ...     maximum=9,
        ...     step=0.1,
        ... )
    """
    return _mirror(dfs, x_column, y_column, minimum, maximum, step, sign=-1)
