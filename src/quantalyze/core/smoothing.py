import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as scipy_savgol_filter


def _grid(minimum, maximum, width) -> np.ndarray:
    """Evenly spaced points from `minimum` up to (at most) `maximum`, `width` apart."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    if maximum < minimum:
        raise ValueError(f"maximum ({maximum}) must not be less than minimum ({minimum}).")
    count = int(np.floor((maximum - minimum) / width + 1e-9)) + 1
    # When minimum is a whole number of widths, build the grid from integer multiples
    # of width so that points such as 0 come out exactly (and symmetric grids stay symmetric).
    offset = minimum / width
    if np.isclose(offset, round(offset), rtol=1e-9, atol=1e-9):
        return (round(offset) + np.arange(count, dtype=float)) * width
    return minimum + width * np.arange(count, dtype=float)


def bin(dfs, column, minimum, maximum, width) -> pd.DataFrame:
    """
    Average data onto an evenly spaced grid of `column` values.

    The grid runs from `minimum` to `maximum` in steps of `width`. Every row whose
    `column` value falls within `width / 2` of a grid point is assigned to that point,
    and all other columns are averaged within each bin. Use it to put noisy or
    irregularly sampled data onto a regular grid, or to merge several sweeps.

    Args:
        dfs (pandas.DataFrame or list of pandas.DataFrame): The data. A list of DataFrames
            is concatenated and binned together.
        column (str): The column that defines the bins (the x-axis), e.g. `'field'`.
        minimum (float): The first grid point.
        maximum (float): The last grid point. If `maximum - minimum` is not a whole number
            of widths, the grid stops at the last point that does not exceed `maximum`.
        width (float): The spacing between grid points (and the width of each bin).

    Returns:
        pandas.DataFrame: One row per grid point with `column` set to the grid point and
            every other column averaged over the bin. Bins with no data give rows of NaN.
            The input DataFrame(s) are not modified.

    Raises:
        ValueError: If `width` is not positive or `maximum` is less than `minimum`.

    Examples:
        >>> import quantalyze as qz
        >>> binned = qz.bin(df, 'field', minimum=-9, maximum=9, width=0.1)

        Combine an up-sweep and a down-sweep onto one grid:

        >>> binned = qz.bin([up, down], 'field', minimum=-9, maximum=9, width=0.1)
    """
    midpoints = _grid(minimum, maximum, width)
    edges = np.append(midpoints - width / 2, midpoints[-1] + width / 2)

    def bin_single_df(df):
        bins = pd.cut(df[column], bins=edges, include_lowest=True).rename('bin')
        binned = df.groupby(bins, observed=False).mean().reset_index(drop=True)
        binned[column] = midpoints
        return binned

    if isinstance(dfs, list):
        combined_df = pd.concat(dfs, ignore_index=True)
        return bin_single_df(combined_df)
    else:
        return bin_single_df(dfs)


def window(df, column, window_size=5, window_type='triang') -> pd.Series:
    """
    Smooth a column with a centred rolling-window average.

    Each value is replaced by a weighted mean of the `window_size` rows around it.
    The window works on row order, not on x values, so sort the DataFrame by x first
    and use evenly spaced data (see `bin`) for a physically meaningful result.

    Args:
        df (pandas.DataFrame): The data.
        column (str): The column to smooth.
        window_size (int, optional): Number of rows in the window. Larger values smooth
            more. Defaults to 5.
        window_type (str, optional): The window weighting, passed to
            `pandas.DataFrame.rolling(win_type=...)`, e.g. `'triang'`, `'hann'`,
            `'boxcar'` (flat). Defaults to `'triang'`.

    Returns:
        pandas.Series: The smoothed values, aligned with `df.index`. The first and last
            `window_size // 2` values are NaN because the window does not fit there.

    Examples:
        >>> import quantalyze as qz
        >>> df['resistance_smooth'] = qz.window(df, 'resistance', window_size=15)
    """
    return df[column].rolling(window=window_size, win_type=window_type, center=True).mean()


def savgol_filter(df, column, window_size=5, order=2) -> pd.Series:
    """
    Smooth a column with a Savitzky-Golay filter.

    Fits a polynomial of degree `order` to each run of `window_size` rows. Compared
    with `window`, it keeps the height and width of peaks much better and has no
    NaN at the ends. Like `window`, it works on row order, so use sorted, evenly
    spaced data.

    Args:
        df (pandas.DataFrame): The data.
        column (str): The column to smooth.
        window_size (int, optional): Number of rows in each fit. Must be greater than
            `order`; an odd number keeps the filter centred. Defaults to 5.
        order (int, optional): Degree of the fitted polynomial. Defaults to 2.

    Returns:
        pandas.Series: The smoothed values, aligned with `df.index`.

    Examples:
        >>> import quantalyze as qz
        >>> df['resistance_smooth'] = qz.savgol_filter(df, 'resistance', window_size=11, order=3)
    """
    return pd.Series(scipy_savgol_filter(df[column], window_size, order), index=df.index)
