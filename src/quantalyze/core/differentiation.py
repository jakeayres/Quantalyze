import pandas as pd


def forward_difference(df, x_column, y_column) -> pd.Series:
    """
    Slope from each row to the next row: `(y[i+1] - y[i]) / (x[i+1] - x[i])`.

    Args:
        df (pandas.DataFrame): The data, ordered by `x_column` (ascending or descending).
        x_column (str): The column to differentiate with respect to, e.g. `'field'`.
        y_column (str): The column to differentiate, e.g. `'resistance'`.

    Returns:
        pandas.Series: dy/dx, aligned with `df.index`. The last value is NaN.

    Examples:
        >>> import quantalyze as qz
        >>> df['dR/dB'] = qz.forward_difference(df, 'field', 'resistance')
    """
    forward_diff = df[y_column].diff().shift(-1) / df[x_column].diff().shift(-1)
    return forward_diff


def backward_difference(df, x_column, y_column) -> pd.Series:
    """
    Slope from the previous row to each row: `(y[i] - y[i-1]) / (x[i] - x[i-1])`.

    Args:
        df (pandas.DataFrame): The data, ordered by `x_column` (ascending or descending).
        x_column (str): The column to differentiate with respect to, e.g. `'field'`.
        y_column (str): The column to differentiate, e.g. `'resistance'`.

    Returns:
        pandas.Series: dy/dx, aligned with `df.index`. The first value is NaN.

    Examples:
        >>> import quantalyze as qz
        >>> df['dR/dB'] = qz.backward_difference(df, 'field', 'resistance')
    """
    backward_diff = df[y_column].diff() / df[x_column].diff()
    return backward_diff


def central_difference(df, x_column, y_column) -> pd.Series:
    """
    The average of the forward and backward differences at each row.

    More accurate than either one-sided difference: for evenly spaced data it is exact
    for any quadratic.

    Args:
        df (pandas.DataFrame): The data, ordered by `x_column` (ascending or descending).
        x_column (str): The column to differentiate with respect to, e.g. `'field'`.
        y_column (str): The column to differentiate, e.g. `'resistance'`.

    Returns:
        pandas.Series: dy/dx, aligned with `df.index`. The first and last values are NaN.

    Examples:
        >>> import quantalyze as qz
        >>> df['dR/dB'] = qz.central_difference(df, 'field', 'resistance')
    """
    forward_diff = forward_difference(df, x_column, y_column)
    backward_diff = backward_difference(df, x_column, y_column)
    central_diff = (forward_diff + backward_diff) / 2
    return central_diff


def derivative(df, x_column, y_column) -> pd.Series:
    """
    The numerical derivative dy/dx. Start here: it is the same as `central_difference`.

    Rows must be ordered by `x_column` and each x value must appear only once (bin
    repeated sweeps with `bin` first). Differentiation amplifies noise, so smooth noisy
    data (e.g. with `savgol_filter`) before differentiating.

    Args:
        df (pandas.DataFrame): The data, ordered by `x_column` (ascending or descending).
        x_column (str): The column to differentiate with respect to, e.g. `'field'`.
        y_column (str): The column to differentiate, e.g. `'resistance'`.

    Returns:
        pandas.Series: dy/dx, aligned with `df.index`. The first and last values are NaN.

    Examples:
        >>> import quantalyze as qz
        >>> df['dR/dB'] = qz.derivative(df, 'field', 'resistance')
    """
    return central_difference(df, x_column, y_column)
