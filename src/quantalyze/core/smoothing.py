import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as scipy_savgol_filter


def bin(dfs, column, minimum, maximum, width) -> pd.DataFrame:
    """
    Bins the data in the specified column of the DataFrame(s) into equal-width bins and computes the mean of each bin.

    Args:
        dfs (pandas.DataFrame or list of pandas.DataFrame): The input DataFrame(s) containing the data to be binned.
        column (str): The name of the column in the DataFrame(s) to be binned.
        minimum (float): The minimum value of the range to be binned.
        maximum (float): The maximum value of the range to be binned.
        width (float): The width of each bin.

    Returns:
        pandas.DataFrame: A DataFrame containing the mean values of each bin, with the bin midpoints as the values in the specified column.
    """
    def bin_single_df(df):
        bin_edges = np.arange(minimum - width / 2, maximum + width / 2 + width, width)
        bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
        df['bin'] = pd.cut(df[column], bins=bin_edges, include_lowest=True)
        binned = df.groupby('bin').agg('mean').reset_index()
        binned[column] = bin_midpoints
        binned = binned.drop('bin', axis=1)
        return binned

    if isinstance(dfs, list):
        combined_df = pd.concat(dfs, ignore_index=True)
        return bin_single_df(combined_df)
    else:
        return bin_single_df(dfs)


def window(df, column, window_size=5, window_type='triang') -> pd.Series:
    """
    Applies a rolling window smoothing to a specified column in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to which the rolling window smoothing will be applied.
        window_size (int, optional): The size of the rolling window. Default is 5.
        window_type (str, optional): The type of window to use. Default is 'triang'.

    Returns:
        pandas.Series: A Series with the smoothed values.
    """
    return df[column].rolling(window=window_size, win_type=window_type, center=True).mean()


def savgol_filter(df, column, window_size=5, order=2) -> pd.Series:
    """
    Applies a Savitzky-Golay filter to a specified column in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to which the Savitzky-Golay filter will be applied.
        window_size (int, optional): The size of the window. Default is 5.
        order (int, optional): The order of the polynomial used to fit the samples. Default is 2.

    Returns:
        pandas.Series: A Series with the smoothed values.
    """
    return pd.Series(scipy_savgol_filter(df[column], window_size, order), index=df.index)


