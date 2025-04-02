import pandas as pd
import numpy as np
from typing import Union, Sequence
from scipy.interpolate import interp1d

def interpolate(df: pd.DataFrame, x_column: str, onto: Union[np.ndarray, Sequence], method: str = 'linear') -> pd.DataFrame:
    """
    Interpolates all columns of a DataFrame onto new x values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_column (str): The name of the column to use as the x-axis.
        onto (Union[np.ndarray, Sequence]): The new x values to interpolate onto.
        method (str): The interpolation method to use (default is 'linear'). Is passed to np.interp1d(kind=...).

    Returns:
        pd.DataFrame: A new DataFrame with interpolated values.
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