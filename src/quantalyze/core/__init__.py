from .fitting import Fit, fit
from .symmetrize import symmetrize, antisymmetrize
from .smoothing import bin, window, savgol_filter



def forward_difference(df, x_column, y_column):
    """
    Calculate the forward difference of a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column representing the x-values.
    y_column (str): The name of the column representing the y-values.

    Returns:
    pandas.Series: A Series containing the forward differences of the y-values with respect to the x-values.
    """
    forward_diff = df[y_column].diff().shift(-1) / df[x_column].diff().shift(-1)
    return forward_diff


def backward_difference(df, x_column, y_column):
    """
    Calculate the backward difference of a DataFrame column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column to use as the x-values.
    y_column (str): The name of the column to use as the y-values.

    Returns:
    pandas.Series: The backward difference of the y_column with respect to the x_column.
    """
    backward_diff = df[y_column].diff() / df[x_column].diff()
    return backward_diff


def central_difference(df, x_column, y_column):
    """
    Calculate the central difference for a given DataFrame.

    The central difference is computed as the average of the forward difference
    and the backward difference for the specified columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the column representing the x-values.
    y_column (str): The name of the column representing the y-values.

    Returns:
    pandas.Series: A Series containing the central difference values.
    """
    forward_diff = forward_difference(df, x_column, y_column)
    backward_diff = backward_difference(df, x_column, y_column)
    central_diff = (forward_diff + backward_diff) / 2
    return central_diff



def derivative(df, x_column, y_column):
    """
    Calculate the derivative of the y_column with respect to the x_column.

    This method computes the numerical derivative of a given y_column with respect to an x_column
    in a pandas DataFrame using central difference method.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the column representing the x-axis.
    y_column (str): The name of the column representing the y-axis.

    Returns:
    pandas.Series: A Series with the derivative values.
    """
    return central_difference(df, x_column, y_column)
    