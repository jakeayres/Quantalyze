import pandas as pd


def forward_difference(df, x_column, y_column) -> pd.Series:
    """
    Calculates the forward difference of a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x_column (str): The name of the column representing the x-values.
        y_column (str): The name of the column representing the y-values.

    Returns:
        pandas.Series: A Series containing the forward differences of the y-values with respect to the x-values.
        
    Examples:
        >>> import quantalize as qz
        >>> df['forward_diff'] = qz.forward_difference(df, 'x', 'y')
    """
    forward_diff = df[y_column].diff().shift(-1) / df[x_column].diff().shift(-1)
    return forward_diff


def backward_difference(df, x_column, y_column) -> pd.Series:
    """
    Calculates the backward difference of a DataFrame column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x_column (str): The name of the column to use as the x-values.
        y_column (str): The name of the column to use as the y-values.

    Returns:
        pandas.Series: The backward difference of the y_column with respect to the x_column.
        
    Examples:
        >>> import quantalize as qz
        >>> df['backward_diff'] = qz.backward_difference(df, 'x', 'y')
    """
    backward_diff = df[y_column].diff() / df[x_column].diff()
    return backward_diff


def central_difference(df, x_column, y_column) -> pd.Series:
    """
    Calculates the central difference for a given DataFrame.

    The central difference is computed as the average of the forward difference
    and the backward difference for the specified columns.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        x_column (str): The name of the column representing the x-values.
        y_column (str): The name of the column representing the y-values.

    Returns:
        pandas.Series: A Series containing the central difference values.
        
    Examples:
        >>> import quantalize as qz
        >>> df['central_diff'] = qz.central_difference(df, 'x', 'y')
    """
    forward_diff = forward_difference(df, x_column, y_column)
    backward_diff = backward_difference(df, x_column, y_column)
    central_diff = (forward_diff + backward_diff) / 2
    return central_diff



def derivative(df, x_column, y_column) -> pd.Series:
    """
    Calculates the derivative of the y_column with respect to the x_column.

    This method computes the numerical derivative of a given y_column with respect to an x_column
    in a pandas DataFrame using the central difference method.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        x_column (str): The name of the column representing the x-axis.
        y_column (str): The name of the column representing the y-axis.

    Returns:
        pandas.Series: A Series with the derivative values.
        
    Examples:
        >>> import quantalize as qz
        >>> df['derivative'] = qz.derivative(df, 'x', 'y')
    """
    return central_difference(df, x_column, y_column)
