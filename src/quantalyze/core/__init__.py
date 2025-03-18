from .fitting import Fit, fit
from .symmetrize import symmetrize, antisymmetrize
from .smoothing import bin, window, savgol_filter



def derivative(df, x_column, y_column):
    """
    Calculate the derivative of the y_column with respect to the x_column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the column representing the x-axis.
    y_column (str): The name of the column representing the y-axis.

    Returns:
    pandas.Series: A Series with the derivative values.
    """
    forward_diff = df[y_column].diff().shift(-1) / df[x_column].diff().shift(-1)
    backward_diff = df[y_column].diff() / df[x_column].diff()
    
    derivative = (forward_diff + backward_diff) / 2

    return derivative
    