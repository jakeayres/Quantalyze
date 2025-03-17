import pandas as pd
from scipy.optimize import curve_fit

def fit(func, df, x_column, y_column, x_min=None, x_max=None):
    """
    Fits a given function to data in a DataFrame within an optional x range.
    Parameters:
    func (callable): The function to fit to the data. It should take x data as the first argument and parameters to fit as subsequent arguments.
    df (pandas.DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the column in the DataFrame to use as the x data.
    y_column (str): The name of the column in the DataFrame to use as the y data.
    x_min (float, optional): The minimum value of x to include in the fitting. Defaults to None.
    x_max (float, optional): The maximum value of x to include in the fitting. Defaults to None.
    Returns:
    tuple: A tuple containing:
        - popt (array): Optimal values for the parameters so that the sum of the squared residuals of `func(x_data, *popt) - y_data` is minimized.
        - pcov (2-D array): The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
    """
    # Filter the dataframe based on x_min and x_max
    if x_min is not None:
        df = df[df[x_column] >= x_min]
    if x_max is not None:
        df = df[df[x_column] <= x_max]
    
    # Extract x and y data
    x_data = df[x_column].values
    y_data = df[y_column].values
    
    # Perform curve fitting
    popt, pcov = curve_fit(func, x_data, y_data)
    
    return popt, pcov


def evaluate(func, popt, x):
    """
    Evaluates the fitted function with the optimal parameters at given x values.
    Parameters:
    func (callable): The function to evaluate. It should take x data as the first argument and parameters to fit as subsequent arguments.
    popt (array): Optimal values for the parameters obtained from the fitting.
    x (array-like): The x values at which to evaluate the function.
    Returns:
    array: The y values of the function evaluated at the given x values.
    """
    return func(x, *popt)