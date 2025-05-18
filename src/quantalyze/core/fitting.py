from scipy.optimize import curve_fit
from inspect import signature
import pandas as pd


class Fit:
    """
    Represents the result of a fitting operation.

    Attributes:
        function (callable): The fitted function.
        parameters (array-like): Optimal values for the parameters.
        covariance (2D array): The estimated covariance of `parameters`.
    """

    def __init__(self, function, parameters, covariance):
        """
        Initialize a Fit instance.

        Args:
            function (callable): The fitted function.
            parameters (array-like): Optimal values for the parameters.
            covariance (2D array): The estimated covariance of `parameters`.
        """
        self.function = function
        self.parameters = parameters
        self.covariance = covariance

    def evaluate(self, x):
        """
        Evaluate the fitted function at given data points.

        Args:
            x (array-like): The input data points where the function should be evaluated.

        Returns:
            array-like: The evaluated values of the fitted function at the given data points.
        """
        return self.function(x, *self.parameters)

    def plot(self, ax, x, **kwargs) -> None:
        """
        Plot the evaluated function on the given axes.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot.
            x (array-like): The x values to evaluate the function.
            **kwargs: Additional keyword arguments passed to ax.plot().

        Returns:
            None
        """
        ax.plot(x, self.evaluate(x), **kwargs)


def fit(
    function: callable, 
    df: pd.DataFrame, 
    x_column, 
    y_column, 
    x_min=None, 
    x_max=None, 
    y_min=None, 
    y_max=None, 
    p0=None
) -> Fit:
    """
    Fits a given function to data in a DataFrame within an optional x and y range.

    Args:
        function (callable): The function to fit to the data. It should take x data as the first argument 
            and parameters to fit as subsequent arguments.
        df (pandas.DataFrame): The input DataFrame containing the data.
        x_column (str): The name of the column in the DataFrame to use as the x data.
        y_column (str): The name of the column in the DataFrame to use as the y data.
        x_min (float, optional): The minimum value of x to include in the fitting. Defaults to None.
        x_max (float, optional): The maximum value of x to include in the fitting. Defaults to None.
        y_min (float, optional): The minimum value of y to include in the fitting. Defaults to None.
        y_max (float, optional): The maximum value of y to include in the fitting. Defaults to None.
        p0 (array-like, optional): Initial guess for the parameters. Must have a length equal to that of the number of free parameters in `function` Defaults to None.

    Returns:
        Fit: An instance of the Fit class containing the fitted function and parameters.

    Raises:
        RuntimeError: If there is no data in the specified x or y range to fit.
        ValueError: If the initial guess `p0` does not have the correct length.
    """
    # Filter the dataframe based on x_min and x_max
    if x_min is not None:
        df = df[df[x_column] >= x_min]
    if x_max is not None:
        df = df[df[x_column] <= x_max]
    
    # Filter the dataframe based on y_min and y_max
    if y_min is not None:
        df = df[df[y_column] >= y_min]
    if y_max is not None:
        df = df[df[y_column] <= y_max]
    
    # Extract x and y data
    x_data = df[x_column].values
    y_data = df[y_column].values
    
    # Check if there is data to fit
    if len(x_data) == 0 or len(y_data) == 0:
        raise RuntimeError("No data in the specified x or y range to fit.")
    
    # Check if p0 is provided and has the correct length
    if p0 is not None:
        num_params = len(signature(function).parameters) - 1  # Subtract 1 for the x parameter
        if len(p0) != num_params:
            raise ValueError(f"Initial guess p0 must have length {num_params}, but got {len(p0)}.")
    
    # Perform curve fitting
    parameters, covariance = curve_fit(function, x_data, y_data, p0=p0)
    
    return Fit(function=function, parameters=parameters, covariance=covariance)