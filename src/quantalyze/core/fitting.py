from scipy.optimize import curve_fit
from inspect import signature


class Fit:
    
    def __init__(self, func, popt, pcov):
        self.func = func
        self.popt = popt
        self.pcov = pcov


    def evaluate(self, x):
        """
        Evaluate the fitted function at given data points.

        Parameters:
        x : array-like
            The input data points where the function should be evaluated.

        Returns:
        array-like
            The evaluated values of the fitted function at the given data points.
        """
        return self.func(x, *self.popt)


    def plot(self, ax, x, **kwargs):
        """
        Plot the evaluated function on the given axes.

        Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        x (array-like): The x values to evaluate the function.
        **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
        None
        """
        ax.plot(x, self.evaluate(x), **kwargs)


def fit(func, df, x_column, y_column, x_min=None, x_max=None, y_min=None, y_max=None, p0=None):
    """
    Fits a given function to data in a DataFrame within an optional x and y range.
    
    Parameters:
    func (callable): The function to fit to the data. It should take x data as the first argument and parameters to fit as subsequent arguments.
    df (pandas.DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the column in the DataFrame to use as the x data.
    y_column (str): The name of the column in the DataFrame to use as the y data.
    x_min (float, optional): The minimum value of x to include in the fitting. Defaults to None.
    x_max (float, optional): The maximum value of x to include in the fitting. Defaults to None.
    y_min (float, optional): The minimum value of y to include in the fitting. Defaults to None.
    y_max (float, optional): The maximum value of y to include in the fitting. Defaults to None.
    p0 (array-like, optional): Initial guess for the parameters. Defaults to None.
    
    Returns:
    Fit: An instance of the Fit class containing the fitted function and parameters.
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
        num_params = len(signature(func).parameters) - 1  # Subtract 1 for the x parameter
        if len(p0) != num_params:
            raise ValueError(f"Initial guess p0 must have length {num_params}, but got {len(p0)}.")
    
    # Perform curve fitting
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)
    
    return Fit(func=func, popt=popt, pcov=pcov)