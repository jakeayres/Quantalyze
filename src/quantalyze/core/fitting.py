from scipy.optimize import curve_fit


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
    
    # Check if there is data to fit
    if len(x_data) == 0 or len(y_data) == 0:
        raise RuntimeError("No data in the specified x range to fit.")
    
    # Perform curve fitting
    popt, pcov = curve_fit(func, x_data, y_data)
    
    return Fit(func=func, popt=popt, pcov=pcov)