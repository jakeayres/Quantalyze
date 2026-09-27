from scipy.optimize import curve_fit
from inspect import signature
import pandas as pd


class Fit:
    """
    The result of `fit`: the fitted function, its best-fit parameters and their covariance.

    You get one back from `qz.fit(...)`; you don't normally create it yourself.

    Attributes:
        function (callable): The function that was fitted.
        parameters (numpy.ndarray): Best-fit parameter values, in the order they appear in
            `function`'s signature (after x).
        covariance (numpy.ndarray): The estimated covariance of `parameters`. One-standard-
            deviation errors are `numpy.sqrt(numpy.diag(covariance))`. Entries of `inf`
            mean the fit could not estimate them, a sign of a bad fit.

    Examples:
        >>> result = qz.fit(lambda T, rho0, A: rho0 + A * T**2, df, 'temperature', 'resistivity')
        >>> result['A']                  # a parameter by name
        >>> result[0]                    # or by position
        >>> result.parameters            # all of them
        >>> result.evaluate(4.2)         # the fitted curve at T = 4.2
        >>> result.plot(ax, np.linspace(0, 40, 200))
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
        The fitted curve at `x`: `function(x, *parameters)`.

        Args:
            x (float, array-like or pandas.Series): Where to evaluate it.

        Returns:
            float, numpy.ndarray or pandas.Series: The fitted values, the same shape and type as `x`.

        Examples:
            >>> result.evaluate(0)                      # extrapolate to x = 0
            >>> df['fit'] = result.evaluate(df['temperature'])
        """
        return self.function(x, *self.parameters)

    def plot(self, ax, x, **kwargs) -> None:
        """
        Draw the fitted curve on a Matplotlib axes.

        Args:
            ax (matplotlib.axes.Axes): The axes to draw on.
            x (array-like): The x values to draw the curve at. They can extend beyond the
                fitted range, e.g. `numpy.linspace(0, 40, 200)`.
            **kwargs: Passed to `ax.plot()`, e.g. `color`, `linestyle`, `label`.

        Returns:
            None

        Examples:
            >>> fig, ax = plt.subplots()
            >>> ax.plot(df['temperature'], df['resistivity'], '.')
            >>> result.plot(ax, np.linspace(0, 40, 200), color='k', label='fit')
        """
        ax.plot(x, self.evaluate(x), **kwargs)


    def _function_args(self):
        """
        Get the names of the parameters of the fitted function.

        Returns:
            list: A list of parameter names.
        """
        return list(signature(self.function).parameters.keys())[1:]
    

    def __getitem__(self, key):
        """
        A best-fit parameter, by name (`result['A']`) or by position (`result[1]`).

        Args:
            key (str or int): The parameter's name in the fitted function's signature, or
                its position among the parameters (0 is the first after x).

        Returns:
            float: The parameter's best-fit value.

        Raises:
            KeyError: If no parameter has that name.
            IndexError: If the position is out of range.
        """

        if isinstance(key, str):
            if key in self._function_args():
                return self.parameters[self._function_args().index(key)]
            else:
                raise KeyError(f"Parameter '{key}' not found in fitted function.")
        elif isinstance(key, int):
            if 0 <= key < len(self.parameters):
                return self.parameters[key]
            else:
                raise IndexError(f"Index {key} out of range for parameters.")
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
        
def fit(
    function: callable, 
    df: pd.DataFrame, 
    x_column, 
    y_column, 
    x_min=None, 
    x_max=None, 
    y_min=None, 
    y_max=None, 
    p0=None,
    **kwargs,
) -> Fit:
    """
    Least-squares fit of `function` to two columns of a DataFrame.

    A thin wrapper around `scipy.optimize.curve_fit` that takes column names, can
    restrict the fit to a range of x and y, and returns a `Fit` you can query by
    parameter name, evaluate and plot.

    Args:
        function (callable): The model, written as `f(x, param1, param2, ...)`. The first
            argument is x; every other argument is a parameter to fit. Parameter names
            become the keys of the returned `Fit` (e.g. `result['param1']`).
            A `lambda` works too.
        df (pandas.DataFrame): The data.
        x_column (str): The column to use as x, e.g. `'temperature'`.
        y_column (str): The column to use as y, e.g. `'resistivity'`.
        x_min (float, optional): Only fit rows with x >= x_min.
        x_max (float, optional): Only fit rows with x <= x_max.
        y_min (float, optional): Only fit rows with y >= y_min.
        y_max (float, optional): Only fit rows with y <= y_max.
        p0 (array-like, optional): Starting guesses for the parameters, in signature order.
            Defaults to 1 for every parameter. Give one whenever the answer is far from 1
            or the model is nonlinear (peaks, exponentials, oscillations): a bad start can
            silently converge to a wrong answer.
        **kwargs: Passed to `scipy.optimize.curve_fit`, e.g. `bounds`, `sigma`, `maxfev`.

    Returns:
        Fit: The result. Use `result['name']`, `result.parameters`, `result.covariance`,
            `result.evaluate(x)` and `result.plot(ax, x)`.

    Raises:
        RuntimeError: If no rows are left after applying the x and y limits, or if
            `curve_fit` fails to converge.
        ValueError: If `p0` does not have one value per parameter.

    Examples:
        >>> import quantalyze as qz
        >>> def fermi_liquid(temperature, rho0, A):
        ...     return rho0 + A * temperature**2
        >>> result = qz.fit(fermi_liquid, df, 'temperature', 'resistivity', x_max=20)
        >>> result['A']
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
    parameters, covariance = curve_fit(function, x_data, y_data, p0=p0, **kwargs)
    
    return Fit(function=function, parameters=parameters, covariance=covariance)