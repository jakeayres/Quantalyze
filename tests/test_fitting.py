import pytest
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from quantalyze.core.fitting import fit, Fit


def test_fit_linear_function():
    # Define a linear function
    def linear_func(x, a, b):
        return a * x + b

    # Create a DataFrame with linear data
    df = pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': 3 * np.linspace(0, 10, 100) + 2
    })

    # Fit the linear function to the data
    fit_result = fit(linear_func, df, 'x', 'y')

    # Assert that the optimal parameters are close to the expected values
    np.testing.assert_allclose(fit_result.popt, [3, 2], rtol=1e-2)


def test_fit_with_x_range():
    # Define a quadratic function
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    # Create a DataFrame with quadratic data
    df = pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': 2 * np.linspace(0, 10, 100)**2 + 3 * np.linspace(0, 10, 100) + 1
    })

    # Fit the quadratic function to the data within a specific x range
    fit_result = fit(quadratic_func, df, 'x', 'y', x_min=2, x_max=8)

    # Assert that the optimal parameters are close to the expected values
    np.testing.assert_allclose(fit_result.popt, [2, 3, 1], rtol=1e-2)


def test_fit_with_no_data_in_range():
    # Define a linear function
    def linear_func(x, a, b):
        return a * x + b

    # Create a DataFrame with linear data
    df = pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': 3 * np.linspace(0, 10, 100) + 2
    })

    # Fit the linear function to the data with an x range that has no data
    with pytest.raises(RuntimeError):
        fit(linear_func, df, 'x', 'y', x_min=20, x_max=30)