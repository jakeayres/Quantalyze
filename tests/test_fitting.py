import pytest
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from quantalyze.core.fitting import fit


def test_fit_linear_function():
    # Define a linear function
    def linear_func(x, a, b):
        return a * x + b

    # Create a DataFrame with linear data
    data = {
        'x': np.linspace(0, 10, 100),
        'y': 3 * np.linspace(0, 10, 100) + 2
    }
    df = pd.DataFrame(data)

    # Fit the linear function to the data
    popt, pcov = fit(linear_func, df, 'x', 'y')

    # Assert the optimal parameters are close to the expected values
    assert np.isclose(popt[0], 3, atol=1e-2)
    assert np.isclose(popt[1], 2, atol=1e-2)

def test_fit_with_x_range():
    # Define a quadratic function
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    # Create a DataFrame with quadratic data
    data = {
        'x': np.linspace(0, 10, 100),
        'y': 2 * np.linspace(0, 10, 100)**2 + 3 * np.linspace(0, 10, 100) + 1
    }
    df = pd.DataFrame(data)

    # Fit the quadratic function to the data within a specific x range
    popt, pcov = fit(quadratic_func, df, 'x', 'y', x_min=2, x_max=8)

    # Assert the optimal parameters are close to the expected values
    assert np.isclose(popt[0], 2, atol=1e-2)
    assert np.isclose(popt[1], 3, atol=1e-2)
    assert np.isclose(popt[2], 1, atol=1e-2)

def test_fit_exponential_function():
    # Define an exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    # Create a DataFrame with exponential data
    data = {
        'x': np.linspace(0, 5, 100),
        'y': 2 * np.exp(1.5 * np.linspace(0, 5, 100))
    }
    df = pd.DataFrame(data)

    # Fit the exponential function to the data
    popt, pcov = fit(exponential_func, df, 'x', 'y')

    # Assert the optimal parameters are close to the expected values
    assert np.isclose(popt[0], 2, atol=1e-2)
    assert np.isclose(popt[1], 1.5, atol=1e-2)
