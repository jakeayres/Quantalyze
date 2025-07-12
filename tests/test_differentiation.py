import pytest
import pandas as pd
from quantalyze.core.differentiation import derivative
from quantalyze.core.differentiation import forward_difference, backward_difference, central_difference, derivative


def test_forward_difference():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 4, 9, 16]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([1, 3, 5, 7, None])
    result = forward_difference(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_backward_difference():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 4, 9, 16]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([None, 1, 3, 5, 7])
    result = backward_difference(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_central_difference():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 4, 9, 16]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([None, 2, 4, 6, None])
    result = central_difference(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_derivative_linear():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([None, 1, 1, 1, None])
    result = derivative(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_derivative_quadratic():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 4, 9, 16]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([None, 2, 4, 6, None])
    result = derivative(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_derivative_constant():
    data = {
        'x': [0, 1, 2, 3, 4],
        'y': [5, 5, 5, 5, 5]
    }
    df = pd.DataFrame(data)
    expected = pd.Series([None, 0, 0, 0, None])
    result = derivative(df, 'x', 'y')
    pd.testing.assert_series_equal(result, expected, check_names=False)