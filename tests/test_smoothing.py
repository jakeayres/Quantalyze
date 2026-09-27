import warnings

import numpy as np
import pandas as pd
import pytest
from quantalyze.core.smoothing import bin, window, savgol_filter


def test_bin_averages_within_bins():
    df = pd.DataFrame({'x': [0.0, 0.1, 0.9, 1.1, 2.0], 'y': [1.0, 3.0, 4.0, 6.0, 7.0]})
    result = bin(df, 'x', minimum=0, maximum=2, width=1)
    expected = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [2.0, 5.0, 7.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_bin_does_not_modify_input():
    df = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [1.0, 2.0, 3.0]})
    original = df.copy()
    bin(df, 'x', minimum=0, maximum=2, width=1)
    pd.testing.assert_frame_equal(df, original)


def test_bin_empty_bins_are_nan():
    df = pd.DataFrame({'x': [0.0, 2.0], 'y': [1.0, 3.0]})
    result = bin(df, 'x', minimum=0, maximum=2, width=1)
    assert np.isnan(result['y'].iloc[1])


def test_bin_list_of_dataframes():
    df1 = pd.DataFrame({'x': [0.0, 1.0], 'y': [1.0, 1.0]})
    df2 = pd.DataFrame({'x': [0.0, 1.0], 'y': [3.0, 5.0]})
    result = bin([df1, df2], 'x', minimum=0, maximum=1, width=1)
    np.testing.assert_allclose(result['y'], [2.0, 3.0])


@pytest.mark.parametrize('maximum, width', [(9, 0.1), (9, 0.01), (14, 0.02), (1, 0.2), (0.5, 0.1)])
def test_bin_grid_ends_at_maximum(maximum, width):
    df = pd.DataFrame({'x': np.linspace(-maximum, maximum, 101), 'y': 1.0})
    result = bin(df, 'x', minimum=-maximum, maximum=maximum, width=width)
    assert result['x'].iloc[0] == pytest.approx(-maximum)
    assert result['x'].iloc[-1] == pytest.approx(maximum)
    np.testing.assert_allclose(result['x'], -result['x'].values[::-1], atol=1e-12)


def test_bin_grid_stops_below_maximum_when_not_a_whole_number_of_widths():
    df = pd.DataFrame({'x': np.linspace(0, 1, 11), 'y': 1.0})
    result = bin(df, 'x', minimum=0, maximum=1, width=0.3)
    np.testing.assert_allclose(result['x'], [0.0, 0.3, 0.6, 0.9])


def test_bin_zero_is_exact():
    df = pd.DataFrame({'x': np.linspace(-9, 9, 11), 'y': 1.0})
    result = bin(df, 'x', minimum=-9, maximum=9, width=0.1)
    assert 0.0 in result['x'].values


def test_bin_no_future_warning():
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [1.0, 2.0]})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bin(df, 'x', minimum=0, maximum=1, width=0.5)


@pytest.mark.parametrize('minimum, maximum, width', [(0, 1, 0), (0, 1, -0.1), (1, 0, 0.1)])
def test_bin_invalid_arguments(minimum, maximum, width):
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [1.0, 2.0]})
    with pytest.raises(ValueError):
        bin(df, 'x', minimum=minimum, maximum=maximum, width=width)


def test_window_smooths_and_leaves_nan_edges():
    df = pd.DataFrame({'y': [0.0, 0.0, 3.0, 0.0, 0.0]})
    result = window(df, 'y', window_size=3, window_type='boxcar')
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[-1])
    np.testing.assert_allclose(result.iloc[1:4], [1.0, 1.0, 1.0])


def test_savgol_filter_preserves_polynomial():
    x = np.linspace(0, 1, 21)
    df = pd.DataFrame({'y': x**2}, index=np.arange(100, 121))
    result = savgol_filter(df, 'y', window_size=5, order=2)
    np.testing.assert_allclose(result.values, x**2, atol=1e-12)
    assert (result.index == df.index).all()
