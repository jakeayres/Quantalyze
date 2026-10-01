import numpy as np
import pandas as pd
import pytest
from quantalyze.core.interpolation import interpolate


def test_interpolate_linear_between_points():
    df = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [0.0, 10.0, 0.0], 'z': [1.0, 2.0, 3.0]})
    result = interpolate(df, 'x', onto=[0.5, 1.5])
    expected = pd.DataFrame({'x': [0.5, 1.5], 'y': [5.0, 5.0], 'z': [1.5, 2.5]})
    pd.testing.assert_frame_equal(result, expected)


def test_interpolate_unsorted_input_and_onto():
    df = pd.DataFrame({'x': [2.0, 0.0, 1.0], 'y': [20.0, 0.0, 10.0]})
    result = interpolate(df, 'x', onto=[1.5, 0.25])
    np.testing.assert_allclose(result['y'], [15.0, 2.5])


def test_interpolate_does_not_modify_input():
    df = pd.DataFrame({'x': [2.0, 0.0, 1.0, np.nan], 'y': [20.0, 0.0, 10.0, 5.0]})
    original = df.copy()
    interpolate(df, 'x', onto=[0.5])
    pd.testing.assert_frame_equal(df, original)


def test_interpolate_fresh_index():
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]}, index=[10, 20])
    onto = pd.Series([0.25, 0.75], index=[5, 6])
    result = interpolate(df, 'x', onto=onto)
    assert list(result.index) == [0, 1]


def test_interpolate_outside_range_is_nan_by_default():
    df = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0]})
    result = interpolate(df, 'x', onto=[-1.0, 0.0, 2.0, 3.0])
    np.testing.assert_allclose(result['y'], [np.nan, 0.0, 2.0, np.nan])


@pytest.mark.parametrize('method', ['linear', 'cubic', 'pchip', 'akima', 'nearest'])
def test_interpolate_extrapolate(method):
    df = pd.DataFrame({'x': np.arange(6.0), 'y': 2 * np.arange(6.0)})
    result = interpolate(df, 'x', onto=[-1.0, 6.0], method=method, extrapolate=True)
    assert np.all(np.isfinite(result['y']))
    if method != 'nearest':
        np.testing.assert_allclose(result['y'], [-2.0, 12.0])


def test_interpolate_extrapolate_matches_previous_behaviour():
    df = pd.DataFrame({'x': [0.0, 1.0, 2.0, 4.0], 'y': [1.0, 3.0, 2.0, 5.0]})
    onto = np.array([-1.0, 0.5, 3.0, 6.0])
    from scipy.interpolate import interp1d
    for method in ['linear', 'cubic']:
        old = interp1d(df['x'], df['y'], kind=method, bounds_error=False, fill_value='extrapolate')(onto)
        new = interpolate(df, 'x', onto=onto, method=method, extrapolate=True)['y']
        np.testing.assert_allclose(new, old)


def test_interpolate_nan_rows_dropped_per_column():
    df = pd.DataFrame({
        'x': [0.0, 1.0, 2.0, 3.0, np.nan],
        'y': [0.0, np.nan, 2.0, 3.0, 100.0],
        'z': [5.0, 5.0, 5.0, np.nan, 5.0],
    })
    result = interpolate(df, 'x', onto=[0.5, 1.0, 2.5])
    np.testing.assert_allclose(result['y'], [0.5, 1.0, 2.5])
    # z has no data above x = 2, so 2.5 is outside its range.
    np.testing.assert_allclose(result['z'], [5.0, 5.0, np.nan])


def test_interpolate_repeated_x_values_averaged():
    df = pd.DataFrame({'x': [0.0, 0.0, 1.0, 1.0, 2.0], 'y': [1.0, 3.0, 4.0, 6.0, 8.0]})
    result = interpolate(df, 'x', onto=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(result['y'], [2.0, 3.5, 5.0])


def test_interpolate_repeated_x_values_cubic():
    x = np.repeat(np.arange(6.0), 2)
    df = pd.DataFrame({'x': x, 'y': x**2})
    result = interpolate(df, 'x', onto=[2.5], method='cubic')
    np.testing.assert_allclose(result['y'], [6.25])


def test_interpolate_cubic_exact_for_cubic():
    x = np.linspace(0, 3, 7)
    df = pd.DataFrame({'x': x, 'y': x**3 - 2 * x})
    onto = np.linspace(0, 3, 50)
    result = interpolate(df, 'x', onto=onto, method='cubic')
    np.testing.assert_allclose(result['y'], onto**3 - 2 * onto, atol=1e-12)


@pytest.mark.parametrize('method', ['pchip', 'akima'])
def test_interpolate_local_methods_do_not_overshoot(method):
    # A step: a cubic spline rings either side of it, pchip and akima stay within the data.
    x = np.arange(10.0)
    df = pd.DataFrame({'x': x, 'y': (x >= 5).astype(float)})
    onto = np.linspace(0, 9, 500)
    y = interpolate(df, 'x', onto=onto, method=method)['y']
    assert y.min() >= 0.0 and y.max() <= 1.0
    cubic = interpolate(df, 'x', onto=onto, method='cubic')['y']
    assert cubic.min() < 0.0 or cubic.max() > 1.0


def test_interpolate_pchip_and_akima_pass_through_points():
    x = np.array([0.0, 1.0, 3.0, 4.0, 7.0])
    df = pd.DataFrame({'x': x, 'y': np.sin(x)})
    for method in ['pchip', 'akima']:
        result = interpolate(df, 'x', onto=x, method=method)
        np.testing.assert_allclose(result['y'], np.sin(x), atol=1e-14)


def test_interpolate_integer_spline_order():
    x = np.linspace(0, 2, 6)
    df = pd.DataFrame({'x': x, 'y': x**2})
    result = interpolate(df, 'x', onto=[0.3], method=2)
    np.testing.assert_allclose(result['y'], [0.09])


def test_interpolate_missing_column_raises():
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]})
    with pytest.raises(ValueError, match="not found"):
        interpolate(df, 'temperature', onto=[0.5])


def test_interpolate_non_numeric_column_raises():
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0], 'label': ['a', 'b']})
    with pytest.raises(ValueError, match="'label' is not numeric"):
        interpolate(df, 'x', onto=[0.5])


@pytest.mark.parametrize('method', ['spline', 'Linear', 1.5, -1, True])
def test_interpolate_unknown_method_raises(method):
    df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]})
    with pytest.raises(ValueError, match="Unknown method"):
        interpolate(df, 'x', onto=[0.5], method=method)


@pytest.mark.parametrize('method, points', [('linear', 1), ('cubic', 3), ('quadratic', 2), ('akima', 2)])
def test_interpolate_too_few_points_raises(method, points):
    df = pd.DataFrame({'x': np.arange(float(points)), 'y': np.arange(float(points))})
    with pytest.raises(ValueError, match="'y' has"):
        interpolate(df, 'x', onto=[0.5], method=method)


def test_interpolate_too_few_points_counts_distinct_non_nan_x():
    df = pd.DataFrame({'x': [0.0, 0.0, 1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0, np.nan, 4.0]})
    # x = 0 appears twice and x = 2 has no y: three distinct points, too few for a cubic.
    with pytest.raises(ValueError, match="has 3 distinct"):
        interpolate(df, 'x', onto=[0.5], method='cubic')


def test_interpolate_all_nan_column_raises():
    df = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0], 'empty': np.nan})
    with pytest.raises(ValueError, match="'empty' has 0"):
        interpolate(df, 'x', onto=[0.5])
