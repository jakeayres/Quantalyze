import pytest
import pandas as pd
import pandas.testing as pdt
from quantalyze.transport import (
    calculate_resistance,
    calculate_resistivity,
    calculate_hall_resistance,
    calculate_hall_resistivity,
)


# Fixture for sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "voltage_col": [10, 20, 30],
        "current_col": [2, 4, 6],
        "gain_col": [1, 2, 3],
        "resistance_col": [5, 10, 15],
    })


def test_calculate_resistance_with_column_names(sample_df):
    result = calculate_resistance(sample_df, "voltage_col", "current_col")
    expected = pd.Series([5.0, 5.0, 5.0], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistance_with_constant_current(sample_df):
    result = calculate_resistance(sample_df, "voltage_col", 2)
    expected = pd.Series([5.0, 10.0, 15.0], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistance_with_gain_column(sample_df):
    result = calculate_resistance(sample_df, "voltage_col", "current_col", gain="gain_col")
    expected = pd.Series([5.0, 2.5, 1.6667], index=sample_df.index)
    assert result.round(4).equals(expected.round(4))


def test_calculate_resistance_with_constant_gain(sample_df):
    result = calculate_resistance(sample_df, "voltage_col", "current_col", gain=2)
    expected = pd.Series([2.5, 2.5, 2.5], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistance_with_constant_current_and_gain(sample_df):
    result = calculate_resistance(sample_df, "voltage_col", 2, gain=2)
    expected = pd.Series([2.5, 5.0, 7.5], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistivity_with_resistance(sample_df):
    result = calculate_resistivity(
        sample_df, resistance="resistance_col", length=100e-6, width=300e-6, thickness=11e-6
    )
    expected = pd.Series([0.000165, 0.00033, 0.000495], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistivity_with_voltage_and_current(sample_df):
    result = calculate_resistivity(
        sample_df, voltage="voltage_col", current="current_col", length=100e-6, width=300e-6, thickness=11e-6
    )
    expected = pd.Series([0.000165, 0.000165, 0.000165], index=sample_df.index)
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistivity_with_constant_current(sample_df):
    result = calculate_resistivity(
        sample_df, voltage="voltage_col", current=2, length=100e-6, width=300e-6, thickness=11e-6
    )
    expected = pd.Series([0.000165, 0.00033, 0.000495], index=sample_df.index)  # Already correct
    pdt.assert_series_equal(result, expected, check_names=False)


def test_calculate_resistivity_with_invalid_inputs(sample_df):
    with pytest.raises(ValueError, match="Only one of resistance or voltage can be provided, not both."):
        calculate_resistivity(
            sample_df, resistance="resistance_col", voltage="voltage_col", current="current_col"
        )

    with pytest.raises(ValueError, match="Either resistance or both voltage and current must be provided."):
        calculate_resistivity(sample_df, length=100e-6, width=300e-6, thickness=11e-6)

    with pytest.raises(ValueError, match="Gain cannot be applied when resistance is directly supplied."):
        calculate_resistivity(
            sample_df, resistance="resistance_col", length=100e-6, width=300e-6, thickness=11e-6, gain="gain_col"
        )

