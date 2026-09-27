import numpy as np
import pandas as pd
import pytest
from quantalyze.core.symmetrization import symmetrize, antisymmetrize


@pytest.fixture
def sweeps():
    field = np.linspace(-9, 9, 18001)
    even = 1 + 0.01 * field**2
    odd = 0.05 * field
    up = pd.DataFrame({'field': field, 'signal': even + odd})
    down = pd.DataFrame({'field': field[::-1], 'signal': (even + odd)[::-1]})
    return up, down, field


@pytest.mark.parametrize('maximum, step', [(9, 0.1), (9, 0.05), (8, 0.25)])
def test_symmetrize_recovers_even_part(sweeps, maximum, step):
    up, down, _ = sweeps
    result = symmetrize([up, down], 'field', 'signal', -maximum, maximum, step)
    # The outermost bins are only half filled (the data stop at +-9), so skip them.
    valid = result.iloc[1:-1]
    expected = 1 + 0.01 * valid['field'] ** 2
    np.testing.assert_allclose(valid['signal'], expected, atol=1e-3)


@pytest.mark.parametrize('maximum, step', [(9, 0.1), (9, 0.05), (8, 0.25)])
def test_antisymmetrize_recovers_odd_part(sweeps, maximum, step):
    up, down, _ = sweeps
    result = antisymmetrize([up, down], 'field', 'signal', -maximum, maximum, step)
    valid = result.iloc[1:-1]
    np.testing.assert_allclose(valid['signal'], 0.05 * valid['field'], atol=1e-3)
    assert result.loc[result['field'] == 0, 'signal'].iloc[0] == 0


def test_grid_is_symmetric_when_maximum_is_not_a_whole_number_of_steps(sweeps):
    up, down, _ = sweeps
    result = symmetrize([up, down], 'field', 'signal', -7.3, 7.3, 0.25)
    np.testing.assert_allclose(result['field'], -result['field'].values[::-1], atol=1e-12)
    assert result['field'].iloc[-1] == pytest.approx(7.25)


def test_accepts_single_dataframe(sweeps):
    up, _, _ = sweeps
    result = symmetrize(up, 'field', 'signal', -9, 9, 0.1)
    assert list(result.columns) == ['field', 'signal']


def test_minimum_only_trims_input(sweeps):
    up, down, _ = sweeps
    result = symmetrize([up, down], 'field', 'signal', 0, 9, 0.1)
    assert result['field'].iloc[0] == pytest.approx(-9)
