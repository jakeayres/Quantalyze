import pytest
import pandas as pd
import numpy as np
from quantalyze.core.fft import fft, Window

def test_fft_no_window():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 100),
        'signal': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz sine wave
    })

    # Perform FFT without a window
    result = fft(df, x_column='time', y_column='signal')

    # Check that the result contains the expected frequency
    assert not result.empty
    assert 10 in result['frequency'].round().values
    assert result['amplitude'].iloc[result['frequency'].round().tolist().index(10)] > 0


def test_fft_with_hann_window():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 100),
        'signal': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz sine wave
    })

    # Perform FFT with a Hann window
    result = fft(df, x_column='time', y_column='signal', window=Window.HANN)

    # Check that the result contains the expected frequency
    assert not result.empty
    assert 10 in result['frequency'].round().values
    assert result['amplitude'].iloc[result['frequency'].round().tolist().index(10)] > 0


def test_fft_with_kaiser_window():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 100),
        'signal': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz sine wave
    })

    # Perform FFT with a Kaiser window
    result = fft(df, x_column='time', y_column='signal', window=Window.KAISER, beta=14)

    # Check that the result contains the expected frequency
    assert not result.empty
    assert 10 in result['frequency'].round().values
    assert result['amplitude'].iloc[result['frequency'].round().tolist().index(10)] > 0


def test_fft_invalid_window():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 100),
        'signal': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz sine wave
    })

    # Attempt to use an invalid window
    with pytest.raises(ValueError, match="Unsupported window type"):
        fft(df, x_column='time', y_column='signal', window="INVALID")


def test_fft_kaiser_window_missing_beta():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 100),
        'signal': np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz sine wave
    })

    # Attempt to use a Kaiser window without providing beta
    with pytest.raises(ValueError, match="Beta parameter must be provided for Kaiser window"):
        fft(df, x_column='time', y_column='signal', window=Window.KAISER)