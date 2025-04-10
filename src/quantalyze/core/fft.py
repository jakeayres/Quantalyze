import numpy as np
import pandas as pd
from enum import Enum


class Window(Enum):
    """
    An enumeration representing different types of window functions for signal processing.
    
    Attributes:
        HANN (str): Hann window.
        HAMMING (str): Hamming window.
        BLACKMAN (str): Blackman window.
        BARTLETT (str): Bartlett window.
        KAISER (str): Kaiser window, requires an additional beta parameter.
        
    Methods:
        get_window(length: int, beta: Optional[float] = None) -> np.ndarray:
            Generates the specified window function as a NumPy array.
            Parameters:
                length (int): The length of the window.
                beta (Optional[float]): The beta parameter for the Kaiser window. 
                                        Required only for the Kaiser window.
            Returns:
                np.ndarray: The generated window function.
            Raises:
                ValueError: If the Kaiser window is selected but the beta parameter is not provided.
                ValueError: If an unsupported window type is specified.
    """
    HANN = 'hann'
    HAMMING = 'hamming'
    BLACKMAN = 'blackman'
    BARTLETT = 'bartlett'
    KAISER = 'kaiser'

    def get_window(self, length, beta=None):
        if self == Window.HANN:
            return np.hanning(length)
        elif self == Window.HAMMING:
            return np.hamming(length)
        elif self == Window.BLACKMAN:
            return np.blackman(length)
        elif self == Window.BARTLETT:
            return np.bartlett(length)
        elif self == Window.KAISER:
            if beta is None:
                raise ValueError("Beta parameter must be provided for Kaiser window.")
            return np.kaiser(length, beta)
        else:
            raise ValueError(f"Unsupported window type: {self}")


def fft(df: pd.DataFrame, x_column: str, y_column: str, n: int = None, window: Window = None, beta: float = None) -> pd.DataFrame:
    """
    Computes the Fast Fourier Transform (FFT) of a signal in a DataFrame. Interpolates onto equally spaced x values.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the signal.
        x_column (str): The name of the column representing the x-axis (time or frequency).
        y_column (str): The name of the column representing the y-axis (signal values).
        n (int, optional): Number of points to compute the FFT. If None, uses the length of the signal.
        window (Window, optional): Window function to apply before FFT. If None, no window is applied.
        beta (float, optional): Beta parameter for the Kaiser window. Required if the Kaiser window is used.
        
    Returns:
        pd.DataFrame: A DataFrame containing the frequency and FFT amplitude.
    """
    
    # Sort the DataFrame by the x_column
    df = df.sort_values(by=x_column)
    
    # Interpolate onto equally spaced x values
    x_values = np.linspace(df[x_column].min(), df[x_column].max(), len(df))
    y_values = np.interp(x_values, df[x_column], df[y_column])
    
    # Apply the window function if provided
    if window is not None:
        if not isinstance(window, Window):
            raise ValueError("Unsupported window type")
        win = window.get_window(len(y_values), beta=beta)
        y_values = y_values * win
        
    # Perform the FFT
    fft_result = np.fft.fft(y_values, n=n)
    fft_freq = np.fft.fftfreq(len(fft_result), d=(x_values[1] - x_values[0]))
    
    # Take the positive frequencies and corresponding FFT amplitudes
    positive_freqs = fft_freq[:len(fft_freq) // 2]
    positive_amplitudes = np.abs(fft_result[:len(fft_result) // 2])
    
    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        'frequency': positive_freqs,
        'amplitude': positive_amplitudes
    })
    return result_df