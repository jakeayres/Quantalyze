import numpy as np
import pandas as pd
from enum import Enum


class Window(Enum):
    """
    Window functions to taper the signal before `fft`, reducing spectral leakage.

    Without a window, the abrupt start and end of the data spread every peak into
    broad "skirts" that can hide weaker peaks. Pass one of these as `fft(window=...)`.

    Attributes:
        HANN: A good default.
        HAMMING: Similar to Hann, slightly narrower peaks, higher far-off leakage.
        BLACKMAN: Lower leakage than Hann, broader peaks.
        BARTLETT: Triangular window.
        KAISER: Adjustable: needs `beta` (0 is no window; larger values give lower
            leakage and broader peaks, ~5-10 is typical).

    Examples:
        >>> import quantalyze as qz
        >>> spectrum = qz.fft(df, 'inverse_field', 'signal', window=qz.Window.HANN)
        >>> spectrum = qz.fft(df, 'inverse_field', 'signal', window=qz.Window.KAISER, beta=8)
    """
    HANN = 'hann'
    HAMMING = 'hamming'
    BLACKMAN = 'blackman'
    BARTLETT = 'bartlett'
    KAISER = 'kaiser'

    def get_window(self, length, beta=None):
        """
        The window's weights as an array (you don't normally need to call this).

        Args:
            length (int): Number of points.
            beta (float, optional): Shape parameter; required for `Window.KAISER`.

        Returns:
            numpy.ndarray: The window weights, `length` values between 0 and 1.

        Raises:
            ValueError: If `Window.KAISER` is used without `beta`.
        """
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
    The frequency spectrum (FFT amplitude) of `y_column` as a function of `x_column`.

    The data are sorted by x and linearly resampled onto `len(df)` evenly spaced x
    values first, so irregular x is fine. Frequencies come out in units of 1/x: for
    quantum oscillations use x = 1/B (in 1/T) and the frequencies are in tesla.

    Args:
        df (pd.DataFrame): The data. Subtract any smooth background from `y_column` first,
            or it shows up as a large peak near zero frequency.
        x_column (str): The column to transform over, e.g. `'inverse_field'` or `'time'`.
        y_column (str): The signal, e.g. `'signal'`.
        n (int, optional): Length of the FFT. Values above `len(df)` zero-pad the signal,
            giving a finer frequency grid (smoother-looking peaks) but not better
            resolution. Defaults to `len(df)`.
        window (Window, optional): A window to apply before transforming, e.g.
            `qz.Window.HANN`. Defaults to no window.
        beta (float, optional): Shape parameter; required with `window=qz.Window.KAISER`.

    Returns:
        pd.DataFrame: Columns `'frequency'` (0 and positive frequencies only, evenly
            spaced by `1 / (x range)`) and `'amplitude'` (the magnitude of the FFT; not
            normalised, so compare peaks within one spectrum rather than between spectra
            with different lengths or windows).

    Raises:
        ValueError: If `window` is not a `Window`, or `Window.KAISER` is used without `beta`.

    Examples:
        >>> import quantalyze as qz
        >>> df['inverse_field'] = 1 / df['field']
        >>> spectrum = qz.fft(df, 'inverse_field', 'signal', window=qz.Window.HANN)
        >>> spectrum.loc[spectrum['amplitude'].idxmax(), 'frequency']
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
