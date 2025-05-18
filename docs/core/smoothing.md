# Smoothing

The `smoothing` module offers three practical methods for data smoothing: `bin`, `window`, and `savgol_filter`. The `bin` method groups data into equal-width bins and computes the mean for each bin, ideal for reducing noise in large datasets. The `window` method applies a rolling window smoothing technique, allowing users to specify the window size and type for flexible smoothing. The `savgol_filter` method uses the Savitzky-Golay filter to smooth data while preserving features like peaks, making it suitable for signal processing tasks.

::: quantalyze.core.smoothing.bin

::: quantalyze.core.smoothing.window

::: quantalyze.core.smoothing.savgol_filter
