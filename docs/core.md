# Core

The **Core** module of the Quantalyze package provides essential utilities for scientific data analysis. These utilities include functions for fitting, symmetrization, smoothing, differentiation, and Fourier transforms. 

All of the more specific modules, such as `transport`, derive much of their functionality from this foundational module. By leveraging the robust tools provided here, these specialized modules build upon the core capabilities to address domain-specific requirements.

---

## Fitting

The `fitting` module provides tools for fitting mathematical functions to data. It includes:

- **`fit(func, df, x_column, y_column, ...)`**: Fits a given function to data in a DataFrame, with optional filtering based on x and y ranges. Returns a `Fit` object containing the fitted function and parameters.
- **`Fit` class**: Encapsulates the results of a fit, allowing you to:
  - Evaluate the fitted function at specific points using `evaluate(x)`.
  - Plot the fitted function using `plot(ax, x, **kwargs)`.

### fit()

## ::: quantalyze.core.fitting.fit

### Example

```python
import pandas as pd
import numpy as np
from quantalyze.core import fit

# Example DataFrame
df = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': 3 * np.sin(2 * np.linspace(0, 10, 100)) + 0.5 * np.random.randn(100)
})

# Define a sine function to fit
def sine_func(x, a, b, c):
    return a * np.sin(b * x) + c

# Perform the fit
fit_result = fit(sine_func, df, 'x', 'y')

# Evaluate the fitted function at specific points
x_values = np.linspace(0, 10, 50)
y_fitted = fit_result.evaluate(x_values)

# Print the fitted parameters
print("Fitted parameters:", fit_result.parameters)
```

---

## Symmetrization

The `symmetrize` module provides tools for symmetrizing and antisymmetrizing data:

- **`symmetrize(dfs, x_column, y_column, minimum, maximum, step)`**: Combines multiple DataFrames, filters data, and averages y-values with their reversed counterparts to produce a symmetrized dataset.
- **`antisymmetrize(dfs, x_column, y_column, minimum, maximum, step)`**: Similar to `symmetrize`, but calculates the antisymmetric part of the data.

---

## Smoothing

The `smoothing` module provides methods for smoothing noisy data:

- **`bin(dfs, column, minimum, maximum, width)`**: Bins data into equal-width bins and computes the mean for each bin.
- **`window(df, column, window_size=5, window_type='triang')`**: Applies a rolling window smoothing to a specified column in a DataFrame.
- **`savgol_filter(df, column, window_size=5, order=2)`**: Applies a Savitzky-Golay filter to smooth data while preserving features like peaks.

---

## Differentiation

The `differentiation` module provides methods for calculating numerical derivatives:

- **`forward_difference(df, x_column, y_column)`**: Computes the forward difference of y-values with respect to x-values.
- **`backward_difference(df, x_column, y_column)`**: Computes the backward difference of y-values with respect to x-values.
- **`central_difference(df, x_column, y_column)`**: Computes the central difference, averaging forward and backward differences.
- **`derivative(df, x_column, y_column)`**: Calculates the derivative using the central difference method.

---

## Fourier Transforms (FFT)

The `fft` module (currently empty) is intended to provide utilities for performing Fast Fourier Transforms (FFT) on data. This will allow users to analyze frequency components of signals.

---

### Usage Example

Here’s an example of how to use the core utilities:

```python
import pandas as pd
import numpy as np
from quantalyze.core import fit, symmetrize, savgol_filter, derivative

# Example DataFrame
df = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
})

# Fitting a sine function
def sine_func(x, a, b, c):
    return a * np.sin(b * x) + c

fit_result = fit(sine_func, df, 'x', 'y')

# Smoothing the data
df['y_smooth'] = savgol_filter(df, 'y', window_size=7, order=2)

# Calculating the derivative
df['y_derivative'] = derivative(df, 'x', 'y_smooth')

# Symmetrizing the data
symmetrized_df = symmetrize([df], 'x', 'y', minimum=0, maximum=10, step=0.1)
```