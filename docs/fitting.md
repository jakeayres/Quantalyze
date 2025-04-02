# Fitting

The `fitting` module provides a simple wrapper for `scipy.curve_fit()` that facilitates the rapid fitting to data held in a `pandas.DataFrame` and subsequent evaluation and plotting of these fits. The expected work flow is to call `quantalyze.fit(...)` and either retrieve the result of the fit from the attributes of the returned `Fit` object or call one of the `Fit` object's utility methods (e.g. `evaluate(...)` or `plot(...)`).


## Examples


### Example data

```python
import pandas as pd
import numpy as np


# Example DataFrame - a straight line with noise
df = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': 3 * np.linspace(0, 10, 100), + 0.5 * np.random.randn(100)
})
```

### Fit and print parameters

```python
from quantalyze.core import fit


# Define a function to fit
f = lambda x, a, b: a + b*x

# Perform the fit
result = fit(f, df, 'x', 'y')

# Print the fitted parameters
print("Fitted parameters:", result.parameters)
```


### Fit and evaluate

```python
from quantalyze.core import fit


# Define a function to fit
f = lambda x, a, b: a + b*x

# Perform the fit and evaluate the function at x=20
result = fit(f, df, 'x', 'y').evaluate(20)

print("result:", result)
```


### Fit and plot

```python
from quantalyze.core import fit
from matplotlib import pyplot as plt

# Create a figure and plot the test data.
fig, ax = plt.subplots()
ax.plot(df['x'], df['y'], 'o')

# Define a function to fit
f = lambda x, a, b: a + b*x

# Perform the fit and plot the result onto pyplot axes
result = fit(f, df, 'x', 'y').plot(ax, np.linspace(0, 20, 10), color='k', linewidth=0.75, linestyle='--')

# Show the figure
fig.show()
```

::: quantalyze.core.fitting.fit


::: quantalyze.core.fitting.Fit

