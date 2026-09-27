# Fitting

Fit any model to two columns of a DataFrame. `qz.fit` wraps [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html): you pass column names instead of arrays, you can limit the fit to a range of x or y, and you get back a [`Fit`](#the-fit-result) object that you can query by parameter name, evaluate and plot.

The workflow in three lines:

```python
def model(x, a, b): ...                             # 1. x first, then the parameters
result = qz.fit(model, df, "x_column", "y_column")  # 2. fit
result["a"]                                         # 3. read a parameter by name
```

??? example "Example data used on this page"

    `rt` is the resistivity of a metal from 2 to 40 K. It follows ρ₀ + A T² (a Fermi liquid, with true values ρ₀ = 2.0 and A = 0.002) below about 20 K, and bends away above that. `transition` is a heat-capacity peak at 10 K.

    ```python
    --8<-- "core/fitting/_data.py"
    ```

## `fit` { #fit }

**Write the model as a Python function whose first argument is x and whose other arguments are the parameters.** Here only the rows with T ≤ 20 K are fitted (`x_max=20`), because that is where the T² law holds.

```python
--8<-- "core/fitting/fit_basic.py:example"
```

```text title="Output"
--8<-- "core/fitting/fit_basic.txt"
```

![Resistivity data with a T-squared fit to the region below 20 kelvin](../examples/core/fitting/fit_basic.png#only-light)
![Resistivity data with a T-squared fit to the region below 20 kelvin](../examples/core/fitting/fit_basic-dark.png#only-dark)

### Restricting the range, and passing options to `curve_fit`

`x_min`, `x_max`, `y_min` and `y_max` keep only the rows inside those limits (inclusive). Any other keyword is passed straight to `scipy.optimize.curve_fit`, for example `bounds`, `sigma` or `maxfev`.

```python
--8<-- "core/fitting/fit_kwargs.py:example"
```

```text title="Output"
--8<-- "core/fitting/fit_kwargs.txt"
```

### Starting guesses with `p0`

Every parameter starts at 1 unless you give a starting guess. For models that aren't close to linear (peaks, exponentials, oscillations), a bad start can end in the wrong minimum. **This fails silently: you get an answer, just a wrong one.** Pass `p0`, one value per parameter in signature order.

```python
--8<-- "core/fitting/fit_p0.py:example"
```

```text title="Output"
--8<-- "core/fitting/fit_p0.txt"
```

![A peak fitted without a starting guess (wrong) and with one (right)](../examples/core/fitting/fit_p0.png#only-light)
![A peak fitted without a starting guess (wrong) and with one (right)](../examples/core/fitting/fit_p0-dark.png#only-dark)

The bad fit also prints SciPy's `OptimizeWarning: Covariance of the parameters could not be estimated`. Treat that warning as a sign the fit has gone wrong.

## The `Fit` result { #the-fit-result }

`qz.fit` returns a `Fit`. Everything you can do with it:

| You want | Write |
|---|---|
| A parameter by name | `result["A"]` |
| A parameter by position | `result[1]` |
| All parameters, as an array | `result.parameters` |
| Their covariance matrix | `result.covariance` |
| The fitted curve at some x | `result.evaluate(x)` |
| The fitted curve drawn on a plot | `result.plot(ax, x, **kwargs)` |
| The model function | `result.function` |

### Uncertainties

The one-standard-deviation error on each parameter is the square root of the diagonal of the covariance matrix.

```python
--8<-- "core/fitting/fit_uncertainty.py:example"
```

```text title="Output"
--8<-- "core/fitting/fit_uncertainty.txt"
```

### Evaluating the fit

`evaluate` accepts a number, an array or a whole column. This example also shows that a `lambda` works as the model: its argument names (`rho0`, `A`) still become the parameter names.

```python
--8<-- "core/fitting/fit_evaluate.py:example"
```

```text title="Output"
--8<-- "core/fitting/fit_evaluate.txt"
```

### Plotting the fit

`plot` draws the curve on a Matplotlib axes at whatever x values you give it (they can extend past the fitted range). Extra keywords are passed to `ax.plot`.

```python
--8<-- "core/fitting/fit_plot.py:example"
```

![Resistivity data with the fitted curve drawn by Fit.plot](../examples/core/fitting/fit_plot.png#only-light)
![Resistivity data with the fitted curve drawn by Fit.plot](../examples/core/fitting/fit_plot-dark.png#only-dark)

!!! warning "Watch out"

    - **x must be the first argument** of your model, and every other argument is fitted. To hold something fixed, don't make it an argument: use a constant, or wrap the model in a `lambda`.
    - **Parameter names come from the function signature.** `result["A"]` only works if the function has an argument called `A`.
    - **`RuntimeError: No data in the specified x or y range to fit`** means your `x_min`/`x_max`/`y_min`/`y_max` removed every row.
    - **`ValueError` about `p0`** means it doesn't have exactly one value per parameter.

## API reference

??? info "`qz.fit`"

    ::: quantalyze.core.fitting.fit
        options:
          heading_level: 3

??? info "`Fit`"

    ::: quantalyze.core.fitting.Fit
        options:
          heading_level: 3
