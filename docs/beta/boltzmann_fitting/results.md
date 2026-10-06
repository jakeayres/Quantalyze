# Reading a result

`fit_transport` returns a `TransportFit`: the best-fit parameters, their uncertainties, and
the diagnostics that say whether to believe them. This page goes through it, then shows
two checks worth making on every fit: the residuals, and the uncertainty of τ(φ) itself.

??? example "Example data used on this page"

    The running sample and its prepared `data` (see [Preparing your data](prepare.md)).

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## What is in a result

```python
--8<-- "beta/boltzmann_fitting/result_tour.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/result_tour.txt"
```

| Attribute | What it is |
|---|---|
| `result["gamma0"]`, `result.parameters` | Best-fit values: one by name, or all of them (held ones included) as a dict |
| `result.names` | The fitted parameters, in model order |
| `result.errors` | One-standard-deviation errors of the fitted parameters |
| `result.covariance`, `result.correlation` | Their covariance and correlation matrices (DataFrames) |
| `result.summary()` | Value, error and whether fitted, for every parameter |
| `result.chi_squared`, `degrees_of_freedom`, `reduced_chi_squared` | Goodness of fit: χ²/ν ≈ 1 when the model and the errors are right |
| `result.absolute` | True if the residuals were weighted by your errors (see below) |
| `result.singular_values` | How many independent combinations of parameters the data constrain: a value near 0 is one they barely see |
| `result.starts` | Where each starting point ended up, and its χ² |
| `result.evaluate(field)` | The fitted model at any fields: a DataFrame like `bzf.transport`'s |
| `result.surface` | The fitted Fermi surface, with `tau` = 1/rate at every point |
| `result.result` | scipy's `OptimizeResult`, for the optimiser's details |

**With your errors** (`result.absolute` is True), χ² is a real test of the fit and the
error bars are the statistical errors. **Without them**, each channel is weighted by its
RMS and the covariance is scaled so that χ²/ν = 1, as `scipy.optimize.curve_fit` does
without `sigma`. That is right only if the noise is uniform within each channel and the
same fraction of each channel's RMS; see [Honest error bars](error_bars.md).

**Several starting points** that end in the same place, as here, are good evidence the
minimum is the only one. If they end in different places that fit equally well,
`fit_transport` warns: see [What the data can't tell you](limits.md).

## Residuals

**Plot (data − model)/error against field.** For the right model they scatter about zero,
mostly within ±1. A wrong model leaves structure that χ² alone hides.

```python
--8<-- "beta/boltzmann_fitting/residuals.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/residuals.txt"
```

![Normalised residuals against field: noise for the right model, a smooth swing for the wrong one](../../examples/beta/boltzmann_fitting/residuals.png#only-light)
![Normalised residuals against field: noise for the right model, a smooth swing for the wrong one](../../examples/beta/boltzmann_fitting/residuals-dark.png#only-dark)

The wrong model misses ρ_xx systematically at low field, where the weak-field
magnetoresistance is set, and drifts at high field. The shape of the residuals says where
the model fails, which a single χ²/ν does not.

## The uncertainty of τ(φ)

**Draw parameter sets from the covariance and compute τ(φ) for each.** The spread of the
curves at each angle is the uncertainty of τ there, correlations included.

```python
--8<-- "beta/boltzmann_fitting/tau_band.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/tau_band.txt"
```

![The 95% band of tau relative to the best fit, narrowest near 25 and 65 degrees, with the truth inside it](../../examples/beta/boltzmann_fitting/tau_band.png#only-light)
![The 95% band of tau relative to the best fit, narrowest near 25 and 65 degrees, with the truth inside it](../../examples/beta/boltzmann_fitting/tau_band-dark.png#only-dark)

The band is narrowest near 25° and 65°, where the 97% anticorrelation of Γ₀ and Γ₁ cancels
(a bit more of one and less of the other leaves the rate there unchanged), and widest along
the diagonals. Quote τ(φ) with its band, not just the parameters.

!!! warning "Watch out"

    - **Check `result.result.success`, or heed the "did not converge" warning.** The
      numbers of a fit that stopped early are not a minimum.
    - **The band is linearised**, like the errors: reliable while the parameters are well
      determined, too narrow when they are nearly degenerate.
    - **The band is statistical only.** A wrong Fermi velocity or thickness moves τ(φ)
      without widening it: see [Systematic errors](limits.md#systematic-errors).
