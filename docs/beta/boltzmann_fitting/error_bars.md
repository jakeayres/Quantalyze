# Honest error bars

An error bar is a promise about how often the truth lies within it. Here that promise is
tested directly: make 200 noisy copies of the same measurement, fit each, and count. If
the errors are right, the **pull** (fit − truth)/error follows a standard normal
distribution: mean 0, standard deviation 1, and |pull| < 1 in 68% of fits. When the
residuals are weighted by measured errors, χ² should follow the χ² distribution with
ν = (number of data) − (number of parameters) degrees of freedom.

??? example "Example data used on this page"

    The running sample of the [guide](prepare.md), at 256 points for speed. The noise here
    is drawn afresh for every trial, from the clean model curves.

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## 200 fits, two weighting modes

```python
--8<-- "beta/boltzmann_fitting/error_bars.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/error_bars.txt"
```

![Histogram of pulls following a standard normal curve, and of chi-squared following the chi-squared distribution with 60 degrees of freedom](../../examples/beta/boltzmann_fitting/error_bars.png#only-light)
![Histogram of pulls following a standard normal curve, and of chi-squared following the chi-squared distribution with 60 degrees of freedom](../../examples/beta/boltzmann_fitting/error_bars-dark.png#only-dark)

**With measured errors** the pulls have mean 0 and spread 1 within the statistical
uncertainty of 200 trials (about ±0.07 and ±0.05), 68% of the 1σ intervals contain the
truth, and the mean χ² matches ν. The ± after "expected" is the uncertainty of a mean of
200 values of χ². So the fitted values are unbiased, `result.errors` and
`result.covariance` are the right size, and a χ²/ν far from 1 on your own data really
does mean the model or the errors are wrong.

**Without errors** each channel is weighted by its RMS, and the covariance is scaled by
χ²/ν, as `scipy.optimize.curve_fit` does without `sigma`. The error bars come out right
here too, because the noise was made in proportion to each channel's RMS: exactly what
that weighting assumes.

!!! warning "Watch out"

    - **Without errors, the error bars rely on an assumption.** Scaling by χ²/ν is right
      only when the noise in each channel is the same at every field and in the same
      proportion to the channel's RMS for both channels. Real noise rarely is, so give
      measured errors whenever you can; [Preparing your data](prepare.md) shows how to
      estimate them from the raw scatter.
    - **These are statistical errors only.** They assume the model is right. A wrong
      Fermi velocity, thickness or rate model shifts the answer without widening the
      error bar: see [What the data can't tell you](limits.md).
    - **The errors are linearised.** They come from the curvature of χ² at the best fit,
      which is what they should be while the parameters are well determined. When two
      parameters are almost degenerate the χ² valley is curved, and the linear errors
      can be far too small; `fit_transport` warns when that happens.
