# Discretisation

The Fermi surface is sampled at N points, and the solver's error falls as N⁻²
([Boltzmann transport](../boltzmann.md)). A fit inherits that error as a **bias**: a shift of
the fitted parameters that no amount of data averages away. It matters when it is
comparable to the statistical error, and precise data have small statistical errors.

??? example "Example data used on this page"

    The running sample of the [guide](prepare.md). The data here are made at the continuum
    limit (4096 points, extrapolated in N), so that any bias comes from the fit alone.

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## Bias against the number of points

**Fit continuum-limit data with N = 64 … 1024 points, with and without `extrapolate`,** and
compare each parameter's bias with its statistical error in the running sample's fit.

```python
--8<-- "beta/boltzmann_fitting/discretisation.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/discretisation.txt"
```

![Bias against N on log axes: plain fits fall as N to the minus 2, extrapolated fits as N to the minus 4](../../examples/beta/boltzmann_fitting/discretisation.png#only-light)
![Bias against N on log axes: plain fits fall as N to the minus 2, extrapolated fits as N to the minus 4](../../examples/beta/boltzmann_fitting/discretisation-dark.png#only-dark)

The bias falls as N⁻², exactly as the solver's error does. But for this sample, whose
ρ_xx is known to 0.06%, it is **more than one error bar at N = 512** and still a third of
one at N = 1024. With `extrapolate=True` (Richardson extrapolation in N, 1.5× the cost) it
falls as N⁻⁴ and is below 0.03σ already at N = 256.

That is why every fit in this guide passes `extrapolate=True`.

## Choosing N

- **Pass `extrapolate=True`** unless the contour is noisy or irregularly sampled
  (measured points, say), where the error does not fall smoothly with N. It needs an even
  number of points, at least 32.
- **Check by doubling.** Fit at N and at 2N: if the parameters move by much less than
  their error bars, N is enough. Without extrapolation the bias at N is about 4/3 of
  that change.
- **Resolve sharp features.** A hot spot narrower than a few points is under-resolved
  whatever the extrapolation, and `bz.conductivity` warns about it.

!!! warning "Watch out"

    - **Synthetic tests can hide the bias.** Data made with the same N as the fit carry the
      same discretisation error, so the fit recovers its input perfectly. Real data come
      from the continuum; test your set-up on data made at a much larger N, as here.
    - **The bias grows with precision, not with noise.** Halving the error bars doubles the
      bias in units of σ. Recheck N when you add data or average more sweeps.
