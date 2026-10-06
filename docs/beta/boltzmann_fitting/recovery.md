# Recovering known answers

A fit is only as good as the model it fits and the machinery around it: the residuals,
their weights, the optimiser and the covariance. This page checks them on problems whose
answer is known. The model itself, `bz.conductivity`, is checked separately against
textbook results and an independent brute-force calculation (see
[Boltzmann transport](../boltzmann.md)).

Every number on the correctness pages is also checked by the test suite
(`tests/test_boltzmann_fitting_docs.py`), which runs these same examples.

??? example "Example data used on this page"

    The running sample of the [guide](prepare.md): a hole pocket about the zone corner, with
    1/τ = Γ₀ + Γ₁ cos²2φ.

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## Exact data give back the truth

**Make data from the model itself, with no noise, and fit them from a poor start.** The
fit must return the parameters the data were made with, whether it fits only the
scattering rate or the Fermi surface as well.

```python
--8<-- "beta/boltzmann_fitting/recover_exact.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/recover_exact.txt"
```

Both fits return the truth to rounding (the test suite measures 2×10⁻¹⁶). This checks the
plumbing: the parameters reach the model, the model reaches the residuals, and the
optimiser finds the minimum, with the Fermi surface rebuilt at every step.

## An independent check: two Drude pockets

**Fit the same data two ways that share no code.** Two circular pockets with a constant
relaxation time each are the two-band Drude model, which has a closed form. The data here
are made from that closed form, with noise, so the solver plays no part in making them.
`bzf` fits them with the Boltzmann solver; plain `scipy.optimize.least_squares` fits them
with the closed form.

```python
--8<-- "beta/boltzmann_fitting/two_band_check.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/two_band_check.txt"
```

![Two-band data with the bzf and closed-form fits on top of each other](../../examples/beta/boltzmann_fitting/two_band_check.png#only-light)
![Two-band data with the bzf and closed-form fits on top of each other](../../examples/beta/boltzmann_fitting/two_band_check-dark.png#only-dark)

With `extrapolate=True`, the two fits agree to a few millionths of an error bar, and so do
the error bars themselves. That confirms the solver, the weighting by the measured errors
and the covariance all at once: the closed-form fit has none of them in common.

The `N=512` column is the same fit without extrapolation. Its parameters move by up to
0.1σ, because at 0.2% noise the solver's own O(N⁻²) error is no longer negligible.
[Discretisation](discretisation.md) shows how to tell when that matters.

!!! warning "Watch out"

    - **Recovering exact data says nothing about noise.** These checks show the fit finds
      the minimum. Whether the minimum is well defined, and whether the error bars are
      right, are separate questions: see [Honest error bars](error_bars.md) and
      [What the data can't tell you](limits.md).
    - **Converge fully on exact data.** With no noise the default tolerances stop early,
      at about 1e-8 in the cost; pass tighter ones through `least_squares_options`, as
      above. With real data the defaults are far below the statistical error.
