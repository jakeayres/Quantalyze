# Boltzmann fitting

Fit Boltzmann-transport models to measured magnetotransport. From ρ_xx(B) and the Hall
resistivity ρ_H(B) at one temperature, `boltzmann_fitting` finds the parameters of a
scattering rate 1/τ(k) on a known Fermi surface, the parameters of the Fermi surface
itself, or both. The model is computed exactly by [`bz.conductivity`](../boltzmann.md), so
it holds at every ω_cτ.

```python
from quantalyze.beta import boltzmann_fitting as bzf
```

## Quick start

**Write the scattering rate as a function, give starting values, and fit.** Arguments named
after columns of the surface (here `phi`) receive them; the others are fitted.

??? example "Example data used on this page"

    A hole pocket about the zone corner of a square-lattice band, and synthetic
    measurements of it with 1/τ = Γ₀ + Γ₁cos²2φ (Γ₀ = 3, Γ₁ = 6, in 10¹² s⁻¹), prepared as
    in [Preparing your data](prepare.md).

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

```python
--8<-- "beta/boltzmann_fitting/quick_start.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/quick_start.txt"
```

![Measured rho_xx and Hall resistivity against field with the fitted curves through them](../../examples/beta/boltzmann_fitting/quick_start.png#only-light)
![Measured rho_xx and Hall resistivity against field with the fitted curves through them](../../examples/beta/boltzmann_fitting/quick_start-dark.png#only-dark)

Both rates come back within their errors of the truth, and χ²/ν ≈ 1 says the errors are
right.

## The functions

| Function | What it does | Returns |
|---|---|---|
| `bzf.fit_transport` | Least-squares fit of a rate and/or a Fermi surface to ρ_xx(B) and ρ_H(B) | a `TransportFit` |
| `bzf.transport` | The model at given parameters: ρ_xx(B) and ρ_H(B) | a `DataFrame` |
| `bzf.build_surface` | The Fermi surface at given parameters, with τ = 1/rate | `DataFrame`s |
| `bzf.polar_angle` | The polar angle φ of each Fermi-surface point about a centre | a `Series` (rad) |

Full signatures are in the [API reference](api.md).

## The guide

1. **[Preparing your data](prepare.md):** from raw ±B resistance sweeps to ρ_xx, ρ_H and
   their errors; checking the Hall sign.
2. **[Surfaces and rate models](models.md):** the argument-name rule, choosing a rate
   model, your own Fermi surface, several pockets, open sheets.
3. **[Fitting the Fermi surface](fermi_surface.md):** surface functions, fitting in
   stages, priors from other measurements.
4. **[Reading a result](results.md):** everything in a `TransportFit`, residuals, and the
   uncertainty of τ(φ).

## Why you can trust it

Each of these pages makes a claim, and the test suite checks it on the same code:

- **[Recovering known answers](recovery.md):** exact data give back the truth to rounding,
  and fits agree with the closed-form two-band model fitted independently.
- **[Honest error bars](error_bars.md):** over 200 noisy fits the pulls are standard
  normal, 68% of 1σ intervals contain the truth, and χ² follows its distribution.
- **[Discretisation](discretisation.md):** how the number of points biases a fit, and how
  `extrapolate=True` removes it.
- **[What the data can't tell you](limits.md):** identifiability, degeneracies no data can
  break, how much field is needed, and which systematic errors χ² does and does not catch.

!!! warning "Watch out"

    - **Use `extrapolate=True`** for precise data. At N = 512 points the solver's own
      error can bias the parameters by more than their error bars; see
      [Discretisation](discretisation.md).
    - **A good χ² is not proof.** Two curves at one temperature determine only a few
      numbers, so check correlations and several starting points, and remember the data
      fix ℓ = vτ, not τ; see [What the data can't tell you](limits.md).
