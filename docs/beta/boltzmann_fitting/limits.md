# What the data can't tell you

A good fit is not the same as a right answer. ρ_xx(B) and ρ_H(B) at one temperature carry a
limited amount of information: roughly the zero-field resistivity, the weak-field Hall and
magnetoresistance coefficients, and how they change up to the highest ω_cτ reached. This
page shows how to see what the data do and do not determine, and which mistakes no fit
can detect.

??? example "Example data used on this page"

    The running sample of the [guide](prepare.md): a hole pocket about the zone corner, with
    1/τ = Γ₀ + Γ₁ cos²2φ, measured to 60 T.

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## A well-determined fit

**Map χ² around the best fit and compare it with the covariance.** For a well-determined
fit the χ² surface is a bowl, and the covariance describes it: χ² rises by 1 one standard
deviation away in any direction.

```python
--8<-- "beta/boltzmann_fitting/landscape.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/landscape.txt"
```

![Contours of chi-squared lying on top of the covariance ellipses, a long thin valley around the truth](../../examples/beta/boltzmann_fitting/landscape.png#only-light)
![Contours of chi-squared lying on top of the covariance ellipses, a long thin valley around the truth](../../examples/beta/boltzmann_fitting/landscape-dark.png#only-dark)

The contours lie on the ellipses, so the error bars can be trusted. The valley is long and
thin: Γ₀ and Γ₁ are 97% anticorrelated, because a bit more of one can be traded for a bit
less of the other. The data constrain their combination far better than either alone.

## Too many parameters

**Add one more harmonic than the data were made with.** The fit is just as good, but the
new parameter is almost a copy of an old one.

```python
--8<-- "beta/boltzmann_fitting/too_many_parameters.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/too_many_parameters.txt"
```

χ²/ν is unchanged, Γ₁'s error is more than ten times larger, and the smallest singular value (0.007)
says one combination of the three parameters is barely constrained. `fit_transport` warns.
The extra parameter is consistent with zero, so the data give no reason to keep it. When
a rate model needs more freedom than this, the data must supply it: a higher field, more
temperatures, or other measurements, rather than more parameters.

## Things no data can fix

**Some changes to the model leave the data exactly the same.** No fit can detect them.

```python
--8<-- "beta/boltzmann_fitting/degeneracies.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/degeneracies.txt"
```

1. **Only the mean free path ℓ = vτ is measured.** Scaling every Fermi velocity by μ and
   every τ by 1/μ leaves σ unchanged, so the fitted rates carry the error of your Fermi
   velocities one for one. Calibrate the velocities independently: ARPES, or the cyclotron
   mass from quantum oscillations.
2. **Mirror images are indistinguishable.** On a pocket with a mirror plane, τ(φ) and
   τ(−φ) give identical ρ_xx and Hall resistivity, so the sign of any term odd under the
   mirror (sin 4φ here) cannot be found. The two starts reach ±Γ₂ with the same χ², and
   `fit_transport` warns. Use only terms that respect the crystal's mirrors.

## How far in field you need to go

**Fit the same data cut off at lower and lower fields.**

```python
--8<-- "beta/boltzmann_fitting/field_range.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/field_range.txt"
```

![Relative errors of both rates falling as the highest field in the fit grows](../../examples/beta/boltzmann_fitting/field_range.png#only-light)
![Relative errors of both rates falling as the highest field in the fit grows](../../examples/beta/boltzmann_fitting/field_range-dark.png#only-dark)

The anisotropy Γ₁ needs field: up to 5 T (ω_c⟨τ⟩ ≈ 0.06) it is known to 2%, up to 60 T
(ω_c⟨τ⟩ ≈ 0.8) to 0.09%.
At low field only the weak-field coefficients are measured, which leaves the shape of τ(φ)
poorly constrained. Each extra parameter needs data from further along the curve.

## Systematic errors

**Fit with something wrong, and look at χ².**

```python
--8<-- "beta/boltzmann_fitting/systematics.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/systematics.txt"
```

- **A wrong thickness** scales ρ_xx and ρ_H together. On a fixed Fermi surface the
  high-field Hall slope 1/(nq) cannot follow, and χ²/ν explodes. If the Fermi surface is
  fitted too, μ moves to absorb part of it (n changes) and χ²/ν is "only" 12: still
  flagged, but the parameters are well off.
- **A wrong Fermi velocity** is invisible (χ²/ν = 1.1) and moves every rate by the same
  10%: the ℓ = vτ degeneracy above.
- **A wrong rate model** usually shows: an isotropic rate cannot fit this sample at all.

!!! warning "Watch out"

    - **χ²/ν near 1 means consistent, not correct.** Errors that rescale ℓ, or a model with
      more freedom than the data constrain, fit just as well. Check `correlation`,
      `singular_values` and several starting points, and say which inputs (Fermi velocity,
      thickness) the answer depends on.
    - **χ²/ν ≫ 1 needs explaining before the parameters mean anything.** Check the
      thickness and the Hall sign, then the Fermi surface, then the rate model.
