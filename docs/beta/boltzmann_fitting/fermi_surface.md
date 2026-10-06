# Fitting the Fermi surface

**Pass a function that builds the surface instead of the surface itself.** Its arguments
are fitted along with the rate's: the surface is rebuilt at every step of the fit.

```python
def band(mu):                              # mu is fitted
    surface = bz.generators.tight_binding(512, tau=1.0, chemical_potential=mu, ...)
    surface["phi"] = bzf.polar_angle(surface, center=corner)   # columns the rate needs
    return surface
```

The same rules apply as for rates: arguments with defaults are held unless `p0` names them,
and an argument shared by the surface and the rate is one parameter.

??? example "Example data used on this page"

    The running sample and its `band(mu)` function (see [Preparing your data](prepare.md)).

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## The chemical potential

```python
--8<-- "beta/boltzmann_fitting/fit_surface.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/fit_surface.txt"
```

The chemical potential comes back to a fraction of a meV. Most of that information is the
high-field Hall resistivity, which tends to B/(nq): it fixes the area of the pocket, and
so μ. `bz.carrier_density(result.surface, ...)` gives the n it implies.

## Hoppings, fitted in stages, and priors

**Shape parameters are harder.** The data see the area of the pocket far better than its
shape, so a hopping trades off against μ (here they are 99% correlated). Two habits make
such fits work:

1. **Fit in stages.** First fit the rates on your best-guess surface, then free the
   surface, starting from those rates. A first stage with a poor χ²/ν tells you the
   surface needs fitting at all.
2. **Use what you know.** `priors={"t_prime": (mean, std)}` adds independent knowledge (band
   structure, ARPES, a quantum-oscillation frequency through μ) as one more residual,
   (value − mean)/std.

```python
--8<-- "beta/boltzmann_fitting/priors.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/priors.txt"
```

The surface held at a hopping 10% off gives χ²/ν = 145; fitted, it comes back to the truth
(μ = −120 meV, t′ = −50 meV). The prior narrows σ(t′) exactly as independent information
should: 1/σ² = 1/σ_data² + 1/σ_prior². It narrows σ(μ) too, through their correlation.

!!! warning "Watch out"

    - **Start close.** Starting the rates far off makes the optimiser move the surface to
      compensate, and it can wander until the surface can no longer be built (a pocket
      that stops being closed or star-shaped). Such steps are turned back, but a fit that
      ends with a "did not converge" warning and a huge χ²/ν has stalled there. Fit in
      stages.
    - **Watch the correlations.** μ and a shape parameter are often more than 99%
      correlated; `fit_transport` warns, and the [limits page](limits.md) explains what
      that means for the error bars.
    - **A wrong thickness moves μ.** The fit adjusts n to absorb a scale error in ρ_H; see
      [Systematic errors](limits.md#systematic-errors).
    - **Rebuilding the surface costs time** (a few milliseconds per step for 512 points of a
      tight-binding band). A fit takes seconds rather than a fraction of one; a fixed surface
      is faster.
