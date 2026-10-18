# Fitting several datasets

`bzf.fit_joint` fits several datasets at once, as one least-squares problem: one sample at
several temperatures, several samples, or one sample under different conditions. Each
dataset has its own data, Fermi surface, rate model and known constants. Parameters with
the same name are shared by every dataset; those listed in `separate` are fitted to each
dataset, or to each group of datasets, on its own.

| Function | What it does | Returns |
|---|---|---|
| `bzf.Dataset` | One dataset: its data, surface, rate, constants, layer spacing and options | a `Dataset` |
| `bzf.fit_joint` | Least-squares fit of every dataset at once | a `JointFit` |
| `bzf.joint_transport` | The joint model at given parameters: ρ_xx(B) and ρ_H(B) of each dataset | `DataFrame`s by dataset name |

??? example "Example data used on this page"

    The running sample's hole pocket (see [Preparing your data](prepare.md)) at five
    temperatures, with 1/τ = Γ_imp + αT² + Γ₁(T) cos²2φ. Each curve has noise of 0.2% of its
    RMS, and no error bars.

    ```python
    --8<-- "beta/boltzmann_fitting/_joint_data.py"
    ```

## Five temperatures at once

**Make one `Dataset` per measurement, with its known values as `constants`.** An argument of
the rate named after a constant (here `T`) receives that dataset's value. Every other
argument that is not a column of the surface is a parameter, shared by all the datasets
unless `separate` names it.

```python
--8<-- "beta/boltzmann_fitting/joint_temperatures.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/joint_temperatures.txt"
```

![rho_xx and the Hall resistivity against field at five temperatures, with the joint fit through every curve](../../examples/beta/boltzmann_fitting/joint_temperatures.png#only-light)
![rho_xx and the Hall resistivity against field at five temperatures, with the joint fit through every curve](../../examples/beta/boltzmann_fitting/joint_temperatures-dark.png#only-dark)

Γ_imp and α are fitted to all five temperatures together, and Γ₁ to each temperature on
its own; every value comes back within its error of the truth. `fit_quality` has one row
per curve (one channel of one dataset): the scale its residuals were divided by (`sigma`),
its χ² and share of the total, and its misfit relative to its own size (`relative_rms`), so
a curve the model fits worse than the others stands out.

## Linking parameters

**Parameters are matched by name across datasets,** extending the
[argument-name rule](models.md#the-argument-name-rule) of a single fit:

| Link | How |
|---|---|
| A term shared by every dataset, another fitted to each | `rate(phi, gamma_imp, gamma_in)` with `separate=["gamma_in"]` |
| A law across datasets, e.g. Γ(T) = Γ₀ + αT | `rate(phi, T, gamma0, alpha)`, with `constants={"T": ...}` in each dataset |
| A different surface for each dataset | Give each `Dataset` its own `surface`; the rate's parameters are still shared by name |
| One value per group of datasets, e.g. per sample | `constants={"sample": "A", ...}` in each dataset, and `separate={"gamma_imp": "sample"}` |

The links combine freely, and apply to a surface's parameters as much as to a rate's: with
a surface function `band(mu)`, `separate=["mu"]` fits one chemical potential per dataset,
and leaving `mu` out of `separate` shares one between them.

**The copies of a separate parameter are named after their dataset or group:**
`gamma1[30 K]`, `gamma_imp[A]`. In `p0`, `fixed`, `bounds` and `priors`, the plain name sets
every copy, and a copy's own name sets that copy only, taking precedence:

```python
bzf.fit_joint(
    datasets, separate=["gamma1"],
    p0={"gamma_imp": 1e12, "alpha": 1e8, "gamma1": 1e12},  # every gamma1 starts at 1e12...
    fixed={"gamma1[10 K]": 1.5e12},                         # ...except this one, which is held
    bounds={"gamma1": (0, None)},                           # every copy is kept positive
)
```

A dataset's `name` defaults to its position in the list ("0", "1", ...). Everything else
that can differ between datasets is set on its `Dataset`: `layer_spacing`, the data's
column names and errors, and options for `bz.conductivity` such as `period`, `kz` and
`extrapolate`. A dataset may have only ρ_xx (`hall_resistivity=None`) or only the Hall
resistivity.

## Weighting

**With error bars for every curve, each residual is (model − data)/error,** and χ² is a test
of the fit, as with `fit_transport`.

**Without them, the weighting decides how much each curve counts,** so `fit_joint` makes the
choice explicit. The residuals of each curve are divided by one number:

- **`weighting="curve"`** (the default): the curve's RMS × √(its points / the mean number of
  points per curve). Every curve counts equally, however large its values and however many
  points it has: a curve's χ² is the mean square of its relative misfit, times the mean
  number of points.
- **`weighting="point"`:** the curve's RMS, as `fit_transport` does. Every point counts
  equally relative to its curve, so a curve with more points counts for more.

A dataset's `weight` multiplies the contributions of its curves to χ². `fit_quality` shows
the scale each curve was given (`sigma`) and its share of χ². The covariance is then scaled
so that χ²/ν = 1, which assumes the misfit of every curve is noise of the same relative
size; the `relative_rms` of the curves shows whether it is.

With one dataset, `fit_joint(..., weighting="point")` does exactly what `fit_transport`
does; with the default, the two agree whenever both channels have the same number of
points.

## Reading a joint result

A `JointFit` has everything a `TransportFit` has (see [Reading a result](results.md)), and
results by dataset:

| Attribute | What it is |
|---|---|
| `result["gamma1[30 K]"]`, `result.parameters` | Best-fit values: one by name, or all of them as a dict |
| `result.summary()` | Value, error and whether fitted for every parameter, with the model argument (`parameter`) and the dataset or group (`group`) of each copy |
| `result.fit_quality` | One row per curve: points, `sigma`, `chi_squared`, `share`, `rms_residual` and `relative_rms` |
| `result.weighting` | "errors", "curve" or "point" |
| `result.evaluate()` | Each dataset's fitted model at its measured fields, as a dict of DataFrames by dataset name: exactly what the fit compared with the data |
| `result.evaluate(field)` | The same at any fields |
| `result.evaluate(datasets=[...])` | Other datasets with the fitted parameters, e.g. to predict a measurement held back from the fit |
| `result.surfaces` | Each dataset's fitted Fermi surface, with `tau` = 1/rate |
| `result.datasets` | The names of the datasets, in order |

`bzf.joint_transport(datasets, parameters, separate=...)` computes the same model at any
parameters: to make synthetic data, or to predict a dataset that needs a value the fit did
not have, such as a new copy of a separate parameter.

## What a joint fit can and cannot determine

**Fit each temperature alone, then all of them together, and compare.**

```python
--8<-- "beta/boltzmann_fitting/joint_vs_single.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/joint_vs_single.txt"
```

![Left: the isotropic rate of each temperature fitted alone, on the joint fit's law. Right: the error of Gamma_1 at each temperature, smaller in the joint fit](../../examples/beta/boltzmann_fitting/joint_vs_single.png#only-light)
![Left: the isotropic rate of each temperature fitted alone, on the joint fit's law. Right: the error of Gamma_1 at each temperature, smaller in the joint fit](../../examples/beta/boltzmann_fitting/joint_vs_single-dark.png#only-dark)

- **A link can determine what no single dataset can.** At one temperature Γ_imp and αT² are
  both constants, so a fit of one dataset finds only their sum, and says so. Across five
  temperatures, the law Γ_imp + αT² separates them.
- **Shared parameters use every dataset,** so the parameters correlated with them are better
  determined too: here every Γ₁ has a smaller error than when its temperature is fitted
  alone.
- **A copy of a separate parameter is determined by its own dataset,** and by the others
  only through the shared parameters it is correlated with.
- **A degeneracy that is in every dataset stays.** Two parameters that enter every dataset
  only as a sum, the scale of the Fermi velocity against the rates, or a mirror image of
  τ(φ) that leaves every curve unchanged (see [What the data can't tell you](limits.md)) are
  no better determined jointly. `fit_joint` warns about them as `fit_transport` does.
- **The fit determines the parameters of the joint model it is given.** The per-curve
  `share` and `relative_rms` show how well that model describes each curve.

!!! warning "Watch out"

    - **Every copy needs a starting value of the right size.** The plain name in `p0` gives
      every copy the same start; give a copy its own where they differ by a lot.
    - **Separate parameters add to the number of free parameters with every dataset.** A
      difference step in a parameter recomputes only the datasets that use it, so a
      separate parameter costs one dataset's model per step, and a shared one costs all of
      them.
    - **Give a constant to every dataset whose model uses it.** A name that is a constant in
      some datasets and would be a parameter in others is an error.
    - **Without error bars, the weighting is a choice.** Look at `fit_quality` before
      reading the errors.
