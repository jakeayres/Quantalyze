# Preparing your data

`fit_transport` takes one DataFrame: the field, ρ_xx and the Hall resistivity ρ_H in SI
units, one row per field, ideally with their errors. This page goes from raw resistance
sweeps to that DataFrame. Every later page of the guide fits the `data` made here.

| Column | Meaning | Units |
|---|---|---|
| `field` | B along ẑ, usually 0 and up | T |
| `resistivity` | ρ_xx, even in field | Ω·m |
| `hall_resistivity` | ρ_H = R_H B, odd in field, **negative for electrons** in B > 0 | Ω·m |
| `resistivity_error`, `hall_resistivity_error` | One standard deviation (optional, but give them) | Ω·m |

Other column names work too: pass them as `field=`, `resistivity=`, `hall_resistivity=`
and so on.

## From raw sweeps to resistivities

The "measurement" here is synthetic, so that every step can be checked against the truth:
a 1 mm × 0.5 mm × 20 μm bar, swept to +60 T and to −60 T. Each voltage pair picks up some of
the other (misaligned contacts), the longitudinal resistance carries quantum oscillations
at high field, and every raw point has 1% noise.

??? example "How the raw sweeps were made"

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

The preparation is four steps:

```python
--8<-- "beta/boltzmann_fitting/_data.py:prepare"
```

1. **(Anti)symmetrise in field.** ρ_xx is even in B and ρ_H is odd, so
   `qz.symmetrize` and `qz.antisymmetrize` remove the pickup of one into the other. They
   need sweeps in both field directions.
2. **Bin.** Both functions average onto a grid. A bin several quantum-oscillation periods
   wide averages the oscillations away; here a 1 T bin holds about four periods at 60 T.
3. **Convert to resistivities** with `calculate_resistivity` and
   `calculate_hall_resistivity` from `quantalyze.transport`: ρ_xx = R_xx·w·t/l and
   ρ_H = R_xy·t, in Ω·m (1 μΩ·cm = 10⁻⁸ Ω·m).
4. **Estimate the errors** from the raw scatter. Differences between neighbouring raw
   points cancel the smooth signal and leave √2 × the noise. Each binned value averages
   many raw points, which shrinks it by √(number averaged).

## Checking the result

```python
--8<-- "beta/boltzmann_fitting/prepare_data.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/prepare_data.txt"
```

![Raw resistance sweeps, lopsided in field from contact pickup, with the symmetrised and antisymmetrised points](../../examples/beta/boltzmann_fitting/prepare_data.png#only-light)
![Raw resistance sweeps, lopsided in field from contact pickup, with the symmetrised and antisymmetrised points](../../examples/beta/boltzmann_fitting/prepare_data-dark.png#only-dark)

The raw sweeps (grey) are lopsided in field: that is the pickup. The processed points are
not. Against the true resistivities the residuals give χ²/n near 1 in both channels, so
the estimated errors are the right size: what [Honest error bars](error_bars.md) needs.

**Check the Hall sign before fitting.** The sign of the Hall resistivity depends on which
way round the Hall contacts and the field are wired. The model fixes the convention:
ρ_H = R_H B is negative for an electron pocket and positive for a hole pocket in B > 0. If
the model at your starting values disagrees with the data in sign, flip the data's sign
(or recheck the Fermi surface) before fitting.

!!! warning "Watch out"

    - **SI units only.** Resistivities in Ω·m, fields in T. Data in μΩ·cm are 10⁸ times too
      large, and the fit will not tell you; it will just return absurd rates.
    - **The thickness matters.** It scales ρ_xx and ρ_H together, and a 10% error in it does
      not fit away: see [Systematic errors](limits.md#systematic-errors).
    - **Bins narrower than an oscillation period leave the oscillations in.** Check that
      the binned data are smooth at your highest fields, or fit only up to where they are.
    - **Don't over-sample.** A few hundred points at most: neighbouring bins should be
      independent, and more points only cost time.
    - **Correlated noise** (drifts, a background that is not even or odd in field) is not
      in the error bars. Look at the residuals after fitting: see
      [Reading a result](results.md).
