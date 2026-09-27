# Interpolation

`qz.interpolate` resamples **every column** of a DataFrame onto x values you choose. Use it to:

- put irregularly logged data onto a regular grid, or
- put two measurements onto the *same* x values, so you can subtract, divide or plot them against each other row by row.

??? example "Example data used on this page"

    `cooldown` and `warmup` are two resistance-versus-temperature runs through a hysteretic phase transition, each logged at its own irregular temperatures. `calibration` is a sparse thermometer calibration table.

    ```python
    --8<-- "core/interpolation/_data.py"
    ```

## `interpolate` { #interpolate }

**Give it the DataFrame, the x column and the new x values (`onto`).** You get back a new DataFrame with one row per value in `onto`.

```python
--8<-- "core/interpolation/interpolate_basic.py:example"
```

```text title="Output"
--8<-- "core/interpolation/interpolate_basic.txt"
```

![Irregularly spaced resistance data and the same data interpolated onto a 10 kelvin grid](../examples/core/interpolation/interpolate_basic.png#only-light)
![Irregularly spaced resistance data and the same data interpolated onto a 10 kelvin grid](../examples/core/interpolation/interpolate_basic-dark.png#only-dark)

The input doesn't need to be sorted: `cooldown` runs from hot to cold.

### Comparing two measurements

Two runs logged at different temperatures can't be subtracted row by row until they share the same x values. Interpolate both onto one grid:

```python
--8<-- "core/interpolation/interpolate_compare.py:example"
```

```text title="Output"
--8<-- "core/interpolation/interpolate_compare.txt"
```

![Cooling and warming curves on a common grid, and their difference showing the hysteresis](../examples/core/interpolation/interpolate_compare.png#only-light)
![Cooling and warming curves on a common grid, and their difference showing the hysteresis](../examples/core/interpolation/interpolate_compare-dark.png#only-dark)

### Smoother curves between sparse points: `method`

`method` is passed to [`scipy.interpolate.interp1d(kind=...)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html). The default, `"linear"`, joins points with straight lines. `"cubic"` follows a smoothly curving function much better when the points are sparse. Other options are `"nearest"` and `"quadratic"`.

```python
--8<-- "core/interpolation/interpolate_method.py:example"
```

```text title="Output"
--8<-- "core/interpolation/interpolate_method.txt"
```

![A sparse calibration table interpolated linearly and with cubic splines, compared with the true curve](../examples/core/interpolation/interpolate_method.png#only-light)
![A sparse calibration table interpolated linearly and with cubic splines, compared with the true curve](../examples/core/interpolation/interpolate_method-dark.png#only-dark)

For noisy data, stay with `"linear"`: a cubic passes exactly through every point, noise included, and can overshoot between them.

### Outside the data range: extrapolation

!!! danger "`interpolate` extrapolates without warning"

    New x values outside the measured range are **not** set to NaN. The curve is extended from the nearest points, which can be wildly wrong.

```python
--8<-- "core/interpolation/interpolate_extrapolation.py:example"
```

```text title="Output"
--8<-- "core/interpolation/interpolate_extrapolation.txt"
```

The resistance at 400 K comes out as 13.5 Ω, extended from the last two noisy points, when the true value is about 2.1 Ω. Keep `onto` inside the measured range, as in the last three lines.

!!! warning "Watch out"

    - **Every column is interpolated,** so every column other than the x column must be numeric. Drop text or timestamp columns first, e.g. `df[["temperature", "resistance"]]`.
    - **The result has a new 0, 1, 2, … index,** not the index of your input.

## API reference

??? info "`qz.interpolate`"

    ::: quantalyze.core.interpolation.interpolate
        options:
          heading_level: 3
