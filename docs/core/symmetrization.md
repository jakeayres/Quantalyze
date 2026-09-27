# Symmetrization

Split a signal measured in both field directions into its even and odd parts:

| Function | Keeps | Formula | Typical use |
|---|---|---|---|
| [`qz.symmetrize`](#symmetrize) | the even part | ( y(x) + y(−x) ) / 2 | longitudinal resistance, magnetoresistance |
| [`qz.antisymmetrize`](#antisymmetrize) | the odd part | ( y(x) − y(−x) ) / 2 | Hall resistance |

Both combine your sweeps, [`bin`](smoothing.md#bin) them onto an evenly spaced grid that is symmetric about zero, then pair each grid point with its mirror image.

??? example "Example data used on this page"

    `up` (−9 T → +9 T) and `down` (+9 T → −9 T) are two field sweeps of the same sample, each logged at irregular field values. They measure `rxx` (even in field) and `rxy` (odd in field), but misaligned contacts mix a little of each into the other: `rxx` picks up a slope and `rxy` picks up an offset.

    ```python
    --8<-- "core/symmetrization/_data.py"
    ```

## `symmetrize` { #symmetrize }

**Keep only the part of the signal that is the same at +x and −x.** Pass the sweeps as a list, the two column names, and the grid: `minimum`, `maximum` and `step`.

```python
--8<-- "core/symmetrization/symmetrize_basic.py:example"
```

```text title="Output"
--8<-- "core/symmetrization/symmetrize_basic.txt"
```

![Measured rxx is lopsided; the symmetrized curve is symmetric about zero field](../examples/core/symmetrization/symmetrize_basic.png#only-light)
![Measured rxx is lopsided; the symmetrized curve is symmetric about zero field](../examples/core/symmetrization/symmetrize_basic-dark.png#only-dark)

The result has only two columns, the grid (`field`) and the symmetrized values (`rxx`), one row per grid point from −9 to +9 T. A single DataFrame works as well as a list, provided it covers both positive and negative x.

## `antisymmetrize` { #antisymmetrize }

**Keep only the part of the signal that flips sign between +x and −x.** It takes the same arguments as `symmetrize`.

```python
--8<-- "core/symmetrization/antisymmetrize_basic.py:example"
```

```text title="Output"
--8<-- "core/symmetrization/antisymmetrize_basic.txt"
```

![Measured rxy is offset from zero; the antisymmetrized curve passes through the origin](../examples/core/symmetrization/antisymmetrize_basic.png#only-light)
![Measured rxy is offset from zero; the antisymmetrized curve passes through the origin](../examples/core/symmetrization/antisymmetrize_basic-dark.png#only-dark)

The offset from the `rxx` pickup is gone, and the result is exactly zero at zero field.

### Both at once

Use the same grid for both and merge the results on the x column:

```python
--8<-- "core/symmetrization/symmetrize_both.py:example"
```

```text title="Output"
--8<-- "core/symmetrization/symmetrize_both.txt"
```

## How `minimum`, `maximum` and `step` work { #grid }

This is the most common source of confusion:

- **The output grid always runs from `−maximum` to `+maximum`** in steps of `step`, so it is symmetric about zero and includes 0. If `maximum` isn't a whole number of steps, the grid stops at the last whole step below it.
- **`minimum` does not set the start of the grid.** It only discards input rows with x below `minimum`. In most cases, pass `minimum=-maximum`.

Setting `minimum=0` is a mistake: it throws away all the negative-field data, so nothing has a mirror partner.

```python
--8<-- "core/symmetrization/symmetrize_grid.py:example"
```

```text title="Output"
--8<-- "core/symmetrization/symmetrize_grid.txt"
```

!!! warning "Watch out"

    - **You need data on both sides of zero,** over the whole range out to `maximum`. Grid points with no data at +x or −x come out NaN.
    - **Other columns are dropped.** Only the x column and the one y column are returned. Call the function again for each column you need, as in [Both at once](#both-at-once).
    - **The outermost points can be slightly off** if your data stop exactly at ±`maximum`, because those bins are only half full (see [`bin`](smoothing.md#bin)). Choose `maximum` a little inside your data range if the end points matter.

## API reference

??? info "`qz.symmetrize`"

    ::: quantalyze.core.symmetrization.symmetrize
        options:
          heading_level: 3

??? info "`qz.antisymmetrize`"

    ::: quantalyze.core.symmetrization.antisymmetrize
        options:
          heading_level: 3
