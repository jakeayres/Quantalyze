# Smoothing

Three ways to reduce noise. Each takes a DataFrame and a column name.

| Function | What it does | Returns |
|---|---|---|
| [`qz.bin`](#bin) | Averages scattered or repeated data onto a regular grid | a new `DataFrame` |
| [`qz.window`](#window) | Rolling average over neighbouring rows | a `Series` |
| [`qz.savgol_filter`](#savgol_filter) | Rolling polynomial fit, which keeps peaks sharp | a `Series` |

A typical chain is `bin` first (to get an evenly spaced grid), then `window` or `savgol_filter` if it is still too noisy.

??? example "Example data used on this page"

    Every example below starts from these DataFrames: `sweep` and `repeat` are two noisy 0–9 T magnetoresistance sweeps logged at irregular fields, and `transition` is a heat-capacity peak. Run this first if you are copying the examples.

    ```python
    --8<-- "core/smoothing/_data.py"
    ```

## `bin` { #bin }

**Average data onto an evenly spaced grid.** Give it the column that defines the grid (here `field`), the first and last grid points, and the spacing. Every row within half a spacing of a grid point is averaged into that point, for every column.

```python
--8<-- "core/smoothing/bin_basic.py:example"
```

```text title="Output"
--8<-- "core/smoothing/bin_basic.txt"
```

![Raw sweep and binned sweep between 5 and 9 tesla](../examples/core/smoothing/bin_basic.png#only-light)
![Raw sweep and binned sweep between 5 and 9 tesla](../examples/core/smoothing/bin_basic-dark.png#only-dark)

The `temperature` column was averaged too: `bin` averages every column, not just the one you smooth.

### Choosing the width

A wider bin averages more points, so it is less noisy, but it also blurs real features narrower than the bin.

```python
--8<-- "core/smoothing/bin_width.py:example"
```

```text title="Output"
--8<-- "core/smoothing/bin_width.txt"
```

![The same sweep binned with widths of 0.01, 0.05 and 0.25 tesla](../examples/core/smoothing/bin_width.png#only-light)
![The same sweep binned with widths of 0.01, 0.05 and 0.25 tesla](../examples/core/smoothing/bin_width-dark.png#only-dark)

### Combining several sweeps

Pass a list of DataFrames to pool them before averaging. This is the easy way to merge repeated sweeps, or an up-sweep and a down-sweep.

```python
--8<-- "core/smoothing/bin_combine.py:example"
```

```text title="Output"
--8<-- "core/smoothing/bin_combine.txt"
```

!!! warning "Watch out"

    - **Empty bins give NaN rows.** If the grid is finer than your data, some grid points have no data. Drop them with `binned.dropna()`.
    - **The grid stops at `maximum` or just below it.** `minimum=0, maximum=1, width=0.3` gives 0, 0.3, 0.6, 0.9.
    - **The end bins may be only half full.** The bin at `maximum` collects data from `maximum - width/2` to `maximum + width/2`. If your data stop exactly at `maximum`, that bin only sees the lower half, which biases it slightly.
    - **Every other column must be numeric,** because they are all averaged. Drop text columns first.

## `window` { #window }

**Replace each value with a weighted average of the rows around it.** `window_size` is a number of rows, not an x distance, so use it on evenly spaced data (for example, the output of `bin`).

```python
--8<-- "core/smoothing/window_basic.py:example"
```

```text title="Output"
--8<-- "core/smoothing/window_basic.txt"
```

![Binned sweep and its rolling-window average](../examples/core/smoothing/window_basic.png#only-light)
![Binned sweep and its rolling-window average](../examples/core/smoothing/window_basic-dark.png#only-dark)

The shape of the weighting is set by `window_type`, which is passed to pandas' `rolling(win_type=...)`. The default is `"triang"`; others include `"hann"` and `"boxcar"` (a plain unweighted average).

!!! warning "Watch out"

    - **The first and last `window_size // 2` values are NaN,** because the window doesn't fit there (see the output above).
    - **It works on row order.** Sort by x first, and bin irregularly spaced data first, or the average mixes points that are far apart in x.
    - **It flattens peaks.** A peak narrower than the window loses height. Use [`savgol_filter`](#savgol_filter) if peak heights matter.

## `savgol_filter` { #savgol_filter }

**Smooth by fitting a polynomial to each run of rows.** A [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) removes noise about as well as `window` but keeps the height and width of peaks, and it has no NaN at the ends. `window_size` is again a number of rows, and `order` is the polynomial degree.

```python
--8<-- "core/smoothing/savgol_basic.py:example"
```

```text title="Output"
--8<-- "core/smoothing/savgol_basic.txt"
```

![A noisy peak smoothed by window and by savgol_filter; savgol_filter keeps the peak height](../examples/core/smoothing/savgol_basic.png#only-light)
![A noisy peak smoothed by window and by savgol_filter; savgol_filter keeps the peak height](../examples/core/smoothing/savgol_basic-dark.png#only-dark)

The true peak height is 0.70. With the same 15-row window, `savgol_filter` gets 0.65, while `window` flattens it to 0.59.

!!! warning "Watch out"

    - **`window_size` must be larger than `order`.** Use an odd `window_size` so the fit is centred on each point.
    - **It works on row order.** As with `window`, use sorted, evenly spaced data.
    - **Higher `order` follows the data more closely**, including more of the noise. `order=2` or `3` is usually right.

## API reference

??? info "`qz.bin`"

    ::: quantalyze.core.smoothing.bin
        options:
          heading_level: 3

??? info "`qz.window`"

    ::: quantalyze.core.smoothing.window
        options:
          heading_level: 3

??? info "`qz.savgol_filter`"

    ::: quantalyze.core.smoothing.savgol_filter
        options:
          heading_level: 3
