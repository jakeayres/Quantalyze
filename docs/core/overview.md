# Core

The core functions are general-purpose tools for measured data: fitting, smoothing, differentiating, symmetrizing, interpolating and Fourier transforming. Everything is available straight from the package:

```python
import quantalyze as qz
```

## How every function works

- **Pass a DataFrame and the names of its columns.** For example, `qz.derivative(df, "field", "resistance")`. You never pull out arrays yourself.
- **Your DataFrame is never modified.** Functions that compute one value per row return a `pandas.Series` lined up with your rows, so you can store it as a new column: `df["dR/dB"] = qz.derivative(...)`. Functions that change the rows (resampling, binning, transforming) return a new `DataFrame`.
- **Several measurements?** `bin`, `symmetrize` and `antisymmetrize` also accept a list of DataFrames, which they combine first.

## What's in core

| Function | Use it to | Returns | Page |
|---|---|---|---|
| `qz.fit` | fit a model to two columns | a `Fit` | [Fitting](fitting.md) |
| `qz.derivative` | take dy/dx | `Series` | [Differentiation](differentiation.md) |
| `qz.forward_difference`, `qz.backward_difference`, `qz.central_difference` | one-sided or central slopes | `Series` | [Differentiation](differentiation.md#one-sided) |
| `qz.bin` | average data onto a regular grid, or merge sweeps | `DataFrame` | [Smoothing](smoothing.md#bin) |
| `qz.window` | rolling-average smoothing | `Series` | [Smoothing](smoothing.md#window) |
| `qz.savgol_filter` | smoothing that keeps peaks | `Series` | [Smoothing](smoothing.md#savgol_filter) |
| `qz.symmetrize` | keep the part even in x (e.g. rxx) | `DataFrame` | [Symmetrization](symmetrization.md#symmetrize) |
| `qz.antisymmetrize` | keep the part odd in x (e.g. Hall) | `DataFrame` | [Symmetrization](symmetrization.md#antisymmetrize) |
| `qz.interpolate` | resample all columns onto new x values | `DataFrame` | [Interpolation](interpolation.md) |
| `qz.fft`, `qz.Window` | frequency spectrum (e.g. quantum oscillations) | `DataFrame` | [FFT](fft.md) |
| `qz.constants` | physical constants in SI units | — | [Constants](constants.md) |

## A complete example

The functions are designed to chain together. Here, two raw Hall sweeps of a 100 nm film become a carrier density in three steps.

??? example "Example data"

    `up` and `down` are field sweeps of the Hall resistance `rxy`, with an offset from misaligned contacts.

    ```python
    --8<-- "core/overview/_data.py"
    ```

```python
--8<-- "core/overview/pipeline.py:example"
```

```text title="Output"
--8<-- "core/overview/pipeline.txt"
```

![Raw Hall sweeps, the antisymmetrized Hall resistance, and the straight-line fit](../examples/core/overview/pipeline.png#only-light)
![Raw Hall sweeps, the antisymmetrized Hall resistance, and the straight-line fit](../examples/core/overview/pipeline-dark.png#only-dark)

Every example in these docs can be copied and run as it stands. Open the "Example data" box on each page and run that code first.
