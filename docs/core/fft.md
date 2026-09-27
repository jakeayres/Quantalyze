# FFT

`qz.fft` turns a signal into its frequency spectrum. The typical use in this field is finding quantum-oscillation frequencies: the oscillations are periodic in 1/B, so you transform against **1/B** and the frequencies come out in **tesla**.

It returns a DataFrame with two columns, `frequency` and `amplitude`.

??? example "Example data used on this page"

    `oscillations` holds quantum oscillations from two Fermi-surface pockets, at 150 T and 420 T, with the smooth background already subtracted. It has columns `field`, `inverse_field` (1/B) and `signal`.

    ```python
    --8<-- "core/fft/_data.py"
    ```

## `fft` { #fft }

**Give it the DataFrame, the x column and the signal column.**

```python
--8<-- "core/fft/fft_basic.py:example"
```

```text title="Output"
--8<-- "core/fft/fft_basic.txt"
```

![The oscillating signal against inverse field, and its spectrum with peaks at 150 and 420 tesla](../examples/core/fft/fft_basic.png#only-light)
![The oscillating signal against inverse field, and its spectrum with peaks at 150 and 420 tesla](../examples/core/fft/fft_basic-dark.png#only-dark)

The strongest peak comes out at 148 T rather than 150 T because the spectrum is only sampled every 7.8 T (see [Resolution and `n`](#resolution)).

What `fft` does, step by step:

1. Sorts the rows by x. They can be in any order, and don't need to be evenly spaced: data taken evenly in B are uneven in 1/B.
2. Resamples the signal onto `len(df)` evenly spaced x values, using linear interpolation.
3. Applies the [window](#windows), if you gave one.
4. Takes the FFT and returns the zero and positive frequencies, with the magnitude of each as `amplitude`.

## Windows { #windows }

The data start and stop abruptly, and that spreads every peak into broad "skirts" (spectral leakage) that can bury weaker peaks. A **window** tapers the signal smoothly to zero at both ends. Pass one of the `qz.Window` values:

```python
--8<-- "core/fft/fft_window.py:example"
```

![Spectra with no window, a Hann window and a Kaiser window, on a log scale](../examples/core/fft/fft_window.png#only-light)
![Spectra with no window, a Hann window and a Kaiser window, on a log scale](../examples/core/fft/fft_window-dark.png#only-dark)

On the log scale, the windowed spectra fall away from each peak far faster, so the weaker 420 T peak stands out cleanly.

The available windows:

- `qz.Window.HANN`: a good default.
- `qz.Window.HAMMING`: like Hann, with slightly narrower peaks but more leakage far from them.
- `qz.Window.BLACKMAN`: less leakage than Hann, but broader peaks.
- `qz.Window.BARTLETT`: triangular.
- `qz.Window.KAISER`: adjustable. **Needs `beta=`:** 0 is no window, and larger values give less leakage but broader peaks (5–10 is typical).

## Resolution and `n` { #resolution }

The spectrum's frequency spacing is 1 / (range of x). Here that is 1 / (1/5 T − 1/14 T) ≈ 7.8 T. To separate two frequencies you need more range in x, meaning data down to lower field.

`n` sets the length of the FFT. Values larger than `len(df)` zero-pad the signal. This samples the spectrum more finely, which makes peaks smooth and lets you read off their positions more precisely. It **doesn't** separate peaks that are closer than 1 / (range of x).

```python
--8<-- "core/fft/fft_padding.py:example"
```

```text title="Output"
--8<-- "core/fft/fft_padding.txt"
```

![The 150 tesla peak with the default length and with zero padding](../examples/core/fft/fft_padding.png#only-light)
![The 150 tesla peak with the default length and with zero padding](../examples/core/fft/fft_padding-dark.png#only-dark)

With padding, the peak sits at 150 T.

!!! warning "Watch out"

    - **Transform against the right x.** Quantum oscillations are periodic in 1/B: make a column `df["inverse_field"] = 1 / df["field"]` and transform against that. Transforming against B smears every peak.
    - **Subtract the background first.** A smooth background (such as magnetoresistance) becomes a huge peak near zero frequency that swamps everything else. Fit and subtract it, for example with [`qz.fit`](fitting.md), before calling `fft`.
    - **`amplitude` isn't normalised.** Compare peak heights within one spectrum. Heights change with the number of points and the window, so don't compare them between spectra computed differently.
    - **`Window.KAISER` without `beta`** raises `ValueError`.

## API reference

??? info "`qz.fft`"

    ::: quantalyze.core.fft.fft
        options:
          heading_level: 3

??? info "`qz.Window`"

    ::: quantalyze.core.fft.Window
        options:
          heading_level: 3
