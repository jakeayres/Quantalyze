# Differentiation

Numerical derivatives of one column with respect to another. Each function returns a `Series` lined up with your DataFrame, so you can assign it straight to a new column.

| Function | Slope at row *i* uses | NaN at |
|---|---|---|
| [`qz.derivative`](#derivative) | rows *i − 1* and *i + 1* (same as `central_difference`) | first and last row |
| [`qz.central_difference`](#one-sided) | rows *i − 1* and *i + 1* | first and last row |
| [`qz.forward_difference`](#one-sided) | rows *i* and *i + 1* | last row |
| [`qz.backward_difference`](#one-sided) | rows *i − 1* and *i* | first row |

Use `derivative` unless you have a reason not to.

??? example "Example data used on this page"

    `clean` is a magnetoresistance curve R = 1 + 0.004 B² on an evenly spaced field grid, so the exact answer is dR/dB = 0.008 B. `noisy` is the same curve with measurement noise.

    ```python
    --8<-- "core/differentiation/_data.py"
    ```

## `derivative` { #derivative }

**dy/dx at every row.**

```python
--8<-- "core/differentiation/derivative_basic.py:example"
```

```text title="Output"
--8<-- "core/differentiation/derivative_basic.txt"
```

![Resistance versus field, and its derivative compared with the exact answer](../examples/core/differentiation/derivative_basic.png#only-light)
![Resistance versus field, and its derivative compared with the exact answer](../examples/core/differentiation/derivative_basic-dark.png#only-dark)

The first and last rows are NaN because they have a neighbour on only one side.

### Noisy data: smooth first

Differentiation turns small point-to-point noise into large slope errors. Smooth first (see [Smoothing](smoothing.md)), then differentiate the smoothed column.

```python
--8<-- "core/differentiation/derivative_noisy.py:example"
```

```text title="Output"
--8<-- "core/differentiation/derivative_noisy.txt"
```

![Derivative of noisy data, with and without smoothing first](../examples/core/differentiation/derivative_noisy.png#only-light)
![Derivative of noisy data, with and without smoothing first](../examples/core/differentiation/derivative_noisy-dark.png#only-dark)

!!! warning "Watch out"

    - **Rows must be in x order.** Ascending and descending both work (a down-sweep is fine), but a DataFrame holding an up-sweep followed by a down-sweep isn't. [`bin`](smoothing.md#bin) them onto one grid first.
    - **Repeated x values give `inf` or NaN,** because the step in x is zero. `bin` fixes this too.
    - **Noise is amplified.** Smooth first, as shown above.

## `forward_difference`, `backward_difference`, `central_difference` { #one-sided }

**The building blocks.** Forward and backward differences are one-sided slopes; the central difference is their average, and is what `derivative` uses. Here are all three on y = x², where the exact answer is 2x:

```python
--8<-- "core/differentiation/differences.py:example"
```

```text title="Output"
--8<-- "core/differentiation/differences.txt"
```

The central difference is exact for this quadratic. The one-sided differences are not, because each is really the slope half a step away from its row: the forward difference at x = 1 is 3, the exact slope at x = 1.5, and the backward difference is the slope at x = 0.5. That is why `derivative` uses the central difference.

## API reference

??? info "`qz.derivative`"

    ::: quantalyze.core.differentiation.derivative
        options:
          heading_level: 3

??? info "`qz.central_difference`"

    ::: quantalyze.core.differentiation.central_difference
        options:
          heading_level: 3

??? info "`qz.forward_difference`"

    ::: quantalyze.core.differentiation.forward_difference
        options:
          heading_level: 3

??? info "`qz.backward_difference`"

    ::: quantalyze.core.differentiation.backward_difference
        options:
          heading_level: 3
