# Symmetrization

This document provides an overview of the symmetrization methods implemented in the `Quantalyze` library.

## ::: quantalyze.core.symmetrize.symmetrize

## ::: quantalyze.core.symmetrize.antisymmetrize

## Usage


```python
from quantalyze.core import symmetrize


# Define a function to fit
f = lambda x, a, b: a + b*x

# Perform the fit and evaluate the function at x=20
result = fit(f, df, 'x', 'y').evaluate(20)

print("result:", result)
```
