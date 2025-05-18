# Differentiation

The `differentiation` module provides tools for calculating numerical derivatives of data in `pandas.DataFrame` objects. It includes methods `forward_difference` and `backward_difference` for approximating derivatives using one-sided differences, and `central_difference` for a more accurate two-sided approach. The `derivative` function simply wraps `central_difference`.

::: quantalyze.core.differentiation.forward_difference

::: quantalyze.core.differentiation.backward_difference

::: quantalyze.core.differentiation.central_difference

::: quantalyze.core.differentiation.derivative