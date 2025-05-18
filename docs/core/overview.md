# Core

The **Core** module of the Quantalyze package provides essential utilities for scientific data analysis. These utilities include functions for fitting, symmetrization, smoothing, differentiation, and Fourier transforms. All of the more specific modules, such as `transport`, derive much of their functionality from this foundational module.

All core submodules are importable from `quantalyze` 

## Differentiation

The `differentiation` module provides tools for calculating numerical derivatives of data in `pandas.DataFrame` objects. It includes methods `forward_difference` and `backward_difference` for approximating derivatives using one-sided differences, and `central_difference` for a more accurate two-sided approach. The `derivative` function simply wraps `central_difference`.

## Interpolation

The `interpolation` module provides tools for interpolating data in pandas DataFrames. It allows users to map data onto new x-values using methods like linear, quadratic, or cubic interpolation. The `interpolate` function works seamlessly with pandas objects, making it easy to handle missing data or resample datasets for analysis.

## Smoothing

The `smoothing` module provides practical methods for reducing noise in data. It includes `bin` for grouping data into bins, `window` for rolling window smoothing, and `savgol_filter` for preserving features like peaks during smoothing.

## Fitting

The `fitting` module provides a simple wrapper for `scipy.curve_fit()` that facilitates the rapid fitting to data held in a `pandas.DataFrame` and subsequent evaluation and plotting of these fits. The expected work flow is to call `quantalyze.fit(...)` and either retrieve the result of the fit from the attributes of the returned `Fit` object or call one of the `Fit` object's utility methods (e.g. `evaluate(...)` or `plot(...)`).

## Symmetrization

The `symmetrization` module provides tools for processing data to enforce symmetry or antisymmetry. The `symmetrize` function averages values with their reversed counterparts to create symmetric datasets, while the `antisymmetrize` function computes the difference between values and their reversed counterparts to create antisymmetric datasets.

