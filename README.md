# Quantalyze

If your analysis and visualization workflow is some combination of `pandas`, `numpy`, `scipy` and `matplotlib`, it is likely that `quantalyze` can save you a lot of time. `Quantalyze` is a set of utility functions that facilitate the analysis of scientific data. Most of the operations are designed to work on `pandas.DataFrame` objects and functionality is pulled from a combination of `numpy`, `scipy`.

Non-domain-specific functionality is contained within the `core` module. Domain-specific utilities are contained within their respective modules and import from the `core` module. For example, the `transport` module contains a function that calculates the magnetoresistance from resistivity data taken in magnetic fields that pulls its functionality from `core.smoothing` (binning onto equally spaced x values) and `core.symmetrization` (the symmetrization in field).


[Documentation on ReadtheDocs](https://quantalyze.readthedocs.io/en/latest/)
[Package on PyPI](https://pypi.org/project/quantalyze/)