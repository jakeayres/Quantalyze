"""Magnetoconductivity of quasi-2D metals from the Boltzmann equation.

A general-purpose solver for your own Fermi surfaces: `FermiSurface` computes
σ_ij(B) with the Shockley–Chambers tube integral in the relaxation-time
approximation. The submodules build Fermi-surface contours from your own band
or from analytic test cases (`generators`), provide relaxation-time models
(`scattering`) and convert common band-structure units into SI (`units`).

Examples:
    >>> import numpy as np
    >>> from quantalyze.beta import boltzmann as bz
    >>> from quantalyze.core.constants import ELECTRON_MASS
    >>> df = bz.generators.polar(512, k_fermi=lambda phi: 7e9 * (1 + 0.05 * np.cos(4 * phi)),
    ...                          mass=ELECTRON_MASS, tau=1e-13)
"""
# Public modules are the namespaces bz.generators, bz.scattering and bz.units. Modules
# whose names start with an underscore are private; anything public in them is
# re-exported here.
from . import generators, scattering, units
from ._legacy import FermiSurface

__all__ = ["FermiSurface", "generators", "scattering", "units"]
