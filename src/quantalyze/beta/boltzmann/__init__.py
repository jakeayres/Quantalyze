"""Magnetoconductivity of quasi-2D metals from the Boltzmann equation.

A general-purpose solver for your own Fermi surfaces. `conductivity` computes
σ_ij(B) with the Shockley–Chambers tube integral in the relaxation-time
approximation, exactly in ω_cτ, and `resistivity`, `hall_coefficient` and
`magnetoresistance` follow from it. The submodules build Fermi-surface contours
from your own band or from analytic test cases (`generators`), provide
relaxation-time models (`scattering`) and convert common band-structure units
into SI (`units`).

Examples:
    >>> import numpy as np
    >>> from quantalyze.beta import boltzmann as bz
    >>> from quantalyze.core.constants import ELECTRON_MASS
    >>> df = bz.generators.polar(512, k_fermi=lambda phi: 7e9 * (1 + 0.05 * np.cos(4 * phi)),
    ...                          mass=ELECTRON_MASS, tau=1e-13)
    >>> sigma = bz.conductivity(df, np.linspace(0, 30, 31), layer_spacing=1e-9)
    >>> r_h = bz.hall_coefficient(sigma)
"""
# Public modules are the namespaces bz.generators, bz.scattering and bz.units. Modules
# whose names start with an underscore are private; anything public in them is
# re-exported here.
from . import generators, scattering, units
from ._legacy import FermiSurface
from ._response import (
    carrier_density,
    conductivity,
    conductivity_tensor,
    hall_coefficient,
    magnetoresistance,
    resistivity,
)

__all__ = [
    "conductivity",
    "resistivity",
    "hall_coefficient",
    "magnetoresistance",
    "carrier_density",
    "conductivity_tensor",
    "FermiSurface",
    "generators",
    "scattering",
    "units",
]
