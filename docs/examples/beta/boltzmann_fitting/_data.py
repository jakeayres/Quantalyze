import numpy as np
import pandas as pd

import quantalyze as qz
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.quantum_oscillations import extremal_frequency
from quantalyze.transport import calculate_hall_resistivity, calculate_resistivity

# The running sample: a hole pocket about the zone corner of a square-lattice band.
d = 0.6e-9                             # interlayer spacing (m)
a = bz.units.angstrom_to_meter(3.87)   # lattice constant (m)
t = bz.units.ev_to_joule(0.25)         # nearest-neighbour hopping (J)
t_prime = -0.2 * t                     # next-nearest-neighbour hopping (J)
corner = (np.pi / a, np.pi / a)        # the pocket's centre


def band(mu, n_points=512):
    """The Fermi surface at chemical potential mu (J), with a polar-angle column "phi"."""
    surface = bz.generators.tight_binding(n_points, tau=1.0, lattice_constant=a, hopping=t, next_hopping=t_prime,
                                          chemical_potential=mu, center=corner)
    surface["phi"] = bzf.polar_angle(surface, center=corner)
    return surface


def rate(phi, gamma0, gamma1):
    """Scattering rate 1/τ (s⁻¹): weakest along the diagonals, strongest along the axes."""
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2


# The truth the "measurements" are made from: what the fits should find.
true_mu = bz.units.ev_to_joule(-0.12)
true_rate = {"gamma0": 3e12, "gamma1": 6e12}  # s⁻¹
surface = band(true_mu)

# The measurement: a 1 mm × 0.5 mm × 20 μm bar, swept up to +60 T and down to −60 T.
length, width, thickness = 1e-3, 0.5e-3, 20e-6  # m
rng = np.random.default_rng(0)
field = np.linspace(0, 60, 4001)  # T
# The sample's true resistivities: the continuum limit (extrapolated in the number of points).
truth = bzf.transport(surface, field, rate=rate, parameters=true_rate, layer_spacing=d, extrapolate=True)
area = 4 * np.pi**2 * bz.carrier_density(surface, layer_spacing=d) * d / 2  # the pocket's k-space area (m⁻²)
sweeps = []
for sign in (1, -1):
    r_xx = truth["resistivity"].to_numpy() * length / (width * thickness)   # Ω
    r_xy = sign * truth["hall_resistivity"].to_numpy() / thickness           # Ω, odd in B
    with np.errstate(divide="ignore", invalid="ignore"):
        # Shubnikov–de Haas oscillations, periodic in 1/B and growing with field.
        oscillation = np.where(field > 0, 0.004 * np.exp(-150 / field)
                               * np.cos(2 * np.pi * extremal_frequency(area) / field), 0.0)
    sweeps.append(pd.DataFrame({
        "field": sign * field,
        # Each voltage pair also picks up some of the other: misaligned contacts.
        "r_xx": r_xx * (1 + oscillation) + 0.1 * r_xy + 0.01 * r_xx[0] * rng.standard_normal(field.size),
        "r_xy": r_xy + 0.05 * r_xx + 0.01 * r_xx[0] * rng.standard_normal(field.size),
    }))

# --8<-- [start:prepare]
top, step = 60.0, 1.0  # T: the field range, and the bin width (several quantum-oscillation periods)

# The longitudinal signal is even in B and the Hall signal odd: (anti)symmetrising removes
# the pickup between them, and averaging into bins removes the oscillations.
r_xx = qz.symmetrize(sweeps, "field", "r_xx", -top, top, step)
r_xy = qz.antisymmetrize(sweeps, "field", "r_xy", -top, top, step)
data = pd.DataFrame({
    "field": r_xx["field"],
    "resistivity": calculate_resistivity(r_xx, length=length, width=width, thickness=thickness, resistance="r_xx"),
    "hall_resistivity": calculate_hall_resistivity(r_xy, hall_resistance="r_xy", thickness=thickness),
})

# Errors: the scatter of the raw points (from differences of neighbours, which cancel the
# smooth signal), divided by √(number of raw points averaged into each value).
counts = np.histogram(pd.concat(sweeps)["field"], np.arange(-top - step / 2, top + step, step))[0]
averaged = np.where(data["field"] == 0, counts, 4 / (1 / counts + 1 / counts[::-1]))
to_resistivity = {"r_xx": width * thickness / length, "r_xy": thickness}  # Ω → Ω·m
for column, raw in (("resistivity", "r_xx"), ("hall_resistivity", "r_xy")):
    scatter = np.mean([np.std(np.diff(s[raw])) for s in sweeps]) / np.sqrt(2)  # Ω per raw point
    data[f"{column}_error"] = scatter * to_resistivity[raw] / np.sqrt(averaged)
data = data[data["field"] >= 0].reset_index(drop=True)
# --8<-- [end:prepare]
