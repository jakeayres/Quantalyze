# --8<-- [start:example]
import numpy as np
import pandas as pd
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf

# Quasi-one-dimensional chains along y: two open sheets near k_y = ±π/2b that run along k_x.
a = b = 4e-10                                             # lattice constants (m)
t_a, t_b = bz.units.ev_to_joule(0.02), bz.units.ev_to_joule(0.25)
sheets, period = bz.generators.open_sheets_from_dispersion(
    256, period=(2 * np.pi / a, 0.0), across=(-np.pi / b, np.pi / b), tau=1.0,
    energy=lambda kx, ky: -2 * t_a * np.cos(kx * a) - 2 * t_b * np.cos(ky * b),
    gradient=lambda kx, ky: (2 * t_a * a * np.sin(kx * a), 2 * t_b * b * np.sin(ky * b)))


def rate(kx, gamma0, gamma1):  # sheets have no centre: write the rate in terms of k
    return gamma0 + gamma1 * np.cos(kx * a) ** 2


# "Measured" data, made here with 0.2% noise. Pass period= exactly as to bz.conductivity.
truth = {"gamma0": 2e12, "gamma1": 4e12}
field = np.linspace(0, 30, 31)
clean = bzf.transport(sheets, field, rate=rate, parameters=truth, layer_spacing=1e-9, period=period)
print("Hall resistivity zero (to 1e-12 of ρ_xx):",
      bool(np.all(np.abs(clean["hall_resistivity"]) < 1e-12 * clean["resistivity"])))
rng = np.random.default_rng(5)
data = clean.assign(resistivity_error=2e-3 * clean["resistivity"])
data["resistivity"] += data["resistivity_error"] * rng.standard_normal(field.size)

# The two sheets are mirror images, so there is no Hall signal to fit: fit ρ_xx alone.
result = bzf.fit_transport(data, sheets, rate=rate, p0={"gamma0": 1e12, "gamma1": 1e12}, layer_spacing=1e-9,
                           period=period, hall_resistivity=None, resistivity_error="resistivity_error")
for name in result.names:
    print(f"{name}: {result[name] / 1e12:.3f} ± {result.errors[name] / 1e12:.3f} ×10¹² s⁻¹ "
          f"(truth {truth[name] / 1e12:.0f})")
# --8<-- [end:example]
