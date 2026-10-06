from _data import d, data, rate, surface, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.core.constants import ELECTRON_MASS

# A second, electron-like pocket about the zone centre, with its own scattering rate.
electrons = bz.generators.circle(512, k_fermi=3e9, mass=0.8 * ELECTRON_MASS, tau=1.0)


def electron_rate(gamma_e):
    return gamma_e


pockets = [surface, electrons]  # the hole pocket of the running sample, and the electron pocket
rates = [rate, electron_rate]   # one rate per pocket, in the same order

# "Measured" data for the two pockets together (made here, with the running sample's errors).
truth = {**true_rate, "gamma_e": 1e13}
clean = bzf.transport(pockets, data["field"], rate=rates, parameters=truth, layer_spacing=d, extrapolate=True)
rng = np.random.default_rng(4)
two_pockets = data.assign(
    resistivity=clean["resistivity"] + data["resistivity_error"] * rng.standard_normal(len(data)),
    hall_resistivity=clean["hall_resistivity"] + data["hall_resistivity_error"] * rng.standard_normal(len(data)))

result = bzf.fit_transport(two_pockets, pockets, rate=rates, layer_spacing=d, extrapolate=True,
                           p0={"gamma0": 1e12, "gamma1": 1e12, "gamma_e": 1e12},
                           resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
for name in result.names:
    print(f"{name:8s} {result[name] / 1e12:.3f} ± {result.errors[name] / 1e12:.3f} ×10¹² s⁻¹ "
          f"(truth {truth[name] / 1e12:.0f})")
print(f"χ²/ν = {result.reduced_chi_squared:.2f}")
# --8<-- [end:example]
