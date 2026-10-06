from _data import band, d, data, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf


def rate(phi, gamma0, gamma1):
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2


result = bzf.fit_transport(
    data, band, rate=rate, layer_spacing=d,  # band(mu) builds the surface: mu is fitted too
    p0={"mu": bz.units.ev_to_joule(-0.05), "gamma0": 1e12, "gamma1": 1e12},
    bounds={"gamma0": (0, None), "gamma1": (0, None)},
    resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error",
    extrapolate=True,  # cancel the solver's O(N⁻²) error: precise data need it (see Discretisation)
)
print(f"mu     = {bz.units.joule_to_ev(result['mu']) * 1e3:.2f} ± "
      f"{bz.units.joule_to_ev(result.errors['mu']) * 1e3:.2f} meV")
for name in ("gamma0", "gamma1"):
    print(f"{name} = {result[name] / 1e12:.4f} ± {result.errors[name] / 1e12:.4f} ×10¹² s⁻¹")
print(f"n      = {bz.carrier_density(result.surface, layer_spacing=d):.4g} m⁻³")
print(f"χ²/ν   = {result.reduced_chi_squared:.2f}")
# --8<-- [end:example]
