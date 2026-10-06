from _data import d, data, rate, surface, true_rate

# --8<-- [start:example]
import warnings
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
options = dict(rate=rate, layer_spacing=d, p0=true_rate, extrapolate=True, **errors)

# 1. Every Fermi velocity 10% too high: the fit makes every rate 10% higher, at the same χ².
faster = surface.assign(vx=1.1 * surface["vx"], vy=1.1 * surface["vy"])
right, wrong = (bzf.fit_transport(data, s, **options) for s in (surface, faster))
print(f"rates ×{wrong['gamma0'] / right['gamma0']:.6f} and ×{wrong['gamma1'] / right['gamma1']:.6f}; "
      f"same χ² to 1e-10: {abs(wrong.chi_squared / right.chi_squared - 1) < 1e-10}")


# 2. A pocket with mirror planes cannot tell τ(φ) from τ(−φ): the sign of a sin 4φ term is lost.
def chiral(phi, gamma0, gamma1, gamma2):
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2 + gamma2 * np.sin(4 * phi)


made = {**true_rate, "gamma2": 1.5e12}
rng = np.random.default_rng(3)
clean = bzf.transport(surface, data["field"], rate=chiral, parameters=made, layer_spacing=d, extrapolate=True)
chiral_data = data.assign(
    resistivity=clean["resistivity"] + data["resistivity_error"] * rng.standard_normal(len(data)),
    hall_resistivity=clean["hall_resistivity"] + data["hall_resistivity_error"] * rng.standard_normal(len(data)))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = bzf.fit_transport(chiral_data, surface, rate=chiral, layer_spacing=d, extrapolate=True, **errors,
                               p0=[{**true_rate, "gamma2": 1e12}, {**true_rate, "gamma2": -1e12}],
                               bounds={"gamma2": (-2.5e12, 2.5e12)})
print(result.starts.to_string(float_format="{:.4g}".format))
print("warned of a different solution:", any("different solution" in str(w.message) for w in caught))
# --8<-- [end:example]
