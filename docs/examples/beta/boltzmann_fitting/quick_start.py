from _data import band, d, data, true_mu

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

surface = band(true_mu)  # the known Fermi surface, with a polar-angle column "phi"


def rate(phi, gamma0, gamma1):  # 1/τ (s⁻¹): phi is a column, gamma0 and gamma1 are fitted
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2


result = bzf.fit_transport(
    data, surface, rate=rate, layer_spacing=d,
    p0={"gamma0": 1e12, "gamma1": 1e12},
    bounds={"gamma0": (0, None), "gamma1": (0, None)},
    resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error",
    extrapolate=True,  # cancel the solver's O(N⁻²) error: precise data need it (see Discretisation)
)
for name in result.names:
    print(f"{name} = {result[name] / 1e12:.4f} ± {result.errors[name] / 1e12:.4f} ×10¹² s⁻¹")
print(f"χ²/ν = {result.reduced_chi_squared:.2f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fit = result.evaluate(np.linspace(0, 60, 121))
fig, (left, right) = plt.subplots(1, 2)
left.plot(data.field, data.resistivity * 1e8, ".", ms=3, label="data")
left.plot(fit.field, fit.resistivity * 1e8, label="fit")
left.set(xlabel="B (T)", ylabel=r"$\rho_{xx}$ (μΩ cm)")
right.plot(data.field, data.hall_resistivity * 1e8, ".", ms=3)
right.plot(fit.field, fit.hall_resistivity * 1e8)
right.set(xlabel="B (T)", ylabel=r"$\rho_H$ (μΩ cm)")
left.legend()
fig.tight_layout()
