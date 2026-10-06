from _data import band, d, data, rate, surface, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.core.constants import ELEMENTARY_CHARGE, HBAR


# ω_c⟨τ⟩ at each top field, with the cyclotron mass m_c = (ħ²/2π) dA/dε from the pocket's area.
def area(mu):
    return 2 * np.pi**2 * bz.carrier_density(band(mu), layer_spacing=d) * d  # A = 4π² n d / g_s


step = bz.units.ev_to_joule(1e-3)
mass = HBAR**2 / (2 * np.pi) * abs(area(true_mu + step) - area(true_mu - step)) / (2 * step)
tau = np.mean(1 / rate(surface["phi"], **true_rate))

errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
tops = [5, 10, 20, 30, 45, 60]  # T
relative = []
print("B_max (T)  ω_c<τ>   σ(Γ0)/Γ0   σ(Γ1)/Γ1")
for top in tops:
    result = bzf.fit_transport(data[data["field"] <= top], surface, rate=rate, layer_spacing=d, p0=true_rate,
                               extrapolate=True, **errors)
    relative.append([result.errors[n] / result[n] for n in true_rate])
    print(f"{top:9d}  {ELEMENTARY_CHARGE * top / mass * tau:6.2f}   {relative[-1][0]:.2e}   {relative[-1][1]:.2e}")
relative = np.array(relative)  # (tops, parameters)
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(tops, relative[:, 0], "o-", label=r"$\Gamma_0$")
ax.semilogy(tops, relative[:, 1], "s-", label=r"$\Gamma_1$")
ax.set(xlabel="Highest field in the fit (T)", ylabel="Relative error")
ax.legend()
