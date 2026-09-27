from _data import anisotropic, d, field_per_omega_c_tau

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.core.constants import ELEMENTARY_CHARGE

omega_c_tau = np.concatenate([[0], np.geomspace(0.01, 100, 81)])
sigma = bz.conductivity(anisotropic, omega_c_tau * field_per_omega_c_tau, layer_spacing=d)

n = bz.carrier_density(anisotropic, layer_spacing=d)
hall = bz.hall_coefficient(sigma) * n * -ELEMENTARY_CHARGE  # in units of 1/(nq)
mr = bz.magnetoresistance(sigma)

print(f"low field:  R_H = {hall.iloc[1]:.3f} / nq,  MR = {mr.iloc[1]:.1e}")
print(f"high field: R_H = {hall.iloc[-1]:.3f} / nq,  MR = {mr.iloc[-1]:.3f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogx(omega_c_tau[1:], hall.iloc[1:], label=r"$R_H\,nq$")
ax.semilogx(omega_c_tau[1:], 1 + mr.iloc[1:], label=r"$\rho_{xx}(B)/\rho_{xx}(0)$")
ax.axhline(1.25, color="0.6", lw=0.8, ls="--")
ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
ax.set(xlabel=r"$\omega_c\tau_0$", ylabel="Normalised value")
ax.legend()
