from _data import circle, d

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

field = np.linspace(0, 60, 61)  # T
sigma = bz.conductivity(circle, field, layer_spacing=d)
rho = bz.resistivity(sigma)
hall = bz.hall_coefficient(sigma)

print(sigma.iloc[[0, 10, 60]])
print(f"rho_xx: {rho.rho_xx.min():.4e} to {rho.rho_xx.max():.4e} Ω·m")
print(f"R_H at 10 T: {hall.iloc[10]:.4e} m³/C")
# --8<-- [end:example]

import matplotlib.pyplot as plt

from quantalyze.core.constants import ELEMENTARY_CHARGE

n = bz.carrier_density(circle, layer_spacing=d)
fig, ax = plt.subplots()
ax.plot(field, rho.rho_xx / rho.rho_xx.iloc[0], label=r"$\rho_{xx}(B)/\rho_{xx}(0)$")
ax.plot(field[1:], -hall.iloc[1:] * n * ELEMENTARY_CHARGE, label=r"$-R_H\,ne$")
ax.set(xlabel="Field (T)", ylabel="Normalised value", ylim=(0.9, 1.1))
ax.legend()
