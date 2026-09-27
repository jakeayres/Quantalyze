from _data import d, k_F, mass, tau

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

electrons = bz.generators.circle(512, k_fermi=k_F, mass=mass, tau=tau)
holes = bz.generators.circle(512, k_fermi=k_F, mass=2 * mass, tau=tau / 2, carrier="hole")

field = np.linspace(0, 60, 61)
sigma = bz.conductivity([electrons, holes], field, layer_spacing=d)  # σ summed over pockets
mr = bz.magnetoresistance(sigma)
hall = bz.hall_coefficient(sigma)

print(f"MR at 30 T: {mr.iloc[30]:.3f}, at 60 T: {mr.iloc[60]:.3f} (ratio {mr.iloc[60] / mr.iloc[30]:.3f})")
print(f"R_H from 1 T to 60 T: {hall.iloc[1]:.4e} to {hall.iloc[60]:.4e} m³/C")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(field, mr, label="compensated electron + hole pockets")
ax.set(xlabel="Field (T)", ylabel="Magnetoresistance")
ax.legend()
