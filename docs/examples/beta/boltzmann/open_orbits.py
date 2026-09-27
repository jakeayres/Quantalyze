from _data import d

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

a = bz.units.angstrom_to_meter(3.87)
G = 2 * np.pi / a  # the sheets repeat every G along k_y

# Two warped sheets near k_x = ±0.5 Å⁻¹, each sampled over exactly one period
sheets = bz.generators.open_sheets(512, k0=5e9, velocity=2e5, tau=1e-13, period=G, warping=1e9)

field = np.linspace(0, 60, 61)
sigma = bz.conductivity(sheets, field, layer_spacing=d, period=(0.0, G))
rho = bz.resistivity(sigma)

print(sigma.iloc[[0, 30, 60], :3])
print(f"MR along x at 60 T: {rho.rho_xx.iloc[60] / rho.rho_xx.iloc[0] - 1:.3f}")
print(f"MR along y at 60 T: {rho.rho_yy.iloc[60] / rho.rho_yy.iloc[0] - 1:.1f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(field, rho.rho_yy / rho.rho_yy.iloc[0] - 1, label=r"along $y$: $\rho_{yy}$")
ax.plot(field, rho.rho_xx / rho.rho_xx.iloc[0] - 1, label=r"along $x$: $\rho_{xx}$")
ax.set(xlabel="Field (T)", ylabel="Magnetoresistance")
ax.legend()
