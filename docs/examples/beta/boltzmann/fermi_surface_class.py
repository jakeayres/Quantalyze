from _data import d, k_F, mass, tau

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
fs = bz.FermiSurface(theta, np.full(512, k_F), mass, tau, d)

sxx, sxy, syx, syy = fs.calculate_conductivity(10.0)
print(f"sigma_xx = {sxx:.4e} S/m, sigma_xy = {sxy:.4e} S/m")

sigma = bz.conductivity(fs.contour(), 10.0, layer_spacing=d)  # the same calculation
print(sigma)
# --8<-- [end:example]
