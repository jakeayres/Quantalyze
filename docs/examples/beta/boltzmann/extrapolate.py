from _data import d, field_per_omega_c_tau, k_F, mass, tau

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.core.constants import ELEMENTARY_CHARGE

n = k_F**2 / (2 * np.pi * d)  # carrier density of the circle (spin-degenerate)
exact = n * ELEMENTARY_CHARGE**2 * tau / mass / 2  # Drude σ_xx at ω_cτ = 1

sizes = [32, 64, 128, 256, 512]
plain, extrapolated = [], []
for size in sizes:
    circle = bz.generators.circle(size, k_fermi=k_F, mass=mass, tau=tau)
    for errors, extrapolate in ((plain, False), (extrapolated, True)):
        sigma = bz.conductivity(circle, field_per_omega_c_tau, layer_spacing=d, extrapolate=extrapolate)
        errors.append(abs(sigma.sigma_xx.iloc[0] / exact - 1))
    print(f"N = {size:3d}: error {plain[-1]:.1e} as computed, {extrapolated[-1]:.1e} extrapolated")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.loglog(sizes, plain, "o-", label="as computed (error ∝ N⁻²)")
ax.loglog(sizes, extrapolated, "s-", label="extrapolate=True (error ∝ N⁻⁴)")
ax.set(xlabel="Number of points N", ylabel=r"Relative error of $\sigma_{xx}$")
ax.legend()
