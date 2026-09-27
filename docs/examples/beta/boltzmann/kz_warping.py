from _data import d, k_F, mass

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.core.constants import HBAR

a = bz.units.angstrom_to_meter(3.87)
c2 = HBAR**2 / (2 * mass)
t_z = bz.units.ev_to_joule(0.01)  # interlayer hopping
e_F = c2 * k_F**2


def energy(kx, ky, kz):
    # interlayer hopping ∝ (cos k_x a − cos k_y a)²: largest along the axes, zero on the diagonals
    form = (np.cos(kx * a) - np.cos(ky * a)) ** 2 / 4
    return c2 * (kx**2 + ky**2) - 2 * t_z * np.cos(kz * d) * form - e_F


def gradient(kx, ky, kz):
    difference = np.cos(kx * a) - np.cos(ky * a)
    hop = t_z * np.cos(kz * d) * difference * a
    return (2 * c2 * kx + hop * np.sin(kx * a),
            2 * c2 * ky - hop * np.sin(ky * a),
            2 * t_z * d * np.sin(kz * d) * difference**2 / 4)


surface = bz.generators.from_dispersion_3d(
    256, 8, energy=energy, gradient=gradient, tau=1e-13, max_radius=2 * k_F, layer_spacing=d)

field = np.linspace(0, 60, 61)
sigma = bz.conductivity(surface, field, layer_spacing=d, kz="kz")
interlayer = bz.magnetoresistance(sigma, component="zz")
in_plane = bz.magnetoresistance(sigma, component="xx")

print(f"sigma_zz / sigma_xx at 0 T: {sigma.sigma_zz.iloc[0] / sigma.sigma_xx.iloc[0]:.2e}")
print(f"MR at 60 T: in-plane {in_plane.iloc[60]:.2e}, interlayer {interlayer.iloc[60]:.3f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(field, interlayer, label=r"interlayer, $\rho_{zz}$")
ax.plot(field, in_plane, label=r"in-plane, $\rho_{xx}$")
ax.set(xlabel="Field along $c$ (T)", ylabel="Magnetoresistance")
ax.legend()
