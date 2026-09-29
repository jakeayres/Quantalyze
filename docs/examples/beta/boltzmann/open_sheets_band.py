from _data import d

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

# Chains along y, coupled weakly across them (x) and between the layers (z): the Fermi
# surface is two warped sheets near k_y = ±π/2b, open along k_x and along k_z.
a, b = bz.units.angstrom_to_meter(7.7), bz.units.angstrom_to_meter(3.6)
t_y, t_x = bz.units.ev_to_joule(0.25), bz.units.ev_to_joule(0.025)
t_z, t_d = bz.units.ev_to_joule(0.002), bz.units.ev_to_joule(0.002)  # to the layer above: straight up, next chain over


def energy(kx, ky, kz):  # J, measured from the Fermi level (a half-filled band)
    return (-2 * t_y * np.cos(ky * b) - 2 * t_x * np.cos(kx * a)
            - 2 * t_z * np.cos(kz * d) - 2 * t_d * np.cos(kz * d + kx * a))


def gradient(kx, ky, kz):  # ∂ε/∂k (J·m)
    return (2 * a * (t_x * np.sin(kx * a) + t_d * np.sin(kz * d + kx * a)),
            2 * b * t_y * np.sin(ky * b),
            2 * d * (t_z * np.sin(kz * d) + t_d * np.sin(kz * d + kx * a)))


sheets, period = bz.generators.open_sheets_from_dispersion(
    256, energy=energy, gradient=gradient, tau=1e-12,
    period=(2 * np.pi / a, 0.0),     # the sheets repeat every 2π/a along k_x
    across=(-np.pi / b, np.pi / b),  # search the whole zone along k_y
    n_kz=8, layer_spacing=d,
)
field = np.linspace(0, 30, 61)
sigma = bz.conductivity(sheets, field, layer_spacing=d, kz="kz", period=period)
mr = {axis: bz.magnetoresistance(sigma, component=axis * 2) for axis in "xyz"}

print(f"{len(sheets)} sheets of {len(sheets[0])} nodes (256 per k_z slice)")
print(sheets[1].iloc[[544, 576, 608]].to_string(float_format="{:.4e}".format))
print("MR at 30 T: " + ", ".join(f"along {axis} {mr[axis].iloc[-1]:.3g}" for axis in "xyz"))
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(field[1:], mr["x"].iloc[1:], label=r"across the chains, $\rho_{xx}$")
ax.plot(field[1:], mr["z"].iloc[1:], label=r"between the layers, $\rho_{zz}$")
ax.plot(field[1:], mr["y"].iloc[1:], label=r"along the chains, $\rho_{yy}$")
ax.set(xlabel="Field along $z$ (T)", ylabel="Magnetoresistance", yscale="log", ylim=(1e-7, 1e3))
ax.legend()
