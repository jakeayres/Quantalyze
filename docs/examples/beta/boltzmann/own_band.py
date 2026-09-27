from _data import d

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz

# A square-lattice band, with parameters in the units a band-structure fit gives
a = bz.units.angstrom_to_meter(3.87)
t, t_prime, mu = (bz.units.ev_to_joule(e) for e in (0.25, -0.06, 0.0))


def energy(kx, ky):  # J, measured from the Fermi level
    x, y = kx * a, ky * a
    return -2 * t * (np.cos(x) + np.cos(y)) - 4 * t_prime * np.cos(x) * np.cos(y) - mu


def gradient(kx, ky):  # ∂ε/∂k (J·m)
    x, y = kx * a, ky * a
    return (a * (2 * t * np.sin(x) + 4 * t_prime * np.sin(x) * np.cos(y)),
            a * (2 * t * np.sin(y) + 4 * t_prime * np.cos(x) * np.sin(y)))


pocket = bz.generators.from_dispersion(
    512, energy=energy, gradient=gradient, tau=1e-13,
    max_radius=np.pi / a, center=(np.pi / a, np.pi / a),  # a hole pocket about the zone corner
)
n = bz.carrier_density(pocket, layer_spacing=d)
sigma = bz.conductivity(pocket, [0.0, 10.0], layer_spacing=d)

print(pocket.head(3))
print(f"n = {n:.3e} m⁻³, R_H(10 T) = {bz.hall_coefficient(sigma).iloc[1]:.3e} m³/C")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
qx, qy = (pocket.kx - np.pi / a) * a / np.pi, (pocket.ky - np.pi / a) * a / np.pi
ax.plot(qx, qy)
every = slice(None, None, 32)
ax.quiver(qx[every], qy[every], pocket.vx[every], pocket.vy[every], color="C1", width=0.006)
ax.set(xlabel=r"$(k_x - \pi/a)\,a/\pi$", ylabel=r"$(k_y - \pi/a)\,a/\pi$", aspect="equal")
