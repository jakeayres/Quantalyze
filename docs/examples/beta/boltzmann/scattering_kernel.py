from _data import d

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.core.constants import HBAR

# Chains along y, weakly coupled across them: two open sheets near k_y = ±π/2b. The
# next-nearest interchain hopping t′ spoils the nesting of the sheets by Q = (π/a, π/b).
a, b = bz.units.angstrom_to_meter(7.7), bz.units.angstrom_to_meter(3.6)
t_y, t_x, t_p = (bz.units.ev_to_joule(e) for e in (0.25, 0.025, 0.01))


def energy(kx, ky):  # J, from the Fermi level (a half-filled band)
    return -2 * t_y * np.cos(ky * b) - 2 * t_x * np.cos(kx * a) - 2 * t_p * np.cos(2 * kx * a)


def gradient(kx, ky):  # ∂ε/∂k (J·m)
    return 2 * a * (t_x * np.sin(kx * a) + 2 * t_p * np.sin(2 * kx * a)), 2 * b * t_y * np.sin(ky * b)


sheets, period = bz.generators.open_sheets_from_dispersion(
    256, energy=energy, gradient=gradient, period=(2 * np.pi / a, 0.0), across=(-np.pi / b, np.pi / b),
    tau=1e-12,  # background scattering (impurities)
)

# Spin-density-wave fluctuations at Q = (π/a, 2k_F), scaled so the hottest spot scatters at 1e13 s⁻¹.
shape = dict(wavevector=(np.pi / a, np.pi / b), correlation_length=(4 * a, 20 * b), lattice_constants=(a, b))
unit = bz.out_scattering_rate(sheets, bz.scattering.spin_fluctuation_kernel(1.0, **shape),
                              layer_spacing=d, period=period)
strength = 1e13 / max(r.max() for r in unit)
kernel = bz.scattering.spin_fluctuation_kernel(strength, **shape)
rates = [strength * r for r in unit]

# The same rates as a relaxation time: what the kernel's out-scattering alone would give.
rta = [df.assign(tau=1 / (1 / df.tau + rate)) for df, rate in zip(sheets, rates)]
field = np.linspace(0, 30, 31)
full = bz.conductivity(sheets, field, layer_spacing=d, period=period, scattering_kernel=kernel)
relaxation = bz.conductivity(rta, field, layer_spacing=d, period=period)

meV = lambda rate: HBAR * rate / 1.602176634e-22  # noqa: E731
print(f"hbar Gamma from {meV(min(r.min() for r in rates)):.2f} (cold) to {meV(1e13):.2f} meV (hot spots)")
print(f"sigma_yy(0) along the chains: kernel {full.sigma_yy[0]:.4e}, relaxation time {relaxation.sigma_yy[0]:.4e} S/m")
mr_full, mr_rta = (bz.magnetoresistance(s, component="xx") for s in (full, relaxation))
print(f"MR across the chains at 30 T: kernel {mr_full.iloc[-1]:.3f}, relaxation time {mr_rta.iloc[-1]:.3f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(field, mr_full, label="scattering kernel")
ax.plot(field, mr_rta, label="relaxation time, same rates", linestyle="--")
ax.set(xlabel="Field along $z$ (T)", ylabel=r"Magnetoresistance, $\rho_{xx}$")
ax.legend()
