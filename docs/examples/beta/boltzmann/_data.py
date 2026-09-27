import numpy as np

from quantalyze.beta import boltzmann as bz
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

d = 0.6e-9      # interlayer spacing (m)
k_F = 7e9       # Fermi wavevector (m⁻¹), 0.7 Å⁻¹
tau = 1e-13     # relaxation time (s), 0.1 ps
mass = 2 * ELECTRON_MASS

# A circular electron pocket: the textbook Drude metal.
circle = bz.generators.circle(512, k_fermi=k_F, mass=mass, tau=tau)

# The same circle with a fourfold scattering rate, 1/τ = (1/τ₀)(1 + 0.6 cos 4φ).
anisotropic = bz.generators.circle(
    512, k_fermi=k_F, mass=mass, tau=lambda phi: bz.scattering.cos4phi(phi, tau, anisotropy=0.6)
)

# ω_cτ = eBτ/m: the field at which carriers complete one radian of orbit per scattering time.
field_per_omega_c_tau = mass / (ELEMENTARY_CHARGE * tau)  # T
