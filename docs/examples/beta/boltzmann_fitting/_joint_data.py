import numpy as np

from quantalyze.beta import boltzmann_fitting as bzf
from _data import band, d, true_mu

# The running sample's hole pocket, measured at five temperatures.
surface = band(true_mu)
temperatures = [10, 30, 50, 70, 90]  # K


def rate(phi, T, gamma_imp, alpha, gamma1):
    """Scattering rate 1/τ (s⁻¹): a constant, a T² term, and an anisotropic part Γ₁ that changes with T."""
    return gamma_imp + alpha * T**2 + gamma1 * np.cos(2 * phi) ** 2


# The truth the "measurements" are made from: what the fits should find.
true_shared = {"gamma_imp": 2e12, "alpha": 4e8}             # s⁻¹ and s⁻¹ K⁻²
true_gamma1 = {T: 1e12 + 5e10 * T for T in temperatures}    # s⁻¹

# ρ_xx(B) and ρ_H(B) up to 60 T at each temperature, with noise of 0.2% of each curve's RMS
# and no error bars.
rng = np.random.default_rng(1)
field = np.linspace(0, 60, 61)  # T
measured = {}
for T in temperatures:
    truth = bzf.transport(surface, field, rate=rate, parameters={**true_shared, "T": T, "gamma1": true_gamma1[T]},
                          layer_spacing=d, extrapolate=True)
    measured[T] = truth.assign(**{
        column: truth[column] + 2e-3 * np.sqrt(np.mean(truth[column] ** 2)) * rng.standard_normal(field.size)
        for column in ("resistivity", "hall_resistivity")
    })
