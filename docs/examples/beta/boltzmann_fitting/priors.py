from _data import a, corner, d, data, rate, t, t_prime, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf


def band(mu, t_prime):  # the chemical potential and the next-nearest-neighbour hopping are fitted
    surface = bz.generators.tight_binding(512, tau=1.0, lattice_constant=a, hopping=t, next_hopping=t_prime,
                                          chemical_potential=mu, center=corner)
    surface["phi"] = bzf.polar_angle(surface, center=corner)
    return surface


errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
guess = {"mu": true_mu, "t_prime": 0.9 * t_prime}  # say band structure gives t′ 10% off

# Stage 1: the rates on the band-structure surface. The poor χ²/ν says the surface is wrong.
stage1 = bzf.fit_transport(data, band(**guess), rate=rate, p0={"gamma0": 1e12, "gamma1": 1e12}, layer_spacing=d,
                           extrapolate=True, **errors)
print(f"stage 1, surface held: χ²/ν = {stage1.reduced_chi_squared:.0f}")

# Stage 2: free the surface, starting from stage 1. Then again with a 1% prior on t′.
p0 = {**guess, "gamma0": stage1["gamma0"], "gamma1": stage1["gamma1"]}
alone = bzf.fit_transport(data, band, rate=rate, p0=p0, layer_spacing=d, extrapolate=True, **errors)
prior = 0.01 * abs(t_prime)
informed = bzf.fit_transport(data, band, rate=rate, p0=p0, layer_spacing=d, extrapolate=True, **errors,
                             priors={"t_prime": (t_prime, prior)})

meV = bz.units.ev_to_joule(1e-3)
print(f"stage 2, surface fitted: χ²/ν = {alone.reduced_chi_squared:.2f}\n")
print("          transport alone     with the prior on t′")
for name, unit, label in (("mu", meV, "meV"), ("t_prime", meV, "meV"), ("gamma0", 1e12, "10¹² s⁻¹"),
                          ("gamma1", 1e12, "10¹² s⁻¹")):
    print(f"{name:8s} {alone[name] / unit:8.2f} ± {alone.errors[name] / unit:.2f}   "
          f"{informed[name] / unit:8.2f} ± {informed.errors[name] / unit:.2f}   {label}")

# A prior adds information: 1/σ² = 1/σ_data² + 1/σ_prior² for the parameter it is on.
expected = 1 / np.sqrt(1 / alone.errors["t_prime"] ** 2 + 1 / prior**2)
print(f"\nσ(t′) with the prior: {informed.errors['t_prime'] / meV:.3f} meV; "
      f"1/√(1/σ_data² + 1/σ_prior²) = {expected / meV:.3f} meV")
# --8<-- [end:example]
