from _joint_data import d, measured, rate, surface, temperatures, true_gamma1, true_shared

# --8<-- [start:example]
import warnings

import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf


# Each temperature alone: Γ_imp + αT² is one constant there, so fit it as one number.
def rate_at_one_temperature(phi, gamma_iso, gamma1):
    return gamma_iso + gamma1 * np.cos(2 * phi) ** 2


alone = {T: bzf.fit_transport(measured[T], surface, rate=rate_at_one_temperature, layer_spacing=d,
                              p0={"gamma_iso": 1e12, "gamma1": 1e12}, extrapolate=True)
         for T in temperatures}

# All five together, linked by the law Γ_imp + αT².
datasets = [bzf.Dataset(measured[T], surface, rate=rate, constants={"T": T}, name=f"{T} K", layer_spacing=d,
                        extrapolate=True) for T in temperatures]
p0 = {"gamma_imp": 1e12, "alpha": 1e8, "gamma1": 1e12}
joint = bzf.fit_joint(datasets, p0=p0, separate=["gamma1"])

# The same joint model on a single temperature cannot tell Γ_imp from αT².
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bzf.fit_joint(datasets[:1], p0=p0, separate=["gamma1"])
print("one temperature:", caught[0].message)

for name, unit in (("gamma_imp", "s⁻¹"), ("alpha", "s⁻¹ K⁻²")):
    print(f"joint: {name} = {joint[name]:.4g} ± {joint.errors[name]:.2g} {unit} (truth {true_shared[name]:.4g})")
print("\n  T   error of Γ₁ (10¹⁰ s⁻¹)\n      alone  joint")
for T in temperatures:
    print(f"{T:3d}  {alone[T].errors['gamma1'] / 1e10:6.2f} {joint.errors[f'gamma1[{T} K]'] / 1e10:6.2f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

T = np.array(temperatures)
fig, (left, right) = plt.subplots(1, 2)
iso = [alone[t]["gamma_iso"] / 1e12 for t in temperatures]
left.errorbar(T, iso, [alone[t].errors["gamma_iso"] / 1e12 for t in temperatures], fmt="o", ms=4,
              label="each T alone")
line = np.linspace(0, 100, 101)
left.plot(line, (joint["gamma_imp"] + joint["alpha"] * line**2) / 1e12, label=r"joint: $\Gamma_{imp} + \alpha T^2$")
left.set(xlabel="T (K)", ylabel=r"isotropic rate ($10^{12}$ s$^{-1}$)")
left.legend(fontsize=8)
right.plot(T, [alone[t].errors["gamma1"] / 1e10 for t in temperatures], "o-", ms=4, label="each T alone")
right.plot(T, [joint.errors[f"gamma1[{t} K]"] / 1e10 for t in temperatures], "s-", ms=4, label="joint")
right.set(xlabel="T (K)", ylabel=r"error of $\Gamma_1$ ($10^{10}$ s$^{-1}$)", ylim=(0, None))
right.legend(fontsize=8)
fig.tight_layout()
