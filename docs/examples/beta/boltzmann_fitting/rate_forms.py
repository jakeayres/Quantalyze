from _data import d, data, surface, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf


def isotropic(gamma0):
    return gamma0


def harmonics(phi, gamma0, a4):  # the C4-symmetric Fourier series, to first order
    return gamma0 * (1 + a4 * np.cos(4 * phi))


def anisotropic(phi, gamma0, gamma1):  # the form the data were made with
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2


def hot_spots(phi, gamma0, height, width=0.3):  # peaks on the axes; width is held at 0.3 rad
    peaks = sum(np.exp((np.cos(phi - p) - 1) / width**2) for p in (0, np.pi / 2, np.pi, 3 * np.pi / 2))
    return gamma0 + height * peaks


def mean_free_path(vx, vy, ell):  # the same mean free path everywhere: 1/τ = |v|/ℓ
    return np.hypot(vx, vy) / ell


forms = {
    isotropic: {"gamma0": 1e12},
    harmonics: {"gamma0": 1e12, "a4": 0.1},
    anisotropic: {"gamma0": 1e12, "gamma1": 1e12},
    hot_spots: {"gamma0": 1e12, "height": 1e12},
    mean_free_path: {"ell": 1e-8},
}
errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
results = {}
for form, p0 in forms.items():
    results[form.__name__] = bzf.fit_transport(data, surface, rate=form, p0=p0, layer_spacing=d, extrapolate=True,
                                               bounds={"a4": (-0.95, 0.95)} if "a4" in p0 else None, **errors)
    result = results[form.__name__]
    values = ", ".join(f"{n} = {result[n]:.3g}" for n in result.names)
    print(f"{form.__name__:15s} χ²/ν = {result.reduced_chi_squared:8.1f}   {values}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

phi = np.linspace(0, np.pi / 2, 200)


def label(result):
    chi = result.reduced_chi_squared
    return f"χ²/ν = {chi:.1f}" if chi < 100 else f"χ²/ν = {chi:,.0f}"


fig, ax = plt.subplots()
ax.plot(np.degrees(phi), anisotropic(phi, **true_rate) / 1e12, color="k", lw=3, alpha=0.25, label="truth")
for form in forms:
    result = results[form.__name__]
    if form is mean_free_path:
        continue  # depends on |v|, not φ alone: shown at the nodes below
    gamma = np.broadcast_to(form(phi, **{n: result[n] for n in result.parameters}) if form is not isotropic
                            else result["gamma0"], phi.shape)
    ax.plot(np.degrees(phi), gamma / 1e12, label=f"{form.__name__} ({label(result)})")
quarter = surface["phi"] <= np.pi / 2
gamma = mean_free_path(surface["vx"], surface["vy"], results["mean_free_path"]["ell"])
ax.plot(np.degrees(surface["phi"][quarter]), gamma[quarter] / 1e12, ":",
        label=f"mean_free_path ({label(results['mean_free_path'])})")
ax.set(xlabel="φ (degrees)", ylabel=r"$1/\tau$ ($10^{12}$ s$^{-1}$)")
ax.set_ylim(2.5, 13)
ax.legend(fontsize=7, ncol=2, loc="upper center")
