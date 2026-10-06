from _data import d, data, rate, surface

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf


def hot_spots(phi, gamma0, height, width=0.3):  # a plausible but wrong model
    return gamma0 + height * sum(np.exp((np.cos(phi - p) - 1) / width**2)
                                 for p in (0, np.pi / 2, np.pi, 3 * np.pi / 2))


errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
fits = {
    "right model": bzf.fit_transport(data, surface, rate=rate, p0={"gamma0": 1e12, "gamma1": 1e12},
                                     layer_spacing=d, extrapolate=True, **errors),
    "wrong model": bzf.fit_transport(data, surface, rate=hot_spots, p0={"gamma0": 1e12, "height": 1e12},
                                     layer_spacing=d, extrapolate=True, **errors),
}

# Normalised residuals, (data − model)/error: noise if the model is right, structure if not.
residuals = {}
for label, result in fits.items():
    model = result.evaluate(data["field"])
    residuals[label] = {c: (data[c] - model[c]) / data[f"{c}_error"] for c in ("resistivity", "hall_resistivity")}
    print(f"{label}: χ²/ν = {result.reduced_chi_squared:.1f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, sharey=True)
for ax, column, title in zip(axes, ("resistivity", "hall_resistivity"), (r"$\rho_{xx}$", r"$\rho_H$")):
    for label in fits:
        ax.plot(data["field"], residuals[label][column], ".-", ms=3, lw=0.6, label=label)
    ax.axhspan(-1, 1, color="0.5", alpha=0.15, lw=0)
    ax.set(xlabel="B (T)", title=title)
axes[0].set_ylabel("(data − model) / error")
axes[0].legend(fontsize=8)
fig.tight_layout()
