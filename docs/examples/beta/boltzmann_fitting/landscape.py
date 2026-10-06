from _data import d, data, rate, surface, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
result = bzf.fit_transport(data, surface, rate=rate, layer_spacing=d, p0=true_rate, extrapolate=True, **errors)


def chi_squared(gamma0, gamma1):
    """χ² of the data against the model at any parameters."""
    model = bzf.transport(surface, data["field"], rate=rate, parameters={"gamma0": gamma0, "gamma1": gamma1},
                          layer_spacing=d, extrapolate=True)
    return sum(np.sum(((model[c] - data[c]) / data[f"{c}_error"]) ** 2) for c in ("resistivity", "hall_resistivity"))


# The covariance predicts Δχ² = 1 one standard deviation away along its principal axes.
best = np.array([result["gamma0"], result["gamma1"]])
values, vectors = np.linalg.eigh(result.covariance.to_numpy())
steps = [sign * np.sqrt(value) * vector for value, vector in zip(values, vectors.T) for sign in (1, -1)]
rises = [chi_squared(*(best + step)) - result.chi_squared for step in steps]
print("Δχ² one σ along each principal axis, both ways:", " ".join(f"{rise:.3f}" for rise in rises))

# The whole landscape, out to 4σ along each principal axis.
u, v = np.meshgrid(np.linspace(-4, 4, 21), np.linspace(-4, 4, 21))
gamma0, gamma1 = (best[:, None, None] + vectors[:, :1, None] * np.sqrt(values[0]) * u
                  + vectors[:, 1:, None] * np.sqrt(values[1]) * v)
landscape = np.vectorize(chi_squared)(gamma0, gamma1) - result.chi_squared
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.contour(gamma0 / 1e12, gamma1 / 1e12, landscape, levels=[1, 4, 9], colors="C0")
ax.plot([], [], color="C0", label="Δχ² = 1, 4, 9")
angle = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(angle), np.sin(angle)])
for k in (1, 2, 3):
    ellipse = best[:, None] + vectors @ (k * np.sqrt(values)[:, None] * circle)
    ax.plot(ellipse[0] / 1e12, ellipse[1] / 1e12, "--", color="C1", lw=1, label="covariance: 1, 2, 3σ" if k == 1 else None)
ax.plot(true_rate["gamma0"] / 1e12, true_rate["gamma1"] / 1e12, "k+", ms=10, label="truth")
ax.set(xlabel=r"$\Gamma_0$ ($10^{12}$ s$^{-1}$)", ylabel=r"$\Gamma_1$ ($10^{12}$ s$^{-1}$)")
ax.legend(fontsize=8)
