from _data import d, data, rate, surface, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

result = bzf.fit_transport(data, surface, rate=rate, p0={"gamma0": 1e12, "gamma1": 1e12}, layer_spacing=d,
                           extrapolate=True, resistivity_error="resistivity_error",
                           hall_resistivity_error="hall_resistivity_error")

# Draw parameter sets from the fit's covariance and compute τ(φ) for each.
rng = np.random.default_rng(6)
best = np.array([result[n] for n in result.names])
draws = rng.multivariate_normal(best, result.covariance.to_numpy(), size=2000)  # (draws, parameters)
phi = np.linspace(0, np.pi / 2, 91)
tau = np.array([1 / rate(phi, *draw) for draw in draws])  # (draws, angles), s
low, high = np.percentile(tau, [2.5, 97.5], axis=0)       # the 95% band at each angle
tau_best = 1 / rate(phi, *best)

true_tau = 1 / rate(phi, **true_rate)
print(f"τ ranges from {tau_best.min() * 1e15:.0f} fs (along the axes) to {tau_best.max() * 1e15:.0f} fs "
      "(along the diagonals)")
print(f"95% band half-width: {100 * np.max((high - low) / 2 / tau_best):.2f}% at most")
print(f"truth inside the 95% band at every angle: {bool(np.all((low <= true_tau) & (true_tau <= high)))}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
degrees = np.degrees(phi)
ax.fill_between(degrees, 100 * (low / tau_best - 1), 100 * (high / tau_best - 1), alpha=0.3, label="95% band")
ax.plot(degrees, 100 * (true_tau / tau_best - 1), color="k", label="truth")
ax.axhline(0, color="C0", label="best fit")
ax.set(xlabel="φ (degrees)", ylabel=r"$\tau / \tau_{\rm best} - 1$ (%)")
ax.legend(fontsize=8)
