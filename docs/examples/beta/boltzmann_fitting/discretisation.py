from _data import band, d, data, rate, surface, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

# Data at the continuum limit: 4096 points, extrapolated in N (its own error is ~1e-12).
field = data["field"].to_numpy()
limit = bzf.transport(band(true_mu, n_points=4096), field, rate=rate, parameters=true_rate, layer_spacing=d,
                      extrapolate=True)

# Fit them with N points: the parameters are off by the solver's own error at that N.
options = dict(rate=rate, layer_spacing=d, p0=true_rate,
               least_squares_options={"ftol": 1e-15, "xtol": 1e-15, "gtol": 1e-15})
sizes = [64, 128, 256, 512, 1024]
bias = {extrapolate: np.array([[bzf.fit_transport(limit, band(true_mu, n_points=n), extrapolate=extrapolate,
                                                  **options)[name] / true_rate[name] - 1 for name in true_rate]
                               for n in sizes])
        for extrapolate in (False, True)}  # (sizes, parameters)

# Compare with the statistical error of the running sample's fit.
sigma = bzf.fit_transport(data, surface, rate=rate, layer_spacing=d, p0=true_rate, extrapolate=True,
                          resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
relative_error = np.array([sigma.errors[name] / sigma[name] for name in true_rate])
slopes = np.polyfit(np.log(sizes), np.log(np.abs(bias[False])), 1)[0]

print("    N    |bias| / σ  (gamma0, gamma1)")
print("         plain           extrapolated")
for i, n in enumerate(sizes):
    plain, extrapolated = (np.abs(bias[x][i]) / relative_error for x in (False, True))
    print(f"{n:5d}   {plain[0]:.4f} {plain[1]:.4f}   {extrapolated[0]:.5f} {extrapolated[1]:.5f}")
print(f"bias ∝ N^{slopes[0]:.2f} (gamma0), N^{slopes[1]:.2f} (gamma1)")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for i, name in enumerate(["$\\Gamma_0$", "$\\Gamma_1$"]):
    ax.loglog(sizes, np.abs(bias[False][:, i]) / relative_error[i], "o-", label=f"{name}")
    ax.loglog(sizes, np.abs(bias[True][:, i]) / relative_error[i], "s--", mfc="none",
              label=f"{name}, extrapolate=True")
ax.loglog(sizes, 0.5 * (sizes[0] / np.array(sizes)) ** 2, ":", color="0.5", label=r"$N^{-2}$")
ax.loglog(sizes, 0.05 * (sizes[0] / np.array(sizes)) ** 4, "-.", color="0.5", label=r"$N^{-4}$")
ax.axhline(0.1, color="0.6", lw=0.8)
ax.text(sizes[0], 0.13, "0.1σ", color="0.5")
ax.set(xlabel="Points on the Fermi surface, N", ylabel="|bias| / statistical error")
ax.legend(fontsize=8)
