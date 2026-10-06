# docs: slow
from _data import band, d, rate, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
import pandas as pd
from quantalyze.beta import boltzmann_fitting as bzf

surface = band(true_mu, n_points=256)
field = np.linspace(0, 60, 31)
clean = bzf.transport(surface, field, rate=rate, parameters=true_rate, layer_spacing=d)
names = list(true_rate)
rng = np.random.default_rng(2)


def fit(noise_xx, noise_h, with_errors):
    """Fit one noisy copy of the clean curves; return (pulls, χ²)."""
    data = clean.assign(resistivity=clean["resistivity"] + noise_xx * rng.standard_normal(field.size),
                        hall_resistivity=clean["hall_resistivity"] + noise_h * rng.standard_normal(field.size),
                        resistivity_error=noise_xx, hall_resistivity_error=noise_h)
    errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
    result = bzf.fit_transport(data, surface, rate=rate, layer_spacing=d, p0={"gamma0": 1e12, "gamma1": 1e12},
                               bounds={"gamma0": (0, None), "gamma1": (0, None)}, **(errors if with_errors else {}))
    return [(result[n] - true_rate[n]) / result.errors[n] for n in names], result.chi_squared


trials = 200
# With measured errors: 0.2% of ρ_xx at each field, and 0.2% of the largest Hall resistivity.
absolute = [fit(2e-3 * clean["resistivity"].to_numpy(), 2e-3 * clean["hall_resistivity"].abs().max(), True)
            for _ in range(trials)]
# Without errors: each channel weighted by its RMS, so its noise must be in that proportion.
rms = {c: np.sqrt(np.mean(clean[c] ** 2)) for c in ("resistivity", "hall_resistivity")}
relative = [fit(2e-3 * rms["resistivity"], 2e-3 * rms["hall_resistivity"], False) for _ in range(trials)]

pulls_absolute = np.array([p for p, _ in absolute])  # (trials, 2)
pulls_relative = np.array([p for p, _ in relative])
chi_squared = np.array([c for _, c in absolute])
nu = 2 * field.size - len(names)

table = pd.DataFrame({
    mode: {**{f"mean pull {n}": f"{pulls[:, i].mean():+.2f}" for i, n in enumerate(names)},
           **{f"std pull {n}": f"{pulls[:, i].std():.2f}" for i, n in enumerate(names)},
           "1σ coverage": f"{np.mean(np.abs(pulls) < 1):.0%}"}
    for mode, pulls in (("with errors", pulls_absolute), ("without errors", pulls_relative))
})
print(table.to_string())
print(f"\nmean χ² = {chi_squared.mean():.1f} for ν = {nu} (expected {nu} ± {np.sqrt(2 * nu / trials):.1f})")
# --8<-- [end:example]

import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

fig, (left, right) = plt.subplots(1, 2)
x = np.linspace(-4, 4, 200)
left.hist(pulls_absolute.ravel(), bins=np.linspace(-4, 4, 25), density=True, alpha=0.6, label="pulls")
left.plot(x, norm.pdf(x), label="N(0, 1)")
left.set(xlabel=r"(fit $-$ truth) / error", ylabel="Density")
left.legend()
c = np.linspace(20, 110, 200)
right.hist(chi_squared, bins=20, density=True, alpha=0.6, label=r"$\chi^2$")
right.plot(c, chi2.pdf(c, nu), label=rf"$\chi^2_{{{nu}}}$")
right.set(xlabel=r"$\chi^2$", ylabel="Density")
right.legend()
fig.tight_layout()
