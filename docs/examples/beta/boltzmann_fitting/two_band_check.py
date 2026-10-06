# --8<-- [start:example]
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE as e

d = 1e-9                                             # layer spacing (m)
m_e, m_h = ELECTRON_MASS, 2 * ELECTRON_MASS          # band masses (held fixed)
truth = {"k_e": 6e9, "k_h": 8e9, "gamma_e": 2e12, "gamma_h": 3e12}  # k_F (m⁻¹), 1/τ (s⁻¹)


def drude(field, k_e, k_h, gamma_e, gamma_h):
    """The closed-form two-band Drude model: ρ_xx and ρ_H = σ_xy/(σ_xx² + σ_xy²)."""
    sigma_xx = sigma_xy = 0
    for k, m, gamma, q in ((k_e, m_e, gamma_e, -e), (k_h, m_h, gamma_h, e)):
        n = k**2 / (2 * np.pi * d)                   # carriers per m³, g_s = 2
        beta = q * field / (m * gamma)               # signed ω_cτ
        sigma_xx = sigma_xx + n * e**2 / (m * gamma) / (1 + beta**2)
        sigma_xy = sigma_xy + n * e**2 / (m * gamma) * beta / (1 + beta**2)
    return sigma_xx / (sigma_xx**2 + sigma_xy**2), sigma_xy / (sigma_xx**2 + sigma_xy**2)


# Noisy data made from the closed form, so the solver plays no part in making it.
field = np.linspace(0, 60, 61)
rho_xx, rho_h = drude(field, **truth)
rng = np.random.default_rng(1)
data = pd.DataFrame({"field": field, "resistivity_error": 2e-3 * rho_xx,
                     "hall_resistivity_error": 2e-3 * np.abs(rho_h).max()})
data["resistivity"] = rho_xx + data["resistivity_error"] * rng.standard_normal(field.size)
data["hall_resistivity"] = rho_h + data["hall_resistivity_error"] * rng.standard_normal(field.size)
p0 = {"k_e": 5e9, "k_h": 9e9, "gamma_e": 1e12, "gamma_h": 1e12}

# 1. bzf: two circular pockets, one relaxation time each, solved by the Boltzmann code.
def pockets(k_e, k_h):
    return [bz.generators.circle(512, k_fermi=k_e, mass=m_e, tau=1.0),
            bz.generators.circle(512, k_fermi=k_h, mass=m_h, tau=1.0, carrier="hole")]


errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
fits = {extrapolate: bzf.fit_transport(data, pockets, rate=[lambda gamma_e: gamma_e, lambda gamma_h: gamma_h],
                                       p0=p0, layer_spacing=d, extrapolate=extrapolate, **errors)
        for extrapolate in (False, True)}

# 2. The closed form, fitted with plain scipy: no code in common with the solver.
names = list(p0)
scale = np.array([p0[n] for n in names])


def residuals(x):
    model_xx, model_h = drude(field, *(x * scale))
    return np.concatenate([(model_xx - data["resistivity"]) / data["resistivity_error"],
                           (model_h - data["hall_resistivity"]) / data["hall_resistivity_error"]])


reference = least_squares(residuals, np.ones(len(names)))
covariance = np.linalg.inv(reference.jac.T @ reference.jac) * np.outer(scale, scale)
reference_values = dict(zip(names, reference.x * scale))
reference_errors = dict(zip(names, np.sqrt(np.diag(covariance))))

print("            closed form           |bzf − closed form| / σ   bzf σ / closed-form σ")
print("                                   N=512    extrapolated")
for n in names:
    shifts = [abs(fits[x][n] - reference_values[n]) / reference_errors[n] for x in (False, True)]
    print(f"{n:8s} {reference_values[n]:.4e} ± {reference_errors[n]:.1e}   {shifts[0]:.3f}    {shifts[1]:.4f}"
          f"         {fits[True].errors[n] / reference_errors[n]:.4f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt

curve = fits[True].evaluate(np.linspace(0, 60, 121))
closed_xx, closed_h = drude(curve["field"].to_numpy(), **reference_values)
fig, (left, right) = plt.subplots(1, 2)
left.plot(field, data["resistivity"] * 1e8, ".", ms=3, label="data")
left.plot(curve["field"], curve["resistivity"] * 1e8, label="bzf")
left.plot(curve["field"], closed_xx * 1e8, "--", color="0.5", label="closed form")
left.set(xlabel="B (T)", ylabel=r"$\rho_{xx}$ (μΩ cm)")
right.plot(field, data["hall_resistivity"] * 1e8, ".", ms=3)
right.plot(curve["field"], curve["hall_resistivity"] * 1e8)
right.plot(curve["field"], closed_h * 1e8, "--", color="0.5")
right.set(xlabel="B (T)", ylabel=r"$\rho_H$ (μΩ cm)")
left.legend()
fig.tight_layout()
