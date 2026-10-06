from _data import d, data, rate, surface, sweeps, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

print(data.head(3).to_string(float_format="{:.4g}".format))


# Before fitting: the model at your starting values must have the data's Hall sign. This is a
# hole pocket, so ρ_H > 0 for B > 0; reversed Hall contacts or field give the opposite sign.
def hall_sign_agrees(data, surface, rate, start):
    top = data["field"].idxmax()
    model = bzf.transport(surface, data["field"][top], rate=rate, parameters=start, layer_spacing=d)
    return np.sign(model["hall_resistivity"].iloc[0]) == np.sign(data["hall_resistivity"][top])


start = {"gamma0": 1e12, "gamma1": 1e12}
print("Hall sign agrees:", hall_sign_agrees(data, surface, rate, start))
print("...and with the Hall sign flipped:",
      hall_sign_agrees(data.assign(hall_resistivity=-data["hall_resistivity"]), surface, rate, start))

# Synthetic data only: compare with the truth, using the estimated errors.
truth = bzf.transport(surface, data["field"], rate=rate, parameters=true_rate, layer_spacing=d, extrapolate=True)
chi = {c: np.mean(((data[c] - truth[c]) / data[f"{c}_error"]) ** 2) for c in ("resistivity", "hall_resistivity")}
print(f"against the truth: χ²/n = {chi['resistivity']:.2f} (ρ_xx), {chi['hall_resistivity']:.2f} (ρ_H)")
# --8<-- [end:example]

import matplotlib.pyplot as plt
from _data import length, thickness, width

fig, (left, right) = plt.subplots(1, 2)
for sweep in sweeps:
    left.plot(sweep["field"], sweep["r_xx"] * 1e3, ",", color="0.6", alpha=0.5)
    right.plot(sweep["field"], sweep["r_xy"] * 1e3, ",", color="0.6", alpha=0.5)
left.plot(data["field"], data["resistivity"] * length / (width * thickness) * 1e3, ".", ms=4, label="symmetrised")
right.plot(data["field"], data["hall_resistivity"] / thickness * 1e3, ".", ms=4, label="antisymmetrised")
left.plot([], [], ".", color="0.6", label="raw sweeps")
left.set(xlabel="B (T)", ylabel=r"$R_{xx}$ (mΩ)")
right.set(xlabel="B (T)", ylabel=r"$R_{xy}$ (mΩ)")
left.legend(fontsize=8)
right.legend(fontsize=8)
fig.tight_layout()
