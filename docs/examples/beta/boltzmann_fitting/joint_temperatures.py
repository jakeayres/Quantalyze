from _joint_data import d, measured, rate, surface, temperatures, true_gamma1, true_shared

# --8<-- [start:example]
import pandas as pd
from quantalyze.beta import boltzmann_fitting as bzf

# One Dataset per temperature: the rate's argument T receives each dataset's constant.
datasets = [
    bzf.Dataset(measured[T], surface, rate=rate, constants={"T": T}, name=f"{T} K", layer_spacing=d,
                extrapolate=True)
    for T in temperatures
]

result = bzf.fit_joint(
    datasets,
    p0={"gamma_imp": 1e12, "alpha": 1e8, "gamma1": 1e12},
    separate=["gamma1"],  # one gamma1 per temperature; gamma_imp and alpha are shared
    bounds={"gamma_imp": (0, None), "alpha": (0, None), "gamma1": (0, None)},
)

truth = pd.Series({**true_shared, **{f"gamma1[{T} K]": value for T, value in true_gamma1.items()}})
table = result.summary()[["value", "error"]].assign(truth=truth)
print(table.to_string(float_format="{:.4g}".format))
print(f"\nweighting: {result.weighting}")
print(result.fit_quality.to_string(float_format="{:.3g}".format))
# --8<-- [end:example]

import matplotlib.pyplot as plt

curves = result.evaluate()  # each dataset's model at its measured fields
fig, (left, right) = plt.subplots(1, 2)
for dataset in datasets:
    data, fit = dataset.data, curves[dataset.name]
    (line,) = left.plot(fit.field, fit.resistivity * 1e8)
    left.plot(data.field[::3], data.resistivity[::3] * 1e8, ".", ms=3, color=line.get_color())
    right.plot(fit.field, fit.hall_resistivity * 1e8, color=line.get_color(), label=dataset.name)
    right.plot(data.field[::3], data.hall_resistivity[::3] * 1e8, ".", ms=3, color=line.get_color())
left.set(xlabel="B (T)", ylabel=r"$\rho_{xx}$ (μΩ cm)")
right.set(xlabel="B (T)", ylabel=r"$\rho_H$ (μΩ cm)")
right.legend(fontsize=8, loc="upper left")
fig.tight_layout()
