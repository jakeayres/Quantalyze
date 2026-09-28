from _data import rt

# --8<-- [start:example]
import quantalyze as qz


def fermi_liquid(temperature, rho0, A):
    return rho0 + A * temperature**2


result = qz.fit(fermi_liquid, rt, "temperature", "resistivity", x_max=20)

print(result.parameters)  # in the order they appear in fermi_liquid
print(f"rho0 = {result['rho0']:.6f}")  # or look them up by name
print(f"A    = {result['A']:.6f}")
# --8<-- [end:example]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.axvspan(0, 20, color="C0", alpha=0.08, lw=0)
ax.text(1, 4.9, "fitted range: x_max=20", color="C0", va="top")
ax.plot(rt.temperature, rt.resistivity, ".", ms=3, color="0.6", label="data")
result.plot(ax, np.linspace(0, 40, 200), color="C0", lw=1.6, label="fit: rho0 + A T²")
ax.set(xlabel="Temperature (K)", ylabel="Resistivity (µΩ cm)", xlim=(0, 40))
ax.legend(loc="lower right")
