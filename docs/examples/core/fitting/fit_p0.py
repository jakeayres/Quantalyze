from _data import transition

# --8<-- [start:example]
import quantalyze as qz


def peak(temperature, height, centre, width, slope):
    """A Lorentzian peak on a sloping background."""
    return height / (1 + ((temperature - centre) / width) ** 2) + slope * temperature


# Without a starting guess, every parameter starts at 1 and the fit wanders off.
bad = qz.fit(peak, transition, "temperature", "heat_capacity")

# A rough guess read off a plot, in the same order as peak's parameters.
good = qz.fit(peak, transition, "temperature", "heat_capacity", p0=[0.5, 10, 0.5, 0])

print(f"without p0: centre = {bad['centre']:.2f} K, width = {bad['width']:.2g} K")
print(f"with p0:    centre = {good['centre']:.2f} K, width = {good['width']:.2g} K")
# --8<-- [end:example]

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(5, 15, 1000)
fig, ax = plt.subplots()
ax.plot(transition.temperature, transition.heat_capacity, ".", ms=3, color="0.6", label="data")
bad.plot(ax, x, color="C1", lw=1.4, label="no p0 (wrong answer, no error)")
good.plot(ax, x, color="C0", lw=1.4, label="p0=[0.5, 10, 0.5, 0]")
ax.set(xlabel="Temperature (K)", ylabel="Heat capacity (arb. units)")
ax.legend(loc="upper right")
