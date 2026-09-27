from _data import transition

# --8<-- [start:example]
import quantalyze as qz

transition["window"] = qz.window(transition, "heat_capacity", window_size=15)
transition["savgol"] = qz.savgol_filter(transition, "heat_capacity", window_size=15, order=3)

print(transition.loc[transition["temperature"].between(9.9, 10.1)])
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(transition.temperature, transition.heat_capacity, ".", ms=3, color="0.6", label="data")
ax.plot(transition.temperature, transition.window, color="C1", lw=1.4, label="window(window_size=15)")
ax.plot(transition.temperature, transition.savgol, color="C0", lw=1.4, label="savgol_filter(window_size=15, order=3)")
ax.set(xlabel="Temperature (K)", ylabel="Heat capacity (arb. units)", xlim=(7, 13))
ax.legend(loc="upper left")
