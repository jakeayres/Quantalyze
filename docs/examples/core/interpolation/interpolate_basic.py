from _data import cooldown

# --8<-- [start:example]
import numpy as np
import quantalyze as qz

grid = np.arange(10, 300, 10)  # 10, 20, ..., 290 K
regular = qz.interpolate(cooldown, "temperature", onto=grid)

print(regular.head())
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(cooldown.temperature, cooldown.resistance, ".", ms=3, color="0.6", label="measured (irregular temperatures)")
ax.plot(regular.temperature, regular.resistance, "o", ms=5, mfc="none", color="C0", label="interpolated onto 10, 20, ..., 290 K")
ax.set(xlabel="Temperature (K)", ylabel="Resistance (Ω)")
ax.legend(loc="lower right")
