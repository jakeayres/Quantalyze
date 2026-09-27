from _data import calibration

# --8<-- [start:example]
import numpy as np
import quantalyze as qz

grid = np.linspace(2, 300, 1000)
linear = qz.interpolate(calibration, "temperature", onto=grid)  # method="linear" is the default
cubic = qz.interpolate(calibration, "temperature", onto=grid, method="cubic")

print(cubic.iloc[::200])
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.loglog(grid, 50 + 3000 / grid**0.8, "--", color="0.5", lw=3, label="true sensor curve")
ax.loglog(linear.temperature, linear.sensor_resistance, color="C1", label='method="linear"')
ax.loglog(cubic.temperature, cubic.sensor_resistance, color="C0", label='method="cubic"')
ax.loglog(calibration.temperature, calibration.sensor_resistance, "o", color="0.4", ms=5, label="calibration points")
ax.set(xlabel="Temperature (K)", ylabel="Sensor resistance (Ω)")
ax.legend()
