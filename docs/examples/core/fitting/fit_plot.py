from _data import rt

# --8<-- [start:example]
import matplotlib.pyplot as plt
import numpy as np
import quantalyze as qz

result = qz.fit(lambda T, rho0, A: rho0 + A * T**2, rt, "temperature", "resistivity", x_max=20)

fig, ax = plt.subplots()
ax.plot(rt["temperature"], rt["resistivity"], ".", color="0.6", label="data")
result.plot(ax, np.linspace(0, 40, 200), color="C0", label="T² fit")  # extra keywords go to ax.plot
ax.set(xlabel="Temperature (K)", ylabel="Resistivity (µΩ cm)")
ax.legend()
# --8<-- [end:example]
