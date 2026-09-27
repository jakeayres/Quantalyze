from _data import up, down

# --8<-- [start:example]
import quantalyze as qz

# 1. Keep the part of the Hall signal that is odd in field (removes the rxx pickup).
hall = qz.antisymmetrize([up, down], "field", "rxy", minimum=-9, maximum=9, step=0.1)

# 2. Fit a straight line through the origin.
line = qz.fit(lambda field, slope: slope * field, hall, "field", "rxy")

# 3. Turn the slope into a carrier density using the film thickness.
thickness = 100e-9                                 # m
hall_coefficient = line["slope"] * thickness       # m³/C
density = 1 / (qz.constants.E * hall_coefficient)  # m⁻³

print(f"slope = {line['slope']:.4f} Ω/T")
print(f"n     = {density / 1e6:.3g} cm⁻³")
# --8<-- [end:example]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(up.field, up.rxy, ".", ms=1.5, color="0.6", label="raw up + down sweeps")
ax.plot(down.field, down.rxy, ".", ms=1.5, color="0.6")
ax.plot(hall.field, hall.rxy, color="C0", lw=2.5, label="1. antisymmetrize")
line.plot(ax, np.array([-9, 9]), color="C1", lw=1.2, ls="--", label="2. fit")
ax.set(xlabel="Field (T)", ylabel="rxy (Ω)")
ax.legend(loc="upper left")
