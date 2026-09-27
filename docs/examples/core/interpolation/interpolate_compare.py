from _data import cooldown, warmup

# --8<-- [start:example]
import numpy as np
import quantalyze as qz

grid = np.arange(10, 295, 0.5)
cooling = qz.interpolate(cooldown, "temperature", onto=grid)
warming = qz.interpolate(warmup, "temperature", onto=grid)

# Now that both share the same temperatures, compare them row by row.
hysteresis = warming["resistance"] - cooling["resistance"]
print(f"largest difference: {hysteresis.max():.3f} Ω at {grid[hysteresis.argmax()]} K")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, (top, bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(6.4, 4.4), gridspec_kw={"height_ratios": [2, 1]}, layout="constrained"
)
top.plot(cooling.temperature, cooling.resistance, color="C0", label="cooling")
top.plot(warming.temperature, warming.resistance, color="C1", label="warming")
top.set(ylabel="Resistance (Ω)")
top.legend()
bottom.plot(grid, hysteresis, color="C2")
bottom.set(xlabel="Temperature (K)", ylabel="warming − cooling (Ω)")
