from _data import sweep

# --8<-- [start:example]
import quantalyze as qz

binned = qz.bin(sweep, "field", minimum=0, maximum=9, width=0.02)
binned["smoothed"] = qz.window(binned, "resistance", window_size=7)

print(binned[["field", "resistance", "smoothed"]].head())
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(binned.field, binned.resistance, ".", ms=3, color="0.6", label="binned")
ax.plot(binned.field, binned.smoothed, color="C0", lw=1.2, label="window(window_size=7)")
ax.set(xlabel="Field (T)", ylabel="Resistance (Ω)", xlim=(5, 9), ylim=(1.08, 1.34))
ax.legend()
