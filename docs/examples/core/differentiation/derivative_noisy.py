from _data import noisy

# --8<-- [start:example]
import quantalyze as qz

# Differentiating noisy data directly amplifies the noise...
noisy["dR/dB raw"] = qz.derivative(noisy, "field", "resistance")

# ...so smooth first, then differentiate.
noisy["smoothed"] = qz.savgol_filter(noisy, "resistance", window_size=15, order=2)
noisy["dR/dB smoothed"] = qz.derivative(noisy, "field", "smoothed")

print(noisy[["field", "dR/dB raw", "dR/dB smoothed"]].iloc[40:45])
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(noisy.field, noisy["dR/dB raw"], ".-", ms=3, lw=0.6, color="0.6", label="derivative of raw data")
ax.plot(noisy.field, noisy["dR/dB smoothed"], color="C0", lw=1.6, label="smooth, then derivative")
ax.plot(noisy.field, 0.008 * noisy.field, "--", color="C1", lw=1.2, label="exact: 0.008 B")
ax.set(xlabel="Field (T)", ylabel="dR/dB (Ω/T)")
ax.legend(loc="upper left")
