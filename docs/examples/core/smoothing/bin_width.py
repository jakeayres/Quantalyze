from _data import sweep

# --8<-- [start:example]
import quantalyze as qz

fine = qz.bin(sweep, "field", minimum=0, maximum=9, width=0.01)
medium = qz.bin(sweep, "field", minimum=0, maximum=9, width=0.05)
coarse = qz.bin(sweep, "field", minimum=0, maximum=9, width=0.25)

print(len(fine), len(medium), len(coarse))
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(fine.field, fine.resistance, color="0.6", lw=0.8, label="width = 0.01 T (noisy)")
ax.plot(medium.field, medium.resistance, color="C0", lw=1.2, label="width = 0.05 T")
ax.plot(coarse.field, coarse.resistance, "o-", color="C1", ms=3, lw=1.2, label="width = 0.25 T (oscillations lost)")
ax.set(xlabel="Field (T)", ylabel="Resistance (Ω)", xlim=(5, 9), ylim=(1.08, 1.34))
ax.legend()
