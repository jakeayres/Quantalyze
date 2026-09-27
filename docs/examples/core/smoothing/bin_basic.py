from _data import sweep

# --8<-- [start:example]
import quantalyze as qz

binned = qz.bin(sweep, "field", minimum=0, maximum=9, width=0.02)

print(f"{len(sweep)} rows in, {len(binned)} rows out")
print(binned.head())
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(sweep.field, sweep.resistance, ".", ms=2, color="0.6", label="raw sweep")
ax.plot(binned.field, binned.resistance, color="C0", lw=1, label="binned, width = 0.02 T")
ax.set(xlabel="Field (T)", ylabel="Resistance (Ω)", xlim=(5, 9), ylim=(1.08, 1.34))
ax.legend()
