from _data import up, down

# --8<-- [start:example]
import quantalyze as qz

rxx = qz.symmetrize([up, down], "field", "rxx", minimum=-9, maximum=9, step=0.1)

print(rxx.head())
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(up.field, up.rxx, ".", ms=1.5, color="0.6", label="measured (up + down sweeps)")
ax.plot(down.field, down.rxx, ".", ms=1.5, color="0.6")
ax.plot(rxx.field, rxx.rxx, color="C0", lw=1.6, label="symmetrize")
ax.set(xlabel="Field (T)", ylabel="rxx (Ω)")
ax.legend(loc="upper center")
