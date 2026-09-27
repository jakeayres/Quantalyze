from _data import up, down

# --8<-- [start:example]
import quantalyze as qz

rxy = qz.antisymmetrize([up, down], "field", "rxy", minimum=-9, maximum=9, step=0.1)

print(rxy[rxy["field"].between(-0.25, 0.25)])  # passes exactly through zero
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(up.field, up.rxy, ".", ms=1.5, color="0.6", label="measured (offset by rxx pickup)")
ax.plot(down.field, down.rxy, ".", ms=1.5, color="0.6")
ax.plot(rxy.field, rxy.rxy, color="C0", lw=1.6, label="antisymmetrize")
ax.axhline(0, color="0.4", lw=0.8)
ax.axvline(0, color="0.4", lw=0.8)
ax.set(xlabel="Field (T)", ylabel="rxy (Ω)")
ax.legend(loc="upper left")
