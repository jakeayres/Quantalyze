from _data import clean

# --8<-- [start:example]
import quantalyze as qz

clean["dR/dB"] = qz.derivative(clean, "field", "resistance")

print(clean.head())
print(clean.tail(3))
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
left.plot(clean.field, clean.resistance, color="C0")
left.set(xlabel="Field (T)", ylabel="Resistance (Ω)", title="resistance")
right.plot(clean.field, 0.008 * clean.field, color="0.6", lw=4, label="exact: 0.008 B")
right.plot(clean.field, clean["dR/dB"], "o", ms=3, color="C0", label="qz.derivative")
right.set(xlabel="Field (T)", ylabel="dR/dB (Ω/T)", title="dR/dB")
right.legend()
