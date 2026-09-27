from _data import oscillations

# --8<-- [start:example]
import quantalyze as qz

coarse = qz.fft(oscillations, "inverse_field", "signal", window=qz.Window.HANN)
fine = qz.fft(oscillations, "inverse_field", "signal", window=qz.Window.HANN, n=2**16)

print(len(coarse), "points, spacing", round(coarse["frequency"][1], 2), "T")
print(len(fine), "points, spacing", round(fine["frequency"][1], 2), "T")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(coarse.frequency, coarse.amplitude, "o", ms=4, color="0.5", label=f"default n ({len(oscillations)} points)")
ax.plot(fine.frequency, fine.amplitude, color="C0", label="n=2**16 (zero padded)")
ax.set(xlabel="Frequency (T)", ylabel="Amplitude (arb. units)", xlim=(100, 200))
ax.legend()
