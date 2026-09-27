from _data import oscillations

# --8<-- [start:example]
import quantalyze as qz

spectrum = qz.fft(oscillations, "inverse_field", "signal")

print(spectrum.head())

strongest = spectrum.loc[spectrum["amplitude"].idxmax()]
print(f"strongest frequency: {strongest['frequency']:.0f} T")
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
left.plot(oscillations.inverse_field, oscillations.signal, color="0.5", lw=0.6)
left.set(xlabel="1 / Field (1/T)", ylabel="Signal (arb. units)", title="input: signal vs 1/B")
right.plot(spectrum.frequency, spectrum.amplitude, color="C0")
right.set(xlabel="Frequency (T)", ylabel="Amplitude (arb. units)", title="output: qz.fft", xlim=(0, 800))
