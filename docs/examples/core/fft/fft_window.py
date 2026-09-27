from _data import oscillations

# --8<-- [start:example]
import quantalyze as qz

plain = qz.fft(oscillations, "inverse_field", "signal")
hann = qz.fft(oscillations, "inverse_field", "signal", window=qz.Window.HANN)
kaiser = qz.fft(oscillations, "inverse_field", "signal", window=qz.Window.KAISER, beta=8)
# --8<-- [end:example]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(plain.frequency, plain.amplitude / plain.amplitude.max(), color="0.6", label="no window")
ax.semilogy(hann.frequency, hann.amplitude / hann.amplitude.max(), color="C0", label="Window.HANN")
ax.semilogy(kaiser.frequency, kaiser.amplitude / kaiser.amplitude.max(), color="C1", label="Window.KAISER, beta=8")
ax.set(xlabel="Frequency (T)", ylabel="Amplitude (normalised)", xlim=(0, 800), ylim=(1e-4, 2))
ax.legend(loc="upper right")
