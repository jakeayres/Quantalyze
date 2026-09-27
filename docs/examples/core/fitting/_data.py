import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=3)

# Resistivity of a metal: a Fermi-liquid T² term at low temperature, with a
# phonon contribution (~T⁵) that takes over above ~20 K.
temperature = np.linspace(2, 40, 200)
resistivity = (
    2.0 + 0.002 * temperature**2               # rho0 + A T²
    + 1e-8 * temperature**5                    # phonons
    + 0.005 * rng.standard_normal(temperature.size)
)
rt = pd.DataFrame({"temperature": temperature, "resistivity": resistivity})  # µΩ cm vs K

# Heat capacity through a phase transition: a sharp peak at 10 K on a sloping background.
temperature = np.linspace(5, 15, 200)
heat_capacity = (
    0.5 / (1 + ((temperature - 10) / 0.25) ** 2)
    + 0.02 * temperature
    + 0.02 * rng.standard_normal(temperature.size)
)
transition = pd.DataFrame({"temperature": temperature, "heat_capacity": heat_capacity})
