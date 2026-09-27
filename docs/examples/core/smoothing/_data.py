import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=1)


def measure_sweep():
    """A 0-9 T magnetoresistance sweep, recorded at irregular field values."""
    field = np.sort(rng.uniform(0, 9, 3000))
    resistance = (
        1 + 0.004 * field**2                                            # background
        + 0.03 * np.exp(-10 / field) * np.cos(2 * np.pi * 150 / field)  # quantum oscillations
        + 0.003 * rng.standard_normal(field.size)                      # noise
    )
    temperature = 2 + 0.005 * rng.standard_normal(field.size)
    return pd.DataFrame({"field": field, "resistance": resistance, "temperature": temperature})


sweep = measure_sweep()   # one sweep
repeat = measure_sweep()  # a second sweep of the same sample


# Heat capacity through a phase transition at 10 K, evenly spaced in temperature.
temperature = np.arange(5, 15, 0.05)
heat_capacity = (
    0.02 * temperature                               # background
    + 0.5 / (1 + ((temperature - 10) / 0.25) ** 2)   # sharp peak at the transition
    + 0.02 * rng.standard_normal(temperature.size)   # noise
)
transition = pd.DataFrame({"temperature": temperature, "heat_capacity": heat_capacity})
