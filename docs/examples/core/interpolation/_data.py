import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=5)


def resistance(temperature, transition):
    """A metal whose resistance jumps up below a phase transition."""
    return 0.5 + 0.004 * temperature + 0.4 / (1 + np.exp((temperature - transition) / 3))


# The transition is hysteretic: it happens at 145 K on cooling but 155 K on warming.
# Each run was logged at its own, irregular temperatures.
t = np.sort(rng.uniform(5, 300, 400))[::-1]
cooldown = pd.DataFrame({"temperature": t, "resistance": resistance(t, 145) + 0.003 * rng.standard_normal(t.size)})
t = np.sort(rng.uniform(5, 300, 250))
warmup = pd.DataFrame({"temperature": t, "resistance": resistance(t, 155) + 0.003 * rng.standard_normal(t.size)})

# A thermometer calibration table: only a few points, and very curved.
t = np.array([2, 4, 7, 10, 20, 40, 70, 100, 200, 300])
calibration = pd.DataFrame({"temperature": t, "sensor_resistance": 50 + 3000 / t**0.8})
