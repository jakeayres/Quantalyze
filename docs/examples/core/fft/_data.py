import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=6)

# Quantum oscillations from two Fermi-surface pockets, at 150 T and 420 T, after the
# smooth background has been subtracted. They are periodic in 1/B, not in B.
field = np.linspace(5, 14, 4000)
signal = (
    1.0 * np.exp(-20 / field) * np.cos(2 * np.pi * 150 / field)
    + 0.5 * np.exp(-30 / field) * np.cos(2 * np.pi * 420 / field)
    + 0.02 * rng.standard_normal(field.size)
)
oscillations = pd.DataFrame({"field": field, "inverse_field": 1 / field, "signal": signal})
