import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=2)

# Magnetoresistance R = 1 + 0.004 B^2 on an evenly spaced field grid, so dR/dB = 0.008 B exactly.
field = np.linspace(0, 9, 91)
clean = pd.DataFrame({"field": field, "resistance": 1 + 0.004 * field**2})

# The same measurement with a little noise, as it would come off the instrument.
noisy = clean.assign(resistance=clean["resistance"] + 0.001 * rng.standard_normal(field.size))
