import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=7)


def measure_sweep(start, stop):
    """A Hall measurement on a 100 nm film with n = 1e21 cm^-3 and slightly misaligned contacts."""
    field = np.sort(rng.uniform(-9, 9, 1500))
    if start > stop:
        field = field[::-1]
    rxy = field / (1e27 * 1.602176634e-19 * 100e-9)  # B / (n e t)
    rxx = 0.5 * (1 + 0.004 * field**2)
    return pd.DataFrame({"field": field, "rxy": rxy + 0.1 * rxx + 0.003 * rng.standard_normal(field.size)})


up = measure_sweep(-9, 9)
down = measure_sweep(9, -9)
