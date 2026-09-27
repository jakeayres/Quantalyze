import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=4)


def measure_sweep(start, stop):
    """One field sweep from `start` to `stop` tesla, measuring rxx and rxy together."""
    field = np.sort(rng.uniform(-9, 9, 1500))
    if start > stop:
        field = field[::-1]
    rxx = 1 + 0.004 * field**2  # even in field
    rxy = 0.06 * field          # odd in field
    # Misaligned contacts mix a little of each signal into the other.
    return pd.DataFrame({
        "field": field,
        "rxx": rxx + 0.1 * rxy + 0.002 * rng.standard_normal(field.size),
        "rxy": rxy + 0.05 * rxx + 0.002 * rng.standard_normal(field.size),
    })


up = measure_sweep(-9, 9)    # -9 T -> +9 T
down = measure_sweep(9, -9)  # +9 T -> -9 T
