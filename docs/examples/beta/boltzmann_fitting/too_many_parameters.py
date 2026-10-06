from _data import d, data, surface

# --8<-- [start:example]
import textwrap
import warnings
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf


def rate3(phi, gamma0, gamma1, gamma2):  # one more harmonic than the data were made with
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2 + gamma2 * np.cos(2 * phi) ** 4


errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = bzf.fit_transport(
        data, surface, rate=rate3, layer_spacing=d, extrapolate=True, **errors,
        p0=[{"gamma0": 1e12, "gamma1": 1e12, "gamma2": 1e12}, {"gamma0": 5e12, "gamma1": 1e12, "gamma2": 5e12}],
        bounds={"gamma0": (0, None), "gamma1": (-2e13, 2e13), "gamma2": (-2e13, 2e13)},
    )
print(result.summary().to_string(float_format="{:.3g}".format))
print(f"\nχ²/ν = {result.reduced_chi_squared:.2f}")
print("singular values:", np.array2string(result.singular_values, precision=4))
for warning in caught:
    print(textwrap.fill(f"warning: {warning.message}", 88))
# --8<-- [end:example]
