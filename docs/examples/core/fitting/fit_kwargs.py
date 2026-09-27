from _data import rt

# --8<-- [start:example]
import numpy as np
import quantalyze as qz


def fermi_liquid(temperature, rho0, A):
    return rho0 + A * temperature**2


result = qz.fit(
    fermi_liquid, rt, "temperature", "resistivity",
    x_min=5, x_max=20,                  # only rows with 5 <= temperature <= 20
    y_max=3,                            # ...and resistivity <= 3
    bounds=([0, 0], [np.inf, np.inf]),  # passed on to curve_fit
)
print(result.parameters)
# --8<-- [end:example]
