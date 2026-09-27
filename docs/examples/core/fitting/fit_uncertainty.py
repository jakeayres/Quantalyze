from _data import rt

# --8<-- [start:example]
import numpy as np
import quantalyze as qz


def fermi_liquid(temperature, rho0, A):
    return rho0 + A * temperature**2


result = qz.fit(fermi_liquid, rt, "temperature", "resistivity", x_max=20)

errors = np.sqrt(np.diag(result.covariance))  # one standard deviation per parameter

for name, value, error in zip(["rho0", "A"], result.parameters, errors):
    print(f"{name} = {value:.4g} ± {error:.2g}")
# --8<-- [end:example]
