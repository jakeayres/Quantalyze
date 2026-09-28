from _data import rt

# --8<-- [start:example]
import numpy as np
import quantalyze as qz

result = qz.fit(lambda T, rho0, A: rho0 + A * T**2, rt, "temperature", "resistivity", x_max=20)

# A single value...
print(f"{result.evaluate(0):.6f}")  # the residual resistivity, extrapolated to T = 0

# ...an array...
print(result.evaluate(np.array([4.2, 10, 20])))

# ...or a whole column, e.g. to get the residuals.
rt["fit"] = result.evaluate(rt["temperature"])
rt["residual"] = rt["resistivity"] - rt["fit"]
print(rt.head())
# --8<-- [end:example]
