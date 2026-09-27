from _data import cooldown

# --8<-- [start:example]
import numpy as np
import quantalyze as qz

print(cooldown["temperature"].min(), cooldown["temperature"].max())  # the measured range

# Asking for 0 K and 400 K silently extends a straight line from the nearest points:
print(qz.interpolate(cooldown, "temperature", onto=[0, 400]))

# To avoid that, keep the new x values inside the measured range:
grid = np.arange(0, 400, 10)
inside = grid[(grid >= cooldown["temperature"].min()) & (grid <= cooldown["temperature"].max())]
safe = qz.interpolate(cooldown, "temperature", onto=inside)
print(safe["temperature"].min(), safe["temperature"].max())
# --8<-- [end:example]
