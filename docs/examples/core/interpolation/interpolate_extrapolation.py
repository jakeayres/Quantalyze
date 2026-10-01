from _data import cooldown

# --8<-- [start:example]
import quantalyze as qz

print(cooldown["temperature"].min(), cooldown["temperature"].max())  # the measured range

# 0 K and 400 K are outside the measured range, so they come back as NaN:
print(qz.interpolate(cooldown, "temperature", onto=[0, 150, 400]))

# extrapolate=True extends a straight line from the nearest points instead:
print(qz.interpolate(cooldown, "temperature", onto=[0, 150, 400], extrapolate=True))
# --8<-- [end:example]
