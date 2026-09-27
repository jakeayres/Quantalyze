from _data import up, down

# --8<-- [start:example]
import quantalyze as qz

grid = dict(minimum=-9, maximum=9, step=0.1)  # use the same grid for both
rxx = qz.symmetrize([up, down], "field", "rxx", **grid)
rxy = qz.antisymmetrize([up, down], "field", "rxy", **grid)

both = rxx.merge(rxy, on="field")
print(both.head())
# --8<-- [end:example]
