from _data import up, down

# --8<-- [start:example]
import quantalyze as qz

# The grid always runs from -maximum to +maximum, whatever minimum is...
result = qz.symmetrize([up, down], "field", "rxx", minimum=0, maximum=1, step=0.5)
print(result)
# ...and minimum=0 threw away all the negative-field data, so only x = 0 (which is
# its own mirror image) has a partner to be averaged with. Everything else is NaN.
# --8<-- [end:example]
