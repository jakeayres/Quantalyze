# --8<-- [start:example]
import pandas as pd
import quantalyze as qz

df = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 1, 4, 9, 16]})  # y = x², so dy/dx = 2x

df["forward"] = qz.forward_difference(df, "x", "y")    # slope to the next point
df["backward"] = qz.backward_difference(df, "x", "y")  # slope from the previous point
df["central"] = qz.central_difference(df, "x", "y")    # average of the two
df["exact"] = 2 * df["x"]

print(df)
# --8<-- [end:example]
