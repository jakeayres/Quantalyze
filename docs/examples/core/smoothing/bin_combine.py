from _data import sweep, repeat

# --8<-- [start:example]
import quantalyze as qz

combined = qz.bin([sweep, repeat], "field", minimum=0, maximum=9, width=0.02)

print(f"{len(sweep)} + {len(repeat)} rows in, {len(combined)} rows out")
print(combined.head())
# --8<-- [end:example]
