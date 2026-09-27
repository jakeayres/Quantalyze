# --8<-- [start:example]
import quantalyze as qz

c = qz.constants

print(c.HBAR)  # J s

# Thermal energy at 4.2 K, in meV
print(c.KB * 4.2 / c.E * 1000)

# Onsager relation: the extremal Fermi-surface area behind a 150 T oscillation, in Å⁻²
F = 150
area = 2 * c.PI * c.E * F / c.HBAR  # m⁻²
print(area * 1e-20)
# --8<-- [end:example]
