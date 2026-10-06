from _data import d, data, rate, surface

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann_fitting as bzf

result = bzf.fit_transport(
    data, surface, rate=rate, layer_spacing=d, extrapolate=True,
    p0=[{"gamma0": 1e12, "gamma1": 1e12}, {"gamma0": 1e13, "gamma1": 1e11}],  # two starting points
    resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error",
)
print(result.summary().to_string(float_format="{:.5g}".format))
print(f"\nχ² = {result.chi_squared:.1f} for ν = {result.degrees_of_freedom}: χ²/ν = {result.reduced_chi_squared:.2f}"
      f" (absolute errors: {result.absolute})")
print("\ncorrelation:\n" + result.correlation.to_string(float_format="{:+.3f}".format))
print("singular values:", np.array2string(result.singular_values, precision=3))
print("\nstarts:\n" + result.starts.to_string(float_format="{:.5g}".format))
print("\nfitted surface:\n" + result.surface[["kx", "ky", "phi", "tau"]].head(3).to_string(float_format="{:.4g}".format))
print("\nmodel at 10, 30 and 60 T:\n" + result.evaluate([10, 30, 60]).to_string(float_format="{:.5g}".format))
# --8<-- [end:example]
