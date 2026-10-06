from _data import band, d, rate, true_mu, true_rate

# --8<-- [start:example]
import numpy as np
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf

# Exact data: the model itself at the true parameters, with no noise at all.
truth = {"mu": true_mu, **true_rate}
exact = bzf.transport(band, np.linspace(0, 60, 61), rate=rate, parameters=truth, layer_spacing=d)

# With no noise there is nothing to stop at, so converge fully; start far from the truth.
options = dict(rate=rate, layer_spacing=d, bounds={"gamma0": (0, None), "gamma1": (0, None)},
               least_squares_options={"ftol": 1e-15, "xtol": 1e-15, "gtol": 1e-15})
rate_only = bzf.fit_transport(exact, band(true_mu), p0={"gamma0": 1e12, "gamma1": 1e12}, **options)
surface_too = bzf.fit_transport(exact, band, p0={"mu": bz.units.ev_to_joule(-0.05), "gamma0": 1e12,
                                                 "gamma1": 1e12}, **options)

for label, result in (("rate only", rate_only), ("surface and rate", surface_too)):
    errors = {name: abs(result[name] / truth[name] - 1) for name in result.names}
    print(f"{label}: largest relative error below 1e-8: {max(errors.values()) < 1e-8}")
# --8<-- [end:example]
