from _data import band, d, data, rate, surface, true_mu, true_rate

# --8<-- [start:example]
import warnings
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf

errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
thick = data.assign(**{c: 1.1 * data[c] for c in data.columns if c != "field"})  # thickness 10% too large
faster = surface.assign(vx=1.1 * surface["vx"], vy=1.1 * surface["vy"])        # v_F 10% too high
start = {"gamma0": 1e12, "gamma1": 1e12}
cases = {
    "correct": (data, surface, rate, start),
    "thickness +10%": (thick, surface, rate, start),
    "thickness +10%, μ fitted": (thick, band, rate, {"mu": true_mu, **start}),
    "Fermi velocity +10%": (data, faster, rate, start),
    "isotropic rate": (data, surface, lambda gamma0: gamma0, {"gamma0": 1e12}),
}
print(f"{'':25s}  χ²/ν   Γ0/true  Γ1/true  μ − true")
for label, (measured, fermi_surface, model, p0) in cases.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bzf.fit_transport(measured, fermi_surface, rate=model, layer_spacing=d, p0=p0, extrapolate=True, **errors)
    gamma1 = f"{r['gamma1'] / true_rate['gamma1']:.3f}" if "gamma1" in r.parameters else "  —  "
    mu = f"{bz.units.joule_to_ev(r['mu'] - true_mu) * 1e3:+.1f} meV" if "mu" in r.names else ""
    print(f"{label:25s} {r.reduced_chi_squared:6.1f}   {r['gamma0'] / true_rate['gamma0']:.3f}    {gamma1}    {mu}")
# --8<-- [end:example]
