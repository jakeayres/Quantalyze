import numpy as np
from _data import band, bz, corner, d, data, rate, true_mu, true_rate

# A table of Fermi wavevectors and speeds at 90 angles, as from ARPES (made here from the band).
_points = band(true_mu, n_points=90)
arpes = {"phi": _points["phi"].to_numpy(),                                   # rad, about the corner
         "k_fermi": 1e-10 * np.hypot(_points["kx"] - corner[0], _points["ky"] - corner[1]).to_numpy(),  # Å⁻¹
         "v_fermi": bz.units.meter_per_second_to_ev_angstrom(np.hypot(_points["vx"], _points["vy"]).to_numpy())}

# --8<-- [start:example]
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf


def surface_from_table(phi, k_fermi, v_fermi, center, n_points=512, carrier="hole"):
    """A smooth contour through measured k_F(φ) (Å⁻¹) and |v_F|(φ) (eV·Å) about a centre (m⁻¹)."""
    closed = np.append(phi, phi[0] + 2 * np.pi)  # periodic splines through the measured points
    k = CubicSpline(closed, np.append(k_fermi, k_fermi[0]), bc_type="periodic")
    v = CubicSpline(closed, np.append(v_fermi, v_fermi[0]), bc_type="periodic")
    angle = 2 * np.pi * np.arange(n_points) / n_points
    radius, slope = bz.units.per_angstrom_to_per_meter(k(angle)), bz.units.per_angstrom_to_per_meter(k(angle, 1))
    # The tangent of k(φ) = k_F(φ)(cos φ, sin φ), turned by −90°: the outward normal.
    tx = slope * np.cos(angle) - radius * np.sin(angle)
    ty = slope * np.sin(angle) + radius * np.cos(angle)
    nx, ny = ty / np.hypot(tx, ty), -tx / np.hypot(tx, ty)
    speed = bz.units.ev_angstrom_to_meter_per_second(v(angle)) * (-1 if carrier == "hole" else 1)
    return pd.DataFrame({"kx": center[0] + radius * np.cos(angle), "ky": center[1] + radius * np.sin(angle),
                         "vx": speed * nx, "vy": speed * ny, "tau": 1.0, "phi": angle})


surface = surface_from_table(arpes["phi"], arpes["k_fermi"], arpes["v_fermi"], center=corner)
result = bzf.fit_transport(data, surface, rate=rate, p0={"gamma0": 1e12, "gamma1": 1e12}, layer_spacing=d,
                           extrapolate=True, resistivity_error="resistivity_error",
                           hall_resistivity_error="hall_resistivity_error")
for name in result.names:
    print(f"{name}: {result[name] / 1e12:.4f} ± {result.errors[name] / 1e12:.4f} ×10¹² s⁻¹ "
          f"(truth {true_rate[name] / 1e12:.0f})")
# --8<-- [end:example]
