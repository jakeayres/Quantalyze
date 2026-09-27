"""M8: open orbits (period=(G_x, G_y)).

An open orbit is one period of a sheet crossing the Brillouin zone; its last segment
ends at the first node shifted by G. Its drift is physical and is never removed. A
warped sheet must converge to the brute-force reference as O(N⁻²).
"""
import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.beta.boltzmann._response import conductivity_tensor

A = 3.87e-10
G = 2 * np.pi / A
K0 = 5e9
V0 = 2e5
TAU = 1e-13
D = 1e-9
WARPING = 0.1 * K0
PERIOD = (0.0, G)


def arrays(df):
    return df["kx"], df["ky"], df["vx"], df["vy"], df["tau"]


def sheets(n, warping=WARPING):
    return gen.open_sheets(n, k0=K0, velocity=V0, tau=TAU, period=G, warping=warping)


def orbit_x(df, field):
    """ω_cτ over one period: 2π|B| / Σ γ_n s_n."""
    c = prepare_contour(*arrays(df), period=PERIOD)
    return 2 * np.pi * abs(field) / np.sum(c.gamma * c.s)


def tensor(sigma):
    return sigma[["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]].to_numpy().reshape(-1, 2, 2)


def test_drift_is_kept_and_cannot_be_removed():
    df = sheets(128)[0]
    prepared = prepare_contour(*arrays(df), period=PERIOD)
    assert np.array_equal(prepared.drift, [0.0, 0.0])
    assert prepare_contour(*arrays(df), period=PERIOD, remove_drift=False).s.size == 128
    with pytest.raises(ValueError, match="physical"):
        prepare_contour(*arrays(df), period=PERIOD, remove_drift=True)
    with pytest.raises(ValueError, match="physical"):
        bz.conductivity(df, 1.0, layer_spacing=D, period=PERIOD, remove_drift=True)


def test_rejects_a_period_that_does_not_join_the_ends():
    df = sheets(128)[0]
    with pytest.raises(ValueError, match="does not join"):
        prepare_contour(*arrays(df), period=(0.0, 2 * G))
    with pytest.raises(ValueError, match="does not join"):
        prepare_contour(*arrays(df.iloc[:64]), period=PERIOD)  # half a period
    with pytest.raises(ValueError, match="period"):
        prepare_contour(*arrays(df), period=(0.0, 0.0))


def test_order_start_and_sign_of_g_do_not_matter():
    """Reversed input, either sign of G, a repeated end point, and a different starting
    node (with the moved nodes shifted by G so the sheet stays continuous) give the same σ."""
    df = sheets(256)[0]
    fields = np.array([-3.0, 0.0, 0.3, 3.0, 30.0])
    base = conductivity_tensor(*arrays(df), fields, layer_spacing=D, period=PERIOD)
    scale = np.max(np.abs(base))
    rolled = df.copy()
    rolled.loc[:36, "ky"] += G  # nodes 0–36 move to the end of the period
    rolled = rolled.iloc[np.roll(np.arange(len(df)), -37)]
    closing = df.iloc[[0]].assign(ky=df.ky.iloc[0] + G)
    variants = {
        "reversed": (df.iloc[::-1], PERIOD),
        "minus G": (df, (0.0, -G)),
        "reversed, minus G": (df.iloc[::-1], (0.0, -G)),
        "repeated end point": (pd.concat([df, closing]), PERIOD),
        "rotated start": (rolled, PERIOD),
    }
    for label, (variant, period) in variants.items():
        error = np.max(np.abs(conductivity_tensor(*arrays(variant), fields, layer_spacing=D, period=period) - base))
        print(f"{label}: {error / scale:.1e}")
        assert error <= 1e-12 * scale


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_finite_for_twelve_decades_of_field(backend):
    plus, minus = sheets(256)
    per_tesla = orbit_x(plus, 1.0)
    x = np.geomspace(1e-6, 1e6, 25)
    fields = np.concatenate([x, -x, [0.0]]) / per_tesla
    result = tensor(bz.conductivity([plus, minus], fields, layer_spacing=D, period=PERIOD, backend=backend))
    assert np.all(np.isfinite(result)) and np.all(result[:, 0, 0] > 0)


def test_open_orbit_magnetoresistance_does_not_saturate():
    """Sheets normal to x carry current along x at any field, so σ_xx saturates at a finite
    value while σ_yy falls as 1/B²: ρ_yy grows as B² without limit."""
    plus, minus = sheets(512)
    per_tesla = orbit_x(plus, 1.0)
    fields = np.array([0.0, 10.0, 20.0]) / per_tesla
    s = bz.conductivity([plus, minus], fields, layer_spacing=D, period=PERIOD)
    rho = bz.resistivity(s)
    growth = (rho.rho_yy.iloc[2] - rho.rho_yy.iloc[0]) / (rho.rho_yy.iloc[1] - rho.rho_yy.iloc[0])
    print(f"sigma_xx(20)/sigma_xx(10) = {s.sigma_xx.iloc[2] / s.sigma_xx.iloc[1]:.4f}; "
          f"rho_yy - rho_yy(0) grows by {growth:.3f}x from x = 10 to 20")
    assert abs(s.sigma_xx.iloc[2] / s.sigma_xx.iloc[1] - 1) < 1e-2
    assert 3.9 < growth < 4.1


@pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
def test_warped_sheets_converge_to_the_reference_as_n_to_the_minus_two(x):
    """Both warped sheets, N = 64…1024 against the brute-force reference: the fitted
    log–log slope of the error must lie in [−2.1, −1.9]."""
    field = x / orbit_x(sheets(1024)[0], 1.0)
    exact = sum(
        ref.sigma(ref.ParametricContour.open_sheet(k0=K0, velocity=V0, warping=WARPING, period=G, tau=TAU,
                                                   side=side), field, layer_spacing=D)[0]
        for side in (1.0, -1.0)
    )
    sizes = np.array([64, 128, 256, 512, 1024])
    errors = []
    for n in sizes:
        actual = tensor(bz.conductivity(sheets(n), field, layer_spacing=D, period=PERIOD))[0]
        errors.append(np.max(np.abs(actual - exact)) / np.max(np.abs(exact)))
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    print(f"x = {x}: errors " + ", ".join(f"{e:.2e}" for e in errors) + f"; slope {slope:.3f}")
    assert -2.1 <= slope <= -1.9
