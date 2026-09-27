"""M5: the public API (bz.conductivity, resistivity, hall_coefficient, magnetoresistance,
carrier_density, conductivity_tensor).

Physics is checked in the known-results and sanity suites; this file checks the
interface: DataFrame and list input, column names, τ as a float, required
arguments, field signs and ordering, and the enclosed-area accuracy behind
carrier_density.
"""
import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import enclosed_area
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9
K_F = 7e9
TAU = 1e-13
FIELDS = np.array([5.0, -5.0, 0.0, 50.0, -0.5])
SIGMA = ["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]


def circle(**kwargs):
    kwargs.setdefault("tau", TAU)
    return gen.circle(512, k_fermi=K_F, mass=M, **kwargs)


def fourfold(n=512):
    return gen.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * M,
                     tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.3), carrier="hole")


def test_public_names():
    for name in ("conductivity", "resistivity", "hall_coefficient", "magnetoresistance", "carrier_density",
                 "conductivity_tensor", "FermiSurface", "generators", "scattering", "units"):
        assert name in bz.__all__ and hasattr(bz, name)


def test_conductivity_output_format():
    s = bz.conductivity(fourfold(), FIELDS, layer_spacing=D)
    assert list(s.columns) == ["field"] + SIGMA
    assert list(s.index) == list(range(FIELDS.size))
    np.testing.assert_array_equal(s["field"], FIELDS)  # rows follow the input order and sign
    assert all(s[c].dtype == np.float64 for c in s.columns)
    assert np.all(np.isfinite(s.to_numpy()))


def test_single_dataframe_equals_one_element_list_and_pockets_add():
    a, b = circle(), fourfold()
    single = bz.conductivity(a, FIELDS, layer_spacing=D)
    pd.testing.assert_frame_equal(single, bz.conductivity([a], FIELDS, layer_spacing=D))
    both = bz.conductivity([a, b], FIELDS, layer_spacing=D)
    summed = single[SIGMA].to_numpy() + bz.conductivity(b, FIELDS, layer_spacing=D)[SIGMA].to_numpy()
    np.testing.assert_allclose(both[SIGMA].to_numpy(), summed, rtol=1e-14)


def test_tau_as_a_float_matches_a_constant_column():
    df = circle()
    from_column = bz.conductivity(df, FIELDS, layer_spacing=D)
    from_float = bz.conductivity(df.drop(columns="tau"), FIELDS, layer_spacing=D, tau=TAU)
    pd.testing.assert_frame_equal(from_column, from_float)


def test_custom_column_names():
    df = fourfold()
    renamed = df.rename(columns={"kx": "k_x", "ky": "k_y", "vx": "v_x", "vy": "v_y", "tau": "lifetime"})
    expected = bz.conductivity(df, FIELDS, layer_spacing=D)
    actual = bz.conductivity(renamed, FIELDS, layer_spacing=D, kx="k_x", ky="k_y", vx="v_x", vy="v_y", tau="lifetime")
    pd.testing.assert_frame_equal(actual, expected)
    assert bz.carrier_density(renamed, "k_x", "k_y", "v_x", "v_y", layer_spacing=D) == pytest.approx(
        bz.carrier_density(df, layer_spacing=D), rel=1e-15)


def test_layer_spacing_is_required():
    df = circle()
    with pytest.raises(TypeError):
        bz.conductivity(df, 1.0)
    with pytest.raises(TypeError):
        bz.carrier_density(df)
    with pytest.raises(TypeError):
        bz.conductivity_tensor(df.kx, df.ky, df.vx, df.vy, df.tau, 1.0)


def test_scalar_field_and_field_sign():
    df = fourfold()
    one = bz.conductivity(df, 5.0, layer_spacing=D)
    assert len(one) == 1 and one["field"].iloc[0] == 5.0
    s = bz.conductivity(df, [5.0, -5.0], layer_spacing=D)
    positive, negative = s[SIGMA].to_numpy().reshape(2, 2, 2)
    np.testing.assert_array_equal(negative, positive.T)  # Onsager, symmetrised


def test_backends_agree_through_the_public_api():
    df = fourfold()
    python = bz.conductivity(df, FIELDS, layer_spacing=D, backend="python")[SIGMA].to_numpy()
    compiled = bz.conductivity(df, FIELDS, layer_spacing=D)[SIGMA].to_numpy()
    assert np.max(np.abs(compiled - python)) <= 1e-12 * np.max(np.abs(python))


def test_resistivity_inverts_sigma():
    s = bz.conductivity([circle(), fourfold()], FIELDS, layer_spacing=D)
    rho = bz.resistivity(s)
    assert list(rho.columns) == ["field", "rho_xx", "rho_xy", "rho_yx", "rho_yy"]
    product = rho[["rho_xx", "rho_xy", "rho_yx", "rho_yy"]].to_numpy().reshape(-1, 2, 2) @ \
        s[SIGMA].to_numpy().reshape(-1, 2, 2)
    np.testing.assert_allclose(product, np.broadcast_to(np.eye(2), product.shape), atol=1e-12)


def test_hall_coefficient_and_magnetoresistance_series():
    s = bz.conductivity(fourfold(), FIELDS, layer_spacing=D)
    r_h, mr = bz.hall_coefficient(s), bz.magnetoresistance(s)
    assert r_h.name == "hall_coefficient" and mr.name == "magnetoresistance"
    assert list(r_h.index) == list(s.index) == list(mr.index)
    assert np.isnan(r_h.iloc[2]) and np.all(np.isfinite(r_h.drop(index=2)))
    assert mr.iloc[2] == 0.0
    assert r_h.iloc[0] == pytest.approx(r_h.iloc[1], rel=1e-12)  # R_H is even in B
    with pytest.raises(ValueError, match="B = 0"):
        bz.magnetoresistance(s[s["field"] != 0])


def test_period_can_be_given_per_contour():
    """One (G_x, G_y) for every contour, or a list aligned with dfs (None for closed pockets)."""
    g = 2 * np.pi / 3.87e-10
    plus, minus = gen.open_sheets(128, k0=5e9, velocity=2e5, tau=TAU, period=g)
    both = bz.conductivity([plus, minus], FIELDS, layer_spacing=D, period=(0.0, g))
    aligned = bz.conductivity([plus, minus], FIELDS, layer_spacing=D, period=[(0.0, g), (0.0, g)])
    pd.testing.assert_frame_equal(both, aligned)
    mixed = bz.conductivity([plus, circle()], FIELDS, layer_spacing=D, period=[(0.0, g), None])
    separate = (bz.conductivity(plus, FIELDS, layer_spacing=D, period=(0.0, g))[SIGMA].to_numpy()
                + bz.conductivity(circle(), FIELDS, layer_spacing=D)[SIGMA].to_numpy())
    np.testing.assert_allclose(mixed[SIGMA].to_numpy(), separate, rtol=1e-14)
    with pytest.raises(ValueError):
        bz.conductivity([plus, minus], FIELDS, layer_spacing=D, period=[(0.0, g)])


def test_carrier_density():
    """n = g_s A/(4π²d); for a circle g_s k_F²/(4πd)."""
    n = bz.carrier_density(circle(), layer_spacing=D)
    expected = 2 * K_F**2 / (4 * np.pi * D)
    print(f"circle: n / (g k_F^2 / 4 pi d) - 1 = {n / expected - 1:.1e}")
    assert n == pytest.approx(expected, rel=1e-9)
    assert bz.carrier_density(circle(), layer_spacing=D, spin_degeneracy=1) == pytest.approx(n / 2, rel=1e-15)
    assert bz.carrier_density(circle(carrier="hole"), layer_spacing=D) == pytest.approx(n, rel=1e-15)
    with pytest.raises(TypeError):
        bz.carrier_density([circle()], layer_spacing=D)


def lopsided_nodes(phi):
    """Nodes, with exact velocities, on the pocket k_F(φ) = k₀(1 + 0.1 cos2φ + 0.05 sin3φ)."""
    r = K_F * (1 + 0.1 * np.cos(2 * phi) + 0.05 * np.sin(3 * phi))
    dr = K_F * (-0.2 * np.sin(2 * phi) + 0.15 * np.cos(3 * phi))
    vr, va = HBAR * r / M, -HBAR * dr / M
    return (r * np.cos(phi), r * np.sin(phi),
            vr * np.cos(phi) - va * np.sin(phi), vr * np.sin(phi) + va * np.cos(phi))


LOPSIDED_AREA = np.pi * K_F**2 * (1 + 0.1**2 / 2 + 0.05**2 / 2)


@pytest.mark.parametrize("jitter", [0.0, 0.45], ids=["even", "jittered"])
def test_enclosed_area_converges_as_n_to_the_minus_four(jitter):
    """The velocity-corrected area is O(N⁻⁴), also for irregularly spaced nodes; a polygon
    area is O(N⁻²)."""
    rng = np.random.default_rng(3)
    sizes = np.array([64, 128, 256, 512, 1024])
    errors, polygon = [], []
    for n in sizes:
        phi = np.sort((2 * np.pi * (np.arange(n) + rng.uniform(-jitter, jitter, n)) / n) % (2 * np.pi))
        kx, ky, vx, vy = lopsided_nodes(phi)
        errors.append(abs(enclosed_area(kx, ky, vx, vy) / LOPSIDED_AREA - 1))
        polygon.append(abs(0.5 * np.sum(kx * np.roll(ky, -1) - np.roll(kx, -1) * ky) / LOPSIDED_AREA - 1))
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    print("area errors " + ", ".join(f"{e:.1e}" for e in errors) + f"; slope {slope:.2f}; "
          f"polygon at N = 512: {polygon[3]:.1e}")
    assert slope < -3.7 and errors[3] < 1e-8


def test_enclosed_area_does_not_depend_on_direction_or_start():
    kx, ky, vx, vy = lopsided_nodes(np.linspace(0, 2 * np.pi, 300, endpoint=False))
    base = enclosed_area(kx, ky, vx, vy)
    assert enclosed_area(kx[::-1], ky[::-1], vx[::-1], vy[::-1]) == pytest.approx(base, rel=1e-14)
    assert enclosed_area(*(np.roll(a, 41) for a in (kx, ky, vx, vy))) == pytest.approx(base, rel=1e-14)
    assert enclosed_area(kx, ky, -vx, -vy) == pytest.approx(base, rel=1e-14)  # hole-like: v reversed
