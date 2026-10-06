"""beta.boltzmann_fitting: the forward model, parameter recovery, uncertainties and diagnostics."""
import warnings

import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

D = 1e-9
MASS = 2 * ELECTRON_MASS
FIELDS = np.linspace(0, 40, 21)  # ω_cτ up to about 3.5 at Γ = 1e12 s⁻¹


def fourfold(n=128):
    surface = bz.generators.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=MASS, tau=1.0)
    surface["phi"] = bzf.polar_angle(surface)
    return surface


def rate4(phi, gamma0, a4):
    return gamma0 * (1 + a4 * np.cos(4 * phi))


TRUTH = {"gamma0": 1e12, "a4": 0.4}


def noisy(data, seed=0, relative=2e-3):
    """Data with Gaussian noise and its error columns."""
    rng = np.random.default_rng(seed)
    data = data.copy()
    data["resistivity_error"] = relative * data["resistivity"]
    data["hall_resistivity_error"] = relative * np.max(np.abs(data["hall_resistivity"]))
    for column in ("resistivity", "hall_resistivity"):
        data[column] += data[f"{column}_error"] * rng.standard_normal(len(data))
    return data


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def test_polar_angle():
    df = pd.DataFrame({"kx": [1.0, 0.0, -1.0, 0.0, 3.0], "ky": [0.0, 1.0, 0.0, -1.0, 1.0]})
    phi = bzf.polar_angle(df)
    np.testing.assert_allclose(phi[:4], [0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-15)
    assert phi.name == "phi"
    assert bzf.polar_angle(df, center=(2.0, 0.0)).iloc[4] == pytest.approx(np.pi / 4)


def test_transport_is_the_drude_model_on_a_circle():
    surface = bz.generators.circle(512, k_fermi=7e9, mass=MASS, tau=1.0)
    model = bzf.transport(surface, FIELDS, rate=lambda gamma: gamma, parameters={"gamma": 2e12}, layer_spacing=D)
    n = bz.carrier_density(surface, layer_spacing=D)
    rho0 = MASS * 2e12 / (n * ELEMENTARY_CHARGE**2)
    print(f"max rel. error: rho_xx {np.max(np.abs(model.resistivity / rho0 - 1)):.1e}, "
          f"hall {np.max(np.abs(model.hall_resistivity[1:] * n * ELEMENTARY_CHARGE / -FIELDS[1:] - 1)):.1e}")
    np.testing.assert_allclose(model.resistivity, rho0, rtol=1e-4)
    np.testing.assert_allclose(model.hall_resistivity, -FIELDS / (n * ELEMENTARY_CHARGE), rtol=1e-4,
                               atol=1e-12 * rho0)  # 0 at B = 0, to rounding
    assert list(model.columns) == ["field", "resistivity", "hall_resistivity"]


def test_transport_matches_conductivity():
    surface = fourfold()
    model = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    direct = bz.resistivity(bz.conductivity(bzf.build_surface(surface, TRUTH, rate=rate4), FIELDS, layer_spacing=D))
    np.testing.assert_allclose(model.resistivity, direct.rho_xx, rtol=1e-12)
    np.testing.assert_allclose(model.hall_resistivity[1:],
                               bz.hall_coefficient(bz.conductivity(bzf.build_surface(surface, TRUTH, rate=rate4),
                                                                   FIELDS, layer_spacing=D))[1:] * FIELDS[1:],
                               rtol=1e-12)


def test_build_surface_sets_tau_from_rate():
    surface = fourfold()
    built = bzf.build_surface(surface, TRUTH, rate=rate4)
    np.testing.assert_array_equal(built.tau, 1 / rate4(surface.phi.to_numpy(), **TRUTH))
    assert surface.tau.eq(1.0).all()  # the input is not changed

    def rate(a, b, c):  # no column arguments: an isotropic rate from three parameters
        return a + b + c

    np.testing.assert_allclose(bzf.build_surface(surface, {"a": 1e12, "b": 2e12, "c": 3e12}, rate=rate).tau, 1 / 6e12)


def test_recovers_tau_parameters_from_exact_data():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    result = bzf.fit_transport(data, surface, rate=rate4, p0={"gamma0": 3e11, "a4": 0.1},
                               bounds={"a4": (-0.95, 0.95)}, layer_spacing=D)
    print(result)
    assert result.names == ["gamma0", "a4"]
    assert result["gamma0"] == pytest.approx(1e12, rel=1e-7)
    assert result["a4"] == pytest.approx(0.4, rel=1e-7)
    assert not result.absolute


def test_noisy_fit_has_calibrated_errors():
    surface = fourfold()
    data = noisy(bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D))
    result = bzf.fit_transport(data, surface, rate=rate4, p0={"gamma0": 3e11, "a4": 0.1},
                               bounds={"a4": (-0.95, 0.95)}, layer_spacing=D,
                               resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
    pulls = {n: (result[n] - TRUTH[n]) / result.errors[n] for n in TRUTH}
    print(f"chi2/dof = {result.reduced_chi_squared:.3f}, pulls {pulls}, errors {result.errors}")
    assert result.absolute
    assert result.degrees_of_freedom == 2 * FIELDS.size - 2
    assert 0.4 < result.reduced_chi_squared < 1.8
    assert all(abs(p) < 4 for p in pulls.values())
    assert np.all(np.isfinite(result.covariance.to_numpy()))
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert result.singular_values[0] == 1.0 and 0 < result.singular_values[-1] <= 1


def test_covariance_without_errors_is_scaled_by_reduced_chi_squared():
    surface = fourfold()
    data = noisy(bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D))
    options = dict(rate=rate4, p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (-0.95, 0.95)}, layer_spacing=D)
    relative = bzf.fit_transport(data, surface, **options)
    rms = {c: float(np.sqrt(np.mean(data[c] ** 2))) for c in ("resistivity", "hall_resistivity")}
    weighted = bzf.fit_transport(data, surface, resistivity_error=rms["resistivity"],
                                 hall_resistivity_error=rms["hall_resistivity"], **options)
    assert weighted.absolute and not relative.absolute
    for n in TRUTH:
        assert relative[n] == pytest.approx(weighted[n], rel=1e-12)
    np.testing.assert_allclose(relative.covariance, weighted.covariance * weighted.reduced_chi_squared, rtol=1e-10)


def test_fits_the_fermi_surface_and_the_rate_together():
    a = 3.87e-10
    t, tp = bz.units.ev_to_joule(0.25), bz.units.ev_to_joule(-0.0625)

    def surface(mu):
        return bz.generators.tight_binding(128, tau=1.0, lattice_constant=a, hopping=t, next_hopping=tp,
                                           chemical_potential=mu, center=(np.pi / a, np.pi / a))

    def rate(gamma):
        return gamma

    truth = {"mu": bz.units.ev_to_joule(-0.1), "gamma": 5e12}
    data = bzf.transport(surface, np.linspace(0, 60, 25), rate=rate, parameters=truth, layer_spacing=D)
    result = bzf.fit_transport(data, surface, rate=rate, layer_spacing=D,
                               p0={"mu": bz.units.ev_to_joule(-0.05), "gamma": 1e12})
    print(result)
    assert result.names == ["mu", "gamma"]
    assert result["mu"] == pytest.approx(truth["mu"], rel=1e-6)
    assert result["gamma"] == pytest.approx(truth["gamma"], rel=1e-6)
    assert isinstance(result.surface, pd.DataFrame)
    np.testing.assert_allclose(result.surface.tau, 1 / result["gamma"])


def test_several_pockets_with_one_rate_each():
    electrons = bz.generators.circle(128, k_fermi=7e9, mass=MASS, tau=1.0)
    holes = bz.generators.circle(128, k_fermi=5e9, mass=3 * ELECTRON_MASS, tau=1.0, carrier="hole")
    rates = [lambda gamma_e: gamma_e, lambda gamma_h: gamma_h]
    truth = {"gamma_e": 1e12, "gamma_h": 3e12}
    data = bzf.transport([electrons, holes], FIELDS, rate=rates, parameters=truth, layer_spacing=D)
    result = bzf.fit_transport(data, [electrons, holes], rate=rates, layer_spacing=D,
                               p0={"gamma_e": 2e12, "gamma_h": 2e12})
    assert result["gamma_e"] == pytest.approx(1e12, rel=1e-6)
    assert result["gamma_h"] == pytest.approx(3e12, rel=1e-6)
    assert isinstance(result.surface, list) and len(result.surface) == 2


def test_fixed_defaults_bounds_and_priors():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    common = dict(rate=rate4, layer_spacing=D)

    held = bzf.fit_transport(data, surface, p0={"gamma0": 3e11}, fixed={"a4": 0.4}, **common)
    assert held.names == ["gamma0"] and held["a4"] == 0.4 and held["gamma0"] == pytest.approx(1e12, rel=1e-7)
    assert held.summary().loc["a4", "fitted"] == False  # noqa: E712
    assert np.isnan(held.summary().loc["a4", "error"])

    def with_default(phi, gamma0, a4=0.4):
        return rate4(phi, gamma0, a4)

    defaulted = bzf.fit_transport(data, surface, rate=with_default, p0={"gamma0": 3e11}, layer_spacing=D)
    assert defaulted.names == ["gamma0"] and defaulted["a4"] == 0.4

    bounded = quiet(bzf.fit_transport, data, surface, p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (0.0, 0.2)},
                    **common)
    assert bounded["a4"] == pytest.approx(0.2, abs=1e-6)

    pinned = bzf.fit_transport(data, surface, p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (-0.95, 0.95)},
                               priors={"a4": (0.1, 1e-9)}, **common)
    assert pinned["a4"] == pytest.approx(0.1, abs=1e-8)
    assert pinned.degrees_of_freedom == 2 * FIELDS.size + 1 - 2


def test_one_channel_only():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    hall_only = bzf.fit_transport(data, surface, rate=rate4, resistivity=None, p0={"gamma0": 3e11, "a4": 0.1},
                                  bounds={"a4": (-0.95, 0.95)}, layer_spacing=D,
                                  least_squares_options={"ftol": 1e-14, "xtol": 1e-14, "gtol": 1e-14})
    assert hall_only.degrees_of_freedom == FIELDS.size - 2
    assert hall_only["a4"] == pytest.approx(0.4, rel=1e-5)


def test_nan_rows_are_skipped_per_channel():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    data.loc[3, "resistivity"] = np.nan
    data.loc[5, "hall_resistivity"] = np.nan
    result = bzf.fit_transport(data, surface, rate=rate4, p0={"gamma0": 3e11, "a4": 0.1},
                               bounds={"a4": (-0.95, 0.95)}, layer_spacing=D)
    assert result.degrees_of_freedom == 2 * FIELDS.size - 2 - 2
    assert result["a4"] == pytest.approx(0.4, rel=1e-6)


def test_evaluate_matches_transport():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    result = bzf.fit_transport(data, surface, rate=rate4, p0={"gamma0": 3e11, "a4": 0.1},
                               bounds={"a4": (-0.95, 0.95)}, layer_spacing=D)
    fields = np.linspace(-50, 50, 11)
    expected = bzf.transport(surface, fields, rate=rate4, parameters=result.parameters, layer_spacing=D)
    pd.testing.assert_frame_equal(result.evaluate(fields), expected)
    assert "χ²" in repr(result)


def test_exactly_degenerate_parameters_warn():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=lambda a, b: a + b, parameters={"a": 4e11, "b": 6e11},
                         layer_spacing=D)
    with pytest.warns(UserWarning, match="do not determine every parameter|correlated"):
        bzf.fit_transport(data, surface, rate=lambda a, b: a + b, p0={"a": 1e11, "b": 1e11}, layer_spacing=D)


def test_mirror_images_of_tau_are_reported_as_different_solutions():
    # On a contour with a mirror plane, τ(φ) and τ(−φ) give identical ρ_xx and Hall
    # resistivity, so the sign of a sin4φ term cannot be found.
    surface = fourfold()

    def chiral(phi, gamma0, b4):
        return gamma0 * (1 + 0.3 * np.cos(4 * phi) + b4 * np.sin(4 * phi))

    truth = {"gamma0": 1e12, "b4": 0.3}
    data = noisy(bzf.transport(surface, FIELDS, rate=chiral, parameters=truth, layer_spacing=D))
    with pytest.warns(UserWarning, match="different solution"):
        result = bzf.fit_transport(
            data, surface, rate=chiral, layer_spacing=D, bounds={"b4": (-0.6, 0.6)},
            p0=[{"gamma0": 8e11, "b4": 0.2}, {"gamma0": 8e11, "b4": -0.2}],
            resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
    print(result.starts)
    assert len(result.starts) == 2
    np.testing.assert_allclose(sorted(result.starts.b4), [-result.starts.b4.abs().max(), result.starts.b4.abs().max()],
                               rtol=1e-4)
    assert result.starts.chi_squared.iloc[0] == pytest.approx(result.starts.chi_squared.iloc[1], rel=1e-6)


def test_several_starts_find_the_better_minimum():
    # On a pocket this close to a circle, rotating the anisotropy by 45° (a4 → −a4) changes
    # ρ by under 1%: a start at a4 = 0 can slide into the wrong, shallower minimum.
    surface = fourfold()
    data = noisy(bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D))
    errors = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")
    result = bzf.fit_transport(data, surface, rate=rate4, bounds={"a4": (-0.95, 0.95)}, layer_spacing=D,
                               p0=[{"gamma0": 3e11, "a4": -0.2}, {"gamma0": 3e11, "a4": 0.2}], **errors)
    print(result.starts)
    assert result.starts.a4.iloc[0] < 0 < result.starts.a4.iloc[1]
    assert result.starts.chi_squared.iloc[0] > 3 * result.starts.chi_squared.iloc[1]
    assert result["a4"] == pytest.approx(0.4, abs=4 * result.errors["a4"])
    assert result.chi_squared == result.starts.chi_squared.min()


def test_steps_into_an_invalid_model_are_turned_back():
    # Unbounded, the optimiser may try a4 beyond ±1, where the rate is negative.
    surface = fourfold(512)  # resolves the steep τ(φ) of a4 = 0.9
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters={"gamma0": 1e12, "a4": 0.9}, layer_spacing=D)
    result = quiet(bzf.fit_transport, data, surface, rate=rate4, p0={"gamma0": 3e11, "a4": 0.1}, layer_spacing=D)
    assert result["a4"] == pytest.approx(0.9, rel=1e-5)


def test_clear_errors():
    surface = fourfold()
    data = bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    fit = bzf.fit_transport
    with pytest.raises(ValueError, match="no value for parameter 'a4'"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12}, layer_spacing=D)
    with pytest.raises(ValueError, match="unknown parameter"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0, "a8": 0.0}, layer_spacing=D)
    with pytest.raises(ValueError, match="unknown parameter"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0}, bounds={"a5": (0, 1)}, layer_spacing=D)
    with pytest.raises(ValueError, match="both fixed and given a starting value"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0}, fixed={"a4": 0.1}, layer_spacing=D)
    with pytest.raises(ValueError, match="outside the bounds"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0}, bounds={"a4": (0.1, 0.5)}, layer_spacing=D)
    with pytest.raises(ValueError, match="finite and positive"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 2.0}, layer_spacing=D)
    with pytest.raises(ValueError, match="every fitted channel or for none"):
        fit(data, surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0}, resistivity_error=1e-9, layer_spacing=D)
    with pytest.raises(ValueError, match="nothing to fit"):
        fit(data, surface, rate=rate4, p0={}, fixed={"gamma0": 1e12, "a4": 0.0}, layer_spacing=D)
    with pytest.raises(ValueError, match="surface needs a value for 'mu'"):
        fit(data, lambda mu: surface, rate=rate4, p0={"gamma0": 1e12, "a4": 0.0}, layer_spacing=D)
    with pytest.raises(TypeError, match="named arguments"):
        fit(data, surface, rate=lambda *p: p[0], p0={"gamma0": 1e12}, layer_spacing=D)
    with pytest.raises(TypeError, match="return only the sheets"):
        fit(data, lambda g: ([surface], (1.0, 0.0)), rate=lambda gamma0: gamma0, p0={"gamma0": 1e12, "g": 1.0},
            layer_spacing=D)
    with pytest.raises(ValueError, match="column of some contours but not others"):
        bzf.transport([surface, surface.drop(columns="phi")], FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D)
    with pytest.raises(ValueError, match="needs a 'tau' column"):
        bzf.transport(surface.drop(columns="tau"), FIELDS, layer_spacing=D)
