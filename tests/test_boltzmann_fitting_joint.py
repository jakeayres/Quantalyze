"""beta.boltzmann_fitting.fit_joint: several datasets fitted together, with shared and separate parameters.

The synthetic data are made with the single-dataset forward model `bzf.transport`, which
shares no code with the joint machinery beyond the model itself.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta import boltzmann_fitting as bzf
from quantalyze.core.constants import ELECTRON_MASS

# The split-rate model below has strongly correlated parameters by construction; tests that
# are about something else need not repeat its warning.
pytestmark = pytest.mark.filterwarnings("ignore:.*% correlated:UserWarning")

D = 1e-9
MASS = 2 * ELECTRON_MASS
FIELDS = np.linspace(0, 40, 21)  # ω_cτ up to about 3.5 at Γ = 1e12 s⁻¹
ERRORS = dict(resistivity_error="resistivity_error", hall_resistivity_error="hall_resistivity_error")


def fourfold(n=128, k4=0.25e9, carrier="electron"):
    surface = bz.generators.polar(n, k_fermi=lambda p: 7.35e9 - k4 * np.cos(4 * p), mass=MASS, tau=1.0,
                                  carrier=carrier)
    surface["phi"] = bzf.polar_angle(surface)
    return surface


ELECTRONS = bz.generators.circle(128, k_fermi=4e9, mass=0.8 * ELECTRON_MASS, tau=1.0)


def measure(surface, rate, parameters, *, seed=None, relative=2e-3, fields=FIELDS, layer_spacing=D, **options):
    """Data from `bzf.transport`; with `seed`, Gaussian noise and its error columns."""
    data = bzf.transport(surface, fields, rate=rate, parameters=parameters, layer_spacing=layer_spacing, **options)
    if seed is not None:
        rng = np.random.default_rng(seed)
        data["resistivity_error"] = relative * data["resistivity"]
        data["hall_resistivity_error"] = relative * np.max(np.abs(data["hall_resistivity"]))
        for column in ("resistivity", "hall_resistivity"):
            data[column] += data[f"{column}_error"] * rng.standard_normal(len(data))
    return data


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def rate4(phi, gamma0, a4):
    return gamma0 * (1 + a4 * np.cos(4 * phi))


TRUTH = {"gamma0": 1e12, "a4": 0.4}


# Link 1: a term shared by every dataset, and one fitted to each.
def split_rate(phi, gamma_imp, gamma_in, a4):
    return gamma_imp + gamma_in * (1 + a4 * np.cos(4 * phi))


SPLIT = {"gamma_imp": 5e11, "a4": 0.5}
GAMMA_IN = [4e11, 1e12, 2e12]
SPLIT_P0 = {"gamma_imp": 1e12, "gamma_in": 1e12, "a4": 0.1}
SPLIT_BOUNDS = {"a4": (-0.95, 0.95)}


def split_datasets(seed=None, **options):
    surface = fourfold()
    return [bzf.Dataset(measure(surface, split_rate, {**SPLIT, "gamma_in": gamma},
                                seed=None if seed is None else seed + number),
                        surface, rate=split_rate, layer_spacing=D, **options)
            for number, gamma in enumerate(GAMMA_IN)]


# Link 2: a law across datasets, Γ(T) = Γ₀ + αT, with T a constant of each dataset.
def law_rate(phi, T, gamma0, alpha, a4):
    return (gamma0 + alpha * T) * (1 + a4 * np.cos(4 * phi))


LAW = {"gamma0": 1e12, "alpha": 2e10, "a4": 0.3}
LAW_P0 = {"gamma0": 5e11, "alpha": 5e9, "a4": 0.1}


def law_datasets(temperatures=(10, 50, 100), surface=None, truth=LAW, **options):
    surface = fourfold() if surface is None else surface
    return [bzf.Dataset(measure(surface, law_rate, {**truth, "T": T}), surface, rate=law_rate, constants={"T": T},
                        name=f"{T} K", layer_spacing=D, **options)
            for T in temperatures]


# --- What the issue asks for: recovery of a known model -----------------------------------

def hole_rate(phi, T, gamma0, alpha, a4):
    return (gamma0 + alpha * T) * (1 + a4 * np.cos(4 * phi))


def electron_rate(gamma_e):
    return gamma_e


def test_recovers_shared_separate_and_law_parameters_of_two_pockets_within_their_errors():
    """Noisy synthetic data at five temperatures from a known model: a hole pocket with
    1/τ = (Γ₀ + αT)(1 + a₄cos4φ), Γ₀, α and a₄ shared, and an electron pocket with its own rate
    at each temperature. Every parameter comes back within 3σ of the truth, and χ²/ν ≈ 1."""
    temperatures = (10, 25, 50, 75, 100)
    pockets = [fourfold(carrier="hole"), ELECTRONS]
    rates = [hole_rate, electron_rate]
    gamma_e = {T: 2e12 + 4e8 * T**2 for T in temperatures}
    datasets = [bzf.Dataset(measure(pockets, rates, {**LAW, "T": T, "gamma_e": gamma_e[T]}, seed=T), pockets,
                            rate=rates, constants={"T": T}, name=f"{T} K", layer_spacing=D, **ERRORS)
                for T in temperatures]
    result = bzf.fit_joint(datasets, p0={**LAW_P0, "gamma_e": 1e12}, separate=["gamma_e"],
                           bounds={"a4": (-0.95, 0.95)})
    expected = {**LAW, **{f"gamma_e[{T} K]": gamma_e[T] for T in temperatures}}
    pulls = {n: (result[n] - expected[n]) / result.errors[n] for n in result.names}
    print(f"chi2/nu {result.reduced_chi_squared:.3f}, pulls {pulls}")
    assert result.names == ["gamma0", "alpha", "a4", *(f"gamma_e[{T} K]" for T in temperatures)]
    assert result.weighting == "errors" and result.absolute
    assert result.degrees_of_freedom == len(temperatures) * 2 * FIELDS.size - len(expected)
    assert all(abs(p) < 3 for p in pulls.values())
    assert 0.7 < result.reduced_chi_squared < 1.3


# --- One dataset is fit_transport ----------------------------------------------------------

@pytest.mark.parametrize("errors", [True, False], ids=["errors", "no errors"])
def test_a_joint_fit_of_one_dataset_is_fit_transport(errors):
    """With weighting="point", exactly: the same residuals, optimiser and diagnostics."""
    surface = fourfold()
    data = measure(surface, rate4, TRUTH, seed=0)
    data.loc[3, "resistivity"] = np.nan  # the two channels then differ in length
    options = dict(rate=rate4, layer_spacing=D, **(ERRORS if errors else {}))
    fit = dict(p0=[{"gamma0": 3e11, "a4": -0.2}, {"gamma0": 3e11, "a4": 0.2}], bounds={"a4": (-0.95, 0.95)})
    single = bzf.fit_transport(data, surface, **options, **fit)
    joint = bzf.fit_joint([bzf.Dataset(data, surface, **options)], weighting="point", **fit)
    assert joint.parameters == single.parameters
    assert joint.errors == single.errors
    assert joint.chi_squared == single.chi_squared
    np.testing.assert_array_equal(joint.covariance, single.covariance)
    pd.testing.assert_frame_equal(joint.starts, single.starts)
    pd.testing.assert_frame_equal(joint.fit_quality.drop(columns="dataset"), single.fit_quality)
    pd.testing.assert_frame_equal(joint.evaluate(FIELDS)["0"], single.evaluate(FIELDS))
    assert joint.weighting == ("errors" if errors else "point")


def test_curve_weighting_of_one_dataset_is_fit_transport_when_the_channels_are_equally_long():
    surface = fourfold()
    data = measure(surface, rate4, TRUTH, seed=0)
    fit = dict(p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (-0.95, 0.95)})
    single = bzf.fit_transport(data, surface, rate=rate4, layer_spacing=D, **fit)
    joint = bzf.fit_joint([bzf.Dataset(data, surface, rate=rate4, layer_spacing=D)], **fit)
    assert joint.weighting == "curve"
    assert joint.parameters == single.parameters and joint.errors == single.errors


# --- The four kinds of link ----------------------------------------------------------------

def test_link_shared_and_separate_terms():
    """1/τ = γ_imp + γ_in(1 + a₄cos4φ): γ_imp and a₄ shared, γ_in fitted to each dataset. Each
    dataset alone fixes only γ_imp + γ_in and γ_in·a₄; together they fix all five, though γ_imp
    and the γ_in are strongly correlated (and warned about)."""
    with pytest.warns(UserWarning, match="gamma_imp and gamma_in\\[0\\] are .*% correlated"):
        result = bzf.fit_joint(split_datasets(), p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS)
    print(result)
    assert result.names == ["gamma_imp", "gamma_in[0]", "a4", "gamma_in[1]", "gamma_in[2]"]
    for name, value in {**SPLIT, **{f"gamma_in[{i}]": g for i, g in enumerate(GAMMA_IN)}}.items():
        assert result[name] == pytest.approx(value, rel=1e-6)
    summary = result.summary()
    assert summary.loc["gamma_in[2]", "parameter"] == "gamma_in" and summary.loc["gamma_in[2]", "group"] == "2"
    assert pd.isna(summary.loc["a4", "group"])


def test_link_a_law_across_datasets():
    """Γ(T) = Γ₀ + αT with T a constant of each dataset: the law's coefficients are fitted."""
    result = bzf.fit_joint(law_datasets(), p0=LAW_P0, bounds={"a4": (-0.95, 0.95)})
    assert result.names == ["gamma0", "alpha", "a4"]
    for name, value in LAW.items():
        assert result[name] == pytest.approx(value, rel=1e-6)
    assert result.datasets == ["10 K", "50 K", "100 K"]


def test_link_a_different_fixed_surface_for_each_dataset():
    """Three pockets of different shape, one rate model with shared parameters."""
    surfaces = [fourfold(k4=k4) for k4 in (0.0, 0.25e9, 0.5e9)]
    datasets = [bzf.Dataset(measure(surface, rate4, TRUTH), surface, rate=rate4, name=f"k4={i}", layer_spacing=D)
                for i, surface in enumerate(surfaces)]
    result = bzf.fit_joint(datasets, p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (-0.95, 0.95)})
    for name, value in TRUTH.items():
        assert result[name] == pytest.approx(value, rel=1e-6)
    curves = bzf.joint_transport(datasets, TRUTH, field=FIELDS)
    for dataset, surface in zip(datasets, surfaces):
        pd.testing.assert_frame_equal(curves[dataset.name],
                                      bzf.transport(surface, FIELDS, rate=rate4, parameters=TRUTH, layer_spacing=D))
    assert all(isinstance(s, pd.DataFrame) for s in result.surfaces.values())


def test_link_groups_of_datasets_sharing_a_constant():
    """Two samples at two temperatures each: the impurity rate is one per sample, shared by its
    temperatures; α and a₄ are shared by all."""
    def rate(phi, T, gamma_imp, alpha, a4):
        return (gamma_imp + alpha * T) * (1 + a4 * np.cos(4 * phi))

    surface = fourfold()
    impurity = {"A": 5e11, "B": 1.5e12}
    shared = {"alpha": 2e10, "a4": 0.3}
    datasets = [bzf.Dataset(measure(surface, rate, {**shared, "T": T, "gamma_imp": impurity[sample]}), surface,
                            rate=rate, constants={"T": T, "sample": sample}, name=f"{sample} {T} K", layer_spacing=D)
                for sample in impurity for T in (10, 80)]
    result = bzf.fit_joint(datasets, p0={"gamma_imp": 1e12, "alpha": 1e10, "a4": 0.1},
                           separate={"gamma_imp": "sample"}, bounds={"a4": (-0.95, 0.95)})
    assert result.names == ["gamma_imp[A]", "alpha", "a4", "gamma_imp[B]"]
    for sample, value in impurity.items():
        assert result[f"gamma_imp[{sample}]"] == pytest.approx(value, rel=1e-6)
    for name, value in shared.items():
        assert result[name] == pytest.approx(value, rel=1e-6)
    assert result.summary().loc["gamma_imp[B]", "group"] == "B"


def test_surface_parameters_can_be_shared_or_separate():
    a = 3.87e-10
    t, tp = bz.units.ev_to_joule(0.25), bz.units.ev_to_joule(-0.0625)

    def band(mu):
        return bz.generators.tight_binding(128, tau=1.0, lattice_constant=a, hopping=t, next_hopping=tp,
                                           chemical_potential=mu, center=(np.pi / a, np.pi / a))

    def rate(gamma):
        return gamma

    mus = [bz.units.ev_to_joule(-0.1), bz.units.ev_to_joule(-0.15)]
    fields = np.linspace(0, 60, 25)
    shared_gamma = [bzf.Dataset(measure(band, rate, {"mu": mu, "gamma": 5e12}, fields=fields), band, rate=rate,
                                layer_spacing=D) for mu in mus]
    result = bzf.fit_joint(shared_gamma, p0={"mu": bz.units.ev_to_joule(-0.12), "gamma": 1e12}, separate=["mu"])
    assert result.names == ["mu[0]", "gamma", "mu[1]"]
    assert result["mu[0]"] == pytest.approx(mus[0], rel=1e-6) and result["mu[1]"] == pytest.approx(mus[1], rel=1e-6)
    assert result["gamma"] == pytest.approx(5e12, rel=1e-6)
    np.testing.assert_allclose(result.surfaces["1"].tau, 1 / result["gamma"])

    shared_mu = [bzf.Dataset(measure(band, rate, {"mu": mus[0], "gamma": g}, fields=fields), band, rate=rate,
                             layer_spacing=D) for g in (2e12, 6e12)]
    result = bzf.fit_joint(shared_mu, p0={"mu": bz.units.ev_to_joule(-0.05), "gamma": 1e12}, separate=["gamma"])
    assert result["mu"] == pytest.approx(mus[0], rel=1e-6)
    assert result["gamma[0]"] == pytest.approx(2e12, rel=1e-6) and result["gamma[1]"] == pytest.approx(6e12, rel=1e-6)


def test_each_dataset_has_its_own_layer_spacing_options_and_channels():
    surface = fourfold()
    a = bzf.Dataset(measure(surface, rate4, TRUTH), surface, rate=rate4, name="a", layer_spacing=D)
    b_data = measure(surface, rate4, TRUTH, layer_spacing=2 * D, extrapolate=True)
    b = bzf.Dataset(b_data.drop(columns="resistivity"), surface, rate=rate4, name="b", layer_spacing=2 * D,
                    resistivity=None, extrapolate=True)
    result = bzf.fit_joint([a, b], p0={"gamma0": 3e11, "a4": 0.1}, bounds={"a4": (-0.95, 0.95)})
    for name, value in TRUTH.items():
        assert result[name] == pytest.approx(value, rel=1e-6)
    assert list(zip(result.fit_quality.dataset, result.fit_quality.channel)) == [
        ("a", "resistivity"), ("a", "hall_resistivity"), ("b", "hall_resistivity")]
    assert result.degrees_of_freedom == 3 * FIELDS.size - 2


# --- Names: shared names set every copy; a copy's own name takes precedence -----------------

def test_a_copy_named_on_its_own_takes_precedence():
    datasets = split_datasets()
    result = bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS,
                           fixed={"gamma_in[1]": GAMMA_IN[1]}, priors={"gamma_in": (1e12, 1e15)})
    assert "gamma_in[1]" not in result.names and result["gamma_in[1]"] == GAMMA_IN[1]
    assert not result.summary().loc["gamma_in[1]", "fitted"]
    assert result["gamma_in[2]"] == pytest.approx(GAMMA_IN[2], rel=1e-6)
    # The prior on the shared name adds one residual per copy, the held one included.
    assert result.degrees_of_freedom == 3 * 2 * FIELDS.size + 3 - 4
    assert result.fit_quality.channel.iloc[-1] == "priors" and result.fit_quality.points.iloc[-1] == 3

    with pytest.raises(ValueError, match=r"outside the bounds: \{'gamma_in\[0\]'"):
        bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"],
                      bounds={**SPLIT_BOUNDS, "gamma_in": (0, 1e13), "gamma_in[0]": (5e12, 1e13)})
    with pytest.raises(ValueError, match="both fixed and given a starting value"):
        bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"], fixed={"gamma_in": 1e12})
    starts = bzf.fit_joint(datasets, separate=["gamma_in"], bounds=SPLIT_BOUNDS,
                           p0={**SPLIT_P0, "gamma_in[2]": 3e12})  # a copy's own starting value
    assert starts["gamma_in[2]"] == pytest.approx(GAMMA_IN[2], rel=1e-6)


# --- Weighting -----------------------------------------------------------------------------

def two_sizes(seed=0):
    """Two noisy datasets without error columns: 41 fields at Γ₀ = 1e12, and 11 at 3e12 (ρ 3× larger)."""
    surface = fourfold()
    sizes = {"many": (np.linspace(0, 40, 41), 1e12), "few": (np.linspace(0, 40, 11), 3e12)}
    datasets = []
    for number, (name, (fields, gamma0)) in enumerate(sizes.items()):
        data = measure(surface, rate4, {"gamma0": gamma0, "a4": 0.4}, fields=fields, seed=seed + number)
        datasets.append(bzf.Dataset(data.drop(columns=list(ERRORS.values())), surface, rate=rate4, name=name,
                                    layer_spacing=D))
    return datasets


def test_curve_weighting_gives_every_curve_the_same_total_weight():
    """σ = RMS × √(points / mean points): each curve's χ² is the mean square of its relative
    misfit times the mean number of points, whatever its size or length. "point" uses the RMS."""
    datasets = two_sizes()
    p0 = {"gamma0": 2e12, "a4": 0.1}
    for weighting in ("curve", "point"):
        result = bzf.fit_joint(datasets, p0=p0, separate=["gamma0"], weighting=weighting, bounds={"a4": (-1, 1)})
        quality = result.fit_quality
        mean_points = quality.points.mean()
        for row in quality.itertuples():
            data = next(d.data for d in datasets if d.name == row.dataset)
            rms = np.sqrt(np.mean(data[row.channel] ** 2))
            factor = np.sqrt(row.points / mean_points) if weighting == "curve" else 1.0
            assert row.sigma == pytest.approx(rms * factor, rel=1e-12)
        assert quality.share.sum() == pytest.approx(1.0, rel=1e-12)
        assert quality.chi_squared.sum() == pytest.approx(result.chi_squared, rel=1e-12)
        assert result.weighting == weighting and not result.absolute

    heavier = [datasets[0], bzf.Dataset(datasets[1].data, datasets[1].surface, rate=rate4, name="few",
                                        layer_spacing=D, weight=4.0)]
    result = bzf.fit_joint(heavier, p0=p0, separate=["gamma0"], bounds={"a4": (-1, 1)})
    plain = bzf.fit_joint(datasets, p0=p0, separate=["gamma0"], bounds={"a4": (-1, 1)})
    np.testing.assert_allclose(result.fit_quality.sigma.iloc[2:], plain.fit_quality.sigma.iloc[2:] / 2, rtol=1e-12)


def test_with_curve_weighting_repeating_every_row_of_a_dataset_changes_no_parameter():
    """A curve counts the same however many points it has; with "point", repeating a dataset's
    rows doubles its weight and moves the shared parameters."""
    datasets = two_sizes(seed=3)
    doubled = [bzf.Dataset(pd.concat([datasets[0].data] * 2, ignore_index=True), datasets[0].surface, rate=rate4,
                           name="many", layer_spacing=D), datasets[1]]
    options = dict(p0={"gamma0": 2e12, "a4": 0.1}, separate=["gamma0"], bounds={"a4": (-1, 1)},
                   least_squares_options={"xtol": 1e-14, "ftol": 1e-14, "gtol": 1e-14})
    once, twice = bzf.fit_joint(datasets, **options), bzf.fit_joint(doubled, **options)
    for name in once.names:
        assert twice[name] == pytest.approx(once[name], rel=1e-8)
    once, twice = (bzf.fit_joint(d, weighting="point", **options) for d in (datasets, doubled))
    assert twice["a4"] != pytest.approx(once["a4"], rel=1e-5)


def test_weighting_is_validated():
    datasets = split_datasets(seed=0)
    with pytest.raises(ValueError, match="weighting must be one of"):
        bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"], weighting="equal")
    with pytest.raises(ValueError, match="weight applies only without measured errors"):
        weighted = [bzf.Dataset(d.data, d.surface, rate=split_rate, layer_spacing=D, weight=2.0, **ERRORS)
                    for d in datasets]
        bzf.fit_joint(weighted, p0=SPLIT_P0, separate=["gamma_in"])
    mixed = [bzf.Dataset(datasets[0].data, datasets[0].surface, rate=split_rate, layer_spacing=D, **ERRORS),
             datasets[1]]
    with pytest.raises(ValueError, match="every fitted channel of every dataset or for none"):
        bzf.fit_joint(mixed, p0=SPLIT_P0, separate=["gamma_in"])


# --- The result: fit quality, and exactly what the fit evaluated -----------------------------

def test_fit_quality_and_evaluate_are_exactly_what_the_fit_compared():
    datasets = split_datasets(seed=10)
    datasets = [bzf.Dataset(d.data.drop(columns=list(ERRORS.values())), d.surface, rate=split_rate, layer_spacing=D)
                for d in datasets]
    datasets[1].data.loc[4, "hall_resistivity"] = np.nan  # a hole in one curve
    result = bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS,
                           priors={"a4": (0.5, 0.1)})
    curves = result.evaluate()
    offset = 0
    for row in result.fit_quality.itertuples():
        if row.channel == "priors":
            continue
        data = datasets[int(row.dataset)].data
        keep = data[row.channel].notna()
        difference = curves[row.dataset][row.channel][keep] - data[row.channel][keep]
        np.testing.assert_array_equal((difference / row.sigma).to_numpy(), result.result.fun[offset:offset + row.points])
        assert row.rms_residual == pytest.approx(np.sqrt(np.mean(difference**2)), rel=1e-12)
        assert row.relative_rms == pytest.approx(row.rms_residual / np.sqrt(np.mean(data[row.channel][keep] ** 2)))
        offset += row.points
    assert result.fit_quality.points.iloc[-1] == 1 and offset + 1 == result.result.fun.size
    assert list(curves) == ["0", "1", "2"]
    pd.testing.assert_index_equal(curves["1"].index, datasets[1].data.index)
    assert "weighting = 'curve'" in repr(result)


def test_joint_transport_is_each_datasets_own_model():
    datasets = split_datasets()
    parameters = {**SPLIT, "gamma_in": 1e12, "gamma_in[2]": 3e12}  # the shared name sets every copy but [2]
    curves = bzf.joint_transport(datasets, parameters, separate=["gamma_in"], field=FIELDS)
    for number, gamma_in in enumerate([1e12, 1e12, 3e12]):
        expected = bzf.transport(fourfold(), FIELDS, rate=split_rate, parameters={**SPLIT, "gamma_in": gamma_in},
                                 layer_spacing=D)
        pd.testing.assert_frame_equal(curves[str(number)], expected)
    at_data = bzf.joint_transport(datasets, parameters, separate=["gamma_in"])
    np.testing.assert_array_equal(at_data["0"].field, datasets[0].data.field)
    with pytest.raises(ValueError, match="no value for parameter 'a4'"):
        bzf.joint_transport(datasets, {"gamma_imp": 1e12, "gamma_in": 1e12}, separate=["gamma_in"])
    with pytest.raises(ValueError, match="no data to take the fields from"):
        bzf.joint_transport([bzf.Dataset(None, fourfold(), rate=rate4, layer_spacing=D)], TRUTH)


def test_evaluate_predicts_a_held_back_dataset():
    result = bzf.fit_joint(law_datasets((10, 100)), p0=LAW_P0, bounds={"a4": (-0.95, 0.95)})
    held_back = law_datasets((50,))
    predicted = result.evaluate(datasets=held_back)["50 K"]
    measured = held_back[0].data
    np.testing.assert_allclose(predicted.resistivity.to_numpy(), measured.resistivity.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(predicted.hall_resistivity.to_numpy()[1:], measured.hall_resistivity.to_numpy()[1:],
                               rtol=1e-6)

    split = bzf.fit_joint(split_datasets()[:2], p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS)
    with pytest.raises(ValueError, match=r"no value for parameter 'gamma_in\[new\]'"):
        surface = fourfold()
        split.evaluate(datasets=[bzf.Dataset(None, surface, rate=split_rate, name="new", layer_spacing=D)],
                       field=FIELDS)


# --- Speed: dataset-local differences ------------------------------------------------------

def test_the_jacobian_is_forward_differences_of_every_residual(monkeypatch):
    """The Jacobian recomputes only the datasets a parameter affects, and is identical to
    differencing every residual."""
    captured = {}
    real = bzf.least_squares

    def capture(fun, x0, **kwargs):
        captured.update(fun=fun, x0=np.array(x0), jac=kwargs["jac"], bounds=kwargs["bounds"])
        return real(fun, x0, **kwargs)

    monkeypatch.setattr(bzf, "least_squares", capture)
    bzf.fit_joint(split_datasets(), p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS)
    fun, x0, jac = captured["fun"], captured["x0"], captured["jac"]
    lower, upper = captured["bounds"]
    local = jac(x0)
    f0 = fun(x0)
    steps = bzf._difference_steps(x0, lower, upper)
    full = np.empty_like(local)
    for j in range(x0.size):
        shifted = x0.copy()
        shifted[j] += steps[j]
        full[:, j] = (fun(shifted) - f0) / (shifted[j] - x0[j])
    np.testing.assert_array_equal(local, full)
    rows = 2 * FIELDS.size
    assert np.all(local[rows:, 1] == 0) and np.any(local[:rows, 1] != 0)  # gamma_in[0] touches dataset 0 only


def test_dataset_local_differences_need_fewer_model_evaluations(monkeypatch):
    calls = {"n": 0}
    original = bzf._Model.transport

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    datasets = split_datasets(seed=1)
    monkeypatch.setattr(bzf._Model, "transport", counted)
    options = dict(p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS)
    local = bzf.fit_joint(datasets, **options)
    n_local, calls["n"] = calls["n"], 0
    full = bzf.fit_joint(datasets, least_squares_options={"jac": "2-point"}, **options)
    print(f"model evaluations: {n_local} dataset-local, {calls['n']} with scipy's 2-point")
    assert n_local < 0.8 * calls["n"]
    for name in local.names:
        assert local[name] == pytest.approx(full[name], rel=1e-6, abs=1e-12)


def test_difference_steps_stay_inside_the_bounds():
    u = np.array([0.0, 1.0, -2.0, 0.95, 0.5])
    lower = np.array([-np.inf, -np.inf, -np.inf, -1.0, 0.5 - 1e-10])
    upper = np.array([np.inf, 1.0, np.inf, 0.95, 0.5 + 4e-10])
    steps = bzf._difference_steps(u, lower, upper)
    eps = np.sqrt(np.finfo(float).eps)
    np.testing.assert_allclose(steps[:4], [eps, -eps, -2 * eps, -eps], rtol=1e-12)
    assert steps[4] == pytest.approx(2e-10)  # bounds closer than a step on both sides: half way to the further
    assert np.all((u + steps >= lower) & (u + steps <= upper))


# --- Diagnostics, starts and invalid models --------------------------------------------------

def test_degenerate_parameters_warn_across_datasets():
    surface = fourfold()
    datasets = [bzf.Dataset(measure(surface, lambda a, b: a + b, {"a": 4e11, "b": 6e11}), surface,
                            rate=lambda a, b: a + b, name=str(n), layer_spacing=D) for n in range(2)]
    with pytest.warns(UserWarning, match="do not determine every parameter|correlated"):
        bzf.fit_joint(datasets, p0={"a": 1e11, "b": 1e11})


def test_several_starts_and_equally_good_different_solutions():
    """On a contour with a mirror plane, τ(φ) and τ(−φ) give identical data: the sign of a sin4φ
    term cannot be found, in one dataset or in several."""
    def chiral(phi, gamma0, b4):
        return gamma0 * (1 + 0.3 * np.cos(4 * phi) + b4 * np.sin(4 * phi))

    surface = fourfold()
    datasets = [bzf.Dataset(measure(surface, chiral, {"gamma0": g, "b4": 0.3}, seed=n), surface, rate=chiral,
                            layer_spacing=D, **ERRORS) for n, g in enumerate((1e12, 2e12))]
    with pytest.warns(UserWarning, match="different solution"):
        result = bzf.fit_joint(datasets, p0=[{"gamma0": 8e11, "b4": 0.2}, {"gamma0": 8e11, "b4": -0.2}],
                               separate=["gamma0"], bounds={"b4": (-0.6, 0.6)})
    assert len(result.starts) == 2
    assert list(result.starts.columns) == ["gamma0[0]", "b4", "gamma0[1]", "chi_squared", "success"]
    assert result.starts.b4.iloc[0] == pytest.approx(-result.starts.b4.iloc[1], rel=1e-4)


def test_non_convergence_warns():
    with pytest.warns(UserWarning, match="did not converge"):
        bzf.fit_joint(law_datasets(), p0=LAW_P0, bounds={"a4": (-0.95, 0.95)}, least_squares_options={"max_nfev": 1})


def test_steps_into_an_invalid_model_are_turned_back():
    """Unbounded, the optimiser may try a₄ beyond ±1, where the rate is negative."""
    truth = {**LAW, "a4": 0.9}
    datasets = law_datasets((10, 100), surface=fourfold(512), truth=truth)
    result = quiet(bzf.fit_joint, datasets, p0=LAW_P0)
    for name, value in truth.items():
        assert result[name] == pytest.approx(value, rel=1e-5)


# --- Clear errors, and inputs left alone ----------------------------------------------------

def test_clear_errors():
    datasets = split_datasets()
    fit = bzf.fit_joint
    with pytest.raises(TypeError, match="list of bzf.Dataset"):
        fit([d.data for d in datasets], p0=SPLIT_P0)
    with pytest.raises(ValueError, match="at least one Dataset"):
        fit([], p0=SPLIT_P0)
    with pytest.raises(ValueError, match="must be unique"):
        fit([bzf.Dataset(d.data, d.surface, rate=split_rate, name="same", layer_spacing=D) for d in datasets],
            p0=SPLIT_P0)
    with pytest.raises(TypeError, match="not the string 'gamma_in'"):
        fit(datasets, p0=SPLIT_P0, separate="gamma_in")
    with pytest.raises(ValueError, match="separate names \\['gamma_out'\\]"):
        fit(datasets, p0=SPLIT_P0, separate=["gamma_out"])
    with pytest.raises(ValueError, match="by the constant 'sample', but dataset '0' has no constant"):
        fit(datasets, p0=SPLIT_P0, separate={"gamma_in": "sample"})
    with pytest.raises(ValueError, match="unknown parameter 'a8' in p0"):
        fit(datasets, p0={**SPLIT_P0, "a8": 0.0}, separate=["gamma_in"])
    for argument in ("fixed", "bounds", "priors"):
        with pytest.raises(ValueError, match=f"unknown parameter 'a8' in {argument}"):
            fit(datasets, p0=SPLIT_P0, separate=["gamma_in"], **{argument: {"a8": (0.0, 1.0)}})
    with pytest.raises(ValueError, match="no value for parameter 'a4'"):
        fit(datasets, p0={"gamma_imp": 1e12, "gamma_in": 1e12}, separate=["gamma_in"])
    with pytest.raises(ValueError, match="has no data to fit"):
        fit([*datasets[:2], bzf.Dataset(None, fourfold(), rate=split_rate, layer_spacing=D)], p0=SPLIT_P0)

    surface = fourfold()
    with pytest.raises(ValueError, match="'T' is a constant of dataset\\(s\\) \\['0'\\] but a parameter of"):
        fit([bzf.Dataset(measure(surface, law_rate, {**LAW, "T": 10}), surface, rate=law_rate, constants={"T": 10},
                         layer_spacing=D),
             bzf.Dataset(measure(surface, law_rate, {**LAW, "T": 50}), surface, rate=law_rate, layer_spacing=D)],
            p0=LAW_P0)
    with pytest.raises(ValueError, match="both a column of the surface and a constant"):
        fit([bzf.Dataset(measure(surface, rate4, TRUTH), surface, rate=rate4, constants={"phi": 0.0},
                         layer_spacing=D)], p0={"gamma0": 1e12, "a4": 0.0})


def test_dataset_checks_its_arguments():
    surface = fourfold()
    data = measure(surface, rate4, TRUTH)
    with pytest.raises(ValueError, match="layer_spacing must be finite and positive, got -1.0"):
        bzf.Dataset(data, surface, layer_spacing=-1.0)
    with pytest.raises(ValueError, match="weight must be finite and positive, got 0.0"):
        bzf.Dataset(data, surface, layer_spacing=D, weight=0)
    with pytest.raises(ValueError, match="no column 'rho' \\(the resistivity argument\\)"):
        bzf.Dataset(data, surface, layer_spacing=D, resistivity="rho")
    with pytest.raises(ValueError, match="no column 'err' \\(the hall_resistivity_error argument\\)"):
        bzf.Dataset(data, surface, layer_spacing=D, hall_resistivity_error="err")
    with pytest.raises(TypeError, match="constants must be a dict"):
        bzf.Dataset(data, surface, layer_spacing=D, constants=[10.0])
    with pytest.raises(TypeError, match="data must be a DataFrame or None"):
        bzf.Dataset(data.to_numpy(), surface, layer_spacing=D)
    dataset = bzf.Dataset(data, surface, layer_spacing=D, constants={"T": 10.0}, name=10, period=None)
    assert dataset.name == "10" and dataset.options == {"period": None}
    assert "constants={'T': 10.0}" in repr(dataset)


def test_inputs_are_not_modified_and_no_future_warnings():
    datasets = split_datasets(seed=5)
    copies = [(d.data.copy(), d.surface.copy()) for d in datasets]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = bzf.fit_joint(datasets, p0=SPLIT_P0, separate=["gamma_in"], bounds=SPLIT_BOUNDS)
        result.evaluate()
        result.surfaces
        bzf.joint_transport(datasets, result.parameters, separate=["gamma_in"])
    for dataset, (data, surface) in zip(datasets, copies):
        pd.testing.assert_frame_equal(dataset.data, data)
        pd.testing.assert_frame_equal(dataset.surface, surface)
