"""The correctness claims of the boltzmann_fitting docs, checked on the docs examples themselves.

Each test runs a docs example (exactly the code the docs show) and asserts on its variables,
so the numbers on the pages cannot drift from what is tested.
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_doc_examples", ROOT / "scripts" / "build_doc_examples.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

EXAMPLES = builder.EXAMPLES / "beta" / "boltzmann_fitting"


def run(name):
    """Run a docs example and return its variables."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        namespace, _, _ = builder.run_example_namespace(EXAMPLES / f"{name}.py")
    return namespace


# F1: recovering known answers (recovery.md)

def test_exact_data_give_back_the_truth():
    ns = run("recover_exact")
    for result in (ns["rate_only"], ns["surface_too"]):
        errors = {n: abs(result[n] / ns["truth"][n] - 1) for n in result.names}
        print(errors)
        assert max(errors.values()) < 1e-6


def test_bzf_agrees_with_the_closed_form_two_band_fit():
    ns = run("two_band_check")
    fit = ns["fits"][True]  # extrapolated in N
    for n in ns["names"]:
        shift = (fit[n] - ns["reference_values"][n]) / ns["reference_errors"][n]
        ratio = fit.errors[n] / ns["reference_errors"][n]
        print(f"{n}: shift {shift:+.2e} sigma, error ratio {ratio:.6f}")
        assert abs(shift) < 0.01
        assert abs(ratio - 1) < 0.01


# F2: honest error bars (error_bars.md)

@pytest.mark.slow
def test_error_bars_are_calibrated():
    ns = run("error_bars")
    trials, nu = ns["trials"], ns["nu"]
    for label, pulls in (("absolute", ns["pulls_absolute"]), ("relative", ns["pulls_relative"])):
        mean, std = pulls.mean(axis=0), pulls.std(axis=0)
        coverage = np.mean(np.abs(pulls) < 1)
        print(f"{label}: mean pull {mean}, std {std}, 1-sigma coverage {coverage:.3f}")
        assert np.all(np.abs(std - 1) < 0.15)
        if label == "absolute":
            assert np.all(np.abs(mean) < 3 / np.sqrt(trials))
            assert abs(coverage - 0.68) < 0.10
    chi = ns["chi_squared"].mean()
    print(f"mean chi2 {chi:.2f} for nu = {nu}, allowed +- {3 * np.sqrt(2 * nu / trials):.2f}")
    assert abs(chi - nu) < 3 * np.sqrt(2 * nu / trials)


# F3: discretisation and limits (discretisation.md, limits.md)

def test_discretisation_bias_falls_as_n_squared_and_extrapolation_removes_it():
    ns = run("discretisation")
    print(f"slopes {ns['slopes']}")
    assert np.all(np.abs(ns["slopes"] + 2) < 0.1)
    at_256 = np.abs(ns["bias"][True][ns["sizes"].index(256)]) / ns["relative_error"]
    print(f"extrapolated bias at N = 256: {at_256} sigma")
    assert np.all(at_256 < 0.1)


def test_covariance_matches_the_chi_squared_landscape():
    rises = run("landscape")["rises"]
    print(f"chi2 rises one sigma along the principal axes: {rises}")
    assert np.allclose(rises, 1.0, atol=0.05)


def test_velocity_scale_is_degenerate_with_the_rates_and_mirror_images_are_flagged():
    ns = run("degeneracies")
    right, wrong = ns["right"], ns["wrong"]
    ratios = [wrong[n] / right[n] for n in right.names]
    print(f"rate ratios {ratios}, chi2 {right.chi_squared!r} {wrong.chi_squared!r}")
    assert np.allclose(ratios, 1.1, rtol=1e-6)
    assert abs(wrong.chi_squared / right.chi_squared - 1) < 1e-10
    assert any("different solution" in str(w.message) for w in ns["caught"])


def test_an_extra_harmonic_is_flagged_as_correlated():
    caught = run("too_many_parameters")["caught"]
    assert any("correlated" in str(w.message) for w in caught)


def test_errors_shrink_as_the_field_range_grows():
    relative = run("field_range")["relative"]
    print(relative)
    assert np.all(np.diff(relative, axis=0) < 0)


# F4: preparing data (prepare.md)

def test_prepared_data_match_the_truth_and_the_sign_check_works():
    ns = run("prepare_data")
    print(f"chi2/n against the truth: {ns['chi']}")
    assert all(0.6 < value < 1.5 for value in ns["chi"].values())
    data, check = ns["data"], ns["hall_sign_agrees"]
    assert check(data, ns["surface"], ns["rate"], ns["start"])
    assert not check(data.assign(hall_resistivity=-data["hall_resistivity"]), ns["surface"], ns["rate"], ns["start"])


# F5: surfaces, rate models and Fermi-surface fits (models.md, fermi_surface.md)

def within(result, truth, sigmas=3):
    pulls = {n: (result[n] - truth[n]) / result.errors[n] for n in result.names}
    print(f"pulls {pulls}")
    return all(abs(p) < sigmas for p in pulls.values())


def test_the_true_rate_form_recovers_the_truth():
    ns = run("rate_forms")
    result = ns["results"]["anisotropic"]
    print(f"chi2/nu {result.reduced_chi_squared:.3f}")
    assert within(result, ns["true_rate"])
    assert 0.6 < result.reduced_chi_squared < 1.6


@pytest.mark.parametrize("name", ["surface_from_points", "several_pockets", "open_sheets"])
def test_surface_options_recover_the_truth(name):
    ns = run(name)
    assert within(ns["result"], ns["true_rate"] if name == "surface_from_points" else ns["truth"])


def test_fermi_surface_fit_recovers_the_truth():
    ns = run("fit_surface")
    assert within(ns["result"], {"mu": ns["true_mu"], **ns["true_rate"]})


def test_a_prior_adds_information_as_inverse_variances():
    ns = run("priors")
    ratio = ns["informed"].errors["t_prime"] / ns["expected"]
    print(f"sigma with prior / expected = {ratio:.4f}")
    assert abs(ratio - 1) < 0.05
    assert within(ns["informed"], {"mu": ns["true_mu"], "t_prime": ns["t_prime"], **ns["true_rate"]})


# F6: reading a result (results.md)

def test_the_tau_band_from_the_covariance_covers_the_truth():
    ns = run("tau_band")
    inside = (ns["low"] <= ns["true_tau"]) & (ns["true_tau"] <= ns["high"])
    print(f"truth inside the 95% band at {inside.mean():.0%} of angles")
    assert np.all(inside)
