"""M11: Richardson extrapolation in the number of nodes (`extrapolate=True`).

With `extrapolate`, σ = (4σ_N − σ_{N/2})/3, where σ_{N/2} is computed on every other
node. The discretisation error is c/N² + O(N⁻⁴) with the same c at both resolutions
when the nodes sample a smooth contour smoothly, so the result converges as N⁻⁴. The
exception is the magnetoresistance inside the low-field window ω_cτ ≲ 2π/N, where a
small |B|³/N remainder (from the slope kinks of the piecewise-linear model) is not
cancelled; there extrapolation must still never make the MR worse.
"""
import functools
import warnings

import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9
TAU = 1e-13
K_F = 7e9
G = 2 * np.pi / 3.87e-10
SIZES = np.array([64, 128, 256, 512, 1024])
BACKENDS = ["python", "numba"]


def fourfold_k(p):
    return 7.35e9 - 0.25e9 * np.cos(4 * p)


def fourfold_dk(p):
    return 1e9 * np.sin(4 * p)


def fourfold_tau(p):
    return sc.cos4phi(p, TAU, anisotropy=0.3)


def lopsided_k(p):
    return K_F * (1 + 0.08 * np.cos(3 * p) + 0.05 * np.sin(2 * p + 0.3))


def lopsided_dk(p):
    return K_F * (-0.24 * np.sin(3 * p) + 0.10 * np.cos(2 * p + 0.3))


def lopsided_tau(p):
    return sc.hot_spot(p, TAU, strength=3.0, width=0.35, positions=(0.3, 2.0, 4.1))


# name: (sampled contour of N nodes, the same pocket parametrised, cyclotron mass)
POCKETS = {
    "fourfold": (lambda n: gen.polar(n, k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=5 * M, tau=fourfold_tau,
                                     carrier="hole"),
                 ref.ParametricContour.polar(k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=5 * M, tau=fourfold_tau,
                                             carrier="hole"), 5 * M),
    "lopsided": (lambda n: gen.polar(n, k_fermi=lopsided_k, dk_fermi=lopsided_dk, mass=M, tau=lopsided_tau),
                 ref.ParametricContour.polar(k_fermi=lopsided_k, dk_fermi=lopsided_dk, mass=M, tau=lopsided_tau), M),
}
X = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])  # ω_cτ = eBτ₀/m


def arrays(df):
    return df["kx"], df["ky"], df["vx"], df["vy"], df["tau"]


def sigma(df, fields, **kwargs):
    kwargs.setdefault("layer_spacing", D)
    return conductivity_tensor(*arrays(df), fields, **kwargs)


def slope(errors, sizes=SIZES):
    return np.polyfit(np.log(sizes), np.log(errors), 1)[0]


def magnetoresistance_and_hall(s, fields):
    """ρ_xx(B)/ρ_xx(0) − 1 and R_H = ½(ρ_yx − ρ_xy)/B from σ at (0, B₁, B₂, …)."""
    rho = np.linalg.inv(s)
    return rho[1:, 0, 0] / rho[0, 0, 0] - 1, 0.5 * (rho[1:, 1, 0] - rho[1:, 0, 1]) / fields[1:]


@functools.lru_cache(maxsize=None)
def reference(pocket):
    """Brute-force σ at B = 0 and at each ω_cτ in X, and the fields."""
    _, contour, mass = POCKETS[pocket]
    fields = np.concatenate([[0.0], X * mass / (E * TAU)])
    return ref.sigma(contour, fields, layer_spacing=D), fields


@functools.lru_cache(maxsize=None)
def errors(pocket, extrapolate):
    """Relative errors of σ, MR and R_H against the reference, each of shape (n_sizes, nX)."""
    exact, fields = reference(pocket)
    mr_exact, hall_exact = magnetoresistance_and_hall(exact, fields)
    make = POCKETS[pocket][0]
    result = []
    for n in SIZES:
        s = sigma(make(n), fields, extrapolate=extrapolate)
        mr, hall = magnetoresistance_and_hall(s, fields)
        result.append((np.max(np.abs(s[1:] - exact[1:]), axis=(1, 2)) / np.max(np.abs(exact[1:]), axis=(1, 2)),
                       np.abs(mr / mr_exact - 1), np.abs(hall / hall_exact - 1)))
    return tuple(np.array(r) for r in zip(*result))


def show(label, values):
    return f"{label} " + ", ".join(f"{v:.1e}" for v in values) + f" (slope {slope(values):.2f})"


# ---------------------------------------------------------------------------
# What it computes
# ---------------------------------------------------------------------------

def test_off_by_default():
    df = POCKETS["lopsided"][0](128)
    fields = np.array([0.0, 1.0, -3.0])
    np.testing.assert_array_equal(sigma(df, fields), sigma(df, fields, extrapolate=False))


@pytest.mark.parametrize("backend", BACKENDS)
def test_is_the_richardson_combination_of_n_and_every_other_node(backend):
    """(4σ_N − σ_{N/2})/3 with σ_{N/2} on every other node, whatever the input order, with or
    without a repeated closing point, for either sign of B."""
    df = POCKETS["lopsided"][0](128)
    fields = np.array([-30.0, 0.0, 0.3, 3.0, 30.0])
    expected = (4 * sigma(df, fields, backend=backend) - sigma(df.iloc[::2], fields, backend=backend)) / 3
    scale = np.max(np.abs(expected))
    variants = {
        "as generated": df,
        "reversed": df.iloc[::-1],
        "closing point repeated": pd.concat([df, df.iloc[:1]]),
    }
    for label, variant in variants.items():
        error = np.max(np.abs(sigma(variant, fields, backend=backend, extrapolate=True) - expected)) / scale
        print(f"{label}: {error:.1e}")
        assert error <= 1e-13


# ---------------------------------------------------------------------------
# Convergence as N⁻⁴
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pocket", POCKETS)
def test_sigma_and_hall_coefficient_converge_as_n_to_the_minus_four(pocket):
    """σ and R_H against the brute-force reference, N = 64…1024, from ω_cτ = 10⁻³ to 10: the
    fitted slope lies in [−4.5, −3.5] (it is −2 without extrapolation)."""
    sigma_errors, _, hall_errors = errors(pocket, True)
    plain_sigma, _, plain_hall = errors(pocket, False)
    for j, x in enumerate(X):
        print(f"{pocket}, x = {x:g}: " + show("sigma", sigma_errors[:, j]) + "; " + show("R_H", hall_errors[:, j])
              + f"; without, at N = 1024: {plain_sigma[-1, j]:.1e}, {plain_hall[-1, j]:.1e}")
        assert -4.5 <= slope(sigma_errors[:, j]) <= -3.5
        assert -4.5 <= slope(hall_errors[:, j]) <= -3.5


@pytest.mark.parametrize("pocket", POCKETS)
def test_magnetoresistance_converges_as_n_to_the_minus_four_above_the_low_field_window(pocket):
    """MR against the reference at ω_cτ = 0.1, 1 and 10: slope in [−4.5, −3.5]."""
    _, mr_errors, _ = errors(pocket, True)
    for j, x in enumerate(X):
        if x >= 0.1:
            print(f"{pocket}, x = {x:g}: " + show("MR", mr_errors[:, j]))
            assert -4.5 <= slope(mr_errors[:, j]) <= -3.5


@pytest.mark.parametrize("pocket", POCKETS)
def test_magnetoresistance_is_never_worse_in_the_low_field_window(pocket):
    """At ω_cτ = 10⁻³ (inside the window ω_cτ ≲ 2π/N for every N here) the MR error has an
    O(|B|/N) part, from the |B|³/N remainder of the model, that extrapolation cannot cancel.
    It must still be no worse than without, at every N, and within 1e-4 at N = 1024."""
    _, extrapolated, _ = errors(pocket, True)
    _, plain, _ = errors(pocket, False)
    print(f"{pocket}, x = 1e-3: " + show("MR extrapolated", extrapolated[:, 0]) + "; " + show("without", plain[:, 0]))
    assert np.all(extrapolated[:, 0] <= plain[:, 0])
    assert extrapolated[-1, 0] <= 1e-4


def test_drude_circle_converges_as_n_to_the_minus_four():
    """K1's circle at ω_cτ = 1 against the exact Drude tensor, N = 32…1024."""
    sizes = np.array([32, 64, 128, 256, 512, 1024])
    field = M / (E * TAU)
    exact = an.drude_circle(field, density=an.circle_density(K_F, D), mass=M, tau=TAU)[0]
    errors_ = [np.max(np.abs(sigma(gen.circle(n, k_fermi=K_F, mass=M, tau=TAU), field, extrapolate=True)[0] - exact))
               / np.max(np.abs(exact)) for n in sizes]
    print("Drude circle: " + ", ".join(f"{e:.1e}" for e in errors_) + f" (slope {slope(errors_, sizes):.2f})")
    assert -4.5 <= slope(errors_, sizes) <= -3.5
    assert errors_[-2] < 1e-8


def test_warped_open_sheets_converge_as_n_to_the_minus_four():
    """Both warped sheets at ω_cτ = 1 against the reference, N = 64…1024."""
    sheets = lambda n: gen.open_sheets(n, k0=5e9, velocity=2e5, tau=TAU, period=G, warping=0.5e9)  # noqa: E731
    c = prepare_contour(*arrays(sheets(1024)[0]), period=(0.0, G))
    field = np.sum(c.damping) / (2 * np.pi)  # ω_cτ averaged round the period = 1
    exact = sum(ref.sigma(ref.ParametricContour.open_sheet(k0=5e9, velocity=2e5, warping=0.5e9, period=G, tau=TAU,
                                                           side=side), field, layer_spacing=D)[0]
                for side in (1.0, -1.0))
    errors_ = []
    for n in SIZES:
        s = bz.conductivity(sheets(n), field, layer_spacing=D, period=(0.0, G), extrapolate=True)
        errors_.append(np.max(np.abs(s.iloc[0, 1:].to_numpy().reshape(2, 2) - exact)) / np.max(np.abs(exact)))
    print("warped sheets: " + ", ".join(f"{e:.1e}" for e in errors_) + f" (slope {slope(errors_):.2f})")
    assert -4.5 <= slope(errors_) <= -3.5


# ---------------------------------------------------------------------------
# Exact properties survive, and the options combine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_onsager_still_holds(backend):
    """Both resolutions satisfy σ(−B) = σ(B)ᵀ to rounding, so their combination does too:
    to 1e-13 unsymmetrised on a mirror-free pocket, and exactly when symmetrised."""
    df = POCKETS["lopsided"][0](256)
    x = np.geomspace(1e-5, 1e5, 21)
    fields = np.concatenate([x, -x]) / (E * TAU / M)
    raw = sigma(df, fields, symmetrize=False, extrapolate=True, backend=backend)
    defect = np.max(np.abs(raw[x.size:] - np.transpose(raw[:x.size], (0, 2, 1)))) / np.max(np.abs(raw))
    print(f"unsymmetrised: max |sigma(-B) - sigma(B)^T| / max |sigma| = {defect:.1e}")
    assert defect <= 1e-13
    sym = sigma(df, fields, extrapolate=True, backend=backend)
    assert np.array_equal(sym[x.size:], np.transpose(sym[:x.size], (0, 2, 1)))


def test_flat_open_sheets_stay_exact():
    """K11 is exact at any N, so extrapolating keeps it: σ_xx field-independent to 1e-12."""
    sheets = gen.open_sheets(512, k0=5e9, velocity=2e5, tau=TAU, period=G)
    s = bz.conductivity(sheets, [0.0, 1.0, 100.0], layer_spacing=D, period=(0.0, G), extrapolate=True)
    change = np.max(np.abs(s.sigma_xx / s.sigma_xx.iloc[0] - 1))
    print(f"flat sheets: max |sigma_xx(B)/sigma_xx(0) - 1| = {change:.1e}")
    assert change <= 1e-12


def test_kz_slices_are_extrapolated_one_by_one():
    """A k_z-independent surface given as slices still reproduces the 2D result (1e-12)."""
    df = POCKETS["fourfold"][0](256)
    layered = pd.concat([df.assign(kz=-np.pi / D + 2 * np.pi * j / (4 * D), vz=0.0) for j in range(4)],
                        ignore_index=True)
    fields = np.array([-3.0, 0.0, 0.3, 30.0])
    two_d = bz.conductivity(df, fields, layer_spacing=D, extrapolate=True).iloc[:, 1:].to_numpy().reshape(-1, 2, 2)
    three_d = bz.conductivity(layered, fields, layer_spacing=D, kz="kz", extrapolate=True)
    in_plane = three_d.iloc[:, 1:].to_numpy().reshape(-1, 3, 3)[:, :2, :2]
    assert np.max(np.abs(in_plane - two_d)) / np.max(np.abs(two_d)) <= 1e-12


def test_two_pockets_through_the_dataframe_api():
    """K5's compensated two-band metal: extrapolated ρ_xx matches (1 + μ_eμ_h B²)/(ne(μ_e+μ_h))
    far more closely than without (1e-8 against 1e-4 at N = 512)."""
    n = an.circle_density(K_F, D)
    mu_e, mu_h = E * TAU / M, E * (TAU / 2) / (2 * M)
    pockets = [gen.circle(512, k_fermi=K_F, mass=M, tau=TAU),
               gen.circle(512, k_fermi=K_F, mass=2 * M, tau=TAU / 2, carrier="hole")]
    fields = np.array([0.1, 1.0, 10.0]) / mu_e
    rho_xx = bz.resistivity(bz.conductivity(pockets, fields, layer_spacing=D, extrapolate=True)).rho_xx.to_numpy()
    error = np.max(np.abs(rho_xx / ((1 + mu_e * mu_h * fields**2) / (n * E * (mu_e + mu_h))) - 1))
    print(f"two-band rho_xx, extrapolated: max rel error {error:.1e}")
    assert error < 1e-8


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def test_needs_an_even_number_of_nodes_at_least_32():
    make = lambda n: POCKETS["lopsided"][0](n).assign(tau=TAU)  # noqa: E731  (so that so few nodes resolve τ)
    for n in (63, 30):
        with pytest.raises(ValueError, match="even number of distinct nodes, at least 32"):
            sigma(make(n), 1.0, extrapolate=True)
    sigma(make(32), 1.0, extrapolate=True)
    sigma(pd.concat([make(64), make(64).iloc[:1]]), 1.0, extrapolate=True)  # 65 rows, 64 distinct nodes
    df = POCKETS["fourfold"][0](63)
    layered = pd.concat([df.assign(kz=-np.pi / D + np.pi * j / D, vz=0.0) for j in range(2)], ignore_index=True)
    with pytest.raises(ValueError, match="even number"):
        bz.conductivity(layered, 1.0, layer_spacing=D, kz="kz", extrapolate=True)


def test_warns_when_every_other_node_is_too_coarse():
    """A 10:1 ellipse is fine at N = 64 but not at 32: the warning says it is the half-resolution
    contour that is too coarse. At N = 128 neither warns."""
    thin = lambda n: gen.ellipse(n, k_fermi=K_F, mass_x=M, mass_y=100 * M, tau=TAU)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma(thin(64), 1.0)
        sigma(thin(128), 1.0, extrapolate=True)
    with pytest.warns(UserWarning, match="every other node"):
        sigma(thin(64), 1.0, extrapolate=True)
