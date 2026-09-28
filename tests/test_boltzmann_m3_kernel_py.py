"""M3: the plain-NumPy O(N) kernel.

Checks that the kernel converges to the brute-force reference as O(N⁻²) from
ω_cτ = 10⁻⁴ to 10, in σ and in the derived magnetoresistance and Hall coefficient
(and to the exact Jones–Zener limit of the magnetoresistance at 10⁻⁵), that it stays
finite over twelve decades of ω_cτ, and that the field branch joins the zero-field
branch continuously. (The moments p_k it uses are tested with the damping-coordinate
kernel, in M10.)
"""
import functools

import numpy as np
import pandas as pd
import pytest

from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
D = 1e-9  # m
TAU = 1e-13  # s
K00, K40 = 7.35e9, 0.25e9  # m⁻¹
MASS = 5 * ELECTRON_MASS  # cyclotron mass of the polar pocket


def fourfold_k(p):
    return K00 - K40 * np.cos(4 * p)


def fourfold_dk(p):
    return 4 * K40 * np.sin(4 * p)


def fourfold_tau(p):
    return sc.cos4phi(p, TAU, anisotropy=0.3)


def fourfold(n):
    return gen.polar(n, k_fermi=fourfold_k, mass=MASS, tau=fourfold_tau, carrier="hole")


def sigma(df, fields, **kwargs):
    kwargs.setdefault("layer_spacing", D)
    kwargs.setdefault("backend", "python")
    return conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"], fields, **kwargs)


# ---------------------------------------------------------------------------
# Convergence to the reference
# ---------------------------------------------------------------------------

SIZES = np.array([64, 128, 256, 512, 1024])
FOURFOLD = ref.ParametricContour.polar(k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=MASS, tau=fourfold_tau,
                                       carrier="hole")


def lopsided_k(p):
    return 7e9 * (1 + 0.08 * np.cos(3 * p) + 0.05 * np.sin(2 * p + 0.3))


def lopsided_dk(p):
    return 7e9 * (-0.24 * np.sin(3 * p) + 0.10 * np.cos(2 * p + 0.3))


def lopsided_tau(p):
    return sc.hot_spot(p, TAU, strength=3.0, width=0.35, positions=(0.3, 2.0, 4.1))


def lopsided(n):
    """No mirror plane, and hot spots off the axes, so nothing cancels by symmetry."""
    return gen.polar(n, k_fermi=lopsided_k, dk_fermi=lopsided_dk, mass=ELECTRON_MASS, tau=lopsided_tau)


LOPSIDED = ref.ParametricContour.polar(k_fermi=lopsided_k, dk_fermi=lopsided_dk, mass=ELECTRON_MASS,
                                       tau=lopsided_tau)
POCKETS = {"fourfold": (fourfold, FOURFOLD, MASS), "lopsided": (lopsided, LOPSIDED, ELECTRON_MASS)}


@functools.lru_cache(maxsize=None)
def reference(pocket, x):
    """Brute-force σ at B = 0 and at ω_cτ = x (eBτ₀/m), shape (2, 2, 2), and the field."""
    _, contour, mass = POCKETS[pocket]
    field = x * mass / (E * TAU)
    return ref.sigma(contour, [0.0, field], layer_spacing=D), field


def magnetoresistance_and_hall(pair, field):
    """ρ_xx(B)/ρ_xx(0) − 1 and R_H = ½(ρ_yx − ρ_xy)/B from σ at (0, B)."""
    rho = np.linalg.inv(pair)
    return rho[1, 0, 0] / rho[0, 0, 0] - 1, 0.5 * (rho[1, 1, 0] - rho[1, 0, 1]) / field


def convergence(pocket, x):
    """Relative errors of σ(B), MR and R_H against the reference for N in SIZES."""
    exact, field = reference(pocket, x)
    make = POCKETS[pocket][0]
    mr_exact, hall_exact = magnetoresistance_and_hall(exact, field)
    errors = []
    for n in SIZES:
        actual = sigma(make(n), [0.0, field])
        mr, hall = magnetoresistance_and_hall(actual, field)
        errors.append((np.max(np.abs(actual[1] - exact[1])) / np.max(np.abs(exact[1])),
                       abs(mr / mr_exact - 1), abs(hall / hall_exact - 1)))
    return np.array(errors).T, mr_exact  # (3, n_sizes)


def slope(errors):
    return np.polyfit(np.log(SIZES), np.log(errors), 1)[0]


@pytest.mark.parametrize("x", [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
def test_converges_to_the_reference_as_n_to_the_minus_two(x):
    """Error of σ against the brute-force reference on the fourfold contour (anisotropic τ),
    N = 64…1024: the fitted log–log slope must lie in [−2.1, −1.9], from ω_cτ = 10⁻⁴ (where
    the history decays within a segment) to 10."""
    (errors, _, _), _ = convergence("fourfold", x)
    print(f"x = {x}: sigma errors " + ", ".join(f"{e:.2e}" for e in errors) + f"; slope {slope(errors):.3f}")
    assert -2.1 <= slope(errors) <= -1.9


@pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
def test_magnetoresistance_converges_to_the_reference(x):
    """MR = ρ_xx(B)/ρ_xx(0) − 1 on the fourfold contour converges with slope in [−2.1, −1.9]."""
    (_, errors, _), mr = convergence("fourfold", x)
    print(f"x = {x}: MR = {mr:.4e}; relative errors " + ", ".join(f"{e:.2e}" for e in errors)
          + f"; slope {slope(errors):.3f}")
    assert -2.1 <= slope(errors) <= -1.9


@pytest.mark.parametrize("x", [1e-3, 1e-2, 0.1, 1.0, 10.0])
def test_hall_coefficient_converges_to_the_reference(x):
    """R_H on the fourfold contour converges with slope in [−2.1, −1.9] from ω_cτ = 10⁻³ to 10."""
    (_, _, errors), _ = convergence("fourfold", x)
    print(f"x = {x}: R_H relative errors " + ", ".join(f"{e:.2e}" for e in errors) + f"; slope {slope(errors):.3f}")
    assert -2.1 <= slope(errors) <= -1.9


@pytest.mark.parametrize("pocket", ["fourfold", "lopsided"])
def test_low_field_magnetoresistance_converges_to_the_jones_zener_limit(pocket):
    """At ω_cτ = 10⁻⁵ the history decays within a segment for every N here (the regime where a
    trapezoid outer sum gives MR ∝ |B|/N). MR/B² must converge to the exact Jones–Zener
    coefficient with slope in [−2.1, −1.9]."""
    make, contour, mass = POCKETS[pocket]
    zeroth, first, second = an.jones_zener(contour, layer_spacing=D)
    rho0, _, rho2 = an.low_field_resistivity(zeroth, first, second)
    coefficient = rho2[0, 0] / rho0[0, 0]  # MR = coefficient · B² + O(B⁴)
    field = 1e-5 * mass / (E * TAU)
    errors = []
    for n in SIZES:
        rho = np.linalg.inv(sigma(make(n), [0.0, field]))
        errors.append(abs((rho[1, 0, 0] / rho[0, 0, 0] - 1) / (coefficient * field**2) - 1))
    print(f"{pocket}: MR/B^2 = {coefficient:.6e} /T^2; relative errors " + ", ".join(f"{e:.2e}" for e in errors)
          + f"; slope {slope(errors):.3f}")
    assert -2.1 <= slope(errors) <= -1.9


def lopsided_at(phi):
    """The lopsided pocket at any polar angles (the generators space them evenly)."""
    r, dr = lopsided_k(phi), lopsided_dk(phi)
    radial, angular = HBAR * r / ELECTRON_MASS, -HBAR * dr / ELECTRON_MASS
    return pd.DataFrame({"kx": r * np.cos(phi), "ky": r * np.sin(phi),
                         "vx": radial * np.cos(phi) - angular * np.sin(phi),
                         "vy": radial * np.sin(phi) + angular * np.cos(phi), "tau": lopsided_tau(phi)})


@pytest.mark.parametrize("spacing", ["jittered", "random"])
def test_irregularly_spaced_nodes_converge_to_the_reference(spacing):
    """Measured contours (ARPES, DFT) are rarely evenly spaced. Nodes jittered by up to ±45% of
    the spacing, or at sorted uniformly random angles, still converge to the reference: σ at
    B = 0 and ω_cτ = 1 within 1e-4 at N = 1024 (even spacing: ~5e-6), and at least 20× closer at
    N = 4096 than at N = 256."""
    exact, field = reference("lopsided", 1.0)
    rng = np.random.default_rng(0)
    errors = []
    for n in (256, 1024, 4096):
        if spacing == "jittered":
            phi = np.sort(2 * np.pi * (np.arange(n) + rng.uniform(-0.45, 0.45, n)) / n)
        else:
            phi = np.sort(rng.uniform(0.0, 2 * np.pi, n))
        actual = sigma(lopsided_at(phi), [0.0, field])
        errors.append(np.max(np.abs(actual - exact)) / np.max(np.abs(exact)))
    print(f"{spacing}: errors at N = 256, 1024, 4096: " + ", ".join(f"{e:.1e}" for e in errors))
    assert errors[1] <= 1e-4 and errors[2] <= errors[0] / 20


@pytest.mark.slow
@pytest.mark.parametrize("pocket", ["fourfold", "lopsided"])
@pytest.mark.parametrize("x", [1e-3, 1e-2])
def test_low_field_magnetoresistance_matches_the_reference(pocket, x):
    """At N = 1024 the low-field MR agrees with the brute-force reference to 1e-3. (A trapezoid
    outer sum is off by +120% to +170% at ω_cτ = 10⁻³ here.)"""
    exact, field = reference(pocket, x)
    mr_exact, _ = magnetoresistance_and_hall(exact, field)
    mr, _ = magnetoresistance_and_hall(sigma(POCKETS[pocket][0](1024), [0.0, field]), field)
    print(f"{pocket}, x = {x}: MR expected {mr_exact:.6e}, actual {mr:.6e}, rel error {abs(mr / mr_exact - 1):.1e}")
    assert abs(mr / mr_exact - 1) <= 1e-3


# ---------------------------------------------------------------------------
# Robustness and the zero-field branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("df, mass", [
    (gen.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=TAU), ELECTRON_MASS),
    (fourfold(512), MASS),
    (gen.tight_binding(512, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2), lattice_constant=3.87e-10,
                       hopping=0.25 * E, next_hopping=-0.0625 * E, third_hopping=0.02 * E, chemical_potential=0.0,
                       center=(np.pi / 3.87e-10, np.pi / 3.87e-10)), ELECTRON_MASS),
], ids=["circle", "fourfold", "tight-binding"])
@pytest.mark.parametrize("symmetrize", [True, False])
def test_finite_for_twelve_decades_of_field(df, mass, symmetrize):
    x = np.geomspace(1e-6, 1e6, 49)
    fields = np.concatenate([x, -x]) * mass / (E * TAU)
    result = sigma(df, fields, symmetrize=symmetrize)
    assert np.all(np.isfinite(result))
    assert np.all(result[:, 0, 0] > 0) and np.all(result[:, 1, 1] > 0)


@pytest.mark.parametrize("df", [
    gen.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=TAU),
    fourfold(512),
    gen.ellipse(512, k_fermi=7e9, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU, rotation=0.4),
], ids=["circle", "fourfold", "rotated-ellipse"])
def test_low_field_joins_the_zero_field_branch(df):
    """No jump between the B = 0 branch and the field branch.

    The even (symmetric) part of σ at 1e-6 T matches σ(0) to 1e-8 of |σ(0)|. The odd
    (Hall) part ½(σ_xy − σ_yx) is linear in B: σ_H/B agrees at 1e-4 T and 2e-4 T to 1e-8.
    (At 1e-6 T the Hall part is ~1e-8 of σ₀, which double precision resolves only to ~1e-8
    of itself, so its linearity is checked where it is resolved.)
    """
    zero, tiny, low, double = sigma(df, [0.0, 1e-6, 1e-4, 2e-4])
    scale = np.max(np.abs(zero))
    even = np.max(np.abs(0.5 * (tiny + tiny.T) - zero)) / scale
    hall = lambda m, b: 0.5 * (m[0, 1] - m[1, 0]) / b  # noqa: E731
    linear = abs(hall(low, 1e-4) / hall(double, 2e-4) - 1)
    print(f"even part at 1e-6 T vs sigma(0): {even:.1e}; Hall sigma_H/B at 1e-4 vs 2e-4 T: {linear:.1e}")
    assert even <= 1e-8 and linear <= 1e-8


def test_zero_field_branch_is_the_limit_of_the_recursion():
    """σ(0) is the same discrete sum the recursion tends to as B → 0, so there is no
    discretisation jump between the branches. (That σ(B) − σ(0) then grows as B² on the
    diagonal, not as |B|, is checked by the curvature test in M10.)"""
    df = fourfold(256)
    zero = sigma(df, 0.0, symmetrize=False)[0]
    tiny = sigma(df, 1e-9, symmetrize=False)[0]
    np.testing.assert_allclose(np.diag(tiny), np.diag(zero), rtol=1e-12)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError):
        sigma(fourfold(64), 1.0, backend="fortran")


def test_layer_spacing_is_required():
    df = fourfold(64)
    with pytest.raises(TypeError):
        conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"], 1.0)
