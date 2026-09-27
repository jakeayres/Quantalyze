"""M3: the plain-NumPy O(N) kernel.

Checks the exponential-integrator weights φ₁, φ₂ against high-precision values,
that the kernel converges to the brute-force reference as O(N⁻²), that it stays
finite over twelve decades of ω_cτ, and that the field branch joins the
zero-field branch continuously.
"""
from decimal import Decimal, getcontext

import numpy as np
import pytest

from quantalyze.beta.boltzmann import _kernel_py as kp
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

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


def exact_phi(z):
    getcontext().prec = 60
    x = Decimal(float(z))
    e = (-x).exp()
    return float((1 - e) / x), float((1 - (1 + x) * e) / (x * x))


# ---------------------------------------------------------------------------
# φ₁(z) = (1 − e^{−z})/z and φ₂(z) = (1 − (1+z)e^{−z})/z²
# ---------------------------------------------------------------------------

def test_phi_accuracy_against_high_precision():
    z = np.concatenate([np.geomspace(1e-14, 1e3, 400), np.geomspace(0.9e-2, 1.1e-2, 50),
                        np.geomspace(0.9, 1.1, 50)])
    expected = np.array([exact_phi(v) for v in z])
    error_1 = np.max(np.abs(kp.phi1(z) / expected[:, 0] - 1))
    error_2 = np.max(np.abs(kp.phi2(z) / expected[:, 1] - 1))
    print(f"max relative error over z in [1e-14, 1e3]: phi1 {error_1:.1e}, phi2 {error_2:.1e}")
    assert error_1 < 1e-14 and error_2 < 1e-14


@pytest.mark.parametrize("switch", [1e-2, 1.0])
def test_phi_continuous_across_branch_switches(switch):
    below, at = np.nextafter(switch, 0.0), switch
    for name, f in (("phi1", kp.phi1), ("phi2", kp.phi2)):
        jump = abs(f(np.array([below]))[0] / f(np.array([at]))[0] - 1)
        print(f"{name} jump across z = {switch:g}: {jump:.1e}")
        assert jump <= 1e-14


def test_phi_limits_are_finite():
    z = np.array([0.0, 1e-300, 1e-14, 1e300, np.finfo(float).max])
    p1, p2 = kp.phi1(z), kp.phi2(z)
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))
    assert p1[0] == 1.0 and p2[0] == 0.5
    assert p1[3] == pytest.approx(1e-300, rel=1e-15) and p2[3] == 0.0


# ---------------------------------------------------------------------------
# Convergence to the reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
def test_converges_to_the_reference_as_n_to_the_minus_two(x):
    """Error against the brute-force reference on the fourfold contour (anisotropic τ),
    N = 64…1024: the fitted log–log slope must lie in [−2.1, −1.9]."""
    field = x * MASS / (E * TAU)
    exact = ref.sigma(ref.ParametricContour.polar(k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=MASS,
                                                  tau=fourfold_tau, carrier="hole"), field, layer_spacing=D)[0]
    sizes = np.array([64, 128, 256, 512, 1024])
    errors = np.array([np.max(np.abs(sigma(fourfold(n), field)[0] - exact)) / np.max(np.abs(exact))
                       for n in sizes])
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    print(f"x = {x}: errors " + ", ".join(f"{e:.2e}" for e in errors) + f"; slope {slope:.3f}")
    assert -2.1 <= slope <= -1.9


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
    """σ(0) is the same discrete sum the recursion tends to as B → 0, so σ(B) − σ(0)
    is pure physics (O(B) off-diagonal, O(B²) diagonal), with no discretisation jump."""
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
