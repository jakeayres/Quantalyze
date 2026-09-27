"""Known results: the Boltzmann solver must return textbook physics where the answer is exact.

Each test is parametrised over the backend, gives the expected formula and its source
in its docstring, and prints expected, actual and relative error. Unless noted,
N = 512, g_s = 2, d = 1 nm and ω_cτ ∈ {0.01, 0.1, 1, 10, 100}. A failure here is a
physics bug until proven otherwise: do not loosen a tolerance.

Tests that need ρ, R_H or several pockets arrive with the public response functions;
this file currently holds the σ-level results.
"""
import numpy as np
import pytest

from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

E = ELEMENTARY_CHARGE
N = 512
D = 1e-9  # m
K_F = 7.0e9  # m⁻¹
TAU = 1e-13  # s
M = ELECTRON_MASS
X = np.array([0.01, 0.1, 1.0, 10.0, 100.0])

BACKENDS = ["python", "numba"]


def sigma(df, fields, backend, **kwargs):
    kwargs.setdefault("layer_spacing", D)
    return conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"], fields, backend=backend, **kwargs)


def report(label, x, actual, expected):
    """Print and return the per-component relative error, shape (nB, 2, 2)."""
    error = np.abs(actual - expected) / np.abs(expected)
    for xi, a, e, err in zip(np.atleast_1d(x), actual, expected, error):
        print(f"{label} x = {xi:g}: expected xx {e[0, 0]:.8e} xy {e[0, 1]:.8e}; "
              f"actual xx {a[0, 0]:.8e} xy {a[0, 1]:.8e}; max rel error {err.max():.1e}")
    return error


def normwise(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def fourfold(n=N, tau=TAU):
    return gen.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * M,
                     tau=lambda p: sc.cos4phi(p, tau, anisotropy=0.3), carrier="hole")


@pytest.mark.parametrize("backend", BACKENDS)
def test_k1_isotropic_drude_electrons(backend):
    """K1: σ_xx = σ_yy = σ₀/(1+x²), σ_xy = −σ_yx = sσ₀x/(1+x²), with σ₀ = ne²τ/m,
    x = eBτ/m and s = −1 (Drude; Ashcroft & Mermin ch. 1). Tolerance 1e-4."""
    fields = X * M / (E * TAU)
    actual = sigma(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU), fields, backend)
    expected = an.drude_circle(fields, density=an.circle_density(K_F, D), mass=M, tau=TAU)
    assert report("K1", X, actual, expected).max() < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k2_isotropic_drude_holes(backend):
    """K2: a circle with v pointing inwards is K1 with s = +1. An electron circle with
    charge = +e gives exactly the same σ as the hole circle. Tolerance 1e-4; equivalence exact."""
    fields = X * M / (E * TAU)
    hole = sigma(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU, carrier="hole"), fields, backend)
    expected = an.drude_circle(fields, density=an.circle_density(K_F, D), mass=M, tau=TAU, charge=+E)
    assert report("K2", X, hole, expected).max() < 1e-4
    positive = sigma(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU), fields, backend, charge=+E)
    print(f"hole circle vs electron circle with q = +e: max |d sigma| / max |sigma| = {normwise(hole, positive):.1e}")
    assert normwise(hole, positive) <= 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_k3_zero_field(backend):
    """K3: σ(0) = σ₀𝟙 (tolerance 1e-4) and σ_xy(0) = 0 to 1e-12 σ₀; σ(10⁻⁶ T) joins σ(0)
    continuously (diagonal to 1e-8)."""
    df = gen.circle(N, k_fermi=K_F, mass=M, tau=TAU)
    zero, tiny = sigma(df, [0.0, 1e-6], backend)
    sigma_0 = an.circle_density(K_F, D) * E**2 * TAU / M
    error = abs(zero[0, 0] / sigma_0 - 1)
    print(f"K3: sigma_xx(0) expected {sigma_0:.8e}, actual {zero[0, 0]:.8e}, rel error {error:.1e}; "
          f"sigma_xy(0)/sigma_0 = {zero[0, 1] / sigma_0:.1e}")
    assert error < 1e-4 and abs(zero[1, 1] / sigma_0 - 1) < 1e-4
    assert abs(zero[0, 1]) <= 1e-12 * sigma_0 and abs(zero[1, 0]) <= 1e-12 * sigma_0
    assert abs(tiny[0, 0] / zero[0, 0] - 1) <= 1e-8


@pytest.mark.parametrize("backend", BACKENDS)
def test_k4_anisotropic_mass(backend):
    """K4: ellipse with m_y = 4m_x, x = eBτ/√(m_x m_y): σ_xx = ne²τ/m_x/(1+x²),
    σ_yy = ne²τ/m_y/(1+x²), σ_xy = sne²τx/(√(m_x m_y)(1+x²)) (anisotropic-mass Drude).
    Tolerance 1e-4. Rotating the ellipse by 30° gives R σ Rᵀ exactly."""
    mx, my = M, 4 * M
    fields = X * np.sqrt(mx * my) / (E * TAU)
    actual = sigma(gen.ellipse(N, k_fermi=K_F, mass_x=mx, mass_y=my, tau=TAU), fields, backend)
    expected = an.drude_ellipse(fields, density=an.circle_density(K_F, D), mass_x=mx, mass_y=my, tau=TAU)
    assert report("K4", X, actual, expected).max() < 1e-4
    rotated = sigma(gen.ellipse(N, k_fermi=K_F, mass_x=mx, mass_y=my, tau=TAU, rotation=np.pi / 6), fields, backend)
    error = normwise(rotated, an.rotate(actual, np.pi / 6))
    print(f"K4 rotation by 30 deg: max |d sigma| / max |sigma| = {error:.1e}")
    assert error <= 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_k7_anisotropic_tau_zero_field(backend):
    """K7 (σ part): 1/τ = (1/τ₀)(1 + 0.6 cos4φ) on a circle, N = 1024: σ(0) = (ne²/m)⟨τ⟩𝟙
    with ⟨τ⟩ = τ₀/√(1 − 0.6²). Tolerance 1e-3."""
    df = gen.circle(1024, k_fermi=K_F, mass=M, tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6))
    zero = sigma(df, 0.0, backend)[0]
    expected = an.circle_density(K_F, D) * E**2 * (TAU / np.sqrt(1 - 0.36)) / M
    error = normwise(zero, expected * np.eye(2))
    print(f"K7: sigma(0) expected {expected:.8e}, actual {zero[0, 0]:.8e}, rel error {error:.1e}")
    assert error < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_k9_omega_c_tau_scaling(backend, scale):
    """K9: σ depends on B and τ only through ω_cτ, so λσ(λB, τ/λ) = σ(B, τ); and σ(0) ∝ τ. Exact."""
    fields = np.array([0.0, 1.0, 10.0, 100.0])
    base = sigma(fourfold(), fields, backend)
    scaled = sigma(fourfold(tau=TAU / scale), scale * fields, backend)
    field_error = normwise(scale * scaled[1:], base[1:])
    zero_error = normwise(scale * scaled[0], base[0])
    print(f"K9 lambda = {scale}: field {field_error:.1e}, sigma(0) prop. to tau {zero_error:.1e}")
    assert field_error <= 1e-12 and zero_error <= 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_k9_onsager_and_parity(backend):
    """K9: symmetrised, σ_ij(−B) = σ_ji(B) exactly (Onsager). Unsymmetrised on the K1 circle,
    σ_xx is even and σ_xy odd in B (the circle's mirror symmetry)."""
    fields = np.array([0.3, 3.0, 30.0])
    sym = sigma(fourfold(), np.concatenate([fields, -fields]), backend)
    onsager = normwise(sym[3:], np.transpose(sym[:3], (0, 2, 1)))
    raw = sigma(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU), np.concatenate([fields, -fields]), backend,
                symmetrize=False)
    even = normwise(raw[3:, 0, 0], raw[:3, 0, 0])
    odd = normwise(-raw[3:, 0, 1], raw[:3, 0, 1])
    print(f"K9 Onsager {onsager:.1e}; unsymmetrised circle: sigma_xx even {even:.1e}, sigma_xy odd {odd:.1e}")
    assert onsager <= 1e-12 and even <= 1e-12 and odd <= 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_k10_invariances(backend):
    """K10: σ is unchanged by rotating the start index or reversing the input (1e-12),
    rotating the contour by α gives R σ Rᵀ (1e-12), and resampling N → 2N changes σ only
    by the O(N⁻²) discretisation error."""
    fields = np.array([0.0, 0.3, 3.0, 30.0])
    df = fourfold()
    base = sigma(df, fields, backend)
    rolled = sigma(df.iloc[np.roll(np.arange(N), 101)], fields, backend)
    reversed_ = sigma(df.iloc[::-1], fields, backend)
    alpha = 0.3
    c, s = np.cos(alpha), np.sin(alpha)
    turned = df.assign(kx=c * df.kx - s * df.ky, ky=s * df.kx + c * df.ky,
                       vx=c * df.vx - s * df.vy, vy=s * df.vx + c * df.vy)
    rotated = sigma(turned, fields, backend)
    finer = sigma(fourfold(2 * N), fields, backend)
    errors = {
        "start index": normwise(rolled, base),
        "reversed": normwise(reversed_, base),
        "rotation": normwise(rotated, an.rotate(base, alpha)),
        "N -> 2N": normwise(finer, base),
    }
    print("K10: " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))
    assert errors["start index"] <= 1e-12 and errors["reversed"] <= 1e-12 and errors["rotation"] <= 1e-12
    assert errors["N -> 2N"] < 1e-4
