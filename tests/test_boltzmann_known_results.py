"""Known results: the Boltzmann solver must return textbook physics where the answer is exact.

Each test is parametrised over the backend, gives the expected formula and its source
in its docstring, and prints expected, actual and relative error. Unless noted,
N = 512, g_s = 2, d = 1 nm and ω_cτ ∈ {0.01, 0.1, 1, 10, 100}. A failure here is a
physics bug until proven otherwise: do not loosen a tolerance.
"""
import numpy as np
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import prepare_contour
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


# ---------------------------------------------------------------------------
# ρ, R_H, MR and several pockets, through the public API
# ---------------------------------------------------------------------------

def transport(dfs, fields, backend, **kwargs):
    """σ DataFrame plus ρ_xx, ρ_yy and R_H arrays."""
    kwargs.setdefault("layer_spacing", D)
    s = bz.conductivity(dfs, fields, backend=backend, **kwargs)
    rho = bz.resistivity(s)
    return s, rho["rho_xx"].to_numpy(), rho["rho_yy"].to_numpy(), bz.hall_coefficient(s).to_numpy()


def orbit_x(df, field):
    """ω_cτ averaged round the orbit: 2π|B| / Σ γ_n s_n, i.e. 2π over the damping per orbit.
    It is eBτ/m for a circle with constant τ."""
    c = prepare_contour(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"])
    return 2 * np.pi * np.abs(field) / np.sum(c.gamma * c.s)


def print_check(label, expected, actual):
    error = np.max(np.abs(np.asarray(actual) / np.asarray(expected) - 1))
    print(f"{label}: expected {np.ravel(expected)[0]:.8e}, actual {np.ravel(actual)[0]:.8e}, "
          f"max rel error {error:.1e}")
    return error


@pytest.mark.parametrize("backend", BACKENDS)
def test_k1_no_magnetoresistance_and_constant_hall_coefficient(backend):
    """K1: for an isotropic Drude metal ρ_xx(B) = m/(ne²τ) at every B (|MR| < 1e-4) and
    R_H = 1/(nq) = −1/(ne) at every B, including x = 0.01. R_H is unchanged when τ changes
    by 10³ or when m changes at fixed n. Tolerance 1e-4."""
    n = an.circle_density(K_F, D)
    fields = np.concatenate([[0.0], X * M / (E * TAU)])
    s, rho_xx, _, r_h = transport(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU), fields, backend)
    mr = bz.magnetoresistance(s).to_numpy()
    print(f"K1: max |MR| = {np.max(np.abs(mr)):.1e}")
    assert np.max(np.abs(mr)) < 1e-4
    assert print_check("K1 rho_xx = m/(ne^2 tau)", M / (n * E**2 * TAU), rho_xx) < 1e-4
    assert print_check("K1 R_H = -1/(ne)", -1 / (n * E), r_h[1:]) < 1e-4
    for label, df in [("tau x 1e3", gen.circle(N, k_fermi=K_F, mass=M, tau=1e3 * TAU)),
                      ("m x 3 at fixed n", gen.circle(N, k_fermi=K_F, mass=3 * M, tau=TAU))]:
        other = transport(df, fields[1:], backend)[3]
        assert print_check(f"K1 R_H with {label}", -1 / (n * E), other) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k2_hole_hall_coefficient(backend):
    """K2: a hole circle (v inwards) has R_H = +1/(ne) at every B. Tolerance 1e-4."""
    n = an.circle_density(K_F, D)
    r_h = transport(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU, carrier="hole"), X * M / (E * TAU), backend)[3]
    assert print_check("K2 R_H = +1/(ne)", 1 / (n * E), r_h) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k3_hall_and_magnetoresistance_at_zero_field(backend):
    """K3: R_H is NaN at B = 0 (undefined), and MR(0) = 0 exactly."""
    s = bz.conductivity(gen.circle(N, k_fermi=K_F, mass=M, tau=TAU), [0.0, 1.0], layer_spacing=D, backend=backend)
    r_h, mr = bz.hall_coefficient(s), bz.magnetoresistance(s)
    print(f"K3: R_H(0) = {r_h.iloc[0]}, MR(0) = {mr.iloc[0]}")
    assert np.isnan(r_h.iloc[0]) and mr.iloc[0] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_k4_no_magnetoresistance_along_either_axis(backend):
    """K4: for the anisotropic-mass ellipse ρ_xx = m_x/(ne²τ) and ρ_yy = m_y/(ne²τ) at every B,
    and R_H = −1/(ne). Tolerance 1e-4."""
    mx, my = M, 4 * M
    n = an.circle_density(K_F, D)
    fields = np.concatenate([[0.0], X * np.sqrt(mx * my) / (E * TAU)])
    _, rho_xx, rho_yy, r_h = transport(gen.ellipse(N, k_fermi=K_F, mass_x=mx, mass_y=my, tau=TAU), fields, backend)
    assert print_check("K4 rho_xx = m_x/(ne^2 tau)", mx / (n * E**2 * TAU), rho_xx) < 1e-4
    assert print_check("K4 rho_yy = m_y/(ne^2 tau)", my / (n * E**2 * TAU), rho_yy) < 1e-4
    assert print_check("K4 R_H = -1/(ne)", -1 / (n * E), r_h[1:]) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k5_compensated_two_band(backend):
    """K5: electron circle (m_e, τ) plus hole circle (2m_e, τ/2) with n_e = n_h = n. σ is the
    sum of the two Drude tensors; ρ_xx(B) = (1 + μ_eμ_h B²)/(ne(μ_e+μ_h)) grows as B² without
    saturating; R_H = (μ_h − μ_e)/(ne(μ_e+μ_h)) at every B (two-band Drude model). Tolerance 1e-4."""
    n = an.circle_density(K_F, D)
    mu_e, mu_h = E * TAU / M, E * (TAU / 2) / (2 * M)
    pockets = [gen.circle(N, k_fermi=K_F, mass=M, tau=TAU),
               gen.circle(N, k_fermi=K_F, mass=2 * M, tau=TAU / 2, carrier="hole")]
    fields = X / mu_e
    s, rho_xx, _, r_h = transport(pockets, fields, backend)
    expected = an.two_band(fields, n_e=n, mu_e=mu_e, n_h=n, mu_h=mu_h)
    actual = s[["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]].to_numpy().reshape(-1, 2, 2)
    assert report("K5", X, actual, expected).max() < 1e-4
    assert print_check("K5 rho_xx", (1 + mu_e * mu_h * fields**2) / (n * E * (mu_e + mu_h)), rho_xx) < 1e-4
    assert print_check("K5 R_H", (mu_h - mu_e) / (n * E * (mu_e + mu_h)), r_h) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k6_uncompensated_two_band(backend):
    """K6: electron circle (n_e, μ_e) plus hole circle (n_h = n_e/2, μ_h = μ_e/2). ρ = (σ_e + σ_h)⁻¹
    from the analytic Drude tensors at every B; R_H(0) = (n_hμ_h² − n_eμ_e²)/(e(n_hμ_h + n_eμ_e)²)
    (at μB = 10⁻³) and R_H(∞) = 1/((n_h − n_e)e) at μB = 10³. Tolerance 1e-4."""
    k_h = K_F / np.sqrt(2)
    n_e, n_h = an.circle_density(K_F, D), an.circle_density(k_h, D)
    mu_e, mu_h = E * TAU / M, E * TAU / (2 * M)
    params = dict(n_e=n_e, mu_e=mu_e, n_h=n_h, mu_h=mu_h)
    pockets = [gen.circle(N, k_fermi=K_F, mass=M, tau=TAU),
               gen.circle(N, k_fermi=k_h, mass=2 * M, tau=TAU, carrier="hole")]
    fields = np.concatenate([X / mu_e, [1e-3 / mu_e, 1e3 / mu_h]])
    s, _, _, r_h = transport(pockets, fields, backend)
    rho = bz.resistivity(s)[["rho_xx", "rho_xy", "rho_yx", "rho_yy"]].to_numpy().reshape(-1, 2, 2)
    expected_rho = an.resistivity(an.two_band(fields[:5], **params))
    assert report("K6 rho", X, rho[:5], expected_rho).max() < 1e-4
    low = (n_h * mu_h**2 - n_e * mu_e**2) / (E * (n_h * mu_h + n_e * mu_e) ** 2)
    assert print_check("K6 R_H(0) at mu B = 1e-3", low, r_h[5]) < 1e-4
    assert print_check("K6 R_H(inf) at mu B = 1e3", 1 / ((n_h - n_e) * E), r_h[6]) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_k7_anisotropic_tau_hall_and_magnetoresistance(backend):
    """K7: circle with 1/τ = (1/τ₀)(1 + 0.6 cos4φ), N = 1024, x̄ = ω_c⟨τ⟩. Ong: the low-field
    R_H(x̄ = 10⁻³) = (1/nq)⟨τ²⟩/⟨τ⟩² = 1.25/nq (N. P. Ong, PRB 43, 193 (1991)); at x̄ = 10³,
    R_H = 1/nq and ρ_xx/ρ_xx(0) = ⟨1/τ⟩⟨τ⟩ = 1.25 (MR saturates at 0.25). MR > 0 and rises
    monotonically. Tolerance 1e-3."""
    df = gen.circle(1024, k_fermi=K_F, mass=M, tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6))
    nq = an.circle_density(K_F, D) * (-E)
    xbar = np.array([1e-3, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3])
    fields = np.concatenate([[0.0], xbar * M / (E * 1.25 * TAU)])
    s, rho_xx, _, r_h = transport(df, fields, backend)
    mr = bz.magnetoresistance(s).to_numpy()[1:]
    assert print_check("K7 R_H(1e-3) = 1.25/nq", 1.25 / nq, r_h[1]) < 1e-3
    assert print_check("K7 R_H(1e3) = 1/nq", 1 / nq, r_h[-1]) < 1e-3
    assert print_check("K7 rho_xx(1e3)/rho_xx(0) = 1.25", 1.25, rho_xx[-1] / rho_xx[0]) < 1e-3
    print("K7 MR:", ", ".join(f"{x:g}: {m:.4e}" for x, m in zip(xbar, mr)))
    assert np.all(mr > 0) and np.all(np.diff(mr) > 0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("pocket", ["hole-M", "electron-gamma"])
def test_k8_arbitrary_closed_pocket(backend, pocket):
    """K8: tight-binding pocket with hot-spot τ(φ). At x̄ = 10³ (x̄ = 2π|B|/∮γ ds, ω_cτ averaged
    round the orbit) R_H = 1/(nq) with n from the enclosed area (Lifshitz–Azbel–Kaganov), and
    MR has saturated: ρ_xx changes by < 1e-3 between x̄ = 10³ and 2×10³. Tolerance 1e-3."""
    a = 3.87e-10
    centre, mu, sign = ((np.pi / a, np.pi / a), 0.0, +1) if pocket == "hole-M" else ((0.0, 0.0), -0.4 * E, -1)
    df = gen.tight_binding(N, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2), lattice_constant=a,
                           hopping=0.25 * E, next_hopping=-0.0625 * E, third_hopping=0.02 * E,
                           chemical_potential=mu, center=centre)
    per_tesla = orbit_x(df, 1.0)
    _, rho_xx, _, r_h = transport(df, np.array([1e3, 2e3]) / per_tesla, backend)
    n = bz.carrier_density(df, layer_spacing=D)
    assert print_check(f"K8 {pocket}: R_H(1e3) = 1/(nq)", 1 / (n * sign * E), r_h[0]) < 1e-3
    change = abs(rho_xx[1] / rho_xx[0] - 1)
    print(f"K8 {pocket}: rho_xx change between xbar = 1e3 and 2e3: {change:.1e}")
    assert change < 1e-3
