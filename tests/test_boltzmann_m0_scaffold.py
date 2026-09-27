"""M0 scaffold for quantalyze.beta.boltzmann.

Checks that the test contours are what they claim to be (velocities are ∇ε/ħ,
nodes lie on ε = 0, enclosed areas are right), that the scattering models,
unit converters and analytic helpers are correct, and that FermiSurface still
produces exactly what it did before the module became a package.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann import units
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

from quantalyze.beta.boltzmann import _analytic as an

E = ELEMENTARY_CHARGE
N = 512
K_F = 7.0e9  # m⁻¹
TAU = 1.0e-13  # s

# Polar pockets: a rounded square, and a lopsided shape with no mirror symmetry
K00, K40 = 7.35e9, 0.25e9  # m⁻¹
POLAR_MASS = 5 * ELECTRON_MASS
FOURFOLD = lambda phi: K00 - K40 * np.cos(4 * phi)  # noqa: E731
LOPSIDED = lambda phi: K_F * (1 + 0.1 * np.cos(2 * phi) + 0.05 * np.sin(3 * phi))  # noqa: E731

# A user-style band: parabolic plus a fourfold quartic term, SI
C2 = HBAR**2 / (2 * ELECTRON_MASS)  # J m²
C4 = 2e-59  # J m⁴
E_F = 1.9e-19  # J

# Tight-binding parameters (square lattice), SI
A = 3.87e-10  # m
T1 = 0.25 * E  # J
T2 = -0.0625 * E
T3 = 0.02 * E
MU_ELECTRON = -0.4 * E  # closed electron pocket about Γ
MU_HOLE = 0.0  # closed hole pocket about (π/a, π/a)

# Open sheets
K0 = 5.0e9  # m⁻¹
V0 = 2.0e5  # m/s
G = 2 * np.pi / A  # m⁻¹
WARPING = 0.1e9  # m⁻¹


# ---------------------------------------------------------------------------
# Dispersions, written here independently of the generators
# ---------------------------------------------------------------------------

def eps_circle(kx, ky, *, sign):
    return sign * HBAR**2 * (kx**2 + ky**2 - K_F**2) / (2 * ELECTRON_MASS)


def eps_ellipse(kx, ky, *, mass_x, mass_y, rotation):
    c, s = np.cos(rotation), np.sin(rotation)
    qx, qy = c * kx + s * ky, -s * kx + c * ky  # back into the principal frame
    fermi_energy = HBAR**2 * K_F**2 / (2 * np.sqrt(mass_x * mass_y))
    return HBAR**2 * qx**2 / (2 * mass_x) + HBAR**2 * qy**2 / (2 * mass_y) - fermi_energy


def eps_polar(kx, ky, *, sign, k_fermi, mass=POLAR_MASS):
    phi = np.arctan2(ky, kx)
    return sign * HBAR**2 * (kx**2 + ky**2 - k_fermi(phi) ** 2) / (2 * mass)


def eps_quartic(kx, ky):
    return C2 * (kx**2 + ky**2) + C4 * (kx**4 + ky**4) - E_F


def grad_quartic(kx, ky):
    return 2 * C2 * kx + 4 * C4 * kx**3, 2 * C2 * ky + 4 * C4 * ky**3


def eps_tight_binding(kx, ky, *, mu):
    x, y = kx * A, ky * A
    return (
        -2 * T1 * (np.cos(x) + np.cos(y))
        - 4 * T2 * np.cos(x) * np.cos(y)
        - 2 * T3 * (np.cos(2 * x) + np.cos(2 * y))
        - mu
    )


def eps_sheet(kx, ky, *, side, warping):
    return HBAR * V0 * (side * kx - K0 - warping * np.cos(2 * np.pi * ky / G))


def finite_difference_velocity(eps, kx, ky, step):
    """Central-difference ∇ε/ħ (m/s)."""
    vx = (eps(kx + step, ky) - eps(kx - step, ky)) / (2 * step * HBAR)
    vy = (eps(kx, ky + step) - eps(kx, ky - step)) / (2 * step * HBAR)
    return vx, vy


# Each case: (id, contour DataFrame, dispersion, finite-difference step, pocket centre)
def _cases():
    rot = np.pi / 6
    sheets_flat = gen.open_sheets(N, k0=K0, velocity=V0, tau=TAU, period=G)
    sheets_warped = gen.open_sheets(N, k0=K0, velocity=V0, tau=TAU, period=G, warping=WARPING)
    return [
        ("circle-electron", gen.circle(N, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU),
         lambda x, y: eps_circle(x, y, sign=+1), 1e-6 * K_F, (0.0, 0.0)),
        ("circle-hole", gen.circle(N, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU, carrier="hole"),
         lambda x, y: eps_circle(x, y, sign=-1), 1e-6 * K_F, (0.0, 0.0)),
        ("ellipse-rotated", gen.ellipse(N, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS,
                                        tau=TAU, rotation=rot),
         lambda x, y: eps_ellipse(x, y, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, rotation=rot),
         1e-6 * K_F, (0.0, 0.0)),
        ("polar-fourfold-hole", gen.polar(N, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU, carrier="hole"),
         lambda x, y: eps_polar(x, y, sign=-1, k_fermi=FOURFOLD), 1e-6 * K_F, (0.0, 0.0)),
        ("polar-fourfold-electron", gen.polar(N, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU),
         lambda x, y: eps_polar(x, y, sign=+1, k_fermi=FOURFOLD), 1e-6 * K_F, (0.0, 0.0)),
        ("polar-lopsided", gen.polar(N, k_fermi=LOPSIDED, mass=POLAR_MASS, tau=TAU),
         lambda x, y: eps_polar(x, y, sign=+1, k_fermi=LOPSIDED), 1e-6 * K_F, (0.0, 0.0)),
        ("from-dispersion-quartic", _quartic(), eps_quartic, 1e-6 * K_F, (0.0, 0.0)),
        ("tight-binding-gamma", _tb_gamma(), lambda x, y: eps_tight_binding(x, y, mu=MU_ELECTRON),
         1e-6 / A, (0.0, 0.0)),
        ("tight-binding-m", _tb_m(), lambda x, y: eps_tight_binding(x, y, mu=MU_HOLE),
         1e-6 / A, (np.pi / A, np.pi / A)),
        ("sheet-flat-plus", sheets_flat[0], lambda x, y: eps_sheet(x, y, side=+1, warping=0.0),
         1e-6 * K0, None),
        ("sheet-flat-minus", sheets_flat[1], lambda x, y: eps_sheet(x, y, side=-1, warping=0.0),
         1e-6 * K0, None),
        ("sheet-warped-plus", sheets_warped[0], lambda x, y: eps_sheet(x, y, side=+1, warping=WARPING),
         1e-6 * K0, None),
        ("sheet-warped-minus", sheets_warped[1], lambda x, y: eps_sheet(x, y, side=-1, warping=WARPING),
         1e-6 * K0, None),
    ]


def _quartic(n=N, tau=TAU):
    return gen.from_dispersion(n, energy=eps_quartic, gradient=grad_quartic, tau=tau, max_radius=2e10)


def _tb_gamma(n=N, tau=TAU):
    return gen.tight_binding(n, tau=tau, lattice_constant=A, hopping=T1, next_hopping=T2,
                             third_hopping=T3, chemical_potential=MU_ELECTRON)


def _tb_m(n=N, tau=TAU):
    return gen.tight_binding(n, tau=tau, lattice_constant=A, hopping=T1, next_hopping=T2,
                             third_hopping=T3, chemical_potential=MU_HOLE,
                             center=(np.pi / A, np.pi / A))


CASES = _cases()
CASE_IDS = [c[0] for c in CASES]


# ---------------------------------------------------------------------------
# Generators: velocities, on-surface, orientation, format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, df, eps, step, center", CASES, ids=CASE_IDS)
def test_generator_velocity_is_gradient_of_dispersion(name, df, eps, step, center):
    kx, ky = df["kx"].to_numpy(), df["ky"].to_numpy()
    vx_fd, vy_fd = finite_difference_velocity(eps, kx, ky, step)
    speed = np.hypot(vx_fd, vy_fd)
    error = np.max(np.hypot(df["vx"] - vx_fd, df["vy"] - vy_fd) / speed)
    print(f"{name}: max |v - grad(eps)/hbar| / |v| = {error:.2e}")
    assert error < 1e-6


@pytest.mark.parametrize("name, df, eps, step, center", CASES, ids=CASE_IDS)
def test_generator_nodes_lie_on_fermi_surface(name, df, eps, step, center):
    kx, ky = df["kx"].to_numpy(), df["ky"].to_numpy()
    speed = np.hypot(df["vx"], df["vy"]).to_numpy()
    if center is None:
        scale = K0
    else:
        scale = np.hypot(kx - center[0], ky - center[1])
    # |ε| relative to the energy change ħ|v|·|k| over the pocket's own size
    error = np.max(np.abs(eps(kx, ky)) / (HBAR * speed * scale))
    print(f"{name}: max |eps| / (hbar |v| |k|) = {error:.2e}")
    assert error < 1e-12


@pytest.mark.parametrize("name, df, eps, step, center", CASES, ids=CASE_IDS)
def test_generator_format(name, df, eps, step, center):
    assert list(df.columns) == ["kx", "ky", "vx", "vy", "tau"]
    assert len(df) == N
    assert all(df[c].dtype == np.float64 for c in df.columns)
    assert np.all(np.isfinite(df.to_numpy()))
    assert np.all(df["tau"] == TAU)


@pytest.mark.parametrize("df, sign", [
    (gen.circle(64, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU), +1),
    (gen.circle(64, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU, carrier="hole"), -1),
    (gen.ellipse(64, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU), +1),
    (gen.ellipse(64, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU,
                 carrier="hole"), -1),
    (gen.polar(64, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU, carrier="hole"), -1),
    (gen.polar(64, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU), +1),
    (_quartic(64), +1),
])
def test_generator_velocity_points_out_for_electrons_and_in_for_holes(df, sign):
    radial = df["kx"] * df["vx"] + df["ky"] * df["vy"]
    assert np.all(sign * radial > 0)


def test_tight_binding_pockets_have_the_right_character():
    gamma = _tb_gamma(64)
    m = _tb_m(64)
    assert np.all(gamma["kx"] * gamma["vx"] + gamma["ky"] * gamma["vy"] > 0)  # electron-like
    qx, qy = m["kx"] - np.pi / A, m["ky"] - np.pi / A
    assert np.all(qx * m["vx"] + qy * m["vy"] < 0)  # hole-like


def test_generators_take_tau_as_a_function_of_angle():
    df = gen.circle(N, k_fermi=K_F, mass=ELECTRON_MASS,
                    tau=lambda phi: sc.cos4phi(phi, TAU, anisotropy=0.6))
    phi = np.arctan2(df["ky"], df["kx"])
    expected = TAU / (1 + 0.6 * np.cos(4 * phi))
    np.testing.assert_allclose(df["tau"], expected, rtol=1e-14)


def test_tight_binding_tau_uses_angle_about_the_pocket_centre():
    df = gen.tight_binding(64, tau=lambda phi: TAU * (2 + np.cos(phi)), lattice_constant=A,
                           hopping=T1, next_hopping=T2, third_hopping=T3,
                           chemical_potential=MU_HOLE, center=(np.pi / A, np.pi / A))
    phi = np.arctan2(df["ky"] - np.pi / A, df["kx"] - np.pi / A)
    np.testing.assert_allclose(df["tau"], TAU * (2 + np.cos(phi)), rtol=1e-14)


def test_open_sheets_span_one_period():
    plus, minus = gen.open_sheets(N, k0=K0, velocity=V0, tau=TAU, period=G, warping=WARPING)
    for df in (plus, minus):
        ky = df["ky"].to_numpy()
        np.testing.assert_allclose(np.diff(ky), G / N, rtol=1e-12)
        assert ky[0] == pytest.approx(-G / 2, rel=1e-15)
        assert ky[-1] + G / N == pytest.approx(G / 2, rel=1e-12)  # last segment ends at k₀ + G
    assert np.all(plus["kx"] > 0) and np.all(plus["vx"] == V0)
    assert np.all(minus["kx"] < 0) and np.all(minus["vx"] == -V0)


def test_from_dispersion_accepts_a_sample_exactly_on_the_fermi_surface():
    """With k_F = max_radius/2 one of the ray samples lands exactly on ε = 0; that is one
    crossing, not two (it used to be rejected as not star-shaped)."""
    c2 = HBAR**2 / (2 * ELECTRON_MASS)
    df = gen.from_dispersion(128, tau=TAU, max_radius=2 * K_F,
                             energy=lambda kx, ky: c2 * (kx**2 + ky**2 - K_F**2),
                             gradient=lambda kx, ky: (2 * c2 * kx, 2 * c2 * ky))
    np.testing.assert_allclose(np.hypot(df.kx, df.ky), K_F, rtol=1e-15)


def test_generators_reject_bad_arguments():
    with pytest.raises(ValueError):
        gen.circle(64, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU, carrier="positron")
    with pytest.raises(ValueError):  # |anisotropy| ≥ 1 would make τ negative
        sc.cos4phi(np.zeros(3), TAU, anisotropy=1.0)
    with pytest.raises(ValueError):  # μ above the saddle point: the Γ pocket is open
        gen.tight_binding(64, tau=TAU, lattice_constant=A, hopping=T1, next_hopping=T2,
                          third_hopping=T3, chemical_potential=-0.2 * E)
    with pytest.raises(ValueError):  # μ below the band bottom: no pocket at all
        gen.tight_binding(64, tau=TAU, lattice_constant=A, hopping=T1, next_hopping=T2,
                          third_hopping=T3, chemical_potential=-2.0 * E)
    with pytest.raises(ValueError, match="star-shaped"):  # an annulus: each ray crosses twice
        gen.from_dispersion(64, tau=TAU, max_radius=3 * K_F,
                            energy=lambda kx, ky: (kx**2 + ky**2 - K_F**2) * (kx**2 + ky**2 - 4 * K_F**2),
                            gradient=lambda kx, ky: (kx, ky))
    with pytest.raises(ValueError, match="max_radius"):  # pocket larger than the search radius
        gen.from_dispersion(64, energy=eps_quartic, gradient=grad_quartic, tau=TAU, max_radius=1e9)
    with pytest.raises(ValueError):  # k_F must stay positive
        gen.polar(64, k_fermi=lambda phi: K_F * np.cos(phi), mass=POLAR_MASS, tau=TAU)


# ---------------------------------------------------------------------------
# Enclosed areas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("df", [
    gen.circle(N, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU),
    gen.circle(N, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU, carrier="hole"),
    gen.circle(16, k_fermi=K_F, mass=ELECTRON_MASS, tau=TAU),
    gen.ellipse(N, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU,
                rotation=np.pi / 6),
], ids=["circle-electron", "circle-hole", "circle-N16", "ellipse-rotated"])
def test_circle_and_ellipse_enclose_pi_kf_squared(df):
    area = an.spectral_area(df["kx"], df["ky"])
    error = abs(area / (np.pi * K_F**2) - 1)
    print(f"area / (pi k_F^2) - 1 = {error:.2e}")
    assert error < 1e-12


@pytest.mark.parametrize("k_fermi, expected", [
    (FOURFOLD, np.pi * (K00**2 + K40**2 / 2)),  # ½∮(k₀ − k₄ cos4φ)² dφ
    (LOPSIDED, np.pi * K_F**2 * (1 + 0.1**2 / 2 + 0.05**2 / 2)),
], ids=["fourfold", "lopsided"])
def test_polar_encloses_its_analytic_area(k_fermi, expected):
    df = gen.polar(N, k_fermi=k_fermi, mass=POLAR_MASS, tau=TAU)
    error = abs(an.spectral_area(df["kx"], df["ky"]) / expected - 1)
    print(f"polar area relative error = {error:.2e}")
    assert error < 1e-12


def test_polar_spectral_derivative_matches_analytic_derivative():
    spectral = gen.polar(N, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU)
    analytic = gen.polar(N, k_fermi=FOURFOLD, mass=POLAR_MASS, tau=TAU,
                         dk_fermi=lambda phi: 4 * K40 * np.sin(4 * phi))
    error = np.max(np.hypot(spectral["vx"] - analytic["vx"], spectral["vy"] - analytic["vy"])
                   / np.hypot(analytic["vx"], analytic["vy"]))
    print(f"spectral vs analytic dk_F/dphi: max relative velocity difference = {error:.2e}")
    assert error < 1e-12
    pd.testing.assert_frame_equal(spectral[["kx", "ky", "tau"]], analytic[["kx", "ky", "tau"]])


def _slice_area(f):
    """Area of {f < 0} for f(qx, qy) (dimensionless q = k·a) symmetric in ±qx and ±qy and
    increasing in |qx| along qy = 0 and in |qy| at fixed qx, inside |q| < π.

    A = 4∫₀^X Y(qx) dqx, with Y(qx) the boundary at fixed qx and X the extent along qy = 0.
    Substituting qx = X sin u removes the square-root endpoint singularity.
    Shares no code with the radial root-finding in the generator.
    """
    X = brentq(lambda q: f(q, 0.0), 0.0, np.pi, xtol=1e-15, rtol=4 * np.finfo(float).eps)

    def Y(qx):
        if f(qx, 0.0) >= 0:
            return 0.0
        return brentq(lambda q: f(qx, q), 0.0, np.pi, xtol=1e-15, rtol=4 * np.finfo(float).eps)

    integral, _ = quad(lambda u: Y(X * np.sin(u)) * np.cos(u), 0.0, np.pi / 2,
                       epsabs=0.0, epsrel=1e-13, limit=200)
    return 4 * X * integral


@pytest.mark.parametrize("pocket", ["gamma", "m"])
def test_tight_binding_area_matches_numerical_area(pocket):
    if pocket == "gamma":
        df = _tb_gamma()
        expected = _slice_area(lambda qx, qy: eps_tight_binding(qx / A, qy / A, mu=MU_ELECTRON)) / A**2
    else:
        df = _tb_m()
        expected = _slice_area(
            lambda qx, qy: -eps_tight_binding((np.pi + qx) / A, (np.pi + qy) / A, mu=MU_HOLE)) / A**2
    area = an.spectral_area(df["kx"], df["ky"])
    error = abs(area / expected - 1)
    print(f"tight-binding {pocket}: area = {area:.10e} m^-2, numerical = {expected:.10e}, "
          f"relative error = {error:.2e}")
    assert error < 1e-10


# ---------------------------------------------------------------------------
# Scattering models
# ---------------------------------------------------------------------------

def test_constant_scattering():
    phi = np.linspace(0, 2 * np.pi, 7)
    np.testing.assert_array_equal(sc.constant(phi, TAU), np.full(7, TAU))


def test_cos4phi_angular_averages():
    """1/τ = (1/τ₀)(1 + a cos4φ) with a = 0.6 gives ⟨τ⟩ = τ₀/√(1−a²) = 1.25τ₀,
    ⟨τ²⟩/⟨τ⟩² = 1/√(1−a²) = 1.25 and ⟨1/τ⟩⟨τ⟩ = 1.25: the low-field Hall
    enhancement and the saturated ρ_xx(∞)/ρ_xx(0) of the anisotropic-τ circle."""
    tau = lambda phi: sc.cos4phi(phi, TAU, anisotropy=0.6)
    mean_tau = an.angular_average(tau)
    hall_factor = an.angular_average(lambda p: tau(p) ** 2) / mean_tau**2
    mr_factor = an.angular_average(lambda p: 1 / tau(p)) * mean_tau
    print(f"<tau>/tau0 = {mean_tau / TAU:.15f}, <tau^2>/<tau>^2 = {hall_factor:.15f}, "
          f"<1/tau><tau> = {mr_factor:.15f}")
    assert mean_tau / TAU == pytest.approx(1.25, rel=1e-12)
    assert hall_factor == pytest.approx(1.25, rel=1e-12)
    assert mr_factor == pytest.approx(1.25, rel=1e-12)


def test_hot_spot_scattering():
    phi = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4])
    tau = sc.hot_spot(phi, TAU, strength=9.0, width=0.05)
    np.testing.assert_allclose(tau[:4], TAU / 10, rtol=1e-12)  # on a hot spot: 1/τ = (1 + 9)/τ₀
    assert tau[4] == pytest.approx(TAU, rel=1e-12)  # far from every hot spot
    shifted = sc.hot_spot(phi + 0.3, TAU, strength=9.0, width=0.05, positions=[0.3])
    assert shifted[0] == pytest.approx(TAU / 10, rel=1e-12)
    np.testing.assert_allclose(sc.hot_spot(phi + 2 * np.pi, TAU, strength=9.0, width=0.05), tau,
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_unit_converters():
    assert units.per_angstrom_to_per_meter(0.7) == pytest.approx(0.7e10, rel=1e-15)
    assert units.angstrom_to_meter(3.87) == pytest.approx(3.87e-10, rel=1e-15)
    assert units.ev_to_joule(0.25) == pytest.approx(0.25 * E, rel=1e-15)
    # v = (1/ħ) dε/dk: 1 eV·Å → e·10⁻¹⁰/ħ ≈ 1.519×10⁵ m/s
    assert units.ev_angstrom_to_meter_per_second(1.0) == pytest.approx(E * 1e-10 / HBAR, rel=1e-15)
    assert units.ev_angstrom_to_meter_per_second(1.0) == pytest.approx(1.5192674e5, rel=1e-7)
    assert units.electron_mass_to_kilogram(5.0) == pytest.approx(5 * ELECTRON_MASS, rel=1e-15)
    assert units.picosecond_to_second(0.1) == pytest.approx(1e-13, rel=1e-15)


@pytest.mark.parametrize("forward, backward", [
    ("per_angstrom_to_per_meter", "per_meter_to_per_angstrom"),
    ("angstrom_to_meter", "meter_to_angstrom"),
    ("ev_to_joule", "joule_to_ev"),
    ("ev_angstrom_to_meter_per_second", "meter_per_second_to_ev_angstrom"),
    ("electron_mass_to_kilogram", "kilogram_to_electron_mass"),
    ("picosecond_to_second", "second_to_picosecond"),
])
def test_unit_converters_round_trip_and_accept_series(forward, backward):
    values = pd.Series([0.1, 1.0, 7.35])
    result = getattr(units, backward)(getattr(units, forward)(values))
    assert isinstance(result, pd.Series)
    np.testing.assert_allclose(result, values, rtol=1e-15)


# ---------------------------------------------------------------------------
# Analytic helpers: check they reproduce the textbook statements the
# known-results tests rely on
# ---------------------------------------------------------------------------

FIELDS = np.array([0.0, 0.5, 5.0, 50.0, 500.0, -5.0])
D = 1e-9


@pytest.mark.parametrize("charge", [-E, +E])
def test_drude_circle(charge):
    n = an.circle_density(K_F, D)
    m = ELECTRON_MASS
    sigma = an.drude_circle(FIELDS, density=n, mass=m, tau=TAU, charge=charge)
    sigma_0 = n * E**2 * TAU / m
    x = E * FIELDS * TAU / m
    s = np.sign(charge)
    np.testing.assert_allclose(sigma[:, 0, 0], sigma_0 / (1 + x**2), rtol=1e-14)
    np.testing.assert_allclose(sigma[:, 1, 1], sigma_0 / (1 + x**2), rtol=1e-14)
    np.testing.assert_allclose(sigma[:, 0, 1], s * sigma_0 * x / (1 + x**2), rtol=1e-14, atol=0)
    np.testing.assert_allclose(sigma[:, 1, 0], -sigma[:, 0, 1], rtol=0, atol=0)
    rho = an.resistivity(sigma)
    np.testing.assert_allclose(rho[:, 0, 0], m / (n * E**2 * TAU), rtol=1e-12)  # no MR
    r_h = an.hall_coefficient(sigma, FIELDS)
    assert np.isnan(r_h[0])
    np.testing.assert_allclose(r_h[1:], 1 / (n * charge), rtol=1e-12)


def test_drude_ellipse():
    n = an.circle_density(K_F, D)
    mx, my = ELECTRON_MASS, 4 * ELECTRON_MASS
    sigma = an.drude_ellipse(FIELDS, density=n, mass_x=mx, mass_y=my, tau=TAU)
    x = E * FIELDS * TAU / np.sqrt(mx * my)
    np.testing.assert_allclose(sigma[:, 0, 0], n * E**2 * TAU / mx / (1 + x**2), rtol=1e-14)
    np.testing.assert_allclose(sigma[:, 1, 1], n * E**2 * TAU / my / (1 + x**2), rtol=1e-14)
    np.testing.assert_allclose(sigma[:, 0, 1], -n * E**2 * TAU * x / (np.sqrt(mx * my) * (1 + x**2)),
                               rtol=1e-14)
    rho = an.resistivity(sigma)
    np.testing.assert_allclose(rho[:, 0, 0], mx / (n * E**2 * TAU), rtol=1e-12)
    np.testing.assert_allclose(rho[:, 1, 1], my / (n * E**2 * TAU), rtol=1e-12)
    np.testing.assert_allclose(an.hall_coefficient(sigma, FIELDS)[1:], -1 / (n * E), rtol=1e-12)
    rotated = an.drude_ellipse(FIELDS, density=n, mass_x=mx, mass_y=my, tau=TAU, rotation=np.pi / 6)
    np.testing.assert_allclose(rotated, an.rotate(sigma, np.pi / 6), rtol=0, atol=0)
    assert not np.allclose(rotated[:, 0, 0], sigma[:, 0, 0])


def test_two_band_closed_forms_match_inverted_sum():
    params = dict(n_e=1e27, mu_e=0.02, n_h=0.5e27, mu_h=0.005)
    B = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1e4])
    sigma = an.two_band(B, **params)
    rho = an.resistivity(sigma)
    np.testing.assert_allclose(rho[:, 0, 0], an.two_band_rho_xx(B, **params), rtol=1e-12)
    np.testing.assert_allclose(an.hall_coefficient(sigma, B)[1:],
                               an.two_band_hall_coefficient(B[1:], **params), rtol=1e-12)


def test_two_band_compensated_limits():
    """n_e = n_h = n: ρ_xx = (1 + μ_eμ_h B²)/(ne(μ_e+μ_h)) grows as B² without saturating,
    and R_H = (μ_h − μ_e)/(ne(μ_e+μ_h)) at every B."""
    n = an.circle_density(K_F, D)
    mu_e = E * TAU / ELECTRON_MASS
    mu_h = E * (TAU / 2) / (2 * ELECTRON_MASS)
    # Up to ω_cτ = 100. Far beyond that, the electron and hole σ_xy cancel to O(1/(μB)²)
    # and rounding in the summed tensor dominates R_H.
    B = np.array([0.01, 1.0, 10.0, 100.0]) / mu_e
    sigma = an.two_band(B, n_e=n, mu_e=mu_e, n_h=n, mu_h=mu_h)
    np.testing.assert_allclose(an.resistivity(sigma)[:, 0, 0],
                               (1 + mu_e * mu_h * B**2) / (n * E * (mu_e + mu_h)), rtol=1e-12)
    np.testing.assert_allclose(an.hall_coefficient(sigma, B), (mu_h - mu_e) / (n * E * (mu_e + mu_h)),
                               rtol=1e-12)


def test_two_band_uncompensated_limits():
    """n_h = n_e/2: R_H(0) = (n_hμ_h² − n_eμ_e²)/(e(n_hμ_h + n_eμ_e)²), and at μB = 10³
    R_H is within 1e-4 of 1/((n_h − n_e)e)."""
    n_e, mu_e, mu_h = 1e27, 0.02, 0.01
    n_h = n_e / 2
    params = dict(n_e=n_e, mu_e=mu_e, n_h=n_h, mu_h=mu_h)
    r_h_zero = an.two_band_hall_coefficient(1e-12, **params)
    expected_zero = (n_h * mu_h**2 - n_e * mu_e**2) / (E * (n_h * mu_h + n_e * mu_e) ** 2)
    assert r_h_zero == pytest.approx(expected_zero, rel=1e-12)
    B_high = 1e3 / min(mu_e, mu_h)
    r_h_high = an.hall_coefficient(an.two_band(B_high, **params), B_high)[0]
    error = abs(r_h_high * (n_h - n_e) * E - 1)
    print(f"R_H(mu B = 1e3) relative distance from 1/((n_h - n_e)e) = {error:.2e}")
    assert error < 1e-4


# ---------------------------------------------------------------------------
# Package layout and the legacy FermiSurface
# ---------------------------------------------------------------------------

SNAPSHOT = Path(__file__).parent / "data" / "boltzmann_legacy_snapshot.json"


def test_package_layout():
    from quantalyze.beta.boltzmann import FermiSurface
    from quantalyze.beta.boltzmann._legacy import FermiSurface as LegacyFermiSurface

    assert FermiSurface is LegacyFermiSurface
    assert bz.FermiSurface is FermiSurface
    for module in ("generators", "scattering", "units"):
        assert hasattr(bz, module)


@pytest.mark.parametrize("case", ["circle", "fourfold"])
def test_fermi_surface_geometry_unchanged(case):
    """FermiSurface's area and density match the pre-package boltzmann.py. (Its conductivity
    changed on purpose when it moved to the exact solver: see the legacy tests.)"""
    data = json.loads(SNAPSHOT.read_text())["cases"][case]
    fs = bz.FermiSurface(
        np.array(data["theta"]),
        np.array(data["fermi_wavevector"]),
        np.array(data["effective_mass"]),
        np.array(data["relaxation_time"]),
        data["c_axis_length"],
    )
    assert fs.fermi_area() == pytest.approx(data["fermi_area"], rel=1e-14)
    assert fs.carrier_density() == pytest.approx(data["carrier_density"], rel=1e-14)


def test_slow_marker_is_registered(pytestconfig):
    assert any(line.startswith("slow:") for line in pytestconfig.getini("markers"))
