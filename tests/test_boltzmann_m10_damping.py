"""M10: the damping-coordinate kernel.

In the damping coordinate g = ∫ds/τ, with the mean free path ℓ = vτ linear in g on each
segment, both the history step and the outer integral are exact, so the discrete σ is
the Chambers σ of a continuous model. These tests check the consequences: the segment
moments p_k are accurate, Onsager's relation holds without symmetrising, σ_sym − σ(0)
grows as B² at low field (no magnetoresistance linear in |B|), σ_sym is positive
definite, and the runtime check comparing the two orbit orientations warns when (and
only when) they disagree.
"""
import math
import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _kernel, _kernel_py, _response
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9
TAU = 1e-13
A = 3.87e-10
G = 2 * np.pi / A
BACKENDS = ["python", "numba"]
SKEWED = (0.3, 2.0, 4.1)  # hot spots off every mirror line


def skewed_tau(p):
    return sc.hot_spot(p, TAU, strength=3.0, width=0.35, positions=SKEWED)


def lopsided(n=512):
    return gen.polar(n, k_fermi=lambda p: 7e9 * (1 + 0.08 * np.cos(3 * p) + 0.05 * np.sin(2 * p + 0.3)),
                     mass=M, tau=skewed_tau)


def tight_binding(n=512, positions=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    return gen.tight_binding(n, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2, positions=positions),
                             lattice_constant=A, hopping=0.25 * E, next_hopping=-0.0625 * E, third_hopping=0.02 * E,
                             chemical_potential=0.0, center=(np.pi / A, np.pi / A))


def open_sheet(n=512):
    return gen.open_sheets(n, k0=5e9, velocity=2e5, tau=skewed_tau, period=G, warping=0.5e9)[0]


def warped_slices(n=256):
    c2 = HBAR**2 / (2 * M)
    return gen.from_dispersion_3d(
        n, 4, tau=skewed_tau, max_radius=2e10, layer_spacing=D,
        energy=lambda kx, ky, kz: c2 * (kx**2 + ky**2 + 0.3e9 * np.cos(kz * D) * kx) - 1.9e-19,
        gradient=lambda kx, ky, kz: (c2 * (2 * kx + 0.3e9 * np.cos(kz * D)), 2 * c2 * ky,
                                     -c2 * 0.3e9 * D * np.sin(kz * D) * kx))


def fourfold(n=512):
    return gen.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * M,
                     tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.3), carrier="hole")


CLOSED = {
    "circle": lambda: gen.circle(512, k_fermi=7e9, mass=M, tau=TAU),
    "circle-cos4-tau": lambda: gen.circle(512, k_fermi=7e9, mass=M, tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6)),
    "fourfold": fourfold,
    "lopsided": lopsided,
    "tight-binding": tight_binding,
    "tight-binding-skewed": lambda: tight_binding(positions=SKEWED),
}


def arrays(df):
    return df["kx"], df["ky"], df["vx"], df["vy"], df["tau"]


def per_tesla(df, period=None):
    """ω_cτ averaged round the orbit per tesla: 2π / Σ Δg_n."""
    return 2 * np.pi / np.sum(prepare_contour(*arrays(df), period=period).damping)


def per_field(a, b):
    return np.max(np.abs(a - b), axis=(1, 2)) / np.max(np.abs(b), axis=(1, 2))


# ---------------------------------------------------------------------------
# The moments p_k(z) = ∫₀¹ r^k e^{−zr} dr and the overlap weights built from them
# ---------------------------------------------------------------------------

def exact_moments(z):
    """e^{−z}, p₀…p₃, M_aa, M_ab, M_ba to 80 digits: the series for z < 1, closed forms above."""
    getcontext().prec = 80
    x = Decimal(float(z))
    e = (-x).exp()
    p = []
    for k in range(4):
        if x < 1:
            total, term, m = Decimal(0), Decimal(1), 0
            while True:
                piece = term / (k + m + 1)
                total += piece
                m += 1
                term = term * (-x) / m
                if abs(piece) < Decimal(10) ** -70:
                    break
            p.append(total)
        else:
            partial = sum(x**m / math.factorial(m) for m in range(k + 1))
            p.append(math.factorial(k) / x ** (k + 1) * (1 - e * partial))
    p0, p1, p2, p3 = p
    overlap = (p0 / 3 - p1 / 2 + p3 / 6, p0 / 6 - p1 / 2 + p2 / 2 - p3 / 6, p0 / 6 + p1 / 2 - p2 / 2 - p3 / 6)
    return [float(v) for v in (e, *p, *overlap)]


def numpy_moments(z):
    e, p0, p1, p2, p3 = _kernel_py.moments(z)
    return np.stack([e, p0, p1, p2, p3, *_kernel_py.overlaps(p0, p1, p2, p3)], axis=-1)


NAMES = ["e^-z", "p0", "p1", "p2", "p3", "M_aa", "M_ab", "M_ba"]


def test_moments_against_high_precision():
    """p₀…p₃ and M_aa, M_ab, M_ba to 1e-14 relative for z ∈ [1e-14, 1e3], on both backends."""
    z = np.concatenate([np.geomspace(1e-14, 1e3, 500), np.geomspace(0.09, 0.11, 40), np.geomspace(1.9, 2.1, 40)])
    expected = np.array([exact_moments(v) for v in z])
    numpy_values = numpy_moments(z)
    numba_values = np.array([_kernel.moments(v) for v in z])
    numba_values = np.concatenate([numba_values, np.stack(_kernel_py.overlaps(*numba_values[:, 1:].T), axis=1)],
                                  axis=1)
    finite = expected[:, 0] > 0  # e^{−z} underflows at z = 1e3; compare it only where it doesn't
    with np.errstate(invalid="ignore"):
        errors = np.abs(numpy_values / expected - 1)
    errors[~finite, 0] = 0.0
    for name, error in zip(NAMES, errors.max(axis=0)):
        print(f"{name}: max relative error {error:.1e}")
    assert errors.max() <= 1e-14
    np.testing.assert_array_equal(numba_values, numpy_values)


@pytest.mark.parametrize("switch", [_kernel_py.SMALL_Z, _kernel_py.SERIES_Z])
def test_moments_continuous_across_switches(switch):
    values = numpy_moments(np.array([np.nextafter(switch, 0.0), switch]))
    jump = np.max(np.abs(values[0] / values[1] - 1))
    print(f"largest jump across z = {switch:g}: {jump:.1e}")
    assert jump <= 1e-14


def test_moments_limits():
    """z = 0 gives p_k = 1/(k+1) and M = (1/8, 1/24, 5/24); large z gives p_k = k!/z^{k+1} and
    M = (1/3z, 1/6z, 1/6z); everything stays finite."""
    values = numpy_moments(np.array([0.0, 1e-300, 1e300]))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], [1, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 8, 1 / 24, 5 / 24], rtol=1e-15)
    np.testing.assert_allclose(values[1], values[0], rtol=1e-15)
    np.testing.assert_allclose(values[2, [1, 5, 6, 7]], [1e-300, 1e-300 / 3, 1e-300 / 6, 1e-300 / 6], rtol=1e-15)


# ---------------------------------------------------------------------------
# Onsager without symmetrising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["lopsided", "tight-binding-skewed", "open-sheet", "warped-slices"])
def test_onsager_holds_without_symmetrising(backend, name):
    """σ(−B) = σ(B)ᵀ to 1e-13 per field for ω_cτ ∈ [1e-6, 1e6], on contours with no mirror plane
    (where it cannot hold by symmetry). The two orientations are separate kernel runs."""
    x = np.geomspace(1e-6, 1e6, 25)
    if name == "warped-slices":
        df = warped_slices()
        fields = np.concatenate([x, -x]) * per_tesla(df[df.kz == df.kz.min()])
        s = bz.conductivity(df, fields, layer_spacing=D, kz="kz", symmetrize=False, backend=backend)
        sigma = s.iloc[:, 1:].to_numpy().reshape(-1, 3, 3)
    else:
        period = (0.0, G) if name == "open-sheet" else None
        df = open_sheet() if name == "open-sheet" else CLOSED[name]()
        fields = np.concatenate([x, -x]) * per_tesla(df, period)
        sigma = conductivity_tensor(*arrays(df), fields, layer_spacing=D, period=period, symmetrize=False,
                                    backend=backend)
    defect = per_field(sigma[x.size:], np.transpose(sigma[:x.size], (0, 2, 1)))
    print(f"{name}: max |sigma(-B) - sigma(B)^T| / max |sigma| = {defect.max():.1e}")
    assert defect.max() <= 1e-13


# ---------------------------------------------------------------------------
# Low field: σ_sym − σ(0) grows as B², not |B|
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", CLOSED)
def test_low_field_curvature_ratio(backend, name):
    """[σ_sym(2B) − σ(0)] / [σ_sym(B) − σ(0)] = 4 ± 0.01 on the diagonal at ω_cτ = 10⁻⁵, 10⁻⁴ and
    10⁻³ (N = 512, so the history decays within one segment). A spurious term linear in |B|
    would make it 2."""
    df = CLOSED[name]()
    x = np.array([1e-5, 1e-4, 1e-3])
    fields = np.concatenate([[0.0], x, 2 * x]) / per_tesla(df)
    sigma = conductivity_tensor(*arrays(df), fields, layer_spacing=D, backend=backend)
    symmetric = 0.5 * (sigma + np.transpose(sigma, (0, 2, 1)))
    change = symmetric[1:] - symmetric[0]  # (2 nx, 2, 2)
    ratios = np.stack([change[x.size:, i, i] / change[:x.size, i, i] for i in range(2)], axis=1)  # (nx, 2)
    print(f"{name}: ratios (xx, yy) at x = 1e-5, 1e-4, 1e-3: "
          + "; ".join(f"{a:.4f}, {b:.4f}" for a, b in ratios))
    assert np.all(np.abs(ratios - 4) <= 0.01)


# ---------------------------------------------------------------------------
# The symmetric part is positive definite (the dissipation j·E > 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("symmetrize", [True, False])
def test_symmetric_part_is_positive_definite(backend, symmetrize):
    x = np.geomspace(1e-6, 1e6, 49)
    worst = np.inf
    for name, make in CLOSED.items():
        df = make()
        fields = np.concatenate([[0.0], x, -x]) / per_tesla(df)
        sigma = conductivity_tensor(*arrays(df), fields, layer_spacing=D, symmetrize=symmetrize, backend=backend)
        eigenvalues = np.linalg.eigvalsh(0.5 * (sigma + np.transpose(sigma, (0, 2, 1))))
        assert np.all(eigenvalues > 0), name
        worst = min(worst, np.min(eigenvalues[:, 0] / eigenvalues[:, 1]))
    print(f"smallest ratio of the two eigenvalues of sigma_sym: {worst:.1e}")


# ---------------------------------------------------------------------------
# The runtime Onsager check
# ---------------------------------------------------------------------------

def test_runtime_check_warns_when_the_orientations_disagree(monkeypatch):
    """A kernel whose reversed orbit is off by one part in 10⁶ (or NaN) is reported."""
    original = _response._orbit_sums_both

    def corrupted(scale):
        def run(backend, forward, backward, field, both):
            sums = original(backend, forward, backward, field, both)
            sums[1] *= scale
            return sums
        return run

    df = lopsided(128)
    for scale in (1 + 1e-6, np.nan):
        monkeypatch.setattr(_response, "_orbit_sums_both", corrupted(scale))
        with pytest.warns(RuntimeWarning, match="Onsager"):
            conductivity_tensor(*arrays(df), [0.0, 1.0, 10.0], layer_spacing=D)
    monkeypatch.setattr(_response, "_orbit_sums_both", original)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        conductivity_tensor(*arrays(df), [0.0, 1.0, 10.0], layer_spacing=D)


@pytest.mark.parametrize("backend", BACKENDS)
def test_runtime_check_is_silent_on_a_correct_kernel(backend):
    """No warning over twelve decades of field, on every test contour, open orbits and k_z slices."""
    x = np.geomspace(1e-6, 1e6, 49)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for make in CLOSED.values():
            df = make()
            conductivity_tensor(*arrays(df), np.concatenate([[0.0], x]) / per_tesla(df), layer_spacing=D,
                                backend=backend)
        sheet = open_sheet()
        conductivity_tensor(*arrays(sheet), x / per_tesla(sheet, (0.0, G)), layer_spacing=D, period=(0.0, G),
                            backend=backend)
        slices = warped_slices()
        bz.conductivity(slices, x / per_tesla(slices[slices.kz == slices.kz.min()]), layer_spacing=D, kz="kz",
                        backend=backend)


# ---------------------------------------------------------------------------
# The zero-field branch
# ---------------------------------------------------------------------------

def test_zero_field_branch_is_the_integral_of_the_piecewise_linear_mean_free_path():
    """σ(0) = (g_s e³/4π²ħ²d) Σ Δg [⅓(aa + bb) + ⅙(ab + ba)], ∮ ℓℓ dg with ℓ linear in g,
    whichever way round the orbit is taken."""
    df = lopsided(64)
    c = prepare_contour(*arrays(df))
    ell = np.stack([c.lx, c.ly], axis=1)
    following = np.roll(ell, -1, axis=0)
    expected = sum(g * (np.outer(a, a) / 3 + np.outer(b, b) / 3 + np.outer(a, b) / 6 + np.outer(b, a) / 6)
                   for g, a, b in zip(c.damping, ell, following))
    expected = expected * 2 * E**3 / (4 * np.pi**2 * HBAR**2 * D)
    for symmetrize in (True, False):
        actual = conductivity_tensor(*arrays(df), 0.0, layer_spacing=D, symmetrize=symmetrize)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
