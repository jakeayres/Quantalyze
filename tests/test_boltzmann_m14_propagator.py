"""M14: the propagator matrix of one contour, and the two-stage contour preparation.

G_nm = ⟨φ_n, R φ_m⟩ is the relaxation-time kernel written as a matrix between hat
functions in the damping coordinate: ℓ_iᵀ G ℓ_j must be the kernel's own orbit sum, G must
map a constant to the lumped mass m, the reversed orbit must give Gᵀ, and G(0) is the
consistent mass matrix. `prepare_contour` is now `finish_contour(contour_geometry(...))`
and records the input row of each prepared node.
"""
import numpy as np
import pytest

from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._collisions import mass_matrix, propagator
from quantalyze.beta.boltzmann._contour import contour_geometry, finish_contour, prepare_contour
from quantalyze.beta.boltzmann._kernel_py import orbit_sums
from quantalyze.beta.boltzmann._response import _zero_field_sums
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
TAU = 1e-13
A = 3.87e-10
G = 2 * np.pi / A


def lopsided(n=256):
    return gen.polar(n, k_fermi=lambda p: 7e9 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p)), mass=M,
                     tau=lambda p: TAU / (1 + 0.5 * np.cos(2 * p + 0.3)))


def contours():
    """Closed, open and k_z-slice contours, prepared."""
    df = lopsided()
    sheet = gen.open_sheets(128, k0=5e9, velocity=2e5, tau=TAU, period=G, warping=1e9)[0]
    slice_ = gen.circle(128, k_fermi=7e9, mass=M, tau=TAU)
    phi = np.arctan2(slice_.ky, slice_.kx)
    return {
        "lopsided pocket": prepare_contour(df.kx, df.ky, df.vx, df.vy, df.tau),
        "open sheet": prepare_contour(sheet.kx, sheet.ky, sheet.vx, sheet.vy, sheet.tau, period=(0.0, G)),
        "k_z slice": prepare_contour(slice_.kx, slice_.ky, slice_.vx, slice_.vy, slice_.tau,
                                     vz=1e3 * (0.5 + 0.5 * np.cos(4 * phi))),
    }


def paths(c):
    return np.stack([c.lx, c.ly] + ([] if c.lz is None else [c.lz]), axis=1)


X = np.array([1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6])  # ω_cτ


@pytest.mark.parametrize("name", ["lopsided pocket", "open sheet", "k_z slice"])
def test_propagator_reproduces_the_orbit_sums(name):
    """ℓᵀGℓ equals the relaxation-time kernel's orbit sum to 1e-14 of σ(0) at every field, and
    to 1e-12 of itself up to ω_cτ = 1e3 (at 1e6 σ has fallen to ~1e-12 of σ(0), and both
    computations cancel)."""
    c = contours()[name]
    ell = paths(c)
    zero = ell.T @ mass_matrix(c.damping) @ ell
    fields = X * M / (E * TAU)
    for x, b in zip(X, fields):
        matrix = ell.T @ propagator(c.damping, b) @ ell
        kernel = orbit_sums(c.damping, c.lx, c.ly, [b], lz=c.lz)[0]
        of_zero = np.max(np.abs(matrix - kernel)) / np.max(np.abs(zero))
        of_itself = np.max(np.abs(matrix - kernel)) / np.max(np.abs(kernel))
        print(f"{name} x = {x:g}: vs sigma(0) {of_zero:.1e}, vs itself {of_itself:.1e}")
        assert of_zero <= 1e-14
        if x <= 1e3:
            assert of_itself <= 1e-12


@pytest.mark.parametrize("name", ["lopsided pocket", "open sheet", "k_z slice"])
def test_propagator_structure(name):
    """G(0) = M reproduces the zero-field sums (1e-15); G·1 = m, the lumped mass (1e-14 up to
    ω_cτ = 1; 1e-13 at 1e3, where the periodic closure divides by the small damping per orbit
    and amplifies rounding about a hundredfold); the reversed orbit gives the transpose
    (1e-13); the symmetric part tends to M as B → 0 (1e-8 at ω_cτ = 1e-6); nothing is NaN or
    infinite from ω_cτ = 1e-8 to 1e8."""
    c = contours()[name]
    ell = paths(c)
    mass = mass_matrix(c.damping)
    zero = ell.T @ mass @ ell
    assert np.max(np.abs(zero - _zero_field_sums(c.damping, *ell.T))) <= 1e-15 * np.max(np.abs(zero))
    lumped = 0.5 * (np.roll(c.damping, 1) + c.damping)
    order = np.roll(np.arange(c.damping.size)[::-1], 1)
    for x in (1e-3, 1.0, 1e3):
        g = propagator(c.damping, x * M / (E * TAU))
        constant = np.max(np.abs(g @ np.ones(g.shape[0]) - lumped)) / np.max(lumped)
        reverse = propagator(c.damping[::-1], x * M / (E * TAU))
        onsager = np.max(np.abs(reverse - g[np.ix_(order, order)].T)) / np.max(np.abs(g))
        print(f"{name} x = {x:g}: G 1 = m {constant:.1e}; reversed = transpose {onsager:.1e}")
        assert constant <= (1e-14 if x <= 1 else 1e-13) and onsager <= 1e-13
    g = propagator(c.damping, 1e-6 * M / (E * TAU))
    assert np.max(np.abs(0.5 * (g + g.T) - mass)) <= 1e-8 * np.max(np.abs(mass))
    for x in (1e-8, 1e8):
        assert np.all(np.isfinite(propagator(c.damping, x * M / (E * TAU))))


def test_geometry_and_finish_are_prepare_contour():
    """prepare_contour = finish_contour(contour_geometry(...)) with τ in the prepared order."""
    df = lopsided().iloc[::-1]
    whole = prepare_contour(df.kx, df.ky, df.vx, df.vy, df.tau)
    geometry = contour_geometry(df.kx, df.ky, df.vx, df.vy)
    split = finish_contour(geometry, df.tau.to_numpy()[geometry.index])
    for field in ("kx", "ky", "vx", "vy", "tau", "s", "damping", "lx", "ly", "drift", "period", "index"):
        assert np.array_equal(getattr(whole, field), getattr(split, field)), field


@pytest.mark.parametrize("case", ["as given", "reversed", "rotated start", "closing node", "open sheet reversed"])
def test_index_maps_prepared_nodes_to_input_rows(case):
    """index[p] is the input row of prepared node p, whatever order the input came in."""
    df = gen.circle(64, k_fermi=7e9, mass=M, tau=lambda p: TAU * (1 + 0.1 * np.cos(p)))
    period = None
    if case == "reversed":
        df = df.iloc[::-1]
    elif case == "rotated start":
        df = df.iloc[np.roll(np.arange(64), 11)]
    elif case == "closing node":
        df = df.iloc[list(range(64)) + [0]]
    elif case == "open sheet reversed":
        df = gen.open_sheets(64, k0=5e9, velocity=2e5, tau=TAU, period=G, warping=1e9)[1].iloc[::-1]
        period = (0.0, G)
    df = df.reset_index(drop=True)
    c = prepare_contour(df.kx, df.ky, df.vx, df.vy, df.tau, period=period)
    assert c.index.size == c.kx.size and not c.index.flags.writeable
    for field in ("kx", "ky", "vx", "vy", "tau"):
        assert np.array_equal(df[field].to_numpy()[c.index], getattr(c, field))
    assert sorted(c.index) == list(range(c.kx.size))
