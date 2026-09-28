"""M1 input contract: prepare_contour.

prepare_contour is the only place the contour contract is enforced. These tests
check that it rejects bad input, drops a repeated closing point, orients the
nodes along the carriers' motion (ħ dk/dt = q v × B with B = +B ẑ), computes
the segment geometric times s_n, dampings Δg_n and mean free paths ℓ_n = v_n τ_n,
and removes the discretisation drift so that the model's ∮ ℓ dg vanishes.
"""
import warnings

import numpy as np
import pytest

from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import MIN_NODES, PreparedContour, prepare_contour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
K_F = 7.0e9  # m⁻¹
TAU = 1.0e-13  # s
A = 3.87e-10  # m

FIELDS = ("kx", "ky", "vx", "vy", "tau", "s", "damping", "lx", "ly")


def arrays(df):
    return tuple(df[c].to_numpy() for c in ("kx", "ky", "vx", "vy", "tau"))


def circle(n=64, carrier="electron", tau=TAU):
    return gen.circle(n, k_fermi=K_F, mass=ELECTRON_MASS, tau=tau, carrier=carrier)


def lopsided(n=64):
    """No mirror symmetry and anisotropic τ, so nothing cancels by symmetry."""
    return gen.polar(n, k_fermi=lambda p: K_F * (1 + 0.1 * np.cos(2 * p) + 0.05 * np.sin(3 * p)),
                     mass=2 * ELECTRON_MASS, tau=lambda p: sc.cos4phi(p + 0.3, TAU, anisotropy=0.5))


def fourfold(n=512):
    return gen.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * ELECTRON_MASS,
                     tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6), carrier="hole")


def tight_binding_m(n=512):
    return gen.tight_binding(n, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2),
                             lattice_constant=A, hopping=0.25 * E, next_hopping=-0.0625 * E,
                             third_hopping=0.02 * E, chemical_potential=0.0,
                             center=(np.pi / A, np.pi / A))


def cyclic_difference(a: PreparedContour, b: PreparedContour):
    """Largest normwise relative difference between two prepared contours over all
    fields, after rolling b so that its node 0 is a's node 0."""
    assert a.kx.size == b.kx.size
    matches = np.flatnonzero((b.kx == a.kx[0]) & (b.ky == a.ky[0]))
    assert matches.size == 1, "node 0 of one contour is not a node of the other"
    shift = matches[0]
    worst = 0.0
    for name in FIELDS:
        x, y = getattr(a, name), np.roll(getattr(b, name), -shift)
        if name in ("vx", "vy", "lx", "ly"):  # components share one scale
            pair = (a.vx, a.vy) if name in ("vx", "vy") else (a.lx, a.ly)
            scale = np.max(np.abs(np.concatenate(pair)))
        else:
            scale = np.max(np.abs(x))
        worst = max(worst, np.max(np.abs(x - y)) / scale)
    return worst


def drift_sum(c: PreparedContour):
    """|Σ ½ Δg_n (ℓ_n + ℓ_{n+1})| relative to Σ ½ Δg_n (|ℓ_n| + |ℓ_{n+1}|): the model's
    ∮ v dt = ∮ ℓ dg/|B|, relative to its scale."""
    sx = np.sum(0.5 * c.damping * (c.lx + np.roll(c.lx, -1)))
    sy = np.sum(0.5 * c.damping * (c.ly + np.roll(c.ly, -1)))
    path = np.hypot(c.lx, c.ly)
    return np.hypot(sx, sy) / np.sum(0.5 * c.damping * (path + np.roll(path, -1)))


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index", range(5), ids=["kx", "ky", "vx", "vy", "tau"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite(index, bad):
    values = [a.copy() for a in arrays(circle())]
    values[index][7] = bad
    with pytest.raises(ValueError, match="finite"):
        prepare_contour(*values)


@pytest.mark.parametrize("bad", [0.0, -1e-13])
def test_rejects_non_positive_tau(bad):
    kx, ky, vx, vy, tau = arrays(circle())
    tau = tau.copy()
    tau[3] = bad
    with pytest.raises(ValueError, match="positive"):
        prepare_contour(kx, ky, vx, vy, tau)
    with pytest.raises(ValueError, match="positive"):
        prepare_contour(kx, ky, vx, vy, bad)


@pytest.mark.parametrize("index", range(5), ids=["kx", "ky", "vx", "vy", "tau"])
def test_rejects_mismatched_lengths(index):
    values = list(arrays(circle()))
    values[index] = values[index][:-1]
    with pytest.raises(ValueError, match="length"):
        prepare_contour(*values)


def test_rejects_two_dimensional_input():
    kx, ky, vx, vy, tau = arrays(circle())
    with pytest.raises(ValueError, match="1-D"):
        prepare_contour(kx.reshape(8, 8), ky, vx, vy, tau)


def test_rejects_fewer_than_sixteen_nodes():
    assert MIN_NODES == 16
    prepare_contour(*arrays(circle(16)))
    with pytest.raises(ValueError, match="16"):
        prepare_contour(*arrays(circle(15)))
    # 16 points of which the last repeats the first are only 15 nodes
    kx, ky, vx, vy, tau = (np.append(a, a[0]) for a in arrays(circle(15)))
    with pytest.raises(ValueError, match="16"):
        prepare_contour(kx, ky, vx, vy, tau)


def test_rejects_zero_length_segment():
    kx, ky, vx, vy, tau = (np.insert(a, 10, a[10]) for a in arrays(circle()))
    with pytest.raises(ValueError, match="coincide"):
        prepare_contour(kx, ky, vx, vy, tau)


def test_rejects_zero_velocity():
    kx, ky, vx, vy, tau = (a.copy() for a in arrays(circle()))
    vx[5] = vy[5] = 0.0
    with pytest.raises(ValueError, match="velocity"):
        prepare_contour(kx, ky, vx, vy, tau)


def test_rejects_unclosed_contour():
    sheet = gen.open_sheets(64, k0=5e9, velocity=2e5, tau=TAU, period=2 * np.pi / A)[0]
    with pytest.raises(ValueError, match="closed"):
        prepare_contour(*arrays(sheet))
    half = circle(128).iloc[:64]  # a half circle: the "closing" segment is a diameter
    with pytest.raises(ValueError, match="closed"):
        prepare_contour(*arrays(half))


def test_an_open_sheet_is_accepted_with_its_period():
    """The same nodes that are rejected as unclosed are a valid open orbit with period."""
    sheet = gen.open_sheets(64, k0=5e9, velocity=2e5, tau=TAU, period=2 * np.pi / A)[0]
    prepared = prepare_contour(*arrays(sheet), period=(0.0, 2 * np.pi / A))
    assert prepared.kx.size == 64 and np.array_equal(prepared.drift, [0.0, 0.0])


def test_rejects_nodes_out_of_order():
    kx, ky, vx, vy, tau = (a.copy() for a in arrays(circle()))
    for a in (kx, ky, vx, vy, tau):
        a[[20, 21]] = a[[21, 20]]
    with pytest.raises(ValueError, match="order"):
        prepare_contour(kx, ky, vx, vy, tau)


@pytest.mark.parametrize("charge", [0.0, np.nan, 2 * E])
def test_rejects_bad_charge(charge):
    with pytest.raises(ValueError, match="charge"):
        prepare_contour(*arrays(circle()), charge=charge)


def test_warns_on_a_sharply_curved_contour_that_is_too_coarse():
    """A 10:1 ellipse needs more than 16 nodes; the warning goes away as N grows."""
    thin = lambda n: gen.ellipse(n, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=100 * ELECTRON_MASS, tau=TAU)  # noqa: E731
    with pytest.warns(UserWarning, match="coarsely"):
        prepare_contour(*arrays(thin(16)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_contour(*arrays(thin(128)))


def test_warns_when_velocity_is_not_normal_to_the_contour():
    kx, ky, vx, vy, tau = arrays(circle())
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    with pytest.warns(UserWarning, match="normal"):
        prepare_contour(kx, ky, c * vx - s * vy, s * vx + c * vy, tau)


@pytest.mark.parametrize("df", [
    circle(16), circle(16, carrier="hole"), lopsided(16).assign(tau=TAU), fourfold(16).assign(tau=TAU),
    tight_binding_m(16).assign(tau=TAU),
    gen.ellipse(16, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU, rotation=0.4),
    lopsided(128), fourfold(128), tight_binding_m(128),
], ids=["circle", "circle-hole", "lopsided", "fourfold", "tight-binding", "ellipse",
        "lopsided-tau", "fourfold-tau", "tight-binding-tau"])
def test_valid_contours_do_not_warn(df):
    """Coarse shapes (16 nodes, constant τ) and resolved anisotropic τ (128 nodes)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_contour(*arrays(df))


def test_warns_when_tau_changes_too_fast_between_nodes():
    """A hot spot narrower than about two node spacings (τ changing by more than ~1.5× between
    neighbours) warns: at N = 256 a 0.02-rad hot spot puts the MR off by ~25%. Resolved, it is
    silent."""
    narrow = circle(256, tau=lambda p: sc.hot_spot(p, TAU, strength=9.0, width=0.02))
    with pytest.warns(UserWarning, match="under-resolved"):
        prepare_contour(*arrays(narrow))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_contour(*arrays(circle(256, tau=lambda p: sc.hot_spot(p, TAU, strength=9.0, width=0.2))))
        prepare_contour(*arrays(circle(4096, tau=lambda p: sc.hot_spot(p, TAU, strength=9.0, width=0.02))))


# ---------------------------------------------------------------------------
# Closing point, dtype and broadcasting
# ---------------------------------------------------------------------------

def test_drops_repeated_closing_point():
    reference = prepare_contour(*arrays(lopsided()))
    closed = prepare_contour(*(np.append(a, a[0]) for a in arrays(lopsided())))
    assert closed.kx.size == reference.kx.size
    assert cyclic_difference(reference, closed) == 0.0


def test_drops_a_closing_point_that_repeats_the_first_to_rounding():
    """θ = linspace(0, 2π, N) repeats θ = 0 at 2π, but sin(2π) ≠ 0 in floating point."""
    theta = np.linspace(0, 2 * np.pi, 65)
    kx, ky = K_F * np.cos(theta), K_F * np.sin(theta)
    assert ky[-1] != ky[0]
    v = HBAR * K_F / ELECTRON_MASS
    prepared = prepare_contour(kx, ky, v * np.cos(theta), v * np.sin(theta), TAU)
    assert prepared.kx.size == 64


def test_output_is_float64_read_only_and_tau_broadcasts():
    kx, ky, vx, vy, tau = arrays(circle())
    prepared = prepare_contour(*(a.astype(np.float32) for a in (kx, ky, vx, vy)), np.float32(TAU))
    for name in FIELDS:
        array = getattr(prepared, name)
        assert array.dtype == np.float64 and array.shape == (64,)
        assert not array.flags.writeable
    np.testing.assert_array_equal(prepared.tau, np.float64(np.float32(TAU)))
    assert prepared.drift.shape == (2,)


# ---------------------------------------------------------------------------
# Orientation: ħ dk/dt = q v × B with B = +B ẑ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("carrier, charge, counter_clockwise", [
    ("electron", -E, True),   # free electrons orbit anticlockwise seen from +z
    ("hole", -E, False),      # v points inwards, so the orbit reverses
    ("electron", +E, False),  # positive carriers turn the other way
    ("hole", +E, True),
])
def test_orientation(carrier, charge, counter_clockwise):
    for df in (circle(carrier=carrier), circle(carrier=carrier).iloc[::-1]):
        prepared = prepare_contour(*arrays(df), charge=charge)
        angle = np.unwrap(np.arctan2(prepared.ky, prepared.kx))
        steps = np.diff(angle)
        assert np.all(steps > 0) if counter_clockwise else np.all(steps < 0)
        # every segment follows dk/dt ∝ q (v_y, −v_x)
        dkx = np.roll(prepared.kx, -1) - prepared.kx
        dky = np.roll(prepared.ky, -1) - prepared.ky
        assert np.all(np.sign(charge) * (dkx * prepared.vy - dky * prepared.vx) > 0)


def test_hole_circle_is_the_electron_circle_run_backwards():
    """A hole circle with q = −e and an electron circle with q = +e trace the same orbit
    in k-space with v reversed, so they prepare to identical s_n and Δg_n."""
    hole = prepare_contour(*arrays(circle(carrier="hole")), charge=-E)
    electron = prepare_contour(*arrays(circle()), charge=+E)
    assert cyclic_difference(hole, electron) > 0  # velocities differ in sign...
    for name in ("kx", "ky", "tau", "s", "damping"):
        np.testing.assert_array_equal(getattr(hole, name), getattr(electron, name))
    np.testing.assert_array_equal(hole.vx, -electron.vx)


# ---------------------------------------------------------------------------
# Invariance under input order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [lopsided, fourfold, tight_binding_m], ids=["lopsided", "fourfold", "tight-binding"])
@pytest.mark.parametrize("charge", [-E, +E])
def test_reversed_and_rotated_input_give_the_same_contour(make, charge):
    df = make()
    reference = prepare_contour(*arrays(df), charge=charge)
    for label, variant in [
        ("reversed", df.iloc[::-1]),
        ("rotated start", df.iloc[np.roll(np.arange(len(df)), 37)]),
        ("reversed and rotated", df.iloc[np.roll(np.arange(len(df)), 37)[::-1]]),
    ]:
        difference = cyclic_difference(reference, prepare_contour(*arrays(variant), charge=charge))
        print(f"{make.__name__}, q = {np.sign(charge):+.0f}e, {label}: max relative difference = {difference:.2e}")
        assert difference <= 1e-15


# ---------------------------------------------------------------------------
# Segment times, damping, mean free paths and drift removal
# ---------------------------------------------------------------------------

def test_segment_time_damping_and_mean_free_path_formulas():
    """s_n = (ħ|Δk_n|/2e)(1/|v_n| + 1/|v_{n+1}|), Δg_n = 2s_n/(τ_n + τ_{n+1}) and ℓ_n = v_n τ_n."""
    prepared = prepare_contour(*arrays(lopsided()), remove_drift=False)
    kx, ky, vx, vy, tau = prepared.kx, prepared.ky, prepared.vx, prepared.vy, prepared.tau
    length = np.hypot(np.roll(kx, -1) - kx, np.roll(ky, -1) - ky)
    speed = np.hypot(vx, vy)
    np.testing.assert_allclose(prepared.s, HBAR * length / (2 * E) * (1 / speed + 1 / np.roll(speed, -1)),
                               rtol=1e-15)
    np.testing.assert_allclose(prepared.damping, 2 * prepared.s / (tau + np.roll(tau, -1)), rtol=1e-15)
    np.testing.assert_array_equal(prepared.lx, vx * tau)
    np.testing.assert_array_equal(prepared.ly, vy * tau)
    assert prepared.lz is None


def test_orbit_damping_is_the_orbit_average_of_the_scattering_rate():
    """Σ Δg_n → ∮ ds/τ = 2π m_c⟨1/τ⟩/e on a circle, with an O(N⁻²) discretisation error
    (fitted slope in [−2.1, −1.9] over N = 256…4096)."""
    expected = 2 * np.pi * ELECTRON_MASS / (E * TAU)  # ⟨1/τ⟩ = 1/τ₀ for the cos4φ model
    sizes = np.array([256, 512, 1024, 2048, 4096])
    errors = [abs(np.sum(prepare_contour(*arrays(gen.circle(
        n, k_fermi=K_F, mass=ELECTRON_MASS, tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.6)))).damping)
        / expected - 1) for n in sizes]
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    print("sum(damping) / (2 pi m <1/tau> / e) - 1: " + ", ".join(f"{e:.2e}" for e in errors) + f"; slope {slope:.3f}")
    assert -2.1 <= slope <= -1.9


@pytest.mark.parametrize("df, cyclotron_mass", [
    (circle(4096), ELECTRON_MASS),
    (gen.ellipse(4096, k_fermi=K_F, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU, rotation=0.4),
     2 * ELECTRON_MASS),  # √(m_x m_y)
    (lopsided(4096), 2 * ELECTRON_MASS),  # polar(): m is the cyclotron mass for any k_F(φ)
], ids=["circle", "ellipse", "lopsided"])
def test_orbit_geometric_time_is_the_cyclotron_period_times_field(df, cyclotron_mass):
    """Σ s_n = T_c B = 2π m_c / e, up to the O(N⁻²) discretisation error."""
    prepared = prepare_contour(*arrays(df))
    error = abs(np.sum(prepared.s) / (2 * np.pi * cyclotron_mass / E) - 1)
    print(f"sum(s) / (2 pi m_c / e) - 1 = {error:.2e}")
    assert error < 1e-6


@pytest.mark.parametrize("df", [lopsided(64), lopsided(1024), fourfold(), tight_binding_m()],
                         ids=["lopsided-64", "lopsided-1024", "fourfold", "tight-binding"])
def test_drift_removal(df):
    """Symmetric pockets have no drift to remove; the lopsided one has O(N⁻²) drift."""
    raw = prepare_contour(*arrays(df), remove_drift=False)
    removed = prepare_contour(*arrays(df))
    before = drift_sum(raw)
    after = drift_sum(removed)
    print(f"relative drift before = {before:.2e}, after = {after:.2e}")
    assert after <= 1e-14
    # Δg_n keeps the original |ℓ|, and the velocities are left as given; only the mean
    # free paths used downstream are shifted
    for name in ("s", "damping", "vx", "vy"):
        np.testing.assert_array_equal(getattr(removed, name), getattr(raw, name))
    np.testing.assert_array_equal(raw.drift, [0.0, 0.0])
    np.testing.assert_allclose(removed.lx + removed.drift[0], raw.lx, rtol=0, atol=1e-15 * np.max(np.abs(raw.lx)))
    np.testing.assert_allclose(removed.ly + removed.drift[1], raw.ly, rtol=0, atol=1e-15 * np.max(np.abs(raw.ly)))


def test_drift_is_physical_discretisation_error_only():
    """On a coarse lopsided pocket the model's ∮ ℓ dg is visibly non-zero, and it
    shrinks as O(N⁻²) under refinement: it is discretisation error, not physics."""
    drifts = [drift_sum(prepare_contour(*arrays(lopsided(n)), remove_drift=False)) for n in (64, 128, 256)]
    print("relative drift at N = 64, 128, 256:", ", ".join(f"{d:.2e}" for d in drifts))
    assert drifts[0] > 1e-8
    slopes = np.diff(np.log(drifts)) / np.log(2)
    assert np.all(slopes < -1.8)
