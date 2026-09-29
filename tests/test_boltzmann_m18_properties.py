"""M18: properties that must hold for any input, exact scalings, band filling, kernel limits
and field arrays.

The other suites check chosen shapes against known answers. These check what must hold
whatever the input: on seeded random pockets, relaxation times and kernels; under exact
rescalings of k, v, τ and d (which catch a misplaced ħ, e, d or 2π that a consistent test
cannot see); against an independent count of a band's occupied states; in the limits of
the scattering kernel; and for every form a field argument can take. Each test prints what
it checks.
"""
import contourpy
import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann._reference import ParametricContour
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9  # m
TAU = 1e-13  # s


def tensors(sigma):
    """(nB, d, d) from a conductivity DataFrame."""
    values = sigma.iloc[:, 1:].to_numpy()
    dim = int(round(np.sqrt(values.shape[1])))
    return values.reshape(-1, dim, dim)


def relative(actual, expected):
    return np.max(np.abs(np.asarray(actual) - np.asarray(expected))) / np.max(np.abs(np.asarray(expected)))


# ---------------------------------------------------------------------------
# 1. Randomised properties
# ---------------------------------------------------------------------------

HARMONICS = np.arange(2, 6)


def random_case(seed):
    """A random star-shaped pocket, carrier, τ(φ) and symmetric kernel, from a fixed seed."""
    rng = np.random.default_rng(seed)
    k0 = rng.uniform(4e9, 9e9)
    a = rng.uniform(-1, 1, 4)
    a *= rng.uniform(0.02, 0.15) / np.sum(np.abs(a))
    theta = rng.uniform(0, 2 * np.pi, 4)
    b = rng.uniform(-1, 1, 4)
    b *= rng.uniform(0.0, 0.6) / np.sum(np.abs(b))
    psi = rng.uniform(0, 2 * np.pi, 4)
    mass = rng.uniform(0.5, 5.0) * M
    carrier = str(rng.choice(["electron", "hole"]))

    def k_fermi(p):
        p = np.asarray(p, dtype=float)[..., None]
        return k0 * (1 + np.sum(a * np.cos(HARMONICS * p + theta), axis=-1))

    def dk_fermi(p):
        p = np.asarray(p, dtype=float)[..., None]
        return -k0 * np.sum(a * HARMONICS * np.sin(HARMONICS * p + theta), axis=-1)

    def tau(p):
        p = np.asarray(p, dtype=float)[..., None]
        return TAU / (1 + np.sum(b * np.cos(HARMONICS * p + psi), axis=-1))

    angle = rng.uniform(0, 2 * np.pi)
    q = rng.uniform(0.5, 1.8) * k0 * np.array([np.cos(angle), np.sin(angle)])
    width, forward = rng.uniform(0.15, 0.3) * k0, rng.uniform(0.2, 0.4) * k0
    share = rng.uniform(0.2, 0.8)

    def shape(kx, ky, kx2, ky2):
        dx, dy = kx - kx2, ky - ky2
        lobes = (np.exp(-((dx - q[0]) ** 2 + (dy - q[1]) ** 2) / (2 * width**2))
                 + np.exp(-((dx + q[0]) ** 2 + (dy + q[1]) ** 2) / (2 * width**2)))
        return share * lobes + (1 - share) * np.exp(-(dx**2 + dy**2) / (2 * forward**2))

    frame = gen.polar(256, k_fermi=k_fermi, dk_fermi=dk_fermi, mass=mass, tau=1.0, carrier=carrier)
    unit = bz.out_scattering_rate(frame, shape, layer_spacing=D).mean()
    strength = 10 ** rng.uniform(12, 14) / unit

    def kernel(*k):
        return strength * shape(*k)

    return dict(k_fermi=k_fermi, dk_fermi=dk_fermi, tau=tau, mass=mass, carrier=carrier, kernel=kernel,
                parametric=ParametricContour.polar(k_fermi=k_fermi, dk_fermi=dk_fermi, mass=mass, tau=tau,
                                                   carrier=carrier))


def frame_of(case, n, tau=None):
    return gen.polar(n, k_fermi=case["k_fermi"], dk_fermi=case["dk_fermi"], mass=case["mass"],
                     tau=case["tau"] if tau is None else tau, carrier=case["carrier"])


SEEDS = list(range(8))


@pytest.mark.parametrize("seed", SEEDS)
def test_random_pockets_keep_the_universal_properties(seed):
    """On a random pocket, τ(φ) and kernel: unsymmetrised Onsager for the relaxation time
    (1e-12); numba = python (1e-12); σ_sym positive definite at every field, with and without
    the kernel; extrapolate at N = 512 within 1e-6 of the spectral reference, with and without
    the kernel."""
    case = random_case(seed)
    fields = np.array([0.0, 0.1, 1.0, 10.0]) * case["mass"] / (E * TAU)
    df = frame_of(case, 512)
    arrays = [df[c].to_numpy() for c in ("kx", "ky", "vx", "vy", "tau")]
    raw = conductivity_tensor(*arrays, np.concatenate([fields[1:], -fields[1:]]), layer_spacing=D, symmetrize=False)
    onsager = relative(raw[3:], np.transpose(raw[:3], (0, 2, 1)))
    backends = relative(conductivity_tensor(*arrays, fields, layer_spacing=D, backend="python"),
                        conductivity_tensor(*arrays, fields, layer_spacing=D, backend="numba"))
    results = {}
    for label, kernel in (("RTA", None), ("kernel", case["kernel"])):
        options = {} if kernel is None else {"scattering_kernel": kernel}
        extrapolated = tensors(bz.conductivity(df, fields, layer_spacing=D, extrapolate=True, **options))
        reference = ref.collision_sigma([case["parametric"]], fields, kernel=kernel or (lambda *k: 0.0),
                                        layer_spacing=D, points=512)
        symmetric = 0.5 * (extrapolated + np.transpose(extrapolated, (0, 2, 1)))
        smallest = min(np.linalg.eigvalsh(s).min() / np.abs(s).max() for s in symmetric)
        results[label] = (relative(extrapolated, reference), smallest)
    print(f"seed {seed} ({case['carrier']}): Onsager {onsager:.1e}, backends {backends:.1e}; "
          + "; ".join(f"{k}: vs reference {v[0]:.1e}, min eig {v[1]:.1e}" for k, v in results.items()))
    assert onsager <= 1e-12 and backends <= 1e-12
    for error, smallest in results.values():
        assert error <= 1e-6 and smallest > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_random_pockets_high_field_hall_and_isotropic_kernel(seed):
    """High-field R_H = 1/(nq) on a random pocket, with and without its kernel, at ω_cτ ≳ 10³
    with the total rates (1e-3, N = 1024); and an isotropic kernel with constant τ₀ gives the
    relaxation-time result with 1/τ = 1/τ₀ + P·N(E_F) (1e-12)."""
    case = random_case(seed)
    df = frame_of(case, 1024)
    n = bz.carrier_density(df, layer_spacing=D)
    sign = 1.0 if case["carrier"] == "hole" else -1.0
    rates = bz.out_scattering_rate(df, case["kernel"], layer_spacing=D).to_numpy()
    for label, options, fastest in (("RTA", {}, np.max(1 / df.tau)),
                                    ("kernel", {"scattering_kernel": case["kernel"]}, np.max(1 / df.tau + rates))):
        field = 1e3 * case["mass"] * fastest / E
        r_h = bz.hall_coefficient(bz.conductivity(df, [field], layer_spacing=D, **options)).iloc[0]
        print(f"seed {seed} {label}: R_H n q = {r_h * n * sign * E:.6f}")
        assert abs(r_h * n * sign * E - 1) <= 1e-3
    constant = frame_of(case, 256, tau=TAU)
    p_const = 1e13 / bz.density_of_states(constant, layer_spacing=D)
    fields = np.array([0.0, 1.0, 10.0, -10.0]) * case["mass"] / (E * TAU)
    with_kernel = tensors(bz.conductivity(constant, fields, layer_spacing=D, scattering_kernel=lambda *k: p_const))
    rta = tensors(bz.conductivity(constant.assign(tau=1 / (1 / TAU + 1e13)), fields, layer_spacing=D))
    print(f"seed {seed}: isotropic kernel vs RTA {relative(with_kernel, rta):.1e}")
    assert relative(with_kernel, rta) <= 1e-12


# ---------------------------------------------------------------------------
# 2. Exact rescalings
# ---------------------------------------------------------------------------

A_SHEET, G_SHEET = 7.3e-10, 2 * np.pi / 7.3e-10


def lopsided(n=256):
    return gen.polar(n, k_fermi=lambda p: 7e9 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p)), mass=M,
                     tau=lambda p: TAU / (1 + 0.4 * np.cos(2 * p + 0.3)))


def hot(kx, ky, kx2, ky2):
    dx, dy = kx - kx2, ky - ky2
    return (3e-33 * (np.exp(-((dx - 9.9e9) ** 2 + (dy - 1.5e9) ** 2) / (2 * 1.5e9**2))
                     + np.exp(-((dx + 9.9e9) ** 2 + (dy + 1.5e9) ** 2) / (2 * 1.5e9**2)))
            + 1e-33 * np.exp(-(dx**2 + dy**2) / (2 * 2.5e9**2)))


def between_sheets(kx, ky, kx2, ky2):
    """Periodic along the sheets (images k_x ± G) and stronger between them."""
    total = 0.0
    for image in (-1, 0, 1):
        dx = kx - kx2 - image * G_SHEET
        total = total + np.exp(-dx**2 / (2 * 2e9**2))
    return 1e-33 * total * np.where(np.sign(ky) == np.sign(ky2), 1.0, 3.0)


def sheet_frames():
    t = {1: 0.02 * E, 2: 0.006 * E}
    return gen.open_sheets_from_dispersion(
        128, energy=lambda kx, ky: HBAR * 1e5 * (np.abs(ky) - 4e9) - sum(2 * tn * np.cos(n * kx * A_SHEET) for n, tn in t.items()),
        gradient=lambda kx, ky: (sum(2 * tn * n * A_SHEET * np.sin(n * kx * A_SHEET) for n, tn in t.items()),
                                 HBAR * 1e5 * np.sign(ky)),
        period=(G_SHEET, 0.0), across=(-8e9, 8e9), tau=TAU)


def slices():
    df = lopsided(128)
    phi = np.arctan2(df.ky, df.kx)
    parts = [df.assign(kz=-np.pi / D + 2 * np.pi * j / (4 * D), vz=1e3 * np.cos(2 * phi) * np.sin(-np.pi + np.pi * j / 2))
             for j in range(4)]
    return pd.concat(parts, ignore_index=True)


def hot3(kx, ky, kz, kx2, ky2, kz2):
    return hot(kx, ky, kx2, ky2) * (1 + 0.5 * np.cos((kz - kz2) * D))


SYSTEMS = {
    "closed": lambda: (lopsided(), {}, hot),
    "open sheets": lambda: (list(sheet_frames()[0]), {"period": (G_SHEET, 0.0)}, between_sheets),
    "k_z slices": lambda: (slices(), {"kz": "kz"}, hot3),
}


def scaled_k(frames, factor):
    out = [f.assign(kx=f.kx * factor, ky=f.ky * factor) for f in (frames if isinstance(frames, list) else [frames])]
    return out if isinstance(frames, list) else out[0]


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("with_kernel", [False, True], ids=["RTA", "kernel"])
def test_scaling_k_and_field(system, with_kernel):
    """k_∥ → λk_∥ (G → λG), B → λB and P(k, k′) → P(k/λ, k′/λ)/λ give σ → λσ, exactly: the
    density of states grows by λ and every orbit is traversed in the same time."""
    frames, options, kernel = SYSTEMS[system]()
    lam = 3.7
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    base_options = dict(options, **({"scattering_kernel": kernel} if with_kernel else {}))
    base = tensors(bz.conductivity(frames, fields, layer_spacing=D, **base_options))
    scaled_options = dict(options)
    if "period" in options:
        scaled_options["period"] = (lam * G_SHEET, 0.0)
    if with_kernel:
        def scaled_kernel(*k):
            k = list(k)
            half = len(k) // 2
            for i in (0, 1, half, half + 1):
                k[i] = k[i] / lam
            return kernel(*k) / lam
        scaled_options["scattering_kernel"] = scaled_kernel
    scaled = tensors(bz.conductivity(scaled_k(frames, lam), lam * fields, layer_spacing=D, **scaled_options))
    error = relative(scaled, lam * base)
    print(f"{system}, {'kernel' if with_kernel else 'RTA'}: sigma(lambda k, lambda B) vs lambda sigma {error:.1e}")
    assert error <= 1e-12


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("with_kernel", [False, True], ids=["RTA", "kernel"])
def test_scaling_velocity_and_rates(system, with_kernel):
    """v → μv (v_z too), τ₀ → τ₀/μ and P → μ²P at the same B leave σ unchanged, exactly: the
    mean free paths, and ω_cτ, are the same, and the density of states falls by μ."""
    frames, options, kernel = SYSTEMS[system]()
    mu = 2.9
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    extra = {"scattering_kernel": kernel} if with_kernel else {}
    base = tensors(bz.conductivity(frames, fields, layer_spacing=D, **options, **extra))

    def faster(f):
        f = f.assign(vx=f.vx * mu, vy=f.vy * mu, tau=f.tau / mu)
        return f.assign(vz=f.vz * mu) if "vz" in f else f

    moved = [faster(f) for f in frames] if isinstance(frames, list) else faster(frames)
    extra = {"scattering_kernel": lambda *k: mu**2 * kernel(*k)} if with_kernel else {}
    scaled = tensors(bz.conductivity(moved, fields, layer_spacing=D, **options, **extra))
    error = relative(scaled, base)
    print(f"{system}, {'kernel' if with_kernel else 'RTA'}: sigma(mu v, tau/mu) vs sigma {error:.1e}")
    assert error <= 1e-12


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("with_kernel", [False, True], ids=["RTA", "kernel"])
def test_scaling_layer_spacing(system, with_kernel):
    """d → αd and P → αP give σ → σ/α (the k_z slices, and the kernel's k_z argument, are
    rescaled to the new period 2π/αd)."""
    frames, options, kernel = SYSTEMS[system]()
    alpha = 1.9
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    extra = {"scattering_kernel": kernel} if with_kernel else {}
    base = tensors(bz.conductivity(frames, fields, layer_spacing=D, **options, **extra))
    if system == "k_z slices":
        frames = frames.assign(kz=frames.kz / alpha)
        new_kernel = lambda kx, ky, kz, kx2, ky2, kz2: alpha * kernel(kx, ky, kz * alpha, kx2, ky2, kz2 * alpha)  # noqa: E731
    else:
        new_kernel = lambda *k: alpha * kernel(*k)  # noqa: E731
    extra = {"scattering_kernel": new_kernel} if with_kernel else {}
    scaled = tensors(bz.conductivity(frames, fields, layer_spacing=alpha * D, **options, **extra))
    error = relative(scaled, base / alpha)
    print(f"{system}, {'kernel' if with_kernel else 'RTA'}: sigma(alpha d) vs sigma / alpha {error:.1e}")
    assert error <= 1e-12


def test_scaling_of_densities():
    """carrier_density → λ²n and density_of_states → λN under k → λk; N → N/μ under v → μv."""
    df, lam, mu = lopsided(), 3.7, 2.9
    n, dos = bz.carrier_density(df, layer_spacing=D), bz.density_of_states(df, layer_spacing=D)
    assert bz.carrier_density(scaled_k(df, lam), layer_spacing=D) == pytest.approx(lam**2 * n, rel=1e-12)
    assert bz.density_of_states(scaled_k(df, lam), layer_spacing=D) == pytest.approx(lam * dos, rel=1e-12)
    assert bz.density_of_states(df.assign(vx=mu * df.vx, vy=mu * df.vy), layer_spacing=D) == pytest.approx(dos / mu, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Band filling
# ---------------------------------------------------------------------------

LATTICE = 3.87e-10
HOPPINGS = dict(hopping=0.25 * E, next_hopping=-0.0625 * E, third_hopping=0.02 * E)


def band(kx, ky, mu):
    x, y = kx * LATTICE, ky * LATTICE
    return (-2 * HOPPINGS["hopping"] * (np.cos(x) + np.cos(y)) - 4 * HOPPINGS["next_hopping"] * np.cos(x) * np.cos(y)
            - 2 * HOPPINGS["third_hopping"] * (np.cos(2 * x) + np.cos(2 * y)) - mu)


def contour_area(mu, center, points):
    """Area of the pocket about `center` by marching squares on a grid over the zone (m⁻²)."""
    half = np.pi / LATTICE
    axis = np.linspace(-half, half, points)
    kx, ky = np.meshgrid(center[0] + axis, center[1] + axis, indexing="xy")
    lines = contourpy.contour_generator(center[0] + axis, center[1] + axis, band(kx, ky, mu)).lines(0.0)
    areas = [0.5 * abs(np.sum(line[:-1, 0] * line[1:, 1] - line[1:, 0] * line[:-1, 1])
                       + line[-1, 0] * line[0, 1] - line[0, 0] * line[-1, 1]) for line in lines]
    return max(areas)


@pytest.mark.parametrize("mu_ev, pocket", [(-0.6, "electron"), (-0.45, "electron"), (0.0, "hole"), (0.3, "hole")])
def test_band_filling(mu_ev, pocket):
    """carrier_density of a tight_binding pocket equals g_s A/(4π²d) with A from marching squares
    on ε(k) (Richardson in the grid spacing, 1e-7), and the filling A/A_BZ (1 − A/A_BZ for a
    hole pocket) matches a brute-force count of the grid points with ε < 0 (1e-4)."""
    mu = mu_ev * E
    center = (0.0, 0.0) if pocket == "electron" else (np.pi / LATTICE, np.pi / LATTICE)
    df = gen.tight_binding(512, tau=TAU, lattice_constant=LATTICE, chemical_potential=mu, center=center, **HOPPINGS)
    n = bz.carrier_density(df, layer_spacing=D)
    coarse, fine = contour_area(mu, center, 2049), contour_area(mu, center, 4097)
    area = (4 * fine - coarse) / 3
    expected = 2 * area / (4 * np.pi**2 * D)
    zone = (2 * np.pi / LATTICE) ** 2
    grid = (np.arange(4096) + 0.5) / 4096 * 2 * np.pi / LATTICE
    occupied = np.mean(band(*np.meshgrid(grid, grid), mu) < 0)
    filling = area / zone if pocket == "electron" else 1 - area / zone
    print(f"mu = {mu_ev} eV ({pocket}): n vs contour area {n / expected - 1:.1e} (Richardson changed A by "
          f"{fine / area - 1:.1e}); filling {filling:.6f} vs grid count {occupied:.6f}")
    assert abs(n / expected - 1) <= 1e-7 and abs(filling - occupied) <= 1e-4


# ---------------------------------------------------------------------------
# 5. Kernel limits
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_narrow_forward_kernel_does_nothing():
    """A carrier scattered to where it already is carries the same current: with a Gaussian
    kernel of width w in |k − k′| and Γ = 10/τ₀, σ − σ_RTA(τ₀) vanishes as w², with a w⁴
    correction. Halving w from 4e8 to 1e8 m⁻¹ (N = 2048, at least four nodes per width), the
    local slope rises towards 2 and ends in [1.85, 2.15]."""
    df = lopsided(2048).assign(tau=TAU)
    fields = np.array([0.0, 1.0, 10.0]) * M / (E * TAU)
    rta = tensors(bz.conductivity(df, fields, layer_spacing=D))
    widths, differences = (4e8, 2e8, 1e8), []
    for w in widths:
        def shape(kx, ky, kx2, ky2, w=w):
            return np.exp(-((kx - kx2) ** 2 + (ky - ky2) ** 2) / (2 * w**2))
        scale = 10 / TAU / bz.out_scattering_rate(df, shape, layer_spacing=D).mean()
        s = tensors(bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=lambda *k: scale * shape(*k)))
        differences.append(relative(s, rta))
    slopes = [np.log(differences[i] / differences[i + 1]) / np.log(2) for i in range(2)]
    print(f"narrow forward kernel: |sigma - sigma_RTA(tau0)| {', '.join(f'{x:.1e}' for x in differences)}; "
          f"local slopes {slopes[0]:.3f}, {slopes[1]:.3f}")
    assert slopes[0] < slopes[1] and 1.85 <= slopes[1] <= 2.15


def test_kz_independent_surface_with_a_kernel_equals_2d():
    """Identical k_z slices (v_z = 0) with a kernel depending only on k_z − k_z′: every slice
    solves the same problem, so the in-plane σ is the 2D σ with the slice-averaged kernel
    (here the k_z modulation averages to zero), to 1e-12; the z components vanish."""
    df = lopsided(128)
    stacked = pd.concat([df.assign(kz=-np.pi / D + 2 * np.pi * j / (4 * D), vz=0.0) for j in range(4)], ignore_index=True)
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    three = tensors(bz.conductivity(stacked, fields, layer_spacing=D, kz="kz", scattering_kernel=hot3))
    two = tensors(bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=hot))
    error = relative(three[:, :2, :2], two)
    print(f"k_z-independent surface with a kernel: in-plane vs 2D {error:.1e}; max |z parts| "
          f"{max(np.abs(three[:, 2, :]).max(), np.abs(three[:, :, 2]).max()):.1e}")
    assert error <= 1e-12 and not np.any(three[:, 2, :]) and not np.any(three[:, :, 2])


@pytest.mark.parametrize("system", ["closed", "open sheets", "k_z slices", "pure-conserving"])
def test_kernel_path_is_finite_over_sixteen_decades_of_field(system):
    """With a kernel, σ is finite from ω_cτ = 1e-8 to 1e8, and ½(σ + σᵀ) at 1e-8 joins σ(0)
    to 1e-8 of its size."""
    if system == "pure-conserving":
        frames, options, kernel = lopsided().assign(tau=np.inf), {}, hot
    else:
        frames, options, kernel = SYSTEMS[system]()
    per_tesla = E * TAU / M
    fields = np.concatenate([[0.0], np.logspace(-8, 8, 17) / per_tesla])
    s = tensors(bz.conductivity(frames, fields, layer_spacing=D, scattering_kernel=kernel, **options))
    joined = relative(0.5 * (s[1] + s[1].T), s[0])
    print(f"{system}: finite {np.all(np.isfinite(s))}; low field vs sigma(0) {joined:.1e}")
    assert np.all(np.isfinite(s)) and joined <= 1e-8


# ---------------------------------------------------------------------------
# 6. Field arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_kernel", [False, True], ids=["RTA", "kernel"])
def test_field_arrays(with_kernel):
    """Duplicate, unsorted and mixed-sign fields keep their order; equal fields give equal rows
    and a separate call gives the same bits; B and −B rows are transposes; −0.0 is B = 0; a
    scalar, a list, an array and a Series agree; an empty array gives an empty DataFrame."""
    df = lopsided(128)
    options = {"scattering_kernel": hot} if with_kernel else {}
    fields = [3.0, -1.0, 0.0, 3.0, -3.0, 1.0, -0.0]
    s = bz.conductivity(df, fields, layer_spacing=D, **options)
    assert s.field.tolist() == fields
    t = tensors(s)
    assert np.array_equal(t[0], t[3]) and np.array_equal(t[4], t[0].T) and np.array_equal(t[1], t[5].T)
    assert np.array_equal(t[6], t[2])
    for i, b in enumerate(fields):
        assert np.array_equal(tensors(bz.conductivity(df, b, layer_spacing=D, **options))[0], t[i])
    forms = [2.0, [2.0], np.array([2.0]), pd.Series([2.0])]
    results = [bz.conductivity(df, f, layer_spacing=D, **options).to_numpy() for f in forms]
    assert all(np.array_equal(results[0], r) for r in results[1:])
    empty = bz.conductivity(df, [], layer_spacing=D, **options)
    assert empty.shape == (0, 5) and list(empty.columns) == ["field", "sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]


@pytest.mark.parametrize("with_kernel", [False, True], ids=["RTA", "kernel"])
@pytest.mark.parametrize("field, message", [([1.0, np.nan], "finite"), ([np.inf], "finite"), ([[0.0, 1.0]], "1-D")])
def test_bad_fields_raise_the_same_clear_error(with_kernel, field, message):
    options = {"scattering_kernel": hot} if with_kernel else {}
    with pytest.raises(ValueError, match=message):
        bz.conductivity(lopsided(64), field, layer_spacing=D, **options)
