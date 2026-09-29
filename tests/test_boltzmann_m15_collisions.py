"""M15: the collision solver, the Boltzmann equation with a scattering kernel.

With a kernel P(k, k′), `conductivity` solves c = ℓ + Ĵ G_Ω c on the nodes, a Galerkin
method on the exact relaxation-time propagator. These tests check its limits and
structure against the relaxation-time path, the spectral reference and exact identities:
the RTA limit, convergence to the reference, no term linear in |B|, conservation, Onsager,
positivity, continuity as the background vanishes, invariances, errors and warnings. The
closed forms are known results K21–K26. Each test prints what it checks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _collisions
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann._reference import ParametricContour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9  # m
TAU = 1e-13  # s
COLUMNS = ["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]
K0 = 7e9


def tensors(sigma):
    """(nB, d, d) from a conductivity DataFrame."""
    values = sigma.iloc[:, 1:].to_numpy()
    dim = int(round(np.sqrt(values.shape[1])))
    return values.reshape(-1, dim, dim)


def relative(actual, expected):
    return np.max(np.abs(np.asarray(actual) - np.asarray(expected))) / np.max(np.abs(np.asarray(expected)))


# --- contours and kernels shared by the tests --------------------------------------------------------------

def lopsided_k(p):
    return K0 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p))


def lopsided_dk(p):
    return K0 * (0.4 * np.sin(4 * p) + 0.15 * np.cos(3 * p))


def lopsided(n, tau=TAU):
    return gen.polar(n, k_fermi=lopsided_k, mass=M, tau=tau)


Q = (9.9e9, 1.5e9)


def hot_spots(kx, ky, kx2, ky2):
    """Scattering to near k ± Q (hot spots) plus forward scattering; symmetric, no symmetry of its own."""
    dx, dy = kx - kx2, ky - ky2
    return (3e-33 * (np.exp(-((dx - Q[0]) ** 2 + (dy - Q[1]) ** 2) / (2 * 1.5e9**2))
                     + np.exp(-((dx + Q[0]) ** 2 + (dy + Q[1]) ** 2) / (2 * 1.5e9**2)))
            + 1e-33 * np.exp(-(dx**2 + dy**2) / (2 * 2.5e9**2)))


def lobes(kx, ky, kx2, ky2):
    cosine = (kx * kx2 + ky * ky2) / (np.hypot(kx, ky) * np.hypot(kx2, ky2))
    return 2e-33 * np.exp(20 * (cosine - 1)) + 1e-33 * np.exp(8 * (-cosine - 1))


def gaussian(kx, ky, kx2, ky2):
    return 2e-33 * np.exp(-((kx - kx2) ** 2 + (ky - ky2) ** 2) / (2 * 3e9**2))


def fourfold_k(p):
    return 7.35e9 - 0.25e9 * np.cos(4 * p)


def fourfold_dk(p):
    return 1e9 * np.sin(4 * p)


A = 7.3e-10
BAND = dict(velocity=1e5, k0=4e9, lattice_constant=A, hoppings={1: 0.02 * E, 2: 0.006 * E, 3: 0.002 * E})
TZ = 0.002 * E


def band_energy(kx, ky, kz):
    return (HBAR * BAND["velocity"] * (np.abs(ky) - BAND["k0"])
            - sum(2 * t * np.cos(n * kx * A) for n, t in BAND["hoppings"].items()) - 2 * TZ * np.cos(kz * D))


def band_gradient(kx, ky, kz):
    return (sum(2 * t * n * A * np.sin(n * kx * A) for n, t in BAND["hoppings"].items()),
            HBAR * BAND["velocity"] * np.sign(ky), 2 * TZ * D * np.sin(kz * D) + 0 * kx)


def sheet_kernel(kx, ky, kz, kx2, ky2, kz2):
    """Between the sheets, peaked at Δk_x = π/a, and weaker between distant k_z slices."""
    cosine = np.cos((kx - kx2) * A)
    within = 3e-33 * np.exp(4 * (cosine - 1))
    between = 2e-33 * np.exp(6 * (-cosine - 1))
    return np.where(np.sign(ky) == np.sign(ky2), within, between) * (1 + 0.5 * np.cos((kz - kz2) * D))


def warped_sheets(n, n_kz, tau=TAU):
    return gen.open_sheets_from_dispersion(n, energy=band_energy, gradient=band_gradient, period=(2 * np.pi / A, 0.0),
                                           across=(-2 * BAND["k0"], 2 * BAND["k0"]), tau=tau, n_kz=n_kz,
                                           layer_spacing=D)


# --- the relaxation-time limit ----------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["closed", "open", "3D"])
def test_zero_kernel_is_the_relaxation_time_path(case):
    """P ≡ 0 given as a function: σ equals the relaxation-time path bitwise, at every field and
    both signs, because contours the kernel does not touch go through that path unchanged
    (with `extrapolate` the Richardson step is taken on the sum, so equal to 1e-13 instead)."""
    fields = [0.0, 1.0, 30.0, -30.0]
    if case == "closed":
        dfs, options = [lopsided(256, tau=lambda p: TAU / (1 + 0.3 * np.cos(2 * p)))], {}
    elif case == "open":
        g = 2 * np.pi / 3.87e-10
        dfs, options = gen.open_sheets(128, k0=5e9, velocity=2e5, tau=TAU, period=g, warping=1e9), {"period": (0.0, g)}
    else:
        sheets, period = warped_sheets(64, 4)
        dfs, options = sheets, {"period": period, "kz": "kz"}
    rta = bz.conductivity(dfs, fields, layer_spacing=D, **options)
    zero = bz.conductivity(dfs, fields, layer_spacing=D, scattering_kernel=lambda *k: 0.0, **options)
    print(f"{case}: zero kernel vs relaxation time, max difference {np.max(np.abs(zero.to_numpy() - rta.to_numpy()))}")
    assert np.array_equal(zero.to_numpy(), rta.to_numpy())
    rta = bz.conductivity(dfs, fields, layer_spacing=D, extrapolate=True, **options)
    zero = bz.conductivity(dfs, fields, layer_spacing=D, scattering_kernel=lambda *k: 0.0, extrapolate=True, **options)
    assert np.max(np.abs(zero.to_numpy() - rta.to_numpy())) <= 1e-13 * np.max(np.abs(rta.to_numpy()))


# --- against the spectral reference --------------------------------------------------------------------------

def _reference_cases():
    """name: (node DataFrames at N, reference contours, kernel, options, N list, kz, weights)."""
    return {
        "fourfold pocket, forward and backward lobes": (
            lambda n: [gen.polar(n, k_fermi=fourfold_k, mass=M, tau=TAU)],
            [ParametricContour.polar(k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=M, tau=TAU)],
            lobes, {}, (128, 256, 512, 1024), None, None),
        "lopsided pocket, hot spots": (
            lambda n: [lopsided(n)],
            [ParametricContour.polar(k_fermi=lopsided_k, dk_fermi=lopsided_dk, mass=M, tau=TAU)],
            hot_spots, {}, (128, 256, 512, 1024), None, None),
        "electron circle in a hole pocket, coupled": (
            lambda n: [gen.circle(n, k_fermi=3e9, mass=M, tau=TAU),
                       gen.polar(n, k_fermi=fourfold_k, mass=M, tau=TAU, carrier="hole")],
            [ParametricContour.circle(k_fermi=3e9, mass=M, tau=TAU),
             ParametricContour.polar(k_fermi=fourfold_k, dk_fermi=fourfold_dk, mass=M, tau=TAU, carrier="hole")],
            gaussian, {}, (128, 256, 512, 1024), None, None),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_reference_cases()))
def test_converges_to_the_spectral_reference(name):
    """The error falls as N⁻² (slope in [−2.1, −1.9] over N = 128…1024 at ω_cτ = 0.1, 1 and 10),
    and the Richardson value (4σ₁₀₂₄ − σ₅₁₂)/3 agrees with the reference to 1e-6."""
    frames, contours, kernel, options, sizes, kz, weights = _reference_cases()[name]
    fields = np.array([0.1, 1.0, 10.0]) * M / (E * TAU)
    reference = ref.collision_sigma(contours, fields, kernel=kernel, layer_spacing=D, points=512)
    sigma = {n: tensors(bz.conductivity(frames(n), fields, layer_spacing=D, scattering_kernel=kernel, **options))
             for n in sizes}
    for i, x in enumerate((0.1, 1.0, 10.0)):
        errors = [relative(sigma[n][i], reference[i]) for n in sizes]
        slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
        richardson = relative((4 * sigma[sizes[-1]][i] - sigma[sizes[-2]][i]) / 3, reference[i])
        print(f"{name}, omega_c tau = {x}: errors {', '.join(f'{e:.1e}' for e in errors)}; slope {slope:.2f}; "
              f"Richardson {richardson:.1e}")
        assert -2.1 <= slope <= -1.9 and richardson <= 1e-6


@pytest.mark.slow
def test_converges_to_the_spectral_reference_warped_sheets_in_3d():
    """Warped open sheets in four k_z slices, with a kernel between the sheets that also
    depends on k_z − k_z′ (no closed form): N⁻² convergence, and Richardson (256/512) to 1e-6."""
    n_kz = 4
    kzs = [-np.pi / D + 2 * np.pi * j / (n_kz * D) for j in range(n_kz)]
    contours = [ParametricContour.warped_band_sheet(**BAND, side=s, tau=TAU, interlayer_hopping=TZ, kz=kz,
                                                    layer_spacing=D) for kz in kzs for s in (1, -1)]
    fields = np.array([0.1, 1.0, 10.0]) / (E * BAND["velocity"] * A * TAU / HBAR)
    reference = ref.collision_sigma(contours, fields, kernel=sheet_kernel, layer_spacing=D, points=256,
                                    weights=[1 / n_kz] * len(contours), kz=[kz for kz in kzs for _ in (1, -1)])
    sizes = (64, 128, 256, 512)
    sigma = {}
    for n in sizes:
        sheets, period = warped_sheets(n, n_kz)
        sigma[n] = tensors(bz.conductivity(sheets, fields, layer_spacing=D, kz="kz", period=period,
                                           scattering_kernel=sheet_kernel))
    for i, x in enumerate((0.1, 1.0, 10.0)):
        errors = [relative(sigma[n][i], reference[i]) for n in sizes]
        slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
        richardson = relative((4 * sigma[512][i] - sigma[256][i]) / 3, reference[i])
        print(f"3D sheets, omega tau = {x}: errors {', '.join(f'{e:.1e}' for e in errors)}; slope {slope:.2f}; "
              f"Richardson {richardson:.1e}")
        assert -2.1 <= slope <= -1.9 and richardson <= 1e-6


# --- structure ------------------------------------------------------------------------------------------------

def test_no_term_linear_in_field():
    """The low-field curvature ratio [σ_sym(2B) − σ(0)] / [σ_sym(B) − σ(0)] is 4 ± 0.01 on the
    diagonal at ω_cτ = 1e-5, 1e-4 and 1e-3: in-scattering adds no |B|-linear artefact."""
    df = lopsided(256)
    for x in (1e-5, 1e-4, 1e-3):
        b = x * M / (E * TAU)
        s = tensors(bz.conductivity(df, [0.0, b, 2 * b], layer_spacing=D, scattering_kernel=hot_spots))
        sym = 0.5 * (s + np.transpose(s, (0, 2, 1)))
        ratio = np.diagonal((sym[2] - s[0]) / (sym[1] - s[0]))
        print(f"omega_c tau = {x:g}: curvature ratio {ratio}")
        assert np.all(np.abs(ratio - 4) <= 0.01)


def test_conservation_onsager_and_positivity():
    """ĴΩm = τΓ on the nodes to 1e-14 (the kernel conserves particles exactly); σ(0) is
    symmetric; σ(−B) = σ(B)ᵀ; the symmetric part of σ is positive definite at every field."""
    df = lopsided(256)
    contour = _collisions.ContourInput(*(df[c].to_numpy() for c in ("kx", "ky", "vx", "vy")),
                                       tau0=np.full(len(df), TAU))
    system = _collisions.build_system([contour], hot_spots, layer_spacing=D, charge=-E, remove_drift=None)
    component = system.components[0]
    prepared = system.prepared[0]
    mass = 0.5 * (np.roll(prepared.damping, 1) + prepared.damping)
    share = prepared.tau * system.gamma
    print(f"J Omega m vs tau Gamma: {np.max(np.abs(component.jmat @ mass - share)):.1e}")
    assert np.max(np.abs(component.jmat @ mass - share)) <= 1e-14
    fields = np.array([0.0, 0.1, 1.0, 10.0, 100.0]) * M / (E * TAU)
    s = tensors(bz.conductivity(df, np.concatenate([fields, -fields[1:]]), layer_spacing=D, scattering_kernel=hot_spots,
                                symmetrize=False))
    assert np.max(np.abs(s[0] - s[0].T)) <= 1e-15 * np.max(np.abs(s[0]))
    assert np.array_equal(s[5:], np.transpose(s[1:5], (0, 2, 1)))
    smallest = min(np.linalg.eigvalsh(0.5 * (m + m.T)).min() / np.abs(m).max() for m in s)
    print(f"smallest eigenvalue of sigma_sym / max|sigma|: {smallest:.2e}")
    assert smallest > 0


def _noisy_sheets(n=256, noise=0.01):
    """Warped open sheets whose speeds scatter by 1%, as measured contours would."""
    rng = np.random.default_rng(3)
    g, k0, v0, w = 2 * np.pi / 3.87e-10, 5e9, 2e5, 0.5e9
    frames = []
    for side in (1, -1):
        ky = g * (np.arange(n) / n - 0.5)
        speed = 1 + noise * rng.standard_normal(n)
        frames.append(pd.DataFrame({"kx": side * (k0 + w * np.cos(2 * np.pi / g * ky)), "ky": ky,
                                    "vx": side * v0 * speed, "vy": v0 * w * 2 * np.pi / g * np.sin(2 * np.pi / g * ky) * speed,
                                    "tau": np.inf}))
    return frames, (0.0, g)


def _images(kernel_width, g):
    def kernel(kx, ky, kx2, ky2):
        total = 0.0
        for image in (-1, 0, 1):
            total = total + np.exp(-((kx - kx2) ** 2 + (ky - ky2 - image * g) ** 2) / (2 * kernel_width**2))
        return 2e-33 * total
    return kernel


@pytest.mark.parametrize("case", ["lopsided pocket", "noisy open sheets"])
def test_continuous_as_the_background_vanishes(case):
    """Through the charge mode: the difference from τ₀ = ∞ falls as 1/τ₀ (slope −1 ± 0.05 over
    τ₀Γ̄ = 1e6, 1e8, 1e10) and is ≤ 1e-8 at 1e10. The noisy sheets carry a discretisation drift,
    which must be removed whatever the background (otherwise σ grows as D²/b)."""
    if case == "lopsided pocket":
        frames, kernel, options = [lopsided(256, tau=np.inf)], hot_spots, {}
    else:
        frames, period = _noisy_sheets()
        kernel, options = _images(6e9, period[1]), {"period": period}
    fields = [0.0, 1.0 * M / (E * TAU)]
    limit = tensors(bz.conductivity(frames, fields, layer_spacing=D, scattering_kernel=kernel, **options))
    rates = bz.out_scattering_rate(frames, kernel, layer_spacing=D, **options)
    mean_rate = np.mean(np.concatenate([np.asarray(r) for r in rates]))
    ratios, differences = (1e6, 1e8, 1e10), []
    for ratio in ratios:
        with_background = [f.assign(tau=ratio / mean_rate) for f in frames]
        s = tensors(bz.conductivity(with_background, fields, layer_spacing=D, scattering_kernel=kernel, **options))
        differences.append(relative(s, limit))
    slope = np.polyfit(np.log(ratios), np.log(differences), 1)[0]
    print(f"{case}: differences {', '.join(f'{d:.1e}' for d in differences)}; slope {slope:.3f}")
    assert abs(slope + 1) <= 0.05 and differences[-1] <= 1e-8


def test_invariances():
    """Scaling λσ(λB; λP, τ₀/λ) = σ; the start index and the input direction do not matter;
    rotating the contour and the kernel together gives RσRᵀ (all 1e-12)."""
    df = lopsided(256)
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    base = tensors(bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=hot_spots))
    lam = 7.0
    scaled = tensors(bz.conductivity(df.assign(tau=df.tau / lam), lam * fields, layer_spacing=D,
                                     scattering_kernel=lambda *k: lam * hot_spots(*k)))
    rolled = tensors(bz.conductivity(df.iloc[np.roll(np.arange(256), 77)], fields, layer_spacing=D,
                                     scattering_kernel=hot_spots))
    reversed_ = tensors(bz.conductivity(df.iloc[::-1], fields, layer_spacing=D, scattering_kernel=hot_spots))
    alpha = 0.6
    c, s = np.cos(alpha), np.sin(alpha)
    turned = df.assign(kx=c * df.kx - s * df.ky, ky=s * df.kx + c * df.ky, vx=c * df.vx - s * df.vy, vy=s * df.vx + c * df.vy)

    def turned_kernel(kx, ky, kx2, ky2):
        return hot_spots(c * kx + s * ky, -s * kx + c * ky, c * kx2 + s * ky2, -s * kx2 + c * ky2)

    rotated = tensors(bz.conductivity(turned, fields, layer_spacing=D, scattering_kernel=turned_kernel))
    rotation = np.array([[c, -s], [s, c]])
    errors = {"scaling": relative(lam * scaled, base), "start index": relative(rolled, base),
              "reversed": relative(reversed_, base), "rotation": relative(rotated, rotation @ base @ rotation.T)}
    print(", ".join(f"{k} {v:.1e}" for k, v in errors.items()))
    assert all(v <= 1e-12 for v in errors.values())


def test_uncoupled_components_add():
    """Two pockets with kernels that do not couple them: σ is the sum of solving each alone
    (1e-12), including with no background (each is bordered on its own)."""
    electrons = gen.circle(256, k_fermi=3e9, mass=M, tau=np.inf)
    holes = gen.polar(256, k_fermi=fourfold_k, mass=M, tau=np.inf, carrier="hole")

    def separate(kx, ky, kx2, ky2):
        inner, inner2 = np.hypot(kx, ky) < 5e9, np.hypot(kx2, ky2) < 5e9
        return np.where(inner == inner2, gaussian(kx, ky, kx2, ky2), 0.0)

    fields = [0.0, 3.0, 30.0, -30.0]
    both = tensors(bz.conductivity([electrons, holes], fields, layer_spacing=D, scattering_kernel=separate))
    alone = sum(tensors(bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=separate))
                for df in (electrons, holes))
    print(f"together vs apart: {relative(both, alone):.1e}")
    assert relative(both, alone) <= 1e-12


# --- errors and warnings ----------------------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:the scattering kernel changes too fast")
def test_rejects_bad_kernels_and_inputs(monkeypatch):
    df = lopsided(64)
    run = lambda frames, kernel, **kw: bz.conductivity(frames, [0.0, 1.0], layer_spacing=D,  # noqa: E731
                                                       scattering_kernel=kernel, **kw)
    with pytest.raises(ValueError, match="not symmetric"):
        run(df, lambda kx, ky, kx2, ky2: 1e-33 * np.exp(-((kx - kx2 - 3e9) ** 2 + (ky - ky2) ** 2) / 1e19))
    with pytest.raises(ValueError, match="negative"):
        run(df, lambda *k: -1e-33)
    with pytest.raises(ValueError, match="finite"):
        run(df, lambda *k: np.inf)
    with pytest.raises(ValueError, match="broadcasts"):
        run(df, lambda *k: np.ones(3))
    with pytest.raises(ValueError, match="nothing scatters"):
        run(df.assign(tau=np.inf), lambda *k: 0.0)
    sheet = gen.open_sheets(64, k0=5e9, velocity=2e5, tau=np.inf, period=1.6e10, warping=1e9)[0]
    with pytest.raises(ValueError, match="net current"):
        run(sheet, lambda *k: 1e-33, period=(0.0, 1.6e10))
    with pytest.warns(UserWarning, match="net current"):
        run(sheet.assign(tau=TAU), lambda *k: 1e-33, period=(0.0, 1.6e10))
    with pytest.raises(ValueError, match="even number of distinct nodes"):
        run(lopsided(65), hot_spots, extrapolate=True)
    monkeypatch.setattr(_collisions, "MAX_COMPONENT_NODES", 50)
    with pytest.raises(ValueError, match="more than 50"):
        run(df, hot_spots)


def test_resolution_warning():
    """The quadrature self-check warns for a kernel narrower than the node spacing, and stays
    quiet for a resolved one (hot spots on a non-circular pocket)."""
    df = lopsided(256)

    def narrow(kx, ky, kx2, ky2):
        dx, dy = kx - kx2, ky - ky2
        width = 2e7  # the nodes are ~1.7e8 apart
        return 3e-31 * (np.exp(-((dx - Q[0]) ** 2 + (dy - Q[1]) ** 2) / (2 * width**2))
                        + np.exp(-((dx + Q[0]) ** 2 + (dy + Q[1]) ** 2) / (2 * width**2))) + 1e-35

    with pytest.warns(UserWarning, match="changes too fast"):
        bz.conductivity(df, [0.0, 1.0], layer_spacing=D, scattering_kernel=narrow)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bz.conductivity(df, [0.0, 1.0], layer_spacing=D, scattering_kernel=hot_spots)
    assert not [w for w in caught if "changes too fast" in str(w.message)]


def test_onsager_check_fires_on_a_corrupted_propagator(monkeypatch):
    real = _collisions.propagator

    def corrupted(damping, field):
        g = real(damping, field)
        g[0, 1] *= 1 + 1e-6
        return g

    monkeypatch.setattr(_collisions, "propagator", corrupted)
    with pytest.warns(RuntimeWarning, match="Onsager"):
        bz.conductivity(lopsided(64), [1.0], layer_spacing=D, scattering_kernel=hot_spots)


def test_backend_routes_to_the_same_solver():
    df = lopsided(128)
    python = bz.conductivity(df, [0.0, 3.0], layer_spacing=D, scattering_kernel=hot_spots, backend="python")
    numba = bz.conductivity(df, [0.0, 3.0], layer_spacing=D, scattering_kernel=hot_spots, backend="numba")
    assert np.array_equal(python.to_numpy(), numba.to_numpy())
    with pytest.raises(ValueError, match="backend"):
        bz.conductivity(df, [0.0], layer_spacing=D, scattering_kernel=hot_spots, backend="fortran")
