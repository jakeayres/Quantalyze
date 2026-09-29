"""M13: closed forms and the spectral reference for scattering kernels.

`_reference.collision_sigma` solves the linearised Boltzmann equation with a kernel
P(k, k′) by spectral collocation on parametrised contours. Before it is used to check
the fast solver, it must reproduce the brute-force RTA reference with no kernel, and the
closed forms K22–K24 and the isotropic-kernel result with one. Each test prints what it
checks.
"""
import numpy as np
import pytest

from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import _reference as ref
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._reference import ParametricContour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9  # m
TAU = 1e-13  # s


def lobes_kernel(lobes):
    """Σ A exp(κ(s k̂·k̂′ − 1)) for lobes (A, κ, s): rotation invariant on a circle about k = 0."""
    def kernel(kx, ky, kx2, ky2):
        cosine = (kx * kx2 + ky * ky2) / (np.hypot(kx, ky) * np.hypot(kx2, ky2))
        return sum(a * np.exp(k * (s * cosine - 1)) for a, k, s in lobes)
    return kernel


# A pocket with no symmetry, and a kernel with hot spots near k′ = k ± Q plus forward scattering.
K0 = 7e9
LOPSIDED = ParametricContour.polar(k_fermi=lambda p: K0 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p)),
                                   dk_fermi=lambda p: K0 * (0.4 * np.sin(4 * p) + 0.15 * np.cos(3 * p)),
                                   mass=M, tau=TAU)
Q = (9.9e9, 1.5e9)


def hot_spots(kx, ky, kx2, ky2):
    dx, dy = kx - kx2, ky - ky2
    return (3e-33 * (np.exp(-((dx - Q[0]) ** 2 + (dy - Q[1]) ** 2) / (2 * 1.5e9**2))
                     + np.exp(-((dx + Q[0]) ** 2 + (dy + Q[1]) ** 2) / (2 * 1.5e9**2)))
            + 1e-33 * np.exp(-(dx**2 + dy**2) / (2 * 2.5e9**2)))


def no_kernel(*k):
    return 0.0


def report(label, actual, expected):
    error = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    print(f"{label}: max |sigma - expected| / max|expected| = {error:.1e}")
    return error


@pytest.mark.parametrize("name", ["circle", "fourfold hole pocket", "warped open sheet"])
def test_without_kernel_matches_the_brute_force_reference(name):
    """P = 0: the collocation reduces to the relaxation-time approximation, and matches the
    independent brute-force quadrature at ω_cτ = 0, 0.1, 1, 10 and −1 (1e-9)."""
    contour = {
        "circle": ParametricContour.circle(k_fermi=K0, mass=M, tau=TAU),
        "fourfold hole pocket": ParametricContour.polar(
            k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), dk_fermi=lambda p: 1e9 * np.sin(4 * p), mass=5 * M,
            tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.3), carrier="hole"),
        "warped open sheet": ParametricContour.open_sheet(k0=5e9, velocity=2e5, warping=1e9,
                                                          period=2 * np.pi / 3.87e-10, tau=TAU),
    }[name]
    fields = np.array([0.0, 0.1, 1.0, 10.0, -1.0]) * M / (E * TAU)
    actual = ref.collision_sigma([contour], fields, kernel=no_kernel, layer_spacing=D, points=512)
    expected = ref.sigma(contour, fields, layer_spacing=D)
    assert report(name, actual, expected) <= 1e-9


@pytest.mark.parametrize("lobes", [[(2e-33, 20.0, 1)], [(1e-33, 8.0, -1)], [(2e-33, 20.0, 1), (1e-33, 8.0, -1)]],
                         ids=["forward", "backward", "both"])
def test_k22_circle(lobes):
    """K22: on a circle, harmonics diagonalise a rotation-invariant kernel, so σ is Drude with
    1/τ_tr = 1/τ₀ + Γ − Γ₁ at every field (1e-10)."""
    mass, kf = 2 * M, K0
    gamma, gamma1, tau_tr = an.circle_transport_time(tau0=TAU, mass=mass, layer_spacing=D, lobes=lobes)
    print(f"1/tau0 = {1 / TAU:.2e}, Gamma = {gamma:.2e}, Gamma_1 = {gamma1:.2e}, tau_tr = {tau_tr:.3e} s")
    fields = np.array([0.0, 0.1, 1.0, 10.0, -1.0]) * mass / (E * tau_tr)
    actual = ref.collision_sigma([ParametricContour.circle(k_fermi=kf, mass=mass, tau=TAU)], fields,
                                 kernel=lobes_kernel(lobes), layer_spacing=D, points=256)
    expected = an.drude_circle(fields, density=an.circle_density(kf, D), mass=mass, tau=tau_tr)
    assert report("K22", actual, expected) <= 1e-10


@pytest.mark.parametrize("tau0", [TAU, np.inf])
def test_k23_flat_sheets(tau0):
    """K23: flat sheets. Scattering within a sheet leaves the current alone; scattering between
    them relaxes it twice: 1/τ_tr = 1/τ₀ + 2N_s A_opp at every field (1e-10)."""
    k0, v0, g = 5e9, 2e5, 2 * np.pi / 3.87e-10
    a_same, a_opp = 4e-33, 1.5e-33

    def kernel(kx, ky, kx2, ky2):
        return np.where(np.sign(kx) == np.sign(kx2), a_same * (1 + np.cos(2 * np.pi / g * (ky - ky2))), a_opp)

    sheets = [ParametricContour.open_sheet(k0=k0, velocity=v0, warping=0.0, period=g, tau=tau0, side=s) for s in (1, -1)]
    fields = np.array([0.0, 1.0, 1e3, -1e3])
    actual = ref.collision_sigma(sheets, fields, kernel=kernel, layer_spacing=D, points=64)
    expected = an.flat_sheets_collisions(fields, velocity=v0, period=g, layer_spacing=D, tau0=tau0, inter=a_opp)
    assert report(f"K23 tau0 = {tau0}", actual, expected) <= 1e-10


BAND = dict(velocity=1e5, k0=4e9, lattice_constant=7.3e-10, hoppings={1: 0.02 * E, 2: 0.006 * E, 3: 0.002 * E})
INTRA, INTER = (3e-33, 4.0), (2e-33, 6.0)


def band_kernel(kx, ky, *rest):
    """Within a sheet A_s e^{κ_s(cos aΔ − 1)}, between them A_o e^{κ_o(−cos aΔ − 1)}; no k_z dependence."""
    kx2, ky2 = (rest[1], rest[2]) if len(rest) == 4 else (rest[0], rest[1])  # rest is (kz, kx2, ky2, kz2) in 3D
    cosine = np.cos((kx - kx2) * BAND["lattice_constant"])
    return np.where(np.sign(ky) == np.sign(ky2), INTRA[0] * np.exp(INTRA[1] * (cosine - 1)),
                    INTER[0] * np.exp(INTER[1] * (-cosine - 1)))


@pytest.mark.parametrize("three_d", [False, True], ids=["2D", "3D"])
def test_k24_warped_sheets(three_d):
    """K24: the harmonics of the warped sheets decouple; the sheets move in opposite directions
    and the kernel between them couples each harmonic across the pair. σ_xx(B), σ_yy, σ_zz
    and zero off-diagonals from the closed form (1e-10)."""
    tz, nz = 0.002 * E, 6
    tau0 = TAU
    total = 1 / tau0 + 1e13  # rough scale for the fields
    fields = np.array([0.0, 0.3, 3.0, -3.0]) * total / (E * BAND["velocity"] * BAND["lattice_constant"] / HBAR)
    if three_d:
        kzs = [-np.pi / D + 2 * np.pi * j / (nz * D) for j in range(nz)]
        contours = [ParametricContour.warped_band_sheet(**BAND, side=s, tau=tau0, interlayer_hopping=tz, kz=kz,
                                                        layer_spacing=D) for kz in kzs for s in (1, -1)]
        actual = ref.collision_sigma(contours, fields, kernel=band_kernel, layer_spacing=D, points=64,
                                     weights=[1 / nz] * len(contours), kz=[kz for kz in kzs for _ in (1, -1)])
    else:
        contours = [ParametricContour.warped_band_sheet(**BAND, side=s, tau=tau0) for s in (1, -1)]
        actual = ref.collision_sigma(contours, fields, kernel=band_kernel, layer_spacing=D, points=64)
    expected = an.warped_sheets_collisions(
        fields, velocity=BAND["velocity"], lattice_constant=BAND["lattice_constant"], hoppings=BAND["hoppings"],
        layer_spacing=D, tau0=tau0, intra=INTRA, inter=INTER, interlayer_hopping=tz if three_d else None)
    diagonal = np.diagonal(actual, axis1=1, axis2=2)
    error = np.max(np.abs(diagonal / np.diagonal(expected, axis1=1, axis2=2) - 1))
    off = np.max(np.abs(actual - np.einsum("bi,ij->bij", diagonal, np.eye(diagonal.shape[1])))) / np.max(np.abs(actual))
    print(f"K24 {'3D' if three_d else '2D'}: diagonal max rel error {error:.1e}; off-diagonal / max {off:.1e}")
    assert error <= 1e-10 and off <= 1e-10


@pytest.mark.parametrize("tau0", [TAU, np.inf])
def test_isotropic_kernel_is_the_relaxation_time_result(tau0):
    """A constant kernel P: its in-scattering is proportional to the density of the current-
    carrying solution, which vanishes, so σ is the RTA result with 1/τ = 1/τ₀ + P·N(E_F),
    here from the brute-force reference (1e-10)."""
    kfermi, dkfermi = (lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p)), (lambda p: 1e9 * np.sin(4 * p))
    pocket = ParametricContour.polar(k_fermi=kfermi, dk_fermi=dkfermi, mass=5 * M, tau=tau0, carrier="hole")
    phi = 2 * np.pi * np.arange(1024) / 1024
    dkx, dky = pocket.dk(phi)
    vx, vy = pocket.v(phi)
    dos = np.sum(np.hypot(dkx, dky) / np.hypot(vx, vy)) * (2 * np.pi / 1024) / (4 * np.pi**2 * HBAR * D)
    p_const = 1e13 / dos
    tau = 1 / (1 / tau0 + p_const * dos)
    fields = np.array([0.0, 1.0, 30.0, -30.0])
    actual = ref.collision_sigma([pocket], fields, kernel=lambda *k: p_const, layer_spacing=D, points=256)
    rta = ParametricContour.polar(k_fermi=kfermi, dk_fermi=dkfermi, mass=5 * M, tau=tau, carrier="hole")
    assert report(f"isotropic kernel, tau0 = {tau0}", actual, ref.sigma(rta, fields, layer_spacing=D)) <= 1e-10


def test_onsager_without_symmetrising():
    """σ(−B) = σ(B)ᵀ for a symmetric kernel, with no symmetry of the pocket or the kernel (1e-10)."""
    fields = np.array([0.3, 3.0]) * M / (E * TAU)
    plus = ref.collision_sigma([LOPSIDED], fields, kernel=hot_spots, layer_spacing=D, points=256)
    minus = ref.collision_sigma([LOPSIDED], -fields, kernel=hot_spots, layer_spacing=D, points=256)
    assert report("sigma(-B) vs sigma(B)^T", minus, np.transpose(plus, (0, 2, 1))) <= 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("tau0", [TAU, np.inf])
def test_converged_in_the_number_of_points(tau0):
    """Doubling the collocation points from 512 to 1024 changes σ by less than 1e-10."""
    contour = ParametricContour.polar(k_fermi=lambda p: K0 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p)),
                                      dk_fermi=lambda p: K0 * (0.4 * np.sin(4 * p) + 0.15 * np.cos(3 * p)),
                                      mass=M, tau=tau0)
    fields = np.array([0.0, 0.1, 1.0, 10.0]) * M / (E * TAU)
    coarse = ref.collision_sigma([contour], fields, kernel=hot_spots, layer_spacing=D, points=512)
    fine = ref.collision_sigma([contour], fields, kernel=hot_spots, layer_spacing=D, points=1024)
    assert report(f"M 512 -> 1024, tau0 = {tau0}", coarse, fine) <= 1e-10
