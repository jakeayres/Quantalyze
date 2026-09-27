"""Sanity checks for the Boltzmann solver: rerun these after any kernel change.

Fast, broad checks on both backends: the Drude circle, high-field Hall coefficients,
invariance under the input order, and Onsager and mirror symmetries.
"""
import numpy as np
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import _analytic as an
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9
K_F = 7e9
TAU = 1e-13
A = 3.87e-10
BACKENDS = ["python", "numba"]
X = np.array([0.01, 0.1, 1.0, 10.0, 100.0])


def tensor(sigma):
    return sigma[["sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]].to_numpy().reshape(-1, 2, 2)


def fourfold(n=512):
    return gen.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * M,
                     tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.3), carrier="hole")


def tight_binding(n=512):
    return gen.tight_binding(n, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2), lattice_constant=A,
                             hopping=0.25 * E, next_hopping=-0.0625 * E, third_hopping=0.02 * E,
                             chemical_potential=0.0, center=(np.pi / A, np.pi / A))


@pytest.mark.parametrize("backend", BACKENDS)
def test_drude_circle(backend):
    """The K1 circle reproduces Drude σ to 1e-4 at N = 512, with |MR| < 1e-4 and R_H = −1/(ne)."""
    n = an.circle_density(K_F, D)
    fields = np.concatenate([[0.0], X * M / (E * TAU)])
    s = bz.conductivity(gen.circle(512, k_fermi=K_F, mass=M, tau=TAU), fields, layer_spacing=D, backend=backend)
    expected = an.drude_circle(fields[1:], density=n, mass=M, tau=TAU)
    sigma_error = np.max(np.abs(tensor(s)[1:] - expected) / np.abs(expected))
    mr = np.max(np.abs(bz.magnetoresistance(s)))
    hall_error = np.max(np.abs(bz.hall_coefficient(s)[1:] * (-n * E) - 1))
    print(f"sigma {sigma_error:.1e}, |MR| {mr:.1e}, R_H {hall_error:.1e}")
    assert sigma_error < 1e-4 and mr < 1e-4 and hall_error < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("carrier, sign", [("electron", -1), ("hole", +1)])
def test_high_field_hall_coefficient(backend, carrier, sign):
    """At ω_cτ = 100, R_H = ∓1/(ne) for electron- and hole-like contours (1e-3), with n from the area."""
    for df, mass in [(gen.circle(512, k_fermi=K_F, mass=M, tau=TAU, carrier=carrier), M),
                     (gen.polar(512, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * M, tau=TAU,
                                carrier=carrier), 5 * M)]:
        s = bz.conductivity(df, 100 * mass / (E * TAU), layer_spacing=D, backend=backend)
        n = bz.carrier_density(df, layer_spacing=D)
        error = abs(bz.hall_coefficient(s).iloc[0] * sign * n * E - 1)
        print(f"{carrier}: R_H relative error {error:.1e}")
        assert error < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("make", [fourfold, tight_binding], ids=["fourfold", "tight-binding"])
def test_invariant_under_start_index_and_reversal(backend, make):
    """σ is unchanged (1e-12) by rotating the start index or reversing the input order."""
    df = make()
    fields = np.array([-30.0, 0.0, 0.3, 3.0, 30.0])
    base = tensor(bz.conductivity(df, fields, layer_spacing=D, backend=backend))
    scale = np.max(np.abs(base))
    for label, variant in [("rotated start", df.iloc[np.roll(np.arange(len(df)), 77)]), ("reversed", df.iloc[::-1])]:
        other = tensor(bz.conductivity(variant, fields, layer_spacing=D, backend=backend))
        error = np.max(np.abs(other - base)) / scale
        print(f"{label}: {error:.1e}")
        assert error <= 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_onsager_and_mirror_symmetry(backend):
    """Symmetrised, σ(−B) = σ(B)ᵀ exactly. On contours with a mirror plane along x or y,
    σ_yx = −σ_xy to 1e-12."""
    fields = np.array([0.1, 1.0, 10.0, 100.0])
    contours = {
        "circle": gen.circle(512, k_fermi=K_F, mass=M, tau=TAU),
        "ellipse": gen.ellipse(512, k_fermi=K_F, mass_x=M, mass_y=4 * M, tau=TAU),
        "fourfold": fourfold(),
        "tight-binding": tight_binding(),
    }
    for name, df in contours.items():
        s = tensor(bz.conductivity(df, np.concatenate([fields, -fields]), layer_spacing=D, backend=backend))
        assert np.array_equal(s[4:], np.transpose(s[:4], (0, 2, 1))), name
        mirror = np.max(np.abs(s[:4, 1, 0] + s[:4, 0, 1])) / np.max(np.abs(s))
        print(f"{name}: |sigma_yx + sigma_xy| / max |sigma| = {mirror:.1e}")
        assert mirror <= 1e-12
