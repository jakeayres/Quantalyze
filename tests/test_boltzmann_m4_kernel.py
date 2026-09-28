"""M4: the numba kernel.

It must reproduce the NumPy kernel (1e-12) on every test contour and field, give
bitwise-identical results for any thread count, and be compiled without fastmath.
"""
import re

import numba
import numpy as np
import pytest

from quantalyze.beta.boltzmann import _kernel, _kernel_py
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann import scattering as sc
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
TAU = 1e-13
A = 3.87e-10
C2, C4 = HBAR**2 / (2 * ELECTRON_MASS), 2e-59

CONTOURS = {
    "circle": (gen.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=TAU), ELECTRON_MASS),
    "circle-hole": (gen.circle(512, k_fermi=7e9, mass=ELECTRON_MASS, tau=TAU, carrier="hole"), ELECTRON_MASS),
    "ellipse": (gen.ellipse(512, k_fermi=7e9, mass_x=ELECTRON_MASS, mass_y=4 * ELECTRON_MASS, tau=TAU,
                            rotation=0.4), 2 * ELECTRON_MASS),
    "fourfold": (gen.polar(512, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * ELECTRON_MASS,
                           tau=lambda p: sc.cos4phi(p, TAU, anisotropy=0.3), carrier="hole"), 5 * ELECTRON_MASS),
    "lopsided": (gen.polar(300, k_fermi=lambda p: 7e9 * (1 + 0.1 * np.cos(2 * p) + 0.05 * np.sin(3 * p)),
                           mass=2 * ELECTRON_MASS, tau=lambda p: sc.cos4phi(p + 0.3, TAU, anisotropy=0.5)),
                 2 * ELECTRON_MASS),
    "quartic": (gen.from_dispersion(256, tau=TAU, max_radius=2e10,
                                    energy=lambda kx, ky: C2 * (kx**2 + ky**2) + C4 * (kx**4 + ky**4) - 1.9e-19,
                                    gradient=lambda kx, ky: (2 * C2 * kx + 4 * C4 * kx**3,
                                                             2 * C2 * ky + 4 * C4 * ky**3)), ELECTRON_MASS),
    "tight-binding": (gen.tight_binding(512, tau=lambda p: sc.hot_spot(p, TAU, strength=4.0, width=0.2),
                                        lattice_constant=A, hopping=0.25 * E, next_hopping=-0.0625 * E,
                                        third_hopping=0.02 * E, chemical_potential=0.0,
                                        center=(np.pi / A, np.pi / A)), ELECTRON_MASS),
}
X = np.geomspace(1e-6, 1e6, 49)  # ω_cτ


def arrays(df):
    return df["kx"], df["ky"], df["vx"], df["vy"], df["tau"]


def per_field_error(a, b):
    """Normwise relative difference for each field, shape (nB,)."""
    return np.max(np.abs(a - b), axis=(1, 2)) / np.max(np.abs(b), axis=(1, 2))


@pytest.mark.parametrize("name", CONTOURS)
def test_matches_the_numpy_kernel(name):
    """Raw orbit sums on both orientations over twelve decades of ω_cτ."""
    df, mass = CONTOURS[name]
    fields = X * mass / (E * TAU)
    c = prepare_contour(*arrays(df))
    order = np.roll(np.arange(c.damping.size)[::-1], 1)
    worst = 0.0
    for damping, lx, ly in [(c.damping, c.lx, c.ly), (c.damping[::-1], c.lx[order], c.ly[order])]:
        expected = _kernel_py.orbit_sums(damping, lx, ly, fields)
        actual = _kernel.orbit_sums(*(np.ascontiguousarray(a) for a in (damping, lx, ly)), fields)
        worst = max(worst, per_field_error(actual, expected).max())
    print(f"{name}: max per-field relative difference from NumPy over x = 1e-6..1e6: {worst:.1e}")
    assert worst <= 1e-12


@pytest.mark.parametrize("name", CONTOURS)
def test_both_orientations_kernel_matches_the_numpy_kernel(name):
    """orbit_sums_both shares each segment's weights between the orbit and its reverse;
    each half must still match the NumPy kernel run on that orientation."""
    df, mass = CONTOURS[name]
    fields = X * mass / (E * TAU)
    c = prepare_contour(*arrays(df))
    order = np.roll(np.arange(c.damping.size)[::-1], 1)
    both = _kernel.orbit_sums_both(c.damping, c.lx, c.ly, fields)
    forward = _kernel_py.orbit_sums(c.damping, c.lx, c.ly, fields)
    backward = _kernel_py.orbit_sums(c.damping[::-1], c.lx[order], c.ly[order], fields)
    worst = max(per_field_error(both[0], forward).max(), per_field_error(both[1], backward).max())
    print(f"{name}: max per-field relative difference from NumPy, both orientations: {worst:.1e}")
    assert worst <= 1e-12
    np.testing.assert_array_equal(both[0], _kernel.orbit_sums(c.damping, c.lx, c.ly, fields))


@pytest.mark.parametrize("symmetrize", [True, False])
def test_conductivity_tensor_backends_agree(symmetrize):
    """Through the full array layer: ±B, B = 0, symmetrised or not, every contour."""
    worst = 0.0
    for name, (df, mass) in CONTOURS.items():
        x = np.array([0.0, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4])
        fields = np.concatenate([x, -x[1:]]) * mass / (E * TAU)
        python = conductivity_tensor(*arrays(df), fields, layer_spacing=1e-9, symmetrize=symmetrize, backend="python")
        compiled = conductivity_tensor(*arrays(df), fields, layer_spacing=1e-9, symmetrize=symmetrize)
        worst = max(worst, per_field_error(compiled, python).max())
    print(f"symmetrize={symmetrize}: max per-field relative difference = {worst:.1e}")
    assert worst <= 1e-12


def test_bitwise_identical_for_any_thread_count():
    df, mass = CONTOURS["tight-binding"]
    c = prepare_contour(*arrays(df))
    fields = X * mass / (E * TAU)
    original = numba.get_num_threads()
    counts = sorted({1, 2, 3, numba.config.NUMBA_NUM_THREADS})
    try:
        results = {}
        for count in counts:
            numba.set_num_threads(count)
            results[count] = _kernel.orbit_sums(c.damping, c.lx, c.ly, fields)
    finally:
        numba.set_num_threads(original)
    print(f"thread counts compared: {counts}")
    for count in counts[1:]:
        assert np.array_equal(results[count], results[counts[0]]), f"{count} threads differ from 1"


def _fresh_ir():
    """LLVM IR of a fresh compile of the kernel with its own options. (A dispatcher loaded
    from numba's on-disk cache cannot be inspected: it returns invalid IR with a warning.)"""
    kernel = _kernel._orbit_sums_blocks
    options = {k: v for k, v in kernel.targetoptions.items() if k not in ("cache", "nopython")}
    fresh = numba.njit(**options)(kernel.py_func)
    fresh(np.full(64, 1e-2), np.ones(64), np.ones(64), np.array([1.0, 2.0]), 2, True)
    return "\n".join(fresh.inspect_llvm().values())


def test_compiled_without_fastmath():
    """No reassociation or other fast-math flags anywhere in the compiled kernel (the
    moments included)."""
    for dispatcher in (_kernel._orbit_sums_blocks, _kernel._segment_weights, _kernel._sweep, _kernel.moments):
        assert not dispatcher.targetoptions.get("fastmath", False)
    ir = _fresh_ir()
    assert "expm1" in ir and "exp" in ir  # the moments and the closure are compiled into this IR too
    flags = [flag for flag in (" fast ", " reassoc ", " contract ", " nnan ", " ninf ", " nsz ", " afn ", " arcp ")
             if flag in ir]
    print(f"LLVM IR size {len(ir)} characters; fast-math flags found: {flags}")
    assert not flags


def test_no_allocation_inside_the_field_loop():
    """The parallel loop bodies (numba's parfor gufuncs) never allocate; the only
    allocation is the result array, made once before the loop."""
    ir = _fresh_ir()
    bodies = [body for body in re.split(r"\ndefine ", ir) if "__numba_parfor_gufunc" in body.split("{", 1)[0]]
    allocations = [len(re.findall(r"NRT_MemInfo_alloc", body)) for body in bodies]
    print(f"parallel loop bodies: {len(bodies)}, allocation calls in each: {allocations}")
    assert bodies and not any(allocations)


def test_default_backend_is_numba():
    df, mass = CONTOURS["fourfold"]
    fields = np.array([0.0, 1.0, 10.0])
    np.testing.assert_array_equal(conductivity_tensor(*arrays(df), fields, layer_spacing=1e-9),
                                  conductivity_tensor(*arrays(df), fields, layer_spacing=1e-9, backend="numba"))
