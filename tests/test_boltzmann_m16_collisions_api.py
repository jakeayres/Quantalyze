"""M16: the public API for scattering kernels.

`conductivity(..., scattering_kernel=...)` with every input form, and the helpers
`density_of_states`, `out_scattering_rate` and `mean_free_path` (aligned with the input
rows whatever their order) and `scattering.spin_fluctuation_kernel`.
"""
import numpy as np
import pandas as pd
import pytest

from quantalyze.beta import boltzmann as bz
from quantalyze.beta.boltzmann import generators as gen
from quantalyze.beta.boltzmann._contour import prepare_contour
from quantalyze.core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

E = ELEMENTARY_CHARGE
M = ELECTRON_MASS
D = 1e-9
TAU = 1e-13
G = 2 * np.pi / 3.87e-10


def smooth(kx, ky, kx2, ky2):
    return 2e-33 * np.exp(-((kx - kx2) ** 2 + (ky - ky2) ** 2) / (2 * 3e9**2))


def periodic(kx, ky, kx2, ky2):
    """`smooth` summed over the images k_y ± G, so it is periodic along the open sheets."""
    return sum(smooth(kx, ky, kx2, ky2 + image * G) for image in (-1, 0, 1))


def lopsided(n=128, tau=TAU):
    return gen.polar(n, k_fermi=lambda p: 7e9 * (1 - 0.1 * np.cos(4 * p) + 0.05 * np.sin(3 * p)), mass=M, tau=tau)


def test_conductivity_input_forms():
    """tau as a column or one number give the same σ; np.inf means no background; a list of
    contours with a list of periods (None for the closed pocket) works; extrapolate works."""
    df = lopsided()
    fields = [0.0, 3.0, -3.0]
    column = bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=smooth)
    number = bz.conductivity(df.drop(columns="tau"), fields, layer_spacing=D, tau=TAU, scattering_kernel=smooth)
    assert np.array_equal(column.to_numpy(), number.to_numpy())
    assert list(column.columns) == ["field", "sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"]
    none = bz.conductivity(df, fields, layer_spacing=D, tau=np.inf, scattering_kernel=smooth)
    assert np.all(np.isfinite(none.to_numpy())) and np.all(none.sigma_xx > column.sigma_xx)
    sheets = gen.open_sheets(128, k0=5e9, velocity=2e5, tau=TAU, period=G, warping=1e9)
    mixed = bz.conductivity([gen.circle(128, k_fermi=3e9, mass=M, tau=TAU), *sheets], fields, layer_spacing=D,
                            period=[None, (0.0, G), (0.0, G)], scattering_kernel=periodic)
    assert np.all(np.isfinite(mixed.to_numpy()))
    extrapolated = bz.conductivity(df, fields, layer_spacing=D, scattering_kernel=smooth, extrapolate=True)
    print(f"extrapolate changes sigma_xx(0) by {extrapolated.sigma_xx[0] / column.sigma_xx[0] - 1:.1e}")
    assert abs(extrapolated.sigma_xx[0] / column.sigma_xx[0] - 1) < 1e-2  # the O(N⁻²) error at N = 128


def test_density_of_states():
    """Circle: m/(2πħ²d) to 1e-5 at N = 512 (the O(N⁻²) chord error); flat sheets exactly
    2G/(4π²ħv₀d); k_z slices of a k_z-independent surface give the 2D value."""
    mass = 2 * M
    circle = gen.circle(512, k_fermi=7e9, mass=mass, tau=TAU)
    dos = bz.density_of_states(circle, layer_spacing=D)
    expected = mass / (2 * np.pi * HBAR**2 * D)
    print(f"circle: N(E_F) / (m/2 pi hbar^2 d) - 1 = {dos / expected - 1:.1e}")
    assert abs(dos / expected - 1) <= 1e-5
    sheets = gen.open_sheets(128, k0=5e9, velocity=2e5, tau=TAU, period=G)
    flat = bz.density_of_states(sheets, layer_spacing=D, period=(0.0, G))
    assert flat == pytest.approx(2 * G / (4 * np.pi**2 * HBAR * 2e5 * D), rel=1e-13)
    slices = pd.concat([circle.assign(kz=-np.pi / D + 2 * np.pi * j / (4 * D)) for j in range(4)], ignore_index=True)
    assert bz.density_of_states(slices, layer_spacing=D, kz="kz") == pytest.approx(dos, rel=1e-13)


@pytest.mark.parametrize("order", ["reversed", "rotated", "closing row"])
def test_helpers_are_aligned_with_the_input_rows(order):
    """out_scattering_rate and mean_free_path return values for the input rows, whatever
    order the rows came in (the solver works on the nodes ordered along the motion)."""
    df = lopsided()
    rate = bz.out_scattering_rate(df, smooth, layer_spacing=D)
    ell = bz.mean_free_path(df, layer_spacing=D, scattering_kernel=smooth)
    if order == "reversed":
        other = df.iloc[::-1]
    elif order == "rotated":
        other = df.iloc[np.roll(np.arange(len(df)), 29)]
    else:
        other = pd.concat([df, df.iloc[:1]])
    other_rate = bz.out_scattering_rate(other, smooth, layer_spacing=D)
    other_ell = bz.mean_free_path(other, layer_spacing=D, scattering_kernel=smooth)
    assert other_rate.index.equals(other.index) and other_ell.index.equals(other.index)
    rows = other.index[:len(df)]
    print(f"{order}: rate {np.max(np.abs(other_rate.iloc[:len(df)].to_numpy() / rate.loc[rows].to_numpy() - 1)):.1e}, "
          f"L {np.max(np.abs(other_ell.iloc[:len(df)].to_numpy() - ell.loc[rows].to_numpy())) / np.max(np.abs(ell.to_numpy())):.1e}")
    np.testing.assert_allclose(other_rate.iloc[:len(df)].to_numpy(), rate.loc[rows].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(other_ell.iloc[:len(df)].to_numpy(), ell.loc[rows].to_numpy(), rtol=0,
                               atol=1e-12 * np.max(np.abs(ell.to_numpy())))


def test_helpers_with_kz_slices_and_lists():
    """With k_z slices in one DataFrame, and a list of DataFrames: one result per DataFrame,
    aligned with its rows."""
    circle = gen.circle(64, k_fermi=7e9, mass=M, tau=TAU)
    slices = pd.concat([circle.assign(kz=-np.pi / D + 2 * np.pi * j / (4 * D), vz=1e3 * np.cos(j)) for j in range(4)],
                       ignore_index=True)

    def kernel3(kx, ky, kz, kx2, ky2, kz2):
        return smooth(kx, ky, kx2, ky2) * (1 + 0.5 * np.cos((kz - kz2) * D))

    rate = bz.out_scattering_rate(slices, kernel3, layer_spacing=D, kz="kz")
    assert rate.index.equals(slices.index) and np.all(rate > 0)
    ell = bz.mean_free_path(slices, layer_spacing=D, kz="kz", scattering_kernel=kernel3)
    assert list(ell.columns) == ["lx", "ly", "lz"] and ell.index.equals(slices.index)
    pair = bz.out_scattering_rate([circle, lopsided(64)], smooth, layer_spacing=D)
    assert isinstance(pair, list) and len(pair) == 2 and pair[1].index.equals(lopsided(64).index)


def test_mean_free_path_without_a_kernel_is_v_tau():
    """Without a kernel L = vτ, drift-removed on closed contours, aligned with the rows."""
    df = lopsided(tau=lambda p: TAU * (1 + 0.3 * np.cos(p))).iloc[::-1]
    ell = bz.mean_free_path(df, layer_spacing=D)
    c = prepare_contour(df.kx, df.ky, df.vx, df.vy, df.tau)
    expected = np.empty((len(df), 2))
    expected[c.index] = np.stack([c.lx, c.ly], axis=1)
    assert np.array_equal(ell.to_numpy(), expected)


def test_spin_fluctuation_kernel():
    """Symmetric in k ↔ k′, periodic in the reciprocal lattice, peaked at k − k′ = ±Q and at its
    images, with the Ornstein–Zernike shape of the chosen power; 3D with a 3-component Q."""
    a, b = 3.87e-10, 5e-10
    q = (np.pi / a, 0.4 * np.pi / b)
    xi = 4 * a
    kernel = bz.scattering.spin_fluctuation_kernel(2.0, wavevector=q, correlation_length=xi, lattice_constants=(a, b))
    rng = np.random.default_rng(0)
    k, k2 = rng.uniform(-2e10, 2e10, (2, 500)), rng.uniform(-2e10, 2e10, (2, 500))
    assert np.array_equal(kernel(*k, *k2), kernel(*k2, *k))
    np.testing.assert_allclose(kernel(k[0] + 2 * np.pi / a, k[1] - 2 * np.pi / b, *k2), kernel(*k, *k2), rtol=1e-12)
    peak = kernel(q[0], q[1], 0.0, 0.0)
    image = kernel(q[0] - 2 * np.pi / a, q[1], 0.0, 0.0)
    other = 0.5 * 2.0 / (1 + (xi * 2 * q[1]) ** 2 + (xi * 0.0) ** 2)  # the −Q term at k − k′ = +Q, reduced: δ = (0, 2Q_y)
    assert peak == pytest.approx(0.5 * 2.0 * 1 + other, rel=1e-12) and image == pytest.approx(peak, rel=1e-12)
    for power in (1, 2):
        shaped = bz.scattering.spin_fluctuation_kernel(1.0, wavevector=(0.0, 0.0), correlation_length=xi,
                                                       lattice_constants=(a, b), power=power)
        assert shaped(1 / xi, 0.0, 0.0, 0.0) == pytest.approx(0.5 ** power, rel=1e-12)
    three = bz.scattering.spin_fluctuation_kernel(1.0, wavevector=(q[0], q[1], np.pi / D), correlation_length=(xi, xi, D),
                                                  lattice_constants=(a, b, D))
    assert three(q[0], q[1], np.pi / D, 0.0, 0.0, 0.0) > 0.5
    with pytest.raises(TypeError, match="arguments"):
        three(0.0, 0.0, 0.0, 0.0)
    for bad in (dict(wavevector=(1.0,)), dict(lattice_constants=(a,)), dict(correlation_length=-1.0),
                dict(power=3)):
        options = dict(wavevector=q, correlation_length=xi, lattice_constants=(a, b)) | bad
        with pytest.raises(ValueError):
            bz.scattering.spin_fluctuation_kernel(1.0, **options)
