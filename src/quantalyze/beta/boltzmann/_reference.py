"""Brute-force reference for the Chambers conductivity, used to test the kernels.

This module is deliberately independent of the fast code: it works on a contour
given as smooth, 2π-periodic functions of a parameter φ (not on sampled nodes),
does its own geometry, and evaluates the tube integral

    σ_ij = (g_s e³ / 4π²ħ² d) · |B| · ∮ dt v_i(t) w_j(t),
    w_j(t) = ∫_{−∞}^{t} v_j(t′) exp(−∫_{t′}^{t} dt″/τ) dt′,

directly by quadrature. The history over one orbit is integrated adaptively for
every outer point, and earlier orbits are added explicitly, orbit by orbit, until
their weight e^{−KZ} falls below 1e-16. It is slow and meant for tests only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import quad, quad_vec

from ...core.constants import ELEMENTARY_CHARGE, HBAR

TWO_PI = 2 * np.pi
Pair = Tuple[np.ndarray, np.ndarray]

# Eighth-order central difference for dk/dφ when no derivative is supplied: truncation
# error ~h⁸ is negligible, and rounding error is ~1e-13 relative for h = 1e-3.
_FD_STEP = 1e-3
_FD_WEIGHTS = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
_FD_OFFSETS = np.arange(-4, 5)
# The cumulative scattering integral is tabulated on this many cells per orbit; inside a
# cell it is completed with 16-point Gauss–Legendre, exact to rounding for smooth rates.
_CELLS = 256
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
# Weight of the oldest orbit kept in the history sum.
_ORBIT_CUTOFF = 1e-16


def _tau_of_polar_angle(tau, k):
    """τ as a function of the parameter, given τ (float or function of the polar angle)."""
    if callable(tau):
        return lambda phi: np.asarray(tau(np.arctan2(k(phi)[1], k(phi)[0])), dtype=float)
    return lambda phi: np.full(np.shape(phi), float(tau))


@dataclass(frozen=True)
class ParametricContour:
    """A closed Fermi contour as smooth 2π-periodic functions of a parameter φ.

    Attributes:
        k: φ ↦ (k_x, k_y) (m⁻¹), vectorised over arrays of φ.
        v: φ ↦ (v_x, v_y), the group velocity ∇ε/ħ (m/s).
        tau: φ ↦ τ (s).
        dk: Optional φ ↦ (dk_x/dφ, dk_y/dφ) (m⁻¹). Finite differences are used if omitted.
        vz: Optional φ ↦ v_z (m/s) for one k_z slice of a warped surface; σ is then 3×3.
    """

    k: Callable[[np.ndarray], Pair]
    v: Callable[[np.ndarray], Pair]
    tau: Callable[[np.ndarray], np.ndarray]
    dk: Optional[Callable[[np.ndarray], Pair]] = None
    vz: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @classmethod
    def circle(cls, *, k_fermi, mass, tau, carrier="electron"):
        """Circle of radius k_F with v = ±ħk/m; τ is a float or a function of the polar angle."""
        sign = {"electron": 1.0, "hole": -1.0}[carrier]
        k = lambda p: (k_fermi * np.cos(p), k_fermi * np.sin(p))  # noqa: E731
        v = lambda p: (sign * HBAR * k_fermi / mass * np.cos(p), sign * HBAR * k_fermi / mass * np.sin(p))  # noqa: E731
        dk = lambda p: (-k_fermi * np.sin(p), k_fermi * np.cos(p))  # noqa: E731
        return cls(k=k, v=v, tau=_tau_of_polar_angle(tau, k), dk=dk)

    @classmethod
    def ellipse(cls, *, k_fermi, mass_x, mass_y, tau, rotation=0.0, carrier="electron"):
        """Ellipse of area πk_F² for ε = ħ²k_x²/2m_x + ħ²k_y²/2m_y, rotated by `rotation`."""
        sign = {"electron": 1.0, "hole": -1.0}[carrier]
        a, b = k_fermi * (mass_x / mass_y) ** 0.25, k_fermi * (mass_y / mass_x) ** 0.25
        c, s = np.cos(rotation), np.sin(rotation)

        def rotate(x, y):
            return c * x - s * y, s * x + c * y

        k = lambda p: rotate(a * np.cos(p), b * np.sin(p))  # noqa: E731
        v = lambda p: rotate(sign * HBAR * a * np.cos(p) / mass_x, sign * HBAR * b * np.sin(p) / mass_y)  # noqa: E731
        dk = lambda p: rotate(-a * np.sin(p), b * np.cos(p))  # noqa: E731
        return cls(k=k, v=v, tau=_tau_of_polar_angle(tau, k), dk=dk)

    @classmethod
    def polar(cls, *, k_fermi, dk_fermi, mass, tau, carrier="electron"):
        """Star-shaped pocket k_F(φ) of ε = ±ħ²(k² − k_F(φ)²)/2m, parametrised by the polar angle."""
        sign = {"electron": 1.0, "hole": -1.0}[carrier]

        def k(p):
            r = k_fermi(p)
            return r * np.cos(p), r * np.sin(p)

        def dk(p):
            r, dr = k_fermi(p), dk_fermi(p)
            return dr * np.cos(p) - r * np.sin(p), dr * np.sin(p) + r * np.cos(p)

        def v(p):
            radial, angular = sign * HBAR * k_fermi(p) / mass, -sign * HBAR * dk_fermi(p) / mass
            return radial * np.cos(p) - angular * np.sin(p), radial * np.sin(p) + angular * np.cos(p)

        return cls(k=k, v=v, tau=_tau_of_polar_angle(tau, k), dk=dk)

    @classmethod
    def open_sheet(cls, *, k0, velocity, warping, period, tau, side=1.0):
        """One period of the open sheet k_x = s(k0 + δ cos(2πk_y/G)) of ε = ħv₀(s k_x − k0 − δ cos(2πk_y/G)).

        The parameter φ ∈ [0, 2π) maps to k_y = G(φ/2π − ½), so k(φ + 2π) = k(φ) + G ŷ while
        dk/dφ, v and τ (a float, or a function of φ) are 2π-periodic, which is all the
        quadrature uses.
        """
        b = 2 * np.pi / period

        def k(p):
            ky = period * (p / TWO_PI - 0.5)
            return side * (k0 + warping * np.cos(b * ky)), ky

        def dk(p):
            ky = period * (p / TWO_PI - 0.5)
            dky = period / TWO_PI
            return -side * warping * b * np.sin(b * ky) * dky, np.full(np.shape(p), dky)

        def v(p):
            ky = period * (p / TWO_PI - 0.5)
            return np.full(np.shape(p), side * velocity), velocity * warping * b * np.sin(b * ky)

        tau_fn = tau if callable(tau) else (lambda p: np.full(np.shape(p), float(tau)))
        return cls(k=k, v=v, tau=tau_fn, dk=dk)

    @classmethod
    def warped_band_sheet(cls, *, velocity, k0, lattice_constant, hoppings, side, tau, interlayer_hopping=None,
                          kz=0.0, layer_spacing=None):
        """One period of sheet s = ±1 of ε = ħv₀(|k_y| − k₀) − Σₙ 2tₙcos(nk_x a) [− 2t_z cos(k_z d)].

        The parameter maps to k_x = (2π/a)(φ/2π − ½), and k_y = s[k₀ + (Σₙ 2tₙcos(nk_x a)
        + 2t_z cos(k_z d))/(ħv₀)], with v = ∇ε/ħ. With `interlayer_hopping` t_z the contour
        is the slice at `kz` of a warped surface and carries v_z = 2t_z d sin(k_z d)/ħ.
        `hoppings` is {n: tₙ (J)}; `tau` is a float (may be inf) or a function of φ.
        """
        a = lattice_constant
        shift = 0.0 if interlayer_hopping is None else 2 * interlayer_hopping * np.cos(kz * layer_spacing)

        def kx_of(p):
            return np.asarray(p, dtype=float) / a - np.pi / a

        def k(p):
            kx = kx_of(p)
            warp = sum(2 * t * np.cos(n * kx * a) for n, t in hoppings.items())
            return kx, side * (k0 + (warp + shift) / (HBAR * velocity))

        def dk(p):
            kx = kx_of(p)
            slope = -sum(2 * t * n * a * np.sin(n * kx * a) for n, t in hoppings.items()) / (HBAR * velocity)
            return np.full(kx.shape, 1 / a), side * slope / a

        def v(p):
            kx = kx_of(p)
            return (sum(2 * t * n * a * np.sin(n * kx * a) for n, t in hoppings.items()) / HBAR,
                    np.full(kx.shape, side * velocity))

        vz = None
        if interlayer_hopping is not None:
            speed_z = 2 * interlayer_hopping * layer_spacing * np.sin(kz * layer_spacing) / HBAR
            vz = lambda p: np.full(np.shape(p), speed_z)  # noqa: E731
        tau_fn = tau if callable(tau) else (lambda p: np.full(np.shape(p), float(tau)))
        return cls(k=k, v=v, tau=tau_fn, dk=dk, vz=vz)


def _parameter_derivative(contour: ParametricContour, phi) -> Pair:
    """dk/dφ (m⁻¹): the contour's own derivative, or eighth-order central differences."""
    phi = np.asarray(phi, dtype=float)
    if contour.dk is not None:
        dkx, dky = contour.dk(phi)
        return np.broadcast_to(dkx, phi.shape).astype(float), np.broadcast_to(dky, phi.shape).astype(float)
    stencil = phi[..., None] + _FD_OFFSETS * _FD_STEP  # (..., 9)
    kx, ky = contour.k(stencil)
    return kx @ _FD_WEIGHTS / _FD_STEP, ky @ _FD_WEIGHTS / _FD_STEP


class _Geometry:
    """Per-contour quantities that do not depend on the field."""

    def __init__(self, contour: ParametricContour):
        self.contour = contour
        rate = lambda p: float(self.rate(np.array([p]))[0])  # noqa: E731
        edges = np.linspace(0.0, TWO_PI, _CELLS + 1)
        pieces = [quad(rate, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0] for lo, hi in zip(edges[:-1], edges[1:])]
        # Cumulative ∫₀^φ (ds/dφ)/τ dφ at the cell edges, summed without rounding drift.
        self.edges_value = np.array([math.fsum(pieces[:m]) for m in range(_CELLS + 1)])
        self.total = self.edges_value[-1]  # |B|·Z, the damping per orbit times |B| (T)

        phi = np.linspace(0.0, TWO_PI, 64, endpoint=False)
        dkx, dky = self.dk(phi)
        vx, vy = self._v(phi)
        crossing = dkx * vy - dky * vx  # k′(φ)·(v_y, −v_x)
        if not (np.all(crossing > 0) or np.all(crossing < 0)):
            raise ValueError("v must be normal to the contour and point to the same side all the way round")
        self.orientation = 1.0 if crossing[0] > 0 else -1.0

    def _v(self, phi):
        vx, vy = self.contour.v(phi)
        return np.broadcast_to(vx, np.shape(phi)).astype(float), np.broadcast_to(vy, np.shape(phi)).astype(float)

    def dk(self, phi) -> Pair:
        return _parameter_derivative(self.contour, phi)

    def velocity(self, phi) -> Pair:
        return self._v(np.asarray(phi, dtype=float))

    def velocities(self, phi) -> tuple:
        """(v_x, v_y), or (v_x, v_y, v_z) for a slice of a warped surface."""
        phi = np.asarray(phi, dtype=float)
        vx, vy = self._v(phi)
        if self.contour.vz is None:
            return vx, vy
        return vx, vy, np.broadcast_to(self.contour.vz(phi), phi.shape).astype(float)

    def ds(self, phi) -> np.ndarray:
        """Geometric time per unit parameter, ds/dφ = ħ|dk/dφ| / (e|v|) (s·T)."""
        dkx, dky = self.dk(phi)
        vx, vy = self.velocity(phi)
        return HBAR * np.hypot(dkx, dky) / (ELEMENTARY_CHARGE * np.hypot(vx, vy))

    def rate(self, phi) -> np.ndarray:
        """(ds/dφ)/τ: the damping exponent per unit parameter, times |B| (T)."""
        phi = np.asarray(phi, dtype=float)
        tau = np.broadcast_to(self.contour.tau(phi), phi.shape)
        return self.ds(phi) / tau

    def cumulative(self, phi) -> np.ndarray:
        """∫₀^φ (ds/dφ′)/τ dφ′ for any real φ (T), continued across orbits."""
        phi = np.asarray(phi, dtype=float)
        turns = np.floor(phi / TWO_PI)
        r = phi - turns * TWO_PI
        width = TWO_PI / _CELLS
        cell = np.minimum((r / width).astype(int), _CELLS - 1)
        start = cell * width
        half, mid = 0.5 * (r - start), 0.5 * (r + start)
        nodes = mid[..., None] + half[..., None] * _GL_NODES  # (..., 16)
        partial = half * (self.rate(nodes) @ _GL_WEIGHTS)
        return turns * self.total + self.edges_value[cell] + partial

    def damping(self, phi, u, direction) -> np.ndarray:
        """∫ (ds/dφ′)/τ dφ′ over the u of parameter a carrier covered before reaching φ (T).

        Short spans are integrated directly, so the result is accurate relative to itself
        even when it is tiny (very low field); longer spans use the tabulated cumulative
        integral, whose absolute rounding error is then negligible next to the result.
        """
        phi = np.asarray(phi, dtype=float)
        if u > TWO_PI / _CELLS:
            lo, hi = (phi - u, phi) if direction > 0 else (phi, phi + u)
            return self.cumulative(hi) - self.cumulative(lo)
        # The span is exactly u: taking it from the rounded endpoints instead would lose
        # the digits of u that φ ± u cannot hold.
        half = 0.5 * u
        nodes = (phi - direction * half)[..., None] + half * _GL_NODES  # (..., 16)
        return half * (self.rate(nodes) @ _GL_WEIGHTS)


def orbit_count(contour: ParametricContour, field: float) -> int:
    """Number of orbits K the history sum keeps at this field: the first K with e^{−KZ} < 1e-16."""
    z = _Geometry(contour).total / abs(field)
    return int(math.ceil(-math.log(_ORBIT_CUTOFF) / z))


def sigma(
    contour: ParametricContour,
    fields,
    *,
    layer_spacing: float,
    charge: float = -ELEMENTARY_CHARGE,
    spin_degeneracy: float = 2,
    orbits: Optional[int] = None,
    rtol: float = 1e-11,
    max_nodes: int = 4096,
) -> np.ndarray:
    """Conductivity tensor by brute-force quadrature (S/m).

    For each field, the history w_j over one orbit is integrated adaptively (scipy
    `quad_vec`) at every outer point, and the contribution of earlier orbits is added
    explicitly as Σ_{K} e^{−KZ} times that one-orbit history. The outer ∮ dt uses the
    periodic trapezoid rule, which converges exponentially for a smooth closed orbit;
    the number of outer points is doubled until σ changes by less than `rtol`.

    Args:
        contour: The closed contour as smooth functions of a parameter.
        fields: Magnetic field B along ẑ (T), float or array of shape (nB,). B = 0 uses
            σ_ij(0) = (g_s e²/4π²ħd) ∮ |dk| τ v_i v_j / |v|.
        layer_spacing: Interlayer spacing d (m).
        charge: Carrier charge q (C).
        spin_degeneracy: Spin degeneracy g_s.
        orbits: Number of orbits of history to sum. By default, enough that the oldest
            orbit is weighted by less than 1e-16.
        rtol: Relative convergence target for the outer integral (and 1e-2 of it for the
            inner adaptive integrals).
        max_nodes: Largest number of outer points tried before giving up.

    Returns:
        ndarray of shape (nB, 2, 2), or (nB, 3, 3) if the contour has v_z: σ with index
        order [[xx, xy, …], [yx, yy, …], …]. The ds of the orbit uses the in-plane speed.

    Raises:
        RuntimeError: If the outer integral has not converged by `max_nodes` points.
    """
    geometry = _Geometry(contour)
    fields = np.atleast_1d(np.asarray(fields, dtype=float))  # (nB,)
    dim = 2 if contour.vz is None else 3
    result = np.empty((fields.size, dim, dim))
    for index, field in enumerate(fields):
        if field == 0:
            result[index] = _sigma_zero_field(geometry, layer_spacing, spin_degeneracy, rtol)
        else:
            result[index] = _sigma_at_field(geometry, field, layer_spacing, charge, spin_degeneracy,
                                            orbits, rtol, max_nodes)
    return result


def _sigma_zero_field(geometry, layer_spacing, spin_degeneracy, rtol):
    def integrand(phi):
        p = np.array([phi])
        dkx, dky = geometry.dk(p)
        v = np.array(geometry.velocities(p))[:, 0]  # (d,)
        weight = (np.hypot(dkx, dky) * geometry.contour.tau(p) / np.hypot(v[0], v[1]))[0]
        return weight * np.outer(v, v)

    integral, _ = quad_vec(integrand, 0.0, TWO_PI, epsabs=0.0, epsrel=1e-2 * rtol, limit=2000)
    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**2 / (4 * np.pi**2 * HBAR * layer_spacing)
    return prefactor * integral


def _sigma_at_field(geometry, field, layer_spacing, charge, spin_degeneracy, orbits, rtol, max_nodes):
    b = abs(field)
    # Carriers move along ħ dk/dt = q v × B: forwards in φ when this is +1.
    direction = np.sign(charge) * np.sign(field) * geometry.orientation
    z = geometry.total / b
    if orbits is None:
        orbits = int(math.ceil(-math.log(_ORBIT_CUTOFF) / z))
    history_weight = math.fsum(math.exp(-n * z) for n in range(orbits))  # Σ_{K<orbits} e^{−KZ}

    # The one-orbit history decays over Δφ ≈ |B| / max rate; put breakpoints there so the
    # adaptive integral resolves the decay even when it is much narrower than an orbit.
    probe = np.linspace(0.0, TWO_PI, 512, endpoint=False)
    scale = b / np.max(geometry.rate(probe))
    breakpoints = [u for u in scale * 4.0 ** np.arange(-2, 40) if u < TWO_PI]

    prefactor = spin_degeneracy * ELEMENTARY_CHARGE**3 / (4 * np.pi**2 * HBAR**2 * layer_spacing)
    previous = None
    n_nodes = 32
    while n_nodes <= max_nodes:
        phi = TWO_PI * np.arange(n_nodes) / n_nodes  # (M,)

        def history(u, phi=phi):
            earlier = phi - direction * u  # the point the carrier left u earlier in φ
            damping = np.exp(-geometry.damping(phi, u, direction) / b)
            weight = geometry.ds(earlier) * damping
            return np.concatenate([v * weight for v in geometry.velocities(earlier)])  # (dM,)

        integral, _ = quad_vec(history, 0.0, TWO_PI, epsabs=0.0, epsrel=1e-2 * rtol,
                               points=breakpoints or None, limit=10000)
        # w_j = (1/|B|) ∫ v_j ds′ e^{−ΔD} · Σ_K e^{−KZ}, with dt′ = ds′/|B|
        v = np.array(geometry.velocities(phi))  # (d, M)
        w = integral.reshape(v.shape[0], n_nodes) * history_weight / b  # (d, M)
        # |B| ∮ dt = ∮ ds, by the periodic trapezoid rule
        weight = TWO_PI / n_nodes * geometry.ds(phi)
        current = prefactor * np.einsum("m,im,jm->ij", weight, v, w)
        if previous is not None and np.max(np.abs(current - previous)) <= rtol * np.max(np.abs(current)):
            return current
        previous = current
        n_nodes *= 2
    raise RuntimeError(f"the reference σ did not converge to rtol = {rtol} with {max_nodes} outer points")


# ---------------------------------------------------------------------------
# Scattering kernels: a spectral reference for the full linearised Boltzmann equation
#
# With a kernel P(k, k′) the vector mean free path L obeys, along each orbit,
#     dL/dt + L/τ = v + ∫ dμ(k′) P(k, k′) L(k′),     1/τ = 1/τ₀ + ∫ dμ(k′) P(k, k′),
# and σ_ij = g_s e² ∫ dμ v_i L_j. Here dμ = |dk| / (4π²ħ|v| d) is the density of states per
# spin per volume on a contour, times the contour's weight (1/N_z for one of N_z k_z
# slices). Each contour is sampled at M equally spaced parameter values; d/dt = φ̇ d/dφ
# uses the Fourier differentiation matrix, and ∫dμ′ the trapezoid rule, both spectrally
# accurate for smooth periodic data. The rates are the row sums of the same quadrature,
# so the kernel conserves particles exactly. Nothing here is shared with the fast code.
# ---------------------------------------------------------------------------

def _fourier_derivative_matrix(points: int) -> np.ndarray:
    """d/dφ on φ_j = 2πj/M for even M: D_ij = ½(−1)^{i−j} cot((φ_i − φ_j)/2), D_ii = 0."""
    if points % 2:
        raise ValueError(f"points must be even, not {points}")
    offset = np.arange(points)[:, None] - np.arange(points)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = 0.5 * (-1.0) ** offset / np.tan(offset * np.pi / points)
    matrix[np.diag_indices(points)] = 0.0
    return matrix


def _collision_nodes(contours, kernel, *, layer_spacing, points, weights, kz, charge):
    """Nodes, velocities, density-of-states weights, motion per tesla, τ₀ and the kernel."""
    weights = [1.0] * len(contours) if weights is None else list(weights)
    phi = TWO_PI * np.arange(points) / points
    parts = {name: [] for name in ("kx", "ky", "kz", "v", "weight", "motion", "tau0")}
    for index, contour in enumerate(contours):
        kx, ky = (np.broadcast_to(a, phi.shape).astype(float) for a in contour.k(phi))
        dkx, dky = _parameter_derivative(contour, phi)
        vx, vy = (np.broadcast_to(a, phi.shape).astype(float) for a in contour.v(phi))
        velocity = [vx, vy]
        if contour.vz is not None:
            velocity.append(np.broadcast_to(contour.vz(phi), phi.shape).astype(float))
        speed, chord = np.hypot(vx, vy), np.hypot(dkx, dky)
        parts["kx"].append(kx)
        parts["ky"].append(ky)
        parts["kz"].append(np.full(points, np.nan if kz is None else float(kz[index])))
        parts["v"].append(np.stack(velocity, axis=1))  # (M, d)
        parts["weight"].append(weights[index] * (TWO_PI / points) * chord
                               / (4 * np.pi**2 * HBAR * speed * layer_spacing))
        # ħ dk/dt = q v × B = qB (v_y, −v_x): φ̇ per tesla, signed.
        parts["motion"].append(charge / HBAR * (vy * dkx - vx * dky) / chord**2)
        parts["tau0"].append(np.broadcast_to(contour.tau(phi), phi.shape).astype(float))
    nodes = {name: np.concatenate(values) for name, values in parts.items()}
    first = (nodes["kx"][:, None], nodes["ky"][:, None])
    second = (nodes["kx"][None, :], nodes["ky"][None, :])
    if kz is not None:
        first, second = first + (nodes["kz"][:, None],), second + (nodes["kz"][None, :],)
    size = nodes["kx"].size
    nodes["kernel"] = np.array(np.broadcast_to(np.asarray(kernel(*first, *second), dtype=float), (size, size)))
    nodes["rate"] = 1.0 / nodes["tau0"] + nodes["kernel"] @ nodes["weight"]
    return nodes


def _collision_solve(nodes, field, derivative, points):
    """L at every node for the sources v_j, shape (M_total, d)."""
    operator = np.diag(nodes["rate"]) - nodes["kernel"] * nodes["weight"][None, :]
    for start in range(0, nodes["rate"].size, points):
        block = slice(start, start + points)
        operator[block, block] += (nodes["motion"][block] * field)[:, None] * derivative
    if np.any(np.isinf(nodes["tau0"])):
        # Without background anywhere on a connected set the charge mode is a zero mode;
        # σ does not depend on it, and least squares picks one solution.
        return np.linalg.lstsq(operator, nodes["v"], rcond=None)[0]
    return np.linalg.solve(operator, nodes["v"])


def collision_sigma(contours, fields, *, kernel, layer_spacing, points=512, weights=None, kz=None,
                    charge=-ELEMENTARY_CHARGE, spin_degeneracy=2) -> np.ndarray:
    """Conductivity with a scattering kernel by spectral collocation (S/m). Tests only.

    Args:
        contours: `ParametricContour`s, each 2π-periodic in its parameter (closed pockets or
            one period of an open sheet); their `tau` is the background τ₀ and may be inf.
        fields: Magnetic field B along ẑ (T), any sign.
        kernel: P(kx, ky, kx2, ky2), or P(kx, ky, kz, kx2, ky2, kz2) with `kz` (J·m³/s).
        layer_spacing: Interlayer spacing d (m).
        points: Parameter values per contour (even).
        weights: Weight of each contour (1/N_z for k_z slices); 1 by default.
        kz: k_z of each contour (m⁻¹) for slices of a warped surface; σ is then 3×3.
        charge: Carrier charge q (C).
        spin_degeneracy: Spin degeneracy g_s.

    Returns:
        ndarray of shape (nB, d, d).
    """
    nodes = _collision_nodes(contours, kernel, layer_spacing=layer_spacing, points=points, weights=weights,
                             kz=kz, charge=charge)
    derivative = _fourier_derivative_matrix(points)
    source = nodes["v"] * nodes["weight"][:, None]  # (M_total, d)
    fields = np.atleast_1d(np.asarray(fields, dtype=float))
    return np.array([spin_degeneracy * ELEMENTARY_CHARGE**2 * source.T @ _collision_solve(nodes, b, derivative, points)
                     for b in fields])


def collision_mean_free_path(contours, *, kernel, layer_spacing, points=512, weights=None, kz=None,
                             charge=-ELEMENTARY_CHARGE):
    """The vector mean free path L at B = 0 on each contour, by spectral collocation. Tests only.

    Returns:
        (L, directions): per contour, L of shape (d, M) at φ_j = 2πj/M, and the direction
        of motion for B > 0 (+1 along increasing φ). With no background scattering on a
        connected set of contours, L is defined up to a constant there.
    """
    nodes = _collision_nodes(contours, kernel, layer_spacing=layer_spacing, points=points, weights=weights,
                             kz=kz, charge=charge)
    ell = _collision_solve(nodes, 0.0, _fourier_derivative_matrix(points), points)
    starts = range(0, ell.shape[0], points)
    return [ell[s:s + points].T for s in starts], [float(np.sign(nodes["motion"][s])) for s in starts]
