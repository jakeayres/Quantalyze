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
    """

    k: Callable[[np.ndarray], Pair]
    v: Callable[[np.ndarray], Pair]
    tau: Callable[[np.ndarray], np.ndarray]
    dk: Optional[Callable[[np.ndarray], Pair]] = None

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
        phi = np.asarray(phi, dtype=float)
        if self.contour.dk is not None:
            dkx, dky = self.contour.dk(phi)
            return np.broadcast_to(dkx, phi.shape).astype(float), np.broadcast_to(dky, phi.shape).astype(float)
        stencil = phi[..., None] + _FD_OFFSETS * _FD_STEP  # (..., 9)
        kx, ky = self.contour.k(stencil)
        return kx @ _FD_WEIGHTS / _FD_STEP, ky @ _FD_WEIGHTS / _FD_STEP

    def velocity(self, phi) -> Pair:
        return self._v(np.asarray(phi, dtype=float))

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
        ndarray of shape (nB, 2, 2), σ with index order [[xx, xy], [yx, yy]].

    Raises:
        RuntimeError: If the outer integral has not converged by `max_nodes` points.
    """
    geometry = _Geometry(contour)
    fields = np.atleast_1d(np.asarray(fields, dtype=float))  # (nB,)
    result = np.empty((fields.size, 2, 2))
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
        vx, vy = geometry.velocity(p)
        weight = np.hypot(dkx, dky) * geometry.contour.tau(p) / np.hypot(vx, vy)
        return (weight * np.array([vx * vx, vx * vy, vy * vx, vy * vy])[:, 0]).reshape(2, 2)

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
            vx, vy = geometry.velocity(earlier)
            return np.concatenate([vx * weight, vy * weight])  # (2M,)

        integral, _ = quad_vec(history, 0.0, TWO_PI, epsabs=0.0, epsrel=1e-2 * rtol,
                               points=breakpoints or None, limit=10000)
        # w_j = (1/|B|) ∫ v_j ds′ e^{−ΔD} · Σ_K e^{−KZ}, with dt′ = ds′/|B|
        wx, wy = integral.reshape(2, n_nodes) * history_weight / b
        vx, vy = geometry.velocity(phi)
        ds = geometry.ds(phi)
        # |B| ∮ dt = ∮ ds, by the periodic trapezoid rule
        weight = TWO_PI / n_nodes * ds
        current = prefactor * np.array([
            [np.sum(weight * vx * wx), np.sum(weight * vx * wy)],
            [np.sum(weight * vy * wx), np.sum(weight * vy * wy)],
        ])
        if previous is not None and np.max(np.abs(current - previous)) <= rtol * np.max(np.abs(current)):
            return current
        previous = current
        n_nodes *= 2
    raise RuntimeError(f"the reference σ did not converge to rtol = {rtol} with {max_nodes} outer points")
