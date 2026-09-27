"""`FermiSurface`: the original polar-form interface, now running on the exact solver (private)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ...core.constants import ELEMENTARY_CHARGE, HBAR
from ._response import conductivity_tensor


def _periodic_derivative(theta: np.ndarray, values: np.ndarray) -> np.ndarray:
    """d(values)/dθ on a periodic, increasing but possibly uneven θ grid (3-point, O(h²))."""
    before = np.mod(theta - np.roll(theta, 1), 2 * np.pi)  # h₋, (N,)
    after = np.mod(np.roll(theta, -1) - theta, 2 * np.pi)  # h₊, (N,)
    previous, following = np.roll(values, 1), np.roll(values, -1)
    return (before**2 * following - after**2 * previous + (after**2 - before**2) * values) / (
        before * after * (before + after))


class FermiSurface:
    """A single electron-like Fermi pocket in polar form, k_F(θ), with m*(θ) and τ(θ).

    The pocket is the contour k = k_F(θ)(cos θ, sin θ). Its group velocity has magnitude
    ħk_F/m* and points along the outward normal, and `calculate_conductivity` passes it to
    `bz.conductivity` with `layer_spacing = c_axis_length`.

    Changed in the exact-solver release: conductivities now come from the exact O(N)
    Chambers solver, so the numbers differ from earlier versions. Those truncated the
    history integral after one orbit and took dθ/dt = eB/m*(θ), so they were wrong for
    ω_cτ ≳ 1 (σ_xx on a circle was 33% too low at ω_cτ = 5). The integration options of
    `calculate_conductivity` no longer have any effect.

    Args:
        theta: Polar angles θ of the nodes (rad), increasing, covering one turn. A final
            point at θ₀ + 2π that repeats the first is allowed.
        fermi_wavevector: k_F at each θ (m⁻¹).
        effective_mass: m* at each θ (kg), or one value for all.
        relaxation_time: τ at each θ (s), or one value for all.
        c_axis_length: Interlayer spacing used as `layer_spacing` (m). For a body-centred
            cell this is c/2, not the lattice parameter c.

    Examples:
        >>> import numpy as np
        >>> from quantalyze.beta import boltzmann as bz
        >>> from quantalyze.core.constants import ELECTRON_MASS
        >>> theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        >>> fs = bz.FermiSurface(theta, np.full(512, 7e9), ELECTRON_MASS, 1e-13, 1e-9)
        >>> sxx, sxy, syx, syy = fs.calculate_conductivity(10.0)
        >>> rxx = sxx / (sxx * syy - sxy * syx)
    """

    def __init__(self, theta, fermi_wavevector, effective_mass, relaxation_time, c_axis_length):
        theta = np.asarray(theta, dtype=np.float64)
        self.theta = theta
        self.fermi_wavevector = np.broadcast_to(np.asarray(fermi_wavevector, dtype=np.float64), theta.shape)
        self.effective_mass = np.broadcast_to(np.asarray(effective_mass, dtype=np.float64), theta.shape)
        self.relaxation_time = np.broadcast_to(np.asarray(relaxation_time, dtype=np.float64), theta.shape)
        self.c_axis_length = c_axis_length

    def fermi_wavevector_x(self):
        """Compute the x-component of the Fermi wavevector for each theta value."""
        return self.fermi_wavevector * np.cos(self.theta)

    def fermi_wavevector_y(self):
        """Compute the y-component of the Fermi wavevector for each theta value."""
        return self.fermi_wavevector * np.sin(self.theta)

    def reciprocal_lattice_vector(self):
        """Compute the reciprocal c-axis lattice vector."""
        return 2 * np.pi / self.c_axis_length

    def cylotron_frequency(self, magnetic_field):
        """Compute the local cyclotron frequency for a given magnetic field at each theta value."""
        return ELEMENTARY_CHARGE * magnetic_field / self.effective_mass

    def omega_c_tau(self, magnetic_field):
        """Compute ω_cτ for a given magnetic field at each theta value."""
        return self.cylotron_frequency(magnetic_field) * self.relaxation_time

    def mean_free_path(self):
        """Calculate the mean free path ħk_Fτ/m* at each theta value (m)."""
        return HBAR * self.fermi_wavevector * self.relaxation_time / self.effective_mass

    def _integrate_fermi_wavevector(self):
        """Area ½∮k_F² dθ by the trapezoid rule over the given θ points."""
        n = len(self.theta)
        if n < 2:
            return 0.0

        area = 0.0
        for i in range(n):
            next_i = (i + 1) % n
            dtheta = self.theta[next_i] - self.theta[i]
            if dtheta <= 0.0:
                dtheta += 2 * np.pi
            r1 = self.fermi_wavevector[i]
            r2 = self.fermi_wavevector[next_i]
            area += 0.25 * (r1**2 + r2**2) * dtheta
        return area

    def fermi_area(self):
        """Compute the area of the Fermi surface in k-space by integrating the Fermi wavevector over theta."""
        return self._integrate_fermi_wavevector()

    def fermi_volume(self):
        """Fermi area times the reciprocal lattice vector 2π/c (m⁻³)."""
        return self._integrate_fermi_wavevector() * self.reciprocal_lattice_vector()

    def carrier_density(self):
        """Carrier density 2 × Fermi volume / (2π)³ (m⁻³)."""
        return self.fermi_volume() / (2 * np.pi)**3 * 2

    def _distinct_nodes(self):
        """θ, k_F, m*, τ without a final point that repeats the first (θ₀ + 2π)."""
        theta = self.theta
        keep = slice(None)
        if theta.size > 1 and np.isclose(np.mod(theta[-1] - theta[0], 2 * np.pi), 0.0, atol=1e-12) \
                and np.isclose(self.fermi_wavevector[-1], self.fermi_wavevector[0], rtol=1e-12):
            keep = slice(0, -1)
        return (theta[keep], self.fermi_wavevector[keep], self.effective_mass[keep], self.relaxation_time[keep])

    def calculate_normal_vectors(self):
        """Unit outward normals (n_x, n_y) to the Fermi contour at each theta value."""
        theta, k_fermi, _, _ = self._distinct_nodes()
        dk_fermi = _periodic_derivative(theta, k_fermi)  # dk_F/dθ
        # The tangent is k_F′ r̂ + k_F θ̂, so the outward normal is k_F r̂ − k_F′ θ̂.
        radial, angular = k_fermi, -dk_fermi
        norm = np.hypot(radial, angular)
        nx = (radial * np.cos(theta) - angular * np.sin(theta)) / norm
        ny = (radial * np.sin(theta) + angular * np.cos(theta)) / norm
        if theta.size < self.theta.size:  # give the repeated closing point its normal back
            nx, ny = np.append(nx, nx[0]), np.append(ny, ny[0])
        return nx, ny

    def contour(self) -> pd.DataFrame:
        """The pocket as a contour DataFrame (kx, ky, vx, vy, tau), for use with `bz.conductivity`.

        Returns:
            DataFrame with columns kx, ky (m⁻¹), vx, vy (m/s) and tau (s), one row per
            distinct node, with v = ħk_F/m* along the outward normal.
        """
        theta, k_fermi, mass, tau = self._distinct_nodes()
        nx, ny = self.calculate_normal_vectors()
        nx, ny = nx[:theta.size], ny[:theta.size]
        speed = HBAR * k_fermi / mass
        return pd.DataFrame({
            "kx": k_fermi * np.cos(theta),
            "ky": k_fermi * np.sin(theta),
            "vx": speed * nx,
            "vy": speed * ny,
            "tau": np.asarray(tau, dtype=np.float64),
        })

    def calculate_conductivity(self, magnetic_field, start_phi=None, end_phi=None, phi_points=None,
                               exponent_points=None):
        """Conductivity tensor components at one or more fields (S/m).

        Args:
            magnetic_field: Magnetic field B along ẑ (T), a float or array-like.
            start_phi: No longer used; kept so existing calls still work.
            end_phi: No longer used.
            phi_points: No longer used.
            exponent_points: No longer used.

        Returns:
            (sxx, sxy, syx, syy): floats for a float field, arrays for an array of fields.
        """
        if any(option is not None for option in (start_phi, end_phi, phi_points, exponent_points)):
            warnings.warn(
                "start_phi, end_phi, phi_points and exponent_points no longer have any effect: "
                "the conductivity is now computed exactly",
                DeprecationWarning,
                stacklevel=2,
            )
        df = self.contour()
        sigma = conductivity_tensor(df["kx"], df["ky"], df["vx"], df["vy"], df["tau"], magnetic_field,
                                    layer_spacing=self.c_axis_length)  # (nB, 2, 2)
        components = (sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 1, 0], sigma[:, 1, 1])
        if np.ndim(magnetic_field) == 0:
            return tuple(float(c[0]) for c in components)
        return components
