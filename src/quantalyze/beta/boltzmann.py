import numpy as np
from scipy.constants import hbar, pi
from scipy.constants import electron_mass, elementary_charge
from numba import njit
import time


import matplotlib.pyplot as plt


class Circle:


    @staticmethod
    @njit
    def relaxation_time(phi):
        """
        Relaxation time as a function of angle phi.
        """
        return 0.2e-12 - 0.02e-12 * np.cos(4 * phi)
    
    
    @staticmethod
    @njit
    def effective_mass(phi):
        """
        Effective mass as a function of angle phi.
        """
        return 5.0 * electron_mass
    
    
    @staticmethod
    @njit
    def fermi_wavevector(phi):
        """
        Fermi wavevector as a function of angle phi.
        """
        k00 = 0.728e10
        k40 = -3.3e-2 * k00
        return k00 + np.cos(4 * phi) * k40
    
    
    def cyclotron_frequency(self, phi, field):
        """
        Cyclotron frequency as a function of angle phi and magnetic field.
        """
        return elementary_charge * field / self.effective_mass(phi)
    
    
    def zeta(self, phi):
        """
        Zeta (angle between Kf and Vf) as a function of angle phi given
        Vf is enforced to be normal to the Fermi surface.
        """
        dphi = 1e-6
        ln_kf_plus = np.log(self.fermi_wavevector(phi + dphi))
        ln_kf_minus = np.log(self.fermi_wavevector(phi - dphi))
        d_lnkf_dphi = (ln_kf_plus - ln_kf_minus) / (2 * dphi)
        return np.arctan(d_lnkf_dphi)
    
    
    def fermi_velocity(self, phi):
        """
        Fermi velocity as a function of angle phi.
        """
        return self.fermi_wavevector(phi) * hbar / self.effective_mass(phi)
    
    
    def fermi_velocity_x(self, phi):
        """
        Fermi velocity in the x-direction as a function of angle phi.
        """
        return self.fermi_wavevector(phi) * np.cos(phi - self.zeta(phi)) * hbar / self.effective_mass(phi)
    
    
    def fermi_velocity_y(self, phi):
        """
        Fermi velocity in the y-direction as a function of angle phi.
        """
        return self.fermi_wavevector(phi) * np.sin(phi - self.zeta(phi)) * hbar / self.effective_mass(phi)
    

    def damping_exponent(self, theta, phi, field, n_points=150, **kwargs):
        """
        Vectorized approximation of the damping exponent using the trapezoidal rule.
        theta, phi can be arrays (broadcastable).
        """
        # theta = np.asarray(theta)
        # phi = np.asarray(phi)
        lower = theta - phi
        upper = theta

        # Broadcast shapes
        shape = np.broadcast(theta, phi).shape
        lower = np.broadcast_to(lower, shape)
        upper = np.broadcast_to(upper, shape)

        # Integration points along the last axis
        x = np.linspace(0, 1, n_points)
        x = x.reshape((-1,) + (1,) * len(shape))  # shape (n_points, 1, ..., 1)
        # Interpolate between lower and upper for each theta, phi
        xs = lower + (upper - lower) * x  # shape (n_points, ...)

        # Evaluate integrand at each point
        integrand = 1 / (self.cyclotron_frequency(xs, field) * self.relaxation_time(xs, **kwargs))
        # Integrate along the first axis (integration variable)
        integral = np.trapezoid(integrand, xs, axis=0)
        return integral


    def damping_factor(self, theta, phi, field):
        """
        Relaxation damping factor as a function of theta, phi, and field.
        """
        exponent = self.damping_exponent(theta, phi, field)
        return np.exp(-exponent)


    def prefactor(self, field):
        """
        Prefactor for the conductivity calculation. Physical constants
        and c-axis lattice parameter.
        """
        numerator = elementary_charge**3 * field
        denominator = 2 * pi**2 * hbar**2 * 13e-10
        return numerator / denominator
    
    
    def integrand_xx(self, theta, phi, field):
        """
        Integrand for the xx component of the conductivity.
        """
        numerator = self.fermi_velocity_x(theta) * self.fermi_velocity_x(theta - phi)
        denominator = self.cyclotron_frequency(theta, field) * self.cyclotron_frequency(theta - phi, field)
        #damp = self.damping_factor(theta, phi, field)
        return numerator / denominator
    
    
    
    def integrand_xy(self, theta, phi, field):
        """
        Integrand for the xy component of the conductivity.
        """
        numerator = self.fermi_velocity_x(theta) * self.fermi_velocity_y(theta - phi)
        denominator = self.cyclotron_frequency(theta, field) * self.cyclotron_frequency(theta - phi, field)
        #damp = self.damping_factor(theta, phi, field)
        return numerator / denominator
    
    
    def _find_damping_integration_limit(self, theta, field):
        """
        Find the maximum phi for which the damping factor is still significant.
        """
        phi_max = 1e-3
        while self.damping_factor(theta, phi_max, field) > 1e-6:
            phi_max *= 1.5
        return phi_max
    
    
    def _find_phi(self, theta, field, points=100):
        """
        Adaptive phi range/step used for integration. High density of points at
        small phi given that the damping factor decays exponentially.
        """
        phi_max = self._find_damping_integration_limit(theta, field)
        phi_min = 1e-6
        phi = np.geomspace(phi_min, phi_max, points)
        phi = np.insert(phi, 0, 0)
        return phi

    
    def conductivity(
        self, 
        field,
        n_theta=200,
        n_phi=1000,
        n_damping=150,
        **kwargs
    ):
        theta = np.linspace(0, 2 * pi, n_theta)
        phi = self._find_phi(0, field, points=n_phi)
        
        t0 = time.time()
        
        theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
        difference_mesh = theta_mesh - phi_mesh
        
        damp = self.damping_factor(theta_mesh, phi_mesh, field, n_points=n_damping, **kwargs)
        vfx_theta = self.fermi_velocity_x(theta_mesh)
        vfx_theta_minus_phi = self.fermi_velocity_x(difference_mesh)
        vfy_theta_minus_phi = self.fermi_velocity_y(difference_mesh)
        wc_theta = self.cyclotron_frequency(theta_mesh, field)
        wc_theta_minus_phi = self.cyclotron_frequency(difference_mesh, field)
        
        y1 = np.zeros_like(theta_mesh)
        #y1 = self.integrand_xx(theta_mesh, phi_mesh, field) * damp
        y1 = (vfx_theta * vfx_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
        sigma_xx = self.prefactor(field) * np.trapezoid(np.trapezoid(y1, phi, axis=1), theta)
        
        y2 = np.zeros_like(theta_mesh)
        #y2 = self.integrand_xy(theta_mesh, phi_mesh, field) * damp
        y2 = (vfx_theta * vfy_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
        sigma_xy = self.prefactor(field) * np.trapezoid(np.trapezoid(y2, phi, axis=1), theta)

        print(f"Time taken for integrand calculation: {time.time() - t0:.2f} seconds")
        
        return sigma_xx, sigma_xy
    
    
    def resistivity(
        self, 
        field,
        n_theta=200,
        n_phi=1000,
        n_damping=150,
        **kwargs,
    ):
        sigma_xx, sigma_xy = self.conductivity(field, **kwargs)
        rho_xx = sigma_xx / (sigma_xx**2 + sigma_xy**2)
        rho_xy = -sigma_xy / (sigma_xx**2 + sigma_xy**2)
        return rho_xx, rho_xy
    
    
    def plot_relaxation_time(self, ax):
        phi = np.linspace(0, 2 * pi, 100)
        ax.plot(phi, self.relaxation_time(phi), label='Relaxation Time')
        ax.set_xlabel('Phi (radians)')
        ax.set_ylabel('Relaxation Time (s)')
        ax.legend()
        ax.grid()
        
        
    def plot_relaxation_rate(self, ax):
        phi = np.linspace(0, 2 * pi, 100)
        ax.plot(phi, 1 / self.relaxation_time(phi) * 1e-12, label='Relaxation Rate')
        ax.set_xlabel('Phi (radians)')
        ax.set_ylabel('Relaxation Rate (ps$^{-1}$)')
        ax.legend()
        ax.grid()
    






def cyclotron_frequency(effective_mass, field):
    """
    Returns the cyclotron frequency for a given angle phi and magnetic field.
    """
    return elementary_charge * field / effective_mass



def zeta(fermi_wavevector, delta_phi):
    """
    Returns the zeta angle for a given angle phi using central differences.
    fermi_wavevector: 1D numpy array of k_F values at phi grid points.
    phi: 1D numpy array of phi grid points (same length as fermi_wavevector).
    delta_phi: spacing between phi grid points.
    """
    # Use central difference for interior points, forward/backward for edges
    ln_kf = np.log(fermi_wavevector)
    d_lnkf_dphi = np.zeros_like(ln_kf)
    d_lnkf_dphi[1:-1] = (ln_kf[2:] - ln_kf[:-2]) / (2 * delta_phi)
    d_lnkf_dphi[0] = (ln_kf[1] - ln_kf[0]) / delta_phi
    d_lnkf_dphi[-1] = (ln_kf[-1] - ln_kf[-2]) / delta_phi
    return np.arctan(d_lnkf_dphi)



def fermi_velocity(fermi_wavevector, effective_mass, phi):
    """
    Returns the Fermi velocity for a given angle phi.
    """
    return fermi_wavevector(phi) * hbar / effective_mass(phi)



def fermi_velocity_x(fermi_wavevector, zeta, effective_mass, phi):
    """
    Returns the x-component of the Fermi velocity for a given angle phi.
    """
    return fermi_wavevector * np.cos(phi - zeta) * hbar / effective_mass



def fermi_velocity_y(fermi_wavevector, zeta, effective_mass, phi):
    """
    Returns the y-component of the Fermi velocity for a given angle phi.
    """
    return fermi_wavevector * np.cos(phi - zeta) * hbar / effective_mass



def damping_factor(precomputed_wct, theta, phi, delta):
    """
    Efficiently calculates the damping factor using precomputed cyclotron frequency (wc)
    and relaxation time (tau).
    
    precomputed_wct: 1D numpy array of precomputed values of wc * tau at uniform grid points.
    theta: 2D numpy array of angles (meshgrid).
    phi: 2D numpy array of angles (meshgrid).
    delta: Spacing between the uniform grid points used for precomputation.
    
    returns: 2D numpy array of damping factors.
    """

    # Compute lower and upper indices for each (theta, phi) pair
    n_points = len(precomputed_wct)
    lower_index = np.round((theta - phi) / delta).astype(int) % n_points
    upper_index = np.round(theta / delta).astype(int) % n_points
    
    
    # Ensure lower_index <= upper_index for integration direction
    # wrap around indices if necessary
    mask = lower_index > upper_index
    lower_index[mask], upper_index[mask] = upper_index[mask], lower_index[mask]
    
    # Initialize the result array
    ans = np.zeros_like(theta)

    
    return np.exp(-ans)

    # Flatten meshgrids for easier indexing
    # theta_flat = theta.ravel()
    # phi_flat = phi.ravel()

    # # Indices for upper (theta) and lower (theta-phi) bounds
    # idx_upper = np.round(theta_flat / delta).astype(int) % n_points
    # idx_lower = np.round((theta_flat - phi_flat) / delta).astype(int) % n_points

    # # Ensure idx_lower <= idx_upper for integration direction
    # mask = idx_lower > idx_upper
    # idx_lower[mask], idx_upper[mask] = idx_upper[mask], idx_lower[mask]

    # # Compute integral for each pair
    # result = np.zeros_like(theta_flat)
    # for i in range(len(theta_flat)):
    #     if idx_upper[i] == idx_lower[i]:
    #         result[i] = 0.0
    #     else:
    #         # Ensure slicing is correct for 1D array
    #         if idx_upper[i] > idx_lower[i]:
    #             wct_segment = precomputed_wct[idx_lower[i]:idx_upper[i] + 1]
    #         else:
    #             wct_segment = precomputed_wct[idx_upper[i]:idx_lower[i] + 1]
    #         # Integration variable: angle = delta * index
    #         result[i] = np.trapz(1.0 / wct_segment, dx=delta)

    # # Reshape to original meshgrid shape and return exp(-integral)
    # return np.exp(-result.reshape(theta.shape))
    
    # lower = theta - phi
    # upper = theta

    # # Broadcast shapes
    # shape = np.broadcast(theta, phi).shape
    # lower = np.broadcast_to(lower, shape)
    # upper = np.broadcast_to(upper, shape)

    # # Integration points along the last axis
    # x = np.linspace(0, 1, n_points).reshape(-1, 1, 1)  # shape (n_points, 1, 1)
    # xs = lower + (upper - lower) * x  # shape (n_points, ...)

    # # Interpolate precomputed values for wc and tau
    # wc_interp = np.interp(xs, theta, precomputed_wc)
    # tau_interp = np.interp(xs, theta, precomputed_tau)

    # # Evaluate integrand at each point
    # integrand = 1 / (wc_interp * tau_interp)

    # # Integrate along the first axis (integration variable)
    # integral = np.sum((integrand[:-1] + integrand[1:]) * (x[1:] - x[:-1]) / 2, axis=0)
    # return np.exp(-integral)


def prefactor(field):
    """
    Returns the prefactor for the conductivity calculation.
    """
    numerator = elementary_charge**3 * field
    denominator = 2 * pi**2 * hbar**2 * 13e-10
    return numerator / denominator


def find_damping_integration_limit(effective_mass, relaxation_time, theta, field):
    """
    Finds the upper limit for the damping integration based on the damping factor.
    """
    phi_max = 1e-3
    while damping_factor(effective_mass, relaxation_time, theta, phi_max, field) > 1e-6:
        phi_max *= 1.5
    return phi_max


def find_phi(effective_mass, relaxation_time, theta, field, points=100):
    """
    Finds the phi values for integration based on the damping factor.
    """
    phi_max = find_damping_integration_limit(effective_mass, relaxation_time, theta, field)
    phi_min = 1e-6
    phi = np.geomspace(phi_min, phi_max, points)
    phi = np.insert(phi, 0, 0)
    return phi


def calculate_conductivity(
    fermi_wavevector,
    effective_mass,
    relaxation_time,
    field,
):
    t0 = time.time()
    N = 200
    delta_theta = 2 * pi / N
    theta = np.linspace(0, 2 * pi, N)
    phi_max = 2 * pi
    phi_min = 0
    phi = np.linspace(phi_min, phi_max, 1000)


    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
    difference_mesh = theta_mesh - phi_mesh
    print(f"Setup time: {time.time() - t0:.2f} seconds")

    t0 = time.time()

    calculated_kf_theta = fermi_wavevector(theta_mesh)
    calculated_kf_difference = fermi_wavevector(difference_mesh)
    calculated_zeta_theta = zeta(calculated_kf_theta, delta_phi=delta_theta)
    calculated_zeta_difference = zeta(calculated_kf_difference, delta_phi=delta_theta)
    calculated_mass_theta = effective_mass(theta_mesh)
    calculated_mass_difference = effective_mass(difference_mesh)

    wc_theta = cyclotron_frequency(calculated_mass_theta, field)
    tau_theta = relaxation_time(theta_mesh)
    wct_theta = wc_theta * tau_theta
    wc_theta_minus_phi = cyclotron_frequency(calculated_mass_difference, field)

    damp = damping_factor(wct_theta, theta_mesh, phi_mesh, delta=delta_theta)
    vx_theta = fermi_velocity_x(calculated_kf_theta, calculated_zeta_theta, calculated_mass_theta, theta_mesh)
    vx_theta_minus_phi = fermi_velocity_x(calculated_kf_difference, calculated_zeta_difference, calculated_mass_difference, difference_mesh)
    vy_theta_minus_phi = fermi_velocity_y(calculated_kf_difference, calculated_zeta_difference, calculated_mass_difference, difference_mesh)

    # y1 = (vx_theta * vx_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
    # sigma_xx = prefactor(field) * np.trapezoid(np.trapezoid(y1, phi, axis=1), theta)
    # y2 = (vx_theta * vy_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
    # sigma_xy = prefactor(field) * np.trapezoid(np.trapezoid(y2, phi, axis=1), theta)

    print(f"Computed values in {time.time() - t0:.2f} seconds")

    return 0, 0
    
# def calculate_conductivity(
#     fermi_wavevector,
#     effective_mass,
#     relaxation_time,
#     field,
#     n_theta=200,
#     n_phi=1000,
#     n_damping=150,
# ):
#     theta = np.linspace(0, 2 * pi, n_theta)
#     phi = find_phi(effective_mass, relaxation_time, 0, field, points=n_phi)
    
#     t0 = time.time()
#     theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
#     difference_mesh = theta_mesh - phi_mesh
    
    
#     damp = damping_factor(effective_mass, relaxation_time, theta_mesh, phi_mesh, field, n_points=n_damping)
#     vfx_theta = fermi_velocity_x(fermi_wavevector, zeta, effective_mass, theta_mesh)
#     vfx_theta_minus_phi = fermi_velocity_x(fermi_wavevector, zeta, effective_mass, difference_mesh)
#     vfy_theta_minus_phi = fermi_velocity_y(fermi_wavevector, zeta, effective_mass, difference_mesh)
#     wc_theta = cyclotron_frequency(effective_mass, theta_mesh, field)
#     wc_theta_minus_phi = cyclotron_frequency(effective_mass, difference_mesh, field)
#     y1 = (vfx_theta * vfx_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
#     sigma_xx = prefactor(field) * np.trapezoid(np.trapezoid(y1, phi, axis=1), theta)
#     y2 = (vfx_theta * vfy_theta_minus_phi) / (wc_theta * wc_theta_minus_phi) * damp
#     sigma_xy = prefactor(field) * np.trapezoid(np.trapezoid(y2, phi, axis=1), theta)
    
#     print(f"[M2] Time taken for integrand calculation: {time.time() - t0:.2f} seconds")
#     return sigma_xx, sigma_xy
    
    


def calculate_resistivity(
    fermi_wavevector,
    effective_mass,
    relaxation_time,
    field,
    n_theta=200,
    n_phi=1000,
    n_damping=150,
):
    sigma_xx, sigma_xy = calculate_conductivity(
        fermi_wavevector,
        effective_mass,
        relaxation_time,
        field,
        n_theta=n_theta,
        n_phi=n_phi,
        n_damping=n_damping
    )
    rho_xx = sigma_xx / (sigma_xx**2 + sigma_xy**2)
    rho_xy = -sigma_xy / (sigma_xx**2 + sigma_xy**2)
    return rho_xx, rho_xy
    




fs = Circle()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

@njit
def fermi_wavevector(phi):
    """
    Returns the wavevector k for a given angle phi.
    """
    k00 = 0.728e10
    k40 = -3.3e-2 * k00
    return k00 + np.cos(4 * phi) * k40


@njit
def effective_mass(phi):
    """
    Returns the effective mass for a given angle phi.
    """
    return 5.0 * electron_mass


@njit
def relaxation_time(phi):
    """
    Returns the relaxation time for a given angle phi.
    """
    return 0.2e-12 - 0.02e-12 * np.cos(4 * phi)


for B in np.linspace(0.5, 35, 25):
    sxx, sxy = calculate_conductivity(
        fermi_wavevector,
        effective_mass,
        relaxation_time,
        B,
    )
plt.show()