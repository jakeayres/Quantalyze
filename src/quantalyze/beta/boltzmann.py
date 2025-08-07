from scipy.constants import hbar, pi
from scipy.constants import electron_mass, elementary_charge
import numpy as np
from numba import njit, typeof, float64, prange


@njit
def fermi_wavevector(theta):
    """
    Returns the fermi wavevector for a given angle theta.
    Is a user provided input.
    """
    k00 = 0.728e10
    k40 = k00 * -3.3e-2
    return k00 + k40*np.cos(4 * theta)
    


@njit
def effective_mass(theta):
    """
    Returns the effective mass for a given angle theta.
    Is a user provided input.
    """
    return 1 * electron_mass

    

@njit
def relaxation_time(theta, T):
    """
    Returns the product of the cyclotron frequency and the relaxation time
    for a given angle theta and temperature T.
    Is a user provided input.
    """
    
    Aiso = 2.43
    Biso = 3.13e-4
    Aani = 0.135
    Bani = 6.11e-2
    alpha = Aiso + Biso*T*T
    beta = Aani + Bani*T
    
    gamma = beta/(2*alpha + beta)
    tau_zero = ((1-gamma)/alpha) / 45
    tau = tau_zero / (1 + gamma*np.cos(4*theta)) * electron_mass / elementary_charge
    return tau


def build_cyclotron_frequency(effective_mass):
    """
    Builds a function to calculate the cyclotron frequency
    from the effective mass function.

    Returns a function that takes theta and B as inputs.
    """
    @njit
    def cyclotron_frequency(theta, B):
        return elementary_charge * B / effective_mass(theta)
    return cyclotron_frequency


def build_omega_c_tau(cyclotron_frequency, relaxation_time):
    """
    Builds a function to calculate the product of cyclotron frequency
    and relaxation time.
    """
    @njit
    def omega_c_tau(theta, T, B):
        omega_c = cyclotron_frequency(theta, B)
        tau = relaxation_time(theta, T)
        return omega_c * tau
    return omega_c_tau


def build_zeta(fermi_wavevector):
    """
    Builds a function to calculate the zeta angle between the
    fermi wavevector and the fermi velocity. It ensure that
    the fermi velocity is always normal to the fermi surface.
    """
    @njit
    def impl(phi):
        dphi = 1e-8
        ln_kf_plus = np.log(fermi_wavevector(phi + dphi))
        ln_kf_minus = np.log(fermi_wavevector(phi - dphi))
        d_lnkf_dphi = (ln_kf_plus - ln_kf_minus) / (2 * dphi)
        return np.arctan(d_lnkf_dphi)

    return impl


def build_fermi_velocity_x(fermi_wavevector, zeta):
    """
    Builds a function to calculate the x component of the fermi velocity
    from the fermi wavevector and the zeta angle.
    """
    @njit
    def impl(theta):
        return hbar * fermi_wavevector(theta) * np.cos(theta - zeta(theta)) / effective_mass(theta)
    return impl


def build_fermi_velocity_y(fermi_wavevector, zeta):
    """
    Builds a function to calculate the y component of the fermi velocity
    from the fermi wavevector and the zeta angle.
    """
    @njit
    def impl(theta):
        return hbar * fermi_wavevector(theta) * np.sin(theta - zeta(theta)) / effective_mass(theta)
    return impl


def build_damping_exponent_integrand(omega_c_tau):
    @njit
    def damping_exponent_integrand(phi, T, B):
        return 1.0 / omega_c_tau(phi, T, B)
    return damping_exponent_integrand


def build_damping_exponent(damping_exponent_integrand):
    @njit
    def damping_exponent(theta, phi, T, B):
        N = 150
        start = theta - phi
        end = theta
        angles = np.linspace(start, end, N)
        integrand = np.empty_like(angles, dtype=np.float64)
        for i in range(N):
            integrand[i] = damping_exponent_integrand(angles[i], T, B)
        integral = 0.0
        for i in range(N-1):
            integral += 0.5 * (integrand[i] + integrand[i+1]) * (angles[i+1] - angles[i])
        return integral
    return damping_exponent


def build_exponential_damping(damping_exponent):
    """
    Builds a function to calculate the exponential damping factor
    based on the damping exponent.
    """
    @njit
    def exponential_damping(theta, phi, T, B):
        return np.exp(-damping_exponent(theta, phi, T, B))
    return exponential_damping


def build_conductivity(
        fermi_wavevector,
        effective_mass,
        relaxation_time,
        c_axis_length,
    ):
    """
    Returns a function that calculates the conductivity given the input functions.
    """

    cyclotron_frequency = build_cyclotron_frequency(effective_mass)
    omega_c_tau = build_omega_c_tau(cyclotron_frequency, relaxation_time)
    zeta = build_zeta(fermi_wavevector)
    fermi_velocity_x = build_fermi_velocity_x(fermi_wavevector, zeta)
    fermi_velocity_y = build_fermi_velocity_y(fermi_wavevector, zeta)
    damping_exponent_integrand = build_damping_exponent_integrand(omega_c_tau)
    damping_exponent = build_damping_exponent(damping_exponent_integrand)
    exponential_damping = build_exponential_damping(damping_exponent)


    @njit(parallel=True)
    def conductivity(T, B):
        
        N_theta = 1000
        N_phi = 1500
        theta_vals = np.linspace(0, np.pi*2, N_theta)
        phi_vals = np.geomspace(1e-6, np.pi*2, N_phi)


        # Precompute the damping exponent for all theta and phi values
        exp_damping_matrix = np.empty((N_theta, N_phi), dtype=np.float64)
        for i in prange(N_theta):
            for j in range(N_phi):
                exp_damping_matrix[i, j] = exponential_damping(theta_vals[i], phi_vals[j], T, B)


        # Precompute fermi_velocity_x(theta), wc_theta(theta, B)
        vx_theta = np.empty(N_theta, dtype=np.float64)
        wc_theta = np.empty(N_theta, dtype=np.float64)
        for i in prange(N_theta):
            vx_theta[i] = fermi_velocity_x(theta_vals[i])
            wc_theta[i] = cyclotron_frequency(theta_vals[i], B)


        # Precompute fermi_velocity_x(theta - phi) and fermi_velocity_y(theta - phi)
        # and cyclotron_frequency(theta - phi, B)
        vx_theta_phi = np.empty((N_theta, N_phi), dtype=np.float64)
        vy_theta_phi = np.empty((N_theta, N_phi), dtype=np.float64)
        wc_theta_phi = np.empty((N_theta, N_phi), dtype=np.float64)
        for i in prange(N_theta):
            for j in range(N_phi):
                angle = theta_vals[i] - phi_vals[j]
                vx_theta_phi[i, j] = fermi_velocity_x(angle)
                vy_theta_phi[i, j] = fermi_velocity_y(angle)
                wc_theta_phi[i, j] = cyclotron_frequency(angle, B)


        # Main integration loops
        x_integral = 0.0
        y_integral = 0.0
        for i in prange(N_theta - 1):
            dtheta = theta_vals[i+1] - theta_vals[i]
            vx_theta_i = vx_theta[i]
            vx_theta_i_plus_1 = vx_theta[i+1]
            wc_theta_i = wc_theta[i]
            wc_theta_i_plus_1 = wc_theta[i + 1]

            for j in range(N_phi - 1):
                dphi = phi_vals[j+1] - phi_vals[j]
                f00_x = vx_theta_i         * vx_theta_phi[i, j]   * exp_damping_matrix[i, j]         / (wc_theta_i * wc_theta_phi[i, j])
                f01_x = vx_theta_i         * vx_theta_phi[i, j+1] * exp_damping_matrix[i, j+1]       / (wc_theta_i * wc_theta_phi[i, j+1])
                f10_x = vx_theta_i_plus_1  * vx_theta_phi[i+1, j] * exp_damping_matrix[i+1, j]       / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
                f11_x = vx_theta_i_plus_1  * vx_theta_phi[i+1, j+1] * exp_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
                f00_y = vx_theta_i         * vy_theta_phi[i, j]     * exp_damping_matrix[i, j]       / (wc_theta_i * wc_theta_phi[i, j])
                f01_y = vx_theta_i         * vy_theta_phi[i, j+1]   * exp_damping_matrix[i, j+1]     / (wc_theta_i * wc_theta_phi[i, j+1])
                f10_y = vx_theta_i_plus_1  * vy_theta_phi[i+1, j]   * exp_damping_matrix[i+1, j]     / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
                f11_y = vx_theta_i_plus_1  * vy_theta_phi[i+1, j+1] * exp_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
                avg_x = 0.25 * (f00_x + f01_x + f10_x + f11_x)
                avg_y = 0.25 * (f00_y + f01_y + f10_y + f11_y)
                x_integral += avg_x * dtheta * dphi
                y_integral += avg_y * dtheta * dphi

        const = elementary_charge**3 * B / (2 * np.pi**2 * hbar**2 * c_axis_length)
        return (x_integral * const, y_integral * const)

    return conductivity