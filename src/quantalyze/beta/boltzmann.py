
from ..core.constants import hbar, elementary_charge
import numpy as np
from numba import njit

"""
Examples:
    >>> import quantalyze as qz
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> fermi_wavevector = np.ones_like(theta) * 1e10  # Example Fermi wavevector
    >>> effective_mass = np.ones_like(theta) * electron_mass  # Example effective mass
    >>> relaxation_time = np.ones_like(theta) * 1e-14  # Example relaxation time
    >>> c_axis_length = 1e-9  # Example c-axis length
    >>>  
    >>> fermi_surface = qz.FermiSurface(theta, fermi_wavevector, effective_mass, relaxation_time, c_axis_length)
    >>> magnetic_field = 1.0  # Example magnetic field in Tesla
    >>> sxx, sxy, syx, syy = fermi_surface.calculate_conductivity(magnetic_field)
    >>> rxx = sxx / (sxx * syy - sxy * syx)

"""

@njit
def _periodic_quadratic_interpolation(theta_array, value_array, theta):
    """ Quadratic interpolation centered around theta (one point before and one point after)
    Does assume equally spaced theta_array, and does assume that theta_array is periodic and
    covers the range [0, 2*pi].
    """
    L = np.pi * 2
    angle = theta % L

    # Find index
    dtheta = theta_array[1] - theta_array[0]
    index_0 = int(angle/dtheta)
    index_minus_1 = (index_0 - 1) % len(theta_array)
    index_plus_1 = (index_0 + 1) % len(theta_array)

    # Get the three points for interpolation
    theta_0 = theta_array[index_0]
    theta_minus_1 = theta_array[index_minus_1]
    theta_plus_1 = theta_array[index_plus_1]

    value_0 = value_array[index_0]
    value_minus_1 = value_array[index_minus_1]
    value_plus_1 = value_array[index_plus_1]

    # Quadratic interpolation
    a = (value_plus_1 - 2*value_0 + value_minus_1) / (2 * dtheta**2)
    b = (value_plus_1 - value_minus_1) / (2 * dtheta)
    c = value_0

    return a * (angle - theta_0)**2 + b * (angle - theta_0) + c


@njit
def _calculate_zeta_array(theta_array, fermi_wavevector_array):
    """ Use quadratic interpolation to calculate the zeta angle for each theta in the array,
    based on the fermi wavevector array. Take a dtheta that is 100x smaller than the spacing
    of the theta_array using quadratic interpolation to ensure accurate numerical differentiation.
    """
    zeta_array = np.empty_like(theta_array, dtype=np.float64)
    for i in range(len(theta_array)):
        dtheta_plus = theta_array[i] + (theta_array[1] - theta_array[0]) / 100
        dtheta_minus = theta_array[i] - (theta_array[1] - theta_array[0]) / 100
        kf_plus = _periodic_quadratic_interpolation(theta_array, fermi_wavevector_array, dtheta_plus)
        kf_minus = _periodic_quadratic_interpolation(theta_array, fermi_wavevector_array, dtheta_minus)
        ln_kf_plus = np.log(kf_plus)
        ln_kf_minus = np.log(kf_minus)
        d_lnkf_dtheta = (ln_kf_plus - ln_kf_minus) / (dtheta_plus - dtheta_minus)
        zeta_array[i] = np.arctan(d_lnkf_dtheta)
    return zeta_array



@njit
def _calculate_exponential_damping_matrix(
    theta_array,
    phi_array,
    omega_c_tau_array,
    points=150,
    ):
    damping_exponent_matrix = np.empty((len(theta_array), len(phi_array)), dtype=np.float64)
    for i in range(len(theta_array)):
        for j in range(len(phi_array)):
            start = theta_array[i] - phi_array[j]
            end = theta_array[i]
            angles = np.linspace(start, end, points)

            integrand = np.empty_like(angles, dtype=np.float64)
            for k in range(points):
                # Use quadratic interpolation to get omega_c_tau at the angle angles[k]
                omega_c_tau = _periodic_quadratic_interpolation(theta_array, omega_c_tau_array, angles[k])
                integrand[k] = 1.0 / omega_c_tau
            integral = 0.0
            for k in range(points-1):
                integral += 0.5 * (integrand[k] + integrand[k+1]) * (angles[k+1] - angles[k])
            damping_exponent_matrix[i, j] = integral
    exponential_damping_matrix = np.exp(-damping_exponent_matrix)
    return exponential_damping_matrix


@njit
def _bilinear_integration(
    theta,
    phi,
    vx_theta,
    vy_theta,
    wc_theta,
    vx_theta_phi,
    vy_theta_phi,
    wc_theta_phi,
    exponential_damping_matrix,
    field,
    c_axis_length,
    ):

    # Main integration loops
    xx_integral = 0.0
    xy_integral = 0.0
    yx_integral = 0.0
    yy_integral = 0.0

    for i in range(len(theta) - 1):
        dtheta = theta[i+1] - theta[i]
        vx_theta_i = vx_theta[i]
        vy_theta_i = vy_theta[i]
        vx_theta_i_plus_1 = vx_theta[i+1]
        vy_theta_i_plus_1 = vy_theta[i+1]
        wc_theta_i = wc_theta[i]
        wc_theta_i_plus_1 = wc_theta[i + 1]

        for j in range(len(phi) - 1):
            dphi = phi[j+1] - phi[j]
           
            # Bilinear interpolation of the Vx.Vx integrand
            f00_xx = vx_theta_i         * vx_theta_phi[i, j]   * exponential_damping_matrix[i, j]         / (wc_theta_i * wc_theta_phi[i, j])
            f01_xx = vx_theta_i         * vx_theta_phi[i, j+1] * exponential_damping_matrix[i, j+1]       / (wc_theta_i * wc_theta_phi[i, j+1])
            f10_xx = vx_theta_i_plus_1  * vx_theta_phi[i+1, j] * exponential_damping_matrix[i+1, j]       / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
            f11_xx = vx_theta_i_plus_1  * vx_theta_phi[i+1, j+1] * exponential_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
            avg_xx = 0.25 * (f00_xx + f01_xx + f10_xx + f11_xx)

            # Bilinear interpolation of the Vx.Vy integrand
            f00_xy = vx_theta_i         * vy_theta_phi[i, j]     * exponential_damping_matrix[i, j]       / (wc_theta_i * wc_theta_phi[i, j])
            f01_xy = vx_theta_i         * vy_theta_phi[i, j+1]   * exponential_damping_matrix[i, j+1]     / (wc_theta_i * wc_theta_phi[i, j+1])
            f10_xy = vx_theta_i_plus_1  * vy_theta_phi[i+1, j]   * exponential_damping_matrix[i+1, j]     / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
            f11_xy = vx_theta_i_plus_1  * vy_theta_phi[i+1, j+1] * exponential_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
            avg_xy = 0.25 * (f00_xy + f01_xy + f10_xy + f11_xy)

            # Bilinear interpolation of the Vy.Vx integrand
            f00_yx = vy_theta_i         * vx_theta_phi[i, j]     * exponential_damping_matrix[i, j]       / (wc_theta_i * wc_theta_phi[i, j])
            f01_yx = vy_theta_i         * vx_theta_phi[i, j+1]   * exponential_damping_matrix[i, j+1]     / (wc_theta_i * wc_theta_phi[i, j+1])
            f10_yx = vy_theta_i_plus_1  * vx_theta_phi[i+1, j]   * exponential_damping_matrix[i+1, j]     / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
            f11_yx = vy_theta_i_plus_1  * vx_theta_phi[i+1, j+1] * exponential_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
            avg_yx = 0.25 * (f00_yx + f01_yx + f10_yx + f11_yx)

            # Bilinear interpolation of the Vy.Vy integrand
            f00_yy = vy_theta_i         * vy_theta_phi[i, j]     * exponential_damping_matrix[i, j]       / (wc_theta_i * wc_theta_phi[i, j])
            f01_yy = vy_theta_i         * vy_theta_phi[i, j+1]   * exponential_damping_matrix[i, j+1]     / (wc_theta_i * wc_theta_phi[i, j+1])
            f10_yy = vy_theta_i_plus_1  * vy_theta_phi[i+1, j]   * exponential_damping_matrix[i+1, j]     / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j])
            f11_yy = vy_theta_i_plus_1  * vy_theta_phi[i+1, j+1] * exponential_damping_matrix[i+1, j+1]   / (wc_theta_i_plus_1 * wc_theta_phi[i+1, j+1])
            avg_yy = 0.25 * (f00_yy + f01_yy + f10_yy + f11_yy)

            xx_integral += avg_xx * dtheta * dphi
            xy_integral += avg_xy * dtheta * dphi
            yx_integral += avg_yx * dtheta * dphi
            yy_integral += avg_yy * dtheta * dphi

    const = elementary_charge**3 * field / (2 * np.pi**2 * hbar**2 * c_axis_length)
    return (xx_integral * const, xy_integral * const, yx_integral * const, yy_integral * const)



@njit
def _compute_conductivity_from_arrays(
    theta,
    fermi_wavevector,
    effective_mass,
    relaxation_time,
    c_axis_length,
    magnetic_field,
    start_phi=1e-9, # start of phi integration
    end_phi=2*np.pi, # end of phi integration
    phi_points=250, # number of points for phi integration
    exponent_points=150, # number of points for exponent integration
    ):

    """
    Builds a conductivity function from precomputed arrays of the input functions.
    This is useful for cases where the input functions are expensive to compute
    or for which no simple analytical form is available e.g. from ab initio calculation
    or tight binding models. The arrays should be defined on the same grid of theta values
    The conductivity function will interpolate the input arrays to compute the conductivity
    at any given theta.

    Args:
        theta: array of theta values (in radians) for which the input functions are defined.
        fermi_wavevector: array of fermi wavevector values for each theta.
        effective_mass: array of effective mass values for each theta.
        relaxation_time: array of relaxation time values for each theta.
        c_axis_length: length of the c-axis (in meters).
        magnetic_field: magnetic field strength (in Tesla).
        start_phi: start of phi integration (in radians).
        end_phi: end of phi integration (in radians).
        phi_points: number of points for phi integration.
        exponent_points: number of points for exponent integration.

    Returns:
        sxx, sxy, syx, syy: conductivity tensor components.
    """

    # Set up the phi array for integration:
    # Geometric spacing is used to capture the rapid decay of the exponential 
    # damping at small phi values
    phi = np.geomspace(start_phi, end_phi, phi_points)

    # Precompute omega_c and omega_c_tau for all theta values
    wc_theta = elementary_charge * magnetic_field / effective_mass
    omega_c_tau = wc_theta * relaxation_time

    # Precompute the fermi velocity components for all theta values
    zeta = _calculate_zeta_array(theta, fermi_wavevector)
    vx_theta = hbar * fermi_wavevector * np.cos(theta - zeta) / effective_mass
    vy_theta = hbar * fermi_wavevector * np.sin(theta - zeta) / effective_mass

    # Precompute the fermi velocity components for all theta - phi values
    # and omega_c for all theta - phi values
    vx_theta_phi = np.empty((len(theta), len(phi)), dtype=np.float64)
    vy_theta_phi = np.empty((len(theta), len(phi)), dtype=np.float64)
    wc_theta_phi = np.empty((len(theta), len(phi)), dtype=np.float64)
    for i in range(len(theta)):
        for j in range(len(phi)):
            angle = theta[i] - phi[j]
            vx_theta_phi[i, j] = _periodic_quadratic_interpolation(theta, vx_theta, angle)
            vy_theta_phi[i, j] = _periodic_quadratic_interpolation(theta, vy_theta, angle)
            wc_theta_phi[i, j] = _periodic_quadratic_interpolation(theta, wc_theta, angle)

    # Precompute the damping exponent for all theta and phi values
    exponential_damping_matrix = _calculate_exponential_damping_matrix(
        theta,
        phi,
        omega_c_tau,
        points=exponent_points,
        )
    
    # Perform the bilinear integration over theta and phi to compute the 
    # conductivity tensor components.
    sxx, sxy, syx, syy = _bilinear_integration(
        theta,
        phi,
        vx_theta,
        vy_theta,
        wc_theta,
        vx_theta_phi,
        vy_theta_phi,
        wc_theta_phi,
        exponential_damping_matrix,
        magnetic_field,
        c_axis_length,
    )

    return sxx, sxy, syx, syy



class FermiSurface:

    """
    Class to represent a Fermi surface and compute conductivity.
    
    Examples:
        >>> import quantalyze as qz
        >>> theta = np.linspace(0, 2*np.pi, 100)
        >>> fermi_wavevector = np.ones_like(theta) * 1e10  # Example Fermi wavevector
        >>> effective_mass = np.ones_like(theta) * electron_mass  # Example effective mass
        >>> relaxation_time = np.ones_like(theta) * 1e-14  # Example relaxation time
        >>> c_axis_length = 1e-9  # Example c-axis length
        >>>  
        >>> fermi_surface = qz.FermiSurface(theta, fermi_wavevector, effective_mass, relaxation_time, c_axis_length)
        >>> magnetic_field = 1.0  # Example magnetic field in Tesla
        >>> sxx, sxy, syx, syy = fermi_surface.calculate_conductivity(magnetic_field)
        >>> rxx = sxx / (sxx * syy - sxy * syx)
    
    """


    def __init__(
        self,
        theta,
        fermi_wavevector,
        effective_mass,
        relaxation_time,
        c_axis_length,
        ):
        self.theta = theta
        self.fermi_wavevector = fermi_wavevector
        self.effective_mass = effective_mass
        self.relaxation_time = relaxation_time
        self.c_axis_length = c_axis_length


    def fermi_wavevector_x(self):
        """
        Compute the x-component of the Fermi wavevector for each theta value.
        """
        return self.fermi_wavevector * np.cos(self.theta)


    def fermi_wavevector_y(self):
        """
        Compute the y-component of the Fermi wavevector for each theta value.
        """
        return self.fermi_wavevector * np.sin(self.theta)


    def reciprocal_lattice_vector(self):
        """
        Compute the reciprocal c-axis lattice vector.
        """
        return 2 * np.pi / self.c_axis_length


    def cylotron_frequency(self, magnetic_field):
        """
        Compute the local cyclotron frequency for a given magnetic field at each theta value.
        """
        return elementary_charge * magnetic_field / self.effective_mass


    def omega_c_tau(self, magnetic_field):
        """
        Compute the product of the cyclotron frequency and relaxation time for a given magnetic field at each theta value.
        """
        return self.cylotron_frequency(magnetic_field) * self.relaxation_time


    def mean_free_path(self):
        """
        Calculate the mean free path of electrons on the Fermi surface.
        """
        return hbar * self.fermi_wavevector * self.relaxation_time / self.effective_mass


    def _integrate_fermi_wavevector(self):
        """
        Integrate using trapezoidal rule and quadratic interpolation between points.
        This is used to compute the Fermi volume and the average effective mass.
        """
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
        """
        Compute the area of the Fermi surface in k-space by integrating the Fermi wavevector over theta.
        """
        return self._integrate_fermi_wavevector()


    def fermi_volume(self):
        """
        Calculate the volume of the Fermi surface in k-space by integrating the Fermi wavevector 
        and multiplying by the reciprocal lattice vector.
        """
        return self._integrate_fermi_wavevector() * self.reciprocal_lattice_vector()


    def carrier_density(self):
        """
        Calculate the carrier density based on the Fermi volume.
        """
        return self.fermi_volume() / (2 * np.pi)**3 * 2
    

    def calculate_normal_vectors(self):
        """
        Calculate the normal vectors to the Fermi surface at each theta value.
        """
        zeta = _calculate_zeta_array(self.theta, self.fermi_wavevector)
        normal_x = np.cos(self.theta - zeta)
        normal_y = np.sin(self.theta - zeta)
        return normal_x, normal_y


    def calculate_conductivity(
        self,
        magnetic_field,
        start_phi=1e-9, # start of phi integration
        end_phi=2*np.pi, # end of phi integration
        phi_points=250, # number of points for phi integration
        exponent_points=150, # number of points for exponent integration
    ):
        """
        Calculate the conductivity tensor components for the Fermi surface given a magnetic field.
        
        Args:
            magnetic_field: Magnetic field strength (in Tesla).
            start_phi: Start of phi integration (in radians).
            end_phi: End of phi integration (in radians).
            phi_points: Number of points for phi integration.
            exponent_points: Number of points for exponent integration.
            
        Returns:
            sxx, sxy, syx, syy: Conductivity tensor components.
        """
        return _compute_conductivity_from_arrays(
            self.theta,
            self.fermi_wavevector,
            self.effective_mass,
            self.relaxation_time,
            self.c_axis_length,
            magnetic_field=magnetic_field,
            start_phi=start_phi,
            end_phi=end_phi,
            phi_points=phi_points,
            exponent_points=exponent_points,
        )
