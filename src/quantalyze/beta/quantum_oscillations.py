from .fft import fft, Window
from ..core.constants import BOLTZMANN_CONSTANT, REDUCED_PLANCK_CONSTANT, ELEMENTARY_CHARGE, ELECTRON_MASS
from ..core.fitting import fit
import numpy as np


def lifshitz_kosevich_amplitude(T, m_eff, B):
    """
    Calculate the Lifshitz-Kosevich temperature dependence of quantum oscillation amplitude.

    Parameters:
    T (float or np.ndarray): Temperature in Kelvin.
    m_eff (float): Effective mass in units of the electron mass.
    B (float): Magnetic field in Tesla.

    Returns:
    float or np.ndarray: Amplitude of quantum oscillations.
    """
    # Convert effective mass to SI units
    m_eff_si = m_eff * ELECTRON_MASS

    # Calculate the cyclotron frequency
    omega_c = ELEMENTARY_CHARGE * B / m_eff_si

    # Calculate the damping factor
    x = (2 * np.pi**2 * BOLTZMANN_CONSTANT * T) / (REDUCED_PLANCK_CONSTANT * omega_c)
    amplitude = x / np.sinh(x)

    return amplitude


def fit_lk_effective_mass(df, temperature_column, amplitude_column, field, m_eff_initial):
    """
    Fit the Lifshitz-Kosevich effective mass to the quantum oscillation amplitude data.

    Parameters:
    df (pd.DataFrame): DataFrame containing temperature and amplitude data.
    temperature_column (str): Name of the column containing temperature data.
    amplitude_column (str): Name of the column containing amplitude data.
    B (float): Magnetic field in Tesla.
    m_eff_initial (float): Initial guess for the effective mass.

    Returns:
    float: Fitted effective mass.
    """
    def lk_amplitude(T, m_eff):
        return lifshitz_kosevich_amplitude(T, m_eff, field)

    # Use the core fitting function
    fit_result = fit(lk_amplitude, df, temperature_column, amplitude_column, p0=[m_eff_initial])
    
    return fit_result.parameters[0]  # Return the fitted effective mass