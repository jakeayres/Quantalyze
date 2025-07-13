"""
This module provides functions for analyzing quantum oscillation data.
It includes calculations for extremal areas and frequencies, damping factors (Lifshitz-Kosevich and Dingle),
effective mass fitting, and Fast Fourier Transform analysis of quantum oscillation signals.
"""

from .core.constants import BOLTZMANN_CONSTANT, REDUCED_PLANCK_CONSTANT, ELEMENTARY_CHARGE, ELECTRON_MASS, PI
from .core.fitting import fit
from .core.interpolation import interpolate
from .core.fft import fft as qz_fft
from .core.fft import Window
from numpy import sinh, exp, inf, linspace
import pandas as pd

def extremal_area(frequency: float):
    """
    Calculate the extremal area from the frequency of quantum oscillations.

    Args:
        frequency (float): Frequency in Tesla.

    Returns:
        float: Extremal area in units of m^-2.
    """
    return frequency * (2 * PI * ELEMENTARY_CHARGE) / REDUCED_PLANCK_CONSTANT


def extremal_frequency(area: float):
    """
    Calculate the extremal frequency from the extremal area.

    Args:
        area (float): Extremal area in units of m^-2.

    Returns:
        float: Frequency in Tesla.
    """
    return (REDUCED_PLANCK_CONSTANT * area) / (2 * PI * ELEMENTARY_CHARGE)


def lifshitz_kosevich_damping_factor(temperature: float, magnetic_field: float, effective_mass: float, harmonic: int = 1):
    """
    Calculate the Lifshitz-Kosevich damping factor for quantum oscillations.

    Args:
        temperature (float): Temperature in Kelvin.
        magnetic_field (float): Magnetic field in Tesla.
        effective_mass (float): Effective mass in units of the electron mass.
        harmonic (int, optional): Harmonic number. Defaults to 1.

    Returns:
        float: Amplitude of quantum oscillations.
    """
    cyclotron_mass = effective_mass * ELECTRON_MASS
    x = (2 * PI**2 * BOLTZMANN_CONSTANT * cyclotron_mass * temperature * harmonic) / (REDUCED_PLANCK_CONSTANT * ELEMENTARY_CHARGE * magnetic_field)
    amplitude = x / sinh(x)
    return amplitude


def dingle_lifetime(dingle_temperature: float):
    """
    Calculate the Dingle lifetime from the Dingle temperature.

    Args:
        dingle_temperature (float): Dingle temperature in Kelvin.

    Returns:
        float: Dingle lifetime in seconds.
    """
    return REDUCED_PLANCK_CONSTANT / (2 * PI * BOLTZMANN_CONSTANT * dingle_temperature)


def dingle_temperature(dingle_lifetime: float):
    """
    Calculate the Dingle temperature from the Dingle lifetime.

    Args:
        dingle_lifetime (float): Dingle lifetime in seconds.

    Returns:
        float: Dingle temperature in Kelvin.
    """
    return REDUCED_PLANCK_CONSTANT / (2 * PI * BOLTZMANN_CONSTANT * dingle_lifetime)


def dingle_damping_factor(magnetic_field: float, effective_mass: float, lifetime: float, harmonic: int = 1):
    """
    Calculate the Dingle damping factor for quantum oscillations.

    Args:
        magnetic_field (float): Magnetic field in Tesla.
        effective_mass (float): Effective mass in units of the electron mass.
        lifetime (float): Dingle lifetime in seconds.
        harmonic (int, optional): Harmonic number. Defaults to 1.

    Returns:
        float: Dingle damping factor.
    """
    cyclotron_mass = effective_mass * ELECTRON_MASS
    cylcotron_frequency = ELEMENTARY_CHARGE * magnetic_field / cyclotron_mass
    return exp(- (PI * harmonic) / (cylcotron_frequency * lifetime))


def mean_inverse_field(minimum: float, maximum: float):
    """
    Calculate the mean inverse field from the minimum and maximum fields.

    Args:
        minimum (float): Minimum magnetic field in Tesla.
        maximum (float): Maximum magnetic field in Tesla.

    Returns:
        float: Mean inverse field in Tesla^-1.
    """
    return ((1 / minimum + 1 / maximum) / 2)


def fit_effective_mass(df: pd.DataFrame, temperature_column: str, amplitude_column: str, magnetic_field: float, harmonic: int = 1):
    """
    Fit the effective mass from quantum oscillation data.

    Args:
        df (pandas.DataFrame): DataFrame containing quantum oscillation data.
        temperature_column (str): Name of the column containing temperature data.
        amplitude_column (str): Name of the column containing amplitude data.
        magnetic_field (float): Magnetic field in Tesla.
        harmonic (int, optional): Harmonic number. Defaults to 1.

    Returns:
        float: Effective mass in units of the electron mass.
    """

    def model(temperature, prefactor, effective_mass):
        return prefactor * lifshitz_kosevich_damping_factor(temperature, magnetic_field, effective_mass, harmonic)
    
    f = fit(
        model,
        df, 
        x_column=temperature_column,
        y_column=amplitude_column,
        bounds=([0, 0], [inf, inf]),
    )

    return f


def fft(
    df: pd.DataFrame,
    field_column: str,
    signal_column: str,
    minimum_field: float,
    maximum_field: float,
    points: int,
    background_function: callable,
    window: Window = Window.HANN,
    subtract_inverse_field: bool =False,
):
    """
    Perform a Fast Fourier Transform (FFT) on the quantum oscillation data.

    Args:
        df (pandas.DataFrame): DataFrame containing quantum oscillation data.
        field_column (str): Name of the column containing magnetic field data.
        signal_column (str): Name of the column containing signal data.
        minimum_field (float): Minimum magnetic field in Tesla.
        maximum_field (float): Maximum magnetic field in Tesla.
        points (int): Number of points for the FFT.
        background_function (callable): Function to model the background.
        window (Window, optional): Window function to apply to the data. Defaults to Window.HANN.
        subtract_inverse_field (bool, optional): Whether to subtract the inverse field. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing the FFT results.
    """
    
    data = df.copy()
    data = data[[field_column, signal_column]].dropna()
    data = data[(data[field_column] >= minimum_field) & (data[field_column] <= maximum_field)]
    data['inverse_field'] = 1 / data[field_column]

    if subtract_inverse_field:
        data['subtracted'] = data[signal_column] - background_function(data['inverse_field'])
    else:
        data['subtracted'] = data[signal_column] - background_function(data[field_column])

    result = qz_fft(
        data,
        x_column='inverse_field',
        y_column='subtracted',
        window=window,
        n=points,
    )

    return result