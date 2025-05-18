import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


class ChebychevFit:

    def __init__(self, cheb_fit_R_to_T, cheb_fit_T_to_R):
        self.cheb_fit_R_to_T = cheb_fit_R_to_T
        self.cheb_fit_T_to_R = cheb_fit_T_to_R


    def resistance_to_temperature(self, resistance):
        """
        Convert resistance to temperature using the Chebyshev fit.
        Parameters:
            resistance (float): Resistance in ohms.
        Returns:
            float: Temperature in Kelvin.
        """
        log_R = np.log10(resistance)
        log_T = self.cheb_fit_R_to_T(log_R)
        return 10**log_T
    

    def temperature_to_resistance(self, temperature):
        """
        Convert temperature to resistance using the Chebyshev fit.
        Parameters:
            temperature (float): Temperature in Kelvin.
        Returns:
            float: Resistance in ohms.
        """
        log_T = np.log10(temperature)
        log_R = self.cheb_fit_T_to_R(log_T)
        return 10**log_R


def fit_chebyshev_polynomial(df, resistance_column='resistance', temperature_column='temperature', order=7):
    """
    Fits Chebyshev polynomials to log-log data for resistance vs. temperature.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing 'resistance' and 'temperature' columns.
        resistance_column (str): Name of the column containing resistance data.
        temperature_column (str): Name of the column containing temperature data.
        order (int): Order of the Chebyshev polynomial.
    
    Returns:
        tuple: (Chebyshev object for R->T, Chebyshev object for T->R)
    """
    log_R = np.log10(df[resistance_column])
    log_T = np.log10(df[temperature_column])
    
    cheb_fit_R_to_T = Chebyshev.fit(log_R, log_T, order)
    cheb_fit_T_to_R = Chebyshev.fit(log_T, log_R, order)
    
    return ChebychevFit(cheb_fit_R_to_T, cheb_fit_T_to_R)

