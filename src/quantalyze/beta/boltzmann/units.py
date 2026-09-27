"""Unit converters into (and out of) the SI units used by the Boltzmann code.

Band-structure inputs usually come in Å⁻¹, eV and eV·Å. Convert them here
before building a contour: the solver itself only accepts SI (m⁻¹, m/s, s, J, kg).
Every converter works on floats, NumPy arrays and pandas Series.

Examples:
    >>> from quantalyze.beta import boltzmann as bz
    >>> bz.units.per_angstrom_to_per_meter(0.7)
    7000000000.0
"""
from __future__ import annotations

from ...core.constants import ELECTRON_MASS, ELEMENTARY_CHARGE, HBAR

ANGSTROM = 1e-10  # m
PER_ANGSTROM = 1e10  # Å per m; exact in floating point, unlike 1e-10, so divide by Å with it
PICOSECOND = 1e-12  # s


def per_angstrom_to_per_meter(k):
    """Convert a wavevector from Å⁻¹ to m⁻¹.

    Args:
        k: Wavevector (Å⁻¹).

    Returns:
        Wavevector (m⁻¹).
    """
    return k * PER_ANGSTROM


def per_meter_to_per_angstrom(k):
    """Convert a wavevector from m⁻¹ to Å⁻¹.

    Args:
        k: Wavevector (m⁻¹).

    Returns:
        Wavevector (Å⁻¹).
    """
    return k * ANGSTROM


def angstrom_to_meter(length):
    """Convert a length from Å to m.

    Args:
        length: Length (Å).

    Returns:
        Length (m).
    """
    return length * ANGSTROM


def meter_to_angstrom(length):
    """Convert a length from m to Å.

    Args:
        length: Length (m).

    Returns:
        Length (Å).
    """
    return length * PER_ANGSTROM


def ev_to_joule(energy):
    """Convert an energy from eV to J.

    Args:
        energy: Energy (eV).

    Returns:
        Energy (J).
    """
    return energy * ELEMENTARY_CHARGE


def joule_to_ev(energy):
    """Convert an energy from J to eV.

    Args:
        energy: Energy (J).

    Returns:
        Energy (eV).
    """
    return energy / ELEMENTARY_CHARGE


def ev_angstrom_to_meter_per_second(gradient):
    """Convert a band gradient dε/dk in eV·Å into the group velocity v = (1/ħ) dε/dk.

    Args:
        gradient: Energy gradient dε/dk (eV·Å).

    Returns:
        Group velocity (m/s). 1 eV·Å is about 1.519×10⁵ m/s.
    """
    return gradient * (ELEMENTARY_CHARGE * ANGSTROM / HBAR)


def meter_per_second_to_ev_angstrom(velocity):
    """Convert a group velocity into the band gradient ħv in eV·Å.

    Args:
        velocity: Group velocity (m/s).

    Returns:
        Energy gradient dε/dk = ħv (eV·Å).
    """
    return velocity / (ELEMENTARY_CHARGE * ANGSTROM / HBAR)


def electron_mass_to_kilogram(mass):
    """Convert a mass from units of the free-electron mass m_e to kg.

    Args:
        mass: Mass (m_e).

    Returns:
        Mass (kg).
    """
    return mass * ELECTRON_MASS


def kilogram_to_electron_mass(mass):
    """Convert a mass from kg to units of the free-electron mass m_e.

    Args:
        mass: Mass (kg).

    Returns:
        Mass (m_e).
    """
    return mass / ELECTRON_MASS


def picosecond_to_second(time):
    """Convert a time from ps to s.

    Args:
        time: Time (ps).

    Returns:
        Time (s).
    """
    return time * PICOSECOND


def second_to_picosecond(time):
    """Convert a time from s to ps.

    Args:
        time: Time (s).

    Returns:
        Time (ps).
    """
    return time / PICOSECOND
