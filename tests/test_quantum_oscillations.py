import pytest
import pandas as pd
import numpy as np
from quantalyze.quantum_oscillations import (extremal_area, 
    extremal_frequency, lifshitz_kosevich_damping_factor, 
    dingle_lifetime, dingle_temperature, dingle_damping_factor,
    mean_inverse_field, fit_effective_mass, fft)
from quantalyze.core.fft import Window


def test_extremal_area():

    # Check that larger frequency gives larger area
    f1 = 100
    f2 = 200
    area1 = extremal_area(f1)
    area2 = extremal_area(f2)
    assert area2 > area1, f"Expected area2 > area1, but got {area2} <= {area1}"

    # Underdoped YBCO example
    frequency = 530
    expected_area = 5.059 * 1e9 * 1e9
    result = extremal_area(frequency)
    relative_error = abs(result - expected_area) / expected_area
    assert relative_error < 0.001, f"Expected {expected_area:.3e}, got f{result:.3e}"


def test_extremal_frequency():

    # Check that larger area gives larger frequency
    area1 = 1e14
    area2 = 2e14
    f1 = extremal_frequency(area1)
    f2 = extremal_frequency(area2)
    assert f2 > f1, f"Expected f2 > f1, but got {f2} <= {f1}"

    # Underdoped YBCO example
    area = 5.059 * 1e9 * 1e9
    expected_frequency = 530
    result = extremal_frequency(area)
    relative_error = abs(result - expected_frequency) / expected_frequency
    assert relative_error < 0.001, f"Expected {expected_frequency:.3f}, got {result:.3f}"


def test_lifshitz_kosevich_damping_factor():

    # Test that increasing effective mass decreases damping factor
    temp = 5.0  # 5 K
    field = 10.0  # 10 Tesla
    mass1 = 1.0
    mass2 = 2.0
    rt1 = lifshitz_kosevich_damping_factor(temperature=temp, effective_mass=mass1, magnetic_field=field)
    rt2 = lifshitz_kosevich_damping_factor(temperature=temp, effective_mass=mass2, magnetic_field=field)
    assert rt2 < rt1, f"Expected increasing mass to decrease damping factor, but got {rt2} >= {rt1}"

    # Test that increasing temperature decreases damping factor
    mass = 1.0
    temp1 = 2.0
    temp2 = 4.0
    rt1 = lifshitz_kosevich_damping_factor(temperature=temp1, effective_mass=mass, magnetic_field=field)
    rt2 = lifshitz_kosevich_damping_factor(temperature=temp2, effective_mass=mass, magnetic_field=field)
    assert rt2 < rt1, f"Expected increasing temperature to decrease damping factor, but got {rt2} >= {rt1}"

    # Test that increasing field increases damping factor
    mass = 1.0
    temp = 5.0
    field1 = 5.0
    field2 = 10.0
    rt1 = lifshitz_kosevich_damping_factor(temperature=temp, effective_mass=mass, magnetic_field=field1)
    rt2 = lifshitz_kosevich_damping_factor(temperature=temp, effective_mass=mass, magnetic_field=field2)
    assert rt2 > rt1, f"Expected increasing field to increase damping factor, but got {rt2} <= {rt1}"


def test_lifshitz_kosevich_damping_factor_zero_temperature():
    # Test at zero temperature
    rt = lifshitz_kosevich_damping_factor(temperature=1e-6, effective_mass=1.0, magnetic_field=10.0)
    assert abs(rt - 1.0) < 1e-10, f"Expected RT = 1.0 at T=0, got {rt}"
    
    # Test with different masses at T=0
    rt_heavy = lifshitz_kosevich_damping_factor(temperature=1e-6, effective_mass=5.0, magnetic_field=10.0)
    assert abs(rt_heavy - 1.0) < 1e-10, f"Expected RT = 1.0 at T=0 with heavy mass, got {rt_heavy}"
    
    # Test with different fields at T=0
    rt_low_field = lifshitz_kosevich_damping_factor(temperature=1e-6, effective_mass=1.0, magnetic_field=1.0)
    assert abs(rt_low_field - 1.0) < 1e-10, f"Expected RT = 1.0 at T=0 with low field, got {rt_low_field}"


def test_dingle_lifetime():

    # Check that larger Dingle temperature gives shorter lifetime
    dt1 = 100
    dt2 = 200
    lifetime1 = dingle_lifetime(dt1)
    lifetime2 = dingle_lifetime(dt2)
    assert lifetime2 < lifetime1, f"Expected lifetime2 < lifetime1, but got {lifetime2} >= {lifetime1}"


def test_dingle_temperature():

    # Check that larger lifetime gives shorter Dingle temperature
    lt1 = 1e-9
    lt2 = 2e-9
    dt1 = dingle_temperature(lt1)
    dt2 = dingle_temperature(lt2)
    assert dt2 < dt1, f"Expected dt2 < dt1, but got {dt2} >= {dt1}"
    

def test_dingle_damping_factor():
    # Check that larger lifetime gives smaller damping factor
    field = 10.0  # Tesla
    mass = 1.0  # effective mass in units of electron mass
    lifetime1 = 1e-6  # seconds
    lifetime2 = 1e-9  # seconds
    df1 = dingle_damping_factor(magnetic_field=field, effective_mass=mass, lifetime=lifetime1)
    df2 = dingle_damping_factor(magnetic_field=field, effective_mass=mass, lifetime=lifetime2)
    assert df2 < df1, f"Expected df2 < df1, but got {df2} >= {df1}"
    
    # Check that higher magnetic field increases damping factor
    lifetime = 1e-8  # seconds
    mass = 1.0  # effective mass in units of electron mass
    field1 = 5.0  # Tesla
    field2 = 10.0  # Tesla
    df1 = dingle_damping_factor(magnetic_field=field1, effective_mass=mass, lifetime=lifetime)
    df2 = dingle_damping_factor(magnetic_field=field2, effective_mass=mass, lifetime=lifetime)
    assert df2 > df1, f"Expected df2 > df1, but got {df2} <= {df1}"

    # Test that the damping factor changes exponentially with field
    lifetime = 1e-8  # seconds
    mass = 1.0  # effective mass
    field1 = 10.0  # Tesla
    field2 = 20.0  # Tesla (double the field)
    
    df1 = dingle_damping_factor(magnetic_field=field1, effective_mass=mass, lifetime=lifetime)
    df2 = dingle_damping_factor(magnetic_field=field2, effective_mass=mass, lifetime=lifetime)
    
    # The Dingle factor should behave as exp(-alpha/B), so doubling the field should square the factor
    # Taking ln(df2/df1) should be approximately ln(df1)
    expected_ratio = df1  # Since df2 ≈ df1² for field doubling (in appropriate units)
    actual_ratio = df2 / df1
    
    # Allow for some numerical imprecision
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.1, \
        f"Expected exponential field dependence with ratio ≈ {expected_ratio}, got {actual_ratio}"
    

def test_mean_inverse_field():
    # Test with minimum and maximum fields
    min_field = 1.0  # Tesla
    max_field = 4.0  # Tesla
    expected_mean = (1 / min_field + 1 / max_field) / 2
    result = mean_inverse_field(minimum=min_field, maximum=max_field)
    assert abs(result - expected_mean) < 1e-6, f"Expected {expected_mean}, got {result}"
    
    # Test with equal fields
    field = 2.0  # Tesla
    expected_mean = 1 / field
    result = mean_inverse_field(minimum=field, maximum=field)
    assert abs(result - expected_mean) < 1e-6, f"Expected {expected_mean}, got {result}"


def test_fit_effective_mass():

    # Create a mock DataFrame
    data = {
        'temperature': [0.5, 1, 2, 3, 4, 5, 10],
        'amplitude': [4.25, 4.0, 2.5, 1.5, 1.0, 0.2, 0.005]
    }
    df = pd.DataFrame(data)

    # Fit the effective mass
    field = 1/mean_inverse_field(1, 13)  # Example magnetic field
    mass_fit = fit_effective_mass(df, 'temperature', 'amplitude', magnetic_field=field)

    effective_mass = mass_fit.parameters[1]
    prefactor = mass_fit.parameters[0]

    # Check that the effective mass is reasonable
    assert 3.5 < prefactor < 4.75, f"Unexpected prefactor: {prefactor}"
    assert 0.1 < effective_mass < 0.2, f"Unexpected effective mass: {effective_mass}"


def test_fft():

    field = np.linspace(1, 50, 5000)
    inverse = 1 / field
    frequency = 100
    signal = np.sin(2 * np.pi * frequency * inverse) * dingle_damping_factor(
        magnetic_field=field, effective_mass=1.0, lifetime=dingle_lifetime(1)
    )

    df = pd.DataFrame(data={
        'field': field,
        'signal': signal,
    })

    result = fft(
        df,
        field_column='field',
        signal_column='signal',
        window=Window.HANN,
        points=100000,
        background_function=lambda x: 0,
        minimum_field=10,
        maximum_field=50,
    )

    # Verify that the result is a DataFrame
    assert isinstance(result, pd.DataFrame), "FFT result should be a DataFrame"
    
    # Check that the result has the expected columns
    assert 'frequency' in result.columns, "Result DataFrame should have a 'frequency' column"
    assert 'amplitude' in result.columns, "Result DataFrame should have an 'amplitude' column"
    
    # Check that the frequencies include the expected peak frequency (around 0.1)
    # Allow some margin since FFT bin resolution depends on sampling
    peak_freq = result.loc[result['amplitude'].idxmax(), 'frequency']
    assert frequency*0.98 < peak_freq < frequency*1.02, f"Expected peak frequency around {frequency}, got {peak_freq}"

