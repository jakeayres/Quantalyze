import pandas as pd


def calculate_resistance(df, voltage, current) -> pd.Series:
    """
    Calculate the electrical resistance using voltage and current.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        voltage (str): The name of the column in the dataframe that contains the voltage values.
        current (str or float): The name of the column in the dataframe that contains the current values if it's a string,
            or a constant current value if it's a float.

    Returns:
        pandas.Series: A series containing the calculated resistance values.
        
    Examples:
        >>> import quantalize as qz
        >>> df['resistance'] = qz.calculate_resistance(
        >>>     df, 
        >>>     voltage='voltage_col', 
        >>>     current='current_col'
        >>> )
    """
    if isinstance(current, str):
        resistance = df[voltage] / df[current]
    else:
        resistance = df[voltage] / current
    return resistance


def calculate_resistivity(df, length=None, width=None, thickness=None, resistance=None, voltage=None, current=None, geometric_factor=None) -> pd.Series:
    """
    Calculate the electrical resistivity using either resistance or voltage and current, along with length, width, and thickness,
    or a provided geometric factor.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        resistance (str, optional): The name of the column in the dataframe that contains the resistance values.
        voltage (str, optional): The name of the column in the dataframe that contains the voltage values.
        current (str or float, optional): The name of the column in the dataframe that contains the current values if it's a string,
            or a constant current value if it's a float.
        length (float, optional): The length of the material.
        width (float, optional): The width of the material.
        thickness (float, optional): The thickness of the material.
        geometric_factor (float, optional): The pre-calculated geometric factor (width * thickness / length).

    Returns:
        pandas.Series: A series containing the calculated resistivity values.

    Raises:
        ValueError: If neither resistance nor both voltage and current are provided.
        ValueError: If neither geometric_factor nor all of length, width, and thickness are provided.
        
    Examples:
        >>> import quantalize as qz
        >>> df['resistivity'] = qz.calculate_resistivity(
        >>>     df, 
        >>>     resistance='resistance_col', 
        >>>     voltage='voltage_col', 
        >>>     current='current_col', 
        >>>     length=100e-6, 
        >>>     width=300e-6, 
        >>>     thickness=11e-6
        >>> )
    """
    if resistance is None:
        if voltage is not None and current is not None:
            resistance = calculate_resistance(df, voltage, current)
        else:
            raise ValueError("Either resistance or both voltage and current must be provided.")
    else:
        resistance = df[resistance]

    if geometric_factor is None:
        if length is not None and width is not None and thickness is not None:
            geometric_factor = (width * thickness) / length
        else:
            raise ValueError("Either geometric_factor or all of length, width, and thickness must be provided.")

    resistivity = resistance * geometric_factor
    return resistivity


def calculate_hall_resistance(df, hall_voltage, current) -> pd.Series:
    """
    Calculate the Hall resistance using the Hall voltage and current.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        hall_voltage (str): The name of the column in the dataframe that contains the Hall voltage values.
        current (str or float): The name of the column in the dataframe that contains the current values if it's a string,
            or a constant current value if it's a float.

    Returns:
        pandas.Series: A series containing the calculated Hall resistance values.
        
    Examples:
        >>> import quantalize as qz
        >>> df['hall_resistance'] = qz.calculate_hall_resistance(
        >>>     df, 
        >>>     hall_voltage='hall_voltage_col', 
        >>>     current='current_col'
        >>> )
    """
    if isinstance(current, str):
        hall_resistance = df[hall_voltage] / df[current]
    else:
        hall_resistance = df[hall_voltage] / current
    return hall_resistance


def calculate_hall_resistivity(df, hall_resistance=None, hall_voltage=None, current=None, thickness=None) -> pd.Series:
    """
    Calculate the Hall resistivity using either Hall resistance or Hall voltage and current, along with thickness.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        hall_resistance (str, optional): The name of the column in the dataframe that contains the Hall resistance values.
        hall_voltage (str, optional): The name of the column in the dataframe that contains the Hall voltage values.
        current (str or float, optional): The name of the column in the dataframe that contains the current values if it's a string,
            or a constant current value if it's a float.
        thickness (float, optional): The thickness of the material.

    Returns:
        pandas.Series: A series containing the calculated Hall resistivity values.

    Raises:
        ValueError: If neither hall_resistance nor both hall_voltage and current are provided.
        ValueError: If thickness is not provided.
        
    Examples:
        >>> import quantalize as qz
        >>> df['hall_resistivity'] = qz.calculate_hall_resistivity(
        >>>     df, 
        >>>     hall_resistance='hall_resistance_col', 
        >>>     hall_voltage='hall_voltage_col', 
        >>>     current='current_col', 
        >>>     thickness=11e-6
        >>> )
    """
    if hall_resistance is None:
        if hall_voltage is not None and current is not None:
            hall_resistance = calculate_hall_resistance(df, hall_voltage, current)
        else:
            raise ValueError("Either hall_resistance or both hall_voltage and current must be provided.")
    else:
        hall_resistance = df[hall_resistance]

    if thickness is None:
        raise ValueError("Thickness must be provided.")

    hall_resistivity = hall_resistance * thickness
    return hall_resistivity
