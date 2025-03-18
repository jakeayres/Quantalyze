


def calculate_resistance(df, voltage, current):
    """
    Calculate the electrical resistance using voltage and current.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    voltage (str): The name of the column in the dataframe that contains the voltage values.
    current (str or float): The name of the column in the dataframe that contains the current values if it's a string,
                            or a constant current value if it's a float.

    Returns:
    pandas.Series: A series containing the calculated resistance values.
    """
    if isinstance(current, str):
        resistance = df[voltage] / df[current]
    else:
        resistance = df[voltage] / current
    return resistance
 

def calculate_resistivity(df, length, width, thickness, resistance=None, voltage=None, current=None):
    """
    Calculate the electrical resistivity using either resistance or voltage and current, along with length, width, and thickness.
    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    resistance (str, optional): The name of the column in the dataframe that contains the resistance values.
    voltage (str, optional): The name of the column in the dataframe that contains the voltage values.
    current (str or float, optional): The name of the column in the dataframe that contains the current values if it's a string,
                                      or a constant current value if it's a float.
    length (float): The length of the material.
    width (float): The width of the material.
    thickness (float): The thickness of the material.
    Returns:
    pandas.Series: A series containing the calculated resistivity values.
    """
    if resistance is None:
        if voltage is not None and current is not None:
            resistance = calculate_resistance(df, voltage, current)
        else:
            raise ValueError("Either resistance or both voltage and current must be provided.")
    else:
        resistance = df[resistance]
    
    resistivity = resistance * (width * thickness) / length
    return resistivity


