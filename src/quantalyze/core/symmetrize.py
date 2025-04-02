import pandas as pd
from .smoothing import bin


def symmetrize(dfs, x_column, y_column, minimum, maximum, step) -> pd.DataFrame:
    """
    Symmetrizes the given dataframes by combining them, binning the onto evenly spaced values of x_column
    between minimum and maximum, and then averaging the y_column values with their reversed counterparts.

    Args:
        dfs (list[pd.DataFrame]): List of dataframes to be symmetrized.
        x_column (str): The name of the column to be used for filtering and sorting.
        y_column (str): The name of the column to be symmetrized.
        minimum (float): The minimum value for filtering the x_column.
        maximum (float): The maximum value for filtering the x_column.
        step (float): The step size for binning the x_column values.

    Returns:
        pd.DataFrame: A new dataframe with symmetrized y_column values.
    """
    dfs = [df[(df[x_column] >= minimum) & (df[x_column] <= maximum)] for df in dfs]
    combined = pd.concat(dfs)
    combined = combined.sort_values(by=x_column)
    combined = bin(combined, x_column, -maximum, maximum, step)
    new_df = pd.DataFrame(data={x_column: combined[x_column], 'a': combined[y_column], 'b': combined[y_column].values[::-1]})
    new_df[y_column] = (new_df["a"]+new_df["b"])/2
    new_df = new_df.drop(columns=['a', 'b'])
    return new_df


def antisymmetrize(dfs, x_column, y_column, minimum, maximum, step) -> pd.DataFrame:
    """
    Antisymmetrizes the given dataframes by combining them, binning the onto evenly spaced values of x_column
    between minimum and maximum, and then taking the difference between y_column values and their reversed counterparts.

    Args:
        dfs (list[pd.DataFrame]): List of dataframes to be antisymmetrized.
        x_column (str): The name of the column representing the x-axis.
        y_column (str): The name of the column representing the y-axis.
        minimum (float): The minimum value of the x-axis range to consider.
        maximum (float): The maximum value of the x-axis range to consider.
        step (float): The step size for binning the x-axis values.

    Returns:
        pd.DataFrame: A new dataframe with antisymmetrized y-values.
    """
    dfs = [df[(df[x_column] >= minimum) & (df[x_column] <= maximum)] for df in dfs]
    combined = pd.concat(dfs)
    combined = combined.sort_values(by=x_column)
    combined = bin(combined, x_column, -maximum, maximum, step)
    new_df = pd.DataFrame(data={x_column: combined[x_column], 'a': combined[y_column], 'b': combined[y_column].values[::-1]})
    new_df[y_column] = (new_df["a"]-new_df["b"])/2
    new_df = new_df.drop(columns=['a', 'b'])
    return new_df