import pandas as pd
from .smoothing import bin


def symmetrize(dfs, x_column, y_column, B_min, B_max, B_step):
	combined = pd.concat(dfs)
	combined = combined.sort_values(by=x_column)
	combined = bin(combined, x_column, -B_max, B_max, B_step)
	new_df = pd.DataFrame(data={x_column: combined[x_column], 'a': combined[y_column], 'b': combined[y_column].values[::-1]})
	new_df[y_column] = (new_df["a"]+new_df["b"])/2
	return new_df


def antisymmetrize(dfs, x_column, y_column, B_min, B_max, B_step):
	combined = pd.concat(dfs)
	combined = combined.sort_values(by=x_column)
	combined = bin(combined, x_column, -B_max, B_max, B_step)
	new_df = pd.DataFrame(data={x_column: combined[x_column], 'a': combined[y_column], 'b': combined[y_column].values[::-1]})
	new_df[y_column] = (new_df["a"]-new_df["b"])/2
	return new_df