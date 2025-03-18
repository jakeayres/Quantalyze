import pandas as pd
import numpy as np





def bin(df, x_column, minimum, maximum, width):
		bin_edges = np.arange(minimum-width/2, maximum + width/2 + width, width)
		bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
		df['bin'] = pd.cut(df[x_column], bins=bin_edges, include_lowest=True)
		binned = df.groupby('bin').agg('mean').reset_index()
		binned[x_column] = bin_midpoints
		binned = binned.drop('bin', axis=1)
		return binned



def window(df, y_column, window_size=5, window_type='triang'):
	return df[y_column].rolling(window=window_size, win_type=window_type, center=True).mean()


def savgol_filter(df, y_column, window_size=5, order=2):
	pass
	
	

