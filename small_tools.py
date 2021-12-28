import pandas as pd
import numpy as np


def convert_one_dimension(series):
    # Converts pandas column (series) with items as lists to 1D,
    # so we can do calculations like unique, value_counts, etc.
    final_list = []
    for lst in series:
        if isinstance(lst, (list, pd.core.series.Series, np.ndarray)):
            for x in lst:
                if x not in final_list:
                    final_list.append(x)
    return pd.Series(final_list)
