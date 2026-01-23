# This script contains dependencies to preprocess M4 data when loading (load_m4.py)

import pandas as pd

def drop_na(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Drop rows with NA in the value column."""
    return df.dropna(subset=[value_col])

def get_valid_series_ids(
    df: pd.DataFrame,
    grouping_col: str = "M4id",
    value_col: str = 'value',
    time_col: str = "time_idx",
    required_days: int = 731
) -> pd.Index:
    """Return series IDs with at least `required_days` of non-missing data from the start."""
    df_initial = df[df[time_col] < required_days]
    counts = df_initial.groupby(grouping_col)[value_col].count()
    return counts[counts >= required_days].index

