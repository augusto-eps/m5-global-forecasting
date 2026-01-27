# This script contains dependencies to preprocess M4 data when loading (load_m4.py)

import pandas as pd
from typing import Tuple, Union
import plotly.express as px

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

def temporal_train_test_split(
    df: pd.DataFrame,
    id_col: str = "M4id",
    time_col: str = "time_idx",
    value_col: str = "value",
    test_size: Union[float, int] = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets, preserving temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format time series dataframe.
    id_col : str
        Column identifying each time series.
    time_col : str
        Time ordering column.
    test_size : float or int
        If float (<1), proportion of observations to use for test.
        If int (>=1), absolute number of observations per series for test.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """

    if isinstance(test_size, float):
        if not (0 < test_size < 1):
            raise ValueError("test_size as float must be between 0 and 1")
        mode = "proportion"

    elif isinstance(test_size, int):
        if test_size < 1:
            raise ValueError("test_size as int must be >= 1")
        mode = "absolute"

    else:
        raise TypeError("test_size must be float (<1) or int (>=1)")

    df_sorted = df.sort_values([id_col, time_col])

    train_parts = []
    test_parts = []

    for series_id, group in df_sorted.groupby(id_col):

        n_obs = len(group)

        if mode == "proportion":
            n_test = int(n_obs*test_size)

        elif mode == "absolute":
            n_test = test_size

        if n_test > n_obs:
            raise ValueError(
                f"Test size {n_test} is greater than the number of observations {n_obs}"
            )

        split_point = n_obs - n_test
        train_parts.append(group.iloc[:split_point])
        test_parts.append(group.iloc[split_point:])

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    return train_df, test_df

def plot_sampled_series(
    df: pd.DataFrame,
    category: str,
    n_series: int = 10,
    random_state: int = 42,
):
    # Filter category
    df_cat = df[df["category"] == category]

    # Sample series IDs
    sampled_ids = (
        df_cat["M4id"]
        .drop_duplicates()
        .sample(n=min(n_series, df_cat["M4id"].nunique()),
                random_state=random_state)
    )

    df_plot = df_cat[df_cat["M4id"].isin(sampled_ids)]

    fig = px.line(
        df_plot,
        x="time_idx",
        y="value",
        color="M4id",
        title=f"Sampled time series from category: {category}",
        labels={
            "time_idx": "Time",
            "value": "Value",
            "M4id": "Series ID",
        },
    )

    fig.update_layout(
        template="plotly_white",
        legend_title_text="Series",
        hovermode="x unified",
    )

    fig.show()

