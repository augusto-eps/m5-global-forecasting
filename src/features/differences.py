# Rate of change features

import pandas as pd

def add_differences(df, group_col="id"):
    df = df.sort_values("date")
    df["sales_diff"] = df.groupby(group_col)["sales"].diff()
    return df

def add_category_mean_diff(df):
    df["cat_mean_diff"] = (
        df.groupby(["cat_id", "date"])["sales_diff"]
          .transform("mean")
    )
    return df
