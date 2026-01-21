# Script designed to read raw M5 data and output processed table

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

def load_raw_data():
    sales = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(RAW_DATA_DIR / "calendar.csv")
    prices = pd.read_csv(RAW_DATA_DIR / "sell_prices.csv")
    return sales, calendar, prices

def melt_sales(sales):
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_vars = [c for c in sales.columns if c.startswith("d_")]

    return sales.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="d",
        value_name="sales"
    )

def main():
    sales, calendar, prices = load_raw_data()
    sales_long = melt_sales(sales)

    df = (
        sales_long
        .merge(calendar, on="d", how="left")
        .merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    )

    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    df.to_parquet(PROCESSED_DATA_DIR / "m5_long.parquet")

if __name__ == "__main__":
    main()
    