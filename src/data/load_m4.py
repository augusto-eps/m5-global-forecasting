# Script designed to read raw M4 data and output processed table

import logging
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

N_SERIES_PER_CATEGORY = 100
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_m4_info() -> pd.DataFrame:

    """Load M4 metadata."""
    
    path = RAW_DATA_DIR / "M4-info.csv"
    logger.info(f"Reading metadata from {path}")

    info = pd.read_csv(path)
    return info


def sample_m4_ids_by_category(
    info_daily: pd.DataFrame,
    n_per_category: int,
    random_state: int = 42
) -> list[str]:
    
    """Sample M4 series IDs per category."""

    logger.info(
        f"Sampling {n_per_category} series per category "
        f"from Daily M4 metadata"
    )

    sampled = (
        info_daily[["M4id", "category"]]
        .groupby("category", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), n_per_category),
                random_state=random_state
            )
        )
    )

    sampled_ids = sampled["M4id"].tolist()

    logger.info(f"Sampled {len(sampled_ids)} total series")
    return sampled_ids


def load_m4_daily_wide(sampled_ids: list[str] | None = None) -> pd.DataFrame:

    """Load M4 Daily training data in wide format."""

    path = RAW_DATA_DIR / "Daily-train.csv"
    logger.info(f"Reading Daily data from {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={"V1": "M4id"})

    if sampled_ids is not None:
        df = df[df["M4id"].isin(sampled_ids)]
        logger.info(f"Filtered to {df.shape[0]} sampled series")

    return df


def melt_daily_to_long(daily: pd.DataFrame) -> pd.DataFrame:

    """Convert wide Daily M4 data to long format."""

    logger.info("Melting Daily data to long format")

    value_cols = [c for c in daily.columns if c.startswith("V")]

    long_df = daily.melt(
        id_vars=["M4id"],
        value_vars=value_cols,
        var_name="t",
        value_name="value"
    )

    # Extract numeric time index from column name (V2 → 0, V3 → 1, ...)
    long_df["time_idx"] = long_df["t"].str[1:].astype(int) - 2
    long_df = long_df.drop(columns=["t"])

    logger.info(f"Long data shape: {long_df.shape}")
    return long_df


def build_processed_daily_table(
    n_per_category: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    
    """Build long-format Daily M4 table with metadata and sampling."""

    info = load_m4_info()
    info_daily = info[info["SP"] == "Daily"]

    sampled_ids = sample_m4_ids_by_category(
        info_daily,
        n_per_category=n_per_category,
        random_state=random_state
    )

    daily_wide = load_m4_daily_wide(sampled_ids=sampled_ids)
    long_df = melt_daily_to_long(daily_wide)

    long_df = long_df.merge(
        info_daily[["M4id", "category", "SP"]],
        on="M4id",
        how="left"
    )

    return long_df


def main():
    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df = build_processed_daily_table(
            n_per_category=N_SERIES_PER_CATEGORY,
            random_state=RANDOM_STATE
        )

        output_path = PROCESSED_DATA_DIR / "m4_daily_sampled.parquet"
        df.to_parquet(output_path, index=False)

        logger.info(
            f"Saved sampled Daily M4 data to {output_path} "
            f"({df['M4id'].nunique()} series, {df.shape[0]} rows)"
        )

    except Exception:
        logger.exception("Daily M4 ingestion pipeline failed")
        raise


if __name__ == "__main__":
    main()
