"""Steps 2-3: load the raw CSV and clean it (missing values + outliers)."""
from pathlib import Path

import pandas as pd


def load_raw(path: str = "data/raw/building_energy.csv") -> pd.DataFrame:
    """Read the CSV, parse timestamps, sort, and force a regular hourly index."""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    df = df.asfreq("h")  # any skipped hour becomes an explicit NaN row
    return df


def clean(df: pd.DataFrame, spike_ratio: float = 2.0) -> pd.DataFrame:
    """Remove sensor spikes and fill gaps.

    A reading is a spike if it is more than `spike_ratio` x the median of its 7 nearest
    hours (a LOCAL window, so normal daily peaks are not mistaken for faults). Spikes are
    turned into NaN, then ALL NaNs are filled by time-based linear interpolation
    (a sensible choice for smooth meter data).
    """
    df = df.copy()
    report = {"missing_before": int(df["energy_kwh"].isna().sum())}

    rolling_median = df["energy_kwh"].rolling(7, center=True, min_periods=3).median()
    spikes = df["energy_kwh"] > spike_ratio * rolling_median
    report["spikes_removed"] = int(spikes.sum())
    df.loc[spikes, "energy_kwh"] = float("nan")

    df = df.interpolate(method="time", limit_direction="both")
    report["missing_after"] = int(df["energy_kwh"].isna().sum())

    print("Cleaning report:", report)
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/clean_energy.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
