"""Step 4: feature engineering - turn the raw time series into ML-ready columns."""
import numpy as np
import pandas as pd

TARGET = "energy_kwh"

FEATURES = [
    # calendar
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos",
    # weather
    "temperature_c", "humidity_pct",
    # memory of the past (lags)
    "lag_1", "lag_24", "lag_168",
    # recent trend
    "roll_mean_24", "roll_std_24",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical, lag and rolling features.

    IMPORTANT: every lag / rolling feature only uses PAST values (shift >= 1),
    so the model never "peeks" at the answer (no data leakage).
    """
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # sin/cos lets the model know that hour 23 is next to hour 0
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)

    out["lag_1"] = out[TARGET].shift(1)        # 1 hour ago
    out["lag_24"] = out[TARGET].shift(24)      # same hour yesterday
    out["lag_168"] = out[TARGET].shift(168)    # same hour last week

    past = out[TARGET].shift(1)
    out["roll_mean_24"] = past.rolling(24).mean()
    out["roll_std_24"] = past.rolling(24).std()
    return out


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and drop the first rows that have no history yet."""
    return make_features(df).dropna(subset=FEATURES + [TARGET])
