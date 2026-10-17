"""Step 1 - Virtual simulation: generate a realistic hourly smart-building energy dataset.

We have no access to a real utility meter, so we SIMULATE one. The simulated building
behaves like a real commercial building:
  * a constant base load (servers, lifts, security lights)
  * occupancy-driven load: high on weekdays during office hours, low on weekends/holidays
  * weather-driven air-conditioning load: more kWh when it is hot
  * slow growth over time (more equipment) + random noise
  * data-quality problems: missing readings and sensor spikes (like real smart meters)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed-date public holidays (month, day) - India-style, easy to edit
HOLIDAYS = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}


def is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 if the date is a public holiday, else 0."""
    return np.array([1 if (d.month, d.day) in HOLIDAYS else 0 for d in index])


def simulate_weather(index: pd.DatetimeIndex, seed: int = 0):
    """Simulate hourly temperature (deg C) and humidity (%) with seasonal + daily cycles."""
    rng = np.random.default_rng(seed)
    doy = index.dayofyear.values
    hour = index.hour.values
    seasonal = 27 + 6 * np.sin(2 * np.pi * (doy - 100) / 365)   # hottest around July
    daily = 4 * np.sin(2 * np.pi * (hour - 9) / 24)             # peak at 3 PM
    temp = seasonal + daily + rng.normal(0, 1.2, len(index))
    humidity = np.clip(65 - 1.2 * (temp - 27) + rng.normal(0, 5, len(index)), 20, 100)
    return temp, humidity


def simulate_energy(index: pd.DatetimeIndex, temp: np.ndarray, holiday: np.ndarray, seed: int = 1):
    """Simulate hourly electricity consumption (kWh) of one commercial building."""
    rng = np.random.default_rng(seed)
    hour = index.hour.values
    dow = index.dayofweek.values

    occupancy = np.exp(-((hour - 13) / 4.5) ** 2)               # bell curve centred on 1 PM
    day_factor = np.where((dow < 5) & (holiday == 0), 1.0, 0.35)  # weekend/holiday = quiet

    base_load = 40.0
    people_load = 55.0 * occupancy * day_factor
    ac_load = 3.2 * np.maximum(temp - 24, 0) * (0.4 + 0.6 * occupancy * day_factor)
    growth = 1 + 0.02 * (np.arange(len(index)) / (24 * 365))
    noise = rng.lognormal(0, 0.04, len(index))

    return (base_load + people_load + ac_load) * growth * noise


def generate_dataset(start="2023-01-01", end="2024-12-31 23:00", seed=42) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h", name="timestamp")
    temp, humidity = simulate_weather(index, seed=seed)
    holiday = is_holiday(index)
    energy = simulate_energy(index, temp, holiday, seed=seed + 1)

    df = pd.DataFrame(
        {
            "temperature_c": temp.round(2),
            "humidity_pct": humidity.round(1),
            "is_holiday": holiday,
            "energy_kwh": energy.round(3),
        },
        index=index,
    )

    # Inject realistic data problems so the cleaning step has real work to do
    rng = np.random.default_rng(seed + 2)
    n = len(df)
    missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)    # 1% missing readings
    spike_idx = rng.choice(n, size=int(0.002 * n), replace=False)     # 0.2% sensor spikes
    df.iloc[spike_idx, df.columns.get_loc("energy_kwh")] *= 3.5
    df.iloc[missing_idx, df.columns.get_loc("energy_kwh")] = np.nan
    return df


def main(out_path: str = "data/raw/building_energy.csv"):
    df = generate_dataset()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved {len(df):,} hourly rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
