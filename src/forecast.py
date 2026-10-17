"""Step 7: forecast the next N days, one hour at a time (recursive forecasting).

For each future hour we: build features from everything known so far -> predict ->
append the prediction to history -> move to the next hour. Future weather is simulated
(in real life you would plug in a weather-forecast API).
"""
import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET, make_features
from src.generate_data import is_holiday, simulate_weather


def forecast_next(history: pd.DataFrame, model, horizon_hours: int = 168, weather_seed: int = 123) -> pd.Series:
    last = history.index[-1]
    future_index = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", name="timestamp")
    temp, hum = simulate_weather(future_index, seed=weather_seed)
    future = pd.DataFrame(
        {"temperature_c": temp, "humidity_pct": hum, "is_holiday": is_holiday(future_index), TARGET: np.nan},
        index=future_index,
    )

    buffer = history[["temperature_c", "humidity_pct", "is_holiday", TARGET]].tail(400).copy()
    preds = []
    for ts in future_index:
        buffer = pd.concat([buffer, future.loc[[ts]]])
        row = make_features(buffer).iloc[[-1]][FEATURES]
        yhat = float(model.predict(row)[0])
        buffer.loc[ts, TARGET] = yhat        # feed prediction back in as "history"
        preds.append(yhat)

    return pd.Series(preds, index=future_index, name="forecast_kwh")


def add_uncertainty(forecast: pd.Series, rmse: float) -> pd.DataFrame:
    """Simple ~95% band using the test RMSE (assumes roughly normal errors)."""
    return pd.DataFrame(
        {"forecast_kwh": forecast, "lower_95": forecast - 1.96 * rmse, "upper_95": forecast + 1.96 * rmse}
    )


def business_summary(fc: pd.DataFrame, tariff_peak: float = 9.0, tariff_offpeak: float = 6.0) -> dict:
    """Turn the forecast into a business insight: expected cost and the best hours to shift load."""
    hours = fc.index.hour
    peak = (hours >= 10) & (hours < 18)
    cost = float((fc["forecast_kwh"][peak].sum() * tariff_peak) + (fc["forecast_kwh"][~peak].sum() * tariff_offpeak))
    daily = fc["forecast_kwh"].resample("D").sum()
    return {
        "total_kwh_next_period": round(float(fc["forecast_kwh"].sum()), 1),
        "estimated_cost_inr": round(cost, 0),
        "peak_day": str(daily.idxmax().date()),
        "peak_hour_of_day": int(fc["forecast_kwh"].groupby(hours).mean().idxmax()),
    }
