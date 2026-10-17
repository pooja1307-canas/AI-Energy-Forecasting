"""Step 8: all graphs. Every function saves a PNG into images/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed - works on servers too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
IMG = Path("images")


def _save(name: str):
    IMG.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(IMG / name, dpi=150)
    plt.close()
    print(f"saved images/{name}")


def plot_overview(df: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(df.index, df["energy_kwh"], lw=0.4, color="#1f77b4")
    ax[0].set_ylabel("Energy (kWh)")
    ax[0].set_title("Hourly electricity consumption of the simulated building")
    ax[1].plot(df.index, df["temperature_c"], lw=0.4, color="#d62728")
    ax[1].set_ylabel("Temp (°C)")
    _save("01_energy_overview.png")


def plot_patterns(df: pd.DataFrame):
    pivot = df.assign(hour=df.index.hour, dow=df.index.dayofweek).pivot_table(
        values="energy_kwh", index="dow", columns="hour", aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.figure(figsize=(12, 4))
    sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Avg kWh"})
    plt.title("Average consumption by weekday and hour")
    plt.xlabel("Hour of day")
    _save("02_weekly_hourly_heatmap.png")


def plot_model_comparison(results: pd.DataFrame):
    plt.figure(figsize=(9, 4))
    sns.barplot(data=results, y="Model", x="RMSE", color="#4c72b0")
    plt.title("Model comparison on the held-out test period (lower RMSE = better)")
    _save("03_model_comparison.png")


def plot_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series, name: str, days: int = 7):
    end = y_true.index.max()
    sl = slice(end - pd.Timedelta(days=days), end)
    plt.figure(figsize=(12, 4.5))
    plt.plot(y_true.loc[sl], label="Actual", color="#1f77b4")
    plt.plot(y_pred.loc[sl], label=f"Predicted ({name})", color="#ff7f0e", ls="--")
    plt.ylabel("Energy (kWh)")
    plt.title(f"Actual vs predicted - last {days} days of the test set")
    plt.legend()
    _save("04_actual_vs_predicted.png")

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual kWh")
    plt.ylabel("Predicted kWh")
    plt.title("Predicted vs actual (test set)")
    plt.legend()
    _save("05_scatter_pred_vs_actual.png")


def plot_residuals(y_true: pd.Series, y_pred: pd.Series):
    err = y_true - y_pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(err, bins=40, kde=True, ax=ax[0], color="#55a868")
    ax[0].set_title("Prediction error distribution")
    ax[0].set_xlabel("Error (kWh)")
    ax[1].plot(err, lw=0.5)
    ax[1].axhline(0, color="r", lw=1)
    ax[1].set_title("Error over time")
    ax[1].tick_params(axis="x", rotation=30)
    _save("06_residuals.png")


def plot_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, name: str):
    """Permutation importance: how much does the error grow if we scramble one feature?
    Works for any model, so we can explain Linear, Forest and Boosting the same way."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    plt.figure(figsize=(8, 5))
    imp.plot.barh(color="#8172b3")
    plt.xlabel("Increase in RMSE (kWh) when feature is shuffled")
    plt.title(f"Feature importance - {name}")
    _save("07_feature_importance.png")


def plot_forecast(history: pd.Series, fc: pd.DataFrame, days_back: int = 7):
    hist = history.loc[history.index.max() - pd.Timedelta(days=days_back):]
    plt.figure(figsize=(12, 4.5))
    plt.plot(hist, label="Recent actual", color="#1f77b4")
    plt.plot(fc["forecast_kwh"], label="Forecast (next 7 days)", color="#ff7f0e")
    plt.fill_between(fc.index, fc["lower_95"], fc["upper_95"], color="#ff7f0e", alpha=0.2, label="95% band")
    plt.ylabel("Energy (kWh)")
    plt.title("7-day ahead electricity demand forecast")
    plt.legend()
    _save("08_future_forecast.png")
