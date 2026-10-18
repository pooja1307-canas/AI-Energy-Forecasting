"""Streamlit dashboard.  Run:  streamlit run app.py   (opens http://localhost:8501)

Run `python main.py` once first so the data, model and outputs exist.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.forecast import add_uncertainty, business_summary, forecast_next

st.set_page_config(page_title="Energy Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ AI-Powered Energy Consumption Forecasting")
st.caption("Simulated smart-building data · scikit-learn · hourly forecasts")

CLEAN = Path("data/processed/clean_energy.csv")
MODEL = Path("models/best_model.joblib")
METRICS = Path("outputs/metrics.csv")

if not (CLEAN.exists() and MODEL.exists() and METRICS.exists()):
    st.error("Outputs not found. Run `python main.py` first, then restart this app.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN, parse_dates=["timestamp"], index_col="timestamp")


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


data = load_data()
bundle = load_model()
metrics = pd.read_csv(METRICS)
best_rmse = float(metrics.loc[metrics["Model"] == bundle["name"], "RMSE"].iloc[0])

# ---- Sidebar controls ----
st.sidebar.header("Controls")
horizon_days = st.sidebar.slider("Forecast horizon (days)", 1, 14, 7)
tariff_peak = st.sidebar.number_input("Peak tariff 10:00-18:00 (₹/kWh)", 1.0, 30.0, 9.0)
tariff_off = st.sidebar.number_input("Off-peak tariff (₹/kWh)", 1.0, 30.0, 6.0)
history_days = st.sidebar.slider("Days of history to show", 3, 30, 7)

# ---- Forecast ----
with st.spinner("Forecasting..."):
    fc = add_uncertainty(forecast_next(data, bundle["model"], horizon_hours=horizon_days * 24), best_rmse)
summary = business_summary(fc, tariff_peak, tariff_off)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected energy", f"{summary['total_kwh_next_period']:,.0f} kWh")
c2.metric("Estimated cost", f"₹{summary['estimated_cost_inr']:,.0f}")
c3.metric("Peak day", summary["peak_day"])
c4.metric("Peak hour", f"{summary['peak_hour_of_day']}:00")

st.subheader(f"{horizon_days}-day demand forecast")
hist = data["energy_kwh"].iloc[-history_days * 24:].rename("Actual")
chart = pd.concat([hist, fc["forecast_kwh"].rename("Forecast"),
                   fc["lower_95"].rename("Lower 95%"), fc["upper_95"].rename("Upper 95%")], axis=1)
st.line_chart(chart)
st.caption(f"Model: {bundle['name']} · band = ±1.96 × test RMSE ({best_rmse:.2f} kWh)")

left, right = st.columns(2)
with left:
    st.subheader("Model comparison (test set)")
    st.dataframe(metrics.round(3), hide_index=True)
with right:
    st.subheader("Average load by hour")
    by_hour = data.groupby(data.index.hour)["energy_kwh"].mean().rename("Avg kWh")
    st.bar_chart(by_hour)

st.subheader("Saved graphs")
imgs = sorted(Path("images").glob("*.png"))
cols = st.columns(2)
for i, p in enumerate(imgs):
    cols[i % 2].image(str(p), caption=p.stem.replace("_", " "))

st.download_button("Download forecast CSV", fc.round(2).to_csv().encode(), "forecast.csv", "text/csv")
