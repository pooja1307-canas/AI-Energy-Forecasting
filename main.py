"""Run the whole pipeline:  python main.py

Phases: simulate data -> load -> clean -> features -> train -> evaluate -> forecast -> plots.
"""
import json
from pathlib import Path

import pandas as pd

from src import generate_data, preprocess, visualize
from src.features import FEATURES, TARGET, build_dataset
from src.forecast import add_uncertainty, business_summary, forecast_next
from src.models import chronological_split, save_best, train_and_evaluate

RAW = "data/raw/building_energy.csv"


def main():
    Path("outputs").mkdir(exist_ok=True)

    # 1. Data (simulated smart-meter readings)
    if not Path(RAW).exists():
        generate_data.main(RAW)
    raw = preprocess.load_raw(RAW)
    print("\n--- Raw data preview ---")
    print(raw.head())
    print(raw.describe().round(2))

    # 2. Cleaning
    clean = preprocess.clean(raw)
    preprocess.save_processed(clean)

    # 3. Features
    data = build_dataset(clean)
    print(f"\nModelling table: {data.shape[0]:,} rows x {len(FEATURES)} features")

    # 4. Train / evaluate
    train, test = chronological_split(data, test_days=60)
    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train):,} rows)")
    print(f"Test : {test.index.min()} -> {test.index.max()} ({len(test):,} rows)\n")
    fitted, preds, results = train_and_evaluate(train, test)
    print("\n--- Evaluation (test set) ---")
    print(results.round(3).to_string(index=False))
    results.round(4).to_csv("outputs/metrics.csv", index=False)
    best = save_best(fitted, results)

    # 5. Graphs
    visualize.plot_overview(clean)
    visualize.plot_patterns(clean)
    visualize.plot_model_comparison(results)
    visualize.plot_actual_vs_pred(test[TARGET], preds[best], best)
    visualize.plot_residuals(test[TARGET], preds[best])
    visualize.plot_feature_importance(fitted[best], test[FEATURES], test[TARGET], best)

    # 6. Future forecast + business insight
    best_rmse = float(results.loc[results["Model"] == best, "RMSE"].iloc[0])
    fc = add_uncertainty(forecast_next(clean, fitted[best], horizon_hours=168), best_rmse)
    fc.round(2).to_csv("outputs/forecast_next_7_days.csv")
    visualize.plot_forecast(clean[TARGET], fc)
    summary = business_summary(fc)
    json.dump(summary, open("outputs/business_summary.json", "w"), indent=2)
    print("\n--- Business summary for next 7 days ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
