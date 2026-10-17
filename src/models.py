"""Steps 5-6: train several models, evaluate them, keep the best one."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES, TARGET


def chronological_split(data: pd.DataFrame, test_days: int = 60):
    """Time series must be split by TIME (never shuffled): train on past, test on future."""
    cutoff = data.index.max() - pd.Timedelta(days=test_days)
    return data[data.index <= cutoff], data[data.index > cutoff]


def get_models() -> dict:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=18, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    """Fit every model, score on the held-out test period, return fitted models + results."""
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    fitted, predictions, rows = {}, {}, []

    # Baseline: "tomorrow looks like today" (seasonal naive). Every ML model must beat this.
    baseline = test["lag_24"]
    predictions["Baseline (same hour yesterday)"] = baseline
    rows.append({"Model": "Baseline (same hour yesterday)", **compute_metrics(y_test, baseline)})

    for name, model in get_models().items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_test), index=test.index)
        fitted[name], predictions[name] = model, pred
        rows.append({"Model": name, **compute_metrics(y_test, pred)})

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return fitted, predictions, results


def save_best(fitted: dict, results: pd.DataFrame, path: str = "models/best_model.joblib") -> str:
    best_name = results[results["Model"].isin(fitted)].iloc[0]["Model"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best_name, "model": fitted[best_name]}, path)
    print(f"Best model: {best_name} -> saved to {path}")
    return best_name
