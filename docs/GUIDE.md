# Complete Guide: AI-Powered Energy Consumption Forecasting System

This guide covers every section you asked for (A to M). The working code is in this repo; this document explains it. Real numbers below come from an actual run of `python main.py`.

---

## A. Project Explanation

### Simple language
Every building uses electricity in a pattern: more during office hours, less at night, more when it is hot. If we know the pattern, we can **predict tomorrow's electricity use** before it happens. A forecasting system learns from past meter readings and predicts future ones, like a weather forecast, but for electricity.

### Technical language
Short-term load forecasting (STLF) is a supervised time-series regression problem: given a history of consumption `y(t-1), y(t-2), ...` and exogenous variables (temperature, humidity, calendar), estimate `y(t+h)` for horizons `h` from 1 hour to several days. We engineer lag, rolling and cyclical calendar features, train regressors with a chronological split, benchmark against a seasonal-naive baseline, and run recursive multi-step forecasting.

### What problem does it solve?
Electricity must be generated or bought in advance, and it is expensive to store. Wrong forecasts cause:
- over-buying (wasted money) or under-buying (expensive spot purchases, blackouts)
- peak-demand charges and tariff penalties
- inefficient cooling/heating schedules
- poor integration of solar and wind

### Why it matters by industry
| Industry | Why forecasting matters |
|---|---|
| Smart cities | Optimise street lighting, metro and building loads; support demand-response programmes |
| Electricity boards | Day-ahead scheduling, power purchase agreements, avoiding load shedding |
| Manufacturing plants | Shift heavy machines to cheaper tariff hours; avoid maximum-demand penalties |
| Data centers | Plan cooling and capacity; negotiate power contracts |
| Renewable companies | Match variable supply with expected demand; size batteries |

### How companies cut cost with forecasts
1. **Peak shaving**: know the peak in advance, pre-cool the building or shift flexible loads.
2. **Tariff optimisation**: run flexible work in cheap hours (this project's `business_summary` does a simple version).
3. **Procurement**: buy the right quantity in the day-ahead market.
4. **Maintenance**: unusual gaps between forecast and actual flag faulty equipment.
5. **Carbon reduction**: schedule loads when the grid is greener.

### Full workflow
1. **Data collection**: hourly smart-meter readings plus weather and calendar data (here: simulated)
2. **Preprocessing**: fix timestamps, fill missing values, remove sensor spikes
3. **Feature engineering**: hour, weekday, month, lags, rolling means
4. **Model training**: fit several models on the past
5. **Evaluation**: score on a future period the model never saw
6. **Forecasting**: predict next 7 days
7. **Visualization**: graphs for stakeholders

---

## B. Tech Stack Options

| | Option A (easiest) | Option B (intermediate) | Option C (advanced) |
|---|---|---|---|
| Tools | Python, Pandas, NumPy, Matplotlib, Scikit-learn | + Seaborn, Statsmodels, Prophet, XGBoost/LightGBM | + TensorFlow/PyTorch, Optuna, MLflow, FastAPI, Docker |
| Models | Linear Regression, Random Forest, Gradient Boosting | SARIMA, Prophet, LightGBM | LSTM, GRU, Temporal Fusion Transformer |
| Difficulty | Easy | Medium | Hard |
| GPU | No | No | Recommended |
| Output | Metrics + graphs + 7-day forecast | + seasonality decomposition, tuned models | + deployed API/dashboard, experiment tracking |

## C. Selected Approach: Option A

Reasons: runs in about a minute on any laptop, every step is explainable in an interview, scikit-learn models with lag features are genuinely strong for this problem (our R² is 0.979), and the simple stack makes the GitHub repo easy for recruiters to read. Once it works, add one Option B/C extension (e.g. SARIMA or LSTM comparison) as a follow-up commit.

---

## D. Architecture

```
Simulated meter + weather ──► raw CSV ──► Cleaning ──► Feature engineering
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
                           Chronological split
                           (train past / test last 60 days)
                                    │
              ┌──────────┬──────────┼──────────────┐
              ▼          ▼          ▼              ▼
          Baseline    Linear     Random       Gradient
                                 Forest       Boosting
              └──────────┴──────────┼──────────────┘
                                    ▼
                  Evaluation (RMSE, MAE, MAPE, R²) ──► best model saved
                                    │
                                    ▼
                  Recursive 7-day forecast + 95% band
                                    │
                                    ▼
              Graphs (images/) + CSV/JSON (outputs/) + business summary
```

**Modules**
| File | Role |
|---|---|
| `src/generate_data.py` | Simulates the building, weather, holidays, data faults |
| `src/preprocess.py` | Loads CSV, enforces hourly index, removes spikes, interpolates gaps |
| `src/features.py` | Calendar, cyclical, lag and rolling features |
| `src/models.py` | Split, train, evaluate, save best model |
| `src/forecast.py` | Recursive forecasting, uncertainty band, business summary |
| `src/visualize.py` | All graphs |
| `main.py` | Orchestrates the pipeline |

**Data flow:** raw CSV → `load_raw` → `clean` → `build_dataset` (adds 14 features) → `chronological_split` → `train_and_evaluate` → `save_best` → `forecast_next` → plots and outputs.

---

## E. Folder Structure

```
AI-Energy-Forecasting/
├── data/
│   ├── raw/building_energy.csv        original (simulated) readings, untouched
│   └── processed/clean_energy.csv     cleaned data
├── notebooks/                         EDA notebook (optional, add your own)
├── src/                               reusable modules
├── models/best_model.joblib           trained model (git-ignored by default)
├── outputs/                           metrics.csv, forecast_next_7_days.csv, business_summary.json
├── images/                            PNG graphs used in README
├── docs/GUIDE.md                      this guide
├── README.md
├── requirements.txt
├── .gitignore
└── main.py
```
Why this layout: separating raw from processed data protects the original; `src/` holds importable code (not one giant script); `outputs/` and `images/` hold generated proof.

---

## F. Installation

**Python:** 3.10 or newer (3.11/3.12 fine).

`requirements.txt`
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
joblib>=1.3
```

**Windows (PowerShell / CMD)**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify: `python -c "import pandas, sklearn; print('ok')"`.

---

## G. Code (file by file)

All files are in the repo. Summary of what each does and the key ideas to explain in interviews:

**`src/generate_data.py`**: builds the load as `base + people*occupancy*day_factor + AC*(temp-24)+ ...`, times slow growth and lognormal noise. Occupancy is a bell curve centred on 1 PM; weekends/holidays scale to 35%. Injects 1% missing values and 0.2% spikes (3.5x).

**`src/preprocess.py`**: `load_raw` parses timestamps and forces a regular hourly index (`asfreq("h")`). `clean` flags a reading as a spike if it is above 2x the median of its 7 nearest hours (a *local* window; a wider window wrongly flags normal afternoon peaks, which I found and fixed during testing), then does time-based interpolation.

**`src/features.py`**: 14 features: hour, dayofweek, month, is_weekend, is_holiday, hour_sin, hour_cos, temperature, humidity, lag_1, lag_24, lag_168, roll_mean_24, roll_std_24. Every lag/rolling value uses `shift(1)` or more, so there is **no leakage**.

**`src/models.py`**: chronological split (last 60 days = test); models: Linear Regression (with scaling), Random Forest, HistGradientBoosting; baseline = same hour yesterday; metrics RMSE, MAE, MAPE, R²; best model saved with joblib.

**`src/forecast.py`**: recursive loop: for each future hour, build features from history plus earlier predictions, predict, append. Adds a ±1.96×RMSE band and a `business_summary` (kWh, ₹ cost under a two-tier tariff, peak day, peak hour).

**`src/visualize.py`**: eight graphs (see section M). Uses permutation importance so any model can be explained.

**`main.py`**: runs everything.

---

## H. Virtual Simulation

**How the dataset mirrors reality:** office buildings peak around midday on weekdays, drop at night, run at about a third of load on weekends/holidays, and use more power as temperature rises (cooling). Real meters also drop readings and glitch, so the data includes both.

**How forecasting simulates business decisions:** you play the energy manager of this building. Each week you receive the 7-day forecast and decide when to pre-cool, when to run flexible loads, and how much power to procure.

**Step-by-step simulation**
1. Generate two years of hourly data (`generate_data.py`)
2. Inspect it: preview, missing counts, describe()
3. Clean it: spikes and gaps
4. Hold out the last 60 days as "the future"
5. Train models only on data before that
6. Predict the held-out 60 days and compare with what really happened
7. Pick the best model by RMSE
8. Forecast the next 7 days beyond the data
9. Compute expected consumption, cost, peak day and hour
10. Decide: shift flexible loads away from the peak hour (13:00 in our run)

**Graphs generated:** overview, weekday×hour heatmap, model comparison, actual vs predicted, scatter, residuals, feature importance, 7-day forecast.
**Outputs saved:** `metrics.csv`, `forecast_next_7_days.csv`, `business_summary.json`, `best_model.joblib`.
**Proof to capture:** terminal log of the run, the PNGs, the metrics table, GitHub commit history.

---

## I. Execution

```bash
python main.py
```
Expected terminal sequence: raw data preview → `Cleaning report: {'missing_before': 175, 'spikes_removed': 35, 'missing_after': 0}` → train/test date ranges → training logs → metrics table → `Best model: Gradient Boosting` → 8 `saved images/...` lines → business summary.

**Actual results from the run**

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Gradient Boosting | 2.96 | 2.19 | 3.7% | 0.979 |
| Random Forest | 3.12 | 2.25 | 3.8% | 0.977 |
| Linear Regression | 4.31 | 3.29 | 5.8% | 0.956 |
| Baseline | 11.72 | 6.61 | 10.3% | 0.672 |

Business summary: 9,191.2 kWh next 7 days, ₹67,556, peak day 2025-01-06, peak hour 13.

To run single stages, import from `src`, for example `from src import generate_data; generate_data.main()`.

---

## J. GitHub Steps

1. Create a repo on github.com: **New repository**, name `AI-Energy-Forecasting` (alternatives: `smart-building-energy-forecasting`, `energy-demand-forecasting-ml`), Public, no README (you already have one).
2. **Description:** `Machine-learning system that forecasts hourly building electricity demand 7 days ahead, with cost insights. Python, Scikit-learn.`
3. **Topics/tags:** `machine-learning`, `time-series`, `energy-forecasting`, `smart-grid`, `python`, `scikit-learn`, `data-science`, `forecasting`, `climate-tech`, `smart-city`.
4. Push:
```bash
cd AI-Energy-Forecasting
git init
git branch -M main
git add .
git commit -m "Initial project structure"
git remote add origin https://github.com/<your-username>/AI-Energy-Forecasting.git
git push -u origin main
```
5. Make it professional: pin the repo on your profile, keep `images/` committed so README graphs render, add a LICENSE (MIT), use meaningful commit messages, add the metrics table to the README, and add a short "Limitations" section (honesty about simulated data is a strength).

---

## K. README

The complete README is in `README.md` at the repo root (overview, problem, industry relevance, stack, dataset, architecture, installation, usage, results, screenshots, learning outcomes, future work). Replace `<your-username>` with your GitHub username.

---

## L. Commit Plan (7 days)

| Day | Work | Commit messages | Proof |
|---|---|---|---|
| 1 Setup | Folders, venv, requirements, .gitignore, empty README | `chore: initialise project structure` / `chore: add requirements and gitignore` | Repo created, folder screenshot |
| 2 Dataset | `generate_data.py`, create CSV, inspect | `feat: add smart-meter simulation` / `docs: describe dataset in README` | `df.head()` and `describe()` output |
| 3 Preprocessing | `preprocess.py`, cleaning report | `feat: add data loading and cleaning` | Cleaning report log; overview graph |
| 4 Model | `features.py`, `models.py` | `feat: add feature engineering` / `feat: train baseline and ML models` | Training log |
| 5 Evaluation | Metrics, model comparison, save model | `feat: evaluate with RMSE MAE MAPE R2` / `docs: add results table` | Metrics table, comparison graph |
| 6 Visualization | `forecast.py`, `visualize.py`, business summary | `feat: add 7-day recursive forecast` / `feat: add all plots` | All 8 PNGs |
| 7 Upload | README polish, LICENSE, final run, tag release | `docs: finalise README with screenshots` / `chore: v1.0` | Repo preview, commit graph |

Tip: commit after each working step (not one big commit); recruiters read history.

---

## M. Proof Checklist

| Asset | File name | Where it comes from |
|---|---|---|
| Dataset preview | `images/00_dataset_preview.png` (screenshot of terminal) | `main.py` raw preview |
| Preprocessing output | `docs/cleaning_report.png` | `Cleaning report:` line |
| Training logs | `docs/training_log.png` | "Training ..." lines |
| Energy overview | `images/01_energy_overview.png` | auto |
| Weekly pattern heatmap | `images/02_weekly_hourly_heatmap.png` | auto |
| Model comparison | `images/03_model_comparison.png` | auto |
| Actual vs predicted | `images/04_actual_vs_predicted.png` | auto |
| Scatter | `images/05_scatter_pred_vs_actual.png` | auto |
| Error analysis | `images/06_residuals.png` | auto |
| Feature importance | `images/07_feature_importance.png` | auto |
| Future forecast | `images/08_future_forecast.png` | auto |
| Error metrics | `outputs/metrics.csv` | auto |
| Forecast table | `outputs/forecast_next_7_days.csv` | auto |
| Business summary | `outputs/business_summary.json` | auto |
| GitHub repo preview | `docs/github_repo_preview.png` | screenshot of repo page |

**Naming:** numbered, lowercase, underscores. **Storage:** graphs in `images/`, screenshots in `docs/`, generated data in `outputs/`. **README usage:** `![caption](images/04_actual_vs_predicted.png)`.

### Common mistakes and checks per phase
| Phase | Common mistake | Verify |
|---|---|---|
| Setup | Forgetting to activate the venv | `pip list` shows scikit-learn |
| Dataset | Timestamps parsed as text | `df.index` is `DatetimeIndex` |
| Cleaning | Dropping rows instead of interpolating (breaks hourly spacing) | `df.index.to_series().diff().value_counts()` shows only 1 hour |
| Features | Data leakage (using current value as a feature) | Every lag uses `shift(>=1)` |
| Modelling | Random shuffle split | Test dates all later than train dates |
| Evaluation | No baseline | Models beat the baseline RMSE |
| Forecasting | Using true future values | Forecast uses only history + predictions |
| Visualization | Backend errors on servers | `matplotlib.use("Agg")` |
| GitHub | Committing `venv/` | `.gitignore` includes `venv/` |

### Interview talking points
- "I used a chronological split and a seasonal-naive baseline; my model reduced RMSE by about 75%."
- "I avoided leakage by shifting all lag and rolling features."
- "I found my first spike detector flagged 469 normal peaks; a local median window fixed it to exactly the 35 injected faults."
- "Data is simulated; the pipeline accepts real data with the same columns."

### Next upgrades
Swap in a real dataset, add SARIMA/Prophet/LSTM comparison, a Streamlit dashboard, hyperparameter tuning with time-series cross-validation, and a Dockerfile.
