# F1 Analytics Platform

A free, local-first Formula 1 data engineering + machine learning + web application platform. Inspired by [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake), rebuilt with DuckDB instead of Databricks, local Parquet instead of S3, and multiple ML models compared via MLFlow.

## Architecture

```
FastF1 --> Parquet (raw/) --> DuckDB --> Parquet (bronze/silver/gold/) --> ML Models --> Streamlit App
```

The project follows a **medallion architecture**:

| Layer | Path | Description |
|-------|------|-------------|
| **Raw** | `data/raw/` | One Parquet file per FastF1 session (`{year}_{round}_{mode}.parquet`), includes weather data (2018+) |
| **Bronze** | `data/bronze/` | Cleaned and consolidated `results.parquet` with weather columns |
| **Silver** | `data/silver/` | Feature store with temporal windows (`fs_driver_life.parquet`, `fs_driver_last10.parquet`, `fs_driver_last20.parquet`, `fs_driver_last40.parquet`, `fs_driver_all.parquet`) |
| **Gold** | `data/gold/` | Analytical base tables: end-of-year (`abt_champions.parquet`, `abt_teams.parquet`, `abt_departures.parquet`) and in-season (`abt_champions_inseason.parquet`, `abt_teams_inseason.parquet`, `abt_departures_inseason.parquet`). In-season ABTs include clinch detection, momentum features, and clinch proximity |

## Project Structure

```
f1-analytics/
├── app/                    # Streamlit web application
│   ├── main.py             # Entry point with tab layout
│   ├── tab_predictions.py  # ML predictions tab
│   ├── tab_model_comparison.py  # Side-by-side model comparison (ROC/PR curves, confusion matrices)
│   ├── tab_eda.py          # Exploratory data analysis tab
│   ├── tab_duckdb.py       # Interactive DuckDB SQL console (Ctrl+Enter to run)
│   └── helpers.py          # Shared UI utilities
├── etl/                    # ETL pipeline modules
│   ├── collect.py          # FastF1 data collection (results + weather)
│   ├── bronze.py           # Raw to bronze transformation
│   ├── silver.py           # Bronze to silver (feature store)
│   ├── gold.py             # Silver to gold (ABTs)
│   ├── run_pipeline.py     # Full pipeline orchestrator
│   └── sql/                # DuckDB SQL queries
│       ├── fs_driver.sql           # Feature store query (point-in-time correct)
│       ├── fs_all.sql              # Join all temporal windows
│       ├── abt_champions.sql       # End-of-year champion ABT
│       ├── abt_teams.sql           # End-of-year constructor ABT
│       ├── abt_departures.sql      # End-of-year departure ABT
│       ├── abt_champions_inseason.sql  # In-season champion ABT
│       ├── abt_teams_inseason.sql      # In-season constructor ABT
│       └── abt_departures_inseason.sql # In-season departure ABT
├── ml/                     # Machine learning models
│   ├── champion_model.py   # Champion prediction training
│   ├── team_model.py       # Best team prediction training
│   ├── departure_model.py  # Driver departure prediction training
│   ├── model_selection.py  # Candidate model definitions
│   ├── predict.py          # Inference utilities
│   ├── utils.py            # Training, splits, metrics, MLFlow setup
│   ├── evaluate_timesfm.py # TimesFM zero-shot forecast evaluation
│   └── timefm_predictor.py # TimesFM predictor wrapper
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Parquet data files (raw/bronze/silver/gold)
├── mlruns/                 # MLFlow artifact storage
├── mlflow.db               # MLFlow metadata (SQLite backend)
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## Tech Stack

All tools are free and open source:

- **Data collection**: [FastF1](https://github.com/theOehrly/Fast-F1) (results + weather data)
- **SQL engine**: [DuckDB](https://duckdb.org/)
- **Data processing**: pandas
- **ML models**: scikit-learn, XGBoost, LightGBM
- **Class balancing**: imbalanced-learn
- **Hyperparameter tuning**: Optuna (TPE sampler, median pruner)
- **Zero-shot forecasting**: TimesFM (separate venv)
- **Experiment tracking**: MLFlow (SQLite backend)
- **Web app**: Streamlit
- **Visualizations**: Plotly, Matplotlib
- **Containerization**: Docker

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
git clone <repo-url>
cd f1-analytics
pip install -r requirements.txt
```

### Running the Full ETL Pipeline

Collects data from FastF1 (including weather) and builds all medallion layers (raw -> bronze -> silver -> gold):

```bash
python -m etl.run_pipeline --years 2020 2021 2022 2023 2024 2025
```

To re-collect with weather data for years that were previously collected without it (FastF1 weather is available from 2018+):

```bash
python -m etl.run_pipeline --years 2018 2019 2020 2021 2022 2023 2024 2025 --force
```

This runs all four steps sequentially:
1. **Collect** — Downloads session results and weather data from FastF1 API
2. **Bronze** — Cleans and consolidates raw data (handles mixed schemas via `union_by_name`)
3. **Silver** — Builds the feature store with multiple temporal windows (includes weather features)
4. **Gold** — Constructs analytical base tables for ML (end-of-year and in-season variants)

### Running Individual ETL Steps

```bash
# Collect raw data (R = Race, S = Sprint)
python -m etl.collect --years 2024 2025 --modes R S

# Re-collect existing files (e.g., to add weather data)
python -m etl.collect --years 2018 2019 2020 --force

# Build bronze layer
python -m etl.bronze

# Build silver layer (feature store)
python -m etl.silver

# Build gold layer (ABTs)
python -m etl.gold
```

### Training ML Models

Each prediction task trains and compares multiple batch models (LogisticRegression, LightGBM, BalancedRandomForest, XGBoost) with Optuna hyperparameter tuning. All runs are logged to MLFlow. Use `--nologreg` to skip LogisticRegression.

Both champion and team models use curated feature sets that exclude data leakage features (`season_fraction`, `season_race_number`) and zero-importance features. The team model uses a combined scoring metric (PR-AUC + top-1 champion accuracy) to select models that produce meaningful per-event predictions:

```bash
# Champion prediction
python -m ml.champion_model

# Best team prediction
python -m ml.team_model

# Driver departure prediction
python -m ml.departure_model
```

### TimesFM Zero-Shot Forecasts

Evaluate Google's TimesFM foundation model as a zero-shot forecaster on the same prediction targets. Uses a separate virtual environment and logs results to the same MLFlow experiments for direct comparison:

```bash
# Evaluate all 3 targets
.venv-timesfm/bin/python -m ml.evaluate_timesfm

# Evaluate a single target
.venv-timesfm/bin/python -m ml.evaluate_timesfm champion
.venv-timesfm/bin/python -m ml.evaluate_timesfm constructor
.venv-timesfm/bin/python -m ml.evaluate_timesfm departure
```

### Running the Web App

```bash
streamlit run app/main.py
```

The app runs at `http://localhost:8501` and has four tabs:

- **Predictions** — ML model predictions with time-series charts
- **Model Comparison** — Side-by-side metrics table, ROC/PR curves, and confusion matrices for all trained models
- **EDA** — Interactive exploratory data analysis with Plotly charts
- **DuckDB Console** — Run SQL queries directly against the Parquet data (Ctrl+Enter to execute, 13 example queries including weather analysis)

### MLFlow UI

View experiment runs, compare metrics, and inspect model artifacts:

```bash

```

Open `http://localhost:5000` in your browser.

## Docker

Run both the Streamlit app and MLFlow UI with Docker Compose:

```bash
docker-compose up
```

| Service | URL |
|---------|-----|
| Streamlit App | `http://localhost:8501` |
| MLFlow UI | `http://localhost:5000` |

The `data/` and `mlruns/` directories are mounted as volumes, so data persists across container restarts.

## Acknowledgments

- [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake) — Original project and inspiration
- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 telemetry and session data
