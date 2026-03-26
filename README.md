# F1 Analytics Platform

A free, local-first Formula 1 data engineering + machine learning + web application platform. Inspired by [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake), rebuilt with DuckDB instead of Databricks, local Parquet instead of S3, and multiple ML models compared via MLFlow.

## Architecture

```
FastF1 --> Parquet (raw/) --> DuckDB --> Parquet (bronze/silver/gold/) --> ML Models --> Streamlit App
```

The project follows a **medallion architecture**:

| Layer | Path | Description |
|-------|------|-------------|
| **Raw** | `data/raw/` | One Parquet file per FastF1 session (`{year}_{round}_{mode}.parquet`) |
| **Bronze** | `data/bronze/` | Cleaned and consolidated `results.parquet` |
| **Silver** | `data/silver/` | Feature store with temporal windows (`fs_driver_life.parquet`, `fs_driver_last10.parquet`, `fs_driver_last20.parquet`, `fs_driver_last40.parquet`, `fs_driver_all.parquet`) |
| **Gold** | `data/gold/` | Analytical base tables: `abt_champions.parquet`, `abt_teams.parquet`, `abt_departures.parquet` |

## Project Structure

```
f1-analytics/
├── app/                    # Streamlit web application
│   ├── main.py             # Entry point with tab layout
│   ├── tab_predictions.py  # ML predictions tab
│   ├── tab_eda.py          # Exploratory data analysis tab
│   ├── tab_duckdb.py       # Interactive DuckDB SQL console
│   └── helpers.py          # Shared UI utilities
├── etl/                    # ETL pipeline modules
│   ├── collect.py          # FastF1 data collection
│   ├── bronze.py           # Raw to bronze transformation
│   ├── silver.py           # Bronze to silver (feature store)
│   ├── gold.py             # Silver to gold (ABTs)
│   ├── run_pipeline.py     # Full pipeline orchestrator
│   └── sql/                # DuckDB SQL queries
│       ├── fs_driver.sql   # Feature store query
│       ├── fs_all.sql      # All-time features query
│       ├── abt_champions.sql
│       ├── abt_teams.sql
│       └── abt_departures.sql
├── ml/                     # Machine learning models
│   ├── champion_model.py   # Champion prediction training
│   ├── team_model.py       # Best team prediction training
│   ├── departure_model.py  # Driver departure prediction training
│   ├── model_selection.py  # Candidate model definitions
│   ├── predict.py          # Inference utilities
│   └── utils.py            # Training, splits, metrics, MLFlow setup
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Parquet data files (raw/bronze/silver/gold)
├── mlruns/                 # MLFlow experiment tracking
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## Tech Stack

All tools are free and open source:

- **Data collection**: [FastF1](https://github.com/theOehrly/Fast-F1)
- **SQL engine**: [DuckDB](https://duckdb.org/)
- **Data processing**: pandas
- **ML models**: scikit-learn, XGBoost, CatBoost
- **Class balancing**: imbalanced-learn
- **Hyperparameter tuning**: Optuna
- **Experiment tracking**: MLFlow
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

Collects data from FastF1 and builds all medallion layers (raw -> bronze -> silver -> gold):

```bash
python -m etl.run_pipeline --years 2020 2021 2022 2023 2024 2025
```

This runs all four steps sequentially:
1. **Collect** — Downloads session results from FastF1 API
2. **Bronze** — Cleans and consolidates raw data
3. **Silver** — Builds the feature store with multiple temporal windows
4. **Gold** — Constructs analytical base tables for ML

### Running Individual ETL Steps

```bash
# Collect raw data (R = Race, S = Sprint)
python -m etl.collect --years 2024 2025 --modes R S

# Build bronze layer
python -m etl.bronze

# Build silver layer (feature store)
python -m etl.silver

# Build gold layer (ABTs)
python -m etl.gold
```

### Training ML Models

Each prediction task trains and compares multiple model types (Random Forest, XGBoost, CatBoost, Logistic Regression, etc.), logging all runs to MLFlow:

```bash
# Champion prediction
python -m ml.champion_model

# Best team prediction
python -m ml.team_model

# Driver departure prediction
python -m ml.departure_model
```

### Running the Web App

```bash
streamlit run app/main.py
```

The app runs at `http://localhost:8501` and has three tabs:

- **Predictions** — ML model predictions with model comparison
- **EDA** — Interactive exploratory data analysis with Plotly charts
- **DuckDB Console** — Run SQL queries directly against the Parquet data

### MLFlow UI

View experiment runs, compare metrics, and inspect model artifacts:

```bash
mlflow ui --backend-store-uri mlruns/
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
