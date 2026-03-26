# F1 Analytics Platform

## Overview

Free, local-first F1 data engineering + ML + web app replicating [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake). Uses DuckDB instead of Databricks, local Parquet instead of S3, and multiple ML models compared via MLFlow.

## Architecture

```
FastF1 → Parquet (raw/) → DuckDB → Parquet (bronze/silver/gold/) → ML (scikit-learn/XGBoost/LightGBM) → Streamlit App
```

**Medallion layers:**
- `data/raw/` — One Parquet per FastF1 session (`{year}_{round}_{mode}.parquet`)
- `data/bronze/` — Cleaned, consolidated `results.parquet`
- `data/silver/` — Feature store: `fs_driver_life.parquet`, `fs_driver_last10.parquet`, `fs_driver_last20.parquet`, `fs_driver_last40.parquet`, `fs_driver_all.parquet`
- `data/gold/` — ABTs: `abt_champions.parquet`, `abt_teams.parquet`, `abt_departures.parquet`

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full ETL pipeline (collect + bronze + silver + gold)
python -m etl.run_pipeline --years 2020 2021 2022 2023 2024 2025

# Run individual ETL steps
python -m etl.collect --years 2024 2025 --modes R S
python -m etl.bronze
python -m etl.silver
python -m etl.gold

# Train ML models (logs to mlruns/)
python -m ml.champion_model
python -m ml.team_model
python -m ml.departure_model

# Run Streamlit app
streamlit run app/main.py

# MLFlow UI
mlflow ui --backend-store-uri mlruns/

# Docker
docker-compose up
```

## Conventions

- **SQL engine**: DuckDB for all transformations. SQL files in `etl/sql/` use `read_parquet()` to access data.
- **Storage**: Parquet files only. No database server.
- **ML tracking**: MLFlow with local file store (`mlruns/`). Each prediction task (champion, team, departure) is a separate experiment with multiple model runs.
- **Web app**: Streamlit with 3 tabs (Predictions, EDA, DuckDB Console). Models loaded inline via `@st.cache_resource`.
- **Charts**: Plotly for interactive visualizations.

## Tech Stack

All free: FastF1, DuckDB, pandas, scikit-learn, XGBoost, CatBoost, imbalanced-learn, Optuna, MLFlow, Streamlit, Plotly, Docker.
