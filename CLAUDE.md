# F1 Analytics Platform

## Overview

Free, local-first F1 data engineering + ML + web app replicating [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake). Uses DuckDB instead of Databricks, local Parquet instead of S3, and multiple ML models compared via MLFlow.

## Architecture

```
FastF1 → Parquet (raw/) → DuckDB → Parquet (bronze/silver/gold/) → ML (scikit-learn/XGBoost/CatBoost) → Streamlit App
```

**Medallion layers:**
- `data/raw/` — One Parquet per FastF1 session (`{year}_{round}_{mode}.parquet`), includes weather data (2018+)
- `data/bronze/` — Cleaned, consolidated `results.parquet` with weather columns
- `data/silver/` — Feature store: `fs_driver_life.parquet`, `fs_driver_last10.parquet`, `fs_driver_last20.parquet`, `fs_driver_last40.parquet`, `fs_driver_all.parquet`
- `data/gold/` — ABTs (end-of-year and in-season): `abt_champions.parquet`, `abt_teams.parquet`, `abt_departures.parquet`, `abt_champions_inseason.parquet`, `abt_teams_inseason.parquet`, `abt_departures_inseason.parquet`

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full ETL pipeline (collect + bronze + silver + gold)
python -m etl.run_pipeline --years 2020 2021 2022 2023 2024 2025

# Re-collect with weather data (FastF1 weather available from 2018+)
python -m etl.run_pipeline --years 2018 2019 2020 2021 2022 2023 2024 2025 --force

# Run individual ETL steps
python -m etl.collect --years 2024 2025 --modes R Q S     # collect raw data
python -m etl.collect --years 2018 2019 2020 --force      # re-collect with --force to overwrite
python -m etl.bronze                                       # raw → bronze
python -m etl.silver                                       # bronze → silver (feature store)
python -m etl.gold                                         # silver → gold (ABTs)

# Train ML models (logs to mlruns/ and mlflow.db)
python -m ml.champion_model
python -m ml.team_model
python -m ml.departure_model

# Evaluate TimesFM zero-shot forecasts (uses separate venv, logs to same MLflow experiments)
.venv-timesfm/bin/python -m ml.evaluate_timesfm                 # all 3 targets
.venv-timesfm/bin/python -m ml.evaluate_timesfm champion        # champion | constructor | departure

# Run Streamlit app
streamlit run app/main.py

# MLFlow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Docker
docker-compose up
```

## Conventions

- **SQL engine**: DuckDB for all transformations. SQL files in `etl/sql/` use `read_parquet()` to access data.
- **Storage**: Parquet files only. No database server. Bronze uses `union_by_name=true` to handle mixed schemas (pre/post weather).
- **Feature store**: Point-in-time correct — `fs_driver.sql` uses `r.event_date < d.dt_ref` so features include current-season data up to (but not including) each race date. Features evolve race-by-race within a season. Includes qualifying features (avg position, poles, Q3 reach rate) from collected Q sessions.
- **Weather features**: Collected from FastF1 (air/track temp, humidity, pressure, wind speed/direction, rainfall). Available from 2018+, NULL for earlier years. Aggregated per session in collect, per window in feature store.
- **ML tracking**: MLFlow with SQLite backend (`mlflow.db`) and local artifact store (`mlruns/`). Each prediction task (champion, team, departure) is a separate experiment with multiple model runs.
- **ML models**: Batch models (LogisticRegression, LightGBM, BalancedRandomForest, XGBoost). Hyperparameter tuning via Optuna (TPE sampler, median pruner).
- **ABTs**: Two variants per target — end-of-year (one row per driver-year) and in-season (one row per driver-race, for time-series predictions).
- **Web app**: Streamlit with 3 tabs (Predictions, EDA, DuckDB Console). DuckDB Console supports Ctrl+Enter to run queries. Models loaded inline via `@st.cache_resource`.
- **Charts**: Plotly for interactive visualizations.

## Tech Stack

All free: FastF1, DuckDB, pandas, scikit-learn, XGBoost, CatBoost, imbalanced-learn, Optuna, MLFlow, Streamlit, Plotly, Docker.
