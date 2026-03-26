# F1 Analytics Platform — Project Plan

## Goal

Replicate [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake) using 100% free tools. Build a Streamlit web app with ML predictions (champion, best team, driver departures), EDA dashboard, and interactive DuckDB SQL console.

## Phases

### Phase 1: Project Setup + Data Collection
- [x] Create project structure, CLAUDE.md, Plan.md, requirements.txt
- [ ] Implement `etl/collect.py` (FastF1 data collection)
- [ ] Create `data/champions.csv` (historical WDC winners)
- [ ] Run collection for 2020-2025

### Phase 2: ETL Pipeline
- [ ] `etl/bronze.py` — raw to bronze (clean + consolidate)
- [ ] `etl/sql/fs_driver.sql` — parameterized feature store SQL
- [ ] `etl/silver.py` — build 4 temporal windows + join
- [ ] `etl/sql/abt_*.sql` — ABT construction queries
- [ ] `etl/gold.py` — build gold layer ABTs
- [ ] `etl/run_pipeline.py` — full pipeline orchestrator

### Phase 3: ML Models (Multi-Model Comparison)
- [ ] `ml/model_selection.py` — candidate models (RF, XGBoost, LightGBM, LogReg, GBM)
- [ ] `ml/utils.py` — splits, metrics, MLFlow setup
- [ ] `ml/champion_model.py` — train + compare for champion prediction
- [ ] `ml/team_model.py` — train + compare for team prediction
- [ ] `ml/departure_model.py` — train + compare for departure prediction

### Phase 4: Streamlit Web App
- [ ] `app/helpers.py` — shared UI utilities
- [ ] `app/tab_predictions.py` — ML predictions with model comparison
- [ ] `app/tab_eda.py` — interactive EDA visualizations
- [ ] `app/tab_duckdb.py` — SQL console
- [ ] `app/main.py` — tab layout

### Phase 5: Containerization
- [ ] `Dockerfile`
- [ ] `docker-compose.yaml`

## Status

**Current phase**: Phase 1 — Project Setup
