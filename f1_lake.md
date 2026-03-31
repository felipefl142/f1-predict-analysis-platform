# F1 Lake - ETL + Machine Learning Prediction Analysis

## Project Overview

**Repository:** [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake)

This project implements a complete data lakehouse pipeline for Formula 1 data. It collects race results from the FastF1 API, processes them through a medallion architecture (Raw -> Bronze -> Silver -> Gold), engineers features across multiple time windows, and trains a machine learning model to predict **which driver will be the World Champion** in a given season.

The project uses MLFlow for experiment tracking and model registry, and exposes predictions via a Flask API.

---

## Architecture Overview

```
┌──────────────┐
│  FastF1 API  │
└──────┬───────┘
       │
       v
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  collect.py  │────>│  .parquet    │────>│  sender.py -> S3    │
│  (Ingestion) │     │  (Raw Data)  │     │  (Raw Layer)        │
└──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  Nekt Platform      │
                                          │  Bronze (Delta)     │
                                          └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  etl/main.py        │
                                          │  etl/fs_drive.sql   │
                                          │  Silver Layer:      │
                                          │  4 Feature Stores   │
                                          └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  etl/fs_all.sql     │
                                          │  Silver Layer:      │
                                          │  fs_f1_driver_all   │
                                          └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  abt_champions.sql  │
                                          │  download_abt.py    │
                                          │  ABT CSV            │
                                          └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  ml_champion/       │
                                          │  train.py -> MLFlow │
                                          └──────────┬──────────┘
                                                     │
                                                     v
                                          ┌─────────────────────┐
                                          │  ml_champion/       │
                                          │  app.py (Flask API) │
                                          └─────────────────────┘
```

---

## Step-by-Step Pipeline Breakdown

### 1. Data Collection (Raw Layer)

**Script:** `collect.py`

The pipeline begins by pulling race results from the [FastF1](https://docs.fastf1.dev/) library, which wraps the official Formula 1 timing data.

- For each year and round, it fetches results from two session types: **Race** and **Sprint**.
- Results are saved locally as **Parquet** files under the `data/` directory.
- Each file contains per-driver results for a specific session: finishing position, grid position, points scored, status (finished/DNF), etc.

**Script:** `sender.py`

- Uploads the generated Parquet files to **AWS S3**, which serves as the Raw layer of the data lake.
- Deletes local copies after successful upload to keep the environment clean.

**Script:** `main.py` (orchestrator)

- Coordinates the collection and upload cycle.
- Runs on a **6-hour interval**, ensuring fresh data after race weekends.

---

### 2. Bronze Layer

Handled entirely by the **Nekt platform** (external to the repository code).

- Consolidates raw Parquet files from S3 into **Delta format** tables.
- Provides a queryable interface (via Spark SQL) for downstream transformations.
- The key Bronze table is `f1_results`, containing all race/sprint results in a unified schema.

---

### 3. Silver Layer - Feature Engineering

This is where the core analytical value is created. The Silver layer consists of **four Feature Store tables**, each computing the same set of ~45 metrics but over different time horizons. This multi-window design captures both long-term career quality and short-term form.

#### 3.1. The Four Feature Store Tables

| Table Name | Time Window | Purpose |
|---|---|---|
| `fs_f1_driver_life` | All-time (career) | Captures overall career quality and consistency |
| `fs_f1_driver_last_10` | Last 10 rounds | Captures very recent form |
| `fs_f1_driver_last_20` | Last 20 rounds | Captures medium-term form (~half a season) |
| `fs_f1_driver_last_40` | Last 40 rounds | Captures longer-term form (~two seasons) |

#### 3.2. How `fs_f1_driver_life` Is Built

**Script:** `etl/main.py`

For each race date from 1991 through 2024:

1. **Select all results** from Bronze (`f1_results`) up to the current reference date (`dt_ref`).
2. **Filter eligible drivers** — only those who have raced within the last 2 years from `dt_ref`. This prevents computing features for retired or inactive drivers.
3. **Compute aggregate statistics** per driver over their entire career (all results up to `dt_ref`).
4. **Save** the result as a Silver-layer table keyed by `(dt_ref, driverid)`.

This is a **point-in-time** computation: at each race date, features reflect only what was known up to that moment, preventing future data leakage.

#### 3.3. How `fs_f1_driver_last_10/20/40` Are Built

**Script:** `etl/fs_drive.sql`

The logic mirrors the life table, but uses a **sliding window**:

```sql
ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY round DESC) AS rn
...
WHERE rn <= {window_size}
```

- For each driver at each `dt_ref`, rounds are ranked by recency.
- Only the last N rounds (10, 20, or 40) are included in the aggregation.
- The same ~45 metrics are computed over this reduced window.

#### 3.4. How Features Are Combined

**Script:** `etl/fs_all.sql`

The four Feature Store tables are **INNER JOINed** on `(driverid, dt_ref)`:

```
fs_f1_driver_life     ──┐
fs_f1_driver_last_10  ──┤── INNER JOIN ──> fs_f1_driver_all
fs_f1_driver_last_20  ──┤                  (~180 features)
fs_f1_driver_last_40  ──┘
```

This produces the unified table `fs_f1_driver_all` with approximately **180 features** (45 base metrics x 4 time windows), each suffixed with `_life`, `_last10`, `_last20`, or `_last40`.

#### 3.5. The ~45 Base Metrics

Each metric exists in three variants where applicable: **overall**, **race-only**, and **sprint-only**.

| Category | Metrics | Description |
|---|---|---|
| **Session Counts** | `qtd_seasons`, `qtd_sessions`, `qtd_race`, `qtd_sprint` | How many seasons/sessions/races/sprints the driver has participated in |
| **Finishing Stats** | `qtde_sessions_finished`, `qtde_sessions_finished_race`, `qtde_sessions_finished_sprint` | Number of sessions where the driver finished (no DNF) |
| **Wins** | `qtde_1Pos`, `qtde_1Pos_race`, `qtde_1Pos_sprint` | Number of 1st place finishes |
| **Podiums** | `qtde_podios`, `qtde_podios_race`, `qtde_podios_sprint` | Number of top-3 finishes |
| **Top-5 Finishes** | `qtde_pos5`, `qtde_pos5_race`, `qtde_pos5_sprint` | Number of top-5 finishes |
| **Grid Position** | `avg_gridposition`, `qtde_gridpos5`, `qtde_1_gridposition` | Average qualifying position, top-5 grid starts, pole positions |
| **Points** | `qtde_points`, `qtde_points_race`, `qtde_points_sprint` | Total points accumulated |
| **Avg Finishing Position** | `avg_position`, `avg_position_race`, `avg_position_sprint` | Mean finishing position (lower is better) |
| **Pole-to-Win** | `qtde_pole_win`, `qtde_pole_win_race`, `qtde_pole_win_sprint` | Number of times driver started on pole and won |
| **Points-Scoring Sessions** | `qtd_sessions_with_points`, by race and sprint | Number of sessions where the driver scored points |
| **Overtaking** | `qtde_sessions_with_overtake`, `avg_overtake` | Sessions where driver finished ahead of grid position; average positions gained |

---

### 4. Target Variable Construction

**Scripts:** `etl/abt_champions.sql` + `etl/download_abt.py`

The target variable `flChampion` is a **binary classification label**: did this driver win the World Championship that year?

#### The SQL

```sql
SELECT t1.*,
    coalesce(t2.rankdriver, 0) AS flChampion
FROM fs_f1_driver_all AS t1
LEFT JOIN f1_champions AS t2
    ON t1.driverid = t2.driverid
    AND year(t1.dt_ref) = t2.year
WHERE t1.dt_ref >= date('2000-01-01')
    AND t1.dt_ref < date('2026-01-01')
```

#### How It Works

1. The combined feature table `fs_f1_driver_all` is **LEFT JOINed** with the reference table `f1_champions`.
2. `f1_champions` contains historical championship results — specifically `driverid`, `year`, and `rankdriver` (where 1 = champion).
3. The join matches on `driverid` and the year extracted from `dt_ref`.
4. `coalesce(t2.rankdriver, 0)` produces:
   - **1** if the driver was the champion that year (match found, `rankdriver = 1`)
   - **0** if the driver was not the champion (no match, NULL coalesced to 0)
5. Data is filtered to seasons from **2000 to 2025**.

#### Table Grain

The result is one row per **(driver, race-date)**. This means at every race in a season, the model evaluates: **"Will this driver be champion by end of season?"**

The output is saved as a CSV: `data/abt_f1_drivers_champion.csv`.

---

### 5. Model Training

**Script:** `ml_champion/train.py`

The model follows the **SEMMA methodology** (Sample, Explore, Modify, Model, Assess).

#### 5.1. Sampling Strategy

```
Full ABT (2000-2025)
  │
  ├── Out-of-Time (OOT) Test Set: year == 2025
  │
  └── Analytics Set: year < 2025
       │
       ├── Remove last 4 rounds per year (anti-leakage)
       │
       └── Stratified 80/20 split at (driver, year) level
            ├── Train Set (80%)
            └── Test Set (20%)
```

**Key anti-leakage measures:**

- **Year 2025 held out entirely** as an out-of-time validation set to evaluate generalization to unseen seasons.
- **Last 4 rounds of each season removed** from training data. By that point in a season, championship standings are nearly decided and features would be too predictive (essentially leaking the outcome).
- **Stratified split at (driver, year) level** ensures the same driver-season doesn't appear in both train and test sets.

#### 5.2. Feature Handling

```python
features = df_train.columns[4:]  # everything after driverid, dt_ref, flChampion, year
```

- **No explicit feature selection** — all ~180 features are passed to the model.
- **Missing value imputation:** uses feature_engine's `ArbitraryNumberImputer` with a fill value of **-10000**. This extreme value allows tree-based models to easily isolate missing-value splits.

#### 5.3. Model

```python
RandomForestClassifier(
    min_samples_leaf=50,
    n_estimators=500
)
```

- **Algorithm:** Random Forest Classifier — an ensemble of decision trees, robust to overfitting and able to handle high-dimensional feature spaces without feature selection.
- **`min_samples_leaf=50`:** Prevents trees from memorizing individual data points; each leaf must represent at least 50 observations.
- **`n_estimators=500`:** 500 trees in the ensemble for stable predictions.

#### 5.4. Evaluation

- **Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve) — appropriate for imbalanced binary classification (very few champions vs. many non-champions per season).
- Evaluated on three splits: **Train**, **Test**, and **OOT (2025)**.
- All metrics and the model artifact are logged to **MLFlow**.

#### 5.5. Final Model

After evaluation, the model is **re-trained on ALL data** (including 2025) before being saved to the MLFlow model registry. This maximizes the data available for the production model.

---

### 6. Model Serving

**Script:** `ml_champion/app.py`

A **Flask web application** that serves real-time predictions:

- On startup, loads the **latest registered model version** from MLFlow.
- Exposes a `/predict` endpoint that accepts driver feature vectors and returns championship probability scores.
- This allows integration with dashboards, bots, or other applications that want to display real-time championship predictions as the season progresses.

---

## Key Design Decisions

### Point-in-Time Feature Engineering

Features are computed at each race date using **only data available up to that moment**. This is critical for preventing future data leakage — the model never "sees" results that haven't happened yet at the time of prediction.

### Multi-Window Feature Strategy

Using 4 time windows (life, last 10, last 20, last 40 rounds) provides the model with complementary signals:

- **Life:** Is this driver historically elite? (e.g., Hamilton, Verstappen)
- **Last 40:** Has this driver been consistently strong over the past ~2 seasons?
- **Last 20:** Is this driver in strong form this season?
- **Last 10:** Is this driver peaking right now?

A driver might have modest career stats but exceptional recent form (rookie breakout), or great career numbers but declining recent performance (aging champion). The multi-window approach captures both scenarios.

### Eligible Driver Filtering

Only drivers who raced within the last 2 years of the reference date are included. This avoids computing features for retired drivers and keeps the dataset focused on active competitors.

### Anti-Leakage Temporal Splits

Removing the last 4 rounds of each season from training is a subtle but important decision. By round 20+ of a typical 22-race season, the points standings are nearly conclusive. Including these rounds would let the model learn patterns like "driver with 350+ points = champion," which is trivially true but useless for early-season predictions.

### Imbalanced Target

In any given season, only **1 out of ~20 drivers** is the champion, making this a highly imbalanced classification problem (~5% positive rate). The choice of ROC AUC as the evaluation metric is appropriate because it measures discrimination ability regardless of class distribution.
