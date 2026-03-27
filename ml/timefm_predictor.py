"""TimesFM-based predictor for F1 championship, team, and departure probabilities.

Google's TimesFM is a zero-shot time series foundation model. It takes a
historical sequence of numeric values per entity and forecasts future values.

Integration approach:
- For each entity (driver/team), we treat their race-by-race performance
  metric (e.g. rolling points, avg finish) as a time series context.
- TimesFM forecasts the next value in that series.
- Forecasted scores are normalized across all entities for a given race
  to produce probability-like outputs (softmax for championship, sigmoid
  for departure).
- No training is required — TimesFM is used zero-shot.

Requirements (Python 3.10–3.11 only — use .venv-timesfm):
    pip install timesfm   # auto-selects PyTorch backend on Python 3.11

The predictor gracefully falls back if timesfm is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMESFM_AVAILABLE = False
try:
    import timesfm  # noqa: F401
    TIMESFM_AVAILABLE = True
except ImportError:
    pass

# HuggingFace model repo — PyTorch backend, no JAX required
_HF_REPO = "google/timesfm-1.0-200m-pytorch"
_CONTEXT_LEN = 32   # number of past race values fed as context
_HORIZON_LEN = 1    # forecast one step ahead


def _load_timesfm():
    """Load TimesFM model (downloads weights on first call, ~800 MB)."""
    if not TIMESFM_AVAILABLE:
        raise ImportError(
            "timesfm is not installed. Run: pip install timesfm[torch]"
        )
    import timesfm  # noqa: F811

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="pytorch",
            per_core_batch_size=32,
            horizon_len=_HORIZON_LEN,
            context_len=_CONTEXT_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=_HF_REPO,
        ),
    )
    return tfm


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(scores - scores.max())
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_timeseries(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    score_col: str,
    normalize: str = "softmax",
) -> pd.DataFrame:
    """Run TimesFM on entity time series and return normalized probability scores.

    Args:
        df: DataFrame with one row per (entity, date), sorted by date.
        entity_col: Column identifying the entity (e.g. 'driverid', 'teamid').
        date_col: Column with the race/event date.
        score_col: Numeric column to use as the time series signal
                   (e.g. 'avg_points_last10', 'avg_position_life').
        normalize: 'softmax' (championship — probabilities sum to 1 per race)
                   or 'sigmoid' (departure — independent per driver).

    Returns:
        DataFrame with columns [entity_col, date_col, 'prob_timesfm'].
    """
    tfm = _load_timesfm()
    df = df.copy().sort_values([date_col, entity_col])

    entities = df[entity_col].unique()
    race_dates = sorted(df[date_col].unique())

    # Build per-entity time series: list of values up to each race date
    # For each (entity, race_date), context = all prior values for that entity
    records = []
    for entity in entities:
        entity_df = df[df[entity_col] == entity].sort_values(date_col)
        values = entity_df[score_col].fillna(0.0).tolist()
        dates = entity_df[date_col].tolist()

        for i, dt in enumerate(dates):
            context = values[max(0, i + 1 - _CONTEXT_LEN): i + 1]
            # Pad short contexts with the first value
            if len(context) < _CONTEXT_LEN:
                context = [context[0]] * (_CONTEXT_LEN - len(context)) + context
            records.append({
                entity_col: entity,
                date_col: dt,
                "context": context,
            })

    if not records:
        return pd.DataFrame(columns=[entity_col, date_col, "prob_timesfm"])

    # Batch forecast
    contexts = np.array([r["context"] for r in records], dtype=np.float32)
    # TimesFM expects (batch, context_len)
    forecasts, _ = tfm.forecast(contexts, freq=[0] * len(contexts))
    raw_scores = forecasts[:, 0]  # first (and only) forecast step

    results = []
    for idx, r in enumerate(records):
        results.append({
            entity_col: r[entity_col],
            date_col: r[date_col],
            "_raw_score": float(raw_scores[idx]),
        })

    result_df = pd.DataFrame(results)

    # Normalize per race date
    prob_list = []
    for dt in result_df[date_col].unique():
        mask = result_df[date_col] == dt
        scores = result_df.loc[mask, "_raw_score"].values
        if normalize == "softmax":
            probs = _softmax(scores)
        else:
            probs = _sigmoid(scores)
        prob_list.append(result_df[mask].assign(prob_timesfm=probs))

    out = pd.concat(prob_list, ignore_index=True)
    return out[[entity_col, date_col, "prob_timesfm"]]


# ---------------------------------------------------------------------------
# High-level helpers used by predict.py
# ---------------------------------------------------------------------------

def predict_champions_timesfm(silver_dir: str, bronze_path: str, year: int) -> pd.DataFrame:
    """TimesFM champion probability time series for a given season.

    Uses total_points_last10 as the performance signal — higher rolling points
    → higher forecasted score → higher championship probability after softmax.
    """
    import duckdb
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT f.driverid, f.dt_ref,
               COALESCE(f.total_points_last10, 0.0) AS score,
               b.full_name, b.team_name, b.team_color
        FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
        JOIN (
            SELECT DISTINCT driverid, full_name, team_name, team_color, event_date
            FROM read_parquet('{bronze_path}')
            WHERE EXTRACT(YEAR FROM event_date)::INT = {year}
              AND mode IN ('Race', 'Sprint Race', 'Sprint')
        ) b ON f.driverid = b.driverid AND f.dt_ref = b.event_date
        WHERE EXTRACT(YEAR FROM f.dt_ref)::INT = {year}
        ORDER BY f.dt_ref, f.driverid
    """).fetchdf()
    con.close()

    if df.empty:
        return df

    probs = predict_timeseries(df, "driverid", "dt_ref", "score", normalize="softmax")
    meta = df[["driverid", "dt_ref", "full_name", "team_name", "team_color"]].drop_duplicates()
    return probs.merge(meta, on=["driverid", "dt_ref"], how="left")


def predict_teams_timesfm(silver_dir: str, bronze_path: str, constructors_csv: str, year: int) -> pd.DataFrame:
    """TimesFM constructor champion probability time series for a given season."""
    import duckdb
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT
            b.teamid,
            b.team_name,
            b.event_date AS dt_ref,
            SUM(COALESCE(f.total_points_last10, 0.0)) AS score
        FROM read_parquet('{bronze_path}') b
        JOIN read_parquet('{silver_dir}/fs_driver_all.parquet') f
            ON f.driverid = b.driverid AND f.dt_ref = b.event_date
        WHERE EXTRACT(YEAR FROM b.event_date)::INT = {year}
          AND b.mode IN ('Race', 'Sprint Race', 'Sprint')
        GROUP BY b.teamid, b.team_name, b.event_date
        ORDER BY b.event_date, b.teamid
    """).fetchdf()
    con.close()

    if df.empty:
        return df

    probs = predict_timeseries(df, "teamid", "dt_ref", "score", normalize="softmax")
    meta = df[["teamid", "team_name", "dt_ref"]].drop_duplicates()
    return probs.merge(meta, on=["teamid", "dt_ref"], how="left")


def predict_departures_timesfm(silver_dir: str, bronze_path: str, year: int) -> pd.DataFrame:
    """TimesFM departure probability time series for a given season.

    Uses avg_position_life inverted (worse avg position → higher departure risk).
    Signal = -avg_position_life so sigmoid gives higher prob for worse performers.
    """
    import duckdb
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT f.driverid, f.dt_ref,
               -COALESCE(f.avg_position_life, 15.0) AS score,
               b.full_name, b.team_name, b.team_color
        FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
        JOIN (
            SELECT DISTINCT driverid, full_name, team_name, team_color, event_date
            FROM read_parquet('{bronze_path}')
            WHERE EXTRACT(YEAR FROM event_date)::INT = {year}
              AND mode IN ('Race', 'Sprint Race', 'Sprint')
        ) b ON f.driverid = b.driverid AND f.dt_ref = b.event_date
        WHERE EXTRACT(YEAR FROM f.dt_ref)::INT = {year}
        ORDER BY f.dt_ref, f.driverid
    """).fetchdf()
    con.close()

    if df.empty:
        return df

    probs = predict_timeseries(df, "driverid", "dt_ref", "score", normalize="sigmoid")
    meta = df[["driverid", "dt_ref", "full_name", "team_name", "team_color"]].drop_duplicates()
    return probs.merge(meta, on=["driverid", "dt_ref"], how="left")
