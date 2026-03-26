"""Load best models and generate predictions. Used by Streamlit app."""

import os

import duckdb
import mlflow
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze", "results.parquet")


def _get_tracking_uri():
    return f"file://{os.path.abspath(MLFLOW_DIR)}"


def load_best_model(experiment_name):
    """Load the best (final) model from a given MLFlow experiment."""
    mlflow.set_tracking_uri(_get_tracking_uri())
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Train models first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.final_model = 'true'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No final model found in experiment '{experiment_name}'.")

    run = runs[0]
    model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    return model, run.info.run_id


def get_model_comparison(experiment_name):
    """Get comparison metrics for all models in an experiment."""
    mlflow.set_tracking_uri(_get_tracking_uri())
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.final_model != 'true'",
        order_by=["start_time DESC"],
    )

    records = []
    for run in runs:
        records.append({
            "model": run.data.params.get("model_type", "unknown"),
            "mode": run.data.params.get("learning_mode", "batch"),
            "cv_auc": run.data.metrics.get("cv_auc_mean"),
            "tuned_cv_auc": run.data.metrics.get("tuned_cv_auc"),
            "auc_train": run.data.metrics.get("auc_train"),
            "auc_test": run.data.metrics.get("auc_test"),
            "auc_oot": run.data.metrics.get("auc_oot"),
            "is_best": run.data.tags.get("best_model") == "true",
        })

    return pd.DataFrame(records)


def predict_champions(model=None):
    """Generate champion probabilities from the gold ABT (all historical seasons).

    Each row represents a driver's pre-season snapshot for the following year.
    prediction_year = dt_ref.year + 1 (e.g., 2025 stats → 2026 prediction).
    """
    if model is None:
        model, _ = load_best_model("f1_champion")

    con = duckdb.connect()
    abt = con.execute(
        f"SELECT * FROM read_parquet('{os.path.join(GOLD_DIR, 'abt_champions.parquet')}')"
    ).fetchdf()
    driver_latest = con.execute(f"""
        SELECT driverid, full_name, team_name, team_color
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        )
        WHERE rn = 1
    """).fetchdf()
    con.close()

    features = [c for c in abt.columns if c not in
                {"dt_ref", "driverid", "year", "fl_champion"}]
    abt["prob_champion"] = model.predict_proba(abt[features])[:, 1]

    result = abt[["dt_ref", "driverid", "prob_champion"]].merge(
        driver_latest, on="driverid", how="left"
    )
    # prediction_year is the season being predicted (stats year + 1)
    result["prediction_year"] = pd.to_datetime(result["dt_ref"]).dt.year + 1
    return result


def predict_champion_season(year, model=None):
    """Predict 2026+ champion probabilities that update as the season progresses.

    Uses the most recent available features per driver from silver:
    - Before any races are collected: end-of-prior-year stats (pure pre-season)
    - After running etl.collect --years {year} + etl.silver: mid-season stats

    This lets the prediction update race-by-race as new data is collected.
    The model was trained on end-of-season snapshots so treat mid-season values
    as directional rather than calibrated probabilities.
    """
    if model is None:
        model, _ = load_best_model("f1_champion")

    con = duckdb.connect()

    # Get all available feature rows for this prediction year:
    # either from the prior year (pre-season) or from the current year (in-season)
    features_df = con.execute(f"""
        SELECT f.*
        FROM read_parquet('{os.path.join(SILVER_DIR, "fs_driver_all.parquet")}') f
        WHERE EXTRACT(YEAR FROM f.dt_ref)::INT <= {year - 1}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY f.driverid, EXTRACT(YEAR FROM f.dt_ref)::INT
            ORDER BY f.dt_ref DESC
        ) = 1
        AND EXTRACT(YEAR FROM f.dt_ref)::INT = (
            SELECT MAX(EXTRACT(YEAR FROM dt_ref)::INT)
            FROM read_parquet('{os.path.join(SILVER_DIR, "fs_driver_all.parquet")}')
            WHERE EXTRACT(YEAR FROM dt_ref)::INT <= {year - 1}
        )
    """).fetchdf()

    driver_latest = con.execute(f"""
        SELECT driverid, full_name, team_name, team_color
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        )
        WHERE rn = 1
    """).fetchdf()
    con.close()

    _EXCLUDE = {"dt_ref", "driverid", "fl_champion", "fl_constructor_champion", "fl_departed"}
    feature_cols = [c for c in features_df.columns if c not in _EXCLUDE]

    features_df["prob_champion"] = model.predict_proba(features_df[feature_cols])[:, 1]
    features_df["prediction_year"] = year
    features_df["data_as_of"] = features_df["dt_ref"]

    result = features_df[["dt_ref", "driverid", "prob_champion", "prediction_year", "data_as_of"]].merge(
        driver_latest, on="driverid", how="left"
    )
    return result.sort_values("prob_champion", ascending=False)


def predict_teams(model=None):
    """Generate constructor champion probabilities from the gold ABT.

    prediction_year = dt_ref.year + 1 (e.g., 2025 stats → 2026 prediction).
    """
    if model is None:
        model, _ = load_best_model("f1_constructor_champion")

    con = duckdb.connect()
    abt = con.execute(
        f"SELECT * FROM read_parquet('{os.path.join(GOLD_DIR, 'abt_teams.parquet')}')"
    ).fetchdf()
    con.close()

    features = [c for c in abt.columns if c not in
                {"dt_ref", "teamid", "team_name", "year", "fl_constructor_champion",
                 "num_drivers"}]
    abt["prob_constructor_champion"] = model.predict_proba(abt[features])[:, 1]
    abt["prediction_year"] = pd.to_datetime(abt["dt_ref"]).dt.year + 1

    return abt[["dt_ref", "teamid", "team_name", "prob_constructor_champion", "prediction_year"]]


def predict_departures(model=None):
    """Generate driver departure probabilities."""
    if model is None:
        model, _ = load_best_model("f1_departure")

    con = duckdb.connect()
    abt = con.execute(
        f"SELECT * FROM read_parquet('{os.path.join(GOLD_DIR, 'abt_departures.parquet')}')"
    ).fetchdf()

    driver_latest = con.execute(f"""
        SELECT driverid, full_name, team_name, team_color
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        )
        WHERE rn = 1
    """).fetchdf()
    con.close()

    features = [c for c in abt.columns if c not in
                {"dt_ref", "driverid", "year", "fl_departed"}]
    X = abt[features]
    abt["prob_departure"] = model.predict_proba(X)[:, 1]

    result = abt[["dt_ref", "driverid", "prob_departure"]].merge(
        driver_latest, on="driverid", how="left"
    )
    result["year"] = pd.to_datetime(result["dt_ref"]).dt.year
    return result
