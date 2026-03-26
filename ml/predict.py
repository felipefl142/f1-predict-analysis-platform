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
    """Generate champion probabilities for current season."""
    if model is None:
        model, _ = load_best_model("f1_champion")

    con = duckdb.connect()
    abt = con.execute(
        f"SELECT * FROM read_parquet('{os.path.join(GOLD_DIR, 'abt_champions.parquet')}')"
    ).fetchdf()

    # Get driver metadata from bronze
    driver_meta = con.execute(f"""
        SELECT DISTINCT driverid, full_name, team_name, team_color
        FROM read_parquet('{BRONZE_PATH}')
    """).fetchdf()
    # Keep latest team info per driver
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
    X = abt[features]
    abt["prob_champion"] = model.predict_proba(X)[:, 1]

    result = abt[["dt_ref", "driverid", "prob_champion"]].merge(
        driver_latest, on="driverid", how="left"
    )
    result["year"] = pd.to_datetime(result["dt_ref"]).dt.year
    return result


def predict_teams(model=None):
    """Generate constructor champion probabilities."""
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
    X = abt[features]
    abt["prob_constructor_champion"] = model.predict_proba(X)[:, 1]

    return abt[["dt_ref", "teamid", "team_name", "prob_constructor_champion"]]


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
