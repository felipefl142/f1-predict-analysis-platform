"""Load best models and generate time-series predictions. Used by Streamlit app.

All predict_* functions return one row per (entity, race_date) for the given
season, enabling line charts that show how probabilities evolve race by race.

Online (adaptive) predict functions load a saved online model, learn from
completed seasons after its training cutoff, then predict the target year.
"""

import copy
import os
import pickle

import duckdb
import mlflow
import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MLFLOW_DB = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze", "results.parquet")


def _get_tracking_uri():
    return MLFLOW_DB


def load_best_model(experiment_name):
    """Load the best (final) model from a given MLFlow experiment."""
    mlflow.set_tracking_uri(_get_tracking_uri())
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Train models first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.final_model = 'true' AND tags.learning_mode != 'online'",
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


def _driver_meta():
    """Latest name/team per driver from bronze."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT driverid, full_name, team_name, team_color
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        )
        WHERE rn = 1
    """).fetchdf()
    con.close()
    return df


def _model_feature_cols(model):
    """Extract feature column names from a trained sklearn pipeline."""
    first_step = model[0] if hasattr(model, '__getitem__') else None
    if first_step is not None and hasattr(first_step, 'feature_names_in_'):
        return list(first_step.feature_names_in_)
    return None


# ---------------------------------------------------------------------------
# Champion predictions
# ---------------------------------------------------------------------------

def predict_champions(year: int, model=None) -> pd.DataFrame:
    """Race-by-race championship win probabilities for all drivers in `year`.

    Returns one row per (driverid, dt_ref) sorted by date then probability.
    Falls back to the prior season's last snapshot if no in-season data exists.
    """
    if model is None:
        model, _ = load_best_model("f1_champion")

    con = duckdb.connect()
    silver = os.path.join(SILVER_DIR, "fs_driver_all.parquet")

    in_season_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{silver}')
        WHERE EXTRACT(YEAR FROM dt_ref)::INT = {year}
    """).fetchone()[0]

    if in_season_count > 0:
        race_cal = con.execute(f"""
            SELECT event_date AS dt_ref,
                   ROW_NUMBER() OVER (ORDER BY event_date) AS season_race_number,
                   COUNT(*) OVER () AS season_total_races
            FROM (SELECT DISTINCT event_date FROM read_parquet('{BRONZE_PATH}')
                  WHERE mode IN ('Race','Sprint Race','Sprint') AND year = {year})
        """).fetchdf()
        features_df = con.execute(f"""
            SELECT f.* FROM read_parquet('{silver}') f
            WHERE EXTRACT(YEAR FROM f.dt_ref)::INT = {year}
        """).fetchdf()
    else:
        race_cal = con.execute(f"""
            SELECT event_date AS dt_ref,
                   ROW_NUMBER() OVER (ORDER BY event_date) AS season_race_number,
                   COUNT(*) OVER () AS season_total_races
            FROM (SELECT DISTINCT event_date FROM read_parquet('{BRONZE_PATH}')
                  WHERE mode IN ('Race','Sprint Race','Sprint') AND year = {year - 1})
        """).fetchdf()
        features_df = con.execute(f"""
            SELECT f.* FROM read_parquet('{silver}') f
            WHERE EXTRACT(YEAR FROM f.dt_ref)::INT = {year - 1}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY f.driverid ORDER BY f.dt_ref DESC) = 1
        """).fetchdf()

    con.close()

    features_df = features_df.merge(race_cal, on="dt_ref", how="left")
    features_df["season_fraction"] = (
        features_df["season_race_number"] / features_df["season_total_races"]
    ).round(3)

    feat_cols = _model_feature_cols(model)
    features_df["prob_champion"] = model.predict_proba(features_df[feat_cols])[:, 1]

    result = features_df[["dt_ref", "driverid", "prob_champion",
                           "season_race_number", "season_fraction"]].merge(
        _driver_meta(), on="driverid", how="left"
    )
    result["year"] = year
    return result.sort_values(["dt_ref", "prob_champion"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Team predictions
# ---------------------------------------------------------------------------

def predict_teams(year: int, model=None) -> pd.DataFrame:
    """Race-by-race constructor championship probabilities for all teams in `year`.

    Returns one row per (teamid, dt_ref).
    Falls back to prior season snapshot if no in-season data exists.
    """
    if model is None:
        model, _ = load_best_model("f1_constructor_champion")

    con = duckdb.connect()
    abt_path = os.path.join(GOLD_DIR, "abt_teams_inseason.parquet")

    in_season_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{abt_path}')
        WHERE EXTRACT(YEAR FROM dt_ref)::INT = {year}
    """).fetchone()[0]

    if in_season_count > 0:
        abt = con.execute(f"""
            SELECT * FROM read_parquet('{abt_path}')
            WHERE EXTRACT(YEAR FROM dt_ref)::INT = {year}
        """).fetchdf()
    else:
        abt = con.execute(f"""
            SELECT * FROM read_parquet('{abt_path}')
            WHERE EXTRACT(YEAR FROM dt_ref)::INT = {year - 1}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY teamid ORDER BY dt_ref DESC) = 1
        """).fetchdf()

    con.close()

    feat_cols = _model_feature_cols(model)
    abt["prob_constructor_champion"] = model.predict_proba(abt[feat_cols])[:, 1]
    abt["year"] = year

    return abt[["dt_ref", "teamid", "team_name",
                "prob_constructor_champion",
                "season_race_number", "season_fraction", "year"]].sort_values(
        ["dt_ref", "prob_constructor_champion"], ascending=[True, False]
    )


# ---------------------------------------------------------------------------
# Departure predictions
# ---------------------------------------------------------------------------

def predict_departures(year: int | None = None, model=None) -> pd.DataFrame:
    """Race-by-race departure probabilities for all drivers in `year`.

    Defaults to the most recent complete season if year is None.
    Returns one row per (driverid, dt_ref).
    """
    if model is None:
        model, _ = load_best_model("f1_departure")

    con = duckdb.connect()
    abt_path = os.path.join(GOLD_DIR, "abt_departures_inseason.parquet")

    if year is None:
        year = con.execute(f"""
            SELECT MAX(EXTRACT(YEAR FROM dt_ref)::INT) FROM read_parquet('{abt_path}')
        """).fetchone()[0]

    abt = con.execute(f"""
        SELECT * FROM read_parquet('{abt_path}')
        WHERE EXTRACT(YEAR FROM dt_ref)::INT = {year}
    """).fetchdf()
    con.close()

    if abt.empty:
        return pd.DataFrame()

    feat_cols = _model_feature_cols(model)
    abt["prob_departure"] = model.predict_proba(abt[feat_cols])[:, 1]

    result = abt[["dt_ref", "driverid", "prob_departure",
                  "season_race_number", "season_fraction"]].merge(
        _driver_meta(), on="driverid", how="left"
    )
    result["year"] = year
    return result.sort_values(["dt_ref", "prob_departure"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Online (adaptive) models — predict-then-learn
# ---------------------------------------------------------------------------

def load_best_online_model(experiment_name):
    """Load the best online model (pickle artifact) from a given MLFlow experiment.

    Returns (model_info dict, run_id) or (None, None) if not found.
    """
    mlflow.set_tracking_uri(_get_tracking_uri())
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None, None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.final_model = 'true' AND tags.learning_mode = 'online'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, None

    run = runs[0]
    local_dir = client.download_artifacts(run.info.run_id, "model")
    pkl_path = os.path.join(local_dir, "online_model.pkl")
    with open(pkl_path, "rb") as f:
        model_info = pickle.load(f)

    return model_info, run.info.run_id


def _online_learn(model_info, X, y):
    """Update an online model with new labeled data."""
    if model_info["type"] == "sklearn":
        pipeline = model_info["model"]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        X_scaled = scaler.transform(X.fillna(-10000))
        model.partial_fit(X_scaled, y.values, classes=np.array([0, 1]))
    else:
        model = model_info["model"]
        X_filled = X.fillna(-10000)
        for i in range(len(X_filled)):
            model.learn_one(X_filled.iloc[i].to_dict(), int(y.iloc[i]))


def _online_predict(model_info, X):
    """Get probability predictions from an online model."""
    if model_info["type"] == "sklearn":
        pipeline = model_info["model"]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        X_scaled = scaler.transform(X.fillna(-10000))
        # SGDClassifier decision margins can be extreme (±600+), making
        # predict_proba/sigmoid saturate to 0/1.  Normalize margins to
        # zero-mean unit-variance before applying sigmoid so the output
        # spans a useful probability range.
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_scaled)
            std = raw.std()
            if std > 0:
                raw = (raw - raw.mean()) / std
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        return model.predict_proba(X_scaled)[:, 1]
    else:
        model = model_info["model"]
        X_filled = X.fillna(-10000)
        probs = []
        for i in range(len(X_filled)):
            p = model.predict_proba_one(X_filled.iloc[i].to_dict())
            probs.append(p.get(1, 0.0))
        return np.array(probs)


def _predict_online_generic(abt_filename, target_col, id_col, prob_col,
                             experiment_name, year, meta_merge=None):
    """Generic adaptive online prediction for any target.

    1. Loads saved online model (trained on data < training cutoff).
    2. Learns from all completed seasons between cutoff and prediction year.
    3. Predicts the requested year.
    """
    model_info, run_id = load_best_online_model(experiment_name)
    if model_info is None:
        raise ValueError(f"No online model for '{experiment_name}'. Train online models first.")

    # Deep copy so cached/shared references are not mutated
    model_info = copy.deepcopy(model_info)

    # Load full ABT
    con = duckdb.connect()
    abt_path = os.path.join(GOLD_DIR, abt_filename)
    abt = con.execute(
        f"SELECT * FROM read_parquet('{abt_path}') ORDER BY dt_ref"
    ).fetchdf()
    con.close()

    from ml.utils import get_feature_columns
    features = get_feature_columns(abt, exclude_cols=[id_col])

    abt["year"] = pd.to_datetime(abt["dt_ref"]).dt.year

    # Get training cutoff from MLflow run params
    client = mlflow.MlflowClient()
    run_data = client.get_run(run_id)
    train_max_year = int(run_data.data.params.get("train_max_year", 0))

    # Learn from completed seasons after training cutoff
    learn_data = abt[(abt["year"] > train_max_year) & (abt["year"] < year)]
    learn_data = learn_data.sort_values("dt_ref")
    if not learn_data.empty:
        _online_learn(model_info, learn_data[features], learn_data[target_col])

    # Predict for target year
    pred_data = abt[abt["year"] == year]
    if pred_data.empty:
        return pd.DataFrame()

    probs = _online_predict(model_info, pred_data[features])

    # Build result with same columns as batch predict functions
    result = pred_data[["dt_ref", id_col]].copy()
    result[prob_col] = probs
    for col in ("season_race_number", "season_fraction", "team_name"):
        if col in pred_data.columns:
            result[col] = pred_data[col].values

    result["year"] = year

    if meta_merge is not None:
        result = meta_merge(result)

    return result.sort_values(["dt_ref", prob_col], ascending=[True, False])


def predict_champions_online(year: int) -> pd.DataFrame:
    """Race-by-race championship predictions using adaptive online learning."""
    return _predict_online_generic(
        "abt_champions_inseason.parquet", "fl_champion", "driverid",
        "prob_champion", "f1_champion", year,
        meta_merge=lambda df: df.merge(_driver_meta(), on="driverid", how="left"),
    )


def predict_teams_online(year: int) -> pd.DataFrame:
    """Race-by-race constructor championship predictions using adaptive online learning."""
    return _predict_online_generic(
        "abt_teams_inseason.parquet", "fl_constructor_champion", "teamid",
        "prob_constructor_champion", "f1_constructor_champion", year,
    )


def predict_departures_online(year: int) -> pd.DataFrame:
    """Race-by-race departure predictions using adaptive online learning."""
    return _predict_online_generic(
        "abt_departures_inseason.parquet", "fl_departed", "driverid",
        "prob_departure", "f1_departure", year,
        meta_merge=lambda df: df.merge(_driver_meta(), on="driverid", how="left"),
    )
