"""Load best models and generate time-series predictions. Used by Streamlit app.

All predict_* functions return one row per (entity, race_date) for the given
season, enabling line charts that show how probabilities evolve race by race.
"""

import os

import duckdb
import mlflow
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


_ALWAYS_EXCLUDE = {
    "dt_ref", "driverid", "teamid", "team_name", "year",
    "fl_champion", "fl_constructor_champion", "fl_departed",
    "prediction_year", "data_as_of", "season_race_number",
    "season_total_races", "season_fraction",
}


def _feature_cols(df, *extra_exclude):
    return [c for c in df.columns if c not in _ALWAYS_EXCLUDE | set(extra_exclude)]


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

    feat_cols = _feature_cols(features_df)
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

    feat_cols = _feature_cols(abt, "num_drivers")
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

    feat_cols = _feature_cols(abt)
    abt["prob_departure"] = model.predict_proba(abt[feat_cols])[:, 1]

    result = abt[["dt_ref", "driverid", "prob_departure",
                  "season_race_number", "season_fraction"]].merge(
        _driver_meta(), on="driverid", how="left"
    )
    result["year"] = year
    return result.sort_values(["dt_ref", "prob_departure"], ascending=[True, False])
