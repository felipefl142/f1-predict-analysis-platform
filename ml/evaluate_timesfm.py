"""Evaluate TimesFM zero-shot forecasts and log results to MLflow.

Runs against the same gold inseason ABTs and OOT splits used by the
sklearn/XGBoost/CatBoost models so all approaches appear in the same
MLflow experiment for direct comparison.

Must be run with .venv-timesfm (Python 3.11, has timesfm + mlflow + duckdb):
    .venv-timesfm/bin/python -m ml.evaluate_timesfm              # all 3 targets
    .venv-timesfm/bin/python -m ml.evaluate_timesfm champion     # single target
    .venv-timesfm/bin/python -m ml.evaluate_timesfm constructor
    .venv-timesfm/bin/python -m ml.evaluate_timesfm departure
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import roc_auc_score, roc_curve

from ml.timefm_predictor import predict_timeseries, _CONTEXT_LEN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLFLOW_DB  = f"sqlite:///{BASE_DIR}/mlflow.db"
GOLD_DIR   = os.path.join(BASE_DIR, "data", "gold")

ABT_CHAMPION   = os.path.join(GOLD_DIR, "abt_champions_inseason.parquet")
ABT_TEAM       = os.path.join(GOLD_DIR, "abt_teams_inseason.parquet")
ABT_DEPARTURES = os.path.join(GOLD_DIR, "abt_departures_inseason.parquet")

# ---------------------------------------------------------------------------
# Helpers (inlined — cannot import ml.utils, it pulls in optuna which is
# absent from .venv-timesfm)
# ---------------------------------------------------------------------------

def _find_oot_year(df: pd.DataFrame, target_col: str) -> int:
    """Return the most recent year that has both classes in target_col."""
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["dt_ref"]).dt.year
    for year in sorted(tmp["year"].unique(), reverse=True):
        if tmp[tmp["year"] == year][target_col].nunique() >= 2:
            return int(year)
    return int(tmp["year"].max())


def _setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(experiment_name)


def _log_roc_curves(
    y_train: pd.Series | None,
    y_train_prob: np.ndarray | None,
    y_oot: pd.Series | None,
    y_oot_prob: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Compute AUC, log metrics, save ROC PNG artifact. Returns (auc_train, auc_oot)."""
    auc_train = auc_oot = None
    fig, ax = plt.subplots(dpi=100)
    legend = []

    if y_train is not None and y_train_prob is not None and len(np.unique(y_train)) >= 2:
        auc_train = roc_auc_score(y_train, y_train_prob)
        fpr, tpr, _ = roc_curve(y_train, y_train_prob)
        ax.plot(fpr, tpr)
        legend.append(f"Train (pre-OOT): {auc_train:.4f}")
        mlflow.log_metric("auc_train", auc_train)

    if y_oot is not None and y_oot_prob is not None and len(np.unique(y_oot)) >= 2:
        auc_oot = roc_auc_score(y_oot, y_oot_prob)
        fpr, tpr, _ = roc_curve(y_oot, y_oot_prob)
        ax.plot(fpr, tpr)
        legend.append(f"OOT: {auc_oot:.4f}")
        mlflow.log_metric("auc_oot", auc_oot)

    ax.legend(legend)
    ax.grid(True)
    ax.set_title("ROC Curve — TimesFM (zero-shot)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    roc_path = "/tmp/roc_curve_timesfm.png"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(roc_path)

    return auc_train, auc_oot


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _run_evaluation(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    target_col: str,
    signal_col: str,
    normalize: str,
    experiment_name: str,
    logged_signal_col: str | None = None,
) -> dict:
    """Evaluate TimesFM on one target and log a run to MLflow.

    Calls predict_timeseries on the FULL multi-year ABT so that OOT rows
    receive proper historical context from prior seasons, then evaluates
    AUC only on the OOT year rows.
    """
    oot_year = _find_oot_year(df, target_col)
    print(f"  OOT year: {oot_year}")
    print(f"  Running TimesFM predict_timeseries ({len(df)} rows, "
          f"signal={signal_col}, normalize={normalize})...")

    probs_df = predict_timeseries(df, entity_col, date_col, signal_col, normalize)

    # Merge probabilities back onto ABT to get labels alongside predictions
    merged = probs_df.merge(
        df[[entity_col, date_col, target_col]].drop_duplicates(),
        on=[entity_col, date_col],
        how="inner",
    )
    merged["year"] = pd.to_datetime(merged[date_col]).dt.year

    pre_oot = merged[merged["year"] < oot_year]
    oot     = merged[merged["year"] == oot_year]

    print(f"  Pre-OOT rows: {len(pre_oot)}, OOT rows: {len(oot)}")

    _setup_mlflow(experiment_name)

    with mlflow.start_run(run_name="zeroshotforecast_TimesFM"):
        mlflow.log_param("model_type",    "TimesFM")
        mlflow.log_param("learning_mode", "zero_shot")
        mlflow.log_param("signal_col",    logged_signal_col or signal_col)
        mlflow.log_param("normalize",     normalize)
        mlflow.log_param("context_len",   _CONTEXT_LEN)
        mlflow.log_param("oot_year",      oot_year)
        mlflow.log_param("n_oot_rows",    len(oot))
        mlflow.log_param("n_train_rows",  len(pre_oot))

        auc_train, auc_oot = _log_roc_curves(
            pre_oot[target_col] if len(pre_oot) > 0 else None,
            pre_oot["prob_timesfm"].values if len(pre_oot) > 0 else None,
            oot[target_col] if len(oot) > 0 else None,
            oot["prob_timesfm"].values if len(oot) > 0 else None,
        )

        run_id = mlflow.active_run().info.run_id

    print(f"  auc_train={auc_train}  auc_oot={auc_oot}  run_id={run_id}")
    return {
        "experiment": experiment_name,
        "auc_train":  auc_train,
        "auc_oot":    auc_oot,
        "run_id":     run_id,
    }


# ---------------------------------------------------------------------------
# Per-target entry points
# ---------------------------------------------------------------------------

def evaluate_champion() -> dict:
    print("\n--- F1 Driver Champion (TimesFM) ---")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_CHAMPION}')").fetchdf()
    con.close()
    return _run_evaluation(
        df, "driverid", "dt_ref", "fl_champion",
        signal_col="total_points_last10",
        normalize="softmax",
        experiment_name="f1_champion",
    )


def evaluate_constructor() -> dict:
    print("\n--- F1 Constructor Champion (TimesFM) ---")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_TEAM}')").fetchdf()
    con.close()
    return _run_evaluation(
        df, "teamid", "dt_ref", "fl_constructor_champion",
        signal_col="sum_points_last10",
        normalize="softmax",
        experiment_name="f1_constructor_champion",
    )


def evaluate_departure() -> dict:
    print("\n--- F1 Driver Departure (TimesFM) ---")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_DEPARTURES}')").fetchdf()
    con.close()
    # Negate avg_position_life: worse avg position → higher departure risk
    df = df.copy()
    df["_departure_signal"] = -df["avg_position_life"].fillna(15.0)
    return _run_evaluation(
        df, "driverid", "dt_ref", "fl_departed",
        signal_col="_departure_signal",
        normalize="sigmoid",
        experiment_name="f1_departure",
        logged_signal_col="avg_position_life (negated)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TARGETS = {
    "champion":    evaluate_champion,
    "constructor": evaluate_constructor,
    "departure":   evaluate_departure,
}

if __name__ == "__main__":
    requested = sys.argv[1:] or list(TARGETS.keys())

    invalid = [t for t in requested if t not in TARGETS]
    if invalid:
        print(f"Unknown target(s): {invalid}. Choose from: {list(TARGETS.keys())}")
        sys.exit(1)

    print("TimesFM Zero-Shot Evaluation")
    print("=" * 60)

    results = []
    for target in requested:
        results.append(TARGETS[target]())

    print("\n=== Summary ===")
    print(pd.DataFrame(results)[["experiment", "auc_train", "auc_oot"]].to_string(index=False))
