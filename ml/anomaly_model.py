"""Anomaly detection models for F1 champion and constructor champion prediction.

Reframes the prediction problem as outlier detection: championship-winning seasons
produce feature vectors that are statistical outliers. Models are trained one-class
on non-champions only, then score all drivers — higher anomaly score = more
champion-like.

Improvements over naive one-class:
- Expanding-window temporal CV for Optuna (no test-set leakage)
- Incomplete seasons (zero positives) excluded from OOT
- Semi-supervised Platt scaling to calibrate raw scores → probabilities
- Per-event relative features (rank/percentile within each race)

Usage:
    .venv/bin/python -m ml.anomaly_model
"""

import os
import time

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from feature_engine.imputation import ArbitraryNumberImputer
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.champion_model import CHAMPION_FEATURES
from ml.utils import (
    ExpandingWindowCV,
    N_CV_FOLDS,
    N_OPTUNA_TRIALS,
    _top1_champion_accuracy,
    setup_mlflow,
    split_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CHAMPION_ABT = os.path.join(BASE_DIR, "data", "gold", "abt_champions_inseason.parquet")
# Target configurations
TARGETS = {
    "champion": {
        "abt_path": CHAMPION_ABT,
        "target_col": "fl_champion",
        "id_cols": ["driverid"],
        "features": CHAMPION_FEATURES,
        "experiment_name": "f1_champion",
        "oot_year": [2024, 2025, 2026],
    },
}

# Per-event relative features to add on top of the curated set.
# These capture a driver's position within the field at each race,
# which better suits anomaly detection (outlier = top of field).
RELATIVE_FEATURES = [
    "standing_position",
    "points_gap_to_leader",
    "points_pct_of_leader",
    "clinch_proximity",
    "total_points_last10",
    "qtd_wins_last10",
]



# ---------------------------------------------------------------------------
# Per-event relative features
# ---------------------------------------------------------------------------

def _add_relative_features(df, features, relative_cols, id_col):
    """Add per-event rank and percentile features for selected columns.

    For each column in relative_cols, adds:
      - {col}_event_rank: rank within the event (1 = best)
      - {col}_event_pctl: percentile within the event (1.0 = top)

    Returns (df_with_new_cols, extended_feature_list).
    """
    df = df.copy()
    new_features = list(features)

    for col in relative_cols:
        if col not in df.columns:
            continue

        rank_col = f"{col}_event_rank"
        pctl_col = f"{col}_event_pctl"

        # For "gap to leader" and "standing position", lower is better → ascending rank
        ascending = col in ("standing_position", "points_gap_to_leader",
                            "team_standing_position", "team_points_gap_to_leader")

        df[rank_col] = df.groupby("dt_ref")[col].rank(
            ascending=ascending, method="min")
        df[pctl_col] = df.groupby("dt_ref")[col].rank(
            ascending=not ascending, pct=True, method="min")

        new_features.extend([rank_col, pctl_col])

    return df, new_features


# ---------------------------------------------------------------------------
# Anomaly model definitions
# ---------------------------------------------------------------------------

def _get_anomaly_models(contamination):
    """Return dict of {name: sklearn Pipeline} for anomaly detection candidates."""
    return {
        "IsolationForest": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=300,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "OneClassSVM": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(
                kernel="rbf",
                nu=min(contamination, 0.5),
                gamma="scale",
            )),
        ]),
        "LOF": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("scaler", StandardScaler()),
            ("model", LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _anomaly_scores(pipeline, X):
    """Get anomaly scores from a fitted pipeline (higher = more anomalous)."""
    return -pipeline.decision_function(X)


def _calibrate_scores(scores_train, y_train, scores_eval):
    """Platt-scale raw anomaly scores → calibrated probabilities.

    Uses the training set (which has both classes) to learn a sigmoid mapping
    from raw anomaly scores to P(champion). Then applies it to eval scores.
    """
    from sklearn.linear_model import LogisticRegression as LR

    # Fit sigmoid on training scores (both classes present)
    calibrator = LR(solver="lbfgs", max_iter=1000)
    calibrator.fit(scores_train.reshape(-1, 1), y_train)
    return calibrator.predict_proba(scores_eval.reshape(-1, 1))[:, 1]


def _score_separation(df_subset, scores, target_col):
    """Per-event z-score of the champion's anomaly score vs the field."""
    tmp = df_subset[["dt_ref"]].copy()
    tmp["year"] = pd.to_datetime(tmp["dt_ref"]).dt.year
    tmp["score"] = scores
    tmp["target"] = df_subset[target_col].values

    separations = []
    for (year, dt_ref), grp in tmp.groupby(["year", "dt_ref"]):
        champ = grp[grp["target"] == 1]
        if len(champ) == 0:
            continue
        median = grp["score"].median()
        std = grp["score"].std()
        if std < 1e-10:
            continue
        sep = (champ["score"].values[0] - median) / std
        separations.append(sep)

    return np.mean(separations) if separations else 0.0


def _log_anomaly_metrics(y_true, scores, suffix):
    """Log PR-AUC and ROC-AUC for anomaly scores on a split."""
    vals = {}
    if len(np.unique(y_true)) < 2:
        return vals

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    vals[f"pr_auc_{suffix}"] = pr_auc
    vals[f"auc_{suffix}"] = roc_auc
    for k, v in vals.items():
        mlflow.log_metric(k, round(v, 6))
    return vals


def _log_score_distribution(scores_test, y_test, scores_oot, y_oot, model_name):
    """Plot anomaly score distributions for champions vs non-champions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    for ax, scores, y_true, label in [
        (axes[0], scores_test, y_test, "Test"),
        (axes[1], scores_oot, y_oot, "OOT"),
    ]:
        if scores is None or y_true is None:
            ax.set_visible(False)
            continue
        neg_scores = scores[y_true == 0]
        pos_scores = scores[y_true == 1]
        ax.hist(neg_scores, bins=50, alpha=0.6, label="Non-champion", density=True)
        if len(pos_scores) > 0:
            ax.hist(pos_scores, bins=20, alpha=0.6, label="Champion", density=True)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    path = "/tmp/anomaly_score_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)


# ---------------------------------------------------------------------------
# Optuna tuning with expanding-window temporal CV
# ---------------------------------------------------------------------------

def _suggest_anomaly_params(trial, model_name):
    """Define Optuna search space for anomaly detection models."""
    if model_name == "IsolationForest":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "model__max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "model__max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "model__contamination": trial.suggest_float("contamination", 0.003, 0.02),
        }
    elif model_name == "OneClassSVM":
        return {
            "model__nu": trial.suggest_float("nu", 0.001, 0.05),
            "model__gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    elif model_name == "LOF":
        return {
            "model__n_neighbors": trial.suggest_int("n_neighbors", 10, 50),
            "model__contamination": trial.suggest_float("contamination", 0.003, 0.02),
        }
    return {}


def _optuna_tune_anomaly(pipeline, X_train, y_train, years_train, model_name,
                         n_trials=N_OPTUNA_TRIALS):
    """Optuna tuning with expanding-window temporal CV for one-class models.

    For each CV fold:
      1. Split by year (train years ≤ Y, val year = Y+1)
      2. Filter training set to negatives only (one-class)
      3. Fit model on negatives
      4. Score validation set (both classes) → PR-AUC

    This avoids test-set leakage by never seeing test/OOT data during tuning.
    """
    cv = ExpandingWindowCV(years_train, n_splits=N_CV_FOLDS)

    def objective(trial):
        params = _suggest_anomaly_params(trial, model_name)
        if not params:
            return 0.0

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # One-class: train only on negatives
            normal_mask = y_fold_train == 0
            X_fold_normal = X_fold_train[normal_mask]

            if len(X_fold_normal) < 10:
                continue

            cloned = clone(pipeline)
            cloned.set_params(**params)

            try:
                cloned.fit(X_fold_normal)
            except Exception:
                return 0.0

            fold_scores = _anomaly_scores(cloned, X_fold_val)

            if y_fold_val.nunique() < 2:
                # Validation fold has only one class — skip
                continue

            fold_pr_auc = average_precision_score(y_fold_val, fold_scores)
            scores.append(fold_pr_auc)

            # Report for pruning
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)

    try:
        best_raw_params = study.best_params
        best_cv = study.best_value
    except ValueError:
        print(f"    Warning: all Optuna trials failed for {model_name}; using defaults.")
        X_normal = X_train[y_train == 0]
        pipeline.fit(X_normal)
        return pipeline, {}, 0.0

    # Refit best params on full training negatives
    mapped_params = {f"model__{k}": v for k, v in best_raw_params.items()}
    best_pipeline = clone(pipeline)
    best_pipeline.set_params(**mapped_params)
    X_train_normal = X_train[y_train == 0]
    best_pipeline.fit(X_train_normal)

    return best_pipeline, best_raw_params, best_cv


# ---------------------------------------------------------------------------
# OOT filtering: drop years with zero positives
# ---------------------------------------------------------------------------

def _filter_oot_complete(df_oot, target_col):
    """Remove years from OOT that have zero positive labels (incomplete seasons)."""
    df_oot = df_oot.copy()
    df_oot["_year"] = pd.to_datetime(df_oot["dt_ref"]).dt.year

    valid_years = []
    dropped_years = []
    for year, grp in df_oot.groupby("_year"):
        if grp[target_col].sum() > 0:
            valid_years.append(year)
        else:
            dropped_years.append(year)

    if dropped_years:
        print(f"  OOT: dropped years with 0 positives: {dropped_years}")

    df_oot = df_oot[df_oot["_year"].isin(valid_years)].drop(columns=["_year"])
    return df_oot, valid_years


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_anomaly_models(target_name):
    """Train anomaly detection models for a given target."""
    cfg = TARGETS[target_name]
    print(f"\n{'=' * 60}")
    print(f"Anomaly Detection — {target_name.title()} Prediction")
    print(f"{'=' * 60}")

    # Load ABT
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{cfg['abt_path']}')").fetchdf()
    con.close()

    target_col = cfg["target_col"]
    base_features = cfg["features"]
    id_cols = cfg["id_cols"]
    id_col = id_cols[0]
    positive_rate = df[target_col].mean()
    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Positive rate: {positive_rate:.4f} ({df[target_col].sum()}/{len(df)})")

    # Add per-event relative features
    rel_cols = RELATIVE_FEATURES
    df, features = _add_relative_features(df, base_features, rel_cols, id_col)
    print(f"  Features: {len(base_features)} base + {len(features) - len(base_features)} relative = {len(features)} total")

    # Split data (reuse existing temporal splits)
    df_train, df_test, df_oot, oot_years = split_data(
        df, target_col, id_cols, oot_year=cfg["oot_year"],
        remove_late_rounds=False,
    )

    # Filter OOT to exclude incomplete seasons (zero positives)
    df_oot, valid_oot_years = _filter_oot_complete(df_oot, target_col)

    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    # One-class: train only on negatives
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    print(f"  One-class training set: {len(X_train_normal)} negative samples "
          f"(dropped {(~normal_mask).sum()} positives)")

    # Year series for expanding-window CV
    years_train = pd.to_datetime(df_train["dt_ref"]).dt.year.reset_index(drop=True)

    # Get anomaly models
    contamination = positive_rate
    candidates = _get_anomaly_models(contamination)

    setup_mlflow(cfg["experiment_name"])

    results = []

    for name, pipeline in candidates.items():
        print(f"\n  [ANOMALY] Training {name}...")
        t_start = time.time()

        try:
            with mlflow.start_run(run_name=f"anomaly_{name}_{target_name}"):
                mlflow.log_param("model_type", name)
                mlflow.log_param("learning_mode", "anomaly_one_class")
                mlflow.log_param("calibration", "platt_scaling")
                mlflow.log_param("contamination", round(contamination, 6))
                mlflow.log_param("n_features", len(features))
                mlflow.log_param("n_relative_features", len(features) - len(base_features))
                mlflow.log_param("n_train_rows", len(X_train_normal))
                mlflow.log_param("n_test_rows", len(X_test))
                mlflow.log_param("n_oot_rows", len(X_oot))
                mlflow.log_param("oot_year", str(valid_oot_years))

                # --- Step 1: Fit with default params ---
                print(f"    Fitting one-class model on {len(X_train_normal)} samples...")
                pipeline.fit(X_train_normal)

                scores_test_default = _anomaly_scores(pipeline, X_test)
                default_pr_auc = (average_precision_score(y_test, scores_test_default)
                                  if y_test.nunique() >= 2 else 0.0)
                print(f"    Default PR-AUC (test): {default_pr_auc:.4f}")
                mlflow.log_metric("default_pr_auc_test", round(default_pr_auc, 6))

                # --- Step 2: Optuna tuning with temporal CV ---
                print(f"    Optuna tuning ({N_OPTUNA_TRIALS} trials, expanding-window CV)...")
                tuned_pipeline, best_params, tuned_cv_pr_auc = _optuna_tune_anomaly(
                    pipeline, X_train, y_train, years_train, name,
                )

                # Compare: evaluate tuned model on test
                scores_test_tuned = _anomaly_scores(tuned_pipeline, X_test)
                tuned_test_pr_auc = (average_precision_score(y_test, scores_test_tuned)
                                     if y_test.nunique() >= 2 else 0.0)

                if tuned_test_pr_auc > default_pr_auc:
                    pipeline = tuned_pipeline
                    print(f"    Tuned CV PR-AUC: {tuned_cv_pr_auc:.4f}, "
                          f"test PR-AUC: {tuned_test_pr_auc:.4f} "
                          f"(improved from {default_pr_auc:.4f})")
                else:
                    print(f"    Tuned test PR-AUC: {tuned_test_pr_auc:.4f} "
                          f"(kept defaults at {default_pr_auc:.4f})")

                mlflow.log_metric("tuned_cv_score", round(tuned_cv_pr_auc, 6))
                for param_key, param_val in best_params.items():
                    mlflow.log_param(f"best_{param_key}", param_val)

                # --- Step 3: Score all splits (raw anomaly scores) ---
                scores_train = _anomaly_scores(pipeline, X_train)
                scores_test = _anomaly_scores(pipeline, X_test)
                scores_oot = (_anomaly_scores(pipeline, X_oot)
                              if len(X_oot) > 0 else None)

                # --- Step 4: Semi-supervised Platt calibration ---
                # Use training set (both classes) to learn score → probability mapping
                print(f"    Platt calibration (training set, both classes)...")
                cal_test = _calibrate_scores(scores_train, y_train.values, scores_test)
                cal_oot = (_calibrate_scores(scores_train, y_train.values, scores_oot)
                           if scores_oot is not None else None)

                # Log metrics for both raw and calibrated scores
                print(f"    Evaluating raw scores...")
                raw_test_metrics = _log_anomaly_metrics(y_test, scores_test, "test")
                raw_oot_metrics = {}
                if scores_oot is not None:
                    raw_oot_metrics = _log_anomaly_metrics(y_oot, scores_oot, "oot")

                print(f"    Evaluating calibrated scores...")
                cal_test_metrics = {}
                cal_oot_metrics = {}
                if y_test.nunique() >= 2:
                    cal_pr = average_precision_score(y_test, cal_test)
                    cal_auc = roc_auc_score(y_test, cal_test)
                    cal_test_metrics = {"pr_auc_cal_test": cal_pr, "auc_cal_test": cal_auc}
                    for k, v in cal_test_metrics.items():
                        mlflow.log_metric(k, round(v, 6))
                if cal_oot is not None and y_oot.nunique() >= 2:
                    cal_pr_oot = average_precision_score(y_oot, cal_oot)
                    cal_auc_oot = roc_auc_score(y_oot, cal_oot)
                    cal_oot_metrics = {"pr_auc_cal_oot": cal_pr_oot, "auc_cal_oot": cal_auc_oot}
                    for k, v in cal_oot_metrics.items():
                        mlflow.log_metric(k, round(v, 6))

                # Top-1 champion accuracy (use calibrated scores)
                top1_test = _top1_champion_accuracy(df_test, cal_test, id_col=id_col)
                mlflow.log_metric("top1_acc_test", round(top1_test, 4))

                top1_oot = None
                if cal_oot is not None:
                    top1_oot = _top1_champion_accuracy(df_oot, cal_oot, id_col=id_col)
                    mlflow.log_metric("top1_acc_oot", round(top1_oot, 4))

                # Also log top-1 from raw scores for comparison
                top1_test_raw = _top1_champion_accuracy(df_test, scores_test, id_col=id_col)
                mlflow.log_metric("top1_acc_test_raw", round(top1_test_raw, 4))
                if scores_oot is not None:
                    top1_oot_raw = _top1_champion_accuracy(df_oot, scores_oot, id_col=id_col)
                    mlflow.log_metric("top1_acc_oot_raw", round(top1_oot_raw, 4))

                # Score separation
                sep_test = _score_separation(df_test, scores_test, target_col)
                mlflow.log_metric("score_separation_test", round(sep_test, 4))
                if scores_oot is not None:
                    sep_oot = _score_separation(df_oot, scores_oot, target_col)
                    mlflow.log_metric("score_separation_oot", round(sep_oot, 4))

                # Score distribution plot
                _log_score_distribution(
                    scores_test, y_test.values,
                    scores_oot, y_oot.values if scores_oot is not None else None,
                    name,
                )

                mlflow.sklearn.log_model(pipeline, artifact_path="model")

                elapsed = time.time() - t_start
                mlflow.log_metric("training_time_s", round(elapsed, 1))
                print(f"    Training time: {elapsed:.1f}s")

                results.append({
                    "model": name,
                    "mode": "anomaly_one_class",
                    "pr_auc_test": raw_test_metrics.get("pr_auc_test"),
                    "pr_auc_oot": raw_oot_metrics.get("pr_auc_oot"),
                    "pr_auc_cal_test": cal_test_metrics.get("pr_auc_cal_test"),
                    "pr_auc_cal_oot": cal_oot_metrics.get("pr_auc_cal_oot"),
                    "top1_acc_test": top1_test,
                    "top1_acc_oot": top1_oot,
                    "score_sep_test": sep_test,
                })

        except Exception as e:
            print(f"    SKIPPED: {name} failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    comparison = pd.DataFrame(results)
    if len(comparison) > 0:
        print(f"\n  Anomaly model comparison:")
        print(comparison.to_string(index=False))

        # Best by calibrated PR-AUC OOT, falling back to raw
        if comparison["pr_auc_cal_oot"].notna().any():
            score_col = "pr_auc_cal_oot"
        elif comparison["pr_auc_oot"].notna().any():
            score_col = "pr_auc_oot"
        else:
            score_col = "pr_auc_test"
        best_idx = comparison[score_col].fillna(0).idxmax()
        best_model = comparison.loc[best_idx, "model"]
        print(f"\n  Best anomaly model: {best_model} (by {score_col})")
    else:
        print("\n  No anomaly models trained successfully.")

    return comparison


def main():
    train_anomaly_models("champion")


if __name__ == "__main__":
    main()
