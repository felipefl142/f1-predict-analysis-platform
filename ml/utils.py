"""Shared ML utilities: MLFlow setup, data splits, CV, Optuna hyperparam tuning."""

import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn import clone, metrics, model_selection

# Suppress Optuna's verbose trial logs (we log our own summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")          # artifact storage
MLFLOW_DB = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"  # metadata store

N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 30


def setup_mlflow(experiment_name):
    """Set up local MLFlow tracking and return the experiment."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)


def get_feature_columns(df, exclude_cols=None, drop_redundant=True):
    """Return feature column names (everything except metadata and target).

    When drop_redundant=True, removes last20/last40 window features and
    low-importance sprint features to reduce dimensionality.
    """
    if exclude_cols is None:
        exclude_cols = []
    always_exclude = {"dt_ref", "driverid", "year", "teamid", "team_name",
                      "fl_champion", "fl_constructor_champion", "fl_departed",
                      "prediction_year", "data_as_of"}
    exclude = always_exclude | set(exclude_cols)
    features = [c for c in df.columns if c not in exclude]

    if drop_redundant:
        # Drop last20 and last40 windows — keep life and last10 only
        features = [f for f in features
                    if not (f.endswith("_last20") or f.endswith("_last40"))]
        # Keep only high-value sprint features (wins, points, podiums, count)
        _keep_sprint = ("qtd_sprint_", "qtd_wins_sprint_",
                        "total_points_sprint_", "qtd_podiums_sprint_")
        features = [f for f in features
                    if "sprint" not in f or f.startswith(_keep_sprint)]

    return features


def _find_oot_year(df, target_col):
    """Find the most recent year that has both classes in the target."""
    df_temp = df.copy()
    df_temp["year"] = pd.to_datetime(df_temp["dt_ref"]).dt.year

    for year in sorted(df_temp["year"].unique(), reverse=True):
        year_data = df_temp[df_temp["year"] == year]
        if year_data[target_col].nunique() >= 2:
            return year

    return int(df_temp["year"].max())


def split_data(df, target_col, id_cols, oot_year=None, remove_late_rounds=True):
    """Split into train/test/OOT sets using a temporal strategy.

    OOT = most recent year with both classes.
    Test = second most recent year with both classes (before OOT).
    Train = all remaining years before test.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["dt_ref"]).dt.year

    if oot_year is None:
        oot_year = _find_oot_year(df, target_col)
    print(f"  OOT year: {oot_year}")

    df_oot = df[df["year"] == oot_year].copy()
    df_rest = df[df["year"] < oot_year].copy()

    # Test = most recent year before OOT with both classes
    test_year = _find_oot_year(df_rest, target_col)
    print(f"  Test year: {test_year}")

    df_test = df_rest[df_rest["year"] == test_year].copy()
    df_train = df_rest[df_rest["year"] < test_year].copy()

    # Remove last 4 rounds per year from training to avoid late-season
    # label leakage.  Skipped for in-season models (remove_late_rounds=False)
    # and when there is only 1 round per year (pre-season snapshot ABTs).
    if remove_late_rounds:
        df_year_round = df_train[["year", "dt_ref"]].drop_duplicates()
        if df_year_round.groupby("year").size().max() > 5:
            df_year_round["row_number"] = (
                df_year_round.sort_values("dt_ref", ascending=False)
                .groupby("year")
                .cumcount()
            )
            df_year_round = df_year_round[df_year_round["row_number"] > 4]
            df_year_round = df_year_round.drop("row_number", axis=1)
            df_train = df_train.merge(df_year_round, how="inner")

    print(f"  Split: train={len(df_train)}, test={len(df_test)}, oot={len(df_oot)}")
    return df_train, df_test, df_oot, oot_year


def _log_classification_metrics(y_true, y_prob, suffix):
    """Log PR-AUC, F1, log loss, and Brier score for a single split. Returns dict."""
    if len(np.unique(y_true)) < 2:
        return {}
    y_pred = (y_prob >= 0.5).astype(int)
    vals = {
        f"pr_auc_{suffix}": metrics.average_precision_score(y_true, y_prob),
        f"f1_{suffix}": metrics.f1_score(y_true, y_pred),
        f"log_loss_{suffix}": metrics.log_loss(y_true, y_prob),
        f"brier_{suffix}": metrics.brier_score_loss(y_true, y_prob),
    }
    for k, v in vals.items():
        mlflow.log_metric(k, v)
    return vals


def log_roc_curves(y_train, y_train_prob, y_test, y_test_prob,
                   y_oot=None, y_oot_prob=None):
    """Plot ROC + Precision-Recall curves and log metrics to MLFlow."""
    if len(np.unique(y_train)) < 2:
        return None, None, None, {}

    # --- ROC metrics ---
    auc_train = metrics.roc_auc_score(y_train, y_train_prob)
    roc_train = metrics.roc_curve(y_train, y_train_prob)
    mlflow.log_metric("auc_train", auc_train)

    if len(np.unique(y_test)) < 2:
        auc_test, roc_test = None, None
    else:
        auc_test = metrics.roc_auc_score(y_test, y_test_prob)
        roc_test = metrics.roc_curve(y_test, y_test_prob)
        mlflow.log_metric("auc_test", auc_test)

    auc_oot = None
    roc_oot = None
    if y_oot is not None and y_oot_prob is not None and len(np.unique(y_oot)) > 1:
        auc_oot = metrics.roc_auc_score(y_oot, y_oot_prob)
        roc_oot = metrics.roc_curve(y_oot, y_oot_prob)
        mlflow.log_metric("auc_oot", auc_oot)

    # --- PR curves ---
    pr_train = metrics.precision_recall_curve(y_train, y_train_prob)
    ap_train = metrics.average_precision_score(y_train, y_train_prob)

    pr_test, ap_test = None, None
    if roc_test is not None:
        pr_test = metrics.precision_recall_curve(y_test, y_test_prob)
        ap_test = metrics.average_precision_score(y_test, y_test_prob)

    pr_oot, ap_oot = None, None
    if roc_oot is not None:
        pr_oot = metrics.precision_recall_curve(y_oot, y_oot_prob)
        ap_oot = metrics.average_precision_score(y_oot, y_oot_prob)

    # --- Extra metrics (PR-AUC, F1, log loss, Brier) ---
    extra = {}
    extra.update(_log_classification_metrics(y_train, y_train_prob, "train"))
    extra.update(_log_classification_metrics(y_test, y_test_prob, "test"))
    if y_oot is not None and y_oot_prob is not None:
        extra.update(_log_classification_metrics(y_oot, y_oot_prob, "oot"))

    # --- Plot ROC + PR side by side ---
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # ROC subplot
    ax_roc.plot(roc_train[0], roc_train[1], label=f"Train: {auc_train:.4f}")
    if roc_test is not None:
        ax_roc.plot(roc_test[0], roc_test[1], label=f"Test: {auc_test:.4f}")
    if roc_oot is not None:
        ax_roc.plot(roc_oot[0], roc_oot[1], label=f"OOT: {auc_oot:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    ax_roc.grid(True)

    # PR subplot
    ax_pr.plot(pr_train[1], pr_train[0], label=f"Train AP: {ap_train:.4f}")
    if pr_test is not None:
        ax_pr.plot(pr_test[1], pr_test[0], label=f"Test AP: {ap_test:.4f}")
    if pr_oot is not None:
        ax_pr.plot(pr_oot[1], pr_oot[0], label=f"OOT AP: {ap_oot:.4f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()
    ax_pr.grid(True)

    curves_path = "/tmp/roc_pr_curves.png"
    fig.savefig(curves_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(curves_path)

    return auc_train, auc_test, auc_oot, extra


def _log_feature_importance_chart(fi_series, model_name):
    """Log a horizontal bar chart of feature importances to MLflow."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(fi_series) * 0.3)), dpi=100)
    fi_series = fi_series.sort_values(ascending=True)
    ax.barh(fi_series.index, fi_series.values)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances — {model_name}")
    ax.grid(True, axis="x", alpha=0.3)
    fi_chart_path = "/tmp/feature_importances.png"
    fig.savefig(fi_chart_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(fi_chart_path)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(pipeline, X, y, n_folds=N_CV_FOLDS, groups=None):
    """Run stratified K-fold CV and return mean/std AUC scores.

    When groups is provided, uses StratifiedGroupKFold so the same entity
    (e.g. driver) never appears in both train and validation within a fold.
    """
    # Cap folds so each validation fold has at least one positive example
    n_positives = int(y.sum())
    n_folds = min(n_folds, max(2, n_positives))

    if groups is not None:
        # Also cap by number of groups with positive labels
        pos_groups = groups[y == 1].nunique()
        n_folds = min(n_folds, max(2, pos_groups))
        cv = model_selection.StratifiedGroupKFold(n_splits=n_folds)
    else:
        cv = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # GPU models manage their own parallelism; running folds in parallel causes
    # multiple workers to compete for VRAM and OOM-abort. Use n_jobs=1 for GPU models.
    model = pipeline.named_steps.get("model")
    model_params = model.get_params() if hasattr(model, "get_params") else {}
    uses_gpu = (model_params.get("task_type") == "GPU"
                or model_params.get("device") in ("cuda", "gpu"))
    n_jobs = 1 if uses_gpu else -1
    cv_scores = model_selection.cross_val_score(
        pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=n_jobs,
        groups=groups, error_score=0.0,
    )
    # NaN occurs when a validation fold has only one class; treat as baseline
    baseline = y.mean() if hasattr(y, "mean") else 0.0
    cv_scores = np.where(np.isnan(cv_scores), baseline, cv_scores)
    return cv_scores.mean(), cv_scores.std(), cv_scores


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def _suggest_params(trial, model_name):
    """Define Optuna search space per model type. Returns dict of pipeline params."""
    if model_name == "LogisticRegression":
        return {
            "model__C": trial.suggest_float("C", 1e-4, 10, log=True),
            "model__l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "model__solver": "saga",
            "model__max_iter": 10000,
        }
    elif model_name == "BalancedRandomForest":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "model__max_depth": trial.suggest_int("max_depth", 3, 6),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 30, 150),
            "model__max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7],
            ),
        }
    elif model_name == "LightGBM":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "model__max_depth": trial.suggest_int("max_depth", 3, 5),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "model__early_stopping_rounds": 50,
        }
    elif model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "model__max_depth": trial.suggest_int("max_depth", 3, 5),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "model__min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "model__gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "model__early_stopping_rounds": 50,
        }
    return {}


def optuna_tune(pipeline, X, y, model_name, n_trials=N_OPTUNA_TRIALS,
                groups=None):
    """Run Optuna Bayesian optimization with TPE sampler and median pruner.

    When groups is provided, uses StratifiedGroupKFold so the same entity
    never appears in both train and validation within a fold.

    Returns:
        (best_pipeline, best_params, best_cv_auc)
    """
    # Cap folds so each validation fold has at least one positive example
    n_positives = int(y.sum())
    n_folds = min(N_CV_FOLDS, max(2, n_positives))

    if groups is not None:
        pos_groups = groups[y == 1].nunique()
        n_folds = min(n_folds, max(2, pos_groups))
        cv = model_selection.StratifiedGroupKFold(n_splits=n_folds)
    else:
        cv = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def objective(trial):
        params = _suggest_params(trial, model_name)
        if not params:
            return 0.0

        cloned = clone(pipeline)
        cloned.set_params(**params)

        # Use CV with pruning: evaluate fold-by-fold
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            fit_params = {}
            if model_name == "XGBoost":
                # Imputer is first step — transform train/val for eval_set
                imputer = cloned.named_steps["imputer"]
                X_val_imp = imputer.fit_transform(X_fold_val)
                fit_params["model__eval_set"] = [(X_val_imp, y_fold_val)]
                fit_params["model__verbose"] = False
            elif model_name == "LightGBM":
                imputer = cloned.named_steps["imputer"]
                X_val_imp = imputer.fit_transform(X_fold_val)
                fit_params["model__eval_set"] = [(X_val_imp, y_fold_val)]
            cloned.fit(X_fold_train, y_fold_train, **fit_params)
            y_pred = cloned.predict_proba(X_fold_val)[:, 1]
            if y_fold_val.nunique() < 2:
                # Validation fold has only one class; treat as baseline
                fold_score = float(y_fold_val.mean()) if len(y_fold_val) > 0 else 0.0
            else:
                fold_score = metrics.average_precision_score(y_fold_val, y_pred)

            scores.append(fold_score)

            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)

    try:
        best_raw_params = study.best_params
        best_cv_auc = study.best_value
    except ValueError:
        # All trials failed (e.g. dataset too small); fall back to default params
        print(f"    Warning: all Optuna trials failed for {model_name}; using default params.")
        pipeline.fit(X, y)
        return pipeline, {}, 0.0

    # Build best pipeline
    best_params = _suggest_params_from_dict(best_raw_params, model_name)
    best_pipeline = clone(pipeline)
    best_pipeline.set_params(**best_params)
    # Disable early stopping for final refit on full data (no eval_set)
    if model_name in ("XGBoost", "LightGBM"):
        best_pipeline.set_params(model__early_stopping_rounds=None)
    best_pipeline.fit(X, y)

    return best_pipeline, best_raw_params, best_cv_auc


def _suggest_params_from_dict(params_dict, model_name):
    """Convert Optuna's flat best_params dict back to pipeline param format."""
    mapped = {}
    for key, val in params_dict.items():
        mapped[f"model__{key}"] = val

    # Add fixed params not in the search space
    if model_name == "LogisticRegression":
        mapped["model__solver"] = "saga"
        mapped["model__max_iter"] = 10000

    return mapped


# ---------------------------------------------------------------------------
# Batch training with CV + Optuna tuning
# ---------------------------------------------------------------------------

def train_and_compare_batch(df, target_col, id_cols, experiment_name, candidates,
                            oot_year=None, remove_late_rounds=True):
    """Train all batch models with cross-validation and Optuna hyperparameter tuning.

    For each model:
    1. Stratified 5-fold CV with default params → cv_auc_mean
    2. Optuna TPE Bayesian search (30 trials, median pruner) → tuned_cv_auc
    3. Evaluate tuned model on held-out test → auc_test
    4. Evaluate on OOT → auc_oot

    Returns:
        (comparison DataFrame, best model name)
    """
    setup_mlflow(experiment_name)
    features = get_feature_columns(df, exclude_cols=id_cols)
    df_train, df_test, df_oot, oot_year = split_data(
        df, target_col, id_cols, oot_year,
        remove_late_rounds=remove_late_rounds,
    )

    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    # Groups for group-aware CV (prevent same entity in train + validation)
    groups_train = df_train[id_cols[0]] if id_cols else None

    results = []

    for name, pipeline in candidates.items():
        print(f"\n  [BATCH] Training {name}...")
        t_start = time.time()
        try:
            with mlflow.start_run(run_name=f"batch_{name}"):
                mlflow.log_param("model_type", name)
                mlflow.log_param("learning_mode", "batch")
                mlflow.log_param("tuning_method", "optuna_tpe")
                mlflow.log_param("n_features", len(features))
                mlflow.log_param("n_train_rows", len(X_train))
                mlflow.log_param("n_test_rows", len(X_test))
                mlflow.log_param("n_oot_rows", len(X_oot))
                mlflow.log_param("oot_year", oot_year)

                # --- Step 1: CV with default params ---
                print(f"    CV ({N_CV_FOLDS}-fold) default params...")
                cv_mean, cv_std, cv_scores = cross_validate_model(
                    pipeline, X_train, y_train, groups=groups_train,
                )
                mlflow.log_metric("cv_ap_mean", cv_mean)
                mlflow.log_metric("cv_ap_std", cv_std)
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i}", score)
                print(f"    Default CV AP: {cv_mean:.4f} (+/- {cv_std:.4f})")

                # --- Step 2: Optuna tuning ---
                print(f"    Optuna tuning ({N_OPTUNA_TRIALS} trials, TPE + median pruner)...")
                tuned_pipeline, best_params, tuned_cv_auc = optuna_tune(
                    pipeline, X_train, y_train, name,
                    n_trials=N_OPTUNA_TRIALS,
                    groups=groups_train,
                )

                if tuned_cv_auc > cv_mean:
                    pipeline = tuned_pipeline
                    print(f"    Tuned CV AP: {tuned_cv_auc:.4f} (improved from {cv_mean:.4f})")
                else:
                    tuned_cv_auc = cv_mean
                    # Disable early stopping for full fit (no eval_set)
                    if name in ("XGBoost", "LightGBM"):
                        pipeline.set_params(model__early_stopping_rounds=None)
                    pipeline.fit(X_train, y_train)
                    best_params = {}
                    print(f"    Tuned CV AP: {tuned_cv_auc:.4f} (kept defaults)")

                mlflow.log_metric("tuned_cv_ap", tuned_cv_auc)
                for param_key, param_val in best_params.items():
                    mlflow.log_param(f"best_{param_key}", param_val)

                # --- Step 3: Evaluate on test + OOT ---
                y_train_prob = pipeline.predict_proba(X_train)[:, 1]
                y_test_prob = pipeline.predict_proba(X_test)[:, 1]
                y_oot_prob = pipeline.predict_proba(X_oot)[:, 1] if len(X_oot) > 0 else None

                auc_train, auc_test, auc_oot, extra_metrics = log_roc_curves(
                    y_train, y_train_prob,
                    y_test, y_test_prob,
                    y_oot if y_oot_prob is not None else None,
                    y_oot_prob,
                )

                # Log feature importances if available
                model_step = pipeline.named_steps.get("model")
                if hasattr(model_step, "feature_importances_"):
                    fi = pd.Series(model_step.feature_importances_, index=features)
                    fi = fi.sort_values(ascending=False)
                    fi_path = "/tmp/feature_importances.md"
                    fi.head(30).to_markdown(fi_path)
                    mlflow.log_artifact(fi_path)
                    # Log top features as metrics for easy comparison in MLflow UI
                    for feat_name, importance in fi.head(30).items():
                        mlflow.log_metric(f"fi_{feat_name}", round(importance, 6))
                    # Log bar chart
                    _log_feature_importance_chart(fi.head(20), name)

                mlflow.sklearn.log_model(pipeline, artifact_path="model")

                elapsed = time.time() - t_start
                mlflow.log_metric("training_time_s", round(elapsed, 1))
                print(f"    Training time: {elapsed:.1f}s")

                results.append({
                    "model": name,
                    "mode": "batch",
                    "cv_ap": cv_mean,
                    "tuned_cv_ap": tuned_cv_auc,
                    "auc_train": auc_train,
                    "auc_test": auc_test,
                    "auc_oot": auc_oot,
                    "pr_auc_test": extra_metrics.get("pr_auc_test"),
                    "pr_auc_oot": extra_metrics.get("pr_auc_oot"),
                    "log_loss_test": extra_metrics.get("log_loss_test"),
                    "log_loss_oot": extra_metrics.get("log_loss_oot"),
                    "brier_test": extra_metrics.get("brier_test"),
                    "brier_oot": extra_metrics.get("brier_oot"),
                    "run_id": mlflow.active_run().info.run_id,
                })
        except Exception as e:
            print(f"    SKIPPED: {name} failed with error: {type(e).__name__}: {e}")
            continue

    comparison = pd.DataFrame(results)

    # Select best by OOT PR-AUC, fallback to tuned CV AP
    score_col = "pr_auc_oot" if comparison["pr_auc_oot"].notna().any() else "tuned_cv_ap"
    best_idx = comparison[score_col].fillna(0).idxmax()
    best_model_name = comparison.loc[best_idx, "model"]
    best_run_id = comparison.loc[best_idx, "run_id"]

    mlflow.MlflowClient().set_tag(best_run_id, "best_model", "true")

    auc_cols = ["model", "mode", "cv_ap", "tuned_cv_ap", "auc_train", "auc_test", "auc_oot"]
    extra_cols = ["model", "pr_auc_test", "pr_auc_oot", "log_loss_test", "log_loss_oot", "brier_test", "brier_oot"]
    print(f"\n  Batch model comparison (ROC-AUC):")
    print(comparison[auc_cols].to_string(index=False))
    print(f"\n  Batch model comparison (PR-AUC / calibration):")
    print(comparison[[c for c in extra_cols if c in comparison.columns]].to_string(index=False))
    print(f"\n  Best batch model: {best_model_name} (by {score_col})")

    # Retrain best model on ALL data with Optuna tuning
    print(f"  Retraining {best_model_name} on full dataset with Optuna tuning...")
    best_base_pipeline = candidates[best_model_name]
    all_X, all_y = df[features], df[target_col]
    all_groups = df[id_cols[0]] if id_cols else None

    final_pipeline, _, _ = optuna_tune(
        best_base_pipeline, all_X, all_y, best_model_name,
        n_trials=N_OPTUNA_TRIALS, groups=all_groups,
    )

    with mlflow.start_run(run_name=f"batch_{best_model_name}_final"):
        mlflow.log_param("model_type", f"{best_model_name}_final")
        mlflow.log_param("learning_mode", "batch")
        mlflow.log_param("trained_on", "all_data")
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        mlflow.set_tag("best_model", "true")
        mlflow.set_tag("final_model", "true")
        mlflow.set_tag("learning_mode", "batch")

    return comparison, best_model_name
