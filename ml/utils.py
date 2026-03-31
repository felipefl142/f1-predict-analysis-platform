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

    oot_year can be a single int or a list of ints (multi-year OOT).
    OOT = most recent year(s) with both classes (or explicit).
    Test = most recent year before OOT with both classes.
    Train = all remaining years before test.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["dt_ref"]).dt.year

    # Normalize oot_year to a sorted list
    if oot_year is None:
        oot_years = [_find_oot_year(df, target_col)]
    elif isinstance(oot_year, (list, tuple)):
        oot_years = sorted(oot_year)
    else:
        oot_years = [oot_year]
    print(f"  OOT year(s): {oot_years}")

    df_oot = df[df["year"].isin(oot_years)].copy()
    df_rest = df[df["year"] < min(oot_years)].copy()

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
    return df_train, df_test, df_oot, oot_years


def _top1_champion_accuracy(df_subset, y_prob, champions_csv="data/champions.csv"):
    """At each race event, check if the top-predicted driver is the actual champion.

    Returns fraction of events where the model's #1 pick is correct.
    Ignores years not in the champions CSV (e.g. incomplete seasons).
    """
    champs = pd.read_csv(champions_csv)
    champ_map = dict(zip(champs["year"], champs["driverid"]))

    tmp = df_subset[["driverid", "dt_ref"]].copy()
    tmp["year"] = pd.to_datetime(tmp["dt_ref"]).dt.year
    tmp["prob"] = y_prob

    correct = 0
    total = 0
    for (year, dt_ref), grp in tmp.groupby(["year", "dt_ref"]):
        if year not in champ_map:
            continue
        top_driver = grp.loc[grp["prob"].idxmax(), "driverid"]
        if top_driver == champ_map[year]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


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

class ExpandingWindowCV:
    """Expanding-window time-series CV splitter based on year groups.

    Given sorted unique years [Y0, Y1, ..., Yn], generates n_splits folds
    using the last n_splits years as validation sets:
        Fold 1: train = years up to Y(n - n_splits),  val = Y(n - n_splits + 1)
        Fold 2: train = years up to Y(n - n_splits+1), val = Y(n - n_splits + 2)
        ...
        Fold n_splits: train = years up to Y(n-1),     val = Y(n)
    """

    def __init__(self, years, n_splits=N_CV_FOLDS):
        self.years = years
        unique = sorted(years.unique())
        # Need at least 1 training year + n_splits validation years
        max_splits = len(unique) - 1
        self.n_splits = min(n_splits, max(2, max_splits))
        self._unique_years = unique

    def split(self, X=None, y=None, groups=None):
        unique = self._unique_years
        for i in range(self.n_splits):
            val_year = unique[len(unique) - self.n_splits + i]
            train_mask = self.years < val_year
            val_mask = self.years == val_year
            yield np.where(train_mask)[0], np.where(val_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def cross_validate_model(pipeline, X, y, years, n_folds=N_CV_FOLDS,
                         scoring="average_precision"):
    """Run expanding-window time-series CV and return mean/std scores.

    Uses ExpandingWindowCV: each fold trains on all years up to year Y
    and validates on year Y+1, with an expanding training window.

    Args:
        scoring: sklearn scoring string — "average_precision" or "roc_auc".
    """
    cv = ExpandingWindowCV(years, n_splits=n_folds)

    # GPU models manage their own parallelism; running folds in parallel causes
    # multiple workers to compete for VRAM and OOM-abort. Use n_jobs=1 for GPU models.
    model = pipeline.named_steps.get("model")
    model_params = model.get_params() if hasattr(model, "get_params") else {}
    uses_gpu = (model_params.get("task_type") == "GPU"
                or model_params.get("device") in ("cuda", "gpu"))
    n_jobs = 1 if uses_gpu else -1
    cv_scores = model_selection.cross_val_score(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        error_score=0.0,
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
            "model__l1_ratio": trial.suggest_float("l1_ratio", 0.2, 1.0),
            "model__solver": "saga",
            "model__max_iter": 3500,
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
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "model__num_leaves": trial.suggest_int("num_leaves", 4, 20),
            "model__max_depth": trial.suggest_int("max_depth", 2, 5),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "model__early_stopping_rounds": 30,
        }
    elif model_name == "XGBoost":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "model__max_depth": trial.suggest_int("max_depth", 2, 6),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "model__min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "model__gamma": trial.suggest_float("gamma", 0.01, 5.0, log=True),
            "model__early_stopping_rounds": 30,
        }
    elif model_name == "AdaBoost":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
            "model__estimator__max_depth": trial.suggest_int("max_depth", 1, 4),
        }
    return {}


def optuna_tune(pipeline, X, y, model_name, years, n_trials=N_OPTUNA_TRIALS,
                scoring="average_precision"):
    """Run Optuna Bayesian optimization with TPE sampler and median pruner.

    Uses ExpandingWindowCV for temporal cross-validation.

    Args:
        scoring: "average_precision" or "roc_auc".

    Returns:
        (best_pipeline, best_params, best_cv_auc)
    """
    cv = ExpandingWindowCV(years, n_splits=N_CV_FOLDS)

    _SCORE_FN = {
        "average_precision": metrics.average_precision_score,
        "roc_auc": metrics.roc_auc_score,
        "f1": lambda y_true, y_prob: metrics.f1_score(y_true, (y_prob >= 0.5).astype(int)),
    }
    score_fn = _SCORE_FN[scoring]

    def objective(trial):
        params = _suggest_params(trial, model_name)
        if not params:
            return 0.0

        cloned = clone(pipeline)
        cloned.set_params(**params)

        # Use CV with pruning: evaluate fold-by-fold
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
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
            try:
                cloned.fit(X_fold_train, y_fold_train, **fit_params)
            except Exception:
                # LightGBM can crash with certain param combos on small datasets
                # (e.g. left_count assertion failure). Return worst score for trial.
                return 0.0
            y_pred = cloned.predict_proba(X_fold_val)[:, 1]
            if y_fold_val.nunique() < 2:
                # Validation fold has only one class; treat as baseline
                fold_score = float(y_fold_val.mean()) if len(y_fold_val) > 0 else 0.0
            else:
                fold_score = score_fn(y_fold_val, y_pred)

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
    # AdaBoost's max_depth belongs to the base estimator, not AdaBoost itself
    _NESTED_PARAMS = {
        "AdaBoost": {"max_depth": "model__estimator__max_depth"},
    }
    nested = _NESTED_PARAMS.get(model_name, {})

    mapped = {}
    for key, val in params_dict.items():
        if key in nested:
            mapped[nested[key]] = val
        else:
            mapped[f"model__{key}"] = val

    # Add fixed params not in the search space
    if model_name == "LogisticRegression":
        mapped["model__solver"] = "saga"
        mapped["model__max_iter"] = 3500

    return mapped


# ---------------------------------------------------------------------------
# Batch training with CV + Optuna tuning
# ---------------------------------------------------------------------------

def train_and_compare_batch(df, target_col, id_cols, experiment_name, candidates,
                            oot_year=None, remove_late_rounds=True,
                            scoring="average_precision", feature_cols=None):
    """Train all batch models with cross-validation and Optuna hyperparameter tuning.

    For each model:
    1. Stratified 5-fold CV with default params → cv_auc_mean
    2. Optuna TPE Bayesian search (30 trials, median pruner) → tuned_cv_auc
    3. Evaluate tuned model on held-out test → auc_test
    4. Evaluate on OOT → auc_oot

    Args:
        scoring: "average_precision" (optimize PR-AUC) or "roc_auc" (optimize ROC-AUC).
        feature_cols: explicit list of feature columns. If None, auto-detect.

    Returns:
        (comparison DataFrame, best model name)
    """
    setup_mlflow(experiment_name)
    features = feature_cols if feature_cols is not None else get_feature_columns(df, exclude_cols=id_cols)
    df_train, df_test, df_oot, oot_years = split_data(
        df, target_col, id_cols, oot_year,
        remove_late_rounds=remove_late_rounds,
    )

    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    # Year series for expanding-window temporal CV
    years_train = pd.to_datetime(df_train["dt_ref"]).dt.year.reset_index(drop=True)

    results = []

    # Compute scale_pos_weight for gradient boosting models
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  Class balance: {n_neg} neg / {n_pos} pos → scale_pos_weight={scale_pos_weight:.2f}")

    for name, pipeline in candidates.items():
        has_sampler = "sampler" in pipeline.named_steps
        if name in ("XGBoost", "LightGBM") and not has_sampler:
            pipeline.set_params(model__scale_pos_weight=scale_pos_weight)
        # AdaBoost handles imbalance via sample_weight in SAMME, no extra param needed
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
                mlflow.log_param("oot_year", str(oot_years))
                mlflow.log_param("scoring", scoring)

                score_label = "AUC" if scoring == "roc_auc" else "AP"

                # --- Step 1: CV with default params ---
                n_folds_actual = ExpandingWindowCV(years_train).n_splits
                print(f"    CV ({n_folds_actual}-fold expanding window) default params...")
                cv_mean, cv_std, cv_scores = cross_validate_model(
                    pipeline, X_train, y_train, years=years_train,
                    scoring=scoring,
                )
                mlflow.log_metric("cv_score_mean", cv_mean)
                mlflow.log_metric("cv_score_std", cv_std)
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i}", score)
                print(f"    Default CV {score_label}: {cv_mean:.4f} (+/- {cv_std:.4f})")

                # --- Step 2: Optuna tuning ---
                print(f"    Optuna tuning ({N_OPTUNA_TRIALS} trials, TPE + median pruner)...")
                tuned_pipeline, best_params, tuned_cv_auc = optuna_tune(
                    pipeline, X_train, y_train, name,
                    years=years_train,
                    n_trials=N_OPTUNA_TRIALS,
                    scoring=scoring,
                )

                if tuned_cv_auc > cv_mean:
                    pipeline = tuned_pipeline
                    print(f"    Tuned CV {score_label}: {tuned_cv_auc:.4f} (improved from {cv_mean:.4f})")
                else:
                    tuned_cv_auc = cv_mean
                    # Disable early stopping for full fit (no eval_set)
                    if name in ("XGBoost", "LightGBM"):
                        pipeline.set_params(model__early_stopping_rounds=None)
                    pipeline.fit(X_train, y_train)
                    best_params = {}
                    print(f"    Tuned CV {score_label}: {tuned_cv_auc:.4f} (kept defaults)")

                mlflow.log_metric("tuned_cv_score", tuned_cv_auc)
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

                # Top-1 champion accuracy (per-event)
                top1_test = _top1_champion_accuracy(df_test, y_test_prob)
                mlflow.log_metric("top1_acc_test", round(top1_test, 4))
                extra_metrics["top1_acc_test"] = top1_test
                if y_oot_prob is not None:
                    top1_oot = _top1_champion_accuracy(df_oot, y_oot_prob)
                    mlflow.log_metric("top1_acc_oot", round(top1_oot, 4))
                    extra_metrics["top1_acc_oot"] = top1_oot

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
                    "cv_score": cv_mean,
                    "tuned_cv_score": tuned_cv_auc,
                    "auc_train": auc_train,
                    "auc_test": auc_test,
                    "auc_oot": auc_oot,
                    "pr_auc_test": extra_metrics.get("pr_auc_test"),
                    "pr_auc_oot": extra_metrics.get("pr_auc_oot"),
                    "log_loss_test": extra_metrics.get("log_loss_test"),
                    "log_loss_oot": extra_metrics.get("log_loss_oot"),
                    "brier_test": extra_metrics.get("brier_test"),
                    "brier_oot": extra_metrics.get("brier_oot"),
                    "top1_acc_test": extra_metrics.get("top1_acc_test"),
                    "top1_acc_oot": extra_metrics.get("top1_acc_oot"),
                    "run_id": mlflow.active_run().info.run_id,
                })
        except Exception as e:
            print(f"    SKIPPED: {name} failed with error: {type(e).__name__}: {e}")
            continue

    comparison = pd.DataFrame(results)

    # Select best by OOT metric matching the scoring strategy
    if scoring == "roc_auc":
        score_col = "auc_oot" if comparison["auc_oot"].notna().any() else "tuned_cv_score"
    elif scoring == "f1":
        score_col = "f1_oot" if "f1_oot" in comparison.columns and comparison["f1_oot"].notna().any() else "tuned_cv_score"
    else:
        score_col = "pr_auc_oot" if comparison["pr_auc_oot"].notna().any() else "tuned_cv_score"
    best_idx = comparison[score_col].fillna(0).idxmax()
    best_model_name = comparison.loc[best_idx, "model"]
    best_run_id = comparison.loc[best_idx, "run_id"]

    mlflow.MlflowClient().set_tag(best_run_id, "best_model", "true")

    auc_cols = ["model", "mode", "cv_score", "tuned_cv_score", "auc_train", "auc_test", "auc_oot"]
    extra_cols = ["model", "pr_auc_test", "pr_auc_oot", "log_loss_test", "log_loss_oot", "brier_test", "brier_oot"]
    top1_cols = ["model", "top1_acc_test", "top1_acc_oot"]
    print(f"\n  Batch model comparison (ROC-AUC):")
    print(comparison[auc_cols].to_string(index=False))
    print(f"\n  Batch model comparison (PR-AUC / calibration):")
    print(comparison[[c for c in extra_cols if c in comparison.columns]].to_string(index=False))
    if "top1_acc_test" in comparison.columns:
        print(f"\n  Batch model comparison (Top-1 champion accuracy per event):")
        print(comparison[[c for c in top1_cols if c in comparison.columns]].to_string(index=False))
    print(f"\n  Best batch model: {best_model_name} (by {score_col})")

    # Retrain best model on ALL data with Optuna tuning
    print(f"  Retraining {best_model_name} on full dataset with Optuna tuning...")
    best_base_pipeline = candidates[best_model_name]
    all_X, all_y = df[features], df[target_col]
    all_years = pd.to_datetime(df["dt_ref"]).dt.year.reset_index(drop=True)

    final_pipeline, _, _ = optuna_tune(
        best_base_pipeline, all_X, all_y, best_model_name,
        years=all_years, n_trials=N_OPTUNA_TRIALS, scoring=scoring,
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
