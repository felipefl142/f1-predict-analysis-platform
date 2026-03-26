"""Shared ML utilities: MLFlow setup, data splits, CV, Optuna hyperparam tuning, batch + online."""

import copy
import os

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
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 30


def setup_mlflow(experiment_name):
    """Set up local MLFlow tracking and return the experiment."""
    tracking_uri = f"file://{os.path.abspath(MLFLOW_DIR)}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)


def get_feature_columns(df, exclude_cols=None):
    """Return feature column names (everything except metadata and target)."""
    if exclude_cols is None:
        exclude_cols = []
    always_exclude = {"dt_ref", "driverid", "year", "teamid", "team_name",
                      "fl_champion", "fl_constructor_champion", "fl_departed"}
    exclude = always_exclude | set(exclude_cols)
    return [c for c in df.columns if c not in exclude]


def _find_oot_year(df, target_col):
    """Find the most recent year that has both classes in the target."""
    df_temp = df.copy()
    df_temp["year"] = pd.to_datetime(df_temp["dt_ref"]).dt.year

    for year in sorted(df_temp["year"].unique(), reverse=True):
        year_data = df_temp[df_temp["year"] == year]
        if year_data[target_col].nunique() >= 2:
            return year

    return int(df_temp["year"].max())


def split_data(df, target_col, id_cols, oot_year=None, test_size=0.2):
    """Split into train/test/OOT sets with stratification.

    Auto-detects OOT year as the most recent year with both classes.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["dt_ref"]).dt.year

    if oot_year is None:
        oot_year = _find_oot_year(df, target_col)
    print(f"  OOT year: {oot_year}")

    df_oot = df[df["year"] == oot_year].copy()
    df_analytics = df[df["year"] < oot_year].copy()

    # Remove last 4 rounds per year to avoid late-season label leakage
    df_year_round = df_analytics[["year", "dt_ref"]].drop_duplicates()
    df_year_round["row_number"] = (
        df_year_round.sort_values("dt_ref", ascending=False)
        .groupby("year")
        .cumcount()
    )
    df_year_round = df_year_round[df_year_round["row_number"] > 4]
    df_year_round = df_year_round.drop("row_number", axis=1)
    df_analytics = df_analytics.merge(df_year_round, how="inner")

    entity_target = (
        df_analytics[id_cols + ["year", target_col]]
        .drop_duplicates()
    )

    if entity_target[target_col].nunique() < 2:
        train_entities = entity_target
        test_entities = entity_target.sample(frac=test_size, random_state=42)
    else:
        train_entities, test_entities = model_selection.train_test_split(
            entity_target,
            test_size=test_size,
            random_state=42,
            stratify=entity_target[target_col],
        )

    df_train = train_entities.merge(df_analytics, how="inner")
    df_test = test_entities.merge(df_analytics, how="inner")

    print(f"  Split: train={len(df_train)}, test={len(df_test)}, oot={len(df_oot)}")
    return df_train, df_test, df_oot, oot_year


def log_roc_curves(y_train, y_train_prob, y_test, y_test_prob,
                   y_oot=None, y_oot_prob=None):
    """Plot ROC curves and log to MLFlow."""
    auc_train = metrics.roc_auc_score(y_train, y_train_prob)
    roc_train = metrics.roc_curve(y_train, y_train_prob)
    mlflow.log_metric("auc_train", auc_train)

    auc_test = metrics.roc_auc_score(y_test, y_test_prob)
    roc_test = metrics.roc_curve(y_test, y_test_prob)
    mlflow.log_metric("auc_test", auc_test)

    plt.figure(dpi=100)
    plt.plot(roc_train[0], roc_train[1])
    plt.plot(roc_test[0], roc_test[1])
    legend = [f"Train: {auc_train:.4f}", f"Test: {auc_test:.4f}"]

    auc_oot = None
    if y_oot is not None and y_oot_prob is not None and len(np.unique(y_oot)) > 1:
        auc_oot = metrics.roc_auc_score(y_oot, y_oot_prob)
        roc_oot = metrics.roc_curve(y_oot, y_oot_prob)
        mlflow.log_metric("auc_oot", auc_oot)
        plt.plot(roc_oot[0], roc_oot[1])
        legend.append(f"OOT: {auc_oot:.4f}")

    plt.legend(legend)
    plt.grid(True)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    roc_path = "/tmp/roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(roc_path)

    return auc_train, auc_test, auc_oot


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(pipeline, X, y, n_folds=N_CV_FOLDS):
    """Run stratified K-fold CV and return mean/std AUC scores."""
    cv = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = model_selection.cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1, error_score=0.0,
    )
    return cv_scores.mean(), cv_scores.std(), cv_scores


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def _suggest_params(trial, model_name, balanced=False):
    """Define Optuna search space per model type. Returns dict of pipeline params."""
    if model_name == "LogisticRegression":
        return {
            "model__C": trial.suggest_float("C", 1e-4, 100, log=True),
            "model__penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "model__solver": "saga",
        }
    elif model_name in ("RandomForest", "BalancedRandomForest"):
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "model__max_depth": trial.suggest_int("max_depth", 5, 50),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "model__max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7],
            ),
        }
    elif model_name == "XGBoost":
        params = {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "model__max_depth": trial.suggest_int("max_depth", 3, 12),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "model__gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        if balanced:
            params["model__scale_pos_weight"] = trial.suggest_int("scale_pos_weight", 1, 30)
        return params
    elif model_name == "CatBoost":
        params = {
            "model__iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "model__depth": trial.suggest_int("depth", 3, 10),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "model__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "model__bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "model__random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
        }
        return params
    return {}


def optuna_tune(pipeline, X, y, model_name, n_trials=N_OPTUNA_TRIALS, balanced=False):
    """Run Optuna Bayesian optimization with TPE sampler and median pruner.

    Returns:
        (best_pipeline, best_params, best_cv_auc)
    """
    cv = model_selection.StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = _suggest_params(trial, model_name, balanced)
        if not params:
            return 0.0

        cloned = clone(pipeline)
        cloned.set_params(**params)

        # Use CV with pruning: evaluate fold-by-fold
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            cloned.fit(X_fold_train, y_fold_train)
            y_pred = cloned.predict_proba(X_fold_val)[:, 1]
            fold_auc = metrics.roc_auc_score(y_fold_val, y_pred)
            scores.append(fold_auc)

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

    # Build best pipeline
    best_params = _suggest_params_from_dict(study.best_params, model_name, balanced)
    best_pipeline = clone(pipeline)
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X, y)

    return best_pipeline, study.best_params, study.best_value


def _suggest_params_from_dict(params_dict, model_name, balanced=False):
    """Convert Optuna's flat best_params dict back to pipeline param format."""
    mapped = {}
    for key, val in params_dict.items():
        mapped[f"model__{key}"] = val

    # Add fixed params not in the search space
    if model_name == "LogisticRegression":
        mapped["model__solver"] = "saga"

    return mapped


# ---------------------------------------------------------------------------
# Batch training with CV + Optuna tuning
# ---------------------------------------------------------------------------

def train_and_compare_batch(df, target_col, id_cols, experiment_name, candidates,
                            oot_year=None, balanced=False):
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
    )

    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    results = []

    for name, pipeline in candidates.items():
        print(f"\n  [BATCH] Training {name}...")

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
            cv_mean, cv_std, cv_scores = cross_validate_model(pipeline, X_train, y_train)
            mlflow.log_metric("cv_auc_mean", cv_mean)
            mlflow.log_metric("cv_auc_std", cv_std)
            for i, score in enumerate(cv_scores):
                mlflow.log_metric(f"cv_fold_{i}", score)
            print(f"    Default CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

            # --- Step 2: Optuna tuning ---
            has_search_space = _suggest_params.__code__.co_varnames  # always true, but check model
            print(f"    Optuna tuning ({N_OPTUNA_TRIALS} trials, TPE + median pruner)...")
            tuned_pipeline, best_params, tuned_cv_auc = optuna_tune(
                pipeline, X_train, y_train, name,
                n_trials=N_OPTUNA_TRIALS, balanced=balanced,
            )

            if tuned_cv_auc > cv_mean:
                pipeline = tuned_pipeline
                print(f"    Tuned CV AUC: {tuned_cv_auc:.4f} (improved from {cv_mean:.4f})")
            else:
                # Default was better, refit with defaults
                tuned_cv_auc = cv_mean
                pipeline.fit(X_train, y_train)
                best_params = {}
                print(f"    Tuned CV AUC: {tuned_cv_auc:.4f} (kept defaults)")

            mlflow.log_metric("tuned_cv_auc", tuned_cv_auc)
            for param_key, param_val in best_params.items():
                mlflow.log_param(f"best_{param_key}", param_val)

            # --- Step 3: Evaluate on test + OOT ---
            y_train_prob = pipeline.predict_proba(X_train)[:, 1]
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
            y_oot_prob = pipeline.predict_proba(X_oot)[:, 1] if len(X_oot) > 0 else None

            auc_train, auc_test, auc_oot = log_roc_curves(
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

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            results.append({
                "model": name,
                "mode": "batch",
                "cv_auc": cv_mean,
                "tuned_cv_auc": tuned_cv_auc,
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_oot": auc_oot,
                "run_id": mlflow.active_run().info.run_id,
            })

    comparison = pd.DataFrame(results)

    # Select best by OOT AUC, fallback to tuned CV AUC
    score_col = "auc_oot" if comparison["auc_oot"].notna().any() else "tuned_cv_auc"
    best_idx = comparison[score_col].fillna(0).idxmax()
    best_model_name = comparison.loc[best_idx, "model"]
    best_run_id = comparison.loc[best_idx, "run_id"]

    mlflow.MlflowClient().set_tag(best_run_id, "best_model", "true")

    print(f"\n  Batch model comparison:")
    print(comparison.to_string(index=False))
    print(f"\n  Best batch model: {best_model_name} (by {score_col})")

    # Retrain best model on ALL data with Optuna tuning
    print(f"  Retraining {best_model_name} on full dataset with Optuna tuning...")
    best_base_pipeline = candidates[best_model_name]
    all_X, all_y = df[features], df[target_col]

    final_pipeline, _, _ = optuna_tune(
        best_base_pipeline, all_X, all_y, best_model_name,
        n_trials=N_OPTUNA_TRIALS, balanced=balanced,
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


# ---------------------------------------------------------------------------
# Online training
# ---------------------------------------------------------------------------

def _train_sklearn_online(pipeline, X_train, y_train, X_test, y_test,
                          X_oot, y_oot, features, classes):
    """Train an sklearn model with partial_fit, simulating streaming data."""
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_train_scaled = scaler.fit_transform(X_train.fillna(-10000))

    batch_size = max(1, len(X_train_scaled) // 50)
    for start in range(0, len(X_train_scaled), batch_size):
        end = min(start + batch_size, len(X_train_scaled))
        X_batch = X_train_scaled[start:end]
        y_batch = y_train.iloc[start:end]
        model.partial_fit(X_batch, y_batch, classes=classes)

    X_test_scaled = scaler.transform(X_test.fillna(-10000))
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    y_oot_prob = None
    if len(X_oot) > 0:
        X_oot_scaled = scaler.transform(X_oot.fillna(-10000))
        y_oot_prob = model.predict_proba(X_oot_scaled)[:, 1]

    return y_train_prob, y_test_prob, y_oot_prob


def _train_streaming_online(model_factory, X_train, y_train, X_test, y_test,
                            X_oot, y_oot, features):
    """Train a streaming model one sample at a time."""
    model = model_factory()

    X_train_filled = X_train.fillna(-10000)
    for i in range(len(X_train_filled)):
        x = X_train_filled.iloc[i].to_dict()
        y = int(y_train.iloc[i])
        model.learn_one(x, y)

    def predict_proba_batch(model, X_df):
        X_filled = X_df.fillna(-10000)
        probs = []
        for i in range(len(X_filled)):
            x = X_filled.iloc[i].to_dict()
            p = model.predict_proba_one(x)
            probs.append(p.get(1, 0.0))
        return np.array(probs)

    y_train_prob = predict_proba_batch(model, X_train)
    y_test_prob = predict_proba_batch(model, X_test)

    y_oot_prob = None
    if len(X_oot) > 0:
        y_oot_prob = predict_proba_batch(model, X_oot)

    return y_train_prob, y_test_prob, y_oot_prob, model


def train_and_compare_online(df, target_col, id_cols, experiment_name, candidates,
                             oot_year=None):
    """Train all online candidate models, log to MLFlow, return comparison."""
    setup_mlflow(experiment_name)
    features = get_feature_columns(df, exclude_cols=id_cols)
    df_train, df_test, df_oot, oot_year = split_data(
        df, target_col, id_cols, oot_year,
    )

    df_train = df_train.sort_values("dt_ref")
    df_test = df_test.sort_values("dt_ref")
    df_oot = df_oot.sort_values("dt_ref")

    X_train, y_train = df_train[features], df_train[target_col]
    X_test, y_test = df_test[features], df_test[target_col]
    X_oot, y_oot = df_oot[features], df_oot[target_col]

    classes = np.array([0, 1])
    results = []

    for name, config in candidates.items():
        print(f"  [ONLINE] Training {name}...")

        with mlflow.start_run(run_name=f"online_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("learning_mode", "online")
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_test_rows", len(X_test))
            mlflow.log_param("n_oot_rows", len(X_oot))
            mlflow.log_param("oot_year", oot_year)

            if config["type"] == "sklearn":
                pipeline = config["model"]
                y_train_prob, y_test_prob, y_oot_prob = _train_sklearn_online(
                    pipeline, X_train, y_train, X_test, y_test,
                    X_oot, y_oot, features, classes,
                )
            elif config["type"] == "streaming":
                y_train_prob, y_test_prob, y_oot_prob, _ = _train_streaming_online(
                    config["model_factory"],
                    X_train, y_train, X_test, y_test,
                    X_oot, y_oot, features,
                )

            auc_train, auc_test, auc_oot = log_roc_curves(
                y_train, y_train_prob,
                y_test, y_test_prob,
                y_oot if y_oot_prob is not None else None,
                y_oot_prob,
            )

            results.append({
                "model": name,
                "mode": "online",
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_oot": auc_oot,
                "run_id": mlflow.active_run().info.run_id,
            })

    comparison = pd.DataFrame(results)

    score_col = "auc_oot" if comparison["auc_oot"].notna().any() else "auc_test"
    best_idx = comparison[score_col].fillna(0).idxmax()
    best_model_name = comparison.loc[best_idx, "model"]
    best_run_id = comparison.loc[best_idx, "run_id"]

    mlflow.MlflowClient().set_tag(best_run_id, "best_model", "true")
    mlflow.MlflowClient().set_tag(best_run_id, "learning_mode", "online")

    print(f"\n  Online model comparison:")
    print(comparison.to_string(index=False))
    print(f"\n  Best online model: {best_model_name} (by {score_col})")

    return comparison, best_model_name
