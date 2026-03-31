"""Tab: Model Comparison — side-by-side metrics, ROC/PR curves, confusion matrices.

Re-evaluates selected models on the original train/test/OOT splits so we can
render interactive Plotly curves and confusion matrices (MLflow only stores
scalar summaries, not the raw arrays).
"""

import os

import duckdb
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics as sk_metrics
import streamlit as st

from app.helpers import BASE_DIR

MLFLOW_DB = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

EXPERIMENTS = {
    "Champion": "f1_champion",
    "Constructor Champion": "f1_constructor_champion",
    "Driver Departure": "f1_departure",
}

_TARGET_COL = {
    "f1_champion": "fl_champion",
    "f1_constructor_champion": "fl_constructor_champion",
    "f1_departure": "fl_departed",
}

_ABT_FILE = {
    "f1_champion": "abt_champions_inseason.parquet",
    "f1_constructor_champion": "abt_teams_inseason.parquet",
    "f1_departure": "abt_departures_inseason.parquet",
}

_ID_COLS = {
    "f1_champion": ["driverid"],
    "f1_constructor_champion": ["teamid", "team_name"],
    "f1_departure": ["driverid"],
}

# Consistent color palette for models
MODEL_COLORS = {
    "LogisticRegression": "#636EFA",
    "LightGBM": "#00CC96",
    "XGBoost": "#EF553B",
    "BalancedRandomForest": "#AB63FA",
    "AdaBoost": "#FFA15A",
    "CatBoost": "#19D3F3",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _get_runs(experiment_name):
    """Fetch all non-final MLflow runs for an experiment."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"],
    )

    # Keep only the most recent run per model type (runs are ordered by start_time DESC)
    seen = set()
    result = []
    for run in runs:
        model_type = run.data.params.get("model_type", "unknown")
        learning_mode = run.data.params.get("learning_mode", "")
        is_final = run.data.tags.get("final_model") == "true"
        if learning_mode == "zero_shot" or is_final:
            continue
        # Check that this run has a model artifact
        model_arts = client.list_artifacts(run.info.run_id, "model")
        if not model_arts:
            continue
        if model_type in seen:
            continue
        seen.add(model_type)
        result.append({
            "run_id": run.info.run_id,
            "model_type": model_type,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        })
    return result


@st.cache_data(ttl=3600)
def _load_abt(experiment_name):
    """Load the end-of-year ABT for an experiment."""
    abt_path = os.path.join(BASE_DIR, "data", "gold", _ABT_FILE[experiment_name])
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{abt_path}')").fetchdf()
    con.close()
    return df


@st.cache_resource
def _load_model(run_id):
    """Load a sklearn model from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def _get_feature_columns(df, exclude_cols):
    """Reproduce feature selection from ml.utils without importing it."""
    always_exclude = {"dt_ref", "driverid", "year", "teamid", "team_name",
                      "fl_champion", "fl_constructor_champion", "fl_departed",
                      "prediction_year", "data_as_of"}
    exclude = always_exclude | set(exclude_cols)
    features = [c for c in df.columns if c not in exclude]
    features = [f for f in features
                if not (f.endswith("_last20") or f.endswith("_last40"))]
    _keep_sprint = ("qtd_sprint_", "qtd_wins_sprint_",
                    "total_points_sprint_", "qtd_podiums_sprint_")
    features = [f for f in features
                if "sprint" not in f or f.startswith(_keep_sprint)]
    return features


def _split_data(df, target_col, oot_year=None):
    """Reproduce the temporal split from ml.utils."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["dt_ref"]).dt.year

    def _find_oot(d):
        for yr in sorted(d["year"].unique(), reverse=True):
            if d[d["year"] == yr][target_col].nunique() >= 2:
                return yr
        return int(d["year"].max())

    if oot_year is None:
        oot_year = _find_oot(df)

    # Support single int or list of ints
    if isinstance(oot_year, (list, tuple)):
        oot_years = sorted(oot_year)
    else:
        oot_years = [oot_year]

    df_oot = df[df["year"].isin(oot_years)]
    df_rest = df[df["year"] < min(oot_years)]
    test_year = _find_oot(df_rest)
    df_test = df_rest[df_rest["year"] == test_year]
    df_train = df_rest[df_rest["year"] < test_year]
    return df_train, df_test, df_oot, test_year, oot_years


@st.cache_data(ttl=3600)
def _evaluate_model(run_id, experiment_name, oot_year=None):
    """Re-evaluate a model on train/test/OOT and return curve data + predictions."""
    abt = _load_abt(experiment_name)
    target_col = _TARGET_COL[experiment_name]
    id_cols = _ID_COLS[experiment_name]
    features = _get_feature_columns(abt, id_cols)
    df_train, df_test, df_oot, test_year, oot_year = _split_data(abt, target_col, oot_year)

    model = _load_model(run_id)
    model_features = list(model[0].feature_names_in_) if hasattr(model[0], "feature_names_in_") else features

    splits = {}
    for name, split_df in [("train", df_train), ("test", df_test), ("oot", df_oot)]:
        if split_df.empty or split_df[target_col].nunique() < 2:
            continue
        y_true = split_df[target_col].values
        y_prob = model.predict_proba(split_df[model_features])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_prob)
        precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
        cm = sk_metrics.confusion_matrix(y_true, y_pred)

        splits[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "roc_auc": float(sk_metrics.roc_auc_score(y_true, y_prob)),
            "pr_auc": float(sk_metrics.average_precision_score(y_true, y_prob)),
            "f1": float(sk_metrics.f1_score(y_true, y_pred)),
            "log_loss": float(sk_metrics.log_loss(y_true, y_prob)),
            "brier": float(sk_metrics.brier_score_loss(y_true, y_prob)),
            "cm": cm.tolist(),
            "n_samples": len(y_true),
        }

    return splits, test_year, oot_year


# ---------------------------------------------------------------------------
# Metrics table with conditional formatting
# ---------------------------------------------------------------------------

def _build_metrics_table(selected_runs, evaluations):
    """Build a DataFrame comparing metrics across models, with conditional formatting."""
    rows = []
    for run in selected_runs:
        splits = evaluations[run["run_id"]][0]
        row = {"Model": run["model_type"]}
        for split in ("test", "oot"):
            if split not in splits:
                continue
            s = splits[split]
            label = "Test" if split == "test" else "OOT"
            row[f"ROC-AUC ({label})"] = s["roc_auc"]
            row[f"PR-AUC ({label})"] = s["pr_auc"]
            row[f"F1 ({label})"] = s["f1"]
            row[f"Log Loss ({label})"] = s["log_loss"]
            row[f"Brier ({label})"] = s["brier"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def _style_metrics(df):
    """Apply conditional coloring: green=good, red=bad per column."""
    # For these metrics, higher is better
    higher_better = [c for c in df.columns if any(m in c for m in ("ROC-AUC", "PR-AUC", "F1"))]
    # For these, lower is better
    lower_better = [c for c in df.columns if any(m in c for m in ("Log Loss", "Brier"))]

    def _color_higher(s):
        if s.isna().all():
            return [""] * len(s)
        vmin, vmax = s.min(), s.max()
        if vmin == vmax:
            return ["background-color: rgba(76, 175, 80, 0.25)"] * len(s)
        normed = (s - vmin) / (vmax - vmin)
        return [
            f"background-color: rgba(76, 175, 80, {0.1 + 0.4 * v})"
            if not pd.isna(v) else ""
            for v in normed
        ]

    def _color_lower(s):
        if s.isna().all():
            return [""] * len(s)
        vmin, vmax = s.min(), s.max()
        if vmin == vmax:
            return ["background-color: rgba(76, 175, 80, 0.25)"] * len(s)
        normed = (s - vmin) / (vmax - vmin)
        return [
            f"background-color: rgba(76, 175, 80, {0.1 + 0.4 * (1 - v)})"
            if not pd.isna(v) else ""
            for v in normed
        ]

    styler = df.style.format("{:.4f}", na_rep="—")
    if higher_better:
        styler = styler.apply(_color_higher, subset=higher_better)
    if lower_better:
        styler = styler.apply(_color_lower, subset=lower_better)

    # Bold the best value in each column
    def _bold_best(s, higher=True):
        if s.isna().all():
            return [""] * len(s)
        best = s.idxmax() if higher else s.idxmin()
        return ["font-weight: bold" if idx == best else "" for idx in s.index]

    for col in higher_better:
        styler = styler.apply(_bold_best, subset=[col], higher=True)
    for col in lower_better:
        styler = styler.apply(_bold_best, subset=[col], higher=False)

    return styler


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

def _plot_roc_curves(selected_runs, evaluations, split_name):
    """Interactive ROC curves for a given split."""
    fig = go.Figure()
    for run in selected_runs:
        splits = evaluations[run["run_id"]][0]
        if split_name not in splits:
            continue
        s = splits[split_name]
        color = MODEL_COLORS.get(run["model_type"], "#888")
        fig.add_trace(go.Scatter(
            x=s["fpr"], y=s["tpr"],
            mode="lines",
            name=f'{run["model_type"]} (AUC={s["roc_auc"]:.4f})',
            line=dict(color=color, width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
    ))

    label = "Test" if split_name == "test" else "OOT"
    fig.update_layout(
        title=f"ROC Curve ({label})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode="closest",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        height=450,
    )
    return fig


def _plot_pr_curves(selected_runs, evaluations, split_name):
    """Interactive Precision-Recall curves for a given split."""
    fig = go.Figure()
    for run in selected_runs:
        splits = evaluations[run["run_id"]][0]
        if split_name not in splits:
            continue
        s = splits[split_name]
        color = MODEL_COLORS.get(run["model_type"], "#888")
        fig.add_trace(go.Scatter(
            x=s["recall"], y=s["precision"],
            mode="lines",
            name=f'{run["model_type"]} (AP={s["pr_auc"]:.4f})',
            line=dict(color=color, width=2),
        ))

    label = "Test" if split_name == "test" else "OOT"
    fig.update_layout(
        title=f"Precision-Recall Curve ({label})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        hovermode="closest",
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        height=450,
    )
    return fig


def _plot_confusion_matrices(selected_runs, evaluations, split_name):
    """Render confusion matrices side by side using Plotly heatmaps."""
    matrices = []
    for run in selected_runs:
        splits = evaluations[run["run_id"]][0]
        if split_name not in splits:
            continue
        matrices.append((run["model_type"], np.array(splits[split_name]["cm"])))

    if not matrices:
        st.info("No confusion matrix data available for this split.")
        return

    n = len(matrices)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[m[0] for m in matrices],
        horizontal_spacing=0.08,
    )

    labels = ["Negative", "Positive"]
    for i, (model_name, cm) in enumerate(matrices, 1):
        # Normalize for color, show raw counts as text
        cm_norm = cm / cm.sum() if cm.sum() > 0 else cm
        text = [[str(v) for v in row] for row in cm]

        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=16),
                colorscale="Greens",
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="Predicted", row=1, col=i)
        fig.update_yaxes(title_text="Actual" if i == 1 else "", row=1, col=i, autorange="reversed")

    label = "Test" if split_name == "test" else "OOT"
    fig.update_layout(
        title=f"Confusion Matrices ({label}) — threshold = 0.5",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_model_comparison():
    st.header("Model Comparison")

    exp_label = st.selectbox("Experiment", list(EXPERIMENTS.keys()))
    experiment_name = EXPERIMENTS[exp_label]

    runs = _get_runs(experiment_name)
    if not runs:
        st.info("No model runs found. Train models first.")
        return

    # Model multi-select
    model_types = list({r["model_type"] for r in runs})
    selected_types = st.multiselect(
        "Models to compare",
        model_types,
        default=model_types,
    )
    if not selected_types:
        st.warning("Select at least one model.")
        return

    selected_runs = [r for r in runs if r["model_type"] in selected_types]

    # Use oot_year from the first run's params (all runs in an experiment share the same split)
    oot_year_param = None
    for run in selected_runs:
        oot_str = run["params"].get("oot_year")
        if oot_str:
            import ast
            try:
                parsed = ast.literal_eval(oot_str)
                oot_year_param = parsed if isinstance(parsed, list) else int(parsed)
            except (ValueError, SyntaxError):
                oot_year_param = int(oot_str)
            break

    # Evaluate all selected models (cached)
    with st.spinner("Evaluating models on data splits..."):
        evaluations = {}
        for run in selected_runs:
            # Convert list to tuple for st.cache_data hashability
            oot_arg = tuple(oot_year_param) if isinstance(oot_year_param, list) else oot_year_param
            evaluations[run["run_id"]] = _evaluate_model(
                run["run_id"], experiment_name, oot_arg,
            )

    # Show split info
    _, test_year, oot_year = next(iter(evaluations.values()))
    st.caption(f"Test year: **{test_year}** | Out-of-time year: **{oot_year}**")

    # --- Metrics table ---
    st.subheader("Performance Metrics")
    metrics_df = _build_metrics_table(selected_runs, evaluations)
    styled = _style_metrics(metrics_df)
    st.dataframe(styled, use_container_width=True)

    # --- Split selector for curves ---
    available_splits = set()
    for ev in evaluations.values():
        available_splits.update(ev[0].keys())
    available_splits.discard("train")
    split_options = sorted(available_splits)
    split_labels = {"test": "Test", "oot": "Out-of-Time (OOT)"}
    split_choice = st.radio(
        "Evaluation split",
        split_options,
        format_func=lambda x: split_labels.get(x, x),
        horizontal=True,
    )

    # --- ROC + PR curves side by side ---
    st.subheader("ROC & Precision-Recall Curves")
    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig_roc = _plot_roc_curves(selected_runs, evaluations, split_choice)
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_pr:
        fig_pr = _plot_pr_curves(selected_runs, evaluations, split_choice)
        st.plotly_chart(fig_pr, use_container_width=True)

    # --- Confusion matrices ---
    st.subheader("Confusion Matrices")
    _plot_confusion_matrices(selected_runs, evaluations, split_choice)
