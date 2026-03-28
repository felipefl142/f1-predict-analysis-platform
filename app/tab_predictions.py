"""Tab 1: ML Predictions — Champion, Best Team, Driver Departures.

All prediction views show a time-series line chart (x = race date) so you
can see how probabilities evolve round by round through the season.
Optionally overlay TimesFM zero-shot forecasts when timesfm is installed.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.helpers import get_team_color
from ml.timefm_predictor import TIMESFM_AVAILABLE


# ---------------------------------------------------------------------------
# Cached model/data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_model_by_run(run_id):
    from ml.predict import load_model_by_run_id
    return load_model_by_run_id(run_id)


@st.cache_data(ttl=3600)
def _list_models(experiment_name):
    from ml.predict import list_batch_models
    return list_batch_models(experiment_name)


@st.cache_data(ttl=3600)
def _get_champion_predictions(year, run_id):
    from ml.predict import predict_champions
    model = _load_model_by_run(run_id)
    return predict_champions(year, model)


@st.cache_data(ttl=3600)
def _get_team_predictions(year, run_id):
    from ml.predict import predict_teams
    model = _load_model_by_run(run_id)
    return predict_teams(year, model)


@st.cache_data(ttl=3600)
def _get_departure_predictions(year, run_id):
    from ml.predict import predict_departures
    model = _load_model_by_run(run_id)
    return predict_departures(year, model)


@st.cache_data(ttl=3600)
def _get_model_comparison(experiment_name):
    from ml.predict import get_model_comparison
    return get_model_comparison(experiment_name)


@st.cache_data(ttl=3600)
def _get_champion_predictions_online(year):
    from ml.predict import predict_champions_online
    return predict_champions_online(year)


@st.cache_data(ttl=3600)
def _get_team_predictions_online(year):
    from ml.predict import predict_teams_online
    return predict_teams_online(year)


@st.cache_data(ttl=3600)
def _get_departure_predictions_online(year):
    from ml.predict import predict_departures_online
    return predict_departures_online(year)


@st.cache_data(ttl=3600)
def _get_timesfm_champions(year):
    import os
    from ml.predict import SILVER_DIR, BRONZE_PATH
    from ml.timefm_predictor import predict_champions_timesfm
    return predict_champions_timesfm(SILVER_DIR, BRONZE_PATH, year)


@st.cache_data(ttl=3600)
def _get_timesfm_teams(year):
    import os
    from ml.predict import SILVER_DIR, BRONZE_PATH
    from ml.timefm_predictor import predict_teams_timesfm
    _base = os.path.join(os.path.dirname(__file__), "..")
    constructors_csv = os.path.join(_base, "data", "constructors_champions.csv")
    return predict_teams_timesfm(SILVER_DIR, BRONZE_PATH, constructors_csv, year)


@st.cache_data(ttl=3600)
def _get_timesfm_departures(year):
    from ml.predict import SILVER_DIR, BRONZE_PATH
    from ml.timefm_predictor import predict_departures_timesfm
    return predict_departures_timesfm(SILVER_DIR, BRONZE_PATH, year)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _model_selector(experiment_name, key_suffix):
    """Render a selectbox for batch model selection. Returns selected run_id or None."""
    models = _list_models(experiment_name)
    if not models:
        return None
    # Default to first final model, else first model
    default_idx = next((i for i, m in enumerate(models) if m["is_final"]), 0)
    selected = st.selectbox(
        "Model",
        models,
        index=default_idx,
        format_func=lambda m: m["label"],
        key=f"model_select_{key_suffix}",
    )
    return selected["run_id"] if selected else None


def _available_years(abt_filename: str) -> list[int]:
    """Return sorted (desc) list of years present in a gold ABT."""
    import os, duckdb
    from ml.predict import GOLD_DIR
    path = os.path.join(GOLD_DIR, abt_filename)
    try:
        con = duckdb.connect()
        years = con.execute(
            f"SELECT DISTINCT EXTRACT(YEAR FROM dt_ref)::INT AS y FROM read_parquet('{path}') ORDER BY y DESC"
        ).fetchdf()["y"].tolist()
        con.close()
        # Always include next year for current-season projection
        current_year = pd.Timestamp.now().year
        if current_year not in years:
            years = [current_year] + years
        return years
    except Exception:
        return [pd.Timestamp.now().year]


def _line_chart(plot_df, x_col, y_col, color_col, color_map,
                title, y_label, x_label="Race date"):
    """Build a Plotly line+marker chart with discrete race dates on x-axis."""
    fig = go.Figure()
    # Get all unique event dates sorted for discrete x-axis
    all_dates = sorted(plot_df[x_col].unique())
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in all_dates]

    for entity in plot_df[color_col].unique():
        sub = plot_df[plot_df[color_col] == entity].sort_values(x_col)
        color = color_map.get(entity, "#888888")
        x_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sub[x_col]]
        fig.add_trace(go.Scatter(
            x=x_labels, y=sub[y_col],
            mode="lines+markers",
            name=entity,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend_title=color_col,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=date_labels,
        ),
    )
    return fig


def _add_timesfm_traces(fig, tfm_df, x_col, y_col, color_col):
    """Overlay TimesFM forecast lines as dashed traces on an existing figure."""
    for entity in tfm_df[color_col].unique():
        sub = tfm_df[tfm_df[color_col] == entity].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="lines",
            name=f"{entity} (TimesFM)",
            line=dict(dash="dash", width=1.5),
            opacity=0.7,
        ))
    return fig


def _render_model_comparison_table(experiment_name):
    comp = _get_model_comparison(experiment_name)
    if not comp.empty:
        st.dataframe(
            comp.style.format({
                c: "{:.4f}" for c in ["auc_train", "auc_test", "auc_oot",
                                      "cv_auc", "tuned_cv_auc"] if c in comp.columns
            }),
            use_container_width=True,
        )
    else:
        st.info("No model comparison data available.")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_predictions():
    st.header("ML Predictions")

    pred_section = st.selectbox(
        "Select prediction type",
        ["Championship Probabilities", "Best Team Prediction", "Driver Departures"],
    )

    if pred_section == "Championship Probabilities":
        _render_champion_predictions()
    elif pred_section == "Best Team Prediction":
        _render_team_predictions()
    else:
        _render_departure_predictions()


# ---------------------------------------------------------------------------
# Championship
# ---------------------------------------------------------------------------

def _render_champion_predictions():
    st.subheader("Who Will Be Champion?")

    years = _available_years("abt_champions_inseason.parquet")
    selected_year = st.selectbox("Season", years, index=0)

    col1, col2 = st.columns(2)
    with col1:
        use_online = st.toggle("Use adaptive (online) model", value=False, key="online_champ")
    with col2:
        show_tfm = False
        if TIMESFM_AVAILABLE:
            show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False)

    run_id = None
    if not use_online:
        run_id = _model_selector("f1_champion", "champ")
        if run_id is None:
            st.info("No batch models found. Train models first: `python -m ml.champion_model`")
            return

    try:
        if use_online:
            data = _get_champion_predictions_online(selected_year)
        else:
            data = _get_champion_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        if use_online:
            st.info("Train online models first: `python -m ml.champion_model`")
        else:
            st.info("Run the ETL pipeline and train the models first.")
        return

    if data.empty:
        st.warning(f"No data available for {selected_year}.")
        return

    drivers = sorted(data["driverid"].unique())
    driver_names = (
        data[["driverid", "full_name"]].drop_duplicates()
        .set_index("driverid")["full_name"].to_dict()
    )
    latest = data[data["dt_ref"] == data["dt_ref"].max()]
    top3 = latest.nlargest(3, "prob_champion")["driverid"].tolist()
    selected = st.multiselect(
        "Drivers", drivers, default=[d for d in top3 if d in drivers],
        format_func=lambda x: driver_names.get(x, x),
    )
    if not selected:
        st.warning("Select at least one driver.")
        return

    plot = data[data["driverid"].isin(selected)].copy()
    plot["label"] = plot["driverid"].map(driver_names)
    color_map = {
        driver_names.get(row["driverid"], row["driverid"]): get_team_color(None, row.get("team_color"))
        for _, row in plot[["driverid", "team_color"]].drop_duplicates().iterrows()
    }

    model_label = "Online (adaptive)" if use_online else "Batch"
    fig = _line_chart(
        plot, "dt_ref", "prob_champion", "label", color_map,
        title=f"{selected_year} Championship Win Probability ({model_label})",
        y_label="Win Probability",
    )

    if show_tfm:
        try:
            tfm_data = _get_timesfm_champions(selected_year)
            tfm_sel = tfm_data[tfm_data["driverid"].isin(selected)].copy()
            tfm_sel["label"] = tfm_sel["driverid"].map(driver_names)
            fig = _add_timesfm_traces(fig, tfm_sel, "dt_ref", "prob_timesfm", "label")
        except Exception as e:
            st.warning(f"TimesFM prediction failed: {e}")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest standings")
    latest_sel = latest[latest["driverid"].isin(selected)].sort_values("prob_champion", ascending=False)
    st.dataframe(
        latest_sel[["full_name", "prob_champion"]].style.format({"prob_champion": "{:.2%}"}),
        use_container_width=True,
    )

    with st.expander("Model Comparison"):
        _render_model_comparison_table("f1_champion")


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def _render_team_predictions():
    st.subheader("Which Team Will Be Best?")

    years = _available_years("abt_teams_inseason.parquet")
    selected_year = st.selectbox("Season", years, index=0)

    col1, col2 = st.columns(2)
    with col1:
        use_online = st.toggle("Use adaptive (online) model", value=False, key="online_teams")
    with col2:
        show_tfm = False
        if TIMESFM_AVAILABLE:
            show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False, key="tfm_teams")

    run_id = None
    if not use_online:
        run_id = _model_selector("f1_constructor_champion", "teams")
        if run_id is None:
            st.info("No batch models found. Train models first: `python -m ml.team_model`")
            return

    try:
        if use_online:
            data = _get_team_predictions_online(selected_year)
        else:
            data = _get_team_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        if use_online:
            st.info("Train online models first: `python -m ml.team_model`")
        else:
            st.info("Run the ETL pipeline and train the models first.")
        return

    if data.empty:
        st.warning(f"No data available for {selected_year}.")
        return

    teams = sorted(data["teamid"].unique())
    team_names = (
        data[["teamid", "team_name"]].drop_duplicates()
        .set_index("teamid")["team_name"].to_dict()
    )
    latest = data[data["dt_ref"] == data["dt_ref"].max()]
    top3 = latest.nlargest(3, "prob_constructor_champion")["teamid"].tolist()
    selected = st.multiselect(
        "Teams", teams, default=[t for t in top3 if t in teams],
        format_func=lambda x: team_names.get(x, x),
    )
    if not selected:
        st.warning("Select at least one team.")
        return

    plot = data[data["teamid"].isin(selected)].copy()
    plot["label"] = plot["teamid"].map(team_names)
    color_map = {name: get_team_color(name) for name in plot["label"].unique()}

    model_label = "Online (adaptive)" if use_online else "Batch"
    fig = _line_chart(
        plot, "dt_ref", "prob_constructor_champion", "label", color_map,
        title=f"{selected_year} Constructor Championship Probability ({model_label})",
        y_label="Win Probability",
    )

    if show_tfm:
        try:
            tfm_data = _get_timesfm_teams(selected_year)
            tfm_sel = tfm_data[tfm_data["teamid"].isin(selected)].copy()
            tfm_sel["label"] = tfm_sel["teamid"].map(team_names)
            fig = _add_timesfm_traces(fig, tfm_sel, "dt_ref", "prob_timesfm", "label")
        except Exception as e:
            st.warning(f"TimesFM prediction failed: {e}")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest standings")
    latest_sel = latest[latest["teamid"].isin(selected)].sort_values(
        "prob_constructor_champion", ascending=False
    )
    st.dataframe(
        latest_sel[["team_name", "prob_constructor_champion"]].style.format(
            {"prob_constructor_champion": "{:.2%}"}
        ),
        use_container_width=True,
    )

    with st.expander("Model Comparison"):
        _render_model_comparison_table("f1_constructor_champion")


# ---------------------------------------------------------------------------
# Departures
# ---------------------------------------------------------------------------

def _render_departure_predictions():
    st.subheader("Who Might Leave F1?")
    st.caption(
        "Note: This model captures performance-correlated departures only. "
        "Contract decisions, team politics, and personal choices are not captured."
    )

    years = _available_years("abt_departures_inseason.parquet")
    selected_year = st.selectbox("Season", years, index=0)

    col1, col2 = st.columns(2)
    with col1:
        use_online = st.toggle("Use adaptive (online) model", value=False, key="online_departures")
    with col2:
        show_tfm = False
        if TIMESFM_AVAILABLE:
            show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False, key="tfm_departures")

    run_id = None
    if not use_online:
        run_id = _model_selector("f1_departure", "departures")
        if run_id is None:
            st.info("No batch models found. Train models first: `python -m ml.departure_model`")
            return

    try:
        if use_online:
            data = _get_departure_predictions_online(selected_year)
        else:
            data = _get_departure_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        if use_online:
            st.info("Train online models first: `python -m ml.departure_model`")
        else:
            st.info("Run the ETL pipeline and train the models first.")
        return

    if data.empty:
        st.warning(f"No departure data available for {selected_year}.")
        return

    drivers = sorted(data["driverid"].unique())
    driver_names = (
        data[["driverid", "full_name"]].drop_duplicates()
        .set_index("driverid")["full_name"].to_dict()
    )
    latest = data[data["dt_ref"] == data["dt_ref"].max()]
    top5 = latest.nlargest(5, "prob_departure")["driverid"].tolist()
    selected = st.multiselect(
        "Drivers", drivers, default=[d for d in top5 if d in drivers],
        format_func=lambda x: driver_names.get(x, x),
    )
    if not selected:
        st.warning("Select at least one driver.")
        return

    plot = data[data["driverid"].isin(selected)].copy()
    plot["label"] = plot["driverid"].map(driver_names)
    color_map = {
        driver_names.get(row["driverid"], row["driverid"]): get_team_color(None, row.get("team_color"))
        for _, row in plot[["driverid", "team_color"]].drop_duplicates().iterrows()
    }

    model_label = "Online (adaptive)" if use_online else "Batch"
    fig = _line_chart(
        plot, "dt_ref", "prob_departure", "label", color_map,
        title=f"{selected_year} Driver Departure Probability ({model_label})",
        y_label="Departure Probability",
    )

    if show_tfm:
        try:
            tfm_data = _get_timesfm_departures(selected_year)
            tfm_sel = tfm_data[tfm_data["driverid"].isin(selected)].copy()
            tfm_sel["label"] = tfm_sel["driverid"].map(driver_names)
            fig = _add_timesfm_traces(fig, tfm_sel, "dt_ref", "prob_timesfm", "label")
        except Exception as e:
            st.warning(f"TimesFM prediction failed: {e}")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("End-of-season standing")
    latest_sel = latest[latest["driverid"].isin(selected)].sort_values("prob_departure", ascending=False)
    st.dataframe(
        latest_sel[["full_name", "team_name", "prob_departure"]].style.format(
            {"prob_departure": "{:.2%}"}
        ),
        use_container_width=True,
    )

    with st.expander("Model Comparison"):
        _render_model_comparison_table("f1_departure")
