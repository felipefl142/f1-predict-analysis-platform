"""Tab 1: ML Predictions — Champion, Best Team, Driver Departures.

All prediction views show a time-series line chart (x = race date) so you
can see how probabilities evolve round by round through the season.
Optionally overlay TimesFM zero-shot forecasts when timesfm is installed.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import os

from app.helpers import get_duckdb_connection, get_team_color, BASE_DIR, BRONZE_PATH
from ml.timefm_predictor import TIMESFM_AVAILABLE

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Inject CSS once for styled tables: alternating rows + bigger font.
# Uses transparent overlays so it works on both light and dark themes.
_TABLE_CSS = """
<style>
div[data-testid="stDataFrame"] table {
    font-size: 15px !important;
}
div[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
    background-color: rgba(128, 128, 128, 0.08);
}
div[data-testid="stDataFrame"] table tbody tr:nth-child(odd) {
    background-color: rgba(128, 128, 128, 0.00);
}
</style>
"""


def _styled_table(df, fmt=None, hide_index=True):
    """Display a dataframe with alternating row colors and larger font."""
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)
    styler = df.style
    # Alternating row backgrounds via Styler (complements CSS for st.dataframe)
    styler = styler.set_properties(**{"font-size": "15px"})
    row_colors = ["rgba(128,128,128,0.06)", "rgba(128,128,128,0.14)"]
    styler = styler.apply(
        lambda row: [f"background-color: {row_colors[row.name % 2]}"] * len(row),
        axis=1,
    )
    if fmt:
        styler = styler.format(fmt)
    st.dataframe(styler, use_container_width=True, hide_index=hide_index)


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


@st.cache_data(ttl=3600)
def _get_actual_champion(year, experiment_name):
    """Return the actual champion driverid/teamid for a given year."""
    import os
    from app.helpers import BASE_DIR
    csv_map = {
        "f1_champion": "data/champions.csv",
        "f1_constructor_champion": "data/constructors_champions.csv",
    }
    id_col_map = {
        "f1_champion": "driverid",
        "f1_constructor_champion": "teamid",
    }
    csv_path = csv_map.get(experiment_name)
    if not csv_path:
        return None
    full_path = os.path.join(BASE_DIR, csv_path)
    if not os.path.exists(full_path):
        return None
    df = pd.read_csv(full_path)
    row = df[df["year"] == year]
    return row[id_col_map[experiment_name]].iloc[0] if not row.empty else None


def _top1_accuracy_chart(data, id_col, actual_id, title):
    """Render a chart comparing Model's #1 Pick probability vs. Actual Champion probability."""
    # Find model's #1 pick at each race
    prob_col = "prob_champion" if id_col=="driverid" else "prob_constructor_champion"
    top1 = data.loc[data.groupby("dt_ref")[prob_col].idxmax()].copy()
    top1 = top1.sort_values("dt_ref")

    # Prob of actual champion
    actual_data = data[data[id_col] == actual_id].sort_values("dt_ref") if actual_id else pd.DataFrame()

    # Actual standings leader (P1) at each race
    standing_col = "standing_position" if id_col == "driverid" else "team_standing_position"
    name_col = "full_name" if id_col == "driverid" else "team_name"
    standings_leader = pd.DataFrame()
    if standing_col in data.columns:
        p1 = data[data[standing_col] == 1].copy()
        if not p1.empty:
            # One leader per date (take highest prob if tied)
            standings_leader = p1.loc[p1.groupby("dt_ref")[prob_col].idxmax()].sort_values("dt_ref")

    fig = go.Figure()

    all_dates = sorted(data["dt_ref"].unique())
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in all_dates]

    # Trace 1: Model's top pick
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in top1["dt_ref"]],
        y=top1[prob_col],
        mode="lines+markers",
        name="Model's #1 Pick",
        line=dict(color="#EF553B", width=3),
        marker=dict(size=8),
        hovertemplate="Model's #1: %{customdata}<br>Prob: %{y:.1%}<extra></extra>",
        customdata=top1[name_col]
    ))

    # Trace 2: Actual champion (if known)
    if not actual_data.empty:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in actual_data["dt_ref"]],
            y=actual_data[prob_col],
            mode="lines+markers",
            name="Actual Champion",
            line=dict(color="#00CC96", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Actual Champ Prob: %{y:.1%}<extra></extra>"
        ))

    # Trace 3: Actual standings leader (P1)
    if not standings_leader.empty:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in standings_leader["dt_ref"]],
            y=standings_leader[prob_col],
            mode="lines+markers",
            name="Standings Leader (P1)",
            line=dict(color="#FFA15A", width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="Standings P1: %{customdata}<br>Prob: %{y:.1%}<extra></extra>",
            customdata=standings_leader[name_col]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Race date",
        yaxis_title="Win Probability",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        xaxis=dict(type="category", categoryorder="array", categoryarray=date_labels),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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
                                      "cv_ap", "tuned_cv_ap", "top1_acc_test", "top1_acc_oot",
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

    st.divider()
    _render_current_standings()
    _render_last_race_results()


# ---------------------------------------------------------------------------
# Championship
# ---------------------------------------------------------------------------

def _render_champion_predictions():
    st.subheader("Who Will Be Champion?")

    years = _available_years("abt_champions_inseason.parquet")
    selected_year = st.selectbox("Season", years, index=0)

    show_tfm = False
    if TIMESFM_AVAILABLE:
        show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False)

    run_id = _model_selector("f1_champion", "champ")
    if run_id is None:
        st.info("No models found. Train models first: `python -m ml.champion_model`")
        return

    try:
        data = _get_champion_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
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
    top5 = latest.nlargest(5, "prob_champion")["driverid"].tolist()
    selected = st.multiselect(
        "Drivers", drivers, default=[d for d in top5 if d in drivers],
        format_func=lambda x: driver_names.get(x, x),
    )
    if not selected:
        st.warning("Select at least one driver.")
        return

    # Add Top-1 Accuracy Evolution Chart
    actual_champ = _get_actual_champion(selected_year, "f1_champion")
    with st.expander("📊 View Top-1 Prediction Evolution", expanded=False):
        fig_top1 = _top1_accuracy_chart(
            data, "driverid", actual_champ, 
            f"{selected_year} Model #1 Pick vs Actual Champion"
        )
        st.plotly_chart(fig_top1, use_container_width=True)

    plot = data[data["driverid"].isin(selected)].copy()
    plot["label"] = plot["driverid"].map(driver_names)
    color_map = {
        driver_names.get(row["driverid"], row["driverid"]): get_team_color(None, row.get("team_color"))
        for _, row in plot[["driverid", "team_color"]].drop_duplicates().iterrows()
    }

    fig = _line_chart(
        plot, "dt_ref", "prob_champion", "label", color_map,
        title=f"{selected_year} Championship Win Probability",
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
    display_champ = latest_sel[["full_name", "prob_champion"]].rename(
        columns={"full_name": "Driver", "prob_champion": "Win Probability"}
    ).reset_index(drop=True)
    _styled_table(display_champ, fmt={"Win Probability": "{:.2%}"})


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def _render_team_predictions():
    st.subheader("Which Team Will Be Best?")

    years = _available_years("abt_teams_inseason.parquet")
    selected_year = st.selectbox("Season", years, index=0)

    show_tfm = False
    if TIMESFM_AVAILABLE:
        show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False, key="tfm_teams")

    run_id = _model_selector("f1_constructor_champion", "teams")
    if run_id is None:
        st.info("No models found. Train models first: `python -m ml.team_model`")
        return

    try:
        data = _get_team_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
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
    top5 = latest.nlargest(5, "prob_constructor_champion")["teamid"].tolist()
    selected = st.multiselect(
        "Teams", teams, default=[t for t in top5 if t in teams],
        format_func=lambda x: team_names.get(x, x),
    )
    if not selected:
        st.warning("Select at least one team.")
        return

    # Add Top-1 Accuracy Evolution Chart
    actual_team = _get_actual_champion(selected_year, "f1_constructor_champion")
    with st.expander("📊 View Top-1 Prediction Evolution", expanded=False):
        fig_top1 = _top1_accuracy_chart(
            data, "teamid", actual_team, 
            f"{selected_year} Model #1 Pick vs Actual Constructor Champion"
        )
        st.plotly_chart(fig_top1, use_container_width=True)

    plot = data[data["teamid"].isin(selected)].copy()
    plot["label"] = plot["teamid"].map(team_names)
    color_map = {name: get_team_color(name) for name in plot["label"].unique()}

    fig = _line_chart(
        plot, "dt_ref", "prob_constructor_champion", "label", color_map,
        title=f"{selected_year} Constructor Championship Probability",
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
    display_team = latest_sel[["team_name", "prob_constructor_champion"]].rename(
        columns={"team_name": "Team", "prob_constructor_champion": "Win Probability"}
    ).reset_index(drop=True)
    _styled_table(display_team, fmt={"Win Probability": "{:.2%}"})


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

    show_tfm = False
    if TIMESFM_AVAILABLE:
        show_tfm = st.toggle("Overlay TimesFM zero-shot forecast", value=False, key="tfm_departures")

    run_id = _model_selector("f1_departure", "departures")
    if run_id is None:
        st.info("No models found. Train models first: `python -m ml.departure_model`")
        return

    try:
        data = _get_departure_predictions(selected_year, run_id)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
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
    
    # Default selection: Top 6 High Risk, Top 4 Medium Risk, 2 Low Risk
    high_risk = latest[latest["risk_tier"] == "High"].nlargest(6, "prob_departure")["driverid"].tolist()
    med_risk = latest[latest["risk_tier"] == "Medium"].nlargest(4, "prob_departure")["driverid"].tolist()
    low_risk = latest[latest["risk_tier"] == "Low"].nlargest(2, "prob_departure")["driverid"].tolist()
    
    default_selected = high_risk + med_risk + low_risk
    if not default_selected:
        default_selected = latest.nlargest(6, "prob_departure")["driverid"].tolist()

    selected = st.multiselect(
        "Drivers", drivers, default=[d for d in default_selected if d in drivers],
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

    fig = _line_chart(
        plot, "dt_ref", "prob_departure", "label", color_map,
        title=f"{selected_year} Driver Departure Risk",
        y_label="Departure Probability",
    )
    # Add horizontal risk-tier bands
    fig.add_hrect(y0=0, y1=0.25, fillcolor="green", opacity=0.07,
                  annotation_text="Low", annotation_position="top left")
    fig.add_hrect(y0=0.25, y1=0.60, fillcolor="orange", opacity=0.07,
                  annotation_text="Medium", annotation_position="top left")
    fig.add_hrect(y0=0.60, y1=1.0, fillcolor="red", opacity=0.07,
                  annotation_text="High", annotation_position="top left")

    if show_tfm:
        try:
            tfm_data = _get_timesfm_departures(selected_year)
            tfm_sel = tfm_data[tfm_data["driverid"].isin(selected)].copy()
            tfm_sel["label"] = tfm_sel["driverid"].map(driver_names)
            fig = _add_timesfm_traces(fig, tfm_sel, "dt_ref", "prob_timesfm", "label")
        except Exception as e:
            st.warning(f"TimesFM prediction failed: {e}")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Risk Assessment")
    tier_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    latest_sel = latest[latest["driverid"].isin(selected)].sort_values("prob_departure", ascending=False).copy()
    tier_str = latest_sel["risk_tier"].astype(str)
    prob_str = latest_sel["prob_departure"].apply(lambda p: f"({p:.1%})")
    latest_sel["Risk"] = tier_str.map(tier_colors) + " " + tier_str + " " + prob_str
    display_dep = latest_sel[["full_name", "team_name", "Risk"]].rename(
        columns={"full_name": "Driver", "team_name": "Team"}
    ).reset_index(drop=True)
    _styled_table(display_dep)



# ---------------------------------------------------------------------------
# Current Standings & Last Race Results
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _load_current_standings():
    """Compute current-season driver standings from bronze results."""
    con = get_duckdb_connection()
    df = con.execute(f"""
        WITH current_year AS (
            SELECT MAX(year) AS yr FROM read_parquet('{BRONZE_PATH}') WHERE mode = 'Race'
        )
        SELECT
            r.driverid,
            r.full_name,
            r.team_name,
            r.team_color,
            SUM(r.points) AS total_points,
            COUNT(*) AS races,
            SUM(CASE WHEN r.position = 1 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN r.position <= 3 THEN 1 ELSE 0 END) AS podiums,
            ROUND(AVG(r.position), 1) AS avg_position
        FROM read_parquet('{BRONZE_PATH}') r, current_year c
        WHERE r.year = c.yr AND r.mode = 'Race'
        GROUP BY r.driverid, r.full_name, r.team_name, r.team_color
        ORDER BY total_points DESC
    """).fetchdf()
    con.close()
    return df


def _fmt_laptime(ns):
    """Format nanosecond timedelta as m:ss.SSS or ss.SSS."""
    if pd.isna(ns) or ns == 0:
        return ""
    total_ms = ns / 1e6
    mins = int(total_ms // 60000)
    secs = (total_ms % 60000) / 1000
    if mins > 0:
        return f"{mins}:{secs:06.3f}"
    return f"{secs:.3f}"


def _fmt_race_time(ns, is_winner=False):
    """Format race time: absolute for winner, gap for others."""
    if pd.isna(ns) or ns == 0:
        return ""
    if is_winner:
        total_s = ns / 1e9
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = total_s % 60
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"+{ns / 1e9:.3f}s"


@st.cache_data(ttl=3600)
def _load_season_events():
    """Return list of (round_number, event_name, event_date) for the current season."""
    con = get_duckdb_connection()
    df = con.execute(f"""
        WITH current_year AS (
            SELECT MAX(year) AS yr FROM read_parquet('{BRONZE_PATH}') WHERE mode = 'Race'
        )
        SELECT DISTINCT r.round_number, r.event_name, r.event_date, r.year
        FROM read_parquet('{BRONZE_PATH}') r, current_year c
        WHERE r.year = c.yr AND r.mode = 'Race'
        ORDER BY r.round_number DESC
    """).fetchdf()
    con.close()
    return df


@st.cache_data(ttl=3600)
def _load_race_results(year, round_number):
    """Load race results for a specific event, joining raw R + Q files for extra data."""
    con = get_duckdb_connection()
    race_file = os.path.join(RAW_DIR, f"{year}_{round_number:02d}_R.parquet")
    quali_file = os.path.join(RAW_DIR, f"{year}_{round_number:02d}_Q.parquet")

    has_quali = os.path.exists(quali_file)
    if has_quali:
        df = con.execute(f"""
            SELECT
                r.Position AS position,
                r.FullName AS full_name,
                r.TeamName AS team_name,
                r.GridPosition AS grid_position,
                r.Points AS points,
                r.Status AS status,
                r.EventName AS event_name,
                r.Date AS event_date,
                (r.GridPosition - r.Position) AS positions_gained,
                r.Time AS race_time,
                q.Position AS quali_position,
                COALESCE(q.Q3, q.Q2, q.Q1) AS best_quali_time
            FROM read_parquet('{race_file}') r
            LEFT JOIN read_parquet('{quali_file}') q ON r.DriverId = q.DriverId
            ORDER BY r.Position
        """).fetchdf()
    else:
        df = con.execute(f"""
            SELECT
                Position AS position,
                FullName AS full_name,
                TeamName AS team_name,
                GridPosition AS grid_position,
                Points AS points,
                Status AS status,
                EventName AS event_name,
                Date AS event_date,
                (GridPosition - Position) AS positions_gained,
                Time AS race_time,
                NULL AS quali_position,
                NULL AS best_quali_time
            FROM read_parquet('{race_file}')
            ORDER BY Position
        """).fetchdf()
    con.close()
    return df


def _format_race_table(results):
    """Format a race results DataFrame for display."""
    display = results.copy()
    # Cast float columns to nullable int for clean display
    for col in ("position", "grid_position", "quali_position", "points"):
        if col in display.columns:
            display[col] = pd.array(display[col], dtype=pd.Int64Dtype())
    # Format best lap time (race Time column: winner = absolute, rest = gap)
    display["Race Time"] = [
        _fmt_race_time(t, i == 0)
        for i, t in enumerate(display["race_time"])
    ]
    # Format quali time
    display["Quali Time"] = display["best_quali_time"].apply(_fmt_laptime)
    # Format positions gained
    display["+/-"] = display["positions_gained"].apply(
        lambda x: f"+{int(x)}" if pd.notna(x) and x > 0 else (str(int(x)) if pd.notna(x) else "")
    )

    cols = {
        "position": "Pos", "full_name": "Driver", "team_name": "Team",
        "quali_position": "Quali", "grid_position": "Grid", "+/-": "+/-",
        "Quali Time": "Quali Time", "Race Time": "Race Time",
        "points": "Points", "status": "Status",
    }
    out = display[[c for c in cols if c in display.columns]].rename(columns=cols)
    return out


def _render_current_standings():
    try:
        standings = _load_current_standings()
    except Exception:
        return

    if standings.empty:
        return

    st.subheader("Current Driver Standings")
    standings["Pos"] = range(1, len(standings) + 1)
    for col in ("total_points", "races", "wins", "podiums"):
        standings[col] = standings[col].astype(int)
    display = standings[["Pos", "full_name", "team_name", "total_points",
                         "races", "wins", "podiums", "avg_position"]].rename(columns={
        "full_name": "Driver", "team_name": "Team", "total_points": "Points",
        "races": "Races", "wins": "Wins", "podiums": "Podiums", "avg_position": "Avg Pos",
    })
    _styled_table(display, fmt={"Avg Pos": "{:.1f}"})


def _render_last_race_results():
    try:
        events = _load_season_events()
    except Exception:
        return

    if events.empty:
        return

    year = int(events["year"].iloc[0])
    latest_round = int(events["round_number"].iloc[0])

    # Last race
    latest_event = events[events["round_number"] == latest_round].iloc[0]
    race_name = latest_event["event_name"]
    race_date = pd.to_datetime(latest_event["event_date"]).strftime("%Y-%m-%d")

    st.subheader(f"Last Race: {race_name} ({race_date})")
    try:
        results = _load_race_results(year, latest_round)
        _styled_table(_format_race_table(results))
    except Exception as e:
        st.error(f"Could not load results: {e}")

    # Event selector for other races
    if len(events) > 1:
        st.subheader("Season Race Results")
        event_options = {
            f"R{int(row['round_number']):02d} — {row['event_name']} ({pd.to_datetime(row['event_date']).strftime('%Y-%m-%d')})": int(row["round_number"])
            for _, row in events.iterrows()
        }
        selected_label = st.selectbox("Select race", list(event_options.keys()),
                                      index=1, key="race_event_selector")
        selected_round = event_options[selected_label]
        try:
            other_results = _load_race_results(year, selected_round)
            _styled_table(_format_race_table(other_results))
        except Exception as e:
            st.error(f"Could not load results: {e}")
