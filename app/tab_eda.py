"""Tab 2: Exploratory Data Analysis — Interactive F1 visualizations."""

import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.helpers import get_duckdb_connection, get_team_color, BRONZE_PATH, SILVER_DIR, GOLD_DIR


@st.cache_data(ttl=3600)
def _load_results():
    con = get_duckdb_connection()
    df = con.execute(f"SELECT * FROM read_parquet('{BRONZE_PATH}')").fetchdf()
    con.close()
    return df


def render_eda():
    st.header("Exploratory Data Analysis")

    try:
        df = _load_results()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure you've run the ETL pipeline first.")
        return

    viz = st.selectbox(
        "Select visualization",
        [
            "Feature Explorer",
            "Champion Model Features",
            "Constructor Model Features",
            "Departure Model Features",
            "Points Distribution by Season",
            "Win Rate Over Career",
            "Team Comparison",
            "Grid vs Finish Position",
            "Driver Career Trajectories",
            "Season Points Progression",
        ],
    )

    if viz == "Feature Explorer":
        _feature_explorer()
    elif viz == "Champion Model Features":
        _champion_features()
    elif viz == "Constructor Model Features":
        _constructor_features()
    elif viz == "Departure Model Features":
        _departure_features()
    elif viz == "Points Distribution by Season":
        _points_distribution(df)
    elif viz == "Win Rate Over Career":
        _win_rate(df)
    elif viz == "Team Comparison":
        _team_comparison(df)
    elif viz == "Grid vs Finish Position":
        _grid_vs_finish(df)
    elif viz == "Driver Career Trajectories":
        _career_trajectories(df)
    elif viz == "Season Points Progression":
        _season_progression(df)


_FEATURE_GROUPS = {
    "Points": ["total_points", "total_points_race", "total_points_sprint"],
    "Wins": ["qtd_wins", "qtd_wins_race", "qtd_wins_sprint"],
    "Podiums": ["qtd_podiums", "qtd_podiums_race", "qtd_podiums_sprint"],
    "Top 5": ["qtd_top5", "qtd_top5_race", "qtd_top5_sprint"],
    "Poles": ["qtd_poles", "qtd_poles_race", "qtd_poles_sprint"],
    "Pole-to-Win": ["qtd_pole_win", "qtd_pole_win_race", "qtd_pole_win_sprint"],
    "Avg Grid Position": ["avg_grid", "avg_grid_race", "avg_grid_sprint"],
    "Avg Finish Position": ["avg_position", "avg_position_race", "avg_position_sprint"],
    "Avg Overtakes": ["avg_overtakes", "avg_overtakes_race", "avg_overtakes_sprint"],
    "Sessions": ["qtd_sessions", "qtd_finished", "qtd_race", "qtd_sprint",
                  "qtd_sessions_with_points", "qtd_sessions_with_overtake"],
    "Weather": ["avg_air_temp", "avg_track_temp", "avg_humidity", "avg_pressure",
                 "avg_wind_speed", "qtd_sessions_rain", "pct_sessions_rain"],
}

_WINDOWS = ["life", "last10", "last20", "last40"]


@st.cache_data(ttl=3600)
def _load_features():
    fs_path = os.path.join(SILVER_DIR, "fs_driver_all.parquet")
    con = get_duckdb_connection()
    df = con.execute(f"""
        SELECT fs.*, b.full_name, b.team_name
        FROM read_parquet('{fs_path}') fs
        LEFT JOIN (
            SELECT driverid, full_name, team_name,
                   ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        ) b ON fs.driverid = b.driverid AND b.rn = 1
    """).fetchdf()
    con.close()
    df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    df["year"] = df["dt_ref"].dt.year
    df["full_name"] = df["full_name"].fillna(df["driverid"])
    return df


def _feature_explorer():
    st.subheader("Feature Explorer")
    st.caption("Browse feature-store metrics race by race. Metrics are point-in-time "
               "correct (computed from data before each race).")

    try:
        fs = _load_features()
    except Exception as e:
        st.error(f"Could not load feature store: {e}")
        st.info("Run the ETL pipeline first: `python -m etl.run_pipeline`")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        group = st.selectbox("Metric group", list(_FEATURE_GROUPS.keys()), key="fe_group")
    with col2:
        window = st.selectbox("Window", _WINDOWS, key="fe_window",
                              format_func=lambda w: {"life": "Career", "last10": "Last 10",
                                                     "last20": "Last 20", "last40": "Last 40"}[w])
    with col3:
        metrics_in_group = _FEATURE_GROUPS[group]
        available = [f"{m}_{window}" for m in metrics_in_group if f"{m}_{window}" in fs.columns]
        metric = st.selectbox("Metric", available, key="fe_metric",
                              format_func=lambda c: c.replace(f"_{window}", ""))

    if not metric:
        st.warning("No matching metric found for this group/window combination.")
        return

    mode = st.radio("View by", ["Drivers", "Teams"], horizontal=True, key="fe_mode")

    # --- Season range ---
    years = sorted(fs["year"].unique())
    min_yr, max_yr = int(years[0]), int(years[-1])

    full_career = st.checkbox("Full career", value=False, key="fe_full_career")
    if full_career:
        yr_lo, yr_hi = min_yr, max_yr
    else:
        yr_lo, yr_hi = st.slider("Seasons", min_yr, max_yr,
                                 (max_yr, max_yr), key="fe_years")

    fs_filtered = fs[(fs["year"] >= yr_lo) & (fs["year"] <= yr_hi)].sort_values("dt_ref")
    year_label = f"{yr_lo}—{yr_hi}" if yr_lo != yr_hi else str(yr_lo)
    metric_label = metric.replace(f"_{window}", "")

    if mode == "Drivers":
        all_drivers = sorted(fs_filtered["full_name"].unique())
        default = all_drivers[:5] if len(all_drivers) > 5 else all_drivers
        selected = st.multiselect("Select drivers", all_drivers, default=default,
                                  key="fe_drivers")
        if not selected:
            return
        plot = fs_filtered[fs_filtered["full_name"].isin(selected)].copy()
        if plot.empty:
            st.warning("No data for the selected drivers/seasons.")
            return
        color_map = {
            row["full_name"]: get_team_color(row["team_name"])
            for _, row in plot[["full_name", "team_name"]].drop_duplicates().iterrows()
        }
        fig = px.line(
            plot, x="dt_ref", y=metric, color="full_name",
            markers=True, color_discrete_map=color_map,
            labels={"dt_ref": "Race Date", metric: metric_label,
                    "full_name": "Driver"},
            title=f"{metric_label} ({window}) — {year_label}",
        )
    else:
        team_agg = (
            fs_filtered.groupby(["dt_ref", "team_name"])[metric]
            .mean()
            .reset_index()
        )
        all_teams = sorted(team_agg["team_name"].dropna().unique())
        default_teams = all_teams[:5] if len(all_teams) > 5 else all_teams
        selected_teams = st.multiselect("Select teams", all_teams,
                                        default=default_teams, key="fe_teams")
        if not selected_teams:
            return
        plot = team_agg[team_agg["team_name"].isin(selected_teams)].copy()
        if plot.empty:
            st.warning("No data for the selected teams/seasons.")
            return
        color_map = {t: get_team_color(t) for t in selected_teams}
        fig = px.line(
            plot, x="dt_ref", y=metric, color="team_name",
            markers=True, color_discrete_map=color_map,
            labels={"dt_ref": "Race Date", metric: metric_label,
                    "team_name": "Team"},
            title=f"{metric_label} ({window}) — {year_label} (team avg)",
        )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Champion Model Features (curated 17 features)
# ---------------------------------------------------------------------------

CHAMPION_FEATURES = [
    "avg_position_last10", "avg_grid_last10", "avg_overtakes_last10",
    "total_points_last10", "qtd_wins_last10", "qtd_podiums_last10",
    "qtd_top5_last10", "qtd_poles_last10", "qtd_pole_win_last10",
    "qtd_sessions_with_points_last10", "standing_position",
    "points_pct_of_leader", "gap_momentum_3r", "points_accel",
    "pct_leader_x_wins", "pct_leader_x_podiums", "pct_leader_x_points",
]

_CHAMPION_LABELS = {
    "avg_position_last10": "Avg Finish Pos (L10)",
    "avg_grid_last10": "Avg Grid Pos (L10)",
    "avg_overtakes_last10": "Avg Overtakes (L10)",
    "total_points_last10": "Total Points (L10)",
    "qtd_wins_last10": "Wins (L10)",
    "qtd_podiums_last10": "Podiums (L10)",
    "qtd_top5_last10": "Top 5 Finishes (L10)",
    "qtd_poles_last10": "Poles (L10)",
    "qtd_pole_win_last10": "Pole-to-Win (L10)",
    "qtd_sessions_with_points_last10": "Points Finishes (L10)",
    "standing_position": "Standings Position",
    "points_pct_of_leader": "Points % of Leader",
    "gap_momentum_3r": "Gap Momentum (3-race)",
    "points_accel": "Points Acceleration",
    "pct_leader_x_wins": "Leader% x Wins",
    "pct_leader_x_podiums": "Leader% x Podiums",
    "pct_leader_x_points": "Leader% x Points",
}


@st.cache_data(ttl=3600)
def _load_champion_abt():
    path = os.path.join(GOLD_DIR, "abt_champions_inseason.parquet")
    con = get_duckdb_connection()
    df = con.execute(f"""
        SELECT c.*, b.full_name, b.team_name
        FROM read_parquet('{path}') c
        LEFT JOIN (
            SELECT driverid, full_name, team_name,
                   ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        ) b ON c.driverid = b.driverid AND b.rn = 1
    """).fetchdf()
    con.close()
    df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    df["year"] = df["dt_ref"].dt.year
    df["full_name"] = df["full_name"].fillna(df["driverid"])
    return df


def _champion_features():
    st.subheader("Champion Model Features")
    st.caption("The 17 curated features used by the champion prediction model. "
               "Visualize how these evolve race-by-race for championship contenders.")

    try:
        df = _load_champion_abt()
    except Exception as e:
        st.error(f"Could not load champion ABT: {e}")
        return

    years = sorted(df["year"].unique(), reverse=True)
    year = st.selectbox("Season", years, key="champ_feat_year")
    df_yr = df[df["year"] == year].sort_values("dt_ref")

    # Pick top drivers by latest standing
    latest = df_yr.groupby("full_name").last().reset_index()
    if "standing_position" in latest.columns:
        top = latest.nsmallest(6, "standing_position")["full_name"].tolist()
    else:
        top = latest.nlargest(6, "total_points_last10")["full_name"].tolist()

    all_drivers = sorted(df_yr["full_name"].unique())
    selected = st.multiselect("Drivers", all_drivers, default=top, key="champ_feat_drivers")
    if not selected:
        return

    plot_df = df_yr[df_yr["full_name"].isin(selected)]

    # Feature selector
    avail = [f for f in CHAMPION_FEATURES if f in plot_df.columns]
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.selectbox("Feature (top)", avail, index=avail.index("points_pct_of_leader") if "points_pct_of_leader" in avail else 0,
                              format_func=lambda f: _CHAMPION_LABELS.get(f, f), key="champ_f1")
    with col2:
        default_idx = avail.index("standing_position") if "standing_position" in avail else min(1, len(avail)-1)
        feat2 = st.selectbox("Feature (bottom)", avail, index=default_idx,
                              format_func=lambda f: _CHAMPION_LABELS.get(f, f), key="champ_f2")

    color_map = {
        row["full_name"]: get_team_color(row["team_name"])
        for _, row in plot_df[["full_name", "team_name"]].drop_duplicates().iterrows()
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[_CHAMPION_LABELS.get(feat1, feat1),
                                        _CHAMPION_LABELS.get(feat2, feat2)])
    for driver in selected:
        d = plot_df[plot_df["full_name"] == driver]
        color = color_map.get(driver, "#888")
        fig.add_trace(go.Scatter(x=d["dt_ref"], y=d[feat1], mode="lines+markers",
                                  name=driver, line=dict(color=color), legendgroup=driver,
                                  showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["dt_ref"], y=d[feat2], mode="lines+markers",
                                  name=driver, line=dict(color=color), legendgroup=driver,
                                  showlegend=False), row=2, col=1)

    reverse_y2 = feat2 in ("standing_position", "avg_position_last10", "avg_grid_last10")
    if reverse_y2:
        fig.update_yaxes(autorange="reversed", row=2, col=1)

    fig.update_layout(height=600, hovermode="x unified",
                      title=f"Champion Features — {year}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap of champion features
    st.subheader("Feature Correlation (Champion Model)")
    corr_df = df_yr[avail].corr()
    labels = [_CHAMPION_LABELS.get(f, f) for f in avail]
    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_df.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig_corr.update_layout(height=550, title=f"Feature Correlations — {year}",
                           xaxis_tickangle=-45)
    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------------------------
# Constructor Model Features (curated 11 features)
# ---------------------------------------------------------------------------

TEAM_FEATURES = [
    "sum_wins_last10", "sum_podiums_last10", "sum_points_last10",
    "avg_position_last10", "avg_grid_last10", "team_standing_position",
    "team_points_pct_of_leader", "team_points_accel",
    "team_pct_leader_x_wins", "team_pct_leader_x_podiums",
    "team_pct_leader_x_points",
]

_TEAM_LABELS = {
    "sum_wins_last10": "Wins (L10 races)",
    "sum_podiums_last10": "Podiums (L10 races)",
    "sum_points_last10": "Points (L10 races)",
    "avg_position_last10": "Avg Finish Pos (L10)",
    "avg_grid_last10": "Avg Grid Pos (L10)",
    "team_standing_position": "Standings Position",
    "team_points_pct_of_leader": "Points % of Leader",
    "team_points_accel": "Points Acceleration",
    "team_pct_leader_x_wins": "Leader% x Wins",
    "team_pct_leader_x_podiums": "Leader% x Podiums",
    "team_pct_leader_x_points": "Leader% x Points",
}


@st.cache_data(ttl=3600)
def _load_team_abt():
    path = os.path.join(GOLD_DIR, "abt_teams_inseason.parquet")
    con = get_duckdb_connection()
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    con.close()
    df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    df["year"] = df["dt_ref"].dt.year
    return df


def _constructor_features():
    st.subheader("Constructor Model Features")
    st.caption("The 11 curated features used by the constructor champion prediction model.")

    try:
        df = _load_team_abt()
    except Exception as e:
        st.error(f"Could not load team ABT: {e}")
        return

    years = sorted(df["year"].unique(), reverse=True)
    year = st.selectbox("Season", years, key="team_feat_year")
    df_yr = df[df["year"] == year].sort_values("dt_ref")

    all_teams = sorted(df_yr["team_name"].dropna().unique())
    latest = df_yr.groupby("team_name").last().reset_index()
    if "team_standing_position" in latest.columns:
        top = latest.nsmallest(5, "team_standing_position")["team_name"].tolist()
    else:
        top = all_teams[:5]

    selected = st.multiselect("Teams", all_teams, default=top, key="team_feat_teams")
    if not selected:
        return

    plot_df = df_yr[df_yr["team_name"].isin(selected)]
    avail = [f for f in TEAM_FEATURES if f in plot_df.columns]

    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.selectbox("Feature (top)", avail,
                              index=avail.index("team_points_pct_of_leader") if "team_points_pct_of_leader" in avail else 0,
                              format_func=lambda f: _TEAM_LABELS.get(f, f), key="team_f1")
    with col2:
        default_idx = avail.index("team_standing_position") if "team_standing_position" in avail else min(1, len(avail)-1)
        feat2 = st.selectbox("Feature (bottom)", avail, index=default_idx,
                              format_func=lambda f: _TEAM_LABELS.get(f, f), key="team_f2")

    color_map = {t: get_team_color(t) for t in selected}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[_TEAM_LABELS.get(feat1, feat1),
                                        _TEAM_LABELS.get(feat2, feat2)])
    for team in selected:
        d = plot_df[plot_df["team_name"] == team]
        color = color_map.get(team, "#888")
        fig.add_trace(go.Scatter(x=d["dt_ref"], y=d[feat1], mode="lines+markers",
                                  name=team, line=dict(color=color), legendgroup=team,
                                  showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["dt_ref"], y=d[feat2], mode="lines+markers",
                                  name=team, line=dict(color=color), legendgroup=team,
                                  showlegend=False), row=2, col=1)

    reverse_y2 = feat2 in ("team_standing_position", "avg_position_last10", "avg_grid_last10")
    if reverse_y2:
        fig.update_yaxes(autorange="reversed", row=2, col=1)

    fig.update_layout(height=600, hovermode="x unified",
                      title=f"Constructor Features — {year}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation (Constructor Model)")
    corr_df = df_yr[avail].corr()
    labels = [_TEAM_LABELS.get(f, f) for f in avail]
    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_df.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig_corr.update_layout(height=500, title=f"Feature Correlations — {year}",
                           xaxis_tickangle=-45)
    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------------------------
# Departure Model Features (curated 24 features)
# ---------------------------------------------------------------------------

DEPARTURE_FEATURES = [
    "avg_position_last10", "avg_quali_position_last10", "qtd_top5_last10",
    "qtd_quali_top10_last10", "total_points_last10", "total_points_race_last10",
    "total_points_life", "total_points_race_life", "qtd_wins_life",
    "qtd_top5_life", "qtd_podiums_life", "qtd_grid_top5_life",
    "driver_age", "seasons_since_last_podium", "seasons_since_last_win",
    "team_tenure_years", "career_distinct_teams", "teammate_position_gap",
    "teammate_grid_gap", "team_points_share", "season_points_current",
    "season_dnf_rate", "trend_win_rate", "trend_podium_rate",
]

_DEPARTURE_LABELS = {
    "avg_position_last10": "Avg Finish Pos (L10)",
    "avg_quali_position_last10": "Avg Quali Pos (L10)",
    "qtd_top5_last10": "Top 5 Finishes (L10)",
    "qtd_quali_top10_last10": "Quali Top-10 (L10)",
    "total_points_last10": "Total Points (L10)",
    "total_points_race_last10": "Race Points (L10)",
    "total_points_life": "Career Points",
    "total_points_race_life": "Career Race Points",
    "qtd_wins_life": "Career Wins",
    "qtd_top5_life": "Career Top 5",
    "qtd_podiums_life": "Career Podiums",
    "qtd_grid_top5_life": "Career Grid Top 5",
    "driver_age": "Driver Age",
    "seasons_since_last_podium": "Seasons Since Last Podium",
    "seasons_since_last_win": "Seasons Since Last Win",
    "team_tenure_years": "Team Tenure (years)",
    "career_distinct_teams": "Career Teams",
    "teammate_position_gap": "Teammate Finish Gap",
    "teammate_grid_gap": "Teammate Grid Gap",
    "team_points_share": "Team Points Share",
    "season_points_current": "Season Points",
    "season_dnf_rate": "Season DNF Rate",
    "trend_win_rate": "Win Rate Trend",
    "trend_podium_rate": "Podium Rate Trend",
}

_DEPARTURE_GROUPS = {
    "Recent Form (L10)": ["avg_position_last10", "avg_quali_position_last10", "qtd_top5_last10",
                          "qtd_quali_top10_last10", "total_points_last10", "total_points_race_last10"],
    "Career Stats": ["total_points_life", "total_points_race_life", "qtd_wins_life",
                     "qtd_top5_life", "qtd_podiums_life", "qtd_grid_top5_life"],
    "Driver Profile": ["driver_age", "seasons_since_last_podium", "seasons_since_last_win",
                        "team_tenure_years", "career_distinct_teams"],
    "Team Dynamics": ["teammate_position_gap", "teammate_grid_gap", "team_points_share",
                      "season_points_current", "season_dnf_rate"],
    "Performance Trends": ["trend_win_rate", "trend_podium_rate"],
}


@st.cache_data(ttl=3600)
def _load_departure_abt():
    path = os.path.join(GOLD_DIR, "abt_departures_inseason.parquet")
    con = get_duckdb_connection()
    df = con.execute(f"""
        SELECT d.*, b.full_name, b.team_name
        FROM read_parquet('{path}') d
        LEFT JOIN (
            SELECT driverid, full_name, team_name,
                   ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY event_date DESC) AS rn
            FROM read_parquet('{BRONZE_PATH}')
        ) b ON d.driverid = b.driverid AND b.rn = 1
    """).fetchdf()
    con.close()
    df["dt_ref"] = pd.to_datetime(df["dt_ref"])
    df["year"] = df["dt_ref"].dt.year
    df["full_name"] = df["full_name"].fillna(df["driverid"])
    return df


def _departure_features():
    st.subheader("Departure Model Features")
    st.caption("The 24 curated features used by the driver departure model, organized by category.")

    try:
        df = _load_departure_abt()
    except Exception as e:
        st.error(f"Could not load departure ABT: {e}")
        return

    years = sorted(df["year"].unique(), reverse=True)
    year = st.selectbox("Season", years, key="dep_feat_year")
    df_yr = df[df["year"] == year].sort_values("dt_ref")

    # Default: pick a mix of drivers
    all_drivers = sorted(df_yr["full_name"].unique())
    default = all_drivers[:6] if len(all_drivers) >= 6 else all_drivers
    selected = st.multiselect("Drivers", all_drivers, default=default, key="dep_feat_drivers")
    if not selected:
        return

    plot_df = df_yr[df_yr["full_name"].isin(selected)]

    # Group selector
    group = st.selectbox("Feature group", list(_DEPARTURE_GROUPS.keys()), key="dep_group")
    group_feats = [f for f in _DEPARTURE_GROUPS[group] if f in plot_df.columns]

    if not group_feats:
        st.warning("No features available for this group.")
        return

    color_map = {
        row["full_name"]: get_team_color(row["team_name"])
        for _, row in plot_df[["full_name", "team_name"]].drop_duplicates().iterrows()
    }

    # Plot up to 2 features side by side if group has multiple
    n_plots = min(2, len(group_feats))
    cols = st.columns(n_plots)
    feat_selections = []
    for i, col in enumerate(cols):
        with col:
            f = st.selectbox(f"Feature {i+1}", group_feats,
                             index=min(i, len(group_feats)-1),
                             format_func=lambda f: _DEPARTURE_LABELS.get(f, f),
                             key=f"dep_f{i}")
            feat_selections.append(f)

    fig = make_subplots(rows=1, cols=n_plots, shared_xaxes=True,
                        subplot_titles=[_DEPARTURE_LABELS.get(f, f) for f in feat_selections],
                        horizontal_spacing=0.08)
    for driver in selected:
        d = plot_df[plot_df["full_name"] == driver]
        color = color_map.get(driver, "#888")
        for i, feat in enumerate(feat_selections):
            fig.add_trace(go.Scatter(
                x=d["dt_ref"], y=d[feat], mode="lines+markers",
                name=driver, line=dict(color=color), legendgroup=driver,
                showlegend=(i == 0),
            ), row=1, col=i+1)
            if feat in ("avg_position_last10", "avg_quali_position_last10"):
                fig.update_yaxes(autorange="reversed", row=1, col=i+1)

    fig.update_layout(height=450, hovermode="x unified",
                      title=f"Departure Features ({group}) — {year}")
    st.plotly_chart(fig, use_container_width=True)

    # End-of-season snapshot: bar chart comparing drivers on selected feature
    st.subheader("End-of-Season Snapshot")
    snap_feat = st.selectbox("Feature to compare", group_feats,
                             format_func=lambda f: _DEPARTURE_LABELS.get(f, f),
                             key="dep_snap_feat")
    latest = plot_df.groupby("full_name").last().reset_index()
    latest = latest.sort_values(snap_feat, ascending=False)
    colors = [color_map.get(d, "#888") for d in latest["full_name"]]
    fig_bar = go.Figure(go.Bar(
        x=latest["full_name"], y=latest[snap_feat],
        marker_color=colors,
    ))
    fig_bar.update_layout(
        title=f"{_DEPARTURE_LABELS.get(snap_feat, snap_feat)} — Latest Values ({year})",
        xaxis_title="Driver", yaxis_title=_DEPARTURE_LABELS.get(snap_feat, snap_feat),
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def _points_distribution(df):
    st.subheader("Points Distribution by Season")
    races = df[df["mode"] == "Race"].copy()
    year_range = st.slider(
        "Year range",
        int(races["year"].min()),
        int(races["year"].max()),
        (int(races["year"].max()) - 5, int(races["year"].max())),
    )
    filtered = races[(races["year"] >= year_range[0]) & (races["year"] <= year_range[1])]

    season_points = (
        filtered.groupby(["year", "driverid", "full_name"])["points"]
        .sum()
        .reset_index()
    )

    fig = px.box(
        season_points,
        x="year",
        y="points",
        labels={"year": "Season", "points": "Total Points"},
        title="Driver Points Distribution per Season",
    )
    st.plotly_chart(fig, use_container_width=True)


def _win_rate(df):
    st.subheader("Win Rate Over Career")
    races = df[df["mode"] == "Race"].copy()

    # Calculate cumulative win rate per driver
    drivers = sorted(races["full_name"].dropna().unique())
    selected = st.multiselect("Select drivers", drivers, default=drivers[:5])
    if not selected:
        return

    filtered = races[races["full_name"].isin(selected)].sort_values("event_date")
    filtered["is_win"] = (filtered["position"] == 1).astype(int)
    filtered["cum_races"] = filtered.groupby("driverid").cumcount() + 1
    filtered["cum_wins"] = filtered.groupby("driverid")["is_win"].cumsum()
    filtered["win_rate"] = filtered["cum_wins"] / filtered["cum_races"]

    fig = px.line(
        filtered,
        x="cum_races",
        y="win_rate",
        color="full_name",
        labels={
            "cum_races": "Career Races",
            "win_rate": "Cumulative Win Rate",
            "full_name": "Driver",
        },
        title="Cumulative Win Rate Over Career",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def _team_comparison(df):
    st.subheader("Team Comparison — Total Points per Season")
    year_range = st.slider(
        "Year range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].max()) - 3, int(df["year"].max())),
        key="team_year",
    )
    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    filtered = filtered[filtered["mode"] == "Race"]

    team_points = (
        filtered.groupby(["year", "team_name"])["points"].sum().reset_index()
    )

    fig = px.bar(
        team_points,
        x="year",
        y="points",
        color="team_name",
        barmode="group",
        labels={"year": "Season", "points": "Total Points", "team_name": "Team"},
        title="Team Points Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)


def _grid_vs_finish(df):
    st.subheader("Grid Position vs Finish Position")
    races = df[df["mode"] == "Race"].copy()
    year = st.selectbox(
        "Season",
        sorted(races["year"].unique(), reverse=True),
        key="grid_year",
    )
    filtered = races[races["year"] == year].dropna(subset=["grid_position", "position"])

    fig = px.scatter(
        filtered,
        x="grid_position",
        y="position",
        color="team_name",
        hover_data=["full_name", "event_name"],
        labels={
            "grid_position": "Grid Position",
            "position": "Finish Position",
            "team_name": "Team",
        },
        title=f"Grid vs Finish Position — {year}",
    )
    # Add diagonal reference line
    max_pos = max(filtered["grid_position"].max(), filtered["position"].max())
    fig.add_shape(
        type="line", x0=1, y0=1, x1=max_pos, y1=max_pos,
        line=dict(dash="dash", color="gray"),
    )
    fig.update_layout(
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points below the diagonal = gained positions. Above = lost positions.")


def _career_trajectories(df):
    st.subheader("Driver Career Trajectories")
    races = df[df["mode"] == "Race"].copy()

    drivers = sorted(races["full_name"].dropna().unique())
    selected = st.multiselect("Select drivers", drivers, default=drivers[:4], key="career_drivers")
    if not selected:
        return

    filtered = races[races["full_name"].isin(selected)]
    avg_pos = (
        filtered.groupby(["year", "full_name"])["position"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        avg_pos,
        x="year",
        y="position",
        color="full_name",
        markers=True,
        labels={
            "year": "Season",
            "position": "Average Finish Position",
            "full_name": "Driver",
        },
        title="Average Finish Position by Season",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


def _season_progression(df):
    st.subheader("Season Points Progression")
    races = df[df["mode"] == "Race"].copy()

    year = st.selectbox(
        "Season",
        sorted(races["year"].unique(), reverse=True),
        key="prog_year",
    )
    filtered = races[races["year"] == year].sort_values("round_number")

    # Cumulative points per driver over rounds
    filtered["cum_points"] = filtered.groupby("driverid")["points"].cumsum()

    # Show top 10 drivers by total points
    top_drivers = (
        filtered.groupby("full_name")["points"].sum()
        .nlargest(10).index.tolist()
    )
    filtered = filtered[filtered["full_name"].isin(top_drivers)]

    fig = px.line(
        filtered,
        x="round_number",
        y="cum_points",
        color="full_name",
        markers=True,
        labels={
            "round_number": "Round",
            "cum_points": "Cumulative Points",
            "full_name": "Driver",
        },
        title=f"Season Points Progression — {year}",
    )
    st.plotly_chart(fig, use_container_width=True)
