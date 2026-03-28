"""Tab 2: Exploratory Data Analysis — Interactive F1 visualizations."""

import os

import streamlit as st
import pandas as pd
import plotly.express as px

from app.helpers import get_duckdb_connection, get_team_color, BRONZE_PATH, SILVER_DIR


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
