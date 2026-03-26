"""Tab 2: Exploratory Data Analysis — Interactive F1 visualizations."""

import streamlit as st
import pandas as pd
import plotly.express as px

from app.helpers import get_duckdb_connection, BRONZE_PATH


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
            "Points Distribution by Season",
            "Win Rate Over Career",
            "Team Comparison",
            "Grid vs Finish Position",
            "Driver Career Trajectories",
            "Season Points Progression",
        ],
    )

    if viz == "Points Distribution by Season":
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
