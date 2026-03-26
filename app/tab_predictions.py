"""Tab 1: ML Predictions — Champion, Best Team, Driver Departures."""

import streamlit as st
import pandas as pd
import plotly.express as px

from app.helpers import get_team_color


@st.cache_resource
def _load_champion_model():
    from ml.predict import load_best_model
    return load_best_model("f1_champion")


@st.cache_resource
def _load_team_model():
    from ml.predict import load_best_model
    return load_best_model("f1_constructor_champion")


@st.cache_resource
def _load_departure_model():
    from ml.predict import load_best_model
    return load_best_model("f1_departure")


@st.cache_data(ttl=3600)
def _get_champion_predictions():
    from ml.predict import predict_champions
    model, _ = _load_champion_model()
    return predict_champions(model)


@st.cache_data(ttl=3600)
def _get_champion_season(year):
    from ml.predict import predict_champion_season
    model, _ = _load_champion_model()
    return predict_champion_season(year, model)


@st.cache_data(ttl=3600)
def _get_team_predictions():
    from ml.predict import predict_teams
    model, _ = _load_team_model()
    return predict_teams(model)


@st.cache_data(ttl=3600)
def _get_departure_predictions():
    from ml.predict import predict_departures
    model, _ = _load_departure_model()
    return predict_departures(model)


@st.cache_data(ttl=3600)
def _get_model_comparison(experiment_name):
    from ml.predict import get_model_comparison
    return get_model_comparison(experiment_name)


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


def _render_champion_predictions():
    st.subheader("Who Will Be Champion?")

    try:
        data = _get_champion_predictions()
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        st.info("Make sure you've run the ETL pipeline and trained the models first.")
        return

    # Season selector — uses prediction_year (the year being predicted, not stats year)
    years = sorted(data["prediction_year"].unique(), reverse=True)
    selected_year = st.selectbox("Season", years, index=0)

    # For the selected year, offer in-season view if available
    current_year = pd.Timestamp.now().year
    if selected_year >= current_year:
        view_mode = st.radio(
            "Prediction mode",
            ["Pre-season (based on prior year stats)", "Current (latest available data)"],
            horizontal=True,
        )
        if view_mode.startswith("Current"):
            try:
                season_data = _get_champion_season(selected_year)
                as_of = pd.to_datetime(season_data["data_as_of"].max()).strftime("%b %d, %Y")
                st.caption(f"Data as of: {as_of}. Run `etl.collect --years {selected_year - 1} {selected_year}` + `etl.silver` to update.")
                _render_champion_bar(season_data, selected_year)
            except Exception as e:
                st.error(f"Could not load current-season data: {e}")
            with st.expander("Model Comparison"):
                _render_model_comparison_table("f1_champion")
            return

    data_year = data[data["prediction_year"] == selected_year]

    drivers = sorted(data_year["driverid"].unique())
    driver_names = (
        data_year[["driverid", "full_name"]]
        .drop_duplicates()
        .set_index("driverid")["full_name"]
        .to_dict()
    )
    top3 = data_year.nlargest(3, "prob_champion")["driverid"].tolist()
    default_drivers = [d for d in top3 if d in drivers]
    selected_drivers = st.multiselect(
        "Drivers",
        drivers,
        default=default_drivers,
        format_func=lambda x: driver_names.get(x, x),
    )
    if not selected_drivers:
        st.warning("Select at least one driver.")
        return

    plot_data = data_year[data_year["driverid"].isin(selected_drivers)]

    # Since each driver has one pre-season point per year, show a bar chart
    _render_champion_bar(plot_data, selected_year)

    with st.expander("Model Comparison"):
        _render_model_comparison_table("f1_champion")


def _render_champion_bar(data, year):
    """Bar chart of championship win probabilities for a single season."""
    plot = data.sort_values("prob_champion", ascending=False)
    color_map = {}
    for _, row in plot[["driverid", "team_color"]].drop_duplicates().iterrows():
        color_map[row.get("full_name", row["driverid"])] = get_team_color(None, row.get("team_color"))

    name_col = "full_name" if "full_name" in plot.columns else "driverid"
    fig = px.bar(
        plot,
        x=name_col,
        y="prob_champion",
        color=name_col,
        color_discrete_map=color_map,
        labels={name_col: "Driver", "prob_champion": "Championship Win Probability"},
        title=f"{year} Championship Win Probability (pre-season model)",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        plot[[name_col, "prob_champion"]].style.format({"prob_champion": "{:.2%}"}),
        use_container_width=True,
    )


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


def _render_team_predictions():
    st.subheader("Which Team Will Be Best?")

    try:
        data = _get_team_predictions()
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        st.info("Make sure you've run the ETL pipeline and trained the models first.")
        return

    years = sorted(data["prediction_year"].unique(), reverse=True)
    selected_year = st.selectbox("Season", years, index=0)

    year_data = data[data["prediction_year"] == selected_year]
    latest = year_data[year_data["dt_ref"] == year_data["dt_ref"].max()]
    latest = latest.sort_values("prob_constructor_champion", ascending=False)

    fig = px.bar(
        latest,
        x="team_name",
        y="prob_constructor_champion",
        color="team_name",
        labels={
            "team_name": "Team",
            "prob_constructor_champion": "Constructor Championship Probability",
        },
        title=f"Constructor Championship Probability — {selected_year}",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        latest[["team_name", "prob_constructor_champion"]].style.format(
            {"prob_constructor_champion": "{:.2%}"}
        ),
        use_container_width=True,
    )

    with st.expander("Model Comparison"):
        _render_model_comparison_table("f1_constructor_champion")


def _render_departure_predictions():
    st.subheader("Who Might Leave F1?")
    st.caption(
        "Note: This model captures performance-correlated departures only. "
        "Contract decisions, team politics, and personal choices are not captured."
    )

    try:
        data = _get_departure_predictions()
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        st.info("Make sure you've run the ETL pipeline and trained the models first.")
        return

    # Show latest predictions for most recent year
    latest_year = data["year"].max()
    latest = data[data["year"] == latest_year]
    latest_date = latest[latest["dt_ref"] == latest["dt_ref"].max()]
    latest_date = latest_date.sort_values("prob_departure", ascending=False)

    fig = px.bar(
        latest_date,
        x="full_name",
        y="prob_departure",
        color="team_name",
        labels={
            "full_name": "Driver",
            "prob_departure": "Departure Probability",
            "team_name": "Team",
        },
        title=f"Driver Departure Probability — After {latest_year} Season",
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        latest_date[["full_name", "team_name", "prob_departure"]].style.format(
            {"prob_departure": "{:.2%}"}
        ),
        use_container_width=True,
    )

    with st.expander("Model Comparison"):
        comp = _get_model_comparison("f1_departure")
        if not comp.empty:
            st.dataframe(comp, use_container_width=True)
