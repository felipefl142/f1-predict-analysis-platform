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

    # Year filter
    years = sorted(data["year"].unique(), reverse=True)
    selected_years = st.multiselect("Season", years, default=[years[0]] if years else [])
    if not selected_years:
        st.warning("Select at least one season.")
        return

    data_filtered = data[data["year"].isin(selected_years)]

    # Get top 3 drivers by latest probability as defaults
    latest = data_filtered[data_filtered["dt_ref"] == data_filtered["dt_ref"].max()]
    top3 = latest.nlargest(3, "prob_champion")["driverid"].tolist()

    drivers = sorted(data_filtered["driverid"].unique())
    driver_names = (
        data_filtered[["driverid", "full_name"]]
        .drop_duplicates()
        .set_index("driverid")["full_name"]
        .to_dict()
    )
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

    plot_data = data_filtered[data_filtered["driverid"].isin(selected_drivers)]

    # Build color map
    color_map = {}
    for _, row in plot_data[["driverid", "team_color"]].drop_duplicates().iterrows():
        color_map[row["driverid"]] = get_team_color(None, row["team_color"])

    # Line chart
    fig = px.line(
        plot_data,
        x="dt_ref",
        y="prob_champion",
        color="full_name",
        color_discrete_map={
            driver_names.get(d, d): color_map.get(d, "#999")
            for d in selected_drivers
        },
        labels={
            "dt_ref": "Post-Race Date",
            "prob_champion": "Championship Win Probability",
            "full_name": "Driver",
        },
        title="Championship Win Probability Over Time",
    )
    fig.update_layout(yaxis_tickformat=".0%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    pivot = plot_data.pivot_table(
        index="dt_ref", columns="full_name", values="prob_champion"
    ).reset_index()
    st.dataframe(
        pivot.style.format(
            {c: "{:.2%}" for c in pivot.columns if c != "dt_ref"}
        ),
        use_container_width=True,
    )

    # Model comparison expander
    with st.expander("Model Comparison"):
        comp = _get_model_comparison("f1_champion")
        if not comp.empty:
            st.dataframe(
                comp.style.format({
                    "auc_train": "{:.4f}",
                    "auc_test": "{:.4f}",
                    "auc_oot": "{:.4f}",
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

    data["year"] = pd.to_datetime(data["dt_ref"]).dt.year
    years = sorted(data["year"].unique(), reverse=True)
    selected_year = st.selectbox("Season", years, index=0)

    year_data = data[data["year"] == selected_year]
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
        comp = _get_model_comparison("f1_constructor_champion")
        if not comp.empty:
            st.dataframe(comp, use_container_width=True)


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
