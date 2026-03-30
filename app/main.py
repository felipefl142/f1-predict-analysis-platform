"""F1 Analytics Platform — Streamlit entry point."""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="F1 Analytics Platform",
    page_icon=":racing_car:",
    layout="wide",
)

st.title("F1 Analytics Platform :checkered_flag:")
st.markdown(
    "Data engineering + ML predictions for Formula 1. "
    "Built with FastF1, DuckDB, scikit-learn, and Streamlit. "
    "Inspired by [TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake)."
)

tab_pred, tab_models, tab_eda, tab_sql = st.tabs(
    [":crystal_ball: Predictions", ":microscope: Model Comparison",
     ":bar_chart: EDA", ":duck: DuckDB Console"]
)

with tab_pred:
    from app.tab_predictions import render_predictions
    render_predictions()

with tab_models:
    from app.tab_model_comparison import render_model_comparison
    render_model_comparison()

with tab_eda:
    from app.tab_eda import render_eda
    render_eda()

with tab_sql:
    from app.tab_duckdb import render_duckdb
    render_duckdb()
