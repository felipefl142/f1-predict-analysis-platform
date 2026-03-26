"""Shared UI helpers for the Streamlit app."""

import os

import duckdb

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
BRONZE_PATH = os.path.join(DATA_DIR, "bronze", "results.parquet")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")


def get_duckdb_connection():
    """Return a fresh DuckDB in-memory connection."""
    return duckdb.connect()


def format_team_color(hex_str):
    """Normalize team color hex string."""
    if hex_str is None or str(hex_str) == "nan":
        return "#999999"
    hex_str = str(hex_str).strip()
    if hex_str.startswith("#"):
        return hex_str.lower()
    return f"#{hex_str}".lower()


# Fallback team colors in case FastF1 data is missing
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
}


def get_team_color(team_name, team_color=None):
    """Get team color, using fallback if needed."""
    if team_color and str(team_color) != "nan":
        return format_team_color(team_color)
    return TEAM_COLORS.get(team_name, "#999999")


AVAILABLE_TABLES = {
    "Bronze — Race Results": f"read_parquet('{BRONZE_PATH}')",
    "Silver — Features (Lifetime)": f"read_parquet('{SILVER_DIR}/fs_driver_life.parquet')",
    "Silver — Features (Last 10)": f"read_parquet('{SILVER_DIR}/fs_driver_last10.parquet')",
    "Silver — Features (Last 20)": f"read_parquet('{SILVER_DIR}/fs_driver_last20.parquet')",
    "Silver — Features (Last 40)": f"read_parquet('{SILVER_DIR}/fs_driver_last40.parquet')",
    "Silver — Features (All Windows)": f"read_parquet('{SILVER_DIR}/fs_driver_all.parquet')",
    "Gold — Champion ABT": f"read_parquet('{GOLD_DIR}/abt_champions.parquet')",
    "Gold — Teams ABT": f"read_parquet('{GOLD_DIR}/abt_teams.parquet')",
    "Gold — Departures ABT": f"read_parquet('{GOLD_DIR}/abt_departures.parquet')",
}
