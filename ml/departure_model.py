"""Train and compare batch models for F1 driver departure prediction.

Uses the in-season ABT (one row per driver per race) so predictions can be
plotted as a time series evolving race by race through the season.
"""

import argparse
import os

import duckdb

from ml.model_selection import get_batch_models
from ml.utils import train_and_compare_batch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ABT_PATH = os.path.join(BASE_DIR, "data", "gold", "abt_departures_inseason.parquet")

# Curated feature set — 24 features
# Removed: all weather features (no causal link to departures), sprint-specific
# features, redundant last20/last40 windows, season_fraction/season_race_number/
# season_total_races (leakage via season progress).
DEPARTURE_FEATURES = [
    # Recent performance (last 10 sessions)
    "avg_position_last10",
    "avg_quali_position_last10",
    "qtd_top5_last10",
    "qtd_quali_top10_last10",
    "total_points_last10",
    "total_points_race_last10",
    # Lifetime performance (selective)
    "total_points_life",
    "total_points_race_life",
    "qtd_wins_life",
    "qtd_top5_life",
    "qtd_podiums_life",
    "qtd_grid_top5_life",
    # Career stage & departure-specific
    "driver_age",
    "seasons_since_last_podium",
    "seasons_since_last_win",
    "team_tenure_years",
    "career_distinct_teams",
    # Teammate comparison
    "teammate_position_gap",
    "teammate_grid_gap",
    "team_points_share",
    # Season context
    "season_points_current",
    "season_dnf_rate",
    # Performance trends
    "trend_win_rate",
    "trend_podium_rate",
]


def train_departure_models(skip_logreg=False):
    print("=" * 60)
    print("F1 Driver Departure Prediction — Multi-Model Training (in-season ABT)")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Departure rate: {df['fl_departed'].mean():.4f}")

    batch_candidates = get_batch_models(skip_logreg=skip_logreg)
    comparison, best = train_and_compare_batch(
        df=df,
        target_col="fl_departed",
        id_cols=["driverid"],
        experiment_name="f1_departure",
        candidates=batch_candidates,
        remove_late_rounds=True,
        oot_year=2025,
        scoring="roc_auc",
        feature_cols=DEPARTURE_FEATURES,
    )

    print(f"\nDone. Best model: {best}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nologreg", action="store_true", help="Skip LogisticRegression")
    args = parser.parse_args()
    train_departure_models(skip_logreg=args.nologreg)
