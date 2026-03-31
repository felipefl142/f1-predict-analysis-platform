"""Train and compare batch models for F1 constructor champion prediction.

Uses the in-season ABT (one row per team per race) so predictions can be
plotted as a time series evolving race by race through the season.
"""

import argparse
import os

import duckdb

from ml.model_selection import get_batch_models
from ml.utils import train_and_compare_batch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ABT_PATH = os.path.join(BASE_DIR, "data", "gold", "abt_teams_inseason.parquet")

# Curated feature set — pruned for the team model
# Removed: season_fraction/season_race_number/season_total_races (data leakage
# via season progress), num_drivers/sum_sessions_life/avg_sessions_life/
# sum_quali_last10 (zero or near-zero importance), all last20 window features
# (redundant with last10 + life).
TEAM_FEATURES = [
    # Core performance (last 10 sessions)
    "avg_position_last10",
    "avg_grid_last10",
    "sum_points_last10",
    "sum_wins_last10",
    "sum_podiums_last10",
    # Lifetime performance
    "avg_position_life",
    "avg_podiums_life",
    "avg_wins_life",
    "avg_points_life",
    "sum_podiums_life",
    "sum_wins_life",
    "sum_points_life",
    # Qualifying
    "sum_quali_top10_life",
    "sum_quali_top3_life",
    "avg_quali_position_last10",
    "avg_quali_position_life",
    "sum_quali_life",
    "sum_quali_pole_life",
    "sum_quali_pole_last10",
    # Season context (standings only, no season progress)
    "team_standing_position",
    "team_points_gap_to_leader",
    "team_points_pct_of_leader",
    # Momentum
    "team_standings_momentum_3r",
    "team_gap_momentum_3r",
    "team_points_accel",
    # Clinch proximity
    "team_clinch_proximity",
    # Interactions
    "team_pct_leader_x_wins",
    "team_pct_leader_x_podiums",
    "team_pct_leader_x_points",
]


def train_team_models(skip_logreg=False):
    print("=" * 60)
    print("F1 Constructor Champion Prediction — Multi-Model Training (in-season ABT)")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Constructor champion rate: {df['fl_constructor_champion'].mean():.4f}")

    batch_candidates = get_batch_models(skip_logreg=skip_logreg, oversampling=True)
    comparison, best = train_and_compare_batch(
        df=df,
        target_col="fl_constructor_champion",
        id_cols=["teamid"],
        experiment_name="f1_constructor_champion",
        candidates=batch_candidates,
        remove_late_rounds=False,
        oot_year=2025,
        scoring="combined",
        feature_cols=TEAM_FEATURES,
    )

    print(f"\nDone. Best model: {best}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nologreg", action="store_true", help="Skip LogisticRegression")
    args = parser.parse_args()
    train_team_models(skip_logreg=args.nologreg)
