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

# Curated feature set — clinch_proximity/team_points_gap_to_leader removed
# (tautological post-clinch). team_standing_position and team_points_pct_of_leader
# are NOT leakage — they reflect publicly available standings at each race date
# and naturally grow more informative as the season progresses.
TEAM_FEATURES = [
    # Team performance (last 10 sessions)
    "sum_wins_last10",
    "sum_podiums_last10",
    "sum_points_last10",
    "avg_position_last10",
    "avg_grid_last10",
    # Standings context (point-in-time, available at prediction time)
    "team_standing_position",
    "team_points_pct_of_leader",
    # Momentum
    "team_points_accel",
    # Interactions
    "team_pct_leader_x_wins",
    "team_pct_leader_x_podiums",
    "team_pct_leader_x_points",
]


def train_team_models(skip_logreg=False, skip_boosting=False):
    print("=" * 60)
    print("F1 Constructor Champion Prediction — Multi-Model Training (in-season ABT)")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Constructor champion rate: {df['fl_constructor_champion'].mean():.4f}")

    batch_candidates = get_batch_models(skip_logreg=skip_logreg, skip_boosting=skip_boosting, oversampling=False)
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
    parser.add_argument("--noboosting", action="store_true", help="Skip LightGBM and XGBoost")
    args = parser.parse_args()
    train_team_models(skip_logreg=args.nologreg, skip_boosting=args.noboosting)
