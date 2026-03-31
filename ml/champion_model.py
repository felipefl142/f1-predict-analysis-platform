"""Train and compare batch models for F1 champion prediction.

Uses the in-season ABT (one row per driver per race) so predictions can be
plotted as a time series evolving race by race through the season.
"""

import argparse
import os

import duckdb

from ml.model_selection import get_batch_models
from ml.utils import train_and_compare_batch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ABT_PATH = os.path.join(BASE_DIR, "data", "gold", "abt_champions_inseason.parquet")

# Curated feature set — 15 features
# Removed: season_fraction/season_race_number (target leakage via season progress),
# total_points_sprint_last10/qtd_seasons_last10 (zero importance),
# quali features except qtd_pole_win_last10 (negligible importance),
# standings_momentum_3r (zero importance in all models),
# qtd_finished_last10 (non-discriminative — nearly all drivers finish in modern F1),
# standing_position/points_gap_to_leader/points_pct_of_leader/clinch_proximity
# (tautological with post-clinch target; removed to let model learn real performance signal).
CHAMPION_FEATURES = [
    # Core performance (last 10 sessions)
    "avg_position_last10",
    "avg_grid_last10",
    "avg_overtakes_last10",
    "total_points_last10",
    # Wins & podiums
    "qtd_wins_last10",
    "qtd_podiums_last10",
    "qtd_top5_last10",
    "qtd_poles_last10",
    "qtd_pole_win_last10",
    # Consistency
    "qtd_sessions_with_points_last10",
    # Momentum
    "gap_momentum_3r",
    "points_accel",
    # Interactions
    "pct_leader_x_wins",
    "pct_leader_x_podiums",
    "pct_leader_x_points",
]


def train_champion_models(skip_logreg=False, skip_boosting=False):
    print("=" * 60)
    print("F1 Champion Prediction — Multi-Model Training (in-season ABT)")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Champion rate: {df['fl_champion'].mean():.4f}")

    batch_candidates = get_batch_models(skip_logreg=skip_logreg, skip_boosting=skip_boosting, oversampling=False)
    comparison, best = train_and_compare_batch(
        df=df,
        target_col="fl_champion",
        id_cols=["driverid"],
        experiment_name="f1_champion",
        candidates=batch_candidates,
        remove_late_rounds=False,
        oot_year=[2024, 2025, 2026],
        scoring="average_precision",
        feature_cols=CHAMPION_FEATURES,
    )

    print(f"\nDone. Best model: {best}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nologreg", action="store_true", help="Skip LogisticRegression")
    parser.add_argument("--noboosting", action="store_true", help="Skip LightGBM and XGBoost")
    args = parser.parse_args()
    train_champion_models(skip_logreg=args.nologreg, skip_boosting=args.noboosting)
