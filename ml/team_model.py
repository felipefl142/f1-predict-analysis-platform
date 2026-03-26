"""Train and compare batch + online models for F1 constructor champion prediction."""

import os

import duckdb

from ml.model_selection import get_batch_models, get_online_models
from ml.utils import train_and_compare_batch, train_and_compare_online

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ABT_PATH = os.path.join(BASE_DIR, "data", "gold", "abt_teams.parquet")


def train_team_models():
    print("=" * 60)
    print("F1 Constructor Champion Prediction — Multi-Model Training")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Constructor champion rate: {df['fl_constructor_champion'].mean():.4f}")

    # --- Batch models ---
    print("\n--- BATCH MODELS ---")
    batch_candidates = get_batch_models(balanced=True)
    batch_comparison, best_batch = train_and_compare_batch(
        df=df,
        target_col="fl_constructor_champion",
        id_cols=["teamid"],
        experiment_name="f1_constructor_champion",
        candidates=batch_candidates,
        balanced=True,
    )

    # --- Online models ---
    print("\n--- ONLINE MODELS ---")
    online_candidates = get_online_models(balanced=True)
    online_comparison, best_online = train_and_compare_online(
        df=df,
        target_col="fl_constructor_champion",
        id_cols=["teamid"],
        experiment_name="f1_constructor_champion",
        candidates=online_candidates,
    )

    print(f"\nDone. Best batch: {best_batch} | Best online: {best_online}")
    return batch_comparison, online_comparison


if __name__ == "__main__":
    train_team_models()
