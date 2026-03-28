"""Train and compare batch models for F1 driver departure prediction.

Uses the in-season ABT (one row per driver per race) so predictions can be
plotted as a time series evolving race by race through the season.
"""

import os

import duckdb

from ml.model_selection import get_batch_models
from ml.utils import train_and_compare_batch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ABT_PATH = os.path.join(BASE_DIR, "data", "gold", "abt_departures_inseason.parquet")


def train_departure_models():
    print("=" * 60)
    print("F1 Driver Departure Prediction — Multi-Model Training (in-season ABT)")
    print("=" * 60)

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{ABT_PATH}')").fetchdf()
    con.close()

    print(f"ABT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Departure rate: {df['fl_departed'].mean():.4f}")

    batch_candidates = get_batch_models()
    comparison, best = train_and_compare_batch(
        df=df,
        target_col="fl_departed",
        id_cols=["driverid"],
        experiment_name="f1_departure",
        candidates=batch_candidates,
        remove_late_rounds=False,
        oot_year=2025,
    )

    print(f"\nDone. Best model: {best}")
    return comparison


if __name__ == "__main__":
    train_departure_models()
