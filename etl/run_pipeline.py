"""Run the full ETL pipeline: collect -> bronze -> silver -> gold."""

import argparse

from etl.collect import CollectResults
from etl.bronze import build_bronze
from etl.silver import build_silver
from etl.gold import build_gold


def run_pipeline(years=None, modes=None):
    if years is None:
        years = list(range(2020, 2026))
    if modes is None:
        modes = ["R", "S"]

    print("=" * 60)
    print("F1 Analytics — Full ETL Pipeline")
    print("=" * 60)

    print("\nStep 1/4: Collecting data from FastF1...")
    collector = CollectResults(years=years, modes=modes)
    collector.process_years()

    print("\nStep 2/4: Building bronze layer...")
    build_bronze()

    print("\nStep 3/4: Building silver layer (feature store)...")
    build_silver()

    print("\nStep 4/4: Building gold layer (ABTs)...")
    build_gold()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full F1 ETL pipeline")
    parser.add_argument(
        "--years", "-y", nargs="+", type=int, default=list(range(2020, 2026))
    )
    parser.add_argument("--modes", "-m", nargs="+", default=["R", "S"])
    args = parser.parse_args()

    run_pipeline(years=args.years, modes=args.modes)
