"""Collect F1 session results via FastF1 and save as Parquet files to data/raw/."""

import argparse
import os
import time

import fastf1
from fastf1.exceptions import RateLimitExceededError
import pandas as pd
from tqdm import tqdm

# Enable FastF1 cache to speed up repeated calls
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "ff1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# FastF1 allows 500 calls/hour. Wait this long (seconds) when rate-limited.
RATE_LIMIT_WAIT = 120
MAX_RETRIES = 5


class CollectResults:
    def __init__(self, years=None, modes=None, force=False):
        self.years = years or [2024, 2025]
        self.modes = modes or ["R", "S"]
        self.force = force
        os.makedirs(RAW_DIR, exist_ok=True)

    def _already_collected(self, year, gp, mode):
        if self.force:
            return False
        filename = os.path.join(RAW_DIR, f"{year}_{gp:02d}_{mode}.parquet")
        return os.path.exists(filename)

    def get_data(self, year, gp, mode) -> pd.DataFrame:
        for attempt in range(MAX_RETRIES):
            try:
                session = fastf1.get_session(year, gp, mode)
                break
            except RateLimitExceededError:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"\n  Rate limited! Waiting {wait}s before retry ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            except ValueError:
                return pd.DataFrame()
        else:
            print(f"\n  Max retries exceeded for {year} round {gp} {mode}. Skipping.")
            return pd.DataFrame()

        for attempt in range(MAX_RETRIES):
            try:
                session.load(telemetry=False, weather=True, messages=False)
                break
            except RateLimitExceededError:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"\n  Rate limited on load! Waiting {wait}s before retry ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        df = session.results
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["Year"] = session.date.year
        df["Date"] = session.date
        df["Mode"] = session.name
        df["RoundNumber"] = session.event["RoundNumber"]
        df["OfficialEventName"] = session.event["OfficialEventName"]
        df["EventName"] = session.event["EventName"]
        df["Country"] = session.event["Country"]
        df["Location"] = session.event["Location"]

        # Aggregate weather data to session-level summary
        try:
            weather = session.weather_data
        except Exception:
            weather = None
        if weather is not None and not weather.empty:
            df["AirTemp"] = weather["AirTemp"].mean()
            df["TrackTemp"] = weather["TrackTemp"].mean()
            df["Humidity"] = weather["Humidity"].mean()
            df["Pressure"] = weather["Pressure"].mean()
            df["WindSpeed"] = weather["WindSpeed"].mean()
            df["WindDirection"] = weather["WindDirection"].mean()
            df["Rainfall"] = int(weather["Rainfall"].any())
        else:
            for col in ("AirTemp", "TrackTemp", "Humidity", "Pressure",
                        "WindSpeed", "WindDirection", "Rainfall"):
                df[col] = None

        return df

    def save_data(self, df: pd.DataFrame, year: int, gp: int, mode: str):
        filename = os.path.join(RAW_DIR, f"{year}_{gp:02d}_{mode}.parquet")
        df.to_parquet(filename, index=False)

    def process(self, year, gp, mode):
        if self._already_collected(year, gp, mode):
            return True
        df = self.get_data(year, gp, mode)
        if df.empty:
            return False
        self.save_data(df, year, gp, mode)
        time.sleep(2)
        return True

    def process_year_modes(self, year):
        for i in tqdm(range(1, 50), desc=f"{year}"):
            for mode in self.modes:
                if not self.process(year, i, mode) and mode == "R":
                    return

    def process_years(self):
        for year in self.years:
            print(f"Collecting data for {year}...")
            self.process_year_modes(year)
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect F1 session results")
    parser.add_argument("--years", "-y", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--modes", "-m", nargs="+", default=["R", "S"])
    parser.add_argument("--force", "-f", action="store_true", help="Re-collect even if files exist")
    args = parser.parse_args()

    collector = CollectResults(years=args.years, modes=args.modes, force=args.force)
    collector.process_years()
    print("Collection complete.")
