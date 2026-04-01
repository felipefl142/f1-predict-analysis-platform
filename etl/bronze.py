"""Build bronze layer: clean and consolidate raw Parquet files into a single results.parquet."""

import os

import duckdb

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")


def build_bronze():
    os.makedirs(BRONZE_DIR, exist_ok=True)
    raw_pattern = os.path.join(RAW_DIR, "*.parquet")

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                CAST("DriverNumber" AS INTEGER) AS driver_number,
                "BroadcastName" AS broadcast_name,
                "FullName" AS full_name,
                "Abbreviation" AS abbreviation,
                "DriverId" AS driverid,
                "TeamName" AS team_name,
                "TeamColor" AS team_color,
                "TeamId" AS teamid,
                TRY_CAST("GridPosition" AS INTEGER) AS grid_position,
                TRY_CAST("Position" AS INTEGER) AS position,
                "ClassifiedPosition" AS classified_position,
                COALESCE(TRY_CAST("Points" AS DOUBLE), 0) AS points,
                "Status" AS status,
                "Year" AS year,
                CAST("Date" AS DATE) AS event_date,
                "Mode" AS mode,
                CAST("RoundNumber" AS INTEGER) AS round_number,
                "OfficialEventName" AS official_event_name,
                "EventName" AS event_name,
                "Country" AS country,
                "Location" AS location,
                CASE
                    WHEN "Status" = 'Finished' OR "Status" LIKE '+%' THEN 1
                    ELSE 0
                END AS is_finished,
                CASE
                    WHEN TRY_CAST("GridPosition" AS INTEGER) IS NOT NULL
                         AND TRY_CAST("Position" AS INTEGER) IS NOT NULL
                    THEN TRY_CAST("GridPosition" AS INTEGER) - TRY_CAST("Position" AS INTEGER)
                    ELSE 0
                END AS overtakes,
                TRY_CAST("AirTemp" AS DOUBLE) AS air_temp,
                TRY_CAST("TrackTemp" AS DOUBLE) AS track_temp,
                TRY_CAST("Humidity" AS DOUBLE) AS humidity,
                TRY_CAST("Pressure" AS DOUBLE) AS pressure,
                TRY_CAST("WindSpeed" AS DOUBLE) AS wind_speed,
                TRY_CAST("WindDirection" AS DOUBLE) AS wind_direction,
                COALESCE(TRY_CAST("Rainfall" AS INTEGER), 0) AS rainfall
            FROM read_parquet('{raw_pattern}', union_by_name=true)
            WHERE "DriverId" IS NOT NULL AND CAST("DriverId" AS VARCHAR) != 'nan'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY "DriverId", "Year", "RoundNumber", "Mode"
                ORDER BY "Date" DESC
            ) = 1
        ) TO '{os.path.join(BRONZE_DIR, "results.parquet")}' (FORMAT PARQUET)
    """)
    con.close()

    row_count = duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{os.path.join(BRONZE_DIR, 'results.parquet')}')"
    ).fetchone()[0]
    print(f"Bronze layer built: {row_count} rows in results.parquet")


if __name__ == "__main__":
    build_bronze()
