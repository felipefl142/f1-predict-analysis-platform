"""Build gold layer: Analytical Base Tables for ML models."""

import os

import duckdb

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze", "results.parquet")
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")
CHAMPIONS_CSV = os.path.join(BASE_DIR, "data", "champions.csv")
CONSTRUCTORS_CSV = os.path.join(BASE_DIR, "data", "constructors_champions.csv")

ABTS = {
    "abt_champions": {
        "sql_file": "abt_champions.sql",
        "replacements": {
            "{silver_dir}": SILVER_DIR,
            "{champions_csv}": CHAMPIONS_CSV,
        },
    },
    "abt_teams": {
        "sql_file": "abt_teams.sql",
        "replacements": {
            "{silver_dir}": SILVER_DIR,
            "{bronze_path}": BRONZE_PATH,
            "{constructors_csv}": CONSTRUCTORS_CSV,
        },
    },
    "abt_departures": {
        "sql_file": "abt_departures.sql",
        "replacements": {
            "{silver_dir}": SILVER_DIR,
            "{bronze_path}": BRONZE_PATH,
        },
    },
}


def build_gold():
    os.makedirs(GOLD_DIR, exist_ok=True)
    con = duckdb.connect()

    print("Building gold layer...")
    for name, config in ABTS.items():
        print(f"  Building {name}...")
        sql = open(os.path.join(SQL_DIR, config["sql_file"])).read()
        for placeholder, value in config["replacements"].items():
            sql = sql.replace(placeholder, value)

        output_path = os.path.join(GOLD_DIR, f"{name}.parquet")
        con.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET)")

        result = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
        ).fetchone()
        col_count = len(
            con.execute(f"DESCRIBE SELECT * FROM read_parquet('{output_path}')").fetchall()
        )
        print(f"    -> {result[0]} rows, {col_count} columns")

    con.close()
    print("Gold layer complete.")


if __name__ == "__main__":
    build_gold()
