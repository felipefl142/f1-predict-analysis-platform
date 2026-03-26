"""Build silver layer: feature store with 4 temporal windows joined into a single feature set."""

import os

import duckdb

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze", "results.parquet")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")

WINDOWS = {
    "life": 9999,
    "last10": 10,
    "last20": 20,
    "last40": 40,
}


def build_feature_stores():
    os.makedirs(SILVER_DIR, exist_ok=True)
    sql_template = open(os.path.join(SQL_DIR, "fs_driver.sql")).read()

    con = duckdb.connect()

    for name, window_size in WINDOWS.items():
        print(f"  Building fs_driver_{name} (window={window_size})...")
        sql = sql_template.replace("{window_size}", str(window_size))
        sql = sql.replace("{suffix}", name)
        sql = sql.replace("{bronze_path}", BRONZE_PATH)

        output_path = os.path.join(SILVER_DIR, f"fs_driver_{name}.parquet")
        con.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET)")

        row_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
        ).fetchone()[0]
        print(f"    -> {row_count} rows")

    con.close()


def build_fs_all():
    print("  Joining all windows into fs_driver_all...")
    sql_template = open(os.path.join(SQL_DIR, "fs_all.sql")).read()
    sql = sql_template.replace("{silver_dir}", SILVER_DIR)

    con = duckdb.connect()
    output_path = os.path.join(SILVER_DIR, "fs_driver_all.parquet")
    con.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET)")

    result = con.execute(
        f"SELECT COUNT(*) AS rows, COUNT(*) - 2 AS feature_cols FROM read_parquet('{output_path}')"
    ).fetchone()
    col_count = len(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{output_path}')").fetchall()
    )
    print(f"    -> {result[0]} rows, {col_count} columns")
    con.close()


def build_silver():
    print("Building silver layer...")
    build_feature_stores()
    build_fs_all()
    print("Silver layer complete.")


if __name__ == "__main__":
    build_silver()
