-- Join all temporal windows into a single feature set per (dt_ref, driverid).
-- Uses DuckDB EXCLUDE clause to avoid duplicate join key columns.

SELECT
    life.*,
    l10.* EXCLUDE (dt_ref, driverid),
    l20.* EXCLUDE (dt_ref, driverid),
    l40.* EXCLUDE (dt_ref, driverid)
FROM read_parquet('{silver_dir}/fs_driver_life.parquet') life
INNER JOIN read_parquet('{silver_dir}/fs_driver_last10.parquet') l10
    ON life.driverid = l10.driverid AND life.dt_ref = l10.dt_ref
INNER JOIN read_parquet('{silver_dir}/fs_driver_last20.parquet') l20
    ON life.driverid = l20.driverid AND life.dt_ref = l20.dt_ref
INNER JOIN read_parquet('{silver_dir}/fs_driver_last40.parquet') l40
    ON life.driverid = l40.driverid AND life.dt_ref = l40.dt_ref
