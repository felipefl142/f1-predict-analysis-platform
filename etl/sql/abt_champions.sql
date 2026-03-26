-- Analytical Base Table for champion prediction.
-- Uses end-of-prior-year stats to predict current-year WDC.
-- One row per (driver, year): features are the season-end snapshot (last race
-- of year Y), label is whether that driver won the championship in year Y+1.
-- This eliminates within-season leakage where late-season cumulative stats
-- (wins, points) already encode the championship outcome.

WITH season_end AS (
    SELECT
        f.*,
        EXTRACT(YEAR FROM f.dt_ref)::INT AS stats_year
    FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY f.driverid, EXTRACT(YEAR FROM f.dt_ref)::INT
        ORDER BY f.dt_ref DESC
    ) = 1
)

SELECT
    s.dt_ref,
    s.driverid,
    s.* EXCLUDE (dt_ref, driverid, stats_year),
    COALESCE(c.is_champion, 0) AS fl_champion
FROM season_end s
LEFT JOIN (
    SELECT driverid, year, 1 AS is_champion
    FROM read_csv('{champions_csv}', header=true, auto_detect=true)
) c
    ON s.driverid = c.driverid
    AND s.stats_year + 1 = c.year
WHERE s.stats_year >= 1999
ORDER BY s.dt_ref DESC, s.driverid
