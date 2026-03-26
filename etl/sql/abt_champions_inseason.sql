-- In-season ABT: one row per (driver, race event) per season.
-- Features are point-in-time from the silver feature store at each race date,
-- so no future data is ever included for a given row.
-- Added season context: race number, total races, and season fraction so the
-- model can calibrate its confidence based on how far through the season we are.
-- Label: fl_champion = 1 for ALL rows of the eventual season champion.

WITH race_calendar AS (
    SELECT
        year,
        event_date AS dt_ref,
        ROW_NUMBER() OVER (PARTITION BY year ORDER BY event_date) AS season_race_number,
        COUNT(*)    OVER (PARTITION BY year)                      AS season_total_races
    FROM (
        SELECT DISTINCT year, event_date
        FROM read_parquet('{bronze_path}')
        WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
    )
)

SELECT
    f.*,
    rc.season_race_number,
    rc.season_total_races,
    ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
    COALESCE(c.is_champion, 0) AS fl_champion
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN race_calendar rc
    ON f.dt_ref = rc.dt_ref
LEFT JOIN (
    SELECT driverid, year, 1 AS is_champion
    FROM read_csv('{champions_csv}', header=true, auto_detect=true)
) c
    ON f.driverid = c.driverid
    AND EXTRACT(YEAR FROM f.dt_ref)::INT = c.year
WHERE f.dt_ref >= DATE '2000-01-01'
ORDER BY f.dt_ref DESC, f.driverid
