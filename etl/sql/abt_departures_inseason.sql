-- In-season ABT for driver departure prediction.
-- One row per (driver, race_event) per season.
-- Features are point-in-time from the silver feature store at each race date.
-- Label: fl_departed = 1 if driver did not race in the following year.
-- Current year is excluded because departure labels are not yet available.

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
),

driver_years AS (
    SELECT DISTINCT driverid, year
    FROM read_parquet('{bronze_path}')
),

departure_labels AS (
    SELECT
        dy.driverid,
        dy.year,
        CASE
            WHEN next_yr.driverid IS NULL THEN 1
            ELSE 0
        END AS fl_departed
    FROM driver_years dy
    LEFT JOIN driver_years next_yr
        ON dy.driverid = next_yr.driverid
        AND dy.year + 1 = next_yr.year
)

SELECT
    f.*,
    rc.season_race_number,
    rc.season_total_races,
    ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
    dl.fl_departed
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN race_calendar rc ON f.dt_ref = rc.dt_ref
INNER JOIN departure_labels dl
    ON f.driverid = dl.driverid AND rc.year = dl.year
WHERE rc.year >= 2000
  AND rc.year < EXTRACT(YEAR FROM CURRENT_DATE)
ORDER BY f.dt_ref DESC, f.driverid
