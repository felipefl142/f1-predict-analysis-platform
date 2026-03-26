-- Analytical Base Table for driver departure prediction.
-- Target: fl_departed = 1 if driver raced in year Y but not in year Y+1.

WITH driver_years AS (
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
),

-- Get last race date per driver per year as the reference date
last_race AS (
    SELECT driverid, year, MAX(event_date) AS dt_ref
    FROM read_parquet('{bronze_path}')
    GROUP BY driverid, year
)

SELECT
    f.*,
    dl.fl_departed
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN last_race lr
    ON f.driverid = lr.driverid AND f.dt_ref = lr.dt_ref
INNER JOIN departure_labels dl
    ON f.driverid = dl.driverid AND lr.year = dl.year
WHERE f.dt_ref >= DATE '2000-01-01'
  AND lr.year < EXTRACT(YEAR FROM CURRENT_DATE)  -- exclude current year (no label yet)
ORDER BY f.dt_ref DESC, f.driverid
