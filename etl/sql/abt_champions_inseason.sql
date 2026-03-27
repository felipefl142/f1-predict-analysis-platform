-- In-season ABT: one row per (driver, race event) per season.
-- Features are point-in-time from the silver feature store at each race date,
-- so no future data is ever included for a given row.
-- Added season context: race number, total races, and season fraction so the
-- model can calibrate its confidence based on how far through the season we are.
-- Label: fl_champion = 1 only from the race where the championship is
-- mathematically clinched (leader's gap > max remaining points) onwards.

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

-- Max points any driver scored at each event (accounts for race vs sprint scoring)
event_max_points AS (
    SELECT year, event_date, MAX(points) AS max_pts
    FROM read_parquet('{bronze_path}')
    WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY year, event_date
),

-- Max remaining points available after each event_date in the season
remaining_points AS (
    SELECT
        e1.year,
        e1.event_date AS dt_ref,
        COALESCE(SUM(e2.max_pts) FILTER (WHERE e2.event_date > e1.event_date), 0) AS max_remaining_pts
    FROM event_max_points e1
    JOIN event_max_points e2 ON e1.year = e2.year
    GROUP BY e1.year, e1.event_date
),

-- Cumulative points per driver at each event_date
cumulative_points AS (
    SELECT
        year,
        event_date AS dt_ref,
        driverid,
        SUM(points) OVER (
            PARTITION BY year, driverid ORDER BY event_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cum_points
    FROM (
        SELECT year, event_date, driverid, SUM(points) AS points
        FROM read_parquet('{bronze_path}')
        WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
        GROUP BY year, event_date, driverid
    )
),

-- At each event_date: gap between leader and 2nd place
standings AS (
    SELECT
        year,
        dt_ref,
        MAX(cum_points) - (
            SELECT cp2.cum_points
            FROM cumulative_points cp2
            WHERE cp2.year = cp.year AND cp2.dt_ref = cp.dt_ref
            ORDER BY cp2.cum_points DESC
            LIMIT 1 OFFSET 1
        ) AS leader_gap,
        ARG_MAX(driverid, cum_points) AS leader_driverid
    FROM cumulative_points cp
    GROUP BY year, dt_ref
),

-- First event where championship is clinched
clinch_event AS (
    SELECT
        s.year,
        s.leader_driverid AS champion_driverid,
        MIN(s.dt_ref) AS clinch_date
    FROM standings s
    JOIN remaining_points rp ON s.year = rp.year AND s.dt_ref = rp.dt_ref
    WHERE s.leader_gap > rp.max_remaining_pts
    GROUP BY s.year, s.leader_driverid
)

SELECT
    f.*,
    rc.season_race_number,
    rc.season_total_races,
    ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
    CASE
        WHEN f.driverid = cl.champion_driverid AND f.dt_ref >= cl.clinch_date THEN 1
        ELSE 0
    END AS fl_champion
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN race_calendar rc
    ON f.dt_ref = rc.dt_ref
LEFT JOIN clinch_event cl
    ON EXTRACT(YEAR FROM f.dt_ref)::INT = cl.year
WHERE f.dt_ref >= DATE '2000-01-01'
ORDER BY f.dt_ref DESC, f.driverid
