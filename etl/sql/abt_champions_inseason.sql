-- In-season ABT: one row per (driver, race event) per season.
-- Features are point-in-time from the silver feature store at each race date,
-- so no future data is ever included for a given row.
-- Label: fl_champion = 1 at EVERY race for the eventual season champion.
-- Uses champions CSV as source of truth so incomplete seasons (no CSV entry)
-- never produce false champions.

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

-- Driver standings at each event: position, gap to leader, momentum
driver_standings AS (
    SELECT
        year,
        dt_ref,
        driverid,
        cum_points,
        RANK() OVER (PARTITION BY year, dt_ref ORDER BY cum_points DESC) AS standing_position,
        MAX(cum_points) OVER (PARTITION BY year, dt_ref) - cum_points AS points_gap_to_leader,
        ROUND(cum_points * 1.0 / NULLIF(MAX(cum_points) OVER (PARTITION BY year, dt_ref), 0), 4) AS points_pct_of_leader
    FROM cumulative_points
),

-- Momentum: change in standings over the last 3 race events
driver_momentum AS (
    SELECT
        year,
        dt_ref,
        driverid,
        cum_points,
        standing_position,
        points_gap_to_leader,
        points_pct_of_leader,
        -- Positive = improving (moving up in standings)
        LAG(standing_position, 3) OVER w - standing_position AS standings_momentum_3r,
        LAG(standing_position, 1) OVER w - standing_position AS standings_momentum_1r,
        -- Positive = closing the gap (or extending lead)
        LAG(points_gap_to_leader, 3) OVER w - points_gap_to_leader AS gap_momentum_3r,
        -- Points scored in last 3 events vs prior 3 events
        cum_points - LAG(cum_points, 3) OVER w AS points_last3,
        LAG(cum_points, 3) OVER w - LAG(cum_points, 6) OVER w AS points_prev3,
    FROM driver_standings
    WINDOW w AS (PARTITION BY year, driverid ORDER BY dt_ref)
),

-- Champions lookup (eventual champion per year)
champions AS (
    SELECT year, driverid AS champion_driverid
    FROM read_csv_auto('{champions_csv}')
)

SELECT
    f.*,
    rc.season_race_number,
    rc.season_total_races,
    ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
    COALESCE(dm.standing_position, 99) AS standing_position,
    COALESCE(dm.points_gap_to_leader, 0) AS points_gap_to_leader,
    COALESCE(dm.points_pct_of_leader, 0) AS points_pct_of_leader,
    -- Momentum features
    COALESCE(dm.standings_momentum_3r, 0) AS standings_momentum_3r,
    COALESCE(dm.standings_momentum_1r, 0) AS standings_momentum_1r,
    COALESCE(dm.gap_momentum_3r, 0) AS gap_momentum_3r,
    COALESCE(dm.points_last3, 0) AS points_last3,
    COALESCE(dm.points_prev3, 0) AS points_prev3,
    COALESCE(dm.points_last3 - dm.points_prev3, 0) AS points_accel,
    -- Interaction features
    COALESCE(dm.points_pct_of_leader * f.qtd_wins_last10, 0) AS pct_leader_x_wins,
    COALESCE(dm.points_pct_of_leader * f.qtd_podiums_last10, 0) AS pct_leader_x_podiums,
    COALESCE(dm.points_pct_of_leader * f.total_points_last10, 0) AS pct_leader_x_points,
    -- Target: 1 at every race for the eventual season champion
    CASE
        WHEN f.driverid = ch.champion_driverid THEN 1
        ELSE 0
    END AS fl_champion
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN race_calendar rc
    ON f.dt_ref = rc.dt_ref
LEFT JOIN driver_momentum dm
    ON dm.driverid = f.driverid AND dm.dt_ref = f.dt_ref
LEFT JOIN standings st
    ON EXTRACT(YEAR FROM f.dt_ref)::INT = st.year AND f.dt_ref = st.dt_ref
LEFT JOIN champions ch
    ON EXTRACT(YEAR FROM f.dt_ref)::INT = ch.year
WHERE f.dt_ref >= DATE '2000-01-01'
ORDER BY f.dt_ref DESC, f.driverid
