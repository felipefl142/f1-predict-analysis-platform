-- In-season ABT for constructor champion prediction.
-- One row per (team, race_event) per season.
-- Features are team-aggregated driver stats from the silver feature store at each race date.
-- Label: fl_constructor_champion = 1 only from the race where the constructor
-- championship is mathematically clinched (leader's gap > max remaining points) onwards.
-- Also includes team standing features (position, gap to leader).

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

-- Points per team per event
team_event_points AS (
    SELECT year, event_date, teamid, SUM(points) AS points
    FROM read_parquet('{bronze_path}')
    WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY year, event_date, teamid
),

-- Max points any team scored at each event (for remaining-points calculation)
event_max_team_points AS (
    SELECT year, event_date, MAX(points) AS max_team_pts
    FROM team_event_points
    GROUP BY year, event_date
),

-- Max remaining points available after each event_date in the season
remaining_points AS (
    SELECT
        e1.year,
        e1.event_date AS dt_ref,
        COALESCE(SUM(e2.max_team_pts) FILTER (WHERE e2.event_date > e1.event_date), 0) AS max_remaining_pts
    FROM event_max_team_points e1
    JOIN event_max_team_points e2 ON e1.year = e2.year
    GROUP BY e1.year, e1.event_date
),

-- Cumulative points per team at each event_date
cumulative_team_points AS (
    SELECT
        year,
        event_date AS dt_ref,
        teamid,
        SUM(points) OVER (
            PARTITION BY year, teamid ORDER BY event_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cum_points
    FROM team_event_points
),

-- Team standings at each event: position, gap to leader
team_standings AS (
    SELECT
        year,
        dt_ref,
        teamid,
        cum_points,
        RANK() OVER (PARTITION BY year, dt_ref ORDER BY cum_points DESC) AS team_standing_position,
        MAX(cum_points) OVER (PARTITION BY year, dt_ref) - cum_points AS team_points_gap_to_leader,
        ROUND(cum_points * 1.0 / NULLIF(MAX(cum_points) OVER (PARTITION BY year, dt_ref), 0), 4) AS team_points_pct_of_leader
    FROM cumulative_team_points
),

-- At each event_date: gap between leading team and 2nd place team
team_leader_gap AS (
    SELECT
        year,
        dt_ref,
        MAX(cum_points) - (
            SELECT cp2.cum_points
            FROM cumulative_team_points cp2
            WHERE cp2.year = cp.year AND cp2.dt_ref = cp.dt_ref
            ORDER BY cp2.cum_points DESC
            LIMIT 1 OFFSET 1
        ) AS leader_gap,
        ARG_MAX(teamid, cum_points) AS leader_teamid
    FROM cumulative_team_points cp
    GROUP BY year, dt_ref
),

-- First event where constructor championship is mathematically clinched
clinch_event AS (
    SELECT
        s.year,
        s.leader_teamid AS champion_teamid,
        MIN(s.dt_ref) AS clinch_date
    FROM team_leader_gap s
    JOIN remaining_points rp ON s.year = rp.year AND s.dt_ref = rp.dt_ref
    WHERE s.leader_gap > rp.max_remaining_pts
    GROUP BY s.year, s.leader_teamid
),

driver_at_race AS (
    SELECT DISTINCT driverid, event_date AS dt_ref, teamid, team_name
    FROM read_parquet('{bronze_path}')
    WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
),

team_features AS (
    SELECT
        rc.dt_ref,
        rc.year,
        rc.season_race_number,
        rc.season_total_races,
        ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
        dar.teamid,
        dar.team_name,

        SUM(f.qtd_sessions_life)     AS sum_sessions_life,
        AVG(f.qtd_sessions_life)     AS avg_sessions_life,
        SUM(f.qtd_wins_life)         AS sum_wins_life,
        AVG(f.qtd_wins_life)         AS avg_wins_life,
        SUM(f.qtd_podiums_life)      AS sum_podiums_life,
        AVG(f.qtd_podiums_life)      AS avg_podiums_life,
        SUM(f.total_points_life)     AS sum_points_life,
        AVG(f.total_points_life)     AS avg_points_life,
        AVG(f.avg_position_life)     AS avg_position_life,
        AVG(f.avg_grid_life)         AS avg_grid_life,

        SUM(f.qtd_sessions_last10)   AS sum_sessions_last10,
        SUM(f.qtd_wins_last10)       AS sum_wins_last10,
        SUM(f.qtd_podiums_last10)    AS sum_podiums_last10,
        SUM(f.total_points_last10)   AS sum_points_last10,
        AVG(f.avg_position_last10)   AS avg_position_last10,
        AVG(f.avg_grid_last10)       AS avg_grid_last10,

        SUM(f.qtd_sessions_last20)   AS sum_sessions_last20,
        SUM(f.qtd_wins_last20)       AS sum_wins_last20,
        SUM(f.qtd_podiums_last20)    AS sum_podiums_last20,
        SUM(f.total_points_last20)   AS sum_points_last20,
        AVG(f.avg_position_last20)   AS avg_position_last20,
        AVG(f.avg_grid_last20)       AS avg_grid_last20,

        SUM(f.qtd_wins_race_last20)     AS sum_wins_race_last20,
        SUM(f.qtd_podiums_race_last20)  AS sum_podiums_race_last20,
        SUM(f.total_points_race_last20) AS sum_points_race_last20,
        AVG(f.avg_position_race_last20) AS avg_position_race_last20,

        SUM(f.qtd_quali_life)              AS sum_quali_life,
        AVG(f.avg_quali_position_life)     AS avg_quali_position_life,
        SUM(f.qtd_quali_pole_life)         AS sum_quali_pole_life,
        SUM(f.qtd_quali_top3_life)         AS sum_quali_top3_life,
        SUM(f.qtd_quali_top10_life)        AS sum_quali_top10_life,

        SUM(f.qtd_quali_last10)            AS sum_quali_last10,
        AVG(f.avg_quali_position_last10)   AS avg_quali_position_last10,
        SUM(f.qtd_quali_pole_last10)       AS sum_quali_pole_last10,

        SUM(f.qtd_quali_last20)            AS sum_quali_last20,
        AVG(f.avg_quali_position_last20)   AS avg_quali_position_last20,
        SUM(f.qtd_quali_pole_last20)       AS sum_quali_pole_last20,

        COUNT(DISTINCT dar.driverid) AS num_drivers

    FROM race_calendar rc
    INNER JOIN driver_at_race dar ON rc.dt_ref = dar.dt_ref
    INNER JOIN read_parquet('{silver_dir}/fs_driver_all.parquet') f
        ON f.driverid = dar.driverid AND f.dt_ref = dar.dt_ref
    GROUP BY
        rc.dt_ref, rc.year, rc.season_race_number, rc.season_total_races,
        ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3),
        dar.teamid, dar.team_name
)

SELECT
    tf.*,
    ts.team_standing_position,
    ts.team_points_gap_to_leader,
    ts.team_points_pct_of_leader,
    CASE
        WHEN tf.teamid = cl.champion_teamid AND tf.dt_ref >= cl.clinch_date THEN 1
        ELSE 0
    END AS fl_constructor_champion
FROM team_features tf
LEFT JOIN team_standings ts
    ON tf.teamid = ts.teamid AND tf.dt_ref = ts.dt_ref
LEFT JOIN clinch_event cl
    ON tf.year = cl.year
WHERE tf.year >= 2000
ORDER BY tf.dt_ref DESC, tf.teamid
