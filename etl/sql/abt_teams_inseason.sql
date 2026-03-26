-- In-season ABT for constructor champion prediction.
-- One row per (team, race_event) per season.
-- Features are team-aggregated driver stats from the silver feature store at each race date.
-- Label: fl_constructor_champion = 1 for ALL rows of the eventual WCC team that season.

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
    tf.dt_ref,
    tf.teamid,
    tf.team_name,
    tf.season_race_number,
    tf.season_total_races,
    tf.season_fraction,
    tf.sum_sessions_life,
    tf.avg_sessions_life,
    tf.sum_wins_life,
    tf.avg_wins_life,
    tf.sum_podiums_life,
    tf.avg_podiums_life,
    tf.sum_points_life,
    tf.avg_points_life,
    tf.avg_position_life,
    tf.avg_grid_life,
    tf.sum_sessions_last10,
    tf.sum_wins_last10,
    tf.sum_podiums_last10,
    tf.sum_points_last10,
    tf.avg_position_last10,
    tf.avg_grid_last10,
    tf.sum_sessions_last20,
    tf.sum_wins_last20,
    tf.sum_podiums_last20,
    tf.sum_points_last20,
    tf.avg_position_last20,
    tf.avg_grid_last20,
    tf.sum_wins_race_last20,
    tf.sum_podiums_race_last20,
    tf.sum_points_race_last20,
    tf.avg_position_race_last20,
    tf.num_drivers,
    COALESCE(cc.is_champion, 0) AS fl_constructor_champion
FROM team_features tf
LEFT JOIN (
    SELECT teamid, year, 1 AS is_champion
    FROM read_csv('{constructors_csv}', header=true, auto_detect=true)
) cc
    ON tf.teamid = cc.teamid
    AND tf.year = cc.year
WHERE tf.year >= 2000
ORDER BY tf.dt_ref DESC, tf.teamid
