-- Analytical Base Table for team/constructor champion prediction.
-- Uses end-of-prior-year driver stats aggregated to team level to predict
-- current-year WCC. One row per (team, year): features are the season-end
-- snapshot (last race of year Y), label is whether that team won the WCC
-- in year Y+1. This eliminates within-season leakage.

WITH season_end_drivers AS (
    SELECT
        f.*,
        EXTRACT(YEAR FROM f.dt_ref)::INT AS stats_year
    FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY f.driverid, EXTRACT(YEAR FROM f.dt_ref)::INT
        ORDER BY f.dt_ref DESC
    ) = 1
),

team_features AS (
    SELECT
        s.dt_ref,
        s.stats_year,
        b.teamid,
        b.team_name,

        -- Aggregate driver features to team level (sum and avg)
        SUM(s.qtd_sessions_life) AS sum_sessions_life,
        AVG(s.qtd_sessions_life) AS avg_sessions_life,
        SUM(s.qtd_wins_life) AS sum_wins_life,
        AVG(s.qtd_wins_life) AS avg_wins_life,
        SUM(s.qtd_podiums_life) AS sum_podiums_life,
        AVG(s.qtd_podiums_life) AS avg_podiums_life,
        SUM(s.total_points_life) AS sum_points_life,
        AVG(s.total_points_life) AS avg_points_life,
        AVG(s.avg_position_life) AS avg_position_life,
        AVG(s.avg_grid_life) AS avg_grid_life,

        SUM(s.qtd_sessions_last10) AS sum_sessions_last10,
        SUM(s.qtd_wins_last10) AS sum_wins_last10,
        SUM(s.qtd_podiums_last10) AS sum_podiums_last10,
        SUM(s.total_points_last10) AS sum_points_last10,
        AVG(s.avg_position_last10) AS avg_position_last10,
        AVG(s.avg_grid_last10) AS avg_grid_last10,

        SUM(s.qtd_sessions_last20) AS sum_sessions_last20,
        SUM(s.qtd_wins_last20) AS sum_wins_last20,
        SUM(s.qtd_podiums_last20) AS sum_podiums_last20,
        SUM(s.total_points_last20) AS sum_points_last20,
        AVG(s.avg_position_last20) AS avg_position_last20,
        AVG(s.avg_grid_last20) AS avg_grid_last20,

        SUM(s.qtd_wins_race_last20) AS sum_wins_race_last20,
        SUM(s.qtd_podiums_race_last20) AS sum_podiums_race_last20,
        SUM(s.total_points_race_last20) AS sum_points_race_last20,
        AVG(s.avg_position_race_last20) AS avg_position_race_last20,

        COUNT(*) AS num_drivers

    FROM season_end_drivers s
    INNER JOIN (
        SELECT DISTINCT driverid, teamid, team_name, event_date
        FROM read_parquet('{bronze_path}')
    ) b
        ON s.driverid = b.driverid AND s.dt_ref = b.event_date
    GROUP BY s.dt_ref, s.stats_year, b.teamid, b.team_name
)

SELECT
    tf.dt_ref,
    tf.teamid,
    tf.team_name,
    tf.* EXCLUDE (dt_ref, stats_year, teamid, team_name),
    COALESCE(cc.is_champion, 0) AS fl_constructor_champion
FROM team_features tf
LEFT JOIN (
    SELECT teamid, year, 1 AS is_champion
    FROM read_csv('{constructors_csv}', header=true, auto_detect=true)
) cc
    ON tf.teamid = cc.teamid
    AND tf.stats_year + 1 = cc.year
WHERE tf.stats_year >= 1999
ORDER BY tf.dt_ref DESC, tf.teamid
