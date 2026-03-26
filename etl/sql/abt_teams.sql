-- Analytical Base Table for team/constructor champion prediction.
-- Aggregates driver-level features to team level, joins with WCC labels.

WITH team_features AS (
    SELECT
        f.dt_ref,
        b.teamid,
        b.team_name,

        -- Aggregate driver features to team level (sum and avg)
        SUM(f.qtd_sessions_life) AS sum_sessions_life,
        AVG(f.qtd_sessions_life) AS avg_sessions_life,
        SUM(f.qtd_wins_life) AS sum_wins_life,
        AVG(f.qtd_wins_life) AS avg_wins_life,
        SUM(f.qtd_podiums_life) AS sum_podiums_life,
        AVG(f.qtd_podiums_life) AS avg_podiums_life,
        SUM(f.total_points_life) AS sum_points_life,
        AVG(f.total_points_life) AS avg_points_life,
        AVG(f.avg_position_life) AS avg_position_life,
        AVG(f.avg_grid_life) AS avg_grid_life,

        SUM(f.qtd_sessions_last10) AS sum_sessions_last10,
        SUM(f.qtd_wins_last10) AS sum_wins_last10,
        SUM(f.qtd_podiums_last10) AS sum_podiums_last10,
        SUM(f.total_points_last10) AS sum_points_last10,
        AVG(f.avg_position_last10) AS avg_position_last10,
        AVG(f.avg_grid_last10) AS avg_grid_last10,

        SUM(f.qtd_sessions_last20) AS sum_sessions_last20,
        SUM(f.qtd_wins_last20) AS sum_wins_last20,
        SUM(f.qtd_podiums_last20) AS sum_podiums_last20,
        SUM(f.total_points_last20) AS sum_points_last20,
        AVG(f.avg_position_last20) AS avg_position_last20,
        AVG(f.avg_grid_last20) AS avg_grid_last20,

        SUM(f.qtd_wins_race_last20) AS sum_wins_race_last20,
        SUM(f.qtd_podiums_race_last20) AS sum_podiums_race_last20,
        SUM(f.total_points_race_last20) AS sum_points_race_last20,
        AVG(f.avg_position_race_last20) AS avg_position_race_last20,

        COUNT(*) AS num_drivers

    FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
    INNER JOIN (
        SELECT DISTINCT driverid, teamid, team_name, event_date
        FROM read_parquet('{bronze_path}')
    ) b
        ON f.driverid = b.driverid AND f.dt_ref = b.event_date
    GROUP BY f.dt_ref, b.teamid, b.team_name
)

SELECT
    tf.*,
    COALESCE(cc.is_champion, 0) AS fl_constructor_champion
FROM team_features tf
LEFT JOIN (
    SELECT teamid, year, 1 AS is_champion
    FROM read_csv('{constructors_csv}', header=true, auto_detect=true)
) cc
    ON tf.teamid = cc.teamid
    AND EXTRACT(YEAR FROM tf.dt_ref) = cc.year
WHERE tf.dt_ref >= DATE '2000-01-01'
ORDER BY tf.dt_ref DESC, tf.teamid
