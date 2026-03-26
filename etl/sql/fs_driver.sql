-- Feature store: driver-level aggregated stats over a rolling window.
-- Parameters: {window_size} (number of rounds), {suffix} (column name suffix)
-- Reads from: data/bronze/results.parquet

WITH dim_dates AS (
    SELECT DISTINCT event_date AS dt_ref, year
    FROM read_parquet('{bronze_path}')
    WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
),

past_sessions AS (
    SELECT
        d.dt_ref,
        d.year AS ref_year,
        r.*
    FROM dim_dates d
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.event_date <= d.dt_ref
),

eligible_drivers AS (
    SELECT DISTINCT dt_ref, driverid
    FROM past_sessions
    WHERE ref_year - year <= 2
),

distinct_rounds AS (
    SELECT DISTINCT dt_ref, year, round_number, event_date
    FROM past_sessions
),

ranked_rounds AS (
    SELECT
        dt_ref,
        year,
        round_number,
        ROW_NUMBER() OVER (PARTITION BY dt_ref ORDER BY event_date DESC) AS rn
    FROM distinct_rounds
),

last_rounds AS (
    SELECT dt_ref, year, round_number
    FROM ranked_rounds
    WHERE rn <= {window_size}
),

tb_results AS (
    SELECT ps.*
    FROM past_sessions ps
    INNER JOIN eligible_drivers ed
        ON ps.dt_ref = ed.dt_ref AND ps.driverid = ed.driverid
    INNER JOIN last_rounds lr
        ON ps.dt_ref = lr.dt_ref AND ps.year = lr.year AND ps.round_number = lr.round_number
),

tb_stats AS (
    SELECT
        dt_ref,
        driverid,

        -- Session counts
        COUNT(DISTINCT year) AS qtd_seasons_{suffix},
        COUNT(*) AS qtd_sessions_{suffix},
        SUM(is_finished) AS qtd_finished_{suffix},

        -- Race vs Sprint breakdown
        SUM(CASE WHEN mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_race_{suffix},
        SUM(CASE WHEN mode IN ('Race') AND is_finished = 1 THEN 1 ELSE 0 END) AS qtd_finished_race_{suffix},
        SUM(CASE WHEN mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_sprint_{suffix},
        SUM(CASE WHEN mode IN ('Sprint Race', 'Sprint') AND is_finished = 1 THEN 1 ELSE 0 END) AS qtd_finished_sprint_{suffix},

        -- Wins
        SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS qtd_wins_{suffix},
        SUM(CASE WHEN position = 1 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_wins_race_{suffix},
        SUM(CASE WHEN position = 1 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_wins_sprint_{suffix},

        -- Podiums (top 3)
        SUM(CASE WHEN position <= 3 THEN 1 ELSE 0 END) AS qtd_podiums_{suffix},
        SUM(CASE WHEN position <= 3 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_podiums_race_{suffix},
        SUM(CASE WHEN position <= 3 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_podiums_sprint_{suffix},

        -- Top 5
        SUM(CASE WHEN position <= 5 THEN 1 ELSE 0 END) AS qtd_top5_{suffix},
        SUM(CASE WHEN position <= 5 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_top5_race_{suffix},
        SUM(CASE WHEN position <= 5 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_top5_sprint_{suffix},

        -- Grid position stats
        SUM(CASE WHEN grid_position <= 5 THEN 1 ELSE 0 END) AS qtd_grid_top5_{suffix},
        SUM(CASE WHEN grid_position <= 5 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_grid_top5_race_{suffix},
        SUM(CASE WHEN grid_position <= 5 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_grid_top5_sprint_{suffix},

        -- Poles
        SUM(CASE WHEN grid_position = 1 THEN 1 ELSE 0 END) AS qtd_poles_{suffix},
        SUM(CASE WHEN grid_position = 1 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_poles_race_{suffix},
        SUM(CASE WHEN grid_position = 1 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_poles_sprint_{suffix},

        -- Pole to win
        SUM(CASE WHEN grid_position = 1 AND position = 1 THEN 1 ELSE 0 END) AS qtd_pole_win_{suffix},
        SUM(CASE WHEN grid_position = 1 AND position = 1 AND mode IN ('Race') THEN 1 ELSE 0 END) AS qtd_pole_win_race_{suffix},
        SUM(CASE WHEN grid_position = 1 AND position = 1 AND mode IN ('Sprint Race', 'Sprint') THEN 1 ELSE 0 END) AS qtd_pole_win_sprint_{suffix},

        -- Points
        SUM(points) AS total_points_{suffix},
        SUM(CASE WHEN mode IN ('Race') THEN points ELSE 0 END) AS total_points_race_{suffix},
        SUM(CASE WHEN mode IN ('Sprint Race', 'Sprint') THEN points ELSE 0 END) AS total_points_sprint_{suffix},
        SUM(CASE WHEN points > 0 THEN 1 ELSE 0 END) AS qtd_sessions_with_points_{suffix},
        SUM(CASE WHEN mode IN ('Race') AND points > 0 THEN 1 ELSE 0 END) AS qtd_sessions_with_points_race_{suffix},
        SUM(CASE WHEN mode IN ('Sprint Race', 'Sprint') AND points > 0 THEN 1 ELSE 0 END) AS qtd_sessions_with_points_sprint_{suffix},

        -- Averages
        AVG(grid_position) AS avg_grid_{suffix},
        AVG(CASE WHEN mode IN ('Race') THEN grid_position END) AS avg_grid_race_{suffix},
        AVG(CASE WHEN mode IN ('Sprint Race', 'Sprint') THEN grid_position END) AS avg_grid_sprint_{suffix},
        AVG(position) AS avg_position_{suffix},
        AVG(CASE WHEN mode IN ('Race') THEN position END) AS avg_position_race_{suffix},
        AVG(CASE WHEN mode IN ('Sprint Race', 'Sprint') THEN position END) AS avg_position_sprint_{suffix},

        -- Overtakes
        SUM(CASE WHEN position < grid_position THEN 1 ELSE 0 END) AS qtd_sessions_with_overtake_{suffix},
        SUM(CASE WHEN mode IN ('Race') AND position < grid_position THEN 1 ELSE 0 END) AS qtd_sessions_with_overtake_race_{suffix},
        SUM(CASE WHEN mode IN ('Sprint Race', 'Sprint') AND position < grid_position THEN 1 ELSE 0 END) AS qtd_sessions_with_overtake_sprint_{suffix},
        AVG(grid_position - position) AS avg_overtakes_{suffix},
        AVG(CASE WHEN mode IN ('Race') THEN grid_position - position END) AS avg_overtakes_race_{suffix},
        AVG(CASE WHEN mode IN ('Sprint Race', 'Sprint') THEN grid_position - position END) AS avg_overtakes_sprint_{suffix}

    FROM tb_results
    GROUP BY dt_ref, driverid
)

SELECT * FROM tb_stats
ORDER BY dt_ref DESC, driverid
