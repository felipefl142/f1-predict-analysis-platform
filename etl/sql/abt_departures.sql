-- Analytical Base Table for driver departure prediction.
-- Target: fl_departed = 1 if driver raced in year Y but not in year Y+1.
-- Includes departure-specific features: performance trends, teammate comparison,
-- team tenure, and DNF rate — computed here to avoid affecting other ABTs.

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
),

-- Driver's team in each year (team at last race)
driver_team_year AS (
    SELECT DISTINCT ON (r.driverid, r.year)
        r.driverid,
        r.year,
        r.teamid
    FROM read_parquet('{bronze_path}') r
    INNER JOIN last_race lr ON r.driverid = lr.driverid AND r.event_date = lr.dt_ref
    WHERE r.mode IN ('Race', 'Sprint Race', 'Sprint')
),

-- Season-level stats per driver-team for the full year
driver_season_stats AS (
    SELECT
        r.driverid,
        r.teamid,
        r.year,
        SUM(r.points) AS season_points,
        AVG(r.position) AS season_avg_position,
        AVG(r.grid_position) AS season_avg_grid,
        COUNT(*) AS season_races,
        SUM(CASE WHEN r.is_finished = 0 THEN 1 ELSE 0 END) AS season_dnfs
    FROM read_parquet('{bronze_path}') r
    WHERE r.mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY r.driverid, r.teamid, r.year
),

-- Team totals per year
team_season_stats AS (
    SELECT
        teamid,
        year,
        SUM(season_points) AS team_total_points,
        AVG(season_avg_position) AS team_avg_position
    FROM driver_season_stats
    GROUP BY teamid, year
),

-- Teammate comparison features
teammate_features AS (
    SELECT
        ds.driverid,
        ds.year,
        CASE WHEN ts.team_total_points > 0
             THEN ds.season_points / ts.team_total_points
             ELSE 0.5
        END AS team_points_share,
        ds.season_avg_position - ts.team_avg_position AS teammate_position_gap,
        ds.season_avg_grid - ts.team_avg_position AS teammate_grid_gap,
        CASE WHEN ds.season_races > 0
             THEN ds.season_dnfs * 1.0 / ds.season_races
             ELSE 0
        END AS season_dnf_rate,
        ds.season_points AS season_points_current
    FROM driver_season_stats ds
    INNER JOIN driver_team_year dty
        ON ds.driverid = dty.driverid AND ds.year = dty.year
    INNER JOIN team_season_stats ts
        ON dty.teamid = ts.teamid AND ds.year = ts.year
),

-- Driver age from external DOB data
driver_dob AS (
    SELECT driverid, CAST(date_of_birth AS DATE) AS date_of_birth
    FROM read_csv('{drivers_dob_csv}', auto_detect=true)
),

-- Revolving door: count of distinct teams in career up to end of year
career_teams AS (
    SELECT
        driverid,
        year,
        COUNT(DISTINCT teamid) AS career_distinct_teams
    FROM (
        SELECT driverid, teamid, year
        FROM read_parquet('{bronze_path}')
        WHERE mode IN ('Race', 'Sprint Race', 'Sprint')
        UNION
        SELECT r.driverid, r.teamid, dy.year
        FROM read_parquet('{bronze_path}') r
        INNER JOIN (SELECT DISTINCT driverid, year FROM read_parquet('{bronze_path}')) dy
            ON r.driverid = dy.driverid AND r.year < dy.year
        WHERE r.mode IN ('Race', 'Sprint Race', 'Sprint')
    )
    GROUP BY driverid, year
),

-- Late-career plateau: seasons since last win and last podium
last_achievement AS (
    SELECT
        lr.driverid,
        lr.year,
        MAX(CASE WHEN r.position = 1 THEN r.year END) AS last_win_year,
        MAX(CASE WHEN r.position <= 3 THEN r.year END) AS last_podium_year
    FROM last_race lr
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.driverid = lr.driverid
        AND r.event_date <= lr.dt_ref
        AND r.mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY lr.driverid, lr.year
),

-- Team tenure: consecutive seasons with same constructor
team_stints AS (
    SELECT
        driverid,
        teamid,
        year,
        year - ROW_NUMBER() OVER (PARTITION BY driverid, teamid ORDER BY year) AS stint_group
    FROM (
        SELECT DISTINCT driverid, teamid, year
        FROM read_parquet('{bronze_path}')
    )
),

team_tenure_by_year AS (
    SELECT
        driverid,
        teamid,
        year,
        ROW_NUMBER() OVER (PARTITION BY driverid, teamid, stint_group ORDER BY year) AS team_tenure_years
    FROM team_stints
)

SELECT
    f.*,
    -- Performance trend: last10 vs life (positive = declining)
    f.avg_position_last10 - f.avg_position_life AS trend_avg_position,
    f.avg_grid_last10 - f.avg_grid_life AS trend_avg_grid,
    f.avg_overtakes_last10 - f.avg_overtakes_life AS trend_avg_overtakes,
    CASE WHEN f.qtd_sessions_last10 > 0
         THEN f.qtd_wins_last10 * 1.0 / f.qtd_sessions_last10
         ELSE 0 END
    - CASE WHEN f.qtd_sessions_life > 0
           THEN f.qtd_wins_life * 1.0 / f.qtd_sessions_life
           ELSE 0 END AS trend_win_rate,
    CASE WHEN f.qtd_sessions_last10 > 0
         THEN f.qtd_podiums_last10 * 1.0 / f.qtd_sessions_last10
         ELSE 0 END
    - CASE WHEN f.qtd_sessions_life > 0
           THEN f.qtd_podiums_life * 1.0 / f.qtd_sessions_life
           ELSE 0 END AS trend_podium_rate,
    CASE WHEN f.qtd_sessions_last10 > 0
         THEN (f.qtd_sessions_last10 - f.qtd_finished_last10) * 1.0 / f.qtd_sessions_last10
         ELSE 0 END
    - CASE WHEN f.qtd_sessions_life > 0
           THEN (f.qtd_sessions_life - f.qtd_finished_life) * 1.0 / f.qtd_sessions_life
           ELSE 0 END AS trend_dnf_rate,
    -- Teammate comparison features
    COALESCE(tm.team_points_share, 0.5) AS team_points_share,
    COALESCE(tm.teammate_position_gap, 0) AS teammate_position_gap,
    COALESCE(tm.teammate_grid_gap, 0) AS teammate_grid_gap,
    COALESCE(tm.season_dnf_rate, 0) AS season_dnf_rate,
    COALESCE(tm.season_points_current, 0) AS season_points_current,
    -- Team tenure
    COALESCE(tt.team_tenure_years, 1) AS team_tenure_years,
    -- Driver age at end of season (years)
    ROUND(DATE_DIFF('day', dob.date_of_birth, lr.dt_ref) / 365.25, 1) AS driver_age,
    -- Revolving door: distinct teams in career
    COALESCE(ct.career_distinct_teams, 1) AS career_distinct_teams,
    -- Late-career plateau: seasons since last win / podium
    COALESCE(lr.year - la.last_win_year, f.qtd_seasons_life + 1) AS seasons_since_last_win,
    COALESCE(lr.year - la.last_podium_year, f.qtd_seasons_life + 1) AS seasons_since_last_podium,
    dl.fl_departed
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN last_race lr
    ON f.driverid = lr.driverid AND f.dt_ref = lr.dt_ref
INNER JOIN departure_labels dl
    ON f.driverid = dl.driverid AND lr.year = dl.year
LEFT JOIN teammate_features tm
    ON f.driverid = tm.driverid AND lr.year = tm.year
LEFT JOIN driver_team_year dty
    ON f.driverid = dty.driverid AND lr.year = dty.year
LEFT JOIN team_tenure_by_year tt
    ON dty.driverid = tt.driverid AND dty.teamid = tt.teamid AND lr.year = tt.year
LEFT JOIN driver_dob dob
    ON f.driverid = dob.driverid
LEFT JOIN career_teams ct
    ON f.driverid = ct.driverid AND lr.year = ct.year
LEFT JOIN last_achievement la
    ON f.driverid = la.driverid AND lr.year = la.year
WHERE f.dt_ref >= DATE '2000-01-01'
  AND lr.year < EXTRACT(YEAR FROM CURRENT_DATE)
ORDER BY f.dt_ref DESC, f.driverid
