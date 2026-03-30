-- In-season ABT for driver departure prediction.
-- One row per (driver, race_event) per season.
-- Features are point-in-time from the silver feature store at each race date.
-- Includes departure-specific features: performance trends, teammate comparison,
-- team tenure, and DNF rate — computed here to avoid affecting other ABTs.
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
),

-- Teammate comparison: point-in-time stats vs teammate at each race date.
-- Uses bronze results before each dt_ref, same point-in-time logic as the feature store.
driver_team_current AS (
    -- Find the driver's current team at each race date
    -- (most recent race entry before dt_ref)
    SELECT DISTINCT
        rc.dt_ref,
        r.driverid,
        r.teamid
    FROM race_calendar rc
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.event_date = rc.dt_ref
        AND r.mode IN ('Race', 'Sprint Race', 'Sprint')
),

-- Season-level stats per driver-team up to each dt_ref (point-in-time)
driver_season_stats AS (
    SELECT
        rc.dt_ref,
        r.driverid,
        r.teamid,
        SUM(r.points) AS season_points,
        AVG(r.position) AS season_avg_position,
        AVG(r.grid_position) AS season_avg_grid,
        COUNT(*) AS season_races,
        SUM(CASE WHEN r.is_finished = 0 THEN 1 ELSE 0 END) AS season_dnfs
    FROM race_calendar rc
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.year = rc.year
        AND r.event_date < rc.dt_ref
        AND r.mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY rc.dt_ref, r.driverid, r.teamid
),

-- Team totals per dt_ref (for computing shares)
team_season_stats AS (
    SELECT
        dt_ref,
        teamid,
        SUM(season_points) AS team_total_points,
        AVG(season_avg_position) AS team_avg_position,
        COUNT(*) AS team_n_drivers
    FROM driver_season_stats
    GROUP BY dt_ref, teamid
),

-- Teammate comparison features (only for current team at each dt_ref)
teammate_features AS (
    SELECT
        ds.dt_ref,
        ds.driverid,
        -- Points share within team (0-1, low = underperforming teammate)
        CASE WHEN ts.team_total_points > 0
             THEN ds.season_points / ts.team_total_points
             ELSE 0.5
        END AS team_points_share,
        -- Position gap vs team average (positive = worse than teammate)
        ds.season_avg_position - ts.team_avg_position AS teammate_position_gap,
        -- Grid gap vs team average
        ds.season_avg_grid - ts.team_avg_position AS teammate_grid_gap,
        -- DNF rate this season
        CASE WHEN ds.season_races > 0
             THEN ds.season_dnfs * 1.0 / ds.season_races
             ELSE 0
        END AS season_dnf_rate,
        -- Season points (useful on its own)
        ds.season_points AS season_points_current
    FROM driver_season_stats ds
    INNER JOIN driver_team_current dtc
        ON ds.dt_ref = dtc.dt_ref AND ds.driverid = dtc.driverid
        AND ds.teamid = dtc.teamid
    INNER JOIN team_season_stats ts
        ON ds.dt_ref = ts.dt_ref AND dtc.teamid = ts.teamid
),

-- Driver age from external DOB data
driver_dob AS (
    SELECT driverid, CAST(date_of_birth AS DATE) AS date_of_birth
    FROM read_csv('{drivers_dob_csv}', auto_detect=true)
),

-- Revolving door: count of distinct teams in career up to each dt_ref
career_teams AS (
    SELECT
        rc.dt_ref,
        r.driverid,
        COUNT(DISTINCT r.teamid) AS career_distinct_teams
    FROM race_calendar rc
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.event_date < rc.dt_ref
        AND r.mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY rc.dt_ref, r.driverid
),

-- Late-career plateau: seasons since last win and last podium (point-in-time)
last_achievement AS (
    SELECT
        rc.dt_ref,
        r.driverid,
        MAX(CASE WHEN r.position = 1 THEN r.year END) AS last_win_year,
        MAX(CASE WHEN r.position <= 3 THEN r.year END) AS last_podium_year
    FROM race_calendar rc
    INNER JOIN read_parquet('{bronze_path}') r
        ON r.event_date < rc.dt_ref
        AND r.mode IN ('Race', 'Sprint Race', 'Sprint')
    GROUP BY rc.dt_ref, r.driverid
),

-- Team tenure: how many consecutive seasons with the same constructor
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
    rc.season_race_number,
    rc.season_total_races,
    ROUND(rc.season_race_number * 1.0 / rc.season_total_races, 3) AS season_fraction,
    -- Performance trend: last10 vs life (positive = declining)
    f.avg_position_last10 - f.avg_position_life AS trend_avg_position,
    f.avg_grid_last10 - f.avg_grid_life AS trend_avg_grid,
    f.avg_overtakes_last10 - f.avg_overtakes_life AS trend_avg_overtakes,
    -- Win/podium rate trend
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
    -- DNF rate trend: last10 vs life
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
    -- Driver age at race date (years)
    ROUND(DATE_DIFF('day', dob.date_of_birth, f.dt_ref) / 365.25, 1) AS driver_age,
    -- Revolving door: distinct teams in career
    COALESCE(ct.career_distinct_teams, 1) AS career_distinct_teams,
    -- Late-career plateau: seasons since last win / podium (NULL if never achieved → large number)
    COALESCE(rc.year - la.last_win_year, f.qtd_seasons_life + 1) AS seasons_since_last_win,
    COALESCE(rc.year - la.last_podium_year, f.qtd_seasons_life + 1) AS seasons_since_last_podium,
    dl.fl_departed
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
INNER JOIN race_calendar rc ON f.dt_ref = rc.dt_ref
INNER JOIN departure_labels dl
    ON f.driverid = dl.driverid AND rc.year = dl.year
LEFT JOIN teammate_features tm
    ON f.dt_ref = tm.dt_ref AND f.driverid = tm.driverid
LEFT JOIN driver_team_current dtc
    ON f.dt_ref = dtc.dt_ref AND f.driverid = dtc.driverid
LEFT JOIN team_tenure_by_year tt
    ON dtc.driverid = tt.driverid AND dtc.teamid = tt.teamid AND rc.year = tt.year
LEFT JOIN driver_dob dob
    ON f.driverid = dob.driverid
LEFT JOIN career_teams ct
    ON f.dt_ref = ct.dt_ref AND f.driverid = ct.driverid
LEFT JOIN last_achievement la
    ON f.dt_ref = la.dt_ref AND f.driverid = la.driverid
WHERE rc.year >= 2000
  AND rc.year < EXTRACT(YEAR FROM CURRENT_DATE)
ORDER BY f.dt_ref DESC, f.driverid
