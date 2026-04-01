"""Tab 3: Interactive DuckDB SQL Console."""

import streamlit as st

from app.helpers import get_duckdb_connection, AVAILABLE_TABLES

EXAMPLE_QUERIES = {
    "-- Select an example --": "",

    # Bronze — race results
    "Top 20 drivers by race count": (
        "SELECT driverid, full_name, COUNT(*) AS races\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race'\n"
        "GROUP BY driverid, full_name\n"
        "ORDER BY races DESC LIMIT 20"
    ),
    "Points per season (2024)": (
        "SELECT driverid, full_name, team_name, SUM(points) AS total_points\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE year = 2024 AND mode = 'Race'\n"
        "GROUP BY driverid, full_name, team_name\n"
        "ORDER BY total_points DESC"
    ),
    "DNF rate by driver (min 50 races)": (
        "SELECT driverid, full_name,\n"
        "    COUNT(*) AS races,\n"
        "    SUM(1 - is_finished) AS dnfs,\n"
        "    ROUND(SUM(1 - is_finished) * 100.0 / COUNT(*), 1) AS dnf_pct\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race'\n"
        "GROUP BY driverid, full_name\n"
        "HAVING races >= 50\n"
        "ORDER BY dnf_pct DESC LIMIT 20"
    ),
    "Wet race results (rainfall sessions)": (
        "SELECT year, event_name, driverid, full_name, team_name,\n"
        "    grid_position, position, points, air_temp, humidity\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE rainfall = 1 AND mode = 'Race'\n"
        "ORDER BY year DESC, position LIMIT 50"
    ),
    "Average weather by circuit": (
        "SELECT location, country,\n"
        "    COUNT(DISTINCT year) AS seasons,\n"
        "    ROUND(AVG(air_temp), 1) AS avg_air_temp,\n"
        "    ROUND(AVG(track_temp), 1) AS avg_track_temp,\n"
        "    ROUND(AVG(humidity), 1) AS avg_humidity,\n"
        "    ROUND(AVG(wind_speed), 1) AS avg_wind_speed,\n"
        "    SUM(rainfall) AS rain_sessions\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race' AND air_temp IS NOT NULL\n"
        "GROUP BY location, country\n"
        "ORDER BY avg_air_temp DESC"
    ),
    "Team win counts by season": (
        "SELECT year, team_name,\n"
        "    SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race'\n"
        "GROUP BY year, team_name\n"
        "HAVING wins > 0\n"
        "ORDER BY year DESC, wins DESC"
    ),
    "Pole-to-win conversion rate": (
        "SELECT driverid, full_name,\n"
        "    SUM(CASE WHEN grid_position = 1 THEN 1 ELSE 0 END) AS poles,\n"
        "    SUM(CASE WHEN grid_position = 1 AND position = 1 THEN 1 ELSE 0 END) AS pole_wins,\n"
        "    ROUND(SUM(CASE WHEN grid_position = 1 AND position = 1 THEN 1 ELSE 0 END)\n"
        "        * 100.0 / NULLIF(SUM(CASE WHEN grid_position = 1 THEN 1 ELSE 0 END), 0), 1) AS conversion_pct\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race'\n"
        "GROUP BY driverid, full_name\n"
        "HAVING poles >= 3\n"
        "ORDER BY conversion_pct DESC"
    ),
    "Biggest overtakers (avg positions gained)": (
        "SELECT driverid, full_name,\n"
        "    COUNT(*) AS races,\n"
        "    ROUND(AVG(grid_position - position), 2) AS avg_positions_gained,\n"
        "    MAX(grid_position - position) AS best_recovery\n"
        "FROM read_parquet('data/bronze/results.parquet')\n"
        "WHERE mode = 'Race' AND is_finished = 1\n"
        "    AND grid_position IS NOT NULL AND position IS NOT NULL\n"
        "GROUP BY driverid, full_name\n"
        "HAVING races >= 30\n"
        "ORDER BY avg_positions_gained DESC LIMIT 20"
    ),

    # Silver — feature store
    "Verstappen career features (evolution)": (
        "SELECT dt_ref, qtd_wins_life, qtd_podiums_life,\n"
        "    total_points_life, avg_position_life\n"
        "FROM read_parquet('data/silver/fs_driver_all.parquet')\n"
        "WHERE driverid = 'max_verstappen'\n"
        "ORDER BY dt_ref DESC LIMIT 20"
    ),
    "Feature comparison: top 5 drivers (last 10 races)": (
        "WITH latest AS (\n"
        "    SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY dt_ref DESC) AS rn\n"
        "    FROM read_parquet('data/silver/fs_driver_all.parquet')\n"
        ")\n"
        "SELECT driverid, qtd_wins_last10, qtd_podiums_last10,\n"
        "    avg_position_last10, total_points_last10\n"
        "FROM latest\n"
        "WHERE rn = 1\n"
        "ORDER BY total_points_last10 DESC LIMIT 5"
    ),

    # Gold — ABTs
    "Champion ABT sample": (
        "SELECT dt_ref, driverid, fl_champion,\n"
        "    qtd_wins_life, total_points_last20\n"
        "FROM read_parquet('data/gold/abt_champions.parquet')\n"
        "WHERE fl_champion = 1\n"
        "ORDER BY dt_ref DESC LIMIT 20"
    ),
    "Class balance per ABT": (
        "SELECT 'champion' AS abt,\n"
        "    SUM(fl_champion) AS positives,\n"
        "    COUNT(*) AS total,\n"
        "    ROUND(SUM(fl_champion) * 100.0 / COUNT(*), 2) AS pct\n"
        "FROM read_parquet('data/gold/abt_champions.parquet')\n"
        "UNION ALL\n"
        "SELECT 'constructor',\n"
        "    SUM(fl_constructor_champion), COUNT(*),\n"
        "    ROUND(SUM(fl_constructor_champion) * 100.0 / COUNT(*), 2)\n"
        "FROM read_parquet('data/gold/abt_teams.parquet')\n"
        "UNION ALL\n"
        "SELECT 'departure',\n"
        "    SUM(fl_departed), COUNT(*),\n"
        "    ROUND(SUM(fl_departed) * 100.0 / COUNT(*), 2)\n"
        "FROM read_parquet('data/gold/abt_departures.parquet')"
    ),

    # --- Champion model curated features ---
    "Champion features: current standings race-by-race (2024)": (
        "SELECT dt_ref, driverid, standing_position,\n"
        "    ROUND(points_pct_of_leader, 3) AS points_pct_of_leader,\n"
        "    ROUND(gap_momentum_3r, 3) AS gap_momentum_3r,\n"
        "    ROUND(points_accel, 3) AS points_accel,\n"
        "    qtd_wins_last10, qtd_podiums_last10, total_points_last10,\n"
        "    ROUND(avg_position_last10, 2) AS avg_position_last10\n"
        "FROM read_parquet('data/gold/abt_champions_inseason.parquet')\n"
        "WHERE YEAR(dt_ref) = 2024 AND standing_position <= 5\n"
        "ORDER BY dt_ref, standing_position"
    ),
    "Champion features: interaction features by season": (
        "SELECT YEAR(dt_ref) AS season, driverid,\n"
        "    MAX(ROUND(pct_leader_x_wins, 3)) AS max_ldr_x_wins,\n"
        "    MAX(ROUND(pct_leader_x_podiums, 3)) AS max_ldr_x_podiums,\n"
        "    MAX(ROUND(pct_leader_x_points, 3)) AS max_ldr_x_points,\n"
        "    MIN(standing_position) AS best_standing,\n"
        "    fl_champion\n"
        "FROM read_parquet('data/gold/abt_champions_inseason.parquet')\n"
        "WHERE fl_champion IS NOT NULL\n"
        "GROUP BY season, driverid, fl_champion\n"
        "HAVING fl_champion = 1\n"
        "ORDER BY season DESC"
    ),
    "Champion features: momentum leaders per race (2024)": (
        "WITH ranked AS (\n"
        "    SELECT dt_ref, driverid, standing_position,\n"
        "        ROUND(gap_momentum_3r, 4) AS gap_momentum_3r,\n"
        "        ROUND(points_accel, 4) AS points_accel,\n"
        "        ROW_NUMBER() OVER (PARTITION BY dt_ref ORDER BY gap_momentum_3r DESC) AS momentum_rank\n"
        "    FROM read_parquet('data/gold/abt_champions_inseason.parquet')\n"
        "    WHERE YEAR(dt_ref) = 2024\n"
        ")\n"
        "SELECT * FROM ranked WHERE momentum_rank <= 3\n"
        "ORDER BY dt_ref, momentum_rank"
    ),

    # --- Constructor model curated features ---
    "Constructor features: team standings race-by-race (2024)": (
        "SELECT dt_ref, team_name, team_standing_position,\n"
        "    ROUND(team_points_pct_of_leader, 3) AS team_pct_leader,\n"
        "    ROUND(team_points_accel, 3) AS team_points_accel,\n"
        "    sum_wins_last10, sum_podiums_last10, sum_points_last10,\n"
        "    ROUND(avg_position_last10, 2) AS avg_pos_l10\n"
        "FROM read_parquet('data/gold/abt_teams_inseason.parquet')\n"
        "WHERE YEAR(dt_ref) = 2024 AND team_standing_position <= 5\n"
        "ORDER BY dt_ref, team_standing_position"
    ),
    "Constructor features: interaction features by season": (
        "SELECT YEAR(dt_ref) AS season, team_name,\n"
        "    MAX(ROUND(team_pct_leader_x_wins, 3)) AS max_ldr_x_wins,\n"
        "    MAX(ROUND(team_pct_leader_x_podiums, 3)) AS max_ldr_x_podiums,\n"
        "    MAX(ROUND(team_pct_leader_x_points, 3)) AS max_ldr_x_points,\n"
        "    MIN(team_standing_position) AS best_standing,\n"
        "    fl_constructor_champion\n"
        "FROM read_parquet('data/gold/abt_teams_inseason.parquet')\n"
        "WHERE fl_constructor_champion IS NOT NULL\n"
        "GROUP BY season, team_name, fl_constructor_champion\n"
        "HAVING fl_constructor_champion = 1\n"
        "ORDER BY season DESC"
    ),

    # --- Departure model curated features ---
    "Departure features: driver risk profile (latest per driver, 2024)": (
        "WITH latest AS (\n"
        "    SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY dt_ref DESC) AS rn\n"
        "    FROM read_parquet('data/gold/abt_departures_inseason.parquet')\n"
        "    WHERE YEAR(dt_ref) = 2024\n"
        ")\n"
        "SELECT driverid, driver_age, team_tenure_years, career_distinct_teams,\n"
        "    seasons_since_last_win, seasons_since_last_podium,\n"
        "    ROUND(teammate_position_gap, 2) AS teammate_pos_gap,\n"
        "    ROUND(team_points_share, 3) AS team_pts_share,\n"
        "    ROUND(season_dnf_rate, 3) AS dnf_rate,\n"
        "    fl_departed\n"
        "FROM latest WHERE rn = 1\n"
        "ORDER BY fl_departed DESC, teammate_position_gap"
    ),
    "Departure features: teammate dynamics race-by-race (2024)": (
        "SELECT dt_ref, driverid,\n"
        "    ROUND(teammate_position_gap, 2) AS teammate_pos_gap,\n"
        "    ROUND(teammate_grid_gap, 2) AS teammate_grid_gap,\n"
        "    ROUND(team_points_share, 3) AS team_pts_share,\n"
        "    season_points_current\n"
        "FROM read_parquet('data/gold/abt_departures_inseason.parquet')\n"
        "WHERE YEAR(dt_ref) = 2024\n"
        "    AND driverid IN ('daniel_ricciardo', 'logan_sargeant', 'kevin_magnussen')\n"
        "ORDER BY driverid, dt_ref"
    ),
    "Departure features: performance trends for departed drivers": (
        "WITH departed AS (\n"
        "    SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid ORDER BY dt_ref DESC) AS rn\n"
        "    FROM read_parquet('data/gold/abt_departures_inseason.parquet')\n"
        "    WHERE fl_departed = 1\n"
        ")\n"
        "SELECT driverid, YEAR(dt_ref) AS season,\n"
        "    ROUND(trend_win_rate, 4) AS trend_win_rate,\n"
        "    ROUND(trend_podium_rate, 4) AS trend_podium_rate,\n"
        "    ROUND(avg_position_last10, 2) AS avg_pos_l10,\n"
        "    total_points_last10, driver_age, team_tenure_years\n"
        "FROM departed WHERE rn = 1\n"
        "ORDER BY season DESC, avg_position_last10"
    ),
    "Departure features: career longevity vs departure": (
        "WITH latest AS (\n"
        "    SELECT *, ROW_NUMBER() OVER (PARTITION BY driverid, YEAR(dt_ref) ORDER BY dt_ref DESC) AS rn\n"
        "    FROM read_parquet('data/gold/abt_departures_inseason.parquet')\n"
        ")\n"
        "SELECT fl_departed,\n"
        "    ROUND(AVG(driver_age), 1) AS avg_age,\n"
        "    ROUND(AVG(team_tenure_years), 1) AS avg_tenure,\n"
        "    ROUND(AVG(career_distinct_teams), 1) AS avg_teams,\n"
        "    ROUND(AVG(seasons_since_last_win), 1) AS avg_seasons_since_win,\n"
        "    ROUND(AVG(teammate_position_gap), 2) AS avg_teammate_gap,\n"
        "    COUNT(DISTINCT driverid) AS n_drivers\n"
        "FROM latest WHERE rn = 1 AND fl_departed IS NOT NULL\n"
        "GROUP BY fl_departed\n"
        "ORDER BY fl_departed"
    ),
}


def render_duckdb():
    st.header("DuckDB SQL Console")
    st.caption(
        "Query the F1 data lake directly with SQL. "
        "Press **Ctrl+Enter** (Cmd+Enter on Mac) to run."
    )

    # Show available tables
    with st.expander("Available Data Tables"):
        for name, path_expr in AVAILABLE_TABLES.items():
            st.code(f"-- {name}\nSELECT * FROM {path_expr} LIMIT 10", language="sql")

    # Example queries
    example_label = st.selectbox("Example queries", list(EXAMPLE_QUERIES.keys()))
    default_query = EXAMPLE_QUERIES[example_label]

    with st.form("sql_form"):
        query = st.text_area(
            "SQL Query",
            value=default_query,
            height=180,
            placeholder="SELECT * FROM read_parquet('data/bronze/results.parquet') LIMIT 10",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            run_clicked = st.form_submit_button("Run Query", type="primary")
        with col2:
            limit_results = st.checkbox("Limit to 1000 rows", value=True)

    if run_clicked and query.strip():
        try:
            con = get_duckdb_connection()
            if limit_results and "limit" not in query.lower():
                exec_query = f"SELECT * FROM ({query}) LIMIT 1000"
            else:
                exec_query = query
            result = con.execute(exec_query).fetchdf()
            con.close()

            st.success(f"{len(result)} rows returned")
            st.dataframe(result, use_container_width=True)

            # Download button
            csv = result.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                file_name="query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Query error: {e}")
