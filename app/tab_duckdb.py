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
