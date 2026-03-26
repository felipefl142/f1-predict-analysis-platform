"""Tab 3: Interactive DuckDB SQL Console."""

import streamlit as st

from app.helpers import get_duckdb_connection, AVAILABLE_TABLES

EXAMPLE_QUERIES = {
    "-- Select an example --": "",
    "Top 20 drivers by race count": (
        "SELECT driverid, full_name, COUNT(*) AS races "
        "FROM read_parquet('data/bronze/results.parquet') "
        "WHERE mode = 'Race' "
        "GROUP BY driverid, full_name "
        "ORDER BY races DESC LIMIT 20"
    ),
    "Points per season (2024)": (
        "SELECT driverid, full_name, team_name, SUM(points) AS total_points "
        "FROM read_parquet('data/bronze/results.parquet') "
        "WHERE year = 2024 AND mode = 'Race' "
        "GROUP BY driverid, full_name, team_name "
        "ORDER BY total_points DESC"
    ),
    "Verstappen career features": (
        "SELECT dt_ref, qtd_wins_life, qtd_podiums_life, total_points_life, "
        "avg_position_life "
        "FROM read_parquet('data/silver/fs_driver_all.parquet') "
        "WHERE driverid = 'max_verstappen' "
        "ORDER BY dt_ref DESC LIMIT 10"
    ),
    "Champion ABT sample": (
        "SELECT dt_ref, driverid, fl_champion, qtd_wins_life, total_points_last20 "
        "FROM read_parquet('data/gold/abt_champions.parquet') "
        "WHERE fl_champion = 1 "
        "ORDER BY dt_ref DESC LIMIT 20"
    ),
    "Team win counts by season": (
        "SELECT year, team_name, "
        "SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) AS wins "
        "FROM read_parquet('data/bronze/results.parquet') "
        "WHERE mode = 'Race' "
        "GROUP BY year, team_name "
        "HAVING wins > 0 "
        "ORDER BY year DESC, wins DESC"
    ),
}


def render_duckdb():
    st.header("DuckDB SQL Console")
    st.caption("Query the F1 data lake directly with SQL. DuckDB reads Parquet files natively.")

    # Show available tables
    with st.expander("Available Data Tables"):
        for name, path_expr in AVAILABLE_TABLES.items():
            st.code(f"-- {name}\nSELECT * FROM {path_expr} LIMIT 10", language="sql")

    # Example queries
    example_label = st.selectbox("Example queries", list(EXAMPLE_QUERIES.keys()))
    default_query = EXAMPLE_QUERIES[example_label]

    query = st.text_area(
        "SQL Query",
        value=default_query,
        height=150,
        placeholder="SELECT * FROM read_parquet('data/bronze/results.parquet') LIMIT 10",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_clicked = st.button("Run Query", type="primary")
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
