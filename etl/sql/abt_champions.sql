-- Analytical Base Table for champion prediction.
-- Joins driver features with historical champion labels.

SELECT
    f.*,
    COALESCE(c.is_champion, 0) AS fl_champion
FROM read_parquet('{silver_dir}/fs_driver_all.parquet') f
LEFT JOIN (
    SELECT driverid, year, 1 AS is_champion
    FROM read_csv('{champions_csv}', header=true, auto_detect=true)
) c
    ON f.driverid = c.driverid
    AND EXTRACT(YEAR FROM f.dt_ref) = c.year
WHERE f.dt_ref >= DATE '2000-01-01'
ORDER BY f.dt_ref DESC, f.driverid
