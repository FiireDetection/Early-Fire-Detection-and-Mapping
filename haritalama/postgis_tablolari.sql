CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS drone_frame_points (
    id BIGSERIAL PRIMARY KEY,
    frame_idx INTEGER,
    video_time_s DOUBLE PRECISION,
    prediction TEXT,
    fire_probability DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    altitude_m DOUBLE PRECISION,
    location_source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    geom geometry(Point, 4326)
);

CREATE INDEX IF NOT EXISTS idx_drone_frame_points_geom
    ON drone_frame_points USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_drone_frame_points_frame_idx
    ON drone_frame_points (frame_idx);

DROP VIEW IF EXISTS drone_flight_path;

CREATE VIEW drone_flight_path AS
SELECT
    1 AS id,
    ST_MakeLine(geom ORDER BY video_time_s ASC NULLS LAST, frame_idx ASC NULLS LAST) AS geom,
    COUNT(*) AS total_points,
    MIN(video_time_s) AS start_time,
    MAX(video_time_s) AS end_time,
    AVG(altitude_m) AS avg_altitude_m
FROM drone_frame_points
WHERE geom IS NOT NULL;

DROP VIEW IF EXISTS fire_heatmap_points;

CREATE VIEW fire_heatmap_points AS
SELECT
    id,
    frame_idx,
    video_time_s,
    fire_probability,
    GREATEST(COALESCE(fire_probability, 0), 0) AS heatmap_weight,
    geom
FROM drone_frame_points
WHERE geom IS NOT NULL
  AND COALESCE(fire_probability, 0) >= 0.5;
