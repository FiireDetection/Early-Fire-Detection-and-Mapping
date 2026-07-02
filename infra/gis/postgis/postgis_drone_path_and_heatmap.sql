-- Drone flight path as LineString from drone_frame_points.
-- Creates a time-ordered line showing where the drone flew.

DROP VIEW IF EXISTS drone_flight_path;

CREATE VIEW drone_flight_path AS
SELECT
    1 AS id,
    ST_MakeLine(
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
        ORDER BY video_time_s ASC
    ) AS geom,
    COUNT(*) AS total_points,
    MIN(video_time_s) AS start_time,
    MAX(video_time_s) AS end_time,
    AVG(altitude_m) AS avg_altitude_m
FROM drone_frame_points
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Active fire heatmap view for GeoServer heatmap styling.
DROP VIEW IF EXISTS fire_heatmap_points;

CREATE VIEW fire_heatmap_points AS
SELECT
    id AS fire_track_id,
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) AS geom,
    fire_probability AS last_probability,
    COALESCE(fire_probability, 0) AS heatmap_weight,
    video_time_s AS last_video_time_s,
    1 AS observations,
    NULL::DOUBLE PRECISION AS max_area_m2,
    NULL::TEXT AS overlay_url
FROM drone_frame_points
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL
  AND COALESCE(fire_probability, 0) >= 0.5
ORDER BY fire_probability DESC;
