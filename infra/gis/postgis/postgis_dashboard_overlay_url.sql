-- Adds image URL columns used by the dashboard when richer fire observation
-- tables exist. Safe to run repeatedly.

ALTER TABLE IF EXISTS fire_observations
    ADD COLUMN IF NOT EXISTS overlay_url TEXT;

ALTER TABLE IF EXISTS active_fire_tracks
    ADD COLUMN IF NOT EXISTS overlay_url TEXT;
