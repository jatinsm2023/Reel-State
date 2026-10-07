-- TMDB poster path (e.g. /abc123.jpg). The browser loads images from TMDB's image CDN.
-- IF NOT EXISTS: scripts/export_posters.sh may already have added the column by hand.
ALTER TABLE movies ADD COLUMN IF NOT EXISTS poster_path text;
