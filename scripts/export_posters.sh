#!/usr/bin/env bash
# For a database that already has the film catalogue but not the posters. Writes posters.sql, a SELF-CONTAINED
# file (it adds the poster_path column if missing, then loads every poster in one go inside one transaction):
#     psql "<database url>" -v ON_ERROR_STOP=1 -f posters.sql
# A fresh database does not need this: export_catalog.sh already includes posters.
set -euo pipefail
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" LC_ALL="en_US.UTF-8"
cd "$(dirname "$0")/.."
SRC="${SOURCE_URL:-postgresql://$USER@localhost:5432/reelstate}"
{
  echo "BEGIN;"
  echo "ALTER TABLE movies ADD COLUMN IF NOT EXISTS poster_path text;"
  echo "CREATE TEMP TABLE p (movie_id integer, poster_path text, overview text, runtime_min integer) ON COMMIT DROP;"
  echo "COPY p FROM STDIN;"
  psql "$SRC" -c "COPY (SELECT movie_id, poster_path, overview, runtime_min FROM movies WHERE poster_path IS NOT NULL) TO STDOUT"
  echo '\.'
  echo "UPDATE movies m SET poster_path = p.poster_path, overview = COALESCE(m.overview, p.overview), runtime_min = COALESCE(m.runtime_min, p.runtime_min) FROM p WHERE m.movie_id = p.movie_id;"
  echo "SELECT count(poster_path) AS films_with_poster FROM movies;"
  echo "COMMIT;"
} > posters.sql
ls -lh posters.sql | awk '{print "wrote posters.sql (" $5 ")"}'
