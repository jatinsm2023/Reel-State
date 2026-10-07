#!/usr/bin/env bash
# Load catalog.sql.gz into a database. Usage: scripts/import_catalog.sh "<database url>"
# Refuses to run twice (the tables must be empty), so it cannot duplicate or wipe anything.
set -euo pipefail
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" LC_ALL="en_US.UTF-8"
cd "$(dirname "$0")/.."
URL="${1:?usage: scripts/import_catalog.sh \"<database url>\"}"
[ -f catalog.sql.gz ] || { echo "catalog.sql.gz not found: run scripts/export_catalog.sh first"; exit 1; }
EXISTING=$(psql "$URL" -Atc "select count(*) from movies")
[ "$EXISTING" = "0" ] || { echo "movies already has $EXISTING rows; not importing again"; exit 1; }
gunzip -c catalog.sql.gz | psql "$URL" -v ON_ERROR_STOP=1 --single-transaction -q
psql "$URL" -c "ANALYZE movies" -c "ANALYZE movie_tags"
psql "$URL" -c "select count(*) as movies, count(content_emb) as with_vectors from movies"
