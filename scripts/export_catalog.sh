#!/usr/bin/env bash
# Dump the film catalogue (movies + movie_tags, including the 384-number vectors) from your LOCAL database
# into catalog.sql.gz, ready to load into Render. Takes a few seconds.
set -euo pipefail
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" LC_ALL="en_US.UTF-8"
cd "$(dirname "$0")/.."
SRC="${SOURCE_URL:-postgresql://$USER@localhost:5432/reelstate}"
pg_dump "$SRC" --data-only --no-owner --no-privileges -t movies -t movie_tags \
  | sed '/^SET transaction_timeout/d' | gzip > catalog.sql.gz
ls -lh catalog.sql.gz
