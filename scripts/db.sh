#!/usr/bin/env bash
# Start/stop/status for the local Reel State Postgres (Homebrew postgresql@17).
# LC_ALL must be a valid locale or Postgres refuses to start on macOS.
set -euo pipefail
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
export LC_ALL="en_US.UTF-8"
DATA="/opt/homebrew/var/postgresql@17"
LOG="$(cd "$(dirname "$0")/.." && pwd)/db/postgres.log"
case "${1:-status}" in
  start)  pg_ctl -D "$DATA" -l "$LOG" start ;;
  stop)   pg_ctl -D "$DATA" stop ;;
  status) pg_ctl -D "$DATA" status ;;
  psql)   shift; psql reelstate "$@" ;;
  *) echo "usage: $0 {start|stop|status|psql}"; exit 1 ;;
esac
