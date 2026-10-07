#!/usr/bin/env bash
# Run the Reel State web app on http://localhost:8000
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec .venv/bin/python -m uvicorn reelstate.api:app --host 127.0.0.1 --port 8000
