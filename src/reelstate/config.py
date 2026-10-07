import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/reelstate")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
DATA_DIR = ROOT / "data"
MIGRATIONS_DIR = ROOT / "db" / "migrations"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
