"""Phase 1b: enrich movies with TMDB overview / runtime / language.

Responses are cached one-file-per-movie under data/tmdb_cache, so re-running only
fetches what is missing. The API key is read from .env and never printed.
Accepts either a v3 API key (query param) or a v4 read-access token (Bearer header).
"""
import asyncio
import json

import httpx

from .config import DATA_DIR, TMDB_API_KEY
from .db import connect

CACHE = DATA_DIR / "tmdb_cache"
BASE = "https://api.themoviedb.org/3/movie/"
CONCURRENCY = 12


def auth() -> tuple[dict, dict]:
    if TMDB_API_KEY.startswith("eyJ"):
        return {"Authorization": f"Bearer {TMDB_API_KEY}"}, {}
    return {}, {"api_key": TMDB_API_KEY}


async def fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, tmdb_id: int, params: dict) -> str:
    path = CACHE / f"{tmdb_id}.json"
    if path.exists():
        return "cached"
    async with sem:
        for attempt in range(5):
            try:
                r = await client.get(BASE + str(tmdb_id), params=params)
            except httpx.HTTPError:
                await asyncio.sleep(1 + attempt)
                continue
            if r.status_code == 200:
                d = r.json()
                path.write_text(json.dumps({k: d.get(k) for k in ("id", "overview", "runtime", "original_language", "tagline")}))
                return "ok"
            if r.status_code == 404:
                path.write_text("{}")
                return "missing"
            if r.status_code == 429:
                await asyncio.sleep(2 + attempt * 2)
                continue
            if r.status_code in (401, 403):
                raise SystemExit(f"TMDB rejected the key (HTTP {r.status_code}); check TMDB_API_KEY in .env")
            await asyncio.sleep(1 + attempt)
        return "failed"


async def run() -> None:
    if not TMDB_API_KEY:
        raise SystemExit("TMDB_API_KEY is empty in .env")
    CACHE.mkdir(exist_ok=True)
    with connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT tmdb_id FROM movies WHERE tmdb_id IS NOT NULL ORDER BY movie_id")]
    headers, params = auth()
    sem = asyncio.Semaphore(CONCURRENCY)
    counts: dict[str, int] = {}
    async with httpx.AsyncClient(headers=headers, timeout=20) as client:
        for i in range(0, len(ids), 500):
            res = await asyncio.gather(*(fetch_one(client, sem, t, params) for t in ids[i : i + 500]))
            for s in res:
                counts[s] = counts.get(s, 0) + 1
            print(f"{min(i + 500, len(ids))}/{len(ids)}  {counts}", flush=True)


def apply_to_db() -> None:
    rows = []
    for p in CACHE.glob("*.json"):
        d = json.loads(p.read_text())
        if d.get("id"):
            rows.append((d.get("overview") or None, d.get("runtime") or None, d.get("original_language"), int(d["id"])))
    with connect() as conn, conn.transaction():
        conn.cursor().executemany(
            "UPDATE movies SET overview = %s, runtime_min = %s, language = %s WHERE tmdb_id = %s", rows
        )
        print(conn.execute(
            "SELECT count(*) FILTER (WHERE overview IS NOT NULL), count(*) FILTER (WHERE runtime_min IS NOT NULL), "
            "count(*) FILTER (WHERE language IS NOT NULL) FROM movies"
        ).fetchone())


if __name__ == "__main__":
    asyncio.run(run())
    apply_to_db()
