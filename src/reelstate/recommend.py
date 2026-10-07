"""Recommendation: choose a strategy, turn it into a target mood, retrieve and rank movies.

Pipeline for one request
  1. current mood (already fused/drifted)                       -> (v, a)
  2. policy.choose over this user's arms for the mood quadrant  -> 'match' | 'regulate'
  3. target mood for that strategy                              -> (tv, ta)
  4. affect_space.embed(tv, ta)                                 -> query vector in the movie space
  5. ONE SQL statement: relational filters + ANN ordering by cosine distance, top CANDIDATES
  6. rerank candidates by  cosine  -  W_AFFECT * ||affect - target||^2  +  W_QUALITY * quality
  7. attach a plain-language explanation and persist the slate
"""
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import affect, affect_space, policy, service
from .quiz import to_va

CANDIDATES = 400        # ANN pool; large enough that a quality floor still leaves real choice
SLATE = 5
W_AFFECT = 0.55         # how strongly the movie's own affect must sit near the target
W_QUALITY = 0.10        # weight on the (Bayesian) quality margin above the catalog mean
QUALITY_FLOOR = 3.4     # only films whose Bayesian-average MovieLens rating clears this are eligible
BAYES_M = 300           # prior strength (ratings) for the Bayesian average
CATALOG_MEAN = 3.28     # mean MovieLens rating across the catalog
SLATE_POOL = 25         # slates are sampled from the top of the ranking, not always the same top 5
SLATE_TAU = 0.05        # sampling temperature over scores (lower = closer to deterministic top-k)
W_COLLAB = 0.20         # boost for films that worked for people whose mood right now resembles yours


@dataclass
class Filters:
    max_runtime: int | None = None
    language: str | None = None
    genres: list[str] | None = None            # any-overlap
    exclude_movie_ids: list[int] = field(default_factory=list)


CANDIDATE_SQL = """
SELECT movie_id, title, year, genres, runtime_min, language, valence, arousal, overview,
       avg_rating, n_ratings, poster_path, 1 - (content_emb <=> %(q)s) AS cosine
FROM movies
WHERE content_emb IS NOT NULL
  AND (%(max_runtime)s::int IS NULL OR runtime_min <= %(max_runtime)s)
  AND (%(language)s::text IS NULL OR language = %(language)s)
  AND (%(genres)s::text[] IS NULL OR genres && %(genres)s)
  AND NOT (movie_id = ANY(%(exclude)s))
ORDER BY content_emb <=> %(q)s
LIMIT %(k)s
"""


def retrieve(conn, target: tuple[float, float], filters: Filters, k: int = CANDIDATES) -> list[dict]:
    q = affect_space.embed(*target)
    cur = conn.execute(CANDIDATE_SQL, {
        "q": q, "max_runtime": filters.max_runtime, "language": filters.language,
        "genres": filters.genres, "exclude": filters.exclude_movie_ids, "k": k,
    })
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def quality(c: dict) -> float:
    """Bayesian-average MovieLens rating: shrinks films with few ratings toward the catalog mean."""
    n, avg = c["n_ratings"] or 0, c["avg_rating"] or CATALOG_MEAN
    return (n * avg + BAYES_M * CATALOG_MEAN) / (n + BAYES_M)


def apply_floor(cands: list[dict]) -> list[dict]:
    """Keep only well-regarded films; if filters leave none, fall back rather than return nothing."""
    good = [c for c in cands if quality(c) >= QUALITY_FLOOR]
    return good or cands


def rerank(cands: list[dict], target: tuple[float, float], collab: dict[int, float] | None = None) -> list[dict]:
    collab = collab or {}
    cands = apply_floor(cands)
    for c in cands:
        d2 = (c["valence"] - target[0]) ** 2 + (c["arousal"] - target[1]) ** 2
        c["affect_dist"] = float(d2 ** 0.5)
        c["collab"] = collab.get(c["movie_id"], 0.0)
        c["score"] = c["cosine"] - W_AFFECT * d2 + W_QUALITY * (quality(c) - CATALOG_MEAN) + W_COLLAB * c["collab"]
    return sorted(cands, key=lambda c: c["score"], reverse=True)


def pick_slate(ranked: list[dict], k: int, rng, diversify: bool = True) -> list[dict]:
    """Sample k films from the top of the ranking (softmax over scores) so nights differ; order by score."""
    if not diversify or len(ranked) <= k:
        return ranked[:k]
    top = ranked[:SLATE_POOL]
    s = np.array([c["score"] for c in top])
    p = np.exp((s - s.max()) / SLATE_TAU)
    p /= p.sum()
    idx = rng.choice(len(top), size=min(k, len(top)), replace=False, p=p)
    return sorted((top[i] for i in idx), key=lambda c: c["score"], reverse=True)


def fetch_by_ids(conn, ids: list[int], target: tuple[float, float]) -> list[dict]:
    if not ids:
        return []
    cur = conn.execute(
        "SELECT movie_id, title, year, genres, runtime_min, language, valence, arousal, overview, avg_rating, n_ratings, poster_path, "
        "1 - (content_emb <=> %s) AS cosine FROM movies WHERE movie_id = ANY(%s)", (affect_space.embed(*target), ids))
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def explain(conn, movie: dict, strategy: str, mood: tuple[float, float], target: tuple[float, float]) -> dict:
    tags = dict(conn.execute("SELECT tag, relevance FROM movie_tags WHERE movie_id = %s", (movie["movie_id"],)).fetchall())
    word = lambda x, lo, hi: hi if x > 0.25 else lo if x < -0.25 else None
    mood_words = [w for w in (word(mood[0], "low", "upbeat"), word(mood[1], "calm", "wound-up")) if w]
    if strategy == "match":
        why = f"Matches how you're feeling ({' and '.join(mood_words) or 'balanced'})."
    else:
        goal = [w for w in (word(target[0], "heavier", "uplifting"), word(target[1], "slower", "livelier")) if w]
        why = f"A change of pace from feeling {' and '.join(mood_words) or 'balanced'}: aiming for {' and '.join(goal) or 'balance'}."
    return {
        "strategy": strategy,
        "why": why,
        "feel_tags": affect.top_affect_tags(tags, 3),
        "movie_affect": [round(movie["valence"], 2), round(movie["arousal"], 2)],
        "target_affect": [round(target[0], 2), round(target[1], 2)],
        "affect_distance": round(movie["affect_dist"], 3),
        "cosine": round(movie["cosine"], 3),
        "similar_mood_boost": round(movie.get("collab", 0.0), 3),
        "quality": round(quality(movie), 2),
    }


def recommend(conn, user_id: int, now: datetime, filters: Filters | None = None, rng=None,
              force_strategy: str | None = None, slate: int = SLATE, use_collab: bool = True, diversify: bool = True) -> dict:
    filters = filters or Filters()
    rng = rng or np.random.default_rng()
    mood, _ = service.current_mood(conn, user_id, now)
    service.save_state(conn, user_id, now, mood)      # the belief at request time is a trajectory point
    mood_ts = now
    v, a = to_va(mood.v), to_va(mood.a)
    q = policy.quadrant(v, a)
    arms = service.load_arms(conn, user_id, q)
    if force_strategy:
        strategy, draws = force_strategy, {}
    else:
        strategy, draws = policy.choose(arms, rng)
    target = policy.target_for(strategy, v, a)

    seen = [r[0] for r in conn.execute("SELECT movie_id FROM recommendations WHERE user_id = %s", (user_id,))]
    filters.exclude_movie_ids = list(set(filters.exclude_movie_ids) | set(seen))
    cands = retrieve(conn, target, filters)
    collab = service.collab_scores(conn, user_id, now) if use_collab else {}
    have = {c["movie_id"] for c in cands}
    extra = [m for m in collab if m not in have and m not in filters.exclude_movie_ids]
    cands += [c for c in fetch_by_ids(conn, extra, target)
              if (filters.max_runtime is None or (c["runtime_min"] or 0) <= filters.max_runtime)
              and (filters.language is None or c["language"] == filters.language)
              and (not filters.genres or set(c["genres"]) & set(filters.genres))]
    ranked = pick_slate(rerank(cands, target, collab), slate, rng, diversify)

    out = []
    for rank, m in enumerate(ranked, start=1):
        ex = explain(conn, m, strategy, (v, a), target)
        rec_id = conn.execute(
            "INSERT INTO recommendations (user_id, ts, mood_ts, movie_id, strategy, rank, score, explanation) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING rec_id",
            (user_id, now, mood_ts, m["movie_id"], strategy, rank, m["score"], json.dumps(ex)),
        ).fetchone()[0]
        out.append({"rec_id": rec_id, "movie_id": m["movie_id"], "title": m["title"], "year": m["year"],
                    "genres": m["genres"], "runtime_min": m["runtime_min"], "overview": m["overview"],
                    "poster_path": m.get("poster_path"), "avg_rating": m["avg_rating"], "n_ratings": m["n_ratings"],
                    "score": round(m["score"], 4), "explanation": ex})
    return {"strategy": strategy, "quadrant": q, "mood": {"valence": v, "arousal": a},
            "target": {"valence": target[0], "arousal": target[1]},
            "arm_means": {s: round(arms[s].mean, 3) for s in arms}, "arm_obs": {s: arms[s].obs for s in arms}, "slate": out}
