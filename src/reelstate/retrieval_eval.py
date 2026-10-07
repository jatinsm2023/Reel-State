"""Offline retrieval quality: does the slate FIT the mood, and are the films GOOD and varied?

Compared over random mood points (no users, no policy):
  random              a random catalog film
  ann_only            top-5 by cosine to the mood embedding
  ann+affect          + affect-distance rerank (the first version of this system)
  final               + Bayesian quality floor/weight and sampled slates (shipped)
Metrics: affect distance of the slate to the target (lower = better mood fit), mean MovieLens rating,
share of films rated >= 3.5, and how many distinct films appear across all moods (variety).
"""
import json

import numpy as np

from . import recommend as R
from .config import ROOT
from .db import connect

N_POINTS = 300
SLATE = 5


def main() -> None:
    rng = np.random.default_rng(8)
    pts = rng.uniform(-0.9, 0.9, size=(N_POINTS, 2))
    with connect() as c:
        cat = c.execute("SELECT movie_id, valence, arousal, avg_rating, n_ratings FROM movies").fetchall()
        pool = [dict(zip(("movie_id", "valence", "arousal", "avg_rating", "n_ratings"), r)) for r in cat]
        cache = [R.retrieve(c, (float(v), float(a)), R.Filters(), k=R.CANDIDATES) for v, a in pts]
    keep = (R.QUALITY_FLOOR, R.W_QUALITY)

    def run(name):
        dist, rate, seen = [], [], set()
        r2 = np.random.default_rng(1)
        for (v, a), cands in zip(pts, cache):
            tgt = (float(v), float(a))
            if name == "random":
                sl = [pool[i] for i in r2.choice(len(pool), SLATE, replace=False)]
            elif name == "ann_only":
                sl = sorted(cands, key=lambda c: c["cosine"], reverse=True)[:SLATE]
            elif name == "ann+affect":
                R.QUALITY_FLOOR, R.W_QUALITY = 0.0, 0.0
                sl = R.rerank([dict(c) for c in cands[:100]], tgt)[:SLATE]
                R.QUALITY_FLOOR, R.W_QUALITY = keep
            else:
                sl = R.pick_slate(R.rerank([dict(c) for c in cands], tgt), SLATE, r2)
            dist += [float(np.hypot(x["valence"] - v, x["arousal"] - a)) for x in sl]
            rate += [x["avg_rating"] for x in sl]
            seen |= {x["movie_id"] for x in sl}
        rate = np.array(rate)
        return {"affect_distance": float(np.mean(dist)), "mean_rating": float(rate.mean()), "share_ge_3_5": float((rate >= 3.5).mean()),
                "distinct_films": len(seen), "slots": N_POINTS * SLATE}

    res = {n: run(n) for n in ("random", "ann_only", "ann+affect", "final")}
    L = ["# Retrieval quality (offline, 300 random moods x 5 films)", "",
         "| Pipeline | Affect distance (lower = better fit) | Mean MovieLens rating | Films rated >= 3.5 | Distinct films (of 1500 slots) |", "|---|---|---|---|---|"]
    for n, r in res.items():
        L.append(f"| {n} | {r['affect_distance']:.3f} | {r['mean_rating']:.2f} | {r['share_ge_3_5']:.0%} | {r['distinct_films']} |")
    (ROOT / "reports" / "retrieval_summary.md").write_text("\n".join(L) + "\n")
    (ROOT / "reports" / "retrieval_results.json").write_text(json.dumps(res))
    print("\n".join(L))


if __name__ == "__main__":
    main()
