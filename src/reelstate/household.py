"""Household mode: one screen, several people, each with their own mood and learned strategy.

  1. every member's current mood and Thompson-sampled strategy give that member a TARGET mood
  2. member targets are blended into one group target, weighting
         confidence  (a confident reading counts for more than a guess)
         need        (someone feeling low gets extra say; a crude 'least misery' nudge)
  3. one ANN query retrieves candidates near the group target
  4. candidates are ranked for FAIRNESS, not just the average: the penalty is the mean squared
     affect distance to each member's target PLUS a heavier penalty on the worst-served member
  5. each member gets their own recommendations row (their own strategy, their own check-in), so
     every person's policy learns from their own felt outcome

Credit assignment is per member: a member's arm updates from what THAT member reports.
"""
import json
from datetime import datetime

import numpy as np

from . import policy, service
from .quiz import to_va
from .recommend import (CANDIDATES, CATALOG_MEAN, W_AFFECT, W_QUALITY, Filters, apply_floor, pick_slate, quality,  # noqa: F401
                        retrieve)

W_FAIR = 0.30           # extra penalty on the worst-served member's affect distance
NEED_BOOST = 0.5        # how much extra say a low-valence member gets


def member_view(conn, uid: int, now: datetime, rng, force_strategy: str | None = None, fair: bool = True) -> dict:
    mood, _ = service.current_mood(conn, uid, now)
    service.save_state(conn, uid, now, mood)
    v, a = to_va(mood.v), to_va(mood.a)
    q = policy.quadrant(v, a)
    strategy = force_strategy or policy.choose(service.load_arms(conn, uid, q), rng)[0]
    target = policy.target_for(strategy, v, a)
    sd = ((mood.var_v + mood.var_a) / 2) ** 0.5
    weight = (1.0 / (1.0 + sd)) * (1.0 + NEED_BOOST * max(0.0, -v)) if fair else 1.0
    name = conn.execute("SELECT display_name FROM users WHERE user_id = %s", (uid,)).fetchone()[0]
    return {"user_id": uid, "name": name, "mood": (v, a), "quadrant": q, "strategy": strategy, "target": target, "weight": weight}


def group_target(members: list[dict]) -> tuple[float, float]:
    w = np.array([m["weight"] for m in members])
    t = np.array([m["target"] for m in members])
    g = (w[:, None] * t).sum(axis=0) / w.sum()
    return float(g[0]), float(g[1])


def rank_for_group(cands: list[dict], members: list[dict], fair: bool = True) -> list[dict]:
    cands = apply_floor(cands)
    for c in cands:
        d2 = np.array([(c["valence"] - m["target"][0]) ** 2 + (c["arousal"] - m["target"][1]) ** 2 for m in members])
        c["member_dist"] = [float(x ** 0.5) for x in d2]
        c["score"] = c["cosine"] - W_AFFECT * d2.mean() - (W_FAIR * d2.max() if fair else 0.0) + W_QUALITY * (quality(c) - CATALOG_MEAN)
    return sorted(cands, key=lambda c: c["score"], reverse=True)


def recommend_group(conn, user_ids: list[int], now: datetime, filters: Filters | None = None, rng=None, slate: int = 5,
                    force_strategies: dict[int, str] | None = None, fair: bool = True, diversify: bool = True) -> dict:
    if len(user_ids) < 2:
        raise ValueError("household mode needs at least two people")
    rng = rng or np.random.default_rng()
    filters = filters or Filters()
    members = [member_view(conn, uid, now, rng, (force_strategies or {}).get(uid), fair) for uid in user_ids]
    target = group_target(members)
    seen = {r[0] for r in conn.execute("SELECT DISTINCT movie_id FROM recommendations WHERE user_id = ANY(%s)", (user_ids,))}
    filters.exclude_movie_ids = list(set(filters.exclude_movie_ids) | seen)
    ranked = pick_slate(rank_for_group(retrieve(conn, target, filters, k=CANDIDATES), members, fair), slate, rng, diversify)

    out = []
    for rank, m in enumerate(ranked, start=1):
        rec_ids = {}
        for mem, dist in zip(members, m["member_dist"]):
            ex = {
                "strategy": mem["strategy"], "group": True,
                "why": f"For {mem['name']}: " + ("matches how they feel." if mem["strategy"] == "match" else "a change of pace for them."),
                "movie_affect": [round(m["valence"], 2), round(m["arousal"], 2)],
                "target_affect": [round(mem["target"][0], 2), round(mem["target"][1], 2)],
                "affect_distance": round(dist, 3), "cosine": round(m["cosine"], 3),
            }
            rec_ids[mem["user_id"]] = conn.execute(
                "INSERT INTO recommendations (user_id, ts, mood_ts, movie_id, strategy, rank, score, explanation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING rec_id",
                (mem["user_id"], now, now, m["movie_id"], mem["strategy"], rank, m["score"], json.dumps(ex)),
            ).fetchone()[0]
        out.append({"movie_id": m["movie_id"], "title": m["title"], "year": m["year"], "genres": m["genres"],
                    "runtime_min": m["runtime_min"], "overview": m["overview"], "score": round(m["score"], 4),
                    "rec_ids": rec_ids, "member_distance": {mem["name"]: round(d, 2) for mem, d in zip(members, m["member_dist"])},
                    "movie_affect": [round(m["valence"], 2), round(m["arousal"], 2)]})
    return {
        "members": [{"user_id": m["user_id"], "name": m["name"], "mood": {"valence": m["mood"][0], "arousal": m["mood"][1]},
                     "strategy": m["strategy"], "weight": round(m["weight"], 3)} for m in members],
        "group_target": {"valence": target[0], "arousal": target[1]},
        "slate": out,
    }
