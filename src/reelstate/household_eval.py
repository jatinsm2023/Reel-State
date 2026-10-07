"""Household aggregation experiment: does fairness-aware blending serve the worst-off member better?

Pairs of synthetic people (with their hidden response types) share a screen for one evening. Each
member's strategy is set to the best one for their hidden type (a stand-in for a trained policy),
so the comparison isolates the AGGREGATION rule, not learning.

  fair     confidence + need weighting, plus a penalty on the worst-served member   (Reel State)
  average  equal weights, mean distance only
  solo_A   only the first member's own recommendation (what a single-profile app does)
Metrics (true wellbeing gain, noise-free): mean over members, and MIN over members (the worst-off person).
"""
import json
from datetime import datetime, timezone

import numpy as np

from . import household, policy, service
from .config import ROOT
from .db import connect
from .quiz import to_va
from .recommend import Filters, recommend
from .simulate import make_population, outcome, cleanup


def best_strategy(person, true_va) -> str:
    q = policy.quadrant(*true_va)
    return "match" if (person.types[q] == "cathartic" and true_va[0] < 0) else "regulate"


def main(n_pairs: int = 150, seed: int = 21) -> None:
    people = make_population(2 * n_pairs, seed)
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)
    res = {m: {"mean": [], "min": []} for m in ("fair", "average", "solo_A")}
    with connect() as conn:
        uids = [service.create_user(conn, f"hh{p.idx}", synthetic=True) for p in people]
        conn.commit()
        for k in range(n_pairs):
            a, b = people[2 * k], people[2 * k + 1]
            ua, ub = uids[2 * k], uids[2 * k + 1]
            th = {ua: a.session_theta(0), ub: b.session_theta(0)}
            truth = {ua: np.array([to_va(th[ua][0]), to_va(th[ua][1])]), ub: np.array([to_va(th[ub][0]), to_va(th[ub][1])])}
            for uid in (ua, ub):          # the system knows each member's mood well (isolates aggregation)
                service.observe(conn, uid, "quiz", {"valence": th[uid][0], "arousal": th[uid][1], "var_valence": 0.15, "var_arousal": 0.15}, t0)
            strat = {ua: best_strategy(a, truth[ua]), ub: best_strategy(b, truth[ub])}
            for method in res:
                conn.execute("DELETE FROM recommendations WHERE user_id = ANY(%s)", ([ua, ub],))
                now = t0.replace(minute=5 + list(res).index(method))
                if method == "solo_A":
                    r = recommend(conn, ua, now, Filters(), rng=rng, force_strategy=strat[ua], slate=1, use_collab=False)
                    movie = np.array(r["slate"][0]["explanation"]["movie_affect"])
                else:
                    r = household.recommend_group(conn, [ua, ub], now, Filters(), rng=rng, slate=1,
                                                  force_strategies=strat, fair=(method == "fair"))
                    movie = np.array(r["slate"][0]["movie_affect"])
                gains = []
                for uid, p in ((ua, a), (ub, b)):
                    _, _ = None, None
                    after, _rep = outcome(np.random.default_rng(p.seed), p, truth[uid], movie)
                    gains.append(policy.wellbeing_gain(tuple(truth[uid]), tuple(after)))
                res[method]["mean"].append(float(np.mean(gains)))
                res[method]["min"].append(float(np.min(gains)))
            conn.commit()
        cleanup(conn, uids)
    summary = {}
    for m, d in res.items():
        summary[m] = {k: (float(np.mean(v)), float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))) for k, v in d.items()}
        print(f"{m:8s} mean-member gain {summary[m]['mean'][0]:+.3f} ± {summary[m]['mean'][1]:.3f} | worst-off member {summary[m]['min'][0]:+.3f} ± {summary[m]['min'][1]:.3f}")
    (ROOT / "reports" / "household_results.json").write_text(json.dumps({"n_pairs": n_pairs, "summary": summary, "raw": res}))


if __name__ == "__main__":
    main()
