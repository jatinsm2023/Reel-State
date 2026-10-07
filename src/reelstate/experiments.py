"""Run the simulator experiments and write results (JSON) + figures to reports/.

A  policy comparison   learned vs always-match vs always-regulate vs random   (gain, learning curve, regret)
B  sensing ablation    context-only / typing / adaptive quiz / fixed quiz / fused  (mood error, questions, gain)
C  cold start          learned policy with vs without the population prior
D  similar-mood signal collaborative boost on vs off

Usage: python -m reelstate.experiments [--users 60 --days 40 --seeds 1 2 3]
"""
import argparse
import json

import numpy as np

from . import service
from .config import ROOT
from .db import connect
from .simulate import run_condition

OUT = ROOT / "reports"


def run(conn, **kw):
    return run_condition(conn, **kw)["records"]


def by_day(records, key, n_days):
    out = np.zeros(n_days)
    for d in range(n_days):
        out[d] = np.mean([r[key] for r in records if r["day"] == d])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    R = {"A": {}, "B": {}, "C": {}, "D": {}, "config": vars(a)}
    with connect() as conn:
        service.seed_quiz_items(conn)
        conn.commit()
        for seed in a.seeds:
            kw = dict(n_users=a.users, n_days=a.days, seed=seed)
            print(f"seed {seed}: A policy comparison", flush=True)
            for mode in ("learned", "always_match", "always_regulate", "random"):
                R["A"].setdefault(mode, []).append(run(conn, policy_mode=mode, sensing="fused", **kw))
            print(f"seed {seed}: B sensing ablation", flush=True)
            for sens in ("context_only", "typing", "quiz_adaptive", "quiz_fixed5", "fused"):
                recs = R["A"]["learned"][-1] if sens == "fused" else run(conn, policy_mode="learned", sensing=sens, **kw)
                R["B"].setdefault(sens, []).append(recs)
            print(f"seed {seed}: C cold start + D similar-mood", flush=True)
            R["C"].setdefault("no_prior", []).append(run(conn, policy_mode="learned", sensing="fused", population_prior=False, **kw))
            R["D"].setdefault("no_collab", []).append(run(conn, policy_mode="learned", sensing="fused", use_collab=False, **kw))
    (OUT / "sim_results.json").write_text(json.dumps(R))
    print("wrote", OUT / "sim_results.json")


if __name__ == "__main__":
    main()
