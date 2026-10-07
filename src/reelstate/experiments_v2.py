"""Final policy evaluation on FRESH seeds (tuning used seed 99; these use 11, 12, 13).

Scenarios
  regulate_dominant   nothing to personalise: measures what exploration COSTS
  strong              a cathartic minority is hurt by being cheered up: measures whether it can be FOUND
Policies
  learned          hierarchical Thompson sampling (population prior + pooling across a person's moods)
  learned_flat     same without pooling (ablation)
  always_match / always_regulate / random
Horizon study (strong scenario, 30 users, 120 days): where does personalisation break even?

Usage: python -m reelstate.experiments_v2 [--users 60 --days 40 --seeds 11 12 13 --horizon-days 120]
"""
import argparse
import json

from . import service
from .config import ROOT
from .db import connect
from .simulate import run_condition

OUT = ROOT / "reports" / "v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--horizon-days", type=int, default=120)
    ap.add_argument("--horizon-users", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"config": vars(a), "main": {}, "horizon": {}}
    with connect() as conn:
        service.seed_quiz_items(conn)
        conn.commit()
        for scen in ("regulate_dominant", "strong"):
            for seed in a.seeds:
                kw = dict(sensing="fused", n_users=a.users, n_days=a.days, seed=seed, scenario=scen)
                print(f"{scen} seed {seed}", flush=True)
                for name, extra in (("learned", {}), ("learned_flat", {"pooling": False}),
                                    ("always_match", {}), ("always_regulate", {}), ("random", {})):
                    mode = name if name in ("always_match", "always_regulate", "random") else "learned"
                    R["main"].setdefault(scen, {}).setdefault(name, []).append(
                        run_condition(conn, policy_mode=mode, **kw, **extra)["records"])
        for seed in a.seeds:
            print(f"horizon seed {seed}", flush=True)
            kw = dict(sensing="fused", n_users=a.horizon_users, n_days=a.horizon_days, seed=seed, scenario="strong")
            for name, mode in (("learned", "learned"), ("always_regulate", "always_regulate"), ("always_match", "always_match")):
                R["horizon"].setdefault(name, []).append(run_condition(conn, policy_mode=mode, **kw)["records"])
    (OUT / "sim_results_v2.json").write_text(json.dumps(R))
    print("wrote", OUT / "sim_results_v2.json")


if __name__ == "__main__":
    main()
