"""Evaluate the shipped policy (switch margin) on the fresh seeds, reusing the baselines from experiments_v2.

Same seeds and the same random draws as the baselines, so differences are due to the policy alone.
Margin was tuned on throwaway seed 99 (see the report); results here are on seeds 11-13.
"""
import json

from . import policy, service
from .config import ROOT
from .db import connect
from .simulate import run_condition

OUT = ROOT / "reports" / "v2"


def main() -> None:
    base = json.loads((OUT / "sim_results_v2.json").read_text())["config"]
    R = {"margin": policy.SWITCH_MARGIN, "main": {}, "horizon": []}
    with connect() as conn:
        service.seed_quiz_items(conn)
        conn.commit()
        for scen in ("regulate_dominant", "strong"):
            for seed in base["seeds"]:
                print(scen, seed, flush=True)
                R["main"].setdefault(scen, []).append(run_condition(
                    conn, policy_mode="learned", sensing="fused", n_users=base["users"], n_days=base["days"], seed=seed, scenario=scen)["records"])
        for seed in base["seeds"]:
            print("horizon", seed, flush=True)
            R["horizon"].append(run_condition(conn, policy_mode="learned", sensing="fused", n_users=base["horizon_users"],
                                              n_days=base["horizon_days"], seed=seed, scenario="strong")["records"])
    (OUT / "sim_results_margin.json").write_text(json.dumps(R))
    print("wrote", OUT / "sim_results_margin.json")


if __name__ == "__main__":
    main()
