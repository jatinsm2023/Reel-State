"""Analyse reports/v2/sim_results_v2.json -> figures + reports/v2/sim_summary_v2.md.

Key quantities
  gain            true wellbeing gain per session (noise-free)
  ceiling         'personalised oracle': for each person and mood quadrant, the better of the two fixed
                  strategies in hindsight (an upper bound for any per-person policy that picks one
                  strategy per quadrant)
  headroom        ceiling - best single fixed strategy: the MOST personalisation could possibly add
  exploration     always_regulate - learned (in the no-heterogeneity scenario this is pure cost)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AMBER, COLORS, GREY, INK, TEAL, WINE, mean_ci, smooth
from .config import ROOT

OUT = ROOT / "reports" / "v2"
LABEL = {"learned": "Learned (pooled)", "learned_flat": "Learned (no pooling)", "always_match": "Always match",
         "always_regulate": "Always regulate", "random": "Random"}
COL = {"learned": TEAL, "learned_flat": "#7FB3B7", "always_match": GREY, "always_regulate": AMBER, "random": WINE}
SCEN = {"regulate_dominant": "No heterogeneity (changing mood always wins)", "strong": "Strong heterogeneity (a minority needs matching)"}


def per_day(runs, n_days):
    out = np.zeros((len(runs), n_days))
    for i, recs in enumerate(runs):
        for d in range(n_days):
            out[i, d] = np.mean([r["gain"] for r in recs if r["day"] == d])
    return out


def ceiling_gain(fixed_match, fixed_reg):
    """Per-seed: mean gain of the per-(user, quadrant) hindsight-best fixed strategy."""
    cells = {}
    for tag, recs in (("m", fixed_match), ("r", fixed_reg)):
        for r in recs:
            cells.setdefault((r["user"], r["true_quadrant"]), {"m": [], "r": []})[tag].append(r["gain"])
    num = den = 0.0
    for c in cells.values():
        for tag in ("m", "r"):
            pass
        best = max(np.mean(c["m"]) if c["m"] else -9, np.mean(c["r"]) if c["r"] else -9)
        n = max(len(c["m"]), len(c["r"]))
        num += best * n
        den += n
    return num / den


def main() -> None:
    R = json.loads((OUT / "sim_results_v2.json").read_text())
    n_days = R["config"]["days"]
    last = slice(n_days - 10, n_days)
    L = ["# Final simulation results (fresh seeds)", "",
         f"{R['config']['users']} synthetic users x {n_days} days x seeds {R['config']['seeds']}. Gain = true change in wellbeing, "
         "noise-free, in [-1, 1]. Intervals are 95% t-intervals over seeds. Policy hyperparameters were tuned on a separate seed (99).", ""]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scen in zip(axes, ("regulate_dominant", "strong")):
        M = R["main"][scen]
        L += [f"## Scenario: {SCEN[scen]}", "", "| Policy | Gain, last 10 days | Gain, all days |", "|---|---|---|"]
        curves = {}
        for name in ("always_match", "random", "learned_flat", "learned", "always_regulate"):
            g = per_day(M[name], n_days)
            curves[name] = g
            lm, lc = mean_ci(g[:, last].mean(axis=1)); am, ac = mean_ci(g.mean(axis=1))
            L.append(f"| {LABEL[name]} | {lm:+.3f} ± {lc:.3f} | {am:+.3f} ± {ac:.3f} |")
            ax.plot(range(1, n_days + 1), smooth(g.mean(axis=0), 5), color=COL[name], lw=2.4 if name == "learned" else 1.5, label=LABEL[name])
        ceil = [ceiling_gain(M["always_match"][s], M["always_regulate"][s]) for s in range(len(M["learned"]))]
        best_fixed = [max(np.mean([r["gain"] for r in M["always_match"][s]]), np.mean([r["gain"] for r in M["always_regulate"][s]]))
                      for s in range(len(M["learned"]))]
        reg_all = curves["always_regulate"].mean(axis=1)
        learned_all = curves["learned"].mean(axis=1)
        cm, cc = mean_ci(ceil); bm, bc = mean_ci(best_fixed)
        hm, hc = mean_ci(np.array(ceil) - np.array(best_fixed)); em, ec = mean_ci(reg_all - learned_all)
        L += ["", f"* Personalised-oracle ceiling (all days): **{cm:+.3f} ± {cc:.3f}**; best single fixed strategy: {bm:+.3f} ± {bc:.3f}.",
              f"* **Headroom** (most that personalisation could add): **{hm:+.3f} ± {hc:.3f}**.",
              f"* **Exploration cost** (always-regulate minus learned, all days): **{em:+.3f} ± {ec:.3f}**.", ""]
        pm, pc = mean_ci(curves["learned"].mean(axis=1) - curves["learned_flat"].mean(axis=1))
        L += [f"* Pooling ablation (pooled minus flat, all days): {pm:+.3f} ± {pc:.3f}.", ""]
        ax.set_title(SCEN[scen], loc="left", fontsize=10); ax.set_xlabel("Day of use")
    axes[0].set_ylabel("Mean true wellbeing gain (5-day smoothing)"); axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig5_policy_scenarios.png"); plt.close(fig)

    # ---- horizon
    H = R["horizon"]
    hd = R["config"]["horizon_days"]
    g = {k: per_day(v, hd) for k, v in H.items()}
    diff = g["learned"] - g["always_regulate"]                              # (seeds, days)
    cum = np.cumsum(diff, axis=1).mean(axis=0)
    be = next((d + 1 for d in range(hd) if cum[d] >= 0 and d > 10), None)
    win = lambda a, b: mean_ci(diff[:, a:b].mean(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in ("always_match", "always_regulate", "learned"):
        axes[0].plot(range(1, hd + 1), smooth(g[name].mean(axis=0), 9), color=COL[name], lw=2.4 if name == "learned" else 1.5, label=LABEL[name])
    axes[0].set_xlabel("Day of use"); axes[0].set_ylabel("Mean true gain (9-day smoothing)"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Long horizon, strong heterogeneity", loc="left", fontsize=10)
    axes[1].plot(range(1, hd + 1), cum, color=TEAL, lw=2); axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_xlabel("Day of use"); axes[1].set_ylabel("Cumulative gain: learned minus always-regulate")
    axes[1].set_title("Where personalisation pays back its exploration", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig6_horizon.png"); plt.close(fig)
    a0, a1 = win(0, 30), win(30, 60); a2, a3 = win(60, 90), win(90, hd)
    L += ["## Horizon study (strong heterogeneity, 30 users)", "",
          "Per-session gain of learned minus always-regulate, by period:", "",
          "| Days | learned − always_regulate |", "|---|---|",
          f"| 1–30 | {a0[0]:+.3f} ± {a0[1]:.3f} |", f"| 31–60 | {a1[0]:+.3f} ± {a1[1]:.3f} |",
          f"| 61–90 | {a2[0]:+.3f} ± {a2[1]:.3f} |", f"| 91–{hd} | {a3[0]:+.3f} ± {a3[1]:.3f} |", "",
          f"* Cumulative break-even day: **{be if be else 'not reached within ' + str(hd) + ' days'}**.", ""]

    # adaptation by type at the end of the horizon
    L += ["Share of low-mood evenings where the learned policy picks MATCH (last 30 days of the horizon run):", "",
          "| Hidden type | Share picking match |", "|---|---|"]
    for pt in ("cathartic", "regulator", "neutral"):
        fr = []
        for recs in H["learned"]:
            sel = [r for r in recs if r["day"] >= hd - 30 and r["ptype"] == pt and r["true_quadrant"].startswith("lo_v")]
            if sel:
                fr.append(np.mean([r["strategy"] == "match" for r in sel]))
        m, c = mean_ci(fr)
        L.append(f"| {pt} | {m:.2f} ± {c:.2f} |")
    (OUT / "sim_summary_v2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
