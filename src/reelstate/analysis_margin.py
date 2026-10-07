"""Shipped policy (switch margin) vs baselines and the margin-0 run, on the fresh seeds."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AMBER, GREY, INK, TEAL, mean_ci, smooth
from .analysis_v2 import SCEN, ceiling_gain, per_day
from .config import ROOT

OUT = ROOT / "reports" / "v2"


def main() -> None:
    V = json.loads((OUT / "sim_results_v2.json").read_text())
    M = json.loads((OUT / "sim_results_margin.json").read_text())
    n_days, hd = V["config"]["days"], V["config"]["horizon_days"]
    last = slice(n_days - 10, n_days)
    L = [f"# Shipped policy (switch margin {M['margin']}) on fresh seeds", "",
         "Same seeds and random draws as the baselines in `sim_summary_v2.md`; margin tuned on throwaway seed 99.", ""]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scen in zip(axes, ("regulate_dominant", "strong")):
        g_new = per_day(M["main"][scen], n_days)
        g0 = per_day(V["main"][scen]["learned"], n_days)
        greg = per_day(V["main"][scen]["always_regulate"], n_days)
        gm = per_day(V["main"][scen]["always_match"], n_days)
        ceil = np.mean([ceiling_gain(V["main"][scen]["always_match"][s], V["main"][scen]["always_regulate"][s]) for s in range(len(greg))])
        L += [f"## {SCEN[scen]}", "", "| Policy | Gain, last 10 days | Gain, all days | Gap to always-regulate (all days) |", "|---|---|---|---|"]
        for name, g in (("Always match", gm), ("Learned, margin 0", g0), (f"**Learned, margin {M['margin']}**", g_new), ("Always regulate", greg)):
            lm, lc = mean_ci(g[:, last].mean(axis=1)); am, ac = mean_ci(g.mean(axis=1)); dm, dc = mean_ci(g.mean(axis=1) - greg.mean(axis=1))
            L.append(f"| {name} | {lm:+.3f} ± {lc:.3f} | {am:+.3f} ± {ac:.3f} | {dm:+.3f} ± {dc:.3f} |")
        L += ["", f"Personalised-oracle ceiling (all days): {ceil:+.3f}.", ""]
        for name, g, col in (("Always match", gm, GREY), ("Learned, margin 0", g0, "#7FB3B7"), (f"Learned, margin {M['margin']}", g_new, TEAL), ("Always regulate", greg, AMBER)):
            ax.plot(range(1, n_days + 1), smooth(g.mean(axis=0), 5), color=col, lw=2.4 if col == TEAL else 1.5, label=name)
        ax.set_title(SCEN[scen], loc="left", fontsize=10); ax.set_xlabel("Day of use")
    axes[0].set_ylabel("Mean true wellbeing gain"); axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig7_shipped_policy.png"); plt.close(fig)

    H = V["horizon"]
    gl, gr = per_day(M["horizon"], hd), per_day(H["always_regulate"], hd)
    diff = gl - gr
    cum = np.cumsum(diff, axis=1).mean(axis=0)
    be = next((d + 1 for d in range(hd) if cum[d] >= 0 and d > 10), None)
    gm_h = per_day(H["always_match"], hd)
    figh, axh = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, g_, col in (("Always match", gm_h, GREY), ("Always regulate", gr, AMBER), (f"Learned, margin {M['margin']}", gl, TEAL)):
        axh[0].plot(range(1, hd + 1), smooth(g_.mean(axis=0), 9), color=col, lw=2.4 if col == TEAL else 1.5, label=name)
    axh[0].set_xlabel("Day of use"); axh[0].set_ylabel("Mean true gain (9-day smoothing)"); axh[0].legend(frameon=False, fontsize=8)
    axh[0].set_title("120 days, strong heterogeneity", loc="left", fontsize=10)
    axh[1].plot(range(1, hd + 1), cum, color=TEAL, lw=2); axh[1].axhline(0, color=INK, lw=0.8)
    axh[1].set_xlabel("Day of use"); axh[1].set_ylabel("Cumulative gain: learned minus always-regulate")
    axh[1].set_title("Cost is paid back slowly; still negative at day 120", loc="left", fontsize=10)
    figh.tight_layout(); figh.savefig(OUT / "fig11_horizon_shipped.png"); plt.close(figh)
    L += ["## Horizon (strong heterogeneity, 30 users)", "", "| Days | learned − always_regulate |", "|---|---|"]
    for a, b in ((0, 30), (30, 60), (60, 90), (90, hd)):
        m, c = mean_ci(diff[:, a:b].mean(axis=1))
        L.append(f"| {a + 1}–{b} | {m:+.3f} ± {c:.3f} |")
    L += ["", f"* Cumulative break-even day: **{be if be else 'not reached within ' + str(hd) + ' days'}**.", "",
          "Share of low-mood evenings where the policy picks MATCH (last 30 days of horizon):", "", "| Hidden type | Share picking match |", "|---|---|"]
    for pt in ("cathartic", "regulator", "neutral"):
        fr = []
        for recs in M["horizon"]:
            sel = [r for r in recs if r["day"] >= hd - 30 and r["ptype"] == pt and r["true_quadrant"].startswith("lo_v")]
            if sel:
                fr.append(np.mean([r["strategy"] == "match" for r in sel]))
        m, c = mean_ci(fr)
        L.append(f"| {pt} | {m:.2f} ± {c:.2f} |")
    (OUT / "sim_summary_margin.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
