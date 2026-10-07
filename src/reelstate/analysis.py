"""Turn reports/sim_results.json into figures and a markdown summary (reports/sim_summary.md)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ROOT

OUT = ROOT / "reports"
TEAL, AMBER, WINE, GREY, INK = "#2A6A70", "#B8752A", "#8B4A52", "#8A8D92", "#22242A"
COLORS = {"learned": TEAL, "always_match": GREY, "always_regulate": AMBER, "random": WINE}
LABEL = {"learned": "Learned policy (Reel State)", "always_match": "Always match", "always_regulate": "Always regulate", "random": "Random"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.15})


def per_day(runs, key, n_days):
    """runs: list over seeds of record lists -> array (seeds, days) of the daily mean of `key`."""
    out = np.zeros((len(runs), n_days))
    for i, recs in enumerate(runs):
        for d in range(n_days):
            out[i, d] = np.mean([r[key] for r in recs if r["day"] == d])
    return out


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    from scipy import stats
    return m, stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def smooth(y, w=3):
    return np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def main() -> None:
    R = json.loads((OUT / "sim_results.json").read_text())
    n_days = R["config"]["days"]
    last = slice(n_days - 10, n_days)
    lines = ["# Simulation results", "",
             f"{R['config']['users']} synthetic users x {n_days} days x {len(R['config']['seeds'])} seeds. "
             "Gain = true change in wellbeing (valence up, arousal less extreme), noise-free, range [-1, 1]. "
             "Intervals are 95% t-intervals over seeds.", ""]

    # ---- Fig 1: learning curves
    fig, ax = plt.subplots(figsize=(7, 4))
    A = R["A"]
    for mode in ("always_match", "random", "always_regulate", "learned"):
        g = per_day(A[mode], "gain", n_days)
        m = smooth(g.mean(axis=0))
        ax.plot(range(1, n_days + 1), m, color=COLORS[mode], lw=2.4 if mode == "learned" else 1.6, label=LABEL[mode])
        if len(g) > 1:
            se = g.std(axis=0, ddof=1) / np.sqrt(len(g))
            ax.fill_between(range(1, n_days + 1), m - 1.96 * smooth(se), m + 1.96 * smooth(se), color=COLORS[mode], alpha=0.12, lw=0)
    ax.set_xlabel("Day of use"); ax.set_ylabel("Mean true wellbeing gain"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("The policy learns what works for each person", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig1_learning_curve.png"); plt.close(fig)

    # ---- table A
    lines += ["## A. Does learning match-vs-regulate help?", "", f"| Policy | Gain, last 10 days | Gain, all days | Regret vs hindsight-best fixed |", "|---|---|---|---|"]
    fixed = {}
    for mode in ("always_match", "always_regulate"):
        fixed[mode] = [{(r["user"], r["day"]): r["gain"] for r in recs} for recs in A[mode]]
    for mode in ("learned", "always_match", "always_regulate", "random"):
        g = per_day(A[mode], "gain", n_days)
        lastm, lastci = mean_ci(g[:, last].mean(axis=1))
        allm, allci = mean_ci(g.mean(axis=1))
        regrets = []
        for s, recs in enumerate(A[mode]):
            best = [max(fixed["always_match"][s][(r["user"], r["day"])], fixed["always_regulate"][s][(r["user"], r["day"])]) - r["gain"]
                    for r in recs if r["day"] >= n_days - 10]
            regrets.append(np.mean(best))
        rm, rci = mean_ci(regrets)
        lines.append(f"| {LABEL[mode]} | {lastm:+.3f} ± {lastci:.3f} | {allm:+.3f} ± {allci:.3f} | {rm:.3f} ± {rci:.3f} |")
    lines.append("")

    # ---- Fig 2: who gets what (adaptation by hidden person type)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, ptype in zip(axes, ("regulator", "cathartic")):
        fr = []
        for recs in A["learned"]:
            sel = [r for r in recs if r["day"] >= n_days - 15 and r["ptype"] == ptype and r["true_quadrant"].startswith("lo_v")]
            fr.append(np.mean([r["strategy"] == "match" for r in sel]) if sel else np.nan)
        m, ci = mean_ci([x for x in fr if not np.isnan(x)])
        ax.bar(["picks match"], [m], yerr=[ci], color=TEAL if ptype == "cathartic" else AMBER, width=0.5, capsize=4)
        ax.set_title(f"{ptype.title()} users, low-mood evenings", fontsize=10, loc="left"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of picks that match mood")
    fig.suptitle("Same system, different strategy per person (days 26–40)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig2_adaptation.png"); plt.close(fig)
    lines += ["## Adaptation by hidden person type (low-valence evenings, last 15 days)", "", "| Hidden type | Share of picks that were 'match' |", "|---|---|"]
    for ptype in ("regulator", "cathartic", "neutral"):
        fr = []
        for recs in A["learned"]:
            sel = [r for r in recs if r["day"] >= n_days - 15 and r["ptype"] == ptype and r["true_quadrant"].startswith("lo_v")]
            if sel:
                fr.append(np.mean([r["strategy"] == "match" for r in sel]))
        m, ci = mean_ci(fr)
        lines.append(f"| {ptype} | {m:.2f} ± {ci:.2f} |")
    lines.append("")

    # ---- Fig 3 + table B: sensing ablation
    B = R["B"]
    order = ["context_only", "typing", "quiz_fixed5", "quiz_adaptive", "fused"]
    names = {"context_only": "Context only", "typing": "Typing + context", "quiz_fixed5": "Fixed 5-question quiz", "quiz_adaptive": "Adaptive quiz", "fused": "Everything fused"}
    stats = {k: {"err": [], "q": [], "gain": []} for k in order}
    for k in order:
        for recs in B[k]:
            stats[k]["err"].append(np.mean([r["est_err"] for r in recs if r["day"] >= 5]))
            stats[k]["q"].append(np.mean([r["n_questions"] for r in recs if r["day"] >= 5]))
            stats[k]["gain"].append(np.mean([r["gain"] for r in recs if r["day"] >= n_days - 10]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, key, title in zip(axes, ("err", "q", "gain"), ("Mood estimate error (lower is better)", "Questions asked per session", "Wellbeing gain, last 10 days")):
        ms = [mean_ci(stats[k][key]) for k in order]
        ax.barh([names[k] for k in order], [m for m, _ in ms], xerr=[c for _, c in ms], color=[TEAL if k == "fused" else GREY for k in order], capsize=3)
        ax.set_title(title, fontsize=9.5, loc="left"); ax.invert_yaxis()
        if key != "err":
            ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(OUT / "fig3_sensing_ablation.png"); plt.close(fig)
    lines += ["## B. Which sensors earn their place?", "", "| Sensing | Mood error (VA distance) | Questions / session | Gain, last 10 days |", "|---|---|---|---|"]
    for k in order:
        e, ec = mean_ci(stats[k]["err"]); q, qc = mean_ci(stats[k]["q"]); g, gc = mean_ci(stats[k]["gain"])
        lines.append(f"| {names[k]} | {e:.3f} ± {ec:.3f} | {q:.2f} ± {qc:.2f} | {g:+.3f} ± {gc:.3f} |")
    lines.append("")

    # ---- Fig 4 + table C/D: cold start and similar-mood signal
    C = {"with prior": A["learned"], "no prior": R["C"]["no_prior"]}
    first = slice(0, 7)
    lines += ["## C. Cold start: does borrowing the population's experience help new users?", "", "| Variant | Gain, first 7 days | Gain, last 10 days |", "|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, runs, col in (("With population prior", C["with prior"], TEAL), ("Without", C["no prior"], GREY)):
        g = per_day(runs, "gain", n_days)
        ax.plot(range(1, n_days + 1), smooth(g.mean(axis=0)), color=col, lw=2, label=name)
        f, fc = mean_ci(g[:, first].mean(axis=1)); l, lc = mean_ci(g[:, last].mean(axis=1))
        lines.append(f"| {name} | {f:+.3f} ± {fc:.3f} | {l:+.3f} ± {lc:.3f} |")
    ax.set_xlabel("Day of use"); ax.set_ylabel("Mean true wellbeing gain"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cold start", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig4_cold_start.png"); plt.close(fig)
    lines += ["", "## D. Does the 'people feeling like you' signal help?", "", "| Variant | Gain, all days | Gain, last 10 days |", "|---|---|---|"]
    for name, runs in (("With similar-mood boost", A["learned"]), ("Without", R["D"]["no_collab"])):
        g = per_day(runs, "gain", n_days)
        a_, ac = mean_ci(g.mean(axis=1)); l, lc = mean_ci(g[:, last].mean(axis=1))
        lines.append(f"| {name} | {a_:+.3f} ± {ac:.3f} | {l:+.3f} ± {lc:.3f} |")
    (OUT / "sim_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
