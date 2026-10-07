"""Figures and tables from reports/bench_e{1,2,3}_*.json -> reports/bench/."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AMBER, GREY, INK, TEAL, WINE
from .config import ROOT

R = ROOT / "reports"
OUT = R / "bench"
STYLE = {"exact": (INK, "o"), "ivfflat": (AMBER, "s"), "hnsw": (TEAL, "^"), "hnsw+iterative": ("#7FB3B7", "v")}


def load(prefix):
    f = sorted(glob.glob(str(R / f"bench_{prefix}_*.json")))
    return json.loads(open(f[-1]).read()) if f else None


def pname(p):
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in p.items() if k.split('.')[-1] in ("probes", "ef_search"))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    L = ["# Systems benchmark results", ""]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.15})

    e1 = load("e1")
    if e1:
        n = e1[0]["n"]
        fig, ax = plt.subplots(figsize=(6.2, 4))
        for name in ("ivfflat", "hnsw"):
            pts = [r for r in e1 if r["index"] == name]
            ax.plot([p["p50_ms"] for p in pts], [p["recall"] for p in pts], color=STYLE[name][0], marker=STYLE[name][1], lw=1.8, label=name)
        ex = next(r for r in e1 if r["index"] == "exact")
        ax.scatter([ex["p50_ms"]], [1.0], color=INK, zorder=5, label="exact scan")
        ax.set_xscale("log"); ax.set_xlabel("Median query latency (ms, log)"); ax.set_ylabel("Recall@10")
        ax.set_title(f"Mood-to-movie kNN, {n:,} vectors, 384-d", loc="left", fontsize=10); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(OUT / "fig8_recall_latency.png"); plt.close(fig)
        L += [f"## E1. Unfiltered kNN ({n:,} vectors)", "", "| Index | Setting | Recall@10 | p50 ms | p95 ms | Build s | Size MB |", "|---|---|---|---|---|---|---|"]
        for r in e1:
            L.append(f"| {r['index']} | {pname(r['params']) or '-'} | {r['recall']:.3f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['build_s']:.0f} | {r['index_mb']:.0f} |")
        L.append("")

    e2 = load("e2")
    if e2:
        sels = sorted({r["selectivity"] for r in e2}, reverse=True)
        L += ["## E2. Filtered kNN (predicate keeps the given fraction of rows)", "",
              "| Index | Setting | " + " | ".join(f"recall @ {s:g}" for s in sels) + " | short results @ " + f"{sels[-1]:g}" + " |", "|---|---|" + "---|" * (len(sels) + 1)]
        configs = []
        for r in e2:
            key = (r["index"], pname(r["params"]))
            if key not in configs:
                configs.append(key)
        fig, ax = plt.subplots(figsize=(6.6, 4))
        for key in configs:
            rs = {r["selectivity"]: r for r in e2 if (r["index"], pname(r["params"])) == key}
            L.append(f"| {key[0]} | {key[1] or '-'} | " + " | ".join(f"{rs[s]['recall']:.3f}" for s in sels) + f" | {rs[sels[-1]]['short_results']}/150 |")
            if key[1] in ("", "ef_search=40", "probes=20", "ef_search=160") or key[0] == "exact":
                col, mk = STYLE.get(key[0], (GREY, "o"))
                ax.plot(sels, [rs[s]["recall"] for s in sels], color=col, marker=mk, lw=1.6, label=f"{key[0]} {key[1]}".strip(), alpha=0.9)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Filter selectivity (fraction of rows kept; smaller = more selective)"); ax.set_ylabel("Recall@10 vs exact filtered")
        ax.set_title("Filtered search quality", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "fig9_filtered.png"); plt.close(fig)
        L.append("")

    e3 = load("e3")
    if e3:
        variants = []
        for r in e3:
            if r["variant"] not in variants:
                variants.append(r["variant"])
        n = e3[0]["n"]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        pal = {"hnsw": TEAL, "hnsw+vacuum": "#7FB3B7", "ivfflat": AMBER, "ivfflat+reindex": WINE}
        for v in variants:
            pts = [r for r in e3 if r["variant"] == v]
            axes[0].plot([p["cumulative_churn"] * 100 for p in pts], [p["recall"] for p in pts], marker="o", color=pal.get(v, GREY), lw=1.7, label=v)
            axes[1].plot([p["cumulative_churn"] * 100 for p in pts], [p["p50_ms"] for p in pts], marker="o", color=pal.get(v, GREY), lw=1.7, label=v)
            axes[2].plot([p["cumulative_churn"] * 100 for p in pts], [p["index_mb"] for p in pts], marker="o", color=pal.get(v, GREY), lw=1.7, label=v)
        axes[0].set_xlabel("Cumulative % of vectors replaced"); axes[0].set_ylabel("Recall@10")
        axes[0].set_title("Recall (not comparable across rounds:\nthe data drift toward the queries)", loc="left", fontsize=9); axes[0].legend(frameon=False, fontsize=8)
        axes[1].set_xlabel("Cumulative % of vectors replaced"); axes[1].set_ylabel("Median query latency (ms)"); axes[1].set_yscale("log"); axes[1].set_title("Latency", loc="left", fontsize=9)
        axes[2].set_xlabel("Cumulative % of vectors replaced"); axes[2].set_ylabel("Index size (MB)"); axes[2].set_title("Index growth", loc="left", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "fig10_churn.png"); plt.close(fig)
        L += [f"## E3. Update churn ({n:,} vectors, 10% replaced per round, drifting population)", "",
              "| Variant | Round 0 recall | Final recall | Final p50 ms | Update rows/s (mean) | Maintenance s/round | Index MB start → end |", "|---|---|---|---|---|---|---|"]
        for v in variants:
            pts = [r for r in e3 if r["variant"] == v]
            ups = [p["update_rows_per_s"] for p in pts if p["update_rows_per_s"]]
            mt = [p["maintain_s"] for p in pts if p["round"] > 0]
            L.append(f"| {v} | {pts[0]['recall']:.3f} | {pts[-1]['recall']:.3f} | {pts[-1]['p50_ms']:.2f} | {np.mean(ups):,.0f} | {np.mean(mt):.1f} | {pts[0]['index_mb']:.0f} → {pts[-1]['index_mb']:.0f} |")
        L.append("")
    (OUT / "bench_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
