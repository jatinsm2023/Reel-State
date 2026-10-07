"""Pilot study tooling: summarise real participants, export anonymised data, honour deletion requests.

  python -m reelstate.pilot summary            per-strategy outcomes from real (non-synthetic) users
  python -m reelstate.pilot export out.csv     one row per check-in, user ids replaced by P01, P02, ...
  python -m reelstate.pilot delete <user_id>   remove EVERYTHING stored about a participant

Outcome definition matches policy.wellbeing_gain: valence up and arousal less extreme, from the
participant's own before (system belief at recommendation time) and after (self-report) readings.
"""
import csv
import sys

import numpy as np

from . import policy
from .db import connect
from .simulate import cleanup

SQL = """
SELECT r.user_id, r.strategy, s.valence, s.arousal, c.valence_after, c.arousal_after, c.reward, c.liked, c.ts
FROM checkins c
JOIN recommendations r USING (rec_id)
JOIN mood_state s ON s.user_id = r.user_id AND s.ts = r.mood_ts
JOIN users u ON u.user_id = r.user_id
WHERE NOT u.is_synthetic
ORDER BY c.ts
"""


def rows(conn):
    out = []
    for uid, strat, vb, ab, va, aa, rew, liked, ts in conn.execute(SQL):
        out.append({"user_id": uid, "strategy": strat, "v_before": vb, "a_before": ab, "v_after": va, "a_after": aa,
                    "reward": rew, "liked": liked, "ts": ts, "gain": policy.wellbeing_gain((vb, ab), (va, aa)),
                    "quadrant": policy.quadrant(vb, ab)})
    return out


def summary() -> None:
    with connect() as conn:
        R = rows(conn)
    if not R:
        print("no real check-ins yet")
        return
    users = sorted({r["user_id"] for r in R})
    print(f"{len(R)} check-ins from {len(users)} participants\n")
    print(f"{'strategy':10s} {'n':>4s} {'mean gain':>10s} {'mean reward':>12s} {'liked':>7s}")
    for s in policy.STRATEGIES:
        sub = [r for r in R if r["strategy"] == s]
        if sub:
            liked = [r["liked"] for r in sub if r["liked"] is not None]
            print(f"{s:10s} {len(sub):4d} {np.mean([r['gain'] for r in sub]):+10.3f} {np.mean([r['reward'] for r in sub]):12.3f} {np.mean(liked) if liked else float('nan'):7.2f}")
    # within-person contrast: for each participant with both strategies, mean gain(regulate) - mean gain(match)
    diffs = []
    for u in users:
        m = [r["gain"] for r in R if r["user_id"] == u and r["strategy"] == "match"]
        g = [r["gain"] for r in R if r["user_id"] == u and r["strategy"] == "regulate"]
        if m and g:
            diffs.append(np.mean(g) - np.mean(m))
    if len(diffs) >= 3:
        d = np.array(diffs)
        print(f"\nwithin-person (regulate - match): mean {d.mean():+.3f}, sd {d.std(ddof=1):.3f}, n={len(d)}; "
              f"{(d > 0).sum()} favour regulate, {(d < 0).sum()} favour match")
        print("a large spread across people is the case for per-person policies")


def export(path: str) -> None:
    with connect() as conn:
        R = rows(conn)
    ids = {u: f"P{i + 1:02d}" for i, u in enumerate(sorted({r["user_id"] for r in R}))}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "ts", "strategy", "quadrant", "v_before", "a_before", "v_after", "a_after", "gain", "reward", "liked"])
        for r in R:
            w.writerow([ids[r["user_id"]], r["ts"].isoformat(), r["strategy"], r["quadrant"], round(r["v_before"], 3), round(r["a_before"], 3),
                        round(r["v_after"], 3), round(r["a_after"], 3), round(r["gain"], 3), round(r["reward"], 3), r["liked"]])
    print(f"wrote {len(R)} rows to {path} (names removed)")


def delete(user_id: int) -> None:
    with connect() as conn:
        name = conn.execute("SELECT display_name FROM users WHERE user_id = %s", (user_id,)).fetchone()
        if not name:
            print("no such user")
            return
        cleanup(conn, [user_id])
        print(f"deleted all data for user {user_id} ({name[0]})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    {"summary": summary, "export": lambda: export(sys.argv[2]), "delete": lambda: delete(int(sys.argv[2]))}[cmd]()
