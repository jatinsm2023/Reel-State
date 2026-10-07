"""Application logic that ties sensing, fusion, the quiz and the policy to the database.

Every function takes an open psycopg connection and is explicit about time (`now`) so the
simulator can replay weeks of use in seconds with the same code the web app runs.
"""
import json
from datetime import datetime, timedelta

import numpy as np

from . import affect_space, fusion, policy, quiz
from .fusion import Mood
from .quiz_bank import BANK, BY_ID
from .typing_features import FEATURES, Baseline, extract, to_observation

QUIZ_REPEAT_COOLDOWN = timedelta(hours=24)
POPULATION_PRIOR = True        # ablation switch: new users start from other users' averages
POOLING = True                 # ablation switch: borrow a user's evidence across their other mood quadrants


# ------------------------------------------------------------------ users
def create_user(conn, name: str, household_id: int | None = None, synthetic: bool = False) -> int:
    return conn.execute(
        "INSERT INTO users (display_name, household_id, is_synthetic) VALUES (%s, %s, %s) RETURNING user_id",
        (name, household_id, synthetic),
    ).fetchone()[0]


# ------------------------------------------------------------------- mood
def user_baseline(conn, user_id: int) -> tuple[float, float]:
    """Long-run theta average of a user, shrunk toward 0 until there is enough history."""
    row = conn.execute(
        "SELECT avg(theta_v), avg(theta_a), count(*) FROM "
        "(SELECT theta_v, theta_a FROM mood_state WHERE user_id = %s ORDER BY ts DESC LIMIT 30) s",
        (user_id,),
    ).fetchone()
    if not row[2]:
        return 0.0, 0.0
    w = row[2] / (row[2] + 10.0)
    return w * row[0], w * row[1]


def current_mood(conn, user_id: int, now: datetime) -> tuple[Mood, datetime | None]:
    """Latest stored belief, advanced to `now` (mean reverts, uncertainty grows)."""
    row = conn.execute(
        "SELECT ts, theta_v, theta_a, var_v, var_a FROM mood_state WHERE user_id = %s AND ts <= %s "
        "ORDER BY ts DESC LIMIT 1",
        (user_id, now),
    ).fetchone()
    if row is None:
        return Mood(), None
    last = Mood(row[1], row[2], row[3], row[4])
    return fusion.drift(last, fusion.hours_between(row[0], now), user_baseline(conn, user_id)), row[0]


def save_state(conn, user_id: int, ts: datetime, mood: Mood) -> tuple[float, float]:
    v, a = quiz.to_va(mood.v), quiz.to_va(mood.a)
    conn.execute(
        "INSERT INTO mood_state (user_id, ts, valence, arousal, var_v, var_a, mood_emb, theta_v, theta_a) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (user_id, ts) DO UPDATE SET valence = EXCLUDED.valence, arousal = EXCLUDED.arousal, "
        "var_v = EXCLUDED.var_v, var_a = EXCLUDED.var_a, mood_emb = EXCLUDED.mood_emb, "
        "theta_v = EXCLUDED.theta_v, theta_a = EXCLUDED.theta_a",
        (user_id, ts, v, a, mood.var_v, mood.var_a, emb := affect_space.embed(v, a), mood.v, mood.a),
    )
    conn.execute(
        "INSERT INTO mood_current (user_id, ts, valence, arousal, var_v, var_a, mood_emb) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET ts = EXCLUDED.ts, valence = EXCLUDED.valence, "
        "arousal = EXCLUDED.arousal, var_v = EXCLUDED.var_v, var_a = EXCLUDED.var_a, mood_emb = EXCLUDED.mood_emb "
        "WHERE mood_current.ts <= EXCLUDED.ts",
        (user_id, ts, v, a, mood.var_v, mood.var_a, emb),
    )
    return v, a


def observe(conn, user_id: int, source: str, obs: dict, now: datetime, features: dict | None = None) -> Mood:
    """Fuse one sensor observation (theta units) into the user's mood and persist event + new state."""
    mood, _ = current_mood(conn, user_id, now)
    new = fusion.fuse(
        mood,
        v_obs=obs.get("valence"), a_obs=obs.get("arousal"),
        var_v_obs=obs.get("var_valence"), var_a_obs=obs.get("var_arousal"),
    )
    feats = dict(features or {})
    feats.update(var_valence=obs.get("var_valence"), var_arousal=obs.get("var_arousal"))
    conn.execute(
        "INSERT INTO mood_events (user_id, ts, source, features, valence_obs, arousal_obs, obs_var) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, now, source, json.dumps(feats), obs.get("valence"), obs.get("arousal"),
         min(x for x in (obs.get("var_valence"), obs.get("var_arousal")) if x is not None)),
    )
    save_state(conn, user_id, now, new)
    return new


def set_state_from_posterior(conn, user_id: int, source: str, mood: Mood, now: datetime, features: dict) -> None:
    """For sources whose output already includes the prior (the quiz): write the posterior as-is."""
    conn.execute(
        "INSERT INTO mood_events (user_id, ts, source, features, valence_obs, arousal_obs, obs_var) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, now, source, json.dumps(features), mood.v, mood.a, min(mood.var_v, mood.var_a)),
    )
    save_state(conn, user_id, now, mood)


def observe_context(conn, user_id: int, now: datetime) -> Mood:
    c = fusion.context_observation(now)
    return observe(conn, user_id, "context", c, now, c["features"])


# ------------------------------------------------------------------- typing
def load_baselines(conn, user_id: int) -> dict[str, Baseline]:
    out = {f: Baseline() for f in FEATURES}
    for f, n, mean, m2 in conn.execute("SELECT feature, n, mean, m2 FROM typing_baseline WHERE user_id = %s", (user_id,)):
        out[f] = Baseline(n, mean, m2)
    return out


def observe_typing(conn, user_id: int, events: list[tuple[float, str]], now: datetime) -> dict | None:
    feats = extract(events)
    if feats is None:
        return None
    baselines = load_baselines(conn, user_id)
    obs = to_observation(feats, baselines)
    mood = observe(conn, user_id, "typing", obs, now, {"values": feats.values, "z": obs["z"], "n_keys": feats.n_keys})
    for f in FEATURES:                                   # update the baseline AFTER scoring against it
        b = baselines[f]
        b.update(feats.values[f])
        conn.execute(
            "INSERT INTO typing_baseline (user_id, feature, n, mean, m2) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (user_id, feature) DO UPDATE SET n = EXCLUDED.n, mean = EXCLUDED.mean, m2 = EXCLUDED.m2",
            (user_id, f, b.n, b.mean, b.m2),
        )
    return {"features": feats.values, "mood": mood}


# --------------------------------------------------------------------- quiz
def _replay(conn, user_id: int, answers: list[dict], now: datetime) -> quiz.QuizState:
    mood, _ = current_mood(conn, user_id, now)
    st = quiz.QuizState.start(mood.v, mood.var_v, mood.a, mood.var_a)
    for ans in answers:
        st.record(BY_ID[ans["item_id"]], ans["answer"], ans.get("latency_ms"))
    return st


def quiz_next(conn, user_id: int, answers: list[dict], now: datetime, force: bool = False) -> dict:
    """Stateless adaptive step: replay answers so far on top of the current mood, pick what to ask."""
    mood, _ = current_mood(conn, user_id, now)
    if not answers and not force and not mood.needs_question():
        return {"done": True, "reason": "fresh_enough", "item": None}
    st = _replay(conn, user_id, answers, now)
    recent = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT item_id FROM quiz_responses WHERE user_id = %s AND ts > %s",
            (user_id, now - QUIZ_REPEAT_COOLDOWN),
        )
    }
    item, why = quiz.next_item(st, exclude=recent)
    return {
        "done": item is None,
        "reason": why,
        "item": None if item is None else {"item_id": item.item_id, "prompt": item.prompt, "options": list(item.options)},
        "estimate": {"valence": quiz.to_va(st.valence.mean), "arousal": quiz.to_va(st.arousal.mean),
                     "sd_valence": st.valence.var ** 0.5, "sd_arousal": st.arousal.var ** 0.5},
    }


def quiz_commit(conn, user_id: int, answers: list[dict], now: datetime) -> Mood:
    st = _replay(conn, user_id, answers, now)
    for ans in answers:
        conn.execute(
            "INSERT INTO quiz_responses (user_id, item_id, ts, answer, latency_ms) VALUES (%s, %s, %s, %s, %s)",
            (user_id, ans["item_id"], now, ans["answer"], ans.get("latency_ms")),
        )
    mood = Mood(st.valence.mean, st.arousal.mean, st.valence.var, st.arousal.var)
    set_state_from_posterior(conn, user_id, "quiz", mood, now, {"n_items": len(answers)})
    return mood


def seed_quiz_items(conn) -> None:
    for it in BANK:
        conn.execute(
            "INSERT INTO quiz_items (item_id, prompt, options, axis, discrimination, thresholds) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (item_id) DO UPDATE SET prompt = EXCLUDED.prompt, "
            "options = EXCLUDED.options, axis = EXCLUDED.axis, discrimination = EXCLUDED.discrimination, "
            "thresholds = EXCLUDED.thresholds",
            (it.item_id, it.prompt, json.dumps(it.options), it.axis, it.a, json.dumps(it.b)),
        )


# ------------------------------------------------------------------- policy
def population_means(conn, user_id: int, q: str) -> dict[str, float | None]:
    """Average reward of each strategy among OTHER users in this quadrant (same kind of user only)."""
    out = {s: None for s in policy.STRATEGIES}
    if not POPULATION_PRIOR:
        return out
    for s, m in conn.execute(
        "SELECT p.strategy, sum(p.r_sum) / sum(p.n_obs) FROM policy_state p JOIN users u USING (user_id) "
        "WHERE p.quadrant = %s AND p.user_id <> %s AND p.n_obs >= 2 "
        "AND u.is_synthetic = (SELECT is_synthetic FROM users WHERE user_id = %s) GROUP BY p.strategy",
        (q, user_id, user_id),
    ):
        out[s] = float(m)
    return out


def load_arms(conn, user_id: int, q: str) -> dict[str, policy.Arm]:
    own = {(qq, s): (n, float(r)) for qq, s, n, r in conn.execute(
        "SELECT quadrant, strategy, n_obs, r_sum FROM policy_state WHERE user_id = %s", (user_id,))}
    pop = population_means(conn, user_id, q)
    arms = {}
    for s in policy.STRATEGIES:
        n, r = own.get((q, s), (0, 0.0))
        pn = sum(v[0] for (qq, ss), v in own.items() if ss == s and qq != q)
        pr = sum(v[1] for (qq, ss), v in own.items() if ss == s and qq != q)
        arms[s] = policy.build_arm(pop[s], n, r, pn, pr, policy.POOL_WEIGHT if POOLING else 0.0)
    return arms


def save_arm(conn, user_id: int, q: str, strategy: str, arm: policy.Arm, now: datetime) -> None:
    conn.execute(
        "INSERT INTO policy_state (user_id, quadrant, strategy, alpha, beta, n_obs, r_sum, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (user_id, quadrant, strategy) DO UPDATE SET alpha = EXCLUDED.alpha, beta = EXCLUDED.beta, "
        "n_obs = EXCLUDED.n_obs, r_sum = EXCLUDED.r_sum, updated_at = EXCLUDED.updated_at",
        (user_id, q, strategy, arm.alpha, arm.beta, arm.obs, arm.r_sum, now),
    )


# ------------------------------------------------------------------ check-in
def record_checkin(conn, rec_id: int, valence_after: float, arousal_after: float, liked: bool | None, now: datetime) -> dict:
    """Close the loop: compute the felt outcome, update the policy, fold the new mood into the trajectory."""
    row = conn.execute(
        "SELECT r.user_id, r.strategy, s.valence, s.arousal FROM recommendations r "
        "JOIN mood_state s ON s.user_id = r.user_id AND s.ts = r.mood_ts WHERE r.rec_id = %s",
        (rec_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown recommendation {rec_id}")
    user_id, strategy, vb, ab = row
    r = policy.reward((vb, ab), (valence_after, arousal_after), liked)
    q = policy.quadrant(vb, ab)
    arm = load_arms(conn, user_id, q)[strategy]
    arm.update(r)
    save_arm(conn, user_id, q, strategy, arm, now)
    conn.execute(
        "INSERT INTO checkins (rec_id, user_id, ts, valence_after, arousal_after, reward, liked) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (rec_id, user_id, now, valence_after, arousal_after, r, liked),
    )
    th = lambda x: 1.5 * float(np.arctanh(np.clip(x, -0.98, 0.98)))
    observe(conn, user_id, "checkin",
            {"valence": th(valence_after), "arousal": th(arousal_after), "var_valence": 0.3, "var_arousal": 0.3},
            now, {"rec_id": rec_id})
    return {"reward": r, "quadrant": q, "strategy": strategy, "arm_mean": arm.mean}


# ------------------------------------------------------------ similar moods
def similar_moods(conn, user_id: int, now: datetime, k: int = 20, within: timedelta = timedelta(hours=48)) -> list[tuple[int, float]]:
    """Nearest users in the live mood index (cosine), most recent readings only. One ANN query."""
    rows = conn.execute(
        "SELECT o.user_id, 1 - (o.mood_emb <=> me.mood_emb) AS sim "
        "FROM mood_current me, LATERAL ("
        "  SELECT c.user_id, c.mood_emb FROM mood_current c "
        "  WHERE c.user_id <> me.user_id AND c.ts > %s AND c.ts <= %s "
        "  ORDER BY c.mood_emb <=> me.mood_emb LIMIT %s) o "
        "WHERE me.user_id = %s",
        (now - within, now, k, user_id),
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def collab_scores(conn, user_id: int, now: datetime, k: int = 20, min_reward: float = 0.6) -> dict[int, float]:
    """movie_id -> how well it worked for people whose mood right now resembles this user's (0..1)."""
    nbrs = similar_moods(conn, user_id, now, k)
    if not nbrs:
        return {}
    sims = dict(nbrs)
    rows = conn.execute(
        "SELECT r.user_id, r.movie_id, c.reward FROM checkins c JOIN recommendations r USING (rec_id) "
        "WHERE r.user_id = ANY(%s) AND c.reward >= %s AND c.ts <= %s",
        (list(sims), min_reward, now),
    ).fetchall()
    acc: dict[int, list[float]] = {}
    for uid, mid, rew in rows:
        acc.setdefault(mid, []).append(sims[uid] * rew)
    return {m: float(np.mean(v)) for m, v in acc.items()}
