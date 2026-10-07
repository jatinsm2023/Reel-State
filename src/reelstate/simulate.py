"""Synthetic users for evaluating the full loop without waiting weeks for real data.

Each synthetic person has HIDDEN traits the system never sees:
  * a baseline mood and day-to-day mood swings
  * a personal typing style, and how weakly typing speed/errors track their mood
  * a quiz response bias (some people rate everything high)
  * per mood quadrant, a response TYPE to what they watch:
        regulator  - a change of pace lifts them, but mood-matching sad content deepens the low
        cathartic  - mood-matching content genuinely helps (they 'need a good cry')
        neutral    - they are simply pulled toward whatever mood the film has
They are driven through the SAME service/recommend code the web app uses, with explicit timestamps,
so weeks of use run in minutes. This shows the MECHANISM works under stated assumptions; it is
evidence about the system, not about real people (that is what the pilot is for).

Common random numbers: the same seed produces the same people and the same mood sequences in every
condition, so conditions differ only in what the system did.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from . import fusion, policy, quiz, service
from .quiz import GRID, category_probs, to_va
from .quiz_bank import BY_ID
from .recommend import Filters, recommend

TYPES = ("regulator", "cathartic", "neutral")
TYPE_PROBS = (0.45, 0.35, 0.20)
SCENARIOS = ("regulate_dominant", "heterogeneous", "strong")
# regulate_dominant : v1. Response types are drawn independently per mood quadrant and the catharsis
#                     effect is small, so changing the mood beats matching it for EVERYONE. Used to
#                     measure what personalisation COSTS when there is nothing to personalise.
# heterogeneous     : response style is a stable person-level trait (people differ in whether they
#                     'engage with' or 'get away from' a low mood) with some per-mood variation, and a
#                     cathartic person genuinely resists being cheered up while benefiting from
#                     mood-matching content. The effect size is an ASSUMPTION; the point is to test
#                     whether the system can find heterogeneity if it exists, and how fast.
# strong            : as 'heterogeneous', but a cathartic person also finds incongruent cheerfulness
#                     dismissive when low (a 'reactance' penalty), so regulating actively backfires.
#                     Defined to bracket the plausible range of effect sizes from above.
START = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


@dataclass
class Person:
    idx: int
    base: np.ndarray                 # baseline theta (v, a)
    swing: float                     # day-to-day sd
    bias: np.ndarray                 # questionnaire answer bias (theta units)
    typing_offset: float             # personal log-speed offset
    types: dict[str, str]
    seed: int
    scenario: str = "regulate_dominant"

    def session_theta(self, day: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1000 + day)
        return self.base + rng.normal(0, self.swing, 2)


def make_population(n: int, seed: int, scenario: str = "regulate_dominant") -> list[Person]:
    rng = np.random.default_rng(seed)
    people = []
    for i in range(n):
        if scenario in ("heterogeneous", "strong"):
            trait = str(rng.choice(TYPES, p=(0.45, 0.40, 0.15)))
            types = {q: (trait if rng.random() < 0.8 else str(rng.choice(TYPES, p=TYPE_PROBS))) for q in policy.QUADRANTS}
        else:
            types = {q: str(rng.choice(TYPES, p=TYPE_PROBS)) for q in policy.QUADRANTS}
        people.append(Person(i, rng.normal(0, 0.5, 2), float(rng.uniform(0.6, 1.0)), rng.normal(0, 0.3, 2),
                             float(rng.normal(0, 0.25)), types, seed * 100 + i, scenario))
    return people


# --------------------------------------------------------------- behaviours
def answer_quiz(rng, person: Person, theta: np.ndarray, item) -> int:
    t = theta[0 if item.axis == "valence" else 1] + person.bias[0 if item.axis == "valence" else 1]
    p = category_probs(item)[np.abs(GRID - t).argmin()]
    return int(rng.choice(5, p=p / p.sum()))


def typing_events(rng, person: Person, theta: np.ndarray, n: int = 90) -> list[tuple[float, str]]:
    median = np.exp(np.log(230.0) + person.typing_offset - 0.10 * theta[1])
    back_p = float(np.clip(0.08 - 0.015 * theta[0], 0.01, 0.3))
    pause_p = float(np.clip(0.07 - 0.010 * theta[0], 0.01, 0.3))
    ev = [(0.0, "char")]
    for _ in range(n):
        gap = rng.lognormal(np.log(median), 0.4)
        if rng.random() < pause_p:
            gap = rng.uniform(1200, 4000)
        cls = "backspace" if rng.random() < back_p else ("space" if rng.random() < 0.15 else "char")
        ev.append((float(gap), cls))
    return ev


def outcome(rng, person: Person, before: np.ndarray, movie: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True felt mood after watching, and the (noisy) self-report. All in valence-arousal space."""
    q = policy.quadrant(*before)
    resonance = float(np.exp(-np.sum((movie - before) ** 2) / (2 * 0.5 ** 2)))     # how congruent film and mood are
    t = person.types[q]
    if person.scenario in ("heterogeneous", "strong") and before[0] < 0 and t == "cathartic":
        strong = person.scenario == "strong"
        pull = (0.05 if strong else 0.15) + (0.45 if strong else 0.35) * resonance   # resists being cheered up
        after = before + pull * (movie - before)
        after[0] += 0.45 * resonance                 # catharsis from congruent content
        if strong:
            after[0] -= 0.20 * (1.0 - resonance)     # reactance: incongruent cheer feels dismissive
    else:
        after = before + 0.5 * (movie - before)
        if before[0] < 0:
            if t == "cathartic":
                after[0] += 0.35 * resonance
            elif t == "regulator":
                after[0] -= 0.25 * resonance         # rumination when the content mirrors a low mood
    after = np.clip(after + rng.normal(0, 0.06, 2), -1, 1)
    reported = np.clip(after + rng.normal(0, 0.10, 2), -1, 1)
    return after, reported


# ------------------------------------------------------------------ runner
SENSING = {
    "context_only":  dict(typing=False, quiz=False),
    "typing":        dict(typing=True,  quiz=False),
    "quiz_adaptive": dict(typing=False, quiz=True),
    "quiz_fixed5":   dict(typing=False, quiz="fixed"),
    "fused":         dict(typing=True,  quiz=True),
}
FIXED5 = [1, 11, 2, 12, 10]                 # one fixed short form, alternating axes


def run_condition(conn, *, policy_mode: str, sensing: str, n_users: int, n_days: int, seed: int,
                  slate: int = 1, progress=None, use_collab: bool = True, population_prior: bool = True,
                  pooling: bool = True, scenario: str = "regulate_dominant") -> dict:
    """policy_mode: learned | always_match | always_regulate | random.  Returns per-session records."""
    service.POPULATION_PRIOR = population_prior
    service.POOLING = pooling
    people = make_population(n_users, seed, scenario)
    uids = [service.create_user(conn, f"sim{p.idx}", synthetic=True) for p in people]
    cfg = SENSING[sensing]
    rec_rng = np.random.default_rng(seed + 7)
    records = []
    for day in range(n_days):
        now = START + timedelta(days=day)
        for p, uid in zip(people, uids):
            rng = np.random.default_rng(p.seed * 7919 + day)           # same draws across conditions
            theta = p.session_theta(day)
            true_v, true_a = to_va(theta[0]), to_va(theta[1])
            t = now + timedelta(minutes=int(rng.integers(0, 90)))
            service.observe_context(conn, uid, t)
            n_questions = 0
            if cfg["typing"]:
                service.observe_typing(conn, uid, typing_events(rng, p, theta), t + timedelta(seconds=30))
            if cfg["quiz"] == "fixed":
                answers = [{"item_id": i, "answer": answer_quiz(rng, p, theta, BY_ID[i]), "latency_ms": 2500} for i in FIXED5]
                service.quiz_commit(conn, uid, answers, t + timedelta(minutes=1))
                n_questions = len(answers)
            elif cfg["quiz"]:
                answers = []
                while True:
                    r = service.quiz_next(conn, uid, answers, t + timedelta(minutes=1))
                    if r["done"]:
                        break
                    it = BY_ID[r["item"]["item_id"]]
                    answers.append({"item_id": it.item_id, "answer": answer_quiz(rng, p, theta, it), "latency_ms": 2500})
                if answers:
                    service.quiz_commit(conn, uid, answers, t + timedelta(minutes=1))
                n_questions = len(answers)
            mood, _ = service.current_mood(conn, uid, t + timedelta(minutes=2))
            est = np.array([to_va(mood.v), to_va(mood.a)])
            force = {"always_match": "match", "always_regulate": "regulate"}.get(policy_mode)
            if policy_mode == "random":
                force = str(rec_rng.choice(policy.STRATEGIES))
            res = recommend(conn, uid, t + timedelta(minutes=2), Filters(), rng=rec_rng, force_strategy=force, slate=slate, use_collab=use_collab)
            top = res["slate"][0]
            movie = np.array(top["explanation"]["movie_affect"])
            before = np.array([true_v, true_a])
            after, reported = outcome(rng, p, before, movie)
            gain = policy.wellbeing_gain(tuple(before), tuple(after))
            liked = bool(rng.random() < 1 / (1 + np.exp(-(4 * gain - 0.3))))
            ck = service.record_checkin(conn, top["rec_id"], float(reported[0]), float(reported[1]), liked,
                                        t + timedelta(hours=2.5))
            records.append({
                "user": p.idx, "day": day, "strategy": res["strategy"], "true_quadrant": policy.quadrant(*before),
                "ptype": p.types[policy.quadrant(*before)], "gain": gain, "reward_seen": ck["reward"],
                "est_err": float(np.linalg.norm(est - before)), "n_questions": n_questions,
            })
        conn.commit()
        if progress:
            progress(day + 1, n_days)
    cleanup(conn, uids)
    service.POPULATION_PRIOR = True
    service.POOLING = True
    return {"records": records}


def cleanup(conn, user_ids: list[int]) -> None:
    for tbl in ("checkins", "recommendations", "policy_state", "quiz_responses", "typing_baseline", "mood_events", "mood_state", "mood_current"):
        conn.execute(f"DELETE FROM {tbl} WHERE user_id = ANY(%s)", (user_ids,))
    conn.execute("DELETE FROM users WHERE user_id = ANY(%s)", (user_ids,))
    conn.commit()
