"""Regulation policy: per user and mood quadrant, learn whether to MATCH the mood or REGULATE it.

Background: Mood Management Theory (Zillmann) says people use media to feel better, but the best
way differs by person and state - some want content that mirrors a feeling (catharsis), some want
the opposite. We treat that as a two-armed bandit per (user, quadrant) and learn it from outcomes.

  match     target mood = the current mood
  regulate  target mood = pleasant valence, arousal pushed toward the opposite pole
  reward    in [0, 1], from how the user actually felt AFTER watching (see `reward`)

Arms are Beta posteriors; each decision is a Thompson-sampling draw. Because rewards are
fractional we update with alpha += r, beta += 1 - r.
"""
from dataclasses import dataclass

import numpy as np

STRATEGIES = ("match", "regulate")
QUADRANTS = ("hi_v_hi_a", "hi_v_lo_a", "lo_v_hi_a", "lo_v_lo_a")
PRIOR_STRENGTH = 12.0       # pseudo-observations the population prior is worth for a new user (tuned on a throwaway seed)
POOL_WEIGHT = 1.0           # how much of a user's evidence from OTHER mood quadrants counts toward this one (tuned on a throwaway seed)


def quadrant(v: float, a: float) -> str:
    return f"{'hi' if v >= 0 else 'lo'}_v_{'hi' if a >= 0 else 'lo'}_a"


def target_for(strategy: str, v: float, a: float) -> tuple[float, float]:
    """Target point in the valence-arousal square (both axes in [-1, 1])."""
    if strategy == "match":
        return v, a
    if strategy == "regulate":
        tv = max(v, 0.3) if v >= 0 else 0.4 + 0.5 * abs(v)     # never aim below pleasant
        ta = -0.6 * a                                           # pull arousal to the opposite pole
        return float(np.clip(tv, -1, 1)), float(np.clip(ta, -1, 1))
    raise ValueError(strategy)


def wellbeing_gain(before: tuple[float, float], after: tuple[float, float]) -> float:
    """Recovery toward a pleasant, balanced state: valence up, arousal less extreme. In [-1, 1].

    This is an explicit design assumption (stated in the report): 'better' means more pleasant and
    less extreme. It is the only place a notion of 'good outcome' enters the system.
    """
    dv = after[0] - before[0]
    da = abs(before[1]) - abs(after[1])
    return float(np.clip((dv + 0.5 * da) / 1.5, -1, 1))


def reward(before: tuple[float, float], after: tuple[float, float], liked: bool | None) -> float:
    """Blend felt outcome (60%) with explicit enjoyment (40%) into [0, 1]."""
    gain01 = (wellbeing_gain(before, after) + 1) / 2
    if liked is None:
        return gain01
    return 0.6 * gain01 + 0.4 * (1.0 if liked else 0.0)


@dataclass
class Arm:
    alpha: float = 1.0
    beta: float = 1.0
    obs: int = 0                # real observations in THIS quadrant; alpha/beta also hold prior and pooled evidence
    r_sum: float = 0.0          # sum of those observations' rewards

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))

    def update(self, r: float) -> None:
        self.alpha += r
        self.beta += 1.0 - r
        self.obs += 1
        self.r_sum += r

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def arm_from_prior(pop_mean: float | None) -> Arm:
    """Start a new user's arm from the population's average success rate in that quadrant."""
    if pop_mean is None:
        return Arm()
    m = float(np.clip(pop_mean, 0.05, 0.95))
    return Arm(1.0 + PRIOR_STRENGTH * m, 1.0 + PRIOR_STRENGTH * (1 - m))


def build_arm(pop_mean: float | None, own_n: int, own_r: float, pool_n: int = 0, pool_r: float = 0.0,
              pool_weight: float = POOL_WEIGHT) -> Arm:
    """Posterior for one (user, quadrant, strategy): population prior + own evidence + pooled evidence.

    Partial pooling: people's response style tends to be a stable trait, so what worked for this user
    in other moods is (down-weighted) evidence about what works now. With ~5 observations per arm in a
    few weeks of use, borrowing strength this way is what makes per-person learning feasible at all.
    """
    base = arm_from_prior(pop_mean)
    a = base.alpha + own_r + pool_weight * pool_r
    b = base.beta + (own_n - own_r) + pool_weight * (pool_n - pool_r)
    return Arm(a, b, own_n, own_r)


SWITCH_MARGIN = 0.08        # evidence margin required to leave DEFAULT_STRATEGY (tuned on a throwaway seed)
DEFAULT_STRATEGY = "regulate"


def choose(arms: dict[str, Arm], rng: np.random.Generator, margin: float | None = None) -> tuple[str, dict[str, float]]:
    """Thompson sampling with a default: leave DEFAULT_STRATEGY only if the other arm's draw beats it by `margin`.

    A plain Thompson draw explores wastefully when one strategy is better for almost everyone. Requiring
    a margin makes the default sticky until a person's own evidence is clearly different.
    """
    m = SWITCH_MARGIN if margin is None else margin
    draws = {s: arms[s].sample(rng) for s in STRATEGIES}
    other = next(s for s in STRATEGIES if s != DEFAULT_STRATEGY)
    pick = other if draws[other] > draws[DEFAULT_STRATEGY] + m else DEFAULT_STRATEGY
    return pick, draws
