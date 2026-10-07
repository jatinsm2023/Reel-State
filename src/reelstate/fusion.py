"""Mood fusion: one Gaussian belief per axis, in theta (z-score) units.

Mood is a hidden state that drifts and is observed through several noisy sensors. We model each
axis as an Ornstein-Uhlenbeck process (mean-reverting random walk) and fuse observations by
precision weighting - the scalar Kalman update. Consequences that the rest of the system uses:
  * confidence decays with time since the last observation, so a stale mood triggers a question
  * a precise source (quiz, post-watch check-in) outweighs weak ones (typing, context)
  * every source is optional; the state is simply whatever evidence is available
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, log, pi, exp

HALF_LIFE_H = 6.0
LAMBDA = log(2) / HALF_LIFE_H
STATIONARY_VAR = 1.0         # long-run uncertainty about a stranger's mood on the theta scale
SD_ASK = 0.75                # above this posterior sd on either axis, the app should ask a question


@dataclass(frozen=True)
class Mood:
    v: float = 0.0
    a: float = 0.0
    var_v: float = STATIONARY_VAR
    var_a: float = STATIONARY_VAR

    def needs_question(self) -> bool:
        return self.var_v ** 0.5 > SD_ASK or self.var_a ** 0.5 > SD_ASK


def drift(mood: Mood, hours: float, baseline: tuple[float, float] = (0.0, 0.0)) -> Mood:
    """Advance the OU process by `hours`: mean relaxes to the user's baseline, variance relaxes to stationary."""
    if hours <= 0:
        return mood
    k = exp(-LAMBDA * hours)
    k2 = exp(-2 * LAMBDA * hours)
    return Mood(
        v=baseline[0] + (mood.v - baseline[0]) * k,
        a=baseline[1] + (mood.a - baseline[1]) * k,
        var_v=STATIONARY_VAR + (mood.var_v - STATIONARY_VAR) * k2,
        var_a=STATIONARY_VAR + (mood.var_a - STATIONARY_VAR) * k2,
    )


def _fuse_axis(mean: float, var: float, obs: float | None, obs_var: float | None) -> tuple[float, float]:
    if obs is None or obs_var is None:
        return mean, var
    prec = 1.0 / var + 1.0 / obs_var
    return (mean / var + obs / obs_var) / prec, 1.0 / prec


def fuse(mood: Mood, v_obs=None, a_obs=None, var_v_obs=None, var_a_obs=None) -> Mood:
    v, vv = _fuse_axis(mood.v, mood.var_v, v_obs, var_v_obs)
    a, va = _fuse_axis(mood.a, mood.var_a, a_obs, var_a_obs)
    return Mood(v, a, vv, va)


def context_observation(ts: datetime) -> dict:
    """Very weak circadian/weekly prior: arousal peaks mid-afternoon, weekends feel slightly better."""
    h = ts.hour + ts.minute / 60
    return {
        "arousal": 0.5 * cos(2 * pi * (h - 16.0) / 24.0),
        "valence": 0.15 if ts.weekday() >= 5 else 0.0,
        "var_arousal": 3.0,
        "var_valence": 4.0,
        "features": {"hour": ts.hour, "weekday": ts.weekday()},
    }


def hours_between(t0: datetime, t1: datetime) -> float:
    return max((t1 - t0).total_seconds() / 3600.0, 0.0)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
