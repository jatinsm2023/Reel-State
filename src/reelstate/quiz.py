"""Computerized adaptive mood quiz: graded response model (GRM) on two latent axes.

State per axis is a posterior over theta on a fixed grid (so no closed-form assumptions break).
After each answer the posterior is multiplied by the answer's likelihood. The next item is the one
with the largest expected reduction in posterior variance (MEPV). The quiz stops once both axes
are confident enough, or when asking more questions no longer pays.

theta lives on the same z-scale as the movie affect scores (see affect.py); to map to a point in
the valence-arousal square use `to_va` (tanh(theta / 1.5)), the same squashing movies use.
"""
from dataclasses import dataclass, field

import numpy as np

from .quiz_bank import BANK, Item

GRID = np.linspace(-4.0, 4.0, 161)
SD_TARGET = 0.60        # stop when both axes have posterior sd below this (theta units); see sweep in reports/
MAX_ITEMS = 6
MIN_GAIN = 0.01         # stop when the best question would cut variance by less than this


def to_va(theta: float) -> float:
    return float(np.tanh(theta / 1.5))


def category_probs(item: Item, a_mult: float = 1.0) -> np.ndarray:
    """P(response = k | theta) on GRID, shape (len(GRID), 5)."""
    a = item.a * a_mult
    cum = 1.0 / (1.0 + np.exp(-a * (GRID[:, None] - np.asarray(item.b)[None, :])))   # P(X >= k), k=1..4
    upper = np.concatenate([np.ones((len(GRID), 1)), cum], axis=1)
    lower = np.concatenate([cum, np.zeros((len(GRID), 1))], axis=1)
    return np.clip(upper - lower, 1e-12, 1.0)


def latency_multiplier(latency_ms: int | None) -> float:
    """Down-weight careless (very fast) or agonised (very slow) answers.

    Hesitation and speed-clicking are both weaker evidence about the true state than a normal answer.
    """
    if latency_ms is None:
        return 1.0
    if latency_ms < 600:
        return 0.5
    if latency_ms > 15000:
        return 0.8
    return 1.0


@dataclass
class AxisPosterior:
    weights: np.ndarray
    mean: float = 0.0
    var: float = 1.0

    @classmethod
    def from_prior(cls, mean: float = 0.0, var: float = 1.0) -> "AxisPosterior":
        w = np.exp(-0.5 * (GRID - mean) ** 2 / var)
        return cls._normalised(w)

    @classmethod
    def _normalised(cls, w: np.ndarray) -> "AxisPosterior":
        w = w / w.sum()
        m = float((w * GRID).sum())
        v = float((w * (GRID - m) ** 2).sum())
        return cls(w, m, v)

    def update(self, item: Item, answer: int, a_mult: float = 1.0) -> "AxisPosterior":
        return self._normalised(self.weights * category_probs(item, a_mult)[:, answer])

    def expected_var_after(self, item: Item) -> float:
        P = category_probs(item)                                  # (G, 5)
        joint = self.weights[:, None] * P                         # (G, 5)
        pk = joint.sum(axis=0)                                    # (5,)
        post = joint / pk
        m = (post * GRID[:, None]).sum(axis=0)
        v = (post * (GRID[:, None] - m) ** 2).sum(axis=0)
        return float((pk * v).sum())


@dataclass
class QuizState:
    valence: AxisPosterior
    arousal: AxisPosterior
    answered: list[int] = field(default_factory=list)

    @classmethod
    def start(cls, v_mean=0.0, v_var=1.0, a_mean=0.0, a_var=1.0) -> "QuizState":
        return cls(AxisPosterior.from_prior(v_mean, v_var), AxisPosterior.from_prior(a_mean, a_var))

    def axis(self, name: str) -> AxisPosterior:
        return self.valence if name == "valence" else self.arousal

    def record(self, item: Item, answer: int, latency_ms: int | None = None) -> None:
        mult = latency_multiplier(latency_ms)
        if item.axis == "valence":
            self.valence = self.valence.update(item, answer, mult)
        else:
            self.arousal = self.arousal.update(item, answer, mult)
        self.answered.append(item.item_id)


def next_item(
    state: QuizState,
    exclude: set[int] | None = None,
    bank=BANK,
    sd_target: float = SD_TARGET,
    max_items: int = MAX_ITEMS,
) -> tuple[Item | None, str]:
    """Pick the most informative unanswered item, or None with the reason for stopping."""
    if (
        len(state.answered) > 0
        and state.valence.var ** 0.5 <= sd_target
        and state.arousal.var ** 0.5 <= sd_target
    ):
        return None, "confident"
    if len(state.answered) >= max_items:
        return None, "max_items"
    skip = set(state.answered) | (exclude or set())
    best, best_gain = None, 0.0
    for it in bank:
        if it.item_id in skip:
            continue
        post = state.axis(it.axis)
        gain = post.var - post.expected_var_after(it)
        if gain > best_gain:
            best, best_gain = it, gain
    if best is None or best_gain < MIN_GAIN:
        return None, "no_gain"
    return best, "ask"
