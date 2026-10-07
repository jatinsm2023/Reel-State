"""Typing-dynamics sensing: derived features from keystroke timing, and their weak mood evidence.

PRIVACY: the client sends only (gap since previous key in ms, key class). It never sends which
character was typed. The raw event list is discarded after `extract`; only the derived features
below are stored (mood_events.features).

The mapping from features to mood is deliberately weak. Published keystroke-emotion work finds
typing speed tracks arousal and error/pause behaviour tracks negative affect, but accuracies are
modest (roughly 70-80% on coarse valence, much lower on fine-grained states). So every feature is
compared with the user's OWN baseline (z-score) and enters fusion as a high-variance observation.
The constants in OBS_MODEL are literature-informed priors to be calibrated on pilot data.
"""
from dataclasses import dataclass

import numpy as np

KEY_CLASSES = ("char", "space", "backspace", "enter", "other")
MIN_KEYS = 25
BURST_GAP_MS = 2000      # gaps longer than this are 'thinking', not typing speed
PAUSE_GAP_MS = 1000

FEATURES = ("log_iki_median", "log_iki_sd", "pause_rate", "backspace_rate", "chars_per_sec")

# population defaults used until a user has enough history of their own: (mean, sd)
POPULATION = {
    "log_iki_median": (np.log(230.0), 0.35),
    "log_iki_sd": (0.55, 0.18),
    "pause_rate": (0.08, 0.06),
    "backspace_rate": (0.08, 0.05),
    "chars_per_sec": (3.6, 1.2),
}

# evidence model: weights of z-scored features on the latent mood axes (theta units)
OBS_MODEL = {
    "arousal": {"weights": {"log_iki_median": -0.50, "chars_per_sec": 0.25, "log_iki_sd": 0.10}, "var": 1.8},
    "valence": {"weights": {"backspace_rate": -0.30, "pause_rate": -0.20}, "var": 2.5},
}


@dataclass
class TypingFeatures:
    n_keys: int
    duration_s: float
    values: dict[str, float]


def extract(events: list[tuple[float, str]]) -> TypingFeatures | None:
    """events: [(gap_ms_since_previous_key, key_class), ...]. Returns None if too little data."""
    if len(events) < MIN_KEYS:
        return None
    gaps = np.array([g for g, _ in events[1:]], dtype=float)
    classes = [c for _, c in events]
    gaps = gaps[(gaps > 5) & (gaps < 120_000)]
    if len(gaps) < MIN_KEYS - 1:
        return None
    burst = gaps[gaps < BURST_GAP_MS]
    if len(burst) < 10:
        return None
    n_back = sum(c == "backspace" for c in classes)
    n_chars = sum(c in ("char", "space") for c in classes)
    active_s = burst.sum() / 1000.0
    values = {
        "log_iki_median": float(np.log(np.median(burst))),
        "log_iki_sd": float(np.std(np.log(burst))),
        "pause_rate": float((gaps > PAUSE_GAP_MS).mean()),
        "backspace_rate": n_back / len(classes),
        "chars_per_sec": n_chars / max(active_s, 1e-6),
    }
    return TypingFeatures(len(events), float(gaps.sum() / 1000.0), values)


@dataclass
class Baseline:
    """Welford running mean/variance per feature for one user."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def stats(self, prior: tuple[float, float], prior_weight: int = 5) -> tuple[float, float]:
        """Shrink toward the population value until the user has enough sessions."""
        pm, ps = prior
        if self.n == 0:
            return pm, ps
        w = self.n / (self.n + prior_weight)
        mean = w * self.mean + (1 - w) * pm
        sd_user = (self.m2 / max(self.n - 1, 1)) ** 0.5 if self.n > 1 else ps
        sd = max(w * sd_user + (1 - w) * ps, 0.25 * ps)
        return mean, sd


def to_observation(feats: TypingFeatures, baselines: dict[str, Baseline]) -> dict:
    """Turn features into a (valence, arousal) observation in theta units with noise variances."""
    z = {}
    for f in FEATURES:
        mean, sd = baselines.get(f, Baseline()).stats(POPULATION[f])
        z[f] = (feats.values[f] - mean) / sd
    z = {k: float(np.clip(v, -3, 3)) for k, v in z.items()}
    out = {"z": z}
    for axis, spec in OBS_MODEL.items():
        out[axis] = sum(w * z[f] for f, w in spec["weights"].items())
        out[f"var_{axis}"] = spec["var"]
    return out
