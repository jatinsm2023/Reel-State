from datetime import datetime, timezone

import numpy as np

from reelstate import fusion
from reelstate.typing_features import Baseline, MIN_KEYS, extract, to_observation


def stream(rng, n, median_ms, backspace_p=0.05, pause_p=0.03):
    ev = [(0.0, "char")]
    for _ in range(n):
        gap = rng.lognormal(np.log(median_ms), 0.4)
        if rng.random() < pause_p:
            gap = rng.uniform(1200, 4000)
        cls = "backspace" if rng.random() < backspace_p else ("space" if rng.random() < 0.15 else "char")
        ev.append((gap, cls))
    return ev


def test_too_few_keys_is_rejected():
    assert extract([(100.0, "char")] * (MIN_KEYS - 1)) is None


def test_faster_typing_means_higher_arousal_evidence():
    rng = np.random.default_rng(0)
    fast = to_observation(extract(stream(rng, 200, 120)), {})
    slow = to_observation(extract(stream(rng, 200, 400)), {})
    assert fast["arousal"] > slow["arousal"]


def test_more_errors_and_pauses_mean_lower_valence_evidence():
    rng = np.random.default_rng(1)
    calm = to_observation(extract(stream(rng, 300, 220, backspace_p=0.02, pause_p=0.01)), {})
    rough = to_observation(extract(stream(rng, 300, 220, backspace_p=0.25, pause_p=0.15)), {})
    assert rough["valence"] < calm["valence"]


def test_baseline_shrinks_toward_population_then_personalises():
    b = Baseline()
    assert b.stats((10.0, 2.0)) == (10.0, 2.0)
    for x in [20.0] * 40:
        b.update(x)
    mean, _ = b.stats((10.0, 2.0))
    assert 18 < mean <= 20


def test_fusion_precision_weighting():
    m = fusion.Mood()
    strong = fusion.fuse(m, v_obs=1.0, var_v_obs=0.2)
    weak = fusion.fuse(m, v_obs=1.0, var_v_obs=3.0)
    assert strong.v > weak.v > 0
    assert strong.var_v < weak.var_v < m.var_v


def test_more_evidence_shrinks_uncertainty():
    m = fusion.Mood()
    for _ in range(3):
        m = fusion.fuse(m, a_obs=0.5, var_a_obs=1.5)
    assert m.var_a < 0.4


def test_drift_reverts_mean_and_restores_uncertainty():
    sure = fusion.Mood(v=1.5, a=-1.0, var_v=0.1, var_a=0.1)
    later = fusion.drift(sure, 12.0)
    assert abs(later.v) < abs(sure.v) and abs(later.a) < abs(sure.a)
    assert later.var_v > sure.var_v
    assert fusion.drift(sure, 500.0).var_v > 0.99


def test_stale_mood_triggers_a_question_fresh_one_does_not():
    fresh = fusion.Mood(v=0.4, a=0.2, var_v=0.2, var_a=0.2)
    assert not fresh.needs_question()
    assert fusion.drift(fresh, 24.0).needs_question()


def test_context_observation_is_weak_and_circadian():
    afternoon = fusion.context_observation(datetime(2026, 10, 7, 16, 0, tzinfo=timezone.utc))
    night = fusion.context_observation(datetime(2026, 10, 7, 4, 0, tzinfo=timezone.utc))
    assert afternoon["arousal"] > night["arousal"]
    assert afternoon["var_arousal"] >= 3.0
