import numpy as np

from reelstate import policy as P


def test_quadrants():
    assert P.quadrant(0.5, 0.5) == "hi_v_hi_a"
    assert P.quadrant(-0.5, -0.1) == "lo_v_lo_a"
    assert P.quadrant(0.1, -0.1) == "hi_v_lo_a"


def test_match_targets_current_mood_and_regulate_lifts_valence_and_opposes_arousal():
    assert P.target_for("match", -0.6, 0.7) == (-0.6, 0.7)
    tv, ta = P.target_for("regulate", -0.6, 0.7)
    assert tv > 0 and ta < 0
    tv, ta = P.target_for("regulate", 0.8, -0.5)
    assert tv >= 0.8 and ta > 0                      # happy and sluggish: keep the high, add energy


def test_reward_orders_outcomes():
    b = (-0.6, 0.8)
    better = P.reward(b, (0.2, 0.1), liked=True)
    same = P.reward(b, b, liked=None)
    worse = P.reward(b, (-0.9, 0.95), liked=False)
    assert better > same > worse
    assert 0 <= worse <= better <= 1


def test_thompson_sampling_finds_the_better_strategy():
    rng = np.random.default_rng(3)
    arms = {"match": P.Arm(), "regulate": P.Arm()}
    truth = {"match": 0.35, "regulate": 0.70}
    picks = []
    for _ in range(200):
        s, _ = P.choose(arms, rng)
        picks.append(s)
        arms[s].update(float(rng.random() < truth[s]))
    assert picks[-60:].count("regulate") / 60 > 0.85


def test_population_prior_gives_new_user_a_head_start_but_data_overrides_it():
    rng = np.random.default_rng(5)
    arm = P.arm_from_prior(0.8)
    assert arm.mean > 0.6
    prior_mean = arm.mean
    for _ in range(60):
        arm.update(0.0)
    assert arm.mean < prior_mean / 3          # enough contrary evidence overrides even a strong prior


def test_pooling_lets_other_moods_inform_this_one_but_less_than_own_data():
    flat = P.build_arm(None, 0, 0.0)
    pooled = P.build_arm(None, 0, 0.0, pool_n=10, pool_r=9.0)          # great results in OTHER quadrants
    assert pooled.mean > flat.mean + 0.2
    half = P.build_arm(None, 0, 0.0, pool_n=10, pool_r=9.0, pool_weight=0.5)
    assert flat.mean < half.mean < pooled.mean                         # the weight controls how much is borrowed
    assert P.build_arm(None, 10, 9.0).mean >= pooled.mean              # own evidence counts at least as much
    assert pooled.obs == 0                                             # pooled evidence is not 'observations here'
    own_bad = P.build_arm(None, 40, 4.0, pool_n=10, pool_r=9.0)
    assert own_bad.mean < 0.3                                          # enough own data overrides the pool
