import numpy as np

from reelstate.quiz import GRID, QuizState, category_probs, next_item, to_va
from reelstate.quiz_bank import BANK


def simulate_answer(rng, item, theta):
    p = category_probs(item)
    p_theta = p[np.abs(GRID - theta).argmin()]
    return int(rng.choice(5, p=p_theta / p_theta.sum()))


def run_session(rng, tv, ta, adaptive=True, n_random=None):
    st = QuizState.start()
    if adaptive:
        while True:
            it, _ = next_item(st)
            if it is None:
                break
            st.record(it, simulate_answer(rng, it, tv if it.axis == "valence" else ta))
    else:
        for it in rng.permutation(np.array(BANK, dtype=object))[:n_random]:
            st.record(it, simulate_answer(rng, it, tv if it.axis == "valence" else ta))
    return st


def test_category_probs_are_distributions():
    for it in BANK:
        p = category_probs(it)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        assert (p >= 0).all()


def test_higher_theta_shifts_answers_up():
    it = BANK[0]
    p = category_probs(it)
    lo, hi = p[np.abs(GRID + 2).argmin()], p[np.abs(GRID - 2).argmin()]
    assert (lo * np.arange(5)).sum() < (hi * np.arange(5)).sum()


def test_adaptive_quiz_beats_random_at_equal_length():
    rng = np.random.default_rng(7)
    n = 400
    err_cat, err_rand, lens = [], [], []
    for _ in range(n):
        tv, ta = rng.normal(size=2)
        s = run_session(rng, tv, ta, adaptive=True)
        lens.append(len(s.answered))
        err_cat += [(s.valence.mean - tv) ** 2, (s.arousal.mean - ta) ** 2]
        r = run_session(rng, tv, ta, adaptive=False, n_random=len(s.answered))
        err_rand += [(r.valence.mean - tv) ** 2, (r.arousal.mean - ta) ** 2]
    rmse_cat, rmse_rand = np.sqrt(np.mean(err_cat)), np.sqrt(np.mean(err_rand))
    print(f"\nmean items {np.mean(lens):.2f}  RMSE adaptive {rmse_cat:.3f}  random(same n) {rmse_rand:.3f}")
    assert np.mean(lens) <= 6
    assert rmse_cat < rmse_rand
    assert rmse_cat < 0.65


def test_confident_prior_means_no_questions():
    st = QuizState.start(v_mean=0.5, v_var=0.1, a_mean=-0.3, a_var=0.1)
    st.answered.append(1)           # simulate "already in a session"
    item, why = next_item(st)
    assert item is None and why == "confident"


def test_to_va_range():
    assert -1 < to_va(-3) < to_va(0) == 0 < to_va(3) < 1
