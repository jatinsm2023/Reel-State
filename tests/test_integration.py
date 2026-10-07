"""End-to-end through the real database: sense -> fuse -> recommend -> check in -> policy learns."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from reelstate import policy, recommend, service, simulate
from reelstate.db import connect

T0 = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn_user():
    with connect() as conn:
        service.seed_quiz_items(conn)
        uid = service.create_user(conn, "itest", synthetic=True)
        conn.commit()
        yield conn, uid
        conn.rollback()
        simulate.cleanup(conn, [uid])


def test_quiz_commit_sets_a_confident_mood_and_next_day_it_asks_again(conn_user):
    conn, uid = conn_user
    answers = []
    while True:
        r = service.quiz_next(conn, uid, answers, T0)
        if r["done"]:
            break
        answers.append({"item_id": r["item"]["item_id"], "answer": 0, "latency_ms": 2000})   # lowest option everywhere
    mood = service.quiz_commit(conn, uid, answers, T0)
    assert 2 <= len(answers) <= 6
    assert mood.v < -0.5 and mood.a < -0.5
    assert not service.current_mood(conn, uid, T0 + timedelta(minutes=5))[0].needs_question()
    assert service.current_mood(conn, uid, T0 + timedelta(days=2))[0].needs_question()


def test_state_history_is_append_only_and_mood_current_tracks_latest(conn_user):
    conn, uid = conn_user
    service.observe_context(conn, uid, T0)
    service.observe(conn, uid, "quiz", {"valence": 1.0, "arousal": 1.0, "var_valence": 0.3, "var_arousal": 0.3}, T0 + timedelta(minutes=1))
    n = conn.execute("SELECT count(*) FROM mood_state WHERE user_id = %s", (uid,)).fetchone()[0]
    cur = conn.execute("SELECT valence, ts FROM mood_current WHERE user_id = %s", (uid,)).fetchall()
    assert n == 2 and len(cur) == 1 and cur[0][0] > 0.2


def test_no_raw_keystroke_data_is_persisted(conn_user):
    conn, uid = conn_user
    rng = np.random.default_rng(0)
    events = [(float(rng.lognormal(np.log(200), 0.4)), "char") for _ in range(80)]
    assert service.observe_typing(conn, uid, events, T0) is not None
    feats = conn.execute("SELECT features FROM mood_events WHERE user_id = %s AND source = 'typing'", (uid,)).fetchone()[0]
    from reelstate.typing_features import FEATURES
    assert set(feats["values"]) == set(FEATURES) and set(feats["z"]) == set(FEATURES)
    assert all(isinstance(x, (int, float)) for x in feats["values"].values())

    def has_list(o):
        return isinstance(o, list) or (isinstance(o, dict) and any(has_list(v) for v in o.values()))
    assert not has_list(feats)                       # nothing that could hold raw timings or keys


def test_recommend_then_checkin_updates_policy_and_reads_back_mood_before(conn_user):
    conn, uid = conn_user
    service.observe(conn, uid, "quiz", {"valence": -1.5, "arousal": 1.2, "var_valence": 0.2, "var_arousal": 0.2}, T0)
    res = recommend.recommend(conn, uid, T0 + timedelta(minutes=1), rng=np.random.default_rng(1), force_strategy="regulate")
    assert len(res["slate"]) == 5 and res["quadrant"] == "lo_v_hi_a"
    top = res["slate"][0]
    assert top["explanation"]["strategy"] == "regulate" and top["explanation"]["why"]
    out = service.record_checkin(conn, top["rec_id"], 0.5, -0.2, True, T0 + timedelta(hours=3))
    assert out["reward"] > 0.6 and out["quadrant"] == "lo_v_hi_a"
    arm = service.load_arms(conn, uid, "lo_v_hi_a")["regulate"]
    assert arm.obs == 1 and arm.mean > 0.5
    assert service.load_arms(conn, uid, "lo_v_hi_a")["match"].obs == 0


def test_already_recommended_movies_are_not_repeated(conn_user):
    conn, uid = conn_user
    service.observe(conn, uid, "quiz", {"valence": 0.8, "arousal": 0.8, "var_valence": 0.3, "var_arousal": 0.3}, T0)
    a = recommend.recommend(conn, uid, T0, rng=np.random.default_rng(2), force_strategy="match")
    b = recommend.recommend(conn, uid, T0 + timedelta(minutes=1), rng=np.random.default_rng(2), force_strategy="match")
    assert not {m["movie_id"] for m in a["slate"]} & {m["movie_id"] for m in b["slate"]}


def test_filters_are_respected(conn_user):
    conn, uid = conn_user
    res = recommend.recommend(conn, uid, T0, recommend.Filters(max_runtime=95, genres=["Comedy"]), rng=np.random.default_rng(3))
    assert res["slate"]
    assert all(m["runtime_min"] <= 95 and "Comedy" in m["genres"] for m in res["slate"])


def test_match_and_regulate_target_different_films_for_a_low_mood(conn_user):
    conn, uid = conn_user
    service.observe(conn, uid, "quiz", {"valence": -1.8, "arousal": -1.0, "var_valence": 0.2, "var_arousal": 0.2}, T0)
    m = recommend.recommend(conn, uid, T0, rng=np.random.default_rng(4), force_strategy="match")
    r = recommend.recommend(conn, uid, T0 + timedelta(minutes=1), rng=np.random.default_rng(4), force_strategy="regulate")
    mv = np.mean([x["explanation"]["movie_affect"][0] for x in m["slate"]])
    rv = np.mean([x["explanation"]["movie_affect"][0] for x in r["slate"]])
    assert rv > mv + 0.4          # regulating lifts valence of what is suggested


def test_household_serves_both_people_and_each_can_check_in(conn_user):
    from reelstate import household
    conn, uid = conn_user
    other = service.create_user(conn, "itest2", synthetic=True)
    try:
        service.observe(conn, uid, "quiz", {"valence": 1.6, "arousal": 1.2, "var_valence": 0.2, "var_arousal": 0.2}, T0)     # upbeat, wired
        service.observe(conn, other, "quiz", {"valence": -1.6, "arousal": -1.0, "var_valence": 0.2, "var_arousal": 0.2}, T0)  # low, tired
        res = household.recommend_group(conn, [uid, other], T0 + timedelta(minutes=1), rng=np.random.default_rng(6))
        assert len(res["slate"]) == 5 and {m["user_id"] for m in res["members"]} == {uid, other}
        top = res["slate"][0]
        assert set(top["rec_ids"]) == {uid, other}
        # the low-mood member gets extra say, so the group target is not simply the midpoint
        low = next(m for m in res["members"] if m["user_id"] == other)
        high = next(m for m in res["members"] if m["user_id"] == uid)
        assert low["weight"] > high["weight"] * 0.9
        for u in (uid, other):
            assert service.record_checkin(conn, top["rec_ids"][u], 0.2, 0.0, True, T0 + timedelta(hours=3))["reward"] >= 0
        with pytest.raises(ValueError):
            household.recommend_group(conn, [uid], T0)
    finally:
        conn.rollback()
        simulate.cleanup(conn, [other])
