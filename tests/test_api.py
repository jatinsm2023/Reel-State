"""HTTP-level tests of the demo API (real database)."""
import pytest
from fastapi.testclient import TestClient

from reelstate import simulate
from reelstate.api import app
from reelstate.db import connect


@pytest.fixture()
def client():
    with TestClient(app) as c:                 # runs the startup migration + quiz seeding
        yield c


@pytest.fixture()
def profile(client):
    r = client.post("/api/users", json={"name": "apitest"}).json()
    yield r["user_id"], {"X-Token": r["token"]}
    with connect() as conn:
        simulate.cleanup(conn, [r["user_id"]])


def test_health_and_status(client):
    assert client.get("/healthz").json() == {"ok": True}
    s = client.get("/api/status").json()
    assert s["ready"] is True and s["movies"] == s["movies_with_vectors"] > 10000


def test_user_endpoints_need_the_owners_token(client, profile):
    uid, headers = profile
    assert client.get(f"/api/readings/{uid}").status_code == 403
    assert client.get(f"/api/readings/{uid}", headers={"X-Token": "wrong"}).status_code == 403
    assert client.get(f"/api/readings/{uid}", headers=headers).status_code == 200


def test_readings_describe_each_sensor_in_plain_terms(client, profile):
    uid, headers = profile
    client.post(f"/api/context/{uid}", headers=headers)
    client.post("/api/quiz/commit", headers=headers, json={"user_id": uid, "answers": [
        {"item_id": 1, "answer": 1, "latency_ms": 2000}, {"item_id": 11, "answer": 0, "latency_ms": 2000}]})
    rows = client.get(f"/api/readings/{uid}", headers=headers).json()
    assert [r["source"] for r in rows][:2] == ["quiz", "context"]          # newest first
    quiz = rows[0]
    assert quiz["n_items"] == 2 and -1 <= quiz["valence"] <= 1 and -1 <= quiz["arousal"] <= 1


def test_recommendations_carry_what_the_results_screen_needs(client, profile):
    uid, headers = profile
    client.post(f"/api/context/{uid}", headers=headers)
    res = client.post("/api/recommend", headers=headers, json={"user_id": uid, "max_runtime": 130}).json()
    assert len(res["slate"]) == 5 and "arm_obs" in res and "target" in res
    for m in res["slate"]:
        assert {"poster_path", "avg_rating", "n_ratings", "overview", "explanation"} <= set(m)
        assert m["explanation"]["why"] and m["runtime_min"] <= 130


def test_checkin_requires_the_owners_token(client, profile):
    uid, headers = profile
    client.post(f"/api/context/{uid}", headers=headers)
    rec = client.post("/api/recommend", headers=headers, json={"user_id": uid}).json()["slate"][0]["rec_id"]
    body = {"rec_id": rec, "valence_after": 0.3, "arousal_after": -0.2, "liked": True}
    assert client.post("/api/checkin", json=body).status_code == 403
    assert client.post("/api/checkin", json=body, headers=headers).status_code == 200
