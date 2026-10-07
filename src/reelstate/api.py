"""FastAPI app: thin HTTP layer over service.py / recommend.py, plus the static web client.

Run:  PYTHONPATH=src .venv/bin/uvicorn reelstate.api:app --reload
"""
import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, household, migrate, policy, recommend, service
from .config import ROOT
from .db import connect
from .quiz import to_va

log = logging.getLogger("reelstate")
WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bring a fresh database up to date. Both steps are idempotent, so this is safe on every start."""
    try:
        migrate.main()
        with connect() as c:
            service.seed_quiz_items(c)
    except Exception:                       # keep serving /healthz so the logs can be read
        log.exception("startup database setup failed")
    yield


app = FastAPI(title="Reel State", lifespan=lifespan)


@contextmanager
def db():
    with connect() as conn:          # psycopg commits on clean exit, rolls back on error
        yield conn


def now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ models
class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    household_id: int | None = None


class Answer(BaseModel):
    item_id: int
    answer: int = Field(ge=0, le=4)
    latency_ms: int | None = None


class QuizReq(BaseModel):
    user_id: int
    answers: list[Answer] = []
    force: bool = False


class TypingReq(BaseModel):
    user_id: int
    events: list[tuple[float, str]]        # (gap_ms since previous key, key class) - never the characters


class RecommendReq(BaseModel):
    user_id: int
    max_runtime: int | None = None
    language: str | None = None
    genres: list[str] | None = None


class Member(BaseModel):
    user_id: int
    token: str


class HouseholdReq(BaseModel):
    members: list[Member] = Field(min_length=2, max_length=6)
    max_runtime: int | None = None
    language: str | None = None
    genres: list[str] | None = None


class CheckinReq(BaseModel):
    rec_id: int
    valence_after: float = Field(ge=-1, le=1)
    arousal_after: float = Field(ge=-1, le=1)
    liked: bool | None = None


# --------------------------------------------------------------- endpoints
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/status")
def status():
    """Deployment check: is the database reachable, is pgvector there, is the film catalogue loaded?"""
    with db() as c:
        ver = c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        n, emb = c.execute("SELECT count(*), count(content_emb) FROM movies").fetchone()
    v = ver[0] if ver else None
    # a SET of an unknown hnsw.* setting is accepted silently, so judge support by the extension version
    iterative = bool(v) and tuple(int(x) for x in v.split(".")[:2]) >= (0, 8)
    return {"pgvector": v, "iterative_scan_supported": iterative, "movies": n, "movies_with_vectors": emb,
            "ready": bool(v and n and n == emb)}


@app.post("/api/users")
def create_user(u: NewUser):
    with db() as c:
        uid = service.create_user(c, u.name, u.household_id)
        return {"user_id": uid, "token": auth.issue(c, uid)}


def _mood_payload(c, user_id: int, t: datetime) -> dict:
    mood, last = service.current_mood(c, user_id, t)
    return {
        "valence": to_va(mood.v), "arousal": to_va(mood.a),
        "sd_valence": mood.var_v ** 0.5, "sd_arousal": mood.var_a ** 0.5,
        "needs_question": mood.needs_question(), "last_update": last,
    }


@app.get("/api/mood/{user_id}")
def get_mood(user_id: int, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, user_id, x_token)
        return _mood_payload(c, user_id, now())


@app.get("/api/trajectory/{user_id}")
def trajectory(user_id: int, limit: int = 60, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, user_id, x_token)
        rows = c.execute(
            "SELECT ts, valence, arousal FROM mood_state WHERE user_id = %s ORDER BY ts DESC LIMIT %s", (user_id, limit)
        ).fetchall()
    return [{"ts": r[0], "valence": r[1], "arousal": r[2]} for r in reversed(rows)]


@app.post("/api/context/{user_id}")
def context(user_id: int, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, user_id, x_token)
        service.observe_context(c, user_id, now())
        return _mood_payload(c, user_id, now())


@app.post("/api/typing")
def typing(req: TypingReq, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, req.user_id, x_token)
        res = service.observe_typing(c, req.user_id, req.events, now())
        return {"used": res is not None, "mood": _mood_payload(c, req.user_id, now())}


@app.post("/api/quiz/next")
def quiz_next(req: QuizReq, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, req.user_id, x_token)
        return service.quiz_next(c, req.user_id, [a.model_dump() for a in req.answers], now(), req.force)


@app.post("/api/quiz/commit")
def quiz_commit(req: QuizReq, x_token: str | None = Header(None)):
    if not req.answers:
        raise HTTPException(400, "no answers to commit")
    with db() as c:
        auth.verify(c, req.user_id, x_token)
        service.quiz_commit(c, req.user_id, [a.model_dump() for a in req.answers], now())
        return _mood_payload(c, req.user_id, now())


@app.post("/api/recommend")
def recommend_endpoint(req: RecommendReq, x_token: str | None = Header(None)):
    with db() as c:
        auth.verify(c, req.user_id, x_token)
        f = recommend.Filters(max_runtime=req.max_runtime, language=req.language, genres=req.genres)
        return recommend.recommend(c, req.user_id, now(), f)


@app.post("/api/household/recommend")
def household_recommend(req: HouseholdReq):
    with db() as c:
        for m in req.members:                       # every person on the sofa must have authorised their own profile
            auth.verify(c, m.user_id, m.token)
        f = recommend.Filters(max_runtime=req.max_runtime, language=req.language, genres=req.genres)
        return household.recommend_group(c, [m.user_id for m in req.members], now(), f)


@app.post("/api/checkin")
def checkin(req: CheckinReq, x_token: str | None = Header(None)):
    with db() as c:
        owner = c.execute("SELECT user_id FROM recommendations WHERE rec_id = %s", (req.rec_id,)).fetchone()
        if owner is None:
            raise HTTPException(404, f"unknown recommendation {req.rec_id}")
        auth.verify(c, owner[0], x_token)
        try:
            return service.record_checkin(c, req.rec_id, req.valence_after, req.arousal_after, req.liked, now())
        except KeyError as e:
            raise HTTPException(404, str(e))


@app.get("/api/policy/{user_id}")
def get_policy(user_id: int, x_token: str | None = Header(None)):
    """What the system has learned about this person: per mood quadrant, how well each strategy works."""
    out = {}
    with db() as c:
        auth.verify(c, user_id, x_token)
        for q in policy.QUADRANTS:
            arms = service.load_arms(c, user_id, q)
            out[q] = {s: {"mean": round(arms[s].mean, 3), "observations": arms[s].obs} for s in arms}
    return out


@app.get("/api/genres")
def genres():
    with db() as c:
        return [r[0] for r in c.execute("SELECT g FROM (SELECT unnest(genres) g, count(*) n FROM movies GROUP BY g) s WHERE n > 30 ORDER BY g")]


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")
