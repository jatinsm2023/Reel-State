"""Put a mood into the movie embedding space.

Movies have both a 384-d content embedding and a 2-d affect position (valence, arousal). We fit a
smooth map  (v, a) -> E[embedding | affect]  by ridge regression on quadratic features. Embedding a
mood is then just evaluating that map: "what does a typical movie that feels like this look like?"
The result is L2-normalised so cosine distance against `movies.content_emb` is meaningful, which
is what lets mood-to-movie retrieval be one ordinary ANN query inside the database.
"""
import numpy as np

from .config import DATA_DIR, ROOT
from .db import as_np, connect

MAP_PATH = DATA_DIR / "affect_map.npz"
ASSET_PATH = ROOT / "assets" / "affect_map.npz"       # shipped with the repo so a fresh deploy needs no fitting
RIDGE = 1e-2


def features(v, a) -> np.ndarray:
    v, a = np.asarray(v, dtype=np.float64), np.asarray(a, dtype=np.float64)
    return np.stack([np.ones_like(v), v, a, v * v, a * a, v * a], axis=-1)


def fit(save: bool = True) -> np.ndarray:
    with connect() as conn:
        rows = conn.execute("SELECT valence, arousal, content_emb FROM movies WHERE content_emb IS NOT NULL").fetchall()
    V = np.array([r[0] for r in rows])
    A = np.array([r[1] for r in rows])
    E = np.stack([as_np(r[2]).astype(np.float64) for r in rows])
    X = features(V, A)
    W = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ E)        # (6, 384)
    if save:
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(MAP_PATH, W=W)
    return W


_W: np.ndarray | None = None


def load() -> np.ndarray:
    global _W
    if _W is None:
        path = next((q for q in (ASSET_PATH, MAP_PATH) if q.exists()), None)
        _W = np.load(path)["W"] if path else fit()
    return _W


def embed(v: float, a: float) -> np.ndarray:
    e = features(v, a) @ load()
    return (e / np.linalg.norm(e)).astype(np.float32)
