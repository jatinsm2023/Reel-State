"""Phase 1c: compute movie valence/arousal and 384-d content embeddings.

Embedding text blends what a movie is (title, genres, plot overview) with how it feels
(its strongest genome tags), so that mood-conditioned queries land near it in one space.
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .affect import affect_scores
from .config import DATA_DIR, EMBED_MODEL
from .db import connect

TAGS_IN_TEXT = 12


def movie_text(title: str, year, genres: list[str], overview: str | None, tags: list[str]) -> str:
    parts = [f"{title} ({year})" if year else title]
    if genres:
        parts.append("Genres: " + ", ".join(genres) + ".")
    if overview:
        parts.append(overview)
    if tags:
        parts.append("Feels: " + ", ".join(tags) + ".")
    return " ".join(parts)


def main() -> None:
    z = np.load(DATA_DIR / "genome_matrix.npz", allow_pickle=True)
    val, aro = affect_scores(z["tags"], z["matrix"])
    row_of = {int(m): i for i, m in enumerate(z["movie_ids"])}

    with connect() as conn:
        movies = conn.execute("SELECT movie_id, title, year, genres, overview FROM movies ORDER BY movie_id").fetchall()
        tag_rows = conn.execute(
            "SELECT movie_id, array_agg(tag ORDER BY relevance DESC) FROM movie_tags GROUP BY movie_id"
        ).fetchall()
    tags_of = {m: t[:TAGS_IN_TEXT] for m, t in tag_rows}

    texts = [movie_text(t, y, g, o, tags_of.get(m, [])) for m, t, y, g, o in movies]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"embedding {len(texts)} movies on {device} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    rows = [
        (emb[i], float(val[row_of[m]]), float(aro[row_of[m]]), m)
        for i, (m, *_rest) in enumerate(movies)
    ]
    with connect() as conn, conn.transaction():
        conn.cursor().executemany(
            "UPDATE movies SET content_emb = %s, valence = %s, arousal = %s WHERE movie_id = %s", rows
        )
        print(conn.execute("SELECT count(content_emb), count(valence) FROM movies").fetchone())


if __name__ == "__main__":
    main()
