"""Phase 1a: build the movie catalog from MovieLens 25M.

Catalog = movies that have Tag Genome scores and at least MIN_RATINGS ratings.
Writes: movies (core columns), movie_tags (top tags per movie), and a dense genome
matrix cache (data/genome_matrix.npz) used later for valence/arousal scoring.
"""
import re

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .db import connect

ML = DATA_DIR / "ml-25m"
MIN_RATINGS = 50
TOP_TAGS = 30


def parse_title(raw: str) -> tuple[str, int | None]:
    m = re.search(r"\((\d{4})\)\s*$", raw)
    if not m:
        return raw.strip(), None
    return raw[: m.start()].strip(), int(m.group(1))


def main() -> None:
    movies = pd.read_csv(ML / "movies.csv")
    links = pd.read_csv(ML / "links.csv", dtype={"imdbId": str})
    ratings = pd.read_csv(ML / "ratings.csv", usecols=["movieId", "rating"])
    stats = ratings.groupby("movieId").rating.agg(n_ratings="count", avg_rating="mean")
    del ratings

    print("reading genome scores ...")
    gs = pd.read_csv(ML / "genome-scores.csv", dtype={"movieId": np.int32, "tagId": np.int16, "relevance": np.float32})
    gtags = pd.read_csv(ML / "genome-tags.csv").set_index("tagId").tag

    keep = stats[(stats.n_ratings >= MIN_RATINGS)].index.intersection(gs.movieId.unique())
    gs = gs[gs.movieId.isin(keep)]
    cat = (
        movies[movies.movieId.isin(keep)]
        .merge(links, on="movieId", how="left")
        .merge(stats, left_on="movieId", right_index=True)
        .sort_values("movieId")
        .reset_index(drop=True)
    )
    print(f"catalog size: {len(cat)}")

    # dense genome matrix, rows aligned with cat order, columns with tag id order
    tag_ids = np.sort(gs.tagId.unique())
    row_of = {m: i for i, m in enumerate(cat.movieId)}
    col_of = {t: j for j, t in enumerate(tag_ids)}
    mat = np.zeros((len(cat), len(tag_ids)), dtype=np.float32)
    mat[gs.movieId.map(row_of).to_numpy(), gs.tagId.map(col_of).to_numpy()] = gs.relevance.to_numpy()
    np.savez_compressed(
        DATA_DIR / "genome_matrix.npz",
        movie_ids=cat.movieId.to_numpy(),
        tags=np.array([gtags[t] for t in tag_ids]),
        matrix=mat,
    )

    # top tags per movie -> movie_tags
    top = gs.sort_values(["movieId", "relevance"], ascending=[True, False]).groupby("movieId").head(TOP_TAGS)
    top = top.assign(tag=top.tagId.map(gtags))[["movieId", "tag", "relevance"]]

    with connect() as conn, conn.transaction():
        conn.execute("TRUNCATE movie_tags, movies CASCADE")
        with conn.cursor().copy(
            "COPY movies (movie_id, title, year, genres, imdb_id, tmdb_id, n_ratings, avg_rating) FROM STDIN"
        ) as cp:
            for r in cat.itertuples(index=False):
                title, year = parse_title(r.title)
                genres = [] if r.genres == "(no genres listed)" else r.genres.split("|")
                tmdb = None if pd.isna(r.tmdbId) else int(r.tmdbId)
                imdb = None if pd.isna(r.imdbId) else r.imdbId
                cp.write_row((int(r.movieId), title, year, genres, imdb, tmdb, int(r.n_ratings), float(r.avg_rating)))
        with conn.cursor().copy("COPY movie_tags (movie_id, tag, relevance) FROM STDIN") as cp:
            for r in top.itertuples(index=False):
                cp.write_row((int(r.movieId), r.tag, float(r.relevance)))

    with connect() as conn:
        print(conn.execute("SELECT count(*), count(tmdb_id), count(year) FROM movies").fetchone())
        print(conn.execute("SELECT count(*) FROM movie_tags").fetchone())


if __name__ == "__main__":
    main()
