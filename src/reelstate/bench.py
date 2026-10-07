"""Systems benchmark: vector index behaviour under the workloads Reel State actually generates.

E1  unfiltered kNN      recall@10 vs latency for exact / IVFFlat / HNSW, plus build time and size
E2  filtered kNN        the same, with a relational predicate of controlled selectivity
E3  update churn        the live mood index is rewritten constantly: recall/latency/size as vectors
                        are replaced, with and without maintenance (VACUUM / REINDEX)

Data are a SYNTHETIC scale-up of the real movie embeddings (sample a real movie, add Gaussian noise,
renormalise) so cluster structure is realistic; this is stated in the report. Queries are mood
embeddings from affect_space.embed over random valence-arousal points, i.e. the real query
distribution, which is concentrated in a low-dimensional subspace.

Ground truth is exact search (index scans disabled) over the same rows and predicate.
Usage:  python -m reelstate.bench e1|e2|e3 [--n 100000]
"""
import argparse
import json
import time

import numpy as np

from . import affect_space
from .config import ROOT
from .db import as_np, connect

OUT = ROOT / "reports"
K = 10
N_QUERIES = 150


def pct(xs, p):
    return float(np.percentile(xs, p))


def make_queries(rng, n=N_QUERIES) -> list[np.ndarray]:
    return [affect_space.embed(float(v), float(a)) for v, a in rng.uniform(-0.95, 0.95, size=(n, 2))]


def build_table(conn, name: str, n: int, seed: int = 0, with_attrs: bool = True) -> None:
    rng = np.random.default_rng(seed)
    base = np.stack([as_np(r[0]) for r in conn.execute("SELECT content_emb FROM movies WHERE content_emb IS NOT NULL")])
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"CREATE TABLE {name} (id integer PRIMARY KEY, emb vector(384), bucket integer)")
    with conn.cursor().copy(f"COPY {name} (id, emb, bucket) FROM STDIN") as cp:
        for start in range(0, n, 20000):
            m = min(20000, n - start)
            parent = base[rng.integers(0, len(base), m)]
            v = parent + rng.normal(0, 0.03, parent.shape).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            for i in range(m):
                cp.write_row((start + i, v[i], int(rng.integers(0, 10000))))
    conn.commit()
    conn.execute(f"ANALYZE {name}")
    conn.commit()


def exact_topk(conn, table, q, where="TRUE", k=K) -> list[int]:
    conn.execute("SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off")
    rows = conn.execute(f"SELECT id FROM {table} WHERE {where} ORDER BY emb <=> %s LIMIT {k}", (q,)).fetchall()
    conn.rollback()
    return [r[0] for r in rows]


def ann_topk(conn, table, q, where="TRUE", k=K) -> tuple[list[int], float]:
    t = time.perf_counter()
    rows = conn.execute(f"SELECT id FROM {table} WHERE {where} ORDER BY emb <=> %s LIMIT {k}", (q,)).fetchall()
    return [r[0] for r in rows], (time.perf_counter() - t) * 1000


def evaluate(conn, table, queries, truths, where="TRUE") -> dict:
    rec, lat, short = [], [], 0
    for q, gt in zip(queries, truths):
        got, ms = ann_topk(conn, table, q, where)
        lat.append(ms)
        rec.append(len(set(got) & set(gt)) / max(len(gt), 1))
        short += len(got) < min(K, len(gt))
    return {"recall": float(np.mean(rec)), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "short_results": short}


def index_size_mb(conn, name):
    r = conn.execute("SELECT pg_relation_size(%s::regclass)", (name,)).fetchone()
    return r[0] / 1e6


# --------------------------------------------------------------------- E1 / E2
def e1_e2(n: int, filtered: bool) -> list[dict]:
    rng = np.random.default_rng(1)
    queries = make_queries(rng)
    results = []
    with connect() as conn:
        build_table(conn, "bench_v", n)
        sels = [0.5, 0.1, 0.01, 0.001] if filtered else [1.0]
        truths = {}
        for s in sels:
            where = "TRUE" if s == 1.0 else f"bucket < {int(s * 10000)}"
            truths[s] = [exact_topk(conn, "bench_v", q, where) for q in queries]
        configs = [("exact", None, [{}])]
        lists = max(int(n ** 0.5), 10)
        configs += [("ivfflat", f"CREATE INDEX bench_idx ON bench_v USING ivfflat (emb vector_cosine_ops) WITH (lists = {lists})",
                     [{"ivfflat.probes": p} for p in (1, 5, 10, 20, 40)])]
        configs += [("hnsw", "CREATE INDEX bench_idx ON bench_v USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                     [{"hnsw.ef_search": e} for e in (10, 20, 40, 80, 160)])]
        if filtered:
            configs += [("hnsw+iterative", "CREATE INDEX bench_idx ON bench_v USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                         [{"hnsw.ef_search": e, "hnsw.iterative_scan": "relaxed_order", "hnsw.max_scan_tuples": 100000} for e in (40, 160)])]
        for name, ddl, settings in configs:
            conn.execute("DROP INDEX IF EXISTS bench_idx")
            conn.commit()
            build_s = 0.0
            size = 0.0
            if ddl:
                conn.execute("SET maintenance_work_mem = '1GB'")
                t = time.perf_counter()
                conn.execute(ddl)
                conn.commit()
                build_s = time.perf_counter() - t
                size = index_size_mb(conn, "bench_idx")
            for st in settings:
                for k, v in st.items():
                    conn.execute(f"SET {k} = '{v}'")
                for s in sels:
                    where = "TRUE" if s == 1.0 else f"bucket < {int(s * 10000)}"
                    if ddl is None:
                        conn.execute("SET enable_indexscan = off")
                    else:
                        conn.execute("RESET enable_indexscan")
                    r = evaluate(conn, "bench_v", queries, truths[s], where)
                    r.update(index=name, params=st, selectivity=s, n=n, build_s=build_s, index_mb=size)
                    results.append(r)
                    print(json.dumps(r), flush=True)
                conn.execute("RESET ALL")
            conn.commit()
        conn.execute("DROP TABLE IF EXISTS bench_v")
        conn.commit()
    return results


# -------------------------------------------------------------------------- E3
def e3(n: int, rounds: int = 10, churn: float = 0.10) -> list[dict]:
    """Replace `churn` of all vectors each round (a user's mood moved) and watch the index."""
    rng = np.random.default_rng(3)
    queries = make_queries(rng)
    results = []
    variants = [
        ("hnsw", "hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)", {"hnsw.ef_search": 40}, False),
        ("hnsw+vacuum", "hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)", {"hnsw.ef_search": 40}, True),
        ("ivfflat", f"ivfflat (emb vector_cosine_ops) WITH (lists = {max(int(n ** 0.5), 10)})", {"ivfflat.probes": 10}, False),
        ("ivfflat+reindex", f"ivfflat (emb vector_cosine_ops) WITH (lists = {max(int(n ** 0.5), 10)})", {"ivfflat.probes": 10}, True),
    ]
    with connect() as conn:
        for name, idx_def, settings, maintain in variants:
            build_table(conn, "bench_m", n, seed=5)
            conn.execute("SET maintenance_work_mem = '1GB'")
            conn.execute(f"CREATE INDEX bench_m_idx ON bench_m USING {idx_def}")
            conn.commit()
            for k, v in settings.items():
                conn.execute(f"SET {k} = '{v}'")
            shift = rng.uniform(-0.9, 0.9, size=2)                 # the whole population drifts to a new mood region
            for rnd in range(rounds + 1):
                if rnd > 0:
                    ids = rng.choice(n, int(n * churn), replace=False)
                    mv = np.clip(shift + rng.normal(0, 0.3, (len(ids), 2)), -0.95, 0.95)
                    vecs = np.stack([affect_space.embed(float(a), float(b)) for a, b in mv])
                    vecs += rng.normal(0, 0.02, vecs.shape).astype(np.float32)
                    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                    t = time.perf_counter()
                    with conn.cursor() as cur:
                        cur.executemany("UPDATE bench_m SET emb = %s WHERE id = %s", [(vecs[i], int(ids[i])) for i in range(len(ids))])
                    conn.commit()
                    upd_s = time.perf_counter() - t
                    if maintain:
                        t = time.perf_counter()
                        if name.startswith("hnsw"):
                            conn.autocommit = True
                            conn.execute("VACUUM bench_m")
                            conn.autocommit = False
                        else:
                            conn.autocommit = True
                            conn.execute("REINDEX INDEX bench_m_idx")
                            conn.autocommit = False
                        maint_s = time.perf_counter() - t
                    else:
                        maint_s = 0.0
                    conn.execute("ANALYZE bench_m")
                    conn.commit()
                else:
                    upd_s, maint_s = 0.0, 0.0
                truths = [exact_topk(conn, "bench_m", q) for q in queries]
                for k, v in settings.items():
                    conn.execute(f"SET {k} = '{v}'")
                r = evaluate(conn, "bench_m", queries, truths)
                r.update(variant=name, round=rnd, n=n, cumulative_churn=rnd * churn, update_rows_per_s=(n * churn / upd_s) if upd_s else None,
                         maintain_s=maint_s, index_mb=index_size_mb(conn, "bench_m_idx"), table_mb=index_size_mb(conn, "bench_m"))
                results.append(r)
                print(json.dumps(r), flush=True)
            conn.execute("DROP TABLE IF EXISTS bench_m")
            conn.commit()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2", "e3"])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = e1_e2(a.n, filtered=False) if a.exp == "e1" else e1_e2(a.n, filtered=True) if a.exp == "e2" else e3(a.n, a.rounds)
    (OUT / f"bench_{a.exp}_{a.n}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
