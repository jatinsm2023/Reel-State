import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL


def connect(**kwargs) -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, **kwargs)
    register_vector(conn)
    # Filter-aware HNSW: keep scanning until enough rows survive the WHERE clause. Iterative scan needs
    # pgvector >= 0.8; on an older server these settings do not exist, so each one is optional.
    for setting in ("hnsw.iterative_scan = relaxed_order", "hnsw.ef_search = 100", "hnsw.max_scan_tuples = 20000"):
        try:
            conn.execute(f"SET {setting}")
        except psycopg.Error:
            conn.rollback()
    conn.commit()
    return conn


def as_np(v):
    """pgvector's Vector -> float32 numpy array (passes numpy arrays through)."""
    import numpy as np

    return v.to_numpy().astype(np.float32) if hasattr(v, "to_numpy") else np.asarray(v, dtype=np.float32)
