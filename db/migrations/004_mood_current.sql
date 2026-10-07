-- One live row per user: the latest mood vector. Rewritten on every reading, so its ANN index
-- sees a continuous stream of updates (the workload the systems benchmark stresses).
-- mood_state stays append-only (the history); mood_current is the "right now" view.
CREATE TABLE mood_current (
    user_id  integer PRIMARY KEY REFERENCES users,
    ts       timestamptz NOT NULL,
    valence  real        NOT NULL,
    arousal  real        NOT NULL,
    var_v    real        NOT NULL,
    var_a    real        NOT NULL,
    mood_emb vector(384) NOT NULL
);
CREATE INDEX mood_current_hnsw ON mood_current USING hnsw (mood_emb vector_cosine_ops);
