-- ANN index for mood -> movie retrieval. Queries run with hnsw.iterative_scan so that relational
-- filters (runtime, language, genre, already-seen) cannot starve the candidate list.
CREATE INDEX movies_content_hnsw ON movies USING hnsw (content_emb vector_cosine_ops) WITH (m = 16, ef_construction = 64);
