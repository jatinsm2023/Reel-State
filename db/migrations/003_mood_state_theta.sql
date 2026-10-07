-- Keep the exact latent (z-scale) belief next to the squashed valence/arousal used for retrieval,
-- so a stored mood can be restored for drift + fusion without inverting tanh.
ALTER TABLE mood_state ADD COLUMN theta_v real NOT NULL DEFAULT 0;
ALTER TABLE mood_state ADD COLUMN theta_a real NOT NULL DEFAULT 0;
