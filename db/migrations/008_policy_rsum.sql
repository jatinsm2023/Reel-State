-- Raw sufficient statistics per arm (own observations only). alpha/beta are now derived on load as
-- prior + own evidence + down-weighted evidence pooled from the same user's other mood quadrants.
ALTER TABLE policy_state ADD COLUMN r_sum real NOT NULL DEFAULT 0;
