-- Real observation count per arm, separate from the prior pseudo-counts folded into alpha/beta.
ALTER TABLE policy_state ADD COLUMN n_obs integer NOT NULL DEFAULT 0;
