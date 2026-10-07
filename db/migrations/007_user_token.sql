-- Per-profile secret. Only a hash is stored; the plaintext token lives on the participant's device.
ALTER TABLE users ADD COLUMN token_hash text;
