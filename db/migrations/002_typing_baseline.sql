-- Per-user running baseline of typing features (Welford). Derived statistics only: no raw timings.
CREATE TABLE typing_baseline (
    user_id integer NOT NULL REFERENCES users,
    feature text    NOT NULL,
    n       integer NOT NULL DEFAULT 0,
    mean    double precision NOT NULL DEFAULT 0,
    m2      double precision NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, feature)
);
