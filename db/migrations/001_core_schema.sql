-- Reel State core schema.
-- Design rules:
--   * raw keystroke timings are never stored; mood_events keeps derived features only (privacy by design)
--   * mood_state is append-only: a trajectory, not an overwritten snapshot
--   * mood vectors and movie vectors share one 384-d space so retrieval is plain cosine ANN

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- content
CREATE TABLE movies (
    movie_id      integer PRIMARY KEY,           -- MovieLens id
    title         text        NOT NULL,
    year          smallint,
    genres        text[]      NOT NULL DEFAULT '{}',
    imdb_id       text,
    tmdb_id       integer,
    overview      text,
    runtime_min   smallint,
    language      text,
    n_ratings     integer     NOT NULL DEFAULT 0,
    avg_rating    real,
    valence       real,                          -- [-1, 1]  unpleasant -> pleasant
    arousal       real,                          -- [-1, 1]  calm -> intense
    content_emb   vector(384)
);

CREATE TABLE movie_tags (                         -- Tag Genome relevance scores
    movie_id   integer NOT NULL REFERENCES movies,
    tag        text    NOT NULL,
    relevance  real    NOT NULL,
    PRIMARY KEY (movie_id, tag)
);

-- ------------------------------------------------------------------ people
CREATE TABLE households (
    household_id serial PRIMARY KEY,
    name         text NOT NULL
);

CREATE TABLE users (
    user_id      serial PRIMARY KEY,
    display_name text        NOT NULL,
    household_id integer REFERENCES households,
    is_synthetic boolean     NOT NULL DEFAULT false,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------- mood
CREATE TABLE mood_events (                        -- one row per sensed observation
    event_id    bigserial PRIMARY KEY,
    user_id     integer     NOT NULL REFERENCES users,
    ts          timestamptz NOT NULL DEFAULT now(),
    source      text        NOT NULL CHECK (source IN ('typing', 'quiz', 'context', 'checkin')),
    features    jsonb       NOT NULL DEFAULT '{}',  -- derived features only
    valence_obs real,
    arousal_obs real,
    obs_var     real                              -- observation noise variance (lower = more trusted)
);
CREATE INDEX mood_events_user_ts ON mood_events (user_id, ts DESC);

CREATE TABLE mood_state (                         -- append-only trajectory
    user_id   integer     NOT NULL REFERENCES users,
    ts        timestamptz NOT NULL DEFAULT now(),
    valence   real        NOT NULL,
    arousal   real        NOT NULL,
    var_v     real        NOT NULL,               -- posterior variance, drives "should we ask a question?"
    var_a     real        NOT NULL,
    mood_emb  vector(384) NOT NULL,
    PRIMARY KEY (user_id, ts)
);

-- ------------------------------------------------------------ adaptive quiz
CREATE TABLE quiz_items (
    item_id        serial PRIMARY KEY,
    prompt         text  NOT NULL,
    options        jsonb NOT NULL,                -- ordered list of answer labels
    axis           text  NOT NULL CHECK (axis IN ('valence', 'arousal')),
    discrimination real  NOT NULL,                -- IRT 'a'
    thresholds     jsonb NOT NULL                 -- IRT ordered 'b' per category boundary
);

CREATE TABLE quiz_responses (
    response_id bigserial PRIMARY KEY,
    user_id     integer     NOT NULL REFERENCES users,
    item_id     integer     NOT NULL REFERENCES quiz_items,
    ts          timestamptz NOT NULL DEFAULT now(),
    answer      smallint    NOT NULL,
    latency_ms  integer
);

-- --------------------------------------------------- recommend + feedback loop
CREATE TABLE recommendations (
    rec_id       bigserial PRIMARY KEY,
    user_id      integer     NOT NULL REFERENCES users,
    ts           timestamptz NOT NULL DEFAULT now(),
    mood_ts      timestamptz NOT NULL,            -- which mood_state row it was based on
    movie_id     integer     NOT NULL REFERENCES movies,
    strategy     text        NOT NULL CHECK (strategy IN ('match', 'regulate')),
    rank         smallint    NOT NULL,
    score        real        NOT NULL,
    explanation  jsonb       NOT NULL DEFAULT '{}'
);
CREATE INDEX recs_user_ts ON recommendations (user_id, ts DESC);

CREATE TABLE checkins (                           -- the post-watch ground truth
    checkin_id    bigserial PRIMARY KEY,
    rec_id        bigint      NOT NULL REFERENCES recommendations,
    user_id       integer     NOT NULL REFERENCES users,
    ts            timestamptz NOT NULL DEFAULT now(),
    valence_after real        NOT NULL,
    arousal_after real        NOT NULL,
    reward        real        NOT NULL,           -- progress toward the target mood, in [-1, 1]
    liked         boolean
);

CREATE TABLE policy_state (                       -- Thompson-sampling Beta posteriors
    user_id    integer NOT NULL REFERENCES users,
    quadrant   text    NOT NULL CHECK (quadrant IN ('hi_v_hi_a', 'hi_v_lo_a', 'lo_v_hi_a', 'lo_v_lo_a')),
    strategy   text    NOT NULL CHECK (strategy IN ('match', 'regulate')),
    alpha      real    NOT NULL DEFAULT 1,
    beta       real    NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, quadrant, strategy)
);
