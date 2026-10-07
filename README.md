# Reel State

A movie recommender that decides, per person and per moment, whether to **match** your mood or **move** it, and learns which one to do from how you actually felt afterwards.

Built as an Advanced Database Systems project. The database is not storage for an app: mood is a first-class, continuously updated vector object, retrieval is one hybrid SQL query, and the systems experiments (`bench.py`) stress the exact workload this design creates.

## How it works

```
 typing rhythm ─┐                                        ┌─ similar-mood neighbours (live ANN index)
 adaptive quiz ─┼─► fusion (Kalman/OU) ─► mood_state ───►│
 time of day  ──┘     per-axis belief       (append-only) └─► policy (Thompson sampling) ─► match | regulate
                                            mood_current        │
                                                                ▼
                    target mood ─► affect_space.embed ─► HNSW ANN + filters ─► rerank ─► explained slate
                                                                                              │
                         policy update ◄── post-watch check-in (the felt outcome) ◄───────────┘
```

| Layer | Module | What it does |
|---|---|---|
| Movie affect | `affect.py` | Valence/arousal per movie from the Tag Genome via a hand-built, documented tag lexicon |
| Mood in movie space | `affect_space.py` | Ridge map (valence, arousal) → expected content embedding, so a mood is a vector in the same space as movies |
| Adaptive quiz | `quiz.py`, `quiz_bank.py` | Graded-response IRT, two latent axes, picks the question that most reduces expected variance, stops when confident |
| Typing sensing | `typing_features.py` | Derived timing features only (never characters), compared against the user's own baseline |
| Fusion | `fusion.py` | Ornstein-Uhlenbeck drift + precision-weighted updates; stale mood ⇒ ask a question |
| Policy | `policy.py` | Per user and mood quadrant, a Beta bandit over {match, regulate}; population prior for cold start |
| Retrieval | `recommend.py` | One SQL statement (filters + ANN), affect rerank, similar-mood boost, explanation |
| Household | `household.py` | Fairness-aware blending of several people's targets |
| Evaluation | `simulate.py`, `experiments.py`, `analysis.py`, `household_eval.py` | Synthetic users driven through the real code path |
| Systems | `bench.py` | HNSW vs IVFFlat vs exact: filtered search and update churn |

## Setup

```bash
brew install postgresql@17 pgvector
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/db.sh start                      # Postgres needs LC_ALL=en_US.UTF-8; the script sets it
createdb reelstate                       # (with postgresql@17 on PATH)
echo "TMDB_API_KEY=..." >> .env          # free key from themoviedb.org

export PYTHONPATH=src
.venv/bin/python -m reelstate.migrate
.venv/bin/python -m reelstate.load_movielens   # needs data/ml-25m (grouplens.org/datasets/movielens/25m)
.venv/bin/python -m reelstate.fetch_tmdb
.venv/bin/python -m reelstate.build_embeddings
```

## Run

```bash
scripts/serve.sh                          # http://localhost:8000
PYTHONPATH=src .venv/bin/python -m pytest tests
PYTHONPATH=src .venv/bin/python -m reelstate.experiments && .venv/bin/python -m reelstate.analysis
PYTHONPATH=src .venv/bin/python -m reelstate.household_eval
PYTHONPATH=src .venv/bin/python -m reelstate.bench e1|e2|e3 --n 100000
```

## Results

The full write-up, with every number traced to a file under `reports/`, is in [REPORT.md](REPORT.md). In one paragraph: the system works end to end on 13,174 real films; recommendations fit the mood and are well rated (mean 3.76 vs 3.30 random); the adaptive quiz plus fusion cut questions per session by 43%; a learned match-vs-regulate policy correctly identifies who needs matching but only reaches parity with "always regulate" (the headroom is smaller than the cost of learning it); household blending showed no benefit; and HNSW with iterative scan is the right index for filtered retrieval while HNSW writes are 56x slower than IVFFlat's. Nothing has been tested on real people yet: `pilot/PILOT.md` is the protocol.

## Honest limits

* Typing and quiz parameters are literature-informed **priors**, not fitted values; `quiz_responses` and `mood_events` collect the data to calibrate them from the pilot.
* The valence/arousal lexicon is hand-assigned and partly debatable (it scores *The Shawshank Redemption* as pleasant). It is isolated in one table so it can be ablated or replaced.
* The simulator encodes assumptions (how people respond to matching vs regulating). It shows the mechanism works when people differ; **it is not evidence about real people.** That is what the pilot is for.
* "Better" is defined as more pleasant and less extreme (`policy.wellbeing_gain`), an explicit design choice.
* Benchmark data beyond 13k movies are a synthetic scale-up of real embeddings.
