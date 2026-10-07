# Retrieval quality (offline, 300 random moods x 5 films)

| Pipeline | Affect distance (lower = better fit) | Mean MovieLens rating | Films rated >= 3.5 | Distinct films (of 1500 slots) |
|---|---|---|---|---|
| random | 0.873 | 3.30 | 39% | 1429 |
| ann_only | 0.532 | 3.20 | 20% | 50 |
| ann+affect | 0.179 | 3.23 | 31% | 166 |
| final | 0.231 | 3.76 | 94% | 283 |
