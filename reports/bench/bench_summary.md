# Systems benchmark results

## E1. Unfiltered kNN (100,000 vectors)

| Index | Setting | Recall@10 | p50 ms | p95 ms | Build s | Size MB |
|---|---|---|---|---|---|---|
| exact | - | 1.000 | 38.79 | 41.51 | 0 | 0 |
| ivfflat | probes=1 | 0.137 | 0.48 | 0.65 | 14 | 165 |
| ivfflat | probes=5 | 0.482 | 1.43 | 1.64 | 14 | 165 |
| ivfflat | probes=10 | 0.664 | 2.55 | 3.17 | 14 | 165 |
| ivfflat | probes=20 | 0.787 | 4.89 | 5.71 | 14 | 165 |
| ivfflat | probes=40 | 0.883 | 10.00 | 11.81 | 14 | 165 |
| hnsw | ef_search=10 | 0.655 | 0.29 | 0.50 | 51 | 205 |
| hnsw | ef_search=20 | 0.824 | 0.39 | 0.49 | 51 | 205 |
| hnsw | ef_search=40 | 0.943 | 0.59 | 0.71 | 51 | 205 |
| hnsw | ef_search=80 | 0.988 | 1.00 | 1.13 | 51 | 205 |
| hnsw | ef_search=160 | 0.999 | 1.79 | 2.03 | 51 | 205 |

## E2. Filtered kNN (predicate keeps the given fraction of rows)

| Index | Setting | recall @ 0.5 | recall @ 0.1 | recall @ 0.01 | recall @ 0.001 | short results @ 0.001 |
|---|---|---|---|---|---|---|
| exact | - | 1.000 | 1.000 | 1.000 | 1.000 | 0/150 |
| ivfflat | probes=1 | 0.184 | 0.144 | 0.092 | 0.023 | 150/150 |
| ivfflat | probes=5 | 0.469 | 0.403 | 0.270 | 0.118 | 150/150 |
| ivfflat | probes=10 | 0.619 | 0.558 | 0.426 | 0.231 | 146/150 |
| ivfflat | probes=20 | 0.770 | 0.731 | 0.587 | 0.455 | 27/150 |
| ivfflat | probes=40 | 0.880 | 0.860 | 0.741 | 0.655 | 0/150 |
| hnsw | ef_search=10 | 0.471 | 0.107 | 0.007 | 0.000 | 150/150 |
| hnsw | ef_search=20 | 0.752 | 0.202 | 0.014 | 0.000 | 150/150 |
| hnsw | ef_search=40 | 0.913 | 0.354 | 0.024 | 0.001 | 150/150 |
| hnsw | ef_search=80 | 0.978 | 0.678 | 0.047 | 0.003 | 150/150 |
| hnsw | ef_search=160 | 0.998 | 0.949 | 0.127 | 0.003 | 150/150 |
| hnsw+iterative | ef_search=40 | 0.911 | 0.946 | 0.972 | 0.840 | 0/150 |
| hnsw+iterative | ef_search=160 | 0.999 | 0.965 | 0.973 | 0.845 | 0/150 |

## E3. Update churn (50,000 vectors, 10% replaced per round, drifting population)

| Variant | Round 0 recall | Final recall | Final p50 ms | Update rows/s (mean) | Maintenance s/round | Index MB start → end |
|---|---|---|---|---|---|---|
| hnsw | 0.945 | 0.994 | 0.59 | 202 | 0.0 | 102 → 184 |
| hnsw+vacuum | 0.957 | 0.998 | 0.58 | 194 | 150.8 | 102 → 113 |
| ivfflat | 0.591 | 0.989 | 17.66 | 11,330 | 0.0 | 83 → 149 |
| ivfflat+reindex | 0.548 | 1.000 | 9.86 | 10,805 | 4.7 | 83 → 83 |

