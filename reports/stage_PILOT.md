# Stage PILOT report

## Provenance

- Date: 2026-10-17T19:10:21.275587+00:00
- Git commit: `aa713c5d6d4d628b605141c07514c6ac6de3b7c2-dirty`
- Hardware id: `1d3d9b66e7d3e1ac` (AMD EPYC-Milan Processor, 16 cores, 58.8 GB)
- Source files: `results/raw/builds`, `results/raw/sweeps`, `results/tuned/tuned.parquet`

## Builds

65 build rows across 4 (N,D) cells, index types: ['flat', 'hnsw', 'ivf_flat', 'ivf_pq', 'ivf_pq_refine'].

## Tuned winners (cheapest setting reaching each R; measured on test queries)

| index_type    |      N |    D |    R |   ef_search |   nprobe |   k_factor |   recall_test | meets_R_on_test   |   lat_p95_us |       qps |   rss_bytes |   index_bytes |   build_seconds_mean |   n_builds |
|:--------------|-------:|-----:|-----:|------------:|---------:|-----------:|--------------:|:------------------|-------------:|----------:|------------:|--------------:|---------------------:|-----------:|
| flat          | 100000 |  768 | 0.9  |         nan |      nan |        nan |      1        | True              |    21043.7   |  2837.99  | 3.15601e+08 |   3.072e+08   |            0.304936  |          3 |
| flat          | 100000 |  768 | 0.95 |         nan |      nan |        nan |      1        | True              |    21043.7   |  2837.99  | 3.15601e+08 |   3.072e+08   |            0.304936  |          3 |
| flat          | 100000 |  768 | 0.99 |         nan |      nan |        nan |      1        | True              |    21043.7   |  2837.99  | 3.15601e+08 |   3.072e+08   |            0.304936  |          3 |
| flat          | 100000 | 1536 | 0.9  |         nan |      nan |        nan |      1        | True              |    43182.1   |  1716.04  | 6.22445e+08 |   6.144e+08   |            0.672872  |          3 |
| flat          | 100000 | 1536 | 0.95 |         nan |      nan |        nan |      1        | True              |    43182.1   |  1716.04  | 6.22445e+08 |   6.144e+08   |            0.672872  |          3 |
| flat          | 100000 | 1536 | 0.99 |         nan |      nan |        nan |      1        | True              |    43182.1   |  1716.04  | 6.22445e+08 |   6.144e+08   |            0.672872  |          3 |
| flat          | 100000 |  384 | 0.9  |         nan |      nan |        nan |      1        | True              |    13596.1   |  3804.66  | 1.59076e+08 |   1.536e+08   |            0.169129  |          3 |
| flat          | 100000 |  384 | 0.95 |         nan |      nan |        nan |      1        | True              |    13596.1   |  3804.66  | 1.59076e+08 |   1.536e+08   |            0.169129  |          3 |
| flat          | 100000 |  384 | 0.99 |         nan |      nan |        nan |      1        | True              |    13596.1   |  3804.66  | 1.59076e+08 |   1.536e+08   |            0.169129  |          3 |
| flat          | 100000 |  128 | 0.9  |         nan |      nan |        nan |      1        | True              |     3450.29  |  4813.27  | 5.53124e+07 |   5.12e+07    |            0.0507611 |          3 |
| flat          | 100000 |  128 | 0.95 |         nan |      nan |        nan |      1        | True              |     3450.29  |  4813.27  | 5.53124e+07 |   5.12e+07    |            0.0507611 |          3 |
| flat          | 100000 |  128 | 0.99 |         nan |      nan |        nan |      1        | True              |     3450.29  |  4813.27  | 5.53124e+07 |   5.12e+07    |            0.0507611 |          3 |
| hnsw          | 100000 | 1536 | 0.9  |          16 |      nan |        nan |      0.898475 | False             |      337.645 |   nan     | 6.29531e+08 |   6.28833e+08 |           65.8       |          3 |
| hnsw          | 100000 | 1536 | 0.95 |          32 |      nan |        nan |      0.9569   | True              |      490.482 |   nan     | 6.29531e+08 |   6.28833e+08 |           65.8       |          3 |
| hnsw          | 100000 | 1536 | 0.99 |          96 |      nan |        nan |      0.991125 | True              |     1134.06  |  2859.21  | 6.29531e+08 |   6.28833e+08 |           65.8       |          3 |
| hnsw          | 100000 |  384 | 0.9  |          24 |      nan |        nan |      0.923075 | True              |      174.838 |   nan     | 1.69071e+08 |   1.68033e+08 |           11.1414    |          3 |
| hnsw          | 100000 |  384 | 0.95 |          48 |      nan |        nan |      0.969362 | True              |      274.401 |   nan     | 1.69071e+08 |   1.68033e+08 |           11.1414    |          3 |
| hnsw          | 100000 |  384 | 0.99 |          96 |      nan |        nan |      0.988862 | False             |      473.512 | 13813.6   | 1.69071e+08 |   1.68033e+08 |           11.1414    |          3 |
| hnsw          | 100000 |  768 | 0.9  |          16 |      nan |        nan |      0.895537 | False             |      193.584 |   nan     | 3.22978e+08 |   3.21633e+08 |           26.9404    |          3 |
| hnsw          | 100000 |  768 | 0.95 |          32 |      nan |        nan |      0.9549   | True              |      292.13  |   nan     | 3.22978e+08 |   3.21633e+08 |           26.9404    |          3 |
| hnsw          | 100000 |  768 | 0.99 |          96 |      nan |        nan |      0.991187 | True              |      680.817 |  5425.08  | 3.22978e+08 |   3.21633e+08 |           26.9404    |          3 |
| hnsw          | 100000 |  128 | 0.9  |          48 |      nan |        nan |      0.931225 | True              |      174.919 |   nan     | 6.6433e+07  |   6.56333e+07 |            3.72369   |          3 |
| hnsw          | 100000 |  128 | 0.95 |          64 |      nan |        nan |      0.951963 | True              |      218.764 |   nan     | 6.6433e+07  |   6.56333e+07 |            3.72369   |          3 |
| hnsw          | 100000 |  128 | 0.99 |         192 |      nan |        nan |      0.990737 | True              |      564.149 | 30884     | 6.6433e+07  |   6.56333e+07 |            3.72369   |          3 |
| ivf_flat      | 100000 | 1536 | 0.9  |         nan |       32 |        nan |      0.916687 | True              |     3167.41  |   nan     | 6.20409e+08 |   6.19088e+08 |            4.46374   |          3 |
| ivf_flat      | 100000 | 1536 | 0.95 |         nan |       64 |        nan |      0.9515   | True              |     5504.79  |   nan     | 6.20409e+08 |   6.19088e+08 |            4.46374   |          3 |
| ivf_flat      | 100000 | 1536 | 0.99 |         nan |      256 |        nan |      0.992513 | True              |    17275.7   |   127.093 | 6.20409e+08 |   6.19088e+08 |            4.46374   |          3 |
| ivf_flat      | 100000 |  128 | 0.9  |         nan |       32 |        nan |      0.912687 | True              |      203.213 |   nan     | 5.30186e+07 |   5.23288e+07 |            0.640211  |          3 |
| ivf_flat      | 100000 |  128 | 0.95 |         nan |       64 |        nan |      0.959263 | True              |      407.477 |   nan     | 5.30186e+07 |   5.23288e+07 |            0.640211  |          3 |
| ivf_flat      | 100000 |  128 | 0.99 |         nan |      256 |        nan |      0.996925 | True              |     1719.15  |  8695.29  | 5.30186e+07 |   5.23288e+07 |            0.640211  |          3 |
| ivf_flat      | 100000 |  384 | 0.9  |         nan |       32 |        nan |      0.928563 | True              |      743.645 |   nan     | 1.56266e+08 |   1.55376e+08 |            1.55842   |          3 |
| ivf_flat      | 100000 |  384 | 0.95 |         nan |       64 |        nan |      0.963363 | True              |     1608.61  |   nan     | 1.56266e+08 |   1.55376e+08 |            1.55842   |          3 |
| ivf_flat      | 100000 |  384 | 0.99 |         nan |      256 |        nan |      0.995862 | True              |     5553.21  |   756.357 | 1.56266e+08 |   1.55376e+08 |            1.55842   |          3 |
| ivf_flat      | 100000 |  768 | 0.9  |         nan |       32 |        nan |      0.928113 | True              |     1409.73  |   nan     | 3.11165e+08 |   3.09947e+08 |            2.52139   |          3 |
| ivf_flat      | 100000 |  768 | 0.95 |         nan |       64 |        nan |      0.960838 | True              |     2692.91  |   nan     | 3.11165e+08 |   3.09947e+08 |            2.52139   |          3 |
| ivf_flat      | 100000 |  768 | 0.99 |         nan |      256 |        nan |      0.9952   | True              |     9112.09  |   293.524 | 3.11165e+08 |   3.09947e+08 |            2.52139   |          3 |
| ivf_pq_refine | 100000 |  384 | 0.9  |         nan |       32 |         16 |      0.89825  | False             |      329.319 | 36407.3   | 1.75481e+08 |   1.58169e+08 |            5.53388   |          3 |
| ivf_pq_refine | 100000 | 1536 | 0.9  |         nan |       32 |         16 |      0.905012 | True              |     1073     |   nan     | 6.94018e+08 |   6.30261e+08 |           18.6556    |          3 |
| ivf_pq_refine | 100000 | 1536 | 0.95 |         nan |      128 |         16 |      0.949788 | False             |     2645.96  |  4883.08  | 6.94018e+08 |   6.30261e+08 |           18.6556    |          3 |
| ivf_pq_refine | 100000 |  768 | 0.9  |         nan |       32 |         16 |      0.910663 | True              |      554.693 |   nan     | 3.48738e+08 |   3.15533e+08 |           11.0065    |          3 |
| ivf_pq_refine | 100000 |  768 | 0.95 |         nan |      128 |         16 |      0.9474   | False             |     1358.72  | 11254.1   | 3.48738e+08 |   3.15533e+08 |           11.0065    |          3 |

## Tuning-to-test generalization gap

6 of 41 tuned winners were selected on the tuning queries but fall short of R on the held-out test queries (worst -0.0045, median -0.0016). This is selection noise, not a harness fault: the reported recall is the test value, and these rows must not be cited as having achieved R.

| index_type    |      N |    D |    R |   recall_test |   recall_gap_vs_R |
|:--------------|-------:|-----:|-----:|--------------:|------------------:|
| hnsw          | 100000 |  768 | 0.9  |      0.895537 |        -0.0044625 |
| ivf_pq_refine | 100000 |  768 | 0.95 |      0.9474   |        -0.0026    |
| ivf_pq_refine | 100000 |  384 | 0.9  |      0.89825  |        -0.00175   |
| hnsw          | 100000 | 1536 | 0.9  |      0.898475 |        -0.001525  |
| hnsw          | 100000 |  384 | 0.99 |      0.988862 |        -0.0011375 |
| ivf_pq_refine | 100000 | 1536 | 0.95 |      0.949788 |        -0.0002125 |

## Unreachable (reachable = false)

| index_type    |      N |    D |    R |
|:--------------|-------:|-----:|-----:|
| ivf_pq        | 100000 |  384 | 0.9  |
| ivf_pq        | 100000 |  384 | 0.95 |
| ivf_pq        | 100000 |  384 | 0.99 |
| ivf_pq        | 100000 | 1536 | 0.9  |
| ivf_pq        | 100000 | 1536 | 0.95 |
| ivf_pq        | 100000 | 1536 | 0.99 |
| ivf_pq        | 100000 |  128 | 0.9  |
| ivf_pq        | 100000 |  128 | 0.95 |
| ivf_pq        | 100000 |  128 | 0.99 |
| ivf_pq        | 100000 |  768 | 0.9  |
| ivf_pq        | 100000 |  768 | 0.95 |
| ivf_pq        | 100000 |  768 | 0.99 |
| ivf_pq_refine | 100000 |  384 | 0.95 |
| ivf_pq_refine | 100000 |  384 | 0.99 |
| ivf_pq_refine | 100000 | 1536 | 0.99 |
| ivf_pq_refine | 100000 |  128 | 0.9  |
| ivf_pq_refine | 100000 |  128 | 0.95 |
| ivf_pq_refine | 100000 |  128 | 0.99 |
| ivf_pq_refine | 100000 |  768 | 0.99 |

## For the paper

- Feeds the Stage PILOT results section. Tuned winners above are the cheapest-at-recall configurations; cite `results/tuned/tuned.parquet`.
- Headline numbers (`recall_test`, `lat_p95_us`, `qps`) are measured on the 8K held-out test queries; parameters were chosen on tuning queries only (§0.8).
