# Stage C report: filtered search

## Provenance

- Date: 2026-11-17T00:12:04+00:00
- Git commit: `1980cef6a1d63ed03322b38ca8ba5fa388306854-dirty`
- Data hardware id: `1d3d9b66e7d3e1ac` (AMD EPYC-Milan Processor, 16 cores, 58.8 GB)
- Generated on: `5a5f956b01cd41d6` (aarch64; not the data host)
- Source: `results/raw/filter_runs.parquet`

924 rows over 308 jobs: 4 cells x 3 strategies x 11 filter variants x 3 recall targets. 584 reachable, 340 unreachable.

## Acceptance checks

- Pre-filter exact scan recall is 1.0 at every selectivity: **PASS** (min 1.000000 over 132 rows)
- Mean measured selectivity within 20% of target: **FAIL at 2 level(s)**

| correlation   |   target_S |   measured_S_mean |   rel_err_pct |
|:--------------|-----------:|------------------:|--------------:|
| negative      |      0.001 |          0.001544 |     54.368    |
| negative      |      0.01  |          0.010542 |      5.42491  |
| negative      |      0.1   |          0.100543 |      0.543085 |
| none          |      0.001 |          0.000975 |     -2.5      |
| none          |      0.01  |          0.010065 |      0.65     |
| none          |      0.1   |          0.099884 |     -0.1155   |
| none          |      0.5   |          0.499852 |     -0.0297   |
| none          |      1     |          1        |      0        |
| positive      |      0.001 |          0.001611 |     61.1157   |
| positive      |      0.01  |          0.010657 |      6.57089  |
| positive      |      0.1   |          0.100668 |      0.668187 |

The failing levels are the smallest correlated targets. Correlated predicates union whole k-means clusters, so with 1,000 clusters the smallest admissible union is already near 0.1% of the corpus and a 0.001 target cannot be hit from above. S* at that level applies to the measured selectivity rather than the nominal one.

## Reachability by strategy and correlation

| strategy              | correlation   |   attempts |   reachable |   unreachable |
|:----------------------|:--------------|-----------:|------------:|--------------:|
| in_index_filtered     | negative      |         72 |           0 |            72 |
| in_index_filtered     | none          |        120 |          81 |            39 |
| in_index_filtered     | positive      |         72 |          72 |             0 |
| post_filter_overfetch | negative      |        144 |           0 |           144 |
| post_filter_overfetch | none          |        240 |         155 |            85 |
| post_filter_overfetch | positive      |        144 |         144 |             0 |
| pre_filter_exact      | negative      |         36 |          36 |             0 |
| pre_filter_exact      | none          |         60 |          60 |             0 |
| pre_filter_exact      | positive      |         36 |          36 |             0 |

**Under anti-correlated predicates, approximate search never reaches the recall target**: 0 of 216 attempts across in-index filtered search and post-filter over-fetch, at any swept effort. Pre-filter exact scan succeeds in every case. This is structural: both index families concentrate effort near the query, which is exactly where such a predicate admits nothing.

## Crossover S*

20 of 46 (index, cell, correlation, target) combinations have a crossover inside the measured S range; elsewhere one strategy dominates throughout.

| index_type   |       N |    D | correlation   |    R |   crossover_S |   crossover_S_lo |   crossover_S_hi |
|:-------------|--------:|-----:|:--------------|-----:|--------------:|-----------------:|-----------------:|
| hnsw         |  500000 |  384 | none          | 0.9  |    0.0186477  |       0.0175505  |       0.0198458  |
| hnsw         |  500000 |  384 | none          | 0.95 |    0.0221351  |       0.0206647  |       0.0237402  |
| hnsw         |  500000 |  384 | positive      | 0.9  |    0.00169006 |       0.0016091  |       0.00194356 |
| ivf_flat     |  500000 |  384 | positive      | 0.9  |    0.002679   |       0.00222757 |       0.00324686 |
| hnsw         |  500000 |  384 | positive      | 0.95 |    0.00256679 |       0.00225345 |       0.0029161  |
| ivf_flat     |  500000 |  384 | positive      | 0.95 |    0.00306705 |       0.00250014 |       0.00373809 |
| hnsw         |  500000 |  384 | positive      | 0.99 |    0.00503302 |       0.0044433  |       0.005694   |
| ivf_flat     |  500000 |  384 | positive      | 0.99 |    0.00584227 |       0.00482845 |       0.00695104 |
| hnsw         |  500000 | 1536 | none          | 0.95 |    0.01206    |       0.011095   |       0.0130914  |
| ivf_flat     |  500000 | 1536 | positive      | 0.9  |    0.00324098 |       0.00282216 |       0.00373993 |
| hnsw         |  500000 | 1536 | positive      | 0.95 |    0.00189997 |       0.00171402 |       0.00210851 |
| ivf_flat     |  500000 | 1536 | positive      | 0.95 |    0.00377211 |       0.00317444 |       0.00446717 |
| hnsw         |  500000 | 1536 | positive      | 0.99 |    0.00354859 |       0.00321178 |       0.00393523 |
| ivf_flat     |  500000 | 1536 | positive      | 0.99 |    0.00469104 |       0.00398897 |       0.00554959 |
| hnsw         | 1000000 |  384 | none          | 0.95 |    0.0101422  |       0.0100324  |       0.0113538  |
| ivf_flat     | 1000000 |  384 | positive      | 0.9  |    0.00437004 |       0.00372618 |       0.00512559 |
| ivf_flat     | 1000000 |  384 | positive      | 0.95 |    0.00456624 |       0.00380475 |       0.0054964  |
| hnsw         | 1000000 |  384 | positive      | 0.99 |    0.00349686 |       0.00314681 |       0.00390235 |
| ivf_flat     | 1000000 |  384 | positive      | 0.99 |    0.0106346  |       0.00848538 |       0.0133945  |
| hnsw         | 1000000 | 1536 | positive      | 0.99 |    0.00204866 |       0.00184174 |       0.00227217 |

- median S* (none): 0.01535
- median S* (positive): 0.00352

## Unreachable entries (reachable = false)

340 rows; grouped:

| strategy              | index_type   | correlation   |   target_S |    R |   n |
|:----------------------|:-------------|:--------------|-----------:|-----:|----:|
| in_index_filtered     | hnsw         | negative      |      0.001 | 0.9  |   4 |
| in_index_filtered     | hnsw         | negative      |      0.001 | 0.95 |   4 |
| in_index_filtered     | hnsw         | negative      |      0.001 | 0.99 |   4 |
| in_index_filtered     | hnsw         | negative      |      0.01  | 0.9  |   4 |
| in_index_filtered     | hnsw         | negative      |      0.01  | 0.95 |   4 |
| in_index_filtered     | hnsw         | negative      |      0.01  | 0.99 |   4 |
| in_index_filtered     | hnsw         | negative      |      0.1   | 0.9  |   4 |
| in_index_filtered     | hnsw         | negative      |      0.1   | 0.95 |   4 |
| in_index_filtered     | hnsw         | negative      |      0.1   | 0.99 |   4 |
| in_index_filtered     | hnsw         | none          |      0.001 | 0.9  |   4 |
| in_index_filtered     | hnsw         | none          |      0.001 | 0.95 |   4 |
| in_index_filtered     | hnsw         | none          |      0.001 | 0.99 |   4 |
| in_index_filtered     | hnsw         | none          |      0.01  | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.001 | 0.9  |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.001 | 0.95 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.001 | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.01  | 0.9  |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.01  | 0.95 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.01  | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.1   | 0.9  |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.1   | 0.95 |   4 |
| in_index_filtered     | ivf_flat     | negative      |      0.1   | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | none          |      0.001 | 0.9  |   3 |
| in_index_filtered     | ivf_flat     | none          |      0.001 | 0.95 |   4 |
| in_index_filtered     | ivf_flat     | none          |      0.001 | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | none          |      0.01  | 0.95 |   3 |
| in_index_filtered     | ivf_flat     | none          |      0.01  | 0.99 |   4 |
| in_index_filtered     | ivf_flat     | none          |      0.1   | 0.99 |   3 |
| in_index_filtered     | ivf_flat     | none          |      0.5   | 0.99 |   2 |
| post_filter_overfetch | hnsw         | negative      |      0.001 | 0.9  |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.001 | 0.95 |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.001 | 0.99 |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.01  | 0.9  |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.01  | 0.95 |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.01  | 0.99 |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.1   | 0.9  |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.1   | 0.95 |   8 |
| post_filter_overfetch | hnsw         | negative      |      0.1   | 0.99 |   8 |
| post_filter_overfetch | hnsw         | none          |      0.001 | 0.9  |   8 |
| post_filter_overfetch | hnsw         | none          |      0.001 | 0.95 |   8 |
| post_filter_overfetch | hnsw         | none          |      0.001 | 0.99 |   8 |
| post_filter_overfetch | hnsw         | none          |      0.01  | 0.95 |   1 |
| post_filter_overfetch | hnsw         | none          |      0.01  | 0.99 |   8 |
| post_filter_overfetch | hnsw         | none          |      0.1   | 0.99 |   4 |
| post_filter_overfetch | ivf_flat     | negative      |      0.001 | 0.9  |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.001 | 0.95 |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.001 | 0.99 |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.01  | 0.9  |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.01  | 0.95 |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.01  | 0.99 |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.1   | 0.9  |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.1   | 0.95 |   8 |
| post_filter_overfetch | ivf_flat     | negative      |      0.1   | 0.99 |   8 |
| post_filter_overfetch | ivf_flat     | none          |      0.001 | 0.9  |   6 |
| post_filter_overfetch | ivf_flat     | none          |      0.001 | 0.95 |   8 |
| post_filter_overfetch | ivf_flat     | none          |      0.001 | 0.99 |   8 |
| post_filter_overfetch | ivf_flat     | none          |      0.01  | 0.95 |   6 |
| post_filter_overfetch | ivf_flat     | none          |      0.01  | 0.99 |   8 |
| post_filter_overfetch | ivf_flat     | none          |      0.1   | 0.99 |   7 |
| post_filter_overfetch | ivf_flat     | none          |      0.5   | 0.99 |   5 |

## For the paper

- Feeds sections 5.7 and 5.8: the crossover and the anti-correlation failure. Cite `results/summary/crossover.csv`, `crossover_h2.csv` and `results/raw/filter_runs.parquet`.
- Every row records `base_index_run_id`, the Stage B index that answered it.
