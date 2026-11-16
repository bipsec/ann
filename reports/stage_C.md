# Stage C report: filtered search

## Provenance

- Date: 2026-11-05T01:52:00+00:00
- Git commit: `fb49bad79afa291bc645e94b8e655c0428581d6c-dirty`
- Hardware id: `1d3d9b66e7d3e1ac` (AMD EPYC-Milan Processor, 16 cores, 58.8 GB)
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

22 of 46 (index, cell, correlation, target) combinations have a crossover inside the measured S range; elsewhere one strategy dominates throughout.

| index_type   |       N |    D | correlation   |    R |   crossover_S |   crossover_S_lo |   crossover_S_hi |
|:-------------|--------:|-----:|:--------------|-----:|--------------:|-----------------:|-----------------:|
| hnsw         |  500000 |  384 | none          | 0.9  |    0.0119332  |       0.0118835  |       0.0119837  |
| hnsw         |  500000 |  384 | none          | 0.95 |    0.0127548  |       0.0126864  |       0.0128006  |
| hnsw         |  500000 |  384 | positive      | 0.9  |    0.00101981 |       0.00101306 |       0.00102565 |
| ivf_flat     |  500000 |  384 | positive      | 0.9  |    0.00120843 |       0.00119823 |       0.00121834 |
| hnsw         |  500000 |  384 | positive      | 0.95 |    0.00126638 |       0.00125416 |       0.0012774  |
| ivf_flat     |  500000 |  384 | positive      | 0.95 |    0.00129028 |       0.0012771  |       0.00130263 |
| hnsw         |  500000 |  384 | positive      | 0.99 |    0.00251441 |       0.00247048 |       0.00256095 |
| ivf_flat     |  500000 |  384 | positive      | 0.99 |    0.00235542 |       0.00230691 |       0.00241556 |
| hnsw         |  500000 | 1536 | none          | 0.95 |    0.0105856  |       0.0105673  |       0.0106039  |
| ivf_flat     |  500000 | 1536 | positive      | 0.9  |    0.00131075 |       0.00129747 |       0.00132396 |
| hnsw         |  500000 | 1536 | positive      | 0.95 |    0.00105266 |       0.00104743 |       0.00105878 |
| ivf_flat     |  500000 | 1536 | positive      | 0.95 |    0.00136723 |       0.00134919 |       0.00138513 |
| hnsw         |  500000 | 1536 | positive      | 0.99 |    0.0015828  |       0.00156301 |       0.00160292 |
| ivf_flat     |  500000 | 1536 | positive      | 0.99 |    0.00162751 |       0.00160062 |       0.00165384 |
| hnsw         | 1000000 |  384 | none          | 0.95 |    0.0100355  |       0.0100192  |       0.0100563  |
| ivf_flat     | 1000000 |  384 | positive      | 0.9  |    0.00198682 |       0.00193573 |       0.00203641 |
| ivf_flat     | 1000000 |  384 | positive      | 0.95 |    0.00203117 |       0.00195857 |       0.00209325 |
| hnsw         | 1000000 |  384 | positive      | 0.99 |    0.00208224 |       0.00205291 |       0.00211017 |
| ivf_flat     | 1000000 |  384 | positive      | 0.99 |    0.00413867 |       0.00397833 |       0.00428426 |
| ivf_flat     | 1000000 | 1536 | none          | 0.9  |    0.00113838 |       0.00113779 |       0.00113899 |
| hnsw         | 1000000 | 1536 | positive      | 0.99 |    0.00112451 |       0.00111489 |       0.00113333 |
| ivf_flat     | 1000000 | 1536 | positive      | 0.99 |    0.00111699 |       0.00110208 |       0.00113076 |

- median S* (none): 0.01059
- median S* (positive): 0.00137

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
