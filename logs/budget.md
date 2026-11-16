# Compute Budget

Estimates and actuals for long-running jobs. Updated before each major run.

## Measured anchors (this host, from the pilot)

| index_type    |       N |    D |   build_seconds |   threads |   n_settings |   mean_ms_per_query |
|:--------------|--------:|-----:|----------------:|----------:|-------------:|--------------------:|
| flat          |  100000 |  128 |           0.049 |        16 |            7 |               3.457 |
| flat          |  500000 |  128 |           0.264 |        16 |            6 |              17.548 |
| flat          |  100000 |  384 |           0.166 |        16 |            7 |              11.454 |
| flat          |  500000 |  384 |           0.669 |        16 |            6 |              58.731 |
| flat          | 1000000 |  384 |           1.270 |        16 |            5 |             107.137 |
| flat          |  100000 |  768 |           0.297 |        16 |            7 |              21.681 |
| flat          |  500000 |  768 |           1.350 |        16 |            6 |             104.267 |
| flat          |  100000 | 1536 |           0.640 |        16 |            7 |              40.936 |
| flat          |  500000 | 1536 |           2.424 |        16 |            6 |             208.745 |
| flat          | 1000000 | 1536 |           6.556 |        16 |            5 |             380.480 |
| hnsw          |  100000 |  128 |           3.758 |        16 |           25 |               0.417 |
| hnsw          |  500000 |  128 |          47.573 |        16 |           54 |               1.022 |
| hnsw          |  100000 |  384 |          38.461 |        16 |           51 |               1.014 |
| hnsw          |  500000 |  384 |         295.733 |        16 |           54 |               1.394 |
| hnsw          | 1000000 |  384 |         227.231 |        16 |           51 |               1.441 |
| hnsw          |  100000 |  768 |          26.999 |        16 |           25 |               0.771 |
| hnsw          |  500000 |  768 |         444.691 |        16 |           54 |               1.741 |
| hnsw          |  100000 | 1536 |         135.981 |        16 |           51 |               2.763 |
| hnsw          |  500000 | 1536 |         315.313 |        16 |           54 |               2.312 |
| hnsw          | 1000000 | 1536 |         474.056 |        16 |           51 |               2.420 |
| ivf_flat      |  100000 |  128 |           0.645 |        16 |           22 |               0.544 |
| ivf_flat      |  500000 |  128 |          22.552 |        16 |           45 |               0.882 |
| ivf_flat      |  100000 |  384 |           3.004 |        16 |           51 |               1.292 |
| ivf_flat      |  500000 |  384 |          31.652 |        16 |           51 |               2.908 |
| ivf_flat      | 1000000 |  384 |         151.705 |        16 |           42 |               3.755 |
| ivf_flat      |  100000 |  768 |           2.542 |        16 |           22 |               3.739 |
| ivf_flat      |  500000 |  768 |          90.632 |        16 |           25 |               3.885 |
| ivf_flat      |  100000 | 1536 |           9.787 |        16 |           51 |               6.102 |
| ivf_flat      |  500000 | 1536 |         101.004 |        16 |           45 |              11.503 |
| ivf_flat      | 1000000 | 1536 |         165.552 |        16 |           42 |              14.128 |
| ivf_pq        |  100000 |  128 |           6.127 |         1 |           10 |               0.229 |
| ivf_pq        |  500000 |  128 |          63.320 |         1 |           90 |               0.523 |
| ivf_pq        |  100000 |  384 |          25.423 |         1 |           90 |               0.880 |
| ivf_pq        |  500000 |  384 |         167.859 |         1 |           90 |               1.519 |
| ivf_pq        | 1000000 |  384 |         439.060 |         1 |           90 |               1.853 |
| ivf_pq        |  100000 |  768 |          36.080 |         1 |           10 |               0.972 |
| ivf_pq        |  500000 |  768 |         323.805 |         1 |           70 |               2.485 |
| ivf_pq        |  100000 | 1536 |          98.508 |         1 |           90 |               4.295 |
| ivf_pq        |  500000 | 1536 |         640.447 |         1 |           90 |               6.700 |
| ivf_pq        | 1000000 | 1536 |        1736.018 |         1 |           90 |               8.661 |
| ivf_pq_refine |  100000 |  128 |           6.338 |         1 |           20 |               0.232 |
| ivf_pq_refine |  500000 |  128 |          28.419 |        16 |          195 |               0.595 |
| ivf_pq_refine |  100000 |  384 |          13.416 |        16 |          194 |               0.949 |
| ivf_pq_refine |  500000 |  384 |          64.452 |        16 |          195 |               1.590 |
| ivf_pq_refine | 1000000 |  384 |         163.642 |        16 |          192 |               1.933 |
| ivf_pq_refine |  100000 |  768 |          11.208 |        16 |           29 |               1.033 |
| ivf_pq_refine |  500000 |  768 |          89.645 |        16 |          201 |               3.064 |
| ivf_pq_refine |  100000 | 1536 |          72.184 |        16 |          193 |               4.219 |
| ivf_pq_refine |  500000 | 1536 |         190.651 |        16 |          195 |               7.351 |
| ivf_pq_refine | 1000000 | 1536 |         573.084 |        16 |          192 |               9.081 |

## Projection

Latency sweeps are single-threaded and run one per core, so their cost is divided by 16 cores. The T-thread build/throughput pass uses every core and therefore runs sequentially — it is NOT divided.

| Stage | Specs/jobs | Sweep core-h | /16 cores | T-thread build h (serial) | Re-measure core-h | **Projected wall h** |
|---|---|---|---|---|---|---|
| stage_a | 100 | 12.5 | 0.8 | 4.3 | 0.2 | **5.3** |
| stage_b | 150 | 27.3 | 1.7 | 10.3 | 0.3 | **12.3** |
| stage_c | 308 | 75.0 | 4.7 | 0.0 | 0.0 | **4.7** |
| stage_d | 3 | 0.8 | 0.0 | 0.7 | 0.0 | **0.8** |
| **total** | | | | | | **23.0** |

A single job cannot be split across shards, so its duration is a hard floor on that stage's wall time regardless of how many cores are used:
- `stage_c`: longest single job 5.7 h

## Calibration against measured actuals

| Stage | Projected by this model | Actual | Ratio |
|---|---|---|---|
| stage_a | 5.2 h | 7.7 h | **1.48x** |

The model is optimistic by up to **1.48x**, chiefly because effective parallelism is memory-limited rather than core-limited: admission control runs 6-12 concurrent jobs at D=1536, not 16. Multiply the projections above by ~1.5 for a planning figure:
- `stage_a`: 7.8 h calibrated
- `stage_b`: 18.2 h calibrated
- `stage_c`: 6.9 h calibrated
- `stage_d`: 1.1 h calibrated

- stage_a: 0 failures; wall clock was 10.6 h including a 2.5 h stall from a disk-full fault, excluded here

## Assumptions

- Build cost scales with N as: flat: N^0.88, hnsw: N^1.0, ivf_flat: N^1.86, ivf_pq: N^1.0, ivf_pq_refine: N^0.95
- Per-query search cost scales with N as: flat: N^1.0, ivf_flat: N^0.24, ivf_pq: N^0.62, ivf_pq_refine: N^0.64, hnsw: N^0.4
- 2000 tuning queries, 8000 test queries, 3 passes.
- Every D is measured directly by the pilot, so only N is extrapolated.
- Build anchors for ivf_pq@D128, ivf_pq@D384, ivf_pq@D768, ivf_pq@D1536, ivf_pq@D128, ivf_pq@D384, ivf_pq@D768, ivf_pq@D1536, ivf_pq@D384, ivf_pq@D1536, ivf_pq_refine@D128 are SINGLE-threaded (those configurations never won a recall target, so no T-thread build exists). Their build projections are therefore conservative by roughly the 3-6x threading speedup measured for the other index types.
- Excludes filtered ground-truth computation for Stage C and the index-only memory pass; both are small next to the measurement itself.
- A projection is not a measurement. Treat these as the input to the Gate 3 go / trim decision, and record actuals in the table below as stages run.

## Actuals

| Date | Job | Estimated | Actual | Notes |
|------|-----|-----------|--------|-------|
| 2026-10-16 | GT N=500K D=1536 (compute+verify) | — | 36s compute, +193s verify | FAISS flat L2, 10K queries, k=100, 24 cores; verify = 100-query numpy brute force (N≤500K only) |
| 2026-10-16 | GT N=1M all 4 D (compute only) | ~3 min | 274.7s (4.6 min) | Completes Phase 1 ground truth; all 12 (N,D) GT files now present |
| 2026-10-17 | PILOT pass1 (20 builds, 216 search settings, 16 pinned cores) | — | 12 min | N=100K, all 4 D, 1 build variant per index; 0 failures |
| 2026-10-17 | PILOT repeats + memory (45 T-thread builds, 45 memory jobs) | — | 35 min | 3 builds per winner, sequential at T=16; index-only RSS |
| 2026-10-17 | PILOT test re-measure (15 winner builds, 8K test queries) | — | 18 min | single-thread pinned, 16-way pool |
| 2026-10-17 | PILOT total (incl. tune/report/analysis) | — | ~70 min | Whole pipeline end to end; projection table above is derived from it |
| 2026-10-17 | acceptance --checks all (incl. efConstruction 100/200/400) | — | ~4 min | 11/11 pass |
| 2026-10-17/18 | **STAGE A actual** (N=500K, 100 builds, 1360 search settings) | 9.4 h | **~7.7 h compute** (10.6 h wall) | Under projection. Wall includes a 2.5 h stall from the disk-full crash. Phases: pass1 3.4 h, repeats 2.35 h (126 T-thread builds), memory backfill 0.2 h, remeasure 1.7 h. 0 failures |
| 2026-10-18 | Stage A remeasure: Flat D=1536 single job | — | ~24 min observed, 91 min predicted | 8K test queries x 3 passes; this job is the hard floor on the remeasure phase and cannot be sharded |
| 2026-10-18/19 | **STAGE B actual** (N in {100K,500K,1M} x D in {384,1536}, 90 new specs of 150) | 16.0 h calibrated | **~17 h** | 0 failures. pass1 9.1 h (N=1M/D=1536 jobs 3700-4240 s each, 2-3 concurrent at 16-19 GB), repeats 3.2 h (180 builds, 60 winners), remeasure 4.6 h. 2 builds skipped on disk and later backfilled |
| 2026-10-25/26 | **STAGE C actual** (308 jobs, 4 cells x 3 strategies x 11 filters) | 7.0 h calibrated | **~27 h** | 0 failures after a cache-race fix. Far over projection: the N=1M cells admit only 3-12 shards because each holds the corpus AND an index, so there is much less parallelism than at the small cells. 20 jobs were lost to a cache write race and re-run |
| 2026-10-28 | **STAGE D actual** (3 index types x 10 insert batches + rebuild) | 1.1 h calibrated | **~4.5 h** | Re-run after the first pass measured completeness rather than drift. Cost dominated by the single-threaded p95 measurement over 8K test queries after every batch |
| 2026-10-28 | **WHOLE STUDY** | — | **~56 h compute** | Pilot 1.2 h + A 7.7 h + B 17 h + C 27 h + D 4.5 h, 0 unexplained failures |

