# EXPERIMENT_PLAN.md: ANN index selection study

Execution plan for Claude Code. The research rationale lives in the companion workshop-paper document; this file says what to build, run, and check, in order.

**Study in one sentence.** For each workload cell (N vectors, D dimensions, filter selectivity S, required recall R, memory budget M), find the cheapest configuration of Flat, HNSW, IVF-Flat, and IVF-PQ that reaches recall@10 >= R, then find where the winner changes and fit a small decision rule that is tested on held-out cells.

---

## 0. Rules for the agent (read first)

1. Work phase by phase. Do not start a phase until the previous phase's acceptance checks pass. Phases ending in **GATE** require a human decision before you continue. Stop, summarize in under 15 lines, and wait.
2. Never invent a number. Every figure in a report, table, or paper draft must come from a file under `results/`. If a result is missing, say so.
3. Before any job expected to take more than 10 minutes, estimate its runtime from a measured small run and write the estimate in `logs/budget.md`. Ask the human before any job projected above 2 hours, and before any paid API call.
4. Every run writes its config, git commit, hardware snapshot, library versions, and seeds next to its results. A result without this metadata does not count.
5. Checkpoint everything long-running (embedding shards, one result row per finished configuration). Every script must be resumable and must skip work already recorded.
6. Do not edit files under `results/raw/`. Fix problems by rerunning, and mark superseded rows with a `superseded_by` column instead of deleting them.
7. Verify library APIs against the installed version (`help(...)`, `dir(...)`) before relying on them. FAISS filtering APIs in particular differ across versions. Do not trust memory of function signatures.
8. Keep tuning and test queries strictly separate (section 3). Parameters are chosen on tuning queries only. Reported numbers come from test queries only.
9. If a result looks too good or contradicts published behavior (for example HNSW recall of 1.0 at efSearch = 16), assume a harness bug first and investigate before continuing.
10. Prefer small, tested modules over one large script. Run unit tests after every change to the harness.

---

## 1. Fixed decisions

| Item | Value |
| --- | --- |
| Corpus | MS MARCO passages; 1,000,000 passages sampled uniformly with seed 0; nested prefixes of that sample for N = 100K and 500K |
| Queries | 10,000 sampled uniformly (seed 1) from MS MARCO training queries; 2,000 tuning queries, 8,000 test queries |
| Embedding | One Matryoshka-trained model; keep first 1536 dims; derive D = 768, 384, 128 by prefix slice and L2 renormalization |
| N values | 100K, 500K, 1M |
| D values | 128, 384, 768, 1536 |
| Metric | L2 on unit-norm vectors in every index (rank-equivalent to cosine and inner product) |
| k | 10 for all headline results; k in {1, 100} as a side check at N = 500K, D = 768 |
| Recall targets R | 0.90, 0.95, 0.99 |
| Memory budgets M | unbounded; 1.5x, 0.5x, and 0.1x of raw size (raw = N x D x 4 bytes) |
| Threads | latency: 1 thread pinned to 1 core; throughput and builds: T threads where T = min(physical cores, 16); record T |
| Indexes | Flat, HNSW, IVF-Flat, IVF-PQ (with and without refine); hnswlib as a cross-check at one cell |
| Excluded | DiskANN, GPU indexes, deletions, multi-node search |

Open decision for the human at Phase 1: which embedding model and route (open weights on GPU, or hosted API). Do not choose silently. See section 4, Phase 1.

---

## 2. Repository layout

```
ann-study/
  EXPERIMENT_PLAN.md
  README.md                  # reproduction steps, written last
  pyproject.toml             # pinned dependencies
  configs/
    data.yaml                # corpus, sample sizes, seeds, model
    stage_a.yaml  stage_b.yaml  stage_c.yaml  stage_d.yaml
  src/annstudy/
    data.py                  # load corpus, sampling, splits
    embed.py                 # sharded, resumable embedding
    truncate.py              # prefix slice + renormalize, LID estimation
    groundtruth.py           # exact top-100 per (N, D)
    indexes.py               # factories and build parameters
    runner.py                # build, save, measure, sweep (subprocess entry point)
    metrics.py               # recall, latency percentiles, bootstrap
    filters.py               # attribute generation and filter strategies
    updates.py               # insert study
    hardware.py              # hardware and version snapshot
    analysis/
      winners.py  scaling.py  crossover.py  rule.py  figures.py
  tests/
  data/                      # gitignored: raw text, embeddings, ground truth
  results/
    raw/                     # append-only parquet, one file per run
    tuned/                   # cheapest setting per (cell, index, R)
    summary/                 # derived tables
  figures/
  logs/
    budget.md  decisions.md  gate_reports/
  notes/related_work.md
```

Create `logs/decisions.md` at the start. Append one dated line for every non-obvious choice you make (library version, deviation from this plan, bug found).

---

## 3. Measurement protocol (applies to every phase)

**Query split.** Query ids 0..1999 are tuning, 2000..9999 are test. Store the split in `data/query_split.json` and assert it in the runner.

**Recall.** Ground truth is exact top-100 from a flat L2 index on the same truncated vectors, stored as ids and squared distances in `data/gt/gt_N{N}_D{D}.npz`. Compute recall@k by recomputing the true squared distance of each returned id from the raw vectors, then counting returned items whose true distance is within a relative tolerance of 1e-5 of the k-th ground-truth distance. This avoids penalizing equally good neighbors that tie. Never use the distances returned by a compressed index for scoring.

**Latency.**
- Run in a fresh subprocess pinned with `taskset -c <core>`, with `OMP_NUM_THREADS=1` and `faiss.omp_set_num_threads(1)`.
- Batch size 1. Warm up with 200 tuning queries first.
- Time each query with `time.perf_counter_ns()` around `index.search`. Do 3 passes over the test queries and take the per-query median across passes.
- Report mean, p50, p95, p99 over the 8,000 per-query values. Python call overhead is included for every index equally. State this in the paper.

**Throughput (QPS).** Same subprocess approach with T threads, queries in batches of 1,000, 3 passes, median QPS.

**Build time.** Fresh subprocess, T threads. Record train time, add time, and total separately. Record training sample size.

**Memory.** In a fresh subprocess: record RSS baseline after imports, load the index with `faiss.read_index` (no memory mapping), run the 200-query warm-up, record RSS again. RAM = difference. Also record serialized size in bytes with `os.path.getsize`.

**Repeats.**
- Build every build-parameter variant once.
- Rebuild the selected (tuned-winner) configuration of each index two more times with different seeds, so each reported build time and recall has 3 builds.
- Multi-threaded HNSW builds are not bit-reproducible even with a seed. Record the seed anyway and rely on the 3 repeats to show variance.

**Uncertainty.** Bootstrap over queries (1,000 resamples) for recall and latency percentiles. Store the 2.5th and 97.5th percentiles.

**Housekeeping.** Delete serialized indexes after measurement unless a later stage reuses them (Stage B indexes are reused by Stage C; keep those). Check free disk before every build and refuse to start if free space is below 2x the expected index size.

---

## 4. Phases

### Phase 0: Positioning (human-led, agent assists) **GATE**

Tasks
- Search for and read the closest prior work: ANN-Benchmarks, VIBE, the 2025 to 2026 filtered-ANN benchmark and system papers, any paper varying dimension on one corpus.
- Fill `notes/related_work.md` with one row per paper: what it varies, what it controls for, datasets, indexes, what it does not cover.
- Specific question to answer: does any existing paper measure how the filter-selectivity crossover between filtered ANN search and exact scan of the filtered subset changes with dimension, on a single corpus?

Acceptance: table exists with at least 10 papers, each with a link and a one-line verdict.

**GATE 0.** Human decides: keep the D x S headline, or shift the headline to the recall-constrained framing and decision rule. Record in `logs/decisions.md`.

### Phase 1: Data and ground truth

Tasks
1. `hardware.py`: write a snapshot (CPU model, physical and logical cores, RAM, OS, Python, library versions, git commit) to `results/hardware.json`. Check that RAM is at least 64 GB and warn if not.
2. Download MS MARCO passages and training queries. Check the license terms and record them in `README.md`. Sample 1,000,000 passages (seed 0) and 10,000 queries (seed 1). Save sampled ids and text.
3. **Ask the human** which embedding route to use. Candidate routes: an open-weights Matryoshka model that supports at least 1536 output dims (for example a Qwen3-Embedding variant; read the model card to confirm supported dims and the required query instruction format), or a hosted API model with a dimensions parameter. Before any paid or long GPU run, compute total tokens and estimated cost or GPU hours and wait for approval.
4. `embed.py`: embed passages and queries in shards of 10,000 with resume. Save the first 1536 dims per shard, then assemble `data/emb/passages_D1536.f32` and `queries_D1536.f32` (float32 memmap). Apply the model's documented query instruction to queries only if its card requires one.
5. `truncate.py`: create D = 768, 384, 128 files by slicing prefixes of the 1536 vectors and L2-normalizing. Also renormalize the 1536 file. Estimate local intrinsic dimensionality (MLE estimator with k = 20) on a 10,000-passage sample for each D and save to `results/summary/lid.csv`.
6. `groundtruth.py`: for each N in {100K, 500K, 1M} and D in {128, 384, 768, 1536}, compute exact top-100 for all 10,000 queries in batches, using FAISS flat L2 (or a GPU only if one is available and verified identical on a sample).

Acceptance checks
- Embedding files have shape (1,000,000, 1536) and (10,000, 1536) and contain no NaN or inf. All row norms are within 1e-4 of 1.
- The 100K and 500K sets are exact prefixes of the 1M set (assert on ids).
- Independent check: for 100 random queries, a plain numpy brute-force top-10 over the 100K set matches the stored ground truth ids (up to ties).
- LID table has 4 rows (one per D), and values are finite.
- Peek at the 5 nearest passages for 3 queries at D = 1536 and D = 128 and write a one-paragraph sanity note in `logs/decisions.md`. Neighbors at 1536 should be topically sensible.

### Phase 2: Harness

Tasks
1. `indexes.py`: factory functions using the FAISS Python classes for Flat, HNSW (`IndexHNSWFlat`), IVF-Flat (`IndexIVFFlat` over a flat L2 quantizer), IVF-PQ (`IndexIVFPQ`, 8 bits per code), and IVF-PQ with refine (`IndexRefineFlat` wrapping it, with a `k_factor` setting). Metric is L2 everywhere.
2. Build-parameter grids (table below). Training samples: IVF centroids use a random sample of min(N, 256 x nlist) vectors; PQ codebooks use a random sample of min(N, 256,000) vectors. Record both.
3. `runner.py`: a subprocess entry point taking a config path and one build spec. It builds, writes one row to `results/raw/builds/`, saves the index, then runs the measurement protocol of section 3 and writes search-sweep rows to `results/raw/sweeps/`.
4. `metrics.py`: recall by distance tolerance, latency percentiles, bootstrap intervals.
5. `tune.py`: from tuning-query sweeps, pick the cheapest search setting (lowest median latency) that reaches each R, then re-measure that setting on test queries and write `results/tuned/`. If no setting reaches R, write a row with `reachable = false`.
6. Unit tests in `tests/`: Flat recall equals 1.0; HNSW on a 5,000-vector toy set reaches recall >= 0.99 at high efSearch; recall function handles ties; latency timer returns plausible values; the split assertion trips if tuning and test ids overlap; RSS measurement increases after loading a known-size index.

Build and search parameter grids

| Index | Build parameters | Search parameters |
| --- | --- | --- |
| Flat | none | none |
| HNSW | M in {16, 32, 64}; efConstruction = 200 | efSearch in {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}, never below k |
| IVF-Flat | nlist = round(c x sqrt(N)) for c in {2, 4, 8} | nprobe in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, capped at nlist |
| IVF-PQ | same nlist values; m in {D/16, D/8, D/4}; nbits = 8 | same nprobe values |
| IVF-PQ + refine | as IVF-PQ | same nprobe values; k_factor in {4, 16} |

Check that every m divides D (it does for the D values above). IVF-PQ plus refine is reported as a separate index because it keeps full vectors in RAM.

Acceptance checks
- All unit tests pass.
- Run Flat, HNSW (M = 32), and IVF-Flat on N = 100K, D = 384. Recall is monotone non-decreasing in efSearch and nprobe (small bootstrap noise aside).
- HNSW reaches recall >= 0.95 at some efSearch <= 256. If not, investigate (normalization, metric, query split, bug) before continuing.
- A repeat of the same run yields p50 latency within 10% (if not, fix pinning and background load).

### Phase 3: Pilot and budget **GATE**

Tasks
1. Stage A at N = 100K for all four D with one build variant per index. Write `reports/pilot_curves.png` of recall versus QPS.
2. Compare shape and rough magnitude against published recall-versus-QPS curves for HNSW and IVF on comparable text-embedding data. Record the comparison and any mismatch in `logs/decisions.md`. A large unexplained mismatch is a harness bug.
3. Timing pilot at N = 1M, D = 1536: build one HNSW (M = 32, efConstruction = 200) and one IVF-PQ (nlist = 4 x sqrt(N), m = D/8). Record wall time and peak RSS. Extrapolate total compute for Stages A to D and write `logs/budget.md` with the number of builds, per-build estimates, and total hours.
4. Run the efConstruction check (100, 200, 400) at N = 100K, D = 384, and keep 200 unless the recall-versus-QPS result is clearly worse.

**GATE 3.** Human approves the budget or trims the plan. Allowed trims, in order: drop M = 64; drop the largest nlist multiplier; run D = 1536 at N <= 500K only; shorten the nprobe and efSearch sweeps; rent a larger machine for builds.

### Phase 4: Experiments

Run stages in order. Each stage has a config under `configs/` and appends to `results/raw/`. After each stage write `reports/stage_X.md` with a table of tuned results and a list of any `reachable = false` entries.

**Stage A: dimension (N = 500K).** D in {128, 384, 768, 1536}. All indexes in the grid table. Output: tuned cheapest setting per (D, index, R). Side check: k in {1, 100} at D = 768 only. Side check: hnswlib HNSW with the same M, efConstruction, efSearch at D = 768 to confirm FAISS-versus-hnswlib recall and latency agree closely.

Acceptance: 4 cells x all indexes complete; at least 3 builds recorded for each selected configuration; reachability table present.

**Stage B: scale.** N in {100K, 500K, 1M} x D in {384, 1536} (reuse Stage A results at N = 500K). Indexes: HNSW, IVF-Flat, IVF-PQ, IVF-PQ + refine, and Flat as the reference. Keep serialized HNSW and IVF-Flat indexes for N in {500K, 1M} at both D for Stage C.

Acceptance: 6 cells complete. Build time, index size, and RAM recorded for each index. Flat latency and memory recorded at every cell.

**Stage C: filtering.** Reuse Stage B indexes (the predicate is applied at search time, so no rebuild). N in {500K, 1M}, D in {384, 1536}.

Filter construction (`filters.py`)
- **Uncorrelated.** Assign each vector an integer attribute drawn uniformly from 0..9999 with a fixed seed. A range predicate `attr < t` gives selectivity S = t / 10000 for S in {1.0, 0.5, 0.1, 0.01, 0.001}.
- **Correlated.** Run k-means (k = 1000) on the D-dimensional vectors of the N-set. The attribute is the cluster id. For a query, positive correlation means the filter admits the cluster whose centroid is nearest to the query; negative correlation means a cluster chosen at random among the 50% farthest from the query. To reach S near 0.1, 0.01, 0.001, take unions of the nearest (or farthest) clusters until the target size is reached, using balanced selection by cluster size. Log the exact measured selectivity per query.
- Run correlated filters at S in {0.1, 0.01, 0.001} only.

Strategies compared (describe exactly in the paper)
1. **Pre-filter and exact scan.** Gather the matching vectors and compute exact top-k by brute force.
2. **In-index filtered search.** Pass an ID selector to the search call (`SearchParametersHNSW`, `SearchParametersIVF`, with `IDSelectorRange`, `IDSelectorBatch`, or `IDSelectorBitmap`). Confirm availability for HNSW in the installed version. If missing, use hnswlib's filter function for the HNSW rows and mark them `impl = hnswlib`.
3. **Post-filter with over-fetch.** Retrieve ceil(safety x k / S) candidates with safety in {1.5, 3}, drop non-matches, keep k.

Filtered ground truth: exact top-100 over the matching subset for every (query, filter). Compute it with the same recall function.

Protocol: for each strategy, index, filter, and S, sweep the search parameter on tuning queries, pick the cheapest setting reaching R in {0.90, 0.95, 0.99}, then measure on test queries. Record `reachable = false` where applicable. Also record recall at one fixed mid-range search effort for every S, to expose silent recall loss.

Acceptance
- At S = 1.0, in-index filtered search matches the unfiltered result within noise.
- Pre-filter exact scan has recall 1.0 at every S (assert).
- Measured selectivity per query is logged; mean measured S is within 20% of target for each correlated level.
- A crossover table exists: for each (N, D, index, R, correlation), the S at which the in-index strategy's latency first exceeds pre-filter exact scan, or `none in range`.

**Stage D: updates (N = 500K, D = 768).** Indexes: HNSW, IVF-Flat, IVF-PQ. Build on 450,000 vectors with parameters tuned for R = 0.95 in a quick pre-run. Insert the remaining 50,000 in 10 batches of 5,000. After each batch, record insert time and vectors per second, then recall@10 and p95 latency at the search parameters fixed before inserting. Compare with an index rebuilt from scratch on the same vectors at the end. For IVF indexes, never retrain centroids during the batches. Add one final point after retraining. Do not test deletions.

Acceptance: 10 batch rows per index plus a rebuild row; recall-drift curve per index.

### Phase 5: Analysis

Write scripts that read `results/tuned/` and `results/raw/` and write tables to `results/summary/` and figures to `figures/`. Each script is deterministic and rerunnable.

1. `winners.py`. For each (N, D, R, memory budget) cell, find the index minimizing p95 latency among reachable configurations that fit the memory budget (RAM measured, compared against budget x raw size). Repeat with build time and with bytes per vector as the objective. Output a winner table and a heatmap panel per objective. If no index qualifies, record `none`.
2. `scaling.py`. For each index, at each R, fit `log y = a + b log N + c log D + d (log N)(log D)` for y in {p95 latency, build time, RAM}. Use the 6 cells from Stages A and B that vary both N and D, plus Stage A for more D values at N = 500K. Bootstrap over queries and over builds for intervals on b, c, d. Report that the fit is descriptive, with few distinct (N, D) points.
3. `crossover.py`. Estimate S* per (index, D, N, correlation, R) by interpolating log-latency versus log S between the pre-filter exact-scan curve and the in-index curve. Bootstrap intervals over queries. Test H2: compare S* at D = 384 versus D = 1536 within each N, and flag non-overlapping 95 percent intervals.
4. `rule.py`. Build one row per (cell, objective) with features: log10 N, D, log10 S (use 0 for unfiltered), correlation type, R, memory budget ratio, update ratio, label = oracle-best index. Fit a decision tree of depth at most 3 on rows with N <= 500K. Evaluate on rows with N = 1M. Baselines: always HNSW; HNSW unless budget is below 0.5x raw, then IVF-PQ. Report top-1 agreement and regret (cost of the chosen index divided by the oracle cost, minus 1). Also report leave-one-D-out cross-validation. Plot the tree.
5. `figures.py`. Produce: cost versus D per index at R = 0.95 and 0.99; winner heatmaps; crossover plots per D with intervals; recall-drift curves for Stage D; the tree figure; one pipeline schematic.

Acceptance
- Every figure and table regenerates from raw data with one command (`make analysis` or a script).
- Every number quoted in the paper draft is traceable to a CSV in `results/summary/`.
- A `reports/findings.md` lists each hypothesis (H1a, H1b, H1c, H2, H2b, H3 from the paper document) with: supported, not supported, or inconclusive; the evidence; the interval.

**GATE 5.** Human reviews `reports/findings.md` and decides which claims go in the paper. Claims not backed by a measured cell go into a future-work paragraph.

### Phase 6: Writing support

Tasks
1. Create `paper/` with section stubs following the outline in the paper document.
2. Generate LaTeX or Markdown tables from `results/summary/`.
3. Write `README.md`: environment setup, how to regenerate embeddings, how to rerun the pilot cell with one command, how to rerun the full study, expected runtime and hardware.
4. Final check: scan `paper/` for numbers and confirm each appears in `results/summary/`.

---

## 5. Result schemas

Store as Parquet. Every table carries `run_id`, `git_commit`, `hardware_id`, `timestamp`, `seed`, `config_hash`.

**builds**
`index_type, N, D, nlist, m, M, ef_construction, nbits, train_size, train_seconds, add_seconds, build_seconds, threads, index_bytes, rss_bytes, repeat_id, selected`

**sweeps** (one row per search setting)
`build_run_id, index_type, N, D, k, ef_search, nprobe, k_factor, query_split, recall_mean, recall_ci_lo, recall_ci_hi, lat_mean_us, lat_p50_us, lat_p95_us, lat_p99_us, qps, threads`

**tuned** (one row per cell, index, R)
`index_type, N, D, k, R, reachable, best_build_run_id, ef_search, nprobe, k_factor, recall_test, lat_p95_us, lat_p95_ci_lo, lat_p95_ci_hi, qps, rss_bytes, index_bytes, build_seconds_mean, build_seconds_min, build_seconds_max`

**filter_runs**
`index_type, strategy, impl, N, D, correlation, target_S, measured_S_mean, R, reachable, search_param, recall_test, lat_p95_us, qps, overfetch_factor`

**update_runs**
`index_type, batch_id, inserted_total, insert_seconds, inserts_per_second, recall_at_fixed_params, lat_p95_us, retrained, rebuild_reference_recall`

---

## 6. Stop conditions and escalation

Stop and report to the human, without trying workarounds, when:
- Free disk or RAM would be exceeded by the next build.
- A build or search job exceeds 3x its budgeted time.
- Recall on a sanity check differs from expectation by more than 0.05 with no identified cause.
- A library does not support a required operation (for example in-index filtering on HNSW) and the fallback changes what is being measured.
- A gate is reached.

When reporting, state: what ran, what passed, what failed, what you think the cause is, and the options. Do not silently reduce scope.

---

## 7. Definition of done

- [ ] Phase 0 table and Gate 0 decision recorded
- [ ] Embeddings, ground truth, LID table, and acceptance checks pass
- [ ] Harness tests pass and pilot curves match expectations
- [ ] Budget approved at Gate 3
- [ ] Stages A to D complete with reachability tables
- [ ] Analysis regenerates from raw data with one command
- [ ] `reports/findings.md` covers every hypothesis
- [ ] README reproduces the pilot cell in one command
- [ ] Every number in the paper traces to `results/summary/`
