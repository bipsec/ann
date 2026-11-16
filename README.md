# ANN index-selection study

For each workload cell — corpus size *N*, embedding dimension *D*, filter selectivity *S*,
required recall *R*, memory budget *M* — this study finds the **cheapest** configuration of
Flat, HNSW, IVF-Flat, IVF-PQ and IVF-PQ+refine that reaches recall@10 ≥ *R*, locates where the
winning index changes, and fits a small decision rule that is tested on held-out cells.

The experiment protocol is specified in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). Every
non-obvious choice, deviation and known gap is dated in [`logs/decisions.md`](logs/decisions.md).
Read that log before trusting any number.

## Status

| Phase | State |
| --- | --- |
| 0. Positioning | Complete; Gate 0 decided the D × S headline (`logs/gate_reports/gate_0.md`) |
| 1. Data and ground truth | Complete and verified: 1M passages × {128, 384, 768, 1536} dims, 10k queries, 12 exact ground-truth files |
| 2. Harness | Complete; 46 tests pass |
| 3. Pilot and budget | Complete; Gate 3 approved (`logs/gate_reports/gate_3.md`) |
| 4. Stages A–D | **Complete** — all four stages run on host `1d3d9b66e7d3e1ac`; reports in `reports/stage_{A,B,C,D}.md` |
| 5. Analysis | Complete; regenerates in one command (`make analysis`); verdicts in `reports/findings.md` |
| 6. Writing | Paper drafted at `paper/main.md` (all results, traceable); Gate 5 reviewed (`logs/gate_reports/gate_5.md`). Open items in `REMAINING_WORK.md` |

See [`FEEDBACK.md`](FEEDBACK.md) for the results summary + what went wrong, and
[`REMAINING_WORK.md`](REMAINING_WORK.md) for the remaining checklist.

## Requirements

- Linux x86-64 with **AVX2**. Verify with
  `python -c "import faiss; print(faiss.get_compile_options())"` — it must mention `AVX2`.
  Latency, throughput and crossover results from a non-SIMD build are not comparable; see the
  AVX2 rows in `logs/decisions.md`.
- Python ≥ 3.11, ~64 GB RAM for the N=1M cells (58 GB has been used, see the decisions log),
  and 16 cores. Free disk: ~17 GB for embeddings and ground truth, plus ~31 GB if Stage B
  indexes are retained for Stage C.
- [`uv`](https://docs.astral.sh/uv/) for installation.

## Setup

```bash
uv venv
uv pip install -e ".[dev,hnswlib]"
make hardware        # writes results/hardware.json; warns if RAM < 64 GB
make test            # 46 tests
```

Results are **scoped by `hardware_id`**: measurements recorded on one machine are never tuned
against or reported alongside another's. Running on a new host re-measures rather than reusing.

## Reproducing the pilot cell in one command

```bash
make pilot
```

This runs Stage A's grid at N=100K for all four dimensions with one build variant per index
(20 builds, 216 search settings), through the full pipeline: pinned single-thread latency
sweeps on the tuning queries → cheapest setting per recall target → re-measurement on the
8,000 held-out test queries → T-thread build/throughput pass with three builds per winner →
index-only memory measurement → `reports/stage_PILOT.md`.

It is resumable: completed configurations are skipped, so an interrupted run can simply be
re-issued. Expect roughly 1–2 hours on 16 cores.

```bash
make analysis        # regenerates every table and figure from results/raw
```

## Regenerating the data (not normally needed)

`data/` is gitignored and holds ~17 GB. The embeddings come from a paid API, so regenerate
only if they are genuinely lost.

```bash
uv pip install -e ".[data-prep,embed-api]"
export OPENAI_API_KEY=...                      # see configs/data.yaml for the model
python -c "from annstudy.data import sample_passages, sample_queries, split_queries; \
           ids = sample_passages(); q = sample_queries(); split_queries(q)"
annstudy-embed --config configs/data.yaml      # prints a cost estimate and waits for approval
python -c "from annstudy.truncate import create_all_truncations; create_all_truncations()"
annstudy-gt --config configs/data.yaml         # exact top-100 per (N, D); ~5 min on 16 cores
```

Acceptance checks that must pass afterwards: embeddings are `(1_000_000, 1536)` and
`(10_000, 1536)` with no NaN and unit norms to 1e-4; the 100K and 500K sets are exact prefixes
of the 1M set; and a numpy brute force over 100 random queries reproduces the stored ground
truth.

## Running the full study

Each stage appends to `results/raw/` and writes `reports/stage_X.md`. **Check
`logs/budget.md` first** — the full grid is measured in days of compute, and
`EXPERIMENT_PLAN.md` section 4 lists the trims to apply in priority order if that is too much.

```bash
make stage-a                     # dimension study at N=500K
make stage-b                     # scale study; keeps indexes for Stage C
make stage-c                     # filtering study; reuses Stage B indexes
make stage-d                     # update study at N=500K, D=768
make analysis
```

Stage C's latency must stay single-threaded, so parallelism comes from sharding across pinned
cores. The shard count cannot simply be the core count: each shard materialises the cell's
corpus *and* loads a full index, which is 13.1 GB at N=1M/D=1536. `make stage-c` drives the
cells in turn at a memory-safe width; inspect the plan first with:

```bash
make stage-c-plan     # per-cell job counts and how many shards fit in memory
make stage-c          # runs them cell by cell, then consolidates
```

## Layout

```
configs/      stage and data configs (grids live in indexes.get_*_grid; see decision M4)
src/annstudy/
  data.py embed.py truncate.py groundtruth.py   Phase 1: corpus, embeddings, ground truth
  indexes.py metrics.py                          index factories, recall/latency/bootstrap
  runner.py                                      one build + sweep, or --mode memory
  coordinate.py                                  stage orchestration, resumable, core-pinned
  filter_runner.py                               Stage C: filtered search
  update_runner.py updates.py                    Stage D: inserts
  analysis/                                      winners, scaling, crossover, rule, figures
results/raw/    append-only parquet (builds, sweeps, memory, filter_runs, update_runs)
results/tuned/  tuned.parquet (test-measured, authoritative) + tuned_tuning.parquet
results/summary/ derived CSVs — every number in the paper must trace to one of these
reports/        per-stage reports, findings.md, analysis_summary.md
logs/           decisions.md, budget.md, gate_reports/, per-runner logs
```

## Reading the results correctly

- **Headline numbers come from the 8,000 test queries only.** Parameters are chosen on the
  2,000 tuning queries; `results/tuned/tuned_tuning.parquet` keeps that selection record and
  `tuned.parquet` holds the test-measured results.
- **Latency is single-threaded and pinned; build time and QPS use T threads.** Rows carry a
  `threads` column — do not compare across different values.
- **Memory**: prefer `rss_bytes` with `rss_source = index_only_subprocess`. A row labelled
  `process_total_inflated` is a fallback that includes the passage array.
- **`reachable = false`** means no setting in the sweep reached that recall target. These rows
  are kept deliberately; absence of a result is a result.
- Recall never uses the distances an index reports. It recomputes true distances from the raw
  vectors and counts neighbours within a 1e-5 relative tolerance of the k-th true distance, so
  equally good neighbours are not penalised.
