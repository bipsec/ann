# Study feedback — results and what went wrong

_ANN index-selection study (`annstudy`). All four stages (A–D) complete on host
`1d3d9b66e7d3e1ac` (AMD EPYC-Milan, 16 cores, 58.8 GB, Linux; **faiss-cpu 1.15.1, AVX2**).
Companion file: [`REMAINING_WORK.md`](REMAINING_WORK.md) (what's left to do)._

---

## 1. Results in one page

The authoritative write-up is [`paper/main.md`](paper/main.md) (machine-generated from `results/`,
every number traceable). Headlines:

**The headline hypothesis is SUPPORTED (H2 — the D×S crossover).** The filter-selectivity crossover
S* — where in-index filtered search and exact scan of the filtered subset swap places — **moves with
dimension in all 8 comparable pairs** (disjoint bootstrap intervals in 8/8), shifting toward lower
selectivity as D rises (median ratio S*(D=1536)/S*(D=384) = 0.76). This contradicts the D-independent
crossover predicted by the closest prior work (Phase-Transition, 2026) and is the study's novel
contribution. S* also moves ~8× with correlation (median 0.0106 uncorrelated vs 0.00137 positive).
Most strikingly, **under anti-correlated predicates approximate search fails outright** — 0 of 216
attempts reach recall@10 ≥ 0.90 at any effort, while exact scan always does.
_Source: `results/summary/crossover_h2.csv`, `crossover.csv`, `reports/stage_C.md`._

**Two honest negatives — more useful than the positive, and reported as prominently:**
- **H1a NOT supported:** the cheapest index does **not** change with D — HNSW wins at every D and
  every R (by 58×–398× over exact scan at R=0.95); its p95 grows only ~1.9× over the 12× D range.
- **H3 NOT supported:** the depth-3 decision rule generalizes (test top-1 = 1.0, regret 0) but only
  **ties the trivial "always-HNSW" baseline**, so it adds no value. Leave-one-D-out CV is weaker
  (mean 0.875; regret up to ~2.9 at D=128/384) — it does not robustly generalize across dimension.
  _Source: `results/summary/rule_accuracy.csv`, `rule_lodo_cv.csv`._

**Other measured results:**
- **IVF-PQ recall ceiling:** plain IVF-PQ is unreachable in all 30 (N,D,R) cells; its ceiling is
  0.888 (D=1536, 16× compression) and ~0.713 in Stage D — set by quantization error, not coverage.
  Exact re-ranking (IVF-PQ+refine) lifts the same configs to 0.999.
- **Sub-raw memory budgets are infeasible at R ≥ 0.90:** the only index small enough (IVF-PQ) can't
  reach the recall, so the 0.5× and 0.1× budgets have no feasible winner at all. Within 1.5×, IVF-Flat
  displaces HNSW only at D=128 (H1c — a real but localized effect, 2 of 30 cells).
- **Updates (Stage D):** inserting 11% more vectors costs negligible recall for every family
  (≤ 0.0004 drift); HNSW ingests ~35–50× slower than the IVF families — the clearest cost/benefit trade.

---

## 2. What went wrong (and its status)

### Correction up front: the study ran on CPU, not GPU
Every artifact — `results/hardware.json` (AMD EPYC CPU, `torch`/`openai`/`sentence-transformers`
**not-installed**), the source (no `cuda`/`gpu`/`device` references), configs, logs, and result rows
(only a `threads` column) — shows **all indexing and search ran on CPU** with AVX2 `faiss-cpu`,
single-thread-pinned latency; embeddings came from the **hosted OpenAI API** (chosen explicitly
because no GPU was available). This matches the methodology, which **excludes GPU indexes on purpose**.
➡️ **Any statement that the experiments "ran on GPU" is incorrect and must not go in the paper.** The
correct framing (already in `paper/main.md`) is CPU / AVX2 / single-thread.

### Harness bugs found and fixed (all FIXED — see `logs/decisions.md`)
| Bug | Impact | Status |
|---|---|---|
| `build_ivf_pq_refine` added vectors before wrapping in `IndexRefineFlat` | one of 5 indexes couldn't build at all | FIXED (wrap-then-add) |
| Stage C filter-cache write race across 16 shards | truncated reads killed ~20 jobs | FIXED (atomic rename + 2 regression tests) |
| `crossover.py` grouped baseline wrong + looked for the wrong sign change | the **headline H2 analysis was dead** | FIXED (direction-agnostic + bootstrap CIs → `crossover_h2.csv`) |
| `rule.py`/`scaling.py` filtered a stale column name (`lat_p95_us_tuning`) | H3 always empty; p95 scaling silently skipped | FIXED |
| `analysis/run_all` swallowed all exceptions | hid the two bugs above; `findings.md` printed "TODO" | FIXED (tracebacks + non-zero exit) |
| `tune` grouped by build-variant not cell | 348 tuned rows at wrong granularity, ~43% inflation | FIXED |
| `config_hash` included the seed | repeat builds never joined to the seed-0 winner (QPS/memory lost) | FIXED (seed-independent key + `finalize` phase) |
| `_expected_rss` under-counted by up to 2.5× | would OOM-kill the pool at D=1536 | FIXED |
| Stage A retained every index → disk 100% | 62 builds skipped as "insufficient disk" | FIXED (retain one per winner; build→measure→delete) |

### Measurement-protocol corrections
- **faiss AVX2 saga (RESOLVED):** Windows faiss-cpu has no AVX2 binary (≈2–4× slower, could shift
  crossovers); conda-forge on Windows gave no speedup and crashed; resolved by moving all measured runs
  to the Linux VM and deciding AVX2 from `faiss.get_compile_options()` rather than filenames.
- **M10 (RESOLVED):** headline numbers now come from the 8K **test** queries (a `remeasure` phase +
  `finalize_tuned`), not the tuning queries.
- **M11 (RESOLVED):** memory is now index-only RSS (load a serialized index in a fresh process), not
  the inflated builder footprint — this is what made the memory-budget question answerable.
- **Occupancy claim refuted:** an early 1.3× "core-occupancy" latency effect did **not** replicate
  (controlled re-test: 1.07×). The real effect is ~1.1× **between-session drift** (cause unisolated).
  ➡️ Never compare absolute microseconds across campaigns or hosts; results are keyed to `hardware_id`.
- **Cross-host contamination:** 127 rows from the earlier 4-core i5 (`0ffbd21964ab1a09`) were silently
  satisfying pilot specs; now excluded from tuning/reporting (kept, not deleted, per plan rule 0.6).

### Accepted caveats (documented, not "fixable")
- **Stage C selectivity at S=0.001 misses target by 54–61%** for correlated predicates — structural
  (k-means with 1000 clusters can't form a 0.1%-selective union); S* there is reported vs *measured*
  selectivity. Self-flagged in `reports/stage_C.md`; a Gate-5 decision asks whether to report S=0.001.
- **Tuning→test generalization gap:** 17/48 Stage A winners miss R on held-out queries (worst −0.012);
  flagged per-row via `meets_R_on_test` — these must not be cited as achieving R.
- **N-scaling exponents are descriptive fits over 2–3 N points**, and were mis-assumed in *both*
  directions in the pre-study budget (e.g. HNSW search measured N^0.40, not the assumed N^0.15).

### Budget: projected vs actual (`logs/budget.md`)
Whole study ≈ **56 h compute** (A 7.7 h, B ~17 h, C ~27 h, D ~4.5 h), 0 unexplained failures.
**Stage C and Stage D each overran ~4×** — Stage C because N=1M cells admit only 3–12 memory-bound
shards (not 16-wide) and 20 jobs were re-run after the cache race; Stage D because of two re-runs
(a completeness-vs-drift scoring error, and a duplicate-process contention bug). The base budget model
ran ~1.48× optimistic.

### Documentation-integrity issues (fixed in this pass / flagged)
- `reports/stage_A.md` and `reports/stage_B.md` were **byte-identical** (a report-generator bug) — being fixed.
- Root `README.md` and `paper/README.md` status tables are **stale** ("Stages A–D: Not started") — being fixed.
- The orphaned hand-written `paper/01…09.md` scaffold still says "PENDING" and `03-method.md`/`08` even
  claim the hnswlib cross-check was *done* (it was not) — being removed; `paper/main.md` is the single source.
- `scaling_fits.csv` has **no confidence-interval columns** though the plan asked for bootstrap
  intervals on b/c/d — see `REMAINING_WORK.md`.
- `findings.md` labels H1a/H1b/H1c as dimension/scale/budget, which differs from the methodology doc's
  RQ1 sub-hypothesis labels — reconcile before submission.
