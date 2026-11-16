# GATE 3 Report: Pilot and budget

_2026-10-17. Execution host: AMD EPYC-Milan, 16 vCPU, 58.8 GB RAM, `hardware_id 1d3d9b66e7d3e1ac`._

Decision required: **approve the Stage A–D budget, or trim the plan.** Everything below is
measured on this host and traceable to a file under `results/`.

## What ran

| Step | Outcome |
| --- | --- |
| Harness repair | Package layout restored; 4 correctness bugs fixed; 46 tests pass |
| Stage C driver | Written (did not exist); validated at N=100K/D=128 |
| Pilot pass 1 | 20/20 builds, 216 search settings, **0 failures** (12 min) |
| Tune | 41 reachable winners, 19 unreachable, 15 distinct winner builds |
| T-thread repeats | 45/45 builds, 3 per winner, **0 failures** (35 min) |
| Memory pass | 45 index-only RSS measurements (gap M11 resolved) |
| Test re-measure | 15/15 winner builds on the 8,000 held-out test queries (18 min) |
| Acceptance checks | **11 passed, 0 failed, 0 skipped** (`results/summary/acceptance_checks.csv`) |
| Analysis | Regenerates in one command, no silent skips, exit 0 |

## What passed

All EXPERIMENT_PLAN section 2 and 3 acceptance checks:

- Flat recall is exactly 1.0 at every D.
- HNSW recall is monotone in efSearch and reaches ≥ 0.95 at efSearch 32–64, far inside the
  required ≤ 256. Minimum-effort recall is 0.80–0.90, i.e. **not** implausibly perfect (rule 0.9).
- IVF-Flat, IVF-PQ and IVF-PQ+refine recall are monotone in nprobe.
- All 4 cells carry all 5 index types.
- **Repeat p50 within 10%:** 282.6 µs → 290.1 µs = **2.7%**, with bit-identical recall.
- **efConstruction check:** mean recall 0.9853 / 0.9894 / 0.9902 for efC 100 / 200 / 400, with
  build time 5 s / 11 s / 23 s. Keeping 200 is justified: efC 400 doubles build time for
  +0.0008 recall.

Measured cost surface at N=100K (test queries, p95 latency, single-threaded):

| D | Flat (exact) | HNSW @ R=0.95 | IVF-Flat @ R=0.95 | IVF-PQ+refine @ R=0.95 |
| --- | --- | --- | --- | --- |
| 128 | 3,450 µs | **219 µs** (ef=64) | 407 µs (nprobe=64) | unreachable |
| 384 | 13,596 µs | **274 µs** (ef=48) | 1,609 µs | unreachable |
| 768 | 21,044 µs | **292 µs** (ef=32) | 2,693 µs | 1,359 µs |
| 1536 | 43,182 µs | **490 µs** (ef=32) | 5,505 µs | 2,646 µs |

Memory is now measured correctly (RSS of a process that loaded only the index), so the
memory-budget winners resolve instead of reading `none`: at the 1.5× budget HNSW wins at every
D; at 0.5× and 0.1× no index qualifies, for the reason in the next section.

## What did not pass, and why it is not a blocker

**1. IVF-PQ is `reachable = false` at every D and every recall target.** This is a consequence
of the pilot's design, not a property of IVF-PQ: `--max-build-variants 1` takes the *first*
entry of each build grid, which for IVF-PQ is the **coarsest** quantization (`m = D/16`, 16×
compression). The full Stage A grid also tests `m = D/8` and `m = D/4`. IVF-PQ+refine, which
re-ranks with exact distances, does reach R=0.95 at D ≥ 768 even at that coarsest setting.
**Do not read the pilot as evidence against IVF-PQ**, and note that it is the only index that
could satisfy the 0.5× and 0.1× memory budgets — so those budget rows stay empty until Stage A
runs the finer quantizations.

**2. H1a/H1b/H1c are all `inconclusive`** in `reports/findings.md`. Correct and expected: the
pilot is a single N with one build variant per index, so there is nothing for a dimension,
scale or budget crossover to be measured against. HNSW wins every unbounded cell.

**3. QPS is present on 23 of 41 reachable rows.** The T-thread pass measured only each winner's
highest-recall-target setting, so lower targets with a different setting have no throughput
number. Fixed for Stage A onward (the pass now measures every setting a winner was chosen
with); the pilot is not re-run for it.

## Validity finding that affects every latency number

Latency on this host depends on **machine-wide core occupancy**. The same spec, same seed,
bit-identical index (recall equal to 5 decimal places at every efSearch) measures:

| Condition | p50 at ef=128 |
| --- | --- |
| Full 16-core pool (how the campaign runs) | 282.6 µs |
| Lone pinned job, 15 cores idle | 365.6 µs |

That is a 1.3× shift from occupancy alone. Repeated measurement *within* a fixed condition is
stable to ~1.2%, and independently rebuilt indexes agree to ~1.1%, so the harness itself is
sound. Consequences, now enforced in code and recorded in `logs/decisions.md`:

- Every latency measurement must run with one pinned runner per core. The coordinator already
  does this; the stability check now generates sibling load so it tests the right condition.
- Latency must never be compared across hosts or across occupancy conditions. Results are
  scoped by `hardware_id`, and the 127 rows transferred from the previous machine are excluded
  rather than deleted (rule 0.6).

## Budget

From measured pilot anchors (`logs/budget.md`, regenerate with `make budget`):

| Stage | Jobs | Projected wall h |
| --- | --- | --- |
| Stage A (dimension, N=500K) | 100 | 9.4 |
| Stage B (scale, N ∈ {100K, 500K, 1M}) | 150 | 20.2 |
| Stage C (filtering, 308 jobs) | 308 | 6.2 (floor 5.6 h — longest single job) |
| Stage D (updates) | 3 | 0.4 |
| **Total** | | **36.1 h** |

**The main caveat:** the pilot measures a single N, so the N-scaling exponents in the
projection are *assumed*, not fitted. Stage A is also a single N (500K), but completing it
gives a second point and lets the budget be re-derived from measurement before committing to
Stage B, which is more than half the total.

Disk is a real constraint: **43 GB free**, and Stage B must retain roughly 31 GB of indexes for
Stage C to reuse. `runner.check_disk_space` refuses a build when free space is below 2× the
expected index size, so this will fail loudly rather than silently.

## Options

1. **Run Stage A, then re-gate (recommended).** 9.4 h. Delivers the dimension cost surface —
   the study's D axis — validates the N-scaling assumptions against a second N, and re-derives
   the budget before the expensive Stage B. Lowest risk of spending 20 h on a mis-projection.
2. **Run Stages A–D as planned.** 36.1 h total. Acceptable on measured numbers, and the full
   grid resolves the IVF-PQ reachability and memory-budget gaps above.
3. **Apply the plan's trims first.** In the plan's own priority order: drop M=64; drop the
   largest nlist multiplier (c=8); run D=1536 at N ≤ 500K only; shorten the nprobe/efSearch
   sweeps. Roughly halves Stage A+B. Recommended only if 36 h is unacceptable — the trims cost
   grid coverage that the headline D × S claim depends on.
4. **Stop here.** The harness, the Stage C driver, the data and the pilot stand on their own as
   a reproducible foundation; Stages A–D remain unrun.

**Recommendation: option 1.** Stage A is the cheapest stage that produces a headline axis, and
it converts the budget's main assumption into a measurement.
