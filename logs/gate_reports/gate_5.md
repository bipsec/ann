# GATE 5 Report: Findings review

_2026-10-28. All four measurement stages complete. Host `1d3d9b66e7d3e1ac`._

Per EXPERIMENT_PLAN Phase 5, the human reviews `reports/findings.md` and decides **which claims
go in the paper**. Claims not backed by a measured cell go to future work. This report states
what is measured, how strongly, and what I would and would not claim.

## What ran

| Stage | Scope | Outcome |
| --- | --- | --- |
| Pilot | N=100K, 4 D, 1 variant/index | 20/20 builds, 0 failures |
| A | N=500K, 4 D, full grid | 100/100 builds, 0 failures, ~7.7 h |
| B | N ∈ {100K,500K,1M} × D ∈ {384,1536} | 90 new specs, 0 failures, ~17 h |
| C | 4 cells × 3 strategies × 11 filters | 308/308 jobs, 924 rows, ~27 h |
| D | N=500K, D=768, 3 index types | 33 rows, 0 failures, ~4.5 h |

Acceptance: 9/9 machine-checked conditions pass (`results/summary/acceptance_checks.csv`).
Analysis regenerates in one command with no silent skips. The paper's traceability check
reports no numeric token that cannot be located in a result file.

## Hypothesis verdicts, and what I would claim

| ID | Verdict | Strength of evidence | Recommended for the paper |
| --- | --- | --- | --- |
| H1a (winner vs D) | not supported | Full grid, 9 comparable groups, 3 corpus sizes | **Yes** — report as a negative result |
| H1b (winner vs N) | not supported | 12 comparable groups; HNSW latency flat across a 10x corpus | **Yes** — negative, with the drift caveat |
| H1c (winner vs memory budget) | supported | Only 2 of 30 cells, both at D=128 | **Yes, but qualified** as localized |
| H2 (S* vs D) | supported | 8/8 pairs differ, 8/8 disjoint intervals, median ratio 0.76 | **Yes — headline** |
| H2b (S* vs correlation) | supported | Median S* 0.0106 uncorrelated vs 0.00137 positive (8x) | **Yes — headline** |
| H3 (decision rule) | not supported | Ties the always-HNSW baseline exactly (1.000 vs 1.000) | **Yes** — as a deflationary finding |

### Claims I would make

1. **S\* moves with dimension** (H2). All eight comparable D-pairs differ with non-overlapping
   intervals, in a consistent direction: the crossover shifts to lower selectivity as D rises.
   This is the gap Phase 0 identified — the closest prior work models the crossover at fixed D.
2. **S\* moves far more with filter/query correlation than with dimension** (H2b): roughly 8x
   versus 0.76x. If a system switches strategy on selectivity alone it will be wrong across a
   tenfold band. I consider this the most actionable result in the study.
3. **Anti-correlated predicates defeat approximate search outright.** In-index filtered search
   and post-filter over-fetch reach R >= 0.90 in 0 of 216 attempts under negative correlation;
   exact scan of the subset succeeds in all 36. Not a slowdown — a failure.
4. **PQ alone cannot reach R >= 0.90** on these embeddings (ceiling 0.888 at 16x compression,
   saturating in nprobe), and consequently **no index satisfies a sub-raw memory budget** at
   these targets. The 0.5x and 0.1x regimes are empty, not unexplored.
5. **HNSW dominates every unfiltered axis**, and its latency at fixed recall is
   indistinguishable from flat across a 10x corpus increase. A learned selection rule therefore
   cannot beat "always HNSW" (H3).
6. **Insertion is nearly free in recall and expensive in ingest**: drift under 11% growth is
   <= 0.00035 for all families, while HNSW ingests ~50x slower than the IVF families.

### Claims I would NOT make

- **No scaling laws.** The N-exponents (`paper/tables/n_exponents.csv`) are fits over two or
  three points and are reported as descriptive. They were used operationally, to correct our own
  compute projection, and should not be read as complexity results.
- **No absolute-latency portability.** Section 4.2: a bit-identical index measured 282.6 us and
  313.0 us two days apart under matched conditions. Relative orderings within a campaign are
  what we defend; microsecond figures are not portable across campaigns or hosts.
- **Nothing about deletions**, k != 10, or a second HNSW implementation. The hnswlib cross-check
  the plan lists was never run.
- **No claim at the smallest correlated selectivity as nominal.** See below.

## Known deviations and gaps, stated for the record

1. **Correlated selectivity at S=0.001 misses the plan's 20% tolerance** (54-61% over target).
   Cause is granularity: correlated predicates union whole k-means clusters, and with k=1000 the
   smallest union is already ~0.1% of the corpus. S* at that level applies to the *measured*
   selectivity, not the nominal one. `paper/tables/measured_selectivity.md`.
2. **Stage D was re-run** after the first pass scored a partially-filled index against
   full-corpus ground truth, which made recall climb mechanically and measured completeness
   rather than drift. Superseded rows retained under
   `results/raw/update_runs_superseded_fullcorpus_gt/` per rule 0.6.
3. **17 of 48 Stage A winners miss their recall target on the held-out queries** (worst -0.0116).
   This is selection noise from a 2,000-query tuning split, flagged per row by
   `meets_R_on_test`. Those rows must not be cited as having achieved R.
4. **One Stage B winner (flat, N=1M, D=1536) briefly had 1 build instead of 3**, after its
   repeats were skipped on disk; backfilled, now 3.
5. **Stage C latency contention risk**: two Stage D processes once ran concurrently and produced
   latencies differing by 66%. Detected, rows discarded, re-run clean. No Stage C row is
   affected, but it illustrates why occupancy discipline matters.
6. **No canary configuration** was interleaved through the campaigns to track drift directly.
   This is the single protocol addition I would recommend before any follow-up run.

## Decision required

1. Confirm the six claims above for the paper, or strike any you consider insufficiently
   supported — in particular whether H1c belongs as a claim given it appears in only 2 of 30
   cells.
2. Decide whether the smallest correlated selectivity (S=0.001) should be reported at all, given
   deviation 1, or dropped to S >= 0.01 for correlated predicates.
3. Decide whether to run the outstanding items before writing: the hnswlib cross-check,
   k ∈ {1, 100}, and a drift canary. None changes the headline; all strengthen it.

**My recommendation:** report all six, with H1c explicitly qualified as localized to low D, and
keep S=0.001 with the measured-versus-nominal caveat stated inline rather than dropping data.
The negative results (H1a, H1b, H3, PQ ceiling, anti-correlation failure) are more useful to a
practitioner than the one positive selection result, and the paper is currently written that way.
