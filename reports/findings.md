# Findings

_Generated 2026-11-17T00:14:22.546867+00:00. Every verdict is computed from a CSV under `results/summary/`; none are entered by hand._

| Hypothesis | Statement | Origin | Status | Evidence | Basis |
|---|---|---|---|---|---|
| H1a | HNSW:IVF-PQ memory ratio widens with dimension D | proposal RQ1 | not supported | `results/summary/index_footprint (paper table)` | at fixed 16x compression the HNSW:IVF-PQ bytes-per-vector ratio is arithmetically ~constant in D (HNSW's extra graph bytes are D-independent, so the ratio narrows slightly with D, not widens) |
| H1b | IVF-PQ recall ceiling FALLS with D at fixed 16x compression | proposal RQ1 | not supported | `results/summary/pq_ceiling (paper table)` | proposal predicted the ceiling falls with D; measured it rises, 0.745 (D=128) -> 0.888 (D=1536) at 16x compression, still < 0.90 so IVF-PQ stays unreachable |
| H1c | Fitted D-exponent of p95 latency differs by index type | proposal RQ1 | supported | `results/summary/scaling_fits.csv` | D-exponent of p95 at N=500K, R=0.95: hnsw 0.20 [0.20, 0.21], ivf_flat 0.95 [0.95, 0.96], flat 0.96 [0.96, 0.96] |
| H2 | The filter-selectivity crossover S* (in-index vs pre-filter exact) moves with D | proposal RQ2 | mixed (index-dependent) | `results/summary/crossover.csv` | 7 D-pairs; hnsw: shift=4 (median ratio 0.65); ivf_flat: inconclusive=3 (median ratio 1.21). Caveat: the original Stage C confounds D with session/load and uses per-D clusterings; the confound-free rerun is configs/stage_c_h2.yaml |
| H2b | S* also depends on filter correlation (positive vs negative) | proposal RQ2 | supported | `results/summary/crossover.csv` | median S* by correlation: none: 0.01535, positive: 0.003523 |
| H3 | A small rule picks the cheapest FILTER STRATEGY from S, D, R and a correlation proxy | exploratory, post-hoc | not supported | `results/summary/strategy_rule_accuracy.csv` | [exploratory, post-hoc] rule top-1/regret 0.667/0.105 on 66 held-out cells vs baselines (top-1/regret): always exact scan 0.439/192.352; always in-index 0.379/0.115; 1% selectivity switch 0.530/0.071; same tree without proxy 0.727/16.493 (the index-choice rule only ties the heuristic: top-1 1.000 vs baseline 1.000 — it never sees the filter axis) |

## Interpretation limits

- A `supported` verdict here means the measured cells show the effect; it is not a significance test. Interval-based claims come from the bootstrap columns in the backing tables.
- `inconclusive` means the stage that would produce the evidence has not run yet, not that the hypothesis failed.

## Pipeline completeness

- skipped `filter_proxy.compute` — missing input: 4 probe base index(es) are not on this host (e.g. data/indexes/stage_b/idx_3ced67cb5e464b1cb070533957b76f9e.faiss); refusing to replace the online proxy with the idealized fallback. Run on the measurement host, or pass allow_fallback=True / --mode idealized explicitly.
