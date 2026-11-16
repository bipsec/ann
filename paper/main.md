# Which ANN Index Is Cheapest at a Given Recall? A Cost-Surface Study over Dimensionality, Scale, and Filter Selectivity

**Draft — all four measurement stages complete:** Stage A (dimension), Stage B (scale), Stage C (filtering) and Stage D (updates), on top of Phase 1 data and a validated harness. Remaining gaps are stated explicitly in section 8 rather than estimated.

_Generated 2026-11-05T01:52:17+00:00 by `annstudy-paper` from files under `results/`. Every number below is rendered from a result file; none are entered by hand. Regenerate with `make paper`._

## Abstract

Choosing an approximate nearest neighbour (ANN) index is usually posed as a recall-versus-throughput trade-off on a fixed dataset. We instead fix the requirement — recall@10 at least R — and ask which index, and which filtering strategy, is cheapest subject to it as the workload varies in corpus size N, embedding dimension D (obtained by truncating one Matryoshka embedding to 128, 384, 768, 1536 dimensions, so D is not confounded with the dataset), and filter selectivity S.

The headline is on the filter axis, as measured on this grid. The selectivity S* at which in-index filtered search and exact scan of the filtered subset swap places **moves with dimension** — disjoint bootstrap intervals in 8 of 8 D-pairs, shifting to lower S as D rises (median ratio 0.76) — and it shifts about 8x more with query-filter correlation. Under anti-correlated predicates, approximate search at the swept effort does not reach recall@10 >= 0.90 anywhere in the grid, while exact scan of the subset always does.

Two negatives follow. Product quantization alone never reaches R=0.90 (ceiling 0.888 at 16x compression; exact re-ranking lifts it to 0.999), so none of the five conventional families meets a sub-raw memory budget at these targets — though a scalar-quantized graph index, added as a follow-up probe, reaches 0.27-0.35x raw while holding recall, and is faster than uncompressed HNSW below R=0.99. Unfiltered, HNSW is the cheapest index at every D and R (58-398x over exact scan at R=0.95; p95 latency grows only 1.6x from D=128 to D=1536), and inserting 11% more vectors costs almost no recall though HNSW ingests ~37x slower than IVF. Absolute latency drifts ~1.1x between sessions, so every comparison is drawn within a campaign and keyed to a hardware identity.

## 1. Introduction

A practitioner choosing a vector index has a requirement, not a curve: *this* query latency, at *this* recall, within *this* memory budget, for *this* corpus. Published benchmarks answer a different question — they plot recall against throughput for many indexes on a handful of datasets, leaving the reader to locate their own operating point. Worse, the datasets differ in dimension, scale and intrinsic structure simultaneously, so the effect of any one axis cannot be isolated.

This study inverts the framing. For each workload cell (N, D, S, R, memory budget M) we find the cheapest configuration of Flat, HNSW, IVF-Flat, IVF-PQ and IVF-PQ+refine that attains recall@10 >= R, then ask where the winner changes. The key methodological move is to vary D *within one corpus*: a Matryoshka-trained embedding model admits prefix truncation followed by renormalisation, so D=128 and D=1536 describe the same million passages under the same model. Dimension stops being confounded with dataset.

**Contributions**, ordered as the paper presents them.

1. The filter-selectivity crossover S* measured as a function of dimension — the gap the survey identified — shifting to lower selectivity as D grows, against the D-independent crossover predicted by prior work.
2. The finding that approximate filtered search fails outright under anti-correlated predicates, where only exact scan of the subset reaches the recall target.
3. A measured product-quantization recall ceiling on modern text embeddings — which rises with D rather than falling as the proposal predicted — and the finding that the sub-raw memory regime is unreachable for the five conventional families yet IS reached by scalar quantization, at 0.27-0.35x raw with no recall loss (section 5.3b).
4. A recall-constrained cost surface over D at fixed N and corpus (unfiltered, HNSW is cheapest throughout), with empirically fitted N-scaling exponents, every figure measured on queries never used for parameter selection.
5. A test of whether a small rule over query-time inputs plus a correlation proxy can pick the filter strategy — it does not beat a one-line selectivity switch, a useful negative, a reproducible resumable harness, and a measurement-validity account of between-session latency drift.

## 2. Related work

The full survey, with one row per paper and an explicit statement of what each does not cover, is in [`notes/related_work.md`](../notes/related_work.md). We summarise the positioning here.

Standard benchmarks — ANN-Benchmarks (Aumüller et al., 2020) and the more recent VIBE (Jääsaari et al., 2025), which targets modern embedding workloads across 19 datasets and 22 implementations — vary dimension only by changing dataset, so D is confounded. Matryoshka Representation Learning (Kusupati et al., 2022) supplies the truncation property our design depends on, but does not study index selection.

On filtering, the closest work to our intended headline is *Filtered ANN as a Phase Transition* (Mandarapu & Kunkunuru, 2026), which models the selectivity crossover between strategies as a phase boundary — but on SIFT1M at a fixed 128 dimensions, with D absent from the resulting formulas. Amanbayev et al. (2026) report that IVF-Flat beats HNSW at low selectivity on 768-d text embeddings, again at a single D. ACORN (Patel et al., 2024) and the arxiv-for-fanns benchmark (Iff et al., 2025) broaden filtered-ANN method coverage without isolating dimension. VADER (Chatzakis et al., 2026) shares our recall-as-a-constraint framing for filtered search, without the dimension axis. Two results inform how we read the dimension axis itself: Munyampirwa et al. (2024) show the HNSW hierarchy is largely redundant in high dimension (hub nodes carry navigation), and Herman (2026) predicts ANN recall from embedding statistics — both bear on why graph cost and the PQ ceiling move with D as they do.

Our Phase 0 survey of 13 papers found none that measures how the filter-selectivity crossover moves with dimension on a single corpus; that gap motivated the study and was recorded as a gate decision before any measurement began ([`logs/gate_reports/gate_0.md`](../logs/gate_reports/gate_0.md)).

## 3. Study design

### 3.1 Corpus, queries and embeddings

One million MS MARCO passages sampled uniformly (seed 0), with nested prefixes giving N = 100,000 and 500,000, so smaller corpora are strict subsets of larger ones. Ten thousand MS MARCO training queries (seed 1) are split **once** into 2,000 tuning and 8,000 test queries; the split is stored and asserted by the runner. Every parameter in this paper was chosen using tuning queries only, and every reported measurement comes from the test queries.

Passages are embedded once with a single hosted Matryoshka-capable model at 1536 dimensions. Lower dimensions are prefix slices of that one embedding, renormalised to unit length. All indexes use L2 on unit-norm vectors, which is rank-equivalent to cosine and inner product, so metric choice is not a free variable.

### 3.2 Dimension is varied, intrinsic difficulty barely is

Truncation changes the ambient dimension far more than it changes the data's intrinsic dimension. A maximum-likelihood estimate (Levina–Bickel, k=20) over 10,000 sampled passages gives:

|    D |   lid_mean |   lid_std |   n_sample |   k |
|-----:|-----------:|----------:|-----------:|----:|
|  128 |      31.54 |     11.43 |      10000 |  20 |
|  384 |      32.39 |     13.1  |      10000 |  20 |
|  768 |      33.12 |     13.6  |      10000 |  20 |
| 1536 |      34.2  |     14.33 |      10000 |  20 |

Ambient D rises 12x from 128 to 1536 while estimated intrinsic dimension moves only from 31.5 to 34.2 — about 8%. Cost differences across D are therefore attributable to representation width rather than to the data becoming intrinsically harder, which is exactly the separation that varying dataset cannot provide. (Source: `results/summary/lid.csv`.)

### 3.3 Indexes and parameter grids

Five index families, all FAISS: exact Flat; HNSW (M in {16, 32, 64}, efConstruction 200); IVF-Flat (nlist = c*sqrt(N), c in {2, 4, 8}); IVF-PQ (same nlist values, m in {D/16, D/8, D/4}, 8 bits per subquantizer); and IVF-PQ wrapped in exact re-ranking (IndexRefineFlat, k_factor in {4, 16}), reported separately because it retains full vectors and therefore has a completely different memory profile. Search effort is swept over efSearch in {16...1024} for HNSW and nprobe in {1...512} for IVF, capped at nlist.

This gives 52 (cell, index) combinations per recall target and, at N=500,000, 426 recorded builds and 3,442 search-sweep rows in total across all cells.

### 3.4 Recall-constrained selection

For each (cell, index family, R) we take the cheapest configuration — jointly over build variants and search settings — whose tuning-query recall reaches R, then re-measure that configuration on the test queries. A cell/index/target with no qualifying setting is recorded as `reachable = false` rather than dropped: **the absence of a feasible configuration is a result**, and it turns out to carry most of this paper's findings.

Because selection happens on tuning queries, a winner can land marginally below R on the test queries. We report this explicitly rather than letting `reachable` be mistaken for an achievement claim:

- 19 of 54 tuned winners at N=500,000 miss their target on the test queries, by at most 0.0116 (median 0.0032). Full table: [`tables/recall_achieved.md`](tables/recall_achieved.md).

### 3.5 Measurement protocol

Latency is measured in a fresh subprocess pinned to one core with a single thread, batch size 1, after a 200-query warm-up, taking the per-query median over three passes and reporting percentiles over the 8,000 test queries. Build time and throughput are measured separately with T = 16 threads, since both are legitimately parallel; rows carry a `threads` column and the two are never mixed. Each selected configuration is built three times with different seeds. Memory is measured in a process that loads only the serialized index and reports the RSS delta after warm-up — not the resident size of a process that also holds the corpus, which overstates index memory by the size of the dataset.

Recall never uses the distances an index reports. We recompute true squared distances from the raw vectors for every returned id and count those within a 1e-5 relative tolerance of the k-th ground-truth distance, so equally good neighbours are not penalised and a compressed index cannot flatter itself.

## 4. Platform and measurement validity

### 4.1 Platform

All measurements in this paper come from one host: AMD EPYC-Milan Processor, 16 cores, 58.8 GB RAM, Linux 7.0.0-28-generic, FAISS 1.15.1 (compiled with AVX2), NumPy 2.5.4, Python 3.14.3. Hardware identity `1d3d9b66e7d3e1ac` is recorded on every result row, and the analysis refuses to mix rows from different hosts.

We verified the installed FAISS reports an AVX2 build rather than assuming it. This matters: a generic build changes absolute latencies enough to move the comparisons, and a wheel that ships a single binary can be either.

### 4.2 Absolute latency drifts between sessions; occupancy is a smaller effect

We first suspected that single-thread latency depended strongly on machine-wide core occupancy, having compared a campaign measurement against a later lone re-measurement. A controlled test does not support that. Alternating 3 samples per condition on a bit-identical index (hnsw, N=100,000, D=128, efSearch=128; recall identical to five decimals across all 6 samples):

| condition    |   samples |   p50_median_us |   p50_min_us |   p50_max_us |
|:-------------|----------:|----------------:|-------------:|-------------:|
| full_pool    |         3 |           313   |        309.8 |        315.5 |
| idle_machine |         3 |           334.6 |        324.5 |        336.1 |

Occupancy accounts for 1.07x, and in the direction opposite to contention: the idle machine is marginally *slower*, plausibly an idle-state or frequency effect in the guest. Within a condition, spread is only 1.02x to 1.04x.

What *is* real is drift between sessions. The same configuration recorded during an earlier campaign measured 282.6 us under full-pool conditions, against 313.0 us today under matched conditions — a 1.11x change two days apart, with recall identical. Our original 1.3x figure conflated this drift with occupancy, and we record the correction rather than the first explanation ([`logs/decisions.md`](../logs/decisions.md)).

The consequences we actually rely on: comparisons must be drawn **within** a measurement campaign, where spread is a few percent; absolute microsecond figures should not be carried across campaigns or hosts; and a reproducibility check must re-measure under matched conditions, which ours does, agreeing to 2.7%. Stage A's cells were measured within a few hours of each other, so the comparisons in section 5 are within-campaign. A longer campaign should interleave a canary configuration to track drift directly — we did not, and flag it as a limitation.

### 4.3 Harness validation

9 of 9 machine-checked acceptance conditions pass (`results/summary/acceptance_checks.csv`). These are not smoke tests; they are the conditions under which the numbers mean anything:

| check                                   | status   |
|:----------------------------------------|:---------|
| flat recall == 1.0                      | pass     |
| hnsw recall monotone in ef_search       | pass     |
| ivf_flat recall monotone in nprobe      | pass     |
| ivf_pq recall monotone in nprobe        | pass     |
| ivf_pq_refine recall monotone in nprobe | pass     |
| HNSW reaches 0.95 at ef <= 256          | pass     |
| no implausibly perfect ANN recall       | pass     |
| index coverage per cell                 | pass     |
| no build failures                       | pass     |

Of these, the most informative is the negative one: HNSW recall at its lowest efSearch is well below 1.0. An approximate index reporting perfect recall at minimum effort is the signature of a broken harness, so we assert its absence.

### 4.4 efConstruction is fixed by measurement, not assumption

We fix efConstruction = 200 throughout. That choice is justified rather than inherited:

|   ef_construction |   mean_recall |   build_seconds |
|------------------:|--------------:|----------------:|
|               100 |        0.9853 |          5.4378 |
|               200 |        0.9894 |         11.3949 |
|               400 |        0.9902 |         22.9806 |

Raising it to 400 roughly doubles build time for a recall gain in the fourth decimal place. (Source: `results/summary/acceptance_ef_construction.csv`.)

## 5. Results

### 5.1 The cost surface over dimension at N=500,000

p95 latency in microseconds, single-threaded, measured on the 8,000 test queries. Lower is better; blank means no configuration of that family reached the target.

|    R | index_type    | 128   |   384 | 768    |   1536 |
|-----:|:--------------|:------|------:|:-------|-------:|
| 0.9  | flat          | 20618 | 65852 | 117738 | 229393 |
| 0.9  | hnsw          | 189   |   251 | 284    |    392 |
| 0.9  | hnsw_sq       | <NA>  |   230 | <NA>   |    349 |
| 0.9  | ivf_flat      | 469   |   639 | 1585   |   4802 |
| 0.9  | ivf_pq_refine | 319   |   585 | 901    |   1807 |
| 0.95 | flat          | 20618 | 65852 | 117738 | 229393 |
| 0.95 | hnsw          | 354   |   295 | 420    |    577 |
| 0.95 | hnsw_sq       | <NA>  |   256 | <NA>   |    518 |
| 0.95 | ivf_flat      | 819   |  1873 | 4811   |   8568 |
| 0.95 | ivf_pq_refine | 857   |   953 | 1835   |   5002 |
| 0.99 | flat          | 20618 | 65852 | 117738 | 229393 |
| 0.99 | hnsw          | 852   |   856 | 954    |   1370 |
| 0.99 | hnsw_sq       | <NA>  |  1349 | <NA>   |   1581 |
| 0.99 | ivf_flat      | 2790  |  6162 | 13740  |  49246 |
| 0.99 | ivf_pq_refine | 2918  |  8611 | 8369   |  23337 |

HNSW is cheapest in every cell. The headline comparison at R=0.95: HNSW runs 58x to 398x faster than exact search, with the margin widening as D grows. More interesting is the slope. From D=128 to D=1536, HNSW's p95 latency rises 1.6x (354 to 577 us) while exact search rises 11.1x. A graph index pays for dimension mainly in distance computations along a traversal whose length is governed by intrinsic structure, and section 3.2 showed intrinsic dimension barely moves under truncation; a linear scan pays the full ambient width.

This is a **negative result for the dimension hypothesis**: the cheapest index does not change with D at this scale. Because the full build grid was measured — not one variant per family — this is a statement about the index families rather than an artifact of limited search. What *does* change with D is the margin, and the ranking of the runners-up: IVF-PQ+refine overtakes IVF-Flat as D grows.

Build time, index size and resident memory for the same configurations are in [`tables/build_time.md`](tables/build_time.md) and [`tables/index_footprint.md`](tables/index_footprint.md).

### 5.2 Reachability, and the product-quantization ceiling

|       N | index_type    |   cells_x_targets |   reachable |   unreachable |
|--------:|:--------------|------------------:|------------:|--------------:|
|  100000 | flat          |                12 |          12 |             0 |
|  100000 | hnsw          |                12 |          12 |             0 |
|  100000 | ivf_flat      |                12 |          12 |             0 |
|  100000 | ivf_pq        |                12 |           0 |            12 |
|  100000 | ivf_pq_refine |                12 |           8 |             4 |
|  500000 | flat          |                12 |          12 |             0 |
|  500000 | hnsw          |                12 |          12 |             0 |
|  500000 | hnsw_sq       |                 6 |           6 |             0 |
|  500000 | ivf_flat      |                12 |          12 |             0 |
|  500000 | ivf_pq        |                12 |           0 |            12 |
|  500000 | ivf_pq_refine |                12 |          12 |             0 |
| 1000000 | flat          |                 6 |           6 |             0 |
| 1000000 | hnsw          |                 6 |           6 |             0 |
| 1000000 | ivf_flat      |                 6 |           6 |             0 |
| 1000000 | ivf_pq        |                 6 |           0 |             6 |
| 1000000 | ivf_pq_refine |                 6 |           6 |             0 |

Every unreachable cell belongs to plain IVF-PQ, which fails at **every** dimension and **every** recall target. This is not a tuning failure. The table below gives the highest recall each of the 12 (dimension, code-size) combinations attains at any swept nprobe, alongside the same codes with exact re-ranking:

|    D |   m |   compression_vs_raw |   ivf_pq |   ivf_pq_refine |
|-----:|----:|---------------------:|---------:|----------------:|
|  128 |   8 |                   64 |   0.3685 |          0.8594 |
|  128 |  16 |                   32 |   0.5407 |          0.9667 |
|  128 |  32 |                   16 |   0.7448 |          0.9986 |
|  384 |  24 |                   64 |   0.5135 |          0.9366 |
|  384 |  48 |                   32 |   0.6692 |          0.9936 |
|  384 |  96 |                   16 |   0.8356 |          0.9978 |
|  768 |  48 |                   64 |   0.5809 |          0.9492 |
|  768 |  96 |                   32 |   0.7227 |          0.9953 |
|  768 | 192 |                   16 |   0.8692 |          0.9976 |
| 1536 |  96 |                   64 |   0.6159 |          0.9548 |
| 1536 | 192 |                   32 |   0.7495 |          0.9959 |
| 1536 | 384 |                   16 |   0.8883 |          0.9971 |

Recall improves monotonically as codes get finer, reaching 0.888 at the finest setting — still short of 0.90. Critically, it saturates in search effort: doubling nprobe from 256 to 512 at the best configuration gains under 0.004, and at nprobe = nlist the scan is exhaustive, so the asymptote is quantization error rather than candidate coverage. Adding exact re-ranking over the identical codes reaches 0.999, confirming the diagnosis: the candidates are there, the approximate distances cannot rank them.

The hypotheses were **revised from the proposal** during the study, and this is one reversal worth reporting as a finding. The proposal predicted the IVF-PQ recall ceiling would *fall* with dimension at a fixed 16x compression; measured, it *rises*, from 0.745 at D=128 to 0.888 at D=1536 — opposite to the prediction, though still below 0.90, so the practical conclusion that PQ alone is unreachable is unchanged (section 6).

The practical reading is that on modern text embeddings, PQ is a *candidate generator*, not a ranker. Any deployment targeting recall@10 >= 0.90 must keep something exact to re-rank against — which has a direct and unwelcome consequence for memory budgets.

### 5.3 Memory budgets: the cheapest index does change, but feasibility binds first

Winner by cell and memory budget at R=0.95, where the budget is a multiple of the raw vector bytes (N x D x 4):

|       N |    D | 0.1x raw   | 0.5x raw   | 1.5x raw   | unbounded   |
|--------:|-----:|:-----------|:-----------|:-----------|:------------|
|  100000 |  128 | none       | none       | hnsw       | hnsw        |
|  100000 |  384 | none       | none       | hnsw       | hnsw        |
|  100000 |  768 | none       | none       | hnsw       | hnsw        |
|  100000 | 1536 | none       | none       | hnsw       | hnsw        |
|  500000 |  128 | none       | none       | ivf_flat   | hnsw        |
|  500000 |  384 | none       | hnsw_sq    | hnsw_sq    | hnsw_sq     |
|  500000 |  768 | none       | none       | hnsw       | hnsw        |
|  500000 | 1536 | none       | hnsw_sq    | hnsw_sq    | hnsw_sq     |
| 1000000 |  128 | none       | none       | none       | none        |
| 1000000 |  384 | none       | none       | hnsw       | hnsw        |
| 1000000 |  768 | none       | none       | none       | none        |
| 1000000 | 1536 | none       | none       | hnsw       | hnsw        |

Two things happen here. First, the winner **does** change with the budget: at D=128, N=500,000 the unbounded winner is HNSW but the 1.5x-raw winner is IVF-Flat. The mechanism is concrete — the tuned HNSW at that cell uses M=64, whose link storage (N x M x 2 x 4 bytes) is comparable to the vectors themselves at low D, pushing the index near 2x raw. Measured footprints are in [`tables/index_footprint.md`](tables/index_footprint.md). (A `none` entry means no configuration of any index met the recall target within that budget.)

Second: within the five families of the main grid, **the 0.5x and 0.1x budgets have no feasible winner anywhere**. The only family small enough is IVF-PQ, and section 5.2 showed it cannot reach R >= 0.90. That result prompted the probe in the next section, which changes the conclusion.

### 5.3b Scalar quantization occupies the sub-raw regime the main grid left empty

The main grid's five families are either larger than the raw vectors (Flat, HNSW, IVF-Flat, IVF-PQ+refine) or unable to reach the recall targets (IVF-PQ), so a sub-raw budget had no feasible answer. That is a statement about the families measured, not about what is possible, so we added one: HNSW over 8-bit scalar-quantized storage, measured **in the same run** as uncompressed HNSW at the same cells, since this host's between-session drift would otherwise be folded into a claim about compression.

| index_type   |    D |    R |   recall_test |   lat_p95_us |   rss_x_raw |
|:-------------|-----:|-----:|--------------:|-------------:|------------:|
| hnsw         |  384 | 0.9  |        0.9346 |      250.762 |       1.096 |
| hnsw_sq      |  384 | 0.9  |        0.9324 |      229.52  |       0.346 |
| hnsw         |  384 | 0.95 |        0.9525 |      294.824 |       1.096 |
| hnsw_sq      |  384 | 0.95 |        0.9492 |      255.833 |       0.346 |
| hnsw         |  384 | 0.99 |        0.9908 |      856.167 |       1.096 |
| hnsw_sq      |  384 | 0.99 |        0.9895 |     1349.09  |       0.346 |
| hnsw         | 1536 | 0.9  |        0.9113 |      392.077 |       1.024 |
| hnsw_sq      | 1536 | 0.9  |        0.9094 |      349.137 |       0.274 |
| hnsw         | 1536 | 0.95 |        0.9604 |      576.735 |       1.024 |
| hnsw_sq      | 1536 | 0.95 |        0.9566 |      518.403 |       0.274 |
| hnsw         | 1536 | 0.99 |        0.9908 |     1369.74  |       1.045 |
| hnsw_sq      | 1536 | 0.99 |        0.9908 |     1580.94  |       0.295 |

Resident memory falls from 1.02-1.10x raw to 0.27-0.35x — inside the 0.5x budget at every measured cell, while still reaching every recall target. **The correct statement is therefore not that no index meets a sub-raw budget, but that none of the five conventional families does: a scalar-quantized graph index qualifies at 0.5x, and nothing we measured qualifies at 0.1x**, which would need roughly 10x compression against SQ8's 4x.

Compression improves with dimension (0.35x at D=384 against 0.27x at D=1536) because SQ8 shrinks only the vectors, by a factor of four, while the graph links are D-independent: at low D that fixed overhead is a larger share of the total.

Compression is also not a pure latency loss. At R=0.90 and 0.95 the quantized index is **faster** — ratio 0.87-0.92 — because fewer bytes per vector means less memory traffic per distance evaluation, which outweighs the decode cost. At R=0.99 it reverses (1.15-1.58x), since compensating for quantization error needs a higher efSearch and the extra traversal costs more than the bytes saved. Quantization trades latency for memory in a way that depends on the recall target, which is exactly the kind of dependence a recall-constrained framing is meant to expose.

This is a two-cell probe, not a full stage, and it tests one compression technique. It is enough to show the regime is reachable; locating the frontier would need 4-bit SQ, finer PQ with re-ranking, and more cells.

### 5.4 The scale axis: HNSW absorbs a 10x corpus at constant cost

p95 latency in microseconds at R=0.95, by index type and corpus size, at the dimensions measured across all three N:

|    D | index_type    | 100000   |   500000 | 1000000   |
|-----:|:--------------|:---------|---------:|:----------|
|  384 | flat          | 11791    |    65852 | 102028    |
|  384 | hnsw          | 281      |      295 | 295       |
|  384 | hnsw_sq       | <NA>     |      256 | <NA>      |
|  384 | ivf_flat      | 703      |     1873 | 2907      |
|  384 | ivf_pq_refine | 565      |      953 | 1116      |
| 1536 | flat          | 44632    |   229393 | 391951    |
| 1536 | hnsw          | 546      |      577 | 532       |
| 1536 | hnsw_sq       | <NA>     |      518 | <NA>      |
| 1536 | ivf_flat      | 3526     |     8568 | 9279      |
| 1536 | ivf_pq_refine | 2647     |     5002 | 5330      |

Over a 10x increase in corpus size, HNSW's p95 latency at fixed recall changes by a factor of 1.01 — no measurable growth — while exact search grows 8.7x, i.e. linearly as expected. IVF-Flat and IVF-PQ+refine grow sub-linearly but measurably. The HNSW figure is non-monotonic across N and the variation is within the between-session drift band of section 4.2, so the honest statement is that we cannot distinguish its growth from zero over this range, not that it is exactly flat.

This has a practical reading: within this range, a graph index's query cost is governed by the recall target and the data's structure rather than by how many vectors are stored. Note the contrast with the next section, where cost averaged over the whole search grid does grow with N at fixed *effort* — a different quantity from the fixed-*recall* cost here, and the two should not be conflated.

### 5.5 How cost scales with corpus size, averaged over the search grid

Exponents fitted across the measured corpus sizes (N in {100K, 500K, 1M}) at matched D (median over D), cost proportional to N^exponent. These are averaged over the whole search grid (all swept efSearch/nprobe), which is a **different quantity** from the fixed-recall cost of section 5.4 — they must not be read as contradicting it:

| quantity      | index_type    |   median_exponent |   n_D |   N_lo |    N_hi |
|:--------------|:--------------|------------------:|------:|-------:|--------:|
| build_seconds | flat          |              0.93 |     4 | 100000 | 1000000 |
| build_seconds | hnsw          |              1.05 |     4 | 100000 | 1000000 |
| build_seconds | ivf_flat      |              1.91 |     4 | 100000 | 1000000 |
| build_seconds | ivf_pq_refine |              0.88 |     3 | 100000 | 1000000 |
| latency_us    | flat          |              1.02 |     4 | 100000 | 1000000 |
| latency_us    | hnsw          |              0.23 |     4 | 100000 | 1000000 |
| latency_us    | ivf_flat      |              0.35 |     4 | 100000 | 1000000 |
| latency_us    | ivf_pq        |              0.42 |     4 | 100000 | 1000000 |
| latency_us    | ivf_pq_refine |              0.44 |     4 | 100000 | 1000000 |

These are measurements, and they differ from the values complexity arguments would suggest. Three are worth naming. IVF-Flat **build** is strongly super-linear, because nlist grows with sqrt(N) and the k-means training sample grows with nlist, so training cost compounds. HNSW **search**, averaged over the grid, grows faster than the logarithmic intuition — more traversal per query as the graph grows at a fixed search effort (this is the grid-averaged exponent; at fixed *recall* the cost is flat, section 5.4). IVF-Flat **search** is flatter than the naive sqrt(N) guess, because a fixed nprobe scans a shrinking fraction of the corpus as nlist grows.

We report these as descriptive fits over three corpus sizes, not as scaling laws: three points constrain a power law only weakly. Their immediate use was operational — they corrected our own compute projection for later stages, which had assumed the theoretical values and was wrong in both directions.

Full log-log fits `log y = a + b log N + c log D + d (log N)(log D)`. The D-exponent **c** is the quantity H1c refers to; it varies markedly across index types (no bootstrap CI columns yet — see limitations):

| index_type    |    R | metric      |          a |         b |         c |            d |   n_points |
|:--------------|-----:|:------------|-----------:|----------:|----------:|-------------:|-----------:|
| flat          | 0.9  | p95_latency | -11.7211   |  1.29757  |  1.59502  | -0.0490919   |         10 |
| flat          | 0.9  | build_time  | -14.2249   |  0.56879  |  0.215864 |  0.0662411   |         10 |
| flat          | 0.95 | p95_latency | -11.7211   |  1.29757  |  1.59502  | -0.0490919   |         10 |
| flat          | 0.95 | build_time  | -14.2249   |  0.56879  |  0.215864 |  0.0662411   |         10 |
| flat          | 0.99 | p95_latency | -11.7211   |  1.29757  |  1.59502  | -0.0490919   |         10 |
| flat          | 0.99 | build_time  | -14.2249   |  0.56879  |  0.215864 |  0.0662411   |         10 |
| hnsw          | 0.9  | p95_latency |   2.4543   |  0.111241 |  0.274004 | -0.000595986 |         10 |
| hnsw          | 0.9  | build_time  | -31.2694   |  2.34128  |  3.65187  | -0.217469    |         10 |
| hnsw          | 0.95 | p95_latency |  -0.652332 |  0.382865 |  0.874435 | -0.0475958   |         10 |
| hnsw          | 0.95 | build_time  | -37.1826   |  2.86578  |  4.50923  | -0.29356     |         10 |
| hnsw          | 0.99 | p95_latency |  -3.90162  |  0.732217 |  1.34005  | -0.0879479   |         10 |
| hnsw          | 0.99 | build_time  | -36.2811   |  2.77283  |  4.12615  | -0.258195    |         10 |
| ivf_flat      | 0.9  | p95_latency |   2.98602  | -0.179093 | -0.160174 |  0.0918386   |         10 |
| ivf_flat      | 0.9  | build_time  | -60.4987   |  4.80123  |  6.79707  | -0.503904    |         10 |
| ivf_flat      | 0.95 | p95_latency |  -7.60295  |  0.745242 |  1.51728  | -0.0452511   |         10 |
| ivf_flat      | 0.95 | build_time  | -44.954    |  3.40455  |  4.22604  | -0.272047    |         10 |
| ivf_flat      | 0.99 | p95_latency |  13.4294   | -0.882314 | -1.75087  |  0.224402    |         10 |
| ivf_flat      | 0.99 | build_time  | -71.4145   |  5.69499  |  8.74023  | -0.662877    |         10 |
| ivf_pq_refine | 0.9  | p95_latency |  -3.21917  |  0.398963 |  0.881332 | -0.0113714   |          9 |
| ivf_pq_refine | 0.9  | build_time  |  -6.79136  |  0.301097 |  0.11909  |  0.0642285   |          9 |
| ivf_pq_refine | 0.95 | p95_latency |  -9.7235   |  0.919704 |  1.976    | -0.0892904   |          9 |
| ivf_pq_refine | 0.95 | build_time  |  -3.6397   |  0.154729 | -0.710484 |  0.117105    |          9 |
| ivf_pq_refine | 0.99 | p95_latency | -19.2574   |  1.77274  |  3.22974  | -0.184584    |          8 |
| ivf_pq_refine | 0.99 | build_time  | -29.8507   |  2.19536  |  3.00661  | -0.169925    |          8 |

### 5.6 A decision rule — on the filter axis, where the answer changes

A depth-3 tree predicting the cheapest **index** from (log10 N, D, R, memory budget) scores 1.000 top-1 on the held-out N=1,000,000 cells, which matches "always choose HNSW" (1.000): unfiltered, one index wins nearly everywhere, so there is little for a selector to learn. The decision only becomes non-trivial on the axis the index rule never sees — the filter.

We therefore (post-hoc) refit the rule to pick the cheapest **filter strategy** (pre-filter exact scan / in-index filtered / post-filter over-fetch) from inputs a system has at query time: selectivity S, dimension D, recall target R, and a **correlation proxy**. Correlation is a property of the predicate-query relationship and is not observable directly, so the proxy estimates it the way a deployed system could: it issues a cheap *approximate* unfiltered top-m probe and takes the fraction of those candidates that satisfy the predicate, divided by the global selectivity (`results/summary/filter_proxy.csv`, `proxy_source = online_probe`; an idealized exact-ground-truth variant is computed only for comparison). The oracle label per cell is the cheapest reachable strategy.

On the 66 held-out N=1,000,000 cells a depth-3 tree over (log10_S,D,R,corr_proxy) scores 0.667 top-1 [95% CI 0.55-0.77] with regret 0.105 (mean extra latency over the per-cell oracle). The fair comparison is against the same tree without the proxy and against a one-line selectivity switch, not only against a fixed strategy:

| policy | top-1 | regret |
|---|---|---|
| proxy rule (S, D, R, corr. proxy) | 0.667 | 0.105 |
| same tree without the proxy (S, D, R) | 0.727 | 16.493 |
| fixed 1%-selectivity switch | 0.530 | 0.071 |
| always exact scan of the subset | 0.439 | 192.352 |
| always in-index filtered | 0.379 | 0.115 |

**The rule does not beat the fair baselines:** on top-1 it is matched or beaten by the same tree without the proxy (S, D, R) (0.727); on regret by the fixed 1%-selectivity switch (0.071). 
What the proxy does buy is protection from the worst choices: without it the tree's regret is 16.5 against 0.105 with it, because S, D and R alone cannot tell an anti-correlated predicate (where only exact scan reaches R) from a benign one, and the rare wrong call there is very expensive. A plain selectivity threshold avoids those calls just as well. The exploratory H3 is therefore **not supported** on this grid; a proxy computed from a cheap probe is not yet worth a learned rule.

The fitted tree is in [`../results/summary/strategy_rule_tree.txt`](../results/summary/strategy_rule_tree.txt).

One caveat the rule inherits: its oracle labels come from the Stage C runs, which include the smallest correlated level (S=0.001) whose *measured* selectivity overshoots the target by 54-61% (section 5.8). Those cells' labels — and any rule split that depends on them — should be read against measured, not nominal, S, and will firm up once the Stage C rerun with finer correlated filters lands.

### 5.7 Filtered search: where the crossover sits, and that it moves with dimension

For each (index, cell, correlation, recall target) we locate S*, the selectivity at which in-index filtered search and exact scan of the filtered subset swap places, by interpolating log-latency against log S between the two measured curves. Intervals propagate the per-row latency bootstrap. A crossover was located in 22 of 46 (index, cell, correlation, target) combinations; elsewhere one curve dominates across the whole measured S range.

| index_type   |       N |    D | correlation   |    R |   crossover_S |   crossover_S_lo |   crossover_S_hi |
|:-------------|--------:|-----:|:--------------|-----:|--------------:|-----------------:|-----------------:|
| hnsw         |  500000 |  384 | none          | 0.9  |    0.0119332  |       0.0118835  |       0.0119837  |
| hnsw         |  500000 |  384 | none          | 0.95 |    0.0127548  |       0.0126864  |       0.0128006  |
| hnsw         |  500000 | 1536 | none          | 0.95 |    0.0105856  |       0.0105673  |       0.0106039  |
| hnsw         | 1000000 |  384 | none          | 0.95 |    0.0100355  |       0.0100192  |       0.0100563  |
| ivf_flat     | 1000000 | 1536 | none          | 0.9  |    0.00113838 |       0.00113779 |       0.00113899 |
| hnsw         |  500000 |  384 | positive      | 0.9  |    0.00101981 |       0.00101306 |       0.00102565 |
| hnsw         |  500000 |  384 | positive      | 0.95 |    0.00126638 |       0.00125416 |       0.0012774  |
| hnsw         |  500000 |  384 | positive      | 0.99 |    0.00251441 |       0.00247048 |       0.00256095 |
| hnsw         |  500000 | 1536 | positive      | 0.95 |    0.00105266 |       0.00104743 |       0.00105878 |
| hnsw         |  500000 | 1536 | positive      | 0.99 |    0.0015828  |       0.00156301 |       0.00160292 |
| hnsw         | 1000000 |  384 | positive      | 0.99 |    0.00208224 |       0.00205291 |       0.00211017 |
| hnsw         | 1000000 | 1536 | positive      | 0.99 |    0.00112451 |       0.00111489 |       0.00113333 |
| ivf_flat     |  500000 |  384 | positive      | 0.9  |    0.00120843 |       0.00119823 |       0.00121834 |
| ivf_flat     |  500000 |  384 | positive      | 0.95 |    0.00129028 |       0.0012771  |       0.00130263 |
| ivf_flat     |  500000 |  384 | positive      | 0.99 |    0.00235542 |       0.00230691 |       0.00241556 |
| ivf_flat     |  500000 | 1536 | positive      | 0.9  |    0.00131075 |       0.00129747 |       0.00132396 |
| ivf_flat     |  500000 | 1536 | positive      | 0.95 |    0.00136723 |       0.00134919 |       0.00138513 |
| ivf_flat     |  500000 | 1536 | positive      | 0.99 |    0.00162751 |       0.00160062 |       0.00165384 |
| ivf_flat     | 1000000 |  384 | positive      | 0.9  |    0.00198682 |       0.00193573 |       0.00203641 |
| ivf_flat     | 1000000 |  384 | positive      | 0.95 |    0.00203117 |       0.00195857 |       0.00209325 |
| ivf_flat     | 1000000 |  384 | positive      | 0.99 |    0.00413867 |       0.00397833 |       0.00428426 |
| ivf_flat     | 1000000 | 1536 | positive      | 0.99 |    0.00111699 |       0.00110208 |       0.00113076 |

The headline numbers: for an uncorrelated predicate S* sits at a median of 0.0106 — in-index search is preferable above roughly 1.1% selectivity and exact scan below it. For a positively correlated predicate S* drops to 0.00137, a factor of 7.7 lower: when the filter admits vectors near the query, in-index search stays the better choice an order of magnitude further down the selectivity range. That is H2b, and it is the single most actionable number in this study — a system tuning a strategy switch on selectivity alone, without regard to correlation, will choose wrongly across a tenfold band.

**H2: does S\* move with dimension?** Comparing the lowest and highest measured D within each (N, index, correlation, target):

|       N | index_type   | correlation   |    R |   D_low |   D_high |   S_star_D_low |   S_star_D_high |   ratio_high_over_low | intervals_disjoint   |
|--------:|:-------------|:--------------|-----:|--------:|---------:|---------------:|----------------:|----------------------:|:---------------------|
|  500000 | hnsw         | none          | 0.95 |     384 |     1536 |     0.0127548  |      0.0105856  |              0.829933 | True                 |
|  500000 | hnsw         | positive      | 0.95 |     384 |     1536 |     0.00126638 |      0.00105266 |              0.831234 | True                 |
|  500000 | hnsw         | positive      | 0.99 |     384 |     1536 |     0.00251441 |      0.0015828  |              0.629492 | True                 |
|  500000 | ivf_flat     | positive      | 0.9  |     384 |     1536 |     0.00120843 |      0.00131075 |              1.08467  | True                 |
|  500000 | ivf_flat     | positive      | 0.95 |     384 |     1536 |     0.00129028 |      0.00136723 |              1.05964  | True                 |
|  500000 | ivf_flat     | positive      | 0.99 |     384 |     1536 |     0.00235542 |      0.00162751 |              0.690967 | True                 |
| 1000000 | hnsw         | positive      | 0.99 |     384 |     1536 |     0.00208224 |      0.00112451 |              0.540049 | True                 |
| 1000000 | ivf_flat     | positive      | 0.99 |     384 |     1536 |     0.00413867 |      0.00111699 |              0.269892 | True                 |

S* differs in all 8 pairs, and the intervals are disjoint in 8 of them, which is the criterion the plan set in advance for treating the effect as more than noise. The median ratio S*(D_high)/S*(D_low) is 0.76: **the crossover moves to lower selectivity as dimension rises**, so at high D in-index filtered search remains preferable over a wider range of predicates than at low D. The mechanism is visible in section 5.1 — exact scan pays the full ambient width of the vector while a graph traversal does not, so raising D penalises the baseline more than the index.

This is the gap the Phase 0 survey identified: the closest prior work models the selectivity crossover as a phase boundary but at a single fixed dimension, with D absent from its formulas. On this corpus, D belongs in them.

### 5.8 Adversarially correlated filters defeat approximate search at the swept effort

Reachability of each strategy by filter correlation, over all (cell, target) combinations:

| strategy              | correlation   |   cells_x_targets |   reachable |   unreachable |
|:----------------------|:--------------|------------------:|------------:|--------------:|
| in_index_filtered     | negative      |                72 |           0 |            72 |
| in_index_filtered     | none          |               120 |          81 |            39 |
| in_index_filtered     | positive      |                72 |          72 |             0 |
| post_filter_overfetch | negative      |               144 |           0 |           144 |
| post_filter_overfetch | none          |               240 |         155 |            85 |
| post_filter_overfetch | positive      |               144 |         144 |             0 |
| pre_filter_exact      | negative      |                36 |          36 |             0 |
| pre_filter_exact      | none          |                60 |          60 |             0 |
| pre_filter_exact      | positive      |                36 |          36 |             0 |

The negative-correlation rows are the strongest result in the filtered study. When the predicate admits only vectors *far* from the query, in-index filtered search and post-filter over-fetch reach recall@10 >= 0.90 in **0 of 216** attempts — that is, never, at any swept effort — while exact scan over the matching subset succeeds in all 36/36. Approximate search is not merely slower under adversarial correlation; within this grid it does not work at all.

The reason is structural rather than a tuning failure. A graph index reaches a neighbourhood by descending towards the query, and an IVF index probes the lists nearest the query; both concentrate effort exactly where an anti-correlated predicate admits nothing. Over-fetching does not rescue it, because the candidates retrieved are drawn from the same wrong region. For a workload whose filters can be anti-correlated with queries, the safe design is to scan the filtered subset exactly and to size the system for that cost.

One boundary to state precisely: the IVF nprobe sweep caps at 512 while nlist is at least 1,414 at these N, so for IVF this is **the swept, non-exhaustive effort failing**, not a proof that an exhaustive nprobe = nlist scan would. An exhaustive anti-correlated IVF check is scheduled in the Stage C rerun. The HNSW result (no exhaustive dial short of full traversal) and the exact-scan success are unaffected by this caveat.

**A measurement caveat we must report.** The plan requires mean measured selectivity within 20% of target for each correlated level:

| correlation   |   target_S |   measured_S_mean |   relative_error_pct | within_20pct   |
|:--------------|-----------:|------------------:|---------------------:|:---------------|
| negative      |      0.001 |          0.001544 |                 54.4 | False          |
| negative      |      0.01  |          0.010542 |                  5.4 | True           |
| negative      |      0.1   |          0.100543 |                  0.5 | True           |
| none          |      0.001 |          0.000975 |                 -2.5 | True           |
| none          |      0.01  |          0.010065 |                  0.6 | True           |
| none          |      0.1   |          0.099884 |                 -0.1 | True           |
| none          |      0.5   |          0.499852 |                 -0   | True           |
| none          |      1     |          1        |                  0   | True           |
| positive      |      0.001 |          0.001611 |                 61.1 | False          |
| positive      |      0.01  |          0.010657 |                  6.6 | True           |
| positive      |      0.1   |          0.100668 |                  0.7 | True           |

2 level(s) fail that check, all at the smallest correlated target (worst 61% over target). The cause is granularity rather than a bug: correlated predicates are built by unioning whole k-means clusters, and with 1,000 clusters the smallest admissible union is already near 0.1% of the corpus, so a 0.001 target cannot be hit from above. The uncorrelated levels, which use an exact range predicate, are within 0.6%. S* estimates at the smallest correlated selectivity should therefore be read as applying to the measured selectivity rather than the nominal one; finer granularity would need more clusters or sub-cluster sampling.

### 5.9 Updates: insertion is nearly free in recall, expensive in throughput

Ten batches of 5,000 vectors inserted into an index built on 450,000, at search parameters fixed before the first insert, with no retraining. Recall is measured against ground truth restricted to the vectors the index actually contains:

| index_type   |   recall_first_batch |   recall_last_batch |    drift |   rebuild_reference |   inserts_per_second_median |   p95_us_median |
|:-------------|---------------------:|--------------------:|---------:|--------------------:|----------------------------:|----------------:|
| hnsw         |              0.9998  |             0.99971 | -9e-05   |             0.99972 |                        1571 |           10890 |
| ivf_flat     |              0.99246 |             0.99256 |  0.0001  |             0.99228 |                       56113 |           26699 |
| ivf_pq       |              0.71321 |             0.71356 |  0.00035 |             0.71245 |                       58576 |           11465 |

Recall drift is negligible for all three families — the largest absolute change over the whole insert run is 0.00035 (ivf_pq) — and each ends within 0.00111 of a from-scratch rebuild on the same vectors. Inserting without retraining costs essentially nothing in recall at this ratio (50,000 added to 450,000, about 11%).

The real asymmetry is throughput. IVF-Flat and IVF-PQ absorb inserts at 56,113-58,576 vectors/second, against 1,571 for HNSW — a factor of roughly 37. HNSW wins every latency comparison in this study and loses the ingest comparison by that margin, which is the clearest cost/benefit trade in our results.

Two caveats on this stage. First, the recall-drift design cannot exhibit IVF centroid staleness: the inserted vectors are a random sample of the **same** corpus, so their distribution matches the training sample and the centroids stay representative by construction. Demonstrating staleness needs distribution shift between the initial and inserted vectors, which we did not test. Second, the absolute p95 latencies in the table (HNSW ~10,890 us) are measured at the search parameters fixed before the first insert — tuned for R=0.95 on the 450,000-vector prefix at D=768, N=500,000 — not re-tuned per batch; they are not comparable to the tuned operating points of section 5.1 and should not be read as HNSW's query latency at this scale. We did not test deletions, and the insert ratio here is modest; a system adding many multiples of its corpus should not read these numbers as licence to never rebuild.

## 6. Hypothesis status

H1a-H2b are the **proposal's original RQ1-RQ3 predictions**, judged against the measured data; one of them, H1b, reverses outright. **H3 is labelled *exploratory* and is post-hoc, not pre-registered**: the proposal's original H3 was a decision rule over the *index* choice, which the data did not support (unfiltered, one index wins everywhere, so it only ties the trivial heuristic); the filter-strategy rule reported as H3 here was formulated *after* that result, on the axis where the choice actually changes. Verdicts are computed from the result files by `annstudy-analysis`, not asserted (`reports/findings.md`):

| Hypothesis | Statement | Origin | Status | Basis |
|---|---|---|---|---|
| H1a | HNSW:IVF-PQ memory ratio widens with dimension D | proposal RQ1 | **not supported** | at fixed 16x compression the HNSW:IVF-PQ bytes-per-vector ratio is arithmetically ~constant in D (HNSW's extra graph bytes are D-independent, so the ratio narrows slightly with D, not widens) |
| H1b | IVF-PQ recall ceiling FALLS with D at fixed 16x compression | proposal RQ1 | **not supported** | proposal predicted the ceiling falls with D; measured it rises, 0.745 (D=128) -> 0.888 (D=1536) at 16x compression, still < 0.90 so IVF-PQ stays unreachable |
| H1c | Fitted D-exponent of p95 latency differs by index type | proposal RQ1 | **supported** | D-exponent (c) of p95 latency spans 2.16 across indexes (no bootstrap CI columns — qualitative; gap C1) |
| H2 | The filter-selectivity crossover S* (in-index vs pre-filter exact) moves with D | proposal RQ2 | **supported** | 8 D-pairs compared; S* differs in 8, with non-overlapping intervals in 8; median ratio S*(D_high)/S*(D_low) = 0.76 |
| H2b | S* also depends on filter correlation (positive vs negative) | proposal RQ2 | **supported** | median S* by correlation: none: 0.01059, positive: 0.001367 |
| H3 | A small rule picks the cheapest FILTER STRATEGY from S, D, R and a correlation proxy | exploratory, post-hoc | **not supported** | [exploratory, post-hoc] rule top-1/regret 0.667/0.105 on 66 held-out cells vs baselines (top-1/regret): always exact scan 0.439/192.352; always in-index 0.379/0.115; 1% selectivity switch 0.530/0.071; same tree without proxy 0.727/16.493 (the index-choice rule only ties the heuristic: top-1 1.000 vs baseline 1.000 — it never sees the filter axis) |

We draw attention to the shape of this table. H1b is not merely unsupported but **reversed** — the PQ ceiling rises with D rather than falling (section 5.2) — which we report as a finding, not hide. The filter-axis hypotheses (H2, H2b) are the supported headline; H3 asks whether a rule can exploit that axis. A paper that reported only the supported rows would be a different and less honest paper.

## 7. Threats to validity

**Single host.** Every number comes from one machine. Absolute latencies are not portable, and we make no claim that they are; the relative ordering is what we argue about. Results are keyed by hardware identity so that cross-host contamination is impossible rather than merely discouraged.

**Between-session drift.** Section 4.2: absolute latency for an identical index moved ~1.1x across two days on this host, cause unisolated (co-tenancy on a shared instance is the obvious suspect). Mitigated by comparing only within a campaign, where spread is a few percent, but it caps the precision of any absolute figure we publish. A canary configuration interleaved through the campaign would measure the drift directly; we did not do this and consider it the most valuable protocol addition for future runs.

**One embedding model, one corpus.** The PQ ceiling and the dimension slopes may be properties of this model's geometry. The Matryoshka design that lets us isolate D also ties us to a model family trained to support truncation; an embedding not trained that way might degrade differently under prefix slicing.

**Three corpus sizes.** The scaling exponents in section 5.5 are three-point fits. They are reported as descriptive and were used operationally, not as laws.

**Tuning-to-test gap.** Section 3.4 reports the winners that miss their target on held-out queries. The shortfalls are small but non-zero, and they are a property of selecting on a 2,000-query sample, not of the indexes.

**Grid boundaries.** `reachable = false` always means 'within the swept grid'. For IVF-PQ we checked that recall saturates in nprobe before concluding the ceiling is intrinsic, but a wider grid — more subquantizers, more bits — could in principle change the conclusion at a memory cost that would defeat the purpose.

## 7b. Figure specifications

These are **specifications**, not the figures: each box names what to plot, the result file it comes from, and the PNG to render. The analysis step writes the PNGs named under `Render` (`make analysis`); drop each in where its box sits before submission.

```text
FIGURE 1. Cost vs dimension at R=0.95
  Plot    : p95 latency (us, log y) vs D in {128,384,768,1536} (log x), one line per index family; HNSW is the lowest line at every D.
  Source  : results/summary/cell_costs.csv (R=0.95, objective lat_p95_us)
  Render  : figures/cost_vs_D_R0.95.png
```

```text
FIGURE 2. Cost vs dimension at R=0.99
  Plot    : as Figure 1 at R=0.99; shows IVF-PQ+refine rising fastest and IVF-PQ absent (unreachable).
  Source  : results/summary/cell_costs.csv (R=0.99, objective lat_p95_us)
  Render  : figures/cost_vs_D_R0.99.png
```

```text
FIGURE 3. Winner map by objective
  Plot    : heatmap of the cheapest index per (D, R) cell; one panel each for p95 latency, build time, and bytes per vector.
  Source  : results/summary/winners.csv
  Render  : figures/winner_heatmap_lat_p95_us.png, figures/winner_heatmap_build_seconds_mean.png, figures/winner_heatmap_index_bytes.png
```

```text
FIGURE 4. IVF-PQ recall ceiling vs search effort
  Plot    : recall@10 vs nprobe (log x) for IVF-PQ at each D; curves saturate below 0.90, showing the ceiling is quantization error not candidate coverage.
  Source  : results/raw/sweeps/ (index_type=ivf_pq, N=500000)
  Render  : figures/pq_ceiling.png (generate; not yet rendered)
```

```text
FIGURE 5. Filter-selectivity crossover S* vs dimension (H2 headline)
  Plot    : p95 latency vs selectivity S (log-log) for in-index filtered search and pre-filter exact scan, one panel per D in {384,1536}; mark S* where the curves cross and show the leftward shift as D rises.
  Source  : results/summary/crossover.csv, crossover_h2.csv; results/raw/filter_runs.parquet
  Render  : figures/crossover.png (generate; not yet rendered)
```

```text
FIGURE 6. Update recall drift (Stage D)
  Plot    : recall@10 at fixed search params vs inserted-vector count, one line per index family, with the from-scratch rebuild reference; shows negligible drift.
  Source  : results/raw/update_runs/*.parquet
  Render  : figures/update_drift.png (generate; not yet rendered)
```

```text
FIGURE 7. Filter-strategy decision tree (H3)
  Plot    : the fitted depth-3 tree (features log10 S, D, R, correlation proxy) with the cheapest-strategy leaf labels; show the corr-proxy split that separates the anti-correlated regime (exact scan) from the rest.
  Source  : results/summary/strategy_rule_tree.txt, strategy_rule_accuracy.csv
  Render  : figures/strategy_tree.png (generate; not yet rendered)
```

## 8. What is not yet measured

All four stages (A-D) are complete; what remains are specific robustness checks and one rerun the filter headline depends on. Stated plainly, because the gap between what a harness can do and what has been run is where benchmark papers usually mislead:

- **Stage C rerun (finer correlated filters).** The smallest correlated level (S=0.001) overshoots its target by 54-61% because whole k-means clusters (k=1000) are too coarse to form so small a subset (section 5.8). A rerun with more clusters and extra selectivity levels (`configs/stage_c_rerun.yaml`) is needed before the crossover S* and the strategy rule's labels at low S are final; the direction of every filter result is expected to hold, but the exact low-S numbers may move.
- **Exhaustive anti-correlated IVF.** The nprobe sweep capped below nlist, so the anti-correlation result (section 5.8) is 'the swept effort fails', pending an nprobe = nlist check.
- **Beyond 4x compression.** Scalar quantization answered whether the sub-raw regime is reachable (section 5.3b: yes, at 0.5x). Locating the frontier — and whether anything meets 0.1x at R >= 0.90 — needs 4-bit SQ or finer PQ with re-ranking, and more cells than the two-cell probe used here.
- **Online vs idealized correlation proxy.** The proxy is now computed from an approximate probe (section 5.6); the exact-GT variant is retained only as an upper bound.
- **Update drift under distribution shift.** Stage D inserts a random sample of the same corpus, which cannot exhibit IVF centroid staleness (section 5.9); a shifted-distribution insert stream would.
- **hnswlib cross-check** at one cell, and **k in {1, 100}** (all results here are k=10) — side checks the plan lists that were not run.
- **Archival release** with a persistent identifier is prepared but not yet minted.

## 9. Reproducibility

The harness is resumable and append-only: completed configurations are skipped on restart, raw rows are never edited, and superseded rows are marked rather than deleted. Every row carries run id, git commit, hardware id, timestamp, seed and a configuration hash.

```
uv venv && uv pip install -e ".[dev,hnswlib]"
make hardware      # record the platform
make test          # harness tests
make pilot         # reproduce the pilot cell end to end
make acceptance    # the checks in section 4.3
make stage-a       # the measurements in section 5
make analysis      # regenerate every table and figure
make paper         # regenerate this document
```

Decisions, deviations and defects found along the way — including several that would have corrupted results and were caught by the acceptance checks — are dated in [`logs/decisions.md`](../logs/decisions.md). We consider that log part of the paper's evidence, not supporting material.

## References

Links and per-paper coverage notes in [`notes/related_work.md`](../notes/related_work.md).

- Aumüller, Bernhardsson, Faithfull. ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms. 2020. arXiv:1807.05614
- Jääsaari et al. VIBE: Vector Index Benchmark for Embeddings. 2025. arXiv:2505.17810
- Malkov, Yashunin. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE TPAMI, 2018. arXiv:1603.09320
- Jégou, Douze, Schmid. Product quantization for nearest neighbor search. IEEE TPAMI, 2011
- Douze et al. The Faiss library. 2024. arXiv:2401.08281
- Kusupati et al. Matryoshka Representation Learning. 2022. arXiv:2205.13147
- Patel et al. ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data. 2024. arXiv:2403.04871
- Mandarapu, Kunkunuru. Filtered ANN as a Phase Transition: When Selectivity-Estimation Error Causes Plan Regret. 2026. arXiv:2606.16341
- Amanbayev et al. Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis. 2026. arXiv:2602.11443
- Iff et al. Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors. 2025. arXiv:2507.21989
- Chatzakis et al. VADER: Filtered Vector Search with Declarative Recall. 2026. arXiv:2609.38053
- Munyampirwa et al. Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'. 2024. arXiv:2412.01940
- Herman. Closed Forms and Synthetic Twins: Predicting Approximate Nearest Neighbor Recall from Embedding Statistics. 2026. arXiv:2609.00364

## Appendix A. Provenance of every table

| Table | Generated from |
|---|---|
| [`tables/cost_surface.md`](tables/cost_surface.md) | `results/tuned/tuned.parquet` |
| [`tables/recall_achieved.md`](tables/recall_achieved.md) | `results/tuned/tuned.parquet` |
| [`tables/reachability.md`](tables/reachability.md) | `results/tuned/tuned.parquet` |
| [`tables/pq_ceiling.md`](tables/pq_ceiling.md) | `results/raw/sweeps/*.parquet + results/raw/builds/*.parquet` |
| [`tables/memory_budget_winners.md`](tables/memory_budget_winners.md) | `results/summary/winners.csv` |
| [`tables/index_footprint.md`](tables/index_footprint.md) | `results/tuned/tuned.parquet (rss from results/raw/memory/)` |
| [`tables/n_exponents.md`](tables/n_exponents.md) | `results/raw/builds/*.parquet + results/raw/sweeps/*.parquet` |
| [`tables/build_time.md`](tables/build_time.md) | `results/tuned/tuned.parquet (T-thread builds)` |
| [`tables/lid.md`](tables/lid.md) | `results/summary/lid.csv` |
| [`tables/acceptance.md`](tables/acceptance.md) | `results/summary/acceptance_checks.csv` |
| [`tables/ef_construction.md`](tables/ef_construction.md) | `results/summary/acceptance_ef_construction.csv` |
| [`tables/scaling_fits.md`](tables/scaling_fits.md) | `results/summary/scaling_fits.csv` |

Platform: `1d3d9b66e7d3e1ac`. Builds recorded: 426. Search-sweep rows: 3,442. Generated 2026-11-05T01:52:17+00:00.
