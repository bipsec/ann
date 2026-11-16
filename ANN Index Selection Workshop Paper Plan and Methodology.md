# ANN Index Selection Workshop Paper: Plan and Methodology

Oct 14, 2026 · @Biplab

## Summary

The paper asks one practical question: given a required recall, which approximate nearest neighbor (ANN) index is cheapest, and how does the answer change with dimensionality, dataset size, and metadata-filter selectivity? Working title: *Which ANN Index Is Cheapest at a Given Recall? A Cost-Surface Study over Dimensionality, Scale, and Filter Selectivity.*

**Claimed contributions**

1. **Recall-constrained cost surfaces.** For each workload cell (N, D, S, R), the cheapest configuration of each index that meets the recall target, compared on latency, memory, and build time. Recall is an input, not a plotted outcome.
2. **Controlled dimensionality.** One corpus is embedded once with a Matryoshka-trained model and truncated to 128, 384, 768, and 1536 dimensions, so D changes while the content does not.
3. **Quantified crossover points.** Where the winning index or filter strategy changes, with bootstrap confidence intervals. The sharpest question is how the filter-selectivity crossover moves with D.
4. **A validated decision rule.** A small interpretable rule fitted on part of the grid and tested on held-out cells, plus a released harness.

**Honest scope.** No new algorithm. FAISS-based, one machine, text embeddings, up to 1M vectors. Each individual axis (size, dimension, filtering) has been studied before. The contribution is the controlled interaction analysis and the tested decision rule. Realistic target: a workshop paper, resource track, or arXiv technical report. Whether the D × S interaction is already published is unverified and is the first thing Phase 0 must settle.

## Motivation and gap

Practitioners choose an index under a service-level target ("recall@10 of at least 0.95") and a memory budget, yet most benchmarks report the opposite view: a recall-versus-throughput curve per dataset. Reading the cheapest point at a required recall off those curves is possible only when the dataset, dimensionality, and size match the reader's own, which they rarely do.

**Three specific gaps this paper targets**

1. **Dimension is confounded with the dataset.** Classic comparisons use different datasets at different dimensions (SIFT at 128, GIST at 960). Differences in intrinsic dimensionality, clustering, and difficulty are mixed into the "effect of D". Truncating one Matryoshka-embedded corpus removes that confound.
2. **Recall is treated as an outcome, not a constraint.** The decision a practitioner faces is "cheapest index that meets R within memory M". That is a constrained optimization, and the answer can flip as R rises from 0.90 to 0.99.
3. **Filtering is studied apart from dimension and scale.** Recent work reports that partition-based indexes can beat graph-based ones at low selectivity, and that filtered-search methods behave differently on different embedding types. Whether those conclusions hold as D grows is not something I could confirm from the material I found.

**A first-order argument worth testing.** Exact search over the filtered subset costs roughly s × N × D operations. A filtered graph traversal must visit roughly 1/s more nodes to find matches, each costing D operations. Both scale linearly in D, so the crossover selectivity s\* should be roughly independent of D if compute dominates. A measured shift in s\* with D would point to memory-hierarchy effects or to the recall-driven search effort changing with intrinsic dimensionality. That makes the null hypothesis clean and the result informative either way. This is my own reasoning, not a published result, and the experiment exists to test it.

## Related work to read first

The papers below are the ones to position against. I saw the first five only as search results and reference lists, not as full texts, so every claim about them is provisional until you read them. The last four are cited from memory and also need checking.

| Work | What it appears to cover | What to verify in Phase 0 |
| --- | --- | --- |
| [ANN-Benchmarks](https://arxiv.org/abs/1807.05614) (Aumüller et al., 2020) | Standard recall-versus-QPS comparison across many algorithms and datasets | Which datasets vary D, and whether any control for data distribution |
| [VIBE](https://arxiv.org/abs/2505.17810) (Jääsaari et al., 2025) | Index benchmark built around modern embedding datasets | Whether it varies D on a single corpus, and whether it frames recall as a constraint |
| [Filtered ANN in vector databases](https://arxiv.org/html/2602.11443) (Feb 2026) | System-level filtered search; reports partition-based indexes beating graph-based ones at low selectivity; proposes a filter–query correlation metric | Whether results are repeated at several D; which engines and versions; how correlation is defined |
| [Filtered ANN on transformer embeddings](https://arxiv.org/html/2507.21989v3) (v3, Apr 2026) | Taxonomy and benchmark of filtered methods on transformer embeddings; large-scale arXiv-based dataset | Whether D is varied in a controlled way, and which index families are covered |
| [The Faiss Library](https://arxiv.org/abs/2401.08281) (Douze et al., 2024) | Reference for the index implementations you will use | Exact index factory strings and filter support in your FAISS version |
| HNSW (Malkov and Yashunin) | Graph index definition and parameters M, efConstruction, efSearch | Cite the journal version |
| Product quantization (Jégou et al., 2011) | Basis of IVF-PQ and its recall ceiling | Cite for the compression-versus-recall trade-off |
| Matryoshka Representation Learning (Kusupati et al., 2022) | Why truncated prefixes remain usable embeddings | Confirm your chosen model was trained this way |
| ACORN (Patel et al., 2024) | Filtered HNSW that stays connected under selective predicates | Decide whether to include it as an optional comparison |

**Search method for Phase 0.** Search the terms "filtered vector search dimensionality", "Matryoshka truncation ANN index", "recall-constrained index selection", and "selectivity crossover pre-filter HNSW IVF". For each hit, record whether it varies D under a controlled corpus. If one does, the headline shifts to the recall-constrained framing plus the tested decision rule.

## Research questions and hypotheses

Each question is paired with a hypothesis that the data can contradict. Stating what would falsify it up front keeps the analysis honest and gives reviewers something concrete to check.

**RQ1. Dimension and index.** At a fixed recall target, how does the cost of HNSW, IVF-Flat, and IVF-PQ change as D grows from 128 to 1536?

- *H1a.* HNSW and IVF-Flat memory grows linearly in D because raw vectors dominate. IVF-PQ memory depends on the code size, so the memory ratio between HNSW and IVF-PQ widens with D. This is close to arithmetic, so the interesting part is H1b.
- *H1b.* At a fixed compression ratio, the recall ceiling of IVF-PQ without re-ranking falls as D grows, so it cannot reach R = 0.99 at high D unless the code size or a re-ranking step is added. *Falsified if* IVF-PQ reaches 0.99 at D = 1536 without re-ranking at 16x compression.
- *H1c.* The latency penalty of high D differs by index: HNSW pays mainly in distance computations along the traversal path, IVF-Flat in scanning larger inverted lists. The fitted D exponent differs between them by more than the bootstrap confidence interval.

**RQ2. Scale, dimension, and filter selectivity.** As selectivity S falls from 100% to 0.1%, where does each filtering strategy collapse, and does that crossover move with D?

- *H2 (null).* The crossover selectivity s\* between filtered ANN search and exact search on the filtered subset is independent of D, because both costs scale linearly in D.
- *H2 (alternative).* s\* shifts with D, which would implicate cache and memory-bandwidth effects or intrinsic-dimension-driven search effort. *Decision:* reject the null if the s\* confidence intervals at D = 384 and D = 1536 do not overlap.
- *H2b.* The partition-based advantage at low selectivity reported in prior work holds for uncorrelated filters but weakens or reverses for query-correlated filters.

**RQ3. Decision rule.** Can a small interpretable rule over (N, D, S, R, memory budget, update ratio) predict the cheapest index on cells it was not fitted on?

- *H3.* A depth-3 decision tree trained on N ≤ 500K cells picks the oracle-best index on the 1M cells with higher accuracy and lower cost regret than the baseline "always use HNSW". *Falsified if* it does not beat that baseline, which is itself a useful negative result.

**Out of scope on purpose.** Distributed or multi-node search, GPU indexes, new index algorithms, and deletion-heavy workloads.

## Methodology

The design goal is that every difference in results traces to one named factor. Anything that cannot be controlled is logged and reported as a limitation.

**Corpus and queries.** Sample 1,000,000 passages at random (fixed seed) from MS MARCO passages. Smaller sets are nested prefixes of that sample, so the 100K set is inside the 500K set, which is inside the 1M set. Sample 10,000 real queries from the MS MARCO training queries. Split them into 2,000 tuning queries and 8,000 test queries. Parameters are chosen on the tuning set only and all reported numbers come from the test set, which avoids tuning on the evaluation data.

**Embeddings and dimensionality.** Embed passages and queries once with a Matryoshka-trained model, keeping the first 1536 dimensions. Derive 768, 384, and 128 by slicing the prefix and L2-normalizing again. Two candidate models, to be checked against their model cards before committing:

- *Open weights* (for example Qwen3-Embedding-4B, which I believe supports user-chosen output sizes up to 2560): best for release and reproducibility, but needs a GPU for about 1M passages.
- *Hosted API* (for example OpenAI text-embedding-3-large, which supports a dimensions parameter): no GPU needed, costs money per token, and the vectors cannot be regenerated if the model is retired.

Record the local intrinsic dimensionality (LID) of each truncation, estimated on a 10K sample, so results can be explained by geometry as well as by D.

**Metric and ground truth.** Vectors are unit length, so cosine, inner product, and L2 rank identically. Use L2 in every index to avoid known quirks of inner-product graph search. Ground truth is exact top-100 neighbors from a FAISS flat index on the same truncated vectors, computed once per (N, D) and stored. This measures ANN fidelity, not retrieval quality, so truncation hurting embedding quality does not matter.

**Indexes (all FAISS; hnswlib as an HNSW cross-check)**

| Index | Build parameters | Search parameters | Notes |
| --- | --- | --- | --- |
| Flat (exact) | none | none | Ground truth and the brute-force baseline |
| HNSW | M in {16, 32, 64}; efConstruction 200 (pilot checks 100 and 400) | efSearch swept from 16 to 1024 | Raw vectors stored in the index |
| IVF-Flat | nlist in {2, 4, 8} × sqrt(N), rounded | nprobe swept from 1 to 512 | Raw vectors in inverted lists |
| IVF-PQ | same nlist; m in {D/16, D/8, D/4}; 8 bits per code | nprobe swept; refine factor in {none, 4, 16} | Report with and without re-ranking |

IVF-PQ with re-ranking needs the full-precision vectors available, in RAM or on disk. Its memory advantage shrinks or vanishes in that case, so report the refined and unrefined variants as separate rows, never merged.

**What "cost" means.** There is no single scalar cost. For each cell, define the constrained problem: among configurations with recall@10 at least R and resident memory at most M, minimize p95 single-thread latency. Report the other metrics (build time, QPS, bytes per vector) alongside, and run the winner analysis separately for each objective so a conclusion does not depend on one weighting.

**Metrics**

| Metric | Definition |
| --- | --- |
| Recall@k | Share of the exact top-k returned; ties in distance resolved by exact distance, not id |
| Latency | Per-query wall time at batch size 1, single thread, after warm-up; mean, p50, p95, p99 over 8,000 queries |
| QPS | Throughput at a fixed thread count (physical cores, capped at 16), large batches |
| Build time | Wall time including training (k-means, PQ codebooks) and adding vectors, with training sample size recorded |
| Index size | Bytes of the serialized index on disk |
| RAM | Resident set size after loading the index and a warm-up pass, minus the baseline process size |
| Insert cost | Vectors per second and recall drift, from the update study |

**Filters.** Attach one integer attribute per vector, drawn so selectivity is controlled. Uncorrelated filters assign values at random, independent of the vector. Query-correlated filters derive the attribute from a k-means cluster id, so a filter can be positively correlated (the query's own neighborhood satisfies it) or negatively correlated (a distant region does). Log the measured selectivity of each query, since cluster sizes vary. Filtered ground truth is exact search on the matching subset.

Three filtering strategies are compared and described precisely in the paper:

1. **Pre-filter then exact scan** of the matching vectors.
2. **In-index filtered search**, where the predicate is applied during graph traversal or list scanning through the library's ID-selector support. Confirm your FAISS version supports it for HNSW; if not, use hnswlib's filter function and say so.
3. **Post-filter with over-fetch**, retrieving k divided by s times a safety factor and discarding non-matches.

FAISS's in-index filter on HNSW is a plain predicate during traversal. It is not the more elaborate connectivity-preserving designs in the literature, so state that, and treat an ACORN comparison as optional.

## Experimental design

A full factorial over every factor would need thousands of index builds and cannot finish on one machine. Instead the study runs four stages, each answering one question, with overlapping cells so interactions can still be estimated.

| Stage | Question | Factors varied | Indexes | Cells |
| --- | --- | --- | --- | --- |
| A: dimension | RQ1: how does D change cost at fixed recall? | D in {128, 384, 768, 1536}; R in {0.90, 0.95, 0.99} | Flat, HNSW, IVF-Flat, IVF-PQ | 4 (N = 500K) |
| B: scale | RQ2: how does N interact with D? | N in {100K, 500K, 1M}; D in {384, 1536} | HNSW, IVF-Flat, IVF-PQ, plus Flat as reference | 6 (the 500K cells reuse Stage A) |
| C: filtering | RQ2: where does each filter strategy collapse, and does it move with D? | S in {100%, 50%, 10%, 1%, 0.1%}; D in {384, 1536}; correlation in {none, positive, negative} | HNSW, IVF-Flat, pre-filter exact scan | 4 (N in {500K, 1M}; reuses Stage B indexes, since the predicate is applied at search time) |
| D: updates | Cost of growing an index | Insert 10% new vectors in 10 batches | HNSW, IVF-Flat, IVF-PQ | 1 (N = 500K, D = 768) |

Side checks, not headline results: k in {1, 100} at one cell in Stage A, and hnswlib against FAISS HNSW at one cell to confirm the implementation does not drive conclusions.

**Parameter search protocol.** Build parameters (M, nlist, m) need a rebuild and are expensive. Search parameters (efSearch, nprobe, refine factor) are free to sweep on one built index. So for each cell and index, build only the few build-parameter variants in the table above, sweep the search parameter on the 2,000 tuning queries to find the cheapest setting that meets each R, then re-measure that setting on the 8,000 test queries. If no setting reaches R, record "not reachable" as a result, since it is evidence for the recall-ceiling hypothesis.

**Update study details.** Build on 90% of the vectors, then insert the remaining 10% in 10 equal batches. After each batch, record insert throughput and recall at the search parameters tuned before the inserts. Compare with an index rebuilt from scratch on the same vectors. For IVF, do not retrain centroids during the study, and add one final retrain point, so centroid staleness is visible. Deletions are excluded: FAISS HNSW does not support removal, so a fair comparison across indexes is not possible without extra machinery.

**Compute and memory budget.** Raw float32 vectors at 1M × 1536 take about 6.1 GB, and HNSW adds graph links on top. A machine with at least 64 GB RAM and 16 physical cores is a comfortable target. Brute-force ground truth for 10,000 queries over 1M vectors is a few minutes of matrix multiplication on a multi-core CPU. Index builds are the real cost, and HNSW at 1M × 1536 is the most expensive. Do not estimate it. Time one HNSW and one IVF-PQ build at that size in the pilot, then scale the plan to the measured numbers.

**Rough run count.** About 9 index builds per cell (3 HNSW, 3 IVF-Flat, 3 IVF-PQ), so roughly 36 builds for Stage A and about 72 across Stages A and B (4 new cells in Stage B), before repeating builds for timing variance.

## Statistical rigor and threats to validity

Benchmark papers are most often rejected because the numbers cannot be trusted or reproduced. These rules are meant to prevent that.

**Measurement rules**

1. **Enough queries for tail latency.** A p99 from 1,000 queries rests on ten samples. Use 8,000 test queries per configuration so p99 rests on about 80.
2. **Repeat builds.** Build every build-parameter variant once, then build the selected configuration of each index three times in total with different seeds. Report build time as a mean with range, and report how much recall varies between builds (HNSW insertion order and k-means seeds both matter).
3. **Resample queries for uncertainty.** Bootstrap over queries (1,000 resamples) for confidence intervals on recall and latency percentiles. Bootstrap over builds for scaling-exponent fits.
4. **Pin the environment.** Fix thread counts, pin processes to cores, record CPU model, RAM, OS, library versions, and the git commit. State whether caches were warm. Latency runs are single-threaded and run alone on the machine.
5. **Equal tuning effort.** Every index gets the same number of build-parameter variants and the same search sweep, so no index wins through more careful tuning.
6. **Separate tuning and test queries**, as described in the methodology.
7. **Sanity-check the harness first.** Before any real run, compare your HNSW and IVF recall-versus-QPS curve against published curves for a similar dataset. A mismatch means a harness bug, not a finding.

**Threats to validity and what to do about each**

| Threat | Why it matters | Mitigation |
| --- | --- | --- |
| One embedding family and one text corpus | Geometry and clustering may differ for images or other models | State plainly; run one cell with a second model as a robustness check if time allows |
| Truncation changes the geometry, not only D | Lower D may also change intrinsic dimensionality, so "D effect" and "geometry effect" blur | Report LID per truncation and discuss; this confound is smaller than across-dataset comparisons but not zero |
| FAISS-specific implementations | Conclusions may reflect library choices, such as its filter handling | Cross-check HNSW with hnswlib; describe filter implementations exactly |
| Synthetic filter attributes | Real metadata has skew and correlation structure | Include query-correlated filters; say synthetic attributes bound, not replace, real behavior |
| Single machine | Cache sizes and memory bandwidth affect the D-dependence results | Report hardware; frame hardware-dependent claims carefully, especially RQ2 |
| Small decision-rule training set | A tree fitted on a few dozen cells can overfit | Keep depth at most 3; validate on cells from a held-out scale; report uncertainty, not only accuracy |
| Ground-truth ties | Near-identical distances can flip neighbor identity | Score recall by distance threshold, not id match |

## Analysis plan

The analysis turns raw sweeps into four outputs: cost surfaces, scaling fits, crossover points, and the decision rule. The earlier "Performance = f(N, D, S, U, R, M)" framing is kept as an organizing idea. With a sparse grid it is not fitted as a six-variable regression.

**1. Cost surfaces and winner maps.** For each (N, D) cell and each R, record which index minimizes p95 latency under the memory budget (unbounded, or 1.5x, 0.5x, or 0.1x of raw vector size, where raw is N × D × 4 bytes), then repeat for build time and for bytes per vector. Show these as winner maps, one panel per objective, so a reader sees at once where the winner changes and whether the choice depends on the objective.

**2. Scaling fits with an explicit interaction term.** For each index, fit a log-log model of each cost on N and D, and test whether the interaction matters.

```latex
\log y = a + b \log N + c \log D + d \, (\log N)(\log D) + \varepsilon
```

Here y is p95 latency, build time, or memory at a fixed recall. The coefficient d is the headline quantity: a nonzero d means scale and dimension do not act independently. Report b, c, and d with bootstrap intervals, and note that with three sizes and four dimensions the fit is descriptive, not predictive.

**3. Filter crossover points.** For each index and D, plot latency at the required recall against selectivity, together with the pre-filter exact-scan curve. Estimate s\* where the curves cross by interpolation, with a bootstrap interval, and compare s\* across D to test H2. Also plot recall at fixed search effort against selectivity, to show where filtered search silently loses recall, which is a failure users often do not notice.

**4. Decision rule and held-out validation.**

1. Label every measured cell and objective with its oracle-best index.
2. Features: N, D, S, R, memory budget, and update ratio where measured.
3. Fit a decision tree of depth at most 3 on cells with N ≤ 500K.
4. Test on the N = 1M cells, which is an extrapolation test of the rule and the scalability claim at once.
5. Compare against two baselines: always use HNSW, and a rule of thumb (HNSW unless memory is tight, then IVF-PQ).
6. Report top-1 agreement with the oracle and **regret**, the extra cost of the rule's choice relative to the oracle's.

Because the grid is small, the resulting tree may be little more than a lookup table. That is acceptable if reported honestly. What the paper offers is the rule together with its measured error, which a plain table of recommendations lacks.

**5. Candidate conclusions.** The table below lists hypotheses to test, not results. Fill the right column only from data.

| Workload condition | Candidate winner to test | Evidence that would confirm or refute |
| --- | --- | --- |
| Low D, high recall, memory not binding | HNSW | Lowest p95 latency at R = 0.99 in Stage A and B |
| High D, tight memory | IVF-PQ, with re-ranking if R is high | Memory per vector versus recall ceiling |
| Very selective filter | Pre-filter exact scan or IVF-Flat | s\* location and recall loss of in-index filtering |
| Frequent inserts | HNSW or IVF-Flat, depending on recall drift | Stage D drift and insert throughput |
| Recall at least 0.99 | Flat or HNSW | Whether IVF-PQ is reachable at all at that target |

## Paper outline

A workshop paper is usually 4 to 8 pages, so check the page limit of your target venue first and trim to fit. The budget below totals about 6.75 pages plus references. Lead with the winner map and decision rule, and treat the sweeps as supporting evidence.

| Section | Pages | Content | Key exhibit |
| --- | --- | --- | --- |
| 1. Introduction | 0.75 | Practitioner problem; why recall as a constraint; the three gaps; contribution list | Small teaser figure of the winner map |
| 2. Background and related work | 0.75 | Index families; benchmarks; filtered search; Matryoshka embeddings | None |
| 3. Method | 1.25 | Corpus, truncation, indexes, filters, metrics, tuning protocol, cost definition | Table of indexes and parameters; pipeline figure |
| 4. RQ1 results: dimension | 1.0 | Cost at fixed recall versus D; PQ recall ceiling; LID per truncation | Cost-versus-D lines per index at R = 0.95 and 0.99 |
| 5. RQ2 results: scale and filtering | 1.25 | Scaling fits with interaction term; crossover s\* per D; correlated filters | Crossover plot per D; table of fitted exponents |
| 6. RQ3 results: decision rule | 0.75 | The tree, held-out accuracy, regret versus baselines | Tree diagram; regret table |
| 7. Update study | 0.25 | Insert throughput and recall drift | One small plot |
| 8. Discussion, limitations | 0.5 | Threats, what does not generalize, practical guidance | None |
| 9. Conclusion | 0.25 | Three sentences answering RQ1 to RQ3 | None |

**Writing guidance**

- State in the introduction exactly what is not new. Reviewers trust a paper that does.
- Every claim in the discussion should point to a figure or table. Anything not backed by a measured cell goes in a clearly labeled "hypotheses for future work" paragraph.
- Report negative results (for example "IVF-PQ never reached 0.99 at D = 1536") with the same prominence as positive ones.
- Include a reproducibility statement: code, configs, seeds, hardware, library versions, and how to regenerate embeddings.

## Timeline, risks, and release

**Timeline.** The plan fits twelve weeks if the pilot goes well. Each gate is a go or no-go check, so effort is not spent on later phases before an earlier assumption has been confirmed.

&#91;embedded content: twelve-week plan · 5 phases, 4 gates\]

Work runs top to bottom; the first gate decides whether the headline claim survives, and the second decides whether the harness can be trusted at scale.

**Risks and fallbacks**

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Prior work already answers the selectivity-by-dimension question | Phase 0 search finds a controlled-D filtered study | Lead with the recall-constrained framing, the Matryoshka control, and the held-out decision rule |
| Results mostly confirm known findings | Winner maps match published intuition everywhere | Emphasize quantified crossovers, the interaction coefficient d, and regret numbers; confirmation with intervals is still citable |
| Compute overruns at 1M × 1536 | Pilot build time times planned builds exceeds the budget | Drop the M = 64 and extra nlist variants, run D = 1536 at 500K only, or rent a larger machine for builds |
| Filter support differs by library version | In-index filtering on HNSW is missing or behaves oddly | Use hnswlib's filter function, document it exactly, and limit claims to that implementation |
| Embedding model too costly or unavailable | No GPU time or API budget | Use a smaller Matryoshka model capped at 768 and drop the 1536 cell, or run the API once and cache the vectors |
| Decision rule is trivial or overfits | Held-out regret is no better than "always HNSW" | Report it as a negative result and make the winner maps the main contribution |

**Target venues.** I cannot verify current deadlines from here, so check the live calls before planning around any date. Candidates to look for: workshops co-located with data-management conferences (for example VLDB or SIGMOD workshops on vector or data systems), information-retrieval reproducibility and resource tracks (for example SIGIR or CIKM), and an arXiv technical report as the baseline outcome. A main-track datasets-and-benchmarks submission would need a larger scope than this plan.

**Release plan**

1. Public repository with a pinned environment and one config file per experiment.
2. Results as tabular files carrying hardware, library versions, seeds, and git commit on every row.
3. A script that regenerates the embeddings, and the vectors themselves only if the model and dataset licenses allow. MS MARCO is licensed for research use, so confirm the terms.
4. A one-command reproduction of the pilot cell, so a reviewer can check the harness in minutes.
5. An archived release with a persistent identifier, cited in the paper.
