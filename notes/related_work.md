# Related Work

## Key question

Does any existing paper measure how the filter-selectivity crossover between filtered ANN search and exact scan of the filtered subset changes with dimension, on a single corpus?

## Papers

| Paper | Year | Link | Varies D? | Controlled corpus? | Datasets | Indexes | Filter study? | What it does NOT cover | Verdict |
|-------|------|------|-----------|-------------------|----------|---------|---------------|----------------------|---------|
| ANN-Benchmarks (Aumüller et al.) | 2020 | https://arxiv.org/abs/1807.05614 | No — uses different datasets at different D | No | SIFT, GloVe, GIST, Fashion-MNIST, others | Many (10+ libraries) | No | Filtering, recall-as-constraint, controlled D | Standard recall-vs-QPS benchmark; D is confounded with dataset |
| VIBE (Jääsaari et al.) | 2025 | https://arxiv.org/abs/2505.17810 | No | No | 19 modern embedding datasets (RAG-focused) | 22 implementations | No | Filtering, controlled D on one corpus, recall-as-constraint | Modern embedding benchmark; broad coverage but does not isolate D |
| Filtered ANN in vector databases (Amanbayev et al.) | 2026 | https://arxiv.org/abs/2602.11443 | No — fixed at 768-d | No | 768-d text embeddings | IVFFlat, HNSW via FAISS/Milvus/pgvector | Yes — reports IVFFlat beats HNSW at low selectivity; proposes GLS correlation metric | D variation, recall-as-constraint, PQ indexes | Closest to our filter question, but D is fixed; partition-based advantage at low S is a key finding to compare against |
| Filtered ANN on transformer embeddings (Iff et al.) | 2026 | https://arxiv.org/abs/2507.21989 | No | No | arxiv-for-fanns (2.7M arXiv abstracts, 11 attributes) | 11 filtered ANN methods | Yes — large-scale filtered benchmark on real text embeddings | Controlled D variation, recall-as-constraint | Broad filtered-ANN taxonomy; does not isolate dimension effects |
| The Faiss Library (Douze et al.) | 2024 | https://arxiv.org/abs/2401.08281 | No | N/A | Various | Flat, IVF, HNSW, PQ, SQ, and more | Documents IDSelector and SearchParameters APIs | Benchmarking D variation | Reference for FAISS index implementations; confirms filter API availability |
| HNSW (Malkov & Yashunin) | 2018 | https://arxiv.org/abs/1603.09320 | No | No | SIFT, GIST, GloVe, random | HNSW only | No | Filtering, D-dependent cost analysis | Foundational graph-index paper; cite the journal version (2018) |
| Product Quantization (Jégou et al.) | 2011 | IEEE TPAMI 2011 | Implicitly (via D/m ratio) | No | SIFT, GIST | IVF-PQ | No | Filtering, recall-as-constraint | Foundational compression paper; PQ recall ceiling depends on D/m ratio — relevant to H1b |
| Matryoshka Representation Learning (Kusupati et al.) | 2022 | https://arxiv.org/abs/2205.13147 | Yes — core contribution is usable embeddings at multiple truncated D | Yes — same data at D from 8 to 2048 | ImageNet, retrieval tasks | Not ANN index focused | No | ANN index selection, filtering | Directly enables our D-truncation methodology; does not study index choice |
| ACORN (Patel et al.) | 2024 | https://arxiv.org/abs/2403.04871 | No | No | Multiple | HNSW with predicate subgraph traversal | Yes — predicate-agnostic filtered HNSW | Dimension variation, cost modeling | Key filtered-graph method (2-1000x gains); compare against as optional baseline |
| Filtered ANN as a Phase Transition (Mandarapu & Kunkunuru) | 2026 | https://arxiv.org/abs/2606.16341 | No — uses SIFT1M (128-d fixed) | No | SIFT1M, synthetic | Pre-filter, post-filter, in-filter strategies | Yes — models selectivity crossover as phase boundaries (s~k/K, s_c~0.83/M) | D as a parameter in crossover formulas | **Most relevant.** Models the exact crossover we study, but D does not appear; our study fills this gap |
| VADER: Filtered Search with Declarative Recall (Chatzakis et al.) | 2026 | https://arxiv.org/abs/2609.38053 | No | No | Multiple | Filter-aware recall predictor + cost model | Yes — "declarative recall" target, filter-aware | D variation | Relevant framing: recall as a constraint with filter awareness; does not study dimension |
| Down with the Hierarchy (Munyampirwa et al.) | 2024 | https://arxiv.org/abs/2412.01940 | Tests high-D datasets | Multiple datasets | High-D embeddings | HNSW, flat NSW | No | Filtering | Shows HNSW hierarchy is redundant in high D (hubs suffice); relevant to understanding D effects on graph topology |
| Closed Forms and Synthetic Twins (Herman) | 2026 | https://arxiv.org/abs/2609.00364 | No | No | Multiple | Predicts recall from embedding statistics | No | Filtering, D-dependent index selection | Predicts ANN recall within 0.03 from label-free stats; methodology could extend to D-dependent prediction |

## Search terms for Phase 0

1. "filtered vector search dimensionality" — no papers found coupling filter selectivity with D
2. "Matryoshka truncation ANN index" — no papers found studying Matryoshka truncation for ANN index selection
3. "recall-constrained index selection" — closest hit: Phase Transition paper and VADER
4. "selectivity crossover pre-filter HNSW IVF" — closest hit: Phase Transition paper

## Findings

**Answer to the key question: No.** After searching arxiv and reviewing 13 papers, no existing paper measures how the filter-selectivity crossover between filtered ANN search and exact scan changes with dimension on a single corpus.

The closest work is **"Filtered ANN as a Phase Transition" (Mandarapu & Kunkunuru, 2026)**, which models the selectivity crossover as phase boundaries with closed-form formulas. However, dimension D does not appear as a parameter in these formulas, and all experiments use fixed-D data (SIFT1M at 128-d). Our study would directly fill this gap by measuring whether s* shifts with D.

**Three gaps confirmed:**
1. **D confounded with dataset** — ANN-Benchmarks, VIBE, and most benchmarks use different datasets at different D. Only Matryoshka (Kusupati 2022) demonstrates controlled truncation, but does not study ANN indexes.
2. **Recall as outcome, not constraint** — Only VADER frames recall as a declared target. No paper computes cost surfaces with recall as an input.
3. **Filtering studied apart from D** — Multiple papers study filtered ANN (Amanbayev 2026, Iff 2026, ACORN, Phase Transition), but none vary D on one corpus.

**The D × S headline is novel.** No prior work does the controlled interaction analysis. The recall-constrained framing and the decision rule are additional contributions not found in the literature.
