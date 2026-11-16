# Literature Review: ANN Index Selection across Dimensionality, Scale, and Filter Selectivity

*Phase 0 deliverable — October 2026*

## 1. Overview

This review surveys the landscape of approximate nearest neighbor (ANN) benchmarking and filtered vector search, with a specific focus on whether any prior work jointly varies embedding dimensionality and filter selectivity on a single controlled corpus. We reviewed 13 papers spanning ANN benchmarks, index algorithms, filtered search systems, and embedding representations.

**Central finding:** No existing paper measures how the filter-selectivity crossover between filtered ANN search and exact scan of the filtered subset changes with dimension on a single corpus. This is the gap our study fills.

---

## 2. ANN Benchmarks

### ANN-Benchmarks (Aumüller, Bernhardsson & Faithfull, 2020)

The de facto standard for comparing ANN algorithms. Evaluates 10+ libraries across datasets including SIFT-128, GloVe-200, GIST-960, and Fashion-MNIST-784. The benchmark plots recall versus queries-per-second, letting the reader eyeball trade-offs.

**Limitation for our purposes:** Dimensionality is confounded with the dataset. SIFT has D=128 and specific clustering structure; GIST has D=960 and entirely different geometry. Any apparent "effect of D" conflates intrinsic dimensionality, data distribution, and vector count. The benchmark does not support filtering, does not frame recall as a constraint, and does not compute cost surfaces.

### VIBE (Jääsaari et al., 2025)

A next-generation benchmark using 19 modern embedding datasets representative of RAG and multimodal retrieval workloads, evaluated across 22 vector-index implementations. VIBE addresses ANN-Benchmarks' aging dataset problem by using contemporary transformer embeddings.

**Limitation:** Like ANN-Benchmarks, VIBE uses different datasets at different dimensions rather than controlling D on a single corpus. It does not study filtered search or frame recall as a constraint.

### NeurIPS'21 Big-ANN Challenge (Simhadri et al., 2022)

A competition-format benchmark at billion scale. The 2023 edition added a filtered track, but the original 2021 challenge does not study filtering. Datasets span multiple dimensions but are not systematically varied on one corpus. Establishes important performance baselines but does not isolate dimension as a variable.

---

## 3. Index Algorithms

### HNSW (Malkov & Yashunin, 2018)

The foundational paper for hierarchical navigable small world graphs, now the dominant graph-based ANN index. Defines the M, efConstruction, and efSearch parameters we use. Evaluated on SIFT, GIST, GloVe, and random data.

**Relevance:** HNSW stores raw vectors in the graph, so its memory scales linearly with D. Its search cost depends on distance computations along the traversal path, making it sensitive to D through a cache/bandwidth channel that our study can measure. Does not study filtering.

### Product Quantization (Jégou, Douze & Schmid, 2011)

The basis of IVF-PQ compression. Splits a D-dimensional vector into m sub-vectors and quantizes each independently to one of 256 centroids (8 bits). The recall ceiling depends on the ratio D/m — higher D with fixed m means each sub-quantizer covers more dimensions and approximation error grows.

**Relevance:** Directly relevant to hypothesis H1b: at a fixed compression ratio, the recall ceiling of IVF-PQ without re-ranking should fall as D grows. The paper does not study this in a controlled manner (it uses SIFT-128 and GIST-960, confounding D with dataset), but the theory predicts it.

### Down with the Hierarchy (Munyampirwa, Lakshman & Coleman, 2024)

Shows that in high-dimensional embedding spaces, the hierarchical layers of HNSW become redundant — a flat NSW graph matches HNSW's performance because "hub" nodes naturally form long-range shortcuts. This finding is relevant to understanding why the cost structure of HNSW may change with D: at high D, the multi-layer navigation contributes less, potentially changing the balance between HNSW and IVF-based indexes.

---

## 4. Filtered ANN Search

### Filtered ANN in Vector Databases (Amanbayev et al., 2026)

The most directly relevant paper for our filtering study. Evaluates FAISS, Milvus, and pgvector on 768-dimensional text embeddings with varying filter selectivity. Key findings:

- **IVFFlat outperforms HNSW at low selectivity** — partition-based indexes handle selective filters better because they can skip entire clusters, while graph traversal struggles when few nodes satisfy the predicate.
- **Proposes the Global-Local Selectivity (GLS) metric** to quantify correlation between filter predicates and query vectors.

**Critical gap:** All experiments use fixed D=768. The paper does not investigate whether the IVFFlat advantage at low selectivity holds, strengthens, or weakens as D changes. Our study tests exactly this with D in {384, 1536}.

### Filtered ANN on Transformer Embeddings (Iff et al., 2026)

The largest filtered-ANN benchmark to date, introducing the arxiv-for-fanns dataset (2.7M arXiv abstracts with 11 real-world attributes) and evaluating 11 filtered ANN methods. Provides a broad taxonomy of filtering strategies on real text embeddings.

**Limitation:** Does not vary D in a controlled way. The dataset has a single fixed embedding dimension. While the breadth of methods evaluated is valuable, dimension effects are not studied.

### ACORN (Patel et al., 2024)

Introduces predicate-agnostic filtered HNSW that maintains graph connectivity under selective predicates by expanding the base HNSW graph. Reports 2–1000x throughput improvements over naive filtered search. An important engineering contribution, but does not study dimension variation. We exclude ACORN from our main comparison (it requires a modified index structure) but note it as an optional baseline.

### Filtered ANN as a Phase Transition (Mandarapu & Kunkunuru, 2026)

**The most relevant paper for our headline question.** Models the selectivity crossover between filtering strategies as phase transitions with closed-form boundary formulas:

- Post-filter cliff at s ~ k/K (where K is the over-fetch factor)
- In-filter cliff at s_c ~ 0.83/M (for HNSW with degree M)

These formulas predict the selectivity at which each strategy breaks down. However, **dimension D does not appear in any of the crossover formulas**, and all experiments use SIFT1M at 128 dimensions.

Our study directly tests whether this D-independence holds. If the crossover selectivity s* shifts with D (our alternative hypothesis H2), it would implicate cache and memory-bandwidth effects or intrinsic-dimension-driven search effort — factors that the Phase Transition paper's analytical model does not capture.

### VADER: Filtered Search with Declarative Recall (Chatzakis et al., 2026)

Introduces "declarative recall" — the user specifies a recall target and the system selects the filtering strategy and search parameters to meet it at minimum cost. Builds a filter-aware recall predictor that generalizes across selectivities. Accepted at VLDB 2027.

**Relevance:** VADER shares our framing of recall as a constraint (not an outcome), making it the closest methodological parallel. However, it does not study dimension as a variable, and its recall predictor is trained per-dataset rather than fitted to understand how D changes the cost surface.

### Other Filtered Search Work

- **Filtered-DiskANN (Gollapudi et al., 2023):** Extends DiskANN graph with label-based filtering. Seminal filtered-graph paper (228 citations). Does not study D variation.
- **UNIFY (Liang et al., 2024):** Single proximity graph supporting pre-, post-, and hybrid filtering seamlessly. Relevant engineering but no D study.
- **PathFinder (Wu & Tang, 2025):** Cost-based optimizer for selecting indexes based on filter complexity. Reports up to 9.8x throughput at recall 0.95. Does not vary D.
- **iFVS (Al-Mamun et al., 2026):** Query-specific PQ codebooks conditioned on both query and filter predicate. Novel approach to PQ under filtering. Does not study D effects.

---

## 5. Embedding Representations

### Matryoshka Representation Learning (Kusupati et al., 2022)

The key enabler for our experimental methodology. Trains embeddings such that truncated prefixes (the first d of D dimensions) remain high-quality representations. Demonstrates usable embeddings from D=8 to D=2048 on the same data, with graceful quality degradation.

**Relevance:** This is what makes controlled D variation possible on a single corpus without the confound of different datasets. We embed MS MARCO passages once at D=1536 and derive D=768, 384, 128 by prefix truncation and L2 renormalization. Matryoshka does not study ANN indexes, filtering, or cost trade-offs — that is our contribution.

### Closed Forms and Synthetic Twins (Herman, 2026)

Predicts ANN recall within 0.03 accuracy from label-free embedding statistics (without building an index). An intriguing methodology that could theoretically be extended to predict how recall changes with D, but the paper itself does not study filtering or dimension-dependent index selection.

---

## 6. Summary of Gaps

| Gap | Evidence | Our contribution |
|-----|----------|-----------------|
| **D confounded with dataset** | ANN-Benchmarks uses SIFT-128 vs GIST-960; VIBE uses 19 different datasets | Matryoshka truncation on one MS MARCO corpus: D changes while content stays fixed |
| **Recall is an outcome, not a constraint** | Only VADER (2026) frames recall as a declared target; all other benchmarks plot recall-vs-QPS curves | Cost surfaces with recall as input: for each (N, D, S, R), the cheapest index configuration |
| **Filtering studied apart from D** | Amanbayev (2026), Iff (2026), Phase Transition (2026) — all at fixed D | Controlled D × S interaction: measure whether s* shifts with D |
| **No tested decision rule** | No prior work fits an interpretable rule to predict index choice across workload cells | Depth-3 decision tree trained on N ≤ 500K, tested on N = 1M |

## 7. Gate 0 Decision

**Headline retained: D × S interaction.** The literature search confirms that the interaction between dimensionality and filter selectivity on a single controlled corpus has not been studied. The Phase Transition paper's D-independent crossover formulas provide a clean null hypothesis (H2) to test.

Secondary contributions — recall-constrained cost surfaces and the validated decision rule — are also novel but less distinctive than the D × S headline.
