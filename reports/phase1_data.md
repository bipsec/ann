# Phase 1 report: data, embeddings, and ground truth

## Provenance
- Date: 2026-10-17. Host `hardware_id 1d3d9b66e7d3e1ac` (AMD EPYC-Milan, 16 cores, 58.8 GB, Linux;
  faiss-cpu 1.15.1 AVX2) — `results/hardware.json`.
- Sources: `data/emb/*.f32`, `data/gt/*.npz`, `results/summary/lid.csv`, `data/query_split.json`.

## Corpus and queries
- 1,000,000 MS MARCO passages sampled uniformly, seed 0. The 100K and 500K sets are exact **nested
  prefixes** of the 1M sample (asserted on ids), so N scales without changing content.
- 10,000 MS MARCO training queries, seed 1; split **2,000 tuning / 2000–9999 test** (positional in
  the embedding files). Parameters are tuned on tuning queries only; all headline numbers use test
  queries (§0.8).

## Embeddings
- Model: OpenAI `text-embedding-3-large`, 1536-d (hosted-API route; chosen for no-GPU
  reproducibility via cached vectors — `logs/decisions.md`).
- Truncations: D = 768, 384, 128 by prefix slice + L2 renorm; the 1536 file re-normalized in place.
- Validation (acceptance): shapes (1,000,000, 1536) and (10,000, 1536); no NaN/inf; all row norms
  within 1e-4 of 1.

## Local intrinsic dimensionality (`results/summary/lid.csv`)
Levina–Bickel MLE, k=20, 10,000-passage sample:

| D | LID mean | LID sd |
|---|---|---|
| 128 | 31.54 | 11.43 |
| 384 | 32.39 | 13.10 |
| 768 | 33.12 | 13.60 |
| 1536 | 34.20 | 14.33 |

LID rises only gently and stays far below the ambient dimension — most of the D range adds
representational dimensions with limited new intrinsic structure. This bounds (does not remove) the
truncation-geometry confound discussed in the paper's limitations.

## Ground truth
- Exact top-100 neighbors via FAISS flat L2 per (N, D); **all 12 cells present**
  (N ∈ {100K, 500K, 1M} × D ∈ {128, 384, 768, 1536}).
- Correctness: numpy brute-force agreement on 100 random queries for N ≤ 500K (ties within tolerance).
- Timing (`logs/budget.md`): N=500K/D=1536 36 s compute; N=1M all D 275 s compute.

## Nearest-neighbor sanity (`logs/decisions.md`)
Top-5 neighbors inspected for 3 test queries at D=1536 vs D=128: at D=1536 all neighbors are
topically correct; at D=128 easy topical queries survive but a harder relational query degrades to
off-topic results — the recall-vs-dimension effect the study measures.

## For the paper
Feeds §3 (Method) — corpus, truncation, metric/ground truth, LID table — and the reproducibility
statement. Cite `results/summary/lid.csv` for the LID table and `results/hardware.json` for the
environment.
