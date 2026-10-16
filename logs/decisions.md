# Decisions Log

One dated line per non-obvious choice.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-14 | Project initialized | ANN index selection study bootstrapped |
| 2026-10-14 | Embedding route: hosted API (OpenAI text-embedding-3-large) | No local GPU; reproducible via cached vectors; dimensions parameter supports Matryoshka-style truncation |
| 2026-10-14 | Full runs on cloud instance (64+ GB RAM, 16+ cores) | Local machine has 7 GB RAM, insufficient for N=1M workloads |
| 2026-10-14 | FAISS 1.15.1 confirmed: SearchParametersHNSW, IDSelectorBatch, IDSelectorBitmap all available | In-index HNSW filtering can use native FAISS; no hnswlib fallback needed |
| 2026-10-14 | Python 3.14.4, faiss-cpu 1.15.1 wheels available | No conda fallback needed |
| 2026-10-14 | GATE 0: Keep D × S headline | Literature search (13 papers) confirms no prior work measures filter-selectivity crossover vs dimension on one corpus; Phase Transition paper (2026) models crossovers at fixed D only |
