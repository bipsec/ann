"""Tests for the 8-bit scalar-quantized HNSW index (sub-raw memory regime probe)."""

import numpy as np

from annstudy.indexes import (
    build_hnsw_sq,
    get_build_grid,
    get_search_grid,
    set_search_params,
)


def test_build_hnsw_sq_builds_trains_searches():
    rng = np.random.default_rng(0)
    D = 32
    xb = rng.random((3000, D)).astype("float32")
    xq = rng.random((5, D)).astype("float32")

    idx, meta = build_hnsw_sq(xb, D, M=32)
    assert meta["index_type"] == "hnsw_sq"
    assert meta["nbits"] == 8
    assert meta["train_size"] == 3000          # SQ is trained on the added vectors
    assert meta["train_seconds"] >= 0 and meta["build_seconds"] >= meta["add_seconds"]

    set_search_params(idx, "hnsw_sq", {"ef_search": 128})
    _, ids = idx.search(xq, 10)
    assert ids.shape == (5, 10)
    # recall@10 against exact flat should be high on easy random data at ef=128
    import faiss
    flat = faiss.IndexFlatL2(D); flat.add(xb)
    _, gt = flat.search(xq, 10)
    overlap = np.mean([len(set(ids[q]) & set(gt[q])) / 10 for q in range(5)])
    assert overlap >= 0.5


def test_hnsw_sq_grids_match_hnsw():
    # Same build grid (M in 16/32/64) and efSearch sweep as plain HNSW.
    assert get_build_grid("hnsw_sq", 500_000, 384) == get_build_grid("hnsw", 500_000, 384)
    g = get_search_grid("hnsw_sq", k=10)
    assert g and all("ef_search" in s for s in g)
