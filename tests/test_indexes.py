"""Tests for index factory functions."""

import numpy as np
import pytest


def test_hnsw_high_recall(medium_vectors, medium_queries):
    """HNSW on 5000 vectors should reach recall >= 0.99 at high efSearch."""
    import faiss
    from annstudy.indexes import build_hnsw, set_search_params
    from annstudy.metrics import recall_at_k

    dim = medium_vectors.shape[1]
    k = 10

    index, meta = build_hnsw(medium_vectors, dim, M=32, ef_construction=200)
    assert meta["index_type"] == "hnsw"
    assert meta["build_seconds"] > 0

    set_search_params(index, "hnsw", {"ef_search": 256})

    gt_index = faiss.IndexFlatL2(dim)
    gt_index.add(np.ascontiguousarray(medium_vectors))
    gt_dists, gt_ids = gt_index.search(medium_queries, k)

    _, ret_ids = index.search(medium_queries, k)
    recalls = recall_at_k(ret_ids, gt_ids, gt_dists, medium_vectors, medium_queries, k=k)
    mean_recall = np.mean(recalls)

    assert mean_recall >= 0.99, f"HNSW recall at efSearch=256 should be >= 0.99, got {mean_recall:.4f}"


def test_ivf_flat_monotone_nprobe(medium_vectors, medium_queries):
    """Recall should be non-decreasing as nprobe increases."""
    import faiss
    from annstudy.indexes import build_ivf_flat, compute_nlist, set_search_params
    from annstudy.metrics import recall_at_k

    dim = medium_vectors.shape[1]
    k = 10
    nlist = compute_nlist(5000, 4)

    index, meta = build_ivf_flat(medium_vectors, dim, nlist=nlist)
    assert meta["index_type"] == "ivf_flat"

    gt_index = faiss.IndexFlatL2(dim)
    gt_index.add(np.ascontiguousarray(medium_vectors))
    gt_dists, gt_ids = gt_index.search(medium_queries, k)

    prev_recall = 0.0
    for nprobe in [1, 4, 16, 64, 256]:
        if nprobe > nlist:
            break
        set_search_params(index, "ivf_flat", {"nprobe": nprobe})
        _, ret_ids = index.search(medium_queries, k)
        recalls = recall_at_k(ret_ids, gt_ids, gt_dists, medium_vectors, medium_queries, k=k)
        mean_recall = np.mean(recalls)

        assert mean_recall >= prev_recall - 0.02, (
            f"Recall decreased from {prev_recall:.4f} to {mean_recall:.4f} "
            f"at nprobe={nprobe} (small noise tolerated)"
        )
        prev_recall = mean_recall


def test_ivf_pq_builds(medium_vectors):
    """IVF-PQ should build without error and m should divide D."""
    from annstudy.indexes import build_ivf_pq, compute_nlist, validate_pq_m

    dim = medium_vectors.shape[1]
    m = dim // 8
    validate_pq_m(dim, m)

    nlist = compute_nlist(5000, 4)
    index, meta = build_ivf_pq(medium_vectors, dim, nlist=nlist, m=m)

    assert meta["index_type"] == "ivf_pq"
    assert meta["train_seconds"] > 0
    assert meta["build_seconds"] > 0


def test_pq_m_validation():
    """validate_pq_m should raise on invalid m."""
    from annstudy.indexes import validate_pq_m

    validate_pq_m(128, 16)
    validate_pq_m(384, 48)

    with pytest.raises(ValueError):
        validate_pq_m(128, 13)


def test_build_grid_coverage():
    """Build grids should produce expected number of configurations."""
    from annstudy.indexes import get_build_grid

    flat_grid = get_build_grid("flat", 500_000, 384)
    assert len(flat_grid) == 1

    hnsw_grid = get_build_grid("hnsw", 500_000, 384)
    assert len(hnsw_grid) == 3

    ivf_flat_grid = get_build_grid("ivf_flat", 500_000, 384)
    assert len(ivf_flat_grid) == 3

    ivf_pq_grid = get_build_grid("ivf_pq", 500_000, 384)
    assert len(ivf_pq_grid) == 9

    for spec in ivf_pq_grid:
        assert 384 % spec["m"] == 0


def test_search_grid_respects_k():
    """Search grid for HNSW should not include efSearch below k."""
    from annstudy.indexes import get_search_grid

    grid = get_search_grid("hnsw", k=10)
    for params in grid:
        assert params["ef_search"] >= 10

    grid_k100 = get_search_grid("hnsw", k=100)
    for params in grid_k100:
        assert params["ef_search"] >= 100
