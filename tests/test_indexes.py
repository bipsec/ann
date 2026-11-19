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


def test_ivf_pq_refine_builds_and_searches(tmp_path, medium_vectors, medium_queries):
    """Regression: IVF-PQ+refine used to fail 100% of builds.

    ``IndexRefineFlat`` keeps its own flat copy and asserts base.ntotal == refine.ntotal, so
    wrapping an already-populated base raised
    ``'base_index->ntotal == refine_index->ntotal' failed``. Every ivf_pq_refine row in
    logs/failures.jsonl was this. The factory must wrap before adding.
    """
    import faiss

    from annstudy.indexes import build_ivf_pq_refine, compute_nlist, set_search_params

    dim = medium_vectors.shape[1]
    nlist = compute_nlist(len(medium_vectors), 4)
    index, meta = build_ivf_pq_refine(medium_vectors, dim, nlist=nlist, m=dim // 8, nbits=8)

    assert meta["index_type"] == "ivf_pq_refine"
    assert meta["add_seconds"] > 0
    assert index.ntotal == len(medium_vectors)

    # refine keeps full vectors, so it must be larger than the raw PQ codes
    for k_factor in (4, 16):
        set_search_params(index, "ivf_pq_refine", {"nprobe": 16, "k_factor": k_factor})
        _, ret_ids = index.search(medium_queries, 10)
        assert ret_ids.shape == (len(medium_queries), 10)
        assert (ret_ids >= 0).all()

    # must survive serialization, since Stage C reuses indexes from disk
    path = tmp_path / "refine.faiss"
    faiss.write_index(index, str(path))
    reloaded = faiss.read_index(str(path))
    assert reloaded.ntotal == index.ntotal


def test_refine_recall_beats_plain_ivf_pq(medium_vectors, medium_queries):
    """Refine re-ranks with exact distances, so it must not be worse than plain IVF-PQ."""
    import faiss

    from annstudy.indexes import (
        build_ivf_pq,
        build_ivf_pq_refine,
        compute_nlist,
        set_search_params,
    )
    from annstudy.metrics import recall_at_k_batch

    dim = medium_vectors.shape[1]
    k, nlist = 10, compute_nlist(len(medium_vectors), 4)

    gt = faiss.IndexFlatL2(dim)
    gt.add(np.ascontiguousarray(medium_vectors))
    gt_dists, _ = gt.search(medium_queries, k)

    def mean_recall(index, index_type, params):
        set_search_params(index, index_type, params)
        _, ids = index.search(medium_queries, k)
        return float(np.mean(recall_at_k_batch(
            ids, gt_dists, medium_vectors, medium_queries, k=k,
        )))

    pq, _ = build_ivf_pq(medium_vectors, dim, nlist=nlist, m=dim // 8)
    rf, _ = build_ivf_pq_refine(medium_vectors, dim, nlist=nlist, m=dim // 8)

    r_pq = mean_recall(pq, "ivf_pq", {"nprobe": 16})
    r_rf = mean_recall(rf, "ivf_pq_refine", {"nprobe": 16, "k_factor": 16})
    assert r_rf >= r_pq - 0.01, f"refine ({r_rf:.4f}) should not trail plain PQ ({r_pq:.4f})"


def test_set_search_params_rejects_unknown_type(medium_vectors):
    """A typo must raise rather than silently measure an untuned index."""
    from annstudy.indexes import build_flat, set_search_params

    index, _ = build_flat(medium_vectors, medium_vectors.shape[1])
    set_search_params(index, "flat", {})  # explicit no-op, exact search has no knobs
    with pytest.raises(ValueError, match="Unknown index type"):
        set_search_params(index, "hnsw_typo", {"ef_search": 32})


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
