"""Correctness tests for the Stage C filter strategies (small inline indexes)."""

import numpy as np
import pytest

from annstudy import filter_runner as fr
from annstudy.filters import compute_filtered_ground_truth
from annstudy.metrics import recall_at_k_batch


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    N, D = 4000, 32
    xb = rng.random((N, D)).astype("float32")
    xq = rng.random((6, D)).astype("float32")
    # Uncorrelated attribute, selectivity ~0.25 via even-5ths
    attrs = rng.integers(0, 10_000, size=N)
    mask = attrs < 2500  # S ~ 0.25
    return N, D, xb, xq, mask


def test_pre_filter_exact_recall_is_one(toy):
    N, D, xb, xq, mask = toy
    matching = np.where(mask)[0]
    k = 10
    gt_ids, gt_d = compute_filtered_ground_truth(xb, xq, lambda qi: mask, k)
    ret = np.stack([fr.strategy_pre_filter_exact(xb, xq[i], matching, k) for i in range(len(xq))])
    rec = recall_at_k_batch(ret, gt_d, xb, xq, k=k)
    assert np.all(rec >= 0.999), f"pre-filter recall should be 1.0, got {rec}"
    # Every returned id must be in the matching set.
    assert set(ret[ret >= 0].tolist()).issubset(set(matching.tolist()))


def test_in_index_filtered_respects_mask_and_improves_with_effort(toy):
    import faiss
    N, D, xb, xq, mask = toy
    matching = np.where(mask)[0]
    k = 10
    h = faiss.IndexHNSWFlat(D, 32)
    h.hnsw.efConstruction = 200
    h.add(xb)
    gt_ids, gt_d = compute_filtered_ground_truth(xb, xq, lambda qi: mask, k)

    sel, bitmap, _ = fr.make_id_selector(matching, N)

    def run(ef):
        params = fr.search_params_filtered("hnsw", sel, {"ef_search": ef})
        return np.stack([fr.strategy_in_index_filtered(h, "hnsw", xq[i], k, params)
                         for i in range(len(xq))])

    ret_lo = run(16)
    ret_hi = run(256)
    # All returned ids are in the allowed set.
    assert set(ret_hi[ret_hi >= 0].tolist()).issubset(set(matching.tolist()))
    rec_lo = recall_at_k_batch(ret_lo, gt_d, xb, xq, k=k).mean()
    rec_hi = recall_at_k_batch(ret_hi, gt_d, xb, xq, k=k).mean()
    assert rec_hi >= rec_lo - 1e-9
    assert rec_hi >= 0.9, f"high-effort filtered HNSW recall too low: {rec_hi}"


def test_post_filter_overfetch_respects_mask(toy):
    import faiss
    N, D, xb, xq, mask = toy
    matching_set = set(np.where(mask)[0].tolist())
    S = mask.mean()
    k = 10
    q = faiss.IndexFlatL2(D)
    ivf = faiss.IndexIVFFlat(q, D, 64)
    ivf.train(xb)
    ivf.add(xb)
    ivf.nprobe = 16
    ret = np.stack([
        fr.strategy_post_filter_overfetch(ivf, "ivf_flat", xq[i], mask, k,
                                          {"nprobe": 16}, safety=3, S=S, N=N)
        for i in range(len(xq))
    ])
    assert set(ret[ret >= 0].tolist()).issubset(matching_set)


def test_id_selector_bitmap_matches_batch_semantics(toy):
    """The bitmap selector must admit exactly the same ids as an IDSelectorBatch."""
    import faiss
    N, D, xb, xq, mask = toy
    matching = np.where(mask)[0]
    sel_bits, bitmap, built_mask = fr.make_id_selector(matching, N)
    sel_batch = faiss.IDSelectorBatch(np.ascontiguousarray(matching.astype(np.int64)))

    assert np.array_equal(built_mask, mask)
    for i in range(0, N, 97):  # sample across the id range
        assert sel_bits.is_member(int(i)) == sel_batch.is_member(int(i))


def test_in_index_at_S1_matches_unfiltered(toy):
    """Plan acceptance check: at S=1.0 filtered search matches unfiltered search."""
    import faiss
    N, D, xb, xq, _ = toy
    k = 10
    h = faiss.IndexHNSWFlat(D, 32)
    h.hnsw.efConstruction = 200
    h.add(xb)

    all_ids = np.arange(N, dtype=np.int64)
    sel, bitmap, _ = fr.make_id_selector(all_ids, N)
    params = fr.search_params_filtered("hnsw", sel, {"ef_search": 96})
    filtered = np.stack([fr.strategy_in_index_filtered(h, "hnsw", xq[i], k, params)
                         for i in range(len(xq))])

    h.hnsw.efSearch = 96
    _, unfiltered = h.search(xq, k)
    assert np.array_equal(filtered, unfiltered), "S=1.0 must reproduce unfiltered results"


def test_pre_filter_exact_paths_agree(toy):
    """The gather path and the scan-and-mask path must return identical top-k.

    The implementation switches between them by subset size, so they have to agree.
    """
    N, D, xb, xq, mask = toy
    matching = np.where(mask)[0]
    k = 10
    for i in range(len(xq)):
        gathered = fr.strategy_pre_filter_exact(xb, xq[i], matching, k, gather_fraction=1.0)
        scanned = fr.strategy_pre_filter_exact(xb, xq[i], matching, k, gather_fraction=0.0)
        assert np.array_equal(gathered, scanned)


def test_make_mask_fn_uncorrelated_selectivity():
    prep = {"uncorrelated": {"attrs": np.arange(1000) % 10_000, "n_attributes": 10_000}}
    mask_fn = fr.make_mask_fn(prep, "none", target_S=0.5, query_vecs=None)
    m = mask_fn(0)
    # attrs 0..999, threshold = 0.5*10000 = 5000 -> all < 5000 -> all True here
    assert m.sum() == 1000


def test_enumerate_filter_jobs_shape():
    """Exact scan is index-independent, so it is enumerated once per cell, not per index."""
    from annstudy.filter_runner import enumerate_filter_jobs, job_key

    cfg = {
        "cells": {"N": [500_000], "D": [384, 1536]},
        "indexes": {"hnsw": {}, "ivf_flat": {}},
        "strategies": ["pre_filter_exact", "in_index_filtered", "post_filter_overfetch"],
        "filters": {
            "uncorrelated": {"selectivities": [1.0, 0.1]},
            "correlated": {"selectivities": [0.01], "types": ["positive", "negative"]},
        },
        "post_filter": {"safety_factors": [1.5, 3]},
        "k": 10,
    }
    jobs = enumerate_filter_jobs(cfg)
    n_variants = 2 + 2  # 2 uncorrelated S + 2 correlated types x 1 S
    by_strategy = {}
    for j in jobs:
        by_strategy.setdefault(j["strategy"], []).append(j)

    assert len(by_strategy["pre_filter_exact"]) == 2 * n_variants          # cells x filters
    assert len(by_strategy["in_index_filtered"]) == 2 * 2 * n_variants     # x index types
    assert len(by_strategy["post_filter_overfetch"]) == 2 * 2 * n_variants * 2  # x safeties
    assert all(j["index_type"] == "exact" for j in by_strategy["pre_filter_exact"])
    assert len({job_key(j) for j in jobs}) == len(jobs), "job keys must be unique"


def test_crossover_detects_positive_to_negative_sign_change():
    """Regression: S* is a positive->negative crossing as S increases.

    In-index latency minus exact-scan latency starts positive (scanning a tiny subset is
    cheap) and ends negative. An implementation that only looked for negative->positive found
    no crossover anywhere and silently reported 'none in range'.
    """
    from annstudy.analysis.crossover import _interp_sign_change

    S = [0.001, 0.01, 0.1, 1.0]
    diff = [500.0, 300.0, -100.0, -800.0]  # crosses between 0.01 and 0.1
    s_star = _interp_sign_change(S, diff)
    assert s_star is not None
    assert 0.01 < s_star < 0.1

    # and the reverse direction is still detected
    assert _interp_sign_change(S, [-5.0, -1.0, 2.0, 9.0]) is not None
    # no crossing at all
    assert _interp_sign_change(S, [5.0, 4.0, 3.0, 2.0]) is None


def test_filter_cache_write_is_atomic_under_concurrency(tmp_path):
    """Regression: concurrent shards writing the same cache produced truncated reads.

    Stage C runs up to 16 shards per cell; several can find a cache missing and compute it at
    once. An in-place np.save let a reader observe a partial array ("could only read 262128 of
    1000000 elements"), which killed shards mid-cell and lost 20 jobs. Publishing via rename
    makes the loser of the race harmless.
    """
    import threading

    import numpy as np

    from annstudy import filter_runner as fr

    path = tmp_path / "attrs.npy"
    payload = np.arange(500_000, dtype=np.int64)
    errors = []

    def writer():
        try:
            fr._save_atomic(path, payload)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        for _ in range(50):
            got = fr._load_cached(path)
            if got is not None and not np.array_equal(got, payload):
                errors.append(AssertionError(f"partial read: {got.shape}"))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert np.array_equal(fr._load_cached(path), payload)
    # no temporary files left behind
    assert not list(tmp_path.glob("*.tmp.npy"))


def test_load_cached_treats_truncated_file_as_a_miss(tmp_path):
    """A truncated cache must be recomputed, not raised through."""
    import numpy as np

    from annstudy import filter_runner as fr

    path = tmp_path / "trunc.npy"
    fr._save_atomic(path, np.arange(10_000, dtype=np.int64))
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 3])  # simulate the observed partial write
    assert fr._load_cached(path) is None
