"""Regression tests for the Stage C measurement fixes made before the H2 rerun.

Each test pins one defect found in the post-study audit (see logs/decisions.md, 2026-11-16).
"""

import time

import numpy as np
import pytest

from annstudy import filter_runner as fr
from annstudy.filters import make_correlated_mask


def _vecs(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _cfg(n_attr=1000, n_clusters=8, cluster_D=None):
    corr = {"n_clusters": n_clusters, "seed": 1, "selectivities": [0.1],
            "types": ["positive", "negative"]}
    if cluster_D is not None:
        corr["cluster_D"] = cluster_D
    return {"uncorrelated": {"n_attributes": n_attr, "seed": 2, "selectivities": [0.1]},
            "correlated": corr}


# ------------------------------------------------------------------ 1.1 cache keys

def test_cluster_cache_is_keyed_on_n_clusters(tmp_path):
    """A config with more clusters must not reuse the old cluster cache."""
    x = _vecs(2000, 16)
    a = fr.prepare_filters(2000, 16, x, _cfg(n_clusters=8), cache_dir=tmp_path)
    b = fr.prepare_filters(2000, 16, x, _cfg(n_clusters=16), cache_dir=tmp_path)
    assert a["correlated"]["centroids"].shape == (8, 16)
    assert b["correlated"]["centroids"].shape == (16, 16)


def test_attribute_cache_is_keyed_on_n_attributes(tmp_path):
    x = _vecs(2000, 16)
    a = fr.prepare_filters(2000, 16, x, _cfg(n_attr=100), cache_dir=tmp_path)
    b = fr.prepare_filters(2000, 16, x, _cfg(n_attr=100_000), cache_dir=tmp_path)
    assert a["uncorrelated"]["attrs"].max() < 100
    assert b["uncorrelated"]["attrs"].max() >= 100  # would be < 100 if the cache were reused


def test_job_key_changes_with_filter_config():
    base = {"cells": {"N": [1000], "D": [16]}, "indexes": {"hnsw": {}},
            "strategies": ["in_index_filtered"], "k": 10}
    j1 = fr.enumerate_filter_jobs(dict(base, filters=_cfg(n_clusters=8)))
    j2 = fr.enumerate_filter_jobs(dict(base, filters=_cfg(n_clusters=16)))
    assert {fr.job_key(j) for j in j1}.isdisjoint({fr.job_key(j) for j in j2})


# --------------------------------------------------------------- 1.2 empty predicates

def test_stale_centroids_raise_instead_of_empty_mask():
    """1000 cached centroids with n_clusters=10_000 used to give an all-False negative mask."""
    ids = np.zeros(100, dtype=np.int32)
    cents = _vecs(10, 4)
    with pytest.raises(ValueError, match="n_clusters"):
        make_correlated_mask(ids, cents, cents[0], 0.1, "negative", n_clusters=100)


def test_empty_predicate_recall_raises():
    with pytest.raises(ValueError, match="no vectors"):
        fr.filtered_recall(np.full((1, 10), -1), np.zeros((1, 10)), _vecs(5, 4), _vecs(1, 4),
                           10, np.array([0]))


# ---------------------------------------------------------- 1.3 same predicate at all D

def test_cluster_D_gives_identical_predicates_across_D(tmp_path):
    """With cluster_D pinned, D=8 and D=16 cells admit exactly the same passages."""
    full = _vecs(3000, 16)
    q16 = _vecs(5, 16, seed=7)
    masks = {}
    for D in (8, 16):
        prefix = full[:, :D] / np.linalg.norm(full[:, :D], axis=1, keepdims=True)
        prep = fr.prepare_filters(3000, D, prefix, _cfg(n_clusters=12, cluster_D=16),
                                  cache_dir=tmp_path, cluster_vectors=full)
        assert prep["correlated"]["query_D"] == 16
        get, static = fr.make_matching_provider(prep, "positive", 0.1, q16, 3000)
        assert not static
        masks[D] = [get(i) for i in range(len(q16))]
    for a, b in zip(masks[8], masks[16], strict=True):
        assert np.array_equal(a, b)


def test_query_dim_mismatch_raises():
    ids = np.arange(100, dtype=np.int32) % 4
    cents = _vecs(4, 16)
    with pytest.raises(ValueError, match="dim"):
        make_correlated_mask(ids, cents, _vecs(1, 8)[0], 0.1, "positive", n_clusters=4)


# --------------------------------------------------------- 1.4 timed region + counters

@pytest.fixture
def hnsw_toy():
    import faiss
    x = _vecs(3000, 16)
    h = faiss.IndexHNSWFlat(16, 16)
    h.add(x)
    return x, h


def test_selector_construction_is_outside_the_timed_region(monkeypatch, hnsw_toy):
    """A deliberately slow selector build must show up in struct_ns, not in the latency."""
    x, h = hnsw_toy
    real = fr.make_id_selector

    def slow(idx, N):
        time.sleep(0.003)
        return real(idx, N)

    monkeypatch.setattr(fr, "make_id_selector", slow)
    q = _vecs(4, 16, seed=3)
    match = [np.arange(i, 3000, 7) for i in range(len(q))]
    _, times, _, extra = fr._measure("in_index_filtered", h, "hnsw", x, q,
                                     lambda qi: match[qi], 10, {"ef_search": 32}, 2, None,
                                     3000, static=False)
    assert np.all(extra["struct_ns"] >= 3_000_000)
    assert np.all(times < 3_000_000), times


def test_ndis_is_recorded_and_grows_with_effort(hnsw_toy):
    x, h = hnsw_toy
    q = _vecs(8, 16, seed=4)
    allidx = np.arange(3000)
    nd = []
    for ef in (16, 256):
        _, _, _, extra = fr._measure("in_index_filtered", h, "hnsw", x, q, lambda qi: allidx,
                                     10, {"ef_search": ef}, 1, None, 3000, static=True)
        assert np.all(extra["ndis"] > 0)
        nd.append(extra["ndis"].mean())
    assert nd[1] > nd[0]
    # exact scan: one distance per matching vector
    _, _, sizes, extra = fr._measure("pre_filter_exact", None, "exact", x, q,
                                     lambda qi: allidx[:500], 10, {}, 1, None, 3000,
                                     static=True)
    assert np.all(extra["ndis"] == 500)


# ------------------------------------------------------------------------ 1.5 threads

def test_static_gt_restores_omp_threads(tmp_path):
    import faiss
    faiss.omp_set_num_threads(1)
    x = _vecs(500, 8)
    fr.filtered_gt_static(x, _vecs(3, 8), np.arange(0, 500, 2), 10, tmp_path / "gt.npz",
                          threads=4)
    assert faiss.omp_get_max_threads() == 1


# ------------------------------------------------------------------- interleaving

def test_interleave_cycles_through_D():
    jobs = [{"D": D, "N": 1, "correlation": "none", "target_S": S, "strategy": "s",
             "index_type": "hnsw", "safety": None}
            for D in (128, 384, 768, 1536) for S in (0.01, 0.02)]
    order = [j["D"] for j in fr.interleave_jobs(jobs)]
    assert order[:4] == [128, 384, 768, 1536]
    assert order[4:] == [128, 384, 768, 1536]


# --------------------------------------------------------------- end-to-end toy job

def test_run_filter_job_end_to_end(monkeypatch, tmp_path, hnsw_toy):
    """One correlated in-index job through the full protocol, checking the new columns."""
    x, h = hnsw_toy
    monkeypatch.setattr(fr, "FILTERED_GT", tmp_path / "gt")
    cfg = {"filters": _cfg(n_clusters=10), "recall_targets": [0.9],
           "measurement": {"latency_passes": 1, "bootstrap_resamples": 20,
                           "warmup_queries": 5}, "k": 10}
    prep = fr.prepare_filters(3000, 16, x, cfg["filters"], cache_dir=tmp_path / "f")
    job = {"N": 3000, "D": 16, "k": 10, "strategy": "in_index_filtered",
           "correlation": "positive", "target_S": 0.1, "index_type": "hnsw", "safety": None,
           "filter_fp": fr.filter_fingerprint(cfg["filters"])}
    rows = fr.run_filter_job(job, cfg, x, prep, _vecs(30, 16, 5), _vecs(30, 16, 6), h)
    (row,) = rows
    assert row["reachable"]
    for col in ("meets_R_on_test", "ndis_mean", "filter_struct_us_p50", "filter_fp"):
        assert col in row
    assert row["ndis_mean"] > 0
    assert row["timing_protocol"].startswith("v2")


# --------------------------------------------------------------- HNSW repeat builds

def test_hnsw_repeat_builds_differ_and_seed0_is_default():
    """FAISS HNSW repeat builds were byte-identical; a seed must now change the graph."""
    import faiss

    from annstudy.indexes import build_hnsw

    x = _vecs(3000, 16)

    def blob(seed):
        idx, _ = build_hnsw(x, 16, 16, 100, seed=seed)
        return faiss.serialize_index(idx).tobytes()

    default = blob(None)
    assert blob(0) == default          # seed 0 reproduces every pre-existing index
    assert blob(1) != default          # repeats are genuinely different graphs
    assert blob(1) != blob(2)
