"""Tests for metrics module."""

import numpy as np
import pytest


def test_flat_recall_is_one(toy_vectors, toy_queries):
    """Flat index should achieve perfect recall."""
    import faiss

    from annstudy.metrics import recall_at_k

    dim = toy_vectors.shape[1]
    k = 10

    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    gt_dists, gt_ids = index.search(toy_queries, k)
    ret_dists, ret_ids = index.search(toy_queries, k)

    recalls = recall_at_k(
        ret_ids, gt_ids, gt_dists,
        toy_vectors, toy_queries, k=k,
    )
    assert np.all(recalls == 1.0), f"Flat recall should be 1.0, got {recalls}"


def test_recall_handles_ties():
    """Recall should handle tied distances correctly."""
    from annstudy.metrics import recall_at_k

    passages = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)

    queries = np.array([[0.0, 0.0]], dtype=np.float32)

    gt_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    gt_distances = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    retrieved_ids = np.array([[0, 4, 2, 3]], dtype=np.int64)

    recalls = recall_at_k(
        retrieved_ids, gt_ids, gt_distances,
        passages, queries, k=4,
    )
    assert recalls[0] == 1.0, (
        f"All 4 retrieved items are within the k-th GT distance, recall should be 1.0, got {recalls[0]}"
    )


def test_bootstrap_ci_covers_mean():
    """Bootstrap CI should contain the sample mean."""
    from annstudy.metrics import bootstrap_ci

    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=1000)

    point, lo, hi = bootstrap_ci(values, np.mean, n_resamples=1000, seed=0)
    assert lo <= point <= hi, f"CI [{lo}, {hi}] doesn't contain point {point}"
    assert lo < hi, "CI should have nonzero width"


def test_latency_stats_plausible():
    """Latency stats should return reasonable values."""
    from annstudy.metrics import latency_stats

    ns_values = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    stats = latency_stats(ns_values)

    assert stats["lat_mean_us"] == pytest.approx(3.0, abs=0.1)
    assert stats["lat_p50_us"] == pytest.approx(3.0, abs=0.1)
    assert stats["lat_p95_us"] > stats["lat_p50_us"]
    assert stats["lat_p99_us"] >= stats["lat_p95_us"]


def test_measure_latency_returns_plausible_values(toy_vectors, toy_queries):
    """Measured latency should be positive and reasonable."""
    import faiss

    from annstudy.metrics import measure_latency

    dim = toy_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    per_query_ns = measure_latency(index, toy_queries, k=5, n_passes=2)
    assert per_query_ns.shape == (toy_queries.shape[0],)
    assert np.all(per_query_ns > 0), "All latencies should be positive"
    assert np.all(per_query_ns < 1e9), "No query should take more than 1 second"


def test_tuned_granularity_is_one_row_per_cell_index_target():
    """Plan section 5: the tuned table is "one row per cell, index, R".

    Selection must be joint over build variants and search settings — the cheapest
    configuration OF THE INDEX TYPE, not of each build variant. Grouping by build_run_id
    emitted one row per variant (e.g. all three HNSW M values), which both departs from the
    schema and multiplies downstream cost: every such row is rebuilt three times at T threads
    and has its memory measured.
    """
    import pandas as pd

    from annstudy.tune import find_cheapest_setting

    # Two HNSW build variants in one cell. M=32 is cheaper at the same recall, so it must win
    # and only ONE row may be emitted for (cell, hnsw, R).
    rows = []
    for build, ef_lat in (("bA", 500.0), ("bB", 300.0)):
        for ef, rec in ((32, 0.92), (64, 0.97)):
            rows.append({
                "index_type": "hnsw", "build_run_id": build, "N": 500_000, "D": 384,
                "k": 10, "query_split": "tuning", "ef_search": ef, "nprobe": None,
                "k_factor": None, "recall_mean": rec,
                "lat_p95_us": ef_lat * (2 if ef == 64 else 1), "qps": None,
            })
    tuned = find_cheapest_setting(pd.DataFrame(rows), [0.90, 0.95])

    assert len(tuned) == 2, f"expected one row per R for this cell/index, got {len(tuned)}"
    assert tuned.groupby(["index_type", "N", "D", "R"]).size().max() == 1

    at90 = tuned[tuned["R"] == 0.90].iloc[0]
    assert at90["best_build_run_id"] == "bB", "must pick the cheaper build variant"
    assert at90["ef_search"] == 32
    at95 = tuned[tuned["R"] == 0.95].iloc[0]
    assert at95["ef_search"] == 64, "R=0.95 needs the higher-effort setting"


def test_unreachable_recorded_per_cell_index_not_per_variant():
    """An index that no variant can push to R yields exactly one unreachable row per R."""
    import pandas as pd

    from annstudy.tune import find_cheapest_setting

    rows = [
        {"index_type": "ivf_pq", "build_run_id": b, "N": 500_000, "D": 1536, "k": 10,
         "query_split": "tuning", "ef_search": None, "nprobe": npr, "k_factor": None,
         "recall_mean": rec, "lat_p95_us": 1000.0, "qps": None}
        for b, rec in (("b1", 0.74), ("b2", 0.888)) for npr in (256, 512)
    ]
    tuned = find_cheapest_setting(pd.DataFrame(rows), [0.90, 0.95])
    assert len(tuned) == 2
    assert not tuned["reachable"].any()


def test_hypothesis_registry_is_the_single_source_for_both_tables():
    """findings.md and the paper must not disagree about what was tested.

    The statements were duplicated in two hand-maintained dicts whose wording had already
    drifted, which leaves a reader comparing two tables that describe different hypotheses.
    """
    from annstudy.analysis.run_all import HYPOTHESES, hypothesis_rows

    rows = hypothesis_rows()
    assert [r[0] for r in rows] == list(HYPOTHESES)
    for key, stmt, origin, evidence in rows:
        assert stmt and not stmt.endswith("."), f"{key}: statements are table cells, no period"
        assert origin, f"{key}: every hypothesis must declare where it came from"
        assert evidence, f"{key}: every hypothesis must name its backing file"
    # a post-hoc hypothesis must never be presentable as pre-registered
    assert "post-hoc" in HYPOTHESES["H3"]["origin"]
    assert all("proposal" in HYPOTHESES[k]["origin"]
               for k in ("H1a", "H1b", "H1c", "H2", "H2b"))
