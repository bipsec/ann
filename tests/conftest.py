"""Shared test fixtures."""

import json

import numpy as np
import pytest

# The real harness hardcodes a 2000/8000 query split (data.load_query_split asserts it), so the
# synthetic dataset below matches that shape even though it is tiny in every other dimension.
TUNING_SIZE = 2_000
TEST_SIZE = 8_000
N_QUERIES = TUNING_SIZE + TEST_SIZE


@pytest.fixture
def tiny_dataset(tmp_path):
    """A complete miniature on-disk dataset: passages, queries, ground truth, query split.

    Lets the runner and the memory pass be exercised end to end without touching the real
    11 GB embeddings. Returns a dict of paths plus N and D.
    """
    import faiss

    N, D, k_gt = 20_000, 32, 100
    rng = np.random.default_rng(7)

    passages = rng.standard_normal((N, D), dtype=np.float32)
    passages /= np.linalg.norm(passages, axis=1, keepdims=True)
    queries = rng.standard_normal((N_QUERIES, D), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    emb = tmp_path / "emb"
    emb.mkdir()
    p_path = emb / f"passages_D{D}.f32"
    q_path = emb / f"queries_D{D}.f32"
    passages.tofile(p_path)
    queries.tofile(q_path)

    flat = faiss.IndexFlatL2(D)
    flat.add(passages)
    gt_dists, gt_ids = flat.search(queries, k_gt)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    gt_path = gt_dir / f"gt_N{N}_D{D}.npz"
    np.savez(gt_path, ids=gt_ids, distances=gt_dists)

    split_path = tmp_path / "query_split.json"
    split_path.write_text(json.dumps({
        "tuning": list(range(TUNING_SIZE)),
        "test": list(range(TUNING_SIZE, N_QUERIES)),
    }))

    return {
        "N": N, "D": D, "root": tmp_path,
        "passages_path": str(p_path), "queries_path": str(q_path),
        "gt_path": str(gt_path), "query_split_path": str(split_path),
        "n_queries": N_QUERIES,
    }


@pytest.fixture
def tiny_spec(tiny_dataset, tmp_path):
    """A runner spec over ``tiny_dataset``, with a short search grid and small warm-up."""
    def _make(**overrides):
        spec = {
            "run_id": "testrun",
            "N": tiny_dataset["N"], "D": tiny_dataset["D"], "k": 10,
            "index_type": "ivf_flat", "build_params": {"nlist": 64},
            "search_grid": [{"nprobe": 4}],
            "seed": 0, "repeat_id": 0,
            "passages_path": tiny_dataset["passages_path"],
            "queries_path": tiny_dataset["queries_path"],
            "gt_path": tiny_dataset["gt_path"],
            "query_split_path": tiny_dataset["query_split_path"],
            "n_queries": tiny_dataset["n_queries"],
            "results_dir": str(tmp_path / "results_raw"),
            "measurement": {
                "warmup_queries": 10, "latency_passes": 1, "bootstrap_resamples": 20,
            },
        }
        spec.update(overrides)
        return spec
    return _make


@pytest.fixture
def toy_vectors():
    """100 random unit-norm vectors of dimension 32."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((100, 32)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def toy_queries():
    """10 random unit-norm query vectors of dimension 32."""
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((10, 32)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def medium_vectors():
    """5000 random unit-norm vectors of dimension 64."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5000, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def medium_queries():
    """100 random unit-norm query vectors of dimension 64."""
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((100, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
