"""FAISS index factory functions and parameter grids."""

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)


def compute_nlist(N: int, c: int) -> int:
    """Return round(c * sqrt(N))."""
    return round(c * math.sqrt(N))


def validate_pq_m(D: int, m: int) -> None:
    """Check that m divides D. Raises ValueError if not."""
    if D % m != 0:
        raise ValueError(f"PQ subquantizers m={m} does not divide D={D}")


def _time_it(fn):
    """Time a function call, return (result, seconds)."""
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def build_flat(vectors: np.ndarray, dim: int) -> tuple:
    """Build IndexFlatL2. Returns (index, metadata)."""
    import faiss

    index = faiss.IndexFlatL2(dim)
    _, add_sec = _time_it(lambda: index.add(np.ascontiguousarray(vectors)))

    meta = {
        "index_type": "flat",
        "train_seconds": 0.0,
        "add_seconds": add_sec,
        "build_seconds": add_sec,
        "train_size": 0,
    }
    return index, meta


FAISS_HNSW_DEFAULT_SEED = 12345  # faiss::HNSW's built-in level-generator seed


def _seed_hnsw_levels(index, seed) -> None:
    """Seed the HNSW level generator so repeat builds are genuinely different graphs.

    FAISS HNSW draws node levels from a generator with a FIXED seed, and its parallel insertion
    turned out to be deterministic too: repeat builds were byte-identical (verified 2026-11-16),
    so "3 builds per configuration" carried no build variance at all for HNSW. ``seed`` is an
    offset from FAISS's default, so seed 0 (or None) reproduces every earlier index exactly.
    """
    import faiss

    if seed:
        index.hnsw.rng = faiss.RandomGenerator(FAISS_HNSW_DEFAULT_SEED + int(seed))


def build_hnsw(
    vectors: np.ndarray,
    dim: int,
    M: int,
    ef_construction: int = 200,
    seed: int | None = None,
) -> tuple:
    """Build IndexHNSWFlat. Returns (index, metadata)."""
    import faiss

    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    _seed_hnsw_levels(index, seed)

    _, add_sec = _time_it(lambda: index.add(np.ascontiguousarray(vectors)))

    meta = {
        "index_type": "hnsw",
        "M": M,
        "ef_construction": ef_construction,
        "train_seconds": 0.0,
        "add_seconds": add_sec,
        "build_seconds": add_sec,
        "train_size": 0,
    }
    return index, meta


def build_hnsw_sq(
    vectors: np.ndarray,
    dim: int,
    M: int,
    ef_construction: int = 200,
    nbits: int = 8,
    seed: int = 0,
) -> tuple:
    """Build IndexHNSWSQ (HNSW over 8-bit scalar-quantized storage). Returns (index, metadata).

    One byte per dimension (~D bytes/vector, 4x compression vs raw float32), so it can fit a
    sub-raw memory budget while keeping a graph index's query profile — the missing point in the
    memory-budget analysis (no scalar-quantized index was in the original grid).
    """
    import faiss

    index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
    index.hnsw.efConstruction = ef_construction
    _seed_hnsw_levels(index, seed)
    vecs = np.ascontiguousarray(vectors)
    _, train_sec = _time_it(lambda: index.train(vecs))  # SQ needs training
    _, add_sec = _time_it(lambda: index.add(vecs))
    meta = {
        "index_type": "hnsw_sq",
        "M": M,
        "ef_construction": ef_construction,
        "nbits": 8,
        "train_seconds": train_sec,
        "add_seconds": add_sec,
        "build_seconds": train_sec + add_sec,
        "train_size": len(vectors),
    }
    return index, meta


def build_ivf_flat(
    vectors: np.ndarray,
    dim: int,
    nlist: int,
    seed: int = 0,
) -> tuple:
    """Build IndexIVFFlat over a flat L2 quantizer. Returns (index, metadata)."""
    import faiss

    n = vectors.shape[0]
    train_size = min(n, 256 * nlist)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(n, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(vectors[train_idx].astype(np.float32))

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    _, train_sec = _time_it(lambda: index.train(train_vecs))
    _, add_sec = _time_it(lambda: index.add(np.ascontiguousarray(vectors)))

    meta = {
        "index_type": "ivf_flat",
        "nlist": nlist,
        "train_seconds": train_sec,
        "add_seconds": add_sec,
        "build_seconds": train_sec + add_sec,
        "train_size": train_size,
    }
    return index, meta


def build_ivf_pq(
    vectors: np.ndarray,
    dim: int,
    nlist: int,
    m: int,
    nbits: int = 8,
    seed: int = 0,
) -> tuple:
    """Build IndexIVFPQ. Returns (index, metadata)."""
    import faiss

    validate_pq_m(dim, m)

    n = vectors.shape[0]
    train_size = min(n, max(256 * nlist, 256_000))

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(n, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(vectors[train_idx].astype(np.float32))

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    _, train_sec = _time_it(lambda: index.train(train_vecs))
    _, add_sec = _time_it(lambda: index.add(np.ascontiguousarray(vectors)))

    meta = {
        "index_type": "ivf_pq",
        "nlist": nlist,
        "m": m,
        "nbits": nbits,
        "train_seconds": train_sec,
        "add_seconds": add_sec,
        "build_seconds": train_sec + add_sec,
        "train_size": train_size,
    }
    return index, meta


def build_ivf_pq_refine(
    vectors: np.ndarray,
    dim: int,
    nlist: int,
    m: int,
    nbits: int = 8,
    seed: int = 0,
) -> tuple:
    """Build IndexIVFPQ wrapped in IndexRefineFlat. Returns (index, metadata)."""
    import faiss

    validate_pq_m(dim, m)

    n = vectors.shape[0]
    train_size = min(n, max(256 * nlist, 256_000))

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(n, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(vectors[train_idx].astype(np.float32))

    quantizer = faiss.IndexFlatL2(dim)
    base_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    _, train_sec = _time_it(lambda: base_index.train(train_vecs))

    # Wrap BEFORE adding. IndexRefineFlat keeps its own flat copy of every vector and its
    # constructor asserts base_index->ntotal == refine_index->ntotal, so adding to the base
    # first makes construction fail. Adding through the wrapper fills both. faiss tracks the
    # sub-index in `referenced_objects`, so returning only the wrapper is safe.
    refine_index = faiss.IndexRefineFlat(base_index)
    _, add_sec = _time_it(lambda: refine_index.add(np.ascontiguousarray(vectors)))

    meta = {
        "index_type": "ivf_pq_refine",
        "nlist": nlist,
        "m": m,
        "nbits": nbits,
        "train_seconds": train_sec,
        "add_seconds": add_sec,
        "build_seconds": train_sec + add_sec,
        "train_size": train_size,
    }
    return refine_index, meta


def get_build_grid(index_type: str, N: int, D: int) -> list[dict]:
    """Return the list of build-parameter dicts for a given index type."""
    if index_type == "flat":
        return [{}]

    if index_type in ("hnsw", "hnsw_sq"):
        return [
            {"M": M, "ef_construction": 200}
            for M in [16, 32, 64]
        ]

    c_values = [2, 4, 8]

    if index_type == "ivf_flat":
        return [
            {"nlist": compute_nlist(N, c)}
            for c in c_values
        ]

    m_divisors = [16, 8, 4]

    if index_type == "ivf_pq":
        grid = []
        for c in c_values:
            for md in m_divisors:
                m = D // md
                if D % m == 0:
                    grid.append({
                        "nlist": compute_nlist(N, c),
                        "m": m,
                        "nbits": 8,
                    })
        return grid

    if index_type == "ivf_pq_refine":
        grid = []
        for c in c_values:
            for md in m_divisors:
                m = D // md
                if D % m == 0:
                    grid.append({
                        "nlist": compute_nlist(N, c),
                        "m": m,
                        "nbits": 8,
                    })
        return grid

    raise ValueError(f"Unknown index type: {index_type}")


def get_search_grid(index_type: str, nlist: int | None = None, k: int = 10) -> list[dict]:
    """Return the list of search-parameter dicts."""
    if index_type == "flat":
        return [{}]

    if index_type in ("hnsw", "hnsw_sq"):
        ef_values = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        return [
            {"ef_search": ef}
            for ef in ef_values
            if ef >= k
        ]

    if index_type in ("ivf_flat", "ivf_pq"):
        nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if nlist is not None:
            nprobe_values = [np for np in nprobe_values if np <= nlist]
        return [{"nprobe": np} for np in nprobe_values]

    if index_type == "ivf_pq_refine":
        nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if nlist is not None:
            nprobe_values = [np for np in nprobe_values if np <= nlist]
        k_factors = [4, 16]
        grid = []
        for np_val in nprobe_values:
            for kf in k_factors:
                grid.append({"nprobe": np_val, "k_factor": kf})
        return grid

    raise ValueError(f"Unknown index type: {index_type}")


def set_search_params(index, index_type: str, params: dict) -> None:
    """Apply search parameters to an index."""
    import faiss

    if index_type in ("hnsw", "hnsw_sq"):
        index.hnsw.efSearch = params["ef_search"]
    elif index_type in ("ivf_flat", "ivf_pq"):
        index.nprobe = params["nprobe"]
    elif index_type == "ivf_pq_refine":
        index.k_factor = params.get("k_factor", 1)
        base = faiss.downcast_index(index.base_index)
        base.nprobe = params["nprobe"]
    elif index_type == "flat":
        pass  # exact search has no tunable parameters
    else:
        raise ValueError(
            f"Unknown index type: {index_type!r} — refusing to measure an untuned index."
        )
