"""Stage C: filtered-ANN search runner.

Compares three strategies for answering a top-k query restricted to a subset of vectors
selected by a predicate of selectivity S:

1. ``pre_filter_exact``     — gather the matching vectors, brute-force exact top-k (recall 1.0).
2. ``in_index_filtered``    — pass an ``IDSelectorBatch`` to the index search (native FAISS).
3. ``post_filter_overfetch`` — over-fetch ``ceil(safety*k/S)`` candidates, drop non-matches.

For each (index, correlation, S, strategy) it sweeps the search parameter on the tuning
queries, picks the cheapest setting reaching each recall target, re-measures on the test
queries, and appends rows to ``results/raw/filter_runs/`` (schema in EXPERIMENT_PLAN §5).

Query slicing is POSITIONAL (decision M8): tuning = positions 0..1999, test = 2000..9999.
"""

import argparse
import logging
import math
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from annstudy.faiss_setup import ensure_opt_level

logger = logging.getLogger(__name__)

FILTER_CACHE = Path("data/filters")
FILTERED_GT = Path("data/gt/filtered")
MATCHING_CACHE_BYTES = 512 * 1024**2  # per-job budget for caching per-query matching sets


# ----------------------------------------------------------------- strategies (pure)

def passage_sq_norms(passages):
    """Squared L2 norms of every passage, for the BLAS-based exact scan. Compute once per cell."""
    return np.einsum("ij,ij->i", passages, passages)


def strategy_pre_filter_exact(passages, query, matching_idx, k, gather_fraction=0.5,
                              sq_norms=None):
    """Exact top-k over the matching subset. Recall is 1.0 by construction.

    This is the baseline the filtered-ANN crossover S* is measured against, so it must be a
    *competent* exact scan — an artificially slow one would shift S* by an order of magnitude
    and corrupt H2. Two deliberate choices:

    - Distances come from a BLAS matvec rather than elementwise numpy, using
      ``||q-x||^2 = ||x||^2 - 2 q.x + ||q||^2``. The final term is constant per query and
      dropped, so ranking by ``||x||^2 - 2 q.x`` is exact for arbitrary vectors (not only
      unit-norm ones). Pass ``sq_norms`` from :func:`passage_sq_norms` to avoid recomputing
      ``||x||^2``; without it the norms are computed per call, which is correct but slower.
    - For a large matching subset, gathering it would copy most of the corpus per query
      (6 GB at N=1M, D=1536, S=1.0), which no real implementation would do. Above
      ``gather_fraction`` of the corpus the scan runs over all vectors and masks the
      non-matching entries instead; below it, the subset is gathered.
    """
    n_match = len(matching_idx)
    if n_match == 0:
        return np.full(k, -1, dtype=np.int64)

    N = passages.shape[0]
    kk = min(k, n_match)
    out = np.full(k, -1, dtype=np.int64)

    if n_match <= gather_fraction * N:
        sub = passages[matching_idx]
        sq = sq_norms[matching_idx] if sq_norms is not None else passage_sq_norms(sub)
        dists = sq - 2.0 * (sub @ query)
        top = np.argpartition(dists, kk - 1)[:kk]
        top = top[np.argsort(dists[top])]
        out[:kk] = matching_idx[top]
        return out

    sq = sq_norms if sq_norms is not None else passage_sq_norms(passages)
    dists = sq - 2.0 * (passages @ query)
    keep = np.zeros(N, dtype=bool)
    keep[matching_idx] = True
    dists = np.where(keep, dists, np.inf)
    top = np.argpartition(dists, kk - 1)[:kk]
    top = top[np.argsort(dists[top])]
    out[:kk] = top
    return out


def make_id_selector(matching_idx, N):
    """Build an ``IDSelectorBitmap`` for ``matching_idx``. Returns (selector, bitmap, mask).

    Bitmap rather than ``IDSelectorBatch``: measured here at N=100K, constructing a Batch
    selector over 100K ids costs ~5700 us, which utterly dominates a ~105 us search and would
    have made Stage C a measurement of selector construction. The bitmap costs ~4.5 us to
    build and searches at ~109 us versus ~104 us unfiltered.

    The returned bitmap array MUST stay alive for as long as the selector is used: faiss keeps
    a borrowed pointer to it and does not copy.
    """
    import faiss

    mask = np.zeros(N, dtype=bool)
    mask[matching_idx] = True
    bitmap = np.packbits(mask, bitorder="little")
    selector = faiss.IDSelectorBitmap(len(bitmap), faiss.swig_ptr(bitmap))
    return selector, bitmap, mask


def search_params_filtered(index_type, selector, search_param):
    """FAISS SearchParameters carrying an ID selector, per index type."""
    import faiss

    if index_type == "hnsw":
        return faiss.SearchParametersHNSW(sel=selector, efSearch=int(search_param["ef_search"]))
    if index_type in ("ivf_flat", "ivf_pq"):
        return faiss.SearchParametersIVF(sel=selector, nprobe=int(search_param["nprobe"]))
    raise ValueError(f"in_index_filtered unsupported for {index_type}")


def strategy_in_index_filtered(index, index_type, query, k, params):
    """Filtered search via a FAISS ID selector carried in ``params``."""
    _, ids = index.search(query[None, :], k, params=params)
    return ids[0]


def strategy_post_filter_overfetch(index, index_type, query, keep_mask, k, search_param,
                                   safety, S, N):
    """Over-fetch ceil(safety*k/S) candidates, drop non-matches, keep first k.

    ``keep_mask`` is a boolean array over all N ids: a direct array lookup, which is both
    faster and closer to how a real system would test membership than a Python set.
    """
    k_fetch = min(N, max(k, int(math.ceil(safety * k / max(S, 1e-9)))))
    _, ids = index.search(query[None, :], k_fetch)
    ids = ids[0]
    ids = ids[ids >= 0]
    kept = ids[keep_mask[ids]][:k]
    out = np.full(k, -1, dtype=np.int64)
    out[:len(kept)] = kept
    return out


# -------------------------------------------------------------- filter preparation

def _save_atomic(path, array) -> None:
    """Write a .npy cache so concurrent readers never observe a partial file.

    Stage C runs many shards at once, and several can find the same cache missing and compute
    it simultaneously. Writing in place let a reader see a half-written array
    ("could only read 262128 of 1000000 elements"), which killed shards mid-cell and cost 20
    jobs. Writing to a unique temporary file in the same directory and then renaming makes the
    publish atomic on POSIX, so the losers of the race simply overwrite with identical bytes.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + ".", suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp, array, allow_pickle=False)
        os.replace(tmp, path)  # atomic
    finally:
        Path(tmp).unlink(missing_ok=True)


def _savez_atomic(path, **arrays) -> None:
    """``np.savez`` published by rename, for the filtered-GT caches shared across shards."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + ".", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _load_gt_cache(path, n_queries):
    """Load a filtered-GT cache if it is readable and has one row per query, else None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = np.load(path)
        if d["ids"].shape[0] == n_queries:
            return d["ids"], d["distances"]
    except Exception as e:  # truncated or corrupt: recompute
        logger.warning(f"Ignoring unreadable GT cache {path} ({e}); recomputing.")
    return None


def _load_cached(path):
    """Load a .npy cache, returning None if it is absent or unreadable (e.g. truncated)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:  # truncated or corrupt: treat as a miss and recompute
        logger.warning(f"Ignoring unreadable cache {path} ({e}); recomputing.")
        return None


def filter_settings(filters_cfg: dict) -> dict:
    """The filter-construction settings that determine WHICH vectors a predicate admits.

    Every cache and every job key derived from a filter must depend on all of these. The
    original caches were keyed on (N[, D]) only, so a config with a finer attribute grid or more
    clusters silently reused the old arrays (``stage_c_rerun.yaml`` would have produced S=1 for
    every uncorrelated S >= 0.1 and empty negative masks).
    """
    uc = filters_cfg.get("uncorrelated", {}) or {}
    cc = filters_cfg.get("correlated", {}) or {}
    return {
        "n_attributes": int(uc.get("n_attributes", 10_000)),
        "attr_seed": int(uc.get("seed", 42)),
        "n_clusters": int(cc.get("n_clusters", 1000)),
        "cluster_seed": int(cc.get("seed", 42)),
        # None = cluster on the cell's own D (the original Stage C behaviour, which makes the
        # correlated predicate differ across D). An integer pins one clustering for every D.
        "cluster_D": (int(cc["cluster_D"]) if cc.get("cluster_D") is not None else None),
    }


def filter_fingerprint(filters_cfg: dict) -> str:
    """Short stable hash of :func:`filter_settings`, used in cache names and job keys."""
    from annstudy.runner import config_hash
    return config_hash(filter_settings(filters_cfg))[:10]


def _load_cluster_vectors(N, cluster_D):
    """First N passages at the clustering dimension (memmap; only read during k-means)."""
    from annstudy.coordinate import passages_path
    return np.asarray(np.memmap(passages_path(cluster_D), dtype=np.float32, mode="r",
                                shape=(N, cluster_D)))


def prepare_filters(N, D, passages, filters_cfg, cache_dir=FILTER_CACHE, cluster_vectors=None):
    """Compute/cache uncorrelated attributes and correlated k-means clusters for a cell.

    With ``correlated.cluster_D`` set, k-means runs once on the vectors at that dimension and
    the SAME cluster ids/centroids serve every D, so a "positive S=0.001" predicate admits
    exactly the same passages at D=128 and D=1536. Matryoshka prefixes share row ids, so this is
    well defined. Masks must then be built from queries at ``cluster_D`` (see ``query_D``).
    """
    from annstudy.filters import generate_correlated_attributes, generate_uncorrelated_attributes

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fs = filter_settings(filters_cfg)
    out = {"settings": fs}

    n_attr, a_seed = fs["n_attributes"], fs["attr_seed"]
    attrs_path = cache_dir / f"attrs_N{N}_A{n_attr}_s{a_seed}.npy"  # D-independent
    attrs = _load_cached(attrs_path)
    if attrs is not None and (len(attrs) != N or attrs.max(initial=0) >= n_attr):
        logger.warning(f"Attribute cache {attrs_path} inconsistent with N/n_attributes; "
                       f"recomputing.")
        attrs = None
    if attrs is None:
        attrs = generate_uncorrelated_attributes(N, n_attr, a_seed)
        _save_atomic(attrs_path, attrs)
    out["uncorrelated"] = {"attrs": attrs, "n_attributes": n_attr}

    if filters_cfg.get("correlated"):
        k, c_seed = fs["n_clusters"], fs["cluster_seed"]
        cD = fs["cluster_D"] or D
        cl_path = cache_dir / f"clusters_N{N}_D{cD}_k{k}_s{c_seed}.npy"
        ce_path = cache_dir / f"centroids_N{N}_D{cD}_k{k}_s{c_seed}.npy"
        cluster_ids, centroids = _load_cached(cl_path), _load_cached(ce_path)
        if cluster_ids is not None and centroids is not None and (
                len(cluster_ids) != N or centroids.shape != (k, cD)):
            logger.warning(f"Cluster cache {cl_path} inconsistent with N/k/D; recomputing.")
            cluster_ids = centroids = None
        if cluster_ids is None or centroids is None:
            if cD == D:
                vecs = passages
            else:
                vecs = cluster_vectors if cluster_vectors is not None \
                    else _load_cluster_vectors(N, cD)
            logger.info(f"k-means (k={k}) on N={N} D={cD} ...")
            cluster_ids, centroids = generate_correlated_attributes(vecs, k, c_seed)
            _save_atomic(cl_path, cluster_ids)
            _save_atomic(ce_path, centroids)
        out["correlated"] = {"cluster_ids": cluster_ids, "centroids": centroids,
                             "n_clusters": k, "query_D": cD}
    return out


def make_mask_fn(prep, correlation, target_S, query_vecs, seed: int = 42):
    """Return mask_fn(qi)->bool mask over N, positional in query_vecs. 'none' = uncorrelated.

    The negative-correlation draw is seeded per query (``seed + qi``) so each query gets a
    different random far cluster while the whole run stays reproducible.
    """
    from annstudy.filters import make_correlated_mask, make_uncorrelated_mask

    if correlation == "none":
        attrs = prep["uncorrelated"]["attrs"]
        n_attr = prep["uncorrelated"]["n_attributes"]
        mask = make_uncorrelated_mask(attrs, target_S, n_attr)  # query-independent
        return lambda qi: mask
    cl = prep["correlated"]
    return lambda qi: make_correlated_mask(
        cl["cluster_ids"], cl["centroids"], query_vecs[qi], target_S, correlation,
        cl["n_clusters"], seed=seed + qi)


def make_matching_provider(prep, correlation, target_S, query_vecs, N, seed: int = 42):
    """Return (get(qi) -> matching_idx, is_static).

    Yields matching INDICES lazily rather than materialising one boolean mask per query:
    at N=1M a mask per query for 10k queries would be ~10 GB. Uncorrelated predicates are
    query-independent, so that subset is computed once and shared.
    """
    mask_fn = make_mask_fn(prep, correlation, target_S, query_vecs, seed=seed)
    if correlation == "none":
        idx = np.where(mask_fn(0))[0]
        return (lambda qi: idx), True
    return (lambda qi: np.where(mask_fn(qi))[0]), False


# --------------------------------------------------------------- per-strategy measure

def _stats_reset(index_type):
    """Zero FAISS's global search counters before one query (single-threaded, so exact)."""
    import faiss
    if index_type == "hnsw":
        faiss.cvar.hnsw_stats.reset()
    elif index_type in ("ivf_flat", "ivf_pq"):
        faiss.cvar.indexIVF_stats.reset()


def _stats_read(index_type, n_match):
    """(distance computations, graph hops or IVF lists visited) for the query just run.

    For the exact scan the work is known in closed form: one distance per matching vector.
    These are hardware-independent measures of search effort, which is what lets a latency
    difference across D be attributed to work done rather than to the memory system.
    """
    import faiss
    if index_type == "hnsw":
        s = faiss.cvar.hnsw_stats
        return int(s.ndis), int(s.nhops)
    if index_type in ("ivf_flat", "ivf_pq"):
        s = faiss.cvar.indexIVF_stats
        return int(s.ndis), int(s.nlist)
    if index_type == "exact":
        return int(n_match), 0
    return -1, -1


def _measure(strategy, index, index_type, passages, queries, get_matching, k, search_param,
             n_passes, safety, N, static: bool = False, sq_norms=None):
    """Run one strategy over all queries.

    Returns (ret_ids [n,k], per_query_ns [n], n_matching [n], extra) where ``extra`` holds
    per-query ``struct_ns`` (time to build the strategy's filter structure, untimed in
    ``per_query_ns``), ``ndis`` (distance computations) and ``nvisit`` (HNSW hops / IVF lists).

    What is inside the timed region, and why. Evaluating the predicate (``get_matching``) is
    always excluded: it is the database's filtering work, identical for every strategy. The
    strategy's filter structure (bitmap selector / boolean keep-mask) is ALSO excluded, for
    static and correlated predicates alike, and its cost is recorded separately in
    ``struct_ns``. The original Stage C built the per-query selector inside the timed region
    for correlated predicates only: an O(N), D-independent cost that is a larger share of a
    cheap low-D search, which biased S*(low D) upward — the direction of the H2 claim. Keeping
    the timed region to "the search call itself" for every strategy removes that confound;
    the separate column lets a reader add it back.

    Passes are OUTER loops over the whole query set (as in ``metrics.measure_latency`` for
    Stages A/B), so the per-query median is not taken over three back-to-back, cache-hot
    repeats of the same query.
    """
    n = len(queries)
    ret = np.full((n, k), -1, dtype=np.int64)
    times = np.zeros((max(1, n_passes), n), dtype=np.int64)
    struct = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    ndis = np.full(n, -1, dtype=np.int64)
    nvisit = np.full(n, -1, dtype=np.int64)

    needs_selector = strategy == "in_index_filtered"
    needs_mask = strategy == "post_filter_overfetch"
    stats_type = "exact" if strategy == "pre_filter_exact" else index_type

    if needs_mask:  # applying the sweep setting is index state, not per-query work
        from annstudy.indexes import set_search_params
        set_search_params(index, index_type, search_param)

    shared = None
    if static and (needs_selector or needs_mask):
        t0 = time.perf_counter_ns()
        idx0 = get_matching(0)
        sel, bitmap, mask = make_id_selector(idx0, N)
        shared = (sel, bitmap, mask,
                  search_params_filtered(index_type, sel, search_param) if needs_selector
                  else None)
        struct[:] = time.perf_counter_ns() - t0  # amortised: built once for every query

    # Correlated predicates differ per query. Cache the matching sets across passes when that
    # fits a modest budget (int32 ids); otherwise recompute them, untimed, on every pass.
    matching_cache = None
    # int32 ids for every predicate type, so the exact scan's gather costs the same per id
    # whether the subset is static or per-query.
    static_idx = get_matching(0).astype(np.int32) if static else None
    if not static and n > 0:
        est_bytes = n * len(get_matching(0)) * 4
        if est_bytes <= MATCHING_CACHE_BYTES:
            matching_cache = [get_matching(qi).astype(np.int32) for qi in range(n)]

    for p in range(max(1, n_passes)):
        for qi in range(n):
            if static:
                matching_idx = static_idx
            elif matching_cache is not None:
                matching_idx = matching_cache[qi]
            else:
                matching_idx = get_matching(qi)
            sizes[qi] = len(matching_idx)
            S_q = len(matching_idx) / N

            # --- untimed: build this query's filter structure
            sel = bitmap = keep = params = None
            if needs_selector or needs_mask:
                if shared is not None:
                    params, keep = shared[3], shared[2]
                else:
                    t_s = time.perf_counter_ns()
                    sel, bitmap, keep = make_id_selector(matching_idx, N)
                    if needs_selector:
                        params = search_params_filtered(index_type, sel, search_param)
                    if p == 0:
                        struct[qi] = time.perf_counter_ns() - t_s

            if p == 0:
                _stats_reset(stats_type)

            # --- timed: the search call itself
            t0 = time.perf_counter_ns()
            if strategy == "pre_filter_exact":
                ids = strategy_pre_filter_exact(passages, queries[qi], matching_idx, k,
                                                sq_norms=sq_norms)
            elif needs_selector:
                ids = strategy_in_index_filtered(index, index_type, queries[qi], k, params)
            else:  # post_filter_overfetch
                ids = strategy_post_filter_overfetch(index, index_type, queries[qi], keep, k,
                                                     search_param, safety, S_q, N)
            times[p, qi] = time.perf_counter_ns() - t0

            if p == 0:
                ndis[qi], nvisit[qi] = _stats_read(stats_type, len(matching_idx))
                ret[qi] = ids
            del sel, bitmap  # the selector borrows the bitmap; drop them together
    per_query = np.median(times, axis=0).astype(np.int64)
    return ret, per_query, sizes, {"struct_ns": struct, "ndis": ndis, "nvisit": nvisit}


def filtered_gt(passages, queries, mask_fn, k, cache_path):
    """Exact filtered top-k GT (cached). Returns (ids, dists)."""
    from annstudy.filters import compute_filtered_ground_truth

    cached = _load_gt_cache(cache_path, len(queries))
    if cached is not None:
        return cached
    ids, dists = compute_filtered_ground_truth(passages, queries, mask_fn, k)
    _savez_atomic(cache_path, ids=ids, distances=dists)
    return ids, dists


def filtered_gt_static(passages, queries, matching_idx, k, cache_path, threads: int = 0):
    """Exact filtered top-k GT for a query-INDEPENDENT subset, via a batched FAISS flat scan.

    Ground truth is reference data, not a measured quantity, so it may use every core. This is
    far faster than the per-query numpy loop: one flat index over the subset answers all
    queries in a single batched call.
    """
    import faiss

    cached = _load_gt_cache(cache_path, len(queries))
    if cached is not None:
        return cached

    n_q = len(queries)
    ids = np.full((n_q, k), -1, dtype=np.int64)
    dists = np.full((n_q, k), np.inf, dtype=np.float32)
    # GT may use every core, but the thread count is process-global: the original code set it
    # here and never restored it, so every later (measured) search in the shard ran with
    # `threads` OMP threads on a single pinned core. Always restore the caller's setting.
    prev_threads = faiss.omp_get_max_threads()
    try:
        if threads:
            faiss.omp_set_num_threads(threads)
        if len(matching_idx) > 0:
            sub = np.ascontiguousarray(passages[matching_idx])
            flat = faiss.IndexFlatL2(sub.shape[1])
            flat.add(sub)
            kk = min(k, len(matching_idx))
            d_sub, i_sub = flat.search(np.ascontiguousarray(queries), kk)
            # map subset-local ids back to global passage ids
            ids[:, :kk] = np.where(i_sub >= 0, matching_idx[np.clip(i_sub, 0, None)], -1)
            dists[:, :kk] = d_sub
    finally:
        faiss.omp_set_num_threads(prev_threads)

    _savez_atomic(cache_path, ids=ids, distances=dists)
    return ids, dists


def filtered_recall(ret_ids, gt_dists, passages, queries, k, n_matching):
    """Recall@k against filtered ground truth, with the denominator capped by subset size.

    When a predicate admits fewer than k vectors, a perfect answer still returns only
    ``n_matching`` ids, so dividing by k would understate recall (and would make the
    "pre-filter exact scan has recall 1.0" assertion fail by construction). Matching uses the
    plan's distance tolerance so equally-good neighbours are not penalised.
    """
    n_q = ret_ids.shape[0]
    recalls = np.zeros(n_q, dtype=np.float64)
    for qi in range(n_q):
        denom = int(min(k, n_matching[qi]))
        if denom <= 0:
            # An empty predicate is a broken filter, not a perfect answer. This used to score
            # 1.0, which is how a stale cluster cache could masquerade as a success.
            raise ValueError(f"query {qi}: predicate admits no vectors; recall is undefined")
        threshold = float(gt_dists[qi, denom - 1]) * (1.0 + 1e-5)
        cand = ret_ids[qi][ret_ids[qi] >= 0]
        if cand.size == 0:
            continue
        d = np.sum((passages[cand].astype(np.float64) - queries[qi].astype(np.float64)) ** 2,
                   axis=1)
        recalls[qi] = float(np.sum(d <= threshold)) / denom
    return recalls


# -------------------------------------------------------------------- job enumeration

def load_stage_c_config(path: str = "configs/stage_c.yaml") -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def enumerate_filter_jobs(cfg: dict) -> list[dict]:
    """Expand the Stage C config into one job per (cell, index, strategy, filter, safety).

    Correlated filters run only at the selectivities the plan lists for them, and the
    over-fetch strategy multiplies out the safety factors.
    """
    cells_N = cfg["cells"]["N"]
    cells_D = cfg["cells"]["D"]
    index_types = list(cfg["indexes"])
    strategies = cfg["strategies"]
    filters_cfg = cfg["filters"]
    safety_factors = cfg.get("post_filter", {}).get("safety_factors", [1.5, 3])

    filter_variants: list[tuple[str, float]] = [
        ("none", float(S))
        for S in filters_cfg.get("uncorrelated", {}).get("selectivities", [])
    ]
    corr = filters_cfg.get("correlated", {})
    for ctype in corr.get("types", []):
        for S in corr.get("selectivities", []):
            filter_variants.append((str(ctype), float(S)))
    # Optional anti-correlated levels with their own S list (e.g. one exhaustive-IVF check),
    # so the negative case need not multiply out the whole positive grid.
    for S in (cfg.get("negative_check") or {}).get("selectivities", []):
        if ("negative", float(S)) not in filter_variants:
            filter_variants.append(("negative", float(S)))

    fp = filter_fingerprint(filters_cfg)
    repeats = list(range(int(cfg.get("repeat_builds", 1)))) if cfg.get("build") else [None]

    jobs = []
    for N in cells_N:
        for D in cells_D:
            for index_type in index_types:
                for strategy in strategies:
                    for correlation, target_S in filter_variants:
                        # Exact scan over the matching subset does not use the index, so it is
                        # enumerated once per cell rather than once per index type.
                        if strategy == "pre_filter_exact" and index_type != index_types[0]:
                            continue
                        safeties = (safety_factors if strategy == "post_filter_overfetch"
                                    else [None])
                        for safety in safeties:
                            for rep_id in repeats:
                                job = {
                                    "N": int(N), "D": int(D),
                                    "index_type": ("exact" if strategy == "pre_filter_exact"
                                                   else index_type),
                                    "strategy": strategy,
                                    "correlation": correlation,
                                    "target_S": target_S,
                                    "safety": safety,
                                    "k": cfg.get("k", 10),
                                    "filter_fp": fp,
                                }
                                if rep_id is not None:
                                    job["build_repeat"] = rep_id
                                jobs.append(job)
    return jobs


def job_key(job: dict) -> str:
    """Stable short hash identifying a job, used for resumable per-job parquet files.

    Includes the filter fingerprint (and build repeat, when a stage builds its own indexes):
    without it, a rerun with a different attribute grid or clustering found the old per-job
    files and skipped every overlapping (S, correlation) job as already done.
    """
    from annstudy.runner import config_hash
    keys = ("N", "D", "index_type", "strategy", "correlation", "target_S", "safety", "k")
    payload = {k: job[k] for k in keys}
    for extra in ("filter_fp", "build_repeat"):
        if job.get(extra) is not None:
            payload[extra] = job[extra]
    return config_hash(payload)


def resolve_index_path(
    N: int,
    D: int,
    index_type: str,
    results_dir: str = "results/raw",
    index_dirs: list[str] | None = None,
) -> str | None:
    """Find a serialized index for this cell, preferring a tuned Stage B winner.

    Stage C applies the predicate at search time, so it reuses the indexes Stage B kept
    (``--keep-indexes``) instead of rebuilding.
    """
    if index_dirs is None:
        index_dirs = ["data/indexes/stage_b", "data/indexes/stage_a", "data/indexes/stage_pilot"]

    available: dict[str, str] = {}
    for d in index_dirs:
        for p in Path(d).glob("idx_*.faiss") if Path(d).exists() else []:
            available[p.stem.replace("idx_", "")] = str(p)
    if not available:
        return None

    builds_dir = Path(results_dir) / "builds"
    builds = [pd.read_parquet(f) for f in builds_dir.glob("build_*.parquet")] \
        if builds_dir.exists() else []
    if builds:
        bdf = pd.concat(builds, ignore_index=True)
        match = bdf[(bdf["N"] == N) & (bdf["D"] == D) & (bdf["index_type"] == index_type)]

        tuned_path = Path("results/tuned/tuned.parquet")
        if tuned_path.exists() and not match.empty:
            tuned = pd.read_parquet(tuned_path)
            t = tuned[(tuned["N"] == N) & (tuned["D"] == D)
                      & (tuned["index_type"] == index_type)]
            if "reachable" in t.columns:
                t = t[t["reachable"].fillna(False)]
            for rid in t.get("best_build_run_id", pd.Series(dtype=str)).dropna():
                if rid in available:
                    return available[rid]

        for rid in match["run_id"]:
            if rid in available:
                return available[rid]
    return None


# ------------------------------------------------------------------------ stage C driver

def job_filtered_gt(cfg, N, D, correlation, target_S, passages, prep, tuning_queries,
                    test_queries, f_tun=None, f_test=None, gt_threads=0, k=10):
    """Filtered ground truth (tuning, test) for one predicate: exact top-k over its subset.

    Shared by the measuring jobs and the campaign's precompute phase, so both use the same
    cache names. The tag carries the filter fingerprint, or a config change would reuse
    stale GT.
    """
    gt_k = max(k, 100)
    gt_dir = Path(cfg.get("gt_dir", FILTERED_GT)) / f"N{N}_D{D}"
    tag = f"{correlation}_S{target_S:g}_f{filter_fingerprint(cfg.get('filters', {}))}"
    corr_seed = cfg.get("filters", {}).get("correlated", {}).get("seed", 42)
    f_tun = tuning_queries if f_tun is None else f_tun
    f_test = test_queries if f_test is None else f_test
    if correlation == "none":
        get_tun, _ = make_matching_provider(prep, correlation, target_S, f_tun, N)
        get_test, _ = make_matching_provider(prep, correlation, target_S, f_test, N)
        return (filtered_gt_static(passages, tuning_queries, get_tun(0), gt_k,
                                   gt_dir / f"{tag}_tuning.npz", threads=gt_threads),
                filtered_gt_static(passages, test_queries, get_test(0), gt_k,
                                   gt_dir / f"{tag}_test.npz", threads=gt_threads))
    return (filtered_gt(passages, tuning_queries,
                        make_mask_fn(prep, correlation, target_S, f_tun, seed=corr_seed),
                        gt_k, gt_dir / f"{tag}_tuning.npz"),
            filtered_gt(passages, test_queries,
                        make_mask_fn(prep, correlation, target_S, f_test,
                                     seed=corr_seed + 100_000),
                        gt_k, gt_dir / f"{tag}_test.npz"))


def precompute_filtered_gt(config_path, gt_threads=0):
    """Compute every filtered-GT cache a stage needs BEFORE measurement starts.

    In the original Stage C, GT was computed inside the measuring shards, so one shard's
    multi-threaded GT ran while its neighbours were timing single-threaded searches.
    """
    from annstudy.data import load_query_split

    cfg = load_stage_c_config(config_path)
    split = load_query_split(QUERY_SPLIT_PATH_DEFAULT)
    cells = _CellCache(cfg, len(split["tuning"]), len(split["test"]), None, None)
    variants = sorted({(j["N"], j["D"], j["correlation"], j["target_S"])
                       for j in enumerate_filter_jobs(cfg)})
    for N, D, corr, S in variants:
        c = cells.get(N, D)
        logger.info(f"GT N={N} D={D} {corr} S={S:g}")
        job_filtered_gt(cfg, N, D, corr, S, c["passages"], c["prep"], c["tuning"], c["test"],
                        c["f_tuning"], c["f_test"], gt_threads=gt_threads,
                        k=cfg.get("k", 10))


def run_filter_job(
    job: dict,
    cfg: dict,
    passages: np.ndarray,
    prep: dict,
    tuning_queries: np.ndarray,
    test_queries: np.ndarray,
    index,
    results_dir: str = "results/raw",
    gt_threads: int = 0,
    sq_norms=None,
    filter_tuning_queries: np.ndarray | None = None,
    filter_test_queries: np.ndarray | None = None,
) -> list[dict]:
    """Measure one Stage C job and return its ``filter_runs`` rows (plan section 5).

    Protocol: sweep the search parameter on tuning queries, take the cheapest setting reaching
    each R, re-measure that setting on the test queries, and additionally record recall at one
    fixed mid-range effort to expose silent recall loss.

    ``filter_*_queries`` are the same queries at the clustering dimension (``cluster_D``); they
    decide WHICH vectors a correlated predicate admits, while ``tuning/test_queries`` (at the
    cell's D) are what is searched. They default to the search queries (original behaviour).
    """
    from annstudy.hardware import snapshot
    from annstudy.indexes import get_search_grid
    from annstudy.metrics import bootstrap_latency_percentile, bootstrap_recall, latency_stats

    N, D, k = job["N"], job["D"], job["k"]
    strategy, correlation, target_S = job["strategy"], job["correlation"], job["target_S"]
    index_type = job["index_type"]
    measurement = cfg.get("measurement", {})
    n_passes = measurement.get("latency_passes", 3)
    n_resamples = measurement.get("bootstrap_resamples", 1000)
    recall_targets = cfg.get("recall_targets", [0.90, 0.95, 0.99])
    corr_seed = cfg.get("filters", {}).get("correlated", {}).get("seed", 42)

    hw = snapshot()
    fp = filter_fingerprint(cfg.get("filters", {}))
    f_tun = tuning_queries if filter_tuning_queries is None else filter_tuning_queries
    f_test = test_queries if filter_test_queries is None else filter_test_queries
    if len(f_tun) != len(tuning_queries) or len(f_test) != len(test_queries):
        raise ValueError("filter queries must be positionally aligned with search queries")

    def provider(fqueries, offset):
        return make_matching_provider(prep, correlation, target_S, fqueries, N,
                                      seed=corr_seed + offset)

    get_tun, static = provider(f_tun, 0)
    get_test, _ = provider(f_test, 100_000)  # distinct seed block from tuning

    gt_tun, gt_test = job_filtered_gt(cfg, N, D, correlation, target_S, passages, prep,
                                      tuning_queries, test_queries, f_tun, f_test,
                                      gt_threads=gt_threads, k=k)

    # Search grid for this strategy. Exact scan has no search knob.
    if strategy == "pre_filter_exact":
        grid = [{}]
    else:
        nlist = None
        if index is not None and index_type in ("ivf_flat", "ivf_pq"):
            nlist = int(getattr(index, "nlist", 0)) or None
        grid = get_search_grid(index_type, nlist=nlist, k=k)
        # Anti-correlated IVF: the default sweep caps nprobe at 512 while nlist >= 1414, so
        # "approximate search fails" could just mean "the scan was never exhaustive". An
        # nprobe = nlist point separates the two.
        if (cfg.get("exhaustive_nprobe_for_negative") and correlation == "negative"
                and nlist and index_type in ("ivf_flat", "ivf_pq")
                and not any(g.get("nprobe") == nlist for g in grid)):
            grid = grid + [{"nprobe": int(nlist)}]

    def run(queries, get_matching, gt, search_param):
        ret, times, sizes, extra = _measure(
            strategy, index, index_type, passages, queries, get_matching, k, search_param,
            n_passes, job["safety"], N, static=static, sq_norms=sq_norms,
        )
        rec = filtered_recall(ret, gt[1], passages, queries, k, sizes)
        return rec, times, sizes, extra

    # Warm-up (untimed, discarded): the plan's warm-up was configured but never applied in
    # Stage C, so the first sweep setting paid cold-cache and page-fault costs.
    n_warm = int(measurement.get("warmup_queries", 0) or 0)
    if n_warm > 0 and len(tuning_queries) > 0:
        w = min(n_warm, len(tuning_queries))
        _measure(strategy, index, index_type, passages, tuning_queries[:w], get_tun, k,
                 grid[0], 1, job["safety"], N, static=static, sq_norms=sq_norms)

    # Different recall targets frequently select the SAME cheapest setting (always so when the
    # grid has one setting, as for the exact scan). Measuring it once per job keeps the reported
    # latency for those rows identical instead of differing by run-to-run noise, and cuts the
    # test-measurement cost, which dominates Stage C.
    _test_cache: dict[str, tuple] = {}

    def run_test(search_param):
        key = json_safe(search_param)
        if key not in _test_cache:
            _test_cache[key] = run(test_queries, get_test, gt_test, search_param)
        return _test_cache[key]

    # --- sweep on tuning queries
    sweep = []
    for sp in grid:
        rec, times, sizes, extra = run(tuning_queries, get_tun, gt_tun, sp)
        stats = latency_stats(times)
        _, rec_lo, _ = bootstrap_recall(rec, n_resamples=min(n_resamples, 500))
        sweep.append({
            "search_param": sp, "recall": float(np.mean(rec)), "recall_lo": float(rec_lo),
            "lat_p95_us": stats["lat_p95_us"], "measured_S": float(np.mean(sizes)) / N,
            "ndis_mean": float(np.mean(extra["ndis"])),
        })
        logger.info(f"    tuning {sp}: recall={np.mean(rec):.4f} "
                    f"p95={stats['lat_p95_us']:.0f}us")

    base = {
        "run_id": uuid.uuid4().hex,
        "job_key": job_key(job),
        "git_commit": hw["git_commit"], "hardware_id": hw["hardware_id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": corr_seed,
        "index_type": index_type, "strategy": strategy,
        "impl": "faiss" if strategy != "pre_filter_exact" else "numpy",
        "N": N, "D": D, "k": k,
        "correlation": correlation, "target_S": target_S,
        "overfetch_factor": job["safety"],
        "n_tuning_queries": len(tuning_queries), "n_test_queries": len(test_queries),
        "threads": 1,
        "filter_fp": fp,
        "cluster_D": prep.get("correlated", {}).get("query_D") if correlation != "none" else None,
        "n_attributes": prep.get("uncorrelated", {}).get("n_attributes"),
        "n_clusters": prep.get("correlated", {}).get("n_clusters"),
        "timing_protocol": "v2_struct_untimed_outer_passes",
        "build_repeat": job.get("build_repeat"),
        "base_index_run_id": (Path(job["base_index_path"]).stem.replace("idx_", "")
                              if job.get("base_index_path") else None),
    }

    # --- fixed mid-range effort, to expose silent recall loss at small S
    rows: list[dict] = []
    mid = sweep[len(sweep) // 2]
    rec_fixed, times_fixed, sizes_fixed, _ = run_test(mid["search_param"])
    fixed_stats = latency_stats(times_fixed)
    fixed = {
        "fixed_effort_param": json_safe(mid["search_param"]),
        "fixed_effort_recall": float(np.mean(rec_fixed)),
        "fixed_effort_lat_p95_us": fixed_stats["lat_p95_us"],
    }

    # --- cheapest setting reaching each R, re-measured on the test queries
    for R in recall_targets:
        # measurement.selection = "lower_ci": the tuning recall's bootstrap lower bound must
        # clear R, so fewer winners miss R on the held-out queries (see tune.meets_target).
        key = "recall_lo" if measurement.get("selection") == "lower_ci" else "recall"
        ok = [s for s in sweep if s[key] >= R]
        row = dict(base, R=float(R), **fixed)
        row["run_id"] = uuid.uuid4().hex
        if not ok:
            best_t = max(sweep, key=lambda s: s["recall"])
            rows.append(dict(
                row, reachable=False,
                search_param=json_safe(best_t["search_param"]),
                recall_tuning=best_t["recall"], best_tuning_recall=best_t["recall"],
                measured_S_mean=best_t["measured_S"],
            ))
            logger.info(f"    R={R}: UNREACHABLE (best tuning recall {best_t['recall']:.4f})")
            continue

        best = min(ok, key=lambda s: s["lat_p95_us"])
        rec, times, sizes, extra = run_test(best["search_param"])
        stats = latency_stats(times)
        _, lo, hi = bootstrap_latency_percentile(times, percentile=95, n_resamples=n_resamples)
        measured_S = float(np.mean(sizes)) / N
        rows.append(dict(
            row, reachable=True,
            search_param=json_safe(best["search_param"]),
            recall_tuning=best["recall"],
            recall_test=float(np.mean(rec)),
            # Reachability is decided on tuning queries; this records whether the held-out
            # queries actually achieve R. The crossover must only use rows where they do.
            meets_R_on_test=bool(np.mean(rec) >= R),
            ndis_mean=float(np.mean(extra["ndis"])),
            ndis_p95=float(np.percentile(extra["ndis"], 95)),
            nvisit_mean=float(np.mean(extra["nvisit"])),
            filter_struct_us_p50=float(np.median(extra["struct_ns"])) / 1000.0,
            lat_p95_us=stats["lat_p95_us"],
            lat_p50_us=stats["lat_p50_us"],
            lat_mean_us=stats["lat_mean_us"],
            lat_p95_ci_lo=lo, lat_p95_ci_hi=hi,
            qps=float("nan"),  # throughput is a T-thread quantity; not measured here
            measured_S_mean=measured_S,
            measured_S_min=float(np.min(sizes)) / N,
            measured_S_max=float(np.max(sizes)) / N,
        ))
        logger.info(f"    R={R}: {best['search_param']} recall_test={np.mean(rec):.4f} "
                    f"p95={stats['lat_p95_us']:.0f}us measured_S={measured_S:.4g}")

    _check_acceptance(rows, job)
    return rows


def json_safe(obj):
    """Compact JSON string for a search-parameter dict, so it fits one parquet column."""
    import json
    return json.dumps(obj, sort_keys=True)


def _check_acceptance(rows: list[dict], job: dict) -> None:
    """Assert the Stage C acceptance checks from EXPERIMENT_PLAN, logging any violation.

    These are the checks that catch a broken filter harness: exact scan must be exact, and the
    predicate must actually have the selectivity it claims.
    """
    for row in rows:
        if not row.get("reachable"):
            continue
        if job["strategy"] == "pre_filter_exact":
            rec = row.get("recall_test")
            if rec is not None and rec < 0.999:
                logger.error(
                    f"ACCEPTANCE FAILURE: pre_filter_exact recall {rec:.6f} < 1.0 at "
                    f"S={job['target_S']} ({job['correlation']}). The exact scan must be exact; "
                    f"suspect the recall denominator or the mask."
                )
        measured = row.get("measured_S_mean")
        target = job["target_S"]
        if measured is not None and target > 0 and abs(measured - target) / target > 0.20:
            logger.warning(
                f"Measured selectivity {measured:.4g} is >20% from target {target:g} "
                f"({job['correlation']}) — expected for correlated unions, "
                f"flagged for the stage report."
            )


def _blas_single_thread():
    """Limit numpy's BLAS pool to one thread for the whole measured run.

    The exact-scan baseline is a BLAS matvec. Shards were pinned with ``taskset`` to one core
    but OpenBLAS/MKL threads were never limited (no ``OMP_NUM_THREADS`` in ``stage_c.sh``), so
    the baseline could oversubscribe its core. Returns the context so the caller can keep it.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception as e:  # pragma: no cover - threadpoolctl ships with scikit-learn
        logger.warning(f"Could not limit BLAS threads ({e}); set OMP_NUM_THREADS=1.")
        return None


def stage_index_dir(cfg: dict) -> Path:
    stage = str(cfg.get("stage", "c")).lower()
    return Path(cfg.get("build", {}).get("index_dir", f"data/indexes/stage_{stage}"))


def build_or_load_index(cfg, N, D, index_type, repeat, passages, build_threads=0):
    """Index for a stage that builds its own (``build:`` section), cached on disk per repeat.

    Repeats exist so the crossover's uncertainty can include build-to-build variance. IVF
    repeats differ by their k-means training sample, HNSW repeats by their level-generator
    seed (both ``seed = repeat``; repeat 0 reproduces the library defaults). Returns
    (index, path).
    """
    import json
    import os

    import faiss

    from annstudy.indexes import build_hnsw, build_ivf_flat, compute_nlist

    bcfg = cfg["build"][index_type]
    out_dir = stage_index_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{index_type}_N{N}_D{D}_r{repeat}.faiss"
    if path.exists():
        return faiss.read_index(str(path)), str(path)

    prev = faiss.omp_get_max_threads()
    try:
        if build_threads:
            faiss.omp_set_num_threads(build_threads)
        vecs = np.ascontiguousarray(passages)
        if index_type == "hnsw":
            index, meta = build_hnsw(vecs, D, int(bcfg.get("M", 32)),
                                     int(bcfg.get("ef_construction", 200)), seed=int(repeat))
        elif index_type == "ivf_flat":
            index, meta = build_ivf_flat(vecs, D, compute_nlist(N, int(bcfg.get("c", 4))),
                                         seed=int(repeat))
        else:
            raise ValueError(f"stage-local build unsupported for {index_type}")
    finally:
        faiss.omp_set_num_threads(prev)

    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)  # atomic publish; concurrent builders of the same spec are harmless
    meta = dict(meta, N=N, D=D, repeat=int(repeat), threads=build_threads or prev,
                timestamp=datetime.now(UTC).isoformat())
    path.with_suffix(".json").write_text(json.dumps(meta, default=str, indent=1))
    logger.info(f"  built {index_type} N={N} D={D} r={repeat} in {meta['build_seconds']:.0f}s")
    return index, str(path)


def interleave_jobs(jobs: list[dict]) -> list[dict]:
    """Order jobs so consecutive jobs cycle through D (and builds) rather than finishing a D.

    The original Stage C ran one cell at a time, so D was confounded with session and with the
    number of concurrently busy cores. Sorting D LAST makes every block of len(D) consecutive
    jobs cover every dimension, so slow drift averages across D instead of aliasing onto it.
    """
    return sorted(jobs, key=lambda j: (
        j.get("build_repeat") or 0, j["correlation"], j["target_S"], j["strategy"],
        j["index_type"], j.get("safety") or 0, j["D"], j["N"]))


class _CellCache:
    """Per-(N, D) heavy state, loaded lazily. Passages stay memory-mapped (shared page cache)."""

    def __init__(self, cfg, n_tun, n_test, max_tq, max_teq):
        self.cfg, self.n_tun, self.n_test = cfg, n_tun, n_test
        self.max_tq, self.max_teq = max_tq, max_teq
        self._cells: dict = {}

    def _queries(self, D):
        from annstudy.coordinate import DEFAULT_N_QUERIES, queries_path
        nq = self.cfg.get("n_queries", DEFAULT_N_QUERIES)
        all_q = np.memmap(queries_path(D), dtype=np.float32, mode="r", shape=(nq, D))
        tun = np.ascontiguousarray(all_q[:self.n_tun][:self.max_tq])
        test = np.ascontiguousarray(all_q[self.n_tun:self.n_tun + self.n_test][:self.max_teq])
        return tun, test

    def get(self, N, D):
        from annstudy.coordinate import passages_path
        if (N, D) not in self._cells:
            passages = np.asarray(np.memmap(passages_path(D), dtype=np.float32, mode="r",
                                            shape=(N, D)))
            tun, test = self._queries(D)
            prep = prepare_filters(N, D, passages, self.cfg["filters"],
                                   cache_dir=self.cfg.get("filter_cache_dir", FILTER_CACHE))
            cD = prep.get("correlated", {}).get("query_D", D)
            f_tun, f_test = (tun, test) if cD == D else self._queries(cD)
            self._cells[(N, D)] = {
                "passages": passages, "tuning": tun, "test": test,
                "f_tuning": f_tun, "f_test": f_test, "prep": prep,
                # Precomputed once per cell so the exact-scan baseline is a single BLAS matvec.
                "sq": passage_sq_norms(passages),
            }
        return self._cells[(N, D)]

    def drop(self, N, D):
        self._cells.pop((N, D), None)


def run_canary(cfg, cells, index_getter, results_dir, shard, jobs_done):
    """Measure a fixed unfiltered configuration and append one timestamped row.

    Between-session drift (~1.1x on the VM) was documented but never tracked. A canary run at
    regular intervals inside the campaign measures it directly, per shard, so any residual
    drift can be modelled rather than assumed away.
    """
    from annstudy.hardware import snapshot
    from annstudy.indexes import set_search_params
    from annstudy.metrics import latency_stats, measure_latency

    c = cfg["canary"]
    N, D, it = int(c["N"]), int(c["D"]), c.get("index_type", "hnsw")
    cell = cells.get(N, D)
    index, _path = index_getter(N, D, it, 0, cell["passages"])
    set_search_params(index, it, c["search_param"])
    q = cell["test"][: int(c.get("n_queries", 1000))]
    stats = latency_stats(measure_latency(index, q, k=cfg.get("k", 10), n_passes=1))
    hw = snapshot()
    row = dict(stats, N=N, D=D, index_type=it, search_param=json_safe(c["search_param"]),
               shard=shard, jobs_done=jobs_done, hardware_id=hw["hardware_id"],
               git_commit=hw["git_commit"], timestamp=datetime.now(UTC).isoformat(),
               stage=cfg.get("stage"))
    out = Path(results_dir) / "canary"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out / f"canary_{uuid.uuid4().hex}.parquet", index=False)
    logger.info(f"  canary p50={stats['lat_p50_us']:.0f}us p95={stats['lat_p95_us']:.0f}us")


def run_stage_c(
    config_path: str = "configs/stage_c.yaml",
    results_dir: str = "results/raw",
    shard: int = 0,
    num_shards: int = 1,
    max_jobs: int | None = None,
    max_tuning_queries: int | None = None,
    max_test_queries: int | None = None,
    gt_threads: int = 0,
    dry_run: bool = False,
    index_source_dir: str = "results/raw",
    only_N: int | None = None,
    only_D: int | None = None,
    build_threads: int = 0,
) -> int:
    """Run Stage C jobs, writing one parquet per job. Resumable and shardable.

    Latency must be single-threaded (plan section 3), so parallelism comes from running
    several shards pinned to different cores:
      ``taskset -c 3 annstudy-run-filters --shard 3 --num-shards 16``

    Two modes. Legacy (no ``build:`` section): reuse Stage B's serialized indexes, one cell at a
    time. Self-contained (``build:`` present, e.g. ``stage_c_h2.yaml``): build ``repeat_builds``
    indexes per cell, and with ``interleave_dimensions`` run jobs in D-interleaved order so D is
    not confounded with session or machine load.
    """
    import faiss

    from annstudy.data import load_query_split

    cfg = load_stage_c_config(config_path)
    jobs = enumerate_filter_jobs(cfg)
    # Cell filters exist because each shard holds the cell's corpus AND index: 13.1 GB at
    # N=1M/D=1536, so only ~3 shards fit in memory there against ~16 at the small cells.
    if only_N is not None:
        jobs = [j for j in jobs if j["N"] == only_N]
    if only_D is not None:
        jobs = [j for j in jobs if j["D"] == only_D]
    interleave = bool(cfg.get("interleave_dimensions"))
    if interleave:
        jobs = interleave_jobs(jobs)
    if num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if i % num_shards == shard]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    out_dir = Path(results_dir) / "filter_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (out_dir / f"filter_{job_key(j)}.parquet").exists()]
    logger.info(f"Stage C: {len(jobs)} jobs in this shard, {len(pending)} pending.")

    if dry_run:
        for j in pending[:20]:
            logger.info(f"  would run {j}")
        return len(pending)

    faiss.omp_set_num_threads(1)  # measured latency is single-threaded
    blas_limit = _blas_single_thread()  # noqa: F841 - kept alive for the whole run

    split = load_query_split(QUERY_SPLIT_PATH_DEFAULT)
    cells = _CellCache(cfg, len(split["tuning"]), len(split["test"]),
                       max_tuning_queries, max_test_queries)
    self_built = bool(cfg.get("build"))
    index_lru: dict = {}  # (N, D, type, repeat) -> (index, path); small, indexes are large
    lru_max = 1 if interleave else 4

    def get_index(N, D, it, repeat, passages):
        key = (N, D, it, repeat)
        if key not in index_lru:
            while len(index_lru) >= lru_max:
                index_lru.pop(next(iter(index_lru)))
            if self_built:
                index_lru[key] = build_or_load_index(cfg, N, D, it, repeat or 0, passages,
                                                     build_threads=build_threads)
            else:
                path = resolve_index_path(N, D, it, results_dir=index_source_dir)
                if path is None:
                    logger.error(f"No serialized {it} index for N={N} D={D}. Stage C reuses "
                                 f"Stage B indexes; run Stage B with --keep-indexes first.")
                    index_lru[key] = (None, None)
                else:
                    logger.info(f"  loading {it} index: {path}")
                    index_lru[key] = (faiss.read_index(path), path)
        return index_lru[key]

    canary_every = int(cfg.get("canary", {}).get("every", 0) or 0)
    done = 0
    prev_cell = None
    for job in pending:
        N, D = job["N"], job["D"]
        if not interleave and prev_cell not in (None, (N, D)):
            cells.drop(*prev_cell)  # free the previous cell's heavy state (legacy mode)
            index_lru.clear()
        prev_cell = (N, D)
        cell = cells.get(N, D)

        index = path = None
        if job["strategy"] != "pre_filter_exact":
            index, path = get_index(N, D, job["index_type"], job.get("build_repeat"),
                                    cell["passages"])
            if index is None:
                continue

        logger.info(f"  job {job['strategy']} {job['index_type']} D={D} "
                    f"{job['correlation']} S={job['target_S']:g} safety={job['safety']} "
                    f"repeat={job.get('build_repeat')}")
        # Record WHICH serialized index answered this job: the choice affects filtered recall.
        job = dict(job, base_index_path=path)
        rows = run_filter_job(job, cfg, cell["passages"], cell["prep"], cell["tuning"],
                              cell["test"], index, results_dir=results_dir,
                              gt_threads=gt_threads, sq_norms=cell["sq"],
                              filter_tuning_queries=cell["f_tuning"],
                              filter_test_queries=cell["f_test"])
        assert faiss.omp_get_max_threads() == 1, "a helper leaked the OMP thread count"
        pd.DataFrame(rows).to_parquet(out_dir / f"filter_{job_key(job)}.parquet", index=False)
        done += 1
        if canary_every and done % canary_every == 0:
            run_canary(cfg, cells, get_index, results_dir, shard, done)

    logger.info(f"Stage C: completed {done} jobs.")
    return done


QUERY_SPLIT_PATH_DEFAULT = "data/query_split.json"


def recommended_shards(N: int, D: int, index_type: str = "hnsw", cores: int = 16) -> int:
    """How many Stage C shards fit in memory for this cell.

    Each shard materialises the corpus (N x D x 4), its squared norms, and a full index, so
    the large cells admit far fewer shards than there are cores.
    """
    from annstudy.coordinate import default_mem_budget

    per = N * D * 4 + N * 8 + (N * D * 4 + N * 32 * 2 * 4) + 500_000_000
    return max(1, min(cores, int(default_mem_budget() // per)))


def plan_shards(config_path: str = "configs/stage_c.yaml", cores: int = 16) -> list[dict]:
    """Per-cell job counts and the shard count that fits in memory."""
    cfg = load_stage_c_config(config_path)
    jobs = enumerate_filter_jobs(cfg)
    out = []
    for (N, D), cell_jobs in _group_by_cell(jobs):
        out.append({"N": N, "D": D, "jobs": len(cell_jobs),
                    "shards": recommended_shards(N, D, cores=cores)})
    return out


def _group_by_cell(jobs: list[dict]):
    """Yield ((N, D), jobs) so heavy per-cell state is loaded once."""
    cells: dict[tuple[int, int], list[dict]] = {}
    for j in jobs:
        cells.setdefault((j["N"], j["D"]), []).append(j)
    for cell in sorted(cells):
        yield cell, cells[cell]


# ------------------------------------------------------------------------- persistence

def consolidate_filter_runs(results_dir="results/raw"):
    """Concatenate per-job filter_runs parquet into results/raw/filter_runs.parquet (M5)."""
    jobs_dir = Path(results_dir) / "filter_runs"
    files = sorted(jobs_dir.glob("filter_*.parquet"))  # sorted: deterministic row order
    if not files:
        logger.warning(f"No filter_runs files in {jobs_dir}")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = Path(results_dir) / "filter_runs.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Consolidated {len(df)} filter_runs rows -> {out}")
    return out


def main():
    ensure_opt_level()
    parser = argparse.ArgumentParser(description="Stage C filtered-search runner")
    parser.add_argument("--config", default="configs/stage_c.yaml")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--consolidate", action="store_true",
                        help="Only concatenate per-job filter_runs into one parquet")
    parser.add_argument("--shard", type=int, default=0,
                        help="This shard's index; pin each shard to its own core")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Run at most this many pending jobs (smoke validation)")
    parser.add_argument("--max-tuning-queries", type=int, default=None,
                        help="Subsample tuning queries (recorded in the output rows)")
    parser.add_argument("--max-test-queries", type=int, default=None,
                        help="Subsample test queries (recorded in the output rows)")
    parser.add_argument("--gt-threads", type=int, default=0,
                        help="Threads for filtered ground truth (reference data, not measured)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pending jobs without measuring")
    parser.add_argument("--index-source-dir", default="results/raw",
                        help="Where the build table for serialized index lookup lives")
    parser.add_argument("--only-N", type=int, default=None, help="Restrict to one corpus size")
    parser.add_argument("--only-D", type=int, default=None, help="Restrict to one dimension")
    parser.add_argument("--build-threads", type=int, default=0,
                        help="Threads for stage-local index builds (not a measured quantity)")
    parser.add_argument("--precompute-gt", action="store_true",
                        help="Compute every filtered-GT cache for the config, then exit")
    parser.add_argument("--plan", action="store_true",
                        help="Print per-cell job counts and memory-safe shard counts, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.plan:
        import json as _json
        print(_json.dumps(plan_shards(args.config), indent=2))
        return

    if args.consolidate:
        consolidate_filter_runs(args.results_dir)
        return

    if args.precompute_gt:
        precompute_filtered_gt(args.config, gt_threads=args.gt_threads)
        return

    run_stage_c(
        config_path=args.config, results_dir=args.results_dir,
        shard=args.shard, num_shards=args.num_shards, max_jobs=args.max_jobs,
        max_tuning_queries=args.max_tuning_queries,
        max_test_queries=args.max_test_queries,
        gt_threads=args.gt_threads, dry_run=args.dry_run,
        index_source_dir=args.index_source_dir,
        only_N=args.only_N, only_D=args.only_D, build_threads=args.build_threads,
    )
    consolidate_filter_runs(args.results_dir)


if __name__ == "__main__":
    main()
