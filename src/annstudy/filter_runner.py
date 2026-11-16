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


def prepare_filters(N, D, passages, filters_cfg, cache_dir=FILTER_CACHE):
    """Compute/cache uncorrelated attributes and correlated k-means clusters for a cell."""
    from annstudy.filters import generate_correlated_attributes, generate_uncorrelated_attributes

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    uc = filters_cfg.get("uncorrelated", {})
    attrs_path = cache_dir / f"attrs_N{N}.npy"  # D-independent
    attrs = _load_cached(attrs_path)
    if attrs is None:
        attrs = generate_uncorrelated_attributes(N, uc.get("n_attributes", 10_000),
                                                  uc.get("seed", 42))
        _save_atomic(attrs_path, attrs)
    out["uncorrelated"] = {"attrs": attrs, "n_attributes": uc.get("n_attributes", 10_000)}

    cc = filters_cfg.get("correlated")
    if cc:
        cl_path = cache_dir / f"clusters_N{N}_D{D}.npy"
        ce_path = cache_dir / f"centroids_N{N}_D{D}.npy"
        cluster_ids, centroids = _load_cached(cl_path), _load_cached(ce_path)
        if cluster_ids is None or centroids is None:
            logger.info(f"k-means (k={cc.get('n_clusters', 1000)}) on N={N} D={D} ...")
            cluster_ids, centroids = generate_correlated_attributes(
                passages, cc.get("n_clusters", 1000), cc.get("seed", 42))
            _save_atomic(cl_path, cluster_ids)
            _save_atomic(ce_path, centroids)
        out["correlated"] = {"cluster_ids": cluster_ids, "centroids": centroids,
                             "n_clusters": cc.get("n_clusters", 1000)}
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

def _measure(strategy, index, index_type, passages, queries, get_matching, k, search_param,
             n_passes, safety, N, static: bool = False, sq_norms=None):
    """Run one strategy over all queries.

    Returns (ret_ids [n,k], per_query_ns [n], n_matching [n]).

    What is inside the timed region, and why. Evaluating the predicate itself
    (``get_matching``) is always excluded: it is the database's filtering work, identical for
    every strategy, and including it would swamp the search cost being compared. The
    strategy's own filter structure (bitmap selector / boolean mask) is built ONCE outside the
    timing when the predicate is query-independent, because a real system would precompute it
    and amortise it across queries; for correlated predicates it genuinely varies per query,
    so it is rebuilt inside the timed region. That asymmetry is real rather than an artifact,
    and it matters when reading H2b.
    """
    n = len(queries)
    ret = np.full((n, k), -1, dtype=np.int64)
    times = np.zeros(n, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    needs_selector = strategy == "in_index_filtered"
    needs_mask = strategy == "post_filter_overfetch"

    if needs_selector or needs_mask:
        from annstudy.indexes import set_search_params
        if needs_mask:  # applying the sweep setting is index state, not per-query work
            set_search_params(index, index_type, search_param)

    shared = None
    if static and (needs_selector or needs_mask):
        idx0 = get_matching(0)
        sel, bitmap, mask = make_id_selector(idx0, N)
        shared = (sel, bitmap, mask,
                  search_params_filtered(index_type, sel, search_param) if needs_selector
                  else None)

    for qi in range(n):
        matching_idx = get_matching(qi)
        sizes[qi] = len(matching_idx)
        S_q = len(matching_idx) / N
        passes = []
        for _ in range(n_passes):
            t0 = time.perf_counter_ns()
            if strategy == "pre_filter_exact":
                ids = strategy_pre_filter_exact(passages, queries[qi], matching_idx, k,
                                                sq_norms=sq_norms)
            elif needs_selector:
                if shared is not None:
                    params = shared[3]
                else:  # correlated: the selector is part of answering this query
                    sel, bitmap, _ = make_id_selector(matching_idx, N)
                    params = search_params_filtered(index_type, sel, search_param)
                ids = strategy_in_index_filtered(index, index_type, queries[qi], k, params)
            else:  # post_filter_overfetch
                keep = shared[2] if shared is not None else None
                if keep is None:
                    keep = np.zeros(N, dtype=bool)
                    keep[matching_idx] = True
                ids = strategy_post_filter_overfetch(index, index_type, queries[qi], keep, k,
                                                     search_param, safety, S_q, N)
            passes.append(time.perf_counter_ns() - t0)
        ret[qi] = ids
        times[qi] = int(np.median(passes))
    return ret, times, sizes


def filtered_gt(passages, queries, mask_fn, k, cache_path):
    """Exact filtered top-k GT (cached). Returns (ids, dists)."""
    from annstudy.filters import compute_filtered_ground_truth

    cache_path = Path(cache_path)
    if cache_path.exists():
        d = np.load(cache_path)
        if d["ids"].shape[0] == len(queries):
            return d["ids"], d["distances"]
    ids, dists = compute_filtered_ground_truth(passages, queries, mask_fn, k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ids=ids, distances=dists)
    return ids, dists


def filtered_gt_static(passages, queries, matching_idx, k, cache_path, threads: int = 0):
    """Exact filtered top-k GT for a query-INDEPENDENT subset, via a batched FAISS flat scan.

    Ground truth is reference data, not a measured quantity, so it may use every core. This is
    far faster than the per-query numpy loop: one flat index over the subset answers all
    queries in a single batched call.
    """
    import faiss

    cache_path = Path(cache_path)
    if cache_path.exists():
        d = np.load(cache_path)
        if d["ids"].shape[0] == len(queries):
            return d["ids"], d["distances"]

    if threads:
        faiss.omp_set_num_threads(threads)

    n_q = len(queries)
    ids = np.full((n_q, k), -1, dtype=np.int64)
    dists = np.full((n_q, k), np.inf, dtype=np.float32)
    if len(matching_idx) > 0:
        sub = np.ascontiguousarray(passages[matching_idx])
        flat = faiss.IndexFlatL2(sub.shape[1])
        flat.add(sub)
        kk = min(k, len(matching_idx))
        d_sub, i_sub = flat.search(np.ascontiguousarray(queries), kk)
        # map subset-local ids back to global passage ids
        ids[:, :kk] = np.where(i_sub >= 0, matching_idx[np.clip(i_sub, 0, None)], -1)
        dists[:, :kk] = d_sub

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ids=ids, distances=dists)
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
            recalls[qi] = 1.0  # nothing matches the predicate: nothing to miss
            continue
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
                            jobs.append({
                                "N": int(N), "D": int(D),
                                "index_type": ("exact" if strategy == "pre_filter_exact"
                                               else index_type),
                                "strategy": strategy,
                                "correlation": correlation,
                                "target_S": target_S,
                                "safety": safety,
                                "k": cfg.get("k", 10),
                            })
    return jobs


def job_key(job: dict) -> str:
    """Stable short hash identifying a job, used for resumable per-job parquet files."""
    from annstudy.runner import config_hash
    return config_hash({k: job[k] for k in
                        ("N", "D", "index_type", "strategy", "correlation", "target_S",
                         "safety", "k")})


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
) -> list[dict]:
    """Measure one Stage C job and return its ``filter_runs`` rows (plan section 5).

    Protocol: sweep the search parameter on tuning queries, take the cheapest setting reaching
    each R, re-measure that setting on the test queries, and additionally record recall at one
    fixed mid-range effort to expose silent recall loss.
    """
    from annstudy.hardware import snapshot
    from annstudy.indexes import get_search_grid
    from annstudy.metrics import bootstrap_latency_percentile, latency_stats

    N, D, k = job["N"], job["D"], job["k"]
    strategy, correlation, target_S = job["strategy"], job["correlation"], job["target_S"]
    index_type = job["index_type"]
    measurement = cfg.get("measurement", {})
    n_passes = measurement.get("latency_passes", 3)
    n_resamples = measurement.get("bootstrap_resamples", 1000)
    recall_targets = cfg.get("recall_targets", [0.90, 0.95, 0.99])
    corr_seed = cfg.get("filters", {}).get("correlated", {}).get("seed", 42)

    hw = snapshot()
    gt_k = max(k, 100)
    gt_dir = FILTERED_GT / f"N{N}_D{D}"
    tag = f"{correlation}_S{target_S:g}"

    def provider(queries, offset):
        return make_matching_provider(prep, correlation, target_S, queries, N,
                                      seed=corr_seed + offset)

    get_tun, static = provider(tuning_queries, 0)
    get_test, _ = provider(test_queries, 100_000)  # distinct seed block from tuning

    # Filtered ground truth: exact top-k over the matching subset for every query.
    if static:
        gt_tun = filtered_gt_static(passages, tuning_queries, get_tun(0), gt_k,
                                    gt_dir / f"{tag}_tuning.npz", threads=gt_threads)
        gt_test = filtered_gt_static(passages, test_queries, get_test(0), gt_k,
                                     gt_dir / f"{tag}_test.npz", threads=gt_threads)
    else:
        gt_tun = filtered_gt(passages, tuning_queries,
                             make_mask_fn(prep, correlation, target_S, tuning_queries,
                                          seed=corr_seed),
                             gt_k, gt_dir / f"{tag}_tuning.npz")
        gt_test = filtered_gt(passages, test_queries,
                              make_mask_fn(prep, correlation, target_S, test_queries,
                                           seed=corr_seed + 100_000),
                              gt_k, gt_dir / f"{tag}_test.npz")

    # Search grid for this strategy. Exact scan has no search knob.
    if strategy == "pre_filter_exact":
        grid = [{}]
    else:
        nlist = None
        if index is not None and index_type in ("ivf_flat", "ivf_pq"):
            nlist = int(getattr(index, "nlist", 0)) or None
        grid = get_search_grid(index_type, nlist=nlist, k=k)

    def run(queries, get_matching, gt, search_param):
        ret, times, sizes = _measure(
            strategy, index, index_type, passages, queries, get_matching, k, search_param,
            n_passes, job["safety"], N, static=static, sq_norms=sq_norms,
        )
        rec = filtered_recall(ret, gt[1], passages, queries, k, sizes)
        return rec, times, sizes

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
        rec, times, sizes = run(tuning_queries, get_tun, gt_tun, sp)
        stats = latency_stats(times)
        sweep.append({
            "search_param": sp, "recall": float(np.mean(rec)),
            "lat_p95_us": stats["lat_p95_us"], "measured_S": float(np.mean(sizes)) / N,
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
        "base_index_path": job.get("base_index_path"),
        "base_index_run_id": (Path(job["base_index_path"]).stem.replace("idx_", "")
                              if job.get("base_index_path") else None),
    }

    # --- fixed mid-range effort, to expose silent recall loss at small S
    rows: list[dict] = []
    mid = sweep[len(sweep) // 2]
    rec_fixed, times_fixed, sizes_fixed = run_test(mid["search_param"])
    fixed_stats = latency_stats(times_fixed)
    fixed = {
        "fixed_effort_param": json_safe(mid["search_param"]),
        "fixed_effort_recall": float(np.mean(rec_fixed)),
        "fixed_effort_lat_p95_us": fixed_stats["lat_p95_us"],
    }

    # --- cheapest setting reaching each R, re-measured on the test queries
    for R in recall_targets:
        ok = [s for s in sweep if s["recall"] >= R]
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
        rec, times, sizes = run_test(best["search_param"])
        stats = latency_stats(times)
        _, lo, hi = bootstrap_latency_percentile(times, percentile=95, n_resamples=n_resamples)
        measured_S = float(np.mean(sizes)) / N
        rows.append(dict(
            row, reachable=True,
            search_param=json_safe(best["search_param"]),
            recall_tuning=best["recall"],
            recall_test=float(np.mean(rec)),
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
) -> int:
    """Run Stage C jobs, writing one parquet per job. Resumable and shardable.

    Latency must be single-threaded (plan section 3), so parallelism comes from running
    several shards pinned to different cores:
      ``taskset -c 3 annstudy-run-filters --shard 3 --num-shards 16``
    """
    import faiss

    from annstudy.coordinate import DEFAULT_N_QUERIES, passages_path, queries_path
    from annstudy.data import load_query_split

    cfg = load_stage_c_config(config_path)
    jobs = enumerate_filter_jobs(cfg)
    # Cell filters exist because each shard holds the cell's corpus AND index: 13.1 GB at
    # N=1M/D=1536, so only ~3 shards fit in memory there against ~16 at the small cells.
    # Stage C is therefore driven one cell at a time with a per-cell shard count.
    if only_N is not None:
        jobs = [j for j in jobs if j["N"] == only_N]
    if only_D is not None:
        jobs = [j for j in jobs if j["D"] == only_D]
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

    split = load_query_split(QUERY_SPLIT_PATH_DEFAULT)
    n_tun, n_test = len(split["tuning"]), len(split["test"])

    done = 0
    # Group by cell so the passage array, filter prep and index are loaded once per cell.
    for (N, D), cell_jobs in _group_by_cell(pending):
        n_queries = cfg.get("n_queries", DEFAULT_N_QUERIES)
        passages = np.asarray(np.memmap(passages_path(D), dtype=np.float32, mode="r",
                                        shape=(N, D)))
        all_q = np.memmap(queries_path(D), dtype=np.float32, mode="r", shape=(n_queries, D))
        tuning_queries = np.ascontiguousarray(all_q[:n_tun][:max_tuning_queries])
        test_queries = np.ascontiguousarray(
            all_q[n_tun:n_tun + n_test][:max_test_queries])

        logger.info(f"Cell N={N} D={D}: {len(cell_jobs)} jobs, "
                    f"{len(tuning_queries)} tuning / {len(test_queries)} test queries")
        prep = prepare_filters(N, D, passages, cfg["filters"])
        # Precomputed once per cell so the exact-scan baseline is a single BLAS matvec.
        sq = passage_sq_norms(passages)

        index_cache: dict[str, object] = {}
        index_paths: dict[str, str] = {}
        for job in cell_jobs:
            index = None
            if job["strategy"] != "pre_filter_exact":
                it = job["index_type"]
                if it not in index_cache:
                    # Indexes and their build table live in the canonical results dir even when
                    # this run writes elsewhere (e.g. a scratch dir for driver validation).
                    path = resolve_index_path(N, D, it, results_dir=index_source_dir)
                    if path is None:
                        logger.error(
                            f"No serialized {it} index for N={N} D={D}. Stage C reuses Stage B "
                            f"indexes; run Stage B with --keep-indexes first."
                        )
                        index_cache[it] = None
                    else:
                        logger.info(f"  loading {it} index: {path}")
                        index_cache[it] = faiss.read_index(path)
                        index_paths[it] = path
                index = index_cache[it]
                if index is None:
                    continue

            logger.info(f"  job {job['strategy']} {job['index_type']} "
                        f"{job['correlation']} S={job['target_S']:g} "
                        f"safety={job['safety']}")
            # Record WHICH serialized index answered this job: several Stage B winners can
            # exist per (cell, index type) and the choice affects filtered recall, so the row
            # must say which one rather than leaving it implicit.
            job = dict(job, base_index_path=index_paths.get(job["index_type"]))
            rows = run_filter_job(job, cfg, passages, prep, tuning_queries, test_queries,
                                  index, results_dir=results_dir, gt_threads=gt_threads,
                                  sq_norms=sq)
            pd.DataFrame(rows).to_parquet(out_dir / f"filter_{job_key(job)}.parquet",
                                          index=False)
            done += 1

        del passages, index_cache, sq  # free the cell's heavy state before the next cell
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
    files = list(jobs_dir.glob("filter_*.parquet"))
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

    run_stage_c(
        config_path=args.config, results_dir=args.results_dir,
        shard=args.shard, num_shards=args.num_shards, max_jobs=args.max_jobs,
        max_tuning_queries=args.max_tuning_queries,
        max_test_queries=args.max_test_queries,
        gt_threads=args.gt_threads, dry_run=args.dry_run,
        index_source_dir=args.index_source_dir,
        only_N=args.only_N, only_D=args.only_D,
    )
    consolidate_filter_runs(args.results_dir)


if __name__ == "__main__":
    main()
