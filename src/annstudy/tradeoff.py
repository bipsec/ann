"""Stage E: truncate vs quantize vs truncate-then-rerank, at matched bytes per vector.

The question only a single-corpus Matryoshka setup can ask cleanly: given a memory budget per
vector, is it cheaper (at a required recall) to keep FEWER DIMENSIONS (index the D' prefix in
float32), FEWER BITS (index all 1536 dimensions scalar- or RaBitQ-quantized), or to search a
cheap prefix index and RE-RANK its candidates against the full-width vectors?

Every arm answers the SAME question — the exact top-10 at D = 1536 — so truncation loss counts
against the arms that truncate. Concretely:

- Every arm receives the full 1536-d query. Truncating arms slice and renormalise it inside the
  timed call, so that (tiny) cost is charged equally to every truncating arm.
- Recall@10 is scored with the plan's distance-tolerance rule against
  ``data/gt/gt_N{N}_D1536.npz`` using the 1536-d vectors, never the index's own distances.
- Bytes per vector = serialized index bytes / N, plus the re-rank store (N x 1536 x 4 for
  float32, N x 1536 for SQ8) when the arm re-ranks. Both totals are reported.

An arm is (index family, index dimension D', re-rank store). Families: ``hnsw`` (float32),
``hnsw_sq8``, ``hnsw_sq4``, ``hnsw_rabitq``; stores: ``none``, ``fp32``, ``sq8``. One graph
configuration (M, efConstruction) is used for every arm so the comparison is about the stored
representation, not about graph tuning.

Protocol per arm: sweep efSearch (x re-rank depth k' for re-ranking arms) on the tuning queries
(1 pass), choose the cheapest setting whose recall clears R (``selection: lower_ci`` by
default), then re-measure that setting on the test queries with the full latency protocol
(single thread, batch 1, warm-up, 3 outer passes). One parquet row per (arm, R) under
``results/raw/tradeoff_runs/``.
"""

import argparse
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from annstudy.faiss_setup import ensure_opt_level

logger = logging.getLogger(__name__)

REF_D = 1536


# --------------------------------------------------------------------------- searcher

class PrefixSearcher:
    """Search an index built on the D' prefix with a full-width query; optionally re-rank.

    ``search(q, k)`` takes queries at REF_D. The index sees ``q[:, :D'] / ||q[:, :D']||``
    (identical to the stored truncated-and-renormalised query files). With a re-rank store,
    the index returns ``k_prime`` candidates whose exact (fp32) or SQ8-decoded distances at
    REF_D decide the final top-k.
    """

    def __init__(self, index, d_index, store=None, store_kind="none", sq=None, k_prime=None):
        self.index, self.d_index = index, int(d_index)
        self.store, self.store_kind, self.sq = store, store_kind, sq
        self.k_prime = k_prime

    def set_ef(self, ef):
        self.index.hnsw.efSearch = int(ef)

    def search(self, q, k):
        qd = q[:, :self.d_index]
        if self.d_index != q.shape[1]:
            qd = qd / np.linalg.norm(qd, axis=1, keepdims=True)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        if self.store_kind == "none":
            return self.index.search(qd, k)
        kp = max(k, int(self.k_prime))
        _, cand = self.index.search(qd, kp)
        out_d = np.full((len(q), k), np.inf, dtype=np.float32)
        out_i = np.full((len(q), k), -1, dtype=np.int64)
        for i in range(len(q)):
            c = cand[i][cand[i] >= 0]
            if self.store_kind == "fp32":
                vecs = self.store[c]
            else:  # sq8 codes, decoded on the fly
                vecs = self.sq.decode(self.store[c])
            # unit-norm vectors: ranking by -2 q.x + ||x||^2 equals ranking by ||q - x||^2
            dist = np.einsum("ij,ij->i", vecs, vecs) - 2.0 * (vecs @ q[i])
            top = np.argsort(dist)[:k]
            out_i[i, :len(top)] = c[top]
            out_d[i, :len(top)] = dist[top]
        return out_d, out_i


# ---------------------------------------------------------------------------- arms

def expand_arms(cfg: dict) -> list[dict]:
    """One dict per arm from the config's ``arms`` list (each entry may list several D')."""
    arms = []
    for a in cfg["arms"]:
        for d in a["D"]:
            for store in a.get("stores", ["none"]):
                arm = {"family": a["family"], "D": int(d), "store": store}
                arm["arm_id"] = f"{arm['family']}@{arm['D']}+{store}"
                arms.append(arm)
    seen = set()
    for a in arms:
        if a["arm_id"] in seen:
            raise ValueError(f"duplicate arm {a['arm_id']}")
        seen.add(a["arm_id"])
    return arms


def _passages(N, D):
    from annstudy.coordinate import passages_path
    return np.memmap(passages_path(D), dtype=np.float32, mode="r", shape=(N, D))


def build_index(family, x, D, M, efc, seed=0):
    import faiss

    from annstudy.indexes import _seed_hnsw_levels

    if family == "hnsw":
        idx = faiss.IndexHNSWFlat(D, M)
    elif family in ("hnsw_sq8", "hnsw_sq4"):
        qt = faiss.ScalarQuantizer.QT_8bit if family == "hnsw_sq8" else faiss.ScalarQuantizer.QT_4bit
        idx = faiss.IndexHNSWSQ(D, qt, M)
    elif family == "hnsw_rabitq":
        idx = faiss.IndexHNSWRaBitQ(D, M)
    else:
        raise ValueError(f"unknown family {family}")
    idx.hnsw.efConstruction = efc
    _seed_hnsw_levels(idx, seed)
    t0 = time.perf_counter()
    if not idx.is_trained:
        idx.train(x)
    idx.add(x)
    return idx, time.perf_counter() - t0


def get_index(cfg, arm, N, build_threads=0):
    """Build or load the arm's base index (cached per family/D under the stage index dir)."""
    import os

    import faiss

    g = cfg.get("graph", {})
    M, efc = int(g.get("M", 32)), int(g.get("ef_construction", 200))
    out_dir = Path(cfg.get("index_dir", "data/indexes/stage_e"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm['family']}_N{N}_D{arm['D']}_M{M}.faiss"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        return faiss.read_index(str(path)), json.loads(meta_path.read_text())
    prev = faiss.omp_get_max_threads()
    try:
        if build_threads:
            faiss.omp_set_num_threads(build_threads)
        x = np.ascontiguousarray(_passages(N, arm["D"]))
        idx, secs = build_index(arm["family"], x, arm["D"], M, efc)
    finally:
        faiss.omp_set_num_threads(prev)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    faiss.write_index(idx, str(tmp))
    os.replace(tmp, path)
    meta = {"build_seconds": secs, "index_bytes": path.stat().st_size, "M": M,
            "ef_construction": efc, "N": N, "D": arm["D"], "family": arm["family"],
            "build_threads": build_threads or prev}
    meta_path.write_text(json.dumps(meta, indent=1))
    logger.info(f"built {arm['family']}@{arm['D']} in {secs:.0f}s, {meta['index_bytes'] / N:.0f} B/vec")
    return idx, meta


def get_store(kind, N, cache_dir):
    """Re-rank store at REF_D: fp32 vectors (memmap) or SQ8 codes (cached)."""
    import faiss
    if kind == "none":
        return None, None, 0
    full = _passages(N, REF_D)
    if kind == "fp32":
        return np.asarray(full), None, REF_D * 4
    if kind != "sq8":
        raise ValueError(f"unknown store {kind}")
    sq = faiss.ScalarQuantizer(REF_D, faiss.ScalarQuantizer.QT_8bit)
    rng = np.random.default_rng(0)
    sq.train(np.ascontiguousarray(full[np.sort(rng.choice(N, min(N, 100_000), replace=False))]))
    p = Path(cache_dir) / f"sq8codes_N{N}_D{REF_D}.npy"
    if p.exists():
        codes = np.load(p)
    else:
        codes = np.empty((N, sq.code_size), dtype=np.uint8)
        for s in range(0, N, 50_000):
            codes[s:s + 50_000] = sq.compute_codes(np.ascontiguousarray(full[s:s + 50_000]))
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, codes)
    return codes, sq, sq.code_size


# ------------------------------------------------------------------------ measurement

def _timed(searcher, q, k, n_passes):
    """Per-query latency (median over outer passes), ids from pass 0, and HNSW ndis."""
    import faiss
    n = len(q)
    times = np.zeros((n_passes, n), dtype=np.int64)
    ids = np.full((n, k), -1, dtype=np.int64)
    ndis = np.zeros(n, dtype=np.int64)
    for p in range(n_passes):
        for i in range(n):
            if p == 0:
                faiss.cvar.hnsw_stats.reset()
            t0 = time.perf_counter_ns()
            _, I = searcher.search(q[i:i + 1], k)
            times[p, i] = time.perf_counter_ns() - t0
            if p == 0:
                ids[i] = I[0]
                ndis[i] = faiss.cvar.hnsw_stats.ndis
    return np.median(times, axis=0), ids, ndis


def run_arm(cfg, arm, N, tun_q, test_q, gt_tun, gt_test, ref_passages, build_threads=0):
    from annstudy.hardware import snapshot
    from annstudy.metrics import (
        bootstrap_latency_percentile,
        bootstrap_recall,
        latency_stats,
        recall_at_k_batch,
    )

    k = int(cfg.get("k", 10))
    meas = cfg.get("measurement", {})
    n_passes = int(meas.get("latency_passes", 3))
    selection = meas.get("selection", "lower_ci")
    efs = cfg.get("ef_search", [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    kps = cfg.get("k_prime", [20, 50, 100]) if arm["store"] != "none" else [None]

    index, meta = get_index(cfg, arm, N, build_threads=build_threads)
    store, sq, store_bpv = get_store(arm["store"], N, cfg.get("index_dir", "data/indexes/stage_e"))
    s = PrefixSearcher(index, arm["D"], store, arm["store"], sq)

    w = int(meas.get("warmup_queries", 200))  # warm-up, untimed and discarded
    s.set_ef(efs[0])
    s.k_prime = kps[0]
    for i in range(min(w, len(tun_q))):
        s.search(tun_q[i:i + 1], k)

    sweep = []
    for kp in kps:
        for ef in efs:
            if ef < k:
                continue
            s.set_ef(ef)
            s.k_prime = kp
            t, ids, nd = _timed(s, tun_q, k, 1)
            rec = recall_at_k_batch(ids, gt_tun, ref_passages, tun_q, k=k)
            _, lo, _ = bootstrap_recall(rec, n_resamples=500)
            sweep.append({"ef_search": ef, "k_prime": kp, "recall": float(rec.mean()),
                          "recall_lo": float(lo), "p95": latency_stats(t)["lat_p95_us"],
                          "ndis": float(nd.mean())})
            logger.info(f"  {arm['arm_id']} ef={ef} k'={kp}: recall={rec.mean():.4f} "
                        f"p95={sweep[-1]['p95']:.0f}us")

    hw = snapshot()
    bpv_index = meta["index_bytes"] / N
    base = {"run_id": None, "stage": cfg.get("stage", "E"), "arm_id": arm["arm_id"],
            "family": arm["family"], "D_index": arm["D"], "store": arm["store"], "N": N,
            "k": k, "M": meta["M"], "ef_construction": meta["ef_construction"],
            "bytes_per_vec_index": bpv_index,
            "bytes_per_vec_total": bpv_index + store_bpv,
            "build_seconds": meta["build_seconds"], "selection": selection,
            "hardware_id": hw["hardware_id"], "git_commit": hw["git_commit"],
            "timestamp": datetime.now(UTC).isoformat(), "eval_D": REF_D,
            "n_tuning_queries": len(tun_q), "n_test_queries": len(test_q)}
    rows, cache = [], {}
    key = "recall_lo" if selection == "lower_ci" else "recall"
    for R in cfg.get("recall_targets", [0.90, 0.95, 0.99]):
        ok = [r for r in sweep if r[key] >= R]
        row = dict(base, run_id=uuid.uuid4().hex, R=float(R),
                   best_tuning_recall=max(r["recall"] for r in sweep))
        if not ok:
            rows.append(dict(row, reachable=False))
            continue
        best = min(ok, key=lambda r: r["p95"])
        ck = (best["ef_search"], best["k_prime"])
        if ck not in cache:
            s.set_ef(best["ef_search"])
            s.k_prime = best["k_prime"]
            t, ids, nd = _timed(s, test_q, k, n_passes)
            cache[ck] = (t, recall_at_k_batch(ids, gt_test, ref_passages, test_q, k=k), nd)
        t, rec, nd = cache[ck]
        st = latency_stats(t)
        _, lo, hi = bootstrap_latency_percentile(t.astype(np.int64), percentile=95)
        rows.append(dict(row, reachable=True, ef_search=best["ef_search"],
                         k_prime=best["k_prime"], recall_tuning=best["recall"],
                         recall_test=float(rec.mean()), meets_R_on_test=bool(rec.mean() >= R),
                         lat_p50_us=st["lat_p50_us"], lat_p95_us=st["lat_p95_us"],
                         lat_mean_us=st["lat_mean_us"], lat_p95_ci_lo=lo, lat_p95_ci_hi=hi,
                         ndis_mean=float(nd.mean())))
        logger.info(f"  {arm['arm_id']} R={R}: ef={best['ef_search']} k'={best['k_prime']} "
                    f"recall_test={rec.mean():.4f} p95={st['lat_p95_us']:.0f}us "
                    f"{base['bytes_per_vec_total']:.0f} B/vec")
    return rows


def run_stage_e(config_path, results_dir="results/raw", shard=0, num_shards=1,
                max_tuning_queries=None, max_test_queries=None, build_threads=0,
                build_only=False):
    import faiss
    import yaml

    from annstudy.coordinate import DEFAULT_N_QUERIES, gt_path, queries_path
    from annstudy.data import load_query_split

    cfg = yaml.safe_load(Path(config_path).read_text())
    N = int(cfg["N"])
    arms = expand_arms(cfg)
    if build_only:
        for arm in arms:
            get_index(cfg, arm, N, build_threads=build_threads)
        for kind in {a["store"] for a in arms}:
            get_store(kind, N, cfg.get("index_dir", "data/indexes/stage_e"))
        return 0
    arms = [a for i, a in enumerate(arms) if i % num_shards == shard]
    out = Path(results_dir) / "tradeoff_runs"
    out.mkdir(parents=True, exist_ok=True)

    faiss.omp_set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        _blas = threadpool_limits(limits=1)  # kept alive for the run
    except ImportError:  # pragma: no cover
        pass

    split = load_query_split("data/query_split.json")
    n_tun = len(split["tuning"])
    allq = np.memmap(queries_path(REF_D), dtype=np.float32, mode="r",
                     shape=(cfg.get("n_queries", DEFAULT_N_QUERIES), REF_D))
    tun_q = np.ascontiguousarray(allq[:n_tun][:max_tuning_queries])
    test_q = np.ascontiguousarray(allq[n_tun:n_tun + len(split["test"])][:max_test_queries])
    gt = np.load(gt_path(N, REF_D))["distances"]
    gt_tun = gt[:n_tun][:len(tun_q)]
    gt_test = gt[n_tun:n_tun + len(split["test"])][:len(test_q)]
    ref = np.asarray(_passages(N, REF_D))

    done = 0
    for arm in arms:
        f = out / f"tradeoff_{cfg.get('stage', 'E')}_{arm['arm_id'].replace('@', '_').replace('+', '_')}_N{N}.parquet"
        if f.exists():
            logger.info(f"skip {arm['arm_id']} (done)")
            continue
        logger.info(f"arm {arm['arm_id']}")
        rows = run_arm(cfg, arm, N, tun_q, test_q, gt_tun, gt_test, ref)
        pd.DataFrame(rows).to_parquet(f, index=False)
        assert faiss.omp_get_max_threads() == 1
        done += 1
    return done


def consolidate(results_dir="results/raw"):
    files = sorted((Path(results_dir) / "tradeoff_runs").glob("tradeoff_*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    p = Path(results_dir) / "tradeoff_runs.parquet"
    df.to_parquet(p, index=False)
    logger.info(f"Consolidated {len(df)} tradeoff rows -> {p}")
    return p


def main():
    ensure_opt_level()
    ap = argparse.ArgumentParser(description="Stage E: truncate vs quantize at matched bytes")
    ap.add_argument("--config", default="configs/stage_e_tradeoff.yaml")
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-tuning-queries", type=int, default=None)
    ap.add_argument("--max-test-queries", type=int, default=None)
    ap.add_argument("--build-threads", type=int, default=0)
    ap.add_argument("--build-only", action="store_true",
                    help="Build every arm's index and re-rank store, then exit (multi-threaded)")
    ap.add_argument("--consolidate", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.consolidate:
        consolidate(a.results_dir)
        return
    run_stage_e(a.config, a.results_dir, a.shard, a.num_shards, a.max_tuning_queries,
                a.max_test_queries, a.build_threads, a.build_only)
    consolidate(a.results_dir)


if __name__ == "__main__":
    main()
