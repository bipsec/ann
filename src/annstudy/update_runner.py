"""Stage D driver: incremental-insert (update) study.

Wraps :func:`annstudy.updates.run_update_study` with config loading from
``configs/stage_d.yaml``, search-parameter pre-tuning for R=0.95 on the initial build,
reproducibility metadata, the ``retrained=True`` flag on the rebuild reference row
(decision M6), and Parquet persistence to ``results/raw/update_runs/``.

Query slicing is POSITIONAL (decision M8). Run pinned to one core with OMP_NUM_THREADS=1
for clean latency; latencies are only AVX2-representative on a Linux faiss-cpu build.
"""

import argparse
import logging
import uuid
from pathlib import Path

import numpy as np
import yaml

from annstudy.faiss_setup import ensure_opt_level

logger = logging.getLogger(__name__)


def _load_split_positions(split_path="data/query_split.json"):
    from annstudy.data import load_query_split
    split = load_query_split(split_path)
    n_tune = len(split["tuning"])
    tuning = list(range(n_tune))
    test = list(range(n_tune, n_tune + len(split["test"])))
    return tuning, test


def make_build_fns(cfg, N):
    """Bind the indexes.py factories with the Stage D config params. Keyed by index type."""
    from annstudy.indexes import build_hnsw, build_ivf_flat, build_ivf_pq, compute_nlist

    hp, ip, pp = cfg["hnsw"], cfg["ivf_flat"], cfg["ivf_pq"]
    return {
        "hnsw": lambda v, d: build_hnsw(v, d, M=hp["M"], ef_construction=hp["ef_construction"]),
        "ivf_flat": lambda v, d: build_ivf_flat(v, d, nlist=compute_nlist(N, ip["c"]), seed=0),
        "ivf_pq": lambda v, d: build_ivf_pq(v, d, nlist=compute_nlist(N, pp["c"]),
                                            m=d // pp["m_divisor"], nbits=pp["nbits"], seed=0),
    }


def pretune_search_param(index_type, build_fn, passages_initial, tuning_queries,
                         gt_dists_tuning, k, R):
    """Build on the initial vectors, sweep search grid on tuning queries, return the cheapest
    setting reaching recall R (lowest p95 latency). Falls back to the max-effort setting."""
    from annstudy.indexes import get_search_grid, set_search_params
    from annstudy.metrics import latency_stats, measure_latency, recall_at_k_batch

    dim = passages_initial.shape[1]
    index, _ = build_fn(passages_initial, dim)
    nlist = getattr(index, "nlist", None)
    grid = get_search_grid(index_type, nlist=nlist, k=k)

    best = None
    fallback = None
    for sp in grid:
        set_search_params(index, index_type, sp)
        _, ids = index.search(tuning_queries, k)
        rec = recall_at_k_batch(ids, gt_dists_tuning, passages_initial, tuning_queries, k=k).mean()
        p95 = latency_stats(measure_latency(index, tuning_queries, k=k, n_passes=1))["lat_p95_us"]
        fallback = sp  # grids are ordered increasing effort -> last is max effort
        if rec >= R and (best is None or p95 < best[1]):
            best = (sp, p95)
    chosen = best[0] if best else fallback
    logger.info(f"Stage D pretune {index_type}: chose {chosen} (R>={R} {'hit' if best else 'MISS->max effort'})")
    return chosen


def run_stage_d(config_path="configs/stage_d.yaml", results_dir="results/raw",
                index_types=None, gt_threads: int = 0):
    ensure_opt_level()
    from annstudy.hardware import snapshot
    from annstudy.updates import run_update_study

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    N, D = cfg["cell"]["N"], cfg["cell"]["D"]
    k = cfg["k"]
    up = cfg["update"]
    initial_n = round(up["initial_fraction"] * N)
    R = up["recall_target_for_tuning"]
    index_types = index_types or cfg["indexes"]

    passages = np.memmap(f"data/emb/passages_D{D}.f32", dtype=np.float32, mode="r", shape=(N, D))
    all_q = np.memmap(f"data/emb/queries_D{D}.f32", dtype=np.float32, mode="r", shape=(10_000, D))
    gt = np.load(f"data/gt/gt_N{N}_D{D}.npz")
    gt_dists = gt["distances"]

    tuning_pos, test_pos = _load_split_positions()
    tuning_queries = np.ascontiguousarray(np.array(all_q[tuning_pos]))
    test_queries = np.ascontiguousarray(np.array(all_q[test_pos]))
    gt_dists_test = gt_dists[test_pos]
    passages_arr = np.array(passages[:N])
    passages_initial = np.ascontiguousarray(passages_arr[:initial_n])

    build_fns = make_build_fns(cfg, N)
    hw = snapshot()
    out_dir = Path(results_dir) / "update_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The pre-tune index holds only the first initial_n vectors, so it must be scored against
    # ground truth over that prefix. Scoring it against full-corpus GT (as originally done)
    # counted the 50K not-yet-inserted vectors as misses, under-estimated recall, and pushed the
    # pre-tune to maximum effort (HNSW recall 0.9998, IVF nprobe=512) where drift cannot show.
    import faiss

    from annstudy.updates import prefix_ground_truth
    _, gt_dists_tuning_prefix = prefix_ground_truth(
        passages_arr, tuning_queries, initial_n, threads=gt_threads, query_set="tuning")
    faiss.omp_set_num_threads(1)  # measured latency is single-threaded

    for index_type in index_types:
        search_param = pretune_search_param(
            index_type, build_fns[index_type], passages_initial,
            tuning_queries, gt_dists_tuning_prefix, k, R)

        df = run_update_study(
            index_type=index_type, build_fn=build_fns[index_type], search_params=search_param,
            passages=passages_arr, queries=test_queries,
            gt_ids=gt["ids"][test_pos], gt_distances=gt_dists_test,
            initial_n=initial_n, n_batches=up["insert_batches"],
            batch_size=up["batch_size"], k=k, seed=0,
            gt_threads=gt_threads)

        # M6: mark the rebuild reference row as retrained; attach reproducibility metadata.
        df["retrained"] = df["rebuild_reference_recall"].notna()
        df["git_commit"] = hw["git_commit"]
        df["hardware_id"] = hw["hardware_id"]
        df["seed"] = 0
        df["N"] = N
        df["D"] = D
        df["search_param"] = str(search_param)

        run_id = uuid.uuid4().hex
        path = out_dir / f"update_{index_type}_{run_id}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Stage D {index_type}: wrote {len(df)} rows -> {path}")


def main():
    ensure_opt_level()
    parser = argparse.ArgumentParser(description="Stage D update study")
    parser.add_argument("--config", default="configs/stage_d.yaml")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--indexes", nargs="+", default=None)
    parser.add_argument("--gt-threads", type=int, default=0,
                        help="Threads for prefix ground truth (reference data, not measured)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_stage_d(args.config, args.results_dir, args.indexes, gt_threads=args.gt_threads)


if __name__ == "__main__":
    main()
