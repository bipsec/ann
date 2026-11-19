"""Subprocess runner for isolated index builds and search sweeps.

Two-tier architecture:
- Coordinator: reads stage config, expands parameter grid, spawns runner subprocesses.
- Runner (this module's main()): builds one index, sweeps search params, writes Parquet.

Thread policy (EXPERIMENT_PLAN sections 1 and 3, decision M7). Latency is measured on ONE
thread pinned to one core, and the grid is parallelised by running many such runners at once:
  taskset -c <core> env OMP_NUM_THREADS=1 python -m annstudy.runner --spec spec.json
Build time and throughput (QPS) are instead measured with T threads, one job at a time, so the
spec carries an explicit ``threads`` field rather than the runner inferring it from the
environment. ``measure_qps`` is therefore off in the pinned latency pass and on in the
T-thread pass.

Memory is measured in its own mode (``--mode memory``), which loads ONLY a serialized index
into a fresh process and reports the RSS delta, since RSS taken after an in-process build is
inflated by the N x D passage array (gap M11).
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def config_hash(spec: dict) -> str:
    """SHA256 of the sorted JSON representation of a build spec."""
    canonical = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_config_key(index_type: str, N: int, D: int, build_params: dict) -> str:
    """Seed-independent identifier for a build configuration.

    ``config_hash`` includes the seed so repeat builds stay distinct rows. But some quantities
    belong to the configuration rather than to one build of it — index size, resident memory,
    and the T-thread throughput of a given search setting — and those must join back to a
    tuned winner that was selected from the seed-0 build. This key is what they join on.
    """
    return config_hash({
        "index_type": index_type, "N": int(N), "D": int(D), "build_params": build_params,
    })


def check_disk_space(path: str, required_bytes: int, safety_factor: float = 2.0) -> bool:
    """Return True if free space > required_bytes * safety_factor."""
    stat = shutil.disk_usage(path)
    needed = required_bytes * safety_factor
    if stat.free < needed:
        logger.error(
            f"Insufficient disk: {stat.free / 1e9:.1f} GB free, "
            f"need {needed / 1e9:.1f} GB at {path}"
        )
        return False
    return True


def measure_rss() -> int:
    """Return current process RSS in bytes."""
    import psutil
    return psutil.Process().memory_info().rss


def run_single_spec(spec: dict) -> dict:
    """Execute one build + sweep specification.

    This function is meant to run in an isolated subprocess.
    It builds the index, saves it, measures memory, and sweeps search parameters.
    """
    from annstudy.faiss_setup import ensure_opt_level
    ensure_opt_level()  # avoid faiss's failed AVX2 probe on generic-only installs

    import faiss

    # Threads come from the spec, not the environment: the pinned latency pass uses 1 and the
    # build/QPS pass uses T. Set it explicitly so faiss cannot inherit a stale OMP setting.
    threads = int(spec.get("threads", 1))
    faiss.omp_set_num_threads(threads)

    from annstudy.data import load_query_split
    from annstudy.hardware import snapshot
    from annstudy.indexes import set_search_params
    from annstudy.metrics import (
        bootstrap_latency_percentile,
        bootstrap_recall,
        latency_stats,
        measure_latency,
        measure_qps,
        recall_at_k_batch,
    )

    run_id = spec.get("run_id", str(uuid.uuid4()))
    hw = snapshot()
    hardware_id = hw["hardware_id"]
    git_commit = hw["git_commit"]
    timestamp = datetime.now(UTC).isoformat()

    N = spec["N"]
    D = spec["D"]
    k = spec.get("k", 10)
    index_type = spec["index_type"]
    build_params = spec["build_params"]
    search_grid = spec["search_grid"]
    seed = spec.get("seed", 0)

    # Which split to MEASURE on. Parameters are chosen on tuning queries (default);
    # headline numbers are re-measured on test queries (measure_split="test"). The query
    # split is POSITIONAL (decision M8): tuning = positions 0..1999, test = 2000..9999.
    measure_split = spec.get("measure_split", "tuning")

    split = load_query_split(spec["query_split_path"])
    tuning_indices = list(range(len(split["tuning"])))
    test_indices = list(range(len(split["tuning"]), len(split["tuning"]) + len(split["test"])))

    passages = np.memmap(spec["passages_path"], dtype=np.float32, mode="r", shape=(N, D))
    all_queries = np.memmap(spec["queries_path"], dtype=np.float32, mode="r",
                            shape=(spec.get("n_queries", 10_000), D))

    tuning_queries = np.ascontiguousarray(np.array(all_queries[tuning_indices]))
    test_queries = np.ascontiguousarray(np.array(all_queries[test_indices]))

    gt_data = np.load(spec["gt_path"])
    gt_ids = gt_data["ids"]
    gt_distances = gt_data["distances"]
    gt_dists_tuning = gt_distances[tuning_indices]
    gt_dists_test = gt_distances[test_indices]

    if measure_split == "test":
        measure_queries = test_queries
        gt_dists_measure = gt_dists_test
    else:
        measure_queries = tuning_queries
        gt_dists_measure = gt_dists_tuning

    passage_arr = np.array(passages[:N])

    logger.info(f"Building {index_type} index: N={N}, D={D}, params={build_params}")

    from annstudy import indexes as idx_mod

    build_fn = {
        "flat": lambda: idx_mod.build_flat(passage_arr, D),
        "hnsw": lambda: idx_mod.build_hnsw(passage_arr, D, **build_params),
        "hnsw_sq": lambda: idx_mod.build_hnsw_sq(passage_arr, D, seed=seed, **build_params),
        "ivf_flat": lambda: idx_mod.build_ivf_flat(passage_arr, D, seed=seed, **build_params),
        "ivf_pq": lambda: idx_mod.build_ivf_pq(passage_arr, D, seed=seed, **build_params),
        "ivf_pq_refine": lambda: idx_mod.build_ivf_pq_refine(passage_arr, D, seed=seed, **build_params),
    }

    index, build_meta = build_fn[index_type]()

    index_save_path = spec.get("index_save_path")
    index_bytes = 0
    if index_save_path:
        Path(index_save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_save_path)
        index_bytes = os.path.getsize(index_save_path)
        # Serialized size is a measured quantity, but the FILE is only needed by a later stage
        # (Stage C reuses Stage B indexes) or the memory pass. Otherwise delete it now rather
        # than holding it for the whole sweep: at N=500K/D=1536 that is ~3 GB per concurrent
        # runner, which would exhaust the disk long before the sweeps finish.
        if not spec.get("keep_index", False):
            Path(index_save_path).unlink(missing_ok=True)
            logger.info(f"Recorded index_bytes={index_bytes} and removed {index_save_path}")

    rss_after_build = measure_rss()  # noqa: F841 (kept for symmetry with warmup RSS)

    warmup_n = spec.get("measurement", {}).get("warmup_queries", 200)
    warmup_q = tuning_queries[:warmup_n]
    for i in range(warmup_n):
        index.search(warmup_q[i : i + 1], k)

    rss_after_warmup = measure_rss()

    chash = config_hash({
        "index_type": index_type, "N": N, "D": D,
        "build_params": build_params, "seed": seed,
    })
    bkey = build_config_key(index_type, N, D, build_params)

    build_row = {
        "run_id": run_id,
        "git_commit": git_commit,
        "hardware_id": hardware_id,
        "timestamp": timestamp,
        "seed": seed,
        "config_hash": chash,
        "build_config_key": bkey,
        "index_type": index_type,
        "N": N,
        "D": D,
        "nlist": build_params.get("nlist"),
        "m": build_params.get("m"),
        "M": build_params.get("M"),
        "ef_construction": build_params.get("ef_construction"),
        "nbits": build_params.get("nbits"),
        "train_size": build_meta["train_size"],
        "train_seconds": build_meta["train_seconds"],
        "add_seconds": build_meta["add_seconds"],
        "build_seconds": build_meta["build_seconds"],
        "threads": threads,
        "index_bytes": index_bytes,
        "rss_bytes": rss_after_warmup,
        "repeat_id": spec.get("repeat_id", 0),
        "selected": False,
        "superseded_by": None,
    }

    results_dir = Path(spec.get("results_dir", "results/raw"))
    builds_dir = results_dir / "builds"
    sweeps_dir = results_dir / "sweeps"
    builds_dir.mkdir(parents=True, exist_ok=True)
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    # In test re-measurement mode the build already has a recorded row from pass 1;
    # don't write a duplicate (would create two rows with the same config_hash).
    if spec.get("write_build_row", True):
        pd.DataFrame([build_row]).to_parquet(builds_dir / f"build_{run_id}.parquet")
        logger.info(f"Build row written: {run_id}")

    measurement = spec.get("measurement", {})
    n_passes = measurement.get("latency_passes", 3)
    n_resamples = measurement.get("bootstrap_resamples", 1000)
    # Default: measure QPS only when running multi-threaded (the plan's throughput condition).
    measure_qps_flag = bool(spec.get("measure_qps", threads > 1))
    # Per-query latency is only meaningful in the pinned single-thread pass. In the T-thread
    # build/throughput pass it is discarded by finalize_tuned anyway, and measuring it over
    # 8000 test queries x 3 passes is the single most expensive thing in that pass, so it is
    # skipped by default there.
    measure_latency_flag = bool(spec.get("measure_latency", threads == 1))

    sweep_rows = []
    for search_params in search_grid:
        set_search_params(index, index_type, search_params)

        if measure_latency_flag:
            per_query_ns = measure_latency(index, measure_queries, k=k, n_passes=n_passes)
            lat = latency_stats(per_query_ns)
            _, lat_p95_lo, lat_p95_hi = bootstrap_latency_percentile(
                per_query_ns, percentile=95, n_resamples=n_resamples,
            )
        else:
            nan = float("nan")
            lat = {"lat_mean_us": nan, "lat_p50_us": nan,
                   "lat_p95_us": nan, "lat_p99_us": nan}
            lat_p95_lo = lat_p95_hi = nan

        _, ret_ids = index.search(measure_queries, k)
        per_q_recall = recall_at_k_batch(
            ret_ids, gt_dists_measure, passage_arr, measure_queries, k=k,
        )
        recall_mean, recall_lo, recall_hi = bootstrap_recall(per_q_recall, n_resamples)

        # QPS is a T-thread measurement (plan section 3). In the pinned 1-thread latency pass it
        # would be a different quantity, so it is recorded as NaN there rather than conflated.
        if measure_qps_flag:
            qps = measure_qps(index, measure_queries, k=k, n_passes=n_passes)
        else:
            qps = float("nan")

        row = {
            "run_id": str(uuid.uuid4()),
            "build_run_id": run_id,
            "git_commit": git_commit,
            "hardware_id": hardware_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "seed": seed,
            "config_hash": chash,
            "build_config_key": bkey,
            "index_type": index_type,
            "N": N,
            "D": D,
            "k": k,
            "ef_search": search_params.get("ef_search"),
            "nprobe": search_params.get("nprobe"),
            "k_factor": search_params.get("k_factor"),
            "query_split": measure_split,
            "recall_mean": recall_mean,
            "recall_ci_lo": recall_lo,
            "recall_ci_hi": recall_hi,
            **lat,
            "lat_p95_ci_lo": lat_p95_lo,
            "lat_p95_ci_hi": lat_p95_hi,
            "qps": qps,
            "threads": threads,
        }
        sweep_rows.append(row)
        logger.info(
            f"  {search_params}: recall={recall_mean:.4f}, "
            f"p95={lat['lat_p95_us']:.0f}us, qps={qps:.0f}"
        )

    if sweep_rows:
        pd.DataFrame(sweep_rows).to_parquet(sweeps_dir / f"sweep_{run_id}.parquet")
        logger.info(f"Sweep rows written: {len(sweep_rows)} settings")

    return {"run_id": run_id, "build_row": build_row, "n_sweep_rows": len(sweep_rows)}


def run_memory_spec(spec: dict) -> dict:
    """Measure an index's incremental RAM in a fresh process (plan section 3, gap M11).

    RSS recorded after an in-process build is inflated by the N x D passage array, so memory is
    measured here instead: baseline RSS after imports, ``faiss.read_index`` (no mmap), the
    200-query warm-up, then RSS again. ``rss_bytes`` is the DELTA, which is what the memory
    budgets in ``analysis/winners.py`` compare against ``budget x N x D x 4``.
    """
    from annstudy.faiss_setup import ensure_opt_level
    ensure_opt_level()

    import faiss

    faiss.omp_set_num_threads(int(spec.get("threads", 1)))

    from annstudy.data import load_query_split
    from annstudy.hardware import snapshot

    index_path = spec["index_path"]
    N, D, k = spec["N"], spec["D"], spec.get("k", 10)

    # Touch the query memmap BEFORE the baseline so the warm-up cannot charge query pages to
    # the index's delta.
    split = load_query_split(spec["query_split_path"])
    n_warmup = spec.get("measurement", {}).get("warmup_queries", 200)
    all_queries = np.memmap(spec["queries_path"], dtype=np.float32, mode="r",
                            shape=(spec.get("n_queries", 10_000), D))
    warmup_q = np.ascontiguousarray(np.array(all_queries[:n_warmup]))
    del all_queries, split

    rss_baseline = measure_rss()
    index = faiss.read_index(index_path)  # no IO_FLAG_MMAP: we want it resident
    for i in range(n_warmup):
        index.search(warmup_q[i : i + 1], k)
    rss_after = measure_rss()

    hw = snapshot()
    row = {
        "run_id": spec.get("run_id", str(uuid.uuid4())),
        "git_commit": hw["git_commit"],
        "hardware_id": hw["hardware_id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "config_hash": spec.get("config_hash"),
        "build_config_key": spec.get("build_config_key"),
        "build_run_id": spec.get("build_run_id"),
        "index_type": spec["index_type"],
        "N": N,
        "D": D,
        "seed": spec.get("seed", 0),
        "repeat_id": spec.get("repeat_id", 0),
        "index_bytes": os.path.getsize(index_path),
        "rss_baseline_bytes": rss_baseline,
        "rss_loaded_bytes": rss_after,
        "rss_bytes": max(0, rss_after - rss_baseline),
        "raw_bytes": N * D * 4,
        "threads": int(spec.get("threads", 1)),
    }

    out_dir = Path(spec.get("results_dir", "results/raw")) / "memory"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out_dir / f"mem_{row['run_id']}.parquet")
    logger.info(
        f"Memory: {spec['index_type']} N={N} D={D} "
        f"rss_delta={row['rss_bytes'] / 1e6:.1f} MB index_bytes={row['index_bytes'] / 1e6:.1f} MB"
    )
    return row


def spawn_runner(spec: dict, core: int = 0) -> subprocess.CompletedProcess:
    """Spawn a runner subprocess with CPU pinning and thread control."""
    spec_path = Path(f"/tmp/annstudy_spec_{uuid.uuid4().hex[:8]}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    threads = int(spec.get("threads", 1))
    tail = [sys.executable, "-m", "annstudy.runner", "--spec", str(spec_path)]
    if threads > 1:  # build/QPS pass: all cores, T threads, no pinning
        cmd = ["env", f"OMP_NUM_THREADS={threads}", *tail]
    else:  # latency pass: one thread pinned to one core
        cmd = ["taskset", "-c", str(core), "env", "OMP_NUM_THREADS=1", *tail]

    logger.info(f"Spawning runner: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    spec_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(f"Runner failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    return result


def is_already_run(
    chash: str,
    results_dir: str = "results/raw/builds",
    hardware_id: str | None = None,
) -> bool:
    """Check if a build with this config_hash already exists.

    ``hardware_id`` scopes the check to one machine. Latency, build time and throughput are
    properties of the host, so a run recorded on different hardware must not satisfy a spec
    here — otherwise a stage would silently inherit another machine's numbers.
    """
    builds_dir = Path(results_dir)
    if not builds_dir.exists():
        return False

    cols = ["config_hash", "superseded_by"]
    if hardware_id is not None:
        cols.append("hardware_id")

    for parquet_file in builds_dir.glob("build_*.parquet"):
        try:
            df = pd.read_parquet(parquet_file, columns=cols)
            match = df[(df["config_hash"] == chash) & (df["superseded_by"].isna())]
            if hardware_id is not None:
                match = match[match["hardware_id"] == hardware_id]
            if not match.empty:
                return True
        except Exception:
            continue
    return False


def main():
    """Subprocess entry point."""
    from annstudy.faiss_setup import ensure_opt_level
    ensure_opt_level()  # must run before faiss is imported anywhere in this process

    parser = argparse.ArgumentParser(description="ANN study runner subprocess")
    parser.add_argument("--spec", required=True, help="Path to spec JSON")
    parser.add_argument(
        "--mode", choices=["sweep", "memory"], default="sweep",
        help="sweep: build + search sweep. memory: load a serialized index and report RSS delta.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.spec) as f:
        spec = json.load(f)

    if args.mode == "memory":
        result = run_memory_spec(spec)
    else:
        result = run_single_spec(spec)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
