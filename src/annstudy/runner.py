"""Subprocess runner for isolated index builds and search sweeps.

Two-tier architecture:
- Coordinator: reads stage config, expands parameter grid, spawns runner subprocesses.
- Runner (this module's main()): builds one index, sweeps search params, writes Parquet.

The runner is invoked as a subprocess:
  taskset -c <core> env OMP_NUM_THREADS=1 python -m annstudy.runner --spec spec.json
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def config_hash(spec: dict) -> str:
    """SHA256 of the sorted JSON representation of a build spec."""
    canonical = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


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
    import faiss

    from annstudy.data import load_query_split
    from annstudy.hardware import load_snapshot, snapshot
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
    timestamp = datetime.now(timezone.utc).isoformat()

    N = spec["N"]
    D = spec["D"]
    k = spec.get("k", 10)
    index_type = spec["index_type"]
    build_params = spec["build_params"]
    search_grid = spec["search_grid"]
    seed = spec.get("seed", 0)

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
    gt_ids_tuning = gt_ids[tuning_indices]
    gt_dists_tuning = gt_distances[tuning_indices]

    passage_arr = np.array(passages[:N])

    logger.info(f"Building {index_type} index: N={N}, D={D}, params={build_params}")

    from annstudy import indexes as idx_mod

    build_fn = {
        "flat": lambda: idx_mod.build_flat(passage_arr, D),
        "hnsw": lambda: idx_mod.build_hnsw(passage_arr, D, **build_params),
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

    rss_after_build = measure_rss()

    warmup_n = spec.get("measurement", {}).get("warmup_queries", 200)
    warmup_q = tuning_queries[:warmup_n]
    for i in range(warmup_n):
        index.search(warmup_q[i : i + 1], k)

    rss_after_warmup = measure_rss()

    chash = config_hash({
        "index_type": index_type, "N": N, "D": D,
        "build_params": build_params, "seed": seed,
    })

    build_row = {
        "run_id": run_id,
        "git_commit": git_commit,
        "hardware_id": hardware_id,
        "timestamp": timestamp,
        "seed": seed,
        "config_hash": chash,
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
        "threads": int(os.environ.get("OMP_NUM_THREADS", 1)),
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

    pd.DataFrame([build_row]).to_parquet(builds_dir / f"build_{run_id}.parquet")
    logger.info(f"Build row written: {run_id}")

    measurement = spec.get("measurement", {})
    n_passes = measurement.get("latency_passes", 3)
    n_resamples = measurement.get("bootstrap_resamples", 1000)

    sweep_rows = []
    for search_params in search_grid:
        set_search_params(index, index_type, search_params)

        per_query_ns = measure_latency(index, tuning_queries, k=k, n_passes=n_passes)
        lat = latency_stats(per_query_ns)

        _, ret_ids = index.search(tuning_queries, k)
        per_q_recall = recall_at_k_batch(
            ret_ids, gt_dists_tuning, passage_arr, tuning_queries, k=k,
        )
        recall_mean, recall_lo, recall_hi = bootstrap_recall(per_q_recall, n_resamples)

        qps = measure_qps(index, tuning_queries, k=k, n_passes=n_passes)

        row = {
            "run_id": str(uuid.uuid4()),
            "build_run_id": run_id,
            "git_commit": git_commit,
            "hardware_id": hardware_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "config_hash": chash,
            "index_type": index_type,
            "N": N,
            "D": D,
            "k": k,
            "ef_search": search_params.get("ef_search"),
            "nprobe": search_params.get("nprobe"),
            "k_factor": search_params.get("k_factor"),
            "query_split": "tuning",
            "recall_mean": recall_mean,
            "recall_ci_lo": recall_lo,
            "recall_ci_hi": recall_hi,
            **lat,
            "qps": qps,
            "threads": int(os.environ.get("OMP_NUM_THREADS", 1)),
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


def spawn_runner(spec: dict, core: int = 0) -> subprocess.CompletedProcess:
    """Spawn a runner subprocess with CPU pinning and thread control."""
    spec_path = Path(f"/tmp/annstudy_spec_{uuid.uuid4().hex[:8]}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    cmd = [
        "taskset", "-c", str(core),
        "env", "OMP_NUM_THREADS=1",
        sys.executable, "-m", "annstudy.runner",
        "--spec", str(spec_path),
    ]

    logger.info(f"Spawning runner: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    spec_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(f"Runner failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    return result


def is_already_run(chash: str, results_dir: str = "results/raw/builds") -> bool:
    """Check if a build with this config_hash already exists."""
    builds_dir = Path(results_dir)
    if not builds_dir.exists():
        return False

    for parquet_file in builds_dir.glob("build_*.parquet"):
        try:
            df = pd.read_parquet(parquet_file, columns=["config_hash", "superseded_by"])
            match = df[
                (df["config_hash"] == chash)
                & (df["superseded_by"].isna())
            ]
            if not match.empty:
                return True
        except Exception:
            continue
    return False


def main():
    """Subprocess entry point."""
    parser = argparse.ArgumentParser(description="ANN study runner subprocess")
    parser.add_argument("--spec", required=True, help="Path to spec JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.spec) as f:
        spec = json.load(f)

    result = run_single_spec(spec)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
