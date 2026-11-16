"""Tests for runner module."""

import tempfile

import numpy as np


def test_config_hash_deterministic():
    """Same spec should always produce the same hash."""
    from annstudy.runner import config_hash

    spec = {"index_type": "hnsw", "N": 500_000, "D": 384, "M": 32}
    h1 = config_hash(spec)
    h2 = config_hash(spec)
    assert h1 == h2

    spec2 = {"D": 384, "index_type": "hnsw", "M": 32, "N": 500_000}
    h3 = config_hash(spec2)
    assert h1 == h3


def test_config_hash_different_for_different_specs():
    """Different specs should produce different hashes."""
    from annstudy.runner import config_hash

    h1 = config_hash({"index_type": "hnsw", "M": 32})
    h2 = config_hash({"index_type": "hnsw", "M": 64})
    assert h1 != h2


def test_disk_space_check():
    """Disk space check should work on a real filesystem."""
    from annstudy.runner import check_disk_space

    assert check_disk_space("/tmp", 1024)
    assert not check_disk_space("/tmp", 10**18)


def test_threads_come_from_spec_not_environment(tiny_spec, monkeypatch):
    """Thread count is a spec field (plan sections 1/3, decision M7), not an env guess.

    Latency is measured single-threaded while builds and QPS use T threads, so a stale
    OMP_NUM_THREADS must not be what lands in the recorded rows.
    """
    import pandas as pd

    from annstudy.runner import run_single_spec

    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # deliberately disagrees with the spec
    spec = tiny_spec(run_id="tthread", threads=2, measure_qps=True)
    run_single_spec(spec)

    builds = pd.read_parquet(f"{spec['results_dir']}/builds/build_tthread.parquet")
    sweeps = pd.read_parquet(f"{spec['results_dir']}/sweeps/sweep_tthread.parquet")
    assert int(builds["threads"].iloc[0]) == 2
    assert int(sweeps["threads"].iloc[0]) == 2


def test_qps_only_measured_multithreaded(tiny_spec):
    """QPS is a T-thread quantity; the pinned 1-thread pass must not report one."""
    import numpy as np
    import pandas as pd

    from annstudy.runner import run_single_spec

    single = tiny_spec(run_id="single", threads=1)
    run_single_spec(single)
    rows = pd.read_parquet(f"{single['results_dir']}/sweeps/sweep_single.parquet")
    assert np.isnan(rows["qps"].iloc[0]), "1-thread pass should leave qps unset"

    multi = tiny_spec(run_id="multi", threads=2, measure_qps=True)
    run_single_spec(multi)
    rows = pd.read_parquet(f"{multi['results_dir']}/sweeps/sweep_multi.parquet")
    assert rows["qps"].iloc[0] > 0


def test_memory_mode_reports_delta_not_process_total(tiny_spec, tmp_path):
    """Regression for gap M11: rss_bytes must be the index's incremental RAM.

    The build-time RSS is whole-process and inflated by the N x D passage array, which made
    every memory-budget winner come out 'none'. The memory pass loads only the serialized
    index, so its delta must be far below the absolute RSS of that process.
    """
    import json
    import subprocess
    import sys

    import pandas as pd

    from annstudy.runner import run_single_spec

    index_path = tmp_path / "idx.faiss"
    build = tiny_spec(run_id="forMem", index_save_path=str(index_path), keep_index=True)
    run_single_spec(build)
    assert index_path.exists()

    mem_spec = {
        "run_id": "memrun", "index_path": str(index_path),
        "build_config_key": "abc123", "index_type": "ivf_flat",
        "N": build["N"], "D": build["D"], "k": 10,
        "queries_path": build["queries_path"],
        "query_split_path": build["query_split_path"],
        "n_queries": build["n_queries"],
        "measurement": {"warmup_queries": 10},
        "results_dir": build["results_dir"], "threads": 1,
    }
    spec_path = tmp_path / "mem_spec.json"
    spec_path.write_text(json.dumps(mem_spec))

    # Must be a FRESH process: measured in-process the delta is absorbed by allocator slack
    # from earlier work, which is exactly why the production path spawns a subprocess.
    proc = subprocess.run(
        [sys.executable, "-m", "annstudy.runner", "--mode", "memory", "--spec", str(spec_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    written = pd.read_parquet(f"{build['results_dir']}/memory/mem_memrun.parquet")
    row = written.iloc[0]
    assert written["build_config_key"].iloc[0] == "abc123"
    assert row["rss_bytes"] > 0
    assert row["rss_bytes"] < row["rss_loaded_bytes"], "delta must be less than process total"
    # A flat-storage IVF index is ~N*D*4, so the resident delta should be in that ballpark and
    # not the hundreds of MB a whole-process reading reports.
    assert row["rss_bytes"] >= 0.3 * row["index_bytes"]
    assert row["rss_bytes"] <= 12 * row["index_bytes"]
    assert row["raw_bytes"] == build["N"] * build["D"] * 4


def test_rss_increases_after_load(toy_vectors):
    """RSS should increase after loading a FAISS index."""
    import faiss

    from annstudy.runner import measure_rss

    rss_before = measure_rss()

    dim = toy_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        faiss.write_index(index, f.name)
        index_path = f.name

    del index
    loaded = faiss.read_index(index_path)
    _ = loaded.search(toy_vectors[:1], 5)

    rss_after = measure_rss()
    assert rss_after >= rss_before, "RSS should not decrease after loading an index"

    import os
    os.unlink(index_path)
