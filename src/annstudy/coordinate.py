"""Stage coordinator: expand a stage config into runner specs and drive them.

Reads a ``configs/stage_*.yaml``, enumerates (N, D) cells x index types x build-parameter
grids (via :mod:`annstudy.indexes`), and runs each as an isolated ``annstudy.runner``
subprocess, one pinned per physical core. Resumes by skipping specs whose build already
exists (``is_already_run``). Then tunes, runs repeat builds for the selected winners, and
writes a per-stage report.

The grid functions in :mod:`annstudy.indexes` are the authoritative parameter grids
(decision M4); the YAML ``indexes`` block is treated as a selection list only.

CLI::

    annstudy-run-stage --config configs/stage_a.yaml --phase all --cores 0-15
"""

import argparse
import json
import logging
import os
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from annstudy import indexes as idx_mod
from annstudy.runner import (
    build_config_key,
    check_disk_space,
    config_hash,
    is_already_run,
)

logger = logging.getLogger(__name__)

EMB_DIR = "data/emb"
GT_DIR = "data/gt"
QUERY_SPLIT_PATH = "data/query_split.json"
DEFAULT_N_QUERIES = 10_000


# --------------------------------------------------------------------------- config

def load_stage_config(path: str | Path) -> dict:
    """Load and return a stage YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def passages_path(D: int) -> str:
    return str(Path(EMB_DIR) / f"passages_D{D}.f32")


def queries_path(D: int) -> str:
    return str(Path(EMB_DIR) / f"queries_D{D}.f32")


def gt_path(N: int, D: int) -> str:
    return str(Path(GT_DIR) / f"gt_N{N}_D{D}.npz")


# ----------------------------------------------------------------------- enumeration

def stage_cells(cfg: dict) -> set[tuple[int, int]]:
    """The (N, D) cells this stage config declares.

    The measurement phases are scoped to these. Without that, running `--phase all` on a
    two-cell config re-measured the winners of every earlier stage: a six-spec probe spent
    hours re-measuring Stages A and B, and wrote latency rows taken under a different machine
    occupancy than the campaign that produced the reported numbers (observed disagreements up
    to 3.8x). Those rows are quarantined under
    `results/raw/sweeps_superseded_offcampaign/`.
    """
    cells = cfg.get("cells") or {}
    if "N" in cells and "D" in cells:
        return {(int(N), int(D)) for N in cells["N"] for D in cells["D"]}
    cell = cfg.get("cell") or {}
    if "N" in cell and "D" in cell:
        return {(int(cell["N"]), int(cell["D"]))}
    return set()


def stage_index_types(cfg: dict) -> set[str]:
    """The index families this stage config declares."""
    return {str(k) for k in (cfg.get("indexes") or {})}


def _restrict_to_cells(tuned: pd.DataFrame, cells: set[tuple[int, int]] | None,
                       what: str, index_types: set[str] | None = None) -> pd.DataFrame:
    """Restrict winners to this stage's cells AND index families.

    Cells alone are not enough: the tuned table holds every family ever measured at a cell, so
    a two-family probe at two cells still picked up 19 winners and would have re-measured the
    other families — including a 1.5 h exact-scan job — writing duplicate rows for results that
    were already measured in their own campaign.
    """
    if tuned is None or tuned.empty:
        return tuned
    keep = pd.Series(True, index=tuned.index)
    if cells:
        keep &= tuned.apply(lambda r: (int(r["N"]), int(r["D"])) in cells, axis=1)
    if index_types:
        keep &= tuned["index_type"].isin(index_types)
    dropped = int((~keep).sum())
    if dropped:
        logger.info(f"{what}: restricted to this stage's "
                    f"{len(cells) if cells else 'all'} cell(s) and "
                    f"{sorted(index_types) if index_types else 'all'} family(ies); "
                    f"ignoring {dropped} winner row(s) owned by other stages.")
    return tuned[keep]


def stage_c_retention_set(path: str = "configs/stage_c.yaml") -> set[tuple[int, int, str]]:
    """The (N, D, index_type) triples Stage C will reuse, read from its own config.

    Plan Stage B says to keep the serialized HNSW and IVF-Flat indexes for the large cells so
    Stage C can apply predicates at search time without rebuilding. It does NOT mean keeping
    every build variant: at Stage B's full grid that is ~205 GB against 43 GB of disk, which is
    how the Stage A repeat pass filled the disk. Retaining only what Stage C names is ~23 GB.
    Deriving the set from `configs/stage_c.yaml` keeps one source of truth.
    """
    p = Path(path)
    if not p.exists():
        return set()
    cfg = yaml.safe_load(p.read_text())
    out = set()
    for N in cfg.get("cells", {}).get("N", []):
        for D in cfg.get("cells", {}).get("D", []):
            for it in cfg.get("indexes", {}):
                out.add((int(N), int(D), str(it)))
    return out

def skip_hash(spec: dict) -> str:
    """Config hash used for skip/resume. Must mirror what the runner records."""
    return config_hash({
        "index_type": spec["index_type"],
        "N": spec["N"],
        "D": spec["D"],
        "build_params": spec["build_params"],
        "seed": spec["seed"],
    })


def enumerate_specs(
    cfg: dict,
    results_dir: str = "results/raw",
    stage: str | None = None,
    seed: int = 0,
    max_build_variants: int | None = None,
    keep_indexes: bool | None = None,
) -> list[dict]:
    """Expand a stage config into a flat list of runner spec dicts (pass 1, repeat_id=0).

    ``max_build_variants`` (if set) truncates each index type's build grid to that many
    variants — used for a quick pilot ("one build variant per index").
    """
    stage = stage or cfg.get("stage", "X")
    # Pass 1 NEVER retains its index. The runner records index_bytes and deletes the file; at
    # Stage B's full grid, retaining them all would want ~205 GB. A stage whose indexes a later
    # stage reuses retains only those, in the repeat/memory phase — see stage_c_retention_set.
    keep_index = False
    _ = keep_indexes
    k = cfg["k"]
    n_queries = cfg.get("n_queries", DEFAULT_N_QUERIES)
    measurement = cfg.get("measurement", {})
    index_dir = Path("data/indexes") / f"stage_{stage.lower()}"

    specs: list[dict] = []
    for N in cfg["cells"]["N"]:
        for D in cfg["cells"]["D"]:
            for index_type in cfg["indexes"]:
                build_grid = idx_mod.get_build_grid(index_type, N, D)
                if max_build_variants is not None:
                    build_grid = build_grid[:max_build_variants]
                for build_params in build_grid:
                    nlist = build_params.get("nlist")
                    search_grid = idx_mod.get_search_grid(index_type, nlist=nlist, k=k)
                    run_id = uuid.uuid4().hex
                    specs.append({
                        "run_id": run_id,
                        "stage": stage,
                        "N": N,
                        "D": D,
                        "k": k,
                        "index_type": index_type,
                        "build_params": build_params,
                        "search_grid": search_grid,
                        "seed": seed,
                        "repeat_id": 0,
                        "query_split_path": QUERY_SPLIT_PATH,
                        "passages_path": passages_path(D),
                        "queries_path": queries_path(D),
                        "n_queries": n_queries,
                        "gt_path": gt_path(N, D),
                        "index_save_path": str(index_dir / f"idx_{run_id}.faiss"),
                        "measurement": measurement,
                        "results_dir": results_dir,
                        # Pass 1 is the pinned single-thread latency sweep; throughput and
                        # build time come from the T-thread pass (plan section 3).
                        "threads": 1,
                        "measure_qps": False,
                        "keep_index": keep_index,
                    })
    return specs


def assert_prerequisites(specs: list[dict]) -> None:
    """Fail fast with a clear message if any required ground-truth file is missing (M9)."""
    missing = sorted({s["gt_path"] for s in specs if not Path(s["gt_path"]).exists()})
    if missing:
        raise FileNotFoundError(
            "Missing ground-truth files required by this stage:\n  "
            + "\n  ".join(missing)
            + "\nGenerate them with `annstudy-gt` before running."
        )


def _expected_index_bytes(spec: dict) -> int:
    """Rough serialized-size estimate for the disk-space pre-check.

    Tolerates specs without ``build_params`` (memory-mode specs describe an index that already
    exists on disk rather than one to be built).
    """
    N, D = spec["N"], spec["D"]
    it = spec["index_type"]
    bp = spec.get("build_params") or {}
    if it == "ivf_pq":
        return N * bp.get("m", D // 8)
    if it == "ivf_pq_refine":
        return N * D * 4 + N * bp.get("m", D // 8)
    return N * D * 4  # flat / hnsw / ivf_flat keep full vectors


ANALYSIS_HOST_ENV = "ANNSTUDY_ANALYSIS_HARDWARE_ID"


@lru_cache(maxsize=1)
def current_hardware_id() -> str:
    """The hardware id that host-scoped reads (tuning, winners, paper) select rows by.

    Normally this host's id. ``ANNSTUDY_ANALYSIS_HARDWARE_ID`` overrides it so the analysis and
    paper for the study host can be regenerated from another machine (e.g. a laptop or the
    aarch64 box) instead of silently scoping to that machine's — empty — rows. Measurement rows
    are always stamped with the real host id by ``hardware.snapshot()``; never set this while
    measuring.
    """
    import os

    from annstudy.hardware import snapshot
    return os.environ.get(ANALYSIS_HOST_ENV) or snapshot()["hardware_id"]


def _expected_train_bytes(spec: dict) -> int:
    """Bytes held by the IVF/PQ training sample, which the factories copy out of the corpus.

    This is NOT negligible and it is alive at the same time as the index: the factories keep
    ``train_vecs`` in scope through ``add()``. At N=500K/D=1536 with nlist >= 2828 the sample
    is the whole corpus, i.e. another 3.07 GB. Omitting it under-estimated real RSS by up to
    2.5x and would have oversubscribed memory on the largest cells.
    """
    it = spec["index_type"]
    N, D = spec["N"], spec["D"]
    nlist = (spec.get("build_params") or {}).get("nlist")
    if not nlist or it in ("flat", "hnsw"):
        return 0
    if it == "ivf_flat":  # indexes.build_ivf_flat
        train_size = min(N, 256 * nlist)
    else:  # build_ivf_pq / build_ivf_pq_refine
        train_size = min(N, max(256 * nlist, 256_000))
    return train_size * D * 4


def _expected_rss(spec: dict, overhead_bytes: int = 500_000_000) -> int:
    """Estimate one runner's peak RSS, used by run_pool's admission control.

    Peak is the sum of three live allocations: the N x D passage array the runner materialises
    (needed to build and to recompute true distances for recall), the training sample, and the
    index itself. At N=500K/D=1536 that is ~9.7 GB for ivf_pq_refine, so a 16-wide pool would
    want far more than this 59 GB machine has.

    A memory-mode spec is different: that process deliberately loads ONLY the serialized index
    (that is the measurement), so it costs the index plus overhead.
    """
    if spec.get("mode") == "memory":
        return _expected_index_bytes(spec) + overhead_bytes
    return (spec["N"] * spec["D"] * 4
            + _expected_train_bytes(spec)
            + _expected_index_bytes(spec)
            + overhead_bytes)


def default_mem_budget(fraction: float = 0.80) -> int:
    """Bytes of RAM the pool may commit, as a fraction of what is currently available."""
    import psutil
    vm = psutil.virtual_memory()
    # Page cache is reclaimable, so budget against available + cached rather than free.
    return int(max(vm.available, vm.total * 0.5) * fraction)


def should_skip(spec: dict, results_dir: str = "results/raw") -> bool:
    return is_already_run(skip_hash(spec), str(Path(results_dir) / "builds"),
                          hardware_id=current_hardware_id())


# -------------------------------------------------------------------------- spawning

def detect_cores() -> list[int]:
    import psutil
    n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    return list(range(n))


def default_threads(cap: int = 16) -> int:
    """T for build/throughput measurement: min(physical cores, cap) per plan section 1."""
    import psutil
    n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    return max(1, min(n, cap))


def parse_cores(spec_str: str | None) -> list[int]:
    """Parse '0-7' or '0,2,4' into a list of core ids; None -> all physical cores."""
    if not spec_str:
        return detect_cores()
    cores: list[int] = []
    for part in spec_str.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return cores


def _spawn(spec: dict, core: int, log_dir: Path):
    """Launch one runner subprocess. Returns (Popen, spec_path, log_path).

    Thread policy (plan sections 1 and 3, decision M7): a spec with ``threads == 1`` is a
    latency job and is pinned to ``core`` with ``OMP_NUM_THREADS=1``. A spec with
    ``threads > 1`` is a build/throughput job and runs unpinned across all cores, so the
    caller must schedule those one at a time.
    """
    fd, spec_path = tempfile.mkstemp(suffix=".json", prefix="annstudy_spec_")
    with os.fdopen(fd, "w") as f:
        json.dump(spec, f, default=str)

    from annstudy.faiss_setup import ensure_opt_level
    ensure_opt_level()  # propagate FAISS_OPT_LEVEL to the child so faiss skips the AVX2 probe

    threads = int(spec.get("threads", 1))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec['run_id']}.log"
    log_fh = open(log_path, "w")

    mode_args = ["--mode", spec["mode"]] if spec.get("mode") else []
    tail = [sys.executable, "-m", "annstudy.runner", "--spec", spec_path, *mode_args]

    pin = threads == 1
    use_taskset = pin and _platform.system() == "Linux" and shutil.which("taskset")
    cmd = ["taskset", "-c", str(core), *tail] if use_taskset else tail

    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    proc._log_fh = log_fh  # keep handle alive for close

    if pin and not use_taskset:
        try:  # best-effort pinning on Windows / non-taskset platforms
            import psutil
            psutil.Process(proc.pid).cpu_affinity([core])
        except Exception as e:  # pragma: no cover - platform dependent
            logger.debug(f"cpu_affinity pin failed on core {core}: {e}")

    return proc, spec_path, log_path


def run_pool(
    specs: list[dict],
    cores: list[int],
    results_dir: str = "results/raw",
    keep_indexes: bool = False,
    dry_run: bool = False,
    skip_done: bool = True,
    mem_budget_bytes: int | None = None,
) -> list[dict]:
    """Run specs across a fixed worker pool, one pinned runner per core. Resumable.

    Admission control: a core stays idle rather than starting a spec whose estimated RSS would
    push the pool over ``mem_budget_bytes``. Without this, the large cells oversubscribe memory
    (16 x 6 GB at N=500K/D=1536) and the run dies part-way through instead of simply using
    fewer cores there. Specs are admitted largest-first within what fits, and a single spec
    bigger than the whole budget still runs alone rather than deadlocking.
    """
    pending = [s for s in specs if not (skip_done and should_skip(s, results_dir))]
    skipped = len(specs) - len(pending)
    logger.info(f"{len(specs)} specs enumerated; {skipped} already done; {len(pending)} to run.")

    if dry_run:
        return [{"run_id": s["run_id"], "index_type": s["index_type"], "N": s["N"], "D": s["D"],
                 "n_search": len(s.get("search_grid", [])),
                 "threads": s.get("threads", 1), "mode": s.get("mode", "sweep")}
                for s in pending]

    log_dir = Path("logs/runner")
    failures_path = Path("logs/failures.jsonl")
    results: list[dict] = []
    queue = list(pending)
    free = list(cores)
    running: dict[int, tuple] = {}

    if mem_budget_bytes is None:
        mem_budget_bytes = default_mem_budget()
    committed = 0
    logger.info(f"Memory budget for this pool: {mem_budget_bytes / 1e9:.1f} GB; "
                f"largest spec needs {max((_expected_rss(s) for s in pending), default=0) / 1e9:.1f} GB")

    while queue or running:
        while free and queue:
            # Pick the first queued spec that fits the remaining memory budget. An oversized
            # spec is admitted only when nothing else is running, so it runs alone instead of
            # deadlocking the pool.
            pick = None
            for i, spec in enumerate(queue):
                need = _expected_rss(spec)
                if not committed or committed + need <= mem_budget_bytes:
                    pick = i
                    break
            if pick is None:
                break  # nothing fits right now; wait for a running job to finish

            spec = queue.pop(pick)
            need = _expected_rss(spec)
            isp = spec.get("index_save_path")
            if isp:
                if not check_disk_space(str(Path(isp).parent.parent),
                                        _expected_index_bytes(spec)):
                    logger.error(f"Skipping {spec['run_id']}: insufficient disk.")
                    continue
                Path(isp).parent.mkdir(parents=True, exist_ok=True)

            core = free.pop(0)
            proc, spec_path, log_path = _spawn(spec, core, log_dir)
            running[core] = (proc, spec, spec_path, log_path, time.time(), need)
            committed += need
            logger.info(f"[core {core}] start {spec.get('mode', 'sweep')} {spec['index_type']} "
                        f"N={spec['N']} D={spec['D']} params={spec.get('build_params')} "
                        f"threads={spec.get('threads', 1)} "
                        f"est={need / 1e9:.1f}GB committed={committed / 1e9:.1f}GB "
                        f"queued={len(queue)}")

        time.sleep(0.3)

        for core, (proc, spec, spec_path, log_path, t0, need) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            proc._log_fh.close()
            Path(spec_path).unlink(missing_ok=True)
            elapsed = time.time() - t0
            if rc == 0:
                logger.info(f"[core {core}] done {spec['run_id']} in {elapsed:.1f}s")
                results.append({"run_id": spec["run_id"], "rc": 0, "seconds": elapsed})
                isp = spec.get("index_save_path")
                if isp and not keep_indexes:
                    Path(isp).unlink(missing_ok=True)
            else:
                tail = Path(log_path).read_text(errors="replace")[-2000:]
                logger.error(f"[core {core}] FAILED {spec['run_id']} rc={rc}; log tail:\n{tail}")
                with open(failures_path, "a") as f:
                    f.write(json.dumps({
                        "run_id": spec["run_id"], "rc": rc, "index_type": spec["index_type"],
                        "N": spec["N"], "D": spec["D"],
                        "build_params": spec.get("build_params"),
                        "mode": spec.get("mode", "sweep"),
                        "log": str(log_path), "timestamp": datetime.now(UTC).isoformat(),
                    }) + "\n")
                results.append({"run_id": spec["run_id"], "rc": rc, "seconds": elapsed})
            del running[core]
            free.append(core)
            committed -= need

    return results


# ------------------------------------------------------------------------ tune/report

def run_tune(results_dir: str, recall_targets: list[float]) -> pd.DataFrame:
    """Pick cheapest settings on TUNING queries. Saved to results/tuned/tuned_tuning.parquet.

    The authoritative results/tuned/tuned.parquet is written later by finalize_tuned,
    after re-measuring the chosen winners on the test queries (decision M10).
    """
    from annstudy.tune import find_cheapest_setting, load_sweep_results
    sweep_df = load_sweep_results(str(Path(results_dir) / "sweeps"),
                                  hardware_id=current_hardware_id())
    tuned = find_cheapest_setting(sweep_df, recall_targets)
    out_dir = Path("results/tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    tuned.to_parquet(out_dir / "tuned_tuning.parquet", index=False)
    logger.info(f"Tuning winners saved: {out_dir / 'tuned_tuning.parquet'} ({len(tuned)} rows)")
    return tuned


def _winner_settings(group: pd.DataFrame, index_type: str) -> list[dict]:
    """Deduplicated list of search-setting dicts chosen for a winner build across R targets."""
    seen, out = set(), []
    for _, row in group.iterrows():
        if index_type in ("hnsw", "hnsw_sq"):
            s = {"ef_search": int(row["ef_search"])}
            key = ("ef", s["ef_search"])
        elif index_type in ("ivf_flat", "ivf_pq"):
            s = {"nprobe": int(row["nprobe"])}
            key = ("np", s["nprobe"])
        elif index_type == "ivf_pq_refine":
            s = {"nprobe": int(row["nprobe"]), "k_factor": int(row["k_factor"])}
            key = ("npkf", s["nprobe"], s["k_factor"])
        else:
            s, key = {}, ("flat",)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def run_test_remeasure(
    cfg: dict,
    tuned: pd.DataFrame,
    results_dir: str,
    cores: list[int],
    stage: str,
    cells: set[tuple[int, int]] | None = None,
) -> None:
    """Re-measure each tuning-chosen winner on the 8K TEST queries (decision M10, plan §3).

    Rebuilds each winning config (seed 0, so the config_hash matches pass 1), measures only
    the chosen search setting(s) on the test split, and appends sweep rows with
    query_split="test". No build row is written (write_build_row=False).
    """
    builds = _load_builds(results_dir)
    reach = tuned[tuned["reachable"]] if "reachable" in tuned else tuned
    reach = _restrict_to_cells(reach, cells, "test re-measurement",
                               index_types=stage_index_types(cfg))
    if builds.empty or reach.empty:
        logger.warning("No winners to re-measure on test queries.")
        return

    specs: list[dict] = []
    for brun, grp in reach.groupby("best_build_run_id"):
        brow = builds[builds["run_id"] == brun]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        index_type = brow["index_type"]
        N, D = int(brow["N"]), int(brow["D"])
        settings = _winner_settings(grp, index_type)
        specs.append({
            "run_id": uuid.uuid4().hex, "stage": stage, "N": N, "D": D, "k": cfg["k"],
            "index_type": index_type, "build_params": _reconstruct_build_params(brow),
            "search_grid": settings, "seed": 0, "repeat_id": 0,
            "measure_split": "test", "write_build_row": False, "index_save_path": None,
            "query_split_path": QUERY_SPLIT_PATH,
            "passages_path": passages_path(D), "queries_path": queries_path(D),
            "n_queries": cfg.get("n_queries", DEFAULT_N_QUERIES), "gt_path": gt_path(N, D),
            "measurement": cfg.get("measurement", {}), "results_dir": results_dir,
            "threads": 1, "measure_qps": False,  # headline latency: 1 thread, pinned
        })

    logger.info(f"Re-measuring {len(specs)} winner builds on the test split.")
    # Test re-measurement writes no build row, so skip-detection must be disabled.
    run_pool(specs, cores, results_dir=results_dir, keep_indexes=False, skip_done=False)


def finalize_tuned(results_dir: str, recall_targets: list[float]) -> pd.DataFrame:
    """Join tuning winners with their test-query re-measurement into results/tuned/tuned.parquet."""
    from annstudy.tune import find_cheapest_setting, load_sweep_results

    sweep_df = load_sweep_results(str(Path(results_dir) / "sweeps"),
                                  hardware_id=current_hardware_id())
    tuning_winners = find_cheapest_setting(
        sweep_df[sweep_df["query_split"] == "tuning"], recall_targets,
    )
    test_rows = sweep_df[sweep_df["query_split"] == "test"].copy()
    builds = _load_builds(results_dir)

    def _hash_of(build_run_id):
        row = builds[builds["run_id"] == build_run_id]
        return row["config_hash"].iloc[0] if not row.empty else None

    def _setting_match(trow, win):
        for col in ("ef_search", "nprobe", "k_factor"):
            wv = win.get(col)
            tv = trow.get(col)
            if pd.isna(wv) and pd.isna(tv):
                continue
            if pd.isna(wv) != pd.isna(tv) or (pd.notna(wv) and int(wv) != int(tv)):
                return False
        return True

    mem_rows = _load_memory_rows(results_dir)

    def _threads_of(row) -> int:
        t = row.get("threads")
        return int(t) if pd.notna(t) else 1

    # The T-thread pass rebuilds each winner with a NEW seed, and config_hash includes the
    # seed, so its throughput rows share no config_hash with the seed-0 winner. Resolve them
    # by the seed-independent build key instead: taken from the row if the runner recorded one,
    # otherwise derived from the row's own build row.
    key_by_run: dict[str, str] = {}
    if not builds.empty:
        for _, br in builds.iterrows():
            key_by_run[br["run_id"]] = build_config_key(
                br["index_type"], int(br["N"]), int(br["D"]), _reconstruct_build_params(br),
            )
    if not test_rows.empty:
        derived = test_rows["build_run_id"].map(key_by_run)
        if "build_config_key" in test_rows.columns:
            test_rows["build_config_key"] = test_rows["build_config_key"].fillna(derived)
        else:
            test_rows["build_config_key"] = derived

    out_rows = []
    for _, win in tuning_winners.iterrows():
        base = {c: win.get(c) for c in ["index_type", "N", "D", "k", "R", "reachable",
                                        "best_build_run_id", "ef_search", "nprobe", "k_factor"]}
        if not win.get("reachable", False):
            out_rows.append(base)
            continue
        chash = _hash_of(win["best_build_run_id"])
        cand = test_rows[test_rows["config_hash"] == chash] if chash else test_rows.iloc[0:0]
        matched = [r for _, r in cand.iterrows() if _setting_match(r, win)]
        brow = builds[builds["run_id"] == win["best_build_run_id"]]

        win_key = key_by_run.get(win["best_build_run_id"])
        key_cand = (test_rows[test_rows["build_config_key"] == win_key]
                    if win_key else test_rows.iloc[0:0])
        key_matched = [r for _, r in key_cand.iterrows() if _setting_match(r, win)]

        # Latency is the single-thread pinned measurement; throughput is the T-thread one
        # (plan section 3). Never let a T-thread row supply the headline latency.
        lat_rows = [r for r in matched if _threads_of(r) == 1]
        qps_rows = [r for r in key_matched if _threads_of(r) > 1]
        if lat_rows:
            # Several campaign measurements of the same configuration can exist (Stage A and
            # Stage B both re-measured shared cells, days apart). Taking the first one made the
            # table depend on file ordering: a re-finalize moved two rows by 1.26x without any
            # new measurement. Take the MEDIAN across duplicates instead, deterministic and
            # using all the evidence, and record the spread so the drift is visible.
            lat_frame = pd.DataFrame(lat_rows)
            n_meas = len(lat_frame)
            m = lat_frame.loc[lat_frame["lat_p95_us"].sort_values().index[n_meas // 2]]
            if n_meas > 1:
                base.update({
                    "n_test_measurements": n_meas,
                    "lat_p95_us_spread": float(lat_frame["lat_p95_us"].max()
                                               / lat_frame["lat_p95_us"].min()),
                })
            else:
                base["n_test_measurements"] = 1
            recall_test = float(m["recall_mean"])
            # The setting was chosen because it reached R on the TUNING queries; the held-out
            # test recall can land just below R. Make that explicit rather than leaving a
            # reader to infer "reachable at R" means "achieved R on test".
            base.update({
                "recall_test": recall_test,
                "recall_gap_vs_R": recall_test - float(win["R"]),
                "meets_R_on_test": bool(recall_test >= float(win["R"])),
                "lat_p95_us": m["lat_p95_us"],
                "lat_p95_ci_lo": m.get("lat_p95_ci_lo"),
                "lat_p95_ci_hi": m.get("lat_p95_ci_hi"),
                "lat_threads": _threads_of(m),
            })
        if qps_rows:
            base["qps"] = qps_rows[0].get("qps")
            base["qps_threads"] = _threads_of(qps_rows[0])

        cfg_key = None
        if not brow.empty:
            b0 = brow.iloc[0]
            cfg_key = build_config_key(
                b0["index_type"], int(b0["N"]), int(b0["D"]), _reconstruct_build_params(b0),
            )
            base["index_bytes"] = int(b0["index_bytes"])

        # Memory: prefer the dedicated measurement (RSS delta of an index-only process). The
        # build row's rss_bytes is whole-process RSS and is inflated by the passage array
        # (gap M11), so it is only a labelled fallback.
        mem = (mem_rows[mem_rows["build_config_key"] == cfg_key]
               if cfg_key and not mem_rows.empty and "build_config_key" in mem_rows
               else pd.DataFrame())
        if not mem.empty:
            base.update({
                "rss_bytes": int(mem["rss_bytes"].median()),
                "rss_source": "index_only_subprocess",
            })
            if pd.isna(base.get("index_bytes")):
                base["index_bytes"] = int(mem["index_bytes"].median())
        elif not brow.empty:
            base.update({
                "rss_bytes": int(brow["rss_bytes"].iloc[0]),
                "rss_source": "process_total_inflated",
            })

        # Build time must come from T-thread builds (plan sections 1 and 3). The seed-0 pass-1
        # build is single-threaded, so it is excluded from the reported statistics.
        same_cfg = builds[builds["config_hash"] == chash] if chash else builds.iloc[0:0]
        if cfg_key is not None and not builds.empty:
            keys = builds.apply(
                lambda r: build_config_key(
                    r["index_type"], int(r["N"]), int(r["D"]), _reconstruct_build_params(r),
                ), axis=1,
            )
            same_cfg = builds[keys == cfg_key]
        if not same_cfg.empty:
            tthread = same_cfg[same_cfg["threads"].fillna(1).astype(int) > 1]
            src = tthread if not tthread.empty else same_cfg
            bs = src["build_seconds"]
            base.update({
                "build_seconds_mean": float(bs.mean()),
                "build_seconds_min": float(bs.min()),
                "build_seconds_max": float(bs.max()),
                "n_builds": len(bs),
                "build_threads": int(src["threads"].fillna(1).astype(int).max()),
            })
        out_rows.append(base)

    final = pd.DataFrame(out_rows)
    out_dir = Path("results/tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_dir / "tuned.parquet", index=False)
    logger.info(f"Final tuned (test-measured) saved: {out_dir / 'tuned.parquet'} ({len(final)} rows)")
    return final


def _load_builds(results_dir: str, this_host_only: bool = True) -> pd.DataFrame:
    """Load build rows, by default restricted to this host (see current_hardware_id)."""
    builds_dir = Path(results_dir) / "builds"
    files = list(builds_dir.glob("build_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if this_host_only and "hardware_id" in df.columns:
        df = df[df["hardware_id"] == current_hardware_id()]
    if "superseded_by" in df.columns:
        df = df[df["superseded_by"].isna()]
    return df.reset_index(drop=True)


def run_repeat_builds(
    cfg: dict,
    tuned: pd.DataFrame,
    results_dir: str,
    cores: list[int],
    stage: str,
    n_repeats: int = 3,
    threads: int | None = None,
    measure_memory: bool = True,
    cells: set[tuple[int, int]] | None = None,
) -> None:
    """Rebuild each selected (tuned-winner) config n_repeats more times with new seeds.

    This is also the authoritative **T-thread** pass (plan sections 1 and 3): build time and
    throughput must be measured with T threads, so these jobs run one at a time, unpinned,
    with ``threads = T``.

    ``n_repeats`` defaults to 3 rather than 2 because the pass-1 build is single-threaded and
    so cannot contribute to a reported build time; all 3 of the builds that plan section 3
    ("Repeats") requires must therefore come from this pass, each with a different seed.

    When ``measure_memory`` is set, each repeat's serialized index is kept just long enough
    for a separate memory-measurement subprocess to load it (gap M11), then deleted.
    """
    builds = _load_builds(results_dir)
    if builds.empty or tuned.empty:
        logger.warning("No builds or no tuned winners; skipping repeat builds.")
        return

    reachable = tuned[tuned["reachable"]] if "reachable" in tuned else tuned
    reachable = _restrict_to_cells(reachable, cells, "repeat builds",
                                   index_types=stage_index_types(cfg))
    winner_ids = sorted(set(reachable.get("best_build_run_id", pd.Series(dtype=str)).dropna()))

    T = threads or default_threads()
    # Only retain indexes if this stage declares that a later stage reuses them.
    retain_set = stage_c_retention_set() if cfg.get("keep_indexes") else set()
    if retain_set:
        logger.info(f"Will retain winner indexes for {len(retain_set)} (N, D, index) triples "
                    f"that Stage C reuses.")
    repeat_specs: list[dict] = []
    selected_hashes: set[str] = set()
    index_dir = Path("data/indexes") / f"stage_{stage.lower()}"

    for brun in winner_ids:
        brow = builds[builds["run_id"] == brun]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        index_type = brow["index_type"]
        N, D = int(brow["N"]), int(brow["D"])
        build_params = _reconstruct_build_params(brow)
        selected_hashes.add(config_hash({
            "index_type": index_type, "N": N, "D": D,
            "build_params": build_params, "seed": 0,
        }))
        # Measure every setting this build won with, not just the highest-R one: the tuned
        # schema carries a qps per (cell, index, R), and QPS can only be measured here (it is
        # a T-thread quantity). One setting would leave the other recall targets without one.
        grp = reachable[reachable["best_build_run_id"] == brun]
        settings = _winner_settings(grp, index_type) or [
            _tuned_search_setting(reachable, brun, index_type)
        ]
        for r in range(1, n_repeats + 1):
            run_id = uuid.uuid4().hex
            repeat_specs.append({
                "run_id": run_id, "stage": stage, "N": N, "D": D, "k": cfg["k"],
                "index_type": index_type, "build_params": build_params,
                "search_grid": settings, "seed": r, "repeat_id": r,
                "query_split_path": QUERY_SPLIT_PATH,
                "passages_path": passages_path(D), "queries_path": queries_path(D),
                "n_queries": cfg.get("n_queries", DEFAULT_N_QUERIES),
                "gt_path": gt_path(N, D),
                "index_save_path": str(index_dir / f"idx_{run_id}.faiss"),
                "measurement": cfg.get("measurement", {}), "results_dir": results_dir,
                # T-thread build + throughput pass, measured on the test split so recall and
                # QPS are comparable with the headline numbers.
                "threads": T, "measure_qps": True, "measure_split": "test",
                # Resident memory is a property of the CONFIGURATION, not of the training
                # seed, so only one repeat per winner needs its index retained for the memory
                # measurement. Retaining all of them filled a 193 GB disk: 126 indexes at
                # N=500K is over 40 GB, and 62 builds were then skipped for lack of space.
                "keep_index": measure_memory and r == 1,
                "winner_build_run_id": brun,
            })

    if repeat_specs:
        logger.info(
            f"Running {len(repeat_specs)} repeat builds for {len(winner_ids)} winners "
            f"at {T} threads, one at a time (build time and QPS must be T-threaded)."
        )
        # Pool size 1: these jobs use every core, so they must not run concurrently.
        if not measure_memory:
            run_pool(repeat_specs, cores[:1], results_dir=results_dir, keep_indexes=False)
        else:
            # Interleave per winner: build its repeats, measure the retained index, delete it.
            # This bounds disk to one index at a time instead of one per repeat build.
            by_winner: dict[str, list[dict]] = {}
            for s in repeat_specs:
                by_winner.setdefault(s["winner_build_run_id"], []).append(s)
            for i, (brun, specs_w) in enumerate(by_winner.items(), start=1):
                logger.info(f"Winner {i}/{len(by_winner)} ({brun}): "
                            f"{len(specs_w)} repeat builds + memory measurement")
                run_pool(specs_w, cores[:1], results_dir=results_dir, keep_indexes=True)
                keepers = [s for s in specs_w if s.get("keep_index")]
                if keepers:
                    run_memory_pass(keepers[:1], results_dir=results_dir, cores=cores,
                                    threads=T, retain=retain_set)

    if selected_hashes:
        sel_path = Path(results_dir) / "selected.parquet"
        rows = [{"config_hash": h, "selected": True} for h in sorted(selected_hashes)]
        if sel_path.exists():
            prev = pd.read_parquet(sel_path)
            rows = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates("config_hash").to_dict("records")
        pd.DataFrame(rows).to_parquet(sel_path, index=False)
        logger.info(f"Recorded {len(selected_hashes)} selected winners in {sel_path}")


def run_memory_pass(
    built_specs: list[dict],
    results_dir: str,
    cores: list[int],
    threads: int = 1,
    retain: set[tuple[int, int, str]] | None = None,
) -> None:
    """Measure index RAM for each serialized index produced by ``built_specs`` (gap M11).

    Plan section 3 wants RAM = RSS(after read_index + warm-up) - RSS(baseline) in a process
    that loaded ONLY the index, so this runs one extra subprocess per index in ``--mode
    memory`` and then removes the serialized file.
    """
    mem_specs = []
    for spec in built_specs:
        isp = spec.get("index_save_path")
        if not isp or not Path(isp).exists():
            continue
        mem_specs.append({
            "run_id": uuid.uuid4().hex,
            "mode": "memory",
            "index_path": isp,
            "build_run_id": spec["run_id"],
            "config_hash": skip_hash(spec),
            "build_config_key": build_config_key(
                spec["index_type"], spec["N"], spec["D"], spec["build_params"],
            ),
            "index_type": spec["index_type"],
            "N": spec["N"], "D": spec["D"], "k": spec.get("k", 10),
            "seed": spec.get("seed", 0), "repeat_id": spec.get("repeat_id", 0),
            "query_split_path": spec["query_split_path"],
            "queries_path": spec["queries_path"],
            "n_queries": spec.get("n_queries", DEFAULT_N_QUERIES),
            "measurement": spec.get("measurement", {}),
            "results_dir": results_dir,
            "threads": threads,
        })

    if not mem_specs:
        logger.warning("No serialized indexes available for the memory pass.")
        return

    logger.info(f"Measuring index RAM for {len(mem_specs)} indexes.")
    # Memory jobs load a whole index each; run them one at a time to keep RSS honest.
    run_pool(mem_specs, cores[:1], results_dir=results_dir, keep_indexes=True, skip_done=False)

    # Delete each index once measured, EXCEPT those a later stage reuses (Stage C applies its
    # predicates to these at search time, so rebuilding them would be wasted work).
    retain = retain or set()
    for spec in mem_specs:
        key = (int(spec["N"]), int(spec["D"]), str(spec["index_type"]))
        if key in retain:
            logger.info(f"Retaining {spec['index_path']} for a later stage ({key})")
            continue
        Path(spec["index_path"]).unlink(missing_ok=True)


def run_memory_backfill(
    cfg: dict,
    tuned: pd.DataFrame,
    results_dir: str,
    cores: list[int],
    stage: str,
    threads: int | None = None,
) -> int:
    """Measure index RAM for any tuned winner that has no memory row yet.

    The memory measurement needs a serialized index, and the repeat pass only retains one per
    winner. If that pass was interrupted (or a winner's repeats were already complete from an
    earlier run), the index is gone and the winner has no memory row. Rather than rebuild the
    whole repeat pass, this rebuilds one index per missing winner, measures it, and deletes it.
    No build row is written: build time is already recorded.
    """
    builds = _load_builds(results_dir)
    mem = _load_memory_rows(results_dir)
    have = set(mem["build_config_key"]) if not mem.empty and "build_config_key" in mem else set()

    reachable = tuned[tuned["reachable"]] if "reachable" in tuned else tuned
    winner_ids = sorted(set(reachable.get("best_build_run_id", pd.Series(dtype=str)).dropna()))
    T = threads or default_threads()
    index_dir = Path("data/indexes") / f"stage_{stage.lower()}"

    missing: list[dict] = []
    for brun in winner_ids:
        brow = builds[builds["run_id"] == brun]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        index_type, N, D = brow["index_type"], int(brow["N"]), int(brow["D"])
        build_params = _reconstruct_build_params(brow)
        key = build_config_key(index_type, N, D, build_params)
        if key in have:
            continue
        have.add(key)  # avoid duplicating when several winners share a configuration
        run_id = uuid.uuid4().hex
        missing.append({
            "run_id": run_id, "stage": stage, "N": N, "D": D, "k": cfg["k"],
            "index_type": index_type, "build_params": build_params,
            "search_grid": [], "seed": 0, "repeat_id": 0,
            "query_split_path": QUERY_SPLIT_PATH,
            "passages_path": passages_path(D), "queries_path": queries_path(D),
            "n_queries": cfg.get("n_queries", DEFAULT_N_QUERIES),
            "gt_path": gt_path(N, D),
            "index_save_path": str(index_dir / f"idx_{run_id}.faiss"),
            "measurement": cfg.get("measurement", {}), "results_dir": results_dir,
            "threads": T, "measure_qps": False, "measure_latency": False,
            "write_build_row": False, "keep_index": True,
            "winner_build_run_id": brun,
        })

    if not missing:
        logger.info("Memory backfill: every tuned winner already has a memory row.")
        return 0

    logger.info(f"Memory backfill: {len(missing)} winner configuration(s) need an index built "
                f"to measure RAM.")
    for i, spec in enumerate(missing, start=1):
        logger.info(f"  backfill {i}/{len(missing)}: {spec['index_type']} "
                    f"N={spec['N']} D={spec['D']} {spec['build_params']}")
        run_pool([spec], cores[:1], results_dir=results_dir, keep_indexes=True,
                 skip_done=False)
        run_memory_pass([spec], results_dir=results_dir, cores=cores, threads=T)
    return len(missing)


def run_retention_backfill(
    cfg: dict,
    tuned: pd.DataFrame,
    results_dir: str,
    cores: list[int],
    stage: str,
    threads: int | None = None,
) -> int:
    """Rebuild and KEEP any index a later stage needs but which was never retained.

    Retention only happens when a repeat build actually runs. A winner whose repeats were
    already complete from an earlier stage is skipped on resume, so no index is written and
    nothing is retained — which is how four of Stage C's eight required indexes went missing
    after Stage B (all at N=500K, repeated back in Stage A).

    Rebuilds the winner's exact configuration and seed, writes NO build row (the build is
    already recorded, and an extra row would inflate the reported build count), and names the
    file after the winner's run id so provenance stays attached. HNSW builds are not
    bit-reproducible when multi-threaded, so this is the same configuration rather than
    necessarily the same graph — which is all Stage C needs, since it applies predicates at
    search time.
    """
    retain = stage_c_retention_set()
    if not retain:
        return 0

    from annstudy.filter_runner import resolve_index_path

    builds = _load_builds(results_dir)
    reachable = tuned[tuned["reachable"]] if "reachable" in tuned else tuned
    T = threads or default_threads()
    index_dir = Path("data/indexes") / f"stage_{stage.lower()}"

    specs: list[dict] = []
    for N, D, it in sorted(retain):
        if resolve_index_path(N, D, it, results_dir=results_dir):
            continue
        cand = reachable[(reachable["N"] == N) & (reachable["D"] == D)
                         & (reachable["index_type"] == it)]
        run_ids = [r for r in cand.get("best_build_run_id", pd.Series(dtype=str)).dropna()]
        brow = None
        for rid in run_ids:
            row = builds[builds["run_id"] == rid]
            if not row.empty:
                brow = row.iloc[0]
                break
        if brow is None:
            logger.error(f"No recorded winner build for ({N}, {D}, {it}); cannot retain.")
            continue
        specs.append({
            "run_id": str(brow["run_id"]), "stage": stage, "N": N, "D": D, "k": cfg["k"],
            "index_type": it, "build_params": _reconstruct_build_params(brow),
            "search_grid": [], "seed": int(brow.get("seed", 0)), "repeat_id": 0,
            "query_split_path": QUERY_SPLIT_PATH,
            "passages_path": passages_path(D), "queries_path": queries_path(D),
            "n_queries": cfg.get("n_queries", DEFAULT_N_QUERIES),
            "gt_path": gt_path(N, D),
            "index_save_path": str(index_dir / f"idx_{brow['run_id']}.faiss"),
            "measurement": cfg.get("measurement", {}), "results_dir": results_dir,
            "threads": T, "measure_qps": False, "measure_latency": False,
            "write_build_row": False, "keep_index": True,
        })

    if not specs:
        logger.info("Retention backfill: every index a later stage needs is already present.")
        return 0

    total = sum(_expected_index_bytes(s) for s in specs)
    logger.info(f"Retention backfill: rebuilding {len(specs)} index(es) a later stage needs "
                f"(~{total / 1e9:.1f} GB to retain).")
    for i, spec in enumerate(specs, start=1):
        logger.info(f"  retain {i}/{len(specs)}: {spec['index_type']} N={spec['N']} "
                    f"D={spec['D']} {spec['build_params']} -> {spec['index_save_path']}")
        run_pool([spec], cores[:1], results_dir=results_dir, keep_indexes=True,
                 skip_done=False)
    return len(specs)


def _load_memory_rows(results_dir: str) -> pd.DataFrame:
    """Load the memory-pass table (results/raw/memory/mem_*.parquet), or an empty frame."""
    mem_dir = Path(results_dir) / "memory"
    files = list(mem_dir.glob("mem_*.parquet")) if mem_dir.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _reconstruct_build_params(brow) -> dict:
    """Rebuild the build_params dict from a build row (only non-null grid fields)."""
    params: dict = {}
    for col in ["nlist", "m", "M", "ef_construction", "nbits"]:
        val = brow.get(col)
        if pd.notna(val):
            params[col] = int(val)
    return params


def _tuned_search_setting(tuned: pd.DataFrame, build_run_id: str, index_type: str) -> dict:
    """Pick one tuned search setting for a winner build (highest R that is reachable)."""
    rows = tuned[(tuned["best_build_run_id"] == build_run_id)]
    if rows.empty:
        return {}
    row = rows.sort_values("R").iloc[-1]
    if index_type in ("hnsw", "hnsw_sq"):
        return {"ef_search": int(row["ef_search"])}
    if index_type in ("ivf_flat", "ivf_pq"):
        return {"nprobe": int(row["nprobe"])}
    if index_type == "ivf_pq_refine":
        return {"nprobe": int(row["nprobe"]), "k_factor": int(row["k_factor"])}
    return {}


def write_stage_report(stage: str, results_dir: str = "results/raw", cells: dict | None = None) -> Path:
    """Write reports/stage_{stage}.md from tuned + builds (pure read; no invented numbers).

    ``cells`` (the stage config's ``cells`` block, i.e. {"N": [...], "D": [...]}) scopes the
    report to that stage's own (N, D) grid. Without it, every stage dumps the full tuned table,
    which made stage_A.md and stage_B.md byte-identical.
    """
    from annstudy.hardware import snapshot

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"stage_{stage}.md"

    final_path = Path("results/tuned/tuned.parquet")
    tuning_path = Path("results/tuned/tuned_tuning.parquet")
    is_final = final_path.exists()
    tuned_path = final_path if is_final else tuning_path
    builds = _load_builds(results_dir)
    hw = snapshot()

    # Scope to this stage's own (N, D) grid so each stage report is distinct.
    allowed = None
    if cells:
        allowed = {(int(n), int(dd)) for n in cells.get("N", []) for dd in cells.get("D", [])}
        if not builds.empty:
            builds = builds[[(int(n), int(dd)) in allowed
                             for n, dd in zip(builds["N"], builds["D"])]]

    lines: list[str] = []
    lines.append(f"# Stage {stage} report\n")
    lines.append("## Provenance\n")
    lines.append(f"- Date: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Git commit: `{hw['git_commit']}`")
    from annstudy.hardware import data_host_line
    lines.append(data_host_line())
    lines.append(f"- Source files: `{results_dir}/builds`, `{results_dir}/sweeps`, `{tuned_path}`\n")

    if not builds.empty:
        n_cells = builds[["N", "D"]].drop_duplicates().shape[0]
        lines.append(f"## Builds\n\n{len(builds)} build rows across {n_cells} (N,D) cells, "
                     f"index types: {sorted(builds['index_type'].unique())}.\n")

    if tuned_path.exists():
        tuned = pd.read_parquet(tuned_path)
        if allowed is not None:
            tuned = tuned[[(int(n), int(dd)) in allowed
                           for n, dd in zip(tuned["N"], tuned["D"])]]
        reach = tuned[tuned["reachable"]] if "reachable" in tuned else tuned
        measured = "test queries" if is_final else "tuning queries (test re-measurement pending)"
        lines.append(f"## Tuned winners (cheapest setting reaching each R; measured on {measured})\n")
        if is_final:
            cols = [c for c in ["index_type", "N", "D", "R", "ef_search", "nprobe", "k_factor",
                                "recall_test", "meets_R_on_test", "lat_p95_us", "qps",
                                "rss_bytes", "index_bytes",
                                "build_seconds_mean", "n_builds"] if c in reach.columns]
        else:
            cols = [c for c in ["index_type", "N", "D", "R", "ef_search", "nprobe", "k_factor",
                                "recall_tuning", "lat_p95_us_tuning", "qps_tuning"] if c in reach.columns]
        lines.append(reach[cols].to_markdown(index=False) if not reach.empty else "_none_")
        lines.append("")
        # Parameters are chosen on tuning queries, so a winner can land just below R on the
        # held-out test queries. Surfacing this prevents "reachable at R" being read as
        # "achieved R on test".
        if is_final and "meets_R_on_test" in reach.columns and not reach.empty:
            # astype(bool) first: the column is object dtype, and ~ on object does a bitwise
            # NOT (giving -1/-2) rather than a logical negation.
            meets = reach["meets_R_on_test"].fillna(True).astype(bool)
            miss = reach[~meets]
            lines.append("## Tuning-to-test generalization gap\n")
            if miss.empty:
                lines.append("Every tuned winner also meets its recall target on the test "
                             "queries.\n")
            else:
                lines.append(
                    f"{len(miss)} of {len(reach)} tuned winners were selected on the tuning "
                    f"queries but fall short of R on the held-out test queries (worst "
                    f"{miss['recall_gap_vs_R'].min():+.4f}, median "
                    f"{miss['recall_gap_vs_R'].median():+.4f}). This is selection noise, not "
                    f"a harness fault: the reported recall is the test value, and these rows "
                    f"must not be cited as having achieved R.\n")
                gcols = [c for c in ["index_type", "N", "D", "R", "recall_test",
                                     "recall_gap_vs_R"] if c in miss.columns]
                lines.append(miss[gcols].sort_values("recall_gap_vs_R")
                             .to_markdown(index=False) + "\n")

        unreach = tuned[~tuned["reachable"]] if "reachable" in tuned else tuned.iloc[0:0]
        lines.append("## Unreachable (reachable = false)\n")
        if unreach.empty:
            lines.append("_none_\n")
        else:
            lines.append(unreach[[c for c in ["index_type", "N", "D", "R"] if c in unreach.columns]]
                         .to_markdown(index=False) + "\n")

    lines.append("## For the paper\n")
    lines.append(f"- Feeds the Stage {stage} results section. Tuned winners above are the "
                 "cheapest-at-recall configurations; cite `results/tuned/tuned.parquet`.")
    if is_final:
        lines.append("- Headline numbers (`recall_test`, `lat_p95_us`, `qps`) are measured on the "
                     "8K held-out test queries; parameters were chosen on tuning queries only (§0.8).\n")
    else:
        lines.append("- NOTE: numbers above are tuning-query measurements; test-query re-measurement "
                     "(gap M10) has not run yet for this stage.\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out


# -------------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Run a stage of the ANN study")
    parser.add_argument("--config", required=True, help="Path to stage_*.yaml")
    parser.add_argument("--phase", default="all",
                        choices=["pass1", "tune", "repeats", "memory", "retain",
                                 "remeasure", "finalize", "report", "all"])
    parser.add_argument("--cores", default=None, help="e.g. '0-15' or '0,2,4' (default: all physical)")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--keep-indexes", action="store_true",
                        help="Keep serialized indexes (needed when a later stage reuses them)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate and print spec count without spawning")
    parser.add_argument("--max-build-variants", type=int, default=None,
                        help="Truncate each index's build grid (e.g. 1 for a quick pilot)")
    parser.add_argument("--threads", type=int, default=None,
                        help="T for the build/throughput pass (default: min(physical cores, 16))")
    parser.add_argument("--force", action="store_true",
                        help="Re-measure specs even if an equivalent run is already recorded. "
                             "Use when a comparison must be made WITHIN one campaign: absolute "
                             "latency on this host drifts between sessions, so reusing an older "
                             "measurement for one arm of a comparison mixes drift into the "
                             "result.")
    parser.add_argument("--no-memory", action="store_true",
                        help="Skip the index-only memory measurement in the T-thread pass")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_stage_config(args.config)
    stage = cfg.get("stage", "X")
    cores = parse_cores(args.cores)
    recall_targets = cfg.get("recall_targets", [0.90, 0.95, 0.99])

    specs = enumerate_specs(cfg, results_dir=args.results_dir, stage=stage,
                            max_build_variants=args.max_build_variants,
                            keep_indexes=args.keep_indexes or None)

    if args.dry_run:
        pending = run_pool(specs, cores, results_dir=args.results_dir, dry_run=True)
        total_search = sum(p["n_search"] for p in pending)
        print(json.dumps({
            "stage": stage, "cores": cores, "total_specs": len(specs),
            "pending_specs": len(pending), "total_search_settings": total_search,
        }, indent=2))
        return

    if args.phase in ("pass1", "all"):
        assert_prerequisites(specs)
        run_pool(specs, cores, results_dir=args.results_dir, keep_indexes=args.keep_indexes,
                 skip_done=not args.force)

    tuned = None
    if args.phase in ("tune", "repeats", "memory", "retain", "remeasure", "finalize",
                      "all"):
        tuned = run_tune(args.results_dir, recall_targets)

    if args.phase in ("repeats", "all"):
        run_repeat_builds(cfg, tuned, args.results_dir, cores, stage,
                          threads=args.threads, measure_memory=not args.no_memory,
                          cells=stage_cells(cfg))

    if args.phase in ("memory", "all") and not args.no_memory:
        run_memory_backfill(cfg, tuned, args.results_dir, cores, stage, threads=args.threads)

    if args.phase in ("retain", "all") and cfg.get("keep_indexes"):
        run_retention_backfill(cfg, tuned, args.results_dir, cores, stage,
                               threads=args.threads)

    if args.phase in ("remeasure", "all"):
        run_test_remeasure(cfg, tuned, args.results_dir, cores, stage,
                           cells=stage_cells(cfg))

    # "finalize" re-joins already-measured rows into results/tuned/tuned.parquet without
    # re-measuring anything, which is what you want after fixing a join or a derived column.
    if args.phase in ("remeasure", "finalize", "all"):
        finalize_tuned(args.results_dir, recall_targets)

    if args.phase in ("report", "all"):
        write_stage_report(stage, args.results_dir, cells=cfg.get("cells"))


if __name__ == "__main__":
    main()
