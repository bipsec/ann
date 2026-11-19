"""Project stage compute cost from measured pilot runs (EXPERIMENT_PLAN rule 0.3, Phase 3).

The plan forbids starting a long job without an estimate derived from a measured small run.
This module reads this host's recorded build and sweep rows, derives per-(index, D) anchors at
the pilot's N, extrapolates to each stage's cells under explicitly stated scaling assumptions,
and writes ``logs/budget.md``.

The projection is deliberately coarse and its assumptions are printed alongside the numbers:
it exists to support a go / trim / stop decision at Gate 3, not to be a result. Anchors are
measured; only the N extrapolation is modelled, since the pilot already covers every D.

CLI: ``annstudy-budget``.
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: How per-build cost grows with N, by index type. MEASURED by fitting N=100K (pilot) against
#: N=500K (Stage A) at matched D, per-index median; regenerate with
#: ``annstudy-budget --fit-exponents``. The notable one is ivf_flat at N^1.86: k-means training
#: is super-linear because nlist grows with sqrt(N) AND the training sample
#: min(N, 256*nlist) grows with it. The earlier assumption of N^1.0 for everything (plus a
#: log2 N factor for HNSW) was wrong in both directions.
BUILD_N_EXPONENT = {"flat": 0.88, "hnsw": 1.00, "ivf_flat": 1.86, "ivf_pq": 1.00,
                    "ivf_pq_refine": 0.95}
#: Empty: HNSW's measured exponent of 1.00 already absorbs the log factor.
BUILD_N_LOG_FACTOR: dict[str, bool] = {}

#: How per-query search cost grows with N at a fixed recall target. MEASURED the same way.
#: Flat is linear as expected. HNSW is N^0.40 rather than the assumed N^0.15 — reaching a fixed
#: recall costs more traversal as N grows. IVF-Flat is flatter than assumed because nlist grows
#: with sqrt(N), so a fixed nprobe scans a shrinking fraction of the corpus.
SEARCH_N_EXPONENT = {"flat": 1.00, "ivf_flat": 0.24, "ivf_pq": 0.62, "ivf_pq_refine": 0.64,
                     "hnsw": 0.40}

#: Build variants that win per (cell, index type): Stage A measured 27 winners / 20 pairs.
WINNER_VARIANTS_PER_PAIR = 1.35

#: Projected-vs-actual for stages that have run, as {name: (projected_h, actual_h, note)}.
#: Used to present a calibrated planning figure rather than a bare model output.
CALIBRATION = {
    "stage_a": (5.2, 7.7, "0 failures; wall clock was 10.6 h including a 2.5 h stall from a "
                          "disk-full fault, excluded here"),
}

TUNING_QUERIES = 2_000
TEST_QUERIES = 8_000


def _anchors(results_dir: str = "results/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measured per-(index_type, D) build seconds and per-setting sweep seconds at the pilot N."""
    from annstudy.coordinate import _load_builds
    from annstudy.tune import load_sweep_results
    from annstudy.coordinate import current_hardware_id

    builds = _load_builds(results_dir)
    if builds.empty:
        raise FileNotFoundError(
            "No build rows for this host; run `make pilot` before projecting a budget."
        )
    sweeps = load_sweep_results(str(Path(results_dir) / "sweeps"),
                                hardware_id=current_hardware_id())

    b = (builds.groupby(["index_type", "N", "D"])
         .agg(build_seconds=("build_seconds", "median"),
              threads=("threads", "max"))
         .reset_index())

    # Per-setting sweep cost: per-query median latency x queries x passes. Use the measured
    # mean latency across the grid, which is what a full sweep actually costs on average.
    s = sweeps.copy()
    s["sec_per_query"] = s["lat_mean_us"] / 1e6
    g = (s.groupby(["index_type", "N", "D"])
         .agg(mean_sec_per_query=("sec_per_query", "mean"),
              n_settings=("sec_per_query", "size"))
         .reset_index())
    return b, g


def _nearest_anchor(anchors: pd.DataFrame, index_type: str, D: int, N: int):
    """The anchor for this (index, D) whose N is closest to the target, in log space.

    Once a larger stage has run, extrapolating from it beats extrapolating from the pilot:
    projecting N=1M from the measured N=500K is a 2x reach instead of 10x, so an error in the
    fitted exponent is amplified far less.
    """
    rows = anchors[(anchors["index_type"] == index_type) & (anchors["D"] == D)]
    if rows.empty:
        return None
    return rows.iloc[(np.log(rows["N"].astype(float)) - math.log(N)).abs().argsort().iloc[0]]


def _scale_build(anchor_seconds: float, index_type: str, N_anchor: int, N: int) -> float:
    factor = (N / N_anchor) ** BUILD_N_EXPONENT.get(index_type, 1.0)
    if BUILD_N_LOG_FACTOR.get(index_type):
        factor *= math.log2(max(N, 2)) / math.log2(max(N_anchor, 2))
    return anchor_seconds * factor


def _scale_search(anchor_sec_per_query: float, index_type: str, N_anchor: int, N: int) -> float:
    return anchor_sec_per_query * (N / N_anchor) ** SEARCH_N_EXPONENT.get(index_type, 0.5)


def project_stage(
    config_path: str,
    build_anchors: pd.DataFrame,
    search_anchors: pd.DataFrame,
    passes: int = 3,
    max_build_variants: int | None = None,
    repeat_builds: int = 3,
) -> dict:
    """Project wall-clock hours for one stage config.

    Returns a dict with the spec count and the three cost components: the pinned latency
    sweeps (parallel across cores), the T-thread build/throughput pass (sequential), and the
    test re-measurement of winners.
    """
    from annstudy.coordinate import enumerate_specs, load_stage_config

    cfg = load_stage_config(config_path)
    specs = enumerate_specs(cfg, stage=cfg.get("stage", "X"),
                            max_build_variants=max_build_variants)

    sweep_core_seconds = 0.0
    build_seconds = 0.0
    remeasure_core_seconds = 0.0
    missing = set()

    for spec in specs:
        it, N, D = spec["index_type"], spec["N"], spec["D"]
        n_settings = len(spec["search_grid"])

        ba = _nearest_anchor(build_anchors, it, D, N)
        sa = _nearest_anchor(search_anchors, it, D, N)
        if ba is None or sa is None:
            missing.add((it, D))
            continue

        # Pass 1: build (single-threaded, inside the pinned runner) + sweep on tuning queries.
        per_q = _scale_search(float(sa["mean_sec_per_query"]), it, int(sa["N"]), N)
        sweep_core_seconds += n_settings * passes * TUNING_QUERIES * per_q
        sweep_core_seconds += _scale_build(float(ba["build_seconds"]), it, int(ba["N"]), N)

        # T-thread pass: per repeat, one build PLUS a recall and QPS measurement over the test
        # queries. Omitting that measurement under-projected Stage A by 2-3 h: at 60-100 s per
        # job it is comparable to the build itself. QPS is batched and multi-threaded, so it
        # is divided by the thread count; recall is a single batched search.
        #
        # Only WINNERS are repeated, not every spec — Stage A repeated 27 of 100 builds. The
        # cost is therefore accumulated per spec and scaled by the winner fraction below;
        # charging every spec over-projected Stage A by 2.2x.
        per_repeat_measure = (2 * TEST_QUERIES * per_q / 16) + (TEST_QUERIES * per_q / 16)
        build_seconds += repeat_builds * (
            _scale_build(float(ba["build_seconds"]), it, int(ba["N"]), N) + per_repeat_measure)

        # Test re-measurement of the winners: assume one chosen setting per recall target,
        # deduplicated to roughly two distinct settings per index.
        remeasure_core_seconds += 2 * passes * TEST_QUERIES * per_q

    # Winner fraction: one configuration wins per (cell, index type), and different recall
    # targets sometimes pick different build variants of it. Stage A measured 27 winners for
    # 20 (cell, index) pairs = 1.35 variants per pair, against 100 specs.
    n_pairs = len({(s["N"], s["D"], s["index_type"]) for s in specs})
    est_winners = min(len(specs), n_pairs * WINNER_VARIANTS_PER_PAIR)
    winner_fraction = est_winners / len(specs) if specs else 0.0

    return {
        "config": config_path,
        "n_specs": len(specs),
        "est_winners": round(est_winners, 1),
        "sweep_core_hours": sweep_core_seconds / 3600,
        "tthread_build_hours": build_seconds * winner_fraction / 3600,
        "remeasure_core_hours": remeasure_core_seconds * winner_fraction / 3600,
        "missing_anchors": sorted(missing),
    }


def project_stage_c(
    build_anchors: pd.DataFrame,
    search_anchors: pd.DataFrame,
    config_path: str = "configs/stage_c.yaml",
    passes: int = 3,
) -> dict:
    """Project Stage C, whose cost is dominated by per-query filtered measurement.

    Each job sweeps its grid on the tuning queries and re-measures the chosen settings on the
    test queries. The exact-scan strategy costs a fraction S of a full flat scan; the in-index
    and post-filter strategies cost roughly an unfiltered search (post-filter more, because it
    over-fetches).
    """
    from annstudy.filter_runner import enumerate_filter_jobs, load_stage_c_config

    cfg = load_stage_c_config(config_path)
    jobs = enumerate_filter_jobs(cfg)

    core_seconds = 0.0
    longest_job = 0.0
    missing = set()
    for job in jobs:
        it, N, D, S = job["index_type"], job["N"], job["D"], job["target_S"]
        lookup_type = "flat" if it == "exact" else it
        sa = _nearest_anchor(search_anchors, lookup_type, D, N)
        if sa is None:
            missing.add((lookup_type, D))
            continue
        per_q = _scale_search(float(sa["mean_sec_per_query"]), lookup_type, int(sa["N"]), N)

        if job["strategy"] == "pre_filter_exact":
            per_q *= S          # scan only the matching fraction
            n_settings = 1
        else:
            n_settings = len(
                [1] * (13 if lookup_type == "hnsw" else 10))
            if job["strategy"] == "post_filter_overfetch":
                per_q *= 1.5    # over-fetch costs more per query
        # tuning sweep + one test re-measurement per distinct chosen setting (~2)
        job_seconds = (n_settings * passes * TUNING_QUERIES * per_q
                       + 2 * passes * TEST_QUERIES * per_q)
        core_seconds += job_seconds
        longest_job = max(longest_job, job_seconds)

    return {
        "config": config_path,
        "n_specs": len(jobs),
        "sweep_core_hours": core_seconds / 3600,
        "tthread_build_hours": 0.0,
        "remeasure_core_hours": 0.0,
        "missing_anchors": sorted(missing),
        # A single job cannot be split across shards, so it is a hard floor on wall time.
        "longest_job_hours": longest_job / 3600,
    }


def project_stage_d(
    build_anchors: pd.DataFrame,
    search_anchors: pd.DataFrame,
    config_path: str = "configs/stage_d.yaml",
    passes: int = 3,
) -> dict:
    """Project Stage D (inserts), whose config describes a single cell rather than a grid.

    Per index: one build on the initial fraction, then ``insert_batches`` batches, each
    followed by a recall and p95 measurement at fixed search parameters, plus a from-scratch
    rebuild at the end and one retrained point.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    N = int(cfg["cell"]["N"])
    D = int(cfg["cell"]["D"])
    upd = cfg.get("update", {})
    n_batches = int(upd.get("insert_batches", 10))
    index_types = cfg.get("indexes", [])

    core_seconds = 0.0
    build_seconds = 0.0
    missing = set()
    for it in index_types:
        ba = _nearest_anchor(build_anchors, it, D, N)
        sa = _nearest_anchor(search_anchors, it, D, N)
        if ba is None or sa is None:
            missing.add((it, D))
            continue
        per_q = _scale_search(float(sa["mean_sec_per_query"]), it, int(sa["N"]), N)
        one_build = _scale_build(float(ba["build_seconds"]), it, int(ba["N"]), N)

        # initial build + end-to-end rebuild reference + one retrained point
        build_seconds += 3 * one_build
        # a pre-tune sweep on tuning queries, then a measurement after each batch
        core_seconds += 10 * passes * TUNING_QUERIES * per_q
        core_seconds += (n_batches + 2) * passes * TEST_QUERIES * per_q

    return {
        "config": config_path,
        "n_specs": len(index_types),
        "sweep_core_hours": core_seconds / 3600,
        "tthread_build_hours": build_seconds / 3600,
        "remeasure_core_hours": 0.0,
        "missing_anchors": sorted(missing),
    }


def write_budget(
    results_dir: str = "results/raw",
    output_path: str = "logs/budget.md",
    cores: int = 16,
) -> Path:
    build_anchors, search_anchors = _anchors(results_dir)

    stages = []
    for cfg_path in ("configs/stage_a.yaml", "configs/stage_b.yaml"):
        if Path(cfg_path).exists():
            stages.append(project_stage(cfg_path, build_anchors, search_anchors))
    if Path("configs/stage_c.yaml").exists():
        stages.append(project_stage_c(build_anchors, search_anchors))
    if Path("configs/stage_d.yaml").exists():
        stages.append(project_stage_d(build_anchors, search_anchors))

    lines = ["# Compute Budget\n",
             "Estimates and actuals for long-running jobs. Updated before each major run.\n"]

    lines.append("## Measured anchors (this host, from the pilot)\n")
    anchors = build_anchors.merge(search_anchors, on=["index_type", "N", "D"], how="outer")
    anchors["mean_ms_per_query"] = anchors["mean_sec_per_query"] * 1e3
    cols = ["index_type", "N", "D", "build_seconds", "threads", "n_settings",
            "mean_ms_per_query"]
    lines.append(anchors[[c for c in cols if c in anchors.columns]]
                 .sort_values(["index_type", "D"]).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Projection\n")
    lines.append(f"Latency sweeps are single-threaded and run one per core, so their cost is "
                 f"divided by {cores} cores. The T-thread build/throughput pass uses every "
                 f"core and therefore runs sequentially — it is NOT divided.\n")
    lines.append("| Stage | Specs/jobs | Sweep core-h | /%d cores | T-thread build h (serial) | "
                 "Re-measure core-h | **Projected wall h** |" % cores)
    lines.append("|---|---|---|---|---|---|---|")
    total = 0.0
    for s in stages:
        wall = (s["sweep_core_hours"] + s["remeasure_core_hours"]) / cores \
            + s["tthread_build_hours"]
        total += wall
        name = Path(s["config"]).stem
        lines.append(
            f"| {name} | {s['n_specs']} | {s['sweep_core_hours']:.1f} | "
            f"{s['sweep_core_hours'] / cores:.1f} | {s['tthread_build_hours']:.1f} | "
            f"{s['remeasure_core_hours'] / cores:.1f} | **{wall:.1f}** |"
        )
    lines.append(f"| **total** | | | | | | **{total:.1f}** |")
    lines.append("")

    floors = [(Path(s["config"]).stem, s["longest_job_hours"])
              for s in stages if s.get("longest_job_hours")]
    if floors:
        lines.append("A single job cannot be split across shards, so its duration is a hard "
                     "floor on that stage's wall time regardless of how many cores are used:")
        for name, h in floors:
            lines.append(f"- `{name}`: longest single job {h:.1f} h")
        lines.append("")

    missing = sorted({m for s in stages for m in s["missing_anchors"]})
    if missing:
        lines.append(f"Missing anchors (projected as zero, so the total is an UNDER-estimate): "
                     f"{missing}\n")

    # Calibration against the one stage whose actual duration is known. A projection that does
    # not say how wrong it was last time is not much use for a go/no-go decision.
    if CALIBRATION:
        lines.append("## Calibration against measured actuals\n")
        lines.append("| Stage | Projected by this model | Actual | Ratio |")
        lines.append("|---|---|---|---|")
        for name, (proj, actual, note) in CALIBRATION.items():
            lines.append(f"| {name} | {proj:.1f} h | {actual:.1f} h | **{actual / proj:.2f}x** |")
        ratio = max(a / p for p, a, _ in CALIBRATION.values())
        lines.append("")
        lines.append(f"The model is optimistic by up to **{ratio:.2f}x**, chiefly because "
                     f"effective parallelism is memory-limited rather than core-limited: "
                     f"admission control runs 6-12 concurrent jobs at D=1536, not 16. "
                     f"Multiply the projections above by ~{ratio:.1f} for a planning figure:")
        for s in stages:
            wall = (s["sweep_core_hours"] + s["remeasure_core_hours"]) / cores \
                + s["tthread_build_hours"]
            lines.append(f"- `{Path(s['config']).stem}`: {wall * ratio:.1f} h calibrated")
        lines.append("")
        for name, (_, _, note) in CALIBRATION.items():
            if note:
                lines.append(f"- {name}: {note}")
        lines.append("")

    lines.append("## Assumptions\n")
    lines.append("- Build cost scales with N as: "
                 + ", ".join(f"{k}: N^{v}" + (" x log2 N" if BUILD_N_LOG_FACTOR.get(k) else "")
                             for k, v in BUILD_N_EXPONENT.items()))
    lines.append("- Per-query search cost scales with N as: "
                 + ", ".join(f"{k}: N^{v}" for k, v in SEARCH_N_EXPONENT.items()))
    lines.append(f"- {TUNING_QUERIES} tuning queries, {TEST_QUERIES} test queries, 3 passes.")
    lines.append("- Every D is measured directly by the pilot, so only N is extrapolated.")
    st = anchors[anchors["threads"] == 1][["index_type", "D"]] if "threads" in anchors else None
    if st is not None and not st.empty:
        pairs = ", ".join(f"{r.index_type}@D{int(r.D)}" for r in st.itertuples())
        lines.append(f"- Build anchors for {pairs} are SINGLE-threaded (those configurations "
                     f"never won a recall target, so no T-thread build exists). Their build "
                     f"projections are therefore conservative by roughly the 3-6x threading "
                     f"speedup measured for the other index types.")
    lines.append("- Excludes filtered ground-truth computation for Stage C and the "
                 "index-only memory pass; both are small next to the measurement itself.")
    lines.append("- A projection is not a measurement. Treat these as the input to the Gate 3 "
                 "go / trim decision, and record actuals in the table below as stages run.\n")

    lines.append("## Actuals\n")
    lines.append("| Date | Job | Estimated | Actual | Notes |")
    lines.append("|------|-----|-----------|--------|-------|")
    prior = Path(output_path)
    if prior.exists():  # keep previously recorded actual rows
        for line in prior.read_text().splitlines():
            if line.startswith("| 2026-") and "---" not in line:
                lines.append(line)
    lines.append("")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Project stage compute budget from pilot data")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output", default="logs/budget.md")
    parser.add_argument("--cores", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = write_budget(args.results_dir, args.output, args.cores)
    print(Path(path).read_text())


if __name__ == "__main__":
    main()
