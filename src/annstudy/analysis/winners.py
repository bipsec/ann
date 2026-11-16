"""Winner analysis: for each (N, D, R, memory budget), find the cheapest index."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_winners(
    tuned_path: str = "results/tuned/tuned.parquet",
    builds_dir: str = "results/raw/builds",
    output_dir: str = "results/summary",
    memory_budgets: list[float | None] | None = None,
) -> pd.DataFrame:
    """Find the index minimizing p95 latency per (N, D, R, memory budget).

    Also computes winners for build_time and bytes_per_vector objectives.
    """
    if memory_budgets is None:
        memory_budgets = [None, 1.5, 0.5, 0.1]

    tuned = pd.read_parquet(tuned_path)
    reachable = tuned[tuned["reachable"] == True].copy()

    builds = pd.concat(
        [pd.read_parquet(f) for f in Path(builds_dir).glob("build_*.parquet")],
        ignore_index=True,
    )
    # Build/latency figures are host properties: never mix hosts, and honour supersession
    # (plan rule 0.6 marks superseded rows rather than deleting them).
    from annstudy.coordinate import current_hardware_id
    if "hardware_id" in builds.columns:
        builds = builds[builds["hardware_id"] == current_hardware_id()]
    if "superseded_by" in builds.columns:
        builds = builds[builds["superseded_by"].isna()]
    builds = builds.reset_index(drop=True)

    # Only pull columns from builds that the (finalized) tuned table doesn't already carry,
    # to avoid _x/_y collisions. The test-measured tuned.parquet already has rss_bytes,
    # index_bytes, and build_seconds_{mean,min,max}.
    extra = [c for c in ["rss_bytes", "index_bytes", "build_seconds"]
             if c not in reachable.columns]
    merged = reachable.merge(
        builds[["run_id"] + extra],
        left_on="best_build_run_id", right_on="run_id", how="left",
    )

    # Column names differ between the finalized (test-measured) and tuning-only schemas.
    lat_col = "lat_p95_us" if "lat_p95_us" in merged.columns else "lat_p95_us_tuning"
    build_col = "build_seconds_mean" if "build_seconds_mean" in merged.columns else "build_seconds"
    objectives = [lat_col, build_col, "index_bytes"]

    rows = []
    cost_rows = []  # every candidate's cost per cell, so regret is traceable to a CSV
    for _, group in merged.groupby(["N", "D", "R"]):
        N = group["N"].iloc[0]
        D = group["D"].iloc[0]
        R = group["R"].iloc[0]
        raw_bytes = N * D * 4

        for budget_mult in memory_budgets:
            if budget_mult is not None:
                budget_bytes = raw_bytes * budget_mult
                candidates = group[group["rss_bytes"] <= budget_bytes]
            else:
                candidates = group
                budget_bytes = None

            for objective in objectives:
                if candidates.empty or objective not in candidates.columns:
                    rows.append({
                        "N": N, "D": D, "R": R,
                        "memory_budget_mult": budget_mult,
                        "objective": objective,
                        "winner": "none",
                    })
                    continue

                if objective in candidates.columns:
                    valid = candidates.dropna(subset=[objective])
                    if valid.empty:
                        rows.append({
                            "N": N, "D": D, "R": R,
                            "memory_budget_mult": budget_mult,
                            "objective": objective,
                            "winner": "none",
                        })
                        continue

                    best = valid.loc[valid[objective].idxmin()]
                    rows.append({
                        "N": N, "D": D, "R": R,
                        "memory_budget_mult": budget_mult,
                        "objective": objective,
                        "winner": best["index_type"],
                        "value": best[objective],
                    })
                    for _, cand in valid.iterrows():
                        cost_rows.append({
                            "N": N, "D": D, "R": R,
                            "memory_budget_mult": budget_mult,
                            "objective": objective,
                            "index_type": cand["index_type"],
                            "value": cand[objective],
                        })

    result = pd.DataFrame(rows)
    output_path = Path(output_dir) / "winners.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Winners saved: {output_path}")

    # Per-candidate costs back the regret figures in analysis/rule.py; without them a regret
    # number would not trace to any CSV (plan rule 0.2).
    costs_path = Path(output_dir) / "cell_costs.csv"
    pd.DataFrame(cost_rows).to_csv(costs_path, index=False)
    logger.info(f"Per-cell candidate costs saved: {costs_path} ({len(cost_rows)} rows)")
    return result
