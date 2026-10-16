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

    merged = reachable.merge(
        builds[["run_id", "rss_bytes", "index_bytes", "build_seconds"]],
        left_on="best_build_run_id",
        right_on="run_id",
        how="left",
    )

    rows = []
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

            for objective in ["lat_p95_us_tuning", "build_seconds", "index_bytes"]:
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

    result = pd.DataFrame(rows)
    output_path = Path(output_dir) / "winners.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Winners saved: {output_path}")
    return result
