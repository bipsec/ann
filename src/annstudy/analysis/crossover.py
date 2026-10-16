"""Crossover analysis: estimate S* where in-index search latency exceeds pre-filter exact scan."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_crossover(
    filter_results_path: str = "results/raw/filter_runs.parquet",
    output_dir: str = "results/summary",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate S* per (index, D, N, correlation, R) by log-space interpolation.

    S* is the selectivity where in-index filtered search latency first
    exceeds pre-filter exact scan latency.
    """
    if not Path(filter_results_path).exists():
        logger.warning(f"Filter results not found: {filter_results_path}")
        return pd.DataFrame()

    df = pd.read_parquet(filter_results_path)

    group_cols = ["index_type", "N", "D", "correlation", "R"]
    rows = []

    for group_key, group in df.groupby(group_cols):
        pre_filter = group[group["strategy"] == "pre_filter_exact"].sort_values("target_S")
        in_index = group[group["strategy"] == "in_index_filtered"].sort_values("target_S")

        if pre_filter.empty or in_index.empty:
            continue

        common_S = sorted(set(pre_filter["target_S"]) & set(in_index["target_S"]))
        if len(common_S) < 2:
            continue

        pf_lat = pre_filter.set_index("target_S")["lat_p95_us"].reindex(common_S)
        ix_lat = in_index.set_index("target_S")["lat_p95_us"].reindex(common_S)

        diff = ix_lat.values - pf_lat.values

        crossover_S = None
        for i in range(len(diff) - 1):
            if diff[i] <= 0 and diff[i + 1] > 0:
                log_s0 = np.log10(common_S[i])
                log_s1 = np.log10(common_S[i + 1])
                frac = -diff[i] / (diff[i + 1] - diff[i])
                crossover_S = 10 ** (log_s0 + frac * (log_s1 - log_s0))
                break

        row = {col: val for col, val in zip(group_cols, group_key)}
        row["crossover_S"] = crossover_S
        row["crossover_found"] = crossover_S is not None
        rows.append(row)

    result = pd.DataFrame(rows)
    output_path = Path(output_dir) / "crossover.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Crossover estimates saved: {output_path}")
    return result
