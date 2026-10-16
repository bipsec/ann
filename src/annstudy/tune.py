"""Select cheapest search settings reaching each recall target, re-measure on test queries."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_sweep_results(sweeps_dir: str = "results/raw/sweeps") -> pd.DataFrame:
    """Load and concatenate all sweep Parquet files."""
    sweeps_dir = Path(sweeps_dir)
    files = list(sweeps_dir.glob("sweep_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sweep files found in {sweeps_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def find_cheapest_setting(
    sweep_df: pd.DataFrame,
    recall_targets: list[float] | None = None,
) -> pd.DataFrame:
    """For each (index_type, build_run_id, R), find the cheapest setting reaching R.

    Operates on tuning-query sweep rows only.
    Returns a DataFrame with one row per (index_type, build_run_id, R),
    including a 'reachable' column.
    """
    if recall_targets is None:
        recall_targets = [0.90, 0.95, 0.99]

    tuning = sweep_df[sweep_df["query_split"] == "tuning"].copy()

    rows = []
    group_cols = ["index_type", "build_run_id", "N", "D"]
    for group_key, group_df in tuning.groupby(group_cols):
        for R in recall_targets:
            meets_target = group_df[group_df["recall_mean"] >= R]

            if meets_target.empty:
                row = {
                    col: group_key[i] for i, col in enumerate(group_cols)
                }
                row["R"] = R
                row["reachable"] = False
                row["k"] = group_df["k"].iloc[0] if "k" in group_df.columns else 10
                rows.append(row)
                continue

            cheapest = meets_target.loc[meets_target["lat_p95_us"].idxmin()]

            row = {
                col: group_key[i] for i, col in enumerate(group_cols)
            }
            row["R"] = R
            row["reachable"] = True
            row["k"] = int(cheapest.get("k", 10))
            row["best_build_run_id"] = cheapest["build_run_id"]
            row["ef_search"] = cheapest.get("ef_search")
            row["nprobe"] = cheapest.get("nprobe")
            row["k_factor"] = cheapest.get("k_factor")
            row["recall_tuning"] = cheapest["recall_mean"]
            row["lat_p95_us_tuning"] = cheapest["lat_p95_us"]
            row["qps_tuning"] = cheapest.get("qps")
            rows.append(row)

    return pd.DataFrame(rows)


def save_tuned_results(
    tuned_df: pd.DataFrame,
    output_dir: str = "results/tuned",
) -> Path:
    """Save the tuned results to Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tuned.parquet"
    tuned_df.to_parquet(output_path, index=False)
    logger.info(f"Tuned results saved: {output_path} ({len(tuned_df)} rows)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Tune: select cheapest settings per recall target")
    parser.add_argument("--sweeps-dir", default="results/raw/sweeps")
    parser.add_argument("--output-dir", default="results/tuned")
    parser.add_argument("--recall-targets", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sweep_df = load_sweep_results(args.sweeps_dir)
    logger.info(f"Loaded {len(sweep_df)} sweep rows")

    tuned = find_cheapest_setting(sweep_df, args.recall_targets)
    logger.info(f"Tuned: {len(tuned)} rows, {tuned['reachable'].sum()} reachable")

    save_tuned_results(tuned, args.output_dir)

    unreachable = tuned[~tuned["reachable"]]
    if not unreachable.empty:
        logger.warning(f"Unreachable cells:\n{unreachable[['index_type', 'N', 'D', 'R']]}")


if __name__ == "__main__":
    main()
