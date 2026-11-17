"""Select cheapest search settings reaching each recall target, re-measure on test queries."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_sweep_results(
    sweeps_dir: str = "results/raw/sweeps",
    hardware_id: str | None = None,
) -> pd.DataFrame:
    """Load and concatenate all sweep Parquet files.

    ``hardware_id`` restricts the result to one machine. Latency and throughput are host
    properties, so rows carried over from another machine must never be tuned against or
    reported alongside this host's measurements.
    """
    sweeps_dir = Path(sweeps_dir)
    files = list(sweeps_dir.glob("sweep_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sweep files found in {sweeps_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if hardware_id is not None and "hardware_id" in df.columns:
        before = len(df)
        df = df[df["hardware_id"] == hardware_id].reset_index(drop=True)
        if len(df) < before:
            logger.info(
                f"Sweep rows: kept {len(df)}/{before} for hardware_id={hardware_id} "
                f"(dropped rows measured on other hosts)"
            )
    return df


SELECTION_MODES = ("point", "lower_ci")


def meets_target(df: pd.DataFrame, R: float, selection: str = "point") -> pd.Series:
    """Rows whose tuning recall reaches R under the selection rule.

    ``point`` (the original rule) compares the point estimate, which over-selects settings
    that only just clear R on 2,000 tuning queries: 17-19 of ~50 winners then missed R on the
    held-out queries. ``lower_ci`` requires the lower end of the tuning recall's bootstrap
    interval to clear R, trading a little extra effort for winners that hold up on test.
    """
    if selection == "point":
        return df["recall_mean"] >= R
    if selection == "lower_ci":
        lo = df["recall_ci_lo"] if "recall_ci_lo" in df.columns else df["recall_mean"]
        return lo.fillna(df["recall_mean"]) >= R
    raise ValueError(f"selection must be one of {SELECTION_MODES}, got {selection!r}")


def find_cheapest_setting(
    sweep_df: pd.DataFrame,
    recall_targets: list[float] | None = None,
    selection: str = "point",
) -> pd.DataFrame:
    """For each (cell, index_type, R), find the cheapest configuration reaching R.

    Selection is joint over BUILD VARIANTS and search settings, which is what the study asks:
    "the cheapest configuration of Flat, HNSW, IVF-Flat and IVF-PQ that reaches recall@10 >= R"
    (plan section 1), with the tuned table defined as "one row per cell, index, R"
    (plan section 5). Grouping by ``build_run_id`` instead would emit one row per build
    variant — e.g. all three HNSW M values — which both departs from that schema and multiplies
    the downstream cost, since every such row is then rebuilt three times at T threads and has
    its memory measured.

    Operates on tuning-query sweep rows only. Returns one row per (index_type, N, D, R) with a
    ``reachable`` column, and ``best_build_run_id`` identifying the chosen build variant.
    """
    if recall_targets is None:
        recall_targets = [0.90, 0.95, 0.99]

    tuning = sweep_df[sweep_df["query_split"] == "tuning"].copy()

    rows = []
    group_cols = ["index_type", "N", "D"]
    for group_key, group_df in tuning.groupby(group_cols):
        for R in recall_targets:
            ok = group_df[meets_target(group_df, R, selection)]

            if ok.empty:
                row = {
                    col: group_key[i] for i, col in enumerate(group_cols)
                }
                row["R"] = R
                row["reachable"] = False
                row["k"] = group_df["k"].iloc[0] if "k" in group_df.columns else 10
                rows.append(row)
                continue

            cheapest = ok.loc[ok["lat_p95_us"].idxmin()]

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
            row["selection"] = selection
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
