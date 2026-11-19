"""Scaling fits: log y = a + b*log(N) + c*log(D) + d*log(N)*log(D)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_scaling_model(
    tuned_path: str = "results/tuned/tuned.parquet",
    builds_dir: str = "results/raw/builds",
    output_dir: str = "results/summary",
    recall_targets: list[float] | None = None,
) -> pd.DataFrame:
    """Fit log-log scaling model for each index type and metric."""
    if recall_targets is None:
        recall_targets = [0.90, 0.95, 0.99]

    tuned = pd.read_parquet(tuned_path)
    reachable = tuned[tuned["reachable"] == True].copy()

    builds = pd.concat(
        [pd.read_parquet(f) for f in Path(builds_dir).glob("build_*.parquet")],
        ignore_index=True,
    )

    merged = reachable.merge(
        builds[["run_id", "rss_bytes", "build_seconds"]],
        left_on="best_build_run_id",
        right_on="run_id",
        how="left",
    )

    rows = []
    for index_type in merged["index_type"].unique():
        for R in recall_targets:
            subset = merged[(merged["index_type"] == index_type) & (merged["R"] == R)]
            if len(subset) < 3:
                continue

            log_N = np.log(subset["N"].values.astype(float))
            log_D = np.log(subset["D"].values.astype(float))

            X = np.column_stack([
                np.ones(len(subset)),
                log_N,
                log_D,
                log_N * log_D,
            ])

            # Column names differ between the finalized (test-measured) tuned table and the
            # tuning-only one; take whichever is present so the latency fit is not silently
            # skipped (it was, before: the finalized table writes lat_p95_us).
            lat_col = next(
                (c for c in ("lat_p95_us", "lat_p95_us_tuning") if c in subset.columns), None,
            )
            build_col = next(
                (c for c in ("build_seconds_mean", "build_seconds") if c in subset.columns), None,
            )
            for metric_col, metric_name in [
                (lat_col, "p95_latency"),
                (build_col, "build_time"),
                ("rss_bytes", "ram"),
            ]:
                if metric_col is None or metric_col not in subset.columns:
                    continue
                y_raw = subset[metric_col].values.astype(float)
                valid = y_raw > 0
                if valid.sum() < 3:
                    continue

                log_y = np.log(y_raw[valid])
                X_valid = X[valid]

                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(X_valid, log_y, rcond=None)
                    a, b, c, d = coeffs
                    rows.append({
                        "index_type": index_type,
                        "R": R,
                        "metric": metric_name,
                        "a": a, "b": b, "c": c, "d": d,
                        "n_points": int(valid.sum()),
                    })
                except np.linalg.LinAlgError:
                    logger.warning(f"Fit failed for {index_type}, R={R}, {metric_name}")

    result = pd.DataFrame(rows)
    output_path = Path(output_dir) / "scaling_fits.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Scaling fits saved: {output_path}")
    return result
