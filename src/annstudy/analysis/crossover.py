"""Crossover analysis: estimate S* where in-index filtered search and exact scan swap places.

S* is the selectivity at which the cheaper strategy changes. Below S*, gathering the matching
subset and scanning it exactly is faster; above it, filtered in-index search wins (the exact
scan's cost grows linearly in S while the index's does not). This is the quantity the study's
headline hypothesis H2 claims moves with dimension.

The exact scan does not use an index, so it is the shared baseline for every index type within
a (N, D, correlation, R) cell rather than being matched per index type.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRE = "pre_filter_exact"
IN_INDEX = "in_index_filtered"


def _interp_sign_change(S_values, diff) -> float | None:
    """First S where ``diff`` crosses zero, interpolated in log10 S. Direction-agnostic.

    ``diff`` is in-index latency minus exact-scan latency. It is positive at low S (the exact
    scan over a tiny subset is cheap) and negative at high S, so the crossing is a
    positive-to-negative transition; the search is written to accept either direction so a
    reversed curve is still detected rather than silently reported as "no crossover".
    """
    diff = np.asarray(diff, dtype=float)
    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if np.isnan(d0) or np.isnan(d1) or d0 == d1:
            continue
        if (d0 <= 0 <= d1) or (d0 >= 0 >= d1):
            log_s0, log_s1 = np.log10(S_values[i]), np.log10(S_values[i + 1])
            frac = -d0 / (d1 - d0)
            frac = float(np.clip(frac, 0.0, 1.0))
            return float(10 ** (log_s0 + frac * (log_s1 - log_s0)))
    return None


def estimate_crossover(
    filter_results_path: str = "results/raw/filter_runs.parquet",
    output_dir: str = "results/summary",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate S* per (index_type, N, D, correlation, R) by interpolation in log S.

    Also returns a conservative interval for S*, obtained by re-running the interpolation on
    the latency bootstrap envelopes already stored per row: the low end compares the in-index
    CI floor against the exact-scan CI ceiling, and the high end does the reverse. This is a
    propagation of the per-row query bootstrap rather than a fresh resample, since the per-query
    timings are not retained in the aggregated table.
    """
    if not Path(filter_results_path).exists():
        logger.warning(f"Filter results not found: {filter_results_path}")
        return pd.DataFrame()

    df = pd.read_parquet(filter_results_path)
    reach = df[df["reachable"].fillna(False)] if "reachable" in df.columns else df

    rows = []
    # The exact scan is index-independent, so group the baseline by cell only.
    for (N, D, corr, R), cell in reach.groupby(["N", "D", "correlation", "R"], dropna=False):
        pre = cell[cell["strategy"] == PRE].sort_values("target_S")
        if pre.empty:
            continue
        for index_type, ix in cell[cell["strategy"] == IN_INDEX].groupby("index_type"):
            ix = ix.sort_values("target_S")
            common_S = sorted(set(pre["target_S"]) & set(ix["target_S"]))
            if len(common_S) < 2:
                continue

            def col(frame, name, S=common_S):  # S bound now, not captured from the loop
                return frame.set_index("target_S")[name].reindex(S).to_numpy(float)

            pf = col(pre, "lat_p95_us")
            ixl = col(ix, "lat_p95_us")
            s_star = _interp_sign_change(common_S, ixl - pf)

            lo = hi = None
            if {"lat_p95_ci_lo", "lat_p95_ci_hi"}.issubset(cell.columns):
                s_a = _interp_sign_change(
                    common_S, col(ix, "lat_p95_ci_lo") - col(pre, "lat_p95_ci_hi"))
                s_b = _interp_sign_change(
                    common_S, col(ix, "lat_p95_ci_hi") - col(pre, "lat_p95_ci_lo"))
                bounds = [b for b in (s_a, s_b) if b is not None]
                if bounds:
                    lo, hi = min(bounds), max(bounds)

            rows.append({
                "index_type": index_type, "N": N, "D": D, "correlation": corr, "R": R,
                "crossover_S": s_star,
                "crossover_S_lo": lo,
                "crossover_S_hi": hi,
                "crossover_found": s_star is not None,
                "n_S_points": len(common_S),
                "S_range_lo": min(common_S),
                "S_range_hi": max(common_S),
            })

    result = pd.DataFrame(rows)
    output_path = Path(output_dir) / "crossover.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Crossover estimates saved: {output_path} ({len(result)} rows)")

    if not result.empty:
        _write_h2_comparison(result, output_dir)
    return result


def _write_h2_comparison(crossover: pd.DataFrame, output_dir: str) -> None:
    """H2: compare S* at the lowest vs highest D within each (N, index, correlation, R).

    Flags pairs whose intervals do not overlap, which is the plan's criterion for treating the
    dimension effect as more than noise.
    """
    found = crossover[crossover["crossover_found"]]
    rows = []
    for (N, it, corr, R), g in found.groupby(["N", "index_type", "correlation", "R"],
                                             dropna=False):
        if g["D"].nunique() < 2:
            continue
        lo_d, hi_d = g["D"].min(), g["D"].max()
        a = g[g["D"] == lo_d].iloc[0]
        b = g[g["D"] == hi_d].iloc[0]
        disjoint = None
        if all(pd.notna(a.get(c)) and pd.notna(b.get(c))
               for c in ("crossover_S_lo", "crossover_S_hi")):
            disjoint = bool(a["crossover_S_hi"] < b["crossover_S_lo"]
                            or b["crossover_S_hi"] < a["crossover_S_lo"])
        rows.append({
            "N": N, "index_type": it, "correlation": corr, "R": R,
            "D_low": lo_d, "D_high": hi_d,
            "S_star_D_low": a["crossover_S"], "S_star_D_high": b["crossover_S"],
            "ratio_high_over_low": (b["crossover_S"] / a["crossover_S"]
                                    if a["crossover_S"] else None),
            "intervals_disjoint": disjoint,
        })
    out = Path(output_dir) / "crossover_h2.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"H2 dimension comparison saved: {out} ({len(rows)} rows)")
