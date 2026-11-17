"""First-order cost model for how the filter crossover S* moves with dimension D.

Near the crossover, model each strategy's p95 latency as a local power law in selectivity S and
dimension D:

    exact scan of the subset:   L_ex(S, D) ~ S^a * D^c_ex
    in-index filtered search:   L_ix(S, D) ~ S^b * D^c_ix

S* solves L_ix = L_ex, so

    S*(D) ~ D^((c_ix - c_ex) / (a - b))     and     S*(D2)/S*(D1) = (D2/D1)^((c_ix - c_ex)/(a - b)).

The D-exponents c_ix and c_ex are taken from the UNFILTERED Stage A/B measurements (tuned p95
at the same N and recall target; Flat for the exact scan), and the S-slopes a, b from the
filtered curves. The prediction therefore uses no information about where the crossovers
actually fell, so comparing it against the measured ratio is a genuine test.

Reading: if the exact scan pays more per unit of D than the index (c_ex > c_ix, true for a
graph index whose traversal length is set by intrinsic, not ambient, dimension), S* falls as D
rises. If the index scales with D like a scan (IVF-Flat scans whole lists, c_ix ~ c_ex), the
model predicts S* roughly D-invariant. The closed-form crossover of prior work (Mandarapu &
Kunkunuru, 2026) has no D term; this is the term it would need, and its sign is index-specific.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from annstudy.analysis.crossover import IN_INDEX, PRE, select_rows

logger = logging.getLogger(__name__)


def predicted_ratio(c_ix, c_ex, a, b, D_lo, D_hi) -> float:
    """S*(D_hi)/S*(D_lo) under the local power-law model."""
    if not np.isfinite(a - b) or (a - b) == 0:
        return float("nan")
    return float(np.exp((c_ix - c_ex) * np.log(D_hi / D_lo) / (a - b)))


def d_exponent(lat_lo, lat_hi, D_lo, D_hi) -> float:
    """Two-point D-exponent of a cost, d log L / d log D."""
    if not (lat_lo and lat_hi) or lat_lo <= 0 or lat_hi <= 0:
        return float("nan")
    return float(np.log(lat_hi / lat_lo) / np.log(D_hi / D_lo))


def s_slope(frame: pd.DataFrame) -> float:
    """Least-squares slope of log p95 against log measured S over a filtered curve."""
    f = frame[(frame["S_x"] > 0) & (frame["lat_p95_us"] > 0)]
    if f["S_x"].nunique() < 2:
        return float("nan")
    return float(np.polyfit(np.log(f["S_x"]), np.log(f["lat_p95_us"]), 1)[0])


def model_table(
    filter_results_path: str = "results/raw/filter_runs.parquet",
    tuned_path: str = "results/tuned/tuned.parquet",
    h2_path: str = "results/summary/crossover_h2.csv",
    output_dir: str = "results/summary",
    s_window: tuple[float, float] = (5e-4, 0.1),
    tag: str = "",
    protocol: str = "legacy",
) -> pd.DataFrame:
    """Predicted vs measured S* ratio for every H2 pair; writes crossover_model{tag}.csv.

    ``s_window`` restricts the S-slope fits to the region where crossovers occur, so a power
    law is a fair local description (the exact scan's curve bends above S ~ 0.1, where it
    switches from gathering to masked full scans).
    """
    for p in (filter_results_path, tuned_path, h2_path):
        if not Path(p).exists():
            logger.warning(f"crossover model: missing {p}")
            return pd.DataFrame()
    rows_df = select_rows(pd.read_parquet(filter_results_path), protocol=protocol)
    rows_df = rows_df[(rows_df["S_x"] >= s_window[0]) & (rows_df["S_x"] <= s_window[1])]
    tuned = pd.read_parquet(tuned_path)
    tuned = tuned[tuned["reachable"].fillna(False).astype(bool)]
    h2 = pd.read_csv(h2_path)

    def unfiltered(it, N, D, R):
        t = tuned[(tuned["index_type"] == it) & (tuned["N"] == N) & (tuned["D"] == D)]
        if it != "flat":
            t = t[t["R"] == R]
        return float(t["lat_p95_us"].median()) if len(t) else float("nan")

    out = []
    for _, h in h2.iterrows():
        N, it, corr, R = h["N"], h["index_type"], h["correlation"], h["R"]
        D_lo, D_hi = float(h["D_low"]), float(h["D_high"])
        c_ix = d_exponent(unfiltered(it, N, D_lo, R), unfiltered(it, N, D_hi, R), D_lo, D_hi)
        c_ex = d_exponent(unfiltered("flat", N, D_lo, R), unfiltered("flat", N, D_hi, R),
                          D_lo, D_hi)
        cell = rows_df[(rows_df["N"] == N) & (rows_df["correlation"] == corr)
                       & (rows_df["R"] == R)]
        a = np.nanmean([s_slope(cell[(cell["strategy"] == PRE) & (cell["D"] == d)])
                        for d in (D_lo, D_hi)])
        b = np.nanmean([s_slope(cell[(cell["strategy"] == IN_INDEX)
                                     & (cell["index_type"] == it) & (cell["D"] == d)])
                        for d in (D_lo, D_hi)])
        pred = predicted_ratio(c_ix, c_ex, a, b, D_lo, D_hi)
        lo, hi = h.get("ratio_ci95_lo", np.nan), h.get("ratio_ci95_hi", np.nan)
        out.append({
            "N": N, "index_type": it, "correlation": corr, "R": R,
            "D_low": int(D_lo), "D_high": int(D_hi),
            "c_index_unfiltered": c_ix, "c_exact_unfiltered": c_ex,
            "a_exact_S_slope": a, "b_index_S_slope": b,
            "predicted_ratio": pred,
            "measured_ratio": h["ratio_high_over_low"],
            "measured_ci95_lo": lo, "measured_ci95_hi": hi,
            "prediction_inside_ci": bool(pd.notna(lo) and lo <= pred <= hi),
            "direction_agrees": bool(np.sign(np.log(pred)) == np.sign(
                np.log(h["ratio_high_over_low"]))) if pred > 0 else False,
            "measured_verdict": h.get("verdict"),
        })
    res = pd.DataFrame(out)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(output_dir) / f"crossover_model{tag}.csv", index=False)
    logger.info(f"Crossover model saved: crossover_model{tag}.csv ({len(res)} rows)")
    return res
