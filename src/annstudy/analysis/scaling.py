"""Scaling fits of recall-constrained cost against corpus size N and dimension D.

Two outputs, both with bootstrap intervals (revised 2026-11-16; the original fit had none, used
uncentred logs — so ``c`` was the D-slope at N = 1, an extrapolation — and silently never fitted
RAM because a merge suffixed the column):

- ``scaling_exponents.csv``: one-axis exponents, the quantities a reader can interpret.
  ``axis = "D"``: d log y / d log D at a fixed N (4 dimensions at N = 100K and 500K).
  ``axis = "N"``: d log y / d log N at a fixed D (3 corpus sizes at D = 384 and 1536).
- ``scaling_fits.csv``: the joint model log y = a + b*lN + c*lD + d*lN*lD with lN, lD CENTRED
  at the reference cell (N = 500K, D = 384), so b and c are local slopes at that cell. With 10
  cells this is descriptive only; the interaction d is fitted when >= 2 levels of each factor
  and >= 6 cells exist (``interaction_identified``).

Intervals cover query sampling and build-to-build variance only. Stage A's cells were measured
within one campaign, so between-session drift (~1.1x) is not in them; treat differences in an
exponent smaller than ~0.1 as within that unmodelled band.

Uncertainty: latency is drawn log-normally from each tuned row's stored p95 bootstrap CI (query
sampling); build time log-uniformly within the [min, max] of its repeat builds (build-to-build);
resident memory is deterministic per configuration. Only rows that meet R on the held-out
queries are used, so a fit is never driven by a configuration that did not achieve the target.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REF_N, REF_D = 500_000, 384
Z95 = 1.959964
METRICS = {  # name -> (value column, draw kind)
    "p95_latency": ("lat_p95_us", "ci"),
    "build_time": ("build_seconds_mean", "range"),
    "ram": ("rss_bytes", "fixed"),
}


def _draw(sub: pd.DataFrame, col: str, kind: str, rng, n_boot: int) -> np.ndarray:
    """(n_boot, n_rows) bootstrap draws of log y."""
    y = np.log(sub[col].to_numpy(float))
    if kind == "ci" and {"lat_p95_ci_lo", "lat_p95_ci_hi"}.issubset(sub.columns):
        lo = np.log(sub["lat_p95_ci_lo"].to_numpy(float))
        hi = np.log(sub["lat_p95_ci_hi"].to_numpy(float))
        sig = np.where(np.isfinite(hi - lo) & (hi > lo), (hi - lo) / (2 * Z95), 0.0)
        return y + rng.normal(0.0, 1.0, (n_boot, len(y))) * sig
    if kind == "range" and {"build_seconds_min", "build_seconds_max"}.issubset(sub.columns):
        lo = np.log(sub["build_seconds_min"].to_numpy(float))
        hi = np.log(sub["build_seconds_max"].to_numpy(float))
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
        lo, hi = np.where(ok, lo, y), np.where(ok, hi, y)
        return lo + rng.random((n_boot, len(y))) * (hi - lo)
    return np.broadcast_to(y, (n_boot, len(y)))


def _slope_with_ci(x, draws, y_point):
    pt = float(np.polyfit(x, y_point, 1)[0])
    sl = np.array([np.polyfit(x, d, 1)[0] for d in draws])
    return pt, float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


def usable_rows(tuned: pd.DataFrame) -> pd.DataFrame:
    t = tuned[tuned["reachable"].fillna(False).astype(bool)].copy()
    if "meets_R_on_test" in t.columns:
        t = t[t["meets_R_on_test"].fillna(True).astype(bool)]
    return t


def fit_exponents(tuned: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = usable_rows(tuned)
    rows = []
    for (it, R), g in t.groupby(["index_type", "R"]):
        for metric, (col, kind) in METRICS.items():
            if col not in g.columns:
                continue
            gm = g[g[col].fillna(0) > 0]
            for axis, fixed_col, var_col in (("D", "N", "D"), ("N", "D", "N")):
                for fixed, sub in gm.groupby(fixed_col):
                    sub = sub.sort_values(var_col)
                    if sub[var_col].nunique() < 3:  # two points fit any line exactly
                        continue
                    x = np.log(sub[var_col].to_numpy(float))
                    pt, lo, hi = _slope_with_ci(x, _draw(sub, col, kind, rng, n_boot),
                                                np.log(sub[col].to_numpy(float)))
                    rows.append({"index_type": it, "R": R, "metric": metric, "axis": axis,
                                 "fixed_at": int(fixed), "exponent": pt, "ci95_lo": lo,
                                 "ci95_hi": hi, "n_points": len(sub),
                                 "levels": ",".join(str(int(v)) for v in sub[var_col])})
    return pd.DataFrame(rows)


def fit_joint(tuned: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    t = usable_rows(tuned)
    rows = []
    for (it, R), g in t.groupby(["index_type", "R"]):
        for metric, (col, kind) in METRICS.items():
            if col not in g.columns:
                continue
            sub = g[g[col].fillna(0) > 0]
            if len(sub) < 5:
                continue
            lN = np.log(sub["N"].to_numpy(float) / REF_N)
            lD = np.log(sub["D"].to_numpy(float) / REF_D)
            ident = sub["N"].nunique() >= 2 and sub["D"].nunique() >= 2 and len(sub) >= 6
            X = np.column_stack([np.ones(len(sub)), lN, lD] + ([lN * lD] if ident else []))
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            y = np.log(sub[col].to_numpy(float))
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            draws = np.array([np.linalg.lstsq(X, d, rcond=None)[0]
                              for d in _draw(sub, col, kind, rng, n_boot)])
            q = np.percentile(draws, [2.5, 97.5], axis=0)
            row = {"index_type": it, "R": R, "metric": metric,
                   "a": coef[0], "b": coef[1], "c": coef[2],
                   "d": coef[3] if ident else np.nan,
                   "b_ci_lo": q[0, 1], "b_ci_hi": q[1, 1],
                   "c_ci_lo": q[0, 2], "c_ci_hi": q[1, 2],
                   "d_ci_lo": q[0, 3] if ident else np.nan,
                   "d_ci_hi": q[1, 3] if ident else np.nan,
                   "interaction_identified": ident, "n_points": len(sub),
                   "reference": f"N={REF_N},D={REF_D} (centred)"}
            rows.append(row)
    return pd.DataFrame(rows)


def fit_scaling_model(
    tuned_path: str = "results/tuned/tuned.parquet",
    builds_dir: str = "results/raw/builds",  # kept for call compatibility; tuned has all inputs
    output_dir: str = "results/summary",
    recall_targets: list[float] | None = None,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Write scaling_exponents.csv and scaling_fits.csv; returns the joint fits."""
    tuned = pd.read_parquet(tuned_path)
    if recall_targets is not None:
        tuned = tuned[tuned["R"].isin(recall_targets) | (tuned["index_type"] == "flat")]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    expo = fit_exponents(tuned, n_boot=n_boot)
    expo.to_csv(out / "scaling_exponents.csv", index=False)
    joint = fit_joint(tuned, n_boot=n_boot)
    joint.to_csv(out / "scaling_fits.csv", index=False)
    logger.info(f"Scaling: {len(expo)} one-axis exponents, {len(joint)} joint fits saved")
    return joint
