"""Crossover analysis: estimate S* where in-index filtered search and exact scan swap places.

S* is the selectivity at which the cheaper strategy changes. Below S*, gathering the matching
subset and scanning it exactly is faster; above it, filtered in-index search wins. H2 asks
whether S* moves with dimension D.

Method (revised 2026-11-16 after the post-study audit; see logs/decisions.md):

- **Interpolation** is linear in log-latency against log *measured* selectivity, as the paper
  states. The original code interpolated the raw latency difference against *nominal* S, which
  moved individual S* by up to 2x and a D-ratio from 0.83 to 0.54. No extrapolation: a crossing
  is only reported between two measured points.
- **Rows** must be reachable AND meet R on the held-out test queries (``meets_R_on_test``, or
  ``recall_test >= R`` for rows written before that column existed).
- **Uncertainty** is a parametric bootstrap over three sources, reported separately so a reader
  can see which one the conclusion depends on:
    1. query sampling — each row's p95 drawn log-normally from its stored bootstrap CI;
    2. build-to-build variance — repeat builds (``build_repeat``) resampled when present;
    3. session/load drift — an independent multiplicative factor per *curve* (one strategy at one
       D), log-uniform in [1/d, d]. A drift common to both strategies cancels in S*; this models
       the part that does not (e.g. a bandwidth-bound exact scan slowed more by co-tenancy than a
       latency-bound graph search). d defaults to 1.12, the median between-session drift
       measured on the study host.
- **H2** is judged per (N, index, correlation, R) on the ratio S*(D_high)/S*(D_low):
  ``shift`` if the 95% interval (with drift) excludes 1; ``equivalent`` if the 90% interval lies
  inside the pre-stated margin [0.8, 1.25] (TOST at alpha = 0.05); otherwise ``inconclusive``.
  With >= 3 dimensions the log-log slope dlogS*/dlogD is also estimated, the quantity the
  first-order cost model predicts (see ``crossover_model.py``).

The exact scan does not use an index, so it is the shared baseline for every index type within
a (N, D, correlation, R) cell.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRE = "pre_filter_exact"
IN_INDEX = "in_index_filtered"
DRIFT_FACTOR = 1.12
EQUIV_MARGIN = (0.8, 1.25)
Z95 = 1.959964


def _interp_sign_change(S_values, diff) -> float | None:
    """First S where ``diff`` crosses zero, interpolated in log10 S. Direction-agnostic.

    Retained for the regression test of the original sign-change bug; the estimator below uses
    :func:`s_star_log`.
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


def s_star_log(S, lat_in_index, lat_exact) -> float | None:
    """S* by linear interpolation of log(in-index / exact) against log S. None if no crossing.

    Points need not be sorted. Only crossings strictly bracketed by two measured points count.
    """
    S = np.asarray(S, float)
    f = np.log(np.asarray(lat_in_index, float)) - np.log(np.asarray(lat_exact, float))
    ok = np.isfinite(S) & np.isfinite(f) & (S > 0)
    S, f = S[ok], f[ok]
    if len(S) < 2:
        return None
    o = np.argsort(S)
    x, f = np.log(S[o]), f[o]
    for i in range(len(f) - 1):
        f0, f1 = f[i], f[i + 1]
        if f0 == 0:
            return float(np.exp(x[i]))
        if f0 * f1 < 0:
            return float(np.exp(x[i] + (-f0 / (f1 - f0)) * (x[i + 1] - x[i])))
    if f[-1] == 0:
        return float(np.exp(x[-1]))
    return None


def select_rows(df: pd.DataFrame, hardware_id: str | None = None,
                protocol: str = "legacy") -> pd.DataFrame:
    """Rows usable for S*: reachable, meeting R on held-out queries, one host, one protocol.

    ``protocol`` keeps campaigns apart: ``legacy`` = the original Stage C rows (no
    ``timing_protocol`` column value), ``v2`` = rows measured with the fixed timing protocol
    (e.g. the ``stage_c_h2.yaml`` campaign). Their latencies are not comparable (the timed
    region changed), so they must never be pooled into one curve.
    """
    d = df.copy()
    if hardware_id is not None and "hardware_id" in d.columns:
        d = d[d["hardware_id"] == hardware_id]
    tp = d["timing_protocol"] if "timing_protocol" in d.columns else pd.Series(None, index=d.index)
    if protocol == "legacy":
        d = d[tp.isna()]
    elif protocol == "v2":
        d = d[tp.fillna("").astype(str).str.startswith("v2")]
    elif protocol != "any":
        raise ValueError(f"unknown protocol {protocol!r}")
    d = d[d["reachable"].fillna(False).astype(bool)]
    if "meets_R_on_test" in d.columns:
        meets = d["meets_R_on_test"]
        fallback = d["recall_test"] >= d["R"]
        d = d[meets.where(meets.notna(), fallback).astype(bool)]
    else:
        d = d[d["recall_test"] >= d["R"]]
    d = d[d["lat_p95_us"].notna() & (d["lat_p95_us"] > 0)]
    if "measured_S_mean" not in d.columns:
        d["measured_S_mean"] = d["target_S"]
    d["S_x"] = d["measured_S_mean"].fillna(d["target_S"])
    if "build_repeat" not in d.columns:
        d["build_repeat"] = 0
    d["build_repeat"] = d["build_repeat"].fillna(0).astype(int)
    return d


def _log_sigma(row) -> float:
    lo, hi = row.get("lat_p95_ci_lo"), row.get("lat_p95_ci_hi")
    if pd.notna(lo) and pd.notna(hi) and lo > 0 and hi > lo:
        return float((np.log(hi) - np.log(lo)) / (2 * Z95))
    return 0.0


def _curve(frame: pd.DataFrame) -> dict:
    """{target_S: [(S_x, log_p95, log_sigma) per repeat]} for one strategy at one cell."""
    out: dict = {}
    for _, r in frame.iterrows():
        out.setdefault(float(r["target_S"]), []).append(
            (float(r["S_x"]), float(np.log(r["lat_p95_us"])), _log_sigma(r)))
    return out


def _point(curve: dict, S_common) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate: mean over repeats of S_x and log-latency at each common S."""
    sx = np.array([np.mean([p[0] for p in curve[s]]) for s in S_common])
    ll = np.array([np.mean([p[1] for p in curve[s]]) for s in S_common])
    return sx, ll


def _draw(curve: dict, S_common, rng, query: bool, drift: float) -> np.ndarray:
    """One bootstrap draw of the curve's log-latency at each common S."""
    vals = []
    for s in S_common:
        reps = curve[s]
        pick = [reps[i] for i in rng.integers(0, len(reps), size=len(reps))]  # build resample
        v = np.mean([p[1] + (rng.normal(0.0, p[2]) if query else 0.0) for p in pick])
        vals.append(v)
    vals = np.asarray(vals)
    if drift and drift > 1:
        vals = vals + rng.uniform(-np.log(drift), np.log(drift))  # one factor per curve
    return vals


def _cell_draws(pre_curve, ix_curve, S_common, n_boot, rng, drift):
    """S* point and bootstrap draws (query-only, and query+build+drift) for one cell."""
    sx_pre, ll_pre = _point(pre_curve, S_common)
    _, ll_ix = _point(ix_curve, S_common)
    point = s_star_log(sx_pre, np.exp(ll_ix), np.exp(ll_pre))
    q_only, full = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for b in range(n_boot):
        a = s_star_log(sx_pre, np.exp(_draw(ix_curve, S_common, rng, True, 0)),
                       np.exp(_draw(pre_curve, S_common, rng, True, 0)))
        c = s_star_log(sx_pre, np.exp(_draw(ix_curve, S_common, rng, True, drift)),
                       np.exp(_draw(pre_curve, S_common, rng, True, drift)))
        q_only[b] = np.nan if a is None else a
        full[b] = np.nan if c is None else c
    return point, q_only, full, sx_pre


def _pct(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def estimate_crossover(
    filter_results_path: str = "results/raw/filter_runs.parquet",
    output_dir: str = "results/summary",
    n_bootstrap: int = 2000,
    seed: int = 0,
    drift: float = DRIFT_FACTOR,
    hardware_id: str | None = None,
    tag: str = "",
    protocol: str = "legacy",
) -> pd.DataFrame:
    """Estimate S* per (index_type, N, D, correlation, R) and write crossover{tag}.csv.

    Also writes crossover_h2{tag}.csv (the H2 dimension comparison). See the module docstring
    for the method.
    """
    if not Path(filter_results_path).exists():
        logger.warning(f"Filter results not found: {filter_results_path}")
        return pd.DataFrame()

    rows_df = select_rows(pd.read_parquet(filter_results_path), hardware_id=hardware_id,
                          protocol=protocol)
    if rows_df.empty:
        logger.info(f"crossover: no {protocol} rows; nothing written for tag {tag!r}")
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    for (N, D, corr, R), cell in rows_df.groupby(["N", "D", "correlation", "R"], dropna=False):
        pre = cell[cell["strategy"] == PRE]
        if pre.empty:
            continue
        pre_curve = _curve(pre)
        for index_type, ix in cell[cell["strategy"] == IN_INDEX].groupby("index_type"):
            ix_curve = _curve(ix)
            S_common = sorted(set(pre_curve) & set(ix_curve))
            base = {"index_type": index_type, "N": N, "D": D, "correlation": corr, "R": R,
                    "n_S_points": len(S_common),
                    "n_build_repeats": int(ix["build_repeat"].nunique())}
            if len(S_common) < 2:
                rows.append(dict(base, crossover_found=False, crossover_S=None))
                continue
            point, q_only, full, sx = _cell_draws(pre_curve, ix_curve, S_common,
                                                  n_bootstrap, rng, drift)
            draws[(index_type, N, D, corr, R)] = full
            rows.append(dict(
                base,
                crossover_found=point is not None,
                crossover_S=point,
                # Query-sampling-only interval: what the original analysis reported (narrow).
                crossover_S_q_lo=_pct(q_only, 2.5), crossover_S_q_hi=_pct(q_only, 97.5),
                # Full interval (query + build + drift): the one conclusions rest on.
                crossover_S_lo=_pct(full, 2.5), crossover_S_hi=_pct(full, 97.5),
                found_frac=float(np.mean(np.isfinite(full))),
                S_range_lo=float(np.min(sx)), S_range_hi=float(np.max(sx)),
                method="log-latency vs log measured S; query+build+drift bootstrap",
                drift_factor=drift,
            ))

    result = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"crossover{tag}.csv", index=False)
    logger.info(f"Crossover estimates saved: crossover{tag}.csv ({len(result)} rows)")
    if not result.empty:
        h2_comparison(result, draws).to_csv(out_dir / f"crossover_h2{tag}.csv", index=False)
    return result


def h2_verdict(ratio_draws: np.ndarray, margin=EQUIV_MARGIN) -> tuple[str, dict]:
    """Classify a bootstrap distribution of S* ratios: shift / equivalent / inconclusive."""
    r = ratio_draws[np.isfinite(ratio_draws) & (ratio_draws > 0)]
    if len(r) < 50:
        return "inconclusive", {"n_valid_draws": int(len(r))}
    lo95, hi95 = np.percentile(r, [2.5, 97.5])
    lo90, hi90 = np.percentile(r, [5.0, 95.0])
    stats = {"ratio_ci95_lo": float(lo95), "ratio_ci95_hi": float(hi95),
             "ratio_ci90_lo": float(lo90), "ratio_ci90_hi": float(hi90),
             "n_valid_draws": int(len(r))}
    if hi95 < 1.0 or lo95 > 1.0:
        return "shift", stats
    if margin[0] < lo90 and hi90 < margin[1]:
        return "equivalent", stats
    return "inconclusive", stats


def h2_comparison(crossover: pd.DataFrame, draws: dict) -> pd.DataFrame:
    """H2 per (N, index, correlation, R): ratio at lowest vs highest D, verdict, D-slope."""
    found = crossover[crossover["crossover_found"].fillna(False).astype(bool)]
    rows = []
    for (N, it, corr, R), g in found.groupby(["N", "index_type", "correlation", "R"],
                                             dropna=False):
        Ds = sorted(g["D"].unique())
        if len(Ds) < 2:
            continue
        lo_d, hi_d = Ds[0], Ds[-1]
        a = g[g["D"] == lo_d].iloc[0]
        b = g[g["D"] == hi_d].iloc[0]
        da, db = draws[(it, N, lo_d, corr, R)], draws[(it, N, hi_d, corr, R)]
        verdict, stats = h2_verdict(db / da)
        disjoint = bool(a["crossover_S_hi"] < b["crossover_S_lo"]
                        or b["crossover_S_hi"] < a["crossover_S_lo"])
        row = {"N": N, "index_type": it, "correlation": corr, "R": R,
               "D_low": lo_d, "D_high": hi_d, "Ds": ",".join(str(int(d)) for d in Ds),
               "S_star_D_low": a["crossover_S"], "S_star_D_high": b["crossover_S"],
               "ratio_high_over_low": b["crossover_S"] / a["crossover_S"],
               "intervals_disjoint": disjoint, "verdict": verdict, **stats}
        if len(Ds) >= 3:
            x = np.log(np.asarray(Ds, float))
            mat = np.vstack([draws[(it, N, d, corr, R)] for d in Ds])  # (nD, B)
            ok = np.all(np.isfinite(mat) & (mat > 0), axis=0)
            slopes = np.array([np.polyfit(x, np.log(mat[:, j]), 1)[0]
                               for j in np.where(ok)[0]])
            pts = np.log(g.set_index("D").loc[Ds, "crossover_S"].to_numpy(float))
            row.update(slope_logS_logD=float(np.polyfit(x, pts, 1)[0]),
                       slope_ci95_lo=_pct(slopes, 2.5), slope_ci95_hi=_pct(slopes, 97.5),
                       slope_n_draws=int(len(slopes)))
        rows.append(row)
    return pd.DataFrame(rows)
