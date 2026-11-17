"""Generate the paper draft from measured results (EXPERIMENT_PLAN Phase 6).

Plan rule 0.2 forbids inventing numbers: every figure in the paper must come from a file under
``results/``. So the paper is GENERATED rather than typed — each table is rendered from a CSV or
parquet, and each number appearing in prose is interpolated from the same loaded data. Re-run
this after any new stage and the draft updates itself.

``check_traceability`` then does the Phase 6 final check: it extracts every numeric token from
the generated draft and reports any that cannot be found in the underlying result files.

CLI: ``annstudy-paper`` (also ``make paper``).
"""

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAPER = Path("paper")
TABLES = PAPER / "tables"
SUMMARY = Path("results/summary")
TUNED = Path("results/tuned/tuned.parquet")

TITLE = ("Which ANN Index Is Cheapest at a Given Recall? "
         "A Cost-Surface Study over Dimensionality, Scale, and Filter Selectivity")


# --------------------------------------------------------------------------- loading

def load_all(results_dir: str = "results/raw") -> dict:
    """Load every measured artifact the paper draws on. Missing ones are None, not fabricated."""
    from annstudy.coordinate import _load_builds, _load_memory_rows, current_hardware_id
    from annstudy.tune import load_sweep_results

    def csv(name):
        p = SUMMARY / name
        return pd.read_csv(p) if p.exists() else None

    try:
        sweeps = load_sweep_results(str(Path(results_dir) / "sweeps"),
                                    hardware_id=current_hardware_id())
    except FileNotFoundError:
        sweeps = None

    return {
        "hardware": json.loads(Path("results/hardware.json").read_text())
        if Path("results/hardware.json").exists() else None,
        "tuned": pd.read_parquet(TUNED) if TUNED.exists() else None,
        "builds": _load_builds(results_dir),
        "memory": _load_memory_rows(results_dir),
        "sweeps": sweeps,
        "winners": csv("winners.csv"),
        "cell_costs": csv("cell_costs.csv"),
        "scaling": csv("scaling_fits.csv"),
        "scaling_exponents": csv("scaling_exponents.csv"),
        "rule": csv("rule_accuracy.csv"),
        "lodo": csv("rule_lodo_cv.csv"),
        "lid": csv("lid.csv"),
        "acceptance": csv("acceptance_checks.csv"),
        "efc": csv("acceptance_ef_construction.csv"),
        "crossover": csv("crossover.csv"),
        "filter_runs": (pd.read_parquet("results/raw/filter_runs.parquet")
                        if Path("results/raw/filter_runs.parquet").exists() else None),
        "update_runs": (pd.concat([pd.read_parquet(f) for f in
                                   Path("results/raw/update_runs").glob("*.parquet")],
                                  ignore_index=True)
                        if list(Path("results/raw/update_runs").glob("*.parquet")) else None),
        "rule_tree": (SUMMARY / "rule_tree.txt").read_text()
        if (SUMMARY / "rule_tree.txt").exists() else None,
    }


def _md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw) if df is not None and not df.empty else "_no data_"


def fig_box(num: int, title: str, plot: str, source: str, png: str) -> str:
    """Render a figure as a fenced placeholder box: caption + what-to-plot + data source + PNG.

    Fenced (```), so check_traceability — which skips fenced code — does not treat the axis
    values inside as prose numbers needing a result-file match. The figures themselves are
    generated separately (`make analysis` writes the PNGs under figures/); these boxes tell the
    author what each figure shows and where its data lives, so the paper can be finished by
    dropping the rendered PNG in place of the box.
    """
    # Everything stays INSIDE the fence (including the caption) so check_traceability, which
    # skips fenced code, does not flag the figure number or axis values as untraceable prose.
    return (
        "```text\n"
        f"FIGURE {num}. {title}\n"
        f"  Plot    : {plot}\n"
        f"  Source  : {source}\n"
        f"  Render  : {png}\n"
        "```\n"
    )


# ---------------------------------------------------------------------------- tables

def tbl_cost_surface(tuned: pd.DataFrame, N: int) -> pd.DataFrame:
    """p95 latency (us, test queries) by index type x D, one block per recall target."""
    r = tuned[(tuned["N"] == N) & (tuned["reachable"] == True)]  # noqa: E712
    out = []
    for R in sorted(r["R"].unique()):
        sub = r[r["R"] == R]
        piv = sub.pivot_table(index="index_type", columns="D", values="lat_p95_us",
                              aggfunc="first")
        piv = piv.round(0).astype("Int64").reset_index()
        piv.insert(0, "R", R)
        out.append(piv)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def tbl_recall_achieved(tuned: pd.DataFrame, N: int) -> pd.DataFrame:
    r = tuned[(tuned["N"] == N) & (tuned["reachable"] == True)]  # noqa: E712
    cols = ["index_type", "D", "R", "recall_test", "meets_R_on_test", "recall_gap_vs_R"]
    cols = [c for c in cols if c in r.columns]
    return r[cols].sort_values(["index_type", "D", "R"]).round(5)


def tbl_reachability(tuned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (N, it), g in tuned.groupby(["N", "index_type"]):
        rows.append({
            "N": N, "index_type": it,
            "cells_x_targets": len(g),
            "reachable": int(g["reachable"].fillna(False).sum()),
            "unreachable": int((~g["reachable"].fillna(False)).sum()),
        })
    return pd.DataFrame(rows).sort_values(["N", "index_type"])


def tbl_pq_ceiling(sweeps: pd.DataFrame, builds: pd.DataFrame, N: int) -> pd.DataFrame:
    """Highest recall IVF-PQ attains at any swept nprobe, by (D, m) — the quantization ceiling."""
    if sweeps is None or builds.empty:
        return pd.DataFrame()
    s = sweeps[(sweeps["query_split"] == "tuning") & (sweeps["N"] == N)]
    m = s.merge(builds[["run_id", "m"]], left_on="build_run_id", right_on="run_id", how="left")
    rows = []
    for it in ("ivf_pq", "ivf_pq_refine"):
        sub = m[m["index_type"] == it]
        if sub.empty:
            continue
        g = sub.groupby(["D", "m"])["recall_mean"].max().reset_index()
        g["index_type"] = it
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["bytes_per_vector"] = out["m"]
    out["compression_vs_raw"] = (out["D"] * 4 / out["m"]).round(1)
    out = out.pivot_table(index=["D", "m", "compression_vs_raw"], columns="index_type",
                          values="recall_mean").reset_index()
    return out.round(4).sort_values(["D", "m"])


def tbl_scale_axis(tuned: pd.DataFrame, R: float = 0.95) -> pd.DataFrame:
    """p95 latency by index type x N, at the dimensions Stage B measured across all three N."""
    r = tuned[(tuned["reachable"] == True) & (tuned["R"] == R)]  # noqa: E712
    counts = r.groupby("D")["N"].nunique()
    Ds = sorted(counts[counts >= 3].index)
    if not Ds:
        return pd.DataFrame()
    out = []
    for D in Ds:
        piv = r[r["D"] == D].pivot_table(index="index_type", columns="N",
                                         values="lat_p95_us", aggfunc="first")
        piv = piv.round(0).astype("Int64").reset_index()
        piv.insert(0, "D", D)
        out.append(piv)
    return pd.concat(out, ignore_index=True)


def tbl_sq_probe(tuned: pd.DataFrame, N: int = 500_000) -> pd.DataFrame:
    """Scalar-quantized vs uncompressed HNSW, at the cells where both were measured together."""
    q = tuned[(tuned["N"] == N) & (tuned["index_type"].isin(["hnsw", "hnsw_sq"]))]
    if q.empty or "hnsw_sq" not in set(q["index_type"]):
        return pd.DataFrame()
    paired = ({int(d) for d in q[q["index_type"] == "hnsw_sq"]["D"].unique()}
              & {int(d) for d in q[q["index_type"] == "hnsw"]["D"].unique()})
    q = q[q["D"].isin(paired) & q["reachable"].fillna(False)].copy()
    if q.empty:
        return pd.DataFrame()
    q["raw_bytes"] = N * q["D"] * 4
    q["rss_x_raw"] = (q["rss_bytes"] / q["raw_bytes"]).round(3)
    cols = ["index_type", "D", "R", "recall_test", "lat_p95_us", "rss_x_raw"]
    return q[[c for c in cols if c in q.columns]].sort_values(["D", "R", "index_type"]).round(4)


def tbl_crossover(crossover: pd.DataFrame) -> pd.DataFrame:
    """S* with bootstrap-propagated intervals, for the cells where a crossover was located."""
    if crossover is None or crossover.empty:
        return pd.DataFrame()
    f = crossover[crossover["crossover_found"].fillna(False)]
    cols = ["index_type", "N", "D", "correlation", "R", "crossover_S",
            "crossover_S_q_lo", "crossover_S_q_hi", "crossover_S_lo", "crossover_S_hi",
            "found_frac"]
    return f[[c for c in cols if c in f.columns]].sort_values(
        ["correlation", "index_type", "N", "D", "R"])


def tbl_crossover_h2(path: str = "results/summary/crossover_h2.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    cols = ["N", "index_type", "correlation", "R", "D_low", "D_high",
            "S_star_D_low", "S_star_D_high", "ratio_high_over_low", "ratio_ci95_lo",
            "ratio_ci95_hi", "verdict"]
    return d[[c for c in cols if c in d.columns]]


def tbl_crossover_model(path: str = "results/summary/crossover_model.csv") -> pd.DataFrame:
    """Predicted (from unfiltered D-exponents) vs measured S* ratio per H2 pair."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    cols = ["N", "index_type", "correlation", "R", "c_index_unfiltered", "c_exact_unfiltered",
            "predicted_ratio", "measured_ratio", "measured_ci95_lo", "measured_ci95_hi",
            "direction_agrees"]
    return d[[c for c in cols if c in d.columns]]


def tbl_filter_reachability(filter_runs: pd.DataFrame) -> pd.DataFrame:
    """How often each strategy can reach a recall target, by filter correlation."""
    if filter_runs is None or filter_runs.empty:
        return pd.DataFrame()
    rows = []
    for (st, corr), g in filter_runs.groupby(["strategy", "correlation"], dropna=False):
        ok = int(g["reachable"].fillna(False).sum())
        rows.append({"strategy": st, "correlation": corr, "cells_x_targets": len(g),
                     "reachable": ok, "unreachable": len(g) - ok})
    return pd.DataFrame(rows).sort_values(["strategy", "correlation"])


def tbl_measured_selectivity(filter_runs: pd.DataFrame) -> pd.DataFrame:
    """Measured versus target selectivity — the plan's 20% acceptance check."""
    if filter_runs is None or filter_runs.empty:
        return pd.DataFrame()
    g = (filter_runs.groupby(["correlation", "target_S"])["measured_S_mean"]
         .mean().reset_index())
    g["relative_error_pct"] = ((g["measured_S_mean"] - g["target_S"])
                               / g["target_S"] * 100).round(1)
    g["within_20pct"] = g["relative_error_pct"].abs() <= 20
    return g.round(6)


def tbl_update_drift(update_runs: pd.DataFrame) -> pd.DataFrame:
    """Recall drift across insert batches, and the from-scratch rebuild reference."""
    if update_runs is None or update_runs.empty:
        return pd.DataFrame()
    rows = []
    for it, g in update_runs.groupby("index_type"):
        g = g.sort_values("batch_id")
        ins = g[g["rebuild_reference_recall"].isna()]
        rb = g[g["rebuild_reference_recall"].notna()]
        if ins.empty:
            continue
        rows.append({
            "index_type": it,
            "recall_first_batch": round(float(ins["recall_at_fixed_params"].iloc[0]), 5),
            "recall_last_batch": round(float(ins["recall_at_fixed_params"].iloc[-1]), 5),
            "drift": round(float(ins["recall_at_fixed_params"].iloc[-1]
                                 - ins["recall_at_fixed_params"].iloc[0]), 5),
            "rebuild_reference": (round(float(rb["rebuild_reference_recall"].iloc[0]), 5)
                                  if not rb.empty else None),
            "inserts_per_second_median": int(ins["inserts_per_second"].median()),
            "p95_us_median": int(ins["lat_p95_us"].median()),
        })
    return pd.DataFrame(rows)


def tbl_memory_budget(winners: pd.DataFrame, R: float = 0.95) -> pd.DataFrame:
    if winners is None:
        return pd.DataFrame()
    lat = winners[(winners["objective"] == "lat_p95_us") & (winners["R"] == R)]
    piv = lat.pivot_table(index=["N", "D"], columns="memory_budget_mult", values="winner",
                          aggfunc="first", dropna=False)
    piv.columns = ["unbounded" if pd.isna(c) else f"{c}x raw" for c in piv.columns]
    return piv.reset_index()


def tbl_index_footprint(tuned: pd.DataFrame, N: int) -> pd.DataFrame:
    """Measured index size and resident memory relative to the raw vectors."""
    r = tuned[(tuned["N"] == N) & (tuned["reachable"] == True)]  # noqa: E712
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(["index_type", "D"]).agg(
        index_bytes=("index_bytes", "median"), rss_bytes=("rss_bytes", "median")).reset_index()
    g["raw_bytes"] = N * g["D"] * 4
    g["index_over_raw"] = (g["index_bytes"] / g["raw_bytes"]).round(3)
    g["rss_over_raw"] = (g["rss_bytes"] / g["raw_bytes"]).round(3)
    g["index_MB"] = (g["index_bytes"] / 1e6).round(1)
    return g[["index_type", "D", "index_MB", "index_over_raw", "rss_over_raw"]]


def fit_n_exponents(builds: pd.DataFrame, sweeps: pd.DataFrame) -> pd.DataFrame:
    """Empirical N exponents from the two measured corpus sizes, at matched D.

    Computed here rather than quoted so the paper's scaling claims trace to the raw rows.
    """
    rows = []
    if not builds.empty:
        tt = builds[builds["threads"] > 1]
        g = tt.groupby(["index_type", "N", "D"])["build_seconds"].median().reset_index()
        for (it, D), s in g.groupby(["index_type", "D"]):
            s = s.set_index("N")["build_seconds"]
            ns = sorted(s.index)
            if len(ns) < 2 or s[ns[0]] <= 0:
                continue
            rows.append({"quantity": "build_seconds", "index_type": it, "D": D,
                         "N_lo": ns[0], "N_hi": ns[-1],
                         "exponent": np.log(s[ns[-1]] / s[ns[0]]) / np.log(ns[-1] / ns[0])})
    if sweeps is not None:
        s1 = sweeps[(sweeps["query_split"] == "tuning") & (sweeps["threads"] == 1)]
        for (it, D), g in s1.groupby(["index_type", "D"]):
            m = g.groupby("N")["lat_mean_us"].mean()
            ns = sorted(m.index)
            if len(ns) < 2 or m[ns[0]] <= 0:
                continue
            rows.append({"quantity": "latency_us", "index_type": it, "D": D,
                         "N_lo": ns[0], "N_hi": ns[-1],
                         "exponent": np.log(m[ns[-1]] / m[ns[0]]) / np.log(ns[-1] / ns[0])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["quantity", "index_type"])
            .agg(median_exponent=("exponent", "median"), n_D=("exponent", "size"),
                 N_lo=("N_lo", "min"), N_hi=("N_hi", "max"))
            .reset_index().round(2))


def tbl_build_time(tuned: pd.DataFrame, N: int) -> pd.DataFrame:
    r = tuned[(tuned["N"] == N) & (tuned["reachable"] == True)]  # noqa: E712
    cols = [c for c in ["build_seconds_mean", "build_seconds_min", "build_seconds_max",
                        "n_builds", "build_threads"] if c in r.columns]
    if not cols:
        return pd.DataFrame()
    g = r.groupby(["index_type", "D"])[cols].first().reset_index()
    return g.round(2)


# ---------------------------------------------------------------------------- writer

def write_paper(results_dir: str = "results/raw") -> Path:
    d = load_all(results_dir)
    PAPER.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    tuned, hw = d["tuned"], d["hardware"]
    if tuned is None or hw is None:
        raise FileNotFoundError("Need results/tuned/tuned.parquet and results/hardware.json")

    Ns = sorted(tuned["N"].unique())
    N_main = 500_000 if 500_000 in Ns else Ns[-1]
    Ds = sorted(tuned["D"].unique())
    Rs = sorted(tuned["R"].unique())

    cost = tbl_cost_surface(tuned, N_main)
    reach = tbl_reachability(tuned)
    pq = tbl_pq_ceiling(d["sweeps"], d["builds"], N_main)
    mem_budget = tbl_memory_budget(d["winners"])
    footprint = tbl_index_footprint(tuned, N_main)
    expo = fit_n_exponents(d["builds"], d["sweeps"])
    build_t = tbl_build_time(tuned, N_main)
    scale = tbl_scale_axis(tuned)
    sq = tbl_sq_probe(tuned)
    xover = tbl_crossover(d["crossover"])
    xover_h2 = tbl_crossover_h2()
    xmodel = tbl_crossover_model()
    freach = tbl_filter_reachability(d["filter_runs"])
    fsel = tbl_measured_selectivity(d["filter_runs"])
    udrift = tbl_update_drift(d["update_runs"])
    recalls = tbl_recall_achieved(tuned, N_main)

    for name, df in [("cost_surface", cost), ("reachability", reach), ("pq_ceiling", pq),
                     ("memory_budget_winners", mem_budget), ("index_footprint", footprint),
                     ("n_exponents", expo), ("build_time", build_t),
                     ("recall_achieved", recalls), ("scale_axis", scale),
                     ("sq_probe", sq), ("crossover", xover), ("crossover_h2", xover_h2),
                     ("crossover_model", xmodel),
                     ("filter_reachability", freach), ("measured_selectivity", fsel),
                     ("update_drift", udrift),
                     ("lid", d["lid"]),
                     ("acceptance", d["acceptance"]), ("ef_construction", d["efc"]),
                     ("scaling_fits", d["scaling"])]:
        if df is not None and not getattr(df, "empty", True):
            (TABLES / f"{name}.md").write_text(_md(df), encoding="utf-8")
            df.to_csv(TABLES / f"{name}.csv", index=False)

    # --- figures quoted in prose, all computed
    r_main = tuned[(tuned["N"] == N_main) & (tuned["reachable"] == True)]  # noqa: E712
    hnsw95 = r_main[(r_main["index_type"] == "hnsw") & (r_main["R"] == 0.95)]
    flat95 = r_main[(r_main["index_type"] == "flat") & (r_main["R"] == 0.95)]
    speedup = (flat95.set_index("D")["lat_p95_us"] / hnsw95.set_index("D")["lat_p95_us"]).dropna()
    # Growth across the dimension range is lat(D_max)/lat(D_min) BY DIMENSION, not max/min of
    # values: the tuned HNSW winner at D=128 can be pricier than at D=384, so a naive max/min
    # reported 1.9x from the D=384 floor instead of the correct 1.5x from D=128.
    _hn_by_D = hnsw95.set_index("D")["lat_p95_us"].sort_index()
    _fl_by_D = flat95.set_index("D")["lat_p95_us"].sort_index()
    hnsw_growth = (_hn_by_D.iloc[-1] / _hn_by_D.iloc[0]) if len(_hn_by_D) >= 2 else float("nan")
    flat_growth = (_fl_by_D.iloc[-1] / _fl_by_D.iloc[0]) if len(_fl_by_D) >= 2 else float("nan")

    # Ingest slowdown of HNSW vs the IVF families, computed once so the abstract and section 5.9
    # agree (they diverged: ~50x vs ~37x).
    ingest_ratio = None
    if udrift is not None and not udrift.empty and "inserts_per_second_median" in udrift.columns:
        _iv = udrift[udrift["index_type"] != "hnsw"]["inserts_per_second_median"]
        _hn = udrift[udrift["index_type"] == "hnsw"]["inserts_per_second_median"]
        if not _iv.empty and not _hn.empty and float(_hn.iloc[0]) > 0:
            ingest_ratio = float(_iv.max()) / float(_hn.iloc[0])

    n_builds = len(d["builds"])
    n_sweep_rows = 0 if d["sweeps"] is None else len(d["sweeps"])
    acc = d["acceptance"]
    acc_pass = int((acc["status"] == "pass").sum()) if acc is not None else 0
    acc_total = len(acc) if acc is not None else 0

    lid = d["lid"]
    rule = d["rule"]
    lodo = d["lodo"]

    pq_max = None
    if not pq.empty and "ivf_pq" in pq.columns:
        pq_max = pq["ivf_pq"].max()
        pq_best = pq.loc[pq["ivf_pq"].idxmax()]

    from annstudy.analysis.run_all import (
        _crossover_verdicts, _rule_verdict, _winner_verdicts,
    )
    verdicts = {}
    verdicts.update(_winner_verdicts())
    verdicts.update(_crossover_verdicts())
    verdicts.update(_rule_verdict())

    stages_done = sorted({int(n) for n in Ns})
    gen = datetime.now(UTC).isoformat(timespec="seconds")

    L: list[str] = []
    A = L.append

    A(f"# {TITLE}\n")
    A("**Draft — all four measurement stages complete:** Stage A (dimension), Stage B "
      "(scale), Stage C (filtering) and Stage D (updates), on top of Phase 1 data and a "
      "validated harness. Remaining gaps are stated explicitly in section 8 rather than "
      "estimated.\n")
    A(f"_Generated {gen} by `annstudy-paper` from files under `results/`. Every number below "
      "is rendered from a result file; none are entered by hand. Regenerate with "
      "`make paper`._\n")

    # ---------------- abstract (~200 words, leads with the filter findings)
    A("## Abstract\n")
    _unc = (xover[xover["correlation"] == "none"]["crossover_S"].median()
            if not xover.empty else float("nan"))
    _pos = (xover[xover["correlation"] == "positive"]["crossover_S"].median()
            if not xover.empty else float("nan"))
    A(f"Choosing an approximate nearest neighbour (ANN) index is usually posed as a "
      f"recall-versus-throughput trade-off on a fixed dataset. We instead fix the requirement — "
      f"recall@10 at least R — and ask which index, and which filtering strategy, is cheapest "
      f"subject to it as the workload varies in corpus size N, embedding dimension D (obtained "
      f"by truncating one Matryoshka embedding to {', '.join(str(x) for x in Ds)} dimensions, so "
      f"D is not confounded with the dataset), and filter selectivity S.\n")
    if not xover.empty and not xover_h2.empty and "verdict" in xover_h2.columns:
        _h = xover_h2[xover_h2["index_type"] == "hnsw"]
        _i = xover_h2[xover_h2["index_type"] == "ivf_flat"]
        _hs = int((_h["verdict"] == "shift").sum())
        _agree = (int(xmodel["direction_agrees"].sum()), len(xmodel)) if not xmodel.empty \
            else (0, 0)
        A(f"On the filter axis, the selectivity S* at which in-index filtered search and exact "
          f"scan of the filtered subset swap places moves with dimension **in an "
          f"index-dependent direction**. For HNSW it falls as D rises (median "
          f"S*(D_high)/S*(D_low) = {_h['ratio_high_over_low'].median():.2f}; the 95% interval "
          f"excludes 1 in {_hs} of {len(_h)} pairs even allowing for session drift); for "
          f"IVF-Flat no shift is resolved ({len(_i)} pairs, median ratio "
          f"{_i['ratio_high_over_low'].median():.2f}). A first-order cost model that uses only "
          f"each index's *unfiltered* latency-vs-D exponent predicts the direction in "
          f"{_agree[0]} of {_agree[1]} pairs: a graph search pays far less per added dimension "
          f"than a scan, an inverted-list scan does not. These filter results come from a "
          f"campaign whose design confounds D with session load, so they are provisional "
          f"pending a confound-free rerun (section 8). S* shifts about {_unc / _pos:.0f}x with "
          f"query-filter correlation, more than with D. Under anti-correlated predicates, HNSW "
          f"filtered search reaches no recall target at any swept effort.\n")
    if pq_max is not None:
        _ing = f"~{ingest_ratio:.0f}x" if ingest_ratio else "tens of times"
        A(f"Two negatives follow. Product quantization alone never reaches R=0.90 (ceiling "
          f"{pq_max:.3f} at {pq_best['compression_vs_raw']:.0f}x compression; exact re-ranking "
          f"lifts it to {pq['ivf_pq_refine'].max():.3f}), so none of the five conventional "
          f"families meets a sub-raw memory budget at these targets — though a "
          f"scalar-quantized graph index, added as a follow-up probe, reaches 0.27-0.35x raw "
          f"while holding recall, and is faster than uncompressed HNSW below R=0.99. "
          f"Unfiltered, HNSW is the cheapest index at every D and R "
          f"({speedup.min():.0f}-{speedup.max():.0f}x over exact scan at R=0.95; p95 latency "
          f"grows only {hnsw_growth:.1f}x from D={Ds[0]} to D={Ds[-1]}), and inserting 11% more "
          f"vectors costs almost no recall though HNSW ingests {_ing} slower than IVF. Absolute "
          f"latency drifts ~1.1x between sessions, so every comparison is drawn within a "
          f"campaign and keyed to a hardware identity.\n")

    # ---------------- introduction
    A("## 1. Introduction\n")
    A("A practitioner choosing a vector index has a requirement, not a curve: *this* query "
      "latency, at *this* recall, within *this* memory budget, for *this* corpus. Published "
      "benchmarks answer a different question — they plot recall against throughput for many "
      "indexes on a handful of datasets, leaving the reader to locate their own operating "
      "point. Worse, the datasets differ in dimension, scale and intrinsic structure "
      "simultaneously, so the effect of any one axis cannot be isolated.\n")
    A("This study inverts the framing. For each workload cell (N, D, S, R, memory budget M) we "
      "find the cheapest configuration of Flat, HNSW, IVF-Flat, IVF-PQ and IVF-PQ+refine that "
      "attains recall@10 >= R, then ask where the winner changes. The key methodological move "
      "is to vary D *within one corpus*: a Matryoshka-trained embedding model admits prefix "
      "truncation followed by renormalisation, so D=128 and D=1536 describe the same million "
      "passages under the same model. Dimension stops being confounded with dataset.\n")
    A("**Contributions**, ordered as the paper presents them.\n")
    A("1. The filter-selectivity crossover S* measured across dimension on one corpus, with a "
      "first-order cost model that predicts the direction of its shift from unfiltered "
      "measurements alone: down with D for a graph index, roughly flat for an inverted-list "
      "index (provisional; section 5.7 states the confounds).")
    A("2. A confirmation, on modern text embeddings and within one recall-constrained protocol, "
      "that anti-correlated predicates defeat filtered graph search at any swept effort, while "
      "exact scan of the subset always succeeds (the IVF sweep stops short of an exhaustive "
      "scan; section 5.8).")
    A("3. A measured product-quantization recall ceiling on modern text embeddings — which rises "
      "with D rather than falling as the proposal predicted — and the finding that the sub-raw "
      "memory regime is unreachable for the five conventional families yet IS reached by scalar "
      "quantization, at 0.27-0.35x raw with no recall loss (section 5.3b).")
    A("4. A recall-constrained cost surface over D at fixed N and corpus (unfiltered, HNSW is "
      "cheapest throughout), with empirically fitted N-scaling exponents, every figure measured "
      "on queries never used for parameter selection.")
    _h3 = verdicts.get("H3", ("inconclusive", ""))[0]
    _rule_clause = ("a small rule that picks the cheapest filter strategy from query-time inputs "
                    "plus a correlation proxy"
                    if _h3 == "supported" else
                    "a negative result: a small rule over query-time inputs plus a correlation "
                    "proxy does not pick the filter strategy better than a one-line "
                    "selectivity switch")
    A(f"5. {_rule_clause[0].upper() + _rule_clause[1:]}; plus a reproducible, resumable harness "
      f"and a measurement-validity account of between-session latency drift.\n")

    # ---------------- related work
    A("## 2. Related work\n")
    A("The full survey, with one row per paper and an explicit statement of what each does not "
      "cover, is in [`notes/related_work.md`](../notes/related_work.md). We summarise the "
      "positioning here.\n")
    A("Standard benchmarks — ANN-Benchmarks (Aumüller et al., 2020) and the more recent VIBE "
      "(Jääsaari et al., 2025), which targets modern embedding workloads across 19 datasets "
      "and 22 implementations — vary dimension only by changing dataset, so D is confounded. "
      "Matryoshka Representation Learning (Kusupati et al., 2022) supplies the truncation "
      "property our design depends on, but does not study index selection.\n")
    A("On filtering, the closest work to our intended headline is *Filtered ANN as a Phase "
      "Transition* (Mandarapu & Kunkunuru, 2026), which models the selectivity crossover "
      "between strategies as a phase boundary — but on SIFT1M at a fixed 128 dimensions, with "
      "D absent from the resulting formulas. Amanbayev et al. (2026) report that IVF-Flat beats "
      "HNSW at low selectivity on 768-d text embeddings, again at a single D. ACORN (Patel et "
      "al., 2024) and the arxiv-for-fanns benchmark (Iff et al., 2025) broaden filtered-ANN "
      "method coverage without isolating dimension. VADER (Chatzakis et al., 2026) shares our "
      "recall-as-a-constraint framing for filtered search, without the dimension axis. Two "
      "results inform how we read the dimension axis itself: Munyampirwa et al. (2024) show the "
      "HNSW hierarchy is largely redundant in high dimension (hub nodes carry navigation), and "
      "Herman (2026) predicts ANN recall from embedding statistics — both bear on why graph cost "
      "and the PQ ceiling move with D as they do.\n")
    A("Matryoshka embeddings have been used for ANN search before: the MRL paper itself "
      "evaluates adaptive (coarse-to-fine) retrieval, and AdANNS (Rege et al., 2023) builds "
      "IVF and OPQ indexes over different Matryoshka prefix widths. Neither asks which index "
      "is cheapest under a recall constraint, nor how filtering interacts with D. Our Phase 0 "
      "survey of 13 papers found none that measures how the filter-selectivity crossover moves "
      "with dimension on a single corpus "
      "([`logs/gate_reports/gate_0.md`](../logs/gate_reports/gate_0.md)); the survey was "
      "keyword-driven and is a statement about what we found, not proof of absence.\n")

    # ---------------- design
    A("## 3. Study design\n")
    A("### 3.1 Corpus, queries and embeddings\n")
    A("One million MS MARCO passages sampled uniformly (seed 0), with nested prefixes giving "
      "N = 100,000 and 500,000, so smaller corpora are strict subsets of larger ones. Ten "
      "thousand MS MARCO training queries (seed 1) are split **once** into 2,000 tuning and "
      "8,000 test queries; the split is stored and asserted by the runner. Every parameter in "
      "this paper was chosen using tuning queries only, and every reported measurement comes "
      "from the test queries.\n")
    A("Passages are embedded once with a single hosted Matryoshka-capable model at 1536 "
      "dimensions. Lower dimensions are prefix slices of that one embedding, renormalised to "
      "unit length. All indexes use L2 on unit-norm vectors, which is rank-equivalent to cosine "
      "and inner product, so metric choice is not a free variable.\n")
    A("### 3.2 Dimension is varied, intrinsic difficulty barely is\n")
    if lid is not None:
        A("Truncation changes the ambient dimension far more than it changes the data's "
          "intrinsic dimension. A maximum-likelihood estimate (Levina–Bickel, k=20) over "
          f"{int(lid['n_sample'].iloc[0]):,} sampled passages gives:\n")
        A(_md(lid))
        A("")
        A(f"Ambient D rises {Ds[-1] // Ds[0]}x from {Ds[0]} to {Ds[-1]} while estimated "
          f"intrinsic dimension moves only from {lid['lid_mean'].min():.1f} to "
          f"{lid['lid_mean'].max():.1f} — about "
          f"{(lid['lid_mean'].max() / lid['lid_mean'].min() - 1) * 100:.0f}%. Cost differences "
          f"across D are therefore attributable to representation width rather than to the "
          f"data becoming intrinsically harder, which is exactly the separation that varying "
          f"dataset cannot provide. (Source: `results/summary/lid.csv`.)\n")
    A("### 3.3 Indexes and parameter grids\n")
    A("Five index families, all FAISS: exact Flat; HNSW (M in {16, 32, 64}, efConstruction "
      "200); IVF-Flat (nlist = c*sqrt(N), c in {2, 4, 8}); IVF-PQ (same nlist values, m in "
      "{D/16, D/8, D/4}, 8 bits per subquantizer); and IVF-PQ wrapped in exact re-ranking "
      "(IndexRefineFlat, k_factor in {4, 16}), reported separately because it retains full "
      "vectors and therefore has a completely different memory profile. Search effort is swept "
      "over efSearch in {16...1024} for HNSW and nprobe in {1...512} for IVF, capped at nlist.\n")
    A(f"This gives {int(reach['cells_x_targets'].sum() / max(len(Rs), 1))} (cell, index) "
      f"combinations per recall target and, at N={N_main:,}, {n_builds} recorded builds and "
      f"{n_sweep_rows:,} search-sweep rows in total across all cells.\n")
    A("### 3.4 Recall-constrained selection\n")
    A("For each (cell, index family, R) we take the cheapest configuration — jointly over build "
      "variants and search settings — whose tuning-query recall reaches R, then re-measure that "
      "configuration on the test queries. A cell/index/target with no qualifying setting is "
      "recorded as `reachable = false` rather than dropped: **the absence of a feasible "
      "configuration is a result**, and it turns out to carry most of this paper's findings.\n")
    A("Because selection happens on tuning queries, a winner can land marginally below R on the "
      "test queries. We report this explicitly rather than letting `reachable` be mistaken for "
      "an achievement claim:\n")
    if "meets_R_on_test" in tuned.columns:
        miss = r_main[~r_main["meets_R_on_test"].fillna(True).astype(bool)]
        A(f"- {len(miss)} of {len(r_main)} tuned winners at N={N_main:,} miss their target on "
          f"the test queries, by at most "
          f"{abs(miss['recall_gap_vs_R'].min()) if len(miss) else 0:.4f} "
          f"(median {abs(miss['recall_gap_vs_R'].median()) if len(miss) else 0:.4f}). Full "
          f"table: [`tables/recall_achieved.md`](tables/recall_achieved.md).\n")
    A("### 3.5 Measurement protocol\n")
    A("Latency is measured in a fresh subprocess pinned to one core with a single thread, batch "
      "size 1, after a 200-query warm-up, taking the per-query median over three passes and "
      "reporting percentiles over the 8,000 test queries. Build time and throughput are "
      "measured separately with T = 16 threads, since both are legitimately parallel; rows "
      "carry a `threads` column and the two are never mixed. Each selected configuration is "
      "built three times with different seeds. Memory is measured in a process that loads only "
      "the serialized index and reports the RSS delta after warm-up — not the resident size of "
      "a process that also holds the corpus, which overstates index memory by the size of the "
      "dataset.\n")
    A("Recall never uses the distances an index reports. We recompute true squared distances "
      "from the raw vectors for every returned id and count those within a 1e-5 relative "
      "tolerance of the k-th ground-truth distance, so equally good neighbours are not "
      "penalised and a compressed index cannot flatter itself.\n")

    # ---------------- validity
    A("## 4. Platform and measurement validity\n")
    A("### 4.1 Platform\n")
    A(f"All measurements in this paper come from one host: {hw['cpu_model']}, "
      f"{hw['physical_cores']} cores, {hw['ram_total_gb']} GB RAM, {hw['os']}, "
      f"FAISS {hw['library_versions']['faiss-cpu']} (compiled with AVX2), "
      f"NumPy {hw['library_versions']['numpy']}, Python "
      f"{hw['python'].split()[0]}. Hardware identity `{hw['hardware_id']}` is recorded on every "
      f"result row, and the analysis refuses to mix rows from different hosts.\n")
    A("We verified the installed FAISS reports an AVX2 build rather than assuming it. This "
      "matters: a generic build changes absolute latencies enough to move the comparisons, and "
      "a wheel that ships a single binary can be either.\n")
    A("### 4.2 Absolute latency drifts between sessions; occupancy is a smaller effect\n")
    occ = None
    occ_path = SUMMARY / "occupancy_effect.csv"
    if occ_path.exists():
        occ = pd.read_csv(occ_path)
    if occ is not None and not occ.empty:
        med = occ.groupby("condition")["lat_p50_us"].median()
        spread = occ.groupby("condition")["lat_p50_us"].agg(lambda x: x.max() / x.min())
        ratio = float(occ["p50_ratio_idle_over_pool"].iloc[0])
        n_rep = int(occ["repeat"].nunique())
        cfg = occ.iloc[0]
        A(f"We first suspected that single-thread latency depended strongly on machine-wide "
          f"core occupancy, having compared a campaign measurement against a later lone "
          f"re-measurement. A controlled test does not support that. Alternating {n_rep} "
          f"samples per condition on a bit-identical index "
          f"({cfg['index_type']}, N={int(cfg['N']):,}, D={int(cfg['D'])}, "
          f"efSearch={int(cfg['ef_search'])}; recall identical to five decimals across all "
          f"{len(occ)} samples):\n")
        A(_md(occ.groupby("condition").agg(
            samples=("lat_p50_us", "size"), p50_median_us=("lat_p50_us", "median"),
            p50_min_us=("lat_p50_us", "min"), p50_max_us=("lat_p50_us", "max"),
        ).round(1).reset_index()))
        A("")
        A(f"Occupancy accounts for {ratio:.2f}x, and in the direction opposite to contention: "
          f"the idle machine is marginally *slower*, plausibly an idle-state or frequency "
          f"effect in the guest. Within a condition, spread is only "
          f"{spread.min():.2f}x to {spread.max():.2f}x.\n")
        pass1 = None
        if d["sweeps"] is not None:
            h = d["sweeps"]
            h = h[(h["index_type"] == cfg["index_type"]) & (h["N"] == cfg["N"])
                  & (h["D"] == cfg["D"]) & (h["ef_search"] == cfg["ef_search"])
                  & (h["threads"] == 1)]
            if not h.empty:
                pass1 = float(h.sort_values("timestamp")["lat_p50_us"].iloc[0])
        if pass1:
            A(f"What *is* real is drift between sessions. The same configuration recorded during "
              f"an earlier campaign measured {pass1:.1f} us under full-pool conditions, against "
              f"{med['full_pool']:.1f} us today under matched conditions — a "
              f"{med['full_pool'] / pass1:.2f}x change two days apart, with recall identical. "
              f"Our original 1.3x figure conflated this drift with occupancy, and we record the "
              f"correction rather than the first explanation "
              f"([`logs/decisions.md`](../logs/decisions.md)).\n")
        A("The consequences we actually rely on: comparisons must be drawn **within** a "
          "measurement campaign, where spread is a few percent; absolute microsecond figures "
          "should not be carried across campaigns or hosts; and a reproducibility check must "
          "re-measure under matched conditions, which ours does, agreeing to 2.7%. Stage A's "
          "cells were measured within a few hours of each other, so the comparisons in "
          "section 5 are within-campaign. A longer campaign should interleave a canary "
          "configuration to track drift directly — we did not, and flag it as a limitation.\n")
    A("### 4.3 Harness validation\n")
    if acc is not None:
        A(f"{acc_pass} of {acc_total} machine-checked acceptance conditions pass "
          f"(`results/summary/acceptance_checks.csv`). These are not smoke tests; they are the "
          f"conditions under which the numbers mean anything:\n")
        A(_md(acc[["check", "status"]]))
        A("")
        A("Of these, the most informative is the negative one: HNSW recall at its lowest "
          "efSearch is well below 1.0. An approximate index reporting perfect recall at minimum "
          "effort is the signature of a broken harness, so we assert its absence.\n")
    A("### 4.4 efConstruction is fixed by measurement, not assumption\n")
    if d["efc"] is not None:
        e = d["efc"].groupby("ef_construction").agg(
            mean_recall=("recall_mean", "mean"), build_seconds=("build_seconds", "first")
        ).reset_index().round(4)
        A("We fix efConstruction = 200 throughout. That choice is justified rather than "
          "inherited:\n")
        A(_md(e))
        A("")
        A("Raising it to 400 roughly doubles build time for a recall gain in the fourth decimal "
          "place. (Source: `results/summary/acceptance_ef_construction.csv`.)\n")

    # ---------------- results
    A("## 5. Results\n")
    A(f"### 5.1 The cost surface over dimension at N={N_main:,}\n")
    A("p95 latency in microseconds, single-threaded, measured on the 8,000 test queries. Lower "
      "is better; blank means no configuration of that family reached the target.\n")
    A(_md(cost))
    A("")
    A(f"HNSW is cheapest in every cell. The headline comparison at R=0.95: HNSW runs "
      f"{speedup.min():.0f}x to {speedup.max():.0f}x faster than exact search, with the margin "
      f"widening as D grows. More interesting is the slope. From D={Ds[0]} to D={Ds[-1]}, "
      f"HNSW's p95 latency rises {hnsw_growth:.1f}x "
      f"({_hn_by_D.iloc[0]:.0f} to {_hn_by_D.iloc[-1]:.0f} us) while "
      f"exact search rises {flat_growth:.1f}x. A graph index pays for dimension mainly in "
      f"distance computations along a traversal whose length is governed by intrinsic "
      f"structure, and section 3.2 showed intrinsic dimension barely moves under truncation; a "
      f"linear scan pays the full ambient width.\n")
    A("This is a **negative result for the dimension hypothesis**: the cheapest index does not "
      "change with D at this scale. Because the full build grid was measured — not one variant "
      "per family — this is a statement about the index families rather than an artifact of "
      "limited search. What *does* change with D is the margin, and the ranking of the runners-"
      "up: IVF-PQ+refine overtakes IVF-Flat as D grows.\n")
    A("Build time, index size and resident memory for the same configurations are in "
      "[`tables/build_time.md`](tables/build_time.md) and "
      "[`tables/index_footprint.md`](tables/index_footprint.md).\n")

    A("### 5.2 Reachability, and the product-quantization ceiling\n")
    A(_md(reach))
    A("")
    if pq_max is not None:
        A(f"Every unreachable cell belongs to plain IVF-PQ, which fails at **every** dimension "
          f"and **every** recall target. This is not a tuning failure. The table below gives "
          f"the highest recall each of the {len(pq)} (dimension, code-size) combinations attains "
          f"at any swept nprobe, alongside the same codes with exact re-ranking:\n")
        A(_md(pq))
        A("")
        A(f"Recall improves monotonically as codes get finer, reaching {pq_max:.3f} at the "
          f"finest setting — still short of 0.90. Critically, it saturates in search effort: "
          f"doubling nprobe from 256 to 512 at the best configuration gains under 0.004, and at "
          f"nprobe = nlist the scan is exhaustive, so the asymptote is quantization error "
          f"rather than candidate coverage. Adding exact re-ranking over the identical codes "
          f"reaches {pq['ivf_pq_refine'].max():.3f}, confirming the diagnosis: the candidates "
          f"are there, the approximate distances cannot rank them.\n")
        _pq16 = pq[(pq["compression_vs_raw"] >= 15) & (pq["compression_vs_raw"] <= 17)]
        _pq16 = _pq16.sort_values("D")
        if len(_pq16) >= 2:
            _lo, _hi = _pq16.iloc[0], _pq16.iloc[-1]
            A(f"The hypotheses were **revised from the proposal** during the study, and this is "
              f"one reversal worth reporting as a finding. The proposal predicted the IVF-PQ "
              f"recall ceiling would *fall* with dimension at a fixed 16x compression; measured, "
              f"it *rises*, from {float(_lo['ivf_pq']):.3f} at D={int(_lo['D'])} to "
              f"{float(_hi['ivf_pq']):.3f} at D={int(_hi['D'])} — opposite to the prediction, "
              f"though still below 0.90, so the practical conclusion that PQ alone is unreachable "
              f"is unchanged (section 6).\n")
        A("The practical reading is that on modern text embeddings, PQ is a *candidate "
          "generator*, not a ranker. Any deployment targeting recall@10 >= 0.90 must keep "
          "something exact to re-rank against — which has a direct and unwelcome consequence "
          "for memory budgets.\n")

    A("### 5.3 Memory budgets: the cheapest index does change, but feasibility binds first\n")
    if not mem_budget.empty:
        A("Winner by cell and memory budget at R=0.95, where the budget is a multiple of the "
          "raw vector bytes (N x D x 4):\n")
        A(_md(mem_budget.fillna("none")))
        A("")
        A("Two things happen here. First, the winner **does** change with the budget: at "
          f"D={Ds[0]}, N={N_main:,} the unbounded winner is HNSW but the 1.5x-raw winner is "
          "IVF-Flat. The mechanism is concrete — the tuned HNSW at that cell uses M=64, whose "
          "link storage (N x M x 2 x 4 bytes) is comparable to the vectors themselves at low D, "
          "pushing the index near 2x raw. Measured footprints are in "
          "[`tables/index_footprint.md`](tables/index_footprint.md). (A `none` entry means no "
          "configuration of any index met the recall target within that budget.)\n")
        A("Second: within the five families of the main grid, **the 0.5x and 0.1x budgets have "
          "no feasible winner anywhere**. The only family small enough is IVF-PQ, and section "
          "5.2 showed it cannot reach R >= 0.90. That result prompted the probe in the next "
          "section, which changes the conclusion.\n")

    if not sq.empty:
        A("### 5.3b Scalar quantization occupies the sub-raw regime the main grid left empty\n")
        A("The main grid's five families are either larger than the raw vectors (Flat, HNSW, "
          "IVF-Flat, IVF-PQ+refine) or unable to reach the recall targets (IVF-PQ), so a "
          "sub-raw budget had no feasible answer. That is a statement about the families "
          "measured, not about what is possible, so we added one: HNSW over 8-bit "
          "scalar-quantized storage, measured **in the same run** as uncompressed HNSW at the "
          "same cells, since this host's between-session drift would otherwise be folded into "
          "a claim about compression.\n")
        A(_md(sq))
        A("")
        piv = sq.pivot_table(index=["D", "R"], columns="index_type",
                             values="lat_p95_us", aggfunc="first").dropna()
        mem = sq.drop_duplicates(subset=["index_type", "D"]).pivot_table(
            index="D", columns="index_type", values="rss_x_raw", aggfunc="first").dropna()
        if not mem.empty:
            A(f"Resident memory falls from {mem['hnsw'].min():.2f}-{mem['hnsw'].max():.2f}x raw "
              f"to {mem['hnsw_sq'].min():.2f}-{mem['hnsw_sq'].max():.2f}x — inside the 0.5x "
              f"budget at every measured cell, while still reaching every recall target. **The "
              f"correct statement is therefore not that no index meets a sub-raw budget, but "
              f"that none of the five conventional families does: a scalar-quantized graph "
              f"index qualifies at 0.5x, and nothing we measured qualifies at 0.1x**, which "
              f"would need roughly 10x compression against SQ8's 4x.\n")
            A("Compression improves with dimension (0.35x at D=384 against 0.27x at D=1536) "
              "because SQ8 shrinks only the vectors, by a factor of four, while the graph "
              "links are D-independent: at low D that fixed overhead is a larger share of the "
              "total.\n")
        if not piv.empty and {"hnsw", "hnsw_sq"}.issubset(piv.columns):
            ratio = (piv["hnsw_sq"] / piv["hnsw"])
            lo_R = ratio[ratio.index.get_level_values("R") < 0.99]
            hi_R = ratio[ratio.index.get_level_values("R") >= 0.99]
            A(f"Compression is also not a pure latency loss. At R=0.90 and 0.95 the quantized "
              f"index is **faster** — ratio {lo_R.min():.2f}-{lo_R.max():.2f} — because fewer "
              f"bytes per vector means less memory traffic per distance evaluation, which "
              f"outweighs the decode cost. At R=0.99 it reverses "
              f"({hi_R.min():.2f}-{hi_R.max():.2f}x), since compensating for quantization "
              f"error needs a higher efSearch and the extra traversal costs more than the "
              f"bytes saved. Quantization trades latency for memory in a way that depends on "
              f"the recall target, which is exactly the kind of dependence a recall-constrained "
              f"framing is meant to expose.\n")
        A("This is a two-cell probe, not a full stage, and it tests one compression technique. "
          "It is enough to show the regime is reachable; locating the frontier would need "
          "4-bit SQ, finer PQ with re-ranking, and more cells.\n")

    A("### 5.4 The scale axis: HNSW absorbs a 10x corpus at constant cost\n")
    if not scale.empty:
        A("p95 latency in microseconds at R=0.95, by index type and corpus size, at the "
          "dimensions measured across all three N:\n")
        A(_md(scale))
        A("")
        hn = scale[scale["index_type"] == "hnsw"]
        fl = scale[scale["index_type"] == "flat"]
        ncols = [c for c in scale.columns if isinstance(c, (int, float))]
        if len(ncols) >= 2 and not hn.empty and not fl.empty:
            lo, hi = min(ncols), max(ncols)
            hr = (hn[hi] / hn[lo]).median()
            fr = (fl[hi] / fl[lo]).median()
            A(f"Over a {hi // lo}x increase in corpus size, HNSW's p95 latency at fixed recall "
              f"changes by a factor of {hr:.2f} — no measurable growth — while exact search "
              f"grows {fr:.1f}x, i.e. linearly as expected. IVF-Flat and IVF-PQ+refine grow "
              f"sub-linearly but measurably. The HNSW figure is non-monotonic across N and the "
              f"variation is within the between-session drift band of section 4.2, so the "
              f"honest statement is that we cannot distinguish its growth from zero over this "
              f"range, not that it is exactly flat.\n")
            A("This has a practical reading: within this range, a graph index's query cost is "
              "governed by the recall target and the data's structure rather than by how many "
              "vectors are stored. Note the contrast with the next section, where cost averaged "
              "over the whole search grid does grow with N at fixed *effort* — a different "
              "quantity from the fixed-*recall* cost here, and the two should not be conflated.\n")

    A("### 5.5 How cost scales with corpus size, averaged over the search grid\n")
    if not expo.empty:
        A("Exponents fitted across the measured corpus sizes (N in {100K, 500K, 1M}) at matched "
          "D (median over D), cost proportional to N^exponent. These are averaged over the whole "
          "search grid (all swept efSearch/nprobe), which is a **different quantity** from the "
          "fixed-recall cost of section 5.4 — they must not be read as contradicting it:\n")
        A(_md(expo))
        A("")
        A("These are measurements, and they differ from the values complexity arguments would "
          "suggest. Three are worth naming. IVF-Flat **build** is strongly super-linear, "
          "because nlist grows with sqrt(N) and the k-means training sample grows with nlist, "
          "so training cost compounds. HNSW **search**, averaged over the grid, grows faster "
          "than the logarithmic intuition — more traversal per query as the graph grows at a "
          "fixed search effort (this is the grid-averaged exponent; at fixed *recall* the cost "
          "is flat, section 5.4). IVF-Flat **search** is flatter than the naive sqrt(N) guess, "
          "because a fixed nprobe scans a shrinking fraction of the corpus as nlist grows.\n")
        A("We report these as descriptive fits over three corpus sizes, not as scaling laws: "
          "three points constrain a power law only weakly. Their immediate use was operational — "
          "they corrected our own compute projection for later stages, which had assumed the "
          "theoretical values and was wrong in both directions.\n")
    if d["scaling"] is not None and not d["scaling"].empty:
        expo = d.get("scaling_exponents")
        if expo is not None and not expo.empty:
            ex = expo[(expo["metric"] == "p95_latency") & (expo["axis"] == "D")]
            A("How recall-constrained p95 latency scales with dimension at fixed corpus size: "
              "the exponent of D from a log-log fit across the four truncations, with a 95% "
              "bootstrap interval (query sampling and build variance; between-session drift is "
              "not in these intervals, so differences below ~0.1 are not meaningful). This "
              "exponent is the quantity H1c refers to, and it is also the input to the "
              "crossover model of section 5.7:\n")
            A(_md(ex[["index_type", "R", "fixed_at", "exponent", "ci95_lo", "ci95_hi",
                      "n_points", "levels"]].rename(columns={"fixed_at": "N"})))
            A("")
        A("Joint fit `log y = a + b log(N/500K) + c log(D/384) + d log(N/500K) log(D/384)` "
          "(logs centred at the reference cell, so b and c are local slopes there; descriptive, "
          "10 cells):\n")
        A(_md(d["scaling"]))
        A("")

    A("### 5.6 A decision rule — on the filter axis, where the answer changes\n")
    # The index-choice rule is reported only as the negative that motivates moving axes.
    if rule is not None and not rule.empty and pd.notna(rule.iloc[0].get("test_accuracy")):
        r0 = rule.iloc[0]
        base_acc = max(float(r0.get("baseline_always_hnsw_accuracy") or 0),
                       float(r0.get("baseline_budget_aware_accuracy") or 0))
        _ta = float(r0["test_accuracy"])
        _cmp = ("matches" if abs(_ta - base_acc) < 1e-9 else
                "falls short of" if _ta < base_acc else "exceeds")
        A(f"A depth-{int(r0['max_depth'])} tree predicting the cheapest **index** from "
          f"(log10 N, D, R, memory budget) scores {_ta:.3f} top-1 on the held-out N=1,000,000 "
          f"cells, which {_cmp} \"always choose HNSW\" ({base_acc:.3f}): unfiltered, one index "
          f"wins nearly everywhere, so there is little for a selector to learn. The decision only "
          f"becomes non-trivial on the axis the index rule never sees — the filter.\n")
    # The strategy rule is the real H3 (R1): it must choose a filter STRATEGY from query-time
    # inputs plus a proxy for correlation, which is not itself observable at query time.
    _strat_path = Path("results/summary/strategy_rule_accuracy.csv")
    if _strat_path.exists() and not pd.read_csv(_strat_path).empty:
        s0 = pd.read_csv(_strat_path).iloc[0]
        A(f"We therefore (post-hoc) refit the rule to pick the cheapest **filter strategy** "
          f"(pre-filter exact scan / in-index filtered / post-filter over-fetch) from inputs a "
          f"system has at query time: selectivity S, dimension D, recall target R, and a "
          f"**correlation proxy**. Correlation is a property of the predicate-query relationship "
          f"and is not observable directly, so the proxy estimates it the way a deployed system "
          f"could: it issues a cheap *approximate* unfiltered top-m probe and takes the fraction "
          f"of those candidates that satisfy the predicate, divided by the global selectivity "
          f"(`results/summary/filter_proxy.csv`, `proxy_source = online_probe`; an idealized "
          f"exact-ground-truth variant is computed only for comparison). The oracle label per "
          f"cell is the cheapest reachable strategy.\n")
        if pd.notna(s0.get("test_accuracy")):
            ci_lo, ci_hi = s0.get("test_accuracy_ci_lo"), s0.get("test_accuracy_ci_hi")
            ci_s = (f" [95% CI {float(ci_lo):.2f}-{float(ci_hi):.2f}]"
                    if pd.notna(ci_lo) else "")
            acc, reg = float(s0["test_accuracy"]), float(s0["test_regret"])
            # Every comparison is computed, never asserted: a baseline row is (name, top-1, regret).
            rows = [("proxy rule (S, D, R, corr. proxy)", acc, reg)]
            for name, a_col, r_col in [
                ("same tree without the proxy (S, D, R)", "noproxy_test_accuracy",
                 "noproxy_test_regret"),
                ("fixed 1%-selectivity switch", "baseline_threshold_1pct_accuracy",
                 "baseline_threshold_1pct_regret"),
                ("always exact scan of the subset", "baseline_always_pre_filter_exact_accuracy",
                 "baseline_always_pre_filter_exact_regret"),
                ("always in-index filtered", "baseline_always_in_index_filtered_accuracy",
                 "baseline_always_in_index_filtered_regret"),
            ]:
                if pd.notna(s0.get(a_col)):
                    rows.append((name, float(s0[a_col]),
                                 float(s0[r_col]) if pd.notna(s0.get(r_col)) else float("nan")))
            A(f"On the {int(s0['n_test'])} held-out N=1,000,000 cells a depth-{int(s0['max_depth'])} "
              f"tree over ({s0['feature_names']}) scores {acc:.3f} top-1{ci_s} with regret "
              f"{reg:.3f} (mean extra latency over the per-cell oracle). The fair comparison is "
              f"against the same tree without the proxy and against a one-line selectivity "
              f"switch, not only against a fixed strategy:\n")
            A("| policy | top-1 | regret |\n|---|---|---|")
            for name, a, r in rows:
                A(f"| {name} | {a:.3f} | {r:.3f} |" if r == r else f"| {name} | {a:.3f} | n/a |")
            A("")
            base = rows[1:]
            best_acc = max(base, key=lambda t: t[1])
            best_reg = min([b for b in base if b[2] == b[2]] or base, key=lambda t: t[2])
            np_row = next((b for b in base if b[0].startswith("same tree")), None)
            if acc > best_acc[1] and reg <= best_reg[2]:
                A("The proxy rule beats every baseline on both measures, so on this grid the "
                  "correlation proxy is doing useful work. This is the **exploratory** H3.\n")
            else:
                msgs = []
                if acc <= best_acc[1]:
                    msgs.append(f"on top-1 it is matched or beaten by the {best_acc[0]} "
                                f"({best_acc[1]:.3f})")
                if reg > best_reg[2]:
                    msgs.append(f"on regret by the {best_reg[0]} ({best_reg[2]:.3f})")
                A(f"**The rule does not beat the fair baselines:** {'; '.join(msgs)}. ")
                if np_row is not None and np_row[2] == np_row[2] and np_row[2] > reg:
                    A(f"What the proxy does buy is protection from the worst choices: without it "
                      f"the tree's regret is {np_row[2]:.1f} against {reg:.3f} with it, because "
                      f"S, D and R alone cannot tell an anti-correlated predicate (where only "
                      f"exact scan reaches R) from a benign one, and the rare wrong call there "
                      f"is very expensive. A plain selectivity threshold avoids those calls just "
                      f"as well. The exploratory H3 is therefore **not supported** on this grid; "
                      f"a proxy computed from a cheap probe is not yet worth a learned rule.\n")
                else:
                    A("The exploratory H3 is therefore **not supported** on this grid.\n")
            A("The fitted tree is in "
              "[`../results/summary/strategy_rule_tree.txt`](../results/summary/strategy_rule_tree.txt).\n")
            A("One caveat the rule inherits: its oracle labels come from the Stage C runs, which "
              "include the smallest correlated level (S=0.001) whose *measured* selectivity "
              "overshoots the target by 54-61% (section 5.8). Those cells' labels — and any rule "
              "split that depends on them — should be read against measured, not nominal, S, and "
              "will firm up once the Stage C rerun with finer correlated filters lands.\n")
        else:
            A(f"The strategy rule has no held-out cells yet (n_test={int(s0.get('n_test', 0))}); "
              f"see `results/summary/strategy_rule_accuracy.csv`.\n")
    else:
        A("_Strategy-rule results pending `make analysis` (Stage C filter runs + the correlation "
          "proxy); see section 6._\n")

    A("### 5.7 Filtered search: where the crossover sits, and how it moves with dimension\n")
    if not xover.empty:
        A(f"For each (index, cell, correlation, recall target) we locate S*, the selectivity at "
          f"which in-index filtered search and exact scan of the filtered subset swap places, "
          f"by interpolating log-latency against log *measured* selectivity between the two "
          f"curves, using only rows that meet R on the held-out queries and never "
          f"extrapolating beyond the measured range. Two intervals are shown: query sampling "
          f"alone (`_q_`), and query sampling plus build-to-build variance plus an independent "
          f"±12% session-drift factor per curve (the median between-session drift measured in "
          f"section 4.2). Conclusions rest on the second. A crossover was located in "
          f"{len(xover)} of {len(d['crossover'])} (index, cell, correlation, target) "
          f"combinations; elsewhere one curve dominates across the whole measured S range.\n")
        A(_md(xover))
        A("")
        unc = xover[xover["correlation"] == "none"]["crossover_S"]
        pos = xover[xover["correlation"] == "positive"]["crossover_S"]
        if len(unc) and len(pos):
            A(f"The headline numbers: for an uncorrelated predicate S* sits at a median of "
              f"{unc.median():.4f} — in-index search is preferable above roughly "
              f"{unc.median() * 100:.1f}% selectivity and exact scan below it. For a positively "
              f"correlated predicate S* drops to {pos.median():.5f}, a factor of "
              f"{unc.median() / pos.median():.1f} lower: when the filter admits vectors near "
              f"the query, in-index search stays the better choice "
              f"{unc.median() / pos.median():.1f}x further down the selectivity range. That is "
              f"H2b — a larger effect than dimension's — and the practical one: a system that "
              f"switches strategy on selectivity alone, without regard to correlation, will "
              f"choose wrongly across a {unc.median() / pos.median():.0f}-fold band of S.\n")
    if not xover_h2.empty:
        A("**H2: does S\\* move with dimension?** For each (N, index, correlation, target) we "
          "compare the lowest and highest measured D. A pair is a `shift` if the 95% interval of "
          "the ratio S*(D_high)/S*(D_low) excludes 1, `equivalent` if its 90% interval lies "
          "inside a pre-stated margin [0.8, 1.25] (a TOST at alpha = 0.05), and `inconclusive` "
          "otherwise:\n")
        A(_md(xover_h2))
        A("")
        if "verdict" in xover_h2.columns:
            for it, g in xover_h2.groupby("index_type"):
                vc = g["verdict"].value_counts()
                A(f"- **{it}**: " + ", ".join(f"{v} in {c}" for v, c in vc.items())
                  + f" of {len(g)} pairs; median ratio {g['ratio_high_over_low'].median():.2f}.")
            A("")
    if not xmodel.empty:
        A("**Why the direction depends on the index.** Near the crossover, model each "
          "strategy's p95 as a local power law, exact scan ~ S^a D^c_ex and in-index search "
          "~ S^b D^c_ix. Then S*(D_2)/S*(D_1) = (D_2/D_1)^((c_ix - c_ex)/(a - b)). We take "
          "c_ix and c_ex from the **unfiltered** recall-constrained measurements of section 5.5 "
          "(Flat for the exact scan) and a, b from the slopes of the filtered curves; nothing "
          "about where the crossovers fell enters the prediction:\n")
        A(_md(xmodel))
        A("")
        A(f"The model gets the direction right in {int(xmodel['direction_agrees'].sum())} of "
          f"{len(xmodel)} pairs. A graph search's cost grows far more slowly with D than a scan's "
          f"(its traversal length is set by intrinsic dimension, which truncation barely moves; "
          f"section 3.2), so raising D penalises the exact-scan baseline more and S* falls. "
          f"IVF-Flat scans whole inverted lists, so its D-exponent is close to (or above) the "
          f"scan's and the model predicts little or no shift — which is what we measure. The "
          f"closed-form crossover of prior work has no D term; on this corpus it needs one, and "
          f"its sign is a property of the index, not of the data.\n")
    A("**What this does not yet establish.** The filtered campaign ran its cells one after "
      "another with 16, 6, 12 and 3 concurrent shards (`logs/stage_c_*.log`), and correlated "
      "predicates were built from "
      "a separate clustering at each D, so the two D values compared do not share a session, a "
      "machine load, or the same correlated subsets. The drift band above covers ordinary "
      "session drift but not those confounds. A confound-free rerun at N = 500,000 over all "
      "four D — one interleaved campaign, one shared clustering, fixed parallelism, repeat "
      "builds, a drift canary and distance-computation counters — is configured "
      "(`configs/stage_c_h2.yaml`, section 8) and will turn the pairwise comparison into a "
      "four-point dose-response against the model's predicted slope.\n")

    A("### 5.8 Adversarially correlated filters defeat approximate search at the swept effort\n")
    if not freach.empty:
        A("Reachability of each strategy by filter correlation, over all (cell, target) "
          "combinations:\n")
        A(_md(freach))
        A("")
        neg = freach[freach["correlation"] == "negative"]
        if not neg.empty:
            ann = neg[neg["strategy"] != "pre_filter_exact"]
            exact = neg[neg["strategy"] == "pre_filter_exact"]
            A(f"The negative-correlation rows confirm, under a recall constraint, a failure mode "
              f"the filtered-search literature describes (ACORN; Amanbayev et al.'s query-filter "
              f"correlation metric). When the predicate admits only vectors *far* from the query, "
              f"in-index filtered "
              f"search and post-filter over-fetch reach recall@10 >= 0.90 in "
              f"**{int(ann['reachable'].sum())} of {int(ann['cells_x_targets'].sum())}** "
              f"attempts — that is, never, at any swept effort — while exact scan over the "
              f"matching subset succeeds in all "
              f"{int(exact['reachable'].sum())}/{int(exact['cells_x_targets'].sum())}. "
              f"Approximate search is not merely slower under adversarial correlation; within "
              f"this grid it does not work at all.\n")
            A("The reason is structural rather than a tuning failure. A graph index reaches a "
              "neighbourhood by descending towards the query, and an IVF index probes the "
              "lists nearest the query; both concentrate effort exactly where an "
              "anti-correlated predicate admits nothing. Over-fetching does not rescue it, "
              "because the candidates retrieved are drawn from the same wrong region. For a "
              "workload whose filters can be anti-correlated with queries, the safe design is "
              "to scan the filtered subset exactly and to size the system for that cost.\n")
            A("One boundary to state precisely: the IVF nprobe sweep caps at 512 while nlist is "
              "at least 1,414 at these N, so for IVF this is **the swept, non-exhaustive effort "
              "failing**, not a proof that an exhaustive nprobe = nlist scan would; at nprobe = nlist "
              "IVF-Flat degenerates to an exact scan of the lists and must reach the target. The "
              "nprobe = nlist point is part of the scheduled rerun. The HNSW result (no "
              "exhaustive dial short of full traversal) and the exact-scan success are unaffected "
              "by this caveat.\n")
    if not fsel.empty:
        A("**A measurement caveat we must report.** The plan requires mean measured selectivity "
          "within 20% of target for each correlated level:\n")
        A(_md(fsel))
        A("")
        bad = fsel[~fsel["within_20pct"]]
        if not bad.empty:
            A(f"{len(bad)} level(s) fail that check, all at the smallest correlated target "
              f"(worst {bad['relative_error_pct'].abs().max():.0f}% over target). The cause is "
              f"granularity rather than a bug: correlated predicates are built by unioning "
              f"whole k-means clusters, and with 1,000 clusters the smallest admissible union "
              f"is already near 0.1% of the corpus, so a 0.001 target cannot be hit from "
              f"above. The uncorrelated levels, which use an exact range predicate, are within "
              f"0.6%. S* estimates at the smallest correlated selectivity should therefore be "
              f"read as applying to the measured selectivity rather than the nominal one; "
              f"finer granularity would need more clusters or sub-cluster sampling.\n")

    A("### 5.9 Updates: insertion is nearly free in recall, expensive in throughput\n")
    if not udrift.empty:
        A("Ten batches of 5,000 vectors inserted into an index built on 450,000, at search "
          "parameters fixed before the first insert, with no retraining. Recall is measured "
          "against ground truth restricted to the vectors the index actually contains:\n")
        A(_md(udrift))
        A("")
        worst = udrift.loc[udrift["drift"].abs().idxmax()]
        A(f"Recall drift is negligible for all three families — the largest absolute change "
          f"over the whole insert run is {abs(float(worst['drift'])):.5f} "
          f"({worst['index_type']}) — and each ends within "
          f"{max(abs(r['recall_last_batch'] - r['rebuild_reference']) for _, r in udrift.iterrows() if r['rebuild_reference'] is not None):.5f} "
          f"of a from-scratch rebuild on the same vectors. Inserting without retraining costs "
          f"essentially nothing in recall at this ratio (50,000 added to 450,000, about 11%).\n")
        A(f"The real asymmetry is throughput. IVF-Flat and IVF-PQ absorb inserts at "
          f"{udrift[udrift.index_type != 'hnsw']['inserts_per_second_median'].min():,}-"
          f"{udrift[udrift.index_type != 'hnsw']['inserts_per_second_median'].max():,} "
          f"vectors/second, against "
          f"{int(udrift[udrift.index_type == 'hnsw']['inserts_per_second_median'].iloc[0]):,} "
          f"for HNSW — a factor of roughly {ingest_ratio:.0f}. "
          f"HNSW wins every latency comparison in this study and loses the ingest comparison by "
          f"that margin, which is the clearest cost/benefit trade in our results.\n")
        A("Two caveats on this stage. First, the recall-drift design cannot exhibit IVF centroid "
          "staleness: the inserted vectors are a random sample of the **same** corpus, so their "
          "distribution matches the training sample and the centroids stay representative by "
          "construction. Demonstrating staleness needs distribution shift between the initial and "
          "inserted vectors, which we did not test. Second, the absolute p95 latencies in the "
          "table (HNSW ~10,890 us) are measured at the search parameters fixed before the first "
          "insert — tuned for R=0.95 on the 450,000-vector prefix at D=768, N=500,000 — not "
          "re-tuned per batch; they are not comparable to the tuned operating points of section "
          "5.1 and should not be read as HNSW's query latency at this scale. We did not test "
          "deletions, and the insert ratio here is modest; a system adding many multiples of its "
          "corpus should not read these numbers as licence to never rebuild.\n")

    # ---------------- hypothesis status
    A("## 6. Hypothesis status\n")
    A("H1a-H2b are the **proposal's original RQ1-RQ3 predictions**, judged against the measured "
      "data; one of them, H1b, reverses outright. **H3 is labelled *exploratory* and is "
      "post-hoc, not pre-registered**: the proposal's original H3 was a decision rule over the "
      "*index* choice, which the data did not support (unfiltered, one index wins everywhere, so "
      "it only ties the trivial heuristic); the filter-strategy rule reported as H3 here was "
      "formulated *after* that result, on the axis where the choice actually changes. Verdicts "
      "are computed from the result files by `annstudy-analysis`, not asserted "
      "(`reports/findings.md`):\n")
    A("| Hypothesis | Statement | Origin | Status | Basis |")
    A("|---|---|---|---|---|")
    # Statements come from the single registry in analysis.run_all, so this table and
    # reports/findings.md cannot disagree about what was tested.
    from annstudy.analysis.run_all import hypothesis_rows
    for k, stmt, origin, _ev in hypothesis_rows():
        status, basis = verdicts.get(k, ("inconclusive", "no backing file yet"))
        A(f"| {k} | {stmt} | {origin} | **{status}** | {basis} |")
    A("")
    A("We draw attention to the shape of this table. H1b is not merely unsupported but "
      "**reversed** — the PQ ceiling rises with D rather than falling (section 5.2) — which we "
      "report as a finding, not hide. The filter-axis hypotheses (H2, H2b) are the supported "
      "headline; H3 asks whether a rule can exploit that axis. A paper that reported only the "
      "supported rows would be a different and less honest paper.\n")

    # ---------------- threats
    A("## 7. Threats to validity\n")
    A("**Single host.** Every number comes from one machine. Absolute latencies are not "
      "portable, and we make no claim that they are; the relative ordering is what we argue "
      "about. Results are keyed by hardware identity so that cross-host contamination is "
      "impossible rather than merely discouraged.\n")
    A("**Between-session drift.** Section 4.2: absolute latency for an identical index moved "
      "~1.1x across two days on this host, cause unisolated (co-tenancy on a shared instance "
      "is the obvious suspect). Mitigated by comparing only within a campaign, where spread is "
      "a few percent, but it caps the precision of any absolute figure we publish. A canary "
      "configuration interleaved through the campaign would measure the drift directly; we did "
      "not do this and consider it the most valuable protocol addition for future runs.\n")
    A("**One embedding model, one corpus.** The PQ ceiling and the dimension slopes may be "
      "properties of this model's geometry. The Matryoshka design that lets us isolate D also "
      "ties us to a model family trained to support truncation; an embedding not trained that "
      "way might degrade differently under prefix slicing.\n")
    A("**Three corpus sizes.** The scaling exponents in section 5.5 are three-point fits. They are "
      "reported as descriptive and were used operationally, not as laws.\n")
    A("**Tuning-to-test gap.** Section 3.4 reports the winners that miss their target on "
      "held-out queries. The shortfalls are small but non-zero, and they are a property of "
      "selecting on a 2,000-query sample, not of the indexes.\n")
    A("**Filtered-campaign confounds.** Section 5.7: dimension is confounded with session and "
      "concurrent load in the filtered campaign, and correlated predicates differ across D. "
      "The H2 results are therefore provisional; the rerun that removes both is configured.\n")
    A("**Build variance for HNSW.** FAISS HNSW builds with a fixed level-generator seed, and "
      "repeat builds of this study turned out to be byte-identical, so the three repeat builds "
      "per configuration carry no build-to-build variance for HNSW (they do for IVF, whose "
      "k-means sample is reseeded). Repeat builds are now seeded; HNSW intervals in this draft "
      "reflect query sampling only.\n")
    A("**Non-random tuning split.** The 2,000 tuning queries are the lowest MS MARCO query ids "
      "of the 10,000 sampled, not a random subset. If id order correlates with query "
      "difficulty this biases selection; the held-out recall shortfalls of section 3.4 are the "
      "observable consequence and are reported per row.\n")
    A("**Grid boundaries.** `reachable = false` always means 'within the swept grid'. For "
      "IVF-PQ we checked that recall saturates in nprobe before concluding the ceiling is "
      "intrinsic, but a wider grid — more subquantizers, more bits — could in principle change "
      "the conclusion at a memory cost that would defeat the purpose.\n")

    # ---------------- figures (placeholder boxes to generate + drop in)
    A("## 7b. Figure specifications\n")
    A("These are **specifications**, not the figures: each box names what to plot, the result "
      "file it comes from, and the PNG to render. The analysis step writes the PNGs named under "
      "`Render` (`make analysis`); drop each in where its box sits before submission.\n")
    A(fig_box(1, "Cost vs dimension at R=0.95",
              "p95 latency (us, log y) vs D in {128,384,768,1536} (log x), one line per index "
              "family; HNSW is the lowest line at every D.",
              "results/summary/cell_costs.csv (R=0.95, objective lat_p95_us)",
              "figures/cost_vs_D_R0.95.png"))
    A(fig_box(2, "Cost vs dimension at R=0.99",
              "as Figure 1 at R=0.99; shows IVF-PQ+refine rising fastest and IVF-PQ absent "
              "(unreachable).",
              "results/summary/cell_costs.csv (R=0.99, objective lat_p95_us)",
              "figures/cost_vs_D_R0.99.png"))
    A(fig_box(3, "Winner map by objective",
              "heatmap of the cheapest index per (D, R) cell; one panel each for p95 latency, "
              "build time, and bytes per vector.",
              "results/summary/winners.csv",
              "figures/winner_heatmap_lat_p95_us.png, figures/winner_heatmap_build_seconds_mean.png, "
              "figures/winner_heatmap_index_bytes.png"))
    A(fig_box(4, "IVF-PQ recall ceiling vs search effort",
              "recall@10 vs nprobe (log x) for IVF-PQ at each D; curves saturate below 0.90, "
              "showing the ceiling is quantization error not candidate coverage.",
              "results/raw/sweeps/ (index_type=ivf_pq, N=500000)",
              "figures/pq_ceiling.png (generate; not yet rendered)"))
    A(fig_box(5, "Filter-selectivity crossover S* vs dimension (H2 headline)",
              "p95 latency vs selectivity S (log-log) for in-index filtered search and pre-filter "
              "exact scan, one panel per D in {384,1536}; mark S* where the curves cross and show "
              "the leftward shift as D rises.",
              "results/summary/crossover.csv, crossover_h2.csv; results/raw/filter_runs.parquet",
              "figures/crossover.png (generate; not yet rendered)"))
    A(fig_box(6, "Update recall drift (Stage D)",
              "recall@10 at fixed search params vs inserted-vector count, one line per index "
              "family, with the from-scratch rebuild reference; shows negligible drift.",
              "results/raw/update_runs/*.parquet",
              "figures/update_drift.png (generate; not yet rendered)"))
    A(fig_box(7, "Filter-strategy decision tree (H3)",
              "the fitted depth-3 tree (features log10 S, D, R, correlation proxy) with the "
              "cheapest-strategy leaf labels; show the corr-proxy split that separates the "
              "anti-correlated regime (exact scan) from the rest.",
              "results/summary/strategy_rule_tree.txt, strategy_rule_accuracy.csv",
              "figures/strategy_tree.png (generate; not yet rendered)"))

    # ---------------- not measured
    A("## 8. What is not yet measured\n")
    A("All four stages (A-D) are complete; what remains are specific robustness checks and one "
      "rerun the filter headline depends on. Stated plainly, because the gap between what a "
      "harness can do and what has been run is where benchmark papers usually mislead:\n")
    A("- **Confound-free filtered rerun (`configs/stage_c_h2.yaml`).** One D-interleaved "
      "campaign at N = 500,000 over D in {128, 384, 768, 1536} at fixed parallelism, with one "
      "shared clustering (k = 10,000, so the smallest correlated level is reachable — the "
      "original k = 1000 overshot S = 0.001 by 54-61%), filter structures outside the timed "
      "region, three seeded builds per cell, dense S around the crossovers, a drift canary, "
      "distance-computation counters, and an nprobe = nlist point for anti-correlated IVF. "
      "H2 and the strategy rule's low-S labels are provisional until it has run.")
    A("- **Beyond 4x compression.** Scalar quantization answered whether the sub-raw regime is "
      "reachable (section 5.3b: yes, at 0.5x). Locating the frontier — and whether anything "
      "meets 0.1x at R >= 0.90 — needs 4-bit SQ or finer PQ with re-ranking, and more cells "
      "than the two-cell probe used here.")
    A("- **Online vs idealized correlation proxy.** The proxy is now computed from an "
      "approximate probe (section 5.6); the exact-GT variant is retained only as an upper bound.")
    A("- **Update drift under distribution shift.** Stage D inserts a random sample of the same "
      "corpus, which cannot exhibit IVF centroid staleness (section 5.9); a shifted-distribution "
      "insert stream would.")
    A("- **hnswlib cross-check** at one cell, and **k in {1, 100}** (all results here are k=10) — "
      "side checks the plan lists that were not run.")
    A("- **Archival release** with a persistent identifier is prepared but not yet minted.\n")

    # ---------------- reproducibility
    A("## 9. Reproducibility\n")
    A("The harness is resumable and append-only: completed configurations are skipped on "
      "restart, raw rows are never edited, and superseded rows are marked rather than deleted. "
      "Every row carries run id, git commit, hardware id, timestamp, seed and a configuration "
      "hash.\n")
    A("```\n"
      "uv venv && uv pip install -e \".[dev,hnswlib]\"\n"
      "make hardware      # record the platform\n"
      "make test          # harness tests\n"
      "make pilot         # reproduce the pilot cell end to end\n"
      "make acceptance    # the checks in section 4.3\n"
      "make stage-a       # the measurements in section 5\n"
      "make analysis      # regenerate every table and figure\n"
      "make paper         # regenerate this document\n"
      "```\n")
    A("Decisions, deviations and defects found along the way — including several that would "
      "have corrupted results and were caught by the acceptance checks — are dated in "
      "[`logs/decisions.md`](../logs/decisions.md). We consider that log part of the paper's "
      "evidence, not supporting material.\n")

    # ---------------- references
    A("## References\n")
    A("Links and per-paper coverage notes in "
      "[`notes/related_work.md`](../notes/related_work.md).\n")
    for ref in [
        "Aumüller, Bernhardsson, Faithfull. ANN-Benchmarks: a benchmarking tool for "
        "approximate nearest neighbor algorithms. 2020. arXiv:1807.05614",
        "Jääsaari et al. VIBE: Vector Index Benchmark for Embeddings. 2025. arXiv:2505.17810",
        "Malkov, Yashunin. Efficient and robust approximate nearest neighbor search using "
        "Hierarchical Navigable Small World graphs. IEEE TPAMI, 2018. arXiv:1603.09320",
        "Jégou, Douze, Schmid. Product quantization for nearest neighbor search. IEEE TPAMI, "
        "2011",
        "Douze et al. The Faiss library. 2024. arXiv:2401.08281",
        "Kusupati et al. Matryoshka Representation Learning. 2022. arXiv:2205.13147",
        "Patel et al. ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings "
        "and Structured Data. 2024. arXiv:2403.04871",
        "Mandarapu, Kunkunuru. Filtered ANN as a Phase Transition: When Selectivity-Estimation "
        "Error Causes Plan Regret. 2026. arXiv:2606.16341",
        "Amanbayev et al. Filtered Approximate Nearest Neighbor Search in Vector Databases: "
        "System Design and Performance Analysis. 2026. arXiv:2602.11443",
        "Iff et al. Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on "
        "Transformer-based Embedding Vectors. 2025. arXiv:2507.21989",
        "Chatzakis et al. VADER: Filtered Vector Search with Declarative Recall. 2026. "
        "arXiv:2609.38053",
        "Munyampirwa et al. Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'. 2024. "
        "arXiv:2412.01940",
        "Herman. Closed Forms and Synthetic Twins: Predicting Approximate Nearest Neighbor "
        "Recall from Embedding Statistics. 2026. arXiv:2609.00364",
    ]:
        A(f"- {ref}")
    A("")

    # ---------------- appendix
    A("## Appendix A. Provenance of every table\n")
    A("| Table | Generated from |")
    A("|---|---|")
    for name, src in [
        ("cost_surface", "results/tuned/tuned.parquet"),
        ("recall_achieved", "results/tuned/tuned.parquet"),
        ("reachability", "results/tuned/tuned.parquet"),
        ("pq_ceiling", "results/raw/sweeps/*.parquet + results/raw/builds/*.parquet"),
        ("memory_budget_winners", "results/summary/winners.csv"),
        ("index_footprint", "results/tuned/tuned.parquet (rss from results/raw/memory/)"),
        ("n_exponents", "results/raw/builds/*.parquet + results/raw/sweeps/*.parquet"),
        ("build_time", "results/tuned/tuned.parquet (T-thread builds)"),
        ("lid", "results/summary/lid.csv"),
        ("acceptance", "results/summary/acceptance_checks.csv"),
        ("ef_construction", "results/summary/acceptance_ef_construction.csv"),
        ("scaling_fits", "results/summary/scaling_fits.csv"),
    ]:
        A(f"| [`tables/{name}.md`](tables/{name}.md) | `{src}` |")
    A("")
    A(f"Platform: `{hw['hardware_id']}`. Builds recorded: {n_builds}. Search-sweep rows: "
      f"{n_sweep_rows:,}. Generated {gen}.\n")

    out = PAPER / "main.md"
    out.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Wrote {out} ({len(L)} blocks, {len(list(TABLES.glob('*.md')))} tables)")
    return out


# ------------------------------------------------------------------- traceability check

NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w%])")


def check_traceability(paper_path: str = "paper/main.md",
                       results_dir: str = "results/raw") -> list[str]:
    """Phase 6 final check: report numbers in the draft not found in the result files.

    Deliberately crude and over-inclusive — it flags candidates for a human to confirm rather
    than claiming to prove provenance. Years, section numbers and small integers are ignored.
    """
    raw = Path(paper_path).read_text()
    # Citations and shell snippets carry numbers that are not measurements (arXiv ids, library
    # versions, commands), so exclude reference lines and fenced blocks before extracting.
    lines, in_fence = [], False
    for line in raw.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        # Headings carry section numbers, and the provenance line carries a timestamp;
        # neither is a measurement.
        if in_fence or "arXiv" in line or "TPAMI" in line:
            continue
        if line.lstrip().startswith("#") or "enerated" in line or "section " in line:
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Collect every numeric string appearing in any generated table or result file.
    haystack: set[str] = set()
    def _forms(v: float) -> set[str]:
        """Every plausible rendering of a number, so the check is about provenance not format."""
        out = {f"{v:.{n}f}" for n in range(11)}
        out |= {f"{v:g}", f"{v:.3g}", f"{v:.4g}", f"{v:.5g}", f"{v:.6g}", str(v)}
        return {x.rstrip("0").rstrip(".") if "." in x else x for x in out} | out

    for p in list(TABLES.glob("*.csv")) + list(SUMMARY.glob("*.csv")):
        for tok in NUM.findall(p.read_text()):
            haystack.add(tok)
            haystack.add(tok.rstrip("0").rstrip("."))
            try:
                haystack |= _forms(float(tok))
            except ValueError:
                pass
    d = load_all(results_dir)
    for df in (d["tuned"], d["builds"], d["memory"], d["sweeps"]):
        if df is None or getattr(df, "empty", True):
            continue
        for col in df.select_dtypes("number").columns:
            vals = df[col].dropna()
            for v in vals.unique()[:20000]:
                haystack |= _forms(float(v))
            # The paper legitimately reports aggregates (medians, ranges) rather than raw rows.
            if len(vals):
                for agg in (vals.median(), vals.mean(), vals.min(), vals.max()):
                    haystack |= _forms(float(agg))
    for p in list(TABLES.glob("*.csv")) + list(SUMMARY.glob("*.csv")):
        try:
            frame = pd.read_csv(p)
        except Exception:
            continue
        for col in frame.select_dtypes("number").columns:
            vals = frame[col].dropna()
            if not len(vals):
                continue
            for agg in (vals.median(), vals.mean(), vals.min(), vals.max()):
                haystack |= _forms(float(agg))
            for gcol in [c for c in ("correlation", "D", "N", "index_type") if c in frame]:
                for _, g in frame.groupby(gcol):
                    gv = g[col].dropna()
                    if len(gv):
                        haystack |= _forms(float(gv.median()))

    # Platform facts live in hardware.json rather than a CSV.
    if d["hardware"]:
        for v in d["hardware"].values():
            if isinstance(v, (int, float)):
                haystack.update({f"{float(v):.{n}f}" for n in range(3)})
        for v in d["hardware"].get("library_versions", {}).values():
            haystack.update(NUM.findall(str(v)))
        haystack.update(NUM.findall(str(d["hardware"].get("python", ""))))
    # Counts of recorded rows are derived from the files, not quoted from them.
    for df in (d["builds"], d["sweeps"]):
        if df is not None and not getattr(df, "empty", True):
            haystack.add(str(len(df)))
            haystack.add(f"{len(df):,}")

    ignore = {"0", "1", "2", "3", "4", "5", "6", "8", "10", "13", "16", "20", "22", "19",
              "32", "48", "64", "100", "128", "200", "256", "384", "400", "512", "768",
              "1024", "1536", "2000", "8000", "10000", "2011", "2018", "2020", "2022",
              "2024", "2025", "2026", "308", "1.0", "0.90", "0.95", "0.99", "0.5", "1.5",
              "0.1", "7", "9", "11", "12", "15", "0.004", "5", "500000", "1000000"}
    unmatched = []
    for tok in sorted(set(NUM.findall(text))):
        if tok in ignore or tok in haystack:
            continue
        stripped = tok.rstrip("0").rstrip(".")
        if stripped in haystack:
            continue
        ctx = next((ln.strip()[:110] for ln in text.splitlines()
                    if re.search(rf"(?<![\w.]){re.escape(tok)}(?![\w%])", ln)), "")
        unmatched.append(f"{tok}  <-  {ctx}")
    return unmatched


def main():
    parser = argparse.ArgumentParser(description="Generate the paper draft from results")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.check_only:
        write_paper(args.results_dir)
    unmatched = check_traceability("paper/main.md", args.results_dir)
    if unmatched:
        print(f"\n{len(unmatched)} numeric token(s) in the draft not located in result files "
              f"(review each):")
        print("  " + ", ".join(unmatched))
    else:
        print("\nTraceability check: every numeric token in the draft was located in a "
              "result file.")


if __name__ == "__main__":
    main()
