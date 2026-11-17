"""Stage E analysis: memory-latency frontier and byte-budget winners at fixed recall.

Reads ``results/raw/tradeoff_runs.parquet`` (written by ``annstudy.tradeoff``) and writes:

- ``results/summary/tradeoff_frontier.csv`` — per R, the arms on the Pareto frontier of
  (bytes per vector, p95 latency) among those that meet R on the held-out queries;
- ``results/summary/tradeoff_winners.csv`` — per (R, budget, accounting), the cheapest arm whose
  bytes per vector fit the budget ("index" counts only the index; "total" adds the re-rank
  store, i.e. the in-RAM view);
- ``figures/tradeoff_frontier_R{R}.png``.

Only rows with ``meets_R_on_test`` count as reaching R, so a winner is never a configuration
that missed the target on queries it was not selected on.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BUDGETS = [512, 1024, 2048, 4096, 8192]  # bytes per vector


def qualifying(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["reachable"].fillna(False).astype(bool)]
    if "meets_R_on_test" in d.columns:
        d = d[d["meets_R_on_test"].fillna(False).astype(bool)]
    return d


def pareto(d: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows not dominated in (x, y), both minimised; sorted by x."""
    d = d.sort_values([x, y])
    keep, best = [], np.inf
    for i, r in d.iterrows():
        if r[y] < best:
            keep.append(i)
            best = r[y]
    return d.loc[keep]


def analyse(runs_path="results/raw/tradeoff_runs.parquet", output_dir="results/summary",
            figures_dir="figures", hardware_id=None) -> pd.DataFrame:
    p = Path(runs_path)
    if not p.exists():
        logger.info(f"tradeoff: {runs_path} not found; skipping")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if hardware_id is not None and "hardware_id" in df.columns:
        df = df[df["hardware_id"] == hardware_id]
    q = qualifying(df)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["R", "arm_id", "family", "D_index", "store", "bytes_per_vec_index",
            "bytes_per_vec_total", "lat_p95_us", "lat_p95_ci_lo", "lat_p95_ci_hi",
            "recall_test", "ef_search", "k_prime"]
    fr = pd.concat([pareto(g, "bytes_per_vec_total", "lat_p95_us").assign(R=R)
                    for R, g in q.groupby("R")], ignore_index=True) if not q.empty \
        else pd.DataFrame(columns=cols)
    fr[[c for c in cols if c in fr.columns]].to_csv(out / "tradeoff_frontier.csv", index=False)

    rows = []
    for R in sorted(df["R"].unique()):
        g = q[q["R"] == R]
        for acct, col in (("total", "bytes_per_vec_total"), ("index", "bytes_per_vec_index")):
            for B in BUDGETS:
                fit = g[g[col] <= B]
                if fit.empty:
                    rows.append({"R": R, "accounting": acct, "budget_bytes": B, "winner": "none"})
                    continue
                w = fit.loc[fit["lat_p95_us"].idxmin()]
                rows.append({"R": R, "accounting": acct, "budget_bytes": B,
                             "winner": w["arm_id"], "lat_p95_us": w["lat_p95_us"],
                             "bytes_per_vec": w[col], "recall_test": w["recall_test"]})
    win = pd.DataFrame(rows)
    win.to_csv(out / "tradeoff_winners.csv", index=False)
    _plot(df, q, figures_dir)
    logger.info(f"tradeoff: {len(fr)} frontier rows, {len(win)} winner rows")
    return win


def _plot(df, q, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    marker = {"none": "o", "fp32": "s", "sq8": "^"}
    for R, g in q.groupby("R"):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for fam, gf in g.groupby("family"):
            for store, gs in gf.groupby("store"):
                ax.scatter(gs["bytes_per_vec_total"], gs["lat_p95_us"],
                           marker=marker.get(store, "o"), label=f"{fam} (+{store} re-rank)"
                           if store != "none" else fam, s=40)
                for _, r in gs.iterrows():
                    ax.annotate(f"@{int(r['D_index'])}", (r["bytes_per_vec_total"],
                                                         r["lat_p95_us"]),
                                fontsize=7, xytext=(3, 3), textcoords="offset points")
        f = pareto(g, "bytes_per_vec_total", "lat_p95_us")
        ax.step(f["bytes_per_vec_total"], f["lat_p95_us"], where="post", color="0.5", lw=1)
        miss = df[(df["R"] == R) & ~df.index.isin(g.index)]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("bytes per vector (index + re-rank store)")
        ax.set_ylabel("p95 latency (us), single thread")
        ax.set_title(f"recall@10 >= {R} (vs exact top-10 at D=1536)\n"
                     f"{len(miss)} arm(s) cannot reach it", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(Path(figures_dir) / f"tradeoff_frontier_R{R}.png", dpi=150)
        plt.close(fig)
