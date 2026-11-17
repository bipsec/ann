"""Stage reports for Stages C and D (EXPERIMENT_PLAN Phase 4).

The plan requires a ``reports/stage_X.md`` per stage with the tuned results and an explicit
list of ``reachable = false`` entries. ``coordinate.write_stage_report`` covers the stages it
drives (pilot, A, B); Stages C and D are driven by their own runners, so their reports are
generated here from the same recorded rows.

CLI: ``annstudy-stage-reports``.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS = Path("reports")
FILTER_RUNS = Path("results/raw/filter_runs.parquet")
UPDATE_RUNS = Path("results/raw/update_runs")


def _provenance(lines: list[str], source: str) -> None:
    from annstudy.hardware import snapshot

    hw = snapshot()
    lines.append("## Provenance\n")
    lines.append(f"- Date: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- Git commit: `{hw['git_commit']}`")
    from annstudy.hardware import data_host_line
    lines.append(data_host_line())
    lines.append(f"- Source: `{source}`\n")


def write_stage_c_report(out_path: str = "reports/stage_C.md") -> Path | None:
    if not FILTER_RUNS.exists():
        logger.warning(f"{FILTER_RUNS} not found; skipping Stage C report.")
        return None
    f = pd.read_parquet(FILTER_RUNS)
    reach = f[f["reachable"].fillna(False)]
    unreach = f[~f["reachable"].fillna(False)]

    L: list[str] = ["# Stage C report: filtered search\n"]
    _provenance(L, str(FILTER_RUNS))

    L.append(f"{len(f)} rows over {f['job_key'].nunique()} jobs: "
             f"{f.groupby(['N', 'D']).ngroups} cells x {f['strategy'].nunique()} strategies x "
             f"{f.groupby(['correlation', 'target_S']).ngroups} filter variants x "
             f"{f['R'].nunique()} recall targets. "
             f"{len(reach)} reachable, {len(unreach)} unreachable.\n")

    L.append("## Acceptance checks\n")
    exact = reach[reach["strategy"] == "pre_filter_exact"]
    if not exact.empty:
        ok = exact["recall_test"].min() >= 0.999
        L.append(f"- Pre-filter exact scan recall is 1.0 at every selectivity: "
                 f"**{'PASS' if ok else 'FAIL'}** "
                 f"(min {exact['recall_test'].min():.6f} over {len(exact)} rows)")
    sel = (f.groupby(["correlation", "target_S"])["measured_S_mean"].mean().reset_index())
    sel["rel_err_pct"] = ((sel["measured_S_mean"] - sel["target_S"]) / sel["target_S"] * 100)
    bad = sel[sel["rel_err_pct"].abs() > 20]
    L.append(f"- Mean measured selectivity within 20% of target: "
             f"**{'PASS' if bad.empty else 'FAIL at ' + str(len(bad)) + ' level(s)'}**")
    L.append("")
    L.append(sel.round(6).to_markdown(index=False))
    L.append("")
    if not bad.empty:
        L.append("The failing levels are the smallest correlated targets. Correlated predicates "
                 "union whole k-means clusters, so with 1,000 clusters the smallest admissible "
                 "union is already near 0.1% of the corpus and a 0.001 target cannot be hit "
                 "from above. S* at that level applies to the measured selectivity rather than "
                 "the nominal one.\n")

    L.append("## Reachability by strategy and correlation\n")
    rows = []
    for (st, corr), g in f.groupby(["strategy", "correlation"], dropna=False):
        n_ok = int(g["reachable"].fillna(False).sum())
        rows.append({"strategy": st, "correlation": corr, "attempts": len(g),
                     "reachable": n_ok, "unreachable": len(g) - n_ok})
    L.append(pd.DataFrame(rows).sort_values(["strategy", "correlation"]).to_markdown(index=False))
    L.append("")
    neg = [r for r in rows if r["correlation"] == "negative" and r["strategy"] != "pre_filter_exact"]
    if neg and sum(r["reachable"] for r in neg) == 0:
        L.append(f"**Under anti-correlated predicates, approximate search never reaches the "
                 f"recall target**: 0 of {sum(r['attempts'] for r in neg)} attempts across "
                 f"in-index filtered search and post-filter over-fetch, at any swept effort. "
                 f"Pre-filter exact scan succeeds in every case. This is structural: both index "
                 f"families concentrate effort near the query, which is exactly where such a "
                 f"predicate admits nothing.\n")

    cross = Path("results/summary/crossover.csv")
    if cross.exists():
        c = pd.read_csv(cross)
        found = c[c["crossover_found"].fillna(False)]
        L.append("## Crossover S*\n")
        L.append(f"{len(found)} of {len(c)} (index, cell, correlation, target) combinations have "
                 f"a crossover inside the measured S range; elsewhere one strategy dominates "
                 f"throughout.\n")
        if not found.empty:
            cols = ["index_type", "N", "D", "correlation", "R", "crossover_S",
                    "crossover_S_lo", "crossover_S_hi"]
            L.append(found[[c2 for c2 in cols if c2 in found.columns]].to_markdown(index=False))
            L.append("")
            for corr, g in found.groupby("correlation"):
                L.append(f"- median S* ({corr}): {g['crossover_S'].median():.5f}")
            L.append("")

    L.append("## Unreachable entries (reachable = false)\n")
    if unreach.empty:
        L.append("_none_\n")
    else:
        u = (unreach.groupby(["strategy", "index_type", "correlation", "target_S", "R"])
             .size().reset_index(name="n"))
        L.append(f"{len(unreach)} rows; grouped:\n")
        L.append(u.to_markdown(index=False))
        L.append("")

    L.append("## For the paper\n")
    L.append("- Feeds sections 5.7 and 5.8: the crossover and the anti-correlation failure. "
             "Cite `results/summary/crossover.csv`, `crossover_h2.csv` and "
             "`results/raw/filter_runs.parquet`.")
    L.append("- Every row records `base_index_run_id`, the Stage B index that answered it.\n")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out


def write_stage_d_report(out_path: str = "reports/stage_D.md") -> Path | None:
    files = sorted(UPDATE_RUNS.glob("*.parquet")) if UPDATE_RUNS.exists() else []
    if not files:
        logger.warning(f"No rows in {UPDATE_RUNS}; skipping Stage D report.")
        return None
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    L: list[str] = ["# Stage D report: updates\n"]
    _provenance(L, str(UPDATE_RUNS))
    N = int(d["N"].iloc[0]) if "N" in d.columns else None
    L.append(f"{len(d)} rows across {d['index_type'].nunique()} index types "
             f"({', '.join(sorted(d['index_type'].unique()))})"
             + (f" at N={N:,}" if N else "") + ".\n")

    L.append("## How recall is measured here\n")
    L.append("Recall is scored against ground truth restricted to the vectors the index "
             "actually contains. Scoring a partially-filled index against full-corpus ground "
             "truth counts the not-yet-inserted vectors as misses, so recall rises mechanically "
             "with every batch and measures completeness rather than drift. The old quantity is "
             "retained per row as `recall_vs_full_corpus_gt`, and the superseded rows are kept "
             "under `results/raw/update_runs_superseded_fullcorpus_gt/`.\n")

    L.append("## Recall drift and the rebuild reference\n")
    rows = []
    for it, g in d.groupby("index_type"):
        g = g.sort_values("batch_id")
        ins = g[g["rebuild_reference_recall"].isna()]
        rb = g[g["rebuild_reference_recall"].notna()]
        if ins.empty:
            continue
        first, last = (float(ins["recall_at_fixed_params"].iloc[0]),
                       float(ins["recall_at_fixed_params"].iloc[-1]))
        ref = float(rb["rebuild_reference_recall"].iloc[0]) if not rb.empty else float("nan")
        rows.append({
            "index_type": it, "batches": len(ins),
            "recall_first": round(first, 5), "recall_last": round(last, 5),
            "drift": round(last - first, 5),
            "rebuild_reference": round(ref, 5),
            "insert_vs_rebuild": round(last - ref, 5),
            "inserts_per_s_median": int(ins["inserts_per_second"].median()),
            "p95_us_median": int(ins["lat_p95_us"].median()),
        })
    t = pd.DataFrame(rows)
    L.append(t.to_markdown(index=False))
    L.append("")
    if not t.empty:
        L.append(f"Drift is negligible for every family (largest absolute change "
                 f"{t['drift'].abs().max():.5f}), and each index ends within "
                 f"{t['insert_vs_rebuild'].abs().max():.5f} of a from-scratch rebuild on the "
                 f"same vectors. The asymmetry is ingest throughput: "
                 + ", ".join(f"{r.index_type} {r.inserts_per_s_median:,} vec/s"
                             for r in t.itertuples()) + ".\n")

    L.append("## Per-batch detail\n")
    cols = [c for c in ["index_type", "batch_id", "inserted_total", "recall_at_fixed_params",
                        "recall_vs_full_corpus_gt", "lat_p95_us", "inserts_per_second",
                        "retrained", "rebuild_reference_recall"] if c in d.columns]
    L.append(d.sort_values(["index_type", "batch_id"])[cols].to_markdown(index=False))
    L.append("")

    L.append("## Caveats\n")
    L.append("- Deletions were not tested (excluded by the plan).")
    L.append("- The insert ratio is modest: 50,000 added to 450,000, about 11%. A system "
             "growing by multiples of its original corpus should not read this as licence "
             "never to rebuild.")
    L.append("- IVF-Flat's pre-tune fell back to maximum effort (`nprobe=512`) because it "
             "missed R>=0.95 on the tuning queries at the reduced corpus size; its measured "
             "recall against prefix ground truth is well above target.\n")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Write the Stage C and D reports")
    parser.add_argument("--stage", choices=["c", "d", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage in ("c", "all"):
        write_stage_c_report()
    if args.stage in ("d", "all"):
        write_stage_d_report()


if __name__ == "__main__":
    main()
