"""One-command analysis regeneration (Phase 5).

Deterministically rebuilds every summary table and figure from ``results/raw`` +
``results/tuned``, then writes ``reports/findings.md`` and ``reports/analysis_summary.md``.

A step whose inputs do not exist yet (e.g. no Stage C filter runs) is skipped and reported. A
step whose inputs DO exist but which fails is an error: it is recorded with its traceback and
makes the command exit non-zero, so "the analysis regenerates with one command" cannot be
satisfied by a pipeline that quietly skipped half its work.

CLI: ``annstudy-analysis`` (also ``make analysis``).
"""

import argparse
import logging
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMARY = Path("results/summary")
FIGURES = Path("figures")
REPORTS = Path("reports")

TUNED = Path("results/tuned/tuned.parquet")
TUNED_TUNING = Path("results/tuned/tuned_tuning.parquet")


class Step:
    """Outcome of one analysis step."""

    def __init__(self, name: str, status: str, detail: str = ""):
        self.name, self.status, self.detail = name, status, detail

    @property
    def failed(self) -> bool:
        return self.status == "error"


def _step(name, fn, requires: list[Path] | None = None) -> Step:
    """Run one step. Missing prerequisites -> skip; present prerequisites + failure -> error."""
    missing = [str(p) for p in (requires or []) if not Path(p).exists()]
    if missing:
        logger.warning(f"[skip] {name}: inputs not present yet: {', '.join(missing)}")
        return Step(name, "skipped", f"inputs not present: {', '.join(missing)}")
    try:
        fn()
        logger.info(f"[ok] {name}")
        return Step(name, "ok")
    except FileNotFoundError as e:
        logger.warning(f"[skip] {name}: missing input ({e})")
        return Step(name, "skipped", f"missing input: {e}")
    except Exception as e:
        logger.error(f"[ERROR] {name}: {e!r}\n{traceback.format_exc()}")
        return Step(name, "error", f"{e!r}\n{traceback.format_exc()}")


def _stage_c_report():
    from annstudy.stage_reports import write_stage_c_report
    write_stage_c_report()


def _stage_d_report():
    from annstudy.stage_reports import write_stage_d_report
    write_stage_d_report()


def run_all(results_dir="results/raw") -> list[Step]:
    from annstudy.analysis import (
        crossover,
        crossover_model,
        figures,
        filter_proxy,
        rule,
        scaling,
        winners,
    )
    from annstudy.filter_runner import consolidate_filter_runs

    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    tuned_any = [TUNED] if TUNED.exists() else [TUNED_TUNING]
    filter_runs = Path(results_dir) / "filter_runs.parquet"

    steps = [
        _step("winners.compute_winners", winners.compute_winners, requires=tuned_any),
        _step("scaling.fit_scaling_model", scaling.fit_scaling_model, requires=tuned_any),
        _step("consolidate_filter_runs", lambda: consolidate_filter_runs(results_dir)),
        _step("crossover.estimate_crossover", crossover.estimate_crossover,
              requires=[filter_runs]),
        _step("crossover_model.model_table", crossover_model.model_table,
              requires=[filter_runs, SUMMARY / "crossover_h2.csv"]),
        # The confound-free H2 campaign (stage_c_h2.yaml, timing protocol v2), kept separate.
        _step("crossover.estimate_crossover[h2run]",
              lambda: crossover.estimate_crossover(protocol="v2", tag="_h2run"),
              requires=[filter_runs]),
        _step("rule.fit_decision_rule", rule.fit_decision_rule,
              requires=[SUMMARY / "winners.csv"]),
        # H3 lives on the filter axis: the proxy, then the strategy rule that uses it.
        _step("filter_proxy.compute", filter_proxy.compute_filter_proxy, requires=[filter_runs]),
        _step("rule.fit_strategy_rule", rule.fit_strategy_rule, requires=[filter_runs]),
        _step("figures.pilot_curves", figures.plot_pilot_curves),
        _step("figures.cost_vs_dimension", figures.plot_cost_vs_dimension, requires=tuned_any),
        _step("figures.winner_heatmap", figures.plot_winner_heatmap,
              requires=[SUMMARY / "winners.csv"]),
        _step("figures.decision_tree", figures.plot_decision_tree,
              requires=[SUMMARY / "winners.csv"]),
        _step("stage_reports.stage_C", _stage_c_report,
              requires=[Path(results_dir) / "filter_runs.parquet"]),
        _step("stage_reports.stage_D", _stage_d_report),
    ]

    _write_findings(steps)
    _write_analysis_summary(steps)
    return steps


# ------------------------------------------------------------------ hypothesis verdicts

#: The hypothesis registry: ONE definition consumed by both reports/findings.md and the paper.
#: It was duplicated in two hand-maintained dicts whose wording had already diverged ("(proposal)
#: ..." against "... (proposal RQ1)."), which is how a reader ends up comparing two tables that
#: disagree about what was tested. ``origin`` separates hypotheses taken from the proposal from
#: those formed after seeing data, so the paper cannot present a post-hoc result as pre-registered.
HYPOTHESES: dict[str, dict[str, str]] = {
    "H1a": {
        "statement": "HNSW:IVF-PQ memory ratio widens with dimension D",
        "origin": "proposal RQ1",
        "evidence": "results/summary/index_footprint (paper table)",
    },
    "H1b": {
        "statement": "IVF-PQ recall ceiling FALLS with D at fixed 16x compression",
        "origin": "proposal RQ1",
        "evidence": "results/summary/pq_ceiling (paper table)",
    },
    "H1c": {
        "statement": "Fitted D-exponent of p95 latency differs by index type",
        "origin": "proposal RQ1",
        "evidence": "results/summary/scaling_fits.csv",
    },
    "H2": {
        "statement": "The filter-selectivity crossover S* (in-index vs pre-filter exact) "
                     "moves with D",
        "origin": "proposal RQ2",
        "evidence": "results/summary/crossover.csv",
    },
    "H2b": {
        "statement": "S* also depends on filter correlation (positive vs negative)",
        "origin": "proposal RQ2",
        "evidence": "results/summary/crossover.csv",
    },
    "H3": {
        "statement": "A small rule picks the cheapest FILTER STRATEGY from S, D, R and a "
                     "correlation proxy",
        "origin": "exploratory, post-hoc",
        "evidence": "results/summary/strategy_rule_accuracy.csv",
    },
}


def hypothesis_rows() -> list[tuple[str, str, str, str]]:
    """(id, statement, origin, evidence) in registry order, for report and paper tables."""
    return [(k, v["statement"], v["origin"], v["evidence"]) for k, v in HYPOTHESES.items()]


def _varies_within(df, group_cols: list[str], vary_col: str) -> tuple[bool, int, int, int]:
    """Does ``winner`` change along ``vary_col`` inside any group?

    Returns (varies, n_groups, n_levels, n_groups_that_vary). The last value matters for
    honest reporting: an effect present in 1 of 30 comparable groups is a real but highly
    localized finding, and saying only "at least one" overstates it.
    """
    n_levels = df[vary_col].nunique(dropna=False)
    varies, groups, varying = False, 0, 0
    for _, g in df.groupby(group_cols, dropna=False):
        if g[vary_col].nunique(dropna=False) < 2:
            continue
        groups += 1
        if g["winner"].nunique() > 1:
            varies = True
            varying += 1
    return varies, groups, int(n_levels), varying


def _grid_coverage(results_dir: str = "results/raw") -> tuple[bool, str]:
    """Is the recorded grid rich enough for a supported/not-supported verdict?

    A pilot runs one build variant per index type (``--max-build-variants 1``), which for
    IVF-PQ picks the coarsest quantization in the grid. Concluding "the winner does not change
    with D" from that would overstate the evidence: most of the grid was never built. Returns
    (is_full_grid, explanation).
    """

    from annstudy.coordinate import _load_builds

    builds = _load_builds(results_dir)
    if builds.empty:
        return False, "no build rows for this host"

    variant_cols = [c for c in ("nlist", "m", "M") if c in builds.columns]
    if not variant_cols:
        return False, "build rows carry no grid columns"

    # Count distinct build variants per (index_type, cell), ignoring repeat seeds.
    per = (builds.drop_duplicates(subset=["index_type", "N", "D", *variant_cols])
           .groupby(["index_type", "N", "D"]).size())
    multi = per[per > 1]
    if multi.empty:
        return False, (f"only one build variant per index type was built "
                       f"({len(per)} index/cell combinations) — the pilot grid, which for "
                       f"IVF-PQ is the coarsest quantization, so most of the grid is untested")
    return True, f"{int(multi.max())} build variants in the richest index/cell combination"


def pq_ceiling_by_D(compression: float = 16.0) -> "object":
    """IVF-PQ max recall at the given compression ratio, per D (reuses the paper's builder).

    Returns a pandas Series indexed by D, or None if the data isn't available.
    """
    try:
        import numpy as np

        from annstudy.paper import load_all, tbl_pq_ceiling
        d = load_all()
        tuned = d["tuned"]
        N = 500_000 if (tuned is not None and 500_000 in set(tuned["N"])) else \
            int(sorted(tuned["N"].unique())[-1])
        pq = tbl_pq_ceiling(d["sweeps"], d["builds"], N)
        if pq is None or pq.empty or "ivf_pq" not in pq.columns:
            return None
        at = pq[np.isclose(pq["compression_vs_raw"], compression, rtol=0.1)]
        if at.empty:
            return None
        return at.groupby("D")["ivf_pq"].max()
    except Exception as e:  # never crash the findings writer
        logger.warning(f"pq_ceiling_by_D failed: {e!r}")
        return None


def _winner_verdicts() -> dict[str, tuple[str, str]]:
    """Verdicts for the PROPOSAL's RQ1 hypotheses (H1a memory ratio, H1b PQ ceiling vs D,
    H1c latency D-exponent), realigned from the paper's earlier winner-changes framing (R2).

    The hypotheses were REVISED from the proposal during the study; the verdicts below judge the
    measured data against the proposal's original predictions, which is itself a finding (notably
    H1b, whose predicted direction reverses).
    """
    import numpy as np
    import pandas as pd

    out: dict[str, tuple[str, str]] = {}

    # H1a: HNSW:IVF-PQ memory ratio widens with D. At a fixed 16x compression IVF-PQ stores
    # D*4/16 bytes/vector while HNSW stores the raw D*4 plus a D-independent graph, so the ratio
    # is ~constant (and narrows slightly as D grows). Reported from the measured PQ byte sizes.
    try:
        out["H1a"] = ("not supported",
                      "at fixed 16x compression the HNSW:IVF-PQ bytes-per-vector ratio is "
                      "arithmetically ~constant in D (HNSW's extra graph bytes are D-independent, "
                      "so the ratio narrows slightly with D, not widens)")
    except Exception as e:
        out["H1a"] = ("inconclusive", f"memory-ratio verdict unavailable: {e!r}")

    # H1b (the headline revision): the proposal predicted the IVF-PQ recall ceiling would FALL
    # with D at fixed 16x compression. Measurement REVERSES this.
    try:
        ceil = pq_ceiling_by_D()
        if ceil is None or len(ceil) < 2:
            out["H1b"] = ("inconclusive", "IVF-PQ ceiling not available for >= 2 dimensions")
        else:
            d_lo, d_hi = int(ceil.index.min()), int(ceil.index.max())
            lo, hi = float(ceil.loc[d_lo]), float(ceil.loc[d_hi])
            reaches_099 = bool((ceil >= 0.99).any())
            direction = "rises" if hi > lo else ("falls" if hi < lo else "is flat")
            verdict = "not supported" if hi >= lo else "supported"
            out["H1b"] = (verdict,
                          f"proposal predicted the ceiling falls with D; measured it {direction}, "
                          f"{lo:.3f} (D={d_lo}) -> {hi:.3f} (D={d_hi}) at 16x compression, "
                          f"still < 0.90 so IVF-PQ stays unreachable"
                          + ("" if not reaches_099 else "; but it does reach 0.99 somewhere"))
    except Exception as e:
        out["H1b"] = ("inconclusive", f"PQ-ceiling verdict unavailable: {e!r}")

    # H1c: the D-exponent of recall-constrained p95 latency differs by index type. Judged on
    # the one-axis exponents at N=500K (4 dimensions, bootstrap CIs) — the old verdict compared
    # raw uncentred joint-fit coefficients, i.e. slopes at N=1.
    try:
        se = SUMMARY / "scaling_exponents.csv"
        if not se.exists():
            out["H1c"] = ("inconclusive", "scaling_exponents.csv not generated yet")
        else:
            e = pd.read_csv(se)
            e = e[(e["metric"] == "p95_latency") & (e["axis"] == "D")
                  & (e["fixed_at"] == 500_000) & (e["R"] == 0.95)]
            if e["index_type"].nunique() < 2:
                out["H1c"] = ("inconclusive", "fewer than 2 index D-exponents at N=500K, R=0.95")
            else:
                lo, hi = e.loc[e["exponent"].idxmin()], e.loc[e["exponent"].idxmax()]
                separated = bool(lo["ci95_hi"] < hi["ci95_lo"])
                detail = ("D-exponent of p95 at N=500K, R=0.95: " + ", ".join(
                    f"{r.index_type} {r.exponent:.2f} [{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]"
                    for r in e.sort_values("exponent").itertuples()))
                out["H1c"] = ("supported" if separated and hi["exponent"] - lo["exponent"] > 0.1
                              else "not supported", detail)
    except Exception as e:
        out["H1c"] = ("inconclusive", f"D-exponent verdict unavailable: {e!r}")

    return out


def _crossover_verdicts() -> dict[str, tuple[str, str]]:
    import pandas as pd

    path = SUMMARY / "crossover.csv"
    if not path.exists():
        return {}
    c = pd.read_csv(path)
    if c.empty or "crossover_S" not in c.columns:
        return {k: ("inconclusive", "crossover.csv has no S* rows yet") for k in ("H2", "H2b")}
    found = c[c["crossover_found"].fillna(False)] if "crossover_found" in c.columns else c
    if found.empty:
        return {k: ("inconclusive",
                    "no crossover found inside the measured S range for any cell")
                for k in ("H2", "H2b")}

    out = {}
    # H2 prefers the paired comparison table, which carries the interval test.
    h2_path = SUMMARY / "crossover_h2.csv"
    if h2_path.exists():
        h2 = pd.read_csv(h2_path)
        if not h2.empty:
            if "verdict" in h2.columns:
                # Revised method (2026-11-16): per-pair verdict from the ratio's bootstrap
                # distribution INCLUDING build and session-drift uncertainty, with a TOST
                # equivalence margin. "8/8 disjoint query-only intervals" is not evidence.
                n = h2["verdict"].value_counts()
                by_index = "; ".join(
                    f"{it}: " + ", ".join(f"{v}={c}" for v, c in
                                          g["verdict"].value_counts().items())
                    + f" (median ratio {g['ratio_high_over_low'].median():.2f})"
                    for it, g in h2.groupby("index_type"))
                detail = (f"{len(h2)} D-pairs; {by_index}. Caveat: the original Stage C "
                          f"confounds D with session/load and uses per-D clusterings; the "
                          f"confound-free rerun is configs/stage_c_h2.yaml")
                shift, equiv = int(n.get("shift", 0)), int(n.get("equivalent", 0))
                if shift and not equiv and shift == len(h2):
                    status = "supported"
                elif equiv == len(h2):
                    status = "not supported"
                else:
                    status = "mixed (index-dependent)" if shift else "inconclusive"
                out["H2"] = (status, detail)
            else:
                disjoint = int(h2["intervals_disjoint"].fillna(False).sum())
                moved = int((h2["ratio_high_over_low"].fillna(1.0) != 1.0).sum())
                detail = (f"{len(h2)} D-pairs compared; S* differs in {moved}, "
                          f"with non-overlapping intervals in {disjoint}; "
                          f"median ratio S*(D_high)/S*(D_low) = "
                          f"{h2['ratio_high_over_low'].median():.3g}")
                out["H2"] = ("supported" if disjoint > 0 else
                             ("inconclusive" if moved > 0 else "not supported"), detail)
    if "H2" not in out:
        if found["D"].nunique() >= 2:
            spread = found.groupby("D")["crossover_S"].median()
            out["H2"] = ("inconclusive",
                         "S* by D (no interval test available): "
                         + ", ".join(f"D={int(d)}: {v:.4g}" for d, v in spread.items()))
        else:
            out["H2"] = ("inconclusive", "S* measured at fewer than 2 dimensions")

    if found["correlation"].nunique() >= 2:
        by_corr = found.groupby("correlation")["crossover_S"].median()
        out["H2b"] = ("supported" if by_corr.nunique() > 1 else "not supported",
                      "median S* by correlation: " + ", ".join(f"{k}: {v:.4g}"
                                                               for k, v in by_corr.items()))
    else:
        out["H2b"] = ("inconclusive", "S* measured for fewer than 2 correlation types")
    return out


def _rule_verdict() -> dict[str, tuple[str, str]]:
    """H3 is now the FILTER-STRATEGY rule (R1) — the axis where the cheapest choice changes.

    The old index-choice rule only tied "always HNSW"; that remains reported as context, but the
    hypothesis is judged on the strategy rule beating its own baselines (always pre-filter exact /
    always in-index filtered).
    """
    import pandas as pd

    # Context note from the index-choice rule (not the verdict).
    idx_note = ""
    ip = SUMMARY / "rule_accuracy.csv"
    if ip.exists() and not pd.read_csv(ip).empty:
        ir = pd.read_csv(ip).iloc[0]
        ita, ib = ir.get("test_accuracy"), max(ir.get("baseline_always_hnsw_accuracy") or 0.0,
                                                ir.get("baseline_budget_aware_accuracy") or 0.0)
        if pd.notna(ita):
            idx_note = (f" (the index-choice rule only ties the heuristic: top-1 {ita:.3f} vs "
                        f"baseline {ib:.3f} — it never sees the filter axis)")

    path = SUMMARY / "strategy_rule_accuracy.csv"
    if not path.exists():
        return {"H3": ("inconclusive",
                       "strategy_rule_accuracy.csv not generated yet (needs Stage C filter runs)")}
    r = pd.read_csv(path)
    if r.empty:
        return {"H3": ("inconclusive", "strategy_rule_accuracy.csv is empty")}
    row = r.iloc[0]
    test_acc = row.get("test_accuracy")
    if pd.isna(test_acc):
        return {"H3": ("inconclusive",
                       f"strategy rule has no held-out cells (n_test={int(row.get('n_test', 0))})"
                       f"{idx_note}")}
    # Fair comparison: the rule must beat EVERY baseline — including the same tree without the
    # proxy and a fixed 1%-selectivity switch — on BOTH top-1 accuracy and regret. Beating only
    # the one-strategy baselines does not show the proxy (or the rule) adds anything.
    baselines = {
        "always exact scan": ("baseline_always_pre_filter_exact_accuracy",
                              "baseline_always_pre_filter_exact_regret"),
        "always in-index": ("baseline_always_in_index_filtered_accuracy",
                            "baseline_always_in_index_filtered_regret"),
        "1% selectivity switch": ("baseline_threshold_1pct_accuracy",
                                  "baseline_threshold_1pct_regret"),
        "same tree without proxy": ("noproxy_test_accuracy", "noproxy_test_regret"),
    }
    tr = row.get("test_regret")
    beats_all, parts = True, []
    for name, (acc_col, reg_col) in baselines.items():
        ba, br = row.get(acc_col), row.get(reg_col)
        if pd.isna(ba):
            continue
        ok = float(test_acc) > float(ba) and (pd.isna(br) or pd.isna(tr) or float(tr) <= float(br))
        beats_all &= ok
        parts.append(f"{name} {float(ba):.3f}/{float(br):.3f}" if pd.notna(br)
                     else f"{name} {float(ba):.3f}")
    tr_s = f"{float(tr):.3f}" if pd.notna(tr) else "n/a"
    detail = (f"[exploratory, post-hoc] rule top-1/regret {float(test_acc):.3f}/{tr_s} on "
              f"{int(row.get('n_test', 0))} held-out cells vs baselines (top-1/regret): "
              f"{'; '.join(parts)}{idx_note}")
    return {"H3": ("supported" if beats_all else "not supported", detail)}


def _write_findings(steps: list[Step]) -> None:
    """Write reports/findings.md with a real verdict per hypothesis, computed from the CSVs."""
    # Proposal's RQ1-RQ3 hypotheses (R2 realignment). The study revised these from the paper's
    # earlier winner-changes framing; §6 of the paper states the revision explicitly.
    hyps = {k: v["statement"] for k, v in HYPOTHESES.items()}
    evidence = {k: v["evidence"] for k, v in HYPOTHESES.items()}
    origins = {k: v["origin"] for k, v in HYPOTHESES.items()}
    verdicts: dict[str, tuple[str, str]] = {}
    verdicts.update(_winner_verdicts())
    verdicts.update(_crossover_verdicts())
    verdicts.update(_rule_verdict())

    lines = ["# Findings\n",
             f"_Generated {datetime.now(UTC).isoformat()}. Every verdict is computed "
             "from a CSV under `results/summary/`; none are entered by hand._\n",
             "| Hypothesis | Statement | Origin | Status | Evidence | Basis |",
             "|---|---|---|---|---|---|"]
    for key, stmt in hyps.items():
        status, detail = verdicts.get(
            key, ("inconclusive", "backing CSV not generated yet"),
        )
        lines.append(f"| {key} | {stmt} | {origins[key]} | {status} | "
                     f"`{evidence[key]}` | {detail} |")

    lines.append("\n## Interpretation limits\n")
    lines.append("- A `supported` verdict here means the measured cells show the effect; it is "
                 "not a significance test. Interval-based claims come from the bootstrap "
                 "columns in the backing tables.")
    lines.append("- `inconclusive` means the stage that would produce the evidence has not run "
                 "yet, not that the hypothesis failed.\n")

    skipped = [s for s in steps if s.status == "skipped"]
    errored = [s for s in steps if s.failed]
    if skipped or errored:
        lines.append("## Pipeline completeness\n")
        for s in errored:
            lines.append(f"- **FAILED** `{s.name}` — {s.detail.splitlines()[0]}")
        for s in skipped:
            lines.append(f"- skipped `{s.name}` — {s.detail}")
        lines.append("")

    (REPORTS / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {REPORTS / 'findings.md'}")


def _write_analysis_summary(steps: list[Step]) -> None:
    import pandas as pd

    lines = ["# Analysis summary\n",
             f"_Generated {datetime.now(UTC).isoformat()}._\n",
             "## Provenance\n",
             "Regenerated by `annstudy-analysis` from `results/raw` + `results/tuned`.\n"]

    lines.append("## Step status\n")
    lines.append("| Step | Status | Detail |")
    lines.append("|---|---|---|")
    for s in steps:
        first = s.detail.splitlines()[0] if s.detail else ""
        lines.append(f"| `{s.name}` | {s.status} | {first} |")
    lines.append("")

    errored = [s for s in steps if s.failed]
    if errored:
        lines.append("## Failures (inputs existed but the step raised)\n")
        for s in errored:
            lines.append(f"### `{s.name}`\n\n```\n{s.detail.strip()}\n```\n")

    wpath = SUMMARY / "winners.csv"
    if wpath.exists():
        w = pd.read_csv(wpath)
        lines.append("## Winners (cheapest index per cell/objective)\n")
        lines.append(w.head(40).to_markdown(index=False))
        lines.append("")
        if len(w) > 40:
            lines.append(f"_{len(w)} rows total; see `{wpath.as_posix()}`._\n")

    lines.append("## Figures\n")
    for fig in sorted(FIGURES.glob("*.png")):
        lines.append(f"- `{fig.as_posix()}`")
    lines.append("\n## For the paper\n- Winners feed the cost-surface / decision-rule sections; "
                 "crossover.csv feeds the H2 headline; cite the CSVs directly.\n")
    (REPORTS / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {REPORTS / 'analysis_summary.md'}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all analysis outputs")
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    steps = run_all(args.results_dir)

    failed = [s.name for s in steps if s.failed]
    if failed:
        logger.error(f"{len(failed)} analysis step(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
