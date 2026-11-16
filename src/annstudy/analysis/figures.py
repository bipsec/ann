"""Figure generation from summary results."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DIR = Path("figures")


# ---------------------------------------------------------------------------- shared style
# Index types get a FIXED categorical slot (color follows the entity, never its rank), from the
# validated reference palette. "none" (no index met R within the budget) is a neutral gray.
INDEX_COLORS = {
    "hnsw": "#2a78d6",           # slot 1 blue
    "ivf_flat": "#eb6834",       # slot 2 orange
    "flat": "#1baf7a",           # slot 3 aqua
    "ivf_pq_refine": "#eda100",  # slot 4 yellow
    "ivf_pq": "#e87ba4",         # slot 5 magenta
    "hnsw_sq": "#008300",        # slot 6 green
}
INDEX_LABELS = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat", "flat": "Flat", "ivf_pq": "IVF-PQ",
                "ivf_pq_refine": "IVF-PQ+refine", "hnsw_sq": "HNSW-SQ8"}
STRATEGY_COLORS = {"pre_filter_exact": "#6250d6", "in_index_filtered": "#eda100",
                   "post_filter_overfetch": "#e34948"}
STRATEGY_LABELS = {"pre_filter_exact": "exact scan of subset",
                   "in_index_filtered": "in-index filtered",
                   "post_filter_overfetch": "post-filter over-fetch"}
NONE_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
OBJECTIVE_TITLES = {
    "lat_p95_us": "cheapest p95 query latency",
    "lat_p95_us_tuning": "cheapest p95 query latency (tuning queries)",
    "build_seconds_mean": "cheapest build time",
    "build_seconds": "cheapest build time",
    "index_bytes": "smallest index",
}


def _study_hardware_id() -> str | None:
    """Hardware id of the measurement host (results/hardware.json), not of whatever machine is
    rendering the figure — plotting on a laptop must not silently switch to its own rows."""
    import json
    p = Path("results/hardware.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("hardware_id")


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------- cost vs D

def plot_cost_vs_dimension(
    tuned_path: str = "results/tuned/tuned.parquet",
    recall_targets: list[float] | None = None,
    output_dir: str | Path = FIGURE_DIR,
    N: int = 500_000,
) -> None:
    """p95 latency at fixed recall vs D, one line per index, at ONE corpus size.

    Fixing N matters: without it every dimension stacked the N=100K/500K/1M points onto one line
    and drew vertical jumps. Hollow markers flag winners that met R on tuning queries but fell
    just short on the held-out test queries. Indexes measured at fewer than two D (e.g. the
    two-cell HNSW-SQ8 probe) are drawn as points, not lines.
    """
    if recall_targets is None:
        recall_targets = [0.95, 0.99]
    tuned = pd.read_parquet(tuned_path)
    tuned = tuned[(tuned["reachable"] == True) & (tuned["N"] == N)]  # noqa: E712
    lat_col = "lat_p95_us" if "lat_p95_us" in tuned.columns else "lat_p95_us_tuning"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["flat", "ivf_flat", "ivf_pq_refine", "hnsw", "hnsw_sq"]

    for R in recall_targets:
        sub = tuned[tuned["R"] == R]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        Ds = sorted(sub["D"].unique())
        for it in [i for i in order if i in set(sub["index_type"])]:
            g = sub[sub["index_type"] == it].sort_values("D")
            c = INDEX_COLORS.get(it, MUTED)
            ax.plot(g["D"], g[lat_col], color=c, lw=2, zorder=2,
                    ls=(0, (4, 2)) if it == "hnsw_sq" else "-")
            meets = (g["meets_R_on_test"].fillna(True).astype(bool)
                     if "meets_R_on_test" in g else pd.Series(True, index=g.index))
            ax.scatter(g["D"][meets], g[lat_col][meets], s=40, color=c, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
            ax.scatter(g["D"][~meets], g[lat_col][~meets], s=40, facecolors=SURFACE,
                       edgecolors=c, linewidths=1.8, zorder=3)
            last = g.iloc[-1]
            ax.annotate(INDEX_LABELS.get(it, it), (last["D"], last[lat_col]),
                        xytext=(8, -9 if it == "hnsw_sq" else 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ds)
        ax.set_xticklabels([str(int(d)) for d in Ds])
        ax.minorticks_off()
        ax.set_xlim(Ds[0] / 1.25, Ds[-1] * 2.1)
        ax.set_xlabel("Embedding dimension D", fontsize=9, color=INK2)
        ax.set_ylabel("p95 latency (us, log scale)", fontsize=9, color=INK2)
        ax.set_title(f"Cost of reaching recall@10 >= {R} vs dimension (N={N:,}, test queries;"
                     f" IVF-PQ never reached R)", fontsize=9.5, color=INK, loc="left")
        ax.text(0.0, -0.17, "Hollow marker: chosen on tuning queries, just below R on the "
                "held-out test queries. Dashed: HNSW-SQ8, measured at two dimensions only.",
                transform=ax.transAxes, fontsize=7, color=MUTED)
        fig.tight_layout()
        path = output_dir / f"cost_vs_D_R{R}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------- pilot curves

def _pareto(g: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Upper-right frontier: for decreasing throughput, the best recall seen so far."""
    g = g.sort_values(x, ascending=False)
    keep, best = [], -1.0
    for _, r in g.iterrows():
        if r[y] > best:
            keep.append(r)
            best = r[y]
    return pd.DataFrame(keep)


def plot_pilot_curves(
    sweeps_dir: str = "results/raw/sweeps",
    query_split: str = "tuning",
    output_path: str | Path = FIGURE_DIR / "pilot_curves.png",
    N: int = 100_000,
) -> Path:
    """Recall@10 vs single-thread throughput at N=100K, one panel per D, one frontier per index.

    Rows are restricted to the measurement host (hardware.json) — mixing an earlier machine's rows
    in left D=768/1536 empty; throughput is 1e6/mean latency (single-thread, batch 1), which every
    sweep row has; and each index is drawn as its Pareto frontier over all build variants, so the
    line does not zig-zag between variants.
    """
    sweeps_dir = Path(sweeps_dir)
    files = list(sweeps_dir.glob("sweep_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sweep files in {sweeps_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["query_split"] == query_split) & (df["N"] == N)]
    hid = _study_hardware_id()
    if hid and "hardware_id" in df.columns:
        df = df[df["hardware_id"] == hid]
    df = df[df["lat_mean_us"] > 0].copy()
    df["throughput"] = 1e6 / df["lat_mean_us"]

    from matplotlib.lines import Line2D

    order = ["flat", "ivf_pq", "ivf_flat", "ivf_pq_refine", "hnsw", "hnsw_sq"]
    dims = sorted(df["D"].unique())
    ncols = 2
    nrows = (len(dims) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.9 * nrows), squeeze=False,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    present = []
    for k, D in enumerate(dims):
        ax = axes[k // ncols][k % ncols]
        _style_axes(ax)
        for R in (0.90, 0.95, 0.99):
            ax.axhline(R, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.text(1.0, R, f" R={R:g}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6.5, color=MUTED)
        sub = df[df["D"] == D]
        for it in [i for i in order if i in set(sub["index_type"])]:
            c = INDEX_COLORS.get(it, MUTED)
            g = sub[sub["index_type"] == it]
            if it == "flat":
                ax.scatter(g["throughput"], g["recall_mean"], s=55, marker="D", color=c,
                           edgecolors=SURFACE, linewidths=1.5, zorder=4)
            else:
                f = _pareto(g, "throughput", "recall_mean")
                ax.plot(f["throughput"], f["recall_mean"], color=c, lw=2, zorder=2)
                ax.scatter(f["throughput"], f["recall_mean"], s=22, color=c,
                           edgecolors=SURFACE, linewidths=1, zorder=3)
            if it not in present:
                present.append(it)
        ax.set_xscale("log")
        ax.set_ylim(0.3, 1.01)
        ax.set_title(f"D = {int(D)}", fontsize=9.5, color=INK, loc="left")
        if k // ncols == nrows - 1:
            ax.set_xlabel("Queries per second, one thread (log scale)", fontsize=8.5,
                          color=INK2)
        if k % ncols == 0:
            ax.set_ylabel("recall@10", fontsize=8.5, color=INK2)
    for j in range(len(dims), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [Line2D([0], [0], color=INDEX_COLORS[i], lw=2,
                      marker="D" if i == "flat" else "o",
                      ls="none" if i == "flat" else "-", markersize=6,
                      label=INDEX_LABELS[i] + (" (exact)" if i == "flat" else ""))
               for i in order if i in present]
    fig_h = fig.get_figheight()
    fig.suptitle(f"Recall vs throughput at N={N:,} ({query_split} queries; best recall per "
                 f"throughput across each index's build variants)", fontsize=10, color=INK,
                 y=1 - 0.12 / fig_h, va="top")
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1 - 0.42 / fig_h))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig_h))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info(f"Saved {output_path}")
    return output_path


# ---------------------------------------------------------------------------- decision trees

def _color_tree_nodes(arts, label_to_color: dict) -> None:
    """Tint each sklearn plot_tree node by the class named IN ITS OWN TEXT.

    plot_tree's annotation order is its drawing order, not node-id order, so pairing annotations
    with tree_.value by position colours the wrong boxes. Reading 'class = X' and 'value = [...]'
    from each node's text is exact. Tints are opaque blends toward the surface (alpha would let
    the edge arrows show through), deeper for purer nodes, capped so dark ink stays legible.
    """
    import re

    from matplotlib.colors import to_rgb

    surf = to_rgb(SURFACE)
    for art in arts:
        txt = art.get_text()
        m = re.search(r"class = (.+)$", txt, flags=re.M)
        box = art.get_bbox_patch()
        if not m or box is None:
            continue
        color = label_to_color.get(m.group(1).strip(), MUTED)
        vm = re.search(r"value = \[([^\]]+)\]", txt)
        share = 0.5
        if vm:
            vals = [float(x) for x in vm.group(1).replace(",", " ").split()]
            share = max(vals) / sum(vals) if sum(vals) else 0.5
        w = 0.22 + 0.5 * share  # 0.22 (mixed) .. 0.72 (pure)
        rgb = tuple(w * c + (1 - w) * s for c, s in zip(to_rgb(color), surf, strict=True))
        box.set_facecolor(rgb)
        box.set_alpha(1.0)
        box.set_edgecolor(color)
        box.set_linewidth(1.2)
        art.set_color(INK)
        # Each edge arrow is drawn with the CHILD annotation and starts at the parent's centre,
        # so stack nodes by height: a parent (higher y) sits above the arrows leaving it.
        art.set_zorder(5 + 10 * float(art.get_position()[1]))


def plot_decision_tree(
    winners_path: str = "results/summary/winners.csv",
    output_dir: str | Path = FIGURE_DIR,
    max_depth: int = 3,
) -> Path | None:
    """Plot the filter-strategy rule (exploratory H3) to decision_tree.png, and the index-choice
    rule (proposal H3, not supported) to decision_tree_index.png. Both are refit in a scratch
    directory so rendering never rewrites result files."""
    import tempfile

    from matplotlib.patches import Patch
    from sklearn.tree import plot_tree

    from annstudy.analysis.rule import fit_decision_rule, fit_strategy_rule

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_src = None
    pp = Path("results/summary/filter_proxy.csv")
    if pp.exists():
        srcs = pd.read_csv(pp).get("proxy_source")
        proxy_src = (srcs.dropna().iloc[0] if srcs is not None and srcs.notna().any()
                     else "idealized_gt")

    with tempfile.TemporaryDirectory() as tmp:
        res = fit_strategy_rule(output_dir=tmp, max_depth=max_depth)
        idx_res = fit_decision_rule(winners_path=winners_path, output_dir=tmp,
                                    max_depth=max_depth) if Path(winners_path).exists() else {}

    path = None
    tree = res.get("_tree")
    if tree is not None:
        classes = [str(c) for c in res["_classes"]]
        names = {"log10_S": "log10 S", "D": "D", "R": "R", "corr_proxy": "corr. proxy"}
        fig, ax = plt.subplots(figsize=(14, 6.8))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        arts = plot_tree(tree, ax=ax, filled=False, rounded=True, impurity=False, fontsize=8,
                         feature_names=[names.get(f, f) for f in res["_feature_names"]],
                         class_names=[STRATEGY_LABELS.get(c, c) for c in classes])
        _color_tree_nodes(arts, {STRATEGY_LABELS.get(c, c): STRATEGY_COLORS.get(c, MUTED)
                                 for c in classes})
        acc = res.get("test_accuracy")
        ci = (f" [95% CI {res['test_accuracy_ci_lo']:.2f}-{res['test_accuracy_ci_hi']:.2f}]"
              if res.get("test_accuracy_ci_lo") is not None else "")
        base = max(res.get("baseline_always_pre_filter_exact_accuracy") or 0,
                   res.get("baseline_always_in_index_filtered_accuracy") or 0)
        proxy_note = ("proxy from exact neighbours: an upper bound"
                      if (proxy_src or "idealized").startswith("idealized") else
                      "proxy from an approximate probe")
        order_note = ", ".join(STRATEGY_LABELS.get(c, c) for c in classes)
        fig.suptitle(f"Filter-strategy rule (exploratory, depth <= {max_depth}), trained on "
                     f"N <= 500K", fontsize=10.5, color=INK, x=0.01, ha="left", y=0.995)
        def _ar(a, r):
            a, r = res.get(a), res.get(r)
            return (f"{a:.2f}/{r:.2f}" if a is not None and r is not None and r == r
                    else (f"{a:.2f}" if a is not None else "n/a"))
        ax.set_title(f"Held-out N=1M, top-1/regret: this rule {_ar('test_accuracy', 'test_regret')}"
                     f"{ci}; no-proxy tree {_ar('noproxy_test_accuracy', 'noproxy_test_regret')}; "
                     f"1% switch {_ar('baseline_threshold_1pct_accuracy', 'baseline_threshold_1pct_regret')}; "
                     f"best fixed strategy {base:.2f}. {res.get('n_test')} cells; {proxy_note}.\n"
                     f"'value' = training cells per [{order_note}]",
                     fontsize=8, color=INK2, loc="left")
        handles = [Patch(facecolor=STRATEGY_COLORS[c], label=STRATEGY_LABELS[c])
                   for c in STRATEGY_COLORS if c in classes]
        fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.995, 1.0),
                   title="box = majority strategy (deeper tint = purer)", title_fontsize=7)
        path = output_dir / "decision_tree.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        logger.info(f"Saved {path}")
    else:
        logger.warning("No filter-strategy tree to plot (needs Stage C filter runs).")

    itree = idx_res.get("_tree") if idx_res else None
    if itree is not None:
        icls = [str(c) for c in idx_res["_classes"]]
        fig, ax = plt.subplots(figsize=(12, 5.6))
        fig.patch.set_facecolor(SURFACE)
        arts = plot_tree(itree, ax=ax, filled=False, rounded=True, impurity=False, fontsize=8,
                         feature_names=idx_res["_feature_names"],
                         class_names=[INDEX_LABELS.get(c, c) for c in icls])
        _color_tree_nodes(arts, {INDEX_LABELS.get(c, c): INDEX_COLORS.get(c, MUTED)
                                 for c in icls})
        ax.set_title("Index-choice rule (proposal H3, not supported): ties 'always HNSW' on "
                     "held-out N=1M cells", fontsize=9.5, color=INK, loc="left")
        p2 = output_dir / "decision_tree_index.png"
        fig.savefig(p2, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        logger.info(f"Saved {p2}")
    return path


# ---------------------------------------------------------------------------- winner heatmaps

def _fmt_value(objective: str, v: float) -> str:
    if v != v:  # NaN
        return ""
    if objective.startswith("lat"):
        return f"{v / 1000:.1f} ms" if v >= 10_000 else f"{v:.0f} us"
    if objective.startswith("build"):
        return f"{v:.1f} s" if v < 100 else f"{v:.0f} s"
    if objective == "index_bytes":
        return f"{v / 1e9:.2f} GB" if v >= 1e9 else f"{v / 1e6:.0f} MB"
    return f"{v:.3g}"


def plot_winner_heatmap(
    winners_path: str = "results/summary/winners.csv",
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    """Winner map per objective, as small multiples: rows = corpus size N, columns = memory
    budget, and within each panel a D x R grid of cells colored by the winning index.

    Pivoting on (D, R) alone with aggfunc="first" collapsed every N and every memory budget into
    one cell and showed an arbitrary row — hiding the structure the figure exists to show.
    """
    from matplotlib.patches import Patch, Rectangle

    winners = pd.read_csv(winners_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [None, 1.5, 0.5, 0.1]
    budget_label = {None: "unbounded memory", 1.5: "1.5x raw", 0.5: "0.5x raw", 0.1: "0.1x raw"}

    for objective in winners["objective"].unique():
        sub = winners[winners["objective"] == objective].copy()
        if sub.empty:
            continue
        Ns = sorted(sub["N"].unique())
        Ds = sorted(sub["D"].unique())
        Rs = sorted(sub["R"].unique())
        fig, axes = plt.subplots(len(Ns), len(budgets), figsize=(3.1 * len(budgets),
                                 0.62 * len(Ds) * len(Ns) + 1.6), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        present = set()
        for ri, N in enumerate(Ns):
            for ci, b in enumerate(budgets):
                ax = axes[ri][ci]
                ax.set_facecolor(SURFACE)
                cell = sub[sub["N"] == N]
                if b is None:
                    cell = cell[cell["memory_budget_mult"].isna()]
                else:
                    cell = cell[cell["memory_budget_mult"] == b]
                for yi, D in enumerate(Ds):
                    for xi, R in enumerate(Rs):
                        row = cell[(cell["D"] == D) & (cell["R"] == R)]
                        if row.empty:
                            continue  # not in the measured grid: leave bare surface
                        win = str(row["winner"].iloc[0])
                        val = row["value"].iloc[0] if "value" in row else float("nan")
                        if win in ("none", "nan"):
                            face, label, ink = NONE_COLOR, "none", INK2
                            present.add("none")
                        else:
                            face = INDEX_COLORS.get(win, MUTED)
                            label = INDEX_LABELS.get(win, win)
                            ink = INK if face == "#eda100" else "#ffffff"
                            present.add(win)
                        ax.add_patch(Rectangle((xi + 0.04, yi + 0.06), 0.92, 0.88,
                                               facecolor=face, edgecolor="none"))
                        txt = (label if win in ("none", "nan")
                               else label + "\n" + _fmt_value(objective, float(val)))
                        ax.text(xi + 0.5, yi + 0.5, txt, ha="center", va="center",
                                fontsize=7, color=ink, linespacing=1.15)
                ax.set_xlim(0, len(Rs))
                ax.set_ylim(0, len(Ds))
                ax.set_xticks([x + 0.5 for x in range(len(Rs))])
                ax.set_xticklabels([f"R={r:g}" for r in Rs], fontsize=7, color=INK2)
                ax.set_yticks([y + 0.5 for y in range(len(Ds))])
                ax.set_yticklabels([f"D={int(d)}" for d in Ds] if ci == 0 else [],
                                   fontsize=7, color=INK2)
                ax.tick_params(length=0)
                for s in ax.spines.values():
                    s.set_visible(False)
                if ri == 0:
                    ax.set_title(budget_label[b], fontsize=9, color=INK)
                if ci == 0:
                    ax.set_ylabel(f"N = {int(N):,}", fontsize=9, color=INK)

        order = [k for k in INDEX_COLORS if k in present]
        handles = [Patch(facecolor=INDEX_COLORS[k], label=INDEX_LABELS[k]) for k in order]
        if "none" in present:
            handles.append(Patch(facecolor=NONE_COLOR, label="none (no index met R within budget)"))
        fig_h = fig.get_figheight()
        title = OBJECTIVE_TITLES.get(objective, objective)
        fig.suptitle(f"Winner by cell — {title} (test queries; blank = cell not measured)",
                     fontsize=10, color=INK, y=1 - 0.15 / fig_h, va="top")
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, 1 - 0.45 / fig_h))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig_h))
        path = output_dir / f"winner_heatmap_{objective}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        logger.info(f"Saved {path}")
