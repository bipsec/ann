"""Figure generation from summary results."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DIR = Path("figures")


def plot_cost_vs_dimension(
    tuned_path: str = "results/tuned/tuned.parquet",
    recall_targets: list[float] | None = None,
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    """Cost (p95 latency) versus D per index at each recall target."""
    if recall_targets is None:
        recall_targets = [0.95, 0.99]

    tuned = pd.read_parquet(tuned_path)
    reachable = tuned[tuned["reachable"] == True]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for R in recall_targets:
        subset = reachable[reachable["R"] == R]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for idx_type, group in subset.groupby("index_type"):
            group_sorted = group.sort_values("D")
            ax.plot(group_sorted["D"], group_sorted["lat_p95_us_tuning"],
                    marker="o", label=idx_type)

        ax.set_xlabel("Dimension (D)")
        ax.set_ylabel("p95 Latency (us)")
        ax.set_title(f"Cost vs Dimension at R={R}")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        path = output_dir / f"cost_vs_D_R{R}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_winner_heatmap(
    winners_path: str = "results/summary/winners.csv",
    output_dir: str | Path = FIGURE_DIR,
) -> None:
    """Winner heatmap per objective."""
    winners = pd.read_csv(winners_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for objective in winners["objective"].unique():
        subset = winners[winners["objective"] == objective]
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="D", columns="R", values="winner", aggfunc="first"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Winner Map: {objective}")

        for i, d_val in enumerate(pivot.index):
            for j, r_val in enumerate(pivot.columns):
                val = pivot.loc[d_val, r_val]
                ax.text(j, i, str(val)[:8], ha="center", va="center", fontsize=8)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Recall target R")
        ax.set_ylabel("Dimension D")

        path = output_dir / f"winner_heatmap_{objective}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
