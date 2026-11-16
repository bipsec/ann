"""Decision rule: fit a small tree to predict the cheapest index (plan section 5.4).

Reports top-1 agreement with the oracle, regret (chosen cost / oracle cost - 1), the plan's two
baselines, and leave-one-D-out cross-validation. Features are added as the data to support them
arrives: filter selectivity needs Stage C, update ratio needs Stage D.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Latency objective as written by analysis/winners.py. The finalized (test-measured) tuned
#: table yields "lat_p95_us"; the tuning-only schema yields "lat_p95_us_tuning".
LATENCY_OBJECTIVES = ("lat_p95_us", "lat_p95_us_tuning")

#: Stand-in for "no memory budget" so the tree can split on it numerically.
UNBOUNDED_BUDGET = 999.0

#: Index types measured only as a follow-up probe at a handful of cells, never across the full
#: grid. The index-choice rule excludes them so it is trained and tested on the same classes.
PROBE_INDEX_TYPES = ("hnsw_sq",)


def _latency_objective(winners: pd.DataFrame) -> str | None:
    present = set(winners["objective"].unique())
    for name in LATENCY_OBJECTIVES:
        if name in present:
            return name
    return None


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Feature matrix per plan section 5.4, including optional Stage C/D columns."""
    feats: dict[str, np.ndarray] = {
        "log10_N": np.log10(df["N"].to_numpy(dtype=float)),
        "D": df["D"].to_numpy(dtype=float),
        "R": df["R"].to_numpy(dtype=float),
        "memory_budget_mult": df["memory_budget_mult"].fillna(UNBOUNDED_BUDGET).to_numpy(float),
    }
    # log10 S with 0 for unfiltered, per the plan's feature definition.
    if "S" in df.columns:
        s = df["S"].to_numpy(dtype=float)
        feats["log10_S"] = np.where(np.isnan(s) | (s <= 0), 0.0, np.log10(np.clip(s, 1e-12, None)))
    if "correlation" in df.columns:
        codes = {"none": 0.0, "uncorrelated": 0.0, "positive": 1.0, "negative": -1.0}
        feats["correlation"] = np.array(
            [codes.get(str(v), 0.0) for v in df["correlation"].fillna("none")], dtype=float,
        )
    if "update_ratio" in df.columns:
        feats["update_ratio"] = df["update_ratio"].fillna(0.0).to_numpy(dtype=float)

    names = list(feats)
    return np.column_stack([feats[n] for n in names]), names


def _regret(
    costs: pd.DataFrame,
    cells: pd.DataFrame,
    predicted: np.ndarray,
    objective: str,
) -> float | None:
    """Mean of (cost of predicted index / cost of oracle index) - 1 over the given cells.

    A cell where the predicted index is unavailable (did not reach R, or did not fit the
    budget) counts as the worst available cost, since in practice that choice would fail.
    """
    if costs.empty:
        return None
    sub = costs[costs["objective"] == objective]
    if sub.empty:
        return None

    ratios = []
    for (_, cell), pred in zip(cells.iterrows(), predicted, strict=True):
        m = sub[
            (sub["N"] == cell["N"]) & (sub["D"] == cell["D"]) & (sub["R"] == cell["R"])
        ]
        if pd.isna(cell["memory_budget_mult"]):
            m = m[m["memory_budget_mult"].isna()]
        else:
            m = m[m["memory_budget_mult"] == cell["memory_budget_mult"]]
        if m.empty:
            continue
        oracle = float(m["value"].min())
        if oracle <= 0:
            continue
        hit = m[m["index_type"] == pred]
        chosen = float(hit["value"].iloc[0]) if not hit.empty else float(m["value"].max())
        ratios.append(chosen / oracle - 1.0)

    return float(np.mean(ratios)) if ratios else None


def _baseline_budget_aware(cells: pd.DataFrame) -> np.ndarray:
    """Plan baseline 2: HNSW unless the budget is below 0.5x raw, then IVF-PQ."""
    budget = cells["memory_budget_mult"].fillna(UNBOUNDED_BUDGET).to_numpy(dtype=float)
    return np.where(budget < 0.5, "ivf_pq", "hnsw")


def fit_decision_rule(
    winners_path: str = "results/summary/winners.csv",
    output_dir: str = "results/summary",
    costs_path: str | None = None,
    max_depth: int = 3,
    train_max_N: int = 500_000,
) -> dict:
    """Fit a depth-``max_depth`` tree on cells with N <= train_max_N and test on larger N."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    winners = pd.read_csv(winners_path)
    winners = winners[winners["winner"] != "none"].copy()
    if winners.empty:
        logger.warning("No winner rows at all; cannot fit a decision rule.")
        return {}

    objective = _latency_objective(winners)
    if objective is None:
        logger.warning(
            f"No latency objective among {sorted(winners['objective'].unique())}; "
            f"expected one of {LATENCY_OBJECTIVES}."
        )
        return {}

    cells = winners[winners["objective"] == objective].copy()
    if cells.empty:
        logger.warning("No latency winners found")
        return {}

    if costs_path is None:
        costs_path = str(Path(output_dir) / "cell_costs.csv")
        if not Path(costs_path).exists():  # e.g. refit into a scratch dir for plotting
            costs_path = str(Path(winners_path).parent / "cell_costs.csv")
    costs = pd.read_csv(costs_path) if Path(costs_path).exists() else pd.DataFrame()

    # Probe index types (measured at a few cells only, e.g. the two-cell HNSW-SQ8 probe) are not
    # part of the grid the rule is trained AND tested on: a class absent from every held-out cell
    # can only be predicted wrongly there. Relabel those cells with the best main-grid index.
    if not costs.empty and cells["winner"].isin(PROBE_INDEX_TYPES).any():
        costs = costs[~costs["index_type"].isin(PROBE_INDEX_TYPES)]
        sub = costs[costs["objective"] == objective]
        keys = ["N", "D", "R", "memory_budget_mult"]
        best = (sub.loc[sub.groupby(keys, dropna=False)["value"].idxmin()]
                [keys + ["index_type"]].rename(columns={"index_type": "_main_winner"}))
        cells = cells.merge(best, on=keys, how="left")
        probe = cells["winner"].isin(PROBE_INDEX_TYPES)
        cells.loc[probe, "winner"] = cells.loc[probe, "_main_winner"]
        cells = cells[cells["winner"].notna()].drop(columns="_main_winner")
        logger.info(f"Index rule: relabelled {int(probe.sum())} probe-index cell(s) "
                    f"({', '.join(PROBE_INDEX_TYPES)}) with the best main-grid index.")
    elif cells["winner"].isin(PROBE_INDEX_TYPES).any():
        cells = cells[~cells["winner"].isin(PROBE_INDEX_TYPES)]

    X, feature_names = _build_features(cells)
    y = cells["winner"].to_numpy()

    train_mask = (cells["N"] <= train_max_N).to_numpy()
    test_mask = ~train_mask
    n_classes = len(np.unique(y[train_mask])) if train_mask.any() else 0

    results: dict = {
        "objective": objective,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_classes_train": int(n_classes),
        "feature_names": ",".join(feature_names),
        "max_depth": max_depth,
        "train_max_N": train_max_N,
    }

    if not train_mask.any() or n_classes < 1:
        logger.warning("Not enough training cells to fit a rule.")
        return results

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X[train_mask], y[train_mask])

    results["train_accuracy"] = float(tree.score(X[train_mask], y[train_mask]))
    results["tree_depth"] = int(tree.get_depth())
    results["train_regret"] = _regret(
        costs, cells[train_mask], tree.predict(X[train_mask]), objective,
    )

    if test_mask.any():
        y_test = y[test_mask]
        pred_test = tree.predict(X[test_mask])
        results["test_accuracy"] = float(np.mean(pred_test == y_test))
        results["test_regret"] = _regret(costs, cells[test_mask], pred_test, objective)

        always_hnsw = np.full(test_mask.sum(), "hnsw")
        results["baseline_always_hnsw_accuracy"] = float(np.mean(always_hnsw == y_test))
        results["baseline_always_hnsw_regret"] = _regret(
            costs, cells[test_mask], always_hnsw, objective,
        )
        budget_aware = _baseline_budget_aware(cells[test_mask])
        results["baseline_budget_aware_accuracy"] = float(np.mean(budget_aware == y_test))
        results["baseline_budget_aware_regret"] = _regret(
            costs, cells[test_mask], budget_aware, objective,
        )
    else:
        logger.info(
            f"No cells with N > {train_max_N}; held-out generalization is untested "
            f"(expected until Stage B runs at N=1M)."
        )

    # Leave-one-D-out CV: does the rule transfer to a dimension it never saw?
    lodo = []
    for d in sorted(cells["D"].unique()):
        hold = (cells["D"] == d).to_numpy()
        if hold.all() or len(np.unique(y[~hold])) < 1:
            continue
        t = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        t.fit(X[~hold], y[~hold])
        lodo.append({
            "D": int(d),
            "accuracy": float(np.mean(t.predict(X[hold]) == y[hold])),
            "regret": _regret(costs, cells[hold], t.predict(X[hold]), objective),
            "n": int(hold.sum()),
        })
    if lodo:
        lodo_df = pd.DataFrame(lodo)
        lodo_path = Path(output_dir) / "rule_lodo_cv.csv"
        lodo_path.parent.mkdir(parents=True, exist_ok=True)
        lodo_df.to_csv(lodo_path, index=False)
        results["lodo_mean_accuracy"] = float(lodo_df["accuracy"].mean())
        logger.info(f"Leave-one-D-out CV saved: {lodo_path}")

    output_path = Path(output_dir) / "rule_accuracy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(output_path, index=False)

    tree_text = export_text(tree, feature_names=feature_names)
    (Path(output_dir) / "rule_tree.txt").write_text(tree_text, encoding="utf-8")
    logger.info(f"Decision rule results saved: {output_path}")

    results["_tree"] = tree  # returned for plotting; not written to CSV
    results["_feature_names"] = feature_names
    results["_classes"] = list(tree.classes_)
    return results


# --------------------------------------------------------------------------- strategy rule (R1)
# The index rule above never sees the filter axis, so it only ties "always HNSW". The filter axis
# is where the cheapest choice actually changes: this rule predicts the cheapest FILTER STRATEGY
# (pre_filter_exact / in_index_filtered / post_filter_overfetch) from query-time-observable inputs
# (selectivity S, dimension D, recall R) plus a correlation PROXY (analysis.filter_proxy), since
# correlation itself is not observable at query time.

STRATEGY_CELL_KEYS = ["N", "D", "R", "correlation", "target_S"]


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion from n trials (small-n honest)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_strategy_frame(
    fr: pd.DataFrame, proxy: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From filter_runs rows, build (cells, costs) for the strategy rule. Pure/testable.

    cells: one row per (N,D,R,correlation,target_S) with the oracle-best reachable strategy,
    features (log10_S, D, R, corr_proxy), and measured_S. costs: per (cell, strategy) the cheapest
    reachable lat_p95_us (collapsing index_type/overfetch), for regret.
    """
    reach = fr[fr["reachable"] == True].copy()  # noqa: E712
    # Cheapest reachable latency per (cell, strategy), collapsing index_type and overfetch_factor.
    costs = (reach.groupby(STRATEGY_CELL_KEYS + ["strategy"])["lat_p95_us"]
             .min().reset_index().rename(columns={"lat_p95_us": "value"}))

    proxy_map = {}
    if proxy is not None and not proxy.empty:
        for _, r in proxy.iterrows():
            proxy_map[(int(r["N"]), int(r["D"]), str(r["correlation"]), float(r["target_S"]))] = \
                float(r["corr_proxy"])

    cell_rows = []
    for key, g in costs.groupby(STRATEGY_CELL_KEYS):
        N, D, R, corr, S = key
        oracle = g.loc[g["value"].idxmin(), "strategy"]
        meas_S = reach[(reach["N"] == N) & (reach["D"] == D) & (reach["R"] == R)
                       & (reach["correlation"] == corr) & (reach["target_S"] == S)]
        measured_S = float(meas_S["measured_S_mean"].mean()) if "measured_S_mean" in meas_S else float(S)
        cell_rows.append({
            "N": int(N), "D": int(D), "R": float(R), "correlation": str(corr),
            "target_S": float(S), "measured_S": measured_S,
            "corr_proxy": proxy_map.get((int(N), int(D), str(corr), float(S)), 0.0),
            "oracle_strategy": oracle,
        })
    return pd.DataFrame(cell_rows), costs


def _strategy_features(cells: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Query-time-observable features: log10 S, D, R, and the correlation proxy."""
    s = cells["measured_S"].to_numpy(dtype=float)
    feats = {
        "log10_S": np.where(s <= 0, 0.0, np.log10(np.clip(s, 1e-12, None))),
        "D": cells["D"].to_numpy(dtype=float),
        "R": cells["R"].to_numpy(dtype=float),
        "corr_proxy": cells["corr_proxy"].to_numpy(dtype=float),
    }
    names = list(feats)
    return np.column_stack([feats[n] for n in names]), names


def _strategy_regret(costs: pd.DataFrame, cells: pd.DataFrame, predicted: np.ndarray) -> float | None:
    """Mean (predicted strategy cost / oracle cost - 1); unreachable choice charged worst cost."""
    ratios = []
    for (_, cell), pred in zip(cells.iterrows(), predicted, strict=True):
        m = costs
        for col in STRATEGY_CELL_KEYS:
            m = m[m[col] == cell[col]]
        if m.empty:
            continue
        oracle = float(m["value"].min())
        if oracle <= 0:
            continue
        hit = m[m["strategy"] == pred]
        chosen = float(hit["value"].iloc[0]) if not hit.empty else float(m["value"].max())
        ratios.append(chosen / oracle - 1.0)
    return float(np.mean(ratios)) if ratios else None


def fit_strategy_rule(
    filter_runs_path: str = "results/raw/filter_runs.parquet",
    proxy_path: str = "results/summary/filter_proxy.csv",
    output_dir: str = "results/summary",
    max_depth: int = 3,
    train_max_N: int = 500_000,
) -> dict:
    """Depth-``max_depth`` tree predicting the cheapest filter strategy from (S, D, R, corr_proxy).

    filter_runs.parquet is a single-campaign artifact (one hardware_id), so no host filtering.
    Primary split holds out N=1M; also reports leave-one-D-out CV. Baselines: always pre-filter
    exact scan (always reaches R) and always in-index filtered.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    if not Path(filter_runs_path).exists():
        logger.warning(f"No filter runs at {filter_runs_path}; cannot fit strategy rule.")
        return {}
    fr = pd.read_parquet(filter_runs_path)
    proxy = pd.read_csv(proxy_path) if Path(proxy_path).exists() else None
    if proxy is None:
        logger.warning("No filter_proxy.csv; corr_proxy will be 0 (correlation unobserved).")

    cells, costs = build_strategy_frame(fr, proxy)
    if cells.empty:
        logger.warning("No reachable filter cells; cannot fit strategy rule.")
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    costs.to_csv(out / "strategy_cell_costs.csv", index=False)

    X, feature_names = _strategy_features(cells)
    y = cells["oracle_strategy"].to_numpy()
    train = (cells["N"] <= train_max_N).to_numpy()
    test = ~train

    results = {
        "n_cells": int(len(cells)), "n_train": int(train.sum()), "n_test": int(test.sum()),
        "n_classes_train": int(len(np.unique(y[train]))) if train.any() else 0,
        "feature_names": ",".join(feature_names), "max_depth": max_depth,
        "train_max_N": train_max_N, "classes": ",".join(sorted(set(y))),
    }
    if not train.any() or results["n_classes_train"] < 1:
        logger.warning("Not enough training cells for the strategy rule.")
        pd.DataFrame([results]).to_csv(out / "strategy_rule_accuracy.csv", index=False)
        return results

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X[train], y[train])
    results["train_accuracy"] = float(tree.score(X[train], y[train]))
    results["train_regret"] = _strategy_regret(costs, cells[train], tree.predict(X[train]))

    if test.any():
        yt = y[test]
        pt = tree.predict(X[test])
        results["test_accuracy"] = float(np.mean(pt == yt))
        results["test_regret"] = _strategy_regret(costs, cells[test], pt)
        lo, hi = _wilson_ci(float(np.mean(pt == yt)), int(test.sum()))
        results["test_accuracy_ci_lo"], results["test_accuracy_ci_hi"] = lo, hi

        # Does the correlation proxy actually help? The honest comparison is the SAME tree without
        # it (S, D, R only) and a one-line selectivity switch — not just "beats always-exact".
        proxy_idx = feature_names.index("corr_proxy")
        keep = [i for i in range(X.shape[1]) if i != proxy_idx]
        t_np = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X[train][:, keep],
                                                                               y[train])
        pnp = t_np.predict(X[test][:, keep])
        results["noproxy_test_accuracy"] = float(np.mean(pnp == yt))
        results["noproxy_test_regret"] = _strategy_regret(costs, cells[test], pnp)

        # Threshold baseline: a fixed switch at ~1% selectivity (exact scan when the subset is
        # tiny, in-index otherwise) — the obvious heuristic a practitioner would try.
        s_test = cells[test]["measured_S"].to_numpy(dtype=float)
        thr = np.where(s_test < 0.01, "pre_filter_exact", "in_index_filtered")
        results["baseline_threshold_1pct_accuracy"] = float(np.mean(thr == yt))
        results["baseline_threshold_1pct_regret"] = _strategy_regret(costs, cells[test], thr)

        for name, const in [("pre_filter_exact", "pre_filter_exact"),
                            ("in_index_filtered", "in_index_filtered")]:
            base = np.full(test.sum(), const)
            results[f"baseline_always_{name}_accuracy"] = float(np.mean(base == yt))
            results[f"baseline_always_{name}_regret"] = _strategy_regret(costs, cells[test], base)

    lodo = []
    for d in sorted(cells["D"].unique()):
        hold = (cells["D"] == d).to_numpy()
        if hold.all() or len(np.unique(y[~hold])) < 1:
            continue
        t = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        t.fit(X[~hold], y[~hold])
        lodo.append({"D": int(d), "accuracy": float(np.mean(t.predict(X[hold]) == y[hold])),
                     "regret": _strategy_regret(costs, cells[hold], t.predict(X[hold])),
                     "n": int(hold.sum())})
    if lodo:
        results["lodo_mean_accuracy"] = float(pd.DataFrame(lodo)["accuracy"].mean())

    pd.DataFrame([results]).to_csv(out / "strategy_rule_accuracy.csv", index=False)
    (out / "strategy_rule_tree.txt").write_text(
        export_text(tree, feature_names=feature_names), encoding="utf-8")
    logger.info(f"Strategy rule saved: {out / 'strategy_rule_accuracy.csv'}")
    results["_tree"] = tree
    results["_feature_names"] = feature_names
    results["_classes"] = list(tree.classes_)
    return results
