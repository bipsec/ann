"""Decision rule: fit a small tree to predict the cheapest index."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_decision_rule(
    winners_path: str = "results/summary/winners.csv",
    output_dir: str = "results/summary",
    max_depth: int = 3,
    train_max_N: int = 500_000,
) -> dict:
    """Fit a depth-3 decision tree on cells with N <= train_max_N, test on larger N."""
    from sklearn.tree import DecisionTreeClassifier

    winners = pd.read_csv(winners_path)
    winners = winners[winners["winner"] != "none"].copy()

    latency_winners = winners[winners["objective"] == "lat_p95_us_tuning"].copy()
    if latency_winners.empty:
        logger.warning("No latency winners found")
        return {}

    features = ["N", "D", "R", "memory_budget_mult"]
    latency_winners["memory_budget_mult"] = latency_winners["memory_budget_mult"].fillna(999)

    X = latency_winners[features].values.astype(float)
    X[:, 0] = np.log10(X[:, 0])
    y = latency_winners["winner"].values

    train_mask = latency_winners["N"] <= train_max_N
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test) if len(X_test) > 0 else None

    baseline_always_hnsw = np.mean(y_test == "hnsw") if len(y_test) > 0 else None

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "baseline_always_hnsw": baseline_always_hnsw,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "tree_depth": tree.get_depth(),
        "feature_names": features,
    }

    output_path = Path(output_dir) / "rule_accuracy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(output_path, index=False)
    logger.info(f"Decision rule results saved: {output_path}")

    return results
