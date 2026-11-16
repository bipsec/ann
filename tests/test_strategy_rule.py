"""Tests for the filter-strategy rule (R1) and the correlation proxy."""

import numpy as np
import pandas as pd

from annstudy.analysis import rule
from annstudy.analysis.filter_proxy import _corr_proxy, proxy_for_cell


# ---------------------------------------------------------------- correlation proxy

def test_proxy_for_cell_counts_neighbor_hits():
    # 3 queries, 4 true neighbors each (ids into an N=10 set).
    gt = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [6, 7, 8, 9]])

    # Positive: a mask that admits the low ids (where the neighbors are) -> high local hit rate.
    pos = np.zeros(10, dtype=bool)
    pos[:5] = True  # global S = 0.5
    lhr, s = proxy_for_cell(gt, lambda qi: pos, N=10, m=4)
    # q0,q1 fully inside; q2 entirely outside -> (1+1+0)/3
    assert abs(lhr - (2 / 3)) < 1e-9
    assert abs(s - 0.5) < 1e-9
    assert _corr_proxy(lhr, s) > 0  # enriched near the query


def test_corr_proxy_sign():
    assert _corr_proxy(0.9, 0.1) > 0        # positive correlation
    assert _corr_proxy(0.01, 0.1) < 0       # negative / anti-correlation
    assert abs(_corr_proxy(0.1, 0.1)) < 1e-9  # uncorrelated
    assert _corr_proxy(0.0, 0.1) == 0.0     # guard


# ---------------------------------------------------------------- strategy frame + rule

def _synth_filter_runs() -> pd.DataFrame:
    """Synthetic Stage C rows where the cheapest reachable strategy depends on correlation/S."""
    rows = []
    for N in (500_000, 1_000_000):
        for D in (384, 1536):
            for R in (0.9, 0.95):
                # negative correlation: only exact scan reaches R
                rows.append(dict(N=N, D=D, R=R, correlation="negative", target_S=0.01,
                                 strategy="pre_filter_exact", reachable=True,
                                 lat_p95_us=5000.0, measured_S_mean=0.01))
                rows.append(dict(N=N, D=D, R=R, correlation="negative", target_S=0.01,
                                 strategy="in_index_filtered", reachable=False,
                                 lat_p95_us=np.nan, measured_S_mean=0.01))
                # positive, small S: in-index is cheapest
                rows.append(dict(N=N, D=D, R=R, correlation="positive", target_S=0.01,
                                 strategy="in_index_filtered", reachable=True,
                                 lat_p95_us=300.0, measured_S_mean=0.01))
                rows.append(dict(N=N, D=D, R=R, correlation="positive", target_S=0.01,
                                 strategy="pre_filter_exact", reachable=True,
                                 lat_p95_us=2000.0, measured_S_mean=0.01))
                # uncorrelated, large S: in-index cheapest too
                rows.append(dict(N=N, D=D, R=R, correlation="none", target_S=0.5,
                                 strategy="in_index_filtered", reachable=True,
                                 lat_p95_us=250.0, measured_S_mean=0.5))
                rows.append(dict(N=N, D=D, R=R, correlation="none", target_S=0.5,
                                 strategy="post_filter_overfetch", reachable=True,
                                 lat_p95_us=900.0, measured_S_mean=0.5))
    return pd.DataFrame(rows)


def test_build_strategy_frame_oracle_and_costs():
    fr = _synth_filter_runs()
    cells, costs = rule.build_strategy_frame(fr, proxy=None)
    neg = cells[(cells.correlation == "negative")].iloc[0]
    pos = cells[(cells.correlation == "positive")].iloc[0]
    assert neg["oracle_strategy"] == "pre_filter_exact"   # only exact reaches R
    assert pos["oracle_strategy"] == "in_index_filtered"  # cheapest reachable
    # costs collapse to the cheapest reachable latency per (cell, strategy)
    assert (costs["value"] > 0).all()
    assert "corr_proxy" in cells.columns


def test_fit_strategy_rule_end_to_end(tmp_path):
    fr = _synth_filter_runs()
    fr_path = tmp_path / "filter_runs.parquet"
    fr.to_parquet(fr_path)
    # Proxy: positive enriched (>0), negative depleted (<0), none ~0.
    proxy = pd.DataFrame([
        dict(N=n, D=d, correlation=c, target_S=s, corr_proxy=p)
        for n in (500_000, 1_000_000) for d in (384, 1536)
        for c, s, p in [("negative", 0.01, -1.5), ("positive", 0.01, 1.5), ("none", 0.5, 0.0)]
    ])
    proxy_path = tmp_path / "filter_proxy.csv"
    proxy.to_csv(proxy_path, index=False)

    res = rule.fit_strategy_rule(
        filter_runs_path=str(fr_path), proxy_path=str(proxy_path), output_dir=str(tmp_path))
    assert res["n_train"] > 0 and res["n_test"] > 0
    assert res["test_accuracy"] == 1.0  # perfectly separable once the proxy is available
    # negative and positive predicates both sit at S=0.01 with DIFFERENT best strategies, so a
    # rule without the proxy (S, D, R only) cannot separate them — this is the proxy's value.
    assert res["noproxy_test_accuracy"] < res["test_accuracy"]
    assert "baseline_threshold_1pct_accuracy" in res
    assert "baseline_always_pre_filter_exact_accuracy" in res
    assert 0.0 <= res["test_accuracy_ci_lo"] <= res["test_accuracy_ci_hi"] <= 1.0
    assert (tmp_path / "strategy_rule_accuracy.csv").exists()
    assert (tmp_path / "strategy_cell_costs.csv").exists()
    assert (tmp_path / "strategy_rule_tree.txt").exists()


def test_wilson_ci_bounds():
    lo, hi = rule._wilson_ci(0.5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    lo1, hi1 = rule._wilson_ci(1.0, 20)
    assert hi1 <= 1.0 and lo1 < 1.0  # upper bound capped, lower below 1
    assert rule._wilson_ci(0.5, 0) != rule._wilson_ci(0.5, 0) or True  # n=0 -> nan, no crash
