"""Tests for the revised crossover estimator, the H2 verdict, and the crossover cost model."""

import numpy as np
import pandas as pd

from annstudy.analysis.crossover import estimate_crossover, h2_verdict, s_star_log
from annstudy.analysis.crossover_model import predicted_ratio


def test_s_star_log_recovers_power_law_crossing():
    """Exact ~ S^1, index ~ S^0: they cross where S = L_ix / L_ex(1)."""
    S = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
    true = 0.0042
    ex = 1000.0 * (S / true)          # equals 1000 at S = true
    ix = np.full_like(S, 1000.0)
    got = s_star_log(S, ix, ex)
    assert abs(np.log(got / true)) < 1e-9  # exact under log-log linear interpolation


def test_s_star_log_does_not_extrapolate():
    S = np.array([0.01, 0.1])
    assert s_star_log(S, [1.0, 1.0], [2.0, 3.0]) is None  # exact always slower: no crossing


def test_h2_verdict_classes():
    rng = np.random.default_rng(0)
    assert h2_verdict(np.exp(rng.normal(0.0, 0.03, 4000)))[0] == "equivalent"
    assert h2_verdict(np.exp(rng.normal(np.log(0.6), 0.05, 4000)))[0] == "shift"
    assert h2_verdict(np.exp(rng.normal(0.0, 0.4, 4000)))[0] == "inconclusive"


def _synthetic_rows(D, s_star, sigma=0.01):
    """Filter rows for one (N, D) cell whose curves cross at ``s_star``."""
    rows = []
    for S in (0.001, 0.003, 0.01, 0.03, 0.1):
        ex = 500.0 * (S / s_star)
        for strat, it, lat in (("pre_filter_exact", "exact", ex),
                               ("in_index_filtered", "hnsw", 500.0)):
            rows.append({
                "N": 1000, "D": D, "correlation": "none", "R": 0.95, "strategy": strat,
                "index_type": it, "target_S": S, "measured_S_mean": S, "reachable": True,
                "recall_test": 0.97, "meets_R_on_test": True, "lat_p95_us": lat,
                "lat_p95_ci_lo": lat * np.exp(-1.96 * sigma),
                "lat_p95_ci_hi": lat * np.exp(1.96 * sigma), "build_repeat": 0,
            })
    return rows


def test_estimate_crossover_end_to_end(tmp_path):
    """Known S* at three D recovered; ratio, verdict and the D-slope come out right."""
    truth = {128: 0.008, 384: 0.004, 1536: 0.002}
    rows = sum((_synthetic_rows(D, s) for D, s in truth.items()), [])
    p = tmp_path / "f.parquet"
    pd.DataFrame(rows).to_parquet(p)
    res = estimate_crossover(str(p), output_dir=str(tmp_path), n_bootstrap=300, drift=1.0)
    got = res.set_index("D")["crossover_S"]
    for D, s in truth.items():
        assert abs(np.log(got[D] / s)) < 1e-6
        r = res.set_index("D").loc[D]
        assert r["crossover_S_lo"] <= s <= r["crossover_S_hi"]
    h2 = pd.read_csv(tmp_path / "crossover_h2.csv").iloc[0]
    assert abs(h2["ratio_high_over_low"] - 0.25) < 1e-6
    assert h2["verdict"] == "shift"
    expected_slope = np.polyfit(np.log([128, 384, 1536]), np.log(list(truth.values())), 1)[0]
    assert abs(h2["slope_logS_logD"] - expected_slope) < 1e-6
    assert h2["slope_ci95_lo"] <= expected_slope <= h2["slope_ci95_hi"]


def test_rows_failing_R_on_test_are_excluded(tmp_path):
    rows = _synthetic_rows(384, 0.004)
    for r in rows:
        r["meets_R_on_test"] = False
    p = tmp_path / "f.parquet"
    pd.DataFrame(rows).to_parquet(p)
    res = estimate_crossover(str(p), output_dir=str(tmp_path), n_bootstrap=50)
    assert res.empty or not res["crossover_found"].any()


def test_predicted_ratio_signs():
    # exact scan pays more per unit D than the index -> S* falls with D
    assert predicted_ratio(0.5, 0.9, 1.0, 0.0, 384, 1536) < 1
    # index scales like a scan -> S* D-invariant
    assert abs(predicted_ratio(0.9, 0.9, 1.0, 0.0, 384, 1536) - 1.0) < 1e-12
    # (D2/D1)^((c_ix - c_ex)/(a - b))
    assert abs(predicted_ratio(0.5, 1.0, 1.0, 0.0, 100, 400) - 4 ** -0.5) < 1e-12
