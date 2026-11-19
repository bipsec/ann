"""Tests for the stage coordinator: spec enumeration and skip-hash parity with the runner."""

import pytest

from annstudy import coordinate, faiss_setup
from annstudy.runner import config_hash


@pytest.fixture
def mini_stage_cfg():
    """A tiny stage config: one small cell, flat + hnsw + ivf_flat only."""
    return {
        "stage": "T",
        "cells": {"N": [1000], "D": [128]},
        "indexes": {"flat": {}, "hnsw": {}, "ivf_flat": {}},
        "k": 10,
        "recall_targets": [0.90, 0.95, 0.99],
        "measurement": {
            "latency_passes": 1,
            "warmup_queries": 5,
            "bootstrap_resamples": 100,
        },
    }


def test_enumerate_specs_covers_grid(mini_stage_cfg):
    specs = coordinate.enumerate_specs(mini_stage_cfg, results_dir="results/raw")
    # flat: 1 build; hnsw: 3 (M in 16/32/64); ivf_flat: 3 (c in 2/4/8) = 7 specs
    types = [s["index_type"] for s in specs]
    assert types.count("flat") == 1
    assert types.count("hnsw") == 3
    assert types.count("ivf_flat") == 3
    assert len(specs) == 7


def test_specs_have_all_runner_keys(mini_stage_cfg):
    specs = coordinate.enumerate_specs(mini_stage_cfg, results_dir="results/raw")
    required = {
        "run_id", "N", "D", "k", "index_type", "build_params", "search_grid",
        "seed", "repeat_id", "query_split_path", "passages_path", "queries_path",
        "n_queries", "gt_path", "measurement", "results_dir",
    }
    for s in specs:
        assert required.issubset(s.keys()), f"missing keys: {required - set(s.keys())}"


def test_paths_follow_convention(mini_stage_cfg):
    specs = coordinate.enumerate_specs(mini_stage_cfg, results_dir="results/raw")
    s = specs[0]
    assert s["passages_path"].replace("\\", "/") == "data/emb/passages_D128.f32"
    assert s["queries_path"].replace("\\", "/") == "data/emb/queries_D128.f32"
    assert s["gt_path"].replace("\\", "/") == "data/gt/gt_N1000_D128.npz"
    assert s["query_split_path"].replace("\\", "/") == "data/query_split.json"


def test_skip_hash_matches_runner(mini_stage_cfg):
    """The coordinator's skip hash must equal the hash the runner records."""
    specs = coordinate.enumerate_specs(mini_stage_cfg, results_dir="results/raw")
    for s in specs:
        coord_hash = coordinate.skip_hash(s)
        runner_hash = config_hash({
            "index_type": s["index_type"],
            "N": s["N"],
            "D": s["D"],
            "build_params": s["build_params"],
            "seed": s["seed"],
        })
        assert coord_hash == runner_hash


def test_search_grid_respects_nlist(mini_stage_cfg):
    specs = coordinate.enumerate_specs(mini_stage_cfg, results_dir="results/raw")
    for s in specs:
        if s["index_type"] == "ivf_flat":
            nlist = s["build_params"]["nlist"]
            for sp in s["search_grid"]:
                assert sp["nprobe"] <= nlist


# --- M10 helpers: test-query re-measurement bookkeeping -----------------------

def test_winner_settings_dedup_hnsw():
    import pandas as pd
    grp = pd.DataFrame([
        {"ef_search": 24, "nprobe": None, "k_factor": None},
        {"ef_search": 24, "nprobe": None, "k_factor": None},  # dup (same build, diff R)
        {"ef_search": 64, "nprobe": None, "k_factor": None},
    ])
    settings = coordinate._winner_settings(grp, "hnsw")
    assert settings == [{"ef_search": 24}, {"ef_search": 64}]


def test_winner_settings_ivf_pq_refine():
    import pandas as pd
    grp = pd.DataFrame([{"ef_search": None, "nprobe": 16, "k_factor": 4}])
    assert coordinate._winner_settings(grp, "ivf_pq_refine") == [{"nprobe": 16, "k_factor": 4}]


def test_ensure_opt_level_respects_explicit(monkeypatch):
    monkeypatch.setenv("FAISS_OPT_LEVEL", "avx2")
    assert faiss_setup.ensure_opt_level() == "avx2"


def test_ensure_opt_level_pins_generic_when_binary_has_no_simd(monkeypatch):
    """Pinning is decided by what the faiss BINARY reports, not by which files ship."""
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)
    monkeypatch.setenv(faiss_setup.COMPILE_OPTIONS_ENV, "GENERIC")
    assert faiss_setup.ensure_opt_level() == "generic"


def test_ensure_opt_level_leaves_unset_when_binary_has_simd(monkeypatch):
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)
    monkeypatch.setenv(faiss_setup.COMPILE_OPTIONS_ENV, "OPTIMIZE DD AVX2")
    assert faiss_setup.ensure_opt_level() is None


def test_unified_avx2_wheel_is_not_downgraded(monkeypatch):
    """Regression: a unified abi3 wheel ships no *_avx2 module but IS an AVX2 build.

    The old filename heuristic pinned generic here, which would have silently disabled the
    SIMD path that every reported latency depends on.
    """
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)
    monkeypatch.setenv(faiss_setup.COMPILE_OPTIONS_ENV, "OPTIMIZE DD AVX2")
    monkeypatch.setattr(faiss_setup, "faiss_has_simd_variant", lambda: False)
    assert faiss_setup.faiss_build_has_simd() is True
    assert faiss_setup.ensure_opt_level() is None


def test_opt_level_falls_back_to_filenames_when_probe_fails(monkeypatch):
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)
    monkeypatch.setenv(faiss_setup.COMPILE_OPTIONS_ENV, "")  # probe failed
    monkeypatch.setattr(faiss_setup, "faiss_has_simd_variant", lambda: False)
    assert faiss_setup.ensure_opt_level() == "generic"


def test_reconstruct_build_params_drops_nulls():
    import pandas as pd
    row = pd.Series({"nlist": 2530.0, "m": None, "M": None,
                     "ef_construction": None, "nbits": None})
    assert coordinate._reconstruct_build_params(row) == {"nlist": 2530}
    row2 = pd.Series({"nlist": None, "m": 48.0, "M": None,
                      "ef_construction": None, "nbits": 8.0})
    assert coordinate._reconstruct_build_params(row2) == {"m": 48, "nbits": 8}



def test_meets_R_on_test_flags_the_generalization_gap(tmp_path, monkeypatch):
    """A winner chosen on tuning queries can fall short of R on the test queries.

    `reachable` records that some setting reached R on the TUNING split. Reporting must not
    let that be read as "achieved R on test", so the finalized table carries an explicit flag
    and signed gap.
    """
    import pandas as pd

    from annstudy import coordinate

    # One winner at R=0.95 whose test recall lands just below it, and one comfortably above.
    tuning = pd.DataFrame([
        {"index_type": "hnsw", "N": 1000, "D": 32, "k": 10, "R": 0.95, "reachable": True,
         "best_build_run_id": "b1", "ef_search": 64, "nprobe": None, "k_factor": None},
        {"index_type": "hnsw", "N": 1000, "D": 32, "k": 10, "R": 0.90, "reachable": True,
         "best_build_run_id": "b1", "ef_search": 32, "nprobe": None, "k_factor": None},
    ])
    test_rows = pd.DataFrame([
        {"build_run_id": "b1", "config_hash": "h1", "threads": 1, "query_split": "test",
         "ef_search": 64, "nprobe": None, "k_factor": None, "recall_mean": 0.947,
         "lat_p95_us": 300.0, "lat_p95_ci_lo": 290.0, "lat_p95_ci_hi": 310.0, "qps": None,
         "index_type": "hnsw", "N": 1000, "D": 32, "k": 10},
        {"build_run_id": "b1", "config_hash": "h1", "threads": 1, "query_split": "test",
         "ef_search": 32, "nprobe": None, "k_factor": None, "recall_mean": 0.962,
         "lat_p95_us": 200.0, "lat_p95_ci_lo": 190.0, "lat_p95_ci_hi": 210.0, "qps": None,
         "index_type": "hnsw", "N": 1000, "D": 32, "k": 10},
    ])
    builds = pd.DataFrame([
        {"run_id": "b1", "config_hash": "h1", "index_type": "hnsw", "N": 1000, "D": 32,
         "M": 16, "ef_construction": 200, "nlist": None, "m": None, "nbits": None,
         "threads": 16, "build_seconds": 1.0, "index_bytes": 123, "rss_bytes": 456,
         "seed": 0, "superseded_by": None},
    ])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coordinate, "_load_builds", lambda *a, **k: builds)
    monkeypatch.setattr(coordinate, "_load_memory_rows", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(coordinate, "current_hardware_id", lambda: "hw")

    import annstudy.tune as tune_mod
    all_rows = pd.concat([test_rows, test_rows.assign(query_split="tuning")], ignore_index=True)
    monkeypatch.setattr(tune_mod, "load_sweep_results", lambda *a, **k: all_rows)
    monkeypatch.setattr(tune_mod, "find_cheapest_setting", lambda *a, **k: tuning)

    final = coordinate.finalize_tuned("results/raw", [0.90, 0.95])

    short = final[final["R"] == 0.95].iloc[0]
    assert short["meets_R_on_test"] is False or not bool(short["meets_R_on_test"])
    assert short["recall_gap_vs_R"] == pytest.approx(0.947 - 0.95)

    ok = final[final["R"] == 0.90].iloc[0]
    assert bool(ok["meets_R_on_test"])
    assert ok["recall_gap_vs_R"] > 0


def test_memory_spec_estimate_needs_no_build_params():
    """Regression: admission control crashed the memory pass with KeyError 'build_params'.

    Memory-mode specs describe an index that already exists on disk, so they carry no build
    grid. They also cost only the index plus overhead, not the corpus, since loading just the
    index IS the measurement.
    """
    from annstudy import coordinate

    mem_spec = {"mode": "memory", "index_type": "ivf_pq_refine", "N": 500_000, "D": 1536,
                "index_path": "/tmp/x.faiss"}
    est = coordinate._expected_rss(mem_spec)
    assert est > 0
    # must be far below a build spec for the same cell, which also holds the corpus and the
    # training sample
    build_spec = dict(mem_spec, mode=None,
                      build_params={"nlist": 2828, "m": 384, "nbits": 8})
    assert est < coordinate._expected_rss(build_spec) / 2
    assert coordinate._expected_index_bytes(mem_spec) > 0
    assert coordinate._expected_train_bytes(mem_spec) == 0


def test_pass1_never_retains_indexes_even_when_stage_keeps_them(tmp_path, monkeypatch):
    """Regression: `keep_indexes: true` must not mean "retain every pass-1 index".

    Stage B's full grid is ~205 GB of indexes against 43 GB of disk. The plan asks only that
    the HNSW and IVF-Flat indexes for the large cells survive for Stage C to reuse; retaining
    all build variants is what filled the disk during Stage A's repeat pass.
    """
    from annstudy import coordinate

    cfg = {
        "stage": "B", "k": 10, "keep_indexes": True,
        "cells": {"N": [1_000_000], "D": [1536]},
        "indexes": {"hnsw": {}, "ivf_flat": {}},
        "measurement": {},
    }
    specs = coordinate.enumerate_specs(cfg, stage="B")
    assert specs, "expected specs"
    assert all(s["keep_index"] is False for s in specs)
    # even an explicit CLI override must not turn pass 1 into a disk bomb
    specs2 = coordinate.enumerate_specs(cfg, stage="B", keep_indexes=True)
    assert all(s["keep_index"] is False for s in specs2)


def test_stage_c_retention_set_is_derived_from_stage_c_config(tmp_path):
    """Retention is read from Stage C's own config, so there is one source of truth."""
    from annstudy import coordinate

    cfgfile = tmp_path / "stage_c.yaml"
    cfgfile.write_text(
        "cells:\n  N: [500000, 1000000]\n  D: [384, 1536]\n"
        "indexes:\n  hnsw: {}\n  ivf_flat: {}\n"
    )
    got = coordinate.stage_c_retention_set(str(cfgfile))
    assert got == {(N, D, it) for N in (500_000, 1_000_000)
                   for D in (384, 1536) for it in ("hnsw", "ivf_flat")}
    assert coordinate.stage_c_retention_set(str(tmp_path / "missing.yaml")) == set()


def test_measurement_phases_are_scoped_to_the_stage_cells():
    """Regression: a small stage must not re-measure every earlier stage's winners.

    `--phase all` on a two-cell config re-measured the winners of all ten measured cells,
    spending hours and writing latency rows under a different machine occupancy than the
    campaign that produced the reported numbers (disagreements up to 3.8x).
    """
    import pandas as pd

    from annstudy import coordinate

    cfg = {"stage": "SQ", "k": 10, "cells": {"N": [500_000], "D": [384, 1536]},
           "indexes": {"hnsw_sq": {}}, "measurement": {}}
    assert coordinate.stage_cells(cfg) == {(500_000, 384), (500_000, 1536)}
    # single-cell form, as Stage D uses
    assert coordinate.stage_cells(
        {"cell": {"N": 500_000, "D": 768}}) == {(500_000, 768)}
    assert coordinate.stage_cells({}) == set()

    tuned = pd.DataFrame([
        {"N": 500_000, "D": 384, "index_type": "hnsw_sq", "reachable": True},
        {"N": 1_000_000, "D": 1536, "index_type": "flat", "reachable": True},
        {"N": 100_000, "D": 128, "index_type": "hnsw", "reachable": True},
    ])
    kept = coordinate._restrict_to_cells(tuned, coordinate.stage_cells(cfg), "test")
    assert len(kept) == 1
    assert kept.iloc[0]["index_type"] == "hnsw_sq"
    # no cell set means no restriction, preserving the old behaviour for full-grid stages
    assert len(coordinate._restrict_to_cells(tuned, None, "test")) == 3
