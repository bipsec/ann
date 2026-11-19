"""Query-time correlation proxy for the filter-strategy rule (R1).

The filter-strategy rule (``analysis.rule.fit_strategy_rule``) must choose among filter
strategies from inputs observable at query time: selectivity S, dimension D, recall target R.
The one axis that strongly moves the answer — whether the predicate is correlated with the query
— is NOT observable at query time. This module computes a proxy for it that a system *could*
estimate online:

    local_hit_rate = mean over queries of ( fraction of the query's true top-m neighbours
                                            that satisfy the predicate )
    corr_proxy     = log10( local_hit_rate / global_selectivity )

local_hit_rate is computed from the UNFILTERED exact ground truth (``data/gt/gt_N{N}_D{D}.npz``,
top-100) and the predicate mask — no ANN search is needed. corr_proxy > 0 means the predicate is
enriched near the query (positive correlation), < 0 means depleted (negative/anti-correlation),
~0 means uncorrelated. At true query time this is estimable from a cheap approximate top-m probe;
here we use the exact GT as the idealised estimate.

Query slicing is POSITIONAL (decision M8): the proxy is measured on the test queries
(positions 2000..9999).

Output: ``results/summary/filter_proxy.csv`` keyed (N, D, correlation, target_S).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

SUMMARY = Path("results/summary")
GT_DIR = Path("data/gt")


def proxy_for_cell(gt_ids: np.ndarray, mask_fn, N: int, m: int = 100) -> tuple[float, float]:
    """Return (local_hit_rate, measured_S) for one (correlation, target_S) predicate.

    gt_ids: (n_queries, k) unfiltered exact neighbour ids (positional into the N-set), already
            sliced to the query positions to measure. mask_fn(qi) -> bool mask of length N.
    Pure and side-effect free, so it is unit-testable without any data files.
    """
    n_q = gt_ids.shape[0]
    mm = min(m, gt_ids.shape[1])
    hit_rates = np.empty(n_q, dtype=np.float64)
    sel = np.empty(n_q, dtype=np.float64)
    for qi in range(n_q):
        mask = mask_fn(qi)
        top = gt_ids[qi, :mm]
        top = top[top >= 0]
        hit_rates[qi] = float(mask[top].mean()) if top.size else 0.0
        sel[qi] = float(mask.mean())
    return float(hit_rates.mean()), float(sel.mean())


def _corr_proxy(local_hit_rate: float, measured_S: float) -> float:
    """log10(local_hit_rate / measured_S); 0 when either is non-positive (treated as uncorrelated)."""
    if local_hit_rate <= 0 or measured_S <= 0:
        return 0.0
    return float(np.log10(local_hit_rate / measured_S))


def _probe_neighbors(index_path: str, queries: np.ndarray, m: int, probe_ef: int):
    """Approximate unfiltered top-m neighbour ids from a saved base index — what a running
    system can compute online (no ground truth). Returns (n_queries, m) ids, or None on failure."""
    try:
        from annstudy.faiss_setup import ensure_opt_level
        ensure_opt_level()
        import faiss
        idx = faiss.read_index(index_path)
        try:
            idx.hnsw.efSearch = max(probe_ef, m)
        except AttributeError:
            try:
                idx.nprobe = max(8, probe_ef // 8)
            except Exception:
                pass
        _, ids = idx.search(np.ascontiguousarray(queries), m)
        return ids
    except Exception as e:
        logger.warning(f"probe failed for {index_path}: {e!r}")
        return None


def compute_filter_proxy(
    filter_runs_path: str = "results/raw/filter_runs.parquet",
    stage_c_config: str = "configs/stage_c.yaml",
    emb_dir: str = "data/emb",
    gt_dir: str | Path = GT_DIR,
    m: int = 100,
    mode: str = "online",
    probe_ef: int = 64,
    output_dir: str | Path = SUMMARY,
) -> pd.DataFrame:
    """Correlation proxy per (N, D, correlation, target_S) that Stage C measured.

    mode="online" (default): neighbour ids come from a cheap APPROXIMATE unfiltered top-m probe of
    the cell's base index (what a deployed system can do; no ground-truth leak). mode="idealized":
    neighbour ids are the exact unfiltered GT — an upper bound, for comparison only. Each row
    records ``proxy_source`` so the paper can state which was used; if an online probe is
    unavailable for a cell it falls back to the idealized GT and says so.
    """
    from annstudy.data import load_query_split
    from annstudy.filter_runner import make_mask_fn, prepare_filters

    if not Path(filter_runs_path).exists():
        logger.warning(f"No filter runs at {filter_runs_path}; skipping proxy.")
        return pd.DataFrame()

    cols = ["N", "D", "correlation", "target_S"]
    frx = pd.read_parquet(filter_runs_path)
    fr = frx[cols].drop_duplicates()
    cells = fr.to_records(index=False)
    # Per (N,D) base index for the online probe: prefer a saved HNSW base index.
    base_index = {}
    if {"index_type", "base_index_path"}.issubset(frx.columns):
        for (N, D), g in frx.groupby(["N", "D"]):
            hn = g[(g["index_type"] == "hnsw") & g["base_index_path"].notna()]
            pick = hn if not hn.empty else g[g["base_index_path"].notna()]
            if not pick.empty:
                base_index[(int(N), int(D))] = str(pick["base_index_path"].iloc[0])

    with open(stage_c_config) as f:
        cfg = yaml.safe_load(f)
    filters_cfg = cfg["filters"]

    split = load_query_split("data/query_split.json")
    n_tune = len(split["tuning"])
    test_lo, test_hi = n_tune, n_tune + len(split["test"])

    rows = []
    # Group by (N, D) so passages/queries/GT and k-means caches load once per cell.
    for N, D in sorted({(int(n), int(d)) for n, d, _, _ in cells}):
        gt_path = Path(gt_dir) / f"gt_N{N}_D{D}.npz"
        passages = np.asarray(np.memmap(Path(emb_dir) / f"passages_D{D}.f32",
                                        dtype=np.float32, mode="r", shape=(N, D)))
        n_q_total = 10_000
        all_q = np.memmap(Path(emb_dir) / f"queries_D{D}.f32", dtype=np.float32, mode="r",
                          shape=(n_q_total, D))
        test_q = np.ascontiguousarray(all_q[test_lo:test_hi])
        prep = prepare_filters(N, D, passages, filters_cfg)

        # Neighbour ids: online probe (preferred) or exact GT.
        neigh, source = None, None
        if mode == "online" and (N, D) in base_index:
            neigh = _probe_neighbors(base_index[(N, D)], test_q, m, probe_ef)
            source = "online_probe" if neigh is not None else None
        if neigh is None:
            if not gt_path.exists():
                logger.warning(f"No probe index and missing GT for N={N} D={D}; skipping.")
                continue
            neigh = np.load(gt_path)["ids"][test_lo:test_hi]
            source = "idealized_gt" if mode == "idealized" else "idealized_gt_fallback"

        for n2, d2, corr, S in cells:
            if int(n2) != N or int(d2) != D:
                continue
            mask_fn = make_mask_fn(prep, corr, float(S), test_q)
            lhr, meas_S = proxy_for_cell(neigh, mask_fn, N, m=m)
            rows.append({
                "N": N, "D": D, "correlation": corr, "target_S": float(S), "m": m,
                "local_hit_rate": lhr, "measured_S": meas_S,
                "corr_proxy": _corr_proxy(lhr, meas_S), "proxy_source": source,
                "n_queries": neigh.shape[0],
            })
            logger.info(f"proxy[{source}] N={N} D={D} {corr} S={S}: lhr={lhr:.4f} "
                        f"S={meas_S:.4f} proxy={_corr_proxy(lhr, meas_S):+.3f}")

    out = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(output_dir) / "filter_proxy.csv", index=False)
    logger.info(f"Filter proxy saved: {Path(output_dir) / 'filter_proxy.csv'} ({len(out)} rows)")
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="Compute the filter correlation proxy")
    p.add_argument("--filter-runs", default="results/raw/filter_runs.parquet")
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--mode", choices=["online", "idealized"], default="online",
                   help="online = approximate probe (deployable); idealized = exact GT (upper bound)")
    p.add_argument("--probe-ef", type=int, default=64)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute_filter_proxy(filter_runs_path=args.filter_runs, stage_c_config=args.config,
                         m=args.m, mode=args.mode, probe_ef=args.probe_ef)


if __name__ == "__main__":
    main()
