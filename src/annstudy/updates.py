"""Insert study for Stage D: measure recall drift as vectors are added."""

import logging
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_update_study(
    index_type: str,
    build_fn,
    search_params: dict,
    passages: np.ndarray,
    queries: np.ndarray,
    gt_ids: np.ndarray,
    gt_distances: np.ndarray,
    initial_n: int = 450_000,
    n_batches: int = 10,
    batch_size: int = 5_000,
    k: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the update study for one index type.

    1. Build on first initial_n vectors.
    2. Insert remaining vectors in batches.
    3. After each batch: measure insert throughput, recall, latency.
    4. Compare with a rebuild from scratch.
    """
    from annstudy.indexes import set_search_params
    from annstudy.metrics import measure_latency, latency_stats, recall_at_k_batch

    dim = passages.shape[1]

    logger.info(f"Building initial {index_type} index on {initial_n} vectors...")
    initial_vecs = np.ascontiguousarray(passages[:initial_n])
    index, build_meta = build_fn(initial_vecs, dim)
    set_search_params(index, index_type, search_params)

    rows = []
    total_inserted = initial_n

    for batch_id in range(n_batches):
        start = initial_n + batch_id * batch_size
        end = start + batch_size
        batch_vecs = np.ascontiguousarray(passages[start:end])

        t0 = time.perf_counter()
        index.add(batch_vecs)
        insert_sec = time.perf_counter() - t0
        total_inserted += batch_size

        per_query_ns = measure_latency(index, queries, k=k, n_passes=3)
        lat = latency_stats(per_query_ns)

        _, ret_ids = index.search(queries, k)
        per_q_recall = recall_at_k_batch(
            ret_ids, gt_distances, passages[:total_inserted], queries, k=k,
        )

        row = {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "index_type": index_type,
            "batch_id": batch_id,
            "inserted_total": total_inserted,
            "insert_seconds": insert_sec,
            "inserts_per_second": batch_size / insert_sec,
            "recall_at_fixed_params": float(np.mean(per_q_recall)),
            "lat_p95_us": lat["lat_p95_us"],
            "retrained": False,
            "rebuild_reference_recall": None,
        }
        rows.append(row)
        logger.info(
            f"  Batch {batch_id}: inserted={total_inserted}, "
            f"recall={row['recall_at_fixed_params']:.4f}, "
            f"p95={lat['lat_p95_us']:.0f}us, "
            f"insert_rate={row['inserts_per_second']:.0f} vec/s"
        )

    logger.info(f"Rebuilding {index_type} from scratch on {total_inserted} vectors...")
    rebuild_vecs = np.ascontiguousarray(passages[:total_inserted])
    rebuild_index, _ = build_fn(rebuild_vecs, dim)
    set_search_params(rebuild_index, index_type, search_params)

    _, ret_ids = rebuild_index.search(queries, k)
    per_q_recall = recall_at_k_batch(
        ret_ids, gt_distances, passages[:total_inserted], queries, k=k,
    )
    rebuild_recall = float(np.mean(per_q_recall))

    rebuild_row = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_type": index_type,
        "batch_id": n_batches,
        "inserted_total": total_inserted,
        "insert_seconds": 0.0,
        "inserts_per_second": 0.0,
        "recall_at_fixed_params": rebuild_recall,
        "lat_p95_us": 0.0,
        "retrained": False,
        "rebuild_reference_recall": rebuild_recall,
    }
    rows.append(rebuild_row)
    logger.info(f"  Rebuild reference recall: {rebuild_recall:.4f}")

    return pd.DataFrame(rows)
