"""Exact ground truth computation and verification."""

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_ground_truth(
    passages_path: str | Path,
    queries_path: str | Path,
    n_passages: int,
    n_queries: int,
    dim: int,
    k: int = 100,
    batch_size: int = 256,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact top-k neighbors using FAISS IndexFlatL2.

    Returns (ids, distances) arrays of shape (n_queries, k).
    Saves to .npz if output_path is given.
    """
    import faiss

    output_path = Path(output_path) if output_path else None
    if output_path and output_path.exists():
        data = np.load(output_path)
        if data["ids"].shape == (n_queries, k):
            logger.info(f"Ground truth already exists: {output_path}, skipping.")
            return data["ids"], data["distances"]

    passages = np.memmap(passages_path, dtype=np.float32, mode="r",
                         shape=(n_passages, dim))
    queries = np.memmap(queries_path, dtype=np.float32, mode="r",
                        shape=(n_queries, dim))

    ram_needed_gb = (n_passages * dim * 4) / (1024**3)
    logger.info(
        f"Building flat index: N={n_passages}, D={dim}, "
        f"estimated RAM: {ram_needed_gb:.1f} GB"
    )

    import psutil
    available_gb = psutil.virtual_memory().available / (1024**3)
    if ram_needed_gb > available_gb * 0.8:
        logger.warning(
            f"Flat index needs ~{ram_needed_gb:.1f} GB but only "
            f"{available_gb:.1f} GB available. This may fail or swap."
        )

    index = faiss.IndexFlatL2(dim)
    add_batch = 100_000
    for start in tqdm(range(0, n_passages, add_batch), desc="Adding to flat index"):
        end = min(start + add_batch, n_passages)
        batch = np.ascontiguousarray(np.array(passages[start:end]))
        index.add(batch)

    all_ids = np.zeros((n_queries, k), dtype=np.int64)
    all_dists = np.zeros((n_queries, k), dtype=np.float32)

    for start in tqdm(range(0, n_queries, batch_size), desc="Searching"):
        end = min(start + batch_size, n_queries)
        query_batch = np.ascontiguousarray(np.array(queries[start:end]))
        dists, ids = index.search(query_batch, k)
        all_ids[start:end] = ids
        all_dists[start:end] = dists

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, ids=all_ids, distances=all_dists)
        logger.info(f"Ground truth saved: {output_path}")

    return all_ids, all_dists


def verify_ground_truth(
    gt_path: str | Path,
    passages_path: str | Path,
    queries_path: str | Path,
    n_passages: int,
    n_queries: int,
    dim: int,
    n_check: int = 100,
    k: int = 10,
    seed: int = 99,
) -> bool:
    """Verify ground truth against numpy brute-force for a random subset."""
    data = np.load(gt_path)
    gt_ids = data["ids"][:, :k]

    passages = np.memmap(passages_path, dtype=np.float32, mode="r",
                         shape=(n_passages, dim))
    queries = np.memmap(queries_path, dtype=np.float32, mode="r",
                        shape=(n_queries, dim))

    rng = np.random.default_rng(seed)
    check_indices = rng.choice(n_queries, size=min(n_check, n_queries), replace=False)

    passage_arr = np.array(passages[:n_passages])

    for qi in tqdm(check_indices, desc="Verifying GT"):
        query = np.array(queries[qi])
        dists = np.sum((passage_arr - query) ** 2, axis=1)
        numpy_top_k = np.argpartition(dists, k)[:k]
        numpy_top_k = numpy_top_k[np.argsort(dists[numpy_top_k])]

        gt_set = set(gt_ids[qi].tolist())
        numpy_set = set(numpy_top_k.tolist())

        if gt_set != numpy_set:
            gt_max_dist = dists[gt_ids[qi, k - 1]]
            numpy_max_dist = dists[numpy_top_k[k - 1]]
            if abs(gt_max_dist - numpy_max_dist) > gt_max_dist * 1e-5:
                logger.error(
                    f"Query {qi}: GT and numpy disagree beyond tie tolerance. "
                    f"GT max dist: {gt_max_dist}, numpy max dist: {numpy_max_dist}"
                )
                return False

    logger.info(f"Ground truth verified for {len(check_indices)} queries at k={k}")
    return True


def compute_all_ground_truths(
    emb_dir: str | Path = "data/emb",
    gt_dir: str | Path = "data/gt",
    n_queries: int = 10_000,
    n_values: list[int] | None = None,
    d_values: list[int] | None = None,
    k: int = 100,
) -> None:
    """Compute and verify ground truth for all (N, D) combinations."""
    if n_values is None:
        n_values = [100_000, 500_000, 1_000_000]
    if d_values is None:
        d_values = [128, 384, 768, 1536]

    emb_dir = Path(emb_dir)
    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for n in n_values:
        for d in d_values:
            passages_path = emb_dir / f"passages_D{d}.f32"
            queries_path = emb_dir / f"queries_D{d}.f32"
            gt_path = gt_dir / f"gt_N{n}_D{d}.npz"

            if not passages_path.exists():
                logger.warning(f"Passages not found: {passages_path}, skipping.")
                continue
            if not queries_path.exists():
                logger.warning(f"Queries not found: {queries_path}, skipping.")
                continue

            logger.info(f"Computing ground truth: N={n}, D={d}")
            compute_ground_truth(
                passages_path, queries_path, n, n_queries, d, k=k,
                output_path=gt_path,
            )

            if n <= 500_000:
                verified = verify_ground_truth(
                    gt_path, passages_path, queries_path,
                    n, n_queries, d, n_check=100, k=10,
                )
                if not verified:
                    raise RuntimeError(
                        f"Ground truth verification failed for N={n}, D={d}"
                    )


def main():
    parser = argparse.ArgumentParser(description="Compute ground truth")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n-values", nargs="+", type=int)
    parser.add_argument("--d-values", nargs="+", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    compute_all_ground_truths(
        n_values=args.n_values or [100_000, 500_000, 1_000_000],
        d_values=args.d_values or config["embedding"]["truncations"],
        k=config["ground_truth"]["k"],
    )


if __name__ == "__main__":
    main()
