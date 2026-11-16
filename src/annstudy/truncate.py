"""Prefix truncation, L2 renormalization, and LID estimation."""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def truncate_and_renorm(
    input_path: str | Path,
    output_path: str | Path,
    n_vectors: int,
    source_dim: int,
    target_dim: int,
    batch_size: int = 50_000,
) -> None:
    """Slice the first target_dim columns and L2-normalize each row.

    Works on memmap files to limit memory usage.
    """
    output_path = Path(output_path)

    if output_path.exists():
        mm = np.memmap(output_path, dtype=np.float32, mode="r",
                       shape=(n_vectors, target_dim))
        sample_norms = np.linalg.norm(np.array(mm[:100]), axis=1)
        if mm.shape == (n_vectors, target_dim) and np.all(np.abs(sample_norms - 1.0) < 1e-4):
            logger.info(f"{output_path} already exists and is valid, skipping.")
            return

    source = np.memmap(input_path, dtype=np.float32, mode="r",
                       shape=(n_vectors, source_dim))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(output_path, dtype=np.float32, mode="w+",
                    shape=(n_vectors, target_dim))

    for start in tqdm(range(0, n_vectors, batch_size),
                      desc=f"Truncate D={target_dim}"):
        end = min(start + batch_size, n_vectors)
        batch = np.array(source[start:end, :target_dim])
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        batch = batch / norms
        out[start:end] = batch

    out.flush()
    logger.info(f"Truncated and renormalized: {output_path} ({n_vectors}, {target_dim})")


def renorm_inplace(
    path: str | Path,
    n_vectors: int,
    dim: int,
    batch_size: int = 50_000,
) -> None:
    """L2-normalize vectors in-place (no truncation needed)."""
    path = Path(path)
    mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(n_vectors, dim))

    sample_norms = np.linalg.norm(np.array(mm[:100]), axis=1)
    if np.all(np.abs(sample_norms - 1.0) < 1e-5):
        logger.info(f"{path} already unit-normalized, skipping.")
        return

    for start in tqdm(range(0, n_vectors, batch_size), desc=f"Renorm D={dim}"):
        end = min(start + batch_size, n_vectors)
        batch = np.array(mm[start:end])
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mm[start:end] = batch / norms

    mm.flush()
    logger.info(f"Renormalized in-place: {path} ({n_vectors}, {dim})")


def estimate_lid(
    vectors_path: str | Path,
    n_vectors: int,
    dim: int,
    n_sample: int = 10_000,
    k: int = 20,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate local intrinsic dimensionality using MLE (Levina & Bickel, 2004).

    Returns (lid_mean, lid_std) over the sample.
    """
    import faiss

    vectors = np.memmap(vectors_path, dtype=np.float32, mode="r",
                        shape=(n_vectors, dim))

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_vectors, size=min(n_sample, n_vectors), replace=False)
    sample = np.array(vectors[sample_idx]).astype(np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(sample))

    distances, _ = index.search(sample, k + 1)
    distances = distances[:, 1:]

    distances = np.maximum(distances, 1e-20)
    sqrt_distances = np.sqrt(distances)

    log_ratios = np.log(sqrt_distances[:, :-1] / sqrt_distances[:, -1:])
    lid_per_point = -k / np.sum(log_ratios, axis=1)

    lid_per_point = lid_per_point[np.isfinite(lid_per_point)]

    return float(np.mean(lid_per_point)), float(np.std(lid_per_point))


def create_all_truncations(
    base_dir: str | Path = "data/emb",
    n_passages: int = 1_000_000,
    n_queries: int = 10_000,
    source_dim: int = 1536,
    target_dims: list[int] | None = None,
    output_lid_path: str = "results/summary/lid.csv",
) -> None:
    """Create all truncated files and compute LID for each dimension."""
    import csv

    base_dir = Path(base_dir)
    if target_dims is None:
        target_dims = [128, 384, 768, 1536]

    for d in target_dims:
        for prefix, n in [("passages", n_passages), ("queries", n_queries)]:
            src = base_dir / f"{prefix}_D{source_dim}.f32"
            dst = base_dir / f"{prefix}_D{d}.f32"

            if d == source_dim:
                renorm_inplace(src, n, d)
            else:
                truncate_and_renorm(src, dst, n, source_dim, d)

    Path(output_lid_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_lid_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["D", "lid_mean", "lid_std", "n_sample", "k"])

        for d in target_dims:
            path = base_dir / f"passages_D{d}.f32"
            lid_mean, lid_std = estimate_lid(
                path, n_passages, d, n_sample=10_000, k=20
            )
            writer.writerow([d, f"{lid_mean:.2f}", f"{lid_std:.2f}", 10_000, 20])
            logger.info(f"LID at D={d}: {lid_mean:.2f} +/- {lid_std:.2f}")

    logger.info(f"LID table saved to {output_lid_path}")
