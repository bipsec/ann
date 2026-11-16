"""Sharded, resumable embedding via OpenAI API."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def estimate_cost(
    texts: list[str],
    model: str = "text-embedding-3-large",
    cost_per_1m_tokens: float = 0.13,
) -> dict:
    """Estimate total tokens and cost for the OpenAI embedding API.

    Uses a rough heuristic of 1 token per 4 characters.
    Returns the estimate dict. Does NOT proceed without human approval.
    """
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars // 4
    estimated_cost = (estimated_tokens / 1_000_000) * cost_per_1m_tokens

    estimate = {
        "n_texts": len(texts),
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "model": model,
        "cost_per_1m_tokens": cost_per_1m_tokens,
        "estimated_cost_usd": round(estimated_cost, 2),
    }

    logger.info(
        f"Embedding estimate: {len(texts):,} texts, "
        f"~{estimated_tokens:,} tokens, ~${estimated_cost:.2f} USD"
    )
    return estimate


def embed_shards_openai(
    texts: list[str],
    output_dir: str | Path,
    model: str = "text-embedding-3-large",
    dimensions: int = 1536,
    shard_size: int = 10_000,
    max_retries: int = 5,
    batch_size: int = 2048,
) -> int:
    """Embed texts in shards via OpenAI API. Resumable: skips existing valid shards.

    Returns the number of shards written (including previously existing ones).
    """
    from openai import OpenAI

    client = OpenAI()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_shards = (len(texts) + shard_size - 1) // shard_size

    for shard_idx in tqdm(range(n_shards), desc="Embedding shards"):
        shard_path = output_dir / f"shard_{shard_idx:05d}.npy"

        start = shard_idx * shard_size
        end = min(start + shard_size, len(texts))
        expected_rows = end - start

        if shard_path.exists():
            try:
                existing = np.load(shard_path)
                if existing.shape == (expected_rows, dimensions):
                    logger.debug(f"Shard {shard_idx} already exists, skipping.")
                    continue
            except Exception:
                pass

        shard_texts = texts[start:end]
        embeddings = []

        for batch_start in range(0, len(shard_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(shard_texts))
            batch = shard_texts[batch_start:batch_end]

            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        input=batch,
                        model=model,
                        dimensions=dimensions,
                    )
                    batch_embs = [item.embedding for item in response.data]
                    embeddings.extend(batch_embs)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(
                        f"API error on shard {shard_idx}, batch {batch_start}, "
                        f"attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed after {max_retries} retries on shard {shard_idx}"
                )

        arr = np.array(embeddings, dtype=np.float32)
        assert arr.shape == (expected_rows, dimensions), (
            f"Unexpected shape {arr.shape}, expected ({expected_rows}, {dimensions})"
        )
        np.save(shard_path, arr)

    return n_shards


def assemble_memmap(
    shard_dir: str | Path,
    output_path: str | Path,
    n_total: int,
    dim: int,
) -> np.memmap:
    """Concatenate shards into a single float32 memmap file.

    Skips if output already exists with correct shape.
    """
    shard_dir = Path(shard_dir)
    output_path = Path(output_path)

    if output_path.exists():
        mm = np.memmap(output_path, dtype=np.float32, mode="r", shape=(n_total, dim))
        if mm.shape == (n_total, dim):
            logger.info(f"Memmap {output_path} already exists with correct shape, skipping.")
            return mm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(n_total, dim))

    shard_files = sorted(shard_dir.glob("shard_*.npy"))
    offset = 0
    for shard_path in tqdm(shard_files, desc="Assembling memmap"):
        shard = np.load(shard_path)
        n_rows = shard.shape[0]
        mm[offset : offset + n_rows] = shard
        offset += n_rows

    assert offset == n_total, f"Assembled {offset} rows, expected {n_total}"
    mm.flush()
    logger.info(f"Assembled memmap: {output_path} shape ({n_total}, {dim})")
    return mm


def validate_embeddings(
    path: str | Path, expected_shape: tuple[int, int], norm_tol: float = 1e-3
) -> bool:
    """Check shape, no NaN/inf, all row norms within norm_tol of 1.0.

    OpenAI embeddings are approximately unit-normalized (~1e-3 precision).
    The truncation step re-normalizes to exact unit norm.
    """
    path = Path(path)
    n, dim = expected_shape
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))

    assert mm.shape == expected_shape, f"Shape {mm.shape} != expected {expected_shape}"
    assert not np.any(np.isnan(mm)), "Found NaN values"
    assert not np.any(np.isinf(mm)), "Found inf values"

    batch_size = 10_000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.array(mm[start:end])
        norms = np.linalg.norm(batch, axis=1)
        max_deviation = np.max(np.abs(norms - 1.0))
        assert max_deviation < norm_tol, (
            f"Row norms deviate from 1.0 by up to {max_deviation} "
            f"in rows {start}:{end}"
        )

    logger.info(f"Validation passed: {path}")
    return True


def main():
    """CLI entry point for embedding."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Embed passages and queries")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--passages", default="data/passages_sampled.jsonl")
    parser.add_argument("--queries", default="data/queries_sampled.jsonl")
    parser.add_argument("--approve-cost", action="store_true",
                        help="Skip cost approval prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    emb_config = config["embedding"]
    model = emb_config["model"]
    max_dim = emb_config["max_dim"]

    from annstudy.data import load_passage_texts, load_query_texts

    passages = load_passage_texts(args.passages)
    queries = load_query_texts(args.queries)

    all_texts = passages + queries
    est = estimate_cost(all_texts, model=model)

    if not args.approve_cost:
        print(json.dumps(est, indent=2))
        response = input("Proceed with embedding? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

    logger.info("Embedding passages...")
    embed_shards_openai(
        passages,
        output_dir="data/emb/passage_shards",
        model=model,
        dimensions=max_dim,
        shard_size=emb_config.get("shard_size", 10_000),
    )

    logger.info("Embedding queries...")
    embed_shards_openai(
        queries,
        output_dir="data/emb/query_shards",
        model=model,
        dimensions=max_dim,
        shard_size=emb_config.get("shard_size", 10_000),
    )

    logger.info("Assembling passage memmap...")
    assemble_memmap(
        "data/emb/passage_shards",
        f"data/emb/passages_D{max_dim}.f32",
        len(passages),
        max_dim,
    )

    logger.info("Assembling query memmap...")
    assemble_memmap(
        "data/emb/query_shards",
        f"data/emb/queries_D{max_dim}.f32",
        len(queries),
        max_dim,
    )

    validate_embeddings(
        f"data/emb/passages_D{max_dim}.f32",
        (len(passages), max_dim),
    )
    validate_embeddings(
        f"data/emb/queries_D{max_dim}.f32",
        (len(queries), max_dim),
    )

    logger.info("Embedding complete.")


if __name__ == "__main__":
    main()
