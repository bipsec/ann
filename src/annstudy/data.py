"""Corpus and query loading, sampling, and splitting."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def load_msmarco_passages():
    """Load MS MARCO passage corpus via ir_datasets.

    Returns the dataset object (iterable via .docs_iter()).
    Each doc has .doc_id (str) and .text (str).
    """
    import ir_datasets

    return ir_datasets.load("msmarco-passage")


def load_msmarco_queries():
    """Load MS MARCO training queries via ir_datasets.

    Returns the dataset object (iterable via .queries_iter()).
    Each query has .query_id (str) and .text (str).
    """
    import ir_datasets

    return ir_datasets.load("msmarco-passage/train")


def sample_passages(
    n: int = 1_000_000,
    seed: int = 0,
    output_dir: str | Path = DATA_DIR,
) -> np.ndarray:
    """Sample n passage doc_ids uniformly. Return sorted array of integer doc_ids.

    The nested prefix property holds because IDs are sorted:
    IDs[:100_000] is a subset of IDs[:500_000] is a subset of IDs[:1_000_000].

    Two-pass approach to avoid holding all 8.8M passage texts in memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "passage_ids.npy"
    text_path = output_dir / "passages_sampled.jsonl"

    if ids_path.exists() and text_path.exists():
        existing_ids = np.load(ids_path)
        if len(existing_ids) == n:
            logger.info(f"Passage sample already exists ({n} IDs), skipping.")
            return existing_ids

    ds = load_msmarco_passages()

    logger.info("Pass 1: collecting all passage doc_ids...")
    all_doc_ids = np.array(
        [int(doc.doc_id) for doc in ds.docs_iter()], dtype=np.int64
    )
    total = len(all_doc_ids)
    logger.info(f"Total passages in corpus: {total}")

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(total, size=n, replace=False)
    sampled_doc_ids = np.sort(all_doc_ids[sampled_indices])

    np.save(ids_path, sampled_doc_ids)
    logger.info(f"Saved {n} passage IDs to {ids_path}")

    logger.info("Pass 2: collecting texts for sampled passages...")
    needed = set(sampled_doc_ids.tolist())
    texts_by_id = {}
    for doc in ds.docs_iter():
        did = int(doc.doc_id)
        if did in needed:
            texts_by_id[did] = doc.text

    with open(text_path, "w") as f:
        for did in sampled_doc_ids:
            record = {"id": int(did), "text": texts_by_id[did]}
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved passage texts to {text_path}")
    del texts_by_id
    return sampled_doc_ids


def sample_queries(
    n: int = 10_000,
    seed: int = 1,
    output_dir: str | Path = DATA_DIR,
) -> np.ndarray:
    """Sample n query IDs uniformly. Return sorted array of integer query_ids."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "query_ids.npy"
    text_path = output_dir / "queries_sampled.jsonl"

    if ids_path.exists() and text_path.exists():
        existing_ids = np.load(ids_path)
        if len(existing_ids) == n:
            logger.info(f"Query sample already exists ({n} IDs), skipping.")
            return existing_ids

    ds = load_msmarco_queries()

    logger.info("Collecting all training query IDs and texts...")
    all_query_ids = []
    all_texts = {}
    for q in ds.queries_iter():
        qid = int(q.query_id)
        all_query_ids.append(qid)
        all_texts[qid] = q.text

    all_query_ids = np.array(all_query_ids, dtype=np.int64)
    total = len(all_query_ids)
    logger.info(f"Total training queries: {total}")

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(total, size=n, replace=False)
    sampled_query_ids = np.sort(all_query_ids[sampled_indices])

    np.save(ids_path, sampled_query_ids)
    logger.info(f"Saved {n} query IDs to {ids_path}")

    with open(text_path, "w") as f:
        for qid in sampled_query_ids:
            record = {"id": int(qid), "text": all_texts[qid]}
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved query texts to {text_path}")
    del all_texts
    return sampled_query_ids


def split_queries(
    query_ids: np.ndarray,
    tuning_size: int = 2_000,
    output_dir: str | Path = DATA_DIR,
) -> dict:
    """Split query indices into tuning (first 2000) and test (remaining 8000).

    Saves to data/query_split.json.
    """
    output_dir = Path(output_dir)
    split_path = output_dir / "query_split.json"

    if split_path.exists():
        logger.info("Query split already exists, loading.")
        return load_query_split(str(split_path))

    tuning = query_ids[:tuning_size].tolist()
    test = query_ids[tuning_size:].tolist()

    assert len(set(tuning) & set(test)) == 0, "Tuning and test sets overlap!"

    split = {"tuning": tuning, "test": test}
    with open(split_path, "w") as f:
        json.dump(split, f)

    logger.info(
        f"Query split saved: {len(tuning)} tuning, {len(test)} test"
    )
    return split


def load_query_split(path: str = "data/query_split.json") -> dict:
    """Load the query split and validate it."""
    with open(path) as f:
        split = json.load(f)

    tuning_set = set(split["tuning"])
    test_set = set(split["test"])
    assert len(tuning_set & test_set) == 0, "Tuning and test sets overlap!"
    assert len(tuning_set) == 2_000, f"Expected 2000 tuning, got {len(tuning_set)}"
    assert len(test_set) == 8_000, f"Expected 8000 test, got {len(test_set)}"

    return split


def load_passage_texts(path: str | Path = "data/passages_sampled.jsonl") -> list[str]:
    """Load passage texts from JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def load_query_texts(path: str | Path = "data/queries_sampled.jsonl") -> list[str]:
    """Load query texts from JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def verify_nested_prefixes(ids: np.ndarray, prefixes: list[int]) -> bool:
    """Verify that smaller prefix sets are subsets of larger ones."""
    sorted_ids = np.sort(ids)
    assert np.array_equal(ids, sorted_ids), "IDs must be sorted"

    for psize in prefixes:
        assert psize <= len(ids), f"Prefix {psize} exceeds total {len(ids)}"
        prefix = ids[:psize]
        assert np.array_equal(prefix, sorted_ids[:psize]), (
            f"Prefix of size {psize} is not the first {psize} sorted IDs"
        )
    return True
