"""Tests for runner module."""

import shutil
import tempfile

import numpy as np
import pytest


def test_config_hash_deterministic():
    """Same spec should always produce the same hash."""
    from annstudy.runner import config_hash

    spec = {"index_type": "hnsw", "N": 500_000, "D": 384, "M": 32}
    h1 = config_hash(spec)
    h2 = config_hash(spec)
    assert h1 == h2

    spec2 = {"D": 384, "index_type": "hnsw", "M": 32, "N": 500_000}
    h3 = config_hash(spec2)
    assert h1 == h3


def test_config_hash_different_for_different_specs():
    """Different specs should produce different hashes."""
    from annstudy.runner import config_hash

    h1 = config_hash({"index_type": "hnsw", "M": 32})
    h2 = config_hash({"index_type": "hnsw", "M": 64})
    assert h1 != h2


def test_disk_space_check():
    """Disk space check should work on a real filesystem."""
    from annstudy.runner import check_disk_space

    assert check_disk_space("/tmp", 1024)
    assert not check_disk_space("/tmp", 10**18)


def test_rss_increases_after_load(toy_vectors):
    """RSS should increase after loading a FAISS index."""
    import faiss

    from annstudy.runner import measure_rss

    rss_before = measure_rss()

    dim = toy_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        faiss.write_index(index, f.name)
        index_path = f.name

    del index
    loaded = faiss.read_index(index_path)
    _ = loaded.search(toy_vectors[:1], 5)

    rss_after = measure_rss()
    assert rss_after >= rss_before, "RSS should not decrease after loading an index"

    import os
    os.unlink(index_path)
