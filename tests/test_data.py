"""Tests for data module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_nested_prefixes():
    """Sorted IDs should have the prefix subset property."""
    from annstudy.data import verify_nested_prefixes

    ids = np.sort(np.random.default_rng(0).choice(10_000_000, size=1_000_000, replace=False))

    assert verify_nested_prefixes(ids, [100_000, 500_000])

    prefix_100k = set(ids[:100_000].tolist())
    prefix_500k = set(ids[:500_000].tolist())
    full = set(ids.tolist())

    assert prefix_100k.issubset(prefix_500k)
    assert prefix_500k.issubset(full)


def test_query_split_disjoint():
    """Tuning and test query sets must not overlap."""
    from annstudy.data import split_queries

    with tempfile.TemporaryDirectory() as tmpdir:
        query_ids = np.arange(10_000)
        split = split_queries(query_ids, tuning_size=2_000, output_dir=tmpdir)

        assert len(split["tuning"]) == 2_000
        assert len(split["test"]) == 8_000
        assert len(set(split["tuning"]) & set(split["test"])) == 0


def test_load_query_split_validates():
    """Loading a corrupted split should fail validation."""
    from annstudy.data import load_query_split

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_split = {"tuning": list(range(100)), "test": list(range(100))}
        path = Path(tmpdir) / "query_split.json"
        with open(path, "w") as f:
            json.dump(bad_split, f)

        with pytest.raises(AssertionError):
            load_query_split(str(path))
