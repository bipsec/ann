"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def toy_vectors():
    """100 random unit-norm vectors of dimension 32."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((100, 32)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def toy_queries():
    """10 random unit-norm query vectors of dimension 32."""
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((10, 32)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def medium_vectors():
    """5000 random unit-norm vectors of dimension 64."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5000, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def medium_queries():
    """100 random unit-norm query vectors of dimension 64."""
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((100, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
