"""Filter attribute generation and strategies for Stage C."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_uncorrelated_attributes(
    N: int,
    n_attributes: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Assign each of N vectors an integer in [0, n_attributes) uniformly at random.

    Selectivity for threshold t is S = t / n_attributes.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_attributes, size=N)


def generate_correlated_attributes(
    vectors: np.ndarray,
    n_clusters: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-means on vectors. Returns (cluster_ids, centroids).

    The attribute is the cluster ID.
    """
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=10_000,
        max_iter=100,
    )
    cluster_ids = kmeans.fit_predict(vectors)
    return cluster_ids.astype(np.int32), kmeans.cluster_centers_.astype(np.float32)


def make_uncorrelated_mask(
    attributes: np.ndarray,
    target_S: float,
    n_attributes: int = 10_000,
) -> np.ndarray:
    """Boolean mask with selectivity ~target_S using a range predicate."""
    threshold = int(target_S * n_attributes)
    return attributes < threshold


def make_correlated_mask(
    cluster_ids: np.ndarray,
    centroids: np.ndarray,
    query: np.ndarray,
    target_S: float,
    correlation: str,
    n_clusters: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Boolean mask for correlated filters.

    For positive: admit clusters nearest to the query.
    For negative: admit clusters among the 50% farthest from the query.
    Unions clusters until target size is reached.

    ``seed`` controls the random cluster choice for negative correlation only. Callers should
    vary it per query (e.g. base_seed + query position) so the negative filters are not the
    same draw for every query, while staying reproducible.
    """
    N = len(cluster_ids)
    target_count = max(1, int(target_S * N))

    dists_to_centroids = np.sum((centroids - query) ** 2, axis=1)
    sorted_cluster_idx = np.argsort(dists_to_centroids)

    cluster_sizes = np.bincount(cluster_ids, minlength=n_clusters)

    if correlation == "positive":
        order = sorted_cluster_idx
    elif correlation == "negative":
        far_half = sorted_cluster_idx[n_clusters // 2:].copy()  # copy: do not shuffle the view
        rng = np.random.default_rng(seed)
        rng.shuffle(far_half)
        order = far_half
    else:
        raise ValueError(f"Unknown correlation type: {correlation}")

    selected_clusters = set()
    accumulated = 0
    for cidx in order:
        selected_clusters.add(cidx)
        accumulated += cluster_sizes[cidx]
        if accumulated >= target_count:
            break

    mask = np.isin(cluster_ids, list(selected_clusters))
    return mask


def compute_filtered_ground_truth(
    passages: np.ndarray,
    queries: np.ndarray,
    mask_fn,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k over the matching subset for every query.

    mask_fn(query_idx) -> boolean mask of shape (N,)
    Returns (ids, distances) arrays.
    """
    n_queries = queries.shape[0]
    all_ids = np.full((n_queries, k), -1, dtype=np.int64)
    all_dists = np.full((n_queries, k), np.inf, dtype=np.float32)

    for qi in range(n_queries):
        mask = mask_fn(qi)
        matching_idx = np.where(mask)[0]

        if len(matching_idx) == 0:
            continue

        matching_vecs = passages[matching_idx]
        query = queries[qi]
        dists = np.sum((matching_vecs - query) ** 2, axis=1)

        actual_k = min(k, len(matching_idx))
        top_k_local = np.argpartition(dists, actual_k - 1)[:actual_k]
        top_k_local = top_k_local[np.argsort(dists[top_k_local])]

        all_ids[qi, :actual_k] = matching_idx[top_k_local]
        all_dists[qi, :actual_k] = dists[top_k_local]

    return all_ids, all_dists
