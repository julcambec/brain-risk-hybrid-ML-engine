"""
Clustering evaluation utilities: Adjusted Rand Index, permutation testing,
and cross-validated model selection across a range of cluster counts.

These functions are algorithm-agnostic: they evaluate any set of cluster
assignments against ground truth or stability criteria.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score


def compute_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Compute the Adjusted Rand Index between two sets of cluster assignments.

    The ARI measures agreement between two partitions, adjusted for chance.
    Values range from -0.5 (worse than random) through 0.0 (random) to
    1.0 (perfect agreement).

    Parameters
    ----------
    labels_a : np.ndarray
        First set of integer cluster labels.
    labels_b : np.ndarray
        Second set of integer cluster labels.

    Returns
    -------
    float
        Adjusted Rand Index.

    Raises
    ------
    ValueError
        If the two label arrays have different lengths.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label arrays must have the same length: {len(labels_a)} != {len(labels_b)}"
        )
    return float(adjusted_rand_score(labels_a, labels_b))


def permutation_test(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_fn: Any,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Permutation test for cluster stability via label randomization.

    Assesses whether the observed clustering is more stable than expected
    by chance. For each permutation, the cluster labels are shuffled and
    the clustering algorithm is re-run. The ARI between the original and
    re-derived labels is compared to a null distribution built from
    permuted-label ARIs.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features); patients only.
    labels : np.ndarray
        Observed cluster assignments to evaluate.
    cluster_fn : callable
        A function with signature ``cluster_fn(features) -> np.ndarray``
        that returns cluster labels for the given feature matrix. This
        allows the test to be agnostic to the clustering algorithm.
    n_permutations : int
        Number of random permutations (default 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``observed_ari`` : float; ARI between two independent runs on
        real data (self-consistency).
        ``null_distribution`` : np.ndarray; ARI values from permuted runs.
        ``p_value`` : float; proportion of null ARIs >= observed ARI.
    """
    rng = np.random.default_rng(seed)

    # Observed self-consistency: cluster twice and measure agreement.
    labels_run1 = cluster_fn(features)
    labels_run2 = cluster_fn(features)
    observed_ari = compute_ari(labels_run1, labels_run2)

    # Null distribution: permute rows, cluster, compare to original labels.
    null_aris: list[float] = []
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(features))
        perm_features = features[perm_idx]
        perm_labels = cluster_fn(perm_features)
        null_aris.append(compute_ari(labels, perm_labels))

    null_distribution = np.array(null_aris)
    p_value = float(np.mean(null_distribution >= observed_ari))

    return {
        "observed_ari": observed_ari,
        "null_distribution": null_distribution,
        "p_value": p_value,
    }


def ari_across_k(
    features: np.ndarray,
    k_range: list[int],
    cluster_factory: Any,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate clustering stability across a range of k values using
    cross-validated ARI.

    For each value of k, the patient feature matrix is split into
    ``n_folds`` folds. On each fold, clustering is performed on the
    training set and the held-out fold is assigned to the nearest cluster.
    The ARI between the held-out assignments and the full-data assignments
    is averaged across folds.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_patients, n_features); patients only.
    k_range : list[int]
        List of cluster counts to evaluate (e.g. ``[2, 3, 4, 5]``).
    cluster_factory : callable
        A function with signature ``cluster_factory(k) -> cluster_fn``
        where ``cluster_fn(features) -> np.ndarray`` performs clustering
        with k clusters. This two-level callable allows the evaluation
        to be agnostic to the algorithm.
    n_folds : int
        Number of cross-validation folds (default 5).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        ``k_values`` : list[int]; evaluated k values.
        ``mean_ari`` : list[float]; mean cross-validated ARI per k.
        ``std_ari`` : list[float]; standard deviation of ARI per k.
        ``best_k`` : int; k with highest mean ARI.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(features)
    indices = np.arange(n_samples)

    mean_aris: list[float] = []
    std_aris: list[float] = []

    for k in k_range:
        cluster_fn = cluster_factory(k)

        # Full-data reference labels
        full_labels = cluster_fn(features)

        fold_aris: list[float] = []
        shuffled = rng.permutation(indices)
        fold_size = n_samples // n_folds

        for fold_idx in range(n_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < n_folds - 1 else n_samples
            val_idx = shuffled[start:end]
            train_idx = np.setdiff1d(shuffled, val_idx)

            train_labels = cluster_fn(features[train_idx])

            # Assign held-out samples to nearest cluster centroid
            from sklearn.metrics import pairwise_distances

            centroids = np.array(
                [
                    features[train_idx][train_labels == c].mean(axis=0)
                    for c in np.unique(train_labels)
                ]
            )
            dists = pairwise_distances(features[val_idx], centroids)
            val_labels = np.unique(train_labels)[dists.argmin(axis=1)]

            # Compare val assignments to the full-data labels for these subjects
            fold_ari = compute_ari(full_labels[val_idx], val_labels)
            fold_aris.append(fold_ari)

        mean_aris.append(float(np.mean(fold_aris)))
        std_aris.append(float(np.std(fold_aris)))

    best_k = k_range[int(np.argmax(mean_aris))]

    return {
        "k_values": k_range,
        "mean_ari": mean_aris,
        "std_ari": std_aris,
        "best_k": best_k,
    }
