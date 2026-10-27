"""Parameter checks and tensor helpers shared by the SSL regularizers."""

import math
from functools import partial

import numpy as np
from loguru import logger


def _positive(method, name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{method} {name} must be finite and positive")
    return value


def _non_negative(method, name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{method} {name} must be finite and non-negative")
    return value


def _probability(method, name, value):
    value = float(value)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"{method} {name} must lie strictly between 0 and 1")
    return value


def _positive_integer(method, name, value):
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{method} {name} must be a positive integer")
    return integer


def parameter_validators(method):
    """``(positive, non_negative, probability, positive_integer)`` checks naming ``method``."""

    return tuple(
        partial(check, method) for check in (_positive, _non_negative, _probability, _positive_integer)
    )


def cross_squared_distances(left, right):
    """Return the ``[len(left), len(right)]`` squared distances between two sets.

    The Gram form keeps memory at ``O(n m)`` instead of ``O(n m d)``. Rounding
    can make the difference slightly negative, so the result is clamped at zero
    before it reaches a softmax.
    """

    left_norms = left.pow(2).sum(dim=1, keepdim=True)
    right_norms = right.pow(2).sum(dim=1, keepdim=True)
    distances = left_norms + right_norms.t() - 2.0 * (left @ right.t())
    return distances.clamp_min(0.0)


def kmeans(
    features,
    k,
    *,
    seed,
    iterations,
    n_init,
    backend,
    use_gpu,
    min_gpu_work,
    purpose,
    require_faiss,
    faiss_gpu_flat_index,
    record,
    return_labels,
):
    """Fit k-means with FAISS (on the GPU when the work warrants it) or scikit-learn.

    ``backend='auto'`` falls back to scikit-learn on hosts without FAISS, and any
    GPU failure falls back to the CPU index. Returns one cluster id per row when
    ``return_labels``, otherwise the ``[k, d]`` float32 centroids. ``record(name)``
    is told which backend actually ran.
    """

    seed = int(seed) % (2**31 - 1)
    if backend != "sklearn":
        try:
            faiss = require_faiss(purpose)
        except ImportError:
            if backend == "faiss":
                raise
            faiss = None
        if faiss is not None:
            prefer_gpu = len(features) * k >= min_gpu_work if use_gpu is None else bool(use_gpu)
            # Held for as long as the index is: a GPU index does not own them.
            gpu_resources = None
            gpu_index = None
            if prefer_gpu:
                built = faiss_gpu_flat_index(faiss, features.shape[1], purpose)
                if built is not None:
                    gpu_index, gpu_resources = built

            def make_clustering():
                return faiss.Kmeans(
                    d=features.shape[1],
                    k=k,
                    niter=int(iterations),
                    seed=seed,
                    # A single unlucky initialization can collapse clusters onto
                    # each other, so keep FAISS's redundant restarts.
                    nredo=max(1, int(n_init) // 5),
                    verbose=False,
                    spherical=False,
                    # The GPU index is attached below instead: gpu=True would
                    # build uncapped resources of its own.
                    gpu=False,
                )

            def result(clustering):
                if return_labels:
                    _, assignments = clustering.index.search(features, 1)
                    return np.asarray(assignments, dtype=np.int64).reshape(-1)
                return np.ascontiguousarray(np.asarray(clustering.centroids, dtype=np.float32))

            if gpu_index is not None:
                try:
                    clustering = make_clustering()
                    clustering.index = gpu_index
                    clustering.train(features)
                    output = result(clustering)
                    record("faiss-gpu")
                    return output
                except Exception as exc:
                    logger.warning(f"{purpose}: FAISS GPU k-means failed ({exc}); retrying on CPU")
                finally:
                    # Drop the index first, then the resources behind it.
                    gpu_index = None
                    clustering = None
                    gpu_resources = None

            clustering = make_clustering()
            clustering.train(features)
            output = result(clustering)
            record("faiss-cpu")
            return output

    from sklearn.cluster import KMeans

    estimator = KMeans(n_clusters=k, n_init=int(n_init), max_iter=int(iterations), random_state=seed)
    if return_labels:
        output = np.asarray(estimator.fit_predict(features), dtype=np.int64)
    else:
        estimator.fit(features)
        output = np.ascontiguousarray(estimator.cluster_centers_.astype(np.float32))
    record("sklearn")
    return output
