"""Roth et al. (2020)'s evaluation metrics, ported from their reference code.

Source: github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch,
``metrics/{e_recall,nmi,f1,mAP_c,mAP_1000}.py`` and ``metrics/__init__.py``
(the CPU path their sample runs use). Everything is computed on the test set
retrieving against itself, with the query dropped as the first neighbour, on
the features as the network outputs them (L2-normalized in their default
architecture; the head here outputs unit vectors too) and with Euclidean
distances. ``tests`` in the docstrings below means: checked against the
original files on random data, see ``python main_eval/roth_metrics.py --verify DIR``.

* ``recall@k`` -- e_recall: the true class among the k nearest neighbours.
* ``nmi`` -- faiss k-means with as many clusters as classes (20 iterations,
  min_points_per_centroid 1, faiss's default seed), every sample assigned to its
  nearest centroid, sklearn's NMI against the labels.
* ``f1`` -- their clustering F1 on the same k-means: each cluster is named after
  its sample closest to the centroid, then pair-counting precision and recall.
* ``map_c`` -- mAP@C: per query, average precision over the first ``freq``
  neighbours, ``freq`` = size of the query's class *including* the query, divided
  by ``freq`` (so a perfect retrieval scores (freq-1)/freq, unlike MAP@R).
* ``map_1000`` -- the same over the first 1000 neighbours, still divided by ``freq``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
from scipy.special import comb


def _neighbours(features, k):
    import faiss

    index = faiss.IndexFlatL2(features.shape[-1])
    index.add(features)
    return index.search(features, int(k + 1))[1][:, 1:]


def kmeans_assignments(features, n_classes):
    import faiss

    index = faiss.IndexFlatL2(features.shape[-1])
    kmeans = faiss.Clustering(features.shape[-1], n_classes)
    kmeans.niter = 20
    kmeans.min_points_per_centroid = 1
    kmeans.max_points_per_centroid = 1000000000
    kmeans.train(features, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])
    search = faiss.IndexFlatL2(centroids.shape[-1])
    search.add(centroids)
    _, assigned = search.search(features, 1)
    return centroids, assigned.reshape(-1)


def clustering_f1(labels, assigned, features, centroids):
    """Their f1.py, vectorized; identical counts."""
    labels = np.asarray(labels).reshape(-1)
    distance = np.linalg.norm(features - centroids[assigned], axis=1)
    named = np.zeros(len(features))
    for cluster in np.unique(assigned):
        members = np.where(assigned == cluster)[0]
        named[members] = members[np.argmin(distance[members])]
    classes, class_counts = np.unique(labels, return_counts=True)
    items, item_index, item_counts = np.unique(named, return_inverse=True, return_counts=True)
    tp_fp = comb(class_counts, 2).sum()
    class_index = np.searchsorted(classes, labels)
    contingency = np.zeros((len(classes), len(items)))
    np.add.at(contingency, (class_index, item_index), 1)
    tp = comb(contingency, 2).sum()
    fp = tp_fp - tp
    fn = comb(item_counts, 2).sum() - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def _average_precision(nn_labels, labels, window):
    """Their mAP_c / mAP_1000 loop: AP over ``window(freq)`` neighbours, divided by freq."""
    classes, freqs = np.unique(labels, return_counts=True)
    freq_of = dict(zip(classes.tolist(), freqs.tolist()))
    out = np.empty(len(labels))
    for row, label in enumerate(labels):
        freq = freq_of[label]
        hits = nn_labels[row, : window(freq)] == label
        ranks = np.arange(1, len(hits) + 1)
        out[row] = np.sum(np.cumsum(hits) * hits / ranks) / freq
    # Their loop visits rows grouped by class; the mean does not depend on the order.
    return float(out.mean())


def roth_metrics(features, labels, recall_ks=(1, 2, 4), with_clustering=True, with_map_1000=True):
    features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    labels = np.asarray(labels).reshape(-1)
    freqs = np.unique(labels, return_counts=True)[1]
    depth = max(max(recall_ks), int(freqs.max()), 1000 if with_map_1000 else 0)
    depth = min(depth, len(features) - 1)
    nn_labels = labels[_neighbours(features, depth)]
    out = {f"recall@{k}": float(np.mean(np.any(nn_labels[:, :k] == labels[:, None], axis=1))) for k in recall_ks}
    out["map_c"] = _average_precision(nn_labels, labels, lambda freq: freq)
    if with_map_1000:
        out["map_1000"] = _average_precision(nn_labels, labels, lambda freq: 1000)
    if with_clustering:
        from sklearn import metrics

        centroids, assigned = kmeans_assignments(features, len(np.unique(labels)))
        out["nmi"] = float(metrics.cluster.normalized_mutual_info_score(assigned, labels))
        out["f1"] = clustering_f1(labels, assigned, features, centroids)
    return out


def verify(reference_dir):
    """Compare against their original metric files on random clustered data."""

    def load(name):
        spec = importlib.util.spec_from_file_location(name, f"{reference_dir}/r_{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    rng = np.random.default_rng(0)
    n, d, c = 3000, 64, 60
    labels = rng.integers(0, c, n)
    features = rng.normal(size=(c, d))[labels] + 4.0 * rng.normal(size=(n, d))
    features = (features / np.linalg.norm(features, axis=1, keepdims=True)).astype(np.float32)
    ours = roth_metrics(features, labels)
    target = labels.reshape(-1, 1)
    closest = labels[_neighbours(features, 4)]
    centroids, assigned = kmeans_assignments(features, c)
    reference = {
        "recall@1": load("e_recall").Metric(1)(target, closest),
        "recall@2": load("e_recall").Metric(2)(target, closest),
        "recall@4": load("e_recall").Metric(4)(target, closest),
        "nmi": load("nmi").Metric()(target, assigned.reshape(-1, 1)),
        "f1": load("f1").Metric()(target.reshape(-1), assigned, features, centroids),
        "map_c": load("mAP_c").Metric()(target, features),
        "map_1000": load("mAP_1000").Metric()(target, features),
    }
    for key, value in reference.items():
        print(f"{key:10s} reference={value:.8f} ours={ours[key]:.8f} diff={abs(value - ours[key]):.1e}")
    return max(abs(value - ours[key]) for key, value in reference.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", required=True, help="folder with their metric files saved as r_<name>.py")
    worst = verify(parser.parse_args().verify)
    sys.exit(0 if worst < 1e-6 else 1)
