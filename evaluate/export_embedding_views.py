"""Export embedding point clouds for the Trial Atlas embedding viewer.

The viewer (``react_trial_visualizer`` → **Embeddings**) draws one run's points
on a canvas in 2D or 3D, with labeled, unlabeled, validation and test samples
toggled independently. It reads ``public/embedding-views/index.json`` and one
JSON file per view, all written here.

Two sources, with different reach:

* **A saved head** (``head_state.pt``, written by ``--save_head_state``). The run
  is rebuilt from its own ``run_config.json``, and the train split, validation
  set and D_test are embedded again from the cached backbone features. The
  labeled/unlabeled split is reconstructed from the recorded seeds and checked
  against ``split/labeled_positions.npy`` when the run saved one. Both a 2D and
  a 3D projection are computed from the same points.
* **A ``test_tsne.csv``** already written by a run. It holds D_test only, in 2D:
  no labeled/unlabeled distinction (the test set has none) and no third axis
  (t-SNE was fitted with two components). This is what seeded final replays
  without a saved head can offer. A CV run's ``test_tsne_concatenated.csv``
  (every fold's test embedding joined) is exported the same way.

Coordinates are rescaled into [-1, 1] per layout; only relative positions carry
meaning in a t-SNE or PaCMAP projection anyway.

Directories passed without a flag are searched for all three sources. Only runs
that finished are exported from a search: a cross-validation fold must be listed
in its ``cv_summary.json``, a concatenated projection needs the CV run's
aggregate, and a run outside cross-validation needs the stats files training
writes last. Every view records a fingerprint of the files it was built from
(names, sizes, modification times), so a view whose sources have not changed
since its last export is skipped; ``--force`` exports it again.

Examples::

    python evaluate/export_embedding_views.py logs/final_tracking
    python evaluate/export_embedding_views.py --run logs/ablations/.../fold_00/2026-09-17_12-00-00
    python evaluate/export_embedding_views.py --root logs/ablations/unlabeled_batch_size
    python evaluate/export_embedding_views.py --tsne-csv-root logs/final_tracking
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORMAT_VERSION = 1
# Part of every view's source fingerprint. Bump it when the exporter starts
# writing different content for the same run files, so --force is not needed to
# bring old views up to date.
EXPORT_REVISION = 1
VIEW_DIR_NAME = "embedding-views"
INDEX_NAME = "index.json"

SUBSET_LABELED = "labeled"
SUBSET_UNLABELED = "unlabeled"
SUBSET_TRAIN = "train"
SUBSET_VALIDATION = "validation"
SUBSET_TEST = "test"
SUBSET_TEST_QUERY = "test_query"
SUBSET_TEST_GALLERY = "test_gallery"
# The order the viewer lists them in.
SUBSET_ORDER = (
    SUBSET_LABELED,
    SUBSET_UNLABELED,
    SUBSET_TRAIN,
    SUBSET_VALIDATION,
    SUBSET_TEST_QUERY,
    SUBSET_TEST_GALLERY,
    SUBSET_TEST,
)
# The scarce group a subsample must not thin out.
KEEP_WHOLE_SUBSETS = (SUBSET_LABELED,)
PROJECTIONS = ("tsne", "pacmap", "pca")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=[],
        help=(
            "Search these directories for every source (saved heads, epoch tracking, projection "
            "CSVs), export what belongs to finished runs, and skip views that are already up to date"
        ),
    )
    parser.add_argument("--run", nargs="*", default=[], help="Run directories holding head_state.pt")
    parser.add_argument(
        "--root",
        nargs="*",
        default=[],
        help="Search these directories for head_state.pt and export every run found",
    )
    parser.add_argument(
        "--tracking-root",
        nargs="*",
        default=[],
        help=(
            "Search these directories for epoch_tracking/ and export each fold as a view with one "
            "layout per tracked epoch (labeled/unlabeled from graph_nodes.csv)"
        ),
    )
    parser.add_argument("--tsne-csv", nargs="*", default=[], help="test_tsne.csv files to export as 2D test views")
    parser.add_argument(
        "--tsne-csv-root",
        nargs="*",
        default=[],
        help=(
            "Search these directories for test_tsne.csv (and a CV run's test_tsne_concatenated.csv) "
            "and export each as a test view"
        ),
    )
    parser.add_argument("--app", default=str(PROJECT_ROOT / "react_trial_visualizer"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Export every view again, even when its source files are unchanged since the last export",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help=(
            "Also export sources a directory search finds in runs that have not finished (still "
            "training, interrupted, or superseded by a retrained fold)"
        ),
    )
    parser.add_argument("--projection", choices=PROJECTIONS, default="tsne")
    parser.add_argument(
        "--tsne-backend",
        default=None,
        help="t-SNE backend for saved-head views (exact, cuml, torchdr, sklearn); defaults to exact",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=25_000,
        help="Subsample to at most this many points per view (labeled samples are kept whole when they fit)",
    )
    parser.add_argument("--no-3d", action="store_true", help="Skip the 3D projection")
    parser.add_argument("--no-validation", action="store_true", help="Leave the validation set out")
    parser.add_argument("--no-test", action="store_true", help="Leave D_test out")
    parser.add_argument("--device", default=None, help="Device to embed on; defaults to the run's own")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subsampling, the projection and k-means")
    parser.add_argument("--knn", type=int, default=10, help="Neighbors per train point in the kNN graph")
    parser.add_argument("--no-graph", action="store_true", help="Skip the kNN graph, clusters and graph metrics")
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=None,
        help="k for k-means on the unlabeled pool; defaults to SLADE's own k, else one cluster per labeled class",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def view_id(source_path, projection):
    digest = hashlib.sha1(f"{Path(source_path).resolve()}|{projection}".encode()).hexdigest()
    return digest[:16]


def normalize_layout(coordinates):
    """Center and scale a layout into [-1, 1], rounded for a compact file."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    center = (low + high) / 2.0
    half_range = float(np.max(high - low)) / 2.0 or 1.0
    normalized = (coordinates - center) / half_range
    return np.round(normalized, 4).reshape(-1).tolist()


def subsample_indices(subsets, max_points, rng):
    """Pick at most ``max_points`` rows, thinning every subset by the same share.

    Labeled samples are the scarce group the viewer exists to compare against, so
    they are kept whole whenever they fit in half the budget.
    """

    subsets = np.asarray(subsets)
    count = len(subsets)
    if max_points <= 0 or count <= max_points:
        return np.arange(count), None
    keep = np.zeros(count, dtype=bool)
    budget = max_points
    for name in KEEP_WHOLE_SUBSETS:
        rows = np.flatnonzero(subsets == name)
        if 0 < len(rows) <= max_points // 2:
            keep[rows] = True
            budget -= len(rows)
    remaining = np.flatnonzero(~keep)
    share = budget / len(remaining)
    for name in np.unique(subsets[remaining]):
        rows = remaining[subsets[remaining] == name]
        take = int(round(len(rows) * share))
        if take:
            keep[rng.choice(rows, size=min(take, len(rows)), replace=False)] = True
    kept = np.flatnonzero(keep)
    note = (
        f"Subsampled {count} points to {len(kept)}"
        + (" (labeled samples kept whole)" if any(np.any(subsets[kept] == n) for n in KEEP_WHOLE_SUBSETS) else "")
    )
    return kept, note


def project(embeddings, n_components, projection, seed):
    import utils

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if projection == "tsne":
        return utils.project_tsne_embeddings(embeddings, seed=seed, n_components=n_components)
    if projection == "pacmap":
        import pacmap

        return np.asarray(
            pacmap.PaCMAP(n_components=n_components, random_state=seed).fit_transform(embeddings),
            dtype=np.float32,
        )
    from sklearn.decomposition import PCA

    return np.asarray(PCA(n_components=n_components, random_state=seed).fit_transform(embeddings), dtype=np.float32)


def encode_classes(names):
    """Return ``(class_names, class_index)`` with names in first-seen order."""

    lookup = {}
    index = np.empty(len(names), dtype=np.int64)
    for row, name in enumerate(names):
        index[row] = lookup.setdefault(name, len(lookup))
    return list(lookup), index


def encode_subsets(subsets):
    present = [name for name in SUBSET_ORDER if name in set(subsets)]
    lookup = {name: position for position, name in enumerate(present)}
    return present, [lookup[name] for name in subsets]


def resolve_sample_path(dataset, index):
    """The image file behind ``dataset[index]``, through Subset/Combined wrappers, or None."""

    index = int(index)
    for _ in range(16):
        if dataset is None:
            return None
        indices = getattr(dataset, "indices", None)
        if indices is not None and hasattr(dataset, "dataset"):
            index = int(indices[index])
            dataset = dataset.dataset
            continue
        children = getattr(dataset, "datasets", None)
        lengths = getattr(dataset, "lengths", None)
        if children is not None and lengths is not None:
            offsets = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))])
            child = int(np.searchsorted(offsets, index, side="right") - 1)
            index -= int(offsets[child])
            dataset = children[child]
            continue
        for attribute in ("paths", "image_paths", "_image_paths"):
            values = getattr(dataset, attribute, None)
            if values is not None and len(values) > index:
                return str(values[index])
        for attribute in ("samples", "imgs", "_samples"):
            values = getattr(dataset, attribute, None)
            if values is not None and len(values) > index:
                item = values[index]
                return str(item[0] if isinstance(item, (tuple, list)) else item)
        dataset = getattr(dataset, "dataset", None)
    return None


def repository_relative(path):
    """A path relative to the repository when it lies inside it; the image server only serves those."""

    if path is None:
        return None
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def dataset_image_paths(dataset, indices):
    try:
        return [repository_relative(resolve_sample_path(dataset, index)) for index in indices]
    except Exception:
        return [None] * len(indices)


def load_run(run_dir, device=None):
    """The run's recorded configuration, as head_evaluation reads it (no head needed)."""

    from training import head_evaluation
    from training.cli import parser as cli_parser

    request_args = cli_parser.parse_args([])
    request_args.explicit_cli_args = []
    if device is not None:
        request_args.device = device
        request_args.ssl_device = device
        request_args.explicit_cli_args = ["device", "ssl_device"]
    return head_evaluation.load_saved_head_run(Path(run_dir), request_args)


def load_saved_clusters(run_dir):
    """SLADE's saved refreshes: ``[(epoch, {position: cluster}, k), ...]`` in refresh order."""

    directory = Path(run_dir) / "slade_clusters"
    refreshes = []
    for path in sorted(directory.glob("refresh_*.npz")):
        with np.load(path) as saved:
            refreshes.append(
                (
                    int(saved["epoch"]),
                    dict(zip(saved["positions"].tolist(), saved["clusters"].tolist())),
                    int(saved["num_clusters"]),
                )
            )
    return refreshes


def load_saved_center_similarity(run_dir):
    """Per SLADE refresh that saved centers: ``[(epoch, {position: cosine to its cluster center}), ...]``."""

    refreshes = []
    for path in sorted((Path(run_dir) / "slade_clusters").glob("refresh_*.npz")):
        with np.load(path) as saved:
            if "center_similarity" not in saved.files:
                continue
            refreshes.append(
                (
                    int(saved["epoch"]),
                    dict(zip(saved["positions"].tolist(), saved["center_similarity"].astype(np.float64).tolist())),
                )
            )
    return refreshes


def encode_mined_pairs(pairs_path, samples, class_count=None):
    """One epoch's slade_pairs sample, reduced to pairs whose both ends the view holds.

    ``pairs``: view-row index pairs (uint16/uint32); ``similarity``: basis cosine
    as a byte over [-1, 1]; ``kind``: a byte per pair, bit 0 set for a mined
    positive, bit 1 for a same-cluster pair, bit 2 when the cluster is unknown.
    The totals tell how much of the epoch's mining the sample stands for.
    """

    row_of_position = {int(position): row for row, position in enumerate(samples)}
    with np.load(pairs_path) as saved:
        left = np.asarray([row_of_position.get(int(position), -1) for position in saved["left"]], dtype=np.int64)
        right = np.asarray([row_of_position.get(int(position), -1) for position in saved["right"]], dtype=np.int64)
        kept = (left >= 0) & (right >= 0)
        similarity = saved["similarity"].astype(np.float64)[kept]
        positive = saved["positive"].astype(np.int64)[kept]
        same_cluster = saved["same_cluster"].astype(np.int64)[kept]
        totals = {
            "sampledPairs": int(len(kept)),
            "minedPositivePairs": float(saved["mined_positive_pairs"]),
            "minedNegativePairs": float(saved["mined_negative_pairs"]),
            "thresholds": [None if not np.isfinite(value) else float(value) for value in saved["thresholds"]],
        }
    index_dtype = "<u2" if len(samples) < (1 << 16) else "<u4"
    kind = (positive == 1).astype(np.uint8) | ((same_cluster == 1).astype(np.uint8) << 1) | ((same_cluster < 0).astype(np.uint8) << 2)
    return {
        **totals,
        "indexBytes": np.dtype(index_dtype).itemsize,
        "pairs": pack(np.stack([left[kept], right[kept]], axis=1), index_dtype),
        "similarity": pack(np.round((np.clip(similarity, -1.0, 1.0) + 1.0) * 127.5), "<u1"),
        "kind": pack(kind, "<u1"),
    }


def label_names(dataset, labels):
    """Class names where the dataset has them, else ``class <label>``."""

    import utils

    names = utils.class_names_for_labels(dataset, labels)
    return [name or f"class {int(label)}" for name, label in zip(names, labels)]


def read_run_metadata(run_dir):
    config_path = Path(run_dir) / "run_config.json"
    if not config_path.is_file():
        return {}, {}
    config = json.loads(config_path.read_text())
    return config.get("args") or {}, config.get("ssl_config") or {}


def campaign_name(save_dir):
    """The campaign a run belongs to, from its save_dir: ``final_tracking/lrml_prx``.

    Fold, CV-timestamp and scenario directories (``class_subset_..._seed_7``) are
    dropped; the first two remaining parts name the campaign. An absolute path is
    read from after its last ``logs`` or ``runs`` directory.
    """

    if not save_dir:
        return None, None
    parts = list(Path(str(save_dir)).parts)
    if Path(str(save_dir)).is_absolute():
        anchors = [index for index, part in enumerate(parts) if part in ("logs", "runs")]
        parts = parts[anchors[-1] + 1 :] if anchors else parts[-2:]
    stamp = next((part[3:] for part in parts if part.startswith("cv_20")), None)
    kept = [
        part
        for part in parts
        if not part.startswith("fold_") and not part.startswith("cv_20") and "_seed_" not in part and not part.startswith("20")
    ]
    campaign = "/".join(kept[:2]) or None
    if stamp:
        date, _, time = stamp.partition("_")
        stamp = f"{date[5:]} {time[:5].replace('-', ':')}"
    return campaign, stamp


def describe_run(run_dir, args, ssl_config):
    method_params = ssl_config.get("method_params") or {}
    method = (method_params.get("regularizer") or ssl_config.get("method")) if args.get("mode") == "ssl" else args.get("mode")
    budget = None
    if ssl_config.get("labeled_per_class") is not None:
        budget = f"{ssl_config['labeled_per_class']}/class"
    elif ssl_config.get("labeled_fraction") not in (None, 1.0):
        budget = f"{float(ssl_config['labeled_fraction']):.0%} labeled"
    campaign, stamp = campaign_name(args.get("save_dir"))
    fold = args.get("cv_fold")
    run_parts = [
        campaign,
        args.get("dataset"),
        method,
        args.get("loss"),
        budget,
        None if args.get("seed") is None else f"seed {args['seed']}",
        None if stamp is None else f"cv {stamp}",
    ]
    run_label = " · ".join(str(part) for part in run_parts if part)
    return {
        "label": run_label if fold is None else f"{run_label} · fold {fold}",
        "runLabel": run_label,
        "dataset": args.get("dataset"),
        "method": method,
        "loss": args.get("loss"),
        "seed": args.get("seed"),
        "fold": fold,
        "labelBudget": budget,
        # The image server rebuilds this run's datasets to find each sample's file.
        "runDir": str(Path(run_dir).resolve()),
    }


# ----------------------------------------------------------------------------
# Source 1: a saved head
# ----------------------------------------------------------------------------


def embed_dataset(model, args, ssl_config, dataset, name):
    import torch

    import utils
    from training import engine

    pin_memory = torch.device(utils.normalize_device_name(args.device)).type == "cuda"
    loader = engine._make_eval_loader(
        args,
        model,
        dataset,
        f"precompute frozen {name} features",
        pin_memory,
        precompute_features=engine.should_precompute_frozen_features(args, ssl_config),
    )
    try:
        embeddings, labels = utils.extract_eval_embeddings(model, loader, name, device=args.device)
    finally:
        utils.shutdown_dataloaders(loader)
    return (
        engine.to_numpy_embeddings(embeddings),
        engine.to_numpy_embeddings(labels).reshape(-1),
    )


def check_saved_split(run_dir, split):
    """Refuse to label points with a split the run did not actually train on."""

    saved = Path(run_dir) / "split" / "labeled_positions.npy"
    if split is None or not saved.is_file():
        return None
    recorded = np.sort(np.load(saved))
    rebuilt = np.sort(np.asarray(split.labeled_positions, dtype=np.int64))
    if not np.array_equal(recorded, rebuilt):
        raise ValueError(
            f"The labeled split rebuilt for {run_dir} ({len(rebuilt)} positions) does not match "
            f"its saved split/labeled_positions.npy ({len(recorded)} positions)"
        )
    return "Labeled/unlabeled split verified against the run's saved split"


def embed_pretrained(model, args, ssl_config, dataset, name):
    """Embed with the projection head replaced by identity: the untrained DINOv2 features."""

    import torch.nn as nn

    original = model.fc
    model.fc = nn.Identity()
    try:
        return embed_dataset(model, args, ssl_config, dataset, f"{name} (pretrained backbone)")
    finally:
        model.fc = original


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def search(database, queries, k, purpose):
    """Exact cosine search; ``(similarities, indices)`` into ``database``."""

    from training.ssl.algorithms import faiss_flat_ip_search

    k = min(int(k), len(database))
    similarities, indices = faiss_flat_ip_search(normalize_rows(database), normalize_rows(queries), k, purpose)
    return np.asarray(similarities, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def knn_excluding_self(features, k, purpose):
    """Each row's ``k`` nearest other rows, as indices into ``features`` (-1 pads)."""

    count = len(features)
    k = min(int(k), count - 1)
    if k <= 0:
        return np.full((count, 0), -1, dtype=np.int64)
    _, indices = search(features, features, k + 1, purpose)
    neighbors = np.full((count, k), -1, dtype=np.int64)
    rows = np.arange(count)
    for row in rows:
        others = indices[row][indices[row] != row][:k]
        neighbors[row, : len(others)] = others
    return neighbors


def nearest_labeled(features, labeled_rows, query_rows):
    """For each query row: its nearest labeled row other than itself, and the similarity."""

    if len(labeled_rows) == 0 or len(query_rows) == 0:
        return np.full(len(query_rows), -1, dtype=np.int64), np.zeros(len(query_rows), dtype=np.float32)
    similarities, indices = search(
        features[labeled_rows], features[query_rows], min(2, len(labeled_rows)), "embedding view nearest labeled"
    )
    first = labeled_rows[indices[:, 0]]
    choice = first
    similarity = similarities[:, 0]
    if indices.shape[1] > 1:
        is_self = first == query_rows
        choice = np.where(is_self, labeled_rows[indices[:, 1]], first)
        similarity = np.where(is_self, similarities[:, 1], similarities[:, 0])
    elif len(labeled_rows) == 1:
        choice = np.where(first == query_rows, -1, first)
    return choice.astype(np.int64), similarity.astype(np.float32)


def resolve_cluster_count(ssl_config, labeled_labels, unlabeled_count, override):
    labeled_classes = len(np.unique(labeled_labels))
    regularizer_params = (ssl_config.method_params or {}).get("regularizer_params") or {}
    if override is not None:
        k, source = int(override), f"--cluster-k {override}"
    elif ssl_config.method == "slade" and regularizer_params.get("num_clusters"):
        k = int(regularizer_params["num_clusters"])
        source = f"SLADE's num_clusters={k}"
    elif ssl_config.method == "slade" and regularizer_params.get("cluster_ratio") is not None:
        ratio = float(regularizer_params["cluster_ratio"])
        k = int(round(ratio * labeled_classes))
        source = f"SLADE's cluster_ratio={ratio:g} x {labeled_classes} labeled classes"
    else:
        k = labeled_classes
        source = f"one cluster per labeled class ({labeled_classes})"
    return max(2, min(k, unlabeled_count)), source


GRAPH_METRICS = (
    (
        "nearest_labeled_accuracy",
        "Nearest labeled is same class · unlabeled",
        True,
        "Share of unlabeled samples whose nearest labeled sample has their true class: a 1-NN classifier "
        "that only knows the labeled support.",
    ),
    (
        "knn_purity_unlabeled",
        "kNN purity · unlabeled",
        True,
        "Share of each unlabeled sample's k nearest train neighbors (labeled or unlabeled) with its true class.",
    ),
    (
        "knn_purity_labeled",
        "kNN purity · labeled",
        True,
        "The same for labeled samples.",
    ),
    (
        "cluster_purity",
        "Cluster purity · unlabeled pool",
        True,
        "k-means on the unlabeled pool: share of samples whose cluster's majority class is their own.",
    ),
    (
        "cluster_pair_precision",
        "Same-cluster pairs of one class",
        True,
        "Of the unlabeled pairs k-means puts in one cluster, the share that share a class.",
    ),
    (
        "cluster_nmi",
        "Cluster NMI",
        True,
        "Normalized mutual information between k-means clusters and true classes.",
    ),
    (
        "top10_anchor_share",
        "Pull of the top 10% labeled",
        None,
        "Share of unlabeled samples whose nearest labeled sample is one of the 10% of labeled samples "
        "nearest to the most unlabeled ones. High means a few labeled samples anchor most of the pool.",
    ),
    (
        "unused_labeled_share",
        "Labeled nearest to no unlabeled",
        None,
        "Share of labeled samples that are the nearest labeled sample of no unlabeled sample.",
    ),
)


def pool_graph_metrics(features, labels, labeled_rows, unlabeled_rows, k, cluster_k, seed, assignments=None):
    """Graph metrics over the whole train pool of one embedding, and its clusters.

    ``assignments`` (one cluster per unlabeled row) replaces the k-means pass when
    the run saved the clusters it actually trained with.
    """

    from training.ssl.slade import kmeans_cluster_labels, pseudo_label_cluster_quality

    values = {}
    train_rows = np.concatenate([labeled_rows, unlabeled_rows])
    neighbors = knn_excluding_self(features[train_rows], k, "embedding view metrics kNN")
    valid = neighbors >= 0
    train_labels = labels[train_rows]
    same = valid & (train_labels[np.where(valid, neighbors, 0)] == train_labels[:, None])
    purity = same.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    if len(labeled_rows):
        values["knn_purity_labeled"] = float(purity[: len(labeled_rows)].mean())
    if len(unlabeled_rows):
        values["knn_purity_unlabeled"] = float(purity[len(labeled_rows):].mean())

    if len(labeled_rows) and len(unlabeled_rows):
        nearest, _ = nearest_labeled(features, labeled_rows, unlabeled_rows)
        values["nearest_labeled_accuracy"] = float(np.mean(labels[nearest] == labels[unlabeled_rows]))
        pulls = np.bincount(np.searchsorted(np.sort(labeled_rows), nearest), minlength=len(labeled_rows))
        top = max(1, int(np.ceil(0.1 * len(labeled_rows))))
        values["top10_anchor_share"] = float(np.sort(pulls)[::-1][:top].sum() / len(unlabeled_rows))
        values["unused_labeled_share"] = float(np.mean(pulls == 0))
    if assignments is None and cluster_k is not None and len(unlabeled_rows) > cluster_k:
        assignments = kmeans_cluster_labels(normalize_rows(features[unlabeled_rows]), cluster_k, seed=seed)
    if assignments is not None:
        known = assignments >= 0
        quality = pseudo_label_cluster_quality(assignments[known], labels[unlabeled_rows][known])
        values["cluster_purity"] = quality["purity"]
        values["cluster_pair_precision"] = quality["pair_precision"]
        values["cluster_nmi"] = quality["nmi"]
    return values, assignments


def head_view(run_dir, options):
    import utils
    from training import engine, head_evaluation
    from training.cli import parser as cli_parser

    run_dir = Path(run_dir)
    request_args = cli_parser.parse_args([])
    request_args.explicit_cli_args = []
    if options.device is not None:
        request_args.device = options.device
        request_args.ssl_device = options.device
        request_args.explicit_cli_args = ["device", "ssl_device"]
    run = head_evaluation.load_saved_head_run(run_dir, request_args)
    model, bundle, metadata = head_evaluation.rebuild_saved_head_model(run)
    model.eval()
    args, ssl_config = run.args, run.ssl_config
    notes = []

    embeddings, subsets, names, samples, label_values = [], [], [], [], []

    train_embeddings, train_labels = embed_dataset(model, args, ssl_config, bundle.train_dataset, "train")
    split = engine._make_training_split(args, ssl_config, bundle.train_dataset)
    train_subsets = np.full(len(train_labels), SUBSET_TRAIN, dtype=object)
    if split is not None:
        train_subsets[:] = SUBSET_UNLABELED
        labeled = np.asarray(split.labeled_positions, dtype=np.int64)
        train_subsets[labeled[labeled < len(train_subsets)]] = SUBSET_LABELED
        verified = check_saved_split(run_dir, split)
        if verified:
            notes.append(verified)
        if not ssl_config.enabled:
            notes.append("Supervised run: unlabeled samples were not used in training")
    embeddings.append(train_embeddings)
    subsets.extend(train_subsets.tolist())
    names.extend(label_names(bundle.train_dataset, train_labels))
    samples.extend(range(len(train_labels)))
    label_values.extend(np.asarray(train_labels).tolist())

    if not options.no_validation and bundle.valid_dataset is not None and len(bundle.valid_dataset):
        valid_embeddings, valid_labels = embed_dataset(model, args, ssl_config, bundle.valid_dataset, "validation")
        embeddings.append(valid_embeddings)
        subsets.extend([SUBSET_VALIDATION] * len(valid_labels))
        names.extend(label_names(bundle.valid_dataset, valid_labels))
        samples.extend(range(len(valid_labels)))
        label_values.extend(np.asarray(valid_labels).tolist())

    if not options.no_test:
        test_embeddings, test_labels = embed_dataset(model, args, ssl_config, bundle.test_dataset, "test")
        test_subsets = np.full(len(test_labels), SUBSET_TEST, dtype=object)
        query_gallery = utils.get_query_gallery_indices(bundle.test_dataset, len(test_labels))
        if query_gallery is not None:
            test_subsets[query_gallery[1]] = SUBSET_TEST_GALLERY
            test_subsets[query_gallery[0]] = SUBSET_TEST_QUERY
        embeddings.append(test_embeddings)
        subsets.extend(test_subsets.tolist())
        names.extend(label_names(bundle.test_dataset, test_labels))
        samples.extend(range(len(test_labels)))
        label_values.extend(np.asarray(test_labels).tolist())

    embeddings = normalize_rows(np.concatenate(embeddings, axis=0))
    subsets_array = np.asarray(subsets, dtype=object)
    label_values = np.asarray(label_values, dtype=np.int64)

    graph = None
    if not options.no_graph and split is not None:
        labeled_rows = np.flatnonzero(subsets_array == SUBSET_LABELED)
        unlabeled_rows = np.flatnonzero(subsets_array == SUBSET_UNLABELED)
        cluster_k, cluster_source = resolve_cluster_count(
            ssl_config, label_values[labeled_rows], len(unlabeled_rows), options.cluster_k
        )
        saved_refreshes = load_saved_clusters(run_dir)
        saved_assignments = None
        if saved_refreshes:
            refresh_epoch, by_position, saved_k = saved_refreshes[-1]
            # Train rows come first and follow training-subset positions one to one.
            saved_assignments = np.asarray([by_position.get(int(row), -1) for row in unlabeled_rows], dtype=np.int64)
            cluster_k = saved_k
            cluster_source = f"the run's own last SLADE refresh (epoch {refresh_epoch}, saved in slade_clusters/)"
        trained_values, assignments = pool_graph_metrics(
            embeddings,
            label_values,
            labeled_rows,
            unlabeled_rows,
            options.knn,
            cluster_k,
            options.seed,
            assignments=saved_assignments,
        )
        pretrained_train, _ = embed_pretrained(model, args, ssl_config, bundle.train_dataset, "train")
        pretrained = np.zeros((len(embeddings), pretrained_train.shape[1]), dtype=np.float32)
        pretrained[: len(pretrained_train)] = normalize_rows(pretrained_train)
        pretrained_values, _ = pool_graph_metrics(
            pretrained, label_values, labeled_rows, unlabeled_rows, options.knn, cluster_k, options.seed
        )
        cluster_by_row = np.full(len(embeddings), -1, dtype=np.int64)
        if assignments is not None:
            cluster_by_row[unlabeled_rows] = assignments
        graph = {
            "k": int(options.knn),
            "clusterK": int(cluster_k) if assignments is not None else None,
            "clusterSource": (
                (
                    f"Clusters from {cluster_source}."
                    if saved_refreshes
                    else f"k-means on the unlabeled pool of this head's embeddings, k from {cluster_source}. "
                    "Recomputed here, so it approximates the run's last pseudo-label refresh."
                )
                if assignments is not None
                else None
            ),
            "metrics": [
                {
                    "name": name,
                    "label": label,
                    "higherIsBetter": higher,
                    "description": description,
                    "trained": trained_values.get(name),
                    "pretrained": pretrained_values.get(name),
                }
                for name, label, higher, description in GRAPH_METRICS
                if name in trained_values
            ],
            "_cluster_by_row": cluster_by_row,
        }
        notes.append(
            f"Graph metrics use every train sample (cosine, k={options.knn}); the pretrained column is the same "
            "backbone without the trained head."
        )

    rng = np.random.default_rng(options.seed)
    kept, note = subsample_indices(subsets_array, options.max_points, rng)
    if note:
        notes.append(note)
    kept_embeddings = embeddings[kept]
    kept_subsets = subsets_array[kept]
    subsets = kept_subsets.tolist()
    names = [names[row] for row in kept]
    samples = [int(samples[row]) for row in kept]

    if graph is not None:
        # Per-point edges are drawn, so they are taken among the points the view holds.
        view_train = np.flatnonzero(np.isin(kept_subsets, [SUBSET_LABELED, SUBSET_UNLABELED]))
        view_labeled = np.flatnonzero(kept_subsets == SUBSET_LABELED)
        knn = np.full((len(kept), graph["k"]), -1, dtype=np.int64)
        neighbors = knn_excluding_self(kept_embeddings[view_train], graph["k"], "embedding view kNN edges")
        if neighbors.shape[1]:
            knn[view_train, : neighbors.shape[1]] = np.where(neighbors >= 0, view_train[np.maximum(neighbors, 0)], -1)
        nearest, similarity = nearest_labeled(kept_embeddings, view_labeled, np.arange(len(kept)))
        graph["knn"] = knn.reshape(-1).tolist()
        graph["nearestLabeled"] = nearest.tolist()
        graph["nearestLabeledSimilarity"] = np.round(similarity, 3).tolist()
        cluster_by_row = graph.pop("_cluster_by_row")
        graph["cluster"] = cluster_by_row[kept].tolist() if graph["clusterK"] is not None else None
        if len(view_train) < int(np.sum(np.isin(subsets_array, [SUBSET_LABELED, SUBSET_UNLABELED]))):
            notes.append("Edges and nearest-labeled links are drawn among the subsampled points only")

    layouts = {"2d": normalize_layout(project(kept_embeddings, 2, options.projection, options.seed))}
    if not options.no_3d:
        layouts["3d"] = normalize_layout(project(kept_embeddings, 3, options.projection, options.seed))

    class_names, class_index = encode_classes(names)
    subset_names, subset_index = encode_subsets(subsets)
    recorded_args, recorded_ssl = read_run_metadata(run_dir)
    view = {
        "version": FORMAT_VERSION,
        "id": view_id(run_dir, options.projection),
        "source": "head_state",
        "sourcePath": repository_relative(run_dir / "head_state.pt"),
        "projection": options.projection,
        "selectedEpoch": metadata.get("selected_epoch"),
        **describe_run(run_dir, recorded_args, recorded_ssl),
        "subsets": subset_names,
        "classNames": class_names,
        "points": {
            "subset": subset_index,
            "class": class_index.tolist(),
            "sample": samples,
        },
        "layouts": layouts,
        "notes": notes,
    }
    if graph is not None:
        view["graph"] = graph
    return view


# ----------------------------------------------------------------------------
# Source 2: an epoch_tracking/ directory
# ----------------------------------------------------------------------------

# graph_nodes.csv columns carried per tracked epoch, as float arrays.
TRACKING_NODE_METRICS = (
    "same_label_neighbor_fraction",
    "degree",
    "distance_to_labeled_node",
    "distance_to_correct_label_labeled_node",
    "labeled_neighbor_count",
    "same_label_labeled_neighbor_count",
    "strongest_neighbor_same_label",
)


def epoch_from_slug(name):
    """``epoch_m001`` is -1, ``epoch_0015`` is 15."""

    number = name.removeprefix("epoch_")
    return -int(number[1:]) if number.startswith("m") else int(number)


def procrustes_align(layout, reference):
    """Rotate/reflect and scale ``layout`` onto ``reference`` (both centered here).

    Each tracked epoch fits its own projection, which is free to rotate or mirror
    the whole picture. Aligning every epoch to the previous one on the shared
    samples removes that arbitrary motion, so what still moves between frames is
    structure the projection actually changed.
    """

    source = layout - layout.mean(axis=0)
    target = reference - reference.mean(axis=0)
    source_scale = np.sqrt((source**2).sum() / len(source)) or 1.0
    target_scale = np.sqrt((target**2).sum() / len(target)) or 1.0
    u, _, vt = np.linalg.svd((source / source_scale).T @ (target / target_scale))
    rotation = u @ vt
    return (source / source_scale) @ rotation * target_scale


def float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def encode_graph_edges(edges_path, position_to_view, count):
    """One epoch's graph_edges.npz as view-index pairs, base64 packed.

    ``pairs`` is little-endian ``[from0, to0, from1, to1, ...]`` with
    ``indexBytes`` per index. A symmetric graph lists each edge once. Weights are
    left out when every edge has the same one (binary kNN), and quantized to a
    byte of ``weightScale`` otherwise.

    Where the run saved its neighbor search, ``knn`` holds each view row's own
    ``k`` neighbors in rank order (``indexBytes`` each, all ones for an empty
    slot) and ``knnSimilarity`` their cosine similarity as a byte over [-1, 1].
    """

    with np.load(edges_path) as saved:
        positions = saved["positions"]
        source = saved["source"].astype(np.int64)
        target = saved["target"].astype(np.int64)
        weight = saved["weight"].astype(np.float64)
        symmetric = bool(saved["symmetric"])
        knn = saved["knn"].astype(np.int64) if "knn" in saved.files else None
        knn_similarity = saved["knn_similarity"].astype(np.float32) if "knn_similarity" in saved.files else None
    view_of_node = np.asarray([position_to_view.get(int(position), -1) for position in positions], dtype=np.int64)
    first, second = view_of_node[source], view_of_node[target]
    kept = (first >= 0) & (second >= 0)
    first, second, weight = first[kept], second[kept], weight[kept]
    # One value short of 2**16 so the all-ones empty-slot marker is never a real row.
    index_dtype = np.dtype("<u2") if count < 1 << 16 else np.dtype("<u4")
    pairs = np.stack([first, second], axis=1).astype(index_dtype).reshape(-1)
    encoded = {
        "edgeCount": int(len(first)),
        "symmetric": symmetric,
        "indexBytes": index_dtype.itemsize,
        "pairs": base64.b64encode(pairs.tobytes()).decode("ascii"),
    }
    if len(weight) and not np.allclose(weight, weight[0]):
        scale = float(np.abs(weight).max()) or 1.0
        encoded["weightScale"] = scale
        quantized = np.round(np.clip(weight / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        encoded["weights"] = base64.b64encode(quantized.tobytes()).decode("ascii")
    else:
        encoded["weight"] = float(weight[0]) if len(weight) else None
    if knn is not None:
        k = knn.shape[1]
        rows = np.full((count, k), -1, dtype=np.int64)
        similarities = np.full((count, k), np.nan, dtype=np.float32)
        mapped = np.where(knn >= 0, view_of_node[np.maximum(knn, 0)], -1)
        placed = view_of_node >= 0
        rows[view_of_node[placed]] = mapped[placed]
        if knn_similarity is not None:
            similarities[view_of_node[placed]] = knn_similarity[placed]
        empty = np.iinfo(index_dtype).max
        encoded["k"] = int(k)
        encoded["knn"] = base64.b64encode(
            np.where(rows >= 0, rows, empty).astype(index_dtype).reshape(-1).tobytes()
        ).decode("ascii")
        if knn_similarity is not None:
            quantized = np.round((np.clip(np.nan_to_num(similarities, nan=-1.0), -1.0, 1.0) + 1.0) * 127.5)
            encoded["knnSimilarity"] = base64.b64encode(quantized.astype(np.uint8).reshape(-1).tobytes()).decode("ascii")
    return encoded


def pack(values, dtype):
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).reshape(-1).tobytes()).decode("ascii")


def quantize_unit(values):
    """[0, 1] floats as little-endian uint16 (NaN and negatives become 0)."""

    return np.round(np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0), 0.0, 1.0) * 65535.0)


def class_index_dtype(class_count):
    # Headroom for the few predicted classes that no tracked row carries.
    return "<i2" if class_count < (1 << 15) - 1024 else "<i4"


def encode_class_evidence(evidence_path, position_to_view, count, class_index_of_label, class_count):
    """One epoch's class_evidence.npz in view-row order.

    Per map (``lp``, ``hoffer``): ``topClass`` (int16/int32 view class indices,
    -1 empty) and ``topProbability`` (uint16 over [0, 1]), ``topK`` per row;
    ``certainty`` and, for label propagation, Iscen's weight ``weight``, both
    uint16 over [0, 1].
    """

    with np.load(evidence_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    rows = np.asarray([position_to_view.get(int(position), -1) for position in arrays["positions"]], dtype=np.int64)
    placed = rows >= 0
    class_dtype = class_index_dtype(class_count)
    encoded = {"classBytes": np.dtype(class_dtype).itemsize}

    def in_view_order(values, fill):
        shaped = np.asarray(values)
        target = np.full((count,) + shaped.shape[1:], fill, dtype=np.float64)
        target[rows[placed]] = shaped[placed]
        return target

    for prefix in ("lp", "hoffer", "prototype"):
        if f"{prefix}_top_classes" not in arrays:
            continue
        top_classes = arrays[f"{prefix}_top_classes"].astype(np.int64)
        view_classes = np.vectorize(lambda label: class_index_of_label(int(label)) if label >= 0 else -1, otypes=[np.int64])(
            top_classes
        )
        entry = {
            "topK": int(top_classes.shape[1]),
            "topClass": pack(in_view_order(view_classes, -1), class_dtype),
            "topProbability": pack(quantize_unit(in_view_order(arrays[f"{prefix}_top_probabilities"], 0.0)), "<u2"),
            "certainty": pack(quantize_unit(in_view_order(arrays[f"{prefix}_certainty"], np.nan)), "<u2"),
        }
        if f"{prefix}_weight" in arrays:
            entry["weight"] = pack(quantize_unit(in_view_order(arrays[f"{prefix}_weight"], np.nan)), "<u2")
        encoded[prefix] = entry
    if "hoffer_distance_scale" in arrays:
        encoded["hofferDistanceScale"] = float(arrays["hoffer_distance_scale"])
    if "prototype_temperature" in arrays:
        encoded["prototypeTemperature"] = float(arrays["prototype_temperature"])
    return encoded


def quantize_byte(values):
    """[0, 1] floats as a byte (NaN and negatives become 0)."""

    return np.round(np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0), 0.0, 1.0) * 255.0)


def encode_propagation(propagation_path, position_to_view, count, class_index_of_label, class_count):
    """One epoch's label_propagation.npz in view-row order.

    The series itself is per node and per step: ``stepClass`` (the class the
    node holds after that many hops, -1 while it holds nothing) and
    ``stepShare`` (its mass then, as a byte over its own final mass), both
    ``steps`` values per row.

    The rest is per node: ``arrivalStep`` (255 never reached), ``parent`` and
    ``parentShare`` (the edge that first delivered the mass and how much of
    that step's arriving mass came over it), ``source`` and ``hop`` (the
    labeled node at the root of that chain and its distance), and
    ``influence``/``influenceShare`` (the strongest contributors to the node's
    final class once the mass settled, ``influenceK`` per row). Every node
    index is a view row, all ones where there is none.
    """

    with np.load(propagation_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    rows = np.asarray([position_to_view.get(int(position), -1) for position in arrays["positions"]], dtype=np.int64)
    placed = rows >= 0
    steps = int(arrays["steps"]) + 1
    index_dtype = np.dtype("<u2") if count < 1 << 16 else np.dtype("<u4")
    empty = np.iinfo(index_dtype).max
    class_dtype = class_index_dtype(class_count)

    def in_view_order(values, fill, dtype=np.float64):
        shaped = np.asarray(values)
        target = np.full((count,) + shaped.shape[1:], fill, dtype=dtype)
        target[rows[placed]] = shaped[placed]
        return target

    def as_view_rows(node_indices):
        """Node indices as view rows, ``empty`` where the node is not in the view."""

        mapped = np.where(node_indices >= 0, rows[np.maximum(node_indices, 0)], -1)
        return np.where(mapped >= 0, mapped, empty)

    classes = np.vectorize(
        lambda label: class_index_of_label(int(label)) if label >= 0 else -1, otypes=[np.int64]
    )(arrays["step_top_classes"].astype(np.int64))
    encoded = {
        "steps": steps,
        "graphSource": str(arrays["graph_source"]) if "graph_source" in arrays else "label_propagation",
        # A backfill over a saved graph knows neither k nor gamma, and json
        # would happily write a bare NaN that no browser can parse back.
        "alpha": float_or_none(arrays["alpha"]),
        "gamma": float_or_none(arrays["gamma"]),
        "nNeighbors": int(arrays["n_neighbors"]),
        "classBytes": np.dtype(class_dtype).itemsize,
        "indexBytes": index_dtype.itemsize,
        "stepClass": pack(in_view_order(classes, -1), class_dtype),
        "stepShare": pack(quantize_byte(in_view_order(arrays["step_share"], 0.0)), "<u1"),
        # 255 is "never reached"; a run recording more than 254 steps would
        # collide with it, which PROPAGATION_STEPS is nowhere near.
        "arrivalStep": pack(np.clip(in_view_order(arrays["arrival_step"], 255), 0, 255), "<u1"),
        "parent": pack(in_view_order(as_view_rows(arrays["parent"].astype(np.int64)), empty, dtype=np.int64), index_dtype),
        "parentShare": pack(quantize_byte(in_view_order(arrays["parent_share"], 0.0)), "<u1"),
        "source": pack(in_view_order(as_view_rows(arrays["source"].astype(np.int64)), empty, dtype=np.int64), index_dtype),
        "hop": pack(np.clip(in_view_order(arrays["hop"], 255), 0, 255), "<u1"),
        "stepAccuracy": [float_or_none(value) for value in arrays["step_accuracy"]],
        "stepReached": [float_or_none(value) for value in arrays["step_reached"]],
        "summary": {name: float_or_none(arrays[name]) for name in sorted(arrays) if name.startswith("lp_")},
    }
    if "influence" in arrays:
        influence = arrays["influence"].astype(np.int64)
        encoded["influenceK"] = int(influence.shape[1])
        encoded["influence"] = pack(
            in_view_order(as_view_rows(influence), empty, dtype=np.int64), index_dtype
        )
        encoded["influenceShare"] = pack(quantize_byte(in_view_order(arrays["influence_share"], 0.0)), "<u1")
    return encoded


def load_pseudo_label_generations(run_dir):
    """The run's saved pseudo-label generations, ``[(epoch, npz arrays), ...]`` in order."""

    generations = []
    for path in sorted((Path(run_dir) / "pseudo_labels").glob("generation_*.npz")):
        with np.load(path) as saved:
            generations.append((int(saved["epoch"]), {name: saved[name] for name in saved.files}))
    return generations


def encode_pseudo_labels(arrays, samples, count, class_index_of_label, class_count):
    """One generation per view row: class (-1 not a candidate), confidence (uint16), accepted (0/1)."""

    row_of_position = {int(position): row for row, position in enumerate(samples)}
    labels = np.full(count, -1, dtype=np.int64)
    confidences = np.zeros(count, dtype=np.float64)
    accepted = np.zeros(count, dtype=np.uint8)
    for position, label, confidence in zip(arrays["positions"], arrays["labels"], arrays["confidences"]):
        row = row_of_position.get(int(position))
        if row is not None:
            labels[row] = class_index_of_label(int(label))
            confidences[row] = confidence
    for position in arrays["accepted_positions"]:
        row = row_of_position.get(int(position))
        if row is not None:
            accepted[row] = 1
    class_dtype = class_index_dtype(class_count)
    return {
        "generation": int(arrays["generation"]),
        "generationEpoch": int(arrays["epoch"]),
        "classBytes": np.dtype(class_dtype).itemsize,
        "label": pack(labels, class_dtype),
        "confidence": pack(quantize_unit(confidences), "<u2"),
        "accepted": pack(accepted, "<u1"),
    }


def tracking_view(tracking_dir):
    tracking_dir = Path(tracking_dir)
    epoch_dirs = sorted(
        (path for path in tracking_dir.glob("epoch_*") if path.is_dir()),
        key=lambda path: epoch_from_slug(path.name),
    )
    frames = []
    propagation_paths = {}
    for epoch_dir in epoch_dirs:
        layout_csv = next(iter(sorted(epoch_dir.glob("embeddings_*.csv"))), None)
        if layout_csv is None:
            continue
        with layout_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        prefix = layout_csv.stem.removeprefix("embeddings_")
        if prefix.endswith("_3d"):
            continue
        coordinates = np.asarray([[float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])] for row in rows])
        coordinates3 = None
        layout3_csv = epoch_dir / f"embeddings_{prefix}_3d.csv"
        if layout3_csv.is_file():
            with layout3_csv.open(newline="") as handle:
                rows3 = list(csv.DictReader(handle))
            if len(rows3) == len(rows):
                coordinates3 = np.asarray(
                    [[float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")] for row in rows3]
                )
        nodes = []
        if (epoch_dir / "graph_nodes.csv").is_file():
            with (epoch_dir / "graph_nodes.csv").open(newline="") as handle:
                nodes = list(csv.DictReader(handle))
        edges_path = epoch_dir / "graph_edges.npz"
        evidence_path = epoch_dir / "class_evidence.npz"
        propagation_path = epoch_dir / "label_propagation.npz"
        if propagation_path.is_file():
            propagation_paths[epoch_from_slug(epoch_dir.name)] = propagation_path
        frames.append(
            (
                epoch_from_slug(epoch_dir.name),
                prefix,
                rows,
                coordinates,
                nodes,
                coordinates3,
                edges_path if edges_path.is_file() else None,
                evidence_path if evidence_path.is_file() else None,
            )
        )
    if not frames:
        raise ValueError(f"{tracking_dir} holds no tracked embedding layouts")

    _, projection, base_rows, _, _, _, _, _ = frames[0]
    count = len(base_rows)
    for epoch, _, rows, _, nodes, _, _, _ in frames:
        if len(rows) != count or [row["label"] for row in rows] != [row["label"] for row in base_rows]:
            raise ValueError(f"Epoch {epoch} of {tracking_dir} tracks different samples than epoch {frames[0][0]}")
        if nodes and [node["label"] for node in nodes] != [row["label"] for row in rows]:
            raise ValueError(f"graph_nodes.csv of epoch {epoch} is not aligned with its embeddings")

    node_rows = next((frame[4] for frame in frames if frame[4]), [])
    if node_rows:
        subsets = [SUBSET_LABELED if node["kind"] == "labeled" else SUBSET_UNLABELED for node in node_rows]
        position_to_view = {int(node["position"]): index for index, node in enumerate(node_rows)}
        samples = [int(node["position"]) for node in node_rows]
    else:
        subsets = [SUBSET_TRAIN] * count
        position_to_view = {}
        samples = [int(row.get("sample_position") or index) for index, row in enumerate(base_rows)]
    names = [row.get("class_name") or f"class {row['label']}" for row in base_rows]

    def align_all(layouts_in):
        aligned = []
        previous = None
        for coordinates in layouts_in:
            current = coordinates if previous is None else procrustes_align(coordinates, previous)
            aligned.append(current - current.mean(axis=0))
            previous = aligned[-1]
        extent = max(float(np.abs(layout).max()) for layout in aligned) or 1.0
        return [np.round(layout / extent, 4).reshape(-1).tolist() for layout in aligned]

    epochs = [frame[0] for frame in frames]
    layouts = dict(zip((str(epoch) for epoch in epochs), align_all([frame[3] for frame in frames])))
    layouts3d = None
    if all(frame[5] is not None for frame in frames):
        layouts3d = dict(zip((str(epoch) for epoch in epochs), align_all([frame[5] for frame in frames])))
    node_metrics = {}
    for epoch, _, _, _, nodes, _, _, _ in frames:
        if not nodes:
            continue
        metrics = {}
        for name in TRACKING_NODE_METRICS:
            if name not in nodes[0]:
                continue
            values = []
            for node in nodes:
                raw = node[name]
                value = 1.0 if raw == "True" else 0.0 if raw == "False" else float_or_none(raw)
                values.append(None if value is None else round(value, 4))
            metrics[name] = values
        if "strongest_neighbor_position" in nodes[0]:
            metrics["strongest_neighbor"] = [
                position_to_view.get(int(float(node["strongest_neighbor_position"])), -1)
                if float_or_none(node["strongest_neighbor_position"]) is not None
                else -1
                for node in nodes
            ]
        node_metrics[str(epoch)] = metrics

    # The whole graph where the run saved it; older runs have only each node's
    # strongest neighbor, which is an arbitrary tie on a binary kNN graph.
    edges_by_epoch = {
        str(epoch): encode_graph_edges(edges_path, position_to_view, len(node_rows))
        for epoch, _, _, _, nodes, _, edges_path, _ in frames
        if nodes and edges_path is not None
    } or None

    epoch_metrics = []
    metrics_csv = tracking_dir / "epoch_metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                epoch_metrics.append({key: float_or_none(value) for key, value in row.items()})

    class_names, class_index = encode_classes(names)
    subset_names, subset_index = encode_subsets(subsets)
    run_dir = tracking_dir.parent
    args, ssl_config = read_run_metadata(run_dir)
    first_layout = layouts[str(epochs[0])]
    notes = []

    view_class_of_label = {int(row["label"]): int(index) for row, index in zip(base_rows, class_index)}

    def class_index_of_label(label):
        # A predicted class no tracked row carries still gets a name.
        if label not in view_class_of_label:
            view_class_of_label[label] = len(class_names)
            class_names.append(f"class {label}")
        return view_class_of_label[label]

    evidence_by_epoch = {
        str(epoch): encode_class_evidence(evidence_path, position_to_view, count, class_index_of_label, len(class_names))
        for epoch, _, _, _, nodes, _, _, evidence_path in frames
        if nodes and evidence_path is not None
    } or None

    # The same diffusion step by step, where the run (or a backfill) saved it.
    propagation_by_epoch = {
        str(epoch): encode_propagation(propagation_paths[epoch], position_to_view, count, class_index_of_label, len(class_names))
        for epoch in epochs
        if node_rows and epoch in propagation_paths
    } or None

    pseudo_labels_by_epoch = None
    generations = load_pseudo_label_generations(run_dir) if node_rows else []
    if generations:
        pseudo_labels_by_epoch = {}
        for epoch in epochs:
            active = [arrays for generation_epoch, arrays in generations if generation_epoch <= epoch]
            if active:
                pseudo_labels_by_epoch[str(epoch)] = encode_pseudo_labels(
                    active[-1], samples, count, class_index_of_label, len(class_names)
                )
        notes.append(
            f"Pseudo labels are the run's own ({len(generations)} generations saved); each epoch shows the "
            "generation it was trained on."
        )

    # Which saved SLADE refresh each tracked epoch was trained against.
    clusters_by_epoch = None
    cluster_k = None
    refresh_epochs = None
    saved_refreshes = load_saved_clusters(run_dir)
    if saved_refreshes and node_rows:
        clusters_by_epoch, refresh_epochs = {}, {}
        for epoch in epochs:
            active = [refresh for refresh in saved_refreshes if refresh[0] <= epoch]
            if not active:
                continue
            refresh_epoch, by_position, cluster_k = active[-1]
            clusters_by_epoch[str(epoch)] = [by_position.get(position, -1) for position in samples]
            refresh_epochs[str(epoch)] = refresh_epoch
        notes.append(
            f"Clusters are the run's own SLADE refreshes ({len(saved_refreshes)} saved); each epoch shows the "
            "refresh it was trained against. SLADE never reads the cluster centers: the IDs only decide which "
            "in-batch pairs fit its positive and negative similarity distributions."
        )

    center_similarity_by_epoch = None
    center_refreshes = load_saved_center_similarity(run_dir) if node_rows else []
    if center_refreshes:
        center_similarity_by_epoch = {}
        for epoch in epochs:
            active = [similarity for refresh_epoch, similarity in center_refreshes if refresh_epoch <= epoch]
            if active:
                values = np.asarray([active[-1].get(position, np.nan) for position in samples], dtype=np.float64)
                # A byte over [-1, 1]; 0 marks a row with no cluster.
                encoded = np.where(np.isfinite(values), np.round((np.clip(values, -1.0, 1.0) + 1.0) * 127.0) + 1, 0)
                center_similarity_by_epoch[str(epoch)] = pack(encoded, "<u1")

    mined_pairs_by_epoch = None
    pair_dir = run_dir / "slade_pairs"
    if node_rows and pair_dir.is_dir():
        mined_pairs_by_epoch = {
            str(epoch): encode_mined_pairs(pair_dir / f"epoch_{epoch:04d}.npz", samples)
            for epoch in epochs
            if epoch >= 0 and (pair_dir / f"epoch_{epoch:04d}.npz").is_file()
        } or None
        if mined_pairs_by_epoch:
            notes.append(
                "Mined pairs: a sample of the pairs SLADE's Eq 7 actually trained on in that epoch (up to 4,096 "
                "positives and 4,096 negatives), kept where both ends are among the tracked points."
            )
    return {
        "version": FORMAT_VERSION,
        "id": view_id(tracking_dir, "tracking"),
        "source": "epoch_tracking",
        "sourcePath": repository_relative(tracking_dir),
        "projection": projection,
        "selectedEpoch": None,
        **describe_run(run_dir, args, ssl_config),
        "subsets": subset_names,
        "classNames": class_names,
        "points": {"subset": subset_index, "class": class_index.tolist(), "sample": samples},
        "layouts": {"2d": first_layout, **({"3d": layouts3d[str(epochs[0])]} if layouts3d else {})},
        "tracking": {
            "epochs": epochs,
            "layouts": layouts,
            "layouts3d": layouts3d,
            "nodeMetrics": node_metrics,
            "epochMetrics": epoch_metrics,
            "clusters": clusters_by_epoch,
            "clusterK": cluster_k,
            "clusterRefreshEpochs": refresh_epochs,
            "edges": edges_by_epoch,
            "evidence": evidence_by_epoch,
            "propagation": propagation_by_epoch,
            "pseudoLabels": pseudo_labels_by_epoch or None,
            "centerSimilarity": center_similarity_by_epoch or None,
            "minedPairs": mined_pairs_by_epoch,
        },
        "notes": notes + [
            f"Epoch tracking: the same {count} pool samples at every tracked epoch, one {projection} fit per "
            "epoch, each aligned to the previous one (rotation, reflection, scale) so the frames are comparable.",
            "Graph metrics per node come from the run's own kNN graph at that epoch (graph_nodes.csv).",
            (
                f"Edges: every edge of that graph ({len(edges_by_epoch)} of {len(epochs)} epochs saved "
                "graph_edges.npz); selecting a point draws all of its neighbors."
                if edges_by_epoch
                else "Edges: only each node's strongest neighbor. This run predates graph_edges.npz, so its "
                "full graph was not saved; on a binary kNN graph the strongest neighbor is an arbitrary tie."
            ),
        ]
        + (
            [
                "Label propagation and Hoffer evidence (class_evidence.npz) are computed on every tracked epoch "
                "for any method: Iscen's label spreading from the labeled points, and Hoffer's reference softmax "
                "averaged over random reference draws. Hidden labels only score them."
            ]
            if evidence_by_epoch
            else []
        )
        + (
            [
                "Label propagation is also saved step by step (label_propagation.npz): the same diffusion as a "
                "series, F_0 = Y and F_t = alpha S F_{t-1} + Y, so each step holds only the label mass that has "
                "travelled that many edges. Per node it gives the step the mass arrived, the edge that delivered "
                "it, the labeled node at the root of that chain, and the strongest contributors to its class once "
                "the mass settled. "
                + (
                    "The graph is label propagation's own, so the last step is what class_evidence.npz holds."
                    if next(iter(propagation_by_epoch.values())).get("graphSource") == "label_propagation"
                    else "The graph is the run's own tracked graph: this was backfilled from graph_edges.npz "
                    "(scripts/backfill_label_propagation.py), whose k and weighting are the method's."
                )
            ]
            if propagation_by_epoch
            else []
        ),
    }


# ----------------------------------------------------------------------------
# Source 3: a test_tsne.csv
# ----------------------------------------------------------------------------


def projection_prefix(columns, csv_path):
    columns = list(columns)
    prefix = next(
        (column[:-2] for column in columns if column.endswith("_x") and f"{column[:-2]}_y" in columns),
        None,
    )
    if prefix is None:
        raise ValueError(f"{csv_path} has no <projection>_x/<projection>_y columns")
    return prefix


def tsne_csv_view(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} holds no points")
    prefix = projection_prefix(rows[0].keys(), csv_path)
    coordinates = np.asarray([[float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])] for row in rows])
    names = [row.get("class_name") or f"class {row.get('label', '?')}" for row in rows]
    class_names, class_index = encode_classes(names)
    # A concatenated-fold projection sits in the CV directory, which has no
    # run_config.json; its folds share the test set, so fold 0 describes it.
    fold_runs = sorted(path.parent for path in csv_path.parent.glob("fold_*/*/run_config.json"))
    concatenated_folds = len(fold_runs) if csv_path.stem.endswith("_concatenated") and fold_runs else None
    run_dir = fold_runs[0] if concatenated_folds else csv_path.parent
    args, ssl_config = read_run_metadata(run_dir)
    layouts = {"2d": normalize_layout(coordinates)}
    twin = csv_path.with_name(f"{csv_path.stem}_3d.csv")
    if twin.is_file():
        with twin.open(newline="") as handle:
            rows3 = list(csv.DictReader(handle))
        if len(rows3) == len(rows):
            layouts["3d"] = normalize_layout(
                np.asarray([[float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")] for row in rows3])
            )
    samples = [int(row.get("sample_position") or index) for index, row in enumerate(rows)]
    description = describe_run(run_dir, args, ssl_config)
    if concatenated_folds:
        description.update(
            label=f"{description['runLabel']} · {concatenated_folds} folds concatenated",
            fold=None,
            concatenatedFolds=concatenated_folds,
        )
    return {
        "version": FORMAT_VERSION,
        "id": view_id(csv_path, prefix),
        "source": "projection_csv",
        "sourcePath": repository_relative(csv_path),
        "projection": prefix,
        "selectedEpoch": None,
        **description,
        "subsets": [SUBSET_TEST],
        "classNames": class_names,
        "points": {
            "subset": [0] * len(rows),
            "class": class_index.tolist(),
            "sample": samples,
        },
        "layouts": layouts,
        "notes": ([
            f"Every fold's D_test embedding, each L2 normalized, joined per sample and normalized again "
            f"({concatenated_folds} folds): the embedding the concatenated test score is computed on."
        ] if concatenated_folds else []) + [
            f"From {csv_path.name}: D_test only"
            + (", with its 3-D twin." if "3d" in layouts else ", 2D only (no test_*_3d.csv next to it).")
            + " Labeled/unlabeled needs epoch tracking or a saved head."
        ],
    }


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------


def index_entry(view, file_name):
    subset_counts = {name: 0 for name in view["subsets"]}
    for code in view["points"]["subset"]:
        subset_counts[view["subsets"][code]] += 1
    return {
        "id": view["id"],
        "file": file_name,
        "label": view["label"],
        "source": view["source"],
        "projection": view["projection"],
        "dataset": view.get("dataset"),
        "method": view.get("method"),
        "loss": view.get("loss"),
        "seed": view.get("seed"),
        "fold": view.get("fold"),
        "concatenatedFolds": view.get("concatenatedFolds"),
        "labelBudget": view.get("labelBudget"),
        "runDir": view.get("runDir"),
        "sourcePath": view.get("sourcePath"),
        "pointCount": len(view["points"]["subset"]),
        "classCount": len(view["classNames"]),
        "subsetCounts": subset_counts,
        "has3d": "3d" in view["layouts"],
        "runLabel": view.get("runLabel") or view["label"],
        "epochs": (view.get("tracking") or {}).get("epochs"),
        "hasGraph": "graph" in view or bool((view.get("tracking") or {}).get("nodeMetrics")),
        "sourceFingerprint": view.get("sourceFingerprint"),
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_views(app_dir, views):
    view_dir = Path(app_dir) / "public" / VIEW_DIR_NAME
    view_dir.mkdir(parents=True, exist_ok=True)
    index_path = view_dir / INDEX_NAME
    existing = json.loads(index_path.read_text()).get("views", []) if index_path.is_file() else []
    entries = {entry["id"]: entry for entry in existing}
    for view in views:
        file_name = f"{view['id']}.json"
        (view_dir / file_name).write_text(json.dumps(view, separators=(",", ":")))
        entries[view["id"]] = index_entry(view, file_name)
    index = {
        "version": FORMAT_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "views": sorted(entries.values(), key=lambda entry: entry["label"]),
    }
    index_path.write_text(json.dumps(index, indent=2))
    return index_path


# ----------------------------------------------------------------------------
# Finding sources, and deciding what still needs exporting
# ----------------------------------------------------------------------------

SOURCE_HEAD = "head_state"
SOURCE_TRACKING = "epoch_tracking"
SOURCE_PROJECTION_CSV = "projection_csv"
# run_training writes these after everything else, test projections included,
# so a run outside cross-validation that has one of them got to the end.
RUN_END_MARKERS = ("frozen_feature_precompute_stats.json", "backbone_cache_stats.json")


def collect_sources(options):
    """``[(kind, path, searched)]`` in export order, without duplicates.

    ``searched`` marks a source a directory search found rather than one named
    directly; only those are held to the finished-run check.
    """

    sources = []

    def add(kind, path, searched):
        sources.append((kind, Path(path), searched))

    for run in options.run:
        add(SOURCE_HEAD, run, False)
    for csv_path in options.tsne_csv:
        add(SOURCE_PROJECTION_CSV, csv_path, False)
    for root in [*options.roots, *options.root]:
        for path in sorted(Path(root).rglob("head_state.pt")):
            add(SOURCE_HEAD, path.parent, True)
    for root in [*options.roots, *options.tracking_root]:
        for path in sorted(Path(root).rglob("epoch_tracking/epoch_metrics.csv")):
            add(SOURCE_TRACKING, path.parent, True)
    for root in [*options.roots, *options.tsne_csv_root]:
        for name in ("test_tsne.csv", "test_tsne_concatenated.csv"):
            for path in sorted(Path(root).rglob(name)):
                add(SOURCE_PROJECTION_CSV, path, True)

    unique = {}
    for kind, path, searched in sources:
        key = (kind, path.resolve())
        # A source named directly keeps that status even if a search finds it too.
        unique[key] = (kind, path, unique[key][2] and searched if key in unique else searched)
    return list(unique.values())


def cross_validation_dir(run_dir):
    """The ``cv_*`` directory of a fold run (``cv_*/fold_NN/<stamp>``), else ``None``."""

    run_dir = Path(run_dir)
    fold_dir = run_dir.parent
    if fold_dir.name.startswith("fold_") and fold_dir.parent.name.startswith("cv_"):
        return fold_dir.parent
    return None


def read_cv_summary(cv_dir):
    path = Path(cv_dir) / "cv_summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, dict) else None


def run_finished(run_dir):
    """``(finished, reason)`` for one training run directory."""

    run_dir = Path(run_dir)
    cv_dir = cross_validation_dir(run_dir)
    if cv_dir is not None:
        summary = read_cv_summary(cv_dir)
        if summary is None:
            return False, f"{cv_dir.name} has not finished any fold yet"
        # Compared by fold and timestamp directory rather than the whole path,
        # which the summary records relative to wherever training ran.
        completed = {
            tuple(Path(str(fold.get("log_dir") or "")).parts[-2:])
            for fold in summary.get("folds") or []
            if isinstance(fold, dict)
        }
        if tuple(run_dir.parts[-2:]) in completed:
            return True, None
        return False, (
            f"not a completed fold of {cv_dir.name} (still training, interrupted, or retrained "
            "in another directory)"
        )
    if any((run_dir / name).is_file() for name in RUN_END_MARKERS):
        return True, None
    return False, "no end-of-run stats file, so it is still training or was interrupted"


def source_finished(kind, path):
    if kind == SOURCE_TRACKING:
        return run_finished(Path(path).parent)
    if kind == SOURCE_PROJECTION_CSV and Path(path).stem.endswith("_concatenated"):
        summary = read_cv_summary(Path(path).parent)
        if summary is not None and summary.get("aggregate") is not None:
            return True, None
        return False, "its cross-validation run has not aggregated its folds yet"
    if kind == SOURCE_PROJECTION_CSV:
        return run_finished(Path(path).parent)
    return run_finished(path)


def source_view_id(kind, path, options):
    """The id the view will get, known without building it."""

    path = Path(path)
    if kind == SOURCE_HEAD:
        return view_id(path, options.projection)
    if kind == SOURCE_TRACKING:
        return view_id(path, "tracking")
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), [])
    return view_id(path, projection_prefix(header, path))


def _files_below(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return [child for child in path.rglob("*") if child.is_file()]
    return []


def source_files(kind, path):
    """Every file a view of this source is built from."""

    path = Path(path)
    if kind == SOURCE_HEAD:
        return [
            *_files_below(path / "head_state.pt"),
            *_files_below(path / "run_config.json"),
            *_files_below(path / "split"),
            *_files_below(path / "slade_clusters"),
        ]
    if kind == SOURCE_TRACKING:
        run_dir = path.parent
        return [
            *_files_below(path),
            *_files_below(run_dir / "run_config.json"),
            *_files_below(run_dir / "pseudo_labels"),
            *_files_below(run_dir / "slade_clusters"),
            *_files_below(run_dir / "slade_pairs"),
        ]
    files = [*_files_below(path), *_files_below(path.with_name(f"{path.stem}_3d.csv"))]
    if path.stem.endswith("_concatenated"):
        # The fold runs name the view and count its folds.
        files.extend(sorted(path.parent.glob("fold_*/*/run_config.json")))
    else:
        files.extend(_files_below(path.parent / "run_config.json"))
    return files


def source_fingerprint(kind, path, options):
    digest = hashlib.sha1(f"{FORMAT_VERSION}|{EXPORT_REVISION}|{kind}".encode())
    if kind == SOURCE_HEAD:
        # A head is embedded and projected again, so these shape the view too.
        settings = {
            name: getattr(options, name, None)
            for name in (
                "projection",
                "tsne_backend",
                "max_points",
                "no_3d",
                "no_validation",
                "no_test",
                "seed",
                "knn",
                "no_graph",
                "cluster_k",
            )
        }
        digest.update(json.dumps(settings, sort_keys=True).encode())
    for file in sorted({Path(file).resolve() for file in source_files(kind, path)}):
        stat = file.stat()
        digest.update(f"|{file}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def build_view(kind, path, options):
    if kind == SOURCE_HEAD:
        return head_view(path, options)
    if kind == SOURCE_TRACKING:
        return tracking_view(path)
    return tsne_csv_view(path)


def read_index_entries(app_dir):
    index_path = Path(app_dir) / "public" / VIEW_DIR_NAME / INDEX_NAME
    if not index_path.is_file():
        return {}
    return {entry["id"]: entry for entry in json.loads(index_path.read_text()).get("views", [])}


def view_is_current(entry, fingerprint, app_dir):
    return (
        entry is not None
        and entry.get("sourceFingerprint") == fingerprint
        and (Path(app_dir) / "public" / VIEW_DIR_NAME / entry["file"]).is_file()
    )


def describe_view(view):
    epochs = (view.get("tracking") or {}).get("epochs")
    detail = f"epochs {epochs[0]}..{epochs[-1]} ({len(epochs)})" if epochs else f"layouts {sorted(view['layouts'])}"
    return f"{view['label']}: {len(view['points']['subset'])} points, {detail}"


def main():
    options = parse_args()
    if options.tsne_backend is not None:
        import utils

        utils.set_tsne_backend(options.tsne_backend, device=options.device)
    sources = collect_sources(options)
    if not sources:
        raise SystemExit(
            "Nothing to export: pass directories to search, --run/--root for saved heads, --tracking-root "
            "for epoch tracking, or --tsne-csv/--tsne-csv-root"
        )

    entries = read_index_entries(options.app)
    counts = Counter()
    index_path = None
    for kind, path, searched in sources:
        shown = f"{kind} {repository_relative(path)}"
        if searched and not options.include_unfinished:
            finished, reason = source_finished(kind, path)
            if not finished:
                print(f"unfinished  {shown}: {reason}")
                counts["unfinished"] += 1
                continue
        try:
            identifier = source_view_id(kind, path, options)
            fingerprint = source_fingerprint(kind, path, options)
        except (OSError, ValueError) as exc:
            print(f"skipped     {shown}: {exc}")
            counts["skipped"] += 1
            continue
        if not options.force and view_is_current(entries.get(identifier), fingerprint, options.app):
            print(f"up to date  {shown}")
            counts["up to date"] += 1
            continue
        print(f"exporting   {shown} ...", flush=True)
        try:
            view = build_view(kind, path, options)
        except ValueError as exc:
            # A run can track test metrics only, with no embedding layouts to draw.
            print(f"skipped     {shown}: {exc}")
            counts["skipped"] += 1
            continue
        except Exception:  # noqa: BLE001 - one broken run must not stop the rest
            print(f"failed      {shown}:\n{traceback.format_exc()}")
            counts["failed"] += 1
            continue
        view["sourceFingerprint"] = fingerprint
        # Written one at a time, so an interrupted export keeps what it finished.
        index_path = write_views(options.app, [view])
        entries[view["id"]] = {
            "id": view["id"],
            "file": f"{view['id']}.json",
            "sourceFingerprint": fingerprint,
        }
        print(f"  {describe_view(view)}")
        counts["exported"] += 1

    summary = ", ".join(f"{counts[name]} {name}" for name in ("exported", "up to date", "unfinished", "skipped", "failed"))
    print(f"{len(sources)} source(s): {summary}")
    if index_path is not None:
        print(f"Index at {index_path}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
