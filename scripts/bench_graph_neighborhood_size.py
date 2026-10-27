"""Measure what raising ``n_neighbors`` actually changes for Iscen and LRML.

The phase-1 studies search ``n_neighbors`` up to 50 (Iscen: 3/5/10/20/50, LRML:
3/6/10/15/25/50) and a few older studies went to 75/100. Reading the objective
across those bins does not answer "does k above 50 do anything", because every
trial also redraws lr, gamma, alpha, the loss parameters and the batch sampler,
so a bin's score is mostly other people's noise. This benchmark answers the
question one level down, where k is the *only* thing that moves: it rebuilds the
real graphs on the real frozen features at a ladder of k and reports what each
method feeds its training signal.

What is measured
----------------
Both arms share the graph the method actually builds - ``make_mixed_label_affinity``
for Iscen, ``build_lrml_knn_graph`` for LRML - on cached DINOv2 features of the
dataset's training pool, split into labeled/unlabeled by the runs' own
``class_subset_k_shot`` rule.

``iscen``
    Runs ``iscen_label_spreading`` end to end and reports the pseudo-labels it
    produces: accuracy over propagated unlabeled rows, coverage at the
    confidence threshold, accuracy among covered rows, and the churn against a
    reference k. Two graph-side numbers explain the outcome. ``effective_neighbors``
    is ``exp(H)`` of each row's normalized ``max(cos, 0) ** gamma`` weights: the
    number of neighbors carrying real weight after the gamma power, which is
    what caps the useful k. ``weight_beyond_reference`` is the share of affinity
    mass sitting in ranks past the reference k - if that is ~0, a larger k is
    arithmetically inert no matter how the training run is tuned.

``lrml``
    Its adjacency is *binary*: the 50th neighbor pulls exactly as hard as the
    1st, and with ``reduction='mean'`` every sampled edge contributes equally.
    So the quantity that decides whether more neighbors help is edge purity -
    the fraction of edges joining two same-class nodes. ``marginal_purity`` is
    the purity of only the edges that k adds on top of the reference k, i.e. the
    quality of the signal being bought. ``mean_pull_scale`` is
    ``mean 1/sqrt(d_i d_j)`` over edges, the per-edge factor the normalized
    Laplacian applies, which says how much a larger k dilutes each edge.

Both arms report wall time, so the cost side of the trade is measured too.

Setup
-----
Frozen DINOv2 ViT-B/14 rows from ``data/<dataset>/backbone_cache`` - the same
matrix a ``--use_cache backbone_tuning=frozen`` run trains on. ``--head`` chooses
the geometry the graph sees: ``identity`` is the backbone itself, ``random`` is
one ``Linear + L2 normalize`` head at init (the graph of the first rebuild), and
``trained`` fits that head on the labeled rows first (the graph after warmup).
``--cv-fold`` reproduces the runs' class-disjoint ``stratified_group_kfold``
training fold, so the node and class counts match a real trial.

Examples::

    python scripts/bench_graph_neighborhood_size.py --datasets Cars196
    python scripts/bench_graph_neighborhood_size.py --datasets DeepFashionInShop --arms lrml
    python scripts/bench_graph_neighborhood_size.py --head trained --gammas 1 3 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.retrieval_model import build_projection_head  # noqa: E402
from scripts.bench_seraph_gradient_contribution import (  # noqa: E402
    load_feature_source,
    make_k_shot_split,
)
from training.ssl.algorithms import (  # noqa: E402
    build_lrml_knn_graph,
    iscen_label_spreading,
    make_mixed_label_affinity,
)
from training.ssl.config import UNLABELED_TARGET  # noqa: E402


# The k-shot budget each dataset's phase-1 runs use, from the study directory
# names (``class_subset_k_shot_label_1_k_<K>``). Cars196 labels 16 per class,
# DeepFashionInShop 2, CIFAR100 24.
DATASET_K_SHOT = {
    "Cars196": 16,
    "DeepFashionInShop": 2,
    "CIFAR100": 24,
    "CUB": 16,
}

# ``ssl_configs/ssl_methods/iscen.json`` values, with the gamma/alpha the phase-1
# space searches around them.
DEFAULT_GAMMA = 3.0
DEFAULT_ALPHA = 0.99
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_CG_RTOL = 1e-5
DEFAULT_CG_MAX_ITER = 2000
DEFAULT_LINEAR_SOLVER = "torch_block_cg"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cars196", "DeepFashionInShop"],
        help="Datasets whose backbone cache to sweep",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["iscen", "lrml"],
        default=["iscen", "lrml"],
    )
    parser.add_argument(
        "--neighbors",
        nargs="+",
        type=int,
        default=[3, 6, 10, 15, 25, 50, 75, 100, 150, 200, 400],
        help="k ladder to sweep",
    )
    parser.add_argument(
        "--reference-k",
        type=int,
        default=50,
        help="k that churn, marginal purity and beyond-rank weight are measured against",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[DEFAULT_GAMMA],
        help="Iscen affinity exponent; the phase-1 space searches int 1..10",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--cg-rtol", type=float, default=DEFAULT_CG_RTOL)
    parser.add_argument("--cg-max-iter", type=int, default=DEFAULT_CG_MAX_ITER)
    parser.add_argument("--linear-solver", default=DEFAULT_LINEAR_SOLVER)
    parser.add_argument(
        "--head",
        choices=["identity", "random", "trained"],
        default="identity",
        help="Geometry the graph is built on: backbone, head at init, or fitted head",
    )
    parser.add_argument("--feat-dim", type=int, default=768)
    parser.add_argument(
        "--head-steps",
        type=int,
        default=500,
        help="Optimizer steps for --head=trained",
    )
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=0,
        help="stratified_group_kfold fold to train on; -1 uses the whole split",
    )
    parser.add_argument("--cv-k", type=int, default=4)
    parser.add_argument("--data-split-seed", type=int, default=7)
    parser.add_argument("--support-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Cap the graph node count; omit to use the whole training fold",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "scripts/reports/bench_graph_neighborhood_size/results.json"),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Node pool
# --------------------------------------------------------------------------- #


def select_training_fold(labels, cv_fold, cv_k, seed):
    """Reproduce the runs' class-disjoint ``stratified_group_kfold`` train fold.

    ``utils.dataset_splits`` passes the labels as both ``y`` and ``groups``, so
    the splitter holds out whole classes. The graph a trial builds therefore
    spans ``(cv_k - 1) / cv_k`` of the classes, not all of them.
    """

    if cv_fold < 0:
        return np.arange(len(labels), dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=cv_k, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.arange(len(labels)), labels, groups=labels))
    train_local, _ = folds[cv_fold]
    return np.sort(np.asarray(train_local, dtype=np.int64))


def subsample_nodes(positions, labels, max_nodes, seed):
    """Keep a class-stratified subset when the full fold is too large to sweep."""

    if max_nodes is None or len(positions) <= max_nodes:
        return positions
    generator = np.random.default_rng(seed)
    keep_fraction = max_nodes / len(positions)
    kept = []
    for value in np.unique(labels[positions]):
        class_positions = positions[labels[positions] == value]
        keep = max(2, int(round(len(class_positions) * keep_fraction)))
        kept.append(generator.choice(class_positions, size=min(keep, len(class_positions)), replace=False))
    return np.sort(np.concatenate(kept))


# --------------------------------------------------------------------------- #
# Embedding geometry
# --------------------------------------------------------------------------- #


def apply_head(features, labels, labeled_mask, args, device):
    """Return the L2-normalized embeddings the graph is built on."""

    if args.head == "identity":
        return F.normalize(features.to(device).float(), p=2.0, dim=1)

    torch.manual_seed(args.seed)
    head = build_projection_head(features.shape[1], args.feat_dim).to(device)
    if args.head == "trained":
        _fit_head(head, features, labels, labeled_mask, args, device)
    with torch.no_grad():
        projected = head(features.to(device).float())
    return F.normalize(projected, p=2.0, dim=1)


def _fit_head(head, features, labels, labeled_mask, args, device):
    """Fit the projection head on the labeled rows with ProxyAnchorLoss.

    This stands in for the run's warmup: the graph a trial builds after
    ``warmup_epochs`` sees a head that has already been pulled toward the
    labeled classes, not the head at init.
    """

    from pytorch_metric_learning.losses import ProxyAnchorLoss

    labeled_positions = np.flatnonzero(labeled_mask)
    labeled_features = features[labeled_positions].to(device).float()
    labeled_labels = torch.as_tensor(labels[labeled_positions], dtype=torch.long, device=device)
    num_classes = int(labels.max()) + 1
    criterion = ProxyAnchorLoss(
        num_classes=num_classes,
        embedding_size=args.feat_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": head.parameters(), "lr": args.head_lr},
            {"params": criterion.parameters(), "lr": args.head_lr * 100},
        ]
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch_size = min(args.head_batch_size, len(labeled_positions))
    for step in range(args.head_steps):
        rows = torch.randperm(len(labeled_positions), generator=generator)[:batch_size]
        embeddings = F.normalize(head(labeled_features[rows]), p=2.0, dim=1)
        loss = criterion(embeddings, labeled_labels[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            logger.debug(f"head fit step {step}: loss={float(loss):.4f}")


# --------------------------------------------------------------------------- #
# Shared edge accounting
# --------------------------------------------------------------------------- #


def upper_triangle_edges(adjacency):
    """Return ``(rows, cols, values)`` of the strict upper triangle."""

    upper = sparse.triu(adjacency, k=1).tocoo()
    return upper.row.astype(np.int64), upper.col.astype(np.int64), upper.data


def edge_keys(rows, cols, num_nodes):
    """Encode undirected edges as sortable integers for set arithmetic."""

    return rows.astype(np.int64) * int(num_nodes) + cols.astype(np.int64)


def purity_stats(rows, cols, labels, values=None):
    """Fraction of edges - and of edge weight - joining two same-class nodes."""

    if len(rows) == 0:
        return {"edge_purity": float("nan"), "weight_purity": float("nan")}
    same_class = labels[rows] == labels[cols]
    stats = {"edge_purity": float(same_class.mean())}
    if values is None:
        stats["weight_purity"] = stats["edge_purity"]
    else:
        total = float(values.sum())
        stats["weight_purity"] = (
            float(values[same_class].sum() / total) if total > 0 else float("nan")
        )
    return stats


def retrieval_stats(neighbor_indices, labels):
    """Score the directed kNN lists the graph is built from.

    Undirected purity conflates two different failures. ``purity_ceiling`` is the
    highest purity any retriever could reach at this k, ``min(k, class_size - 1) / k``
    averaged over nodes: once k passes the class size, most neighbors *must* be
    cross-class and falling purity says nothing about the embedding.
    ``neighbor_recall`` is the complementary half - the share of each node's
    same-class pool that k reaches. When recall is already ~1 at the reference k,
    a larger k provably adds cross-class edges and nothing else.
    """

    k = int(neighbor_indices.shape[1])
    class_counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    hits = (labels[neighbor_indices] == labels[:, None]).sum(axis=1)
    reachable = np.minimum(class_counts[labels] - 1, k)
    has_peers = reachable > 0
    return {
        "directed_purity": float(hits.mean() / k),
        "purity_ceiling": float(reachable.mean() / k),
        "neighbor_recall": (
            float((hits[has_peers] / reachable[has_peers]).mean())
            if np.any(has_peers)
            else float("nan")
        ),
    }


def marginal_purity(rows, cols, labels, reference_keys, num_nodes):
    """Purity of the edges this k adds on top of the reference k.

    A larger k only helps if the edges it buys are worth having. The overall
    purity hides that: it is dominated by the edges both settings already share.
    """

    if reference_keys is None:
        return {"marginal_edges": None, "marginal_purity": None}
    keys = edge_keys(rows, cols, num_nodes)
    added = ~np.isin(keys, reference_keys, assume_unique=False)
    if not np.any(added):
        return {"marginal_edges": 0, "marginal_purity": None}
    same_class = labels[rows[added]] == labels[cols[added]]
    return {
        "marginal_edges": int(added.sum()),
        "marginal_purity": float(same_class.mean()),
    }


# --------------------------------------------------------------------------- #
# Iscen arm
# --------------------------------------------------------------------------- #


def effective_neighbor_count(similarities, gamma):
    """``exp(H)`` of each row's normalized affinity weights, averaged.

    The directed affinity of row ``i`` is ``max(cos, 0) ** gamma`` over its k
    retrieved neighbors. If that distribution concentrates on the first few
    ranks, the remaining neighbors are attached with weight near zero and a
    larger k cannot move the diffusion. This reports how many neighbors are
    effectively carrying the row.
    """

    weights = np.clip(np.asarray(similarities, dtype=np.float64), 0.0, None) ** float(gamma)
    totals = weights.sum(axis=1, keepdims=True)
    positive = totals[:, 0] > 0.0
    if not np.any(positive):
        return float("nan")
    normalized = weights[positive] / totals[positive]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(normalized > 0.0, np.log(normalized), 0.0)
    entropy = -(normalized * logs).sum(axis=1)
    return float(np.exp(entropy).mean())


def weight_beyond_rank(similarities, gamma, rank):
    """Share of a row's affinity mass sitting past ``rank``, averaged over rows."""

    if similarities.shape[1] <= rank:
        return 0.0
    weights = np.clip(np.asarray(similarities, dtype=np.float64), 0.0, None) ** float(gamma)
    totals = weights.sum(axis=1)
    positive = totals > 0.0
    if not np.any(positive):
        return float("nan")
    tail = weights[:, rank:].sum(axis=1)
    return float((tail[positive] / totals[positive]).mean())


def run_iscen(embeddings, labels, targets, args, gamma, reference):
    """Sweep k through the real LP-DeepSSL pipeline and score its pseudo-labels."""

    features = embeddings.detach().cpu().numpy().astype(np.float32)
    num_classes = int(labels.max()) + 1
    unlabeled_mask = targets == UNLABELED_TARGET
    rows_out = []
    reference_state = {}

    for k in args.neighbors:
        if k >= len(features):
            logger.warning(f"skipping k={k}: only {len(features)} nodes in the graph")
            continue

        graph_started_at = time.perf_counter()
        affinity, construction = make_mixed_label_affinity(
            features,
            n_neighbors=k,
            gamma=gamma,
            return_diagnostics=True,
        )
        graph_seconds = time.perf_counter() - graph_started_at

        rows, cols, values = upper_triangle_edges(affinity)
        keys = edge_keys(rows, cols, len(features))
        record = {
            "k": int(k),
            "gamma": float(gamma),
            "search_k": int(construction["search_n_neighbors"]),
            "undirected_edges": int(len(rows)),
            "mean_degree": float(2 * len(rows) / len(features)),
            "graph_seconds": float(graph_seconds),
            "effective_neighbors": effective_neighbor_count(
                construction["neighbor_similarities"], gamma
            ),
            "weight_beyond_reference": weight_beyond_rank(
                construction["neighbor_similarities"], gamma, args.reference_k
            ),
        }
        record.update(purity_stats(rows, cols, labels, values))
        record.update(retrieval_stats(construction["neighbor_indices"], labels))
        record.update(
            marginal_purity(rows, cols, labels, reference_state.get("keys"), len(features))
        )

        propagate_started_at = time.perf_counter()
        probabilities, confidences = iscen_label_spreading(
            features=features,
            targets=targets,
            num_classes=num_classes,
            n_neighbors=k,
            gamma=gamma,
            alpha=args.alpha,
            cg_rtol=args.cg_rtol,
            cg_max_iter=args.cg_max_iter,
            linear_solver=args.linear_solver,
        )
        record["propagate_seconds"] = float(time.perf_counter() - propagate_started_at)

        unlabeled_probabilities = probabilities[unlabeled_mask]
        propagated = unlabeled_probabilities.sum(axis=1) > 0.0
        truth = labels[unlabeled_mask]
        predictions = np.full(len(truth), -1, dtype=np.int64)
        predictions[propagated] = np.argmax(unlabeled_probabilities[propagated], axis=1)
        unlabeled_confidences = confidences[unlabeled_mask]

        record["propagated_fraction"] = float(propagated.mean())
        record["pseudo_accuracy"] = (
            float((predictions[propagated] == truth[propagated]).mean())
            if np.any(propagated)
            else float("nan")
        )
        covered = propagated & (unlabeled_confidences >= args.confidence_threshold)
        record["covered_fraction"] = float(covered.mean())
        record["covered_accuracy"] = (
            float((predictions[covered] == truth[covered]).mean())
            if np.any(covered)
            else float("nan")
        )
        # Accuracy alone hides how much training signal survives the threshold:
        # a k that keeps a third as many rows at the same accuracy is worse.
        record["covered_correct_per_node"] = (
            record["covered_fraction"] * record["covered_accuracy"]
            if np.any(covered)
            else 0.0
        )
        record["mean_confidence"] = float(np.mean(unlabeled_confidences))

        if reference_state.get("predictions") is not None:
            both = propagated & reference_state["propagated"]
            record["argmax_churn_vs_reference"] = (
                float((predictions[both] != reference_state["predictions"][both]).mean())
                if np.any(both)
                else float("nan")
            )
        else:
            record["argmax_churn_vs_reference"] = None

        if int(k) == int(args.reference_k):
            reference_state["keys"] = keys
            reference_state["predictions"] = predictions
            reference_state["propagated"] = propagated

        logger.info(f"iscen k={k} gamma={gamma}: {json.dumps(record)}")
        rows_out.append(record)
    return rows_out


# --------------------------------------------------------------------------- #
# LRML arm
# --------------------------------------------------------------------------- #


def run_lrml(embeddings, labels, args):
    """Sweep k through the real LRML graph and score the edges it feeds the loss."""

    features = embeddings.detach().cpu().numpy().astype(np.float32)
    rows_out = []
    reference_keys = None

    for k in args.neighbors:
        if k >= len(features):
            logger.warning(f"skipping k={k}: only {len(features)} nodes in the graph")
            continue

        started_at = time.perf_counter()
        neighbor_indices, adjacency, degrees = build_lrml_knn_graph(features, n_neighbors=k)
        graph_seconds = time.perf_counter() - started_at

        rows, cols, _ = upper_triangle_edges(adjacency)
        keys = edge_keys(rows, cols, len(features))
        # The normalized Laplacian scales each node by 1/sqrt(deg), so an edge's
        # contribution to the loss carries this factor. It shows how much a
        # larger k dilutes every individual edge.
        pull_scale = 1.0 / np.sqrt(np.maximum(degrees[rows], 1.0) * np.maximum(degrees[cols], 1.0))

        record = {
            "k": int(k),
            "search_k": int(neighbor_indices.shape[1]),
            "undirected_edges": int(len(rows)),
            "mean_degree": float(degrees.mean()),
            "min_degree": float(degrees.min()),
            "max_degree": float(degrees.max()),
            "mean_pull_scale": float(pull_scale.mean()),
            "graph_seconds": float(graph_seconds),
        }
        record.update(purity_stats(rows, cols, labels))
        record.update(retrieval_stats(neighbor_indices, labels))
        record.update(marginal_purity(rows, cols, labels, reference_keys, len(features)))
        # With reduction='mean' the loss is an average over uniformly sampled
        # edges, so purity is exactly the fraction of the gradient that pulls
        # same-class points together. Same-class edges reached per node is the
        # absolute amount of correct signal, which can still grow while the
        # fraction falls.
        record["same_class_edges_per_node"] = (
            record["edge_purity"] * 2 * len(rows) / len(features)
        )

        if int(k) == int(args.reference_k):
            reference_keys = keys

        logger.info(f"lrml k={k}: {json.dumps(record)}")
        rows_out.append(record)
    return rows_out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


ISCEN_COLUMNS = [
    ("k", "k", "{:>5}"),
    ("edges", "undirected_edges", "{:>10}"),
    ("eff_nbrs", "effective_neighbors", "{:>8.1f}"),
    ("w>ref", "weight_beyond_reference", "{:>7.4f}"),
    ("purity", "edge_purity", "{:>7.3f}"),
    ("ceiling", "purity_ceiling", "{:>7.3f}"),
    ("recall", "neighbor_recall", "{:>6.3f}"),
    ("w_purity", "weight_purity", "{:>8.3f}"),
    ("marg_pur", "marginal_purity", "{:>8.3f}"),
    ("prop", "propagated_fraction", "{:>6.3f}"),
    ("acc", "pseudo_accuracy", "{:>6.3f}"),
    ("cover", "covered_fraction", "{:>6.3f}"),
    ("cov_acc", "covered_accuracy", "{:>7.3f}"),
    ("corr/node", "covered_correct_per_node", "{:>9.3f}"),
    ("churn", "argmax_churn_vs_reference", "{:>6.3f}"),
    ("sec", "propagate_seconds", "{:>7.1f}"),
]

LRML_COLUMNS = [
    ("k", "k", "{:>5}"),
    ("edges", "undirected_edges", "{:>10}"),
    ("mean_deg", "mean_degree", "{:>8.1f}"),
    ("purity", "edge_purity", "{:>7.3f}"),
    ("ceiling", "purity_ceiling", "{:>7.3f}"),
    ("recall", "neighbor_recall", "{:>6.3f}"),
    ("marg_pur", "marginal_purity", "{:>8.3f}"),
    ("marg_edges", "marginal_edges", "{:>10}"),
    ("same/node", "same_class_edges_per_node", "{:>9.1f}"),
    ("pull", "mean_pull_scale", "{:>7.4f}"),
    ("sec", "graph_seconds", "{:>7.1f}"),
]


def print_table(title, columns, records):
    print(f"\n{title}")
    header = "  ".join(
        f"{name:>{max(len(name), _column_width(fmt))}}" for name, _, fmt in columns
    )
    print(header)
    print("-" * len(header))
    for record in records:
        cells = []
        for name, key, fmt in columns:
            value = record.get(key)
            width = max(len(name), _column_width(fmt))
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append(f"{'-':>{width}}")
            else:
                cells.append(f"{fmt.format(value):>{width}}")
        print("  ".join(cells))


def _column_width(fmt):
    """Read the field width out of a format spec like ``{:>8.3f}``."""

    spec = fmt.strip("{}").split(":")[-1].lstrip("<>^")
    digits = spec.split(".")[0]
    return int(digits) if digits.isdigit() else 8


def main():
    args = parse_args()
    # Churn, marginal purity and beyond-rank weight are all measured against the
    # reference k, which is captured as the ladder passes it.
    args.neighbors = sorted(set(int(k) for k in args.neighbors))
    if int(args.reference_k) not in args.neighbors:
        raise SystemExit(
            f"--reference-k {args.reference_k} must be one of --neighbors {args.neighbors}"
        )
    device = torch.device(args.device)
    results = {"config": vars(args), "datasets": {}}

    for dataset_name in args.datasets:
        source = load_feature_source(dataset_name, args.split, args.data_root, "cpu")
        labels_all = source.labels.numpy().astype(np.int64)
        fold_positions = select_training_fold(
            labels_all, args.cv_fold, args.cv_k, args.data_split_seed
        )
        fold_positions = subsample_nodes(fold_positions, labels_all, args.max_nodes, args.seed)
        features = source.features[torch.as_tensor(fold_positions)]
        labels = labels_all[fold_positions]

        k_shot = DATASET_K_SHOT.get(dataset_name)
        if k_shot is None:
            raise SystemExit(f"No k-shot budget recorded for {dataset_name}; add one to DATASET_K_SHOT")
        labeled_local, _ = make_k_shot_split(
            torch.as_tensor(labels), k_shot, args.support_seed
        )
        labeled_mask = np.zeros(len(labels), dtype=bool)
        labeled_mask[labeled_local] = True
        targets = np.where(labeled_mask, labels, UNLABELED_TARGET).astype(np.int64)

        embeddings = apply_head(features, labels, labeled_mask, args, device)
        summary = {
            "cache_path": source.cache_path,
            "nn_label_agreement": source.nn_label_agreement,
            "nodes": int(len(labels)),
            "classes": int(len(np.unique(labels))),
            # A node has at most class_size - 1 same-class neighbors to find, so
            # this is the scale at which k stops buying same-class edges.
            "median_class_size": float(np.median(np.bincount(labels)[np.bincount(labels) > 0])),
            "labeled": int(labeled_mask.sum()),
            "unlabeled": int((~labeled_mask).sum()),
            "k_shot": int(k_shot),
            "head": args.head,
        }
        logger.info(f"{dataset_name}: {json.dumps(summary)}")
        print(
            f"\n=== {dataset_name} | {summary['nodes']} nodes, {summary['classes']} classes, "
            f"{summary['labeled']} labeled ({k_shot}-shot), head={args.head} ==="
        )

        dataset_results = {"summary": summary, "arms": {}}
        if "lrml" in args.arms:
            records = run_lrml(embeddings, labels, args)
            dataset_results["arms"]["lrml"] = records
            print_table(f"LRML binary kNN graph - {dataset_name}", LRML_COLUMNS, records)
        if "iscen" in args.arms:
            dataset_results["arms"]["iscen"] = {}
            for gamma in args.gammas:
                records = run_iscen(embeddings, labels, targets, args, gamma, args.reference_k)
                dataset_results["arms"]["iscen"][str(gamma)] = records
                print_table(
                    f"Iscen label spreading (gamma={gamma}, alpha={args.alpha}, "
                    f"threshold={args.confidence_threshold}) - {dataset_name}",
                    ISCEN_COLUMNS,
                    records,
                )
        results["datasets"][dataset_name] = dataset_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
