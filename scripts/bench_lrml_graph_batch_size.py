"""Measure what ``graph_batch_size`` buys the LRML regularizer.

``graph_batch_size`` is a number of *graph edges* per step, not examples.
``GraphEdgeBatchSampler`` permutes the undirected edge list and hands out
contiguous chunks of that size, and ``_reduce_sampled_graph_energy`` divides the
sampled energy by the number of sampled edges under ``reduction='mean'``. So the
expected loss - and the expected gradient - is the same at every batch size. The
knob does not change what LRML optimizes at all. It changes three other things,
and those are what this measures:

``gradient agreement``
    The cosine between the sampled-edge gradient over the model parameters and
    the exact whole-graph Laplacian gradient, plus the relative error of the
    sampled gradient. This is the quantity that saturates: past the batch size
    where agreement flattens, extra edges are bought at full cost for no extra
    signal.
``graph coverage per epoch``
    The sampler runs a fixed ``num_batches=len(supervised_loader)`` steps per
    epoch, so edges seen per epoch is ``graph_batch_size * steps``, against the
    graph's total edge count. At the bottom of the searched ladder this is a
    tiny fraction of the graph, which is a different failure from noisy
    gradients: most edges are simply never visited.
``cost``
    Endpoints are deduplicated into at most ``2 * graph_batch_size`` nodes, and
    every one of them costs a forward pass. The measured dedup ratio says what a
    given batch size actually costs per step.

The whole-graph reference uses the closed form of the same objective. For a
binary symmetric ``W`` with zero diagonal and degree-scaled ``s_i = z_i / sqrt(d_i)``,

    sum_{i<j} W_ij ||s_i - s_j||^2  =  sum_i d_i ||s_i||^2 - tr(S^T W S)

which needs one sparse matmul instead of materializing millions of edge
differences. ``--verify-closed-form`` checks it against the explicit edge sum.

Setup mirrors ``bench_graph_neighborhood_size.py``: cached DINOv2 rows, the runs'
``class_subset_k_shot`` split, the class-disjoint CV training fold, and a
``Linear + L2 normalize`` head. Defaults per dataset come from the best trial of
that dataset's tuned LRML study, so the sweep brackets a real operating point.

Examples::

    python scripts/bench_lrml_graph_batch_size.py --datasets Cars196
    python scripts/bench_lrml_graph_batch_size.py --head trained --replicates 60
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.retrieval_model import build_projection_head  # noqa: E402
from scripts.bench_graph_neighborhood_size import (  # noqa: E402
    DATASET_K_SHOT,
    select_training_fold,
    subsample_nodes,
)
from scripts.bench_seraph_gradient_contribution import (  # noqa: E402
    load_feature_source,
    make_k_shot_split,
)
from training.ssl.algorithms import build_lrml_knn_graph  # noqa: E402
from training.ssl.data import GraphEdgeBatchSampler  # noqa: E402


# Best trial of each dataset's tuned LRML study. In-Shop comes from
# ``ssl/in_shop/lrml_prx/.../hpo/cars_imb_search`` (objective 0.6635); Cars196 has
# no completed LRML study, so it takes the phase-1 config's own defaults with the
# batch sampler the space centers on.
DATASET_OPERATING_POINT = {
    "Cars196": {
        "n_neighbors": 6,
        "batch_size": 128,
        "length_before_new_iter": 6041,
        "graph_batch_size": 256,
    },
    "DeepFashionInShop": {
        "n_neighbors": 3,
        "batch_size": 64,
        "length_before_new_iter": 20706,
        "graph_batch_size": 256,
    },
}

# The ladder ``configs/hpo/ssl_methods/phase_1/lrml/*/*.json`` searches, extended
# two rungs past its top so the saturation point is bracketed rather than assumed.
DEFAULT_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cars196", "DeepFashionInShop"],
    )
    parser.add_argument("--graph-batch-sizes", nargs="+", type=int, default=DEFAULT_LADDER)
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Graph k; defaults to the dataset's tuned value",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=40,
        help="Sampled batches per size; the spread across them is the measurement",
    )
    parser.add_argument(
        "--head",
        choices=["identity", "random", "trained"],
        default="trained",
        help="Geometry the graph and the gradient are taken on",
    )
    parser.add_argument("--feat-dim", type=int, default=768)
    parser.add_argument("--head-steps", type=int, default=500)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--normalized-laplacian", action="store_true", default=True)
    parser.add_argument(
        "--unnormalized-laplacian",
        dest="normalized_laplacian",
        action="store_false",
    )
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--cv-k", type=int, default=4)
    parser.add_argument("--data-split-seed", type=int, default=7)
    parser.add_argument("--support-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--verify-closed-form",
        action="store_true",
        help="Check the closed-form whole-graph energy against the explicit edge sum",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "scripts/reports/bench_graph_neighborhood_size/graph_batch_size.json"
        ),
    )
    return parser.parse_args()


def build_head(features, labels, labeled_mask, args, device):
    """Return the trainable head whose parameters the gradients are taken over."""

    torch.manual_seed(args.seed)
    if args.head == "identity":
        head = torch.nn.Linear(features.shape[1], features.shape[1], bias=False).to(device)
        with torch.no_grad():
            head.weight.copy_(torch.eye(features.shape[1], device=device))
        return head
    head = build_projection_head(features.shape[1], args.feat_dim).to(device)
    if args.head == "trained":
        fit_head(head, features, labels, labeled_mask, args, device)
    return head


def fit_head(head, features, labels, labeled_mask, args, device):
    """Pull the head toward the labeled classes, as warmup does before the graph."""

    from pytorch_metric_learning.losses import ProxyAnchorLoss

    labeled_positions = np.flatnonzero(labeled_mask)
    labeled_features = features[labeled_positions].to(device).float()
    labeled_labels = torch.as_tensor(labels[labeled_positions], dtype=torch.long, device=device)
    criterion = ProxyAnchorLoss(
        num_classes=int(labels.max()) + 1,
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
    for _ in range(args.head_steps):
        rows = torch.randperm(len(labeled_positions), generator=generator)[:batch_size]
        embeddings = F.normalize(head(labeled_features[rows]), p=2.0, dim=1)
        loss = criterion(embeddings, labeled_labels[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def embed(head, features):
    return F.normalize(head(features), p=2.0, dim=1)


def whole_graph_energy(scaled, sparse_adjacency, degrees):
    """Closed-form ``sum_{i<j} W_ij ||s_i - s_j||^2`` over the whole graph."""

    weighted = torch.sparse.mm(sparse_adjacency, scaled)
    cross = (scaled * weighted).sum()
    self_term = (degrees[:, None] * scaled * scaled).sum()
    return self_term - cross


def explicit_edge_energy(scaled, edge_rows, edge_cols):
    """The same quantity as an explicit sum, for verifying the closed form."""

    differences = scaled[edge_rows] - scaled[edge_cols]
    return (differences * differences).sum()


def flat_gradient(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=False)
    return torch.cat([gradient.reshape(-1) for gradient in grads])


def batch_gradient(head, features, node_scale, node_ids, edge_pairs, parameters):
    """Gradient of one sampled edge batch, computed the way the trainer does.

    Only the batch's own nodes are embedded, and the per-node ``1/sqrt(deg)``
    factor is folded into the embeddings before the pairwise differences - this
    is ``LRMLRegularizer.compute_loss`` with ``reduction='mean'``.
    """

    embeddings = embed(head, features[node_ids])
    scaled = embeddings * node_scale[node_ids][:, None]
    left, right = edge_pairs
    differences = scaled[left] - scaled[right]
    loss = (differences * differences).sum() / len(left)
    return flat_gradient(loss, parameters), float(loss.detach())


def sample_batches(sampler, replicates):
    """Draw ``replicates`` batches from the real sampler, as the loader would."""

    batches = []
    iterator = iter(sampler)
    for _ in range(replicates):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(sampler)
            batch = next(iterator)
        node_ids = [item.node_id for item in batch]
        edge_indices = batch[0].edge_indices
        batches.append((node_ids, edge_indices))
    return batches


def run_dataset(dataset_name, args, device):
    operating_point = DATASET_OPERATING_POINT.get(dataset_name, {})
    n_neighbors = args.n_neighbors or operating_point.get("n_neighbors", 6)
    batch_size = operating_point.get("batch_size", 128)
    length_before_new_iter = operating_point.get("length_before_new_iter")
    steps_per_epoch = (
        None
        if length_before_new_iter is None
        else max(1, int(length_before_new_iter) // int(batch_size))
    )

    source = load_feature_source(dataset_name, args.split, args.data_root, "cpu")
    labels_all = source.labels.numpy().astype(np.int64)
    fold_positions = select_training_fold(labels_all, args.cv_fold, args.cv_k, args.data_split_seed)
    fold_positions = subsample_nodes(fold_positions, labels_all, args.max_nodes, args.seed)
    labels = labels_all[fold_positions]
    features = source.features[torch.as_tensor(fold_positions)].to(device).float()

    labeled_local, _ = make_k_shot_split(
        torch.as_tensor(labels), DATASET_K_SHOT[dataset_name], args.support_seed
    )
    labeled_mask = np.zeros(len(labels), dtype=bool)
    labeled_mask[labeled_local] = True

    head = build_head(source.features[torch.as_tensor(fold_positions)], labels, labeled_mask, args, device)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]

    with torch.no_grad():
        graph_embeddings = embed(head, features).cpu().numpy().astype(np.float32)
    _, adjacency, degrees = build_lrml_knn_graph(graph_embeddings, n_neighbors=n_neighbors)

    scale = (
        1.0 / np.sqrt(np.maximum(degrees, 1.0))
        if args.normalized_laplacian
        else np.ones(len(degrees), dtype=np.float64)
    )
    node_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    degree_tensor = torch.as_tensor(degrees, dtype=torch.float32, device=device)

    upper = adjacency.tocoo()
    sparse_adjacency = torch.sparse_coo_tensor(
        np.vstack([upper.row, upper.col]),
        torch.ones(len(upper.data), dtype=torch.float32, device=device),
        size=adjacency.shape,
        device=device,
    ).coalesce()
    num_edges = int(adjacency.nnz // 2)

    scaled = embed(head, features) * node_scale[:, None]
    full_energy = whole_graph_energy(scaled, sparse_adjacency, degree_tensor)
    if args.verify_closed_form:
        from scipy import sparse as scipy_sparse

        triangle = scipy_sparse.triu(adjacency, k=1).tocoo()
        explicit = explicit_edge_energy(
            scaled,
            torch.as_tensor(triangle.row.astype(np.int64), device=device),
            torch.as_tensor(triangle.col.astype(np.int64), device=device),
        )
        torch.testing.assert_close(full_energy, explicit, rtol=1e-4, atol=1e-3)
        logger.info("closed-form whole-graph energy matches the explicit edge sum")
    full_loss = full_energy / num_edges
    full_gradient = flat_gradient(full_loss, parameters)
    full_norm = float(torch.linalg.vector_norm(full_gradient))

    summary = {
        "nodes": int(len(labels)),
        "classes": int(len(np.unique(labels))),
        "n_neighbors": int(n_neighbors),
        "undirected_edges": num_edges,
        "mean_degree": float(degrees.mean()),
        "batch_size": int(batch_size),
        "steps_per_epoch": steps_per_epoch,
        "head": args.head,
        "whole_graph_loss": float(full_loss.detach()),
        "whole_graph_gradient_norm": full_norm,
    }
    logger.info(f"{dataset_name}: {json.dumps(summary)}")
    print(
        f"\n=== {dataset_name} | {summary['nodes']} nodes, k={n_neighbors}, "
        f"{num_edges} undirected edges, {steps_per_epoch} steps/epoch, head={args.head} ==="
    )

    records = []
    for graph_batch_size in args.graph_batch_sizes:
        if graph_batch_size > num_edges:
            logger.warning(
                f"skipping graph_batch_size={graph_batch_size}: graph has {num_edges} edges"
            )
            continue
        sampler = GraphEdgeBatchSampler(
            adjacency=adjacency,
            graph_batch_size=graph_batch_size,
            seed=args.seed,
            num_batches=args.replicates,
        )
        batches = sample_batches(sampler, args.replicates)

        cosines = []
        relative_errors = []
        losses = []
        unique_nodes = []
        started_at = time.perf_counter()
        for node_ids, edge_indices in batches:
            node_index = torch.as_tensor(node_ids, dtype=torch.long, device=device)
            pairs = np.asarray(edge_indices, dtype=np.int64)
            edge_pairs = (
                torch.as_tensor(pairs[:, 0], device=device),
                torch.as_tensor(pairs[:, 1], device=device),
            )
            gradient, loss_value = batch_gradient(
                head, features, node_scale, node_index, edge_pairs, parameters
            )
            cosines.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        gradient, full_gradient, dim=0
                    )
                )
            )
            relative_errors.append(
                float(torch.linalg.vector_norm(gradient - full_gradient)) / full_norm
            )
            losses.append(loss_value)
            unique_nodes.append(len(node_ids))
        seconds = time.perf_counter() - started_at

        edges_per_epoch = (
            None if steps_per_epoch is None else int(graph_batch_size) * steps_per_epoch
        )
        record = {
            "graph_batch_size": int(graph_batch_size),
            "mean_gradient_cosine": float(np.mean(cosines)),
            "min_gradient_cosine": float(np.min(cosines)),
            "mean_relative_error": float(np.mean(relative_errors)),
            "loss_mean": float(np.mean(losses)),
            # With reduction='mean' the estimator is unbiased, so this dispersion
            # is the whole effect of the knob on the objective.
            "loss_cv": float(np.std(losses) / np.mean(losses)) if np.mean(losses) else float("nan"),
            "unique_nodes_per_batch": float(np.mean(unique_nodes)),
            "dedup_ratio": float(np.mean(unique_nodes) / (2 * graph_batch_size)),
            "edges_per_epoch": edges_per_epoch,
            "graph_passes_per_epoch": (
                None if edges_per_epoch is None else edges_per_epoch / num_edges
            ),
            "seconds_per_step": seconds / len(batches),
        }
        logger.info(f"graph_batch_size={graph_batch_size}: {json.dumps(record)}")
        records.append(record)

    return {"summary": summary, "records": records}


COLUMNS = [
    ("gbs", "graph_batch_size", "{:>6}"),
    ("cos(g,g_full)", "mean_gradient_cosine", "{:>13.4f}"),
    ("min cos", "min_gradient_cosine", "{:>8.4f}"),
    ("rel_err", "mean_relative_error", "{:>8.3f}"),
    ("loss", "loss_mean", "{:>8.4f}"),
    ("loss_cv", "loss_cv", "{:>8.4f}"),
    ("nodes/step", "unique_nodes_per_batch", "{:>10.0f}"),
    ("dedup", "dedup_ratio", "{:>6.2f}"),
    ("edges/epoch", "edges_per_epoch", "{:>11}"),
    ("graph/epoch", "graph_passes_per_epoch", "{:>11.3f}"),
    ("sec/step", "seconds_per_step", "{:>8.4f}"),
]


def print_table(title, records):
    from scripts.bench_graph_neighborhood_size import _column_width

    print(f"\n{title}")
    header = "  ".join(
        f"{name:>{max(len(name), _column_width(fmt))}}" for name, _, fmt in COLUMNS
    )
    print(header)
    print("-" * len(header))
    for record in records:
        cells = []
        for name, key, fmt in COLUMNS:
            value = record.get(key)
            width = max(len(name), _column_width(fmt))
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append(f"{'-':>{width}}")
            else:
                cells.append(f"{fmt.format(value):>{width}}")
        print("  ".join(cells))


def main():
    args = parse_args()
    args.graph_batch_sizes = sorted(set(int(value) for value in args.graph_batch_sizes))
    device = torch.device(args.device)
    results = {"config": vars(args), "datasets": {}}
    for dataset_name in args.datasets:
        block = run_dataset(dataset_name, args, device)
        results["datasets"][dataset_name] = block
        print_table(f"LRML edge-batch gradient quality - {dataset_name}", block["records"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
