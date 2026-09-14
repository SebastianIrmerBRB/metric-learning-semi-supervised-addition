"""Label apportioning and pseudo-label generation for SSL training.

The module first divides the current training subset into labeled and unlabeled
positions.  A configured SSL method embeds both groups and predicts mapped
training labels for some or all unlabeled positions.  The accepted
pseudo-labels are then combined with the true labeled samples in a
``RelabeledSubset`` that can be consumed by the normal metric-learning loader.

Positions in this file always refer to offsets inside the current training
dataset, not indices in the original source dataset.
"""

import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from scipy import sparse

import numpy as np
import torch
from loguru import logger

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from losses import metric_losses
from models.retrieval_model import BACKBONE_TUNING_FROZEN, normalize_backbone_tuning

from .ssl import interfaces
from .ssl import gradient_surgery as grad_surgery
from .ssl.algorithms import (
    block_cg as block_cg,
    build_manifold_preserving_graph,
    build_lrml_knn_graph,
    entropy_confidence as entropy_confidence,
    faiss_flat_ip_search,
    faiss_label_spreading as _faiss_label_spreading,
    iscen_label_spreading as _iscen_label_spreading,
    majority_vote,
    make_dissimilarity_affinity as make_dissimilarity_affinity,
    make_mixed_label_affinity as make_mixed_label_affinity,
    mixed_label_propagation as _mixed_label_propagation,
    normalize_linear_solver,
    normalize_mixed_label_rows as normalize_mixed_label_rows,
    require_faiss,
    select_self_training_candidates,
    solve_sparse_label_system as solve_sparse_label_system,
    solve_sparse_label_system_cholmod as solve_sparse_label_system_cholmod,
    suppress_ssl_timing_logs,
)
from .ssl.config import (
    CLASS_OVERLAP_MODES,
    DEFAULT_SUPPORT_SEED,
    GRAPH_BATCH_MODES,
    IN_BATCH_GRAPH_MODES,
    GRAPH_DIAGNOSTICS_CLASS_FOCUS_MODES,
    GRAPH_DIAGNOSTICS_LAYOUTS,
    GRAPH_DIAGNOSTICS_MODES,
    LABEL_SAMPLING_MODES,
    LOSS_DRIVEN_METHODS,
    PSEUDO_LABEL_DIAGNOSTICS_MODES,
    TWO_STREAM_SAMPLER_METHODS,
    UNLABELED_CLASS_SCOPE_LABELED_CLASSES,
    UNLABELED_CLASS_SCOPES,
    UNLABELED_TARGET,
    UPDATE_MODES,
    WARMUP_CHECKPOINT_MODES,
    GraphDiagnosticsRequest as GraphDiagnosticsRequest,
    PseudoLabelResult,
    SemiSupervisedConfig,
    SemiSupervisedSplit as SemiSupervisedSplit,
    sample_scoped_refresh_interval_steps as sample_scoped_refresh_interval_steps,
    should_rebuild_on_epoch,
)
from .ssl.data import (
    CombinedTrainingLoader,
    GraphEdgeBatchSampler,
    HofferReferenceBatchSampler,
    HofferReferenceDataset,
    LRMLGraphDataset,
    UnlabeledSubset,
    collate_graph_edge_batch,
    graph_upper_triangle_edges,
)
from .ssl.embeddings import (
    extract_embeddings,
    make_feature_dataset,
    make_embedding_loader,
    ssl_compute_device,
)
from .ssl.graph_diagnostics import (
    dataset_labels_for_positions,
    make_graph_diagnostics_request,
    maybe_save_graph_diagnostics,
    maybe_update_graph_propagation_diagnostics,
    project_graph_embeddings_2d as project_graph_embeddings_2d,
    save_graph_diagnostics as save_graph_diagnostics,
)
from .ssl.interfaces import BaseSemiSupervisedMethod, BaseTrainingRegularizer
from .ssl.ismlp import IsmlpRegularizer
from .ssl.seraph import SeraphRegularizer
from .ssl.simmatch_v2 import SimMatchV2Regularizer
from .ssl.slade import SladeRegularizer
from .ssl.stml_threshold import STMLThresholdInBatchMethod, STMLThresholdPseudoLabeler
from .ssl.pseudo_labels import (
    PseudoLabelDiagnosticsTracker,
    filter_pseudo_labels,
    make_relabeled_training_dataset,
    summarize_numeric_values,
)
from .ssl.sampling import (
    make_semi_supervised_split,
)

def _sync_timing_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_now(device):
    _sync_timing_device(device)
    return time.perf_counter()


def _timing_start(device, timings):
    return _timing_now(device) if timings is not None else None


def _record_timing(timings, name, start, device):
    if timings is None or start is None:
        return
    timings[name] = timings.get(name, 0.0) + (_timing_now(device) - start)


def _graph_edge_batch(adjacency, node_ids, edge_indices=None):
    """Return local endpoint indices and weights for a sampled edge batch."""

    node_ids = torch.as_tensor(node_ids, dtype=torch.long)
    if node_ids.ndim != 1 or len(node_ids) == 0:
        raise ValueError("graph edge batches must contain graph node IDs")
    if edge_indices is None:
        if len(node_ids) % 2 != 0:
            raise ValueError("graph edge batches must contain adjacent endpoint pairs")
        edge_indices = torch.arange(len(node_ids), dtype=torch.long).reshape(-1, 2)
    else:
        edge_indices = torch.as_tensor(edge_indices, dtype=torch.long)
        if edge_indices.ndim != 2 or edge_indices.shape[1] != 2 or len(edge_indices) == 0:
            raise ValueError("graph edge_indices must have shape [num_edges, 2]")
        if torch.any((edge_indices < 0) | (edge_indices >= len(node_ids))):
            raise ValueError("graph edge_indices refer to a node outside the collated batch")

    left_indices = edge_indices[:, 0]
    right_indices = edge_indices[:, 1]
    # SciPy's sparse advanced indexer mutates NumPy writeability flags, while
    # arrays exported from Torch can be non-writeable views.
    left_nodes = node_ids[left_indices].cpu().numpy().copy()
    right_nodes = node_ids[right_indices].cpu().numpy().copy()
    weights = np.asarray(adjacency[left_nodes, right_nodes]).reshape(-1)
    if len(weights) != len(left_nodes) or np.any(weights <= 0):
        raise RuntimeError("graph edge batch contains a pair absent from the adjacency")
    return left_indices, right_indices, weights


# How often a repulsion partner that turns out to be a graph neighbor is
# redrawn before its pair is dropped. The draw is uniform over the pool, so an
# attempt fails with probability deg/pool_size; below 1/2 - which only an
# in-batch graph whose k approaches its batch size exceeds - nine draws leave
# under 0.2% of the pairs to drop.
PARTNER_REDRAW_ATTEMPTS = 8


def _connected_partner_mask(adjacency, anchors, partners):
    """Flag the drawn pairs eq. (3) has no ``W_in = 0`` branch for.

    True where the graph joins the two nodes, and where the partner *is* the
    anchor: ``L(f_i, f_i, 0)`` is the degenerate case of the same mislabelling,
    a constant ``m^2`` pushing a point away from itself.
    """

    anchors = torch.as_tensor(anchors, dtype=torch.long).cpu()
    partners = torch.as_tensor(partners, dtype=torch.long).cpu()
    # SciPy's sparse advanced indexer mutates NumPy writeability flags, while
    # arrays exported from Torch can be non-writeable views.
    weights = np.asarray(
        adjacency[anchors.numpy().copy(), partners.numpy().copy()]
    ).reshape(-1)
    # Both graph builders eliminate their explicit zeros, so a stored weight is
    # a real edge and the test is exactly W_in != 0.
    return (anchors == partners) | torch.as_tensor(weights != 0)


def _partner_connectivity_test(adjacency, node_ids=None):
    """Bind ``adjacency`` into the test :func:`_sample_repulsion_pairs` wants.

    The sampler works in local batch positions. ``node_ids`` maps them to the
    graph's own node IDs for the whole-pool graph; in-batch mode builds its
    graph over the batch itself, so its positions already are the node IDs.
    """

    if node_ids is not None:
        node_ids = torch.as_tensor(node_ids, dtype=torch.long).cpu()

    def is_connected(anchors, partners):
        if node_ids is None:
            return _connected_partner_mask(adjacency, anchors, partners)
        return _connected_partner_mask(
            adjacency, node_ids[anchors], node_ids[partners]
        )

    return is_connected


def _sample_repulsion_pairs(
    node_count,
    edge_left,
    edge_right,
    partner_start,
    num_pairs,
    generator,
    *,
    is_connected=None,
):
    """Draw the ``L(x_i, x_n, 0)`` pairs of Weston et al. (2012), Algorithm 1.

    Algorithm 1 picks a random pair of neighbors ``x_i, x_j``, steps on
    ``L(x_i, x_j, 1)``, then picks a random example ``x_n`` and steps on
    ``L(x_i, x_n, 0)``: two pairs sharing the anchor ``x_i``, one an edge of the
    graph and one a plain random draw.

    The third argument of eq. (3) is ``W_in``, and only its ``W_in = 0`` branch
    is a repulsion, so a partner the graph joins to the anchor - or that is the
    anchor - must not be drawn: the pair belongs to the attractive branch the
    edge batch already trains. ``is_connected`` is that test. Blocked partners
    are redrawn, which leaves the draw uniform over the anchor's non-neighbors
    and keeps every anchor's pair count, and a pair whose anchor exhausts
    ``PARTNER_REDRAW_ATTEMPTS`` is dropped rather than trained on the wrong
    branch. The pool decides how often this fires: a partner drawn from the
    whole dataset is a neighbor with probability deg/N, while
    ``graph_batch_mode='in_batch'`` draws from one batch, where the same ratio
    is deg/batch_nodes and percent-scale.

    ``edge_left``/``edge_right`` are the sampled edges' local endpoint indices. A
    neighbor pair is unordered while edges arrive as (lower node, higher node),
    so a coin flip decides which endpoint plays ``x_i``. Partners are drawn from
    local positions ``partner_start..node_count-1``: under
    ``graph_batch_mode='global'`` those are the uniform pool draws appended by
    ``GraphEdgeBatchSampler``, while in-batch mode passes the whole batch.
    Returns local endpoint indices, or ``None`` when there is nothing to draw
    from.
    """

    num_pairs = int(num_pairs)
    edge_count = len(edge_left)
    partner_count = int(node_count) - int(partner_start)
    if edge_count < 1 or partner_count < 1 or num_pairs < 1:
        return None

    edge_left = torch.as_tensor(edge_left, dtype=torch.long).cpu()
    edge_right = torch.as_tensor(edge_right, dtype=torch.long).cpu()
    # One repulsion step per neighbor-pair step is Algorithm 1's alternation, so
    # the default count maps the two one to one; any other count picks the edges
    # its anchors come from uniformly.
    edges = (
        torch.arange(edge_count, dtype=torch.long)
        if num_pairs == edge_count
        else torch.randint(edge_count, (num_pairs,), generator=generator)
    )
    anchor_is_right = torch.randint(2, (num_pairs,), generator=generator).bool()
    left_indices = torch.where(anchor_is_right, edge_right[edges], edge_left[edges])
    right_indices = int(partner_start) + torch.randint(
        partner_count, (num_pairs,), generator=generator
    )
    if is_connected is None:
        return left_indices, right_indices

    blocked = is_connected(left_indices, right_indices)
    for _ in range(PARTNER_REDRAW_ATTEMPTS):
        positions = torch.nonzero(blocked, as_tuple=False).reshape(-1)
        if len(positions) == 0:
            break
        redrawn = int(partner_start) + torch.randint(
            partner_count, (len(positions),), generator=generator
        )
        right_indices[positions] = redrawn
        blocked[positions] = is_connected(left_indices[positions], redrawn)
    if bool(blocked.any()):
        # Only a pool small enough to be mostly one anchor's neighborhood gets
        # here, which is an in-batch graph over a small batch.
        keep = ~blocked
        left_indices, right_indices = left_indices[keep], right_indices[keep]
        if len(left_indices) == 0:
            return None
    return left_indices, right_indices


def _repulsion_energy(embeddings, left_indices, right_indices, margin):
    """Sum the ``W_ij = 0`` branch of eq. (3) over the given pairs.

    ``max(0, m - ||f_i - f_j||)^2`` from Weston et al. (2012), following Hadsell
    et al. (2006). The hinge is on the Euclidean distance rather than on its
    square, so ``margin`` lives on the same scale as the embedding space.
    Returns the summed energy and the per-pair hinge behind it.
    """

    embeddings = embeddings.float()
    differences = embeddings[left_indices] - embeddings[right_indices]
    squared_distances = (differences * differences).sum(dim=1)
    # sqrt' is unbounded at zero, which coincident embeddings would hit exactly.
    distances = torch.sqrt(squared_distances.clamp_min(1e-12))
    hinge = (float(margin) - distances).clamp_min(0.0)
    return (hinge * hinge).sum(), hinge


def _reduce_sampled_graph_energy(
    energy,
    sampled_edges,
    graph_edges,
    graph_weight,
    reduction,
):
    """Scale a uniform edge mini-batch to an unbiased graph objective estimate.

    ``sum`` rescales the sampled energy to the full trace, which is unbiased for
    any edge weighting because the sampler draws edges uniformly.

    ``mean`` divides by the number of *sampled edges*, not by their weight. Under
    ``edge_weighting='binary'`` those are the same number and this is the graph's
    weighted mean. Under ``cosine`` it is the mean per edge of an already
    weighted energy, which is the intended reading: the weighting is meant to
    change how hard each edge pulls, and dividing it back out would undo exactly
    that. It also keeps the loss on the same scale as the binary runs, so a
    regularizer_weight carries over. ``graph_weight`` is passed for a caller that
    wants the true weighted mean instead.
    """
    if reduction == "mean":
        return energy / float(sampled_edges) # remove mean, simply sum.
    if reduction == "torch_sum":
        return energy.sum()

    return energy * float(graph_edges) / float(sampled_edges)


def faiss_label_spreading(*args, **kwargs):
    """Call the modular implementation with façade-level helper overrides."""

    kwargs.setdefault("_dependencies", globals())
    return _faiss_label_spreading(*args, **kwargs)


def iscen_label_spreading(*args, **kwargs):
    """Call LP-DeepSSL with facade-level helper overrides."""

    kwargs.setdefault("_dependencies", globals())
    return _iscen_label_spreading(*args, **kwargs)


def mixed_label_propagation(*args, **kwargs):
    """Call the modular implementation with façade-level helper overrides."""

    kwargs.setdefault("_dependencies", globals())
    return _mixed_label_propagation(*args, **kwargs)


def _make_in_batch_graph_affinity(*args, **kwargs):
    """Use a CPU flat index for small, per-step graphs to avoid GPU setup cost."""

    kwargs["prefer_gpu"] = False
    return make_mixed_label_affinity(*args, **kwargs)


IN_BATCH_GRAPH_DEPENDENCIES = {
    "make_mixed_label_affinity": _make_in_batch_graph_affinity,
}


def _make_in_batch_graph_unlabeled_loader(
    regularizer,
    *,
    supervised_loader,
    config,
    device,
    seed,
    num_workers,
    start_method,
    epoch,
    log_dir,
):
    """Pair the labeled stream with sampled unlabeled nodes for local graphs."""

    if regularizer.dataset is None:
        raise RuntimeError("build_dataset must be called before make_loader")
    requested_unlabeled = int(regularizer.graph_unlabeled_batch_size)
    unlabeled_batch_size = min(requested_unlabeled, len(regularizer.dataset))
    if unlabeled_batch_size <= 0:
        raise ValueError(f"{regularizer.name} in-batch graphs need unlabeled samples")
    if unlabeled_batch_size < requested_unlabeled:
        logger.warning(
            f"{regularizer.name} graph_unlabeled_batch_size={requested_unlabeled} "
            f"exceeds the unlabeled pool of {len(regularizer.dataset)}; "
            f"using {unlabeled_batch_size}"
        )

    worker_count = utils.dataloader_num_workers_for_dataset(
        regularizer.dataset,
        num_workers,
    )
    cache_key = (
        "in_batch",
        id(regularizer.dataset),
        unlabeled_batch_size,
        int(worker_count),
        str(start_method),
    )
    if (
        regularizer._regularizer_loader is None
        or regularizer._regularizer_loader_cache_key != cache_key
    ):
        utils.shutdown_dataloaders(regularizer._regularizer_loader)
        regularizer._regularizer_loader = utils.make_unlabeled_stream_loader(
            regularizer.dataset,
            batch_size=unlabeled_batch_size,
            seed=seed,
            num_workers=worker_count,
            start_method=start_method,
            supervised_loader=supervised_loader,
            # Every graph has the configured composition. A ragged final graph
            # would change both neighborhood density and loss scale.
            drop_last=True,
            persistent_workers=True,
            pin_memory=torch.device(device).type == "cuda",
            desc="in-batch graph",
        )
        regularizer._regularizer_loader_cache_key = cache_key

    # A run can contain thousands of local graphs. Preserve useful graph
    # diagnostics without producing one artifact bundle per optimizer step.
    regularizer._in_batch_graph_diagnostics_request = make_graph_diagnostics_request(
        config=config,
        log_dir=log_dir,
        name=f"{regularizer.name}_in_batch_graph",
        epoch=epoch,
        title=f"{regularizer.name} in-batch graph",
    )
    logger.info(
        f"{regularizer.name} in-batch graph streams: "
        f"labeled_nodes_per_graph={regularizer.graph_labeled_batch_size}, "
        f"unlabeled_nodes_per_graph={unlabeled_batch_size}, "
        f"unlabeled_pool={len(regularizer.dataset)}, "
        f"steps={len(supervised_loader)}"
    )
    return CombinedTrainingLoader(
        supervised_loader,
        regularizer._regularizer_loader,
    )


def _in_batch_graph_context(
    regularizer,
    *,
    batch,
    supervised_embeddings,
    supervised_labels,
    regularizer_embeddings,
    supervised_indices,
):
    """Return the exact labeled/unlabeled rows forming one local graph."""

    if supervised_embeddings is None or supervised_labels is None:
        raise ValueError(f"{regularizer.name} in-batch mode needs labeled embeddings")
    if regularizer_embeddings is None:
        raise ValueError(f"{regularizer.name} in-batch mode needs unlabeled embeddings")
    if batch is None or len(batch) < 3:
        raise ValueError(
            f"{regularizer.name} in-batch unlabeled batches must include source positions"
        )

    requested_labeled = int(regularizer.graph_labeled_batch_size)
    labeled_count = min(requested_labeled, len(supervised_embeddings))
    if labeled_count <= 0:
        raise ValueError(f"{regularizer.name} in-batch graphs need labeled samples")
    if labeled_count < requested_labeled:
        logger.warning(
            f"{regularizer.name} received only {labeled_count} labeled rows for an "
            f"in-batch graph requesting {requested_labeled}"
        )

    if labeled_count == len(supervised_embeddings):
        labeled_rows_tensor = torch.arange(
            labeled_count,
            dtype=torch.long,
            device=supervised_embeddings.device,
        )
    else:
        labeled_rows_tensor = torch.randperm(
            len(supervised_embeddings),
            device=supervised_embeddings.device,
        )[:labeled_count]
    labeled_embeddings = supervised_embeddings[labeled_rows_tensor]
    labeled_targets = supervised_labels[labeled_rows_tensor]
    graph_embeddings = torch.cat(
        [labeled_embeddings, regularizer_embeddings],
        dim=0,
    )
    graph_targets = torch.cat(
        [
            labeled_targets,
            torch.full(
                (len(regularizer_embeddings),),
                UNLABELED_TARGET,
                dtype=torch.long,
                device=labeled_targets.device,
            ),
        ],
        dim=0,
    )

    graph_positions = None
    if supervised_indices is not None and regularizer._labeled_positions is not None:
        selected_supervised_indices = torch.as_tensor(
            supervised_indices,
            device=labeled_rows_tensor.device,
        )[labeled_rows_tensor]
        labeled_rows = np.asarray(
            selected_supervised_indices.detach().cpu(),
            dtype=np.int64,
        )
        if np.any((labeled_rows < 0) | (labeled_rows >= len(regularizer._labeled_positions))):
            raise ValueError("labeled in-batch graph indices are outside the support pool")
        labeled_positions = regularizer._labeled_positions[labeled_rows]
        unlabeled_positions = np.asarray(
            torch.as_tensor(batch[2]).detach().cpu(),
            dtype=np.int64,
        )
        graph_positions = np.concatenate(
            [labeled_positions, unlabeled_positions]
        ).astype(np.int64, copy=False)

    return graph_embeddings, graph_targets, graph_positions, labeled_count


def _maybe_save_in_batch_graph(
    regularizer,
    *,
    graph_embeddings,
    adjacency,
    graph_positions,
    labeled_count,
    graph_metadata,
):
    """Save the first local graph of an epoch when diagnostics are enabled."""

    request = getattr(regularizer, "_in_batch_graph_diagnostics_request", None)
    if request is None:
        return
    # Consume the request before writing so a diagnostics failure cannot cause
    # every subsequent batch to retry and flood the output directory.
    regularizer._in_batch_graph_diagnostics_request = None
    if graph_positions is None:
        logger.warning(
            f"Could not save {regularizer.name} in-batch graph diagnostics: "
            "labeled source indices are unavailable"
        )
        return
    labels = regularizer._train_dataset_labels[graph_positions]
    known_mask = np.arange(len(graph_positions), dtype=np.int64) < labeled_count
    maybe_save_graph_diagnostics(
        request=request,
        embeddings=np.ascontiguousarray(
            graph_embeddings.detach().float().cpu().numpy(),
            dtype=np.float32,
        ),
        adjacency=adjacency,
        positions=graph_positions,
        labels=labels,
        known_mask=known_mask,
        graph_metadata=graph_metadata,
    )


def _build_lrml_graph(regularizer, features, *, prefer_gpu=True):
    """Build the regularizer's graph under its configured edge weighting.

    ``mode='manifold_preserving'`` builds Ying et al.'s density-weighted
    Gaussian graph. Otherwise, ``binary`` is Hoi et al.'s W: every kNN edge
    pulls exactly as hard as every other. ``cosine`` calls the affinity Iscen's
    LP-DeepSSL builds, so the two methods construct the same graph from the same
    embeddings -- same internal
    L2 normalization, same ``max(cos, 0) ** gamma`` weights, same ``W + W.T``
    symmetrization giving a mutual kNN pair twice a one-way pair's weight, and
    the same positive-part clipping. That clipping is not cosmetic here: a
    negative weight in ``w * d^2`` is unbounded below, so it would turn an
    attractive edge into a runaway repulsion.

    Returns the directed kNN candidates, the symmetric adjacency, and the node
    degrees, which are neighbor counts under ``binary`` and affinity sums under
    ``cosine``. Manifold-preserving degrees are weighted affinity sums too.
    """

    if regularizer.mode == "manifold_preserving":
        (
            neighbor_indices,
            adjacency,
            degrees,
            construction_metadata,
        ) = build_manifold_preserving_graph(
            features,
            n_neighbors=regularizer.n_neighbors,
            similarity_v=regularizer.similarity_v,
            density_bandwidth=regularizer.density_bandwidth,
            prefer_gpu=prefer_gpu,
        )
        regularizer.graph_construction_metadata = construction_metadata
        return neighbor_indices, adjacency, degrees

    regularizer.graph_construction_metadata = {}
    if regularizer.edge_weighting == "binary":
        return build_lrml_knn_graph(
            features,
            n_neighbors=regularizer.n_neighbors,
            prefer_gpu=prefer_gpu,
        )

    adjacency, construction = make_mixed_label_affinity(
        features,
        n_neighbors=regularizer.n_neighbors,
        gamma=regularizer.gamma,
        return_diagnostics=True,
        prefer_gpu=prefer_gpu,
    )
    degrees = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    return construction["neighbor_indices"], adjacency, degrees


def _filter_lrml_graph_to_oracle_same_class(adjacency, labels):
    """Keep only graph edges whose endpoints have the same hidden true label.

    This is an explicit oracle ablation, not a deployable SSL operation.  It is
    useful for separating failures of the learned neighbourhood graph from
    failures of the Laplacian objective after the graph is correct.  Filtering
    the completed symmetric graph preserves either binary or cosine edge
    weights and therefore changes only which pairs reach the regularizer.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != adjacency.shape[0]:
        raise ValueError("LRML oracle labels must align with the graph nodes")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("LRML oracle filtering requires a square adjacency")

    original = adjacency.tocoo()
    same_class = labels[original.row] == labels[original.col]
    filtered = sparse.coo_matrix(
        (
            original.data[same_class],
            (original.row[same_class], original.col[same_class]),
        ),
        shape=original.shape,
        dtype=original.dtype,
    ).tocsr()
    filtered.setdiag(0)
    filtered.eliminate_zeros()
    if filtered.nnz == 0:
        raise ValueError("LRML oracle same-class filtering removed every graph edge")

    degrees = np.asarray(filtered.sum(axis=1), dtype=np.float64).ravel()
    original_edges = int(adjacency.nnz // 2)
    retained_edges = int(filtered.nnz // 2)
    original_weight = float(adjacency.sum()) / 2.0
    retained_weight = float(filtered.sum()) / 2.0
    return filtered, degrees, {
        "original_undirected_edges": original_edges,
        "retained_undirected_edges": retained_edges,
        "removed_undirected_edges": original_edges - retained_edges,
        "retained_edge_fraction": retained_edges / float(original_edges),
        "original_undirected_weight": original_weight,
        "retained_undirected_weight": retained_weight,
        "retained_weight_fraction": retained_weight / original_weight,
        "isolated_nodes": int(np.sum(degrees == 0.0)),
    }


def _lrml_node_scale(regularizer, degrees):
    """Return the per-node 1/sqrt(deg) factor of the symmetric normalization."""

    if not regularizer.normalized_laplacian:
        return np.ones(len(degrees), dtype=np.float64)
    # A binary graph's degree is an integer count that is always at least one,
    # so its historical floor of 1.0 never bound. A weighted degree is a sum of
    # affinities that is legitimately fractional, and that same floor would
    # silently rescale every node below it.
    floor = (
        1.0
        if regularizer.mode == "lrml" and regularizer.edge_weighting == "binary"
        else regularizer.MIN_WEIGHTED_DEGREE
    )
    return 1.0 / np.sqrt(np.maximum(degrees, floor))


def _local_graph_energy(embeddings, adjacency, reduction):
    """Evaluate every undirected edge of a graph built on the current batch.

    The energy is an unbounded sum of squared distances over every edge, so it
    is held in float32. That is what CUDA autocast already does on its own - it
    promotes ``normalize`` and ``sum``, so the embedding reaching this function
    is float32 there - while CPU autocast promotes neither and would run the
    whole reduction in bfloat16. Pinning it here makes the two agree.

    The evaluated endpoints are returned alongside the energy: the repulsion
    anchors its pairs on them, and rebuilding them would mean walking the
    adjacency twice.
    """

    embeddings = embeddings.float()
    edge_rows, edge_cols, edge_weights = graph_upper_triangle_edges(adjacency)
    left = torch.as_tensor(edge_rows, dtype=torch.long, device=embeddings.device)
    right = torch.as_tensor(edge_cols, dtype=torch.long, device=embeddings.device)
    weights = torch.as_tensor(
        edge_weights,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    squared_distances = ((embeddings[left] - embeddings[right]) ** 2).sum(dim=1)
    energy = (weights * squared_distances).sum()
    if reduction == "mean":
        energy = energy / float(len(edge_rows))
    return energy, len(edge_rows), float(edge_weights.sum()), left, right


LRML_AUXILIARY_MODULE = "lrml_auxiliary_embedding"


def _build_auxiliary_embedding_head(input_dim, output_dim, hidden_dim):
    """One auxiliary embedding network of Weston et al. (2012).

    ``hidden_dim=None`` is eq. (11) as written: a single new set of weights on
    top of the shared layer. An integer instead builds the nonlinear head
    Section 4.3 actually used for its deepest results (there: 50 hidden units
    onto a 10-dimensional embedding space).
    """

    if hidden_dim is None:
        return torch.nn.Linear(int(input_dim), int(output_dim))
    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
    )


class LrmlAuxiliaryEmbeddingHeads(torch.nn.Module):
    """Eq. (11)'s auxiliary networks, one per regularized trunk layer.

    The paper throws these away at test time, and nothing here evaluates
    through them: they exist only so the embedding loss has a space of its own
    to act in, one whose gradient still reaches the layers the retrieval head
    is built on.
    """

    def __init__(self, layers, output_dim, hidden_dim=None, normalize=False):
        super().__init__()
        self.layer_names = [str(name) for name, _ in layers]
        self.heads = torch.nn.ModuleList(
            [
                _build_auxiliary_embedding_head(width, output_dim, hidden_dim)
                for _, width in layers
            ]
        )
        self.output_dim = int(output_dim)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.normalize = bool(normalize)

    def forward(self, activations):
        if len(activations) != len(self.heads):
            raise RuntimeError(
                f"lrml auxiliary embedding expected {len(self.heads)} trunk "
                f"activations, got {len(activations)}"
            )
        embeddings = []
        for head, activation in zip(self.heads, activations):
            embedding = head(activation).float()
            if self.normalize:
                embedding = F.normalize(embedding, p=2.0, dim=1)
            embeddings.append(embedding)
        return embeddings


class LRMLRegularizer(BaseTrainingRegularizer):
    """Deep Laplacian Regularized Metric Learning regularizer (Hoi et al., 2010).

    Replaces LRML's linear projection U^T x with the network embedding f(x), so the
    regularizer is the graph-Laplacian smoothness energy

        g(theta) = (1/2) * sum_ij W_ij * || f(x_i) - f(x_j) ||^2 = tr(Z^T L Z),

    with W a symmetric kNN graph and L = D - W (optionally the symmetric-
    normalized Laplacian). The similar/dissimilar
    loss terms of the original objective are supplied by the configured supervised
    loss; this class only adds the unlabeled Laplacian term.

    ``edge_weighting`` selects W. ``binary`` is the paper's: every kNN edge pulls
    exactly as hard as every other, so a wrong edge costs as much as a right one.
    ``cosine`` instead builds the affinity Iscen's LP-DeepSSL uses, via the same
    ``make_mixed_label_affinity`` call, so the two methods can be compared on one
    graph rather than two: ``max(cos, 0) ** gamma`` weights, ``W + W.T``
    symmetrization that gives a mutual kNN pair twice a one-way pair's weight,
    and positive-part clipping. Measured on Cars196 at k=10 on frozen DINOv2
    features, the two produce the *identical* edge set -- only the weights
    differ -- but the cosine weighting moves the share of edge mass joining two
    same-class nodes from 0.466 to 0.523. It also normalizes internally, which
    the binary builder does not: that one only ranks by inner product, and its
    caller has to guarantee unit norms.

    ``cosine`` pulls hardest on the pairs already closest together, which is
    self-reinforcing. The Laplacian term alone is already minimized by collapse,
    so the ``contrastive`` repulsion matters more here, not less.

    ``mode='manifold_preserving'`` ports the regularity term of Ying et al.
    (2018), Equations (2), (7), and (8). Its directed graph weight is

        W_ij = beta_i * exp(-||z_i-z_j||^2 / (2 sigma^2)),  j in N(i),

    where ``beta_i`` is a normalized Gaussian Parzen density and Section IV's
    ``sigma = min(D) + (max(D)-min(D))/v`` is measured exactly on the graph
    features. Opposite directions are added into one undirected edge because
    their squared-distance energies are identical. This retains the paper's
    density-adaptive regularizer while reusing LRML's edge sampler. The default
    for this mode is the paper's unnormalized, 10-neighbor graph. The paper does
    not specify its Parzen bandwidth ``h``; ``density_bandwidth=None`` ties it to
    ``sigma`` and an explicit positive value can reproduce another choice.

    As with the default deep LRML adapter, the configured supervised metric loss
    replaces the paper's linear labeled-pair/triplet term. The network embedding
    also replaces its Mahalanobis matrix, so Adam updates network parameters
    rather than applying the paper's exponential-map optimizer on the SPD matrix
    manifold. This mode is therefore the paper's semisupervised regularizer, not
    a claim to reproduce its linear optimizer.

    ``GraphEdgeBatchSampler`` samples the symmetric graph's upper-triangle edges
    uniformly. Each step evaluates only those selected pairs and rescales their
    energy to an unbiased estimate of the full trace (or its stable global
    weighted mean). The symmetric-normalized Laplacian is obtained by scaling
    each endpoint embedding by 1/sqrt(deg) before evaluating its edge.

    ``contrastive`` optionally adds the repulsive branch of the semi-supervised
    embedding loss of Weston et al. (2012), eq. (3):

        L(f_i, f_j, W_ij) = || f_i - f_j ||^2                  if W_ij = 1
                            max(0, m - || f_i - f_j ||)^2      if W_ij = 0

    The attractive branch is the Laplacian energy this class already computes,
    so only the ``W_ij = 0`` branch is added:

        g(theta) + mean_{(i,k) random pairs} max(0, m - d_ik)^2

    It carries no weight of its own. The paper has exactly one hyperparameter
    here, the lambda of eq. (9), and Algorithm 1 spends it on both branches --
    ``lambda L(g(x_i), g(x_j), 1)`` and ``lambda L(g(x_i), g(x_n), 0)`` -- so
    ``regularizer_weight`` scales both and nothing rebalances them against each
    other. The reduction below is per pair for both branches, so one repulsion
    pair weighs exactly what one edge weighs. ``contrastive_update_mode='joint'``
    differentiates their sum in the legacy single update; ``'alternating'``
    yields the two graph branches separately after the trainer's supervised
    update, producing three real optimizer steps. Gradient-ratio calibration
    and GradNorm remain available only to joint mode because their measurement
    assumes one shared parameter point.

    ``embed_mode`` selects *where* that embedding loss acts, which is the axis
    Weston et al. actually win on. ``'output'`` is Fig. 1(a) and eq. (9): the
    loss is evaluated on the retrieval embedding itself, alongside the
    supervised metric loss. ``'auxiliary'`` is Fig. 1(c) and eq. (11): a fresh
    set of weights ``g_k(x) = W_AUX h_k(x)`` is attached to each internal layer
    of the trainable trunk, the graph loss is evaluated in those spaces instead,
    and the retrieval embedding carries no graph term at all. Attaching to every
    such layer at once -- ``embed_aux_layers='all'``, the default -- is the paper's
    EmbedALL, whose margin over output embedding grows with depth: on Mnist1h at
    15 layers, 47.7 plain, 11.8 for EmbedO, 9.3 for EmbedALL (Table 4).

    The trunk layers an auxiliary head can reach are the model's
    ``auxiliary_embedding_layers``: the projection head's post-ReLU activations,
    plus the backbone output when backbone blocks are being tuned. A frozen
    backbone with a single linear projection head has none, and the mode refuses
    to configure rather than attach heads that share no weight with the
    retrieval head and therefore regularize nothing. Section 4.3's auxiliary
    embedding is itself nonlinear -- 50 hidden units onto a 10-dimensional
    space -- which is ``embed_aux_hidden_dim``; leaving it ``None`` builds
    eq. (11) as written. ``embed_aux_normalize`` is off by default because
    eq. (3) is a statement about plain distances: the retrieval head's unit
    sphere caps every margin at 2 and, at the default margin of 1, leaves the
    repulsion inactive on most pairs, while an auxiliary space is free to let
    the margin set its own scale. The heads are training-only, as in the paper,
    and are excluded from every gradient comparison against the supervised term
    (see :meth:`private_model_module_names`).

    With the symmetric-normalized Laplacian, the positive branch is degree-
    scaled and is therefore an LRML/Weston hybrid. Set
    ``normalized_laplacian=False`` when the desired positive branch is literally
    Weston et al.'s plain embedding distance.

    It is off by default. Its purpose is the one the paper gives: the pure
    Laplacian term is minimized by collapsing every embedding onto one point,
    which the repulsion inhibits without the balancing constraints of eq. (2).

    Algorithm 1 pairs each ``L(x_i, x_j, 1)`` step on a random neighbor pair with
    an ``L(x_i, x_n, 0)`` step on the *same* anchor and a random ``x_n``, so each
    repulsion pair here hangs off one of the step's sampled edges: the anchor is
    one of that edge's two endpoints (a coin flip, since the pair is unordered)
    and the partner is a uniform draw from the graph's node pool, restricted to
    the nodes ``W_in = 0`` actually holds for: eq. (3) reads its third argument
    as the pair's graph weight, so a partner the graph joins to the anchor is
    redrawn rather than pushed away, and so is a draw that lands on the anchor
    itself. Under
    ``graph_batch_mode='global'``, ``GraphEdgeBatchSampler`` appends those uniform
    draws to each edge batch so their embeddings exist. Alternating mode expands
    the realized pairs into its own fresh batched forward and is supported only
    for this global mode. Under joint ``graph_batch_mode='in_batch'`` only the
    step's own labeled and unlabeled samples are embedded, so the partner comes
    from that batch and the anchor from the per-batch graph's edges. Under
    ``graph_on='all'`` the pool includes the labeled nodes, matching the node set
    the attractive term runs over.
    """

    name = "lrml"
    supports_frozen_feature_precompute = True
    # The whole graph refresh is one make_loader call over a fixed node set, so
    # the engine can repeat it inside an epoch on a sample-count schedule.
    supports_sample_scoped_refresh = True

    DEFAULT_PARAMS = {
        "mode": "lrml",                # "lrml" (Hoi et al.) or Ying et al.'s "manifold_preserving"
        "n_neighbors": 6,             # Hoi et al. use 6; manifold_preserving defaults to 10
        "edge_weighting": "binary",   # Hoi et al.'s binary W, or Iscen's "cosine" affinity
        "gamma": 3.0,                 # exponent of edge_weighting="cosine"; unread when binary
        "similarity_v": 10.0,         # Ying et al. Section IV Gaussian bandwidth parameter
        "density_bandwidth": None,    # Ying et al.'s unspecified Parzen h; None reuses sigma
        "normalized_laplacian": True, # Hoi et al.; manifold_preserving defaults to False
        "graph_on": "all",            # "all" (labeled + unlabeled) or "unlabeled"
        "reduction": "mean",          # stable global weighted mean, or full-trace "sum"
        "graph_batch_size": None,      # sampled graph edges; None matches supervised batch size
        # Diagnostic upper bound that deliberately leaks hidden training labels.
        # It filters the completed kNN graph to same-class pairs without changing
        # the retained edges' binary/cosine weights.
        "oracle_same_class_edges": False,
        "contrastive": False,          # Weston et al. (2012) eq. (3) repulsion, at the paper's shared lambda
        "contrastive_margin": 1.0,     # m of eq. (3), a distance in the L2-normalized embedding space
        "contrastive_pairs": None,     # random pairs drawn per step; None gives one per sampled edge, as in Algorithm 1
        "contrastive_update_mode": "joint",  # "joint" is legacy one-step training; "alternating" takes the paper's three ordered updates
        # Where Weston et al. (2012) attach the embedding loss: "output" is
        # Fig. 1(a)/eq. (9), "auxiliary" is Fig. 1(c)/eq. (11) on every trunk
        # layer at once, which is the EmbedALL their deep results are won with,
        # and "pre_norm" is eq. (10) on the last-but-one representation -- the
        # projection head's output before F.normalize. The paper prescribes that
        # last one whenever the output layer's representation does not suit a
        # 2-norm loss (their example is a softmax; here it is the unit sphere,
        # which caps every margin at 2 and leaves the default one inactive on
        # most pairs). It needs no new parameters and no deeper head, and it
        # reaches every trainable weight the retrieval embedding depends on.
        "embed_mode": "output",
        "embed_aux_layers": "all",     # "all" is EmbedALL; a name list picks single layers
        "embed_aux_dim": None,         # width of eq. (11)'s embedding space; None reuses feat_dim
        "embed_aux_hidden_dim": None,  # None is eq. (11) literally; an int is Section 4.3's nonlinear head
        "embed_aux_normalize": False,  # eq. (3) measures plain distances, not angles on a sphere
    }
    MODE_CHOICES = {"lrml", "manifold_preserving"}
    EMBED_MODE_CHOICES = {"output", "auxiliary", "pre_norm"}
    AUXILIARY_ONLY_PARAMS = (
        "embed_aux_layers",
        "embed_aux_dim",
        "embed_aux_hidden_dim",
        "embed_aux_normalize",
    )
    GRAPH_ON_CHOICES = {"all", "unlabeled"}
    EDGE_WEIGHTING_CHOICES = {"binary", "cosine"}
    REDUCTION_CHOICES = {"mean", "sum", "torch_sum"}
    # A weighted degree is a sum of affinities in (0, 2], so it is legitimately
    # below 1 and must not be clamped there the way an integer neighbor count
    # can be. Only guard against a division by zero.
    MIN_WEIGHTED_DEGREE = 1e-12
    CONTRASTIVE_UPDATE_MODE_CHOICES = {"joint", "alternating"}
    # L2-normalized embeddings are at most this far apart, so a larger margin can
    # never be satisfied and the repulsion never stops pushing.
    MAX_USEFUL_CONTRASTIVE_MARGIN = 2.0

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(regularizer_weight=regularizer_weight, supervised_weight=supervised_weight)
        if "contrastive_weight" in params:
            raise ValueError(
                "lrml contrastive_weight was removed: Weston et al. (2012) weigh both "
                "branches of eq. (3) with the single lambda of eq. (9), which is "
                "regularizer_weight here. Use contrastive=true and scale the pair via "
                "method_params.regularizer_weight or regularizer_target_ratio"
            )
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown lrml regularizer_params: {unknown}")
        mode = str(params.get("mode", self.DEFAULT_PARAMS["mode"]))
        if mode not in self.MODE_CHOICES:
            raise ValueError(f"lrml mode must be one of {sorted(self.MODE_CHOICES)}")
        defaults = dict(self.DEFAULT_PARAMS)
        if mode == "manifold_preserving":
            # Section IV uses ten neighbors and Equation (8) is the ordinary
            # (not symmetrically normalized) graph Laplacian.
            defaults["n_neighbors"] = 10
            defaults["normalized_laplacian"] = False
        merged = {**defaults, **params}

        self.mode = mode

        self.n_neighbors = int(merged["n_neighbors"])
        if self.n_neighbors <= 0:
            raise ValueError("lrml n_neighbors must be positive")
        configured_edge_weighting = str(merged["edge_weighting"])
        if configured_edge_weighting not in self.EDGE_WEIGHTING_CHOICES:
            raise ValueError(
                f"lrml edge_weighting must be one of {sorted(self.EDGE_WEIGHTING_CHOICES)}"
            )
        self.edge_weighting = (
            "density_gaussian"
            if self.mode == "manifold_preserving"
            else configured_edge_weighting
        )
        self.gamma = float(merged["gamma"])
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("lrml gamma must be finite and positive")
        self.similarity_v = float(merged["similarity_v"])
        if not math.isfinite(self.similarity_v) or self.similarity_v <= 0:
            raise ValueError("lrml similarity_v must be finite and positive")
        density_bandwidth = merged["density_bandwidth"]
        self.density_bandwidth = (
            None if density_bandwidth is None else float(density_bandwidth)
        )
        if (
            self.density_bandwidth is not None
            and (
                not math.isfinite(self.density_bandwidth)
                or self.density_bandwidth <= 0
            )
        ):
            raise ValueError(
                "lrml density_bandwidth must be finite and positive when set"
            )
        self.normalized_laplacian = bool(merged["normalized_laplacian"])
        self.graph_on = str(merged["graph_on"])
        if self.graph_on not in self.GRAPH_ON_CHOICES:
            raise ValueError(f"lrml graph_on must be one of {sorted(self.GRAPH_ON_CHOICES)}")
        self.reduction = str(merged["reduction"])
        if self.reduction not in self.REDUCTION_CHOICES:
            raise ValueError(f"lrml reduction must be one of {sorted(self.REDUCTION_CHOICES)}")
        graph_batch_size = merged["graph_batch_size"]
        self.graph_batch_size = None if graph_batch_size is None else int(graph_batch_size)
        if self.graph_batch_size is not None and self.graph_batch_size <= 0:
            raise ValueError("lrml graph_batch_size must be positive when set")
        self.oracle_same_class_edges = bool(merged["oracle_same_class_edges"])
        self.contrastive = bool(merged["contrastive"])
        self.contrastive_margin = float(merged["contrastive_margin"])
        if not math.isfinite(self.contrastive_margin) or self.contrastive_margin <= 0:
            raise ValueError("lrml contrastive_margin must be finite and positive")
        contrastive_pairs = merged["contrastive_pairs"]
        self.contrastive_pairs = None if contrastive_pairs is None else int(contrastive_pairs)
        if self.contrastive_pairs is not None and self.contrastive_pairs <= 0:
            raise ValueError("lrml contrastive_pairs must be positive when set")
        self.embed_mode = str(merged["embed_mode"])
        if self.embed_mode not in self.EMBED_MODE_CHOICES:
            raise ValueError(
                f"lrml embed_mode must be one of {sorted(self.EMBED_MODE_CHOICES)}"
            )
        embed_aux_layers = merged["embed_aux_layers"]
        if isinstance(embed_aux_layers, str):
            if embed_aux_layers != "all":
                raise ValueError(
                    "lrml embed_aux_layers must be \"all\" or a list of layer names"
                )
            self.embed_aux_layers = "all"
        else:
            names = [str(name) for name in embed_aux_layers]
            if not names:
                raise ValueError(
                    "lrml embed_aux_layers must name at least one trunk layer"
                )
            if len(set(names)) != len(names):
                raise ValueError("lrml embed_aux_layers must not repeat a layer")
            self.embed_aux_layers = names
        embed_aux_dim = merged["embed_aux_dim"]
        self.embed_aux_dim = None if embed_aux_dim is None else int(embed_aux_dim)
        if self.embed_aux_dim is not None and self.embed_aux_dim <= 0:
            raise ValueError("lrml embed_aux_dim must be positive when set")
        embed_aux_hidden_dim = merged["embed_aux_hidden_dim"]
        self.embed_aux_hidden_dim = (
            None if embed_aux_hidden_dim is None else int(embed_aux_hidden_dim)
        )
        if self.embed_aux_hidden_dim is not None and self.embed_aux_hidden_dim <= 0:
            raise ValueError("lrml embed_aux_hidden_dim must be positive when set")
        self.embed_aux_normalize = bool(merged["embed_aux_normalize"])
        self.contrastive_update_mode = str(merged["contrastive_update_mode"])
        if self.contrastive_update_mode not in self.CONTRASTIVE_UPDATE_MODE_CHOICES:
            raise ValueError(
                "lrml contrastive_update_mode must be one of "
                f"{sorted(self.CONTRASTIVE_UPDATE_MODE_CHOICES)}"
            )
        if self.contrastive_update_mode == "alternating" and not self.contrastive:
            raise ValueError(
                "lrml contrastive_update_mode='alternating' requires contrastive=true"
            )
        self.uses_separate_optimizer_steps = (
            self.contrastive_update_mode == "alternating"
        )
        if (
            self.contrastive
            and self.repulsion_space_is_normalized
            and self.contrastive_margin > self.MAX_USEFUL_CONTRASTIVE_MARGIN
        ):
            logger.warning(
                f"lrml contrastive_margin={self.contrastive_margin} exceeds the largest "
                f"distance between L2-normalized embeddings "
                f"({self.MAX_USEFUL_CONTRASTIVE_MARGIN}), so no repulsion pair can ever "
                "satisfy it"
            )

        self.dataset = None
        self.graph_positions = None
        self.graph_known_mask = None
        self.neighbor_indices = None
        self.adjacency = None   # scipy CSR, symmetric binary W
        self.degrees = None
        self.node_scale = None  # torch tensor, 1/sqrt(deg) or ones
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self.graph_construction_metadata = {}
        self.oracle_filter_stats = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        self.graph_batch_mode = "global"
        self.graph_labeled_batch_size = None
        self.graph_unlabeled_batch_size = None
        self._labeled_positions = None
        self._train_dataset_labels = None
        self._in_batch_graph_diagnostics_request = None
        self._last_diagnostics = {}
        self._contrastive_generator = None
        self._contrastive_seed = 0
        self.auxiliary_heads = None
        self._auxiliary_layer_positions = None
        self._auxiliary_layer_count = None

    def set_target_ratio(
        self,
        target_ratio,
        calibration_batches=None,
        recalibration_interval_samples=None,
        update_interval_samples=None,
        probe_memory=None,
        ema_alpha=None,
        aggregation=None,
        statistic=None,
        statistic_anchor=None,
        log_diagnostics=False,
    ):
        if target_ratio is not None and self.uses_separate_optimizer_steps:
            raise ValueError(
                "lrml contrastive_update_mode='alternating' cannot use "
                "regularizer_target_ratio: calibration assumes the supervised and "
                "regularizer gradients are evaluated at one shared parameter point. "
                "Use fixed supervised_weight and regularizer_weight values"
            )
        return super().set_target_ratio(
            target_ratio,
            calibration_batches=calibration_batches,
            recalibration_interval_samples=recalibration_interval_samples,
            update_interval_samples=update_interval_samples,
            probe_memory=probe_memory,
            ema_alpha=ema_alpha,
            aggregation=aggregation,
            statistic=statistic,
            statistic_anchor=statistic_anchor,
            log_diagnostics=log_diagnostics,
        )

    def set_grad_norm(
        self,
        alpha,
        lr=None,
        update_interval=None,
        renormalize=None,
        parameterization=None,
        exclude_components=None,
    ):
        if alpha is not None and self.uses_separate_optimizer_steps:
            raise ValueError(
                "lrml contrastive_update_mode='alternating' cannot use GradNorm: "
                "GradNorm assumes the supervised and regularizer losses share one "
                "backward graph. Use fixed supervised_weight and regularizer_weight values"
            )
        return super().set_grad_norm(
            alpha,
            lr=lr,
            update_interval=update_interval,
            renormalize=renormalize,
            parameterization=parameterization,
            exclude_components=exclude_components,
        )

    def configure_graph_batching(self, config):
        self.graph_batch_mode = str(config.graph_batch_mode)
        if self.embed_mode != "output" and self.graph_batch_mode != "global":
            raise ValueError(
                f"lrml embed_mode={self.embed_mode!r} currently requires "
                "graph_batch_mode='global': the in-batch graph reads the embeddings "
                "the trainer's joint forward already produced, which carry none of "
                "the trunk tensors this mode evaluates eq. (3) in"
            )
        if self.uses_separate_optimizer_steps and self.graph_batch_mode != "global":
            raise ValueError(
                "lrml contrastive_update_mode='alternating' currently requires "
                "graph_batch_mode='global' so its positive and random-pair updates "
                "can be forwarded separately"
            )
        if self.graph_batch_mode == "global":
            return
        if self.graph_batch_mode != "in_batch":
            # 'in_batch_merged' folds the supervised term into the regularizer's
            # own loss call, which only stml_threshold implements.
            raise ValueError(
                "lrml graph_batch_mode must be one of ['global', 'in_batch']"
            )
        if self.graph_on != "all":
            raise ValueError(
                "lrml graph_batch_mode='in_batch' requires graph_on='all' so the "
                "configured labeled and unlabeled nodes both participate"
            )
        if self.graph_batch_size is not None:
            raise ValueError(
                "lrml regularizer_params.graph_batch_size samples edges from the "
                "global graph and cannot be combined with graph_batch_mode='in_batch'"
            )
        self.graph_labeled_batch_size = int(config.graph_labeled_batch_size)
        self.graph_unlabeled_batch_size = int(config.graph_unlabeled_batch_size)
        # The graph uses embeddings from the same labeled/unlabeled forward.
        self.uses_joint_forward = True
        self.requires_labeled_indices = True

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("LRML regularization requires batch_size >= 2")
        if (
            self.graph_batch_mode == "in_batch"
            and self.graph_labeled_batch_size > int(args.batch_size)
        ):
            raise ValueError(
                "lrml graph_labeled_batch_size cannot exceed the supervised "
                f"batch_size ({self.graph_labeled_batch_size} > {args.batch_size})"
            )

    def build_dataset(self, train_dataset, split, use_cache=False):
        if self.graph_on == "all":
            positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        else:
            positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        positions = np.unique(np.asarray(positions, dtype=np.int64))  # sorted + deterministic
        if len(positions) < 2:
            raise ValueError("LRML regularization requires at least two graph samples")
        utils.shutdown_dataloaders(
            self._regularizer_loader,
            self._embedding_loader,
        )
        self.graph_positions = positions
        self.graph_known_mask = np.isin(positions, np.asarray(split.labeled_positions, dtype=np.int64))
        regularizer_dataset = self.make_regularizer_source_dataset(train_dataset, use_cache=use_cache)
        self._labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)
        self._train_dataset_labels = np.asarray(train_dataset.labels, dtype=np.int64)
        if self.graph_batch_mode == "in_batch":
            unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
            if len(unlabeled_positions) == 0:
                raise ValueError("LRML in-batch mode requires unlabeled samples")
            self.dataset = UnlabeledSubset(
                regularizer_dataset,
                unlabeled_positions,
                num_views=1,
            )
        else:
            self.dataset = LRMLGraphDataset(regularizer_dataset, positions)
        self.neighbor_indices = None
        self.adjacency = None
        self.degrees = None
        self.node_scale = None
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self.graph_construction_metadata = {}
        self.oracle_filter_stats = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        self._in_batch_graph_diagnostics_request = None
        self._last_diagnostics = {}
        return self.dataset

    def make_loader(
        self, model, train_dataset, supervised_loader, device, config,
        batch_size, seed, num_workers, start_method, epoch,
        log_dir=None,
    ):
        if self.dataset is None or self.graph_positions is None:
            raise RuntimeError("build_dataset must be called before make_loader")
        # Seeded once so repulsion sampling is reproducible for a run, then
        # left alone so later epochs continue the stream instead of repeating it.
        self._contrastive_seed = int(seed)
        if self.contrastive_enabled and self._contrastive_generator is None:
            self._pair_generator()
            logger.info(
                "LRML contrastive repulsion enabled at the shared regularizer weight: "
                f"margin={self.contrastive_margin}, "
                "random_pairs_per_step="
                f"{'one per evaluated edge' if self.contrastive_pairs is None else self.contrastive_pairs}"
                f", each anchored on that edge, update_mode={self.contrastive_update_mode}"
            )
        if self.graph_batch_mode == "in_batch":
            return _make_in_batch_graph_unlabeled_loader(
                self,
                supervised_loader=supervised_loader,
                config=config,
                device=device,
                seed=seed,
                num_workers=num_workers,
                start_method=start_method,
                epoch=epoch,
                log_dir=log_dir,
            )

        has_cached_graph = (
            self.neighbor_indices is not None
            and self.adjacency is not None
            and self.degrees is not None
            and self.node_scale is not None
            and self.graph_edge_count is not None
            and self.graph_weight_sum is not None
        )
        # Consume unconditionally: a request must not survive a call that
        # rebuilt for another reason and force a second rebuild afterwards.
        refresh_requested = self.consume_refresh_request()
        should_rebuild_graph = (
            not has_cached_graph
            or refresh_requested
            or should_rebuild_on_epoch(
                config.update_mode,
                config.update_interval_epochs,
                epoch,
                self._last_graph_rebuild_epoch,
            )
        )
        if not should_rebuild_graph:
            self.node_scale = self.node_scale.to(device)
            logger.info(
                "Reusing LRML graph: "
                f"{self.adjacency.shape[0]} nodes, {self.graph_edge_count} undirected edges, "
                f"mean_degree={float(self.degrees.mean()):.2f}, "
                f"normalized_laplacian={self.normalized_laplacian}"
            )
        else:
            if self._embedding_loader is None:
                self._embedding_loader = make_embedding_loader(
                    train_dataset, self.graph_positions,
                    config.embedding_batch_size, config.embedding_num_workers, seed, start_method,
                )
            embeddings = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.graph_positions,
                device=self.get_ssl_device(device),
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"LRML graph embeddings - epoch {epoch}",
                embedding_kind="default",
                loader=self._embedding_loader,
            )

            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            norms = np.linalg.norm(embeddings, axis=1)
            if not np.all(np.isfinite(norms)):
                raise ValueError("LRML embeddings have non-finite norms")

            if np.any(norms <= 1e-12):
                bad = np.flatnonzero(norms <= 1e-12)
                raise ValueError(
                    "LRML embeddings contain zero-norm vectors at indices "
                    f"{bad[:10].tolist()}"
                )

            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError(
                    "IndexFlatIP requires L2-normalized LRML embeddings; "
                    f"observed norm range [{norms.min():.9f}, {norms.max():.9f}]"
                )
            neighbor_indices, adjacency, degrees = _build_lrml_graph(self, embeddings)
            graph_labels = dataset_labels_for_positions(
                train_dataset,
                self.graph_positions,
            )
            oracle_filter_stats = None
            if self.oracle_same_class_edges:
                adjacency, degrees, oracle_filter_stats = (
                    _filter_lrml_graph_to_oracle_same_class(
                        adjacency,
                        graph_labels,
                    )
                )
                logger.warning(
                    "LRML ORACLE graph filter used hidden true labels: "
                    f"retained {oracle_filter_stats['retained_undirected_edges']} / "
                    f"{oracle_filter_stats['original_undirected_edges']} same-class "
                    "edges "
                    f"({oracle_filter_stats['retained_edge_fraction']:.3%}); "
                    f"isolated_nodes={oracle_filter_stats['isolated_nodes']}"
                )
            edge_rows, _, edge_weights = graph_upper_triangle_edges(adjacency)
            maybe_save_graph_diagnostics(
                request=make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_knn",
                    epoch=epoch,
                    title=(
                        "Manifold-preserving density-weighted kNN graph"
                        if self.mode == "manifold_preserving"
                        else "LRML symmetric kNN graph"
                    ),
                ),
                embeddings=embeddings,
                adjacency=adjacency,
                positions=self.graph_positions,
                labels=graph_labels,
                known_mask=self.graph_known_mask,
                graph_metadata={
                    "graph_kind": (
                        "density_weighted_gaussian_knn"
                        if self.mode == "manifold_preserving"
                        else (
                            "binary_knn_union"
                            if self.edge_weighting == "binary"
                            else "positive_part_cosine_knn"
                        )
                    ),
                    "mode": self.mode,
                    "edge_weighting": (
                        "density_gaussian"
                        if self.mode == "manifold_preserving"
                        else self.edge_weighting
                    ),
                    "gamma": (
                        self.gamma
                        if self.mode == "lrml" and self.edge_weighting == "cosine"
                        else None
                    ),
                    "requested_n_neighbors": self.n_neighbors,
                    "search_n_neighbors": (
                        min(self.n_neighbors, max(len(embeddings) - 1, 0))
                        if neighbor_indices is None
                        else int(neighbor_indices.shape[1])
                    ),
                    "neighbor_indices": neighbor_indices,
                    "oracle_same_class_edges": self.oracle_same_class_edges,
                    "oracle_filter": oracle_filter_stats,
                    **self.graph_construction_metadata,
                },
            )
            self.neighbor_indices = neighbor_indices
            self.adjacency = adjacency
            self.degrees = degrees
            self.graph_edge_count = int(len(edge_rows))
            self.graph_weight_sum = float(edge_weights.sum())
            self.oracle_filter_stats = oracle_filter_stats
            self._last_graph_rebuild_epoch = None if epoch is None else int(epoch)
            scale = _lrml_node_scale(self, degrees)
            self.node_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
            if self.mode == "manifold_preserving":
                weighting_name = "density_gaussian"
                weighting_details = (
                    f" (sigma={self.graph_construction_metadata['similarity_sigma']:.4g})"
                )
            else:
                weighting_name = self.edge_weighting
                weighting_details = (
                    "" if self.edge_weighting == "binary" else f" (gamma={self.gamma})"
                )
            logger.info(
                "Built LRML graph: "
                f"{adjacency.shape[0]} nodes, {self.graph_edge_count} undirected edges, "
                f"mean_degree={float(degrees.mean()):.2f}, "
                f"mode={self.mode}, "
                f"edge_weighting={weighting_name}{weighting_details}, "
                f"mean_edge_weight={float(edge_weights.mean()):.4f}, "
                f"normalized_laplacian={self.normalized_laplacian}, "
                f"oracle_same_class_edges={self.oracle_same_class_edges}"
            )

        num_workers = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        graph_batch_size = int(
            batch_size if self.graph_batch_size is None else self.graph_batch_size
        )
        dataloader_kwargs = utils.make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=True,
        )
        # Algorithm 1's random partner comes from the whole pool, so the pool
        # draws have to be loaded and embedded alongside the edge endpoints.
        random_nodes_per_batch = 0
        if self.contrastive_enabled:
            random_nodes_per_batch = (
                graph_batch_size if self.contrastive_pairs is None else self.contrastive_pairs
            )
        cache_key = (
            id(self.dataset),
            graph_batch_size,
            random_nodes_per_batch,
            int(len(supervised_loader)),
            int(dataloader_kwargs.get("num_workers", 0)),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_sampler = GraphEdgeBatchSampler(
                adjacency=self.adjacency,
                graph_batch_size=graph_batch_size,
                seed=seed,
                num_batches=len(supervised_loader),
                random_nodes_per_batch=random_nodes_per_batch,
            )
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_sampler=self._regularizer_sampler,
                collate_fn=collate_graph_edge_batch,
                **dataloader_kwargs,
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "LRML batch streams: "
                f"supervised_examples_per_batch={int(batch_size)}, "
                f"graph_edges_per_batch={graph_batch_size}, "
                f"max_graph_nodes_before_dedup={2 * graph_batch_size}, "
                f"uniform_pool_nodes_per_batch={random_nodes_per_batch}, "
                f"steps={len(supervised_loader)}"
            )
        else:
            self._regularizer_sampler.set_graph(self.adjacency)

        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    @property
    def contrastive_enabled(self):
        return self.contrastive

    @property
    def repulsion_space_is_normalized(self):
        """Whether eq. (3)'s distances are measured on a unit sphere.

        The retrieval head L2-normalizes, so the output mode's margin is capped
        by that sphere's diameter. An auxiliary embedding has no such cap unless
        it is asked for one, which is the paper's own reading: eq. (3) trades
        collapse for a margin in a free space.
        """

        if self.embed_mode == "output":
            return True
        if self.embed_mode == "pre_norm":
            # The whole point of eq. (10) here: the margin sets its own scale.
            return False
        return self.embed_aux_normalize

    def private_model_module_names(self):
        # The auxiliary heads are eq. (11)'s new weights: the supervised loss
        # never reaches them, and evaluation never runs through them, so no
        # gradient comparison against the supervised term may count them.
        return (
            (LRML_AUXILIARY_MODULE,) if self.embed_mode == "auxiliary" else ()
        )

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        """Attach eq. (11)'s auxiliary heads before the optimizer collects parameters."""

        del train_dataset, split, train_labels_mapper
        if self.embed_mode != "auxiliary" or self.regularizer_weight == 0:
            return None
        if hasattr(student_model, LRML_AUXILIARY_MODULE):
            raise RuntimeError(
                "lrml auxiliary embedding heads are already configured on this model"
            )
        if not hasattr(student_model, "auxiliary_embedding_layers"):
            raise TypeError(
                f"lrml embed_mode='auxiliary' needs a model that exposes its trunk "
                f"activations; {type(student_model).__name__} does not"
            )
        available = list(student_model.auxiliary_embedding_layers())
        if not available:
            raise ValueError(
                "lrml embed_mode='auxiliary' found no trunk layer to attach eq. (11) "
                "to. The backbone is frozen and the projection head is a single "
                "linear map, so an auxiliary head would share no trainable weight "
                "with the retrieval head and could not regularize anything. Set "
                "projection_layers >= 2 (with feat_dim), or unfreeze backbone blocks"
            )
        available_names = [name for name, _ in available]
        if self.embed_aux_layers == "all":
            selected = list(range(len(available)))
        else:
            unknown = sorted(set(self.embed_aux_layers) - set(available_names))
            if unknown:
                raise ValueError(
                    f"lrml embed_aux_layers names {unknown}, which this model does "
                    f"not expose. Available: {available_names}"
                )
            selected = [available_names.index(name) for name in self.embed_aux_layers]
        output_dim = (
            student_model.feat_dim if self.embed_aux_dim is None else self.embed_aux_dim
        )
        heads = LrmlAuxiliaryEmbeddingHeads(
            [available[position] for position in selected],
            output_dim,
            hidden_dim=self.embed_aux_hidden_dim,
            normalize=self.embed_aux_normalize,
        ).to(device)
        student_model.add_module(LRML_AUXILIARY_MODULE, heads)
        self.auxiliary_heads = heads
        self._auxiliary_layer_positions = selected
        self._auxiliary_layer_count = len(available)
        logger.info(
            "Configured LRML auxiliary embedding (Weston et al. 2012, eq. (11)): "
            f"layers={heads.layer_names} of {available_names}, "
            f"embedding_dim={heads.output_dim}, "
            + (
                "linear head"
                if heads.hidden_dim is None
                else f"nonlinear head with {heads.hidden_dim} hidden units"
            )
            + f", normalize={heads.normalize}. The graph loss is evaluated in these "
            "spaces instead of on the retrieval embedding; the heads are training-only"
        )
        return None

    def _pair_generator(self):
        """Keep one RNG stream for the repulsion's pair draws across the whole run."""

        if self._contrastive_generator is None:
            generator = torch.Generator()
            generator.manual_seed(self._contrastive_seed)
            self._contrastive_generator = generator
        return self._contrastive_generator

    def _draw_contrastive_pairs(
        self,
        node_count,
        edge_left,
        edge_right,
        partner_start,
        is_connected,
    ):
        """Draw Algorithm 1's anchored random pairs and describe the draw.

        ``contrastive_pairs`` counts the pairs asked for; the pairs actually
        drawn can fall short of it when an anchor's pool holds nothing eq. (3)
        would call a non-neighbor.
        """

        if not self.contrastive_enabled:
            return None, {}
        requested_pairs = (
            len(edge_left) if self.contrastive_pairs is None else self.contrastive_pairs
        )
        partner_pool = int(node_count) - int(partner_start)
        pairs = _sample_repulsion_pairs(
            node_count,
            edge_left,
            edge_right,
            partner_start,
            requested_pairs,
            self._pair_generator(),
            is_connected=is_connected,
        )
        if pairs is None:
            diagnostics = {
                "train/lrml/contrastive_requested_pairs": float(requested_pairs),
                "train/lrml/contrastive_pairs": 0.0,
                "train/lrml/contrastive_partner_pool": float(partner_pool),
            }
            if self.collect_batch_diagnostics:
                diagnostics["train/lrml/contrastive_active_pairs"] = 0.0
                diagnostics["train/lrml/contrastive_term"] = 0.0
            return pairs, diagnostics
        diagnostics = {
            "train/lrml/contrastive_requested_pairs": float(requested_pairs),
            "train/lrml/contrastive_pairs": float(len(pairs[0])),
            "train/lrml/contrastive_partner_pool": float(partner_pool),
        }
        return pairs, diagnostics

    def _repulsion_from_pairs(self, embeddings, pairs, graph_edges, graph_weight):
        """Evaluate eq. (3)'s ``W_ij = 0`` branch on an already drawn pair set.

        The draw is separate from the evaluation because the auxiliary mode runs
        the same pairs through several embedding spaces: Algorithm 1 loops its
        embedding functions around one anchor, and redrawing per space would only
        add variance to a term whose pairs are already uniform.
        """

        left_indices, right_indices = pairs
        energy, hinge = _repulsion_energy(
            embeddings,
            left_indices.to(embeddings.device),
            right_indices.to(embeddings.device),
            self.contrastive_margin,
        )
        repulsion = _reduce_sampled_graph_energy(
            energy=energy,
            sampled_edges=len(left_indices),
            graph_edges=graph_edges,
            graph_weight=graph_weight,
            reduction=self.reduction,
        )
        return repulsion, hinge

    def _sampled_edge_attraction(
        self,
        embeddings,
        node_ids,
        left_indices,
        right_indices,
        weights,
        device,
    ):
        """The ``W_ij = 1`` branch over one uniform edge mini-batch.

        Folds the per-node 1/sqrt(deg) factor in: normalized-Laplacian energy
        equals the unnormalized energy on degree-scaled embeddings.
        """

        scale = self.node_scale[node_ids.to(device)].to(dtype=embeddings.dtype)
        scaled = embeddings * scale[:, None]
        differences = (
            scaled[left_indices.to(device)] - scaled[right_indices.to(device)]
        )
        squared_distances = (differences * differences).sum(dim=1)
        # W_ij is 1 for every edge only under edge_weighting='binary'. The
        # cosine affinity is what makes this multiply load-bearing, and dropping
        # it would train the weighted graph as if it were the binary one.
        edge_weight = torch.as_tensor(
            weights,
            dtype=squared_distances.dtype,
            device=squared_distances.device,
        )
        energy = (edge_weight * squared_distances).sum()
        return _reduce_sampled_graph_energy(
            energy=energy,
            sampled_edges=len(weights),
            graph_edges=self.graph_edge_count,
            graph_weight=self.graph_weight_sum,
            reduction=self.reduction,
        )

    def _embedding_spaces(self, student_model, inputs, device):
        """Return the named spaces eq. (3) is evaluated in for this input batch.

        ``embed_mode='output'`` yields the one retrieval embedding, which is
        Fig. 1(a). ``'pre_norm'`` yields the projection head's output before
        normalization, which is eq. (10) on the last-but-one representation.
        ``'auxiliary'`` yields one space per configured trunk layer: the
        retrieval embedding is computed by the same forward but carries no graph
        loss, exactly as in Fig. 1(c), where the auxiliary heads are the only
        place the embedding loss acts.

        The graph itself is built on the normalized retrieval embedding in every
        mode, because that is the space the run is evaluated in and holding it
        fixed is what makes the modes comparable. Under ``'pre_norm'`` the
        neighborhood is therefore chosen by cosine while the energy is measured
        by Euclidean distance off the sphere; the two disagree exactly by the
        per-sample norm, which ``train/lrml/pre_norm_radius`` tracks.
        """

        if self.embed_mode == "output":
            embedding = utils.forward_model_inputs(
                student_model,
                inputs,
                device,
                use_cache=self.use_cache,
            ).float()  # see _local_graph_energy: CPU autocast would sum in bfloat16
            return [("output", embedding)]
        _, pre_norm, hidden = utils.forward_model_inputs_with_trunk(
            student_model,
            inputs,
            device,
            use_cache=self.use_cache,
        )
        if self.embed_mode == "pre_norm":
            return [("pre_norm", pre_norm.float())]
        if self.auxiliary_heads is None:
            raise RuntimeError(
                "lrml embed_mode='auxiliary' has no heads: configure_model must run "
                "before training so the optimizer collects eq. (11)'s weights"
            )
        if len(hidden) != self._auxiliary_layer_count:
            raise RuntimeError(
                f"lrml auxiliary embedding was configured against "
                f"{self._auxiliary_layer_count} trunk activations but the forward "
                f"returned {len(hidden)}"
            )
        activations = [hidden[position] for position in self._auxiliary_layer_positions]
        return list(
            zip(self.auxiliary_heads.layer_names, self.auxiliary_heads(activations))
        )

    def _contrastive_repulsion(
        self,
        embeddings,
        edge_left,
        edge_right,
        partner_start,
        graph_edges,
        graph_weight,
        is_connected,
    ):
        """Return eq. (3)'s ``W_ij = 0`` term for one joint-loss step.

        ``embeddings`` are the batch's plain embeddings: the 1/sqrt(deg) scaling
        belongs to the normalized Laplacian, while the margin is a distance in
        the embedding space itself. The attractive term's reduction is reused so
        both branches keep one scale. ``contrastive_update_mode='joint'`` sums
        them before one backward; the alternating mode uses the same pair draw in
        :meth:`separate_optimizer_step_losses` but steps between fresh forwards.
        """

        pairs, diagnostics = self._draw_contrastive_pairs(
            len(embeddings),
            edge_left,
            edge_right,
            partner_start,
            is_connected,
        )
        if pairs is None:
            return None, diagnostics
        repulsion, hinge = self._repulsion_from_pairs(
            embeddings,
            pairs,
            graph_edges,
            graph_weight,
        )
        if self.collect_batch_diagnostics:
            # Both of these read device memory, so they wait for a run that asks
            # for diagnostics instead of syncing on every step.
            diagnostics["train/lrml/contrastive_active_pairs"] = float((hinge > 0).sum())
            diagnostics["train/lrml/contrastive_term"] = float(repulsion.detach())
        return repulsion, diagnostics

    def separate_optimizer_step_losses(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
    ):
        """Yield Weston's batched attraction and repulsion updates in order.

        The trainer has already taken the supervised update when it starts
        consuming this generator. It steps the model after the first yield, then
        resumes the generator, so the random-pair forward below observes the
        post-attraction parameters. Pair examples are expanded before each
        forward instead of reusing the joint mode's endpoint-deduplicated
        embeddings; repeated anchors therefore retain their per-pair weight in
        training-mode layers such as batch normalization.
        """

        del state, timings
        if not self.uses_separate_optimizer_steps:
            raise RuntimeError(
                "LRML separate optimizer losses require "
                "contrastive_update_mode='alternating'"
            )
        if self.graph_batch_mode != "global":
            raise RuntimeError("LRML alternating updates require a global graph batch")
        if len(batch) == 3:
            images, node_ids, edge_indices = batch
        else:
            images, node_ids = batch
            edge_indices = None
        if not torch.is_tensor(images):
            raise TypeError("LRML alternating updates require tensor graph inputs")

        node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        left_indices, right_indices, weights = _graph_edge_batch(
            self.adjacency,
            node_ids,
            edge_indices,
        )

        # Restore the sampled pair multiplicity that graph collation deduplicates.
        # This is still one batched positive update, not one optimizer update per
        # edge; it is the practical mini-batch analogue of Algorithm 1's step.
        positive_indices = torch.stack((left_indices, right_indices), dim=1).reshape(-1)
        positive_inputs = images.index_select(
            0,
            positive_indices.to(images.device),
        )
        positive_node_ids = node_ids.index_select(
            0,
            positive_indices.to(node_ids.device),
        )
        positive_spaces = self._embedding_spaces(student_model, positive_inputs, device)
        positive_scale = self.node_scale[positive_node_ids.to(device)]
        attraction = None
        for _, positive_embeddings in positive_spaces:
            scaled_positive = positive_embeddings * positive_scale.to(
                dtype=positive_embeddings.dtype
            )[:, None]
            positive_differences = scaled_positive[0::2] - scaled_positive[1::2]
            positive_squared_distances = (
                positive_differences * positive_differences
            ).sum(dim=1)
            # Same reason as the joint path: W_ij is 1 on every edge only under
            # edge_weighting='binary'. Dropping it would train the cosine affinity
            # and the manifold-preserving density-Gaussian graph as if both were the
            # binary one.
            positive_edge_weight = torch.as_tensor(
                weights,
                dtype=positive_squared_distances.dtype,
                device=positive_squared_distances.device,
            )
            attraction_energy = (
                positive_edge_weight * positive_squared_distances
            ).sum()
            space_attraction = _reduce_sampled_graph_energy(
                energy=attraction_energy,
                sampled_edges=len(weights),
                graph_edges=self.graph_edge_count,
                graph_weight=self.graph_weight_sum,
                reduction=self.reduction,
            )
            attraction = (
                space_attraction
                if attraction is None
                else attraction + space_attraction
            )
        yield "attraction", attraction

        attraction_detached = attraction.detach()
        del (
            attraction,
            space_attraction,
            attraction_energy,
            positive_edge_weight,
            positive_squared_distances,
            positive_differences,
            scaled_positive,
            positive_scale,
            positive_embeddings,
            positive_spaces,
            positive_node_ids,
            positive_inputs,
            positive_indices,
        )

        # Execution reaches this point only after the trainer has applied the
        # attraction step. Draw the partner now, as Algorithm 1 does, and run a
        # fresh forward at the updated parameters.
        endpoint_count = 1 + int(
            torch.maximum(left_indices.max(), right_indices.max())
        )
        pairs, contrastive_diagnostics = self._draw_contrastive_pairs(
            len(images),
            left_indices,
            right_indices,
            endpoint_count,
            _partner_connectivity_test(self.adjacency, node_ids),
        )
        if pairs is None:
            raise RuntimeError(
                "LRML alternating repulsion drew no partner pair: either the "
                "loader was built without contrastive=true, so no uniform pool "
                "nodes are streamed, or every drawn partner was a graph "
                "neighbor of its anchor"
            )
        repulsion_left, repulsion_right = pairs
        random_pair_indices = torch.stack(
            (repulsion_left, repulsion_right),
            dim=1,
        ).reshape(-1)
        random_pair_inputs = images.index_select(
            0,
            random_pair_indices.to(images.device),
        )
        random_pair_spaces = self._embedding_spaces(
            student_model,
            random_pair_inputs,
            device,
        )
        pair_count = len(repulsion_left)
        repulsion = None
        active_pairs = 0.0
        for _, random_pair_embeddings in random_pair_spaces:
            pair_left = torch.arange(
                0,
                2 * pair_count,
                2,
                dtype=torch.long,
                device=random_pair_embeddings.device,
            )
            pair_right = pair_left + 1
            repulsion_energy, hinge = _repulsion_energy(
                random_pair_embeddings,
                pair_left,
                pair_right,
                self.contrastive_margin,
            )
            space_repulsion = _reduce_sampled_graph_energy(
                energy=repulsion_energy,
                sampled_edges=pair_count,
                graph_edges=self.graph_edge_count,
                graph_weight=self.graph_weight_sum,
                reduction=self.reduction,
            )
            repulsion = (
                space_repulsion if repulsion is None else repulsion + space_repulsion
            )
            if self.collect_batch_diagnostics:
                active_pairs += float((hinge > 0).sum())
        self._last_diagnostics = {
            "train/lrml/optimizer_updates_per_iteration": 3.0,
            **contrastive_diagnostics,
        }
        if self.collect_batch_diagnostics:
            self._last_diagnostics.update(
                {
                    "train/lrml/attraction_term": float(attraction_detached),
                    "train/lrml/contrastive_active_pairs": active_pairs,
                    "train/lrml/contrastive_term": float(repulsion.detach()),
                }
            )
        yield "repulsion", repulsion

    def compute_loss(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
        supervised_embeddings=None,
        supervised_labels=None,
        regularizer_embeddings=None,
        supervised_indices=None,
        **unused_context,
    ):
        if self.uses_separate_optimizer_steps:
            raise RuntimeError(
                "LRML contrastive_update_mode='alternating' must be consumed through "
                "the trainer's ordered optimizer-step path"
            )
        if self.graph_batch_mode == "in_batch":
            (
                graph_embeddings,
                _,
                graph_positions,
                labeled_count,
            ) = _in_batch_graph_context(
                self,
                batch=batch,
                supervised_embeddings=supervised_embeddings,
                supervised_labels=supervised_labels,
                regularizer_embeddings=regularizer_embeddings,
                supervised_indices=supervised_indices,
            )
            graph_features = np.ascontiguousarray(
                graph_embeddings.detach().float().cpu().numpy(),
                dtype=np.float32,
            )
            with suppress_ssl_timing_logs():
                # A per-step graph is small, so a CPU flat index beats paying
                # GPU index setup on every optimizer step.
                neighbor_indices, adjacency, degrees = _build_lrml_graph(
                    self,
                    graph_features,
                    prefer_gpu=False,
                )
            oracle_filter_stats = None
            if self.oracle_same_class_edges:
                if graph_positions is None:
                    raise RuntimeError(
                        "LRML oracle same-class filtering requires graph positions"
                    )
                adjacency, degrees, oracle_filter_stats = (
                    _filter_lrml_graph_to_oracle_same_class(
                        adjacency,
                        self._train_dataset_labels[graph_positions],
                    )
                )
            scaled = graph_embeddings
            if self.normalized_laplacian:
                scale = torch.as_tensor(
                    _lrml_node_scale(self, degrees),
                    dtype=graph_embeddings.dtype,
                    device=graph_embeddings.device,
                )
                scaled = graph_embeddings * scale[:, None]
            loss, edge_count, weight_sum, edge_left, edge_right = _local_graph_energy(
                scaled,
                adjacency,
                self.reduction,
            )
            _maybe_save_in_batch_graph(
                self,
                graph_embeddings=graph_embeddings,
                adjacency=adjacency,
                graph_positions=graph_positions,
                labeled_count=labeled_count,
                graph_metadata={
                    "graph_kind": (
                        "in_batch_density_weighted_gaussian_knn"
                        if self.mode == "manifold_preserving"
                        else (
                            "in_batch_binary_knn_union"
                            if self.edge_weighting == "binary"
                            else "in_batch_positive_part_cosine_knn"
                        )
                    ),
                    "mode": self.mode,
                    "edge_weighting": (
                        "density_gaussian"
                        if self.mode == "manifold_preserving"
                        else self.edge_weighting
                    ),
                    "gamma": (
                        self.gamma
                        if self.mode == "lrml" and self.edge_weighting == "cosine"
                        else None
                    ),
                    "requested_n_neighbors": self.n_neighbors,
                    "search_n_neighbors": int(neighbor_indices.shape[1]),
                    "neighbor_indices": neighbor_indices,
                    "oracle_same_class_edges": self.oracle_same_class_edges,
                    "oracle_filter": oracle_filter_stats,
                    **self.graph_construction_metadata,
                },
            )
            # The step's own labeled and unlabeled samples are the only pool in
            # this mode, so the partner is drawn from all of it that the batch
            # graph does not join to the anchor, while the anchor still comes
            # from that graph's edges. A batch-sized pool is where the
            # restriction bites: deg/batch_nodes is percent-scale.
            repulsion, contrastive_diagnostics = self._contrastive_repulsion(
                graph_embeddings,
                edge_left=edge_left,
                edge_right=edge_right,
                partner_start=0,
                graph_edges=edge_count,
                graph_weight=weight_sum,
                is_connected=_partner_connectivity_test(adjacency),
            )
            self._last_diagnostics = {
                "train/lrml/in_batch_graph_nodes": float(len(graph_embeddings)),
                "train/lrml/in_batch_labeled_nodes": float(labeled_count),
                "train/lrml/in_batch_unlabeled_nodes": float(
                    len(graph_embeddings) - labeled_count
                ),
                "train/lrml/in_batch_graph_edges": float(edge_count),
                "train/lrml/in_batch_graph_weight": float(weight_sum),
                "train/lrml/oracle_retained_edge_fraction": (
                    1.0
                    if oracle_filter_stats is None
                    else oracle_filter_stats["retained_edge_fraction"]
                ),
                **contrastive_diagnostics,
            }
            if repulsion is None:
                return loss
            return loss + repulsion

        if len(batch) == 3:
            images, node_ids, edge_indices = batch
        else:
            images, node_ids = batch
            edge_indices = None

        spaces = self._embedding_spaces(student_model, images, device)
        left_indices, right_indices, weights = _graph_edge_batch(
            self.adjacency, node_ids, edge_indices
        )
        # GraphEdgeBatchSampler assigns the edge endpoints local indices
        # 0..endpoint_count-1 and appends the uniform pool draws behind them, so
        # this split tells the partner draws apart from the anchors' edges
        # without a second index stream.
        endpoint_count = 1 + int(
            torch.maximum(left_indices.max(), right_indices.max())
        )
        pairs, diagnostics = self._draw_contrastive_pairs(
            len(spaces[0][1]),
            left_indices,
            right_indices,
            endpoint_count,
            _partner_connectivity_test(self.adjacency, node_ids),
        )
        # Algorithm 1 spends one lambda on both branches of every embedding
        # function it loops over, so the spaces are summed rather than averaged:
        # each auxiliary head answers for its own layer at full weight.
        total = None
        active_pairs = 0.0
        repulsion_total = 0.0
        for name, embeddings in spaces:
            energy = self._sampled_edge_attraction(
                embeddings,
                node_ids,
                left_indices,
                right_indices,
                weights,
                device,
            )
            if self.collect_batch_diagnostics:
                if len(spaces) > 1:
                    diagnostics[
                        f"train/lrml/embedding_space/{name}/attraction"
                    ] = float(energy.detach())
                if name == "pre_norm":
                    # F.normalize is scale invariant, so the retrieval metric
                    # cannot see this radius: the attraction can be paid for by
                    # shrinking it, and only the margin pins it back. Watch it
                    # to tell a term that is shaping the embedding from one that
                    # is spending itself on a gauge the evaluation ignores.
                    diagnostics["train/lrml/pre_norm_radius"] = float(
                        embeddings.detach().norm(dim=1).mean()
                    )
            if pairs is not None:
                repulsion, hinge = self._repulsion_from_pairs(
                    embeddings,
                    pairs,
                    self.graph_edge_count,
                    self.graph_weight_sum,
                )
                energy = energy + repulsion
                if self.collect_batch_diagnostics:
                    active_pairs += float((hinge > 0).sum())
                    repulsion_total += float(repulsion.detach())
                    if len(spaces) > 1:
                        diagnostics[
                            f"train/lrml/embedding_space/{name}/repulsion"
                        ] = float(repulsion.detach())
            total = energy if total is None else total + energy
        if pairs is not None and self.collect_batch_diagnostics:
            # Summed over spaces, so one number keeps meaning "this step's
            # repulsion" whether the term acts in one space or in five.
            diagnostics["train/lrml/contrastive_active_pairs"] = active_pairs
            diagnostics["train/lrml/contrastive_term"] = repulsion_total
        self._last_diagnostics = diagnostics
        return total

    def batch_diagnostics(self):
        return dict(self._last_diagnostics)

class STMLRegularizer(BaseTrainingRegularizer):
    """Use the existing STML objective as an unlabeled regularization term."""

    name = "stml"
    supports_frozen_feature_precompute = True
    provides_trainable_projection_without_feat_dim = True

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        self.criterion = metric_losses.STMLLoss(**params)
        self.num_views = self.criterion.num_views
        self.num_neighbors = self.criterion.num_neighbors
        self.teacher_momentum = self.criterion.teacher_momentum
        self.normalize_student = self.criterion.normalize_student
        self.dataset = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_sampling_rebuild_epoch = None

    def model_kwargs(self, args):
        return {
            "stml": True,
            "stml_g_dim": getattr(args, "stml_g_dim", None),
            "stml_normalize_student": self.normalize_student,
        }

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("STML regularization requires batch_size >= 2")
        if args.batch_size % self.num_neighbors != 0:
            raise ValueError(
                "STML regularization requires batch_size to be divisible by "
                "method_params.regularizer_params.num_neighbors"
            )
        if args.stml_g_dim is not None and args.stml_g_dim <= 0:
            raise ValueError("stml_g_dim must be positive when set")

    def build_dataset(self, train_dataset, split, use_cache=False):
        if len(split.unlabeled_positions) < 2:
            raise ValueError("STML regularization requires at least two unlabeled samples")
        self.use_cache = bool(use_cache)
        self.dataset = UnlabeledSubset(
            train_dataset,
            split.unlabeled_positions,
            num_views=self.num_views,
        )
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_sampling_rebuild_epoch = None
        return self.dataset

    def make_loader(
        self,
        model,
        train_dataset,
        supervised_loader,
        device,
        config,
        batch_size,
        seed,
        num_workers,
        start_method,
        epoch,
        log_dir=None,
    ):
        if self.dataset is None:
            raise RuntimeError("build_dataset must be called before make_loader")
        effective_num_workers = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        cache_key = (
            id(self.dataset),
            int(batch_size),
            int(self.num_neighbors),
            int(effective_num_workers),
            str(start_method),
        )
        should_rebuild_sampling = (
            self._regularizer_loader is None
            or self._regularizer_loader_cache_key != cache_key
            or should_rebuild_on_epoch(
                config.update_mode,
                config.update_interval_epochs,
                epoch,
                self._last_sampling_rebuild_epoch,
            )
        )
        if should_rebuild_sampling:
            utils.shutdown_dataloaders(self._regularizer_loader)
            sampling_embeddings = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.dataset.positions,
                device=self.get_ssl_device(device),
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"STML sampling embeddings - epoch {epoch}",
                embedding_kind="stml_g",
            )
            self._regularizer_loader = utils.make_stml_train_loader(
                train_dataset=self.dataset,
                sampling_embeddings=sampling_embeddings,
                batch_size=batch_size,
                neighbors_per_query=self.num_neighbors,
                seed=seed,
                num_workers=effective_num_workers,
                start_method=start_method,
                graph_device=self.get_ssl_device(device),
            )
            self._regularizer_loader_cache_key = cache_key
            self._last_sampling_rebuild_epoch = None if epoch is None else int(epoch)
        else:
            logger.info(
                "Reusing STML nearest-neighbor sampler: "
                f"{len(self.dataset)} samples, {self.num_neighbors} neighbors/query"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def initialize_state(self, student_model, device):
        teacher_model = copy.deepcopy(student_model)
        torch.nn.init.orthogonal_(teacher_model.embedding_g.weight)
        torch.nn.init.zeros_(teacher_model.embedding_g.bias)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        logger.info("Initialized STML EMA teacher from the supervised student")
        return teacher_model.to(device)

    def compute_loss(self, student_model, state, batch, device, timings=None):
        teacher_model = state
        if teacher_model is None:
            raise RuntimeError("STML regularization requires an initialized EMA teacher")
        images, _, instance_ids = batch
        if not isinstance(images, (list, tuple)) or len(images) != self.num_views:
            raise ValueError(f"STML batches must contain {self.num_views} augmented views per sample")
        images = torch.cat(list(images), dim=0)
        instance_ids = instance_ids.repeat(self.num_views).to(device)
        student_g, student_f = student_model.forward_stml_cached(images, device)
        with torch.no_grad():
            teacher_g = teacher_model.forward_stml_teacher_cached(images, device)
        return self.criterion(student_f, student_g, teacher_g, instance_ids)

    @torch.no_grad()
    def after_optimizer_step(self, student_model, state):
        teacher_model = state
        if teacher_model is None:
            raise RuntimeError("STML regularization requires an initialized EMA teacher")
        teacher_parameters = dict(teacher_model.named_parameters())
        for name, student_parameter in student_model.named_parameters():
            if name.startswith("fc."):
                continue
            teacher_parameters[name].lerp_(student_parameter.detach(), 1 - self.teacher_momentum)
        teacher_buffers = dict(teacher_model.named_buffers())
        for name, student_buffer in student_model.named_buffers():
            teacher_buffer = teacher_buffers[name]
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.lerp_(student_buffer.detach(), 1 - self.teacher_momentum)
            else:
                teacher_buffer.copy_(student_buffer.detach())


class HofferEntropyRegularizer(BaseTrainingRegularizer):
    """Deep neighbor-embedding entropy regularizer (Hoffer & Ailon, 2016).

    Implements the unlabeled term of "Semi-supervised deep learning by metric
    embedding" (arXiv:1611.01449). Every step draws one labeled reference per
    class (uniform within class, freshly resampled for that step)
    plus a batch of unlabeled samples x_u; all are embedded by the current
    network in a single forward pass. Over the references, the distance softmax

        P_i(x_u) = exp(-||f(x_u) - f(z_i)||^2) / sum_j exp(-||f(x_u) - f(z_j)||^2)

    is formed and the regularization term is the mean Shannon entropy
    H(P(x_u)) over the unlabeled batch. Gradients flow through both the
    unlabeled and the reference embeddings, so the labeled references act as
    class anchors that are pushed away from ambiguous regions.

    Deviations from the paper, mirroring the other deep regularizers here: the
    paper's supervised term (the NCA-style cross entropy -log P_y(x_l) against
    the same references) is supplied by the configured supervised loss instead;
    ``supervised_weight``/``regularizer_weight`` play the role of the paper's
    lambda_L/lambda_U. The shared model forward path supplies the normalized
    embeddings used by the supervised objective and this regularizer; with many
    classes, ``distance_scale`` (an inverse temperature on -d^2) can counteract
    the resulting softmax flattening. ``reference_sets=K`` draws K independent, complete reference
    sets and averages K separate C-way entropies; it never flattens them into a
    K*C-way exemplar softmax. Every class with labeled support supplies one
    reference per set and participates in every denominator; a training class
    without any labeled exemplar (as under ``class_subset_k_shot``) is dropped
    from the reference set with a warning, so its unlabeled samples still get
    an entropy signal over the classes that do have support. For large C this intentionally retains
    the paper's O(C) reference forward pass and O(B*C) distance calculation
    rather than silently changing to sampled softmax.

    Batch layout: each regularizer batch contains ``batch_size`` unlabeled
    samples plus exactly ``reference_sets * C`` reference images.
    """

    name = "hoffer_entropy"
    supports_frozen_feature_precompute = True

    DEFAULT_PARAMS = {
        "distance_scale": 1.0,
        "reference_sets": 1,
        "unlabeled_batch_size": None,
    }

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(regularizer_weight=regularizer_weight, supervised_weight=supervised_weight)
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown hoffer_entropy regularizer_params: {unknown}")
        merged = {**self.DEFAULT_PARAMS, **params}

        self.distance_scale = float(merged["distance_scale"])
        if not math.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("hoffer_entropy distance_scale must be finite and positive")
        self.reference_sets = int(merged["reference_sets"])
        if self.reference_sets <= 0:
            raise ValueError("hoffer_entropy reference_sets must be positive")
        unlabeled_batch_size = merged["unlabeled_batch_size"]
        self.unlabeled_batch_size = None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        if self.unlabeled_batch_size is not None and self.unlabeled_batch_size <= 0:
            raise ValueError("hoffer_entropy unlabeled_batch_size must be positive")

        self.dataset = None
        self.class_candidates = None
        self.reference_class_labels = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None

    def validate_run_args(self, args):
        return None

    def build_dataset(self, train_dataset, split, use_cache=False):
        labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) == 0:
            raise ValueError("hoffer_entropy regularization requires unlabeled samples")
        labels = dataset_labels_for_positions(train_dataset, labeled_positions)
        if labels is None:
            raise ValueError("hoffer_entropy regularization requires a train dataset exposing .labels")

        # Group labeled positions by class; candidate indices live in the joint
        # HofferReferenceDataset index space (references start at num_unlabeled).
        num_unlabeled = len(unlabeled_positions)
        all_labels = dataset_labels_for_positions(
            train_dataset, np.arange(len(train_dataset), dtype=np.int64)
        )
        if all_labels is None:
            raise ValueError("hoffer_entropy could not determine the complete training class set")
        unique_labels = np.unique(all_labels[all_labels != UNLABELED_TARGET]).astype(np.int64)
        reference_labels = np.unique(labels).astype(np.int64)
        missing_classes = np.setdiff1d(unique_labels, reference_labels)
        if len(missing_classes) > 0:
            # Full class support is desirable, not required: the entropy is
            # defined over whatever reference set exists, so classes without a
            # labeled exemplar simply drop out of the softmax and their
            # unlabeled samples are still scored against the remaining classes.
            # Splits such as class_subset_k_shot label only a class subset by
            # construction, so warn instead of refusing the run.
            preview = missing_classes[:20].tolist()
            elided = "" if len(missing_classes) <= len(preview) else ", ..."
            logger.warning(
                "hoffer_entropy has no labeled exemplar for "
                f"{len(missing_classes)} of {len(unique_labels)} training classes; "
                f"the reference softmax spans only the {len(reference_labels)} classes with "
                f"labeled support. Missing classes: {preview}{elided}"
            )
        class_candidates = [
            num_unlabeled + np.flatnonzero(labels == label)
            for label in reference_labels
        ]
        if len(class_candidates) < 2:
            raise ValueError("hoffer_entropy regularization requires at least two labeled classes")
        self.class_candidates = class_candidates
        self.reference_class_labels = reference_labels
        regularizer_dataset = self.make_regularizer_source_dataset(train_dataset, use_cache=use_cache)
        self.dataset = HofferReferenceDataset(regularizer_dataset, unlabeled_positions, labeled_positions)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        return self.dataset

    def make_loader(
        self,
        model,
        train_dataset,
        supervised_loader,
        device,
        config,
        batch_size,
        seed,
        num_workers,
        start_method,
        epoch,
        log_dir=None,
    ):
        if self.dataset is None or self.class_candidates is None:
            raise RuntimeError("build_dataset must be called before make_loader")
        num_workers = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        unlabeled_batch_size = int(batch_size if self.unlabeled_batch_size is None else self.unlabeled_batch_size)
        cache_key = (
            id(self.dataset),
            unlabeled_batch_size,
            int(num_workers),
            str(start_method),
            int(self.reference_sets),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            sampler = HofferReferenceBatchSampler(
                num_unlabeled=self.dataset.num_unlabeled,
                class_candidates=self.class_candidates,
                unlabeled_per_batch=unlabeled_batch_size,
                seed=seed,
                reference_sets=self.reference_sets,
                class_labels=self.reference_class_labels,
                unlabeled_positions=self.dataset.unlabeled_positions,
                labeled_positions=self.dataset.labeled_positions,
            )
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_sampler=sampler,
                **utils.make_dataloader_kwargs(
                    num_workers,
                    seed,
                    start_method,
                    persistent_workers=True,
                ),
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "Hoffer regularizer loader: "
                f"unlabeled_pool={self.dataset.num_unlabeled}, "
                f"labeled_reference_pool={self.dataset.num_labeled}, "
                f"reference_classes={len(self.class_candidates)}, "
                f"supervised_batch_size={int(batch_size)}, "
                f"unlabeled_batch_size={unlabeled_batch_size}, "
                f"reference_classes_per_batch={sampler.reference_classes_per_batch}, "
                f"references_per_class={sampler.references_per_class}, "
                f"reference_batch_size={sampler.references_per_batch}, "
                f"regularizer_forward_batch_size={unlabeled_batch_size + sampler.references_per_batch}"
            )
        regularizer_loader = self._regularizer_loader
        return CombinedTrainingLoader(supervised_loader, regularizer_loader)

    def compute_loss(self, student_model, state, batch, device, timings=None):
        if len(batch) == 3:
            images, is_reference, _ = batch
        else:
            images, is_reference = batch

        t0 = _timing_start(device, timings)
        # float32: CUDA autocast already promotes the squared-distance sum and
        # the log_softmax below, CPU autocast promotes neither, and the entropy
        # is a sum of p*log(p) terms over every reference set.
        embeddings = utils.forward_model_inputs(
            student_model,
            images,
            device,
            use_cache=self.use_cache,
        ).float()
        _record_timing(timings, "hoffer_forward", t0, device)

        role_tensor = torch.as_tensor(is_reference)
        # The loader supplies CPU roles. Compact them there, before transferring
        # integer indices, to avoid two CUDA nonzero synchronizations. Advanced
        # indexing sees exactly the same indices and retains its eager backward.
        role_mask = role_tensor.bool()
        reference_indices = role_mask.nonzero(as_tuple=True)[0].to(device=device, non_blocking=True)
        unlabeled_indices = (~role_mask).nonzero(as_tuple=True)[0].to(device=device, non_blocking=True)
        reference_count = len(reference_indices)
        unlabeled_count = len(unlabeled_indices)
        logger.debug(
            "Hoffer regularizer loss batch: "
            f"unlabeled_count={unlabeled_count}, "
            f"reference_count={reference_count}, "
            f"total_count={int(role_tensor.numel())}"
        )

        references = embeddings[reference_indices]
        unlabeled = embeddings[unlabeled_indices]
        if len(references) < 2 or len(unlabeled) == 0:
            return embeddings.sum() * 0.0  # connected zero so backward stays valid

        t0 = _timing_start(device, timings)
        if len(references) % self.reference_sets != 0:
            raise RuntimeError("hoffer_entropy reference count is not divisible by reference_sets")
        num_classes = (
            len(self.class_candidates)
            if self.class_candidates is not None
            else len(references) // self.reference_sets
        )
        expected_references = num_classes * self.reference_sets
        if len(references) != expected_references:
            raise RuntimeError(
                "hoffer_entropy reference batch does not contain every class in every set: "
                f"expected {expected_references}, got {len(references)}"
            )
        reference_sets = references.reshape(num_classes, self.reference_sets, -1).permute(1, 0, 2)
        squared_distances = (
            unlabeled[None, :, None, :] - reference_sets[:, None, :, :]
        ).pow(2).sum(dim=-1)
        log_probabilities = F.log_softmax(-self.distance_scale * squared_distances, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
        _record_timing(timings, "hoffer_entropy", t0, device)
        return entropy.mean()


class RegularizedSemiSupervisedMethod(BaseSemiSupervisedMethod):
    """Compose a configured supervised loss with a registered regularizer."""

    generates_pseudo_labels = False
    is_regularization_method = True
    # How the two loss weights are set, in order of precedence: GradNorm learns
    # them, regularizer_target_ratio periodically calibrates the regularizer
    # weight, and regularizer_weight is the plain fixed value.
    allowed_params = {
        "regularizer",
        "regularizer_params",
        "regularizer_weight",
        "regularizer_target_ratio",
        "regularizer_target_ratio_batches",
        "regularizer_target_ratio_recalibration_interval_samples",
        "regularizer_target_ratio_update_interval_samples",
        "regularizer_target_ratio_probe_memory",
        "regularizer_target_ratio_ema_alpha",
        "regularizer_target_ratio_aggregation",
        "regularizer_target_ratio_statistic",
        "regularizer_target_ratio_statistic_anchor",
        "regularizer_target_ratio_diagnostics",
        "gradient_surgery",
        "grad_norm_alpha",
        "grad_norm_lr",
        "grad_norm_update_interval",
        "grad_norm_renormalize",
        "grad_norm_parameterization",
        "grad_norm_exclude_components",
        "supervised_weight",
    }
    grad_norm_params = (
        "grad_norm_lr",
        "grad_norm_update_interval",
        "grad_norm_renormalize",
        "grad_norm_parameterization",
        "grad_norm_exclude_components",
    )

    def __init__(self, name, default_regularizer=None):
        self.name = name
        self.default_regularizer = default_regularizer

    def resolve_params(self, config):
        params = dict(config.method_params)
        unknown = sorted(set(params) - self.allowed_params)
        if unknown:
            raise ValueError(f"Unknown {self.name} method_params: {unknown}")
        regularizer_name = params.get("regularizer", self.default_regularizer)
        if regularizer_name is None:
            raise ValueError(f"{self.name} requires method_params.regularizer")
        regularizer_params = params.get("regularizer_params", {})
        if not isinstance(regularizer_params, dict):
            raise ValueError(f"{self.name} method_params.regularizer_params must be an object")
        if params.get("regularizer_target_ratio") is None:
            orphaned = sorted(
                name
                for name in (
                    "regularizer_target_ratio_batches",
                    "regularizer_target_ratio_recalibration_interval_samples",
                    "regularizer_target_ratio_update_interval_samples",
                    "regularizer_target_ratio_probe_memory",
                    "regularizer_target_ratio_ema_alpha",
                    "regularizer_target_ratio_aggregation",
                    "regularizer_target_ratio_statistic",
                    "regularizer_target_ratio_statistic_anchor",
                    "regularizer_target_ratio_diagnostics",
                )
                if params.get(name) is not None
            )
            if orphaned:
                raise ValueError(
                    f"{self.name} method_params {orphaned} "
                    "have no effect without regularizer_target_ratio"
                )
        continuous = sorted(
            name
            for name in (
                "regularizer_target_ratio_probe_memory",
                "regularizer_target_ratio_ema_alpha",
            )
            if params.get(name) is not None
        )
        if continuous:
            windowed = sorted(
                name
                for name in (
                    "regularizer_target_ratio_batches",
                    "regularizer_target_ratio_recalibration_interval_samples",
                )
                if params.get(name) is not None
            )
            if windowed:
                raise ValueError(
                    f"{self.name} method_params {continuous} replaces the "
                    f"calibration window, so {windowed} have no effect; configure "
                    "a windowless schedule or the window, not both"
                )
        if (
            params.get("regularizer_target_ratio_ema_alpha") is not None
            and params.get("regularizer_target_ratio_aggregation") is not None
        ):
            raise ValueError(
                f"{self.name} method_params regularizer_target_ratio_ema_alpha "
                "already defines how probes are aggregated, so "
                "regularizer_target_ratio_aggregation has no effect; configure "
                "only one"
            )
        if (
            params.get("regularizer_target_ratio_batches") is not None
            and params.get("regularizer_target_ratio_update_interval_samples")
            is not None
        ):
            raise ValueError(
                f"{self.name} method_params regularizer_target_ratio_batches and "
                "regularizer_target_ratio_update_interval_samples are alternative "
                "ways to schedule probes; configure only one"
            )
        if params.get("grad_norm_alpha") is None:
            unused = sorted(
                name for name in self.grad_norm_params if params.get(name) is not None
            )
            if unused:
                raise ValueError(
                    f"{self.name} method_params {unused} have no effect without grad_norm_alpha"
                )
        return {
            "regularizer_name": regularizer_name,
            "regularizer_params": regularizer_params,
            "regularizer_weight": params.get("regularizer_weight", 1.0),
            "supervised_weight": params.get("supervised_weight", 1.0),
            "regularizer_target_ratio": params.get("regularizer_target_ratio"),
            "regularizer_target_ratio_batches": params.get(
                "regularizer_target_ratio_batches"
            ),
            "regularizer_target_ratio_recalibration_interval_samples": params.get(
                "regularizer_target_ratio_recalibration_interval_samples"
            ),
            "regularizer_target_ratio_update_interval_samples": params.get(
                "regularizer_target_ratio_update_interval_samples"
            ),
            "regularizer_target_ratio_probe_memory": params.get(
                "regularizer_target_ratio_probe_memory"
            ),
            "regularizer_target_ratio_ema_alpha": params.get(
                "regularizer_target_ratio_ema_alpha"
            ),
            "regularizer_target_ratio_aggregation": params.get(
                "regularizer_target_ratio_aggregation"
            ),
            "regularizer_target_ratio_statistic": params.get(
                "regularizer_target_ratio_statistic"
            ),
            "regularizer_target_ratio_statistic_anchor": params.get(
                "regularizer_target_ratio_statistic_anchor"
            ),
            "regularizer_target_ratio_diagnostics": params.get(
                "regularizer_target_ratio_diagnostics",
                False,
            ),
            "gradient_surgery": params.get("gradient_surgery"),
            "grad_norm_alpha": params.get("grad_norm_alpha"),
            "grad_norm_lr": params.get("grad_norm_lr"),
            "grad_norm_update_interval": params.get("grad_norm_update_interval"),
            "grad_norm_renormalize": params.get("grad_norm_renormalize"),
            "grad_norm_parameterization": params.get("grad_norm_parameterization"),
            "grad_norm_exclude_components": params.get("grad_norm_exclude_components"),
        }

    def make_regularizer(self, config):
        params = self.resolve_params(config)
        regularizer_name = params.pop("regularizer_name")
        # Calibration settings configure the trainer, not the regularizer's own
        # constructor, so they never reach the registered class.
        target_ratio = params.pop("regularizer_target_ratio")
        target_ratio_batches = params.pop("regularizer_target_ratio_batches")
        target_ratio_recalibration_interval_samples = params.pop(
            "regularizer_target_ratio_recalibration_interval_samples"
        )
        target_ratio_update_interval_samples = params.pop(
            "regularizer_target_ratio_update_interval_samples"
        )
        target_ratio_probe_memory = params.pop(
            "regularizer_target_ratio_probe_memory"
        )
        target_ratio_ema_alpha = params.pop(
            "regularizer_target_ratio_ema_alpha"
        )
        target_ratio_aggregation = params.pop(
            "regularizer_target_ratio_aggregation"
        )
        target_ratio_statistic = params.pop("regularizer_target_ratio_statistic")
        target_ratio_statistic_anchor = params.pop(
            "regularizer_target_ratio_statistic_anchor"
        )
        target_ratio_diagnostics = params.pop(
            "regularizer_target_ratio_diagnostics"
        )
        surgery_mode = params.pop("gradient_surgery")
        grad_norm_alpha = params.pop("grad_norm_alpha")
        grad_norm_settings = {
            name.removeprefix("grad_norm_"): params.pop(name)
            for name in self.grad_norm_params
        }
        try:
            regularizer_class = REGULARIZER_REGISTRY[regularizer_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown regularizer {regularizer_name!r}. Available: {sorted(REGULARIZER_REGISTRY)}"
            ) from exc
        regularizer_params = params.pop("regularizer_params")
        configured_weight = params["regularizer_weight"]
        regularizer = regularizer_class(**params, **regularizer_params)
        if target_ratio is not None:
            regularizer.set_target_ratio(
                target_ratio,
                calibration_batches=target_ratio_batches,
                recalibration_interval_samples=(
                    target_ratio_recalibration_interval_samples
                ),
                update_interval_samples=target_ratio_update_interval_samples,
                probe_memory=target_ratio_probe_memory,
                ema_alpha=target_ratio_ema_alpha,
                aggregation=target_ratio_aggregation,
                statistic=target_ratio_statistic,
                statistic_anchor=target_ratio_statistic_anchor,
                log_diagnostics=target_ratio_diagnostics,
            )
            if target_ratio_update_interval_samples == 0:
                update_description = "on every regularized optimizer step"
            elif target_ratio_update_interval_samples is not None:
                update_description = (
                    "approximately every "
                    f"{int(target_ratio_update_interval_samples)} samples"
                )
            else:
                probes = (
                    target_ratio_batches
                    if target_ratio_batches is not None
                    else interfaces.DEFAULT_TARGET_RATIO_BATCHES
                )
                update_description = f"at {int(probes)} probes per window"
            aggregation = regularizer.regularizer_target_ratio_aggregation
            if target_ratio_ema_alpha is not None:
                estimator_description = (
                    "an exponential moving average with "
                    f"alpha={float(target_ratio_ema_alpha):g}"
                )
            elif target_ratio_probe_memory is not None:
                estimator_description = (
                    f"a sliding {aggregation} over the last "
                    f"{int(target_ratio_probe_memory)} probes"
                )
            else:
                window_samples = (
                    target_ratio_recalibration_interval_samples
                    if target_ratio_recalibration_interval_samples is not None
                    else interfaces.DEFAULT_TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES
                )
                estimator_description = (
                    f"a running {aggregation} reset every "
                    f"{int(window_samples)} samples"
                )
            statistic = regularizer.regularizer_target_ratio_statistic
            statistic_description = (
                "Euclidean gradient norms"
                if statistic == "l2"
                else (
                    f"the {statistic!r} gradient statistic (max over the supervised "
                    "gradient, mean over the regularizer one), whose balance point is "
                    "not the Euclidean one -- the same target ratio corresponds to a "
                    "roughly tenfold larger L2 ratio on a head of this size"
                )
            )
            logger.info(
                f"{regularizer.name}: regularizer_target_ratio={float(target_ratio):g} "
                f"overrides the configured regularizer_weight={float(configured_weight):g}; "
                "the weight is calibrated from measured gradient norms once training "
                f"reaches the regularized phase, updated {update_description}, using "
                f"{estimator_description} of {statistic_description}"
            )
        if surgery_mode is not None:
            regularizer.set_gradient_surgery(surgery_mode)
            grad_surgery.log_gradient_surgery_configuration(regularizer)
        if grad_norm_alpha is not None:
            superseded = [
                "regularizer_target_ratio"
                if target_ratio is not None
                else "regularizer_weight"
            ]
            # Named before set_grad_norm clears the ones it takes over.
            ratios_before = sorted(regularizer.extra_target_ratios())
            regularizer.set_grad_norm(grad_norm_alpha, **grad_norm_settings)
            surviving = set(regularizer.extra_target_ratios())
            superseded.extend(name for name in ratios_before if name not in surviving)
            starting = ", ".join(
                (
                    f"supervised={regularizer.supervised_weight:g}",
                    f"regularizer={regularizer.regularizer_weight:g}",
                    *(
                        f"{name}={weight:g}"
                        for name, weight in regularizer.grad_norm_balanced_components().items()
                    ),
                )
            )
            logger.info(
                f"{regularizer.name}: grad_norm_alpha={float(grad_norm_alpha):g} "
                f"supersedes {', '.join(superseded)}; every loss weight is learned "
                f"during training and starts from the configured values ({starting})"
            )
            excluded = regularizer.grad_norm_exclude_components
            if excluded:
                logger.info(
                    f"{regularizer.name}: {', '.join(excluded)} excluded from GradNorm "
                    "balancing; each keeps its configured weight, or its target-ratio "
                    "calibration where it declares one"
                )
        regularizer.configure_graph_batching(config)
        return regularizer

    def validate_config(self, config, source=""):
        try:
            self.make_regularizer(config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid regularization method configuration{source}: {exc}") from exc


class SklearnGraphSSLMethod(BaseSemiSupervisedMethod):
    """Adapter for sklearn graph-based label propagation/spreading."""

    def __init__(self, name, estimator_cls, default_params):
        self.name = name
        self.estimator_cls = estimator_cls
        self.default_params = default_params

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        """Fit a graph SSL estimator on embeddings and predict unlabeled nodes."""

        if len(split.unlabeled_positions) == 0:
            # Preserve the normal result shape/typing so the later filtering and
            # merging pipeline does not need a separate no-unlabeled branch.
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )

        # train_dataset.labels already contains dense mapped labels. Graph SSL
        # therefore predicts directly in the label space used for training.
        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        # The labeled prefix followed by the unlabeled suffix is an important
        # ordering contract used again after estimator.fit.
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        # sklearn recognizes -1 as the unknown target.  Labeled and unlabeled
        # embeddings are concatenated in the same order as ssl_targets.
        ssl_targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )

        # Embeddings are extracted in ssl_positions order, so each feature row
        # lines up with the target at the same ssl_targets offset.
        features = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        # Copy defaults before applying config overrides so the shared registry
        # method object is not mutated between runs.
        params = dict(self.default_params)
        params.update(config.method_params)
        logger.info(f"Fitting {self.name} with params: {params}")
        estimator = self.estimator_cls(**params)
        # The estimator builds a graph over all feature rows and propagates the
        # known prefix labels into rows marked with UNLABELED_TARGET.
        estimator.fit(features, ssl_targets)

        # transduction_ includes predictions for the labeled prefix as well, so
        # retain only the rows corresponding to the unlabeled suffix.
        unlabeled_start = len(split.labeled_positions)
        pseudo_labels = np.asarray(estimator.transduction_[unlabeled_start:], dtype=np.int64)
        distributions = getattr(estimator, "label_distributions_", None)
        confidences = None
        if distributions is not None:
            # Use the highest class probability as a scalar confidence for
            # threshold filtering. Some estimators may not expose distributions.
            confidences = np.asarray(distributions[unlabeled_start:].max(axis=1), dtype=np.float32)

        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=confidences,
        )


class FaissKNNMajorityVotePseudoLabeler(BaseSemiSupervisedMethod):
    """Assign each unlabeled embedding the majority label of labeled neighbors."""

    def __init__(self, name, n_neighbors=10):
        self.name = name
        self.n_neighbors = n_neighbors
        self._embedding_loader = None

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        if len(split.unlabeled_positions) == 0:
            # Return an empty, correctly typed result that downstream code can
            # concatenate/filter without special handling.
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")

        # n_neighbors is the only accepted method-specific option. pop removes
        # it so any remaining keys can be reported as unsupported.
        params = dict(config.method_params)
        n_neighbors = int(params.pop("n_neighbors", self.n_neighbors))
        if params:
            raise ValueError(f"Unknown {self.name} params: {sorted(params)}")
        if n_neighbors <= 0:
            raise ValueError(f"{self.name} n_neighbors must be positive")

        faiss = require_faiss(self.name)

        # FAISS cannot retrieve more labeled neighbors than exist.
        k = min(n_neighbors, len(split.labeled_positions))
        if k < n_neighbors:
            logger.warning(
                f"{self.name} requested n_neighbors={n_neighbors}, but only "
                f"{len(split.labeled_positions)} labeled samples are available; using {k}"
            )

        # Extract all embeddings in one deterministic pass. The labeled prefix
        # is indexed; the unlabeled suffix becomes the query matrix.
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        embeddings = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        # Split the embedding matrix at the same boundary used to build
        # ssl_positions. FAISS expects contiguous float32 arrays.
        num_labeled = len(split.labeled_positions)
        labeled_embeddings = np.ascontiguousarray(embeddings[:num_labeled], dtype=np.float32)
        unlabeled_embeddings = np.ascontiguousarray(embeddings[num_labeled:], dtype=np.float32)

        # Only labels corresponding to indexed/labeled embeddings are visible to
        # the method. Labels of query samples remain unused.
        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        labeled_targets = labels[split.labeled_positions]

        # Inner product between L2-normalized vectors is cosine similarity.
        faiss.normalize_L2(labeled_embeddings)
        faiss.normalize_L2(unlabeled_embeddings)

        # IndexFlatIP performs exact inner-product search. After normalization,
        # the returned similarity is cosine similarity.
        # neighbor_indices has shape [num_unlabeled, k] and contains row offsets
        # into labeled_embeddings/labeled_targets.
        similarities, neighbor_indices = faiss_flat_ip_search(
            database=labeled_embeddings,
            queries=unlabeled_embeddings,
            k=k,
            purpose=self.name,
            faiss_module=faiss,
        )
        request = make_graph_diagnostics_request(
            config=config,
            log_dir=log_dir,
            name=f"{self.name}_labeled_knn",
            epoch=epoch,
            title=f"{self.name} unlabeled-to-labeled FAISS kNN graph",
        )
        if request is not None:
            graph_rows = np.repeat(np.arange(len(split.unlabeled_positions), dtype=np.int64) + num_labeled, k)
            graph_cols = neighbor_indices.reshape(-1).astype(np.int64)
            graph_values = np.maximum(similarities.reshape(-1).astype(np.float64), 0.0)
            graph = sparse.coo_matrix(
                (graph_values, (graph_rows, graph_cols)),
                shape=(len(ssl_positions), len(ssl_positions)),
                dtype=np.float64,
            ).tocsr()
            graph = (graph + graph.T).tocsr()
            maybe_save_graph_diagnostics(
                request=request,
                embeddings=np.vstack([labeled_embeddings, unlabeled_embeddings]),
                adjacency=graph,
                positions=ssl_positions,
                labels=labels[ssl_positions],
                known_mask=np.arange(len(ssl_positions)) < num_labeled,
                graph_metadata={
                    "graph_kind": "unlabeled_to_labeled_positive_cosine_knn",
                    "requested_n_neighbors": n_neighbors,
                    "search_n_neighbors": k,
                    "neighbor_indices": neighbor_indices,
                    "neighbor_similarities": similarities,
                    "query_indices": (
                        np.arange(
                            len(split.unlabeled_positions),
                            dtype=np.int64,
                        )
                        + num_labeled
                    ),
                },
            )

        # Advanced indexing turns neighbor row offsets into a label matrix with
        # the same [num_unlabeled, k] shape.
        neighbor_labels = labeled_targets[neighbor_indices]

        if k == 1:
            pseudo_labels = neighbor_labels[:, 0]
            confidences = similarities[:, 0].astype(np.float32)
        else:
            # For k > 1 confidence is the winning vote fraction rather than a
            # distance-derived score.
            pseudo_labels, vote_counts = majority_vote(neighbor_labels)
            confidences = (vote_counts / k).astype(np.float32)
        if request is not None:
            num_classes = int(labels.max()) + 1
            diagnostic_scores = np.zeros(
                (len(ssl_positions), num_classes),
                dtype=np.float64,
            )
            labeled_rows = np.arange(num_labeled, dtype=np.int64)
            diagnostic_scores[
                labeled_rows,
                labeled_targets,
            ] = 1.0
            unlabeled_scores = diagnostic_scores[num_labeled:]
            np.add.at(
                unlabeled_scores,
                (
                    np.repeat(
                        np.arange(len(unlabeled_scores), dtype=np.int64),
                        k,
                    ),
                    neighbor_labels.reshape(-1),
                ),
                1.0 / float(k),
            )
            diagnostic_confidences = np.ones(
                len(ssl_positions),
                dtype=np.float64,
            )
            diagnostic_confidences[num_labeled:] = confidences
            maybe_update_graph_propagation_diagnostics(
                request=request,
                scores=diagnostic_scores,
                confidences=diagnostic_confidences,
                labels=labels[ssl_positions],
                known_mask=np.arange(len(ssl_positions)) < num_labeled,
                method=self.name,
                confidence_threshold=config.confidence_threshold,
                extra={
                    "vote_neighbor_count": k,
                    "confidence_kind": (
                        "cosine_similarity"
                        if k == 1
                        else "winning_vote_fraction"
                    ),
                },
            )
        logger.info(f"{self.name} confidence distribution: {summarize_numeric_values(confidences)}")

        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=confidences,
        )


class SelfTrainingKNNPseudoLabeler(BaseSemiSupervisedMethod):
    """Cumulative squared-Euclidean 1-NN wrapper from Sahito et al. Algorithm 1.

    Each pseudo-label refresh is one meta-iteration. Newly selected examples
    remain promoted on later refreshes, become 1-NN reference points with their
    fixed pseudo-labels, and are removed from the candidate pool. The outer
    training loop supplies the paper's retraining step between refreshes via
    ``update_mode`` and ``update_interval_epochs``.
    """

    DEFAULT_PARAMS = {
        "selection_fraction": 0.05,
        "selection_strategy": "per_predicted_class",
        "max_meta_iterations": 25,
    }

    def __init__(self, name="self_training_knn"):
        self.name = name
        self.reset_state()

    def reset_state(self):
        """Discard pseudo-label promotions from a previous training run."""

        self._state_signature = None
        self._promoted_positions = np.empty(0, dtype=np.int64)
        self._promoted_labels = np.empty(0, dtype=np.int64)
        self._promoted_confidences = np.empty(0, dtype=np.float32)
        self._meta_iteration = 0
        self._has_generated = False
        self._last_generation_epoch = None

    def resolve_params(self, config):
        params = dict(self.DEFAULT_PARAMS)
        unknown = sorted(set(config.method_params) - set(params))
        if unknown:
            raise ValueError(f"Unknown {self.name} params: {unknown}")
        params.update(config.method_params)

        selection_fraction = params["selection_fraction"]
        if isinstance(selection_fraction, bool):
            raise ValueError(f"{self.name} selection_fraction must be in (0, 1]")
        selection_fraction = float(selection_fraction)
        if not np.isfinite(selection_fraction) or not (0.0 < selection_fraction <= 1.0):
            raise ValueError(f"{self.name} selection_fraction must be in (0, 1]")

        selection_strategy = str(params["selection_strategy"])
        if selection_strategy not in {"global", "per_predicted_class"}:
            raise ValueError(
                f"{self.name} selection_strategy must be one of "
                "['global', 'per_predicted_class']"
            )

        max_meta_iterations = params["max_meta_iterations"]
        if isinstance(max_meta_iterations, bool) or not isinstance(
            max_meta_iterations,
            (int, np.integer),
        ):
            raise ValueError(f"{self.name} max_meta_iterations must be a positive integer")
        max_meta_iterations = int(max_meta_iterations)
        if max_meta_iterations <= 0:
            raise ValueError(f"{self.name} max_meta_iterations must be a positive integer")

        return {
            "selection_fraction": selection_fraction,
            "selection_strategy": selection_strategy,
            "max_meta_iterations": max_meta_iterations,
        }

    def validate_config(self, config, source=""):
        try:
            self.resolve_params(config)
        except ValueError as exc:
            raise ValueError(f"Invalid {self.name} configuration{source}: {exc}") from exc

    def _current_result(self):
        return PseudoLabelResult(
            positions=self._promoted_positions,
            mapped_labels=self._promoted_labels,
            confidences=self._promoted_confidences,
        )

    def _prepare_generation(self, train_dataset, split, epoch):
        """Reset on a new dataset/split and make repeated epoch calls idempotent."""

        signature = (id(train_dataset), id(split))
        if self._state_signature != signature:
            self.reset_state()
            self._state_signature = signature
        elif (
            self._has_generated
            and epoch is not None
            and self._last_generation_epoch is not None
            and int(epoch) < int(self._last_generation_epoch)
        ):
            # This mainly protects direct registry users. The training engine
            # creates a fresh method instance for every fold/run.
            self.reset_state()
            self._state_signature = signature

        return self._has_generated and epoch == self._last_generation_epoch

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        params = self.resolve_params(config)
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")
        if self._prepare_generation(train_dataset, split, epoch):
            return self._current_result()

        if self._meta_iteration >= params["max_meta_iterations"]:
            logger.info(
                f"{self.name} reached max_meta_iterations="
                f"{params['max_meta_iterations']}; reusing "
                f"{len(self._promoted_positions)} promoted pseudo-labels"
            )
            self._has_generated = True
            self._last_generation_epoch = epoch
            return self._current_result()

        labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        remaining_mask = ~np.isin(
            unlabeled_positions,
            self._promoted_positions,
            assume_unique=False,
        )
        remaining_positions = unlabeled_positions[remaining_mask]
        if len(remaining_positions) == 0:
            logger.info(f"{self.name} has promoted the entire unlabeled pool")
            self._has_generated = True
            self._last_generation_epoch = epoch
            return self._current_result()

        support_positions = np.concatenate(
            [labeled_positions, self._promoted_positions]
        )
        ssl_positions = np.concatenate([support_positions, remaining_positions])
        embeddings = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} meta-iteration {self._meta_iteration + 1} embeddings",
        )
        num_support = len(support_positions)
        support_embeddings = np.ascontiguousarray(
            embeddings[:num_support],
            dtype=np.float32,
        )
        candidate_embeddings = np.ascontiguousarray(
            embeddings[num_support:],
            dtype=np.float32,
        )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        support_labels = np.concatenate(
            [labels[labeled_positions], self._promoted_labels]
        )
        faiss = require_faiss(self.name)
        # Ordinary retrieval embeddings are L2-normalized by DinoWrapper. Keep
        # that invariant explicit for alternate callers, then reuse the common
        # exact inner-product search. For unit vectors, squared Euclidean
        # distance is exactly 2 - 2 * cosine similarity.
        faiss.normalize_L2(support_embeddings)
        faiss.normalize_L2(candidate_embeddings)
        similarities, neighbor_indices = faiss_flat_ip_search(
            database=support_embeddings,
            queries=candidate_embeddings,
            k=1,
            purpose=self.name,
            faiss_module=faiss,
        )
        nearest_distances = np.clip(
            2.0 - 2.0 * np.asarray(similarities[:, 0], dtype=np.float64),
            0.0,
            4.0,
        )
        nearest_indices = np.asarray(neighbor_indices[:, 0], dtype=np.int64)
        if np.any((nearest_indices < 0) | (nearest_indices >= len(support_labels))):
            raise RuntimeError(f"{self.name} returned an invalid labeled-neighbor index")
        predicted_labels = support_labels[nearest_indices]

        # Algorithm 1 ranks by raw squared Euclidean distance. The common SSL
        # contract expects confidence to increase with certainty and to lie in
        # [0, 1], so expose the monotone inverse-distance transform.
        confidences = (1.0 / (1.0 + nearest_distances)).astype(np.float32)
        selected_indices = select_self_training_candidates(
            distances=nearest_distances,
            predicted_labels=predicted_labels,
            selection_fraction=params["selection_fraction"],
            strategy=params["selection_strategy"],
            eligible=confidences >= float(config.confidence_threshold),
        )

        selected_positions = remaining_positions[selected_indices]
        selected_labels = predicted_labels[selected_indices].astype(np.int64, copy=False)
        selected_confidences = confidences[selected_indices]
        self._promoted_positions = np.concatenate(
            [self._promoted_positions, selected_positions]
        )
        self._promoted_labels = np.concatenate(
            [self._promoted_labels, selected_labels]
        )
        self._promoted_confidences = np.concatenate(
            [self._promoted_confidences, selected_confidences]
        )
        self._meta_iteration += 1
        self._has_generated = True
        self._last_generation_epoch = epoch

        logger.info(
            f"{self.name} meta-iteration {self._meta_iteration}/"
            f"{params['max_meta_iterations']}: selected {len(selected_positions)} "
            f"of {len(remaining_positions)} remaining samples; "
            f"{len(self._promoted_positions)} promoted in total"
        )
        logger.info(
            f"{self.name} 1-NN squared-distance distribution: "
            f"{summarize_numeric_values(nearest_distances)}"
        )
        if len(selected_positions) == 0:
            logger.warning(
                f"{self.name} selected no samples: the integer selection budget, "
                "per-class allocation, or confidence threshold excluded every candidate"
            )
        return self._current_result()


class FaissLabelSpreadingPseudoLabeler(BaseSemiSupervisedMethod):
    """Zhou et al. label spreading solved directly on a FAISS kNN graph."""

    DEFAULT_PARAMS = {
        "n_neighbors": 10,
        "gamma": 1.0,
        "alpha": 0.2,
        "cg_rtol": 1e-5,
        "cg_max_iter": 1000,
        "linear_solver": "cg",
    }

    def __init__(self, name="faiss_label_spreading"):
        self.name = name

    def validate_config(self, config, source=""):
        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        try:
            validate_faiss_label_spreading_params(params)
        except ValueError as exc:
            raise ValueError(f"Invalid {self.name} configuration{source}: {exc}") from exc

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        if len(split.unlabeled_positions) == 0:
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        validate_faiss_label_spreading_params(params)
        logger.info(f"Running {self.name} with params: {params}")
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])

        features = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings"
        )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )
        probabilities, confidences = faiss_label_spreading(
            features=features,
            targets=targets,
            num_classes=int(labels.max()) + 1,
            graph_diagnostics={
                "request": make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_affinity",
                    epoch=epoch,
                    title=f"{self.name} symmetric affinity graph",
                ),
                "positions": ssl_positions,
                "labels": labels[ssl_positions],
                "known_mask": targets != UNLABELED_TARGET,
                "confidence_threshold": config.confidence_threshold,
            },
            **params,
        )
        unlabeled_start = len(split.labeled_positions)
        unlabeled_probabilities = probabilities[unlabeled_start:]
        pseudo_labels = np.argmax(unlabeled_probabilities, axis=1).astype(np.int64)
        unlabeled_confidences = confidences[unlabeled_start:].astype(np.float32)
        logger.info(f"{self.name} confidence distribution: {summarize_numeric_values(unlabeled_confidences)}")
        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=unlabeled_confidences,
        )


class IscenLabelSpreadingPseudoLabeler(BaseSemiSupervisedMethod):
    """LP-DeepSSL label propagation from Iscen et al., CVPR 2019."""

    # The whole refresh is one build_ssl_training_dataset call, so the engine can
    # repeat it inside an epoch on a sample-count schedule.
    supports_sample_scoped_refresh = True

    DEFAULT_PARAMS = {
        "n_neighbors": 50,
        "gamma": 3.0,
        "alpha": 0.99,
        "cg_rtol": 1e-6,
        "cg_max_iter": 20,
        "linear_solver": "cg",
    }

    def __init__(self, name="iscen_label_spreading"):
        self.name = name

    def validate_config(self, config, source=""):
        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        try:
            validate_iscen_label_spreading_params(params)
        except ValueError as exc:
            raise ValueError(f"Invalid {self.name} configuration{source}: {exc}") from exc

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        if len(split.unlabeled_positions) == 0:
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        validate_iscen_label_spreading_params(params)
        logger.info(f"Running {self.name} with params: {params}")

        # The common graph-method ordering contract makes the unlabeled output
        # a direct suffix slice after propagation.
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        features = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=0,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )
        probabilities, confidences = iscen_label_spreading(
            features=features,
            targets=targets,
            num_classes=int(labels.max()) + 1,
            graph_diagnostics={
                "request": make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_affinity",
                    epoch=epoch,
                    title=f"{self.name} LP-DeepSSL affinity graph",
                ),
                "positions": ssl_positions,
                "labels": labels[ssl_positions],
                "known_mask": targets != UNLABELED_TARGET,
                "confidence_threshold": config.confidence_threshold,
            },
            **params,
        )
        unlabeled_start = len(split.labeled_positions)
        unlabeled_probabilities = probabilities[unlabeled_start:]
        propagated_mask = unlabeled_probabilities.sum(axis=1) > 0.0
        omitted_count = int((~propagated_mask).sum())
        if omitted_count > 0:
            logger.warning(
                f"{self.name} omitted {omitted_count} unlabeled candidates with no "
                "propagated class mass; they will not enter pseudo-label training"
            )
        unlabeled_probabilities = unlabeled_probabilities[propagated_mask]
        pseudo_labels = np.argmax(unlabeled_probabilities, axis=1).astype(np.int64)
        unlabeled_confidences = confidences[unlabeled_start:][propagated_mask].astype(np.float32)
        logger.info(
            f"{self.name} entropy-certainty distribution: "
            f"{summarize_numeric_values(unlabeled_confidences)}"
        )
        return PseudoLabelResult(
            positions=split.unlabeled_positions[propagated_mask],
            mapped_labels=pseudo_labels,
            confidences=unlabeled_confidences,
        )


class MixedLabelPropagationPseudoLabeler(BaseSemiSupervisedMethod):
    """Sparse mixed label propagation from Zhuang and Moulin, CVPR 2023."""

    DEFAULT_PARAMS = {
        "n_neighbors": 50,
        "gamma": 3.0,
        "temperature": 4.0,
        "beta": 1.0,
        "mu": 1.0 / 99.0,
        "cg_rtol": 1e-5,
        "cg_max_iter": 1000,
        "edge_batch_size": 65536,
        "linear_solver": "cg",
    }

    def __init__(self, name="mixed_label_propagation"):
        self.name = name

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        if len(split.unlabeled_positions) == 0:
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        validate_mixed_label_propagation_params(params)
        logger.info(f"Running {self.name} with params: {params}")
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        features = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )
        num_classes = int(labels.max()) + 1
        normalized_scores, confidences = mixed_label_propagation(
            features=features,
            targets=targets,
            num_classes=num_classes,
            graph_diagnostics={
                "request": make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_affinity",
                    epoch=epoch,
                    title=f"{self.name} symmetric affinity graph",
                ),
                "positions": ssl_positions,
                "labels": labels[ssl_positions],
                "known_mask": targets != UNLABELED_TARGET,
                "confidence_threshold": config.confidence_threshold,
            },
            **params,
        )
        unlabeled_start = len(split.labeled_positions)
        unlabeled_scores = normalized_scores[unlabeled_start:]
        pseudo_labels = np.argmax(unlabeled_scores, axis=1).astype(np.int64)
        unlabeled_confidences = confidences[unlabeled_start:].astype(np.float32)
        logger.info(f"{self.name} confidence distribution: {summarize_numeric_values(unlabeled_confidences)}")
        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=unlabeled_confidences,
        )


def validate_mixed_label_propagation_params(params):
    unknown = sorted(set(params) - set(MixedLabelPropagationPseudoLabeler.DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"Unknown mixed_label_propagation params: {unknown}")
    if int(params["n_neighbors"]) <= 0:
        raise ValueError("mixed_label_propagation n_neighbors must be positive")
    for name in ("gamma", "temperature", "beta", "mu", "cg_rtol"):
        if float(params[name]) <= 0:
            raise ValueError(f"mixed_label_propagation {name} must be positive")
    if int(params["cg_max_iter"]) <= 0:
        raise ValueError("mixed_label_propagation cg_max_iter must be positive")
    if int(params["edge_batch_size"]) <= 0:
        raise ValueError("mixed_label_propagation edge_batch_size must be positive")
    try:
        normalize_linear_solver(params["linear_solver"])
    except ValueError as exc:
        raise ValueError(f"mixed_label_propagation {exc}") from exc


def validate_faiss_label_spreading_params(params):
    unknown = sorted(set(params) - set(FaissLabelSpreadingPseudoLabeler.DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"Unknown faiss_label_spreading params: {unknown}")
    if int(params["n_neighbors"]) <= 0:
        raise ValueError("faiss_label_spreading n_neighbors must be positive")
    for name in ("gamma", "cg_rtol"):
        if not np.isfinite(float(params[name])) or float(params[name]) <= 0:
            raise ValueError(f"faiss_label_spreading {name} must be positive")
    alpha = float(params["alpha"])
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("faiss_label_spreading alpha must be in (0, 1)")
    if int(params["cg_max_iter"]) <= 0:
        raise ValueError("faiss_label_spreading cg_max_iter must be positive")
    try:
        normalize_linear_solver(params["linear_solver"])
    except ValueError as exc:
        raise ValueError(f"faiss_label_spreading {exc}") from exc


def validate_iscen_label_spreading_params(params):
    unknown = sorted(set(params) - set(IscenLabelSpreadingPseudoLabeler.DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"Unknown iscen_label_spreading params: {unknown}")
    if int(params["n_neighbors"]) <= 0:
        raise ValueError("iscen_label_spreading n_neighbors must be positive")
    for name in ("gamma", "cg_rtol"):
        if not np.isfinite(float(params[name])) or float(params[name]) <= 0:
            raise ValueError(f"iscen_label_spreading {name} must be positive")
    alpha = float(params["alpha"])
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("iscen_label_spreading alpha must be in (0, 1)")
    if int(params["cg_max_iter"]) <= 0:
        raise ValueError("iscen_label_spreading cg_max_iter must be positive")
    try:
        normalize_linear_solver(params["linear_solver"])
    except ValueError as exc:
        raise ValueError(f"iscen_label_spreading {exc}") from exc


GRAPH_PSEUDO_LABEL_METHODS = frozenset(
    {
        "faiss_majority_vote_knn",
        "sklearn_label_spreading",
        "sklearn_label_propagation",
        "faiss_label_spreading",
        "iscen_label_spreading",
        "mixed_label_propagation",
    }
)


def _in_batch_graph_diagnostics(
    regularizer,
    *,
    graph_positions,
    graph_targets,
):
    request = getattr(regularizer, "_in_batch_graph_diagnostics_request", None)
    # At most the first graph of each epoch gets an artifact bundle.
    regularizer._in_batch_graph_diagnostics_request = None
    if request is None or graph_positions is None:
        return None
    return {
        "request": request,
        "positions": graph_positions,
        "labels": regularizer._train_dataset_labels[graph_positions],
        "known_mask": np.asarray(
            graph_targets.detach().cpu(),
            dtype=np.int64,
        )
        != UNLABELED_TARGET,
        "confidence_threshold": regularizer.confidence_threshold,
    }


def _run_graph_pseudo_label_batch(
    method,
    *,
    features,
    targets,
    num_classes,
    method_params,
    graph_diagnostics,
):
    """Apply one registered graph pseudo-labeler to an already embedded batch."""

    features = np.ascontiguousarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    labeled_count = int(np.sum(targets != UNLABELED_TARGET))
    if labeled_count <= 0 or labeled_count >= len(targets):
        raise ValueError("an in-batch pseudo-label graph needs labeled and unlabeled nodes")
    if (
        np.any(targets[:labeled_count] == UNLABELED_TARGET)
        or np.any(targets[labeled_count:] != UNLABELED_TARGET)
    ):
        raise ValueError(
            "in-batch graph targets must contain the labeled prefix followed by "
            "the unlabeled suffix"
        )
    unlabeled_slice = slice(labeled_count, None)

    if isinstance(method, SklearnGraphSSLMethod):
        params = dict(method.default_params)
        params.update(method_params)
        if params.get("kernel") == "knn":
            requested_neighbors = int(params.get("n_neighbors", 10))
            params["n_neighbors"] = min(
                requested_neighbors,
                max(1, len(features) - 1),
            )
        estimator = method.estimator_cls(**params)
        estimator.fit(features, targets)
        pseudo_labels = np.asarray(
            estimator.transduction_[unlabeled_slice],
            dtype=np.int64,
        )
        distributions = getattr(estimator, "label_distributions_", None)
        confidences = (
            np.ones(len(pseudo_labels), dtype=np.float32)
            if distributions is None
            else np.asarray(
                distributions[unlabeled_slice].max(axis=1),
                dtype=np.float32,
            )
        )
        keep = (pseudo_labels >= 0) & (pseudo_labels < int(num_classes))
        return pseudo_labels, confidences, keep

    if isinstance(method, FaissKNNMajorityVotePseudoLabeler):
        params = dict(method_params)
        n_neighbors = int(params.pop("n_neighbors", method.n_neighbors))
        if params:
            raise ValueError(f"Unknown {method.name} params: {sorted(params)}")
        if n_neighbors <= 0:
            raise ValueError(f"{method.name} n_neighbors must be positive")
        faiss = require_faiss(method.name)
        labeled_features = features[:labeled_count].copy()
        unlabeled_features = features[labeled_count:].copy()
        faiss.normalize_L2(labeled_features)
        faiss.normalize_L2(unlabeled_features)
        k = min(n_neighbors, labeled_count)
        similarities, neighbor_indices = faiss_flat_ip_search(
            database=labeled_features,
            queries=unlabeled_features,
            k=k,
            purpose=method.name,
            faiss_module=faiss,
            prefer_gpu=False,
        )
        neighbor_labels = targets[:labeled_count][neighbor_indices]
        if k == 1:
            pseudo_labels = neighbor_labels[:, 0]
            confidences = similarities[:, 0]
        else:
            pseudo_labels, vote_counts = majority_vote(neighbor_labels)
            confidences = vote_counts / float(k)
        # Confidence is used as an optional sample weight by some metric losses.
        confidences = np.clip(confidences, 0.0, 1.0).astype(np.float32)

        if graph_diagnostics is not None and graph_diagnostics["request"] is not None:
            query_rows = (
                np.repeat(
                    np.arange(len(unlabeled_features), dtype=np.int64)
                    + labeled_count,
                    k,
                )
            )
            neighbor_cols = neighbor_indices.reshape(-1).astype(np.int64)
            values = np.maximum(
                similarities.reshape(-1).astype(np.float64),
                0.0,
            )
            positive = values > 0.0
            if np.any(positive):
                adjacency = sparse.coo_matrix(
                    (
                        values[positive],
                        (query_rows[positive], neighbor_cols[positive]),
                    ),
                    shape=(len(features), len(features)),
                    dtype=np.float64,
                ).tocsr()
                adjacency = (adjacency + adjacency.T).tocsr()
                maybe_save_graph_diagnostics(
                    request=graph_diagnostics["request"],
                    embeddings=np.vstack([labeled_features, unlabeled_features]),
                    adjacency=adjacency,
                    positions=graph_diagnostics["positions"],
                    labels=graph_diagnostics["labels"],
                    known_mask=graph_diagnostics["known_mask"],
                    graph_metadata={
                        "graph_kind": "in_batch_unlabeled_to_labeled_cosine_knn",
                        "requested_n_neighbors": n_neighbors,
                        "search_n_neighbors": k,
                        "neighbor_indices": neighbor_indices,
                        "neighbor_similarities": similarities,
                        "query_indices": np.arange(
                            len(unlabeled_features),
                            dtype=np.int64,
                        )
                        + labeled_count,
                    },
                )
        keep = (pseudo_labels >= 0) & (pseudo_labels < int(num_classes))
        return (
            np.asarray(pseudo_labels, dtype=np.int64),
            confidences,
            keep,
        )

    # Only classes represented by labeled nodes can receive propagated mass.
    # Removing the identically-zero right-hand sides keeps a local graph cheap
    # even when the dataset has thousands of global classes. This compaction
    # must be independent of diagnostics: otherwise saving the first graph of
    # an epoch changes Iscen's entropy denominator and therefore training.
    local_class_lookup = np.unique(targets[:labeled_count]).astype(np.int64)
    algorithm_targets = targets.copy()
    algorithm_targets[:labeled_count] = np.searchsorted(
        local_class_lookup,
        targets[:labeled_count],
    )
    algorithm_num_classes = int(len(local_class_lookup))
    diagnostics = graph_diagnostics
    if diagnostics is not None:
        diagnostics = dict(diagnostics)
        # Propagation operates in the compact class space, while diagnostic
        # artifacts should continue to display dataset-wide mapped labels.
        diagnostics["score_class_labels"] = local_class_lookup
    if isinstance(method, FaissLabelSpreadingPseudoLabeler):
        params = dict(method.DEFAULT_PARAMS)
        params.update(method_params)
        validate_faiss_label_spreading_params(params)
        probabilities, confidences = faiss_label_spreading(
            features=features,
            targets=algorithm_targets,
            num_classes=algorithm_num_classes,
            graph_diagnostics=diagnostics,
            _dependencies=IN_BATCH_GRAPH_DEPENDENCIES,
            **params,
        )
    elif isinstance(method, IscenLabelSpreadingPseudoLabeler):
        params = dict(method.DEFAULT_PARAMS)
        params.update(method_params)
        validate_iscen_label_spreading_params(params)
        probabilities, confidences = iscen_label_spreading(
            features=features,
            targets=algorithm_targets,
            num_classes=algorithm_num_classes,
            graph_diagnostics=diagnostics,
            _dependencies=IN_BATCH_GRAPH_DEPENDENCIES,
            **params,
        )
    elif isinstance(method, MixedLabelPropagationPseudoLabeler):
        params = dict(method.DEFAULT_PARAMS)
        params.update(method_params)
        validate_mixed_label_propagation_params(params)
        probabilities, confidences = mixed_label_propagation(
            features=features,
            targets=algorithm_targets,
            num_classes=algorithm_num_classes,
            graph_diagnostics=diagnostics,
            _dependencies=IN_BATCH_GRAPH_DEPENDENCIES,
            **params,
        )
    else:
        raise TypeError(f"{type(method).__name__} is not an in-batch graph method")

    unlabeled_probabilities = probabilities[unlabeled_slice]
    unlabeled_confidences = np.asarray(
        confidences[unlabeled_slice],
        dtype=np.float32,
    )
    positive_mass = np.asarray(
        unlabeled_probabilities.sum(axis=1) > 0.0,
        dtype=bool,
    )
    pseudo_labels = np.argmax(unlabeled_probabilities, axis=1).astype(np.int64)
    pseudo_labels = local_class_lookup[pseudo_labels]
    keep = (
        positive_mass
        & (pseudo_labels >= 0)
        & (pseudo_labels < int(num_classes))
    )
    return pseudo_labels, unlabeled_confidences, keep


class InBatchGraphPseudoLabelRegularizer(BaseTrainingRegularizer):
    """Run a graph pseudo-labeler and its metric loss inside each training batch."""

    supports_frozen_feature_precompute = True
    uses_joint_forward = True
    requires_labeled_indices = True
    requires_supervised_objective = True

    def __init__(self, method, config):
        super().__init__(regularizer_weight=1.0, supervised_weight=1.0)
        self.method = method
        self.name = method.name
        self.method_params = dict(config.method_params)
        self.graph_labeled_batch_size = int(config.graph_labeled_batch_size)
        self.graph_unlabeled_batch_size = int(config.graph_unlabeled_batch_size)
        self.confidence_threshold = float(config.confidence_threshold)
        self.dataset = None
        self.num_classes = None
        self._labeled_positions = None
        self._train_dataset_labels = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._in_batch_graph_diagnostics_request = None
        self._last_diagnostics = {}

    def validate_run_args(self, args):
        if self.graph_labeled_batch_size > int(args.batch_size):
            raise ValueError(
                f"{self.method.name} graph_labeled_batch_size cannot exceed the "
                f"supervised batch_size ({self.graph_labeled_batch_size} > {args.batch_size})"
            )

    def configure_model(
        self,
        student_model,
        train_dataset,
        split,
        train_labels_mapper,
        device,
    ):
        self.num_classes = int(len(train_labels_mapper))
        self._labeled_positions = np.asarray(
            split.labeled_positions,
            dtype=np.int64,
        )
        self._train_dataset_labels = np.asarray(
            train_dataset.labels,
            dtype=np.int64,
        )

    def build_dataset(self, train_dataset, split, use_cache=False):
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) == 0:
            raise ValueError(f"{self.method.name} in-batch mode needs unlabeled samples")
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset,
            use_cache=use_cache,
        )
        self.dataset = UnlabeledSubset(
            regularizer_dataset,
            unlabeled_positions,
            num_views=1,
        )
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._in_batch_graph_diagnostics_request = None
        self._last_diagnostics = {}
        return self.dataset

    def make_loader(
        self,
        model,
        train_dataset,
        supervised_loader,
        device,
        config,
        batch_size,
        seed,
        num_workers,
        start_method,
        epoch,
        log_dir=None,
    ):
        return _make_in_batch_graph_unlabeled_loader(
            self,
            supervised_loader=supervised_loader,
            config=config,
            device=device,
            seed=seed,
            num_workers=num_workers,
            start_method=start_method,
            epoch=epoch,
            log_dir=log_dir,
        )

    def compute_loss(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
        supervised_embeddings=None,
        supervised_labels=None,
        regularizer_embeddings=None,
        supervised_indices=None,
        supervised_criterion=None,
        supervised_miner=None,
        supervised_is_classification=False,
        **unused_context,
    ):
        if self.num_classes is None:
            raise RuntimeError("configure_model must run before in-batch graph training")
        if supervised_criterion is None:
            raise ValueError("in-batch graph pseudo-labeling needs the supervised objective")
        (
            graph_embeddings,
            graph_targets,
            graph_positions,
            labeled_count,
        ) = _in_batch_graph_context(
            self,
            batch=batch,
            supervised_embeddings=supervised_embeddings,
            supervised_labels=supervised_labels,
            regularizer_embeddings=regularizer_embeddings,
            supervised_indices=supervised_indices,
        )
        graph_diagnostics = _in_batch_graph_diagnostics(
            self,
            graph_positions=graph_positions,
            graph_targets=graph_targets,
        )
        with suppress_ssl_timing_logs():
            pseudo_labels, confidences, propagated = _run_graph_pseudo_label_batch(
                self.method,
                features=graph_embeddings.detach().float().cpu().numpy(),
                targets=graph_targets.detach().cpu().numpy(),
                num_classes=self.num_classes,
                method_params=self.method_params,
                graph_diagnostics=graph_diagnostics,
            )
        confidences = np.asarray(confidences, dtype=np.float32)
        accepted = (
            np.asarray(propagated, dtype=bool)
            & np.isfinite(confidences)
            & (confidences >= self.confidence_threshold)
        )
        accepted_tensor = torch.as_tensor(
            accepted,
            dtype=torch.bool,
            device=regularizer_embeddings.device,
        )
        confidence_tensor = torch.as_tensor(
            np.clip(confidences, 0.0, 1.0),
            dtype=torch.float32,
            device=regularizer_embeddings.device,
        )
        pseudo_label_tensor = torch.as_tensor(
            pseudo_labels,
            dtype=torch.long,
            device=regularizer_embeddings.device,
        )

        if bool(accepted_tensor.any()):
            # Include exactly the labeled graph nodes in the auxiliary metric
            # objective. This gives pair/triplet losses labeled anchors while
            # keeping both graph construction and its loss at the configured
            # B / mu-B sizes.
            loss_embeddings = torch.cat(
                [
                    graph_embeddings[:labeled_count],
                    regularizer_embeddings[accepted_tensor],
                ],
                dim=0,
            )
            loss_labels = torch.cat(
                [
                    graph_targets[:labeled_count],
                    pseudo_label_tensor[accepted_tensor],
                ],
                dim=0,
            )
            supports_sample_weights = getattr(
                supervised_criterion,
                "supports_sample_weights",
                False,
            )
            sample_weights = None
            if supports_sample_weights:
                sample_weights = torch.cat(
                    [
                        torch.ones(
                            labeled_count,
                            dtype=torch.float32,
                            device=loss_embeddings.device,
                        ),
                        confidence_tensor[accepted_tensor],
                    ],
                    dim=0,
                )
            if supervised_is_classification:
                loss = metric_losses.classification_loss_float32(
                    supervised_criterion,
                    loss_embeddings,
                    loss_labels,
                    sample_weights=sample_weights,
                )
            elif supports_sample_weights:
                loss = supervised_criterion(
                    loss_embeddings,
                    loss_labels,
                    sample_weights=sample_weights,
                )
            elif supervised_miner is not None:
                mined = supervised_miner(loss_embeddings, loss_labels)
                loss = supervised_criterion(loss_embeddings, loss_labels, mined)
            else:
                loss = supervised_criterion(loss_embeddings, loss_labels)
        else:
            loss = regularizer_embeddings.sum() * 0.0

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{self.method.name} in-batch graph produced a non-finite loss"
            )
        self._last_diagnostics = {
            f"train/{self.method.name}/in_batch_graph_nodes": float(
                len(graph_embeddings)
            ),
            f"train/{self.method.name}/in_batch_labeled_nodes": float(
                labeled_count
            ),
            f"train/{self.method.name}/in_batch_unlabeled_nodes": float(
                len(regularizer_embeddings)
            ),
            f"train/{self.method.name}/in_batch_accepted": float(accepted.sum()),
            f"train/{self.method.name}/in_batch_accepted_fraction": float(
                accepted.mean()
            ),
            f"train/{self.method.name}/in_batch_mean_confidence": float(
                confidences.mean()
            ),
        }
        return loss

    def batch_diagnostics(self):
        return dict(self._last_diagnostics)


class InBatchGraphSemiSupervisedMethod(BaseSemiSupervisedMethod):
    """Adapt an offline graph pseudo-labeler to the regularized training path."""

    generates_pseudo_labels = False
    is_regularization_method = True

    def __init__(self, method):
        self.method = method
        self.name = method.name

    def make_regularizer(self, config):
        return InBatchGraphPseudoLabelRegularizer(self.method, config)


REGULARIZER_REGISTRY = {
    "stml": STMLRegularizer,
    "lrml": LRMLRegularizer,
    "hoffer_entropy": HofferEntropyRegularizer,
    "simmatch_v2": SimMatchV2Regularizer,
    "slade": SladeRegularizer,
    "seraph": SeraphRegularizer,
    "ismlp": IsmlpRegularizer,
}


METHOD_REGISTRY = {
    "faiss_majority_vote_knn": FaissKNNMajorityVotePseudoLabeler(
        name="faiss_knn",
        n_neighbors=10,
    ),
    "self_training_knn": SelfTrainingKNNPseudoLabeler(),
    # these are basically unusable. Way to much memory usage. Interesting thought though <-_->.
    "sklearn_label_spreading": SklearnGraphSSLMethod(
        name="sklearn_label_spreading",
        estimator_cls=LabelSpreading,
        default_params={"kernel": "knn", "n_neighbors": 10, "alpha": 0.2, "max_iter": 30},
    ),
    "sklearn_label_propagation": SklearnGraphSSLMethod(
        name="sklearn_label_propagation",
        estimator_cls=LabelPropagation,
        default_params={"kernel": "knn", "n_neighbors": 10, "max_iter": 30},
    ),
    "faiss_label_spreading": FaissLabelSpreadingPseudoLabeler(),
    "iscen_label_spreading": IscenLabelSpreadingPseudoLabeler(),
    # STML's teacher relations cut at a confidence threshold instead of
    # applied as soft pairwise weights, which frees them from STML's own loss.
    "stml_threshold": STMLThresholdPseudoLabeler(),
    "mixed_label_propagation": MixedLabelPropagationPseudoLabeler(),
    # Generic composition point for supervised loss + unlabeled regularizer.
    "regularized": RegularizedSemiSupervisedMethod(name="regularized"),
    # Convenience alias; this is still the same supervised-loss + regularizer
    # composition and therefore works with every supported PML loss/miner.
    "simmatch_v2": RegularizedSemiSupervisedMethod(
        name="simmatch_v2",
        default_regularizer="simmatch_v2",
    ),
    # SLADE's student stage is the same composition: the configured pair-based
    # ranking loss on labeled data plus the unlabeled ranking and feature-basis
    # terms of Eq 9.
    "slade": RegularizedSemiSupervisedMethod(
        name="slade",
        default_regularizer="slade",
    ),
    # SERAPH keeps only the unlabeled half of its Eq 7: the configured metric
    # loss replaces the paper's pairwise logistic likelihood on labeled pairs.
    "seraph": RegularizedSemiSupervisedMethod(
        name="seraph",
        default_regularizer="seraph",
    ),
    # ISMLP's unlabeled proxy-assignment entropy (Eq 9), plus Eq 8's labeled
    # proxy alignment where ``labeled_weight`` turns it on -- by default under
    # ``proxy_mode="learned"``. The configured metric loss stays on the labeled
    # stream either way; see the ismlp module docstring, deviations 1 and 9.
    "ismlp": RegularizedSemiSupervisedMethod(
        name="ismlp",
        default_regularizer="ismlp",
    ),
}


def load_ssl_config(config_path, default_seed=0, default_support_seed=None):
    """Load a JSON SSL config and fill in missing runtime/support seeds."""

    if default_support_seed is None:
        default_support_seed = DEFAULT_SUPPORT_SEED

    if config_path is None:
        # No config means fully supervised defaults with resolved seeds attached.
        return SemiSupervisedConfig(seed=default_seed, support_seed=default_support_seed)

    path = Path(config_path)
    with path.open() as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"SSL config must be a JSON object: {path}")

    # Reject misspelled top-level keys instead of silently ignoring them.
    allowed_keys = set(SemiSupervisedConfig.__dataclass_fields__)
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown SSL config keys in {path}: {unknown_keys}. "
            "Put method-specific settings under method_params."
        )

    # Dataclass construction fills every omitted JSON key with its declared
    # default.
    config = SemiSupervisedConfig(**raw_config)
    if config.seed is None:
        # Runtime SSL randomness follows the outer run seed unless the SSL
        # config explicitly asks for a different seed.
        config = replace(config, seed=default_seed)
    if "support_seed" not in raw_config or config.support_seed is None:
        # Labeled support selection has its own seed so run-seed sweeps do not
        # silently change which samples are labeled.
        config = replace(config, support_seed=default_support_seed)
    validate_ssl_config(config, path)
    return config


def sample_scoped_refresh_is_supported(config):
    """Whether this configuration's refresh can be repeated inside an epoch.

    A sample-scoped schedule fires between epoch boundaries, so it only works for
    a method whose entire refresh is one call the engine can make again mid-epoch:
    the lrml regularizer's whole-pool graph rebuild, and the pseudo-label methods
    that rebuild the training dataset in one pass.
    """

    method = METHOD_REGISTRY.get(config.method)
    if method is None:
        return False
    if not method.is_regularization_method:
        return bool(method.supports_sample_scoped_refresh)
    regularizer_name = dict(config.method_params).get(
        "regularizer",
        getattr(method, "default_regularizer", None),
    )
    regularizer_class = REGULARIZER_REGISTRY.get(regularizer_name)
    if regularizer_class is None:
        return False
    return bool(regularizer_class.supports_sample_scoped_refresh)


def sample_scoped_refresh_methods():
    """Names accepted by ``update_mode='every_n_samples'``, for error messages."""

    return sorted(
        name
        for name in METHOD_REGISTRY
        if sample_scoped_refresh_is_supported(
            SemiSupervisedConfig(method=name, seed=0)
        )
    )


def validate_ssl_config(config, path=None):
    """Validate label-selection and method settings before any data is loaded."""

    source = f" in {path}" if path is not None else ""
    if config.method != "none" and config.method not in METHOD_REGISTRY and config.method not in LOSS_DRIVEN_METHODS:
        raise ValueError(f"Unknown SSL method{source}: {config.method}. Available: {available_methods()}")
    if config.update_mode not in UPDATE_MODES:
        raise ValueError(f"Unknown SSL update_mode{source}: {config.update_mode}. Available: {sorted(UPDATE_MODES)}")
    if config.update_interval_epochs <= 0:
        raise ValueError(f"update_interval_epochs must be positive{source}")
    if config.update_interval_samples is not None:
        if isinstance(config.update_interval_samples, bool) or not isinstance(
            config.update_interval_samples,
            (int, np.integer),
        ):
            raise ValueError(f"update_interval_samples must be an integer or null{source}")
        if config.update_interval_samples <= 0:
            raise ValueError(f"update_interval_samples must be positive when set{source}")
    if config.update_mode == "every_n_samples":
        if config.update_interval_samples is None:
            raise ValueError(
                f"update_mode='every_n_samples' requires update_interval_samples{source}"
            )
        if not sample_scoped_refresh_is_supported(config):
            raise ValueError(
                "update_mode='every_n_samples' rebuilds inside an epoch, which is "
                f"implemented for the lrml regularizer and for methods "
                f"{sample_scoped_refresh_methods()}; got method={config.method!r}"
                f"{source}"
            )
        if config.graph_batch_mode != "global":
            raise ValueError(
                "update_mode='every_n_samples' schedules whole-pool graph "
                f"rebuilds and does not apply to graph_batch_mode="
                f"{config.graph_batch_mode!r}{source}, which already builds one "
                "graph per step"
            )
        if config.graph_diagnostics_mode == "save":
            # One artifact bundle is written per (series, epoch), so a second
            # rebuild inside an epoch would silently overwrite the first one's
            # bundle instead of adding its own.
            raise ValueError(
                "graph_diagnostics_mode='save' names its artifact bundles by "
                "epoch and cannot record more than one graph per epoch, which "
                f"update_mode='every_n_samples' produces{source}"
            )
    elif config.update_interval_samples is not None:
        raise ValueError(
            "update_interval_samples has no effect unless "
            f"update_mode='every_n_samples'{source}"
        )
    if config.label_sampling_mode not in LABEL_SAMPLING_MODES:
        raise ValueError(
            f"Unknown label_sampling_mode{source}: {config.label_sampling_mode}. "
            f"Available: {sorted(LABEL_SAMPLING_MODES)}"
        )
    if config.method == "none" and config.update_mode != "once":
        raise ValueError(f"update_mode must be 'once' when method is 'none'{source}")
    if config.warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be non-negative{source}")
    if config.method == "none" and config.warmup_epochs != 0:
        raise ValueError(f"warmup_epochs must be 0 when method is 'none'{source}")
    if not isinstance(config.restart_selection_after_warmup, bool):
        raise ValueError(
            f"restart_selection_after_warmup must be a boolean{source}: "
            f"{config.restart_selection_after_warmup!r}"
        )
    if config.method == "none" and config.restart_selection_after_warmup:
        raise ValueError(
            "restart_selection_after_warmup has nothing to restart when method is "
            f"'none': there is no SSL phase{source}"
        )
    if config.warmup_checkpoint_mode not in WARMUP_CHECKPOINT_MODES:
        raise ValueError(
            f"Unknown warmup_checkpoint_mode{source}: "
            f"{config.warmup_checkpoint_mode!r}. "
            f"Available: {sorted(WARMUP_CHECKPOINT_MODES)}"
        )
    method = METHOD_REGISTRY.get(config.method)
    if config.method in LOSS_DRIVEN_METHODS and config.method_params:
        raise ValueError(
            f"method_params must be empty for loss-driven method {config.method!r}{source}; "
            "configure the loss with loss_params"
        )
    if config.seed is None:
        raise ValueError(f"seed must be resolved before validation{source}")
    if config.support_seed is None:
        raise ValueError(f"support_seed must be resolved before validation{source}")
    if config.labeled_per_class is None and not (0 < config.labeled_fraction <= 1):
        raise ValueError(f"labeled_fraction must be in (0, 1]{source}")
    if config.labeled_per_class is not None and config.labeled_per_class <= 0:
        raise ValueError(f"labeled_per_class must be positive{source}")
    if config.labeled_per_class is not None and config.label_sampling_mode not in {
        "per_class_min",
        "per_class_imbalanced",
        "class_subset_k_shot",
    }:
        raise ValueError(
            f"labeled_per_class is only supported with label_sampling_mode='per_class_min', "
            f"'per_class_imbalanced', or 'class_subset_k_shot'{source}"
        )
    if config.label_sampling_mode == "class_subset_k_shot" and config.labeled_per_class is None:
        raise ValueError(f"class_subset_k_shot requires labeled_per_class to set k-shot{source}")
    if not (0 <= config.confidence_threshold <= 1):
        raise ValueError(f"confidence_threshold must be in [0, 1]{source}")
    if not (0 <= config.pseudo_label_rescue_confidence_floor <= 1):
        raise ValueError(f"pseudo_label_rescue_confidence_floor must be in [0, 1]{source}")
    if config.pseudo_label_rescue_top_k is not None:
        if isinstance(config.pseudo_label_rescue_top_k, bool) or not isinstance(
            config.pseudo_label_rescue_top_k,
            (int, np.integer),
        ):
            raise ValueError(f"pseudo_label_rescue_top_k must be an integer or null{source}")
        if config.pseudo_label_rescue_top_k <= 0:
            raise ValueError(f"pseudo_label_rescue_top_k must be positive when set{source}")
    if config.labeled_batch_size is not None:
        if isinstance(config.labeled_batch_size, bool) or not isinstance(
            config.labeled_batch_size,
            (int, np.integer),
        ):
            raise ValueError(f"labeled_batch_size must be an integer or null{source}")
        if config.labeled_batch_size <= 0:
            raise ValueError(f"labeled_batch_size must be positive when set{source}")
        if config.method not in TWO_STREAM_SAMPLER_METHODS:
            raise ValueError(
                "labeled_batch_size currently enables TwoStreamMPerClassBatchSampler only for "
                f"methods {sorted(TWO_STREAM_SAMPLER_METHODS)}{source}"
            )
    if config.class_overlap not in CLASS_OVERLAP_MODES:
        raise ValueError(
            f"class_overlap must be one of {sorted(CLASS_OVERLAP_MODES)}{source}: "
            f"{config.class_overlap!r}"
        )
    if config.graph_batch_mode not in GRAPH_BATCH_MODES:
        raise ValueError(
            f"graph_batch_mode must be one of {sorted(GRAPH_BATCH_MODES)}{source}"
        )
    for name in ("graph_labeled_batch_size", "graph_unlabeled_batch_size"):
        value = getattr(config, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or null{source}")
        if value <= 0:
            raise ValueError(f"{name} must be positive when set{source}")
    if config.graph_batch_mode in IN_BATCH_GRAPH_MODES:
        if config.labeled_batch_size is not None:
            raise ValueError(
                "labeled_batch_size is the whole-pool post-propagation sampler "
                f"quota and cannot be combined with "
                f"graph_batch_mode={config.graph_batch_mode!r}{source}; "
                "use graph_labeled_batch_size"
            )
        if config.graph_unlabeled_batch_size is None:
            raise ValueError(
                f"graph_unlabeled_batch_size is required for in-batch graphs{source}"
            )
    if config.graph_batch_mode == "in_batch_merged":
        # The merged mode owns the whole objective, so it needs a method that can
        # fold the supervised term into its own loss call.
        if config.method != "stml_threshold":
            raise ValueError(
                "graph_batch_mode='in_batch_merged' is implemented for "
                f"stml_threshold only; got method={config.method!r}{source}"
            )
        if config.graph_labeled_batch_size is not None:
            raise ValueError(
                "graph_labeled_batch_size does not apply to "
                f"graph_batch_mode='in_batch_merged'{source}: the graph spans the "
                "whole supervised batch, so there is no labeled subsample to size"
            )
    elif config.graph_batch_mode == "in_batch":
        if config.graph_labeled_batch_size is None:
            raise ValueError(
                f"graph_labeled_batch_size is required for in-batch graphs{source}"
            )
        if (
            config.method not in GRAPH_PSEUDO_LABEL_METHODS
            and config.method not in {"regularized", "stml_threshold"}
        ):
            raise ValueError(
                "graph_batch_mode='in_batch' is supported by graph pseudo-label "
                "methods, stml_threshold, and the lrml/slrml regularizers; "
                f"got method={config.method!r}{source}"
            )
    if config.pseudo_label_diagnostics_mode not in PSEUDO_LABEL_DIAGNOSTICS_MODES:
        raise ValueError(
            f"pseudo_label_diagnostics_mode must be one of {sorted(PSEUDO_LABEL_DIAGNOSTICS_MODES)}{source}"
        )
    if config.graph_diagnostics_mode not in GRAPH_DIAGNOSTICS_MODES:
        raise ValueError(
            f"graph_diagnostics_mode must be one of {sorted(GRAPH_DIAGNOSTICS_MODES)}{source}"
        )
    if config.graph_diagnostics_max_nodes <= 0:
        raise ValueError(f"graph_diagnostics_max_nodes must be positive{source}")
    if config.graph_diagnostics_max_edges <= 0:
        raise ValueError(f"graph_diagnostics_max_edges must be positive{source}")
    if config.graph_diagnostics_max_labels < 0:
        raise ValueError(f"graph_diagnostics_max_labels must be non-negative{source}")
    if config.graph_diagnostics_layout not in GRAPH_DIAGNOSTICS_LAYOUTS:
        raise ValueError(
            f"graph_diagnostics_layout must be one of {sorted(GRAPH_DIAGNOSTICS_LAYOUTS)}{source}"
        )
    if config.graph_diagnostics_class_focus not in GRAPH_DIAGNOSTICS_CLASS_FOCUS_MODES:
        raise ValueError(
            "graph_diagnostics_class_focus must be one of "
            f"{sorted(GRAPH_DIAGNOSTICS_CLASS_FOCUS_MODES)}{source}"
        )
    if config.graph_diagnostics_class_count <= 0:
        raise ValueError(f"graph_diagnostics_class_count must be positive{source}")
    if config.graph_diagnostics_classes is not None:
        classes = list(config.graph_diagnostics_classes)
        if not classes:
            raise ValueError(
                f"graph_diagnostics_classes must not be empty when set{source}"
            )
        if any(not isinstance(label, int) or isinstance(label, bool) for label in classes):
            raise ValueError(
                f"graph_diagnostics_classes must be a list of integer class ids{source}"
            )
        if any(label < 0 for label in classes):
            raise ValueError(
                f"graph_diagnostics_classes must be non-negative class ids{source}"
            )
        if len(set(classes)) != len(classes):
            raise ValueError(f"graph_diagnostics_classes must be unique{source}")
    if (
        config.graph_diagnostics_class_focus == "explicit"
        and config.graph_diagnostics_classes is None
    ):
        raise ValueError(
            "graph_diagnostics_class_focus='explicit' requires "
            f"graph_diagnostics_classes{source}"
        )
    if (
        config.graph_diagnostics_class_focus not in {"off", "explicit"}
        and config.graph_diagnostics_classes is not None
    ):
        raise ValueError(
            "graph_diagnostics_classes only applies to "
            f"graph_diagnostics_class_focus='explicit'{source}: mode "
            f"{config.graph_diagnostics_class_focus!r} picks the classes itself"
        )
    if (
        config.graph_diagnostics_class_focus != "off"
        and config.graph_diagnostics_mode == "off"
    ):
        raise ValueError(
            "graph_diagnostics_class_focus needs graph_diagnostics_mode='save'"
            f"{source}: nothing writes the class-scoped view otherwise"
        )
    if config.max_unlabeled_samples is not None and config.max_unlabeled_samples <= 0:
        raise ValueError(f"max_unlabeled_samples must be positive{source}")
    if config.unlabeled_class_scope not in UNLABELED_CLASS_SCOPES:
        raise ValueError(
            f"unlabeled_class_scope must be one of {sorted(UNLABELED_CLASS_SCOPES)}"
            f"{source}: {config.unlabeled_class_scope!r}"
        )
    if config.unlabeled_class_scope == UNLABELED_CLASS_SCOPE_LABELED_CLASSES and (
        config.label_sampling_mode == "class_subset"
    ):
        raise ValueError(
            "unlabeled_class_scope='labeled_classes' leaves no unlabeled candidates "
            f"under label_sampling_mode='class_subset'{source}: that mode labels its "
            "selected classes completely, so restricting the pool to those classes "
            "empties it. Use 'class_subset_k_shot' to keep a labeled class subset "
            "with an unlabeled remainder inside it"
        )
    if config.unlabeled_fraction is not None:
        if isinstance(config.unlabeled_fraction, bool) or not isinstance(
            config.unlabeled_fraction,
            (int, float, np.floating, np.integer),
        ):
            raise ValueError(f"unlabeled_fraction must be a number or null{source}")
        if not math.isfinite(float(config.unlabeled_fraction)) or not (
            0 < float(config.unlabeled_fraction) <= 1
        ):
            raise ValueError(
                f"unlabeled_fraction must be in (0, 1]{source}: {config.unlabeled_fraction}"
            )
    if config.embedding_batch_size <= 0:
        raise ValueError(f"embedding_batch_size must be positive{source}")
    if config.embedding_num_workers < 0:
        raise ValueError(f"embedding_num_workers must be non-negative{source}")
    if not isinstance(config.method_params, dict):
        raise ValueError(f"method_params must be an object{source}")
    if method is not None:
        method.validate_config(config, source=source)


def available_methods():
    return ["none", *sorted(METHOD_REGISTRY), *sorted(LOSS_DRIVEN_METHODS)]


def get_method(config):
    """Return the configured registry method, or None when SSL is disabled."""

    if not config.enabled:
        return None
    method = METHOD_REGISTRY.get(config.method)
    if method is not None and config.graph_batch_mode in IN_BATCH_GRAPH_MODES:
        # stml_threshold keeps its own in-batch adapter: the paper's loop needs a
        # per-step EMA teacher and nearest-neighbor batches, neither of which the
        # generic graph adapter provides.
        if config.method == "stml_threshold":
            return STMLThresholdInBatchMethod(method)
        # Only stml_threshold implements the merged single-term objective, so the
        # generic adapter stays on the two-term 'in_batch' mode.
        if config.graph_batch_mode == "in_batch" and config.method in GRAPH_PSEUDO_LABEL_METHODS:
            return InBatchGraphSemiSupervisedMethod(method)
    return method


def create_method(config):
    """Create an isolated method instance for one training run or CV fold."""

    method = get_method(config)
    if method is None:
        return None
    method = copy.deepcopy(method)
    reset_state = getattr(method, "reset_state", None)
    if reset_state is not None:
        reset_state()
    return method


def is_regularization_method(config):
    method = get_method(config)
    return method is not None and method.is_regularization_method


def is_pseudo_label_method(config):
    method = get_method(config)
    return method is not None and method.generates_pseudo_labels


def make_pseudo_label_diagnostics_tracker(log_dir, config, mode=None):
    """Build the tracker, letting an explicit ``mode`` override the SSL config.

    ``mode`` carries --pseudo_label_diagnostics_mode. 'off' returns None, which
    skips the hidden-label audit entirely rather than computing summaries that
    are then discarded.
    """

    if not is_pseudo_label_method(config):
        return None
    resolved_mode = config.pseudo_label_diagnostics_mode if mode is None else mode
    if resolved_mode not in PSEUDO_LABEL_DIAGNOSTICS_MODES:
        raise ValueError(
            f"pseudo_label_diagnostics_mode must be one of {sorted(PSEUDO_LABEL_DIAGNOSTICS_MODES)}"
        )
    if resolved_mode == "off":
        return None
    return PseudoLabelDiagnosticsTracker(log_dir, mode=resolved_mode)


def prepare_ssl_split(train_dataset, config):
    """Create the labeled/unlabeled split only when SSL is enabled."""

    if not config.enabled:
        # A normal fully supervised run does not need a position split.
        return None

    logger.info(f"Using semi-supervised config: {config.to_dict()}")
    return prepare_label_split(train_dataset, config)


def prepare_label_split(train_dataset, config):
    """Apply the configured label budget and log the resulting class coverage."""

    # train_dataset.labels is aligned with positions 0..len(train_dataset)-1,
    # which is the coordinate system returned by the selector.
    split = make_semi_supervised_split(
        labels=train_dataset.labels,
        label_sampling_mode=config.label_sampling_mode,
        labeled_fraction=config.labeled_fraction,
        labeled_per_class=config.labeled_per_class,
        max_unlabeled_samples=config.max_unlabeled_samples,
        seed=config.support_seed,
        unlabeled_class_scope=config.unlabeled_class_scope,
        unlabeled_fraction=config.unlabeled_fraction,
    )
    # Count class coverage after selection because some sampling modes expose
    # only a subset of classes.
    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    labeled_labels = labels[split.labeled_positions]
    num_labeled_classes = int(len(np.unique(labeled_labels))) if len(labeled_labels) > 0 else 0
    num_total_classes = int(len(np.unique(labels)))
    # The unlabeled class count is the one number that shows whether the pool is
    # in-distribution, so it is logged next to the labeled coverage rather than
    # left to be inferred from the sampling mode.
    unlabeled_labels = labels[split.unlabeled_positions]
    num_unlabeled_classes = int(len(np.unique(unlabeled_labels))) if len(unlabeled_labels) > 0 else 0
    logger.info(
        "Semi-supervised split: "
        f"label_mode={config.label_sampling_mode}, "
        f"{len(split.labeled_positions)} labeled across {num_labeled_classes}/{num_total_classes} classes, "
        f"{len(split.unlabeled_positions)} unlabeled candidates across "
        f"{num_unlabeled_classes}/{num_total_classes} classes "
        f"(scope={config.unlabeled_class_scope}, fraction={config.unlabeled_fraction})"
    )
    return split


def build_ssl_training_dataset(
    model,
    train_dataset,
    train_labels_mapper,
    device,
    config,
    split=None,
    epoch=None,
    start_method="spawn",
    diagnostics_tracker=None,
    log_dir=None,
    required_pseudo_label_classes=None,
    required_combined_label_classes=None,
    pseudo_label_rescue_top_k=None,
    method=None,
):
    """Generate, filter, and merge pseudo-labels with the true labeled subset."""

    if not config.enabled:
        # When SSL is disabled and no split-only supervised baseline is needed,
        # use the original training dataset unchanged.
        return train_dataset
    if config.graph_batch_mode == "in_batch":
        raise ValueError(
            "in-batch graph methods generate pseudo-labels during each training "
            "step and must use the regularized training path"
        )
    if config.method in LOSS_DRIVEN_METHODS:
        raise ValueError(
            f"{config.method} is a loss-driven SSL method and does not generate pseudo-labels; "
            "use build_loss_driven_training_dataset"
        )
    method = METHOD_REGISTRY[config.method] if method is None else method
    if not method.generates_pseudo_labels:
        raise ValueError(
            f"{config.method} is a regularization method and does not generate pseudo-labels"
        )

    if split is None:
        # Callers may cache/reuse a split for fair comparisons. If omitted,
        # derive one now from the current training subset.
        split = prepare_ssl_split(train_dataset, config)

    epoch_label = "" if epoch is None else f" for epoch {epoch}"
    logger.info(f"Generating {config.method} pseudo-labels{epoch_label}")

    # Methods predict dense mapped labels because those are the labels used by
    # losses and the M-per-class sampler during training.
    # Keep embedding extraction, FAISS graph construction, and any configured
    # CUDA solver on the same explicitly selected SSL device. The extraction
    # helper restores the model to its training device before this context ends.
    with ssl_compute_device(device):
        raw_pseudo_labels = method.generate_pseudo_labels(
            model,
            train_dataset,
            split,
            device,
            config,
            epoch=epoch,
            start_method=start_method,
            log_dir=log_dir,
        )

    #### TODO: this is probably not faithfully implemented. Check this again for e. g. MLPPL

    labeled_mapped_classes = set()
    if required_combined_label_classes is not None:
        source_labels = np.asarray(train_dataset.orig_labels, dtype=np.int64)
        try:
            labeled_mapped_classes = {
                int(train_labels_mapper[int(source_labels[position])])
                for position in split.labeled_positions
            }
        except KeyError as exc:
            raise ValueError(
                f"True-labeled class {int(exc.args[0])} is absent from the active training label mapper"
            ) from exc

    # Filter before merging so low-confidence/invalid predictions never affect
    # sampler class counts or training batches. The two-stream path also checks
    # coverage across the true/pseudo union needed by the global M-per-class
    # constraint.
    pseudo_labels = filter_pseudo_labels(
        pseudo_labels=raw_pseudo_labels,
        confidence_threshold=method.pseudo_label_filter_threshold(config),
        valid_mapped_labels=set(train_labels_mapper.values()),
        required_classes=required_pseudo_label_classes,
        required_union_classes=required_combined_label_classes,
        union_classes=labeled_mapped_classes,
        rescue_confidence_floor=config.pseudo_label_rescue_confidence_floor,
        rescue_top_k=(
            config.pseudo_label_rescue_top_k
            if config.pseudo_label_rescue_top_k is not None
            else pseudo_label_rescue_top_k
        ),
    )
    if diagnostics_tracker is not None:
        diagnostics_tracker.log(
            raw_pseudo_labels=raw_pseudo_labels,
            accepted_pseudo_labels=pseudo_labels,
            train_dataset=train_dataset,
            config=config,
            epoch=epoch,
        )

    if len(pseudo_labels.positions) == 0:
        logger.info("No pseudo-labels selected; training on labeled subset only")
    else:
        logger.info(f"Selected {len(pseudo_labels.positions)} pseudo-labeled samples")

    # Keep selection confidence intact through the global threshold and the
    # M-per-class capacity rescue. A method may turn the accepted confidence
    # into a different training weight only after those decisions are final.
    training_pseudo_labels = method.prepare_pseudo_labels_for_training(
        pseudo_labels,
        config,
    )

    return make_relabeled_training_dataset(
        train_dataset=train_dataset,
        train_labels_mapper=train_labels_mapper,
        labeled_positions=split.labeled_positions,
        pseudo_labels=training_pseudo_labels,
    )


def build_labeled_training_dataset(
    train_dataset,
    train_labels_mapper,
    split,
    return_indices=False,
):
    """Build the supervised baseline from only the split's labeled positions."""

    # Reuse the same relabeling/merging function as SSL, but provide an empty
    # pseudo-label group so only true-labeled positions remain.
    empty_pseudo_labels = PseudoLabelResult(
        positions=np.array([], dtype=np.int64),
        mapped_labels=np.array([], dtype=np.int64),
        confidences=np.array([], dtype=np.float32),
    )
    return make_relabeled_training_dataset(
        train_dataset=train_dataset,
        train_labels_mapper=train_labels_mapper,
        labeled_positions=split.labeled_positions,
        pseudo_labels=empty_pseudo_labels,
        return_indices=return_indices,
    )


def build_loss_driven_training_dataset(train_dataset, split, num_views=1):
    """Expose labeled and unlabeled candidates without exposing their labels."""

    positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
    if len(positions) < 2:
        raise ValueError("loss-driven SSL requires at least two labeled or unlabeled training samples")
    return UnlabeledSubset(train_dataset, positions, num_views=num_views)
