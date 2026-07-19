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

from .ssl.algorithms import (
    build_lrml_knn_graph,
    build_slrml_graph,
    entropy_confidence as entropy_confidence,
    faiss_flat_ip_search,
    faiss_label_spreading as _faiss_label_spreading,
    iscen_label_spreading as _iscen_label_spreading,
    majority_vote,
    make_dissimilarity_affinity as make_dissimilarity_affinity,
    make_mixed_label_affinity as make_mixed_label_affinity,
    make_slrml_graph_labels,
    mixed_label_propagation as _mixed_label_propagation,
    normalize_mixed_label_rows as normalize_mixed_label_rows,
    require_faiss,
    solve_sparse_label_system as solve_sparse_label_system,
    solve_sparse_label_system_cholmod as solve_sparse_label_system_cholmod,
)
from .ssl.config import (
    DEFAULT_SUPPORT_SEED,
    GRAPH_DIAGNOSTICS_LAYOUTS,
    GRAPH_DIAGNOSTICS_MODES,
    LABEL_SAMPLING_MODES,
    LOSS_DRIVEN_METHODS,
    PSEUDO_LABEL_DIAGNOSTICS_MODES,
    UNLABELED_TARGET,
    UPDATE_MODES,
    GraphDiagnosticsRequest as GraphDiagnosticsRequest,
    PseudoLabelResult,
    SemiSupervisedConfig,
    SemiSupervisedSplit as SemiSupervisedSplit,
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
    make_embedding_loader,
)
from .ssl.graph_diagnostics import (
    dataset_labels_for_positions,
    make_graph_diagnostics_request,
    maybe_save_graph_diagnostics,
    project_graph_embeddings_2d as project_graph_embeddings_2d,
    save_graph_diagnostics as save_graph_diagnostics,
)
from .ssl.interfaces import BaseSemiSupervisedMethod, BaseTrainingRegularizer
from .ssl.simmatch_v2 import SimMatchV2Regularizer
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


def _reduce_sampled_graph_energy(
    energy,
    sampled_edges,
    graph_edges,
    graph_weight,
    reduction,
):
    """Scale a uniform edge mini-batch to an unbiased graph objective estimate."""
    # FIXME: if we don't use LRML, this might not work as intended.
    # Maybe follow edge weight like in https://www.jmlr.org/papers/v7/belkin06a.html
    # this might make sense, because then
    if reduction == "mean":
        return energy / float(sampled_edges)
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

class LRMLRegularizer(BaseTrainingRegularizer):
    """Deep Laplacian Regularized Metric Learning regularizer (Hoi et al., 2010).

    Replaces LRML's linear projection U^T x with the network embedding f(x), so the
    regularizer is the graph-Laplacian smoothness energy

        g(theta) = (1/2) * sum_ij W_ij * || f(x_i) - f(x_j) ||^2 = tr(Z^T L Z),

    with W a binary symmetric kNN graph and L = D - W (optionally the symmetric-
    normalized Laplacian used in the paper's experiments). The similar/dissimilar
    loss terms of the original objective are supplied by the configured supervised
    loss; this class only adds the unlabeled Laplacian term.

    The log-det constraint and the SDP / matrix-inversion solvers are specific to
    the closed-form linear metric and have no SGD analogue; collapse is instead
    avoided by the supervised loss and L2-normalized embeddings.

    ``GraphEdgeBatchSampler`` samples the symmetric graph's upper-triangle edges
    uniformly. Each step evaluates only those selected pairs and rescales their
    energy to an unbiased estimate of the full trace (or its stable global
    weighted mean). The symmetric-normalized Laplacian is obtained by scaling
    each endpoint embedding by 1/sqrt(deg) before evaluating its edge.
    """

    name = "lrml"
    supports_frozen_feature_precompute = True

    DEFAULT_PARAMS = {
        "n_neighbors": 6,             # paper uses 6 nearest neighbors
        "normalized_laplacian": True, # paper adopts the normalized Laplacian in practice
        "graph_on": "all",            # "all" (labeled + unlabeled) or "unlabeled"
        "reduction": "mean",          # stable global weighted mean, or full-trace "sum"
        "graph_batch_size": None,      # sampled graph edges; None matches supervised batch size
    }
    GRAPH_ON_CHOICES = {"all", "unlabeled"}
    REDUCTION_CHOICES = {"mean", "sum"}

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(regularizer_weight=regularizer_weight, supervised_weight=supervised_weight)
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown lrml regularizer_params: {unknown}")
        merged = {**self.DEFAULT_PARAMS, **params}

        self.n_neighbors = int(merged["n_neighbors"])
        if self.n_neighbors <= 0:
            raise ValueError("lrml n_neighbors must be positive")
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

        self.dataset = None
        self.graph_positions = None
        self.graph_known_mask = None
        self.neighbor_indices = None
        self.adjacency = None   # scipy CSR, symmetric binary W
        self.degrees = None
        self.node_scale = None  # torch tensor, 1/sqrt(deg) or ones
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("LRML regularization requires batch_size >= 2")

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
        self.dataset = LRMLGraphDataset(regularizer_dataset, positions)
        self.neighbor_indices = None
        self.adjacency = None
        self.degrees = None
        self.node_scale = None
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        return self.dataset

    def make_loader(
        self, model, train_dataset, supervised_loader, device, config,
        batch_size, seed, num_workers, start_method, epoch,
        log_dir=None,
    ):
        if self.dataset is None or self.graph_positions is None:
            raise RuntimeError("build_dataset must be called before make_loader")

        has_cached_graph = (
            self.neighbor_indices is not None
            and self.adjacency is not None
            and self.degrees is not None
            and self.node_scale is not None
            and self.graph_edge_count is not None
            and self.graph_weight_sum is not None
        )
        should_rebuild_graph = not has_cached_graph or should_rebuild_on_epoch(
            config.update_mode,
            config.update_interval_epochs,
            epoch,
            self._last_graph_rebuild_epoch,
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
                device=device,
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
            deviations = np.abs(norms - 1.0)

            print(
                "LRML embedding norms: "
                f"min={norms.min():.9f}, "
                f"mean={norms.mean():.9f}, "
                f"max={norms.max():.9f}, "
                f"max_abs_deviation={deviations.max():.3e}, "
                f"normalized={np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)}"
            )

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
            neighbor_indices, adjacency, degrees = build_lrml_knn_graph(
                embeddings,
                n_neighbors=self.n_neighbors,
            )
            edge_rows, _, edge_weights = graph_upper_triangle_edges(adjacency)
            maybe_save_graph_diagnostics(
                request=make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_knn",
                    epoch=epoch,
                    title="LRML symmetric kNN graph",
                ),
                embeddings=embeddings,
                adjacency=adjacency,
                positions=self.graph_positions,
                labels=dataset_labels_for_positions(train_dataset, self.graph_positions),
                known_mask=self.graph_known_mask,
            )
            self.neighbor_indices = neighbor_indices
            self.adjacency = adjacency
            self.degrees = degrees
            self.graph_edge_count = int(len(edge_rows))
            self.graph_weight_sum = float(edge_weights.sum())
            self._last_graph_rebuild_epoch = None if epoch is None else int(epoch)
            scale = (
                1.0 / np.sqrt(np.maximum(degrees, 1.0))
                if self.normalized_laplacian
                else np.ones(len(degrees), dtype=np.float64)
            )
            self.node_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
            logger.info(
                "Built LRML graph: "
                f"{adjacency.shape[0]} nodes, {self.graph_edge_count} undirected edges, "
                f"mean_degree={float(degrees.mean()):.2f}, "
                f"normalized_laplacian={self.normalized_laplacian}"
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
        cache_key = (
            id(self.dataset),
            graph_batch_size,
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
                f"steps={len(supervised_loader)}"
            )
        else:
            self._regularizer_sampler.set_graph(self.adjacency)

        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def compute_loss(self, student_model, state, batch, device, timings=None):
        if len(batch) == 3:
            images, node_ids, edge_indices = batch
        else:
            images, node_ids = batch
            edge_indices = None

        embeddings = utils.forward_model_inputs(
            student_model,
            images,
            device,
            use_cache=self.use_cache,
        )

        # Fold the per-node 1/sqrt(deg) factor in: normalized-Laplacian energy equals
        # the unnormalized energy on degree-scaled embeddings.
        scaled = embeddings * self.node_scale[node_ids.to(device)][:, None]
        left_indices, right_indices, weights = _graph_edge_batch(
            self.adjacency, node_ids, edge_indices
        )
        # edge_weight = torch.as_tensor(weights, dtype=embeddings.dtype, device=device) # this is super not needed at all, because weight is always 1.
        differences = (
            scaled[left_indices.to(device)] - scaled[right_indices.to(device)]
        )
        squared_distances = (differences * differences).sum(dim=1)
        energy = squared_distances.sum() # edge_weight *

        return _reduce_sampled_graph_energy(
            energy=energy,
            sampled_edges=len(weights),
            graph_edges=self.graph_edge_count,
            graph_weight=self.graph_weight_sum,
            reduction=self.reduction,
        )


class SLRMLRegularizer(BaseTrainingRegularizer):
    """Deep single-stream SLRML graph regularizer.

    The paper's linear projection ``U^T x`` is replaced by the network embedding
    ``f(x)``.  The regularizer therefore evaluates

        tr(Z^T L^s Z) = sum_{i<j} W^s_ij ||z_i - z_j||^2

    using uniform edge mini-batches. ``W^u`` is the symmetrized kNN graph with
    edge weight ``1/N_k`` and is the default label-free regularizer over all
    labeled and unlabeled graph nodes. ``include_supervised_graph=True``
    explicitly adds ``W^l``, which connects known same-class labeled samples
    with weight ``1/(2N_S)``.

    This is not the paper's multimodal SLRMML algorithm: that requires aligned
    modality-specific inputs, projections, graphs, and learned modality weights.
    """

    name = "slrml"
    supports_frozen_feature_precompute = True

    DEFAULT_PARAMS = {
        "n_neighbors": 6,
        "reduction": "mean",
        "include_supervised_graph": False,
        "graph_batch_size": None,
    }
    REDUCTION_CHOICES = {"mean", "sum"}

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(regularizer_weight=regularizer_weight, supervised_weight=supervised_weight)
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown slrml regularizer_params: {unknown}")
        merged = {**self.DEFAULT_PARAMS, **params}

        self.n_neighbors = int(merged["n_neighbors"])
        if self.n_neighbors <= 0:
            raise ValueError("slrml n_neighbors must be positive")
        self.reduction = str(merged["reduction"])
        if self.reduction not in self.REDUCTION_CHOICES:
            raise ValueError(f"slrml reduction must be one of {sorted(self.REDUCTION_CHOICES)}")
        self.include_supervised_graph = bool(merged["include_supervised_graph"])
        graph_batch_size = merged["graph_batch_size"]
        self.graph_batch_size = None if graph_batch_size is None else int(graph_batch_size)
        if self.graph_batch_size is not None and self.graph_batch_size <= 0:
            raise ValueError("slrml graph_batch_size must be positive when set")

        self.dataset = None
        self.graph_positions = None
        self.graph_labels = None
        self.graph_known_mask = None
        self.adjacency = None
        self.degrees = None
        self.positive_pair_count = 0
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("SLRML regularization requires batch_size >= 2")

    def build_dataset(self, train_dataset, split, use_cache=False):
        positions = np.unique(
            np.concatenate([split.labeled_positions, split.unlabeled_positions]).astype(np.int64)
        )
        if len(positions) < 2:
            raise ValueError("SLRML regularization requires at least two graph samples")
        utils.shutdown_dataloaders(
            self._regularizer_loader,
            self._embedding_loader,
        )
        self.graph_positions = positions
        self.graph_known_mask = np.isin(
            positions, np.asarray(split.labeled_positions, dtype=np.int64)
        )
        self.graph_labels = None
        if self.include_supervised_graph:
            self.graph_labels = make_slrml_graph_labels(
                train_dataset=train_dataset,
                graph_positions=positions,
                labeled_positions=split.labeled_positions,
            )
        regularizer_dataset = self.make_regularizer_source_dataset(train_dataset, use_cache=use_cache)
        self.dataset = LRMLGraphDataset(regularizer_dataset, positions)
        self.adjacency = None
        self.degrees = None
        self.positive_pair_count = 0
        self.graph_edge_count = None
        self.graph_weight_sum = None
        self._last_graph_rebuild_epoch = None
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        return self.dataset

    def make_loader(
        self, model, train_dataset, supervised_loader, device, config,
        batch_size, seed, num_workers, start_method, epoch,
        log_dir=None,
    ):
        if self.dataset is None or self.graph_positions is None:
            raise RuntimeError("build_dataset must be called before make_loader")

        has_cached_graph = (
            self.adjacency is not None
            and self.degrees is not None
            and self.graph_edge_count is not None
            and self.graph_weight_sum is not None
        )
        should_rebuild_graph = not has_cached_graph or should_rebuild_on_epoch(
            config.update_mode,
            config.update_interval_epochs,
            epoch,
            self._last_graph_rebuild_epoch,
        )

        total_start = time.perf_counter()
        embeddings_seconds = 0.0
        graph_seconds = 0.0
        if should_rebuild_graph:
            embeddings_start = time.perf_counter()
            if self._embedding_loader is None:
                self._embedding_loader = make_embedding_loader(
                    train_dataset, self.graph_positions,
                    config.embedding_batch_size, config.embedding_num_workers, seed, start_method,
                )

            embeddings = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.graph_positions,
                device=device,
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"SLRML graph embeddings - epoch {epoch}",
                embedding_kind="default",
                loader=self._embedding_loader,
            )
            embeddings_seconds = time.perf_counter() - embeddings_start

            graph_start = time.perf_counter()
            _, adjacency, degrees, positive_pair_count = build_slrml_graph(
                embeddings=embeddings,
                n_neighbors=self.n_neighbors,
                labels=self.graph_labels,
                include_supervised_graph=self.include_supervised_graph,
            )
            edge_rows, _, edge_weights = graph_upper_triangle_edges(adjacency)
            graph_seconds = time.perf_counter() - graph_start
            maybe_save_graph_diagnostics(
                request=make_graph_diagnostics_request(
                    config=config,
                    log_dir=log_dir,
                    name=f"{self.name}_graph",
                    epoch=epoch,
                    title="SLRML graph",
                ),
                embeddings=embeddings,
                adjacency=adjacency,
                positions=self.graph_positions,
                labels=dataset_labels_for_positions(train_dataset, self.graph_positions),
                known_mask=self.graph_known_mask,
            )
            self.adjacency = adjacency
            self.degrees = degrees
            self.positive_pair_count = positive_pair_count
            self.graph_edge_count = int(len(edge_rows))
            self.graph_weight_sum = float(edge_weights.sum())
            self._last_graph_rebuild_epoch = None if epoch is None else int(epoch)
        else:
            adjacency = self.adjacency
            degrees = self.degrees
            positive_pair_count = self.positive_pair_count
            logger.info(
                "Reusing SLRML graph: "
                f"{adjacency.shape[0]} nodes, {self.graph_edge_count} undirected weighted edges, "
                f"mean_degree={float(degrees.mean()):.4f}, "
                f"positive_pairs={positive_pair_count}, "
                f"n_neighbors={self.n_neighbors}"
            )

        loader_start = time.perf_counter()
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
        cache_key = (
            id(self.dataset),
            graph_batch_size,
            int(len(supervised_loader)),
            int(dataloader_kwargs.get("num_workers", 0)),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_sampler = GraphEdgeBatchSampler(
                adjacency=adjacency,
                graph_batch_size=graph_batch_size,
                seed=seed,
                num_batches=len(supervised_loader),
            )
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_sampler=self._regularizer_sampler,
                collate_fn=collate_graph_edge_batch,
                **dataloader_kwargs,
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SLRML batch streams: "
                f"supervised_examples_per_batch={int(batch_size)}, "
                f"graph_edges_per_batch={graph_batch_size}, "
                f"max_graph_nodes_before_dedup={2 * graph_batch_size}, "
                f"steps={len(supervised_loader)}"
            )
        else:
            self._regularizer_sampler.set_graph(adjacency)
        loader_seconds = time.perf_counter() - loader_start
        total_seconds = time.perf_counter() - total_start
        if should_rebuild_graph:
            logger.info(
                "Built SLRML graph: "
                f"{adjacency.shape[0]} nodes, {self.graph_edge_count} undirected weighted edges, "
                f"mean_degree={float(degrees.mean()):.4f}, "
                f"positive_pairs={positive_pair_count}, "
                f"n_neighbors={self.n_neighbors}, "
                f"timing_embeddings={embeddings_seconds:.4f}s, "
                f"timing_graph={graph_seconds:.4f}s, "
                f"timing_loader={loader_seconds:.4f}s, "
                f"timing_total={total_seconds:.4f}s"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def compute_loss(self, student_model, state, batch, device, timings=None):
        if len(batch) == 3:
            images, node_ids, edge_indices = batch
        else:
            images, node_ids = batch
            edge_indices = None

        t0 = _timing_start(device, timings)
        embeddings = utils.forward_model_inputs(
            student_model,
            images,
            device,
            use_cache=self.use_cache,
        )
        _record_timing(timings, "slrml_forward", t0, device)

        t0 = _timing_start(device, timings)
        left_indices, right_indices, weights = _graph_edge_batch(
            self.adjacency, node_ids, edge_indices
        )
        _record_timing(timings, "slrml_edge_lookup", t0, device)

        t0 = _timing_start(device, timings)
        edge_weight = torch.as_tensor(weights, dtype=embeddings.dtype, device=device) #
        _record_timing(timings, "slrml_tensor_prep", t0, device)

        t0 = _timing_start(device, timings)
        differences = (
            embeddings[left_indices.to(device)] - embeddings[right_indices.to(device)]
        )
        squared_distances = (differences * differences).sum(dim=1)
        energy = (edge_weight * squared_distances).sum()
        loss = _reduce_sampled_graph_energy(
            energy=energy,
            sampled_edges=len(weights),
            graph_edges=self.graph_edge_count,
            graph_weight=self.graph_weight_sum,
            reduction=self.reduction,
        )
        _record_timing(timings, "slrml_energy", t0, device)
        return loss

class STMLRegularizer(BaseTrainingRegularizer):
    """Use the existing STML objective as an unlabeled regularization term."""

    name = "stml"
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
        if args.use_cache:
            raise ValueError(
                "STML regularization requires stochastic multi-view augmentation "
                "and cannot use backbone caching"
            )
        if args.stml_g_dim is not None and args.stml_g_dim <= 0:
            raise ValueError("stml_g_dim must be positive when set")

    def build_dataset(self, train_dataset, split, use_cache=False):
        if len(split.unlabeled_positions) < 2:
            raise ValueError("STML regularization requires at least two unlabeled samples")
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
                device=device,
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

    def compute_loss(self, student_model, teacher_model, batch, device, timings=None):
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
    def after_optimizer_step(self, student_model, teacher_model):
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


class SeraphEntropyRegularizer(BaseTrainingRegularizer):
    """Deep SERAPH-style pairwise entropy regularizer (Niu et al., 2012).

    The original SERAPH probability for a weak pair label ``y in {+1, -1}``
    becomes, for the similar label,

        p(y=+1 | z_i, z_j) = sigmoid((eta - ||z_i - z_j||^2) / temperature).

    """

    name = "seraph_entropy"
    supports_frozen_feature_precompute = True
    uses_joint_forward = True

    DEFAULT_PARAMS = {
        "temperature": 0.1,
        "eta_mode": "fixed",
        "eta": 1.0,
        "unlabeled_ratio": 1.0,
        "unlabeled_batch_size": None,
        "include_labeled_unlabeled": False,
        "entropy_band": None,
    }
    ETA_MODES = {"batch", "fixed"}

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(regularizer_weight=regularizer_weight, supervised_weight=supervised_weight)
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown seraph_entropy regularizer_params: {unknown}")
        merged = {**self.DEFAULT_PARAMS, **params}

        self.temperature = float(merged["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("seraph_entropy temperature must be finite and positive")
        self.eta_mode = str(merged["eta_mode"])
        if self.eta_mode not in self.ETA_MODES:
            raise ValueError(
                f"seraph_entropy eta_mode must be one of {sorted(self.ETA_MODES)}"
            )
        self.eta = float(merged["eta"])
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("seraph_entropy eta must be finite and non-negative")
        self.unlabeled_ratio = float(merged["unlabeled_ratio"])
        if not math.isfinite(self.unlabeled_ratio) or self.unlabeled_ratio <= 0:
            raise ValueError("seraph_entropy unlabeled_ratio must be finite and positive")
        unlabeled_batch_size = merged["unlabeled_batch_size"]
        self.unlabeled_batch_size = (
            None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        )
        if self.unlabeled_batch_size is not None and self.unlabeled_batch_size < 2:
            raise ValueError("seraph_entropy unlabeled_batch_size must be at least 2")
        self.include_labeled_unlabeled = bool(merged["include_labeled_unlabeled"])
        entropy_band = merged["entropy_band"]
        self.entropy_band = None if entropy_band is None else float(entropy_band)
        if self.entropy_band is not None and (
            not math.isfinite(self.entropy_band) or self.entropy_band <= 0
        ):
            raise ValueError("seraph_entropy entropy_band must be finite and positive when set")
        self.dataset = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_diagnostics = {}
        self._last_comparison_diagnostics = {}
        self.collect_comparison_diagnostics = False
        self._warned_eta_fallback = False

    def validate_run_args(self, args):
        if self.eta_mode != "batch":
            return
        sampler_m = int(args.sampler_m)
        batch_size = int(args.batch_size)
        if sampler_m < 2:
            raise ValueError(
                "seraph_entropy eta_mode='batch' requires sampler_m >= 2 "
                "so labeled positive pairs exist"
            )
        if batch_size // sampler_m < 2:
            raise ValueError(
                "seraph_entropy eta_mode='batch' requires at least two classes per "
                "supervised batch so labeled negative pairs exist"
            )

    def build_dataset(self, train_dataset, split, use_cache=False):
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) < 2:
            raise ValueError("seraph_entropy regularization requires at least two unlabeled samples")
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset, use_cache=use_cache
        )
        self.dataset = UnlabeledSubset(regularizer_dataset, unlabeled_positions, num_views=1)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_diagnostics = {}
        self._last_comparison_diagnostics = {}
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
        requested_batch_size = (
            int(round(float(batch_size) * self.unlabeled_ratio))
            if self.unlabeled_batch_size is None
            else self.unlabeled_batch_size
        )
        unlabeled_batch_size = min(requested_batch_size, len(self.dataset))
        if unlabeled_batch_size < 2:
            raise ValueError("seraph_entropy needs at least two unlabeled samples per batch")

        num_workers = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        cache_key = (
            id(self.dataset),
            int(unlabeled_batch_size),
            int(num_workers),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_size=unlabeled_batch_size,
                shuffle=True,
                drop_last=True,
                **utils.make_dataloader_kwargs(
                    num_workers,
                    seed,
                    start_method,
                    persistent_workers=True,
                ),
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SERAPH two-stream loader: "
                f"labeled_batch_size={int(batch_size)}, "
                f"unlabeled_batch_size={unlabeled_batch_size}, "
                f"unlabeled_pool={len(self.dataset)}, "
                f"joint_forward_batch_size={int(batch_size) + unlabeled_batch_size}, "
                f"include_labeled_unlabeled={self.include_labeled_unlabeled}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def _prepare_embeddings(self, embeddings):
        return embeddings.float()

    @staticmethod
    def _upper_triangle_squared_distances(embeddings):
        """Squared distances between distinct L2-normalized embedding rows."""

        if len(embeddings) < 2:
            return embeddings.new_empty((0,))
        pair_indices = torch.triu_indices(
            len(embeddings),
            len(embeddings),
            offset=1,
            device=embeddings.device,
        )
        similarities = embeddings @ embeddings.T
        squared_distances = 2.0 - 2.0 * similarities
        return squared_distances[pair_indices[0], pair_indices[1]]

    @staticmethod
    def _cross_squared_distances(left_embeddings, right_embeddings):
        """Squared distances across two sets of L2-normalized embedding rows."""

        similarities = left_embeddings @ right_embeddings.T
        return (2.0 - 2.0 * similarities).reshape(-1)

    def _batch_eta(
        self,
        supervised_embeddings,
        supervised_labels,
        return_pair_details=False,
    ):
        if supervised_labels is None:
            raise ValueError("seraph_entropy eta_mode='batch' requires supervised labels")
        labels = torch.as_tensor(supervised_labels, device=supervised_embeddings.device).reshape(-1)
        if len(labels) != len(supervised_embeddings):
            raise ValueError("seraph_entropy supervised labels must align with embeddings")

        with torch.no_grad():
            distances = self._upper_triangle_squared_distances(
                supervised_embeddings.detach().float()
            )
            label_pairs = torch.triu_indices(
                len(labels), len(labels), offset=1, device=labels.device
            )
            same_class = labels[label_pairs[0]] == labels[label_pairs[1]]
            if same_class.any() and (~same_class).any():
                positive_mean = distances[same_class].mean()
                negative_mean = distances[~same_class].mean()
                eta = 0.5 * (positive_mean + negative_mean)
                eta_source = "batch_midpoint"
            else:
                eta = supervised_embeddings.new_tensor(self.eta, dtype=torch.float32)
                eta_source = "configured_fallback"

        if eta_source == "batch_midpoint":
            if return_pair_details:
                return eta, {
                    "eta_source": eta_source,
                    "pair_indices": label_pairs,
                    "squared_distances": distances,
                    "same_class": same_class,
                }
            return eta

        if not self._warned_eta_fallback:
            logger.warning(
                "SERAPH could not estimate eta from both positive and negative labeled "
                f"pairs; falling back to configured eta={self.eta}"
            )
            self._warned_eta_fallback = True
        if return_pair_details:
            return eta, {
                "eta_source": eta_source,
                "pair_indices": label_pairs,
                "squared_distances": distances,
                "same_class": same_class,
            }
        return eta

    def _select_pairs(self, squared_distances, eta):
        if self.entropy_band is None or len(squared_distances) == 0:
            return squared_distances
        selection = (squared_distances.detach().float() - eta.detach()).abs() <= self.entropy_band
        return squared_distances[selection]

    @staticmethod
    def _pair_label_entropy_from_logits(logits):
        """Numerically stable entropy with gradients through logits."""
        probabilities = torch.sigmoid(logits)
        entropy = F.softplus(logits) - probabilities * logits
        return entropy, probabilities

    def set_comparison_diagnostics_enabled(self, enabled):
        """Capture exact SERAPH endpoint and pair values for interactive debugging.

        This is deliberately separate from scalar batch diagnostics: a SERAPH
        batch has a quadratic number of pairs, so normal experiment logging
        should not copy every comparison to the host.
        """

        self.collect_comparison_diagnostics = bool(enabled)
        if not self.collect_comparison_diagnostics:
            self._last_comparison_diagnostics = {}

    @staticmethod
    def _comparison_positions(positions, count, name):
        if positions is None:
            return [None] * int(count)
        values = torch.as_tensor(positions).detach().reshape(-1).cpu().tolist()
        if len(values) != count:
            raise ValueError(
                f"SERAPH {name} positions must align with its {count} embeddings"
            )
        return [int(value) for value in values]

    @staticmethod
    def _comparison_samples(role, embeddings, positions, labels=None):
        if embeddings is None:
            return []
        host_embeddings = embeddings.detach().float().cpu().tolist()
        host_positions = SeraphEntropyRegularizer._comparison_positions(
            positions,
            len(host_embeddings),
            role,
        )
        if labels is None:
            host_labels = [None] * len(host_embeddings)
        else:
            host_labels = (
                torch.as_tensor(labels).detach().reshape(-1).cpu().tolist()
            )
            if len(host_labels) != len(host_embeddings):
                raise ValueError(
                    f"SERAPH {role} labels must align with its embeddings"
                )
        return [
            {
                "role": role,
                "batch_index": int(index),
                "source_position": position,
                "mapped_label": None if label is None else int(label),
                "embedding": [float(value) for value in embedding],
            }
            for index, (position, label, embedding) in enumerate(
                zip(host_positions, host_labels, host_embeddings)
            )
        ]

    def _comparison_selection_mask(self, squared_distances, eta):
        if self.entropy_band is None:
            return torch.ones_like(squared_distances, dtype=torch.bool)
        return (
            squared_distances.detach().float() - eta.detach().float()
        ).abs() <= self.entropy_band

    def _capture_comparison_diagnostics(
        self,
        *,
        eta,
        eta_pair_details,
        labeled_embeddings,
        supervised_labels,
        supervised_positions,
        unlabeled_embeddings,
        unlabeled_positions,
        uu_distances,
        lu_distances,
    ):
        """Snapshot the exact comparisons without retaining an autograd graph."""

        labeled_samples = self._comparison_samples(
            "labeled",
            labeled_embeddings,
            supervised_positions,
            supervised_labels,
        )
        unlabeled_samples = self._comparison_samples(
            "unlabeled",
            unlabeled_embeddings,
            unlabeled_positions,
        )

        eta_pairs = []
        eta_summary = {
            "source": "fixed" if eta_pair_details is None else eta_pair_details["eta_source"],
            "positive_pair_count": 0,
            "negative_pair_count": 0,
            "positive_mean_squared_distance": None,
            "negative_mean_squared_distance": None,
        }
        if eta_pair_details is not None:
            pair_indices = eta_pair_details["pair_indices"].detach().cpu().T.tolist()
            distances = (
                eta_pair_details["squared_distances"].detach().float().cpu().tolist()
            )
            same_class = eta_pair_details["same_class"].detach().cpu().tolist()
            for (left, right), distance, same in zip(
                pair_indices,
                distances,
                same_class,
            ):
                eta_pairs.append(
                    {
                        "kind": "ll",
                        "left_role": "labeled",
                        "left_index": int(left),
                        "right_role": "labeled",
                        "right_index": int(right),
                        "squared_distance": float(distance),
                        "same_label": bool(same),
                        "eta_group": "positive" if same else "negative",
                    }
                )
            distance_tensor = eta_pair_details["squared_distances"].detach().float()
            same_tensor = eta_pair_details["same_class"].detach()
            eta_summary.update(
                positive_pair_count=int(same_tensor.sum().item()),
                negative_pair_count=int((~same_tensor).sum().item()),
                positive_mean_squared_distance=(
                    float(distance_tensor[same_tensor].mean().cpu())
                    if same_tensor.any()
                    else None
                ),
                negative_mean_squared_distance=(
                    float(distance_tensor[~same_tensor].mean().cpu())
                    if (~same_tensor).any()
                    else None
                ),
            )

        pair_blocks = []
        if len(uu_distances) > 0:
            uu_indices = torch.triu_indices(
                len(unlabeled_embeddings),
                len(unlabeled_embeddings),
                offset=1,
                device=uu_distances.device,
            )
            pair_blocks.append(
                (
                    "uu",
                    "unlabeled",
                    "unlabeled",
                    uu_indices,
                    uu_distances,
                )
            )
        if len(lu_distances) > 0:
            labeled_count = len(labeled_embeddings)
            unlabeled_count = len(unlabeled_embeddings)
            lu_indices = torch.stack(
                [
                    torch.arange(labeled_count, device=lu_distances.device).repeat_interleave(
                        unlabeled_count
                    ),
                    torch.arange(unlabeled_count, device=lu_distances.device).repeat(
                        labeled_count
                    ),
                ]
            )
            pair_blocks.append(
                (
                    "lu",
                    "labeled",
                    "unlabeled",
                    lu_indices,
                    lu_distances,
                )
            )

        block_values = []
        for kind, left_role, right_role, pair_indices, distances in pair_blocks:
            selection = self._comparison_selection_mask(distances, eta)
            selected_count = int(selection.sum().item())
            logits = (eta.detach().float() - distances.detach().float()) / self.temperature
            entropies, probabilities = self._pair_label_entropy_from_logits(logits)
            block_values.append(
                {
                    "kind": kind,
                    "left_role": left_role,
                    "right_role": right_role,
                    "pair_indices": pair_indices.detach(),
                    "distances": distances.detach().float(),
                    "selection": selection.detach(),
                    "logits": logits.detach(),
                    "probabilities": probabilities.detach(),
                    "entropies": entropies.detach(),
                    "selected_count": selected_count,
                }
            )

        total_selected_count = sum(block["selected_count"] for block in block_values)
        loss_pairs = []
        loss_blocks = []
        for block in block_values:
            selected_count = block["selected_count"]
            block_weight = (
                float(selected_count) / float(total_selected_count)
                if selected_count > 0 and total_selected_count > 0
                else 0.0
            )
            mean_selected_entropy = (
                float(block["entropies"][block["selection"]].mean().cpu())
                if selected_count > 0
                else None
            )
            loss_blocks.append(
                {
                    "kind": block["kind"],
                    "candidate_count": int(len(block["distances"])),
                    "selected_count": selected_count,
                    "mean_selected_entropy": mean_selected_entropy,
                    "block_weight": block_weight,
                }
            )

            pair_indices = block["pair_indices"].cpu().T.tolist()
            distances = block["distances"].cpu().tolist()
            selection = block["selection"].cpu().tolist()
            logits = block["logits"].cpu().tolist()
            probabilities = block["probabilities"].cpu().tolist()
            entropies = block["entropies"].cpu().tolist()
            for (left, right), distance, selected, logit, probability, entropy in zip(
                pair_indices,
                distances,
                selection,
                logits,
                probabilities,
                entropies,
            ):
                contribution = (
                    float(entropy) / float(total_selected_count)
                    if selected and total_selected_count > 0
                    else 0.0
                )
                loss_pairs.append(
                    {
                        "kind": block["kind"],
                        "left_role": block["left_role"],
                        "left_index": int(left),
                        "right_role": block["right_role"],
                        "right_index": int(right),
                        "squared_distance": float(distance),
                        "logit": float(logit),
                        "same_probability": float(probability),
                        "entropy": float(entropy),
                        "selected": bool(selected),
                        "loss_contribution": contribution,
                    }
                )

        self._last_comparison_diagnostics = {
            "eta": float(eta.detach().float().cpu()),
            "temperature": self.temperature,
            "entropy_band": self.entropy_band,
            "eta_summary": eta_summary,
            "samples": {
                "labeled": labeled_samples,
                "unlabeled": unlabeled_samples,
            },
            "eta_pairs": eta_pairs,
            "loss_blocks": loss_blocks,
            "loss_pairs": loss_pairs,
            "reconstructed_loss": float(
                sum(pair["loss_contribution"] for pair in loss_pairs)
            ),
        }

    def _pair_diagnostics(
        self,
        eta,
        uu_distances,
        lu_distances,
        selected_uu_distances,
        selected_lu_distances,
    ):
        """Measure collapse on all candidate pairs, before optional banding."""

        candidate_blocks = [
            block for block in (uu_distances, lu_distances) if block.numel() > 0
        ]
        with torch.no_grad():
            if candidate_blocks:
                diagnostic_distances = torch.cat(candidate_blocks).detach().float()
                diagnostic_probabilities = torch.sigmoid(
                    (eta.detach().float() - diagnostic_distances) / self.temperature
                )
                mean_same_probability = diagnostic_probabilities.mean()
                confident_similar_fraction = (
                    diagnostic_probabilities >= 0.9
                ).float().mean()
                confident_dissimilar_fraction = (
                    diagnostic_probabilities <= 0.1
                ).float().mean()
            else:
                mean_same_probability = eta.detach().new_zeros(())
                confident_similar_fraction = eta.detach().new_zeros(())
                confident_dissimilar_fraction = eta.detach().new_zeros(())

        total_pair_count = int(uu_distances.numel() + lu_distances.numel())
        selected_pair_count = int(
            selected_uu_distances.numel() + selected_lu_distances.numel()
        )
        return {
            # Keep tensors detached on-device here. batch_diagnostics batches
            # their host transfer into one synchronization only when logging is enabled.
            "train/seraph_eta": eta.detach(),
            "train/seraph_mean_same_probability": mean_same_probability,
            "train/seraph_confident_similar_fraction": confident_similar_fraction,
            "train/seraph_confident_dissimilar_fraction": confident_dissimilar_fraction,
            "train/seraph_uu_pair_count": float(uu_distances.numel()),
            "train/seraph_lu_pair_count": float(lu_distances.numel()),
            "train/seraph_selected_uu_pair_count": float(selected_uu_distances.numel()),
            "train/seraph_selected_lu_pair_count": float(selected_lu_distances.numel()),
            "train/seraph_selected_pair_count": float(selected_pair_count),
            "train/seraph_pair_selection_fraction": (
                float(selected_pair_count) / float(total_pair_count)
                if total_pair_count > 0
                else 0.0
            ),
        }

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
        supervised_positions=None,
    ):
        self._last_comparison_diagnostics = {}
        if regularizer_embeddings is None:
            if batch is None:
                raise ValueError("seraph_entropy requires a regularizer batch or embeddings")
            images = batch[0]
            t0 = _timing_start(device, timings)
            regularizer_embeddings = utils.forward_model_inputs(
                student_model,
                images,
                device,
                use_cache=self.use_cache,
            )
            _record_timing(timings, "seraph_forward", t0, device)
        t0 = _timing_start(device, timings)
        unlabeled = self._prepare_embeddings(regularizer_embeddings)
        needs_labeled_embeddings = (
            self.eta_mode == "batch" or self.include_labeled_unlabeled
        )
        if supervised_embeddings is None and needs_labeled_embeddings:
            raise ValueError(
                "seraph_entropy requires supervised embeddings for batch eta "
                "or labeled-unlabeled pairs"
            )
        labeled = (
            None
            if supervised_embeddings is None
            else self._prepare_embeddings(supervised_embeddings)
        )
        eta_pair_details = None
        if self.eta_mode == "fixed":
            eta = unlabeled.new_tensor(self.eta, dtype=torch.float32)
        elif self.collect_comparison_diagnostics:
            eta, eta_pair_details = self._batch_eta(
                labeled,
                supervised_labels,
                return_pair_details=True,
            )
        else:
            eta = self._batch_eta(labeled, supervised_labels)

        all_uu_distances = self._upper_triangle_squared_distances(unlabeled)
        selected_uu_distances = self._select_pairs(all_uu_distances, eta)
        all_lu_distances = unlabeled.new_empty((0,))
        selected_lu_distances = unlabeled.new_empty((0,))
        if self.include_labeled_unlabeled:
            all_lu_distances = self._cross_squared_distances(labeled, unlabeled)
            selected_lu_distances = self._select_pairs(all_lu_distances, eta)
        if self.collect_batch_diagnostics:
            self._last_diagnostics = self._pair_diagnostics(
                eta=eta,
                uu_distances=all_uu_distances,
                lu_distances=all_lu_distances,
                selected_uu_distances=selected_uu_distances,
                selected_lu_distances=selected_lu_distances,
            )
        else:
            self._last_diagnostics = {}

        if self.collect_comparison_diagnostics:
            unlabeled_positions = None
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                unlabeled_positions = batch[2]
            self._capture_comparison_diagnostics(
                eta=eta,
                eta_pair_details=eta_pair_details,
                labeled_embeddings=labeled,
                supervised_labels=supervised_labels,
                supervised_positions=supervised_positions,
                unlabeled_embeddings=unlabeled,
                unlabeled_positions=unlabeled_positions,
                uu_distances=all_uu_distances,
                lu_distances=all_lu_distances,
            )

        pair_blocks = [selected_uu_distances]
        if self.include_labeled_unlabeled:
            pair_blocks.append(selected_lu_distances)
        nonempty_pair_blocks = [block for block in pair_blocks if len(block) > 0]
        if not nonempty_pair_blocks:
            _record_timing(timings, "seraph_entropy", t0, device)
            return regularizer_embeddings.sum() * 0.0

        # Pool first so every selected UU or LU pair receives identical weight.
        selected_distances = torch.cat(nonempty_pair_blocks)
        logits = (eta - selected_distances.float()) / self.temperature
        entropy, _ = self._pair_label_entropy_from_logits(logits)
        loss = entropy.mean()
        sum_loss = entropy.sum()
        _record_timing(timings, "seraph_entropy", t0, device)
        return loss

    def batch_diagnostics(self):
        diagnostics = dict(self._last_diagnostics)
        tensor_items = [
            (name, value)
            for name, value in diagnostics.items()
            if torch.is_tensor(value)
        ]
        if tensor_items:
            host_values = torch.stack(
                [value.detach().float().reshape(()) for _, value in tensor_items]
            ).cpu().tolist()
            diagnostics.update(
                {name: float(value) for (name, _), value in zip(tensor_items, host_values)}
            )
        return diagnostics

    def comparison_diagnostics(self):
        """Return the most recently captured sample-level SERAPH trace."""

        return copy.deepcopy(self._last_comparison_diagnostics)


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
    K*C-way exemplar softmax. All training classes must have labeled support and
    participate in every denominator. For large C this intentionally retains
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
        labeled_classes = np.unique(labels).astype(np.int64)
        missing_classes = np.setdiff1d(unique_labels, labeled_classes)
        if len(missing_classes) > 0:
            raise ValueError(
                "hoffer_entropy requires at least one labeled exemplar for every training class; "
                f"missing classes: {missing_classes.tolist()}"
            )
        class_candidates = [
            num_unlabeled + np.flatnonzero(labels == label)
            for label in unique_labels
        ]
        if len(class_candidates) < 2:
            raise ValueError("hoffer_entropy regularization requires at least two labeled classes")
        self.class_candidates = class_candidates
        self.reference_class_labels = unique_labels
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
        embeddings = utils.forward_model_inputs(
            student_model,
            images,
            device,
            use_cache=self.use_cache,
        )
        _record_timing(timings, "hoffer_forward", t0, device)

        role_tensor = torch.as_tensor(is_reference)
        reference_count = int(role_tensor.bool().sum().item())
        unlabeled_count = int(role_tensor.numel() - reference_count)
        logger.debug(
            "Hoffer regularizer loss batch: "
            f"unlabeled_count={unlabeled_count}, "
            f"reference_count={reference_count}, "
            f"total_count={int(role_tensor.numel())}"
        )

        reference_mask = is_reference.to(device=device).bool()
        references = embeddings[reference_mask]
        unlabeled = embeddings[~reference_mask]
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
    allowed_params = {
        "regularizer",
        "regularizer_params",
        "regularizer_weight",
        "supervised_weight",
    }

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
        return {
            "regularizer_name": regularizer_name,
            "regularizer_params": regularizer_params,
            "regularizer_weight": params.get("regularizer_weight", 1.0),
            "supervised_weight": params.get("supervised_weight", 1.0),
        }

    def make_regularizer(self, config):
        params = self.resolve_params(config)
        regularizer_name = params.pop("regularizer_name")
        try:
            regularizer_class = REGULARIZER_REGISTRY[regularizer_name]
        except KeyError as exc:
            if regularizer_name == "slrmml":
                raise ValueError(
                    "Multimodal SLRMML is not implemented by the single-stream training pipeline; "
                    "use regularizer='slrml' for the one-stream SLRML regularizer"
                ) from exc
            raise ValueError(
                f"Unknown regularizer {regularizer_name!r}. Available: {sorted(REGULARIZER_REGISTRY)}"
            ) from exc
        regularizer_params = params.pop("regularizer_params")
        return regularizer_class(**params, **regularizer_params)

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
        logger.info(f"{self.name} confidence distribution: {summarize_numeric_values(confidences)}")

        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=confidences,
        )


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
            },
            **params,
        )
        unlabeled_start = len(split.labeled_positions)
        unlabeled_probabilities = probabilities[unlabeled_start:]
        pseudo_labels = np.argmax(unlabeled_probabilities, axis=1).astype(np.int64)
        unlabeled_confidences = confidences[unlabeled_start:].astype(np.float32)
        logger.info(
            f"{self.name} entropy-certainty distribution: "
            f"{summarize_numeric_values(unlabeled_confidences)}"
        )
        return PseudoLabelResult(
            positions=split.unlabeled_positions,
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
    if str(params["linear_solver"]) not in {"cholmod", "cg"}:
        raise ValueError("mixed_label_propagation linear_solver must be one of ['cholmod', 'cg']")


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
    if str(params["linear_solver"]) not in {"cholmod", "cg"}:
        raise ValueError("faiss_label_spreading linear_solver must be one of ['cholmod', 'cg']")


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
    if str(params["linear_solver"]) not in {"cg", "cholmod"}:
        raise ValueError("iscen_label_spreading linear_solver must be one of ['cg', 'cholmod']")


REGULARIZER_REGISTRY = {
    "stml": STMLRegularizer,
    "lrml": LRMLRegularizer,
    "slrml": SLRMLRegularizer,
    "seraph_entropy": SeraphEntropyRegularizer,
    "hoffer_entropy": HofferEntropyRegularizer,
    "simmatch_v2": SimMatchV2Regularizer,
}


METHOD_REGISTRY = {
    # first implementation
    "faiss_majority_vote_knn": FaissKNNMajorityVotePseudoLabeler(name="faiss_knn", n_neighbors=10), # find reference anywhere in literature
    # implement distance based faiss implementation
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
    "mixed_label_propagation": MixedLabelPropagationPseudoLabeler(),
    # Generic composition point for supervised loss + unlabeled regularizer.
    "regularized": RegularizedSemiSupervisedMethod(name="regularized"),
    # Convenience alias; this is still the same supervised-loss + regularizer
    # composition and therefore works with every supported PML loss/miner.
    "simmatch_v2": RegularizedSemiSupervisedMethod(
        name="simmatch_v2",
        default_regularizer="simmatch_v2",
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


def validate_ssl_config(config, path=None):
    """Validate label-selection and method settings before any data is loaded."""

    source = f" in {path}" if path is not None else ""
    if config.method != "none" and config.method not in METHOD_REGISTRY and config.method not in LOSS_DRIVEN_METHODS:
        raise ValueError(f"Unknown SSL method{source}: {config.method}. Available: {available_methods()}")
    if config.update_mode not in UPDATE_MODES:
        raise ValueError(f"Unknown SSL update_mode{source}: {config.update_mode}. Available: {sorted(UPDATE_MODES)}")
    if config.update_interval_epochs <= 0:
        raise ValueError(f"update_interval_epochs must be positive{source}")
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
    if config.labeled_batch_size is not None:
        if isinstance(config.labeled_batch_size, bool) or not isinstance(
            config.labeled_batch_size,
            (int, np.integer),
        ):
            raise ValueError(f"labeled_batch_size must be an integer or null{source}")
        if config.labeled_batch_size <= 0:
            raise ValueError(f"labeled_batch_size must be positive when set{source}")
        if config.method != "iscen_label_spreading":
            raise ValueError(
                "labeled_batch_size currently enables TwoStreamMPerClassBatchSampler only for "
                f"method='iscen_label_spreading'{source}"
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
    if config.max_unlabeled_samples is not None and config.max_unlabeled_samples <= 0:
        raise ValueError(f"max_unlabeled_samples must be positive{source}")
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

    return None if not config.enabled else METHOD_REGISTRY.get(config.method)


def is_regularization_method(config):
    method = get_method(config)
    return method is not None and method.is_regularization_method


def is_pseudo_label_method(config):
    method = get_method(config)
    return method is not None and method.generates_pseudo_labels


def make_pseudo_label_diagnostics_tracker(log_dir, config):
    if not is_pseudo_label_method(config):
        return None
    if config.pseudo_label_diagnostics_mode == "off":
        return None
    return PseudoLabelDiagnosticsTracker(
        log_dir,
        mode=config.pseudo_label_diagnostics_mode,
    )


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
    )
    # Count class coverage after selection because some sampling modes expose
    # only a subset of classes.
    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    labeled_labels = labels[split.labeled_positions]
    num_labeled_classes = int(len(np.unique(labeled_labels))) if len(labeled_labels) > 0 else 0
    num_total_classes = int(len(np.unique(labels)))
    logger.info(
        "Semi-supervised split: "
        f"label_mode={config.label_sampling_mode}, "
        f"{len(split.labeled_positions)} labeled across {num_labeled_classes}/{num_total_classes} classes, "
        f"{len(split.unlabeled_positions)} unlabeled candidates"
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
):
    """Generate, filter, and merge pseudo-labels with the true labeled subset."""

    if not config.enabled:
        # When SSL is disabled and no split-only supervised baseline is needed,
        # use the original training dataset unchanged.
        return train_dataset
    if config.method in LOSS_DRIVEN_METHODS:
        raise ValueError(
            f"{config.method} is a loss-driven SSL method and does not generate pseudo-labels; "
            "use build_loss_driven_training_dataset"
        )
    method = METHOD_REGISTRY[config.method]
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

    # Filter before merging so low-confidence/invalid predictions never affect
    # sampler class counts or training batches.
    pseudo_labels = filter_pseudo_labels(
        pseudo_labels=raw_pseudo_labels,
        confidence_threshold=config.confidence_threshold,
        valid_mapped_labels=set(train_labels_mapper.values()),
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

    return make_relabeled_training_dataset(
        train_dataset=train_dataset,
        train_labels_mapper=train_labels_mapper,
        labeled_positions=split.labeled_positions,
        pseudo_labels=pseudo_labels,
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
