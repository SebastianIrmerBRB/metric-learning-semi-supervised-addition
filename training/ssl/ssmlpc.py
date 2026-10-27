"""The unlabeled term of Soleymani Baghshah and Bagheri Shouraki (IJCAI 2009).

*Semi-Supervised Metric Learning Using Pairwise Constraints*, IJCAI 2009,
pages 1217-1222. The paper's full objective is the constrained trace ratio of
its eq. (1)/(2),

    W* = argmax_{W^T W = I}  tr(W^T S_b W) / (tr(W^T S_w W) + gamma J(W)),

where ``S_w`` and ``S_b`` are the scatter matrices of the positive (must-link)
and negative (cannot-link) constraint sets of eqs. (3) and (4). Only ``J(W)``
reads the unlabeled data, and only ``J(W)`` is implemented here: it is the term
that plugs into this repository's ``supervised_loss + w * regularizer`` shape,
with the repository's own supervised metric loss standing in for the
constraint scatters.

Eq. (6) defines it as the locally linear reconstruction error of the
transformed points,

    J(W) = sum_i || y_i - sum_{j in N_k(x_i)} s*_ij y_j ||^2  =  tr(Y E Y^T),
    E = (I - S*)^T (I - S*),

with ``y_i = W^T x_i`` and ``S*`` the eq. (5) reconstruction weights of the
*input* points, which the paper solves once and holds fixed. The idea is
borrowed from Locally Linear Embedding (Roweis and Saul, 2000): the weights
that reconstruct a point from its neighbors describe the local geometry, and
re-using them in the target space asks the transformation to preserve it.

What this term does and does not constrain
------------------------------------------
Eq. (5) forces ``sum_j s*_ij = 1``, which has two consequences worth stating
before anyone reads a training curve.

First, the penalty is *affine invariant*. Because the weights sum to one, the
residual of an affine map ``y = A x + b`` is ``A`` applied to the input's own
reconstruction residual, so any affine image of the data the weights were fit
on carries the energy the fit already left behind and no more. This is a
materially weaker attractor than a graph Laplacian: on L2-normalized
embeddings the Laplacian's ``sum_ij w_ij ||z_i - z_j||^2`` reduces to
``-2 sum_ij w_ij <z_i, z_j>`` plus a constant, so it pays for raising neighbor
cosines and keeps paying as the embedding contracts. Eq. (6) has no such term.

Second, and for the same reason, **a constant embedding is a global minimizer**:
``z_i - sum_j s_ij z_j = c (1 - sum_j s_ij) = 0``. The paper is not exposed to
this, because ``J`` sits in the *denominator* of a ratio whose numerator pushes
cannot-link pairs apart, and because ``W^T W = I`` forbids a rank-deficient
map. Taking the term out of the ratio discards both guards. Nothing in this
module reinstates them; ``train/ssmlpc/centroid_norm`` and
``train/ssmlpc/residual_scale`` are logged so the failure is visible rather
than inferred. See ``docs/ssmlpc.md``.

Where the weights come from
---------------------------
``graph_source='backbone_features'`` is the paper: ``S*`` is solved once, on
the input representation, and never refreshed -- under a frozen backbone that
is the cached DINOv2 feature matrix, which is literally the ``X`` of eq. (5)
while ``W`` is the trained projection head. ``graph_source='embeddings'``
instead re-solves on the current projected embedding on the engine's usual
refresh schedule, which is what the Laplacian regularizers in this repository
do; it is available for comparability, not for fidelity.
"""

import math

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

import utils

from .algorithms import (
    DEFAULT_LLE_RECONSTRUCTION_REGULARIZATION,
    LLE_RECONSTRUCTION_METRICS,
    build_lle_reconstruction_graph,
)
from .config import should_rebuild_on_epoch
from .data import (
    CombinedTrainingLoader,
    GraphNeighborhoodBatchSampler,
    LRMLGraphDataset,
    collate_graph_neighborhood_batch,
)
from .embeddings import extract_embeddings, make_embedding_loader
from .graph_diagnostics import dataset_labels_for_positions
from .interfaces import BaseTrainingRegularizer


def reconstruction_energy(
    embeddings,
    anchor_rows,
    neighbor_columns,
    neighbor_weights,
    reduction,
    *,
    sampled_anchors=None,
    graph_anchors=None,
):
    """Evaluate eq. (6) over one batch of sampled anchors.

    ``neighbor_columns`` and ``neighbor_weights`` are ``[num_anchors, k]``, so
    the reconstruction is a dense gather and a weighted sum along ``k``. That
    ordering is fixed, which makes the result reproducible on CUDA; a
    scatter-add over the shared neighbors would not be.

    The energy is accumulated in float32 whatever autocast is doing, for the
    reason ``_local_graph_energy`` pins the Laplacian there: it is an unbounded
    sum of squares, and CPU autocast promotes neither ``normalize`` nor ``sum``.
    """

    embeddings = embeddings.float()
    weights = neighbor_weights.to(dtype=embeddings.dtype, device=embeddings.device)
    columns = neighbor_columns.to(embeddings.device)
    anchors = anchor_rows.to(embeddings.device)
    # [num_anchors, k, dim] -> [num_anchors, dim]
    reconstructions = (weights[:, :, None] * embeddings[columns]).sum(dim=1)
    residuals = embeddings[anchors] - reconstructions
    squared = residuals.square().sum(dim=1)
    energy = squared.sum()
    if reduction == "mean":
        return energy / float(len(anchors)), residuals
    if reduction == "sum":
        # tr(Y E Y^T) is a sum over every anchor. Anchors are drawn uniformly,
        # so rescaling a sampled batch by the pool size is unbiased for it.
        if sampled_anchors is None or graph_anchors is None:
            raise ValueError("reduction='sum' needs the sampled and total anchor counts")
        return energy * float(graph_anchors) / float(sampled_anchors), residuals
    if reduction == "batch_sum":
        return energy, residuals
    raise ValueError(f"unknown ssmlpc reduction: {reduction!r}")


class SsmlpcRegularizer(BaseTrainingRegularizer):
    """Eq. (6): preserve the locally linear geometry of the unlabeled pool.

    The term is anchored on nodes rather than on edges. Each summand needs one
    node and every neighbor it is reconstructed from at once, so the batching
    goes through :class:`GraphNeighborhoodBatchSampler` instead of the edge
    sampler the Laplacian regularizers share.
    """

    name = "ssmlpc"
    # One deterministic view per node, exactly as the Laplacian path needs: the
    # weights are solved on a fixed representation and the energy is evaluated
    # on the same nodes' embeddings.
    supports_frozen_feature_precompute = True
    # The whole refresh is one make_loader call over a fixed node set.
    supports_sample_scoped_refresh = True

    DEFAULT_PARAMS = {
        # Section 3.1: "The parameters of our algorithm are set to k = 10".
        "n_neighbors": 10,
        # LLE fits in the input space, so its neighborhoods are Euclidean.
        # "cosine" reuses the ranking the Laplacian regularizers build, which
        # lets a study hold the neighborhoods fixed and vary only the weights.
        "neighbor_metric": "euclidean",
        # Roweis and Saul's conditioner for eq. (5)'s local Gram matrix.
        "reconstruction_regularization": DEFAULT_LLE_RECONSTRUCTION_REGULARIZATION,
        # "backbone_features" is the paper: S* is the input data's, solved once.
        # "embeddings" re-solves from the current projection on the engine's
        # refresh schedule, matching this repository's Laplacian regularizers.
        "graph_source": "backbone_features",
        "graph_on": "all",
        # Sampled anchors per step; None matches the supervised batch size.
        "graph_batch_size": None,
        # "mean" is the per-anchor mean. "sum" rescales the sampled batch to
        # eq. (6)'s full trace, which is unbiased under uniform anchor
        # sampling. "batch_sum" is the raw batch sum, whose scale then depends
        # on graph_batch_size.
        "reduction": "mean",
        # Oracle ablation that deliberately reads hidden training labels: drop
        # every different-class neighbor *before* eq. (5) is solved, so the
        # retained columns are re-fitted and the weights still sum to one.
        "oracle_same_class_neighbors": False,
    }
    GRAPH_SOURCE_CHOICES = ("backbone_features", "embeddings")
    GRAPH_ON_CHOICES = ("all", "unlabeled")
    REDUCTION_CHOICES = ("mean", "sum", "batch_sum")

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0, **params):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        unknown = sorted(set(params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"Unknown ssmlpc regularizer_params: {unknown}")
        merged = {**self.DEFAULT_PARAMS, **params}

        self.n_neighbors = int(merged["n_neighbors"])
        if self.n_neighbors <= 0:
            raise ValueError("ssmlpc n_neighbors must be positive")
        self.neighbor_metric = str(merged["neighbor_metric"])
        if self.neighbor_metric not in LLE_RECONSTRUCTION_METRICS:
            raise ValueError(
                "ssmlpc neighbor_metric must be one of "
                f"{list(LLE_RECONSTRUCTION_METRICS)}"
            )
        self.reconstruction_regularization = float(
            merged["reconstruction_regularization"]
        )
        if (
            not math.isfinite(self.reconstruction_regularization)
            or self.reconstruction_regularization < 0
        ):
            raise ValueError(
                "ssmlpc reconstruction_regularization must be finite and "
                "non-negative"
            )
        self.graph_source = str(merged["graph_source"])
        if self.graph_source not in self.GRAPH_SOURCE_CHOICES:
            raise ValueError(
                f"ssmlpc graph_source must be one of {list(self.GRAPH_SOURCE_CHOICES)}"
            )
        self.graph_on = str(merged["graph_on"])
        if self.graph_on not in self.GRAPH_ON_CHOICES:
            raise ValueError(
                f"ssmlpc graph_on must be one of {list(self.GRAPH_ON_CHOICES)}"
            )
        graph_batch_size = merged["graph_batch_size"]
        self.graph_batch_size = (
            None if graph_batch_size is None else int(graph_batch_size)
        )
        if self.graph_batch_size is not None and self.graph_batch_size <= 0:
            raise ValueError("ssmlpc graph_batch_size must be positive when set")
        self.reduction = str(merged["reduction"])
        if self.reduction not in self.REDUCTION_CHOICES:
            raise ValueError(
                f"ssmlpc reduction must be one of {list(self.REDUCTION_CHOICES)}"
            )
        self.oracle_same_class_neighbors = bool(merged["oracle_same_class_neighbors"])
        # configure_graph_batching overwrites both from the shared ssl_config
        # keys. They are set here too so the regularizer is usable without that
        # hook -- a benchmark script, a unit test, or any caller that builds it
        # directly rather than through make_regularizer.
        self.graph_edge_ablation = (
            "oracle_pruned" if self.oracle_same_class_neighbors else "off"
        )
        self.graph_edge_ablation_seed = 0

        self.dataset = None
        self.graph_positions = None
        self.graph_known_mask = None
        self.neighbor_indices = None
        self.neighbor_mask = None
        self.neighbor_weights = None
        self.valid_anchors = None
        self.graph_anchor_count = None
        self.graph_metadata = {}
        self._train_dataset_labels = None
        self._last_graph_rebuild_epoch = None
        self._weights_are_static = False
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        self._last_diagnostics = {}

    # -- configuration -----------------------------------------------------

    def configure_graph_batching(self, config):
        """Apply the shared ``ssl_config.graph_edge_ablation`` key.

        The same hook LRML uses, so an oracle arm is configured identically for
        both regularizers. ``regularizer_params.oracle_same_class_neighbors`` is
        the local shorthand for ``oracle_pruned`` and the two cannot disagree.
        """

        configured_ablation = str(config.graph_edge_ablation)
        if self.oracle_same_class_neighbors and configured_ablation != "off":
            raise ValueError(
                "ssmlpc regularizer_params.oracle_same_class_neighbors cannot be "
                "combined with ssl_config.graph_edge_ablation"
            )
        self.graph_edge_ablation = (
            "oracle_pruned"
            if self.oracle_same_class_neighbors
            else configured_ablation
        )
        ablation_seed = (
            config.seed
            if config.graph_edge_ablation_seed is None
            else config.graph_edge_ablation_seed
        )
        self.graph_edge_ablation_seed = 0 if ablation_seed is None else int(ablation_seed)

    def validate_config(self, config, source=""):
        if (
            self.graph_source == "backbone_features"
            and config.update_mode == "every_n_samples"
        ):
            logger.info(
                f"ssmlpc{source}: graph_source='backbone_features' solves eq. (5) "
                "once, so the sample-scoped refresh schedule has nothing to "
                "rebuild"
            )
        return None

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("ssmlpc regularization requires batch_size >= 2")
        if self.graph_source == "backbone_features" and not args.use_cache:
            raise ValueError(
                "ssmlpc graph_source='backbone_features' needs the cached "
                "frozen-backbone path (--use_cache) so eq. (5) can be solved on "
                "the input representation; use graph_source='embeddings' to fit "
                "the weights on the projected embedding instead"
            )

    def graph_source_is_static(self):
        """Whether eq. (5) is solved once, as the paper solves it."""

        return self.graph_source == "backbone_features"

    # -- dataset -----------------------------------------------------------

    def build_dataset(self, train_dataset, split, use_cache=False):
        if self.graph_on == "all":
            positions = np.concatenate(
                [split.labeled_positions, split.unlabeled_positions]
            )
        else:
            positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        positions = np.unique(np.asarray(positions, dtype=np.int64))
        if len(positions) < 2:
            raise ValueError("ssmlpc regularization requires at least two graph samples")
        utils.shutdown_dataloaders(self._regularizer_loader, self._embedding_loader)

        self.graph_positions = positions
        self.graph_known_mask = np.isin(
            positions,
            np.asarray(split.labeled_positions, dtype=np.int64),
        )
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset,
            use_cache=use_cache,
        )
        self._train_dataset_labels = np.asarray(train_dataset.labels, dtype=np.int64)
        self.dataset = LRMLGraphDataset(regularizer_dataset, positions)

        self.neighbor_indices = None
        self.neighbor_mask = None
        self.neighbor_weights = None
        self.valid_anchors = None
        self.graph_anchor_count = None
        self.graph_metadata = {}
        self._last_graph_rebuild_epoch = None
        self._weights_are_static = False
        self._regularizer_sampler = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._embedding_loader = None
        self._last_diagnostics = {}
        return self.dataset

    # -- eq. (5) -----------------------------------------------------------

    def _backbone_feature_matrix(self, train_dataset, config, seed, start_method):
        """The input representation eq. (5) is solved on, as a dense matrix.

        Under the frozen-backbone path the dataset items *are* the backbone
        features, so this reads them straight through the deterministic
        embedding loader without a model forward. A dataset that still yields
        images has no input representation to fit, and says so rather than
        silently fitting the weights on pixels.
        """

        if self._embedding_loader is None:
            self._embedding_loader = make_embedding_loader(
                train_dataset,
                self.graph_positions,
                config.embedding_batch_size,
                config.embedding_num_workers,
                seed,
                start_method,
            )
        rows = []
        for batch in self._embedding_loader:
            features = batch[0]
            if not torch.is_tensor(features) or features.ndim != 2:
                raise ValueError(
                    "ssmlpc graph_source='backbone_features' needs 2-D cached "
                    "backbone features; this dataset yields "
                    f"{tuple(getattr(features, 'shape', ()))}-shaped items. Run "
                    "with the frozen-feature precompute enabled, or set "
                    "graph_source='embeddings'"
                )
            rows.append(features.detach().float().cpu().numpy())
        features = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
        if len(features) != len(self.graph_positions):
            raise RuntimeError(
                "ssmlpc backbone feature matrix does not cover every graph node"
            )
        return features

    def _solve_graph(self, model, train_dataset, device, config, seed, start_method, epoch):
        if self.graph_source == "backbone_features":
            features = self._backbone_feature_matrix(
                train_dataset,
                config,
                seed,
                start_method,
            )
            source_description = "cached backbone features"
        else:
            features = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.graph_positions,
                device=self.get_ssl_device(device),
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"ssmlpc reconstruction embeddings - epoch {epoch}",
                loader=self._embedding_loader,
            )
            features = np.ascontiguousarray(features, dtype=np.float32)
            source_description = "projected embeddings"

        if not np.all(np.isfinite(features)):
            raise ValueError("ssmlpc reconstruction features must be finite")

        ablation_labels = None
        if self.graph_edge_ablation != "off":
            ablation_labels = dataset_labels_for_positions(
                train_dataset,
                self.graph_positions,
            )
            if ablation_labels is None:
                raise ValueError(
                    f"ssmlpc graph_edge_ablation={self.graph_edge_ablation!r} "
                    "requires dataset labels"
                )

        (
            neighbor_indices,
            neighbor_mask,
            neighbor_weights,
            metadata,
        ) = build_lle_reconstruction_graph(
            features,
            self.n_neighbors,
            metric=self.neighbor_metric,
            regularization=self.reconstruction_regularization,
            ablation_mode=self.graph_edge_ablation,
            ablation_labels=ablation_labels,
            ablation_seed=self.graph_edge_ablation_seed,
        )

        self.neighbor_indices = neighbor_indices
        self.neighbor_mask = neighbor_mask
        self.neighbor_weights = neighbor_weights
        self.valid_anchors = neighbor_mask.any(axis=1)
        self.graph_anchor_count = int(self.valid_anchors.sum())
        self.graph_metadata = {
            **metadata,
            "graph_source": self.graph_source,
            "feature_dimension": int(features.shape[1]),
        }
        if metadata["ablation"] is not None:
            ablation = metadata["ablation"]
            logger.warning(
                "SSMLPC NEIGHBOR ABLATION uses hidden true labels: "
                f"mode={ablation['mode']}, "
                f"raw_neighbor_purity={ablation['neighbor_purity']:.3%}, "
                f"retained {ablation['retained_neighbors']} / "
                f"{ablation['original_neighbors']} neighbors "
                f"({ablation['retained_neighbor_fraction']:.3%}), "
                f"retained_neighbor_purity="
                f"{ablation['retained_neighbor_purity']:.3%}; "
                "eq. (5) was re-solved on the retained columns so the weights "
                "still sum to one; "
                f"isolated_nodes={ablation['isolated_nodes']}"
            )
        logger.info(
            f"Solved ssmlpc eq. (5) on {source_description}: "
            f"{metadata['nodes']} nodes, k={metadata['n_neighbors']} "
            f"({self.neighbor_metric}), "
            f"anchors={self.graph_anchor_count}, "
            f"isolated_nodes={metadata['isolated_nodes']}, "
            f"weight_range=[{metadata['weight_min']:.4g}, "
            f"{metadata['weight_max']:.4g}], "
            f"negative_weights={metadata['negative_weight_fraction']:.2%}, "
            f"regularization={self.reconstruction_regularization:g}"
        )

    # -- loader ------------------------------------------------------------

    def make_loader(
        self, model, train_dataset, supervised_loader, device, config,
        batch_size, seed, num_workers, start_method, epoch,
        log_dir=None,
    ):
        if self.dataset is None or self.graph_positions is None:
            raise RuntimeError("build_dataset must be called before make_loader")

        has_graph = self.neighbor_weights is not None
        # Consume unconditionally: a request must not survive a call that
        # rebuilt for another reason and force a second rebuild afterwards.
        refresh_requested = self.consume_refresh_request()
        if self.graph_source_is_static():
            # The paper solves S* once, on data that does not move. Re-solving
            # it would make the term track the embedding it is supposed to
            # constrain, which is a different method.
            should_rebuild = not has_graph
        else:
            should_rebuild = (
                not has_graph
                or refresh_requested
                or should_rebuild_on_epoch(
                    config.update_mode,
                    config.update_interval_epochs,
                    epoch,
                    self._last_graph_rebuild_epoch,
                )
            )

        if should_rebuild:
            self._solve_graph(
                model, train_dataset, device, config, seed, start_method, epoch,
            )
            self._last_graph_rebuild_epoch = None if epoch is None else int(epoch)
            self._weights_are_static = self.graph_source_is_static()
        elif self._weights_are_static:
            logger.info(
                "Reusing the ssmlpc eq. (5) weights solved on the input "
                f"representation: {self.graph_metadata['nodes']} nodes, "
                f"{self.graph_anchor_count} anchors (held fixed, as in the paper)"
            )
        else:
            logger.info(
                f"Reusing ssmlpc eq. (5) weights: {self.graph_metadata['nodes']} "
                f"nodes, {self.graph_anchor_count} anchors, k={self.n_neighbors}"
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
            self._regularizer_sampler = GraphNeighborhoodBatchSampler(
                neighbor_indices=self.neighbor_indices,
                neighbor_weights=self.neighbor_weights,
                graph_batch_size=graph_batch_size,
                seed=seed,
                num_batches=len(supervised_loader),
                valid_anchors=self.valid_anchors,
            )
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_sampler=self._regularizer_sampler,
                collate_fn=collate_graph_neighborhood_batch,
                **dataloader_kwargs,
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "ssmlpc batch streams: "
                f"supervised_examples_per_batch={int(batch_size)}, "
                f"graph_anchors_per_batch={graph_batch_size}, "
                f"max_graph_nodes_before_dedup="
                f"{graph_batch_size * (1 + self.n_neighbors)}, "
                f"reduction={self.reduction}, "
                f"steps={len(supervised_loader)}"
            )
        else:
            self._regularizer_sampler.set_graph(
                self.neighbor_indices,
                self.neighbor_weights,
                valid_anchors=self.valid_anchors,
            )

        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    # -- eq. (6) -----------------------------------------------------------

    def compute_loss(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
        **unused_context,
    ):
        if len(batch) != 5:
            raise ValueError(
                "ssmlpc expects (images, node_ids, anchor_rows, neighbor_columns, "
                f"neighbor_weights) from its loader, got {len(batch)} fields"
            )
        images, node_ids, anchor_rows, neighbor_columns, neighbor_weights = batch
        if self.graph_anchor_count is None:
            raise RuntimeError("make_loader must run before ssmlpc.compute_loss")

        embeddings = utils.forward_model_inputs(
            student_model,
            images,
            device,
            use_cache=self.use_cache,
        ).float()  # see reconstruction_energy: CPU autocast would sum in bfloat16
        if len(embeddings) != len(node_ids):
            raise RuntimeError(
                "ssmlpc forward returned one embedding per graph node; got "
                f"{len(embeddings)} for {len(node_ids)} nodes"
            )

        energy, residuals = reconstruction_energy(
            embeddings,
            anchor_rows,
            neighbor_columns,
            neighbor_weights,
            self.reduction,
            sampled_anchors=len(anchor_rows),
            graph_anchors=self.graph_anchor_count,
        )

        if self.collect_batch_diagnostics:
            with torch.no_grad():
                detached = embeddings.detach()
                self._last_diagnostics = {
                    "train/ssmlpc/reconstruction_energy": float(energy.detach()),
                    "train/ssmlpc/anchors": float(len(anchor_rows)),
                    "train/ssmlpc/graph_nodes_per_batch": float(len(node_ids)),
                    # The per-anchor residual length. Eq. (6)'s own scale, free
                    # of the reduction and of the calibrated weight.
                    "train/ssmlpc/residual_scale": float(
                        residuals.detach().norm(dim=1).mean()
                    ),
                    # A constant embedding is a global minimizer of eq. (6), so
                    # these two are the failure this term has to be watched for:
                    # a centroid norm climbing toward 1 while the residual
                    # shrinks is the collapse the paper's trace ratio and
                    # orthonormality constraint would have blocked.
                    "train/ssmlpc/centroid_norm": float(
                        detached.mean(dim=0).norm()
                    ),
                    "train/ssmlpc/embedding_norm": float(
                        detached.norm(dim=1).mean()
                    ),
                }
        return energy

    def batch_diagnostics(self):
        return dict(self._last_diagnostics)
