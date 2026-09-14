"""STML teacher relations, cut at a confidence threshold, as pseudo labels.

STML (Kim et al., CVPR 2022) reads the relation between two samples off a
teacher as ``w = (s + c) / 2``: a Gaussian kernel on teacher distances plus the
reciprocal-neighbor contextual similarity of Sec 3.2. Its own objective consumes
``w`` as a *soft pairwise weight* -- the relaxed contrastive loss pulls with
``w`` and pushes with ``1 - w`` -- which ties STML to that one loss function.

This variant keeps ``w`` and replaces the soft weighting with a confidence
threshold, the way Iscen et al. (CVPR 2019) accept only pseudo labels above a
certainty cutoff. ``w`` is already a confidence: it is the teacher's estimate
that two samples share a class. Relations above ``positive_threshold`` become
graph edges, single-linkage propagation carries true labels from the labeled
split along them, and every reached sample leaves with a real class id plus the
bottleneck weight of its path back to a labeled sample as its confidence.

For a two-stream M-per-class loader, STML also propagates a candidate pool down
to ``pseudo_label_rescue_confidence_floor``. The configured thresholds remain
the normal acceptance rule; the lower-scoring candidates are visible only to
the shared class-capacity rescue, which admits the strongest missing classes
when the normal cutoff cannot fill the pseudo-labeled stream.

The result is an ordinary pseudo-label method: predictions go through the shared
confidence filter and relabeled-dataset path, so any configured metric loss
trains on them, proxy and classification losses included.

Two properties of the propagation are deliberate:

* Confidence is the *bottleneck* of the strongest path to a label, not the
  strength of the last edge. A sample reached through a chain of mediocre
  relations therefore scores lower than a direct neighbor, and
  ``confidence_threshold`` prunes exactly those long chains.
* Two labeled components of different classes never merge. Transitive merging is
  the one failure mode a pairwise view does not have, and a single confident but
  wrong edge would otherwise weld two classes into one pseudo class.
"""

from contextlib import contextmanager

import numpy as np
import torch
from loguru import logger
from scipy import sparse
from torch.utils.data import DataLoader

import utils
from losses import metric_losses
from .algorithms import faiss_flat_ip_search, require_faiss
from .config import (
    IN_BATCH_GRAPH_MODES,
    PseudoLabelResult,
    UNLABELED_TARGET,
    should_rebuild_on_epoch,
)
from .data import CombinedTrainingLoader, UnlabeledSubset
from .embeddings import extract_embeddings
from .graph_diagnostics import make_graph_diagnostics_request, maybe_save_graph_diagnostics
from .interfaces import BaseSemiSupervisedMethod, BaseTrainingRegularizer
from .pseudo_labels import summarize_numeric_values


SAMPLE_WEIGHT_MODES = ("confidence", "uniform")


# The paper updates its teacher after every backward pass; the pool-wide method
# updates once per pseudo-label refresh, so each cadence needs its own decay.
PER_STEP_TEACHER_MOMENTUM = 0.999


IN_BATCH_ONLY_PARAMS = {
    "supervised_weight": 1.0,
    "regularizer_weight": 1.0,
}


# Parameters the merged mode never reads, mapped to why. Setting one is almost
# always a config carried over from 'in_batch' that no longer means what it says.
MERGED_INERT_PARAMS = {
    "supervised_weight": (
        "the single loss covers the labeled rows already, so there is no separate "
        "supervised term to weight"
    ),
    "n_neighbors": (
        "the unlabeled rows are drawn uniformly rather than in nearest-neighbor "
        "groups, so no neighbor count is used"
    ),
}


class EMATeacherWeights:
    """An EMA copy of a model's trainable weights, applied by swapping them in.

    STML keeps a teacher that trails the student and reads every relation off
    it. Holding a second full model is both wasteful and impossible here -- a
    cached backbone carries a thread lock that cannot be deep-copied -- so the
    teacher is kept as bare tensors and installed into the student's parameters
    for the duration of a teacher forward.

    Only trainable parameters are tracked: frozen weights are identical in
    student and teacher by definition, so a frozen backbone costs nothing beyond
    the projection head. Buffers stay the student's own.
    """

    def __init__(self, model):
        self.weights = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.updates = 0

    def __len__(self):
        return len(self.weights)

    @torch.no_grad()
    def update(self, model, momentum):
        """Interpolate the teacher toward the student's current weights."""

        student_weights = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        stale = sorted(set(self.weights) - set(student_weights))
        if stale:
            raise RuntimeError(
                f"EMA teacher no longer matches the student's trainable parameters: {stale[:5]}"
            )
        for name, teacher_parameter in self.weights.items():
            student_parameter = student_weights[name].detach().to(teacher_parameter.device)
            teacher_parameter.lerp_(student_parameter, 1.0 - float(momentum))
        self.updates += 1

    @contextmanager
    def applied(self, model):
        """Run a block with the teacher's weights installed in ``model``.

        Only ``.data`` is rebound, so an autograd graph the student already
        built keeps its own saved tensors and the swap cannot leak into a
        pending backward pass.
        """

        originals = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    teacher_parameter = self.weights.get(name)
                    if teacher_parameter is None:
                        continue
                    originals[name] = parameter.data
                    parameter.data = teacher_parameter
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in originals:
                        parameter.data = originals[name]


def stml_relation_graph(
    features,
    n_neighbors,
    sigma,
    context_topk,
    *,
    prefer_gpu=True,
    block_size=4096,
    return_diagnostics=False,
):
    """Build STML's teacher relation ``w = (s + c) / 2`` as a sparse kNN graph.

    This is the dataset-scale form of
    :func:`losses.metric_losses.stml_teacher_pair_weights`, which STML evaluates
    densely inside one batch. Restricting it to a kNN graph is what makes it
    affordable over a whole training pool; pairs outside the graph would carry a
    small Gaussian similarity and no contextual similarity at all, so they never
    clear a useful threshold anyway.

    ``context_topk`` is STML's neighborhood size for the contextual term
    (``num_views * num_neighbors`` there). It counts the sample itself, matching
    the dense implementation, whose top-k always ranks a sample first.
    """

    features = np.ascontiguousarray(features, dtype=np.float32).copy()
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("stml_threshold needs at least two feature rows")
    num_samples = len(features)
    n_neighbors = min(int(n_neighbors), num_samples - 1)
    context_neighbors = min(max(int(context_topk) - 1, 1), num_samples - 1)
    if n_neighbors <= 0:
        raise ValueError("stml_threshold n_neighbors must be positive")

    faiss = require_faiss("stml_threshold relations")
    faiss.normalize_L2(features)
    retrieved = min(max(n_neighbors, context_neighbors) + 1, num_samples)
    similarities, neighbors = faiss_flat_ip_search(
        database=features,
        queries=features,
        k=retrieved,
        purpose="stml_threshold relations",
        faiss_module=faiss,
        prefer_gpu=prefer_gpu,
    )
    neighbors, similarities = _drop_self_matches(neighbors, similarities)

    # s = exp(-d^2 / sigma) on unit-norm features, where d^2 = 2 - 2 cos.
    squared_distances = np.clip(2.0 - 2.0 * similarities[:, :n_neighbors], 0.0, None)
    pair_similarity = _symmetric_csr(
        neighbors[:, :n_neighbors],
        np.exp(-squared_distances / float(sigma)).astype(np.float64),
        num_samples,
    )
    contextual_similarity = _contextual_similarity_graph(
        neighbors[:, :context_neighbors],
        num_samples,
        block_size=block_size,
    )
    relations = ((pair_similarity + contextual_similarity) / 2.0).tocsr()
    relations.setdiag(0.0)
    relations.eliminate_zeros()
    if not return_diagnostics:
        return relations
    return relations, {
        "graph_kind": "stml_teacher_relations",
        "requested_n_neighbors": int(n_neighbors),
        "search_n_neighbors": int(n_neighbors),
        "context_topk": int(context_topk),
        "sigma": float(sigma),
        "neighbor_indices": neighbors[:, :n_neighbors],
        "neighbor_similarities": similarities[:, :n_neighbors],
    }


def _drop_self_matches(neighbors, similarities):
    """Remove each row's self match, which exact search always returns first."""

    num_samples, retrieved = neighbors.shape
    queries = np.arange(num_samples, dtype=np.int64).reshape(-1, 1)
    is_self = neighbors == queries
    # Only the first self hit is dropped; duplicate rows can repeat an index.
    first_self = is_self & (is_self.cumsum(axis=1) == 1)
    keep = ~first_self
    # Rows without a self match lose their weakest neighbor instead, keeping the
    # result rectangular.
    keep[~first_self.any(axis=1), -1] = False
    kept_per_row = retrieved - 1
    return (
        neighbors[keep].reshape(num_samples, kept_per_row),
        similarities[keep].reshape(num_samples, kept_per_row),
    )


def _symmetric_csr(neighbors, values, num_samples):
    """Build a symmetric sparse matrix from directed kNN entries."""

    rows = np.repeat(np.arange(num_samples, dtype=np.int64), neighbors.shape[1])
    directed = sparse.csr_matrix(
        (values.ravel(), (rows, neighbors.ravel().astype(np.int64))),
        shape=(num_samples, num_samples),
    )
    # s is symmetric by construction, so the two directions agree wherever both
    # were retrieved; maximum keeps the value where only one direction was.
    return directed.maximum(directed.T)


def _contextual_similarity_graph(neighbors, num_samples, block_size):
    """Sparse form of STML's contextual similarity with query expansion.

    Mirrors :func:`losses.metric_losses.stml_contextual_similarity`: reciprocal
    nearest neighbors form ``V``, the shared-neighbor overlap ``V V^T`` masked by
    ``V`` and normalized by each row's neighbor count is the contextual
    similarity, and each row is then replaced by the mean of its closest
    ``topk / 2`` rows before symmetrization.
    """

    context_topk = neighbors.shape[1] + 1
    # The neighborhood includes the sample itself, as the dense top-k does.
    neighbor_mask = _binary_csr(neighbors, num_samples, include_self=True)
    reciprocal = neighbor_mask.multiply(neighbor_mask.T).tocsr()
    reciprocal_counts = np.asarray(reciprocal.sum(axis=1)).ravel()
    np.clip(reciprocal_counts, 1.0, None, out=reciprocal_counts)

    shared_blocks = []
    for start in range(0, num_samples, max(int(block_size), 1)):
        block = reciprocal[start : start + max(int(block_size), 1)]
        shared_blocks.append(block.dot(reciprocal.T).multiply(block).tocsr())
    shared = sparse.vstack(shared_blocks, format="csr")
    contextual = sparse.diags(1.0 / reciprocal_counts) @ shared

    half_k = max(1, int(round(context_topk / 2)))
    expansion = _binary_csr(neighbors[:, : half_k - 1], num_samples, include_self=True)
    expansion = sparse.diags(1.0 / np.asarray(expansion.sum(axis=1)).ravel()) @ expansion
    contextual = (expansion @ contextual).tocsr()
    return ((contextual + contextual.T) / 2.0).tocsr()


def _binary_csr(neighbors, num_samples, include_self=False):
    rows = np.repeat(np.arange(num_samples, dtype=np.int64), neighbors.shape[1])
    columns = neighbors.ravel().astype(np.int64)
    if include_self:
        diagonal = np.arange(num_samples, dtype=np.int64)
        rows = np.concatenate([rows, diagonal])
        columns = np.concatenate([columns, diagonal])
    mask = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(num_samples, num_samples),
    )
    # Duplicate entries sum during construction; the mask must stay binary.
    mask.data[:] = 1.0
    return mask


def propagate_labels_along_relations(relations, labels, positive_threshold):
    """Carry labels along confident relations by single linkage.

    Edges are consumed strongest first, so the weight that first connects a
    sample to a labeled component is the bottleneck of its strongest path back
    to a real label, and that weight becomes the sample's confidence. Merging two
    labeled components of different classes is refused rather than resolved.

    ``labels`` holds class ids for labeled nodes and ``-1`` elsewhere. The
    returned vectors cover every node: seeds keep their own label at confidence
    1, unreached nodes stay ``-1`` at confidence 0.
    """

    num_nodes = relations.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != num_nodes:
        raise ValueError("labels must be aligned with the relation graph")

    upper = sparse.triu(relations, k=1).tocoo()
    confident = upper.data >= float(positive_threshold)
    anchors = upper.row[confident]
    partners = upper.col[confident]
    edge_weights = upper.data[confident]
    # Descending weight is what makes the first connecting edge the bottleneck.
    order = np.argsort(-edge_weights, kind="stable")

    pseudo_labels = labels.copy()
    confidences = (labels >= 0).astype(np.float32)
    parents = list(range(num_nodes))
    sizes = [1] * num_nodes
    component_labels = labels.copy()
    # Unlabeled members still waiting for a class, tracked per component root.
    pending = [[] if label >= 0 else [node] for node, label in enumerate(component_labels)]

    def find(node):
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != root:
            parents[node], node = root, parents[node]
        return root

    conflicts = 0
    for index in order:
        anchor_root, partner_root = find(int(anchors[index])), find(int(partners[index]))
        if anchor_root == partner_root:
            continue
        anchor_label = int(component_labels[anchor_root])
        partner_label = int(component_labels[partner_root])
        if anchor_label >= 0 and partner_label >= 0 and anchor_label != partner_label:
            conflicts += 1
            continue
        merged_label = anchor_label if anchor_label >= 0 else partner_label
        if sizes[anchor_root] < sizes[partner_root]:
            anchor_root, partner_root = partner_root, anchor_root
        parents[partner_root] = anchor_root
        sizes[anchor_root] += sizes[partner_root]
        component_labels[anchor_root] = merged_label
        members = pending[anchor_root] + pending[partner_root]
        pending[partner_root] = []
        if merged_label >= 0 and members:
            weight = float(edge_weights[index])
            for member in members:
                pseudo_labels[member] = merged_label
                confidences[member] = weight
            members = []
        pending[anchor_root] = members

    return (
        pseudo_labels,
        confidences,
        {
            "confident_edges": int(len(edge_weights)),
            "graph_edges": int(upper.nnz),
            "label_conflicts": int(conflicts),
        },
    )


class STMLThresholdPseudoLabeler(BaseSemiSupervisedMethod):
    """Pseudo-label from STML teacher relations cut at a confidence threshold."""

    DEFAULT_PARAMS = {
        "n_neighbors": 10,
        "sigma": 3.0,
        "context_topk": 10,
        "positive_threshold": 0.7,
        "sample_weight_mode": "confidence",
        "teacher_momentum": 0.9,
    }

    def __init__(self, name="stml_threshold"):
        self.name = name
        self.reset_state()

    def reset_state(self):
        """Discard the EMA teacher carried over from a previous run or fold."""

        self._teacher = None

    def validate_config(self, config, source=""):
        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        try:
            # Shared validation runs against the registry instance even when the
            # run will take the in-batch path, so the mode decides which params
            # are in scope.
            validate_stml_threshold_params(
                params,
                in_batch=config.graph_batch_mode in IN_BATCH_GRAPH_MODES,
                merged=config.graph_batch_mode == "in_batch_merged",
                explicit_params=config.method_params,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid {self.name} configuration{source}: {exc}") from exc

    def pseudo_label_filter_threshold(self, config):
        """Keep ``positive_threshold`` as STML's normal acceptance boundary.

        Pool-wide capacity rescue may ask propagation to expose candidates
        below the relation cutoff. Taking the maximum here ensures those
        candidates remain rejected unless the shared filter explicitly rescues
        their class.
        """

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        return max(
            float(config.confidence_threshold),
            float(params["positive_threshold"]),
        )

    @staticmethod
    def _candidate_propagation_threshold(config, positive_threshold):
        """Return the edge floor needed to expose two-stream rescue candidates."""

        if config.labeled_batch_size is None:
            return float(positive_threshold)
        return min(
            float(positive_threshold),
            float(config.pseudo_label_rescue_confidence_floor),
        )

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
            raise ValueError(f"{self.name} requires at least one labeled sample to propagate from")

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        validate_stml_threshold_params(params)
        logger.info(f"Running {self.name} with params: {params}")

        # The common graph-method ordering contract makes the unlabeled output
        # a direct suffix slice after propagation.
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        # STML reads relations off a teacher, never off the student being
        # trained, so the student's weights are swapped for the EMA teacher's
        # for the duration of this pass.
        with self.teacher_weights_applied(model, float(params["teacher_momentum"])):
            features = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=ssl_positions,
                device=device,
                batch_size=config.embedding_batch_size,
                num_workers=0,
                seed=config.seed if epoch is None else config.seed + epoch,
                start_method=start_method,
                desc=f"{self.name} teacher embeddings",
            )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )
        relations, graph_metadata = stml_relation_graph(
            features,
            n_neighbors=int(params["n_neighbors"]),
            sigma=float(params["sigma"]),
            context_topk=int(params["context_topk"]),
            return_diagnostics=True,
        )
        self._save_graph_diagnostics(
            config=config,
            log_dir=log_dir,
            epoch=epoch,
            features=features,
            relations=relations,
            ssl_positions=ssl_positions,
            labels=labels,
            targets=targets,
            graph_metadata=graph_metadata,
        )
        positive_threshold = float(params["positive_threshold"])
        propagation_threshold = self._candidate_propagation_threshold(
            config,
            positive_threshold,
        )
        if propagation_threshold < positive_threshold:
            logger.info(
                f"{self.name} extending candidate propagation below positive_threshold="
                f"{positive_threshold:.6g} to the two-stream rescue floor "
                f"{propagation_threshold:.6g}; lower-confidence paths remain rescue-only"
            )
        propagated_labels, confidences, propagation_info = propagate_labels_along_relations(
            relations,
            targets,
            propagation_threshold,
        )

        unlabeled_start = len(split.labeled_positions)
        unlabeled_labels = propagated_labels[unlabeled_start:]
        unlabeled_confidences = confidences[unlabeled_start:]
        reached = unlabeled_labels >= 0
        unreached_count = int((~reached).sum())
        if unreached_count > 0:
            logger.warning(
                f"{self.name} left {unreached_count} unlabeled candidates unconnected to any "
                f"labeled sample at candidate propagation floor={propagation_threshold:.6g}; "
                "they will not enter pseudo-label training"
            )
        if propagation_info["label_conflicts"] > 0:
            logger.info(
                f"{self.name} refused {propagation_info['label_conflicts']} merges between "
                "components carrying different labeled classes"
            )
        logger.info(
            f"{self.name} propagated {int(reached.sum())}/{len(reached)} unlabeled candidates over "
            f"{propagation_info['confident_edges']}/{propagation_info['graph_edges']} confident "
            f"relations at floor {propagation_threshold:.6g}; bottleneck confidence distribution: "
            f"{summarize_numeric_values(unlabeled_confidences[reached])}"
        )

        accepted = reached
        accepted_confidences = unlabeled_confidences[accepted].astype(np.float32)
        return PseudoLabelResult(
            positions=split.unlabeled_positions[accepted],
            mapped_labels=unlabeled_labels[accepted].astype(np.int64),
            confidences=accepted_confidences,
        )

    def prepare_pseudo_labels_for_training(self, pseudo_labels, config):
        """Apply uniform weights only after shared filtering and rescue."""

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.method_params)
        if params["sample_weight_mode"] != "uniform":
            return pseudo_labels
        logger.info(
            f"{self.name} sample_weight_mode='uniform': using equal weight for "
            f"{len(pseudo_labels.positions)} accepted pseudo labels"
        )
        return PseudoLabelResult(
            positions=pseudo_labels.positions,
            mapped_labels=pseudo_labels.mapped_labels,
            confidences=np.ones(len(pseudo_labels.positions), dtype=np.float32),
        )

    @contextmanager
    def teacher_weights_applied(self, model, teacher_momentum):
        """Run a block with the EMA teacher's weights installed in ``model``.

        There is no per-step hook on the pseudo-label path, so this teacher
        advances once per pseudo-label refresh instead of once per optimizer
        step; ``teacher_momentum`` is therefore a per-refresh decay. STML's
        0.999 per step over a ~40-step epoch is roughly 0.96 per epoch, which is
        the range the default sits in. ``teacher_momentum=0`` keeps the student
        as its own teacher, which is plain self-training on the current iterate.
        """

        if teacher_momentum <= 0.0:
            yield model
            return

        if self._teacher is None:
            # The first refresh happens after warmup, so the teacher starts from
            # a student that already fits the labeled split.
            self._teacher = EMATeacherWeights(model)
            logger.info(
                f"{self.name} initialized the EMA teacher from the student "
                f"({len(self._teacher)} trainable tensors, momentum={teacher_momentum} per refresh)"
            )
        else:
            self._teacher.update(model, teacher_momentum)
        with self._teacher.applied(model):
            yield model

    def _save_graph_diagnostics(
        self,
        config,
        log_dir,
        epoch,
        features,
        relations,
        ssl_positions,
        labels,
        targets,
        graph_metadata,
    ):
        request = make_graph_diagnostics_request(
            config=config,
            log_dir=log_dir,
            name=f"{self.name}_relations",
            epoch=epoch,
            title=f"{self.name} STML teacher relation graph",
        )
        if request is None:
            return
        maybe_save_graph_diagnostics(
            request=request,
            embeddings=features,
            adjacency=relations,
            positions=ssl_positions,
            labels=labels[ssl_positions],
            known_mask=targets != UNLABELED_TARGET,
            graph_metadata=graph_metadata,
        )


class STMLThresholdRegularizer(BaseTrainingRegularizer):
    """Per-step STML: relations read off an EMA teacher inside each batch.

    This is the paper's own loop with its objective swapped out. Nearest-neighbor
    batches are rebuilt every epoch, the teacher advances after every optimizer
    step, and ``w`` is computed densely over the batch by the same function
    :class:`STMLLoss` uses -- no kNN approximation, because a batch is small
    enough to compare exhaustively. Only the consumption differs: instead of
    weighting a relaxed contrastive loss by ``w``, relations above
    ``positive_threshold`` become edges, labels propagate along them from the
    labeled rows in the batch, and the configured metric loss trains on the
    result.
    """

    name = "stml_threshold"
    supports_frozen_feature_precompute = True
    uses_joint_forward = True
    requires_supervised_objective = True

    def __init__(self, config):
        params = dict(STMLThresholdPseudoLabeler.DEFAULT_PARAMS)
        params.update(config.method_params)
        self.merged = config.graph_batch_mode == "in_batch_merged"
        validate_stml_threshold_params(
            params,
            in_batch=True,
            merged=self.merged,
            explicit_params=config.method_params,
        )
        super().__init__(
            regularizer_weight=float(params.get("regularizer_weight", 1.0)),
            supervised_weight=float(params.get("supervised_weight", 1.0)),
        )
        self.sigma = float(params["sigma"])
        self.num_neighbors = int(params["n_neighbors"])
        self.context_topk = int(params["context_topk"])
        self.positive_threshold = float(params["positive_threshold"])
        self.sample_weight_mode = str(params["sample_weight_mode"])
        # A per-step teacher needs a much slower decay than a per-refresh one,
        # so the paper's value is the default here rather than the pool-wide one.
        self.teacher_momentum = float(
            config.method_params.get("teacher_momentum", PER_STEP_TEACHER_MOMENTUM)
        )
        if not 0 <= self.teacher_momentum < 1:
            raise ValueError("stml_threshold teacher_momentum must be in [0, 1)")
        # The merged mode puts the whole supervised batch in the graph, so its
        # labeled size is only known per step and stays None here.
        if self.merged:
            self.graph_labeled_batch_size = None
        elif config.graph_labeled_batch_size is None:
            raise ValueError(
                f"{self.name} graph_batch_mode='in_batch' needs a "
                "graph_labeled_batch_size; use graph_batch_mode='in_batch_merged' "
                "to put the whole supervised batch in the graph instead"
            )
        else:
            self.graph_labeled_batch_size = int(config.graph_labeled_batch_size)
        self.graph_unlabeled_batch_size = int(config.graph_unlabeled_batch_size)
        self.confidence_threshold = float(config.confidence_threshold)
        self.dataset = None
        self._labeled_positions = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_sampling_rebuild_epoch = None
        self._last_diagnostics = {}

    def configure_graph_batching(self, config):
        if config.graph_batch_mode not in IN_BATCH_GRAPH_MODES:
            raise ValueError(
                f"{self.name} regularization is the in-batch mode; "
                f"got graph_batch_mode={config.graph_batch_mode!r}"
            )

    def validate_run_args(self, args):
        if self.merged:
            # Neither rule applies: the graph takes the supervised batch whole,
            # and the unlabeled stream is drawn uniformly rather than in groups.
            return
        if self.graph_labeled_batch_size > int(args.batch_size):
            raise ValueError(
                f"{self.name} graph_labeled_batch_size cannot exceed the supervised batch_size "
                f"({self.graph_labeled_batch_size} > {args.batch_size})"
            )
        if self.graph_unlabeled_batch_size % self.num_neighbors != 0:
            raise ValueError(
                f"{self.name} requires graph_unlabeled_batch_size to be divisible by "
                f"method_params.n_neighbors ({self.graph_unlabeled_batch_size} % "
                f"{self.num_neighbors} != 0); the nearest-neighbor sampler fills each batch "
                "with whole neighbor groups"
            )

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        self._labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)

    def build_dataset(self, train_dataset, split, use_cache=False):
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) < 2:
            raise ValueError(f"{self.name} in-batch mode needs at least two unlabeled samples")
        regularizer_dataset = self.make_regularizer_source_dataset(train_dataset, use_cache=use_cache)
        # One view per sample: the joint labeled/unlabeled forward concatenates
        # image tensors, so a two-view batch cannot travel that path. The
        # nearest-neighbor sampler still puts related samples in every batch.
        self.dataset = UnlabeledSubset(regularizer_dataset, unlabeled_positions, num_views=1)
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_sampling_rebuild_epoch = None
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
        if self.dataset is None:
            raise RuntimeError(f"{self.name} build_dataset must run before make_loader")
        unlabeled_batch_size = min(self.graph_unlabeled_batch_size, len(self.dataset))
        effective_num_workers = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        if self.merged:
            return self._make_uniform_loader(
                supervised_loader=supervised_loader,
                unlabeled_batch_size=unlabeled_batch_size,
                num_workers=effective_num_workers,
                seed=seed,
                start_method=start_method,
            )
        cache_key = (
            id(self.dataset),
            int(unlabeled_batch_size),
            int(self.num_neighbors),
            int(effective_num_workers),
            str(start_method),
        )
        should_rebuild = (
            self._regularizer_loader is None
            or self._regularizer_loader_cache_key != cache_key
            or should_rebuild_on_epoch(
                config.update_mode,
                config.update_interval_epochs,
                epoch,
                self._last_sampling_rebuild_epoch,
            )
        )
        if should_rebuild:
            utils.shutdown_dataloaders(self._regularizer_loader)
            # The paper rebuilds its NNBatchSampler every epoch from the current
            # embeddings; batches of unrelated samples would hold no relation
            # confident enough to threshold.
            sampling_embeddings = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.dataset.positions,
                device=self.get_ssl_device(device),
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"{self.name} sampling embeddings - epoch {epoch}",
            )
            self._regularizer_loader = utils.make_stml_train_loader(
                train_dataset=self.dataset,
                sampling_embeddings=sampling_embeddings,
                batch_size=unlabeled_batch_size,
                neighbors_per_query=self.num_neighbors,
                seed=seed,
                num_workers=effective_num_workers,
                start_method=start_method,
                graph_device=self.get_ssl_device(device),
            )
            self._regularizer_loader_cache_key = cache_key
            self._last_sampling_rebuild_epoch = None if epoch is None else int(epoch)
            logger.info(
                f"{self.name} nearest-neighbor loader: pool={len(self.dataset)}, "
                f"batch_size={unlabeled_batch_size}, neighbors/query={self.num_neighbors}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def _make_uniform_loader(
        self,
        supervised_loader,
        unlabeled_batch_size,
        num_workers,
        seed,
        start_method,
    ):
        """Draw the unlabeled rows uniformly instead of in neighbor groups.

        The paper's ``NNBatchSampler`` exists to make batches dense in mutual
        neighbors, which its relation-matching objective needs. This mode
        propagates outward from labeled seeds instead, and the labeled stream is
        drawn by an independent M-per-class sampler, so grouping the unlabeled
        rows among themselves buys nothing here -- and it costs an embedding pass
        over the whole unlabeled pool every epoch.
        """

        if unlabeled_batch_size < 2:
            raise ValueError(f"{self.name} needs at least two unlabeled samples per batch")
        cache_key = (
            id(self.dataset),
            int(unlabeled_batch_size),
            int(num_workers),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_loader = utils.make_unlabeled_stream_loader(
                self.dataset,
                batch_size=unlabeled_batch_size,
                seed=seed,
                num_workers=num_workers,
                start_method=start_method,
                supervised_loader=supervised_loader,
                # A ragged final batch would shrink the graph without saying so.
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
                desc="stml uniform",
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                f"{self.name} uniform unlabeled loader: pool={len(self.dataset)}, "
                f"batch_size={unlabeled_batch_size}, steps={len(supervised_loader)}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def initialize_state(self, student_model, device):
        # Warmup has already run, so the teacher starts from a student that fits
        # the labeled split, exactly as the paper's teacher does.
        teacher = EMATeacherWeights(student_model)
        logger.info(
            f"{self.name} in-batch mode: sigma={self.sigma}, context_topk={self.context_topk}, "
            f"positive_threshold={self.positive_threshold}; EMA teacher seeded from the student "
            f"({len(teacher)} trainable tensors, momentum={self.teacher_momentum} per step)"
        )
        return teacher

    def after_optimizer_step(self, student_model, state):
        if state is None:
            raise RuntimeError(f"{self.name} requires an initialized EMA teacher")
        if self.teacher_momentum <= 0.0:
            return
        # The paper's Momentum_Update, run at the same point in the loop.
        state.update(student_model, self.teacher_momentum)

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
        supervised_inputs=None,
        supervised_indices=None,
        supervised_criterion=None,
        supervised_miner=None,
        supervised_is_classification=False,
        **unused_context,
    ):
        if state is None:
            raise RuntimeError(f"{self.name} requires an initialized EMA teacher")
        if supervised_embeddings is None or supervised_labels is None or regularizer_embeddings is None:
            raise ValueError(f"{self.name} requires the joint labeled/unlabeled forward context")
        if supervised_criterion is None:
            raise ValueError(f"{self.name} requires the configured supervised objective")
        if supervised_inputs is None or batch is None:
            raise ValueError(f"{self.name} needs both input streams to embed with the teacher")

        labeled_rows = self._select_labeled_rows(supervised_embeddings)
        labeled_count = len(labeled_rows)
        student_graph_embeddings = torch.cat(
            [supervised_embeddings[labeled_rows], regularizer_embeddings],
            dim=0,
        )
        graph_labels = np.concatenate(
            [
                supervised_labels[labeled_rows].detach().cpu().numpy().astype(np.int64),
                np.full(len(regularizer_embeddings), UNLABELED_TARGET, dtype=np.int64),
            ]
        )
        relations = self._teacher_relations(
            student_model,
            state,
            # Raw inputs arrive on the loader's device, which is not the
            # embedding device until the model's forward moves them.
            supervised_inputs[labeled_rows.to(supervised_inputs.device)],
            batch[0],
            device,
        )
        pseudo_labels, confidences, propagation_info = propagate_labels_along_relations(
            dense_relation_graph(relations, self.positive_threshold),
            graph_labels,
            self.positive_threshold,
        )

        accepted = np.zeros(len(graph_labels), dtype=bool)
        accepted[labeled_count:] = (pseudo_labels[labeled_count:] >= 0) & (
            confidences[labeled_count:] >= self.confidence_threshold
        )
        self._record_diagnostics(relations, accepted, confidences, propagation_info, labeled_count)
        if not accepted.any():
            if not self.merged:
                return student_graph_embeddings.sum() * 0.0
            # This term is the whole objective here, so returning zero would skip
            # the step entirely -- and early in training, when the teacher accepts
            # almost nothing, that would be most of an epoch. Falling back to the
            # labeled rows makes an unproductive graph cost nothing rather than
            # stalling training.
            return self._apply_supervised_criterion(
                supervised_criterion=supervised_criterion,
                supervised_miner=supervised_miner,
                supervised_is_classification=supervised_is_classification,
                embeddings=supervised_embeddings,
                labels=supervised_labels,
                sample_weights=None,
            )

        accepted_tensor = torch.as_tensor(
            accepted,
            dtype=torch.bool,
            device=student_graph_embeddings.device,
        )
        pseudo_label_tensor = torch.as_tensor(
            pseudo_labels,
            dtype=supervised_labels.dtype,
            device=supervised_labels.device,
        )
        # Labeled rows stay in the batch as anchors, so a pair loss sees real
        # positives for every propagated class.
        loss_embeddings = torch.cat(
            [student_graph_embeddings[:labeled_count], student_graph_embeddings[accepted_tensor]],
            dim=0,
        )
        loss_labels = torch.cat(
            [
                torch.as_tensor(
                    graph_labels[:labeled_count],
                    dtype=supervised_labels.dtype,
                    device=supervised_labels.device,
                ),
                pseudo_label_tensor[accepted_tensor],
            ],
            dim=0,
        )
        loss = self._apply_supervised_criterion(
            supervised_criterion=supervised_criterion,
            supervised_miner=supervised_miner,
            supervised_is_classification=supervised_is_classification,
            embeddings=loss_embeddings,
            labels=loss_labels,
            sample_weights=self._sample_weights(
                supervised_criterion,
                confidences,
                accepted,
                labeled_count,
                loss_embeddings,
            ),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{self.name} produced a non-finite loss")
        return loss

    def _select_labeled_rows(self, supervised_embeddings):
        if self.merged:
            # The whole supervised batch is in the graph, so every labeled row is
            # already an anchor and there is nothing to subsample.
            return torch.arange(
                len(supervised_embeddings),
                dtype=torch.long,
                device=supervised_embeddings.device,
            )
        requested = min(self.graph_labeled_batch_size, len(supervised_embeddings))
        if requested <= 0:
            raise ValueError(f"{self.name} in-batch graphs need labeled samples")
        if requested == len(supervised_embeddings):
            return torch.arange(requested, dtype=torch.long, device=supervised_embeddings.device)
        return torch.randperm(len(supervised_embeddings), device=supervised_embeddings.device)[
            :requested
        ]

    @torch.no_grad()
    def _teacher_relations(self, student_model, teacher, labeled_inputs, unlabeled_inputs, device):
        """Embed the batch with the teacher and return STML's dense ``w``."""

        inputs = torch.cat(
            [
                labeled_inputs.to(device, non_blocking=True),
                unlabeled_inputs.to(device, non_blocking=True),
            ],
            dim=0,
        )
        with teacher.applied(student_model):
            teacher_embeddings = utils.forward_model_inputs(
                student_model,
                inputs,
                device,
                use_cache=self.use_cache,
            )
        # Single-view batches carry no repeated instances to tie together.
        return metric_losses.stml_teacher_pair_weights(
            teacher_embeddings,
            self.sigma,
            self.context_topk,
            instance_ids=None,
        )

    def _sample_weights(
        self,
        supervised_criterion,
        confidences,
        accepted,
        labeled_count,
        loss_embeddings,
    ):
        if self.sample_weight_mode != "confidence":
            return None
        if not getattr(supervised_criterion, "supports_sample_weights", False):
            return None
        accepted_confidences = torch.as_tensor(
            confidences[accepted],
            dtype=torch.float32,
            device=loss_embeddings.device,
        )
        labeled_weights = torch.ones(
            labeled_count,
            dtype=torch.float32,
            device=loss_embeddings.device,
        )
        return torch.cat([labeled_weights, accepted_confidences.clamp(0.0, 1.0)], dim=0)

    @staticmethod
    def _apply_supervised_criterion(
        supervised_criterion,
        supervised_miner,
        supervised_is_classification,
        embeddings,
        labels,
        sample_weights,
    ):
        if supervised_is_classification:
            return metric_losses.classification_loss_float32(
                supervised_criterion,
                embeddings,
                labels,
                sample_weights=sample_weights,
            )
        if sample_weights is not None:
            return supervised_criterion(embeddings, labels, sample_weights=sample_weights)
        if supervised_miner is not None:
            return supervised_criterion(embeddings, labels, supervised_miner(embeddings, labels))
        return supervised_criterion(embeddings, labels)

    def _record_diagnostics(self, relations, accepted, confidences, propagation_info, labeled_count):
        if not self.collect_batch_diagnostics:
            return
        unlabeled_count = max(len(accepted) - labeled_count, 1)
        accepted_count = int(accepted.sum())
        self._last_diagnostics = {
            "train/stml_threshold/nodes": float(len(accepted)),
            "train/stml_threshold/labeled_nodes": float(labeled_count),
            "train/stml_threshold/accepted": float(accepted_count),
            "train/stml_threshold/accepted_fraction": float(accepted_count / unlabeled_count),
            "train/stml_threshold/confident_relations": float(propagation_info["confident_edges"]),
            "train/stml_threshold/label_conflicts": float(propagation_info["label_conflicts"]),
            "train/stml_threshold/mean_relation": float(relations.mean()),
            "train/stml_threshold/mean_confidence": float(
                confidences[accepted].mean() if accepted_count else 0.0
            ),
        }

    def combine_losses(self, supervised_loss, regularization_loss):
        if not self.merged:
            return super().combine_losses(supervised_loss, regularization_loss)
        # The regularizer term already ran the configured metric loss over every
        # labeled row with its true label, so adding the engine's supervised loss
        # on top would count the labeled pairs twice -- with a ratio that drifts
        # as the accepted count changes from step to step.
        return self.regularizer_weight * regularization_loss

    def batch_diagnostics(self):
        return dict(self._last_diagnostics)


class STMLThresholdInBatchMethod(BaseSemiSupervisedMethod):
    """Route ``graph_batch_mode='in_batch'`` to the paper's per-step training loop."""

    name = "stml_threshold"
    generates_pseudo_labels = False
    is_regularization_method = True

    def __init__(self, method):
        self.method = method

    def validate_config(self, config, source=""):
        try:
            STMLThresholdRegularizer(config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {self.name} in-batch configuration{source}: {exc}") from exc

    def make_regularizer(self, config):
        regularizer = STMLThresholdRegularizer(config)
        regularizer.configure_graph_batching(config)
        return regularizer


def dense_relation_graph(relations, positive_threshold):
    """Return the confident half of a dense relation matrix as a sparse graph."""

    weights = relations.detach().float().cpu().numpy()
    np.fill_diagonal(weights, 0.0)
    weights[weights < float(positive_threshold)] = 0.0
    return sparse.csr_matrix(weights)


def validate_stml_threshold_params(params, in_batch=False, merged=False, explicit_params=None):
    allowed = set(STMLThresholdPseudoLabeler.DEFAULT_PARAMS)
    if in_batch:
        allowed |= set(IN_BATCH_ONLY_PARAMS)
    if merged:
        # Silently ignoring these would leave a config that reads as if it still
        # weighted two terms or still sampled neighbor groups.
        configured = set(explicit_params or {})
        for name, reason in MERGED_INERT_PARAMS.items():
            if name in configured:
                raise ValueError(
                    f"stml_threshold {name} has no effect under "
                    f"graph_batch_mode='in_batch_merged': {reason}"
                )
    unknown = sorted(set(params) - allowed)
    if unknown:
        in_batch_only = sorted(set(unknown) & set(IN_BATCH_ONLY_PARAMS))
        if in_batch_only:
            raise ValueError(
                f"stml_threshold params {in_batch_only} apply only to "
                "graph_batch_mode='in_batch'"
            )
        raise ValueError(f"Unknown stml_threshold params: {unknown}")
    for name in ("supervised_weight", "regularizer_weight"):
        if name in params and (not np.isfinite(float(params[name])) or float(params[name]) < 0):
            raise ValueError(f"stml_threshold {name} must be finite and non-negative")
    for name in ("n_neighbors", "context_topk"):
        if int(params[name]) <= 0:
            raise ValueError(f"stml_threshold {name} must be positive")
    sigma = float(params["sigma"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("stml_threshold sigma must be positive")
    positive_threshold = float(params["positive_threshold"])
    if not np.isfinite(positive_threshold) or not 0 < positive_threshold <= 1:
        raise ValueError("stml_threshold positive_threshold must be in (0, 1]")
    teacher_momentum = float(params["teacher_momentum"])
    if not np.isfinite(teacher_momentum) or not 0 <= teacher_momentum < 1:
        raise ValueError("stml_threshold teacher_momentum must be in [0, 1)")
    if params["sample_weight_mode"] not in SAMPLE_WEIGHT_MODES:
        raise ValueError(f"stml_threshold sample_weight_mode must be one of {list(SAMPLE_WEIGHT_MODES)}")
