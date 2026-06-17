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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from scipy import stats
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

import numpy as np
import torch
from loguru import logger

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from tqdm import tqdm

import utils
import metric_losses


UNLABELED_TARGET = -1
UPDATE_MODES = {"once", "every_epoch"}
LABEL_SAMPLING_MODES = {"per_class_min", "global_budget", "class_subset", "class_subset_k_shot"}
LOSS_DRIVEN_METHODS = {"stml"}


@dataclass(frozen=True)
class SemiSupervisedConfig:
    """All settings needed to select labels and generate pseudo-labels."""

    method: str = "none"
    update_mode: str = "once"
    warmup_epochs: int = 0
    label_sampling_mode: str = "global_budget"
    labeled_fraction: float = 1.0
    labeled_per_class: int | None = None
    seed: int | None = None
    confidence_threshold: float = 0.0
    max_unlabeled_samples: int | None = None
    embedding_batch_size: int = 32
    embedding_num_workers: int = 8
    method_params: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self):
        return self.method != "none"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SemiSupervisedSplit:
    """Positions assigned to the known-label and pseudo-label candidate pools."""

    labeled_positions: np.ndarray
    unlabeled_positions: np.ndarray


@dataclass(frozen=True)
class PseudoLabelResult:
    """Pseudo-label predictions aligned with their training-subset positions."""

    positions: np.ndarray
    mapped_labels: np.ndarray
    confidences: np.ndarray | None = None


class PseudoLabelDiagnosticsTracker:
    """Persist pseudo-label quality, distribution, and stability diagnostics."""

    def __init__(self, log_dir):
        self.path = Path(log_dir) / "pseudo_label_diagnostics.jsonl"
        self.previous_raw = None
        self.previous_accepted = None
        self.generation_index = 0

    def log(self, raw_pseudo_labels, accepted_pseudo_labels, train_dataset, config, epoch=None):
        true_labels = np.asarray(train_dataset.labels, dtype=np.int64)
        raw_summary = summarize_pseudo_label_result(raw_pseudo_labels, true_labels)
        accepted_summary = summarize_pseudo_label_result(accepted_pseudo_labels, true_labels)
        raw_changes = summarize_pseudo_label_changes(self.previous_raw, raw_pseudo_labels)
        accepted_changes = summarize_pseudo_label_changes(self.previous_accepted, accepted_pseudo_labels)

        record = {
            "generation_index": self.generation_index,
            "epoch": epoch,
            "method": config.method,
            "confidence_threshold": config.confidence_threshold,
            "raw": raw_summary,
            "accepted": accepted_summary,
            "raw_changes_from_previous_generation": raw_changes,
            "accepted_changes_from_previous_generation": accepted_changes,
            "audit_note": "Hidden labels are used only for post-prediction diagnostics, never for pseudo-label generation.",
        }
        with self.path.open("a") as jsonl_file:
            jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")

        logger.info(
            "Pseudo-label diagnostics: "
            f"raw={raw_summary['count']}, accepted={accepted_summary['count']}, "
            f"accepted_audit_accuracy={format_optional_metric(accepted_summary['audit_accuracy'])}, "
            f"accepted_confidence_mean={format_optional_metric(accepted_summary['confidence']['mean'])}, "
            f"accepted_changes={format_change_summary(accepted_changes)}"
        )
        self.previous_raw = pseudo_labels_to_position_map(raw_pseudo_labels)
        self.previous_accepted = pseudo_labels_to_position_map(accepted_pseudo_labels)
        self.generation_index += 1


class RelabeledSubset(Dataset):
    """Dataset view containing true-labeled and accepted pseudo-labeled samples.

    ``orig_labels`` are returned by ``__getitem__`` because the main training
    loop applies the shared original-to-mapped label dictionary.  ``labels``
    stores the dense mapped labels needed by MPerClassSampler. The third item
    returned by ``__getitem__`` is an optional confidence consumed only by
    confidence-aware losses.
    """

    def __init__(self, dataset, positions, orig_labels, mapped_labels, confidences=None):
        if confidences is None:
            confidences = np.ones(len(positions), dtype=np.float32)
        if not (len(positions) == len(orig_labels) == len(mapped_labels) == len(confidences)):
            raise ValueError("positions, orig_labels, mapped_labels, and confidences must have the same length")
        self.dataset = dataset
        # positions chooses which samples are visible through this view. The two
        # label arrays stay aligned with that exact order.
        self.positions = np.asarray(positions, dtype=np.int64)
        self.orig_labels = [int(label) for label in orig_labels]
        # MPerClassSampler reads this attribute directly without calling
        # __getitem__, so it receives dense true/pseudo labels here.
        self.labels = [int(label) for label in mapped_labels]
        self.confidences = np.asarray(confidences, dtype=np.float32)
        if not np.all(np.isfinite(self.confidences)):
            raise ValueError("confidences must be finite")
        if np.any((self.confidences < 0) | (self.confidences > 1)):
            raise ValueError("confidences must be in [0, 1]")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        # Ignore the label returned by the wrapped dataset because pseudo-labeled
        # samples must expose their predicted label instead of the hidden truth.
        image, _ = self.dataset[int(self.positions[index])]
        return image, self.orig_labels[index], self.confidences[index]


class UnlabeledSubset(Dataset):
    """Dataset view that intentionally hides all labels from loss-driven SSL."""

    def __init__(self, dataset, positions, num_views=1):
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)
        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = int(self.positions[index])
        views = [self.dataset[position][0] for _ in range(self.num_views)]
        image_or_views = views[0] if self.num_views == 1 else views
        return image_or_views, UNLABELED_TARGET, position

class LRMLGraphDataset(Dataset):
    """Expose graph nodes by their graph index for LRML Laplacian regularization.

    Item ``i`` corresponds to graph node ``i`` so the indices produced by
    ``LRMLGraphBatchSampler`` line up with the rows of the precomputed neighbor
    graph and the symmetric adjacency. The training transform is applied (the
    regularizer sees the same augmented views as the supervised loss); only the
    node index is returned because labels stay hidden from the regularizer.
    """

    def __init__(self, dataset, positions):
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        image, _ = self.dataset[int(self.positions[index])]
        return image, index

class BaseSemiSupervisedMethod:
    """Interface implemented by each pseudo-label generation strategy."""

    name = None
    generates_pseudo_labels = True
    is_regularization_method = False

    def validate_config(self, config, source=""):
        return None

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
        raise NotImplementedError


class CombinedTrainingLoader:
    """Pair every regularizer batch with a labeled supervised batch."""

    def __init__(self, supervised_loader, regularizer_loader):
        if len(supervised_loader) == 0:
            raise ValueError("regularized training requires at least one supervised batch")
        self.supervised_loader = supervised_loader
        self.regularizer_loader = regularizer_loader

    def __len__(self):
        return len(self.regularizer_loader)

    def __iter__(self):
        supervised_iterator = iter(self.supervised_loader)
        for regularizer_batch in self.regularizer_loader:
            try:
                supervised_batch = next(supervised_iterator)
            except StopIteration:
                supervised_iterator = iter(self.supervised_loader)
                supervised_batch = next(supervised_iterator)
            yield supervised_batch, regularizer_batch


class LRMLGraphBatchSampler(torch.utils.data.Sampler):
    """Build batches that keep each sampled graph node close to its neighbors."""

    def __init__(self, neighbor_indices, batch_size, seed):
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        if neighbor_indices.ndim != 2:
            raise ValueError("LRML neighbor_indices must be a matrix")
        if len(neighbor_indices) < 2:
            raise ValueError("LRML regularization requires at least two graph samples")
        if batch_size < 2:
            raise ValueError("LRML batch_size must be at least 2")
        if neighbor_indices.shape[1] == 0:
            raise ValueError("LRML graph must contain at least one neighbor per node")

        self.neighbor_indices = torch.as_tensor(neighbor_indices, dtype=torch.long)
        self.num_samples = int(len(neighbor_indices))
        self.batch_size = int(batch_size)
        self.neighbors_per_query = min(self.batch_size - 1, int(neighbor_indices.shape[1]))
        self.queries_per_batch = max(1, self.batch_size // (self.neighbors_per_query + 1))
        self.generator = utils.make_torch_generator(seed)

    def __iter__(self):
        query_order = torch.randperm(self.num_samples, generator=self.generator)
        for start in range(0, self.num_samples, self.queries_per_batch):
            batch_indices = []
            seen = set()
            query_indices = query_order[start : start + self.queries_per_batch]
            for query_index in query_indices.tolist():
                candidates = [query_index]
                candidates.extend(self.neighbor_indices[query_index, : self.neighbors_per_query].tolist())
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    batch_indices.append(candidate)
            yield batch_indices

    def __len__(self):
        return int(math.ceil(self.num_samples / self.queries_per_batch))


class BaseTrainingRegularizer:
    """Pluggable unlabeled regularization term combined with a supervised loss."""

    name = None
    provides_trainable_projection_without_feat_dim = False

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0):
        self.regularizer_weight = float(regularizer_weight)
        self.supervised_weight = float(supervised_weight)
        if not math.isfinite(self.regularizer_weight) or self.regularizer_weight < 0:
            raise ValueError("regularizer_weight must be finite and non-negative")
        if not math.isfinite(self.supervised_weight) or self.supervised_weight < 0:
            raise ValueError("supervised_weight must be finite and non-negative")
        if self.regularizer_weight == 0 and self.supervised_weight == 0:
            raise ValueError("regularizer_weight and supervised_weight cannot both be zero")

    def model_kwargs(self, args):
        return {}

    def validate_run_args(self, args):
        return None

    def build_dataset(self, train_dataset, split):
        raise NotImplementedError

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
    ):
        raise NotImplementedError

    def initialize_state(self, student_model, device):
        return None

    def combine_losses(self, supervised_loss, regularization_loss):
        return (
            self.supervised_weight * supervised_loss
            + self.regularizer_weight * regularization_loss
        )

    def compute_loss(self, student_model, state, batch, device):
        raise NotImplementedError

    def after_optimizer_step(self, student_model, state):
        return None

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

    Because the term couples neighbors, ``LRMLGraphBatchSampler`` keeps every node
    together with its neighbors in a batch, and each step evaluates the Laplacian
    energy of the sub-graph induced on that batch. The symmetric-normalized
    Laplacian is obtained by scaling each node's embedding by 1/sqrt(deg) (a global
    constant) and computing the unnormalized energy on the scaled embeddings.
    """

    name = "lrml"

    DEFAULT_PARAMS = {
        "n_neighbors": 6,             # paper uses 6 nearest neighbors
        "normalized_laplacian": True, # paper adopts the normalized Laplacian in practice
        "normalize_embeddings": True, # DML standard; keeps the energy bounded
        "graph_on": "all",            # "all" (labeled + unlabeled) or "unlabeled"
        "reduction": "mean",          # "mean" over intra-batch edges, or "sum"
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
        self.normalize_embeddings = bool(merged["normalize_embeddings"])
        self.graph_on = str(merged["graph_on"])
        if self.graph_on not in self.GRAPH_ON_CHOICES:
            raise ValueError(f"lrml graph_on must be one of {sorted(self.GRAPH_ON_CHOICES)}")
        self.reduction = str(merged["reduction"])
        if self.reduction not in self.REDUCTION_CHOICES:
            raise ValueError(f"lrml reduction must be one of {sorted(self.REDUCTION_CHOICES)}")

        self.dataset = None
        self.graph_positions = None
        self.adjacency = None   # scipy CSR, symmetric binary W
        self.node_scale = None  # torch tensor, 1/sqrt(deg) or ones

    def validate_run_args(self, args):
        if args.batch_size < 2:
            raise ValueError("LRML regularization requires batch_size >= 2")
        if getattr(args, "use_cache", False):
            raise ValueError(
                "LRML regularization augments graph nodes per step and cannot use "
                "backbone caching; set use_cache=False"
            )

    def build_dataset(self, train_dataset, split):
        if self.graph_on == "all":
            positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        else:
            positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        positions = np.unique(np.asarray(positions, dtype=np.int64))  # sorted + deterministic
        if len(positions) < 2:
            raise ValueError("LRML regularization requires at least two graph samples")
        self.graph_positions = positions
        self.dataset = LRMLGraphDataset(train_dataset, positions)
        return self.dataset

    def make_loader(
        self, model, train_dataset, supervised_loader, device, config,
        batch_size, seed, num_workers, start_method, epoch,
    ):
        if self.dataset is None or self.graph_positions is None:
            raise RuntimeError("build_dataset must be called before make_loader")

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
        )

        neighbor_indices, adjacency, degrees = build_lrml_knn_graph(
            embeddings, n_neighbors=self.n_neighbors, normalize=self.normalize_embeddings,
        )
        self.adjacency = adjacency
        scale = (
            1.0 / np.sqrt(np.maximum(degrees, 1.0))
            if self.normalized_laplacian
            else np.ones(len(degrees), dtype=np.float64)
        )
        self.node_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

        sampler = LRMLGraphBatchSampler(
            neighbor_indices=neighbor_indices, batch_size=batch_size, seed=seed,
        )
        regularizer_loader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            **utils.make_dataloader_kwargs(num_workers, seed, start_method),
        )
        logger.info(
            "Built LRML graph: "
            f"{adjacency.shape[0]} nodes, {adjacency.nnz // 2} undirected edges, "
            f"mean_degree={float(degrees.mean()):.2f}, "
            f"normalized_laplacian={self.normalized_laplacian}"
        )
        return CombinedTrainingLoader(supervised_loader, regularizer_loader)

    def compute_loss(self, student_model, state, batch, device):
        images, node_ids = batch
        embeddings = student_model(images.to(device))
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        # Fold the per-node 1/sqrt(deg) factor in: normalized-Laplacian energy equals
        # the unnormalized energy on degree-scaled embeddings.
        scaled = embeddings * self.node_scale[node_ids.to(device)][:, None]

        rows, cols, weights = induced_subgraph_edges(self.adjacency, node_ids.numpy())
        if len(rows) == 0:
            return embeddings.sum() * 0.0  # connected zero so backward stays valid

        row_index = torch.as_tensor(rows, dtype=torch.long, device=device)
        col_index = torch.as_tensor(cols, dtype=torch.long, device=device)
        edge_weight = torch.as_tensor(weights, dtype=embeddings.dtype, device=device)
        differences = scaled[row_index] - scaled[col_index]
        squared_distances = (differences * differences).sum(dim=1)
        energy = (edge_weight * squared_distances).sum()
        return energy / float(len(rows)) if self.reduction == "mean" else energy

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

    def build_dataset(self, train_dataset, split):
        if len(split.unlabeled_positions) < 2:
            raise ValueError("STML regularization requires at least two unlabeled samples")
        self.dataset = UnlabeledSubset(
            train_dataset,
            split.unlabeled_positions,
            num_views=self.num_views,
        )
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
    ):
        if self.dataset is None:
            raise RuntimeError("build_dataset must be called before make_loader")
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
        regularizer_loader = utils.make_stml_train_loader(
            train_dataset=self.dataset,
            sampling_embeddings=sampling_embeddings,
            batch_size=batch_size,
            neighbors_per_query=self.num_neighbors,
            seed=seed,
            num_workers=num_workers,
            start_method=start_method,
        )
        return CombinedTrainingLoader(supervised_loader, regularizer_loader)

    def initialize_state(self, student_model, device):
        teacher_model = copy.deepcopy(student_model)
        torch.nn.init.orthogonal_(teacher_model.embedding_g.weight)
        torch.nn.init.zeros_(teacher_model.embedding_g.bias)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        logger.info("Initialized STML EMA teacher from the supervised student")
        return teacher_model.to(device)

    def compute_loss(self, student_model, teacher_model, batch, device):
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

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
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

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
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

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(f"{self.name} requires the faiss-cpu package") from exc

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
        index = faiss.IndexFlatIP(labeled_embeddings.shape[1])
        index.add(labeled_embeddings)
        # neighbor_indices has shape [num_unlabeled, k] and contains row offsets
        # into labeled_embeddings/labeled_targets.
        similarities, neighbor_indices = index.search(unlabeled_embeddings, k)

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
    }

    def __init__(self, name="mixed_label_propagation"):
        self.name = name

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
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
        probabilities, confidences = mixed_label_propagation(
            features=features,
            targets=targets,
            num_classes=num_classes,
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


def mixed_label_propagation(
    features,
    targets,
    num_classes,
    n_neighbors=50,
    gamma=3.0,
    temperature=4.0,
    beta=1.0,
    mu=1.0 / 99.0,
    cg_rtol=1e-5,
    cg_max_iter=1000,
    edge_batch_size=65536,
):
    """Run equations (14)-(24) and return mixed-LP probabilities/confidences."""

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be a matrix aligned with targets")
    if len(features) < 2:
        raise ValueError("mixed_label_propagation requires at least two samples")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    labeled = targets != UNLABELED_TARGET
    if not np.any(labeled):
        raise ValueError("mixed_label_propagation requires at least one labeled target")
    if np.any((targets[labeled] < 0) | (targets[labeled] >= num_classes)):
        raise ValueError("labeled targets must be in [0, num_classes)")

    affinity = make_mixed_label_affinity(features, n_neighbors=n_neighbors, gamma=gamma)
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees) - affinity
    anchors = sparse.diags(np.where(labeled, float(mu), 0.0))

    one_hot_targets = np.zeros((len(features), num_classes), dtype=np.float64)
    one_hot_targets[np.flatnonzero(labeled), targets[labeled]] = 1.0
    right_hand_side = anchors @ one_hot_targets
    initial_system = (laplacian + anchors).tocsr()
    initial_labels = solve_sparse_label_system(
        initial_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="initial label propagation",
    )

    dissimilarity = make_dissimilarity_affinity(
        affinity=affinity,
        degrees=degrees,
        propagated_labels=initial_labels,
        temperature=float(temperature),
        edge_batch_size=int(edge_batch_size),
    )
    dissimilarity_degrees = np.asarray(dissimilarity.sum(axis=1)).ravel()
    signless_laplacian = sparse.diags(dissimilarity_degrees) + dissimilarity
    # Equation (24) sums both directions of each symmetric edge, yielding the
    # factor 2 in the derivative of beta/2 * D(G).
    mixed_system = (
        laplacian
        + anchors
        + 2.0 * float(beta) * signless_laplacian
    ).tocsr()
    mixed_labels = solve_sparse_label_system(
        mixed_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="mixed label propagation",
    )
    probabilities = normalize_mixed_label_rows(mixed_labels)
    confidences = entropy_confidence(probabilities)
    return probabilities.astype(np.float32), confidences.astype(np.float32)

def build_lrml_knn_graph(embeddings, n_neighbors, normalize):
    """Build the paper's binary symmetric kNN graph (the W_ij definition) with FAISS.

    Returns the directed kNN matrix (used only to co-locate neighbors in a batch),
    the symmetric binary adjacency W as a CSR matrix, and the degrees D_ii = sum_j W_ij.
    """

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("lrml regularization requires the faiss-cpu package") from exc

    features = np.ascontiguousarray(embeddings, dtype=np.float32).copy()
    if normalize:
        # On L2-normalized vectors the inner-product ranking matches the Euclidean
        # nearest-neighbor ordering the paper uses to define N(x).
        faiss.normalize_L2(features)
    num_nodes = len(features)
    k = min(int(n_neighbors), num_nodes - 1)
    if k <= 0:
        raise ValueError("lrml graph needs at least two samples")

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    # Query k + 1 because the first hit of each row is the node itself.
    _, neighbors = index.search(features, k + 1)

    neighbor_indices = np.empty((num_nodes, k), dtype=np.int64)
    rows, cols = [], []
    for node, neighbor_row in enumerate(neighbors):
        kept = 0
        for neighbor in neighbor_row:
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            neighbor_indices[node, kept] = neighbor
            rows.append(node)
            cols.append(neighbor)
            kept += 1
            if kept == k:
                break
        while kept < k:  # degenerate fallback if FAISS returned the node itself twice
            neighbor_indices[node, kept] = neighbor_indices[node, kept - 1]
            kept += 1

    directed = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    # W_ij = 1 if x_i in N(x_j) OR x_j in N(x_i): union of the directed graph and
    # its transpose, clipped back to a binary adjacency.
    symmetric = (directed + directed.T).tocsr()
    symmetric.data[:] = 1.0
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    degrees = np.asarray(symmetric.sum(axis=1), dtype=np.float64).ravel()
    return neighbor_indices, symmetric, degrees


def induced_subgraph_edges(adjacency, node_ids):
    """Upper-triangular edges of the sub-graph induced on ``node_ids``.

    Row/column indices are local (into the batch order given by ``node_ids``) so
    they index straight into the batch embedding matrix. Taking only the upper
    triangle counts each undirected edge of the symmetric graph exactly once.
    """

    sub = adjacency[node_ids][:, node_ids].tocoo()
    upper = sub.row < sub.col
    return sub.row[upper], sub.col[upper], sub.data[upper]

def make_mixed_label_affinity(features, n_neighbors, gamma):
    """Build equation (15)'s sparse symmetric cosine-affinity graph."""

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("mixed_label_propagation requires the faiss-cpu package") from exc

    normalized = np.ascontiguousarray(features, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    k = min(int(n_neighbors), len(normalized) - 1)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    similarities, neighbors = index.search(normalized, k + 1)

    rows = []
    columns = []
    values = []
    for query_index, (neighbor_row, similarity_row) in enumerate(zip(neighbors, similarities)):
        kept = 0
        for neighbor_index, similarity in zip(neighbor_row, similarity_row):
            if int(neighbor_index) == query_index:
                continue
            rows.append(int(neighbor_index))
            columns.append(query_index)
            values.append(max(float(similarity), 0.0) ** float(gamma))
            kept += 1
            if kept == k:
                break

    directed = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(normalized), len(normalized)),
        dtype=np.float64,
    ).tocsr()
    affinity = (directed + directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity


def make_dissimilarity_affinity(
    affinity,
    degrees,
    propagated_labels,
    temperature,
    edge_batch_size,
):
    """Compute first-order-neighbor hard-negative weights from equations (20)-(22)."""

    upper = sparse.triu(affinity, k=1).tocoo()
    if upper.nnz == 0:
        return sparse.csr_matrix(affinity.shape, dtype=np.float64)

    left = upper.row
    right = upper.col
    edge_weights = upper.data
    weights = np.empty(upper.nnz, dtype=np.float64)
    # Chunk edge/class computations to retain the paper's O(Nk + NC) memory
    # bound instead of materializing one O(NkC) tensor for the whole graph.
    for start in range(0, upper.nnz, int(edge_batch_size)):
        stop = min(start + int(edge_batch_size), upper.nnz)
        chunk_left = left[start:stop]
        chunk_right = right[start:stop]
        chunk_edge_weights = edge_weights[start:stop]
        left_logits = float(temperature) * (
            degrees[chunk_left, None] * propagated_labels[chunk_left]
            - chunk_edge_weights[:, None] * propagated_labels[chunk_right]
        )
        right_logits = float(temperature) * (
            degrees[chunk_right, None] * propagated_labels[chunk_right]
            - chunk_edge_weights[:, None] * propagated_labels[chunk_left]
        )
        left_probabilities = stable_softmax(left_logits)
        right_probabilities = stable_softmax(right_logits)
        dissimilarity_probability = 1.0 - np.sum(left_probabilities * right_probabilities, axis=1)
        weights[start:stop] = (
            entropy_confidence(left_probabilities)
            * entropy_confidence(right_probabilities)
            * dissimilarity_probability
        )

    dissimilarity = sparse.coo_matrix(
        (
            np.concatenate([weights, weights]),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=affinity.shape,
        dtype=np.float64,
    ).tocsr()
    dissimilarity.eliminate_zeros()
    return dissimilarity


def solve_sparse_label_system(matrix, right_hand_side, rtol, max_iter, name):
    """Solve one sparse positive-definite system per class using CG."""

    result = np.zeros_like(right_hand_side, dtype=np.float64)
    for class_index in range(right_hand_side.shape[1]):
        solution, info = sparse_linalg.cg(
            matrix,
            right_hand_side[:, class_index],
            rtol=rtol,
            atol=0.0,
            maxiter=max_iter,
        )
        if info != 0:
            raise RuntimeError(f"{name} conjugate gradient did not converge for class {class_index}: info={info}")
        result[:, class_index] = solution
    return result


def stable_softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def normalize_mixed_label_rows(values):
    """Convert propagated scores to the paper's L1-normalized probabilities."""

    l1_norms = np.linalg.norm(values, ord=1, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(l1_norms.ravel() == 0)
    if len(zero_rows) > 0:
        raise RuntimeError(
            "Mixed label propagation produced zero-L1-norm rows, which cannot be normalized "
            f"as specified by the paper: {zero_rows[:10].tolist()}"
        )
    return values / l1_norms


def entropy_confidence(probabilities):
    """Equation (21): one minus entropy normalized by log(number of classes)."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a matrix")
    if probabilities.shape[1] <= 1:
        return np.ones(probabilities.shape[0], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Equation (21) received non-finite normalized class values")
    if np.any(probabilities < 0):
        raise RuntimeError(
            "Mixed label propagation produced negative normalized class values; "
            "equation (21) is undefined because they are not probabilities"
        )
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise RuntimeError("Equation (21) received class values that do not sum to one")
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    entropy = -np.sum(entropy_terms, axis=1)
    return 1.0 - entropy / np.log(probabilities.shape[1])


def majority_vote(label_rows):
    """Return the most frequent label and its vote count for every row."""

    pseudo_labels = np.empty(label_rows.shape[0], dtype=np.int64)
    vote_counts = np.empty(label_rows.shape[0], dtype=np.int64)

    for row_index, labels in enumerate(label_rows):
        # np.unique sorts labels and returns an aligned occurrence count. In a
        # tie, argmax chooses the first/smallest label deterministically.
        unique_labels, counts = np.unique(labels, return_counts=True)
        best_index = int(np.argmax(counts))
        pseudo_labels[row_index] = unique_labels[best_index]
        vote_counts[row_index] = counts[best_index]

    return pseudo_labels, vote_counts


REGULARIZER_REGISTRY = {
    "stml": STMLRegularizer,
    "lrml": LRMLRegularizer,
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
    "mixed_label_propagation": MixedLabelPropagationPseudoLabeler(),
    # Generic composition point for supervised loss + unlabeled regularizer.
    "regularized": RegularizedSemiSupervisedMethod(name="regularized"),
}


def load_ssl_config(config_path, default_seed=0):
    """Load a JSON SSL config and fill in a missing seed from the run seed."""

    if config_path is None:
        # No config means fully supervised defaults with the run seed attached.
        return SemiSupervisedConfig(seed=default_seed)

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
        # Tie split randomness to the outer run seed unless the SSL config
        # explicitly asks for a different split seed.
        config = replace(config, seed=default_seed)
    validate_ssl_config(config, path)
    return config


def validate_ssl_config(config, path=None):
    """Validate label-selection and method settings before any data is loaded."""

    source = f" in {path}" if path is not None else ""
    if config.method != "none" and config.method not in METHOD_REGISTRY and config.method not in LOSS_DRIVEN_METHODS:
        raise ValueError(f"Unknown SSL method{source}: {config.method}. Available: {available_methods()}")
    if config.update_mode not in UPDATE_MODES:
        raise ValueError(f"Unknown SSL update_mode{source}: {config.update_mode}. Available: {sorted(UPDATE_MODES)}")
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
    if method is not None and method.is_regularization_method and config.update_mode != "once":
        raise ValueError(f"update_mode must be 'once' for regularization method {config.method!r}{source}")
    if config.method in LOSS_DRIVEN_METHODS and config.update_mode != "once":
        raise ValueError(f"update_mode must be 'once' for loss-driven method {config.method!r}{source}")
    if config.method in LOSS_DRIVEN_METHODS and config.method_params:
        raise ValueError(
            f"method_params must be empty for loss-driven method {config.method!r}{source}; "
            "configure the loss with loss_params"
        )
    if config.seed is None:
        raise ValueError(f"seed must be resolved before validation{source}")
    if config.labeled_per_class is None and not (0 < config.labeled_fraction <= 1):
        raise ValueError(f"labeled_fraction must be in (0, 1]{source}")
    if config.labeled_per_class is not None and config.labeled_per_class <= 0:
        raise ValueError(f"labeled_per_class must be positive{source}")
    if config.labeled_per_class is not None and config.label_sampling_mode not in {
        "per_class_min",
        "class_subset_k_shot",
    }:
        raise ValueError(
            f"labeled_per_class is only supported with label_sampling_mode='per_class_min' "
            f"or 'class_subset_k_shot'{source}"
        )
    if config.label_sampling_mode == "class_subset_k_shot" and config.labeled_per_class is None:
        raise ValueError(f"class_subset_k_shot requires labeled_per_class to set k-shot{source}")
    if not (0 <= config.confidence_threshold <= 1):
        raise ValueError(f"confidence_threshold must be in [0, 1]{source}")
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
        seed=config.seed,
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
    )
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


def build_labeled_training_dataset(train_dataset, train_labels_mapper, split):
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
    )


def build_loss_driven_training_dataset(train_dataset, split, num_views=1):
    """Expose labeled and unlabeled candidates without exposing their labels."""

    positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
    if len(positions) < 2:
        raise ValueError("loss-driven SSL requires at least two labeled or unlabeled training samples")
    return UnlabeledSubset(train_dataset, positions, num_views=num_views)


def make_semi_supervised_split(
    labels,
    label_sampling_mode,
    labeled_fraction,
    labeled_per_class,
    max_unlabeled_samples,
    seed,
):
    """Select labeled positions and optionally cap the unlabeled candidate pool."""

    # One RNG instance drives all choices in this split, making it reproducible
    # from a single seed.
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    labeled_positions, unlabeled_by_label = select_labeled_positions(
        labels=labels,
        label_sampling_mode=label_sampling_mode,
        labeled_fraction=labeled_fraction,
        labeled_per_class=labeled_per_class,
        rng=rng,
    )

    # Sort outputs so their order is stable and independent of dictionary/loop
    # traversal after random selection has finished.
    labeled_positions = np.asarray(sorted(labeled_positions), dtype=np.int64)
    unlabeled_positions = concatenate_position_groups(unlabeled_by_label.values())

    # Capping can make embedding extraction and graph methods tractable on
    # large datasets while leaving the labeled budget unchanged.
    if max_unlabeled_samples is not None and len(unlabeled_positions) > max_unlabeled_samples:
        # Sample without replacement, then sort below to restore deterministic
        # dataset order for embedding extraction.
        unlabeled_positions = rng.choice(unlabeled_positions, size=max_unlabeled_samples, replace=False)

    return SemiSupervisedSplit(
        labeled_positions=labeled_positions,
        unlabeled_positions=np.asarray(sorted(unlabeled_positions), dtype=np.int64),
    )


def select_labeled_positions(labels, label_sampling_mode, labeled_fraction, labeled_per_class, rng):
    """Dispatch to one of the supported label-budget semantics."""

    if label_sampling_mode == "per_class_min":
        # Every class receives at least one labeled example (or a fixed k).
        return select_per_class_min_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    if label_sampling_mode == "global_budget":
        # labeled_fraction controls the total number of labeled samples.
        return select_global_budget_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset":
        # labeled_fraction controls how many classes are fully labeled.
        return select_class_subset_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset_k_shot":
        # labeled_fraction selects classes; labeled_per_class selects k examples
        # from each chosen class.
        return select_class_subset_k_shot_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    raise ValueError(f"Unknown label_sampling_mode: {label_sampling_mode}")


def make_permuted_positions_by_label(labels, rng):
    """Group positions by class and independently shuffle every class."""

    positions_by_label = {}
    for label in np.unique(labels):
        # flatnonzero returns positions in the current training subset. Shuffle
        # each class independently so prefixes form random labeled selections.
        class_positions = np.flatnonzero(labels == label)
        positions_by_label[int(label)] = rng.permutation(class_positions)
    return positions_by_label


def select_per_class_min_labeled_positions(labels, labeled_fraction, labeled_per_class, rng):
    """Label a fraction or fixed count within every represented class."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    labeled_positions = []
    unlabeled_by_label = {}

    for label, class_positions in positions_by_label.items():
        if labeled_per_class is None:
            # round approximates the requested per-class fraction; max(1, ...)
            # guarantees every class remains represented in labeled training.
            num_labeled = max(1, int(round(len(class_positions) * labeled_fraction)))
        else:
            # Classes smaller than k contribute all available samples.
            num_labeled = min(labeled_per_class, len(class_positions))

        # Because class_positions was shuffled, its prefix is a random labeled
        # subset and the suffix is the same class's unlabeled pool.
        labeled_positions.extend(class_positions[:num_labeled])
        unlabeled_by_label[int(label)] = class_positions[num_labeled:]

    return labeled_positions, unlabeled_by_label


def select_global_budget_labeled_positions(labels, labeled_fraction, rng):
    """Spend one dataset-wide label budget in a class-balanced round robin."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # Convert the fraction into one global integer budget, guaranteeing at least
    # one labeled sample and never exceeding the dataset.
    target_labeled = max(1, int(np.floor(len(labels) * labeled_fraction)))
    target_labeled = min(target_labeled, len(labels))
    selected_counts = {int(label): 0 for label in unique_labels}
    labeled_positions = []

    # Selecting one item per shuffled class per pass spreads a small global
    # budget across classes before assigning second examples to any class.
    while len(labeled_positions) < target_labeled:
        made_progress = False
        for label in rng.permutation(unique_labels):
            label = int(label)
            selected_count = selected_counts[label]
            class_positions = positions_by_label[label]
            if selected_count >= len(class_positions):
                # Small/exhausted classes stop contributing while larger classes
                # can continue filling the remaining global budget.
                continue

            # Take the next unused position from this class's shuffled list.
            labeled_positions.append(class_positions[selected_count])
            selected_counts[label] = selected_count + 1
            made_progress = True
            if len(labeled_positions) == target_labeled:
                break

        if not made_progress:
            break

    # Everything after each class's consumed prefix remains an unlabeled
    # candidate for pseudo-labeling.
    unlabeled_by_label = {
        int(label): class_positions[selected_counts[int(label)] :]
        for label, class_positions in positions_by_label.items()
    }
    return labeled_positions, unlabeled_by_label


def select_class_subset_labeled_positions(labels, class_fraction, rng):
    """Fully label a random fraction of classes and leave the rest unlabeled."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # Convert class_fraction to a count, retaining at least one selected class.
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            # Selected classes are completely labeled in this mode.
            labeled_positions.extend(class_positions)
            unlabeled_by_label[int(label)] = np.array([], dtype=np.int64)
        else:
            # Every sample from an unselected class becomes an unlabeled
            # candidate.
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def select_class_subset_k_shot_labeled_positions(labels, class_fraction, labeled_per_class, rng):
    """Select a class subset, then label at most k examples in each class."""

    if labeled_per_class is None:
        raise ValueError("class_subset_k_shot requires labeled_per_class to set k-shot")

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # First choose how many classes are visible to labeled training.
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            # Then spend the k-shot budget only inside selected classes.
            num_labeled = min(int(labeled_per_class), len(class_positions))
            labeled_positions.extend(class_positions[:num_labeled])
            unlabeled_by_label[int(label)] = class_positions[num_labeled:]
        else:
            # Classes outside the selected subset have no ground-truth examples
            # exposed to the training method.
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def concatenate_position_groups(groups):
    # Discard empty per-class arrays so np.concatenate always receives at least
    # one nonempty input when there are unlabeled candidates.
    arrays = [np.asarray(group, dtype=np.int64) for group in groups if len(group) > 0]
    if not arrays:
        return np.array([], dtype=np.int64)
    return np.concatenate(arrays).astype(np.int64, copy=False)


def filter_pseudo_labels(pseudo_labels, confidence_threshold, valid_mapped_labels):
    """Drop low-confidence predictions and labels unknown to the train mapping."""

    # Start with predictions that refer to a class represented in the current
    # training label mapping. This protects against invalid estimator outputs.
    keep = np.isin(pseudo_labels.mapped_labels, list(valid_mapped_labels))

    if pseudo_labels.confidences is not None:
        # Combine conditions elementwise so positions, labels, and confidences
        # remain aligned after boolean indexing.
        keep = keep & (pseudo_labels.confidences >= confidence_threshold)

    dropped = int(len(keep) - keep.sum())
    if dropped > 0:
        logger.info(f"Dropped {dropped} pseudo-labels below confidence threshold or outside known classes")

    # Apply the same mask to every aligned result array.
    return PseudoLabelResult(
        positions=pseudo_labels.positions[keep],
        mapped_labels=pseudo_labels.mapped_labels[keep],
        confidences=None if pseudo_labels.confidences is None else pseudo_labels.confidences[keep],
    )


def summarize_pseudo_label_result(pseudo_labels, true_labels):
    """Return JSON-safe distribution and hidden-label audit metrics."""

    positions = np.asarray(pseudo_labels.positions, dtype=np.int64)
    predicted_labels = np.asarray(pseudo_labels.mapped_labels, dtype=np.int64)
    if len(positions) == 0:
        audit_accuracy = None
        true_class_counts = {}
        correct_counts_by_true_class = {}
        audit_accuracy_by_true_class = {}
    else:
        audit_labels = true_labels[positions]
        correct = predicted_labels == audit_labels
        audit_accuracy = float(np.mean(correct))
        true_class_counts = count_values(audit_labels)
        correct_counts_by_true_class = {
            str(int(label)): int(correct[audit_labels == label].sum())
            for label in np.unique(audit_labels)
        }
        audit_accuracy_by_true_class = {
            label: float(correct_counts_by_true_class[label] / count)
            for label, count in true_class_counts.items()
        }

    return {
        "count": int(len(positions)),
        "predicted_class_counts": count_values(predicted_labels),
        "true_class_counts": true_class_counts,
        "correct_counts_by_true_class": correct_counts_by_true_class,
        "audit_accuracy": audit_accuracy,
        "audit_accuracy_by_true_class": audit_accuracy_by_true_class,
        "confidence": summarize_numeric_values(pseudo_labels.confidences),
    }


def summarize_pseudo_label_changes(previous, current):
    """Compare position-aligned pseudo-label predictions across generations."""

    if previous is None:
        return None
    current_map = pseudo_labels_to_position_map(current)
    previous_positions = set(previous)
    current_positions = set(current_map)
    overlapping_positions = previous_positions & current_positions
    changed_count = sum(previous[position] != current_map[position] for position in overlapping_positions)
    return {
        "overlap_count": int(len(overlapping_positions)),
        "changed_count": int(changed_count),
        "changed_fraction": None
        if not overlapping_positions
        else float(changed_count / len(overlapping_positions)),
        "added_count": int(len(current_positions - previous_positions)),
        "removed_count": int(len(previous_positions - current_positions)),
    }


def pseudo_labels_to_position_map(pseudo_labels):
    return {
        int(position): int(label)
        for position, label in zip(pseudo_labels.positions, pseudo_labels.mapped_labels)
    }


def count_values(values):
    values = np.asarray(values)
    if len(values) == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def summarize_numeric_values(values):
    if values is None:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def format_optional_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def format_change_summary(changes):
    if changes is None:
        return "n/a"
    return (
        f"{changes['changed_count']}/{changes['overlap_count']} changed, "
        f"{changes['added_count']} added, {changes['removed_count']} removed"
    )


def make_relabeled_training_dataset(train_dataset, train_labels_mapper, labeled_positions, pseudo_labels):
    """Combine true and pseudo labels into the dataset view used for training."""

    # labels contains dense mapped IDs; orig_labels contains source class IDs.
    # Both arrays are aligned with positions in train_dataset.
    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    orig_labels = np.asarray(train_dataset.orig_labels, dtype=np.int64)
    # Pseudo-labelers predict dense mapped labels.  Convert them back to
    # original labels because RelabeledSubset.__getitem__ follows the same
    # contract as the source training dataset.
    inverse_labels_mapper = {mapped: original for original, mapped in train_labels_mapper.items()}

    # Build all result arrays in the same order: true-labeled samples first,
    # followed by accepted pseudo-labeled samples.
    all_positions = np.concatenate([labeled_positions, pseudo_labels.positions])
    # True-labeled positions use their known dense labels; pseudo-labeled
    # positions use the predictions generated by the SSL method.
    all_mapped_labels = np.concatenate([labels[labeled_positions], pseudo_labels.mapped_labels])
    pseudo_orig_labels = np.asarray(
        [inverse_labels_mapper[int(label)] for label in pseudo_labels.mapped_labels],
        dtype=np.int64,
    )
    # RelabeledSubset returns original IDs, so concatenate known source labels
    # with inverse-mapped predicted source labels in the same sample order.
    all_orig_labels = np.concatenate([orig_labels[labeled_positions], pseudo_orig_labels])
    pseudo_confidences = (
        np.ones(len(pseudo_labels.positions), dtype=np.float32)
        if pseudo_labels.confidences is None
        else np.asarray(pseudo_labels.confidences, dtype=np.float32)
    )
    all_confidences = np.concatenate(
        [np.ones(len(labeled_positions), dtype=np.float32), pseudo_confidences]
    )

    return RelabeledSubset(
        dataset=train_dataset,
        positions=all_positions,
        orig_labels=all_orig_labels,
        mapped_labels=all_mapped_labels,
        confidences=all_confidences,
    )


def extract_embeddings(
    model,
    dataset,
    positions,
    device,
    batch_size,
    num_workers,
    seed,
    start_method,
    desc,
    embedding_kind="default",
):
    """Extract deterministic evaluation-transform embeddings for given positions."""

    # Work on a copy using deterministic feature transforms; training
    # augmentation would make pseudo-labels depend on random image distortions.
    feature_dataset = make_feature_dataset(dataset)
    # Subset preserves the positions order, which all pseudo-label methods rely
    # on when splitting the resulting embedding matrix.
    loader = DataLoader(
        Subset(feature_dataset, [int(position) for position in positions]),
        batch_size=batch_size,
        shuffle=False,
        **utils.make_dataloader_kwargs(num_workers, seed, start_method),
    )

    # Pseudo-labels should use stable evaluation behavior, but restore the
    # caller's previous mode after extraction.
    was_training = model.training
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=desc):
            # Labels are deliberately ignored: pseudo-label generation must use
            # only images/embeddings for the unlabeled candidate pool.
            if embedding_kind == "default":
                forward_cached = getattr(model, "forward_cached", None)
                embeddings = model(images.to(device)) if forward_cached is None else forward_cached(images, device)
            elif embedding_kind == "stml_g":
                forward_stml_cached = getattr(model, "forward_stml_cached", None)
                if forward_stml_cached is None:
                    raise AttributeError("Model does not expose forward_stml_cached")
                embeddings, _ = forward_stml_cached(images, device)
            else:
                raise ValueError(f"Unknown embedding_kind: {embedding_kind}")
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
    if was_training:
        model.train()

    # Concatenation restores one [num_positions, embedding_dim] matrix in loader
    # order.
    return np.concatenate(all_embeddings)


def make_feature_dataset(dataset):
    """Copy a dataset and replace augmentation with its feature transform."""

    # Copy before changing transforms so the real training dataset continues to
    # use stochastic augmentation.
    feature_dataset = copy.deepcopy(dataset)
    feature_transform = getattr(dataset, "feature_transform", None)
    if feature_transform is not None:
        set_nested_transform(feature_dataset, feature_transform)
    return feature_dataset


def set_nested_transform(dataset, transform):
    """Set the transform on the base dataset beneath any Subset wrappers."""

    utils.set_nested_transform(dataset, transform)
