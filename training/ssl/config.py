"""Shared configuration, constants, and value types for SSL."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


UNLABELED_TARGET = -1


UPDATE_MODES = {"once", "every_epoch", "every_n_epochs"}


PSEUDO_LABEL_DIAGNOSTICS_MODES = {"off", "log", "save"}


GRAPH_DIAGNOSTICS_MODES = {"off", "save"}


GRAPH_DIAGNOSTICS_LAYOUTS = {"pacmap", "tsne", "pca"}


LABEL_SAMPLING_MODES = {
    "per_class_min",
    "per_class_imbalanced",
    "global_budget",
    "class_subset",
    "class_subset_k_shot",
}


LOSS_DRIVEN_METHODS = {"stml"}


DEFAULT_SUPPORT_SEED = 7


@dataclass(frozen=True)
class SemiSupervisedConfig:
    """All settings needed to select labels and generate pseudo-labels."""

    method: str = "none"
    update_mode: str = "once"
    update_interval_epochs: int = 1
    warmup_epochs: int = 0
    label_sampling_mode: str = "global_budget"
    labeled_fraction: float = 1.0
    labeled_per_class: int | None = None
    seed: int | None = None
    support_seed: int | None = DEFAULT_SUPPORT_SEED
    confidence_threshold: float = 0.0
    labeled_batch_size: int | None = None
    pseudo_label_diagnostics_mode: str = "save"
    graph_diagnostics_mode: str = "off"
    graph_diagnostics_max_nodes: int = 400
    graph_diagnostics_max_edges: int = 2000
    graph_diagnostics_max_labels: int = 80
    graph_diagnostics_layout: str = "pacmap"
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


@dataclass(frozen=True)
class GraphDiagnosticsRequest:
    """Output settings for one graph visualization."""

    output_dir: Path
    slug: str
    title: str
    max_nodes: int
    max_edges: int
    max_labels: int
    seed: int
    layout: str = "pacmap"


def should_rebuild_on_epoch(update_mode, interval_epochs, epoch, last_rebuild_epoch):
    """Return whether an epoch-scoped SSL artifact should be regenerated."""

    if update_mode not in UPDATE_MODES:
        raise ValueError(f"Unknown update mode: {update_mode}")
    interval_epochs = int(interval_epochs)
    if interval_epochs <= 0:
        raise ValueError("update interval must be positive")
    if update_mode == "every_epoch":
        return True
    if last_rebuild_epoch is None:
        return True
    if update_mode == "once":
        return False
    current_epoch = 0 if epoch is None else int(epoch)
    return current_epoch - int(last_rebuild_epoch) >= interval_epochs
