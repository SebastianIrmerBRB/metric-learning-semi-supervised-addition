"""Shared configuration, constants, and value types for SSL."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


UNLABELED_TARGET = -1


# ``every_n_samples`` schedules refreshes on consumed training samples rather
# than on epochs. See ``sample_scoped_refresh_interval_steps``.
UPDATE_MODES = {"once", "every_epoch", "every_n_epochs", "every_n_samples"}

# What the SSL phase starts from at the ``warmup_epochs`` boundary.
#
# ``restore_best`` reloads the warm-up's best-on-validation checkpoint before the
# boundary epoch builds its loader, so the SSL phase begins from the best
# supervised model the warm-up produced rather than from whatever the last
# warm-up epoch happened to leave behind. On a small labeled split the supervised
# head routinely peaks early and then decays for the rest of the warm-up, which
# otherwise hands the SSL phase a measurably worse starting point -- and, for a
# graph method, a graph built from those worse embeddings. It also makes
# ``warmup_epochs`` cost only compute instead of accuracy, which is what a
# searched hyperparameter should do.
#
# ``keep_last`` keeps training from the final warm-up epoch's weights. It is the
# original behavior and the default: it is what every recorded run predating this
# option did, and ``restore_best`` has not been shown to beat it. The one
# controlled comparison so far went the other way -- on a noisy validation curve
# the warm-up's argmax is a lucky fluctuation, and restoring it discarded six
# real epochs of optimization to chase it. Treat ``restore_best`` as opt-in and
# unvalidated until measured in the regime you care about.
WARMUP_CHECKPOINT_MODES = ("restore_best", "keep_last")


PSEUDO_LABEL_DIAGNOSTICS_MODES = {"off", "log", "save"}


GRAPH_DIAGNOSTICS_MODES = {"off", "save"}


GRAPH_DIAGNOSTICS_LAYOUTS = {"pacmap", "tsne", "pca"}


# How the class-scoped graph view picks the classes it renders and reports on.
# ``off`` keeps the whole-graph edge sample that bounds the plot by itself.
# The others select a class set and report how those classes' nodes -- their
# labeled anchors and their unlabeled members alike, scored against the true
# labels the run withheld -- ended up connected to each other and to the rest.
GRAPH_DIAGNOSTICS_CLASS_FOCUS_MODES = {
    "off",
    "lowest_purity",
    "largest",
    "random",
    "explicit",
}


# ``in_batch_merged`` is ``in_batch`` with the objective folded into one term:
# the graph spans the whole supervised batch plus a uniformly drawn unlabeled
# batch, and the single loss covers every row that ends up with a label, true or
# pseudo. ``in_batch`` instead subsamples the labeled rows and adds its term on
# top of the engine's supervised loss, which counts the labeled rows twice.
GRAPH_BATCH_MODES = {"global", "in_batch", "in_batch_merged"}


# The two per-step modes, as opposed to the whole-pool ``global`` one.
IN_BATCH_GRAPH_MODES = frozenset({"in_batch", "in_batch_merged"})


LABEL_SAMPLING_MODES = {
    "per_class_min",
    "per_class_imbalanced",
    "global_budget",
    "class_subset",
    "class_subset_k_shot",
}


# Which classes may contribute unlabeled candidates. ``all`` is the historical
# behavior: everything in the training split outside the labeled support is a
# candidate, including entire classes the support never covers.
# ``labeled_classes`` keeps only the classes the support does cover, which is the
# in-distribution regime -- the SSL objective then never sees an unseen class, so
# a drop against ``all`` measures class mismatch rather than label scarcity.
UNLABELED_CLASS_SCOPE_ALL = "all"


UNLABELED_CLASS_SCOPE_LABELED_CLASSES = "labeled_classes"


UNLABELED_CLASS_SCOPES = {
    UNLABELED_CLASS_SCOPE_ALL,
    UNLABELED_CLASS_SCOPE_LABELED_CLASSES,
}


LOSS_DRIVEN_METHODS = {"stml"}


# Pool-wide pseudo-label methods that can feed ``TwoStreamMPerClassBatchSampler``
# through ``labeled_batch_size``. The sampler only needs a relabeled dataset that
# keeps the true-labeled and pseudo-labeled positions apart, which every
# whole-pool method produces; the set records which ones have been validated
# against the stream sizing rules rather than a structural limit.
TWO_STREAM_SAMPLER_METHODS = frozenset({"iscen_label_spreading", "stml_threshold"})


# How the two-stream sampler relates the classes drawn by each stream.
# ``independent`` draws each stream's classes on its own, as LP-DeepSSL does, so
# shared classes yield cross-stream positive pairs; it is both the default and
# the behavior every run before this option used. ``disjoint`` keeps the
# combined batch globally M-per-class at the price of never placing a class's
# true- and pseudo-labeled samples in the same batch.
CLASS_OVERLAP_MODES = {"disjoint", "independent"}


DEFAULT_SUPPORT_SEED = 7


@dataclass(frozen=True)
class SemiSupervisedConfig:
    """All settings needed to select labels and generate pseudo-labels."""

    method: str = "none"
    update_mode: str = "once"
    update_interval_epochs: int = 1
    # Only read by update_mode='every_n_samples'. An epoch is a different amount
    # of training on every dataset -- length_before_new_iter is resolved from the
    # fold's own pool, so one epoch is ~46k samples on semi-iNat and a fraction of
    # that on Cars196 -- which makes an epoch-scoped refresh interval mean a very
    # different amount of drift between rebuilds. This is that interval in
    # samples, divided by the batch size once per run to get a step count.
    update_interval_samples: int | None = None
    warmup_epochs: int = 0
    # Model selection and early stopping normally span the whole fold, carrying
    # the warm-up's high-water mark into the SSL phase. That leaves the SSL
    # phase unmeasurable whenever the supervised warm-up already peaks: the
    # reported ``best_valid_*`` is then a warm-up epoch's score, so the method's
    # own hyperparameters cannot move an HPO objective built on it, and the
    # first post-warm-up epoch already starts consuming patience against a
    # target the SSL objective has had no epoch to reach. Setting this restarts
    # selection at ``warmup_epochs``: the SSL phase gets a fresh baseline, its
    # own full patience budget, and a ``best_valid_*`` that describes only
    # itself. The price is deliberate -- the selected checkpoint is the best
    # *SSL* epoch even when a warm-up epoch scored higher, because the question
    # this setting answers is what the SSL phase did, not what the fold's best
    # model was. ``metrics.csv`` still records the full curve either way.
    restart_selection_after_warmup: bool = False
    # Which warm-up checkpoint the SSL phase continues from; see
    # ``WARMUP_CHECKPOINT_MODES``. Independent of
    # ``restart_selection_after_warmup``: that setting decides which epochs may
    # *win* selection, this one decides which weights the SSL phase *starts
    # from*. Inert when ``warmup_epochs`` is 0, and skipped when the run keeps no
    # validation checkpoint to restore (final full-development training).
    warmup_checkpoint_mode: str = "keep_last"
    label_sampling_mode: str = "global_budget"
    labeled_fraction: float = 1.0
    labeled_per_class: int | None = None
    seed: int | None = None
    support_seed: int | None = DEFAULT_SUPPORT_SEED
    confidence_threshold: float = 0.0
    # A two-stream M-per-class batch needs enough distinct pseudo-label classes.
    # When the global threshold removes too many classes, rejected predictions
    # can be rescued class-by-class down to this lower absolute floor.
    pseudo_label_rescue_confidence_floor: float = 0.0
    # ``None`` uses sampler_m, giving every rescued class up to one complete
    # M-per-class group without introducing a second required hyperparameter.
    pseudo_label_rescue_top_k: int | None = None
    labeled_batch_size: int | None = None
    # Only read when labeled_batch_size selects the two-stream sampler.
    class_overlap: str = "independent"
    # ``global`` preserves the original whole-pool graph construction.
    # ``in_batch`` follows the B / mu-B two-stream convention: every training
    # step builds a fresh graph over a sampled labeled and unlabeled set.
    graph_batch_mode: str = "global"
    graph_labeled_batch_size: int | None = None
    graph_unlabeled_batch_size: int | None = None
    pseudo_label_diagnostics_mode: str = "save"
    graph_diagnostics_mode: str = "off"
    graph_diagnostics_max_nodes: int = 400
    graph_diagnostics_max_edges: int = 2000
    graph_diagnostics_max_labels: int = 80
    graph_diagnostics_layout: str = "pacmap"
    # Class-scoped view of the same graph. It changes nothing about training or
    # about the full-graph statistics; it selects which classes the plot and the
    # class-focus report are built around.
    graph_diagnostics_class_focus: str = "off"
    graph_diagnostics_class_count: int = 6
    # Explicit class ids, required by graph_diagnostics_class_focus='explicit'.
    graph_diagnostics_classes: list[int] | None = None
    # Also draw and count the out-of-selection nodes the focus classes attach
    # to. Without them the plot is an induced subgraph, which hides exactly the
    # leaked edges that explain a low-purity class.
    graph_diagnostics_class_context: bool = True
    max_unlabeled_samples: int | None = None
    # Class-balanced share of the unlabeled pool kept after
    # ``unlabeled_class_scope`` has been applied; ``None`` keeps all of it. It
    # differs from ``max_unlabeled_samples`` in both respects that matter for a
    # sweep: the share is apportioned per class rather than drawn from the pool
    # as a whole, and successive values nest, so 0.25 is a subset of 0.5 up to
    # per-class rounding. Both may be set; the fraction applies first.
    unlabeled_fraction: float | None = None
    # See ``UNLABELED_CLASS_SCOPES``.
    unlabeled_class_scope: str = UNLABELED_CLASS_SCOPE_ALL
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
    """Output settings for one graph-diagnostics artifact bundle."""

    output_dir: Path
    slug: str
    title: str
    max_nodes: int
    max_edges: int
    max_labels: int
    seed: int
    layout: str = "pacmap"
    series_slug: str | None = None
    epoch: int | None = None
    class_focus: str = "off"
    class_count: int = 6
    classes: tuple[int, ...] | None = None
    class_context: bool = True


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
    if update_mode == "every_n_samples":
        # The engine owns this schedule: it counts the samples each step consumes,
        # which an epoch index cannot express, and asks for every refresh after
        # this first one explicitly.
        return False
    if update_mode == "once":
        return False
    current_epoch = 0 if epoch is None else int(epoch)
    return current_epoch - int(last_rebuild_epoch) >= interval_epochs


def sample_scoped_refresh_interval_steps(interval_samples, batch_size):
    """Convert a sample-scoped refresh interval into a number of training steps.

    The interval is configured in samples so one value means the same amount of
    training on every dataset, and so it stays independent of whichever batch
    size a trial happens to draw. Both are only known per run, hence the
    conversion here rather than in the config.
    """

    interval_samples = int(interval_samples)
    if interval_samples <= 0:
        raise ValueError("update_interval_samples must be positive")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Rounding rather than flooring keeps an interval below one batch from
    # becoming a per-step rebuild, which is what graph_batch_mode='in_batch' is.
    return max(1, round(interval_samples / batch_size))
