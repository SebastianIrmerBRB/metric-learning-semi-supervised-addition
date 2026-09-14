"""Shared constants and immutable result types for experiment execution."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pytorch_metric_learning import losses

DATASETS = [
    "Cars196",
    "CUB",
    "DeepFashionInShop",
    "Food101",
    "Flowers102",
    "FGVCAircraft",
    "FGVCFungi",
    "CIFAR10",
    "CIFAR100",
    "NABirds",
    "PUMA",
    "PUMAStandard",
    "Pittsburgh30k",
    "Pittsburgh250k",
    "SemiAves",
    "SPED",
    "iNat",
    "iNat2018",
    "StanfordDogs",
    "StanfordOnlineProducts",
    "VehicleID"
]

# Default for ``--warmup_loss``. A labeled-only warm-up should train the run's
# own objective unless told otherwise; naming a specific loss as the default
# meant that whichever loss happened to be the default got trained instead, with
# stock parameters, on every code path that did not go through Optuna.
WARMUP_LOSS_SAME_AS_LOSS = "same_as_loss"

ALL_LOSSES = [
    "AngularLoss",
    "ArcFaceLoss",
    "BaseMetricLossFunction",
    "CircleLoss",
    "ContrastiveLoss",
    "CosFaceLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "GenericPairLoss",
    "HistogramLoss",
    "InstanceLoss",
    "IntraPairVarianceLoss",
    "LargeMarginSoftmaxLoss",
    "GeneralizedLiftedStructureLoss",
    "LiftedStructureLoss",
    "ManifoldLoss",
    "MarginLoss",
    "MixedLabelPropagationProxyLoss",
    "WeightRegularizerMixin",
    "MultiSimilarityLoss",
    "MultipleLosses",
    "NPairsLoss",
    "NCALoss",
    "NormalizedSoftmaxLoss",
    "NTXentLoss",
    "P2SGradLoss",
    "PNPLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "RankedListLoss",
    "SelfSupervisedLoss",
    "SignalToNoiseRatioContrastiveLoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "STMLLoss",
    "SubCenterArcFaceLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletMarginLoss",
    "TupletMarginLoss",
    "VICRegLoss",
]

CLASSIFICATION_LOSSES = [
    "ArcFaceLoss",
    "CosFaceLoss",
    "LargeMarginSoftmaxLoss",
    "MixedLabelPropagationProxyLoss",
    "WeightRegularizerMixin",
    "NormalizedSoftmaxLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
]

ALL_MINERS = [
    "no_miner",
    "AngularMiner",
    "BatchEasyHardMiner",
    "BatchHardMiner",
    "DistanceWeightedMiner",
    "HDCMiner",
    "MultiSimilarityMiner",
    "PairMarginMiner",
    "TripletMarginMiner",
    "UniformHistogramMiner",
]

SELECTION_METRIC_PRECISION_AT_1 = "precision_at_1"

SELECTION_METRIC_MAP_AT_R = "map_at_r"

SELECTION_METRICS = (SELECTION_METRIC_PRECISION_AT_1, SELECTION_METRIC_MAP_AT_R)

OBJECTIVE_METRICS = {
    "best_valid_precision_at_1",
    "best_valid_mean_average_precision_at_r",
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
    "final_train_loss",
}

COMPARISON_FORBIDDEN_HPARAM_KEYS = {
    "dataset",
    "dataset_protocol",
    "cifar_imbalance_factor",
    "cifar_train_fraction",
    "cifar_test_fraction",
    "semi_aves_ood_fraction",
    "semi_aves_ood_seed",
    "semi_inat_ood_fraction",
    "semi_inat_ood_seed",
    "mode",
    "seed",
    "hparam_seed",
    "data_split_seed",
    "support_seed",
    "cv_k",
    "cv_mode",
    "val_mode",
    "ssl.labeled_fraction",
    "ssl_config.labeled_fraction",
    "ssl.label_sampling_mode",
    "ssl_config.label_sampling_mode",
    "ssl.max_unlabeled_samples",
    "ssl_config.max_unlabeled_samples",
    "ssl.unlabeled_fraction",
    "ssl_config.unlabeled_fraction",
    "ssl.unlabeled_class_scope",
    "ssl_config.unlabeled_class_scope",
    "ssl.seed",
    "ssl_config.seed",
    "ssl.support_seed",
    "ssl_config.support_seed",
    "ssl.method",
    "ssl_config.method",
}

LABELED_PER_CLASS_HPARAM_KEYS = {
    "ssl.labeled_per_class",
    "ssl_config.labeled_per_class",
}

BATCH_SAMPLER_HPARAM_KEY = "batch_sampler"

LABELED_BATCH_SIZE_HPARAM_KEY = "labeled_batch_size"

LABELED_BATCH_SIZE_HPARAM_KEYS = {
    LABELED_BATCH_SIZE_HPARAM_KEY,
    "ssl.labeled_batch_size",
    "ssl_config.labeled_batch_size",
}

# Searches the labeled share of the batch instead of its absolute size, so the
# value means the same thing at every batch_sampler choice and never has to be
# filtered against it. Derived into labeled_batch_size per trial; it is not a
# SemiSupervisedConfig field, so the ssl./ssl_config. prefixes are rejected.
LABELED_BATCH_FRACTION_HPARAM_KEY = "labeled_batch_fraction"

LABELED_BATCH_FRACTION_HPARAM_ALIASES = {
    "ssl.labeled_batch_fraction",
    "ssl_config.labeled_batch_fraction",
}

# labeled_batch_size <= unlabeled_batch_size caps the labeled share at half the
# batch; larger fractions would all collapse onto that cap.
# changed it to 0.75 - remove this line - this is better this way trust
MAX_LABELED_BATCH_FRACTION = 0.75

LOSS_HPARAM_PREFIX = "loss."

MINER_HPARAM_PREFIX = "miner."

JOINT_COMPONENT_HPARAM_PREFIX = "__joint_component__."

JOINT_HPARAM_PREFIX = "__joint_hparam__."

JOINT_TWO_STREAM_HPARAM_KEY = f"{JOINT_HPARAM_PREFIX}two_stream_batch_sampler"

JOINT_STML_IN_BATCH_HPARAM_KEY = f"{JOINT_HPARAM_PREFIX}stml_in_batch_graph"

# Graph sizes only the in-batch STML mode reads; the pool-wide mode has no
# per-step graph to size, so a study covering both modes drops them there.
STML_IN_BATCH_ONLY_HPARAM_KEYS = (
    "ssl_config.graph_labeled_batch_size",
    "ssl_config.graph_unlabeled_batch_size",
)

# Sampler and graph sizes the in-batch STML mode constrains against each other.
STML_IN_BATCH_HPARAM_KEYS = (
    *STML_IN_BATCH_ONLY_HPARAM_KEYS,
    "ssl_config.method_params.n_neighbors",
)

# Dimensions the merged in-batch mode never reads: it takes the supervised batch
# whole, draws the unlabeled rows uniformly, and runs one weighted loss. Leaving
# them in would spend trials on values the run ignores -- and the config layer
# rejects them outright, so every such trial would be lost.
STML_MERGED_INERT_HPARAM_KEYS = (
    "ssl_config.graph_labeled_batch_size",
    "ssl_config.method_params.n_neighbors",
    "ssl_config.method_params.supervised_weight",
)

HPO_MODE_KEYS = {"backbone_tuning", "use_cache"}

# Structural keys consumed while loading an HPO config rather than searched over.
HPARAM_CONFIG_EXTENDS_KEY = "extends"

HPARAM_CONFIG_PER_LOSS_KEY = "per_loss"

# Objects merged entry by entry across inheritance layers. Everything else is
# replaced outright so a layer can reset a value to an empty object.
MERGED_HPARAM_CONFIG_KEYS = {"spaces", HPARAM_CONFIG_PER_LOSS_KEY}

SAMPLER_CAPACITY_HPARAM_KEYS = {
    "dataset",
    "dataset_protocol",
    "cifar_imbalance_factor",
    "cifar_train_fraction",
    "cifar_test_fraction",
    "seed",
    "data_split_seed",
    "support_seed",
    "cv_k",
    "cv_mode",
    "val_mode",
    "ssl.label_sampling_mode",
    "ssl_config.label_sampling_mode",
    "ssl.labeled_fraction",
    "ssl_config.labeled_fraction",
    "ssl.labeled_per_class",
    "ssl_config.labeled_per_class",
    "ssl.seed",
    "ssl_config.seed",
    "ssl.support_seed",
    "ssl_config.support_seed",
}

SUPERVISED_SPLIT_SSL_HPARAM_KEYS = {
    *LABELED_PER_CLASS_HPARAM_KEYS,
}

@dataclass(frozen=True)
class TrainingResult:
    """Metrics and artifact locations returned by a training or CV run."""

    log_dir: Path
    metrics_csv: Path
    best_valid_precision_at_1: float | None
    best_valid_mean_average_precision_at_r: float | None
    test_precision_at_1: float | None
    test_mean_average_precision_at_r: float | None
    final_train_loss: float | None
    last_epoch: int
    selected_epoch: int
    global_step: int
    epoch0_test_precision_at_1: float | None = None
    epoch0_test_mean_average_precision_at_r: float | None = None
    cv_k: int = 1
    cv_mode: str | None = None
    cv_fold: int | None = None
    fold_results: list[dict[str, Any]] | None = None
    validation_retrieval_backend: str | None = None
    test_retrieval_backend: str | None = None
    test_pacmap_coordinates: Path | None = None
    test_pacmap_plot: Path | None = None
    test_tsne_coordinates: Path | None = None
    test_tsne_plot: Path | None = None
    # Saved per-fold D_test embeddings, and the two ways the folds' test scores
    # combine: the sample standard deviation of the per-fold scores, and one
    # evaluation of every fold's embeddings concatenated per test sample.
    test_embeddings_path: Path | None = None
    # The same embeddings held in this process instead of on disk, under
    # fold_test_embedding_storage='memory'. Process-local by nature: it is not
    # serialized, so a result read back from JSON never carries it.
    test_embedding_set: dict[str, Any] | None = None
    # The fitted projection head, saved only when --save_head_state asks for it.
    head_state_path: Path | None = None
    # What the supervised warm-up reached on its own, captured at the
    # ``warmup_epochs`` boundary before any selection restart discards it. This
    # is the run's own supervised baseline under identical hyperparameters, so
    # ``best_valid_* - warmup_best_valid_*`` is what the SSL phase actually
    # bought. ``None`` for runs without a warm-up phase.
    warmup_best_valid_precision_at_1: float | None = None
    warmup_best_valid_mean_average_precision_at_r: float | None = None
    warmup_selected_epoch: int | None = None
    warmup_checkpoint_mode: str | None = None
    # The epoch whose weights the SSL phase actually started from: equal to
    # ``warmup_selected_epoch`` under ``restore_best``, and ``None`` under
    # ``keep_last``, where the SSL phase just continued from the last warm-up
    # epoch.
    warmup_restored_epoch: int | None = None
    test_precision_at_1_std: float | None = None
    test_mean_average_precision_at_r_std: float | None = None
    concatenated_test_precision_at_1: float | None = None
    concatenated_test_mean_average_precision_at_r: float | None = None
    concatenated_test_embedding_dim: int | None = None
    # Optional Recall@K, keyed by K. Empty unless --recall_at_k asked for it, so
    # one dict per evaluation keeps the field count independent of how many K a
    # run requests.
    best_valid_recall_at_k: dict[int, float] | None = None
    test_recall_at_k: dict[int, float] | None = None
    epoch0_test_recall_at_k: dict[int, float] | None = None
    concatenated_test_recall_at_k: dict[int, float] | None = None
    # The configured headline list, including optional NMI and R-precision.
    # Legacy P@1/MAP@R fields above remain explicit for selection and backwards
    # compatibility; these mappings preserve the CLI-selected measurement set.
    best_valid_measurements: dict[str, float] | None = None
    test_measurements: dict[str, float] | None = None
    epoch0_test_measurements: dict[str, float] | None = None
    test_measurements_std: dict[str, float] | None = None
    concatenated_test_measurements: dict[str, float] | None = None

@dataclass(frozen=True)
class HParamSearchConfig:
    """Optuna study settings loaded from a JSON configuration."""

    enabled: bool = True
    n_trials: int = 20
    timeout: int | None = None
    n_jobs: int = 1
    direction: str = "maximize"
    metric: str = "best_valid_mean_average_precision_at_r"
    study_name: str | None = None
    study_dir: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    sampler: str = "tpe"
    tpe_startup_trials: int | None = None
    sampler_params: dict[str, Any] = field(default_factory=dict)
    pruner: str = "none"
    pruner_params: dict[str, Any] = field(default_factory=dict)
    retry_failed_trials: bool = False
    spaces: dict[str, Any] = field(default_factory=dict)
    # Per-loss-class overrides applied once the trial's loss is fixed. Keys that
    # are not class-qualified, such as lr and batch_sampler, cannot be scoped by
    # loss the way loss.<ClassName>.<param> spaces are, so they live here.
    per_loss: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class HParamStudyResult:
    """summary of a completed or resumed Optuna study."""

    study_name: str
    study_dir: Path
    trials_csv: Path
    trials_jsonl: Path
    best_trial_number: int | None
    best_value: float | None
    best_params: dict[str, Any] | None
    best_user_attrs: dict[str, Any] | None
    completed_trials: list[dict[str, Any]] | None = None

@dataclass(frozen=True)
class ComparisonScenario:
    """One point in the outer label-budget/loss/miner/seed experiment grid."""

    name: str
    labeled_fraction: float
    labeled_per_class: int | None
    seed: int
    label_sampling_mode: str
    loss: str
    miner: str
    ssl_config_path: Path
    run_seed: int | None = None
    data_split_seed: int | None = None
    support_seed: int | None = None
    hparam_seed: int | None = None
    comparison_seed_targets: tuple[str, ...] = ()
