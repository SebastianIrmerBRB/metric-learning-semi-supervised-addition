"""Shared constants and immutable result types for experiment execution."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATASETS = ["Cars196", "CUB", "DeepFashionInShop", "CIFAR100"]

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

LOSS_HPARAM_PREFIX = "loss."

MINER_HPARAM_PREFIX = "miner."

JOINT_COMPONENT_HPARAM_PREFIX = "__joint_component__."

HPO_MODE_KEYS = {"backbone_tuning", "use_cache"}

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
    test_pacmap_coordinates: Path | None = None
    test_pacmap_plot: Path | None = None

@dataclass(frozen=True)
class HParamSearchConfig:
    """Optuna study settings loaded from a JSON configuration."""

    enabled: bool = True
    n_trials: int = 20
    timeout: int | None = None
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
    spaces: dict[str, Any] = field(default_factory=dict)

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
