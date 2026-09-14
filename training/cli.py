"""Command-line parser and top-level experiment configuration loading."""

import argparse
import json
import os
import sys
from pathlib import Path

import utils
from models.retrieval_model import (
    BACKBONE_TUNING_FULL,
    BACKBONE_TUNING_FROZEN,
    DEFAULT_PROJECTION_LAYERS,
    normalize_backbone_tuning,
)
from tensorboard import default
from utils.dataset_constants import SAME_SOURCE_EVALUATION, VAL_MODE_MATCH_TRAIN

from . import cpu_threads, device_thresholds, semi_supervised
from .ssl import step_visualization
from .types import (
    ALL_LOSSES,
    ALL_MINERS,
    WARMUP_LOSS_SAME_AS_LOSS,
    DATASETS,
    SELECTION_METRIC_MAP_AT_R,
    SELECTION_METRICS,
)

DEFAULT_DATA_SPLIT_SEED = 7
DEFAULT_SUPPORT_SEED = semi_supervised.DEFAULT_SUPPORT_SEED
FINAL_TEST_VISUALIZATION_NONE = "none"
FINAL_TEST_VISUALIZATION_PACMAP = "pacmap"
FINAL_TEST_VISUALIZATION_TSNE = "tsne"
FINAL_TEST_VISUALIZATION_MODES = (
    FINAL_TEST_VISUALIZATION_NONE,
    FINAL_TEST_VISUALIZATION_PACMAP,
    FINAL_TEST_VISUALIZATION_TSNE,
)
FOLD_TEST_EMBEDDING_STORAGE_FILE = "file"
FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY = "temporary"
FOLD_TEST_EMBEDDING_STORAGE_MEMORY = "memory"
FOLD_TEST_EMBEDDING_STORAGES = (
    FOLD_TEST_EMBEDDING_STORAGE_FILE,
    FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY,
    FOLD_TEST_EMBEDDING_STORAGE_MEMORY,
)


# What each storage mode means for the concatenated fold evaluation, phrased for
# the line that announces a per-fold final run.
FOLD_TEST_EMBEDDING_LOG_CLAUSES = {
    FOLD_TEST_EMBEDDING_STORAGE_FILE: (
        "; plus a concatenated-embedding evaluation from each fold's saved test_embeddings.npz"
    ),
    FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY: (
        "; plus a concatenated-embedding evaluation, after which the folds' "
        "test_embeddings.npz files are deleted"
    ),
    FOLD_TEST_EMBEDDING_STORAGE_MEMORY: (
        "; plus a concatenated-embedding evaluation from fold embeddings held in memory, "
        "writing no test_embeddings.npz"
    ),
}


def get_fold_test_embedding_storage(args):
    """Return how a cross-validation run holds its folds' test embeddings.

    Runs recorded before this option existed wrote the files and kept them, so
    a missing value means 'file'.
    """

    storage = getattr(args, "fold_test_embedding_storage", None)
    if storage is None:
        return FOLD_TEST_EMBEDDING_STORAGE_FILE
    if storage not in FOLD_TEST_EMBEDDING_STORAGES:
        raise ValueError(
            f"fold_test_embedding_storage must be one of {FOLD_TEST_EMBEDDING_STORAGES}: {storage}"
        )
    return storage


def normalize_final_test_visualization(value):
    """Return the requested final-test visualizations as an ordered tuple.

    Accepts a bare string because every study saved before this option took
    several modes recorded one, and 'none' stays absorbing: it means no
    visualization, so combining it with a real mode is a contradiction rather
    than a request for both.
    """

    if value is None:
        return ()
    modes = (value,) if isinstance(value, str) else tuple(value)
    unknown = [mode for mode in modes if mode not in FINAL_TEST_VISUALIZATION_MODES]
    if unknown:
        raise ValueError(
            f"final_test_visualization must be one or more of {FINAL_TEST_VISUALIZATION_MODES}: "
            f"{unknown}"
        )
    if FINAL_TEST_VISUALIZATION_NONE in modes:
        if len(set(modes)) > 1:
            raise ValueError(
                f"final_test_visualization={list(modes)} combines "
                f"{FINAL_TEST_VISUALIZATION_NONE!r} with a visualization; request one or the other"
            )
        return ()
    # dict.fromkeys keeps the requested order while dropping repeats.
    return tuple(dict.fromkeys(modes))
RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV = (
    device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV
)
RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV = (
    device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV
)
STUDY_DIR_MODE_FINAL_TRAIN = "final_train"
STUDY_DIR_MODE_TRAIN_VAL = "train_val"
STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL = "cross_seed_train_val"
STUDY_DIR_MODES = (
    STUDY_DIR_MODE_FINAL_TRAIN,
    STUDY_DIR_MODE_TRAIN_VAL,
    STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL,
)
FINAL_EPOCH_AGGREGATION_MEAN = "mean"
FINAL_EPOCH_AGGREGATION_MEDIAN = "median"
FINAL_EPOCH_AGGREGATION_MAX = "max"
FINAL_EPOCH_AGGREGATIONS = (
    FINAL_EPOCH_AGGREGATION_MEAN,
    FINAL_EPOCH_AGGREGATION_MEDIAN,
    FINAL_EPOCH_AGGREGATION_MAX,
)
# How the model that is finally measured on D_test is produced. full_train uses
# every development sample for a duration transferred from validation;
# early_stop_holdout keeps a validation slice out of that fit and lets early
# stopping choose the duration, so no epoch count has to be transferred;
# cross_validation_folds keeps no separate final fit at all and tests each
# fold's own validation-selected checkpoint, the protocol from "A Metric
# Learning Reality Check" (Musgrave et al., ECCV 2020).
FINAL_FIT_MODE_FULL_TRAIN = "full_train"
FINAL_FIT_MODE_EARLY_STOP_HOLDOUT = "early_stop_holdout"
FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS = "cross_validation_folds"
FINAL_FIT_MODES = (
    FINAL_FIT_MODE_FULL_TRAIN,
    FINAL_FIT_MODE_EARLY_STOP_HOLDOUT,
    FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS,
)
COMPARISON_SEED_TARGET_RUNTIME = "seed"
COMPARISON_SEED_TARGET_DATA_SPLIT = "data_split_seed"
COMPARISON_SEED_TARGET_SUPPORT = "support_seed"
COMPARISON_SEED_TARGET_HPARAM = "hparam_seed"
COMPARISON_SEED_TARGETS = (
    COMPARISON_SEED_TARGET_RUNTIME,
    COMPARISON_SEED_TARGET_DATA_SPLIT,
    COMPARISON_SEED_TARGET_SUPPORT,
    COMPARISON_SEED_TARGET_HPARAM,
)
LR_SCHEDULER_NONE = "none"
LR_SCHEDULER_STEP = "step"
LR_SCHEDULER_COSINE = "cosine"
LR_SCHEDULER_COSINE_WARM_RESTARTS = "cosine_warm_restarts"
LR_SCHEDULERS = (
    LR_SCHEDULER_NONE,
    LR_SCHEDULER_STEP,
    LR_SCHEDULER_COSINE,
    LR_SCHEDULER_COSINE_WARM_RESTARTS,
)
MEASUREMENTS = utils.AVAILABLE_MEASUREMENTS
DEFAULT_MEASUREMENTS = utils.DEFAULT_MEASUREMENTS


def parse_json_object(value):
    """Accept a JSON object from CLI while preserving dict values from config files."""

    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_non_negative_int(value):
    """Parse an integer value that may be zero."""

    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("value must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_optional_interval(value):
    """Parse an interval that JSON configs may switch off with ``null``."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("value must be an integer or null")
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer or null") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or null")
    # 0 and null both mean "never", so they collapse to one disabled value.
    return parsed or None


def parse_optional_open_unit_interval(value):
    """Parse a ratio strictly inside (0, 1) that ``null`` leaves at its default."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("value must be a number in (0, 1) or null")
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a number in (0, 1) or null") from exc
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(f"value must be in (0, 1), got {parsed}")
    return parsed


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--experiment_config",
    "--experiment-config",
    type=Path,
    default="configs/experiments/class/cars196_testing_all.json",
    help=(
        "top-level JSON config containing CLI argument values. "
        "Explicit CLI arguments override values from this file."
    ),
)
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=1e-6, help="LR")
parser.add_argument("--classifier_lr", type=float, default=1.0, help="classifier LR (only for classification losses)")
parser.add_argument(
    "--weight_decay",
    "--weight-decay",
    dest="weight_decay",
    type=float,
    default=0.0,
    help=(
        "weight decay for the model optimizer and any classification-loss optimizer; "
        "0 disables weight decay"
    ),
)
parser.add_argument("--sampler_m", type=int, default=4, help="M value for MPerClassSampler")
parser.add_argument(
    "--length_before_new_iter",
    "--length-before-new-iter",
    type=int,
    default=None,
    help=(
        "legacy fixed sampler length retained for config compatibility. "
        "Training overrides it with the current fold's labeled + unlabeled pool size."
    ),
)
parser.add_argument(
    "--length_before_new_iter_override",
    "--length-before-new-iter-override",
    "--override_length_before_new_iter",
    "--override-length-before-new-iter",
    type=int,
    default=None,
    help=(
        "optional fixed sampler length that overrides the current fold's "
        "labeled + unlabeled pool size"
    ),
)
parser.add_argument("--dataset", type=utils.normalize_dataset_name, default="Cars196", choices=DATASETS, help="dataset")
parser.add_argument(
    "--cifar_imbalance_factor",
    type=float,
    default=None,
    help=(
        "CIFAR long-tail factor img_min/img_max. None keeps balanced data; "
        "for example, 0.01 produces approximately 100:1 head-to-tail imbalance."
    ),
)
parser.add_argument(
    "--cifar_train_fraction",
    "--cifar-train-fraction",
    type=float,
    default=0.5,
    help="per-class fraction assigned to development/training by dataset_protocol=cifar_balanced_fraction",
)
parser.add_argument(
    "--cifar_test_fraction",
    "--cifar-test-fraction",
    type=float,
    default=0.5,
    help="per-class fraction assigned to the final test set by dataset_protocol=cifar_balanced_fraction",
)
parser.add_argument(
    "--dataset_protocol",
    choices=utils.DATASET_PROTOCOLS,
    default="official",
    help=utils.format_dataset_protocol_help(),
)
parser.add_argument(
    "--image_resize_mode",
    "--image-resize-mode",
    choices=utils.IMAGE_RESIZE_MODES,
    default=utils.DEFAULT_IMAGE_RESIZE_MODE,
    help=utils.format_image_resize_mode_help(),
)
parser.add_argument(
    "--semi_aves_ood_fraction",
    "--semi-aves-ood-fraction",
    type=float,
    default=utils.SEMI_AVES_DEFAULT_OOD_FRACTION,
    help=(
        "class-mismatch level for dataset_protocol=semi_aves_known_100_100: the share of the "
        "original out-of-class (u_train_out) pool added to the unlabeled data, spread evenly "
        "over the 800 out-of-class species. The remaining out-of-class images are excluded "
        "entirely. The pool is opt-in: the default 0.0 leaves only in-class unlabeled data, "
        "and 1.0 restores the original released pool."
    ),
)
parser.add_argument(
    "--semi_aves_ood_seed",
    "--semi-aves-ood-seed",
    type=int,
    default=utils.SEMI_AVES_DEFAULT_OOD_SEED,
    help="seed selecting which out-of-class images --semi_aves_ood_fraction keeps",
)
parser.add_argument(
    "--semi_inat_ood_fraction",
    "--semi-inat-ood-fraction",
    type=float,
    default=utils.SEMI_INAT_DEFAULT_OOD_FRACTION,
    help=(
        "class-mismatch level for dataset_protocol=semi_inat_known_810: the share of the "
        "original out-of-class (u_train_out) pool added to the unlabeled data, spread evenly "
        "over the 1,629 out-of-class species. The remaining out-of-class images are excluded "
        "entirely. The pool is opt-in: the default 0.0 leaves only in-class unlabeled data, "
        "and 1.0 restores the original released pool."
    ),
)
parser.add_argument(
    "--semi_inat_ood_seed",
    "--semi-inat-ood-seed",
    type=int,
    default=utils.SEMI_INAT_DEFAULT_OOD_SEED,
    help="seed selecting which out-of-class images --semi_inat_ood_fraction keeps",
)
parser.add_argument("--dino_size", type=str, default="b", choices=["s", "b", "l", "g"], help="which Dino to use")
parser.add_argument(
    "--backbone-tuning",
    "--backbone_tuning",
    dest="backbone_tuning",
    type=normalize_backbone_tuning,
    default=BACKBONE_TUNING_FROZEN,
    metavar="POLICY",
    help="DINO tuning policy: full, frozen, or last_N_blocks (for example last_2_blocks)",
)

parser.add_argument(
    "--use-cache",
    "--use_cache",
    dest="use_cache",
    action="store_true",
    default=True,
    help=(
        "use deterministic frozen-backbone features backed by one source-indexed "
        "memory-mapped matrix per dataset/backbone"
    ),
)
parser.add_argument(
    "--frozen-head-only-checkpoint",
    "--frozen_head_only_checkpoint",
    dest="frozen_head_only_checkpoint",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "with backbone_tuning=frozen, keep the selected non-backbone model state in CPU memory "
        "instead of writing the full DINO model to a temporary checkpoint"
    ),
)
parser.add_argument("--loss", type=str, choices=ALL_LOSSES, help="loss")
parser.add_argument("--miner", type=str, choices=ALL_MINERS, help="miner")
parser.add_argument(
    "--loss_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to the selected loss",
)
parser.add_argument(
    "--miner_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to the selected miner",
)
parser.add_argument(
    "--warmup_loss",
    type=str,
    default=WARMUP_LOSS_SAME_AS_LOSS,
    choices=[*ALL_LOSSES, WARMUP_LOSS_SAME_AS_LOSS],
    help=(
        "supervised loss used during labeled-only SSL warmup epochs. The default "
        f"{WARMUP_LOSS_SAME_AS_LOSS!r} warms up with --loss and its parameters, "
        "miner and miner parameters, so the warm-up trains the objective the run "
        "is actually about. Name a loss to warm up with something else"
    ),
)
parser.add_argument(
    "--warmup_miner",
    type=str,
    default="no_miner",
    choices=ALL_MINERS,
    help="miner used with warmup_loss",
)
parser.add_argument(
    "--warmup_loss_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to warmup_loss",
)
parser.add_argument(
    "--warmup_miner_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to warmup_miner",
)
parser.add_argument("--feat_dim", type=int, default=768, help="Output dimensionality. Set to None to use CLS")
parser.add_argument(
    "--projection_layers",
    "--projection-layers",
    dest="projection_layers",
    type=int,
    default=DEFAULT_PROJECTION_LAYERS,
    help=(
        "linear layers in the trainable projection head over the backbone. 1 keeps the single "
        "linear map; N>1 stacks N-1 ReLU-separated hidden layers before the feat_dim output "
        "and requires a non-null feat_dim"
    ),
)
parser.add_argument(
    "--projection_hidden_dim",
    "--projection-hidden-dim",
    dest="projection_hidden_dim",
    type=int,
    default=None,
    help=(
        "width of the projection head's hidden layers when projection_layers > 1. "
        "None uses the DINO backbone dimension."
    ),
)
parser.add_argument(
    "--stml_g_dim",
    type=int,
    default=None,
    help="STML background-head dimension. None uses the DINO backbone dimension.",
)
parser.add_argument(
    "--device",
    type=utils.normalize_device_name,
    default="cuda",
    metavar="DEVICE",
    help="device: cpu, cuda, or an indexed CUDA device such as cuda:2",
)
parser.add_argument(
    "--ssl_device",
    "--ssl-device",
    type=utils.normalize_device_name,
    default=None,
    metavar="DEVICE",
    help=(
        "device used for SSL embedding extraction, GPU graph construction, and "
        "CUDA-backed solvers; defaults to --device"
    ),
)
parser.add_argument("--optim", type=str, default="adam", choices=["adamw", "adam", "rmsprop"], help="optimizer")
parser.add_argument(
    "--lr_scheduler",
    "--lr-scheduler",
    choices=LR_SCHEDULERS,
    default=LR_SCHEDULER_NONE,
    help=(
        "learning-rate scheduler. 'none' keeps the learning rate fixed; 'step' and "
        "'cosine' advance per epoch, while 'cosine_warm_restarts' advances per batch"
    ),
)
parser.add_argument(
    "--lr_scheduler_params",
    "--lr-scheduler-params",
    type=parse_json_object,
    default={},
    help=(
        "JSON object of scheduler parameters. Step defaults: step_size=10, gamma=0.1. "
        "Cosine defaults: T_max=epochs, eta_min=0. Warm-restart defaults: "
        "T_0=10, T_mult=1, eta_min=0"
    ),
)
parser.add_argument("--seed", type=int, default=7, help="random seed for training/runtime randomness")
parser.add_argument(
    "--hparam_seed",
    "--hparam-seed",
    "--hpo_seed",
    "--hpo-seed",
    dest="hparam_seed",
    type=int,
    default=None,
    help="seed for Optuna HPO samplers. Defaults to --seed.",
)
parser.add_argument(
    "--tpe_startup_trials",
    "--tpe-startup-trials",
    type=parse_non_negative_int,
    default=None,
    help=(
        "number of random startup trials for Optuna's TPE sampler. "
        "None uses the HPO config or Optuna default."
    ),
)
parser.add_argument(
    "--data_split_seed",
    type=int,
    default=None,
    help=(
        "seed for dataset protocol, train/validation, and validation downsampling splits. "
        f"Defaults to {DEFAULT_DATA_SPLIT_SEED} and is independent of --seed."
    ),
)
parser.add_argument(
    "--support_seed",
    "--support-seed",
    type=int,
    default=None,
    help=(
        "seed for labeled support/sample selection. "
        f"Defaults to {DEFAULT_SUPPORT_SEED} and is independent of --seed."
    ),
)
parser.add_argument("--epochs", type=int, default=1, help="maximum number of training epochs")
parser.add_argument("--patience", type=int, default=3, help="early-stopping patience after SSL warmup")
parser.add_argument("--cv_k", type=int, default=4, help="number of cross-validation folds. Set to 1 to disable CV")
parser.add_argument(
    "--cv_mode",
    type=str,
    choices=utils.CV_MODES,
    default="kfold",
    help=(
        "cross-validation splitter to use when cv_k > 1. "
        "Use superclass_group_kfold with cifar100_fc100 to hold out complete superclasses"
    ),
)
parser.add_argument(
    "--val_mode",
    type=str,
    choices=utils.VAL_MODES,
    default=VAL_MODE_MATCH_TRAIN,
    help=(
        "validation data mode. 'all' keeps the current behavior and uses all validation samples; "
        "'match_train' downsamples validation to roughly the labeled/fractioned training size; "
        "'split_after_apportion' creates a standard train/validation split after the training data is apportioned."
    ),
)
parser.add_argument(
    "--validation_retrieval_mode",
    type=str,
    choices=utils.VALIDATION_RETRIEVAL_MODES,
    default=SAME_SOURCE_EVALUATION,
    help=(
        "how validation retrieval is scored. 'same_source' keeps the current behavior and uses "
        "every validation embedding as both query and reference; 'query_gallery' derives a "
        "per-class query/gallery partition of the validation split so validation measures the "
        "same protocol as a query/gallery test set such as DeepFashion In-shop. Validation "
        "numbers from the two modes are not comparable, so a study must not mix them."
    ),
)
parser.add_argument(
    "--validation_gallery_fraction",
    type=float,
    default=utils.DEFAULT_VALIDATION_GALLERY_FRACTION,
    help=(
        "fraction of each validation class placed in the gallery when "
        "--validation_retrieval_mode=query_gallery; the rest become queries. Every class keeps "
        "at least one sample on each side"
    ),
)
parser.add_argument(
    "--selection_metric",
    default=SELECTION_METRIC_MAP_AT_R,
    choices=SELECTION_METRICS,
    help="validation metric used for checkpoint selection and early stopping",
)
parser.add_argument(
    "--measurements",
    type=utils.normalize_measurement_name,
    nargs="+",
    choices=MEASUREMENTS,
    default=list(DEFAULT_MEASUREMENTS),
    metavar="MEASUREMENT",
    help=(
        "evaluation measurements to report as a list. The default preserves "
        "precision_at_1 and mean_average_precision_at_r; optionally add NMI and "
        "r_precision (RP). P@1, MAP@R, NMI, and RP spellings are accepted. "
        "Checkpoint selection and early stopping still use --selection_metric"
    ),
)
parser.add_argument(
    "--recall_at_k",
    "--recall-at-k",
    type=int,
    nargs="*",
    default=[],
    metavar="K",
    help=(
        "additional retrieval Recall@K metrics to report, for example 1 2 4 8. Recall@K is the "
        "share of queries with at least one correct neighbour among their K nearest, so Recall@1 "
        "equals precision_at_1, which is in the default measurement list. They are logged for validation and test, "
        "written to metrics.csv and TensorBoard, and stored in the run's results. Selection and "
        "early stopping keep using --selection_metric."
    ),
)
parser.add_argument(
    "--report_test_metrics",
    "--report-test-metrics",
    dest="report_test_metrics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "on by default: report the full retrieval table on D_test only: precision_at_1, r_precision, "
        "mean_average_precision_at_r, mean_average_precision, mean_reciprocal_rank, NMI, AMI "
        "and Recall@"
        + "/".join(str(k) for k in utils.REPORT_TEST_RECALL_AT_K)
        + ". Validation keeps reporting --measurements and --recall_at_k, so the extra "
        "metrics cost one evaluation per run instead of one per epoch. The numbers land in "
        "metrics.csv, cv_results.csv, cv_summary.json and the run's result JSON, for the "
        "per-fold and the concatenated-fold evaluation alike. Pass --no-report_test_metrics to "
        "report only what --measurements and --recall_at_k ask for; --test_measurements and "
        "--test_recall_at_k replace the preset's two lists outright"
    ),
)
parser.add_argument(
    "--test_measurements",
    "--test-measurements",
    dest="test_measurements",
    type=utils.normalize_measurement_name,
    nargs="+",
    choices=MEASUREMENTS,
    default=None,   
    metavar="MEASUREMENT",
    help=(
        "measurements to report on D_test, if they should differ from --measurements. "
        "Unset means the test evaluation reports the same list validation does"
    ),
)
parser.add_argument(
    "--test_recall_at_k",
    "--test-recall-at-k",
    dest="test_recall_at_k",
    type=int,
    nargs="*",
    default=None,
    metavar="K",
    help=(
        "Recall@K values to report on D_test, if they should differ from --recall_at_k, "
        "for example 1 10 20 40 for In-Shop or 1 2 4 8 for Cars196/CUB. Unset means the test "
        "evaluation reports the same ladder validation does; passing it with no values "
        "reports no ladder on test at all"
    ),
)
parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count for training/evaluation")
parser.add_argument(
    "--num_threads",
    type=str,
    default=cpu_threads.DEFAULT_THREAD_BUDGET,
    metavar="N",
    help=(
        "thread budget for CPU compute, applied to every numeric runtime in the process "
        "(torch, numpy, scipy, scikit-learn, faiss) rather than to torch alone. 'auto' uses "
        "the host's physical core count; pass an integer to pin it. Oversubscribing past "
        "physical cores is flat at best and a large regression at worst. "
        "METRIC_LEARNING_NUM_THREADS caps this budget; it does not raise it."
    ),
)
parser.add_argument(
    "--eval_amp",
    action="store_true",
    default=False,
    help=(
        "run evaluation/embedding passes under bfloat16 autocast. Off by default because it "
        "changes stored embeddings and therefore retrieval metrics; A/B it against existing "
        "results before adopting it."
    ),
)
parser.add_argument(
    "--train_amp",
    action="store_true",
    default=False,
    help=(
        "run the training step under bfloat16 autocast. Off by default: with a frozen backbone "
        "the step is one projection matmul plus the loss, so BF16 buys no measurable speed, "
        "while its op list differs between CPU and CUDA and so makes trials that land on "
        "different devices incomparable."
    ),
)
parser.add_argument(
    "--dataloader_start_method",
    type=str,
    default="spawn",
    choices=utils.DATALOADER_START_METHODS,
    help="DataLoader multiprocessing start method for CPU runs or CUDA runs with zero workers.",
)
parser.add_argument(
    "--debug_batch_timing",
    "--debug-batch-timing",
    action="store_true",
    default=False,
    help="log detailed per-batch timing; this synchronizes CUDA and should stay off for benchmarks",
)
parser.add_argument(
    "--debug_batch_timing_interval",
    "--debug-batch-timing-interval",
    type=int,
    default=5,
    help="number of batches between debug timing log lines when --debug_batch_timing is enabled",
)
parser.add_argument(
    "--batch_loss_log_interval",
    "--batch-loss-log-interval",
    type=parse_non_negative_int,
    default=0,
    help=(
        "log batch loss every N optimizer steps and at each epoch's final batch; "
        "0 disables batch-loss logging and its per-batch device synchronization"
    ),
)
parser.add_argument(
    "--log_batch_diagnostics",
    "--log-batch-diagnostics",
    dest="log_batch_diagnostics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "log per-batch diagnostic metrics such as gradient norms and miner counts; "
        "this adds overhead and should stay off for throughput runs"
    ),
)
parser.add_argument(
    "--debug_log",
    "--debug-log",
    dest="debug_log",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "also write a DEBUG-level debug.log next to info.log in the run directory. "
        "Off by default because no tooling reads it and the per-batch records embed "
        "full index lists, which reaches hundreds of megabytes on a single SSL fold"
    ),
)
parser.add_argument(
    "--log_epoch_diagnostics",
    "--log-epoch-diagnostics",
    dest="log_epoch_diagnostics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "write the always-on epoch and evaluation diagnostics (timing, embedding norms, "
        "dataset shape, per-class retrieval breakdowns) to diagnostics.csv and TensorBoard. "
        "Off by default: these dominate diagnostics.csv and are diagnostic-only, so HPO "
        "trials pay for them without reading them. Headline metrics still go to metrics.csv"
    ),
)
parser.add_argument(
    "--tensorboard",
    "--no-tensorboard",
    dest="tensorboard",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "write TensorBoard event files. On by default because react_trial_visualizer reads "
        "them; --no-tensorboard skips the SummaryWriter entirely, which is the single "
        "largest artifact class across a study"
    ),
)
parser.add_argument(
    "--pseudo_label_diagnostics_mode",
    "--pseudo-label-diagnostics-mode",
    dest="pseudo_label_diagnostics_mode",
    type=str,
    default=None,
    choices=sorted(semi_supervised.PSEUDO_LABEL_DIAGNOSTICS_MODES),
    help=(
        "override the SSL config's pseudo_label_diagnostics_mode. 'off' skips the hidden-label "
        "audit entirely (no confidence AUC, no per-class tallies), 'log' computes and logs the "
        "summary without writing pseudo_label_diagnostics.jsonl, 'save' also writes the file. "
        "Unset keeps whatever the SSL config asks for"
    ),
)
parser.add_argument(
    "--ssl_gradient_contribution_log_interval",
    "--ssl-gradient-contribution-log-interval",
    type=parse_non_negative_int,
    default=0,
    metavar="BATCHES",
    help=(
        "measure the weighted supervised and SSL-regularizer gradient contributions "
        "on the first active regularization batch and every N active batches thereafter; "
        "0 disables this opt-in diagnostic"
    ),
)
parser.add_argument(
    "--visualization_interval",
    "--visualization-interval",
    type=parse_optional_interval,
    default=None,
    metavar="STEPS",
    help=(
        "draw the SSL loss debugger's per-step artifacts during a normal run, on the "
        "first active regularization step and every N active steps thereafter. "
        "null (or 0) draws none; 5 draws every fifth step. Only regularizers the "
        "debugger can draw a single step for produce artifacts: "
        f"{list(step_visualization.VISUALIZED_REGULARIZERS)}"
    ),
)
parser.add_argument(
    "--visualization_dir",
    "--visualization-dir",
    type=Path,
    default=None,
    help=(
        "root directory for --visualization_interval artifacts. Each run gets its own "
        "timestamped subdirectory here, with one directory per cross-validation fold "
        "inside it, so reruns and folds never overwrite each other. Defaults to "
        "<log_dir>/step_visualizations inside the run's own directory"
    ),
)
parser.add_argument(
    "--visualization_layout",
    "--visualization-layout",
    choices=step_visualization.VISUALIZATION_LAYOUTS,
    default=step_visualization.DEFAULT_VISUALIZATION_LAYOUT,
    help="2-D projection used by --visualization_interval plots; nonlinear layouts fall back to PCA on failure",
)
parser.add_argument(
    "--visualization_seraph_trace_pairs",
    "--visualization-seraph-trace-pairs",
    type=parse_non_negative_int,
    default=step_visualization.DEFAULT_SERAPH_TRACE_PAIRS,
    help=(
        "pairs per SERAPH block listed in the comparison CSV and drawn on the contact "
        "sheets, spread evenly across the block's distance range. The same count is "
        "listed again for the pairs that share a true label, on their own sheet"
    ),
)
parser.add_argument(
    "--frozen_feature_batch_size",
    type=int,
    default=None,
    help="batch size for one-time frozen-backbone feature extraction; defaults to --batch_size",
)
parser.add_argument(
    "--frozen_feature_residency",
    "--frozen-feature-residency",
    type=str,
    default=utils.FEATURE_RESIDENCY_MMAP,
    choices=utils.FEATURE_RESIDENCIES,
    help=(
        "where precomputed frozen features are read from during training and evaluation; "
        "'mmap' reads rows from the shared on-disk matrix, 'ram' copies the rows each view "
        "uses into process memory once so no step touches the file. Use 'ram' when the "
        "feature matrix is too large to stay in the page cache. 'gpu' additionally holds the "
        "training view in device memory and drops the DataLoader, so a step gathers its "
        "batch on-device with no host transfer; it falls back to 'ram' behavior whenever "
        "the view does not fit the device budget"
    ),
)
parser.add_argument(
    "--evaluation_embedding_residency",
    "--evaluation-embedding-residency",
    type=str,
    default=utils.EVALUATION_EMBEDDING_RESIDENCY_CPU,
    choices=utils.EVALUATION_EMBEDDING_RESIDENCIES,
    help=(
        "where projected validation/test embeddings live while retrieval metrics "
        "are computed. 'cpu' is the low-memory mode: retrieval still uses "
        "FAISS-GPU when available, but projected embeddings stay in host memory, "
        "FAISS reserves no persistent scratch arena, and its GPU index is released "
        "after each evaluation. 'gpu' retains the embeddings and FAISS index for "
        "maximum evaluation speed at a higher per-process GPU-memory cost"
    ),
)
parser.add_argument(
    "--low_memory_mode",
    "--low-memory-mode",
    dest="low_memory_mode",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "cap the per-process buffers that FAISS and the frozen-feature cache "
        "hold for their whole lifetime, without moving anything off the GPU. "
        "FAISS GPU resources otherwise reserve a ~1.5 GB device scratch arena "
        "and a 256 MB pinned host buffer per process, and the in-memory "
        "frozen-feature cache otherwise keeps every view a study materializes. "
        "Retrieval metrics are unchanged. Pass --no-low_memory_mode to restore "
        "the unbounded buffers; see --faiss_temp_memory_mb and "
        "--frozen_feature_cache_max_gb to tune the caps"
    ),
)
parser.add_argument(
    "--faiss_temp_memory_mb",
    "--faiss-temp-memory-mb",
    type=float,
    default=256,
    help=(
        "device scratch left to FAISS under --low_memory_mode when evaluation "
        "embeddings are GPU-resident; defaults to 256. 0 makes FAISS allocate "
        "search scratch on demand. Ignored unless --low_memory_mode is set"
    ),
)
parser.add_argument(
    "--frozen_feature_cache_max_gb",
    "--frozen-feature-cache-max-gb",
    type=float,
    default=8096,
    help=(
        "total budget for the in-memory frozen-feature views shared across the "
        "trials of one study; least-recently-used views are evicted above it. "
        "Defaults to unbounded, or to 8 GB under --low_memory_mode"
    ),
)
parser.add_argument(
    "--frozen_feature_residency_max_gb",
    "--frozen-feature-residency-max-gb",
    type=float,
    default=2.0,
    help=(
        "per-view memory budget for --frozen_feature_residency ram and gpu; a view needing "
        "more than this stays memory-mapped instead of exhausting RAM across concurrent runs"
    ),
)
parser.add_argument(
    "--frozen_feature_train_views",
    type=int,
    default=1,
    help=(
        "number of stochastic training views to precompute per sample when frozen feature precompute is active; "
        "1 preserves deterministic cached features"
    ),
)
parser.add_argument(
    "--compile_train_step",
    "--compile-train-step",
    dest="compile_train_step",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "capture a fixed-shape frozen-backbone projection+loss with torch.compile's "
        "cudagraphs backend. Unlike Inductor fusion, this replays the eager ATen kernels "
        "and is tested for bit-identical losses, gradients, and optimizer trajectories. "
        "Supported paths are ArcFace, ProxyAnchor, and "
        "MultiSimilarityLoss+MultiSimilarityMiner. The first batches "
        "pay one-time graph setup. Unsupported losses, CPU or train-AMP runs, sample "
        "weights, active regularizers, diagnostics, and non-precomputed inputs keep the "
        "eager path"
    ),
)
parser.add_argument(
    "--graph_projection",
    "--graph-projection",
    dest="graph_projection",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "capture the projection head's forward and backward as a CUDA graph with "
        "torch.cuda.make_graphed_callables, so the embeddings feeding an SSL regularizer "
        "cost one graph replay instead of a launch per kernel. Replays the head's own "
        "kernels, so embeddings and gradients are the eager ones bit for bit, and the "
        "capture is verified against an eager forward before it is used. Opt-in and "
        "narrow: CPU or train-AMP runs, image (non-precomputed) batches, and anything "
        "that runs a second backward through the head -- gradient surgery, GradNorm, "
        "target-ratio calibration, gradient-contribution diagnostics -- keep the eager "
        "path and log why"
    ),
)
parser.add_argument(
    "--ssl_config",
    type=Path,
    default=None,
    help=(
        "path to a JSON semi-supervised config, naming the SSL method and its "
        "method_params. Required by --mode ssl and unused by --mode supervised: the "
        "label budget is set by --ssl_label_sampling_modes, --label_budget_grid and "
        "--k_shot_grid, which apply to both modes"
    ),
)
parser.add_argument(
    "--unlabeled_class_scope",
    "--unlabeled-class-scope",
    type=str,
    default=None,
    choices=sorted(semi_supervised.UNLABELED_CLASS_SCOPES),
    help=(
        "which classes may contribute unlabeled candidates. all keeps every training "
        "sample outside the labeled support, including classes the support never "
        "covers; labeled_classes keeps only the support's own classes, making the pool "
        "in-distribution. Omitted leaves the SSL config's value"
    ),
)
parser.add_argument(
    "--unlabeled_fraction",
    "--unlabeled-fraction",
    type=float,
    default=None,
    help=(
        "class-balanced share in (0, 1] of the unlabeled pool to keep, applied after "
        "--unlabeled_class_scope. Successive values nest, so 0.25 is a subset of 0.5. "
        "Omitted leaves the SSL config's value"
    ),
)
parser.add_argument(
    "--max_unlabeled_samples",
    "--max-unlabeled-samples",
    type=int,
    default=None,
    help=(
        "absolute cap on unlabeled candidates, drawn from the pool as a whole after "
        "--unlabeled_fraction. Unlike that fraction it is neither class-balanced nor "
        "nested across values; prefer it only to bound cost. Omitted leaves the SSL "
        "config's value"
    ),
)
parser.add_argument(
    "--unlabeled_source",
    choices=["split", "labeled", "external", "split_and_external"],
    default="split",
    help=(
        "SSL unlabeled pool: the current train split, an external recursive image directory, "
        "or both. labeled is the loss-control ablation -- the unlabeled objective is applied "
        "to the labeled support itself with its labels hidden, so the regularizer is present "
        "but no genuinely unlabeled sample is. Comparing split against labeled separates the "
        "unlabeled objective's own regularizing effect from the information in the unlabeled data"
    ),
)
parser.add_argument(
    "--external_unlabeled_dir",
    type=Path,
    default=None,
    help="recursive external image directory used as unlabeled SSL data, such as a local Fashion200K root",
)
parser.add_argument(
    "--external_unlabeled_filter",
    type=str,
    default=utils.EXTERNAL_UNLABELED_FILTER_NONE,
    choices=utils.EXTERNAL_UNLABELED_FILTERS,
    help=(
        "optional filtering applied to external unlabeled images. "
        "compcars_model_min_count keeps CompCars model-level categories with enough images; "
        "compcars_slade_paper reproduces the SLADE CompCars subset from the official "
        "classification split and checks for 16,537 images across 145 model classes; "
        "compcars_stml_paper is the same pool under the STML name, since STML adopts SLADE's "
        "protocol; nabirds reads the official NABirds metadata and exposes all "
        "listed images as hidden-label samples for CUB."
    ),
)
parser.add_argument(
    "--compcars_min_model_images",
    type=int,
    default=100,
    help="minimum images per inferred CompCars model class for CompCars external-unlabeled filters",
)
parser.add_argument(
    "--compcars_strict_paper_counts",
    action="store_true",
    default=False,
    help=(
        "fail CompCars SLADE/STML-paper filtering unless the filtered pool has exactly "
        "16,537 images across 145 model classes"
    ),
)
parser.add_argument(
    "--compcars_paper_threshold_calibration",
    type=str,
    default=utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_AUTO,
    choices=utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_MODES,
    help=(
        "how CompCars SLADE/STML-paper filtering picks its count threshold. auto keeps "
        "--compcars_min_model_images when it reproduces the published 16,537 images across "
        "145 model classes and otherwise searches for a threshold that does; off always uses "
        "--compcars_min_model_images"
    ),
)
parser.add_argument(
    "--hparam_config",
    type=Path,
    help="path to a JSON Optuna hyperparameter search config. Omit to run a single training job.",
)
parser.add_argument(
    "--ablation_config",
    "--ablation-config",
    dest="ablation_config",
    type=Path,
    default=None,
    help=(
        "path to a JSON ablation config: named variants of sparse changes applied after every "
        "other source, including an HPO study's winning parameters, so a tuned run can be "
        "replayed with one thing changed. A variant is written like an experiment config "
        "('sampler_m', 'batch_size', 'loss_params', an 'ssl_config' object or file path) or with "
        "a search space's dotted names ('loss.CircleLoss.m', "
        "'ssl_config.method_params.regularizer_weight'); 'sweep' expands each value of one key "
        "into its own variant. See docs/ablation_config.md"
    ),
)
parser.add_argument(
    "--ablation_variant",
    "--ablation-variant",
    dest="ablation_variant",
    type=str,
    default=None,
    help=(
        "which variant of --ablation_config this run executes. May be omitted when the config "
        "defines exactly one; the run scheduler sets it for every child it expands"
    ),
)
parser.add_argument(
    "--compare_supervised_ssl",
    action="store_true",
    help=(
        "run two separate Optuna searches with identical budget: a supervised baseline on the labeled "
        "subset only and an SSL run on the same labeled subset plus unlabeled data"
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["supervised", "ssl"],
    default="supervised",
    help="training mode. supervised uses only the labeled split; ssl uses the labeled split plus unlabeled data.",
)
parser.add_argument(
    "--skip_test_during_hpo",
    action="store_true",
    default=True,
    help="do not evaluate D_test inside Optuna trials; use a final retraining run for test evaluation",
)
parser.add_argument(
    "--retry_failed_hpo_trials",
    action="store_true",
    default=False,
    help="enqueue existing failed/pruned Optuna trials as new trials and run only those retries",
)
parser.add_argument(
    "--evaluate_test",
    "--evaluate-test",
    action="store_true",
    default=False,
    help="evaluate D_test after a direct non-HPO training run",
)
parser.add_argument(
    "--final_test_after_hpo",
    action="store_true",
    default=False,
    help=(
        "after each HPO study, train the best configuration once on the development set "
        "for the duration chosen by --final_fit_mode and evaluate D_test"
    ),
)
parser.add_argument(
    "--final_fit_mode",
    "--final-fit-mode",
    dest="final_fit_mode",
    choices=FINAL_FIT_MODES,
    default=FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS,
    help=(
        "how the model measured on D_test is fitted. 'full_train' trains on every development "
        "sample for the fixed epoch count --final_epoch_aggregation transfers from validation; "
        "'early_stop_holdout' keeps a validation slice out of the final fit "
        "(size --holdout_val_ratio), early-stops on it, tests the selected checkpoint, and "
        "therefore needs no transferred epoch count at all; 'cross_validation_folds' trains no "
        "separate final model and instead tests every fold's own validation-selected "
        "checkpoint, reporting their mean/std and the concatenated-embedding score"
    ),
)
parser.add_argument(
    "--save_test_embeddings",
    "--save-test-embeddings",
    dest="save_test_embeddings",
    action="store_true",
    default=False,
    help=(
        "retain each run's D_test embeddings for the concatenated fold evaluation, by default "
        "as test_embeddings.npz in its log directory (see --fold_test_embedding_storage). "
        "final_fit_mode='cross_validation_folds' sets this itself; pass it directly to get the "
        "concatenated-embedding evaluation from a plain --cv_k N --evaluate_test run"
    ),
)
parser.add_argument(
    "--fold_test_embedding_storage",
    "--fold-test-embedding-storage",
    dest="fold_test_embedding_storage",
    choices=FOLD_TEST_EMBEDDING_STORAGES,
    default=FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY,
    help=(
        "where the per-fold D_test embeddings that --save_test_embeddings retains live. "
        "'file' writes test_embeddings.npz per fold and keeps it; 'temporary' writes it and "
        "deletes it once the concatenated fold evaluation has read it; 'memory' writes no file "
        "at all and holds the arrays in the cross-validation process. 'memory' costs "
        "cv_k * len(D_test) * feat_dim float32 values of host RAM and cannot serve a resumed "
        "run, whose earlier folds finished in a process that is gone: that run reports the "
        "per-fold mean only"
    ),
)
parser.add_argument(
    "--save_head_state",
    "--save-head-state",
    dest="save_head_state",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "write the validation-selected projection head of every run to head_state.pt in its log "
        "directory. With a frozen backbone that head is the whole fitted model, so "
        "--evaluate_saved_heads can measure it again without training it again"
    ),
)
parser.add_argument(
    "--evaluate_saved_heads",
    "--evaluate-saved-heads",
    dest="evaluate_saved_heads",
    type=Path,
    default=None,
    help=(
        "skip training entirely and evaluate D_test from the head_state.pt files under this run, "
        "cross-validation, or final directory, reporting the same per-fold mean/std and "
        "concatenated-embedding scores the original run reported"
    ),
)
parser.add_argument(
    "--concatenated_fold_test",
    "--concatenated-fold-test",
    dest="concatenated_fold_test",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "when a cross-validation run saved every fold's test embeddings, additionally evaluate "
        "D_test once on each test sample's per-fold embeddings concatenated and L2 normalized "
        "(cv_k * feat_dim dimensions). final_fit_mode='cross_validation_folds' turns the saving "
        "on for you; pass --no-concatenated_fold_test to report only the per-fold mean and keep "
        "the folds from retaining their embeddings at all"
    ),
)
parser.add_argument(
    "--final_epoch_aggregation",
    "--final-epoch-aggregation",
    dest="final_epoch_aggregation",
    choices=FINAL_EPOCH_AGGREGATIONS,
    default=FINAL_EPOCH_AGGREGATION_MEAN,
    help=(
        "how --final_fit_mode='full_train' turns the selected epoch of every cross-validation "
        "fold (or validation replay) into one training duration: 'mean' and 'median' round the "
        "aggregate half up, 'max' trains for the longest fold's duration. Ignored by "
        "--final_fit_mode='early_stop_holdout'"
    ),
)
parser.add_argument(
    "--holdout_val_ratio",
    "--holdout-val-ratio",
    dest="holdout_val_ratio",
    type=parse_optional_open_unit_interval,
    default=None,
    help=(
        "fraction of development classes reserved for the class-disjoint validation holdout "
        "when cv_k=1. Applies to ordinary holdout runs and to "
        "--final_fit_mode='early_stop_holdout'; null keeps the built-in 80/20 split"
    ),
)
parser.add_argument(
    "--final_test_top_n",
    type=int,
    default=1,
    help=(
        "number of highest-value completed HPO trials to replay; final_train tests every replay, "
        "while train_val and cross_seed_train_val fully retrain/test only the validation winner"
    ),
)
parser.add_argument(
    "--final_test_trial_numbers",
    type=int,
    nargs="*",
    default=None,
    help=(
        "specific completed HPO trial numbers to replay; final_train tests every replay, while "
        "train_val and cross_seed_train_val fully retrain/test only the validation winner"
    ),
)
parser.add_argument(
    "--final_test_study_dir",
    "--final-test-study-dir",
    type=Path,
    default=None,
    help=(
        "existing HPO study directory to load for final-test evaluation. "
        "Use with --final_test_trial_numbers or --final_test_top_n to avoid scheduling new HPO trials."
    ),
)
parser.add_argument(
    "--study_dir_mode",
    "--study-dir-mode",
    "--final_test_study_dir_mode",
    "--final-test-study-dir-mode",
    dest="study_dir_mode",
    choices=STUDY_DIR_MODES,
    default=STUDY_DIR_MODE_TRAIN_VAL,
    help=(
        "how to replay an existing HPO study: final_train trains once on the full development set "
        "and evaluates D_test; train_val ranks selected trials with the study's saved validation protocol "
        "and then fully retrains/tests the winner; cross_seed_train_val uses that same validation protocol "
        "over comparison_seeds and then fully retrains/tests the winner once"
    ),
)
parser.add_argument(
    "--final_test_visualization",
    "--final-test-visualization",
    nargs="+",
    choices=FINAL_TEST_VISUALIZATION_MODES,
    default=[FINAL_TEST_VISUALIZATION_PACMAP],
    help=(
        "optional final D_test embedding visualization artifacts to create after test "
        "evaluation; name several to get them all from the same embeddings, or 'none' alone "
        "to skip"
    ),
)
parser.add_argument(
    "--resume_interrupted_runs",
    "--resume-interrupted-runs",
    dest="resume_interrupted_runs",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "continue an interrupted study replay instead of repeating it: validation and final "
        "runs a previous attempt already finished are reused from their summaries, and an "
        "interrupted cross-validation run continues in its own cv_ directory at the first "
        "unfinished fold. Pass --no-resume_interrupted_runs to train every run again"
    ),
)
parser.add_argument(
    "--label_budget_grid",
    type=float,
    nargs="*",
    help="outer experiment grid over SSL labeled_fraction values, for example 0.01 0.05 0.10 0.25 0.50",
)
parser.add_argument(
    "--k_shot_grid",
    type=int,
    nargs="*",
    help="outer experiment grid over k-shot counts for label_sampling_mode='class_subset_k_shot', for example 1 2 5",
)
parser.add_argument(
    "--loss_miner_grid",
    type=str,
    nargs="*",
    metavar="LOSS:MINER",
    help=(
        "outer experiment grid over paired loss/miner choices, for example "
        "MultiSimilarityLoss:MultiSimilarityMiner TripletMarginLoss:TripletMarginMiner"
    ),
)
parser.add_argument(
    "--comparison_seeds",
    type=int,
    nargs="*",
    default=None,
    help=(
        "outer experiment grid over runtime, dataset-split, labeled-support, "
        "and HPO sampler seeds, for example 0 1 2 3 4"
    ),
)
parser.add_argument(
    "--comparison_seed_targets",
    "--comparison-seed-targets",
    nargs="+",
    choices=COMPARISON_SEED_TARGETS,
    default=list(COMPARISON_SEED_TARGETS),
    help=(
        "seed channels replaced by each --comparison_seeds value. Configure as a JSON array in an "
        "experiment config; omitted defaults to seed, data_split_seed, support_seed, and hparam_seed"
    ),
)
parser.add_argument(
    "--ssl_label_sampling_modes",
    type=str,
    nargs="*",
    default=["class_subset_k_shot"],
    choices=sorted(semi_supervised.LABEL_SAMPLING_MODES),
    help="outer experiment grid over labeled-sample selection modes",
)
parser.add_argument(
    "--save_dir",
    type=Path,
    default="logs",
    help="name of directory in which to save the logs, under logs/save_dir",
)
parser.add_argument(
    "--save_trial_split_data",
    "--save-trial-split-data",
    dest="save_trial_split_data",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "also write the split/ manifest inside HPO trial and trial cross-validation "
        "fold directories. Every trial of a study shares the same dataset/support "
        "seeds, so these arrays are duplicates; final, train_val, and standalone runs "
        "write split/ regardless of this flag"
    ),
)

def parse_args_with_experiment_config(argv=None):
    """Parse CLI arguments after loading optional top-level JSON defaults."""

    explicit_cli_args = collect_explicit_cli_destinations(argv)
    config_path = get_experiment_config_path(argv)
    config_values = load_experiment_config(config_path)
    namespace = argparse.Namespace(**config_values)
    args = parser.parse_args(argv, namespace=namespace)
    args.explicit_cli_args = sorted(explicit_cli_args)
    normalize_backbone_tuning_args(args)
    resolve_scheduler_batch_device(args)
    resolve_ssl_device(args)
    resolve_hparam_seed(args)
    resolve_data_split_seed(args)
    resolve_support_seed(args)
    if config_path is not None:
        args.experiment_config_resolved = config_values
    return args

def collect_explicit_cli_destinations(argv=None):
    """Return argparse destination names set directly by the current CLI."""

    raw_argv = sys.argv[1:] if argv is None else list(argv)
    explicit_dests = set()
    for token in raw_argv:
        if not token.startswith("--") or token == "--":
            continue
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is None or action.dest in {"help", "experiment_config"}:
            continue
        explicit_dests.add(action.dest)
    return explicit_dests


def resolve_scheduler_batch_device(args, ssl_config=None):
    """Apply the matching loss/SSL batch threshold for an assigned GPU."""

    raw_threshold = os.environ.get(RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV)
    if raw_threshold is None:
        return args
    try:
        threshold = int(raw_threshold)
    except ValueError as exc:
        raise ValueError(
            f"{RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV} must be an integer"
        ) from exc
    if threshold < 0:
        raise ValueError(
            f"{RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV} must be non-negative"
        )

    try:
        threshold_rules = device_thresholds.decode_gpu_batch_size_threshold_rules(
            os.environ.get(RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV)
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    threshold, matched_rule = device_thresholds.resolve_gpu_batch_size_threshold(
        default_threshold=threshold,
        rules=threshold_rules,
        loss=getattr(args, "loss", None),
        ssl_config=ssl_config,
    )
    args.gpu_batch_size_threshold = threshold
    args.gpu_batch_size_threshold_rule = (
        None if matched_rule is None else matched_rule.to_spec()
    )
    args.gpu_batch_size_threshold_rules = [
        rule.to_spec() for rule in threshold_rules
    ]
    (
        effective_batch_size,
        effective_batch_size_components,
    ) = device_thresholds.gpu_batch_size_threshold_input(
        getattr(args, "batch_size", None),
        ssl_config,
    )
    args.gpu_batch_size_threshold_effective_batch_size = effective_batch_size
    args.gpu_batch_size_threshold_batch_components = (
        effective_batch_size_components
    )

    scheduled_device = utils.normalize_device_name(
        os.environ.get("RUN_SCHEDULER_DEVICE", "cpu")
    )
    scheduled_ssl_device = utils.normalize_device_name(
        os.environ.get("RUN_SCHEDULER_SSL_DEVICE", scheduled_device)
    )
    if effective_batch_size > threshold:
        # The scheduler exposes one assigned physical GPU as logical cuda:0.
        args.device = "cuda"
        args.ssl_device = "cuda"
    else:
        # HPO reuses one argparse namespace as its base. Resetting both values
        # here lets a smaller trial follow a preceding GPU-sized trial safely.
        args.device = scheduled_device
        args.ssl_device = scheduled_ssl_device
    return args

def get_ssl_device(args):
    """Return the device reserved for out-of-batch SSL computation."""

    ssl_device = getattr(args, "ssl_device", None)
    if ssl_device is None:
        ssl_device = getattr(args, "device", "cpu")
    return utils.normalize_device_name(ssl_device)

def resolve_ssl_device(args):
    """Default SSL computation to the training device unless separated."""

    args.ssl_device = get_ssl_device(args)
    return args

def get_hparam_seed(args):
    """Return the seed used by Optuna's stochastic samplers."""

    hparam_seed = getattr(args, "hparam_seed", None)
    return int(getattr(args, "seed", 7)) if hparam_seed is None else int(hparam_seed)

def resolve_hparam_seed(args):
    """Default HPO sampling to the runtime seed unless explicitly separated."""

    args.hparam_seed = get_hparam_seed(args)
    return args

def resolve_data_split_seed(args):
    """Keep the validation/test split seed independent from the run seed."""

    if getattr(args, "data_split_seed", None) is None:
        args.data_split_seed = DEFAULT_DATA_SPLIT_SEED
    return args

def get_support_seed(args):
    """Return the fixed seed used for labeled support selection."""

    support_seed = getattr(args, "support_seed", None)
    return DEFAULT_SUPPORT_SEED if support_seed is None else int(support_seed)

def resolve_support_seed(args):
    """Keep labeled support selection independent from the run seed."""

    args.support_seed = get_support_seed(args)
    return args

def normalize_backbone_tuning_args(args):
    """Normalize the selected backbone fine-tuning policy."""

    args.backbone_tuning = normalize_backbone_tuning(
        getattr(args, "backbone_tuning", BACKBONE_TUNING_FULL)
    )
    return args

def get_experiment_config_path(argv=None):
    """Read only --experiment-config without validating the remaining CLI."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--experiment_config",
        "--experiment-config",
        type=Path,
    )
    config_args, _ = config_parser.parse_known_args(argv)
    return config_args.experiment_config

def load_experiment_config(config_path):
    """Load and validate JSON values that become argparse defaults."""

    if config_path is None:
        return {}

    path = Path(config_path)
    with path.open() as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {path}")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "experiment_config"}
    }
    unknown_keys = sorted(set(raw_config) - set(actions))
    if unknown_keys:
        raise ValueError(
            f"Unknown experiment config keys in {path}: {unknown_keys}. "
            "Use argparse destination names such as save_dir, ssl_label_sampling_modes, and loss_miner_grid."
        )

    return {
        name: normalize_experiment_config_value(actions[name], value, path)
        for name, value in raw_config.items()
    }

def normalize_experiment_config_value(action, value, path):
    """Apply an argparse action's type and choices to one JSON config value."""

    source = f" for {action.dest!r} in {path}"
    if value is None:
        return None
    if action.nargs == 0:
        if not isinstance(value, bool):
            raise ValueError(f"Experiment config value{source} must be true or false")
        return value

    expects_list = action.nargs in {"*", "+"} or isinstance(action.nargs, int)
    if expects_list:
        if not isinstance(value, list):
            raise ValueError(f"Experiment config value{source} must be a JSON array or null")
        if action.nargs == "+" and not value:
            raise ValueError(f"Experiment config value{source} must not be empty")
        values = [convert_experiment_config_scalar(action, item, source) for item in value]
        validate_experiment_config_choices(action, values, source)
        return values

    if isinstance(value, list):
        raise ValueError(f"Experiment config value{source} must be a scalar or null")
    if isinstance(value, dict) and action.type is None:
        raise ValueError(f"Experiment config value{source} must be a scalar or null")
    value = convert_experiment_config_scalar(action, value, source)
    validate_experiment_config_choices(action, [value], source)
    return value

def convert_experiment_config_scalar(action, value, source):
    if action.type is None:
        return value
    try:
        return action.type(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"Invalid experiment config value{source}: {value!r}") from exc

def validate_experiment_config_choices(action, values, source):
    if action.choices is None:
        return
    invalid_values = [value for value in values if value not in action.choices]
    if invalid_values:
        raise ValueError(
            f"Invalid experiment config value{source}: {invalid_values}. "
            f"Choose from {list(action.choices)}"
        )
