"""Dataset, split, DataLoader, logging, and evaluation utilities.

An important distinction throughout this module is:

* an ``index`` identifies a sample in the underlying source dataset;
* a ``position`` identifies an offset inside the current ``Subset``.

Whenever a split rebuilds a ``Subset``, positions change even though source
indices do not.  The post-apportion helpers therefore return remapped positions
for the new training subset.
"""

import copy
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytorch_metric_learning.samplers as samplers
import torch
from loguru import logger
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import return_results
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from training import gpu_feature_loader

from . import local_datasets
from .dataset_constants import (
    CIFAR100_DEVELOPMENT_CLASSES,
    CIFAR100_FC100_DEVELOPMENT_CLASSES,
    CIFAR100_FC100_DEVELOPMENT_SUPERCLASSES,
    CIFAR100_FC100_TEST_CLASSES,
    CIFAR100_FC100_TEST_SUPERCLASSES,
    CIFAR100_FC100_TRAIN_CLASSES,
    CIFAR100_FC100_TRAIN_SUPERCLASSES,
    CIFAR100_FC100_VALIDATION_CLASSES,
    CIFAR100_FC100_VALIDATION_SUPERCLASSES,
    CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES,
    CIFAR100_FINE_CLASS_DISJOINT_TEST_CLASSES,
    CIFAR100_FINE_CLASS_TO_SUPERCLASS,
    CIFAR100_HELD_OUT_TEST_CLASSES,
    CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES,
    CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES,
    CIFAR100_SUPERCLASS_DISJOINT_TEST_CLASSES,
    CIFAR100_SUPERCLASS_DISJOINT_TEST_SUPERCLASSES,
    CIFAR100_SUPERCLASS_FINE_CLASSES,
    CIFAR100_SUPERCLASS_NAMES,
    CIFAR10_DEVELOPMENT_CLASSES,
    CIFAR10_HELD_OUT_TEST_CLASSES,
    CIFAR_DATASETS,
    CIFAR_LONG_TAIL_SOURCE,
    ANY_DATASET,
    CIFAR_UNSEEN_CLASS_PROTOCOLS,
    CUB_ALTERNATING_CLASS_SPLIT_VERSION,
    CUB_ALTERNATING_DEVELOPMENT_CLASSES,
    CUB_ALTERNATING_HELD_OUT_TEST_CLASSES,
    CUB_CLASS_IDS,
    CUB_EXPECTED_CLASS_COUNT,
    CUB_PROTOCOLS,
    CV_MODES,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
    CV_MODE_SUPERCLASS_GROUP_KFOLD,
    DATASET_PROTOCOLS,
    DATASET_PROTOCOL_INFO,
    DATASET_PROTOCOL_CIFAR100_FC100,
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_CUB_ALTERNATING_100_100,
    DATASET_PROTOCOL_NABIRDS_ALTERNATING_277_278,
    DATASET_PROTOCOL_NABIRDS_PUMA_278_277,
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_HASH_100_100,
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500,
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_HASH_500_500,
    DATASET_PROTOCOL_SEMI_INAT_KNOWN_810,
    DATASET_PROTOCOL_SEMI_INAT_ORACLE_50_50,
    DATASET_PROTOCOL_STANFORD_DOGS_PUMA_60_60,
    STANFORD_DOGS_PROTOCOLS,
    STANFORD_DOGS_PROTOCOL_CLASS_SPLITS,
    STANFORD_DOGS_PUMA_CLASS_SPLIT,
    NABIRDS_ALTERNATING_CLASS_SPLIT,
    NABIRDS_PROTOCOLS,
    NABIRDS_PROTOCOL_CLASS_SPLITS,
    NABIRDS_PUMA_CLASS_SPLIT,
    NATIVE_UNLABELED_SEMI_AVES,
    NATIVE_UNLABELED_SEMI_INAT,
    GROUPED_CV_MODES,
    DEFAULT_VALIDATION_GALLERY_FRACTION,
    POST_APPORTION_VAL_RATIO,
    QUERY_GALLERY_EVALUATION,
    SAME_SOURCE_EVALUATION,
    SEMI_AVES_DEFAULT_OOD_FRACTION,
    SEMI_AVES_DEFAULT_OOD_SEED,
    SEMI_AVES_ALTERNATING_CLASS_SPLIT,
    SEMI_AVES_KNOWN_PROTOCOLS,
    SEMI_AVES_ORACLE_PROTOCOLS,
    SEMI_AVES_PROTOCOL_CLASS_SPLITS,
    SEMI_AVES_PROTOCOLS,
    SEMI_AVES_SHA256_CLASS_SPLIT,
    SEMI_INAT_DEFAULT_OOD_FRACTION,
    SEMI_INAT_DEFAULT_OOD_SEED,
    SEMI_INAT_PROTOCOLS,
    SUPERCLASS_AWARE_CV_MODES,
    VAL_MODES,
    VAL_MODE_ALL,
    VAL_MODE_MATCH_TRAIN,
    VAL_MODE_SPLIT_AFTER_APPORTION,
    VALIDATION_RETRIEVAL_MODES,
)
from .dataset_composition import (
    COMPCARS_PAPER_THRESHOLD_CALIBRATION_AUTO,
    COMPCARS_PAPER_THRESHOLD_CALIBRATION_MODES,
    COMPCARS_PAPER_THRESHOLD_CALIBRATION_OFF,
    CombinedDataset,
    DatasetBundle,
    EXTERNAL_UNLABELED_FILTERS,
    EXTERNAL_UNLABELED_FILTERS_COMPCARS_PAPER,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_SLADE_PAPER,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER,
    EXTERNAL_UNLABELED_FILTER_NABIRDS,
    EXTERNAL_UNLABELED_FILTER_NONE,
    append_external_unlabeled_dataset,
    append_semi_aves_native_unlabeled_dataset,
    append_semi_inat_native_unlabeled_dataset,
    get_nested_transform,
)
from .dataset_protocols import (
    apply_cifar_long_tail,
    describe_dataset_protocols,
    format_dataset_protocol_help,
    get_dataset_class,
    is_dataset_ready,
    load_dataset_protocol_sources as _load_dataset_protocol_sources,
    make_cifar_long_tail_class_counts,
    normalize_dataset_name,
    sanitize_native_unlabeled_pool_fractions,
    validate_cifar_balanced_fraction_protocol,
    validate_cifar_imbalance_factor,
    validate_dataset_protocol,
    validate_semi_aves_ood_fraction,
    validate_semi_inat_ood_fraction,
)
from .puma_unified_dataset import (
    PUMA_ALL_MEMBER_NAMES,
    PUMA_MEMBERS,
    PUMA_PAPER_SPLIT_COUNTS,
    PUMA_STANDARD_MEMBER_NAMES,
    PUMAStandard,
    PUMAUnified,
)
from .dataset_splits import (
    apply_apportioned_cross_validation_split,
    apply_post_apportion_validation_split,
    apply_validation_mode,
    assert_disjoint_dataset_classes,
    build_class_groups_by_superclass,
    cifar100_superclass_labels_for_fine_labels,
    count_dataset_classes,
    count_labels_at_positions,
    derive_query_gallery_indices,
    make_holdout_split_info,
    make_superclass_balanced_group_folds,
    make_superclass_group_folds,
    make_train_valid_subsets,
    remap_positions,
    select_balanced_subset_indices,
    set_nested_transform,
    split_cifar_balanced_by_fraction,
    split_dataset_by_classes,
    split_dataset_by_fixed_classes,
    split_dataset_by_classes_superclass_balanced,
    split_dataset_cross_validation,
    split_positions_class_disjoint_by_label,
    split_positions_cross_validation,
    split_positions_stratified_by_label,
    split_positions_superclass_balanced_holdout,
    subset_dataset_by_classes,
    subset_dataset_by_indices,
    unique_sorted_positions,
    update_apportioned_cross_validation_info,
    update_post_apportion_validation_info,
    update_validation_mode_info,
    validate_group_cv,
    validate_stratified_cv,
)
from .image_transforms import (
    DEFAULT_IMAGE_RESIZE_MODE,
    DEFAULT_IMAGE_SIZE,
    DINOV2_PATCH_SIZE,
    IMAGE_RESIZE_MODES,
    IMAGE_RESIZE_MODE_DINOV2,
    IMAGE_RESIZE_MODE_INFO,
    IMAGE_RESIZE_MODE_SQUASH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    describe_image_resize_modes,
    format_image_resize_mode_help,
    make_test_transform,
    make_train_transform,
    validate_image_resize_mode,
    validate_image_size,
)

PREFERRED_TORCH_SHARING_STRATEGY = "file_descriptor"
TORCH_SHARING_STRATEGY_ENV = "METRIC_LEARNING_TORCH_SHARING_STRATEGY"

TENSORBOARD_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = exc

DATALOADER_START_METHODS = ("spawn", "forkserver", "fork", "default")
# This threshold affects only projection batch sizing. It never decides whether
# retrieval runs on CPU or CUDA; GPU FAISS is used for every evaluation when it
# is available.
FROZEN_LARGE_EVAL_MIN_SAMPLES = 5_000
FROZEN_CUDA_EVAL_BATCH_SIZE = 4_096
FAISS_GPU_MAX_K = 2_048
TORCH_EXACT_KNN_MAX_DISTANCE_BYTES = 256 * 1024 * 1024
EVALUATION_RETRIEVAL_BACKEND_CPU = "cpu"
EVALUATION_RETRIEVAL_BACKEND_CUDA = "cuda"
EVALUATION_RETRIEVAL_BACKENDS = (
    EVALUATION_RETRIEVAL_BACKEND_CPU,
    EVALUATION_RETRIEVAL_BACKEND_CUDA,
)
# Retrieval can still run through FAISS-GPU while projected embeddings remain
# in host memory.  This avoids a second full device copy alongside the FAISS
# index and enables the low-memory FAISS resource policy below.
EVALUATION_EMBEDDING_RESIDENCY_CPU = "cpu"
EVALUATION_EMBEDDING_RESIDENCY_GPU = "gpu"
EVALUATION_EMBEDDING_RESIDENCIES = (
    EVALUATION_EMBEDDING_RESIDENCY_CPU,
    EVALUATION_EMBEDDING_RESIDENCY_GPU,
)
# Device scratch left to FAISS under ``--low_memory_mode`` when embeddings are
# GPU-resident. Measured on a 16 GB card, the default arena is ~1.5 GiB; 256 MiB
# recovers most of that and still lets a search run without falling back to a
# cudaMalloc per call. Host-resident retrieval caps the arena at zero instead,
# which is what the pre-existing low-memory FAISS policy already did.
LOW_MEMORY_FAISS_TEMP_BYTES = 256 * 1024 * 1024
# Where a precomputed frozen-feature view reads its rows from. "mmap" keeps the
# shared on-disk matrix and pays a page fault per uncached row; "ram" copies the
# rows this view actually uses into a private tensor once, so training steps
# never touch the file again. "gpu" additionally uploads the view to the training
# device and drops the DataLoader, so a step gathers its batch on-device.
FEATURE_RESIDENCY_MMAP = "mmap"
FEATURE_RESIDENCY_RAM = "ram"
FEATURE_RESIDENCY_GPU = "gpu"
FEATURE_RESIDENCIES = (
    FEATURE_RESIDENCY_MMAP,
    FEATURE_RESIDENCY_RAM,
    FEATURE_RESIDENCY_GPU,
)


def load_dataset_protocol_sources(*args, **kwargs):
    """Compatibility façade that preserves an overridable dataset resolver."""

    kwargs.setdefault("dataset_class_resolver", get_dataset_class)
    return _load_dataset_protocol_sources(*args, **kwargs)


class MPerClassSamplerCapacityError(ValueError):
    """Raised when the selected labels cannot fill one M-per-class batch."""


class NonFiniteEmbeddingError(ValueError):
    """Raised when evaluation embeddings contain NaN or infinite values."""

    pass


def configure_torch_sharing_strategy(strategy=None):
    """Prefer FD-backed Torch sharing to avoid orphaned torch_shm_manager processes."""

    available = set(torch.multiprocessing.get_all_sharing_strategies())
    requested = strategy or os.environ.get(TORCH_SHARING_STRATEGY_ENV)
    if requested is None:
        if PREFERRED_TORCH_SHARING_STRATEGY not in available:
            return torch.multiprocessing.get_sharing_strategy()
        requested = PREFERRED_TORCH_SHARING_STRATEGY

    if requested not in available:
        raise ValueError(
            f"Torch multiprocessing sharing strategy {requested!r} is not available. "
            f"Available: {sorted(available)}"
        )
    torch.multiprocessing.set_sharing_strategy(requested)
    return requested


configure_torch_sharing_strategy()


def normalize_device_name(device_name):
    # Let torch parse aliases/index syntax first, then restrict accepted device
    # families to the ones supported by this training script.
    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'") from exc

    if device.type == "cpu" and device.index is None:
        return "cpu"
    if device.type == "cuda" and (device.index is None or device.index >= 0):
        return str(device)
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def activate_torch_device(device_name):
    """Make an explicitly indexed CUDA device active for process-wide APIs."""

    device = torch.device(normalize_device_name(device_name))
    if device.type == "cuda" and device.index is not None:
        # Moving a tensor to cuda:N does not change torch.cuda.current_device().
        # FAISS and CUDA calls without a device argument use that current device,
        # so select it before any CUDA initialization or seeding can touch cuda:0.
        torch.cuda.set_device(device)
    return device


def initialize_logger(args):
    """Create or reuse a run directory and configure console/file logging."""

    start_time = datetime.now()
    logger.remove()
    # Mutating args.log_dir makes the concrete path available to all later
    # artifact writers. HPO trials use a stable directory so recovered trials
    # overwrite the same trial folder instead of creating timestamp children.
    args.log_dir = resolve_log_dir(args, start_time)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(args.log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    # Opt-in: the DEBUG sink is the only one below INFO, so leaving it off keeps
    # the lazy debug records below from ever being rendered.
    if getattr(args, "debug_log", False):
        logger.add(args.log_dir / "debug.log", level="DEBUG")
    # Route otherwise uncaught exceptions through the run log, preserving a
    # traceback in interrupted experiment directories.
    sys.excepthook = lambda _, value, tb: logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"The outputs are being saved in {args.log_dir}")


def resolve_log_dir(args, start_time):
    base_dir = Path("logs") / args.save_dir
    if is_hpo_trial_run(args):
        return base_dir
    return base_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")


def is_hpo_trial_run(args):
    return getattr(args, "trial_number", None) is not None or getattr(args, "hparam_study_name", None) is not None


def seed_everything(seed, device="cpu"):
    """Seed Python, NumPy, and Torch for reproducible split/training behavior."""

    device = activate_torch_device(device)
    # Different libraries maintain independent random-number generators.
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed() seeds every CUDA device as well as the CPU. Seed the
    # CPU generator directly so an otherwise unused GPU does not get a context.
    torch.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

def seed_worker(_):
    """Derive deterministic NumPy/Python seeds for each DataLoader worker."""

    # DataLoader assigns each worker a distinct Torch seed derived from its
    # generator. Reuse it for NumPy/Python code executed inside that worker.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


#: Set once from ``--eval_amp``. Both this and the training step's autocast
#: (``--train_amp``) are off by default, so a default run is float32 throughout;
#: this stays opt-in because enabling it changes the stored embeddings and
#: therefore every retrieval metric computed from them.
_EVAL_AUTOCAST_ENABLED = False


def set_eval_autocast_enabled(enabled):
    """Enable or disable bfloat16 autocast for evaluation/embedding passes."""

    global _EVAL_AUTOCAST_ENABLED
    _EVAL_AUTOCAST_ENABLED = bool(enabled)


def eval_autocast_enabled():
    return _EVAL_AUTOCAST_ENABLED


#: Set once from ``--low_memory_mode``. ``StandardGpuResources`` otherwise
#: reserves a ~1.5 GiB device scratch arena and a 256 MiB pinned host buffer per
#: process, both for the process lifetime and both outside the torch caching
#: allocator, so neither training nor a sibling HPO run can reuse them. Capping
#: them changes no metric; it only bounds what FAISS holds between searches.
_FAISS_LOW_MEMORY_ENABLED = False
_FAISS_LOW_MEMORY_TEMP_BYTES = LOW_MEMORY_FAISS_TEMP_BYTES


def set_faiss_low_memory(enabled, temp_memory_bytes=None):
    """Cap the FAISS GPU resource buffers for every retrieval in this process."""

    global _FAISS_LOW_MEMORY_ENABLED, _FAISS_LOW_MEMORY_TEMP_BYTES
    _FAISS_LOW_MEMORY_ENABLED = bool(enabled)
    if temp_memory_bytes is not None:
        if int(temp_memory_bytes) < 0:
            raise ValueError("FAISS temp memory must be non-negative")
        _FAISS_LOW_MEMORY_TEMP_BYTES = int(temp_memory_bytes)


def faiss_low_memory_enabled():
    return _FAISS_LOW_MEMORY_ENABLED


def _faiss_memory_policy(low_memory):
    """Return ``(temp_memory_bytes, pinned_memory_bytes)``; ``None`` keeps FAISS's default.

    ``low_memory`` here is the pre-existing host-resident retrieval policy,
    which already zeroed the scratch arena. Low-memory mode additionally caps
    the arena for GPU-resident retrieval and zeroes the pinned host buffer in
    both, since a one-shot index build plus search does not stream enough to
    benefit from a persistent staging area.
    """

    temp_memory_bytes = 0 if low_memory else None
    if not _FAISS_LOW_MEMORY_ENABLED:
        return temp_memory_bytes, None
    if not low_memory:
        temp_memory_bytes = _FAISS_LOW_MEMORY_TEMP_BYTES
    return temp_memory_bytes, 0


def eval_autocast(device):
    """Autocast context for inference passes, inert unless ``--eval_amp`` is set."""

    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=_EVAL_AUTOCAST_ENABLED,
    )


def effective_num_workers(num_workers, platform_name=None):
    """Disable DataLoader subprocesses on Windows while preserving other platforms."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    platform_name = sys.platform if platform_name is None else platform_name
    return 0 if platform_name.startswith("win") else num_workers


def validate_dataloader_settings(device, num_workers, ssl_embedding_num_workers, start_method):
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if ssl_embedding_num_workers < 0:
        raise ValueError("ssl embedding_num_workers must be non-negative")
    if start_method not in DATALOADER_START_METHODS:
        raise ValueError(f"dataloader_start_method must be one of {DATALOADER_START_METHODS}")
    if start_method != "default" and start_method not in mp.get_all_start_methods():
        raise ValueError(
            f"dataloader_start_method={start_method!r} is not available on this platform. "
            f"Available: {mp.get_all_start_methods()}"
        )
    if device.startswith("cuda") and max(num_workers, ssl_embedding_num_workers) > 0:
        if start_method in {"default", "fork"}:
            raise ValueError(
                "CUDA with multi-worker DataLoaders must not use fork/default multiprocessing. "
                "Use --dataloader_start_method spawn or forkserver, or set --num_workers 0 "
                "and embedding_num_workers 0 in the SSL config."
            )


def make_dataloader_kwargs(
    num_workers,
    seed,
    start_method,
    persistent_workers=False,
    pin_memory=True,
):
    """Build the shared deterministic multiprocessing options for DataLoaders."""

    num_workers = effective_num_workers(num_workers)
    # The generator controls DataLoader/sampler randomness. worker_init_fn then
    # transfers the derived worker seed to NumPy and Python's random module.
    kwargs = {
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": make_torch_generator(seed),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        # persistent_workers avoids process startup each epoch, but it cannot be
        # used when there are no worker processes.
        kwargs["persistent_workers"] = persistent_workers
    if num_workers > 0 and start_method != "default":
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"dataloader_start_method={start_method!r} is not available on this platform. "
                f"Available: {mp.get_all_start_methods()}"
            )
        kwargs["multiprocessing_context"] = start_method
    return kwargs


def shutdown_dataloader_workers(loader):
    """Best-effort shutdown for DataLoader persistent worker iterators."""

    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    if iterator is None and hasattr(loader, "_shutdown_workers"):
        iterator = loader
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        try:
            shutdown()
        except AttributeError as exc:
            # PyTorch can raise this while cleaning a partially initialized
            # multiprocessing iterator after worker startup failed.
            if "_workers_status" not in str(exc):
                raise

    if hasattr(loader, "_iterator"):
        try:
            loader._iterator = None
        except AttributeError:
            pass


def shutdown_dataloaders(*loaders):
    """Shutdown each distinct DataLoader or DataLoader iterator supplied."""

    seen = set()

    def visit(value):
        if value is None:
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                visit(child)
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        shutdown_dataloader_workers(value)

    for loader in loaders:
        visit(loader)


class MetricsLogger:
    """Write training/evaluation metrics to TensorBoard and CSV."""

    #: Diagnostic categories written unconditionally before --log_epoch_diagnostics
    #: existed. They are diagnostic-only, so they are opt-in; the per-batch
    #: categories stay governed by their own upstream interval/flag guards.
    EPOCH_DIAGNOSTIC_CATEGORIES = frozenset(
        {"train_epoch", "eval_epoch", "eval_per_class", "eval_per_class_summary"}
    )

    def __init__(self, log_dir, args):
        self.tensorboard_enabled = bool(getattr(args, "tensorboard", True))
        self.epoch_diagnostics_enabled = bool(getattr(args, "log_epoch_diagnostics", False))
        if self.tensorboard_enabled and SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the tensorboard package. "
                "Install it with `pip install -r requirements.txt`, or pass --no-tensorboard."
            ) from TENSORBOARD_IMPORT_ERROR

        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "metrics.csv"
        self.diagnostics_path = self.log_dir / "diagnostics.csv"
        # TensorBoard is convenient for visualization; CSV keeps metrics easy
        # to inspect and aggregate with ordinary tools.
        self.writer = (
            SummaryWriter(log_dir=str(self.log_dir / "tensorboard")) if self.tensorboard_enabled else None
        )
        # Headline measurement columns follow the CLI list. Recall@K remains a
        # separately parameterized family because each requested K is a column.
        self.measurements = logged_measurements(args)
        self.recall_at_k = logged_recall_at_k(args)
        self.recall_at_k_columns = [recall_at_k_metric_name(k) for k in self.recall_at_k]
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "step",
                "epoch",
                "split",
                "loss",
                *self.measurements,
                *self.recall_at_k_columns,
            ],
        )
        self.csv_writer.writeheader()
        # Opened on the first surviving diagnostic so a run that disables every
        # category leaves no empty diagnostics.csv behind.
        self.diagnostics_file = None
        self.diagnostics_writer = None
        self._configure_tensorboard(args)
        logger.info(f"CSV metrics are being saved in {self.csv_path}")

    def _configure_tensorboard(self, args):
        if not self.tensorboard_enabled:
            logger.info("TensorBoard logging is disabled (--no-tensorboard)")
            return
        self.writer.add_text("run/arguments", "\n".join(f"{key}: {value}" for key, value in vars(args).items()), 0)
        self.writer.add_custom_scalars(
            {
                "Epoch analysis": {
                    "Training loss": ["Multiline", ["epoch/train/loss"]],
                    "Learning rates": [
                        "Multiline",
                        [
                            "epoch/train/learning_rate/model/group_0",
                            "epoch/train/learning_rate/criterion/group_0",
                        ],
                    ],
                    "Validation retrieval": [
                        "Multiline",
                        [
                            *(
                                f"epoch/valid/{measurement}"
                                for measurement in self.measurements
                            ),
                            *(
                                f"epoch/valid/{column}"
                                for column in self.recall_at_k_columns
                            ),
                        ],
                    ],
                    "Validation timing": [
                        "Multiline",
                        [
                            "epoch/valid/timing/embedding_extraction_seconds",
                            "epoch/valid/timing/retrieval_metrics_seconds",
                            "epoch/valid/timing/total_seconds",
                        ],
                    ],
                    "Validation per-class macro means": [
                        "Multiline",
                        [
                            *(
                                f"epoch/valid/per_class_summary/{measurement}_macro_mean"
                                for measurement in self.measurements
                                if measurement in PER_CLASS_RETRIEVAL_METRICS
                            ),
                        ],
                    ],
                }
            }
        )
        logger.info(f"TensorBoard logs are being saved in {self.log_dir / 'tensorboard'}")

    def log_train_batch(self, loss, epoch, step, diagnostics=None):
        # Batch loss uses global optimizer-step count on TensorBoard's x-axis.
        self._add_scalar("train/batch_loss", loss, step)
        self._write_row(step=step, epoch=epoch, split="train_batch", loss=loss)
        self.log_diagnostics(diagnostics, step=step, epoch=epoch, category="train_batch")

    def log_train_epoch(self, loss, epoch, step, diagnostics=None):
        self._add_scalar("train/epoch_loss", loss, step)
        self._add_epoch_scalar("train/loss", loss, epoch)
        self._write_row(step=step, epoch=epoch, split="train_epoch", loss=loss)
        self.log_diagnostics(diagnostics, step=step, epoch=epoch, category="train_epoch")
        self._add_epoch_diagnostics(diagnostics, epoch)

    def log_eval(
        self,
        split,
        precision_at_1,
        mean_average_precision_at_r,
        step,
        epoch=None,
        per_class_metrics=None,
        diagnostics=None,
        recall_at_k=None,
        measurements=None,
    ):
        # The split prefix keeps validation and test curves separate.
        measurement_values = self._checked_measurement_values(
            measurements,
            split,
            precision_at_1=precision_at_1,
            mean_average_precision_at_r=mean_average_precision_at_r,
        )
        for measurement, value in measurement_values.items():
            self._add_scalar(f"{split}/{measurement}", value, step)
            self._add_epoch_scalar(f"{split}/{measurement}", value, epoch)
        recall_at_k = self._checked_recall_values(recall_at_k, split)
        for k, value in recall_at_k.items():
            metric_name = recall_at_k_metric_name(k)
            self._add_scalar(f"{split}/{metric_name}", value, step)
            self._add_epoch_scalar(f"{split}/{metric_name}", value, epoch)
        self._write_row(
            step=step,
            epoch=epoch,
            split=split,
            measurements=measurement_values,
            recall_at_k=recall_at_k,
        )
        qualified_diagnostics = {
            self._qualify_split_name(split, name): value
            for name, value in (diagnostics or {}).items()
        }
        self.log_diagnostics(
            qualified_diagnostics,
            step=step,
            epoch=epoch,
            category="eval_epoch",
        )
        self._add_epoch_diagnostics(qualified_diagnostics, epoch)
        # The per-class breakdown and its summary exist only to feed diagnostics,
        # so skip the numpy work entirely rather than computing and discarding it.
        if not self.epoch_diagnostics_enabled:
            return
        for label, metrics in (per_class_metrics or {}).items():
            class_diagnostics = {
                **{
                    f"{split}/per_class/{label}/{measurement}": metrics[measurement]
                    for measurement in self.measurements
                    if measurement in PER_CLASS_RETRIEVAL_METRICS
                    and measurement in metrics
                },
                f"{split}/per_class/{label}/sample_count": metrics["count"],
            }
            self.log_diagnostics(
                class_diagnostics,
                step=step,
                epoch=epoch,
                category="eval_per_class",
                details={"class_label": label},
            )
            self._add_epoch_diagnostics(class_diagnostics, epoch)

        summary_diagnostics, distributions = self._summarize_per_class_metrics(
            split,
            per_class_metrics,
        )
        self.log_diagnostics(
            summary_diagnostics,
            step=step,
            epoch=epoch,
            category="eval_per_class_summary",
        )
        self._add_epoch_diagnostics(summary_diagnostics, epoch)
        if epoch is not None and self.writer is not None:
            for name, values in distributions.items():
                self.writer.add_histogram(f"epoch/{name}", values, int(epoch))

    def close(self):
        self.csv_file.close()
        if self.diagnostics_file is not None:
            self.diagnostics_file.close()
        if self.writer is not None:
            self.writer.close()

    def log_diagnostics(self, diagnostics, step, epoch, category, details=None):
        if not diagnostics or not self._diagnostics_category_enabled(category):
            return
        rows = []
        for name, value in diagnostics.items():
            if value is None:
                continue
            scalar_value = float(value)
            self._add_scalar(name, scalar_value, step)
            rows.append(
                {
                    "step": step,
                    "epoch": "" if epoch is None else epoch,
                    "category": category,
                    "name": name,
                    "value": scalar_value,
                    "details": "" if details is None else json.dumps(details, sort_keys=True),
                }
            )
        if not rows:
            return
        self._diagnostics_csv_writer().writerows(rows)
        self.diagnostics_file.flush()

    def _diagnostics_category_enabled(self, category):
        # Per-batch categories keep their own upstream guards
        # (--log_batch_diagnostics, --ssl_gradient_contribution_log_interval);
        # only the always-on epoch/eval categories answer to --log_epoch_diagnostics.
        if category in self.EPOCH_DIAGNOSTIC_CATEGORIES:
            return self.epoch_diagnostics_enabled
        return True

    def _diagnostics_csv_writer(self):
        """Open diagnostics.csv on first use so disabled runs leave no empty file."""

        if self.diagnostics_writer is None:
            self.diagnostics_file = self.diagnostics_path.open("w", newline="")
            self.diagnostics_writer = csv.DictWriter(
                self.diagnostics_file,
                fieldnames=["step", "epoch", "category", "name", "value", "details"],
            )
            self.diagnostics_writer.writeheader()
            logger.info(f"Diagnostic metrics are being saved in {self.diagnostics_path}")
        return self.diagnostics_writer

    def _add_scalar(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def _add_epoch_scalar(self, name, value, epoch):
        if epoch is None or value is None or self.writer is None:
            return
        self.writer.add_scalar(f"epoch/{name}", float(value), int(epoch))

    def _add_epoch_diagnostics(self, diagnostics, epoch):
        # Every caller passes an epoch/eval diagnostic dict, so this mirrors the
        # diagnostics.csv gate; headline metrics use _add_epoch_scalar directly.
        if not self.epoch_diagnostics_enabled:
            return
        for name, value in (diagnostics or {}).items():
            self._add_epoch_scalar(name, value, epoch)

    @staticmethod
    def _qualify_split_name(split, name):
        name = str(name).lstrip("/")
        return name if name.startswith(f"{split}/") else f"{split}/{name}"

    @staticmethod
    def _summarize_per_class_metrics(split, per_class_metrics):
        if not per_class_metrics:
            return {}, {}

        diagnostics = {
            f"{split}/per_class_summary/class_count": len(per_class_metrics),
        }
        distributions = {}
        counts = np.asarray(
            [metrics["count"] for metrics in per_class_metrics.values()],
            dtype=np.float64,
        )
        diagnostics.update(
            {
                f"{split}/per_class_summary/sample_count": counts.sum(),
                f"{split}/per_class_summary/samples_per_class_min": counts.min(),
                f"{split}/per_class_summary/samples_per_class_mean": counts.mean(),
                f"{split}/per_class_summary/samples_per_class_max": counts.max(),
            }
        )
        distributions[f"{split}/per_class_distribution/sample_count"] = counts

        for metric_name in PER_CLASS_RETRIEVAL_METRICS:
            if not all(metric_name in metrics for metrics in per_class_metrics.values()):
                continue
            values = np.asarray(
                [metrics[metric_name] for metrics in per_class_metrics.values()],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            prefix = f"{split}/per_class_summary/{metric_name}"
            diagnostics.update(
                {
                    f"{prefix}_min": values.min(),
                    f"{prefix}_macro_mean": values.mean(),
                    f"{prefix}_std": values.std(),
                    f"{prefix}_max": values.max(),
                }
            )
            distributions[f"{split}/per_class_distribution/{metric_name}"] = values
        return diagnostics, distributions

    def _checked_recall_values(self, recall_at_k, split):
        """Reject values for a K this run never configured a column for."""

        recall_at_k = {int(k): value for k, value in (recall_at_k or {}).items()}
        unexpected = sorted(set(recall_at_k) - set(self.recall_at_k))
        if unexpected:
            raise ValueError(
                f"{split} reported Recall@{unexpected} but the run configured "
                f"recall_at_k={list(self.recall_at_k)}"
            )
        return recall_at_k

    def _checked_measurement_values(
        self,
        measurements,
        split,
        *,
        precision_at_1=None,
        mean_average_precision_at_r=None,
    ):
        """Validate one evaluation's values against the configured columns."""

        values = dict(measurements or {})
        legacy_values = {
            MEASUREMENT_PRECISION_AT_1: precision_at_1,
            MEASUREMENT_MAP_AT_R: mean_average_precision_at_r,
        }
        for measurement, value in legacy_values.items():
            if measurement in self.measurements and measurement not in values:
                values[measurement] = value
        unexpected = sorted(set(values) - set(self.measurements))
        if unexpected:
            raise ValueError(
                f"{split} reported unconfigured measurements {unexpected}; "
                f"measurements={list(self.measurements)}"
            )
        return {
            measurement: values[measurement]
            for measurement in self.measurements
            if values.get(measurement) is not None
        }

    def _write_row(
        self,
        step,
        epoch,
        split,
        loss=None,
        precision_at_1=None,
        mean_average_precision_at_r=None,
        measurements=None,
        recall_at_k=None,
    ):
        # Empty strings produce clean sparse CSV columns for rows containing
        # either a loss or retrieval metrics.
        recall_at_k = recall_at_k or {}
        measurement_values = dict(measurements or {})
        if precision_at_1 is not None:
            measurement_values.setdefault(MEASUREMENT_PRECISION_AT_1, precision_at_1)
        if mean_average_precision_at_r is not None:
            measurement_values.setdefault(MEASUREMENT_MAP_AT_R, mean_average_precision_at_r)
        self.csv_writer.writerow(
            {
                "step": step,
                "epoch": "" if epoch is None else epoch,
                "split": split,
                "loss": "" if loss is None else loss,
                **{
                    measurement: measurement_values.get(measurement, "")
                    for measurement in self.measurements
                },
                **{
                    recall_at_k_metric_name(k): recall_at_k.get(k, "")
                    for k in self.recall_at_k
                },
            }
        )
        self.csv_file.flush()


def summarize_miner_outputs(miner_outputs):
    """Return scalar tuple/pair counts for a metric-learning miner output."""

    if miner_outputs is None:
        return {}
    if torch.is_tensor(miner_outputs):
        return {"train/miner/output_count": int(miner_outputs.numel())}
    if not isinstance(miner_outputs, (tuple, list)):
        return {}

    counts = [int(output.numel()) if torch.is_tensor(output) else len(output) for output in miner_outputs]
    if len(counts) == 3:
        return {"train/miner/triplet_count": min(counts)}
    if len(counts) == 4:
        positive_pairs = min(counts[:2])
        negative_pairs = min(counts[2:])
        return {
            "train/miner/positive_pair_count": positive_pairs,
            "train/miner/negative_pair_count": negative_pairs,
            "train/miner/total_pair_count": positive_pairs + negative_pairs,
        }
    return {f"train/miner/output_{index}_count": count for index, count in enumerate(counts)}


def gradient_l2_norm(parameters):
    """Compute the global L2 norm of currently populated gradients."""

    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradients.append(parameter.grad.detach())
    if not gradients:
        return None

    if hasattr(torch, "_foreach_norm"):
        norms = torch._foreach_norm([gradient.float() for gradient in gradients], 2.0)
    else:
        norms = [torch.linalg.vector_norm(gradient.float(), ord=2) for gradient in gradients]
    device = norms[0].device
    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(device) for norm in norms]), ord=2)
    return float(total_norm.item())


def optimizer_learning_rates(optimizer, optimizer_name):
    return {
        f"train/learning_rate/{optimizer_name}/group_{index}": float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def setup_dataset_bundle(
    dataset_name,
    seed,
    data_split_seed=None,
    cv_k=1,
    cv_fold=None,
    cv_mode="group_kfold",
    val_mode=VAL_MODE_ALL,
    dataset_protocol=DATASET_PROTOCOL_OFFICIAL,
    cifar_imbalance_factor=None,
    cifar_train_fraction=0.8,
    cifar_test_fraction=0.2,
    full_train=False,
    holdout_val_ratio=None,
    image_resize_mode=DEFAULT_IMAGE_RESIZE_MODE,
):
    """Load source data and create the initial train/validation/test split.

    ``split_after_apportion`` is special: validation is left empty here and is
    carved from the selected support draw later.
    ``full_train`` uses every development sample and leaves validation empty for
    the one fixed-epoch model trained after HPO.
    ``holdout_val_ratio`` resizes the class-disjoint holdout so a run that early-
    stops on its own validation slice can keep that slice small; ``None`` keeps
    the built-in 80/20 class split.
    ``image_resize_mode`` selects the image geometry every split is loaded with.
    """

    dataset_name = normalize_dataset_name(dataset_name)
    if data_split_seed is None:
        data_split_seed = seed
    if (
        dataset_protocol == DATASET_PROTOCOL_CIFAR100_FC100
        and cv_k > 1
        and cv_mode != CV_MODE_SUPERCLASS_GROUP_KFOLD
    ):
        raise ValueError(
            "dataset_protocol='cifar100_fc100' requires "
            "cv_mode='superclass_group_kfold' when cv_k > 1 so complete "
            "superclasses remain disjoint"
        )
    # Training uses stochastic augmentation. Validation, test, and SSL feature
    # extraction use the deterministic test_transform below. image_resize_mode
    # selects the geometry of both; see utils/image_transforms.py.
    train_transform = make_train_transform(image_resize_mode)
    test_transform = make_test_transform(image_resize_mode)
    data_root = Path("data") / dataset_name
    download = not is_dataset_ready(dataset_name, data_root)
    # change from initial setup --> into new function that handles more data splitting.
    train_val_dataset, test_dataset, protocol_info = load_dataset_protocol_sources(
        dataset_name=dataset_name,
        data_root=data_root,
        train_transform=train_transform,
        test_transform=test_transform,
        dataset_protocol=dataset_protocol,
        download=download,
        cifar_imbalance_factor=cifar_imbalance_factor,
        cifar_train_fraction=cifar_train_fraction,
        cifar_test_fraction=cifar_test_fraction,
        seed=data_split_seed,
    )

    if val_mode not in VAL_MODES:
        raise ValueError(f"val_mode must be one of {VAL_MODES}: {val_mode}")
    if holdout_val_ratio is not None and not 0 < holdout_val_ratio < 1:
        raise ValueError(f"holdout_val_ratio must be in (0, 1), got {holdout_val_ratio}")
    holdout_split_ratio = 0.8 if holdout_val_ratio is None else 1.0 - float(holdout_val_ratio)

    if full_train:
        # Final HPO evaluation trains once on the complete development pool.
        # Validation was already used for parameter and epoch selection during
        # HPO, so no samples are held back from this final fit.
        train_indices = np.arange(len(train_val_dataset), dtype=np.int64).tolist()
        train_dataset, valid_dataset, train_labels_mapper = make_train_valid_subsets(
            train_val_dataset,
            train_indices,
            [],
        )
        split_label = "full development train"
        split_info = {
            "split_kind": "full_development_train",
            "source_train_size": int(len(train_dataset)),
            "validation_size": 0,
        }
    elif val_mode == VAL_MODE_SPLIT_AFTER_APPORTION:
        # Keep all development samples available until label apportioning has
        # happened in run_training.
        train_indices = np.arange(len(train_val_dataset), dtype=np.int64).tolist()
        train_dataset, valid_dataset, train_labels_mapper = make_train_valid_subsets(
            train_val_dataset,
            train_indices,
            [],
        )
        split_label = "post-apportion source train"
        split_info = {
            "split_kind": "post_apportion_source",
            "val_mode": val_mode,
            "source_train_size": int(len(train_dataset)),
            "post_apportion_val_ratio": float(
                POST_APPORTION_VAL_RATIO if holdout_val_ratio is None else holdout_val_ratio
            ),
        }
    elif cv_k > 1:
        # Materialize only the requested fold; run_cross_validation calls this
        # function once per fold with a fresh training run.
        train_dataset, valid_dataset, train_labels_mapper = split_dataset_cross_validation(
            train_val_dataset,
            cv_k=cv_k,
            cv_fold=cv_fold,
            cv_mode=cv_mode,
            seed=data_split_seed,
        )
        split_label = f"{cv_mode} fold {cv_fold + 1}/{cv_k}"
        split_info = {
            "split_kind": "cross_validation",
            "cv_k": int(cv_k),
            "cv_fold": int(cv_fold),
            "cv_mode": cv_mode,
        }
    else:
        # The default metric-learning holdout splits by class, testing whether
        # embeddings generalize to validation classes unseen during training.
        if dataset_protocol == DATASET_PROTOCOL_CIFAR100_FC100:
            if holdout_val_ratio is not None:
                raise ValueError(
                    "dataset_protocol='cifar100_fc100' uses the canonical FC100 validation "
                    "superclasses, so holdout_val_ratio cannot resize its holdout"
                )
            train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_fixed_classes(
                train_val_dataset,
                train_classes=CIFAR100_FC100_TRAIN_CLASSES,
                valid_classes=CIFAR100_FC100_VALIDATION_CLASSES,
            )
            split_label = "canonical FC100 holdout"
        elif dataset_name == "CIFAR100" and cv_mode == CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
            train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes_superclass_balanced(
                train_val_dataset,
                split_ratio=holdout_split_ratio,
                seed=data_split_seed,
            )
            split_label = "superclass-balanced holdout"
        else:
            train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes(
                train_val_dataset,
                split_ratio=holdout_split_ratio,
                seed=data_split_seed,
            )
            split_label = "holdout"
        split_info = make_holdout_split_info(
            train_val_dataset=train_val_dataset,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            val_mode=val_mode,
        )
        split_info["holdout_val_ratio"] = (
            None if holdout_val_ratio is None else float(holdout_val_ratio)
        )
        split_info["holdout_split_ratio"] = float(holdout_split_ratio)
        if split_label == "superclass-balanced holdout":
            split_info["holdout_strategy"] = "superclass_balanced_by_cifar100_superclass"
        elif split_label == "canonical FC100 holdout":
            split_info["holdout_strategy"] = "canonical_fc100_train_validation_superclasses"
    split_info["dataset_protocol"] = protocol_info
    split_info["image_resize_mode"] = image_resize_mode
    # Training keeps augmented images for optimization but exposes a separate
    # deterministic transform for pseudo-label feature extraction.
    train_dataset.feature_transform = test_transform
    # Validation must never receive RandAugment, so replace the transform on the
    # base dataset below its Subset wrapper.
    if len(valid_dataset) > 0:
        set_nested_transform(valid_dataset, test_transform)

    logger.info(
        f"Split: {split_label}. Train size: {len(train_dataset)}, "
        f"Validation size: {len(valid_dataset)}, Test size: {len(test_dataset)}"
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_labels_mapper=train_labels_mapper,
        split_info=split_info,
    )


TWO_STREAM_CLASS_OVERLAP_MODES = ("disjoint", "independent")


class TwoStreamMPerClassBatchSampler(torch.utils.data.Sampler):
    """Build M-per-class batches with fixed true-/pseudo-labeled stream quotas.

    The pseudo-labeled (originally unlabeled) stream is primary: without an
    explicit epoch length, its size determines the number of full batches.
    The true-labeled stream is sampled repeatedly for the same number of
    batches. Each stream always contributes exactly ``m`` samples per class it
    selects; ``class_overlap`` decides how the two class selections relate.

    ``"independent"`` (the default) draws each stream's classes from its own
    label set with no coordination, matching LP-DeepSSL's uncoordinated
    two-stream sampling. Classes drawn by both streams contribute ``2 * m``
    samples and create the cross-stream positive pairs that anchor
    pseudo-labels to true labels; a batch then holds between
    ``unlabeled_batch_size / m`` and ``batch_size / m`` distinct classes.

    ``"disjoint"`` instead coordinates the streams so a class is used by at most
    one of them: every batch holds ``batch_size / m`` distinct classes
    contributing exactly ``m`` samples each, which makes the combined batch
    globally M-per-class. The cost is that a pseudo-labeled sample never shares
    a batch with a true-labeled sample of the same class, so no positive pair
    ever crosses the streams and the metric loss cannot pull pseudo-labeled
    samples onto their class's labeled exemplars.

    Sampling uses replacement when an assigned class has fewer than ``m``
    samples in that stream.
    """

    def __init__(
        self,
        labels,
        labeled_indices,
        unlabeled_indices,
        batch_size,
        labeled_batch_size,
        m,
        seed=0,
        length_before_new_iter=None,
        class_overlap="independent",
    ):
        labels = torch.as_tensor(labels, dtype=torch.long).cpu().numpy()
        if labels.ndim != 1:
            raise ValueError("TwoStreamMPerClassBatchSampler labels must be one-dimensional")

        self.batch_size = int(batch_size)
        self.labeled_batch_size = int(labeled_batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.m_per_class = int(m)
        self.class_overlap = str(class_overlap)
        if self.class_overlap not in TWO_STREAM_CLASS_OVERLAP_MODES:
            raise ValueError(
                "TwoStreamMPerClassBatchSampler class_overlap must be one of "
                f"{list(TWO_STREAM_CLASS_OVERLAP_MODES)}: {class_overlap!r}"
            )
        if self.batch_size <= 0:
            raise ValueError("TwoStreamMPerClassBatchSampler batch_size must be positive")
        if not 0 < self.labeled_batch_size < self.batch_size:
            raise ValueError(
                "TwoStreamMPerClassBatchSampler labeled_batch_size must be greater than zero "
                "and smaller than batch_size"
            )
        if self.m_per_class <= 0:
            raise ValueError("TwoStreamMPerClassBatchSampler m must be positive")
        for stream_name, stream_batch_size in (
            ("labeled", self.labeled_batch_size),
            ("unlabeled", self.unlabeled_batch_size),
        ):
            if stream_batch_size % self.m_per_class != 0:
                raise ValueError(
                    f"TwoStreamMPerClassBatchSampler {stream_name} batch size must be divisible by m: "
                    f"{stream_name}_batch_size={stream_batch_size}, m={self.m_per_class}."
                )

        self.labeled_indices = self._validate_indices(
            labeled_indices,
            len(labels),
            "labeled_indices",
        )
        self.unlabeled_indices = self._validate_indices(
            unlabeled_indices,
            len(labels),
            "unlabeled_indices",
        )
        if np.intersect1d(self.labeled_indices, self.unlabeled_indices).size:
            raise ValueError(
                "TwoStreamMPerClassBatchSampler labeled_indices and unlabeled_indices must be disjoint"
            )
        if len(self.labeled_indices) == 0 or len(self.unlabeled_indices) == 0:
            raise ValueError(
                "TwoStreamMPerClassBatchSampler requires non-empty labeled and unlabeled streams"
            )

        labeled_stream_labels = labels[self.labeled_indices]
        unlabeled_stream_labels = labels[self.unlabeled_indices]
        self._validate_stream_capacity(
            labeled_stream_labels,
            self.labeled_batch_size,
            "labeled",
            self.m_per_class,
        )
        self._validate_stream_capacity(
            unlabeled_stream_labels,
            self.unlabeled_batch_size,
            "unlabeled",
            self.m_per_class,
        )
        self._labeled_by_class = self._group_indices_by_label(
            self.labeled_indices,
            labeled_stream_labels,
        )
        self._unlabeled_by_class = self._group_indices_by_label(
            self.unlabeled_indices,
            unlabeled_stream_labels,
        )
        self._labeled_classes_per_batch = self.labeled_batch_size // self.m_per_class
        self._unlabeled_classes_per_batch = self.unlabeled_batch_size // self.m_per_class
        if self.class_overlap == "disjoint":
            # Independent streams need no preparation: the per-stream capacity
            # checks above already guarantee every batch is fillable.
            self._prepare_disjoint_class_selection()

        if length_before_new_iter is None:
            # Match LP-DeepSSL's primary-stream epoch definition at the batch
            # level. M-per-class balancing means individual pseudo-labeled
            # samples can still be replaced or omitted within that many draws.
            self.num_batches = max(
                1,
                len(self.unlabeled_indices) // self.unlabeled_batch_size,
            )
        else:
            sampler_length = make_sampler_epoch_length(
                len(labels),
                self.batch_size,
                length_before_new_iter=length_before_new_iter,
            )
            self.num_batches = sampler_length // self.batch_size
        self.num_samples = self.num_batches * self.batch_size
        self.generator = np.random.default_rng(seed)

    @staticmethod
    def _validate_indices(indices, num_samples, name):
        indices = torch.as_tensor(indices, dtype=torch.long).cpu().numpy()
        if indices.ndim != 1:
            raise ValueError(f"TwoStreamMPerClassBatchSampler {name} must be one-dimensional")
        if np.any((indices < 0) | (indices >= num_samples)):
            raise ValueError(f"TwoStreamMPerClassBatchSampler {name} contains an out-of-range index")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"TwoStreamMPerClassBatchSampler {name} must not contain duplicates")
        return indices.astype(np.int64, copy=False)

    @staticmethod
    def _validate_stream_capacity(labels, batch_size, stream_name, m_per_class):
        try:
            validate_m_per_class_sampler_capacity(
                labels,
                batch_size,
                sampler_m=m_per_class,
            )
        except MPerClassSamplerCapacityError as exc:
            raise MPerClassSamplerCapacityError(
                f"TwoStreamMPerClassBatchSampler {stream_name} stream cannot fill its batch: {exc}"
            ) from exc

    @staticmethod
    def _group_indices_by_label(indices, labels):
        grouped = {}
        for index, label in zip(indices, labels):
            grouped.setdefault(int(label), []).append(int(index))
        return {
            label: np.asarray(class_indices, dtype=np.int64)
            for label, class_indices in grouped.items()
        }

    @staticmethod
    def _log_combination_count(n, k):
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def _sample_independent_stream(self, grouped_indices, stream_batch_size):
        """Draw one stream's classes and samples, ignoring the other stream.

        This reproduces the pre-``class_overlap`` implementation exactly, down to
        the order in which it consumes the generator: the label array keeps the
        grouping's first-appearance order, classes come from ``permutation``
        rather than ``choice``, and a stream draws its indices before the next
        stream picks its classes. A seed therefore replays batches recorded by
        runs made before disjoint class selection was introduced.
        """

        classes_per_batch = stream_batch_size // self.m_per_class
        labels = np.asarray(list(grouped_indices), dtype=np.int64)
        selected_labels = self.generator.permutation(labels)[:classes_per_batch]
        return self._sample_stream(grouped_indices, selected_labels)

    def _prepare_disjoint_class_selection(self):
        labeled_labels = set(self._labeled_by_class)
        unlabeled_labels = set(self._unlabeled_by_class)
        labeled_classes_per_batch = self._labeled_classes_per_batch
        unlabeled_classes_per_batch = self._unlabeled_classes_per_batch
        classes_per_batch = labeled_classes_per_batch + unlabeled_classes_per_batch

        union_size = len(labeled_labels | unlabeled_labels)
        if union_size < classes_per_batch:
            raise MPerClassSamplerCapacityError(
                "TwoStreamMPerClassBatchSampler cannot make the combined batch globally "
                "M-per-class: "
                f"batch_size={self.batch_size}, m={self.m_per_class} requires "
                f"{classes_per_batch} distinct classes, but the union of the true- and "
                f"pseudo-labeled streams contains only {union_size}. The stream quotas "
                f"require {labeled_classes_per_batch} true-labeled and "
                f"{unlabeled_classes_per_batch} pseudo-labeled class groups. "
                "class_overlap='independent' drops this union requirement because the "
                "streams may then reuse each other's classes."
            )

        self._labeled_only_labels = np.asarray(
            sorted(labeled_labels - unlabeled_labels),
            dtype=np.int64,
        )
        self._unlabeled_only_labels = np.asarray(
            sorted(unlabeled_labels - labeled_labels),
            dtype=np.int64,
        )
        self._shared_labels = np.asarray(
            sorted(labeled_labels & unlabeled_labels),
            dtype=np.int64,
        )
        # An option records how many shared labels are assigned to each stream.
        # Weight options by their number of concrete class assignments so every
        # feasible pair of disjoint stream label sets is sampled uniformly.
        options = []
        log_weights = []
        for shared_labeled in range(min(labeled_classes_per_batch, len(self._shared_labels)) + 1):
            labeled_only = labeled_classes_per_batch - shared_labeled
            if labeled_only > len(self._labeled_only_labels):
                continue
            max_shared_unlabeled = min(
                unlabeled_classes_per_batch,
                len(self._shared_labels) - shared_labeled,
            )
            for shared_unlabeled in range(max_shared_unlabeled + 1):
                unlabeled_only = unlabeled_classes_per_batch - shared_unlabeled
                if unlabeled_only > len(self._unlabeled_only_labels):
                    continue
                options.append((shared_labeled, shared_unlabeled))
                log_weights.append(
                    self._log_combination_count(len(self._labeled_only_labels), labeled_only)
                    + self._log_combination_count(len(self._unlabeled_only_labels), unlabeled_only)
                    + self._log_combination_count(len(self._shared_labels), shared_labeled)
                    + self._log_combination_count(
                        len(self._shared_labels) - shared_labeled,
                        shared_unlabeled,
                    )
                )

        if not options:
            # The separate stream-capacity checks plus the union check above are
            # sufficient for two stream sets, so reaching this branch indicates
            # an internal consistency error rather than a user configuration.
            raise RuntimeError(
                "TwoStreamMPerClassBatchSampler found no feasible disjoint class allocation"
            )
        log_weights = np.asarray(log_weights, dtype=np.float64)
        weights = np.exp(log_weights - np.max(log_weights))
        self._class_allocation_options = tuple(options)
        self._class_allocation_probabilities = weights / weights.sum()

    def _choose_labels(self, labels, count):
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(
            self.generator.choice(labels, size=count, replace=False),
            dtype=np.int64,
        )

    def _select_disjoint_stream_labels(self):
        option_index = int(
            self.generator.choice(
                len(self._class_allocation_options),
                p=self._class_allocation_probabilities,
            )
        )
        shared_labeled, shared_unlabeled = self._class_allocation_options[option_index]
        shared = self._choose_labels(
            self._shared_labels,
            shared_labeled + shared_unlabeled,
        )
        labeled_labels = np.concatenate(
            [
                self._choose_labels(
                    self._labeled_only_labels,
                    self._labeled_classes_per_batch - shared_labeled,
                ),
                shared[:shared_labeled],
            ]
        )
        unlabeled_labels = np.concatenate(
            [
                self._choose_labels(
                    self._unlabeled_only_labels,
                    self._unlabeled_classes_per_batch - shared_unlabeled,
                ),
                shared[shared_labeled:],
            ]
        )
        self.generator.shuffle(labeled_labels)
        self.generator.shuffle(unlabeled_labels)
        return unlabeled_labels, labeled_labels

    def _sample_stream(self, grouped_indices, selected_labels):
        batch = []
        for label in selected_labels:
            candidates = grouped_indices[int(label)]
            sampled = self.generator.choice(
                candidates,
                size=self.m_per_class,
                replace=len(candidates) < self.m_per_class,
            )
            batch.extend(int(index) for index in sampled)
        return batch

    def _make_independent_batch(self):
        return self._sample_independent_stream(
            self._unlabeled_by_class,
            self.unlabeled_batch_size,
        ) + self._sample_independent_stream(
            self._labeled_by_class,
            self.labeled_batch_size,
        )

    def _make_disjoint_batch(self):
        unlabeled_labels, labeled_labels = self._select_disjoint_stream_labels()
        return self._sample_stream(
            self._unlabeled_by_class,
            unlabeled_labels,
        ) + self._sample_stream(
            self._labeled_by_class,
            labeled_labels,
        )

    def __iter__(self):
        # Both modes keep the reference implementation's primary-then-secondary
        # ordering: pseudo-labeled samples first, true-labeled samples last.
        make_batch = (
            self._make_disjoint_batch
            if self.class_overlap == "disjoint"
            else self._make_independent_batch
        )
        for _ in range(self.num_batches):
            yield make_batch()

    def __len__(self):
        return self.num_batches


def make_train_loader(
    train_dataset,
    batch_size,
    sampler_m,
    seed,
    num_workers=8,
    start_method="spawn",
    persistent_workers=True,
    length_before_new_iter=None,
    pin_memory=True,
    labeled_batch_size=None,
    class_overlap="independent",
    gpu_resident_device=None,
    gpu_resident_max_bytes=None,
):
    """Create an M-per-class loader, optionally with labeled/pseudo streams.

    With ``gpu_resident_device`` set and a precomputed-feature dataset, the
    sampler is kept but the DataLoader is replaced by a device-resident batcher.
    """

    num_workers = dataloader_num_workers_for_dataset(train_dataset, num_workers)
    if labeled_batch_size is not None:
        for attribute in ("labeled_indices", "unlabeled_indices"):
            if not hasattr(train_dataset, attribute):
                raise ValueError(
                    "Two-stream M-per-class sampling requires a relabeled dataset exposing "
                    f"{attribute}"
                )
        if len(train_dataset.unlabeled_indices) > 0:
            sampler = TwoStreamMPerClassBatchSampler(
                labels=train_dataset.labels,
                labeled_indices=train_dataset.labeled_indices,
                unlabeled_indices=train_dataset.unlabeled_indices,
                batch_size=batch_size,
                labeled_batch_size=labeled_batch_size,
                m=sampler_m,
                seed=seed,
                length_before_new_iter=length_before_new_iter,
                class_overlap=class_overlap,
            )
            train_loader = None
            if gpu_resident_device is not None:
                train_loader = gpu_feature_loader.try_make_gpu_resident_loader(
                    train_dataset,
                    device=gpu_resident_device,
                    batch_size=batch_size,
                    batch_sampler=sampler,
                    seed=seed,
                    max_bytes=gpu_resident_max_bytes,
                    desc="two-stream train",
                )
            if train_loader is None:
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=sampler,
                    **make_dataloader_kwargs(
                        num_workers,
                        seed,
                        start_method,
                        persistent_workers=persistent_workers,
                        pin_memory=pin_memory,
                    ),
                )
            logger.info(
                "Two-stream train loader: "
                f"{len(train_dataset)} samples, {len(train_dataset.labeled_indices)} true-labeled, "
                f"{len(train_dataset.unlabeled_indices)} pseudo-labeled, "
                f"batch={sampler.unlabeled_batch_size} pseudo + {sampler.labeled_batch_size} true, "
                f"m={sampler.m_per_class} per stream, class_overlap={sampler.class_overlap}, "
                f"{sampler.num_samples} sampled examples/epoch, "
                f"{len(train_loader)} batches/epoch"
            )
            return train_loader
        logger.warning(
            "Two-stream M-per-class sampling was requested, but no pseudo-labeled samples were accepted; "
            "falling back to the labeled-only MPerClassSampler"
        )

    sampler_length = make_sampler_epoch_length(
        len(train_dataset),
        batch_size,
        length_before_new_iter=length_before_new_iter,
    )
    validate_m_per_class_sampler_capacity(train_dataset.labels, batch_size, sampler_m)
    # MPerClassSampler builds batches from batch_size / sampler_m classes, with
    # exactly sampler_m sampled examples contributed by each chosen class.
    sampler = samplers.MPerClassSampler(
        train_dataset.labels,
        m=sampler_m,
        batch_size=batch_size,
        length_before_new_iter=sampler_length,
    )
    train_loader = None
    if gpu_resident_device is not None:
        train_loader = gpu_feature_loader.try_make_gpu_resident_loader(
            train_dataset,
            device=gpu_resident_device,
            batch_size=batch_size,
            sampler=sampler,
            seed=seed,
            max_bytes=gpu_resident_max_bytes,
            desc="train",
        )
    if train_loader is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **make_dataloader_kwargs(
                num_workers,
                seed,
                start_method,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            ),
        )
    logger.info(
        "Train loader: "
        f"{len(train_dataset)} samples, {len(set(train_dataset.labels))} labels, "
        f"{len(sampler)} sampled examples/epoch, {len(train_loader)} batches/epoch"
    )
    return train_loader


class STMLNearestNeighborBatchSampler(torch.utils.data.Sampler):
    """Build STML batches from nearest-neighbor groups, as in the upstream code."""

    def __init__(self, embeddings, batch_size, neighbors_per_query, seed, device="cpu"):
        device = torch.device(normalize_device_name(device))
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        if embeddings.ndim != 2:
            raise ValueError("STML sampling embeddings must be a matrix")
        if neighbors_per_query <= 0:
            raise ValueError("neighbors_per_query must be positive")
        if batch_size % neighbors_per_query != 0:
            raise ValueError("STML batch_size must be divisible by neighbors_per_query")
        if len(embeddings) < batch_size:
            raise ValueError("STML nearest-neighbor sampling requires at least batch_size samples")
        if neighbors_per_query > len(embeddings):
            raise ValueError("neighbors_per_query cannot exceed the STML training dataset size")
        self.num_samples = len(embeddings)
        self.batch_size = int(batch_size)
        self.neighbors_per_query = int(neighbors_per_query)
        self.queries_per_batch = self.batch_size // self.neighbors_per_query
        self.generator = make_torch_generator(seed)
        self.neighbor_indices = self._make_neighbor_indices(embeddings)

    def _make_neighbor_indices(self, embeddings, chunk_size=512):
        neighbors = []
        for start in range(0, len(embeddings), chunk_size):
            distances = torch.cdist(embeddings[start : start + chunk_size], embeddings)
            neighbors.append(distances.topk(self.neighbors_per_query, largest=False).indices.cpu())
        return torch.cat(neighbors, dim=0)

    def __iter__(self):
        for _ in range(len(self)):
            query_indices = torch.randperm(self.num_samples, generator=self.generator)[: self.queries_per_batch]
            yield self.neighbor_indices[query_indices].reshape(-1).tolist()

    def __len__(self):
        return self.num_samples // self.batch_size


def make_stml_train_loader(
    train_dataset,
    sampling_embeddings,
    batch_size,
    neighbors_per_query,
    seed,
    num_workers=8,
    start_method="spawn",
    pin_memory=True,
    graph_device="cpu",
):
    """Create the nearest-neighbor batch loader used by STML."""

    num_workers = dataloader_num_workers_for_dataset(train_dataset, num_workers)
    sampler = STMLNearestNeighborBatchSampler(
        embeddings=sampling_embeddings,
        batch_size=batch_size,
        neighbors_per_query=neighbors_per_query,
        seed=seed,
        device=graph_device,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        **make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=False,
            pin_memory=pin_memory,
        ),
    )
    logger.info(
        f"STML train loader: {len(train_dataset)} samples, {neighbors_per_query} neighbors/query, "
        f"{len(loader)} batches/epoch"
    )
    return loader


def validate_m_per_class_sampler_capacity(labels, batch_size, sampler_m):
    """Check that MPerClassSampler can build a complete training batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sampler_m <= 0:
        raise ValueError("sampler_m must be positive")
    if batch_size % sampler_m != 0:
        raise MPerClassSamplerCapacityError(
            "MPerClassSampler requires batch_size to be divisible by sampler_m: "
            f"batch_size={batch_size}, sampler_m={sampler_m}."
        )

    label_counts = {}
    for label in labels:
        label = int(label)
        label_counts[label] = label_counts.get(label, 0) + 1

    # MPerClassSampler samples with replacement when a class has fewer than m
    # examples. Its hard capacity constraint is enough distinct labels to fill
    # one complete batch.
    num_labels = len(label_counts)
    max_samples_per_sampler_pass = sampler_m * num_labels
    min_required_labels = int(np.ceil(batch_size / sampler_m))
    if max_samples_per_sampler_pass < batch_size:
        raise MPerClassSamplerCapacityError(
            "MPerClassSampler cannot build one training batch from the selected labeled data: "
            f"batch_size={batch_size}, sampler_m={sampler_m}, labeled_classes={num_labels}, "
            f"sampler_m*labeled_classes={max_samples_per_sampler_pass}. "
            f"Need at least {min_required_labels} labeled classes. "
            "For k samples from every training class, use label_sampling_mode='per_class_min'. "
            "For class_subset_k_shot, increase labeled_fraction so the class subset contains enough classes, "
            "or reduce batch_size/sampler_m."
        )


def make_sampler_epoch_length(dataset_size, batch_size, length_before_new_iter=None):
    if dataset_size <= 0:
        raise ValueError("training dataset must not be empty")
    if length_before_new_iter is not None:
        if length_before_new_iter < batch_size:
            raise ValueError("length_before_new_iter must be at least batch_size")
        # MPerClassSampler emits complete batches and applies the same rounding.
        return int(length_before_new_iter) - int(length_before_new_iter) % batch_size
    # Round the active dataset size up so the automatic mode emits full batches.
    return max(batch_size, int(np.ceil(dataset_size / batch_size) * batch_size))


def make_unlabeled_stream_loader(
    dataset,
    batch_size,
    seed,
    num_workers,
    start_method,
    supervised_loader=None,
    drop_last=True,
    persistent_workers=True,
    pin_memory=True,
    desc="regularizer",
):
    """Shuffled unlabeled stream, GPU-resident when the supervised loader is.

    The SSL regularizer streams follow the supervised loader's residency rather
    than threading the residency arguments through every regularizer. A resident
    stream draws its permutation from a ``RandomSampler`` over the DataLoader's
    own generator, so it visits exactly the batches ``shuffle=True`` would have.
    """

    dataloader_kwargs = make_dataloader_kwargs(
        num_workers,
        seed,
        start_method,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    gpu_resident_device = gpu_feature_loader.resident_loader_device(supervised_loader)
    if gpu_resident_device is not None:
        stream_loader = gpu_feature_loader.try_make_gpu_resident_loader(
            dataset,
            device=gpu_resident_device,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(
                dataset,
                generator=dataloader_kwargs["generator"],
            ),
            drop_last=drop_last,
            seed=seed,
            max_bytes=gpu_feature_loader.resident_loader_max_bytes(supervised_loader),
            desc=desc,
        )
        if stream_loader is not None:
            return stream_loader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **dataloader_kwargs,
    )


def make_eval_loader(
    dataset,
    batch_size=1024,
    seed=0,
    num_workers=8,
    start_method="spawn",
    pin_memory=True,
    gpu_resident_device=None,
    gpu_resident_max_bytes=None,
):
    # Evaluation traverses every item exactly once in dataset order.
    if gpu_resident_device is not None:
        # A sequential sampler reproduces shuffle=False exactly, and
        # drop_last=False keeps the ragged final batch.
        eval_loader = gpu_feature_loader.try_make_gpu_resident_loader(
            dataset,
            device=gpu_resident_device,
            batch_size=batch_size,
            sampler=range(len(dataset)),
            drop_last=False,
            seed=seed,
            max_bytes=gpu_resident_max_bytes,
            desc="eval",
            mode=gpu_feature_loader.RESIDENT_MODE_EVAL,
        )
        if eval_loader is not None:
            return eval_loader
    num_workers = dataloader_num_workers_for_dataset(dataset, num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=True,
            pin_memory=pin_memory,
        ),
    )


class PrecomputedBackboneFeatureDataset(Dataset):
    """Dataset backed by fixed backbone features instead of image tensors."""

    def __init__(
        self,
        features,
        labels,
        dense_labels=None,
        sample_weights=None,
        source_dataset=None,
        feature_indices=None,
        feature_matrix_path=None,
        residency=FEATURE_RESIDENCY_MMAP,
        residency_max_bytes=None,
        residency_desc=None,
    ):
        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        self._feature_matrix_path = (
            None
            if feature_matrix_path is None
            else str(Path(feature_matrix_path).resolve())
        )
        self._feature_matrix_shape = tuple(features.shape)
        self._feature_memmap = None
        self._resident = False
        if feature_indices is None:
            if features.ndim not in {2, 3}:
                raise ValueError("precomputed backbone features must be a 2D or 3D tensor")
            if len(features) != len(labels):
                raise ValueError("features and labels must have the same length")
            self._feature_matrix = features.contiguous()
            self.feature_indices = None
        else:
            if features.ndim != 2:
                raise ValueError("indexed precomputed features require a 2D backing matrix")
            feature_indices = torch.as_tensor(feature_indices, dtype=torch.long).cpu()
            if feature_indices.ndim not in {1, 2}:
                raise ValueError("feature_indices must be a 1D or 2D tensor")
            if len(feature_indices) != len(labels):
                raise ValueError("feature_indices and labels must have the same length")
            if feature_indices.numel() and (
                int(feature_indices.min()) < 0
                or int(feature_indices.max()) >= len(features)
            ):
                raise IndexError("feature_indices contains a row outside the backing matrix")
            self._feature_matrix = features
            self.feature_indices = feature_indices.contiguous()
            self._make_resident(residency, residency_max_bytes, residency_desc)
        self.orig_labels = [int(label) for label in labels]
        if dense_labels is None:
            dense_labels = self.orig_labels
        if len(dense_labels) != len(self.orig_labels):
            raise ValueError("dense_labels must match labels length")
        self.labels = [int(label) for label in dense_labels]
        if sample_weights is None:
            self.sample_weights = None
        else:
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32).cpu()
            if len(sample_weights) != len(self.orig_labels):
                raise ValueError("sample_weights must match labels length")
            self.sample_weights = sample_weights.contiguous()

        if source_dataset is not None:
            for attr_name in ("classes", "query_indices", "gallery_indices"):
                if hasattr(source_dataset, attr_name):
                    setattr(self, attr_name, getattr(source_dataset, attr_name))

    def _make_resident(self, residency, residency_max_bytes, desc):
        """Copy the rows this view uses out of the shared mmap, once.

        Random per-sample reads from a feature matrix larger than the page cache
        cost a page fault each, which dominates the step for big datasets. The
        view's own rows are usually a small fraction of the matrix, so lifting
        just those into a private tensor removes the file from the training loop
        without holding the whole matrix in memory.
        """

        if residency not in FEATURE_RESIDENCIES:
            raise ValueError(f"residency must be one of {FEATURE_RESIDENCIES}: {residency!r}")
        if residency == FEATURE_RESIDENCY_MMAP:
            return

        # torch.unique sorts, so the gather below reads the file in ascending
        # row order rather than jumping around it.
        used_rows, local_indices = torch.unique(
            self.feature_indices.reshape(-1),
            return_inverse=True,
        )
        resident_bytes = used_rows.numel() * self._feature_matrix_shape[1] * torch.float32.itemsize
        label = "precomputed features" if desc is None else desc
        if residency_max_bytes is not None and resident_bytes > residency_max_bytes:
            logger.warning(
                f"Keeping {label} memory-mapped: {len(used_rows)} rows would need "
                f"{resident_bytes / 1e9:.2f} GB, above the "
                f"{residency_max_bytes / 1e9:.2f} GB residency budget"
            )
            return

        self._feature_matrix = self._ensure_feature_matrix().index_select(0, used_rows).contiguous()
        self.feature_indices = local_indices.reshape(self.feature_indices.shape).contiguous()
        # The rows now live in this process, so nothing may reopen the file and
        # spawned workers would copy the tensor instead of sharing it.
        self._feature_matrix_path = None
        self._feature_memmap = None
        self._feature_matrix_shape = tuple(self._feature_matrix.shape)
        self._resident = True
        logger.info(
            f"Resident {label}: {len(used_rows)} unique rows "
            f"({resident_bytes / 1e9:.2f} GB) copied from the backbone feature mmap"
        )

    @property
    def is_resident(self):
        """Whether this view reads features from memory instead of the mmap."""

        return self._resident

    @property
    def supports_dataloader_workers(self):
        """Whether spawned workers can reopen storage without copying it."""

        return self._feature_matrix_path is not None

    def _ensure_feature_matrix(self):
        if self._feature_matrix is not None:
            return self._feature_matrix
        if self._feature_matrix_path is None:
            raise RuntimeError("precomputed feature storage is unavailable")

        feature_memmap = np.load(self._feature_matrix_path, mmap_mode="r+")
        if feature_memmap.dtype != np.float32:
            raise ValueError(
                "cached backbone feature matrix must use float32 storage"
            )
        if tuple(feature_memmap.shape) != self._feature_matrix_shape:
            raise ValueError(
                "cached backbone feature matrix shape changed after dataset creation: "
                f"expected {self._feature_matrix_shape}, got {tuple(feature_memmap.shape)}"
            )
        self._feature_memmap = feature_memmap
        self._feature_matrix = torch.from_numpy(feature_memmap)
        return self._feature_matrix

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._feature_matrix_path is not None:
            # Spawned DataLoader workers reopen the mmap directly. Omitting the
            # tensor prevents PyTorch from copying a potentially multi-GB
            # feature matrix into multiprocessing shared memory.
            state["_feature_matrix"] = None
            state["_feature_memmap"] = None
        return state

    @property
    def features(self):
        """Expose aligned features while keeping indexed training access lazy."""

        feature_matrix = self._ensure_feature_matrix()
        if self.feature_indices is None:
            return feature_matrix
        return feature_matrix[self.feature_indices]

    def __len__(self):
        return len(self.orig_labels)

    def __getitem__(self, index):
        feature_matrix = self._ensure_feature_matrix()
        if self.feature_indices is None:
            features = feature_matrix[index]
            if features.ndim == 2:
                view_index = int(torch.randint(features.shape[0], ()).item())
                features = features[view_index]
        else:
            matrix_rows = self.feature_indices[index]
            if matrix_rows.ndim == 0:
                features = feature_matrix[matrix_rows]
            else:
                view_index = int(torch.randint(len(matrix_rows), ()).item())
                features = feature_matrix[matrix_rows[view_index]]
        if self.sample_weights is None:
            return features, self.orig_labels[index]
        return features, self.orig_labels[index], self.sample_weights[index]

    def __getitems__(self, indices):
        """Gather deterministic cached rows once per DataLoader batch.

        Keep stochastic views on the item path to preserve per-sample RNG
        consumption, including wrappers that request multiple views.
        The ordinary collator still owns stacking and worker shared memory.
        """

        feature_matrix = self._ensure_feature_matrix()
        if feature_matrix.ndim == 3 or (
            self.feature_indices is not None and self.feature_indices.ndim == 2
        ):
            return [self[index] for index in indices]
        indices = torch.as_tensor(indices, dtype=torch.long)
        rows = indices if self.feature_indices is None else self.feature_indices[indices]
        features = feature_matrix[rows].unbind(0)
        labels = [self.orig_labels[index] for index in indices.tolist()]
        if self.sample_weights is None:
            return list(zip(features, labels))
        weights = self.sample_weights[indices].unbind(0)
        return list(zip(features, labels, weights))

    def storage_nbytes(self):
        if self.feature_indices is None or self._resident:
            total = int(np.prod(self._feature_matrix_shape)) * torch.float32.itemsize
            if self.feature_indices is not None:
                total += self.feature_indices.numel() * self.feature_indices.element_size()
        else:
            # The float matrix is persistent shared mmap storage, not an
            # allocation owned by this dataset view.
            total = self.feature_indices.numel() * self.feature_indices.element_size()
        if self.sample_weights is not None:
            total += self.sample_weights.numel() * self.sample_weights.element_size()
        return int(total)


class RepeatedAugmentedViewDataset(Dataset):
    """Repeat each dataset item consecutively to materialize stochastic views."""

    def __init__(self, dataset, num_views):
        self.dataset = dataset
        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")

    def __len__(self):
        return len(self.dataset) * self.num_views

    def __getitem__(self, index):
        return self.dataset[int(index) // self.num_views]


def _precomputed_feature_worker_safety(dataset, seen=None):
    """Return ``(found_precomputed_features, all_are_worker_safe)``."""

    if seen is None:
        seen = set()
    object_id = id(dataset)
    if object_id in seen:
        return False, True
    seen.add(object_id)

    if isinstance(dataset, PrecomputedBackboneFeatureDataset):
        return True, dataset.supports_dataloader_workers

    found = False
    worker_safe = True
    child_datasets = getattr(dataset, "datasets", None)
    if child_datasets is not None:
        for child_dataset in child_datasets:
            child_found, child_safe = _precomputed_feature_worker_safety(
                child_dataset,
                seen,
            )
            found = found or child_found
            if child_found:
                worker_safe = worker_safe and child_safe

    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        child_found, child_safe = _precomputed_feature_worker_safety(
            child_dataset,
            seen,
        )
        found = found or child_found
        if child_found:
            worker_safe = worker_safe and child_safe
    return found, worker_safe


def dataset_has_precomputed_backbone_features(dataset):
    """Return True when a dataset is backed by precomputed features."""

    found, _ = _precomputed_feature_worker_safety(dataset)
    return found


def dataloader_num_workers_for_dataset(dataset, num_workers):
    has_precomputed_features, worker_safe = _precomputed_feature_worker_safety(
        dataset
    )
    if has_precomputed_features and not worker_safe:
        # Ordinary in-memory feature tensors can be duplicated or copied into
        # multiprocessing shared memory. Persistent cache datasets instead
        # reopen their mmap in each worker and keep the requested worker count.
        return 0
    return num_workers


def get_nested_feature_transform(dataset):
    """Return the deterministic feature transform attached under wrapper datasets."""

    feature_transform = getattr(dataset, "feature_transform", None)
    if feature_transform is not None:
        return feature_transform
    if isinstance(dataset, Subset):
        return get_nested_feature_transform(dataset.dataset)
    if isinstance(dataset, CombinedDataset):
        for child_dataset in dataset.datasets:
            feature_transform = get_nested_feature_transform(child_dataset)
            if feature_transform is not None:
                return feature_transform
        return None
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        return get_nested_feature_transform(child_dataset)
    return None


def make_feature_transform_dataset(dataset, require_feature_transform=False):
    """Copy a dataset and force its deterministic feature transform."""

    feature_dataset = copy.deepcopy(dataset)
    feature_transform = get_nested_feature_transform(feature_dataset)
    if feature_transform is None and require_feature_transform:
        raise ValueError("Frozen feature precompute requires a deterministic feature_transform")
    if feature_transform is not None:
        set_nested_transform(feature_dataset, feature_transform)
    return feature_dataset


def split_optional_weight_batch(batch):
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels, None
    if len(batch) == 3:
        inputs, labels, sample_weights = batch
        return inputs, labels, sample_weights
    raise ValueError(f"Expected a 2- or 3-item batch, got {len(batch)} items")


def is_precomputed_feature_batch(inputs):
    return torch.is_tensor(inputs) and inputs.ndim == 2


def forward_model_inputs(model, inputs, device, use_cache=False):
    """Forward either image batches or precomputed backbone feature batches."""
    if is_precomputed_feature_batch(inputs) and hasattr(model, "project_features"):
        return model.project_features(inputs.to(device, non_blocking=True))
    if use_cache and hasattr(model, "forward_cached"):
        return model.forward_cached(inputs, device)
    return model(inputs.to(device, non_blocking=True))


def forward_model_inputs_with_trunk(model, inputs, device, use_cache=False):
    """``forward_model_inputs`` plus the trunk tensors an embedding loss can act in.

    Returns ``(embedding, pre_norm, hidden)``: the normalized retrieval
    embedding, the projection head's output before that normalization, and one
    activation per entry of the model's ``auxiliary_embedding_layers``. The three
    input paths mirror :func:`forward_model_inputs` exactly, so a precomputed
    feature batch still enters below the backbone.
    """

    if is_precomputed_feature_batch(inputs) and hasattr(
        model, "project_features_with_trunk"
    ):
        return model.project_features_with_trunk(
            inputs.to(device, non_blocking=True)
        )
    if use_cache and hasattr(model, "forward_cached_with_trunk"):
        return model.forward_cached_with_trunk(inputs, device)
    if not hasattr(model, "forward_with_trunk"):
        raise TypeError(
            f"{type(model).__name__} exposes no forward_with_trunk, so an "
            "embedding loss cannot reach anything but its output"
        )
    return model.forward_with_trunk(inputs.to(device, non_blocking=True))


def _take_dataset_values(values, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if torch.is_tensor(values):
        return values.index_select(0, torch.as_tensor(indices, dtype=torch.long))
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(index)] for index in indices]


def _resolve_aligned_dataset_attribute(dataset, names, seen=None):
    """Resolve metadata through Subset/selection wrappers without loading images."""

    if seen is None:
        seen = set()
    object_id = id(dataset)
    if object_id in seen:
        return None
    seen.add(object_id)

    for name in names:
        value = getattr(dataset, name, None)
        if value is None:
            continue
        try:
            if len(value) == len(dataset):
                return value
        except TypeError:
            continue

    child = getattr(dataset, "dataset", None)
    if child is None or child is dataset:
        return None
    child_values = _resolve_aligned_dataset_attribute(child, names, seen)
    if child_values is None:
        return None
    if isinstance(dataset, Subset):
        return _take_dataset_values(child_values, dataset.indices)
    positions = getattr(dataset, "positions", None)
    if positions is not None and len(positions) == len(dataset):
        return _take_dataset_values(child_values, positions)
    return None


def _precomputed_dataset_metadata(dataset):
    labels = _resolve_aligned_dataset_attribute(
        dataset,
        ("orig_labels", "labels", "targets"),
    )
    if labels is None:
        # This fallback keeps custom Dataset implementations compatible. Normal
        # repository datasets expose aligned label arrays and never take this
        # image-decoding path on mmap hits.
        logger.warning(
            "Dataset exposes no aligned label metadata; reading items to build "
            "the precomputed feature labels"
        )
        labels = []
        weights = []
        has_weights = None
        for index in range(len(dataset)):
            item = dataset[index]
            labels.append(item[1])
            item_has_weights = len(item) >= 3
            if has_weights is None:
                has_weights = item_has_weights
            elif has_weights != item_has_weights:
                raise ValueError("dataset items inconsistently expose sample weights")
            if item_has_weights:
                weights.append(item[2])
        sample_weights = (
            torch.as_tensor(weights, dtype=torch.float32)
            if has_weights
            else None
        )
    else:
        sample_weights = _resolve_aligned_dataset_attribute(
            dataset,
            ("sample_weights", "confidences"),
        )

    dense_labels = getattr(dataset, "labels", None)
    if dense_labels is not None and len(dense_labels) != len(dataset):
        dense_labels = None
    return labels, dense_labels, sample_weights


def _compute_backbone_feature_rows(
    model,
    dataset,
    local_positions,
    *,
    device,
    batch_size,
    seed,
    num_workers,
    start_method,
    pin_memory,
):
    """Decode and forward only the rows claimed missing by the mmap."""

    local_positions = np.asarray(local_positions, dtype=np.int64)
    missing_dataset = Subset(dataset, local_positions.tolist())
    loader = DataLoader(
        missing_dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=False,
            pin_memory=pin_memory,
        ),
    )
    features = []
    try:
        for batch in loader:
            images = batch[0]
            output = model.forward_backbone(images.to(device, non_blocking=True))
            features.append(output.detach().float().cpu())
    finally:
        shutdown_dataloader_workers(loader)
    if not features:
        raise RuntimeError("indexed feature cache requested an empty missing-row computation")
    return torch.cat(features, dim=0)


def precompute_backbone_feature_dataset(
    model,
    dataset,
    device,
    batch_size,
    seed,
    num_workers,
    start_method,
    desc,
    pin_memory=True,
    require_feature_transform=False,
    use_feature_transform=True,
    num_views=1,
    cache_key=None,
    cache_indices=None,
    cache_size=None,
    residency=FEATURE_RESIDENCY_MMAP,
    residency_max_bytes=None,
):
    """Extract frozen raw backbone features once and keep dataset labels aligned."""

    if not hasattr(model, "forward_backbone"):
        raise AttributeError("Model does not expose forward_backbone for feature precompute")
    num_views = int(num_views)
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if use_feature_transform:
        feature_dataset = make_feature_transform_dataset(
            dataset,
            require_feature_transform=require_feature_transform,
        )
    elif require_feature_transform:
        raise ValueError("require_feature_transform cannot be combined with use_feature_transform=False")
    else:
        feature_dataset = copy.deepcopy(dataset)
    source_length = len(feature_dataset)
    loader_dataset = feature_dataset
    if num_views > 1:
        loader_dataset = RepeatedAugmentedViewDataset(feature_dataset, num_views)

    materialize_cached = getattr(model, "materialize_cached_backbone_features", None)
    indexed_cache_enabled = (
        materialize_cached is not None
        and cache_key is not None
        and cache_indices is not None
        and cache_size is not None
    )
    if indexed_cache_enabled:
        cache_indices = np.asarray(cache_indices, dtype=np.int64)
        if len(cache_indices) != len(loader_dataset):
            raise ValueError("cache_indices must align with every precompute input row")

        labels, dense_labels, sample_weights = _precomputed_dataset_metadata(
            feature_dataset
        )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), eval_autocast(device):
                feature_matrix, feature_indices = materialize_cached(
                    cache_key=cache_key,
                    cache_indices=cache_indices,
                    cache_size=cache_size,
                    compute_missing=lambda local_positions: _compute_backbone_feature_rows(
                        model,
                        loader_dataset,
                        local_positions,
                        device=device,
                        batch_size=batch_size,
                        seed=seed,
                        num_workers=num_workers,
                        start_method=start_method,
                        pin_memory=pin_memory,
                    ),
                )
                matrix_path_resolver = getattr(
                    model,
                    "cached_backbone_feature_matrix_path",
                    None,
                )
                feature_matrix_path = (
                    matrix_path_resolver(
                        cache_key=cache_key,
                        cache_size=cache_size,
                    )
                    if callable(matrix_path_resolver)
                    else None
                )
        finally:
            if was_training:
                model.train()

        if num_views > 1:
            feature_indices = feature_indices.reshape(source_length, num_views)
        return PrecomputedBackboneFeatureDataset(
            features=feature_matrix,
            feature_indices=feature_indices,
            labels=labels,
            dense_labels=dense_labels,
            sample_weights=sample_weights,
            source_dataset=dataset,
            feature_matrix_path=feature_matrix_path,
            residency=residency,
            residency_max_bytes=residency_max_bytes,
            residency_desc=desc,
        )

    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=False,
            pin_memory=pin_memory,
        ),
    )

    was_training = model.training
    model.eval()
    all_features = []
    all_labels = []
    all_sample_weights = []
    saw_sample_weights = False
    with torch.no_grad(), eval_autocast(device):
        for batch in tqdm(loader, desc=desc):
            images, labels, sample_weights = split_optional_weight_batch(batch)
            forward_backbone_cached = getattr(model, "forward_backbone_cached", None)
            if forward_backbone_cached is None:
                features = model.forward_backbone(images.to(device, non_blocking=True))
            else:
                features = forward_backbone_cached(images, device)
            all_features.append(features.detach().float().cpu())
            all_labels.append(torch.as_tensor(labels, dtype=torch.long).cpu())
            if sample_weights is not None:
                saw_sample_weights = True
                all_sample_weights.append(torch.as_tensor(sample_weights, dtype=torch.float32).cpu())
    if was_training:
        model.train()

    features = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    sample_weights = torch.cat(all_sample_weights, dim=0) if saw_sample_weights else None
    if num_views > 1:
        expected_rows = source_length * num_views
        if len(features) != expected_rows or len(labels_tensor) != expected_rows:
            raise ValueError("precomputed augmented feature rows do not match dataset length and num_views")
        if features.ndim != 2:
            raise ValueError("augmented backbone features must be a 2D row matrix before grouping")
        features = features.reshape(source_length, num_views, features.shape[1])
        label_groups = labels_tensor.reshape(source_length, num_views)
        if not torch.equal(label_groups, label_groups[:, :1].expand_as(label_groups)):
            raise ValueError("labels changed across augmented views for the same sample")
        labels = label_groups[:, 0].tolist()
        if sample_weights is not None:
            sample_weight_groups = sample_weights.reshape(source_length, num_views)
            if not torch.allclose(sample_weight_groups, sample_weight_groups[:, :1].expand_as(sample_weight_groups)):
                raise ValueError("sample weights changed across augmented views for the same sample")
            sample_weights = sample_weight_groups[:, 0]
    else:
        labels = labels_tensor.tolist()
    dense_labels = getattr(dataset, "labels", None)
    if dense_labels is not None and len(dense_labels) != len(labels):
        raise ValueError("dataset.labels must align with the precomputed feature rows")
    return PrecomputedBackboneFeatureDataset(
        features=features,
        labels=labels,
        dense_labels=dense_labels,
        sample_weights=sample_weights,
        source_dataset=dataset,
    )


def use_feature_transform_for_training(dataset):
    """Replace stochastic training augmentation with the deterministic feature transform."""

    feature_transform = get_nested_feature_transform(dataset)
    if feature_transform is None:
        raise ValueError("Cached frozen-backbone training requires a deterministic feature_transform")
    set_nested_transform(dataset, feature_transform)


def setup_datasets(
    dataset_name,
    batch_size,
    sampler_m,
    seed=0,
    num_workers=8,
    start_method="spawn",
    cv_k=1,
    cv_fold=None,
    cv_mode="group_kfold",
    val_mode=VAL_MODE_ALL,
    dataset_protocol=DATASET_PROTOCOL_OFFICIAL,
    cifar_imbalance_factor=None,
    cifar_train_fraction=0.8,
    cifar_test_fraction=0.2,
    image_resize_mode=DEFAULT_IMAGE_RESIZE_MODE,
):
    dataset_bundle = setup_dataset_bundle(
        dataset_name=dataset_name,
        seed=seed,
        cv_k=cv_k,
        cv_fold=cv_fold,
        cv_mode=cv_mode,
        val_mode=val_mode,
        dataset_protocol=dataset_protocol,
        cifar_imbalance_factor=cifar_imbalance_factor,
        cifar_train_fraction=cifar_train_fraction,
        cifar_test_fraction=cifar_test_fraction,
        image_resize_mode=image_resize_mode,
    )
    train_loader = make_train_loader(dataset_bundle.train_dataset, batch_size, sampler_m, seed, num_workers, start_method)
    valid_loader = make_eval_loader(dataset_bundle.valid_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    test_loader = make_eval_loader(dataset_bundle.test_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    return train_loader, valid_loader, test_loader, dataset_bundle.train_labels_mapper


def _selected_faiss_gpu(device):
    """Return ``(faiss, gpu_id)`` when the requested CUDA device supports FAISS."""

    device = torch.device(normalize_device_name(device))
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        import faiss
    except (ImportError, OSError):
        return None
    required_gpu_api = ("get_num_gpus", "StandardGpuResources", "index_cpu_to_gpu")
    if not all(hasattr(faiss, name) for name in required_gpu_api):
        return None
    try:
        gpu_count = int(faiss.get_num_gpus())
        gpu_id = int(torch.cuda.current_device()) if device.index is None else int(device.index)
    except Exception as exc:
        logger.debug(f"FAISS GPU discovery failed during retrieval setup: {exc}")
        return None
    if not 0 <= gpu_id < gpu_count:
        return None
    return faiss, gpu_id


def available_cuda_retrieval_device(device):
    """Return the requested/current CUDA device when GPU FAISS is available."""

    requested_device = torch.device(normalize_device_name(device))
    cuda_device = (
        requested_device
        if requested_device.type == "cuda"
        else torch.device("cuda")
    )
    if _selected_faiss_gpu(cuda_device) is None:
        return None
    return cuda_device


def frozen_feature_cuda_evaluation_device(dataset, device):
    """Return the CUDA retrieval device for an evaluation, if usable.

    ``device`` is the model/training device. CPU training and small validation
    sets must not force retrieval onto the CPU, so dataset size and storage type
    deliberately do not participate in this decision.
    """

    if dataset is None:
        return None
    return available_cuda_retrieval_device(device)


def resolve_evaluation_retrieval_device(dataset, device, backend=None):
    """Resolve an automatic or explicitly inherited retrieval backend."""

    if backend is None:
        return frozen_feature_cuda_evaluation_device(dataset, device) or torch.device(
            "cpu"
        )
    backend = str(backend).lower()
    if backend not in EVALUATION_RETRIEVAL_BACKENDS:
        raise ValueError(
            "evaluation retrieval backend must be one of "
            f"{EVALUATION_RETRIEVAL_BACKENDS}: {backend!r}"
        )
    if backend == EVALUATION_RETRIEVAL_BACKEND_CPU:
        return torch.device("cpu")
    cuda_device = available_cuda_retrieval_device(device)
    if cuda_device is None:
        raise RuntimeError(
            "CUDA retrieval was inherited from validation, but a GPU FAISS "
            "backend is not available for final evaluation"
        )
    return cuda_device


def frozen_feature_cuda_evaluation_enabled(dataset, device):
    """Whether this evaluation can use the automatic CUDA retrieval backend."""

    return frozen_feature_cuda_evaluation_device(dataset, device) is not None


def frozen_feature_eval_batch_size(dataset, device, default_batch_size=32):
    """Use large projection batches only for large cached frozen-feature sets."""

    device = torch.device(normalize_device_name(device))
    large_precomputed_dataset = (
        dataset is not None
        and len(dataset) > FROZEN_LARGE_EVAL_MIN_SAMPLES
        and dataset_has_precomputed_backbone_features(dataset)
    )
    if large_precomputed_dataset and (
        device.type == "cuda"
        or available_cuda_retrieval_device(device) is not None
    ):
        return min(FROZEN_CUDA_EVAL_BATCH_SIZE, len(dataset))
    return int(default_batch_size)


class _ReusableSingleGpuFaissKNN:
    """Run exact single-GPU FAISS retrieval with an optional low-memory policy."""

    def __init__(
        self,
        faiss_module,
        gpu_id,
        low_memory=False,
        temp_memory_bytes=None,
        pinned_memory_bytes=None,
    ):
        self.faiss = faiss_module
        self.gpu_id = int(gpu_id)
        self.low_memory = bool(low_memory)
        self.resources = self.faiss.StandardGpuResources()
        # StandardGpuResources otherwise reserves 0.5-1.5 GiB of device scratch
        # (depending on GPU size) plus a 256 MiB pinned host buffer in every
        # process, and holds both for its lifetime. Sizing them here keeps
        # concurrent HPO runs from each pinning that much device and host memory.
        if temp_memory_bytes is None and self.low_memory:
            # Host-resident retrieval never needed a persistent arena.
            temp_memory_bytes = 0
        self._set_resource_buffer("setTempMemory", temp_memory_bytes)
        self._set_resource_buffer("setPinnedMemory", pinned_memory_bytes)
        self.index = None
        self.dimension = None
        self._lock = threading.Lock()

    def _set_resource_buffer(self, method_name, size_bytes):
        """Size one FAISS resource buffer, if this build exposes the setter."""

        if size_bytes is None:
            return
        setter = getattr(self.resources, method_name, None)
        if callable(setter):
            setter(int(size_bytes))

    def _reset_index(self, dimension):
        dimension = int(dimension)
        if self.index is None or self.dimension != dimension:
            cpu_index = self.faiss.IndexFlatL2(dimension)
            self.index = self.faiss.index_cpu_to_gpu(
                self.resources,
                self.gpu_id,
                cpu_index,
            )
            self.dimension = dimension
        else:
            self.index.reset()

    def _torch_exact_search(self, query, reference, search_k):
        """Return exact neighbors on CUDA when FAISS-GPU cannot serve k."""

        if not torch.is_tensor(query):
            query = torch.as_tensor(query, dtype=torch.float32)
        if not torch.is_tensor(reference):
            reference = torch.as_tensor(reference, dtype=torch.float32)
        output_device = query.device
        if query.device.type == "cuda":
            compute_device = query.device
        elif torch.cuda.is_available():
            compute_device = torch.device("cuda", self.gpu_id)
        else:
            # This branch primarily keeps CPU-only unit tests useful. A real
            # instance is created only after GPU FAISS/CUDA discovery succeeds.
            compute_device = query.device
        reference_on_compute = reference.to(
            compute_device,
            dtype=torch.float32,
        ).contiguous()

        bytes_per_query = max(
            1,
            len(reference_on_compute) * reference_on_compute.element_size()
            + search_k
            * (reference_on_compute.element_size() + torch.tensor([], dtype=torch.long).element_size()),
        )
        chunk_size = max(
            1,
            min(
                len(query),
                TORCH_EXACT_KNN_MAX_DISTANCE_BYTES // bytes_per_query,
            ),
        )
        distance_chunks = []
        index_chunks = []
        try:
            for start in range(0, len(query), chunk_size):
                query_chunk = query[start : start + chunk_size].to(
                    compute_device,
                    dtype=torch.float32,
                )
                pairwise_distances = torch.cdist(
                    query_chunk,
                    reference_on_compute,
                ).square_()
                distances, indices = torch.topk(
                    pairwise_distances,
                    k=search_k,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                distance_chunks.append(distances.to(output_device))
                index_chunks.append(indices.to(output_device))
            return torch.cat(distance_chunks), torch.cat(index_chunks)
        finally:
            if (
                self.low_memory
                and compute_device.type == "cuda"
                and output_device.type == "cpu"
            ):
                # The high-k fallback uses PyTorch rather than FAISS. Return its
                # cached workspace to the driver so sibling HPO processes can
                # reuse the memory after this evaluation.
                reference_on_compute = None
                query_chunk = None
                pairwise_distances = None
                distances = None
                indices = None
                with torch.cuda.device(compute_device):
                    torch.cuda.empty_cache()

    @staticmethod
    def _faiss_input(value):
        """Give FAISS host NumPy arrays or device-resident torch tensors."""

        if torch.is_tensor(value):
            value = value.detach().contiguous()
            if value.device.type == "cpu":
                return np.ascontiguousarray(value.numpy(), dtype=np.float32)
            return value
        return np.ascontiguousarray(value, dtype=np.float32)

    def _discard_index(self):
        """Drop the gallery allocation instead of retaining it between epochs."""

        index = self.index
        self.index = None
        self.dimension = None
        if index is not None:
            reset = getattr(index, "reset", None)
            if callable(reset):
                reset()

    def __call__(
        self,
        query,
        k,
        reference=None,
        ref_includes_query=False,
    ):
        reference = query if reference is None else reference
        search_k = int(k) + int(ref_includes_query)
        if search_k > FAISS_GPU_MAX_K:
            compute_device = (
                query.device
                if torch.is_tensor(query) and query.device.type == "cuda"
                else torch.device("cuda", self.gpu_id)
            )
            logger.debug(
                f"FAISS GPU supports at most k={FAISS_GPU_MAX_K}; using "
                f"chunked exact torch retrieval on {compute_device} for k={search_k}"
            )
            distances, indices = self._torch_exact_search(
                query,
                reference,
                search_k,
            )
            return return_results(distances, indices, ref_includes_query)
        result_device = query.device if torch.is_tensor(query) else torch.device("cpu")
        query_input = self._faiss_input(query)
        reference_input = self._faiss_input(reference)
        with self._lock:
            self._reset_index(reference_input.shape[1])
            index = self.index
            try:
                # The trainable projection head can change after every optimizer
                # step, so replace the index contents for every evaluation.
                index.add(reference_input)
                distances, indices = index.search(query_input, search_k)
            finally:
                if self.low_memory:
                    self._discard_index()
        distances = torch.as_tensor(distances, device=result_device)
        indices = torch.as_tensor(indices, device=result_device)
        return return_results(distances, indices, ref_includes_query)


_REUSABLE_FAISS_KNN = {}
_REUSABLE_FAISS_KNN_LOCK = threading.Lock()


def _get_reusable_faiss_knn(faiss_module, gpu_id, low_memory=False):
    temp_memory_bytes, pinned_memory_bytes = _faiss_memory_policy(low_memory)
    # The buffer sizes are part of the identity: a wrapper built under a
    # different policy already holds resources sized for that policy.
    cache_key = (
        id(faiss_module),
        int(gpu_id),
        bool(low_memory),
        temp_memory_bytes,
        pinned_memory_bytes,
    )
    with _REUSABLE_FAISS_KNN_LOCK:
        knn = _REUSABLE_FAISS_KNN.get(cache_key)
        if knn is None:
            knn = _ReusableSingleGpuFaissKNN(
                faiss_module,
                gpu_id,
                low_memory=low_memory,
                temp_memory_bytes=temp_memory_bytes,
                pinned_memory_bytes=pinned_memory_bytes,
            )
            _REUSABLE_FAISS_KNN[cache_key] = knn
    return knn


def _cuda_retrieval_backend(embeddings, device):
    del embeddings
    if torch.device(normalize_device_name(device)).type != "cuda":
        return None
    return _selected_faiss_gpu(device)


def extract_eval_embeddings(
    model,
    eval_loader,
    name="test set",
    device="cuda",
    keep_on_device=None,
    output_device=None,
):
    """Embed a dataset once, optionally retaining embeddings on another device."""

    # eval() disables training-only behavior such as dropout and updates to
    # normalization statistics.
    model = model.eval()
    model_device = torch.device(normalize_device_name(device))
    if output_device is not None:
        output_device = torch.device(normalize_device_name(output_device))
    cuda_evaluation_device = None
    if keep_on_device is None:
        if output_device is None:
            cuda_evaluation_device = frozen_feature_cuda_evaluation_device(
                getattr(eval_loader, "dataset", None),
                model_device,
            )
            keep_on_device = cuda_evaluation_device is not None
        else:
            keep_on_device = True
    elif keep_on_device and output_device is None:
        cuda_evaluation_device = frozen_feature_cuda_evaluation_device(
            getattr(eval_loader, "dataset", None),
            model_device,
        )
    if output_device is None:
        output_device = cuda_evaluation_device or model_device
    if keep_on_device:
        dataset = getattr(eval_loader, "dataset", None)
        sample_count = "all" if dataset is None else f"{len(dataset)}"
        logger.info(
            f"{name}: retaining {sample_count} projected embeddings "
            f"on {output_device} for device-resident retrieval "
            f"(model inference: {model_device})"
        )
    all_embeddings = []
    all_labels = []
    # Extract embeddings and labels
    progress = None
    try:
        with torch.no_grad(), eval_autocast(device):
            progress = tqdm(eval_loader, desc=name)
            for images, labels in progress:
                forward_cached = getattr(model, "forward_cached", None)
                embeddings = forward_model_inputs(
                    model,
                    images,
                    device,
                    use_cache=forward_cached is not None,
                )
                if keep_on_device:
                    all_embeddings.append(
                        embeddings.detach().to(
                            output_device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                    )
                    all_labels.append(
                        labels.detach().to(output_device, non_blocking=True)
                    )
                else:
                    # Host-resident evaluation releases accelerator outputs
                    # after every batch, even when retrieval later uses CUDA.
                    # .float() before .numpy(): numpy has no bfloat16, so an
                    # autocast embedding must leave torch as float32.
                    all_embeddings.append(
                        embeddings.detach().float().cpu().numpy().astype(np.float32)
                    )
                    all_labels.append(labels.detach().cpu().numpy())
    except BaseException:
        shutdown_dataloader_workers(eval_loader)
        raise
    finally:
        if progress is not None:
            progress.close()
    if keep_on_device:
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
    else:
        all_embeddings = np.concatenate(all_embeddings)
        all_labels = np.concatenate(all_labels)
    validate_finite_embeddings(all_embeddings, name)
    return all_embeddings, all_labels


def validate_finite_embeddings(all_embeddings, name="test set"):
    if torch.is_tensor(all_embeddings):
        finite = torch.isfinite(all_embeddings)
        if bool(finite.all()):
            return
        total_values = int(all_embeddings.numel())
        nonfinite_values = int(total_values - finite.sum().item())
        nan_values = int(torch.isnan(all_embeddings).sum().item())
        inf_values = int(torch.isinf(all_embeddings).sum().item())
    else:
        finite = np.isfinite(all_embeddings)
        if finite.all():
            return
        total_values = int(all_embeddings.size)
        nonfinite_values = int(total_values - finite.sum())
        nan_values = int(np.isnan(all_embeddings).sum())
        inf_values = int(np.isinf(all_embeddings).sum())
    raise NonFiniteEmbeddingError(
        f"{name} produced non-finite embeddings before retrieval metric calculation: "
        f"{nonfinite_values}/{total_values} values are non-finite "
        f"({nan_values} NaN, {inf_values} +/-Inf). "
        "This usually indicates that the model diverged for the current hyperparameters."
    )


def _as_numpy(values, dtype=None):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def _take_evaluation_rows(values, indices):
    if torch.is_tensor(values):
        indices = torch.as_tensor(indices, dtype=torch.long, device=values.device)
        return values.index_select(0, indices)
    return values[indices]


def get_query_gallery_indices(dataset, num_embeddings):
    """Return query/gallery indices when a dataset exposes a retrieval split."""

    if dataset is None:
        return None
    query_indices = getattr(dataset, "query_indices", None)
    gallery_indices = getattr(dataset, "gallery_indices", None)
    if query_indices is None or gallery_indices is None:
        return None

    query_indices = np.asarray(query_indices, dtype=np.int64)
    gallery_indices = np.asarray(gallery_indices, dtype=np.int64)
    if query_indices.ndim != 1 or gallery_indices.ndim != 1:
        raise ValueError("query_indices and gallery_indices must be one-dimensional")
    if len(query_indices) == 0 or len(gallery_indices) == 0:
        raise ValueError("query/gallery evaluation requires non-empty query and gallery partitions")
    max_index = max(int(query_indices.max()), int(gallery_indices.max()))
    min_index = min(int(query_indices.min()), int(gallery_indices.min()))
    if min_index < 0 or max_index >= num_embeddings:
        raise ValueError("query_indices/gallery_indices are out of range for the evaluated embeddings")
    return query_indices, gallery_indices


def aligned_dataset_labels(dataset):
    """Return the per-row labels an evaluation pass over ``dataset`` will see."""

    return _resolve_aligned_dataset_attribute(dataset, ("labels", "targets", "orig_labels"))


def apply_validation_query_gallery_split(dataset_bundle, gallery_fraction, seed):
    """Give the validation split a derived query/gallery partition.

    Only the test split of a retrieval dataset ships a query/gallery partition,
    so validation is scored same-source by default: every validation embedding
    is both a query and a reference. Deriving a partition here makes validation
    measure the same thing the final test does, at the cost of a smaller
    reference set. The partition is attached to the dataset itself, which is the
    single input :func:`get_query_gallery_indices` reads, so it reaches both the
    plain and the precomputed-feature evaluation paths unchanged.
    """

    valid_dataset = dataset_bundle.valid_dataset
    if valid_dataset is None or len(valid_dataset) == 0:
        raise ValueError(
            "validation_retrieval_mode='query_gallery' requires a non-empty validation split"
        )
    if getattr(valid_dataset, "query_indices", None) is not None:
        # A split that already carries a protocol partition keeps its own.
        return dataset_bundle

    labels = aligned_dataset_labels(valid_dataset)
    if labels is None:
        raise ValueError(
            "validation_retrieval_mode='query_gallery' requires a validation dataset that "
            "exposes per-row labels aligned with its evaluation order"
        )
    query_indices, gallery_indices, info = derive_query_gallery_indices(
        labels,
        gallery_fraction=gallery_fraction,
        seed=seed,
    )
    valid_dataset.query_indices = query_indices
    valid_dataset.gallery_indices = gallery_indices
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["validation_retrieval"] = info
    logger.info(
        "Validation retrieval mode query_gallery: "
        f"{info['num_queries']} queries across {info['num_query_classes']} classes against "
        f"{info['num_gallery']} gallery samples across {info['num_gallery_classes']} classes "
        f"(gallery_fraction={info['gallery_fraction']}, seed={info['seed']})"
    )
    if info["gallery_only_classes"]:
        # Those samples still answer other queries, but nothing can retrieve
        # them, so the query count is below the validation size by more than
        # the gallery fraction alone explains.
        logger.warning(
            f"{info['gallery_only_classes']} validation classes hold a single sample and "
            "became gallery-only; they contribute no queries"
        )
    return dataset_bundle


def make_evaluation_embedding_sets(all_embeddings, all_labels, dataset=None):
    """Build query/reference arrays for same-source or query-gallery retrieval."""

    if torch.is_tensor(all_embeddings):
        all_embeddings = all_embeddings.to(dtype=torch.float32)
        all_labels = torch.as_tensor(
            all_labels,
            device=all_embeddings.device,
        ).reshape(-1)
    else:
        all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
        all_labels = np.asarray(all_labels).reshape(-1)
    if len(all_embeddings) != len(all_labels):
        raise ValueError("embeddings and labels must have the same length")

    query_gallery_indices = get_query_gallery_indices(dataset, len(all_embeddings))
    if query_gallery_indices is None:
        return {
            "mode": SAME_SOURCE_EVALUATION,
            "query_embeddings": all_embeddings,
            "query_labels": all_labels,
            "reference_embeddings": None,
            "reference_labels": None,
            "ref_includes_query": True,
        }

    query_indices, gallery_indices = query_gallery_indices
    return {
        "mode": QUERY_GALLERY_EVALUATION,
        "query_embeddings": _take_evaluation_rows(all_embeddings, query_indices),
        "query_labels": _take_evaluation_rows(all_labels, query_indices),
        "reference_embeddings": _take_evaluation_rows(all_embeddings, gallery_indices),
        "reference_labels": _take_evaluation_rows(all_labels, gallery_indices),
        "ref_includes_query": False,
    }


def _vectorized_get_relevance_mask(
    shape,
    gt_labels,
    ref_includes_query,
    label_counts,
):
    """Drop-in replacement for ``accuracy_calculator.get_relevance_mask``.

    Upstream builds the mask with one Python iteration per unique class, and
    every iteration launches a handful of tiny CUDA kernels. That loop is launch
    bound, so a GPU-resident evaluation costs time linear in the class count:
    ~0.3 s for the ~1000-class In-Shop validation fold against ~0.016 s here,
    and several seconds once sibling runs share the device. R is per row just
    the class count, so the mask is ``column < R`` and vectorises to one
    broadcast comparison.
    """

    unique_labels, match_counts = label_counts
    # ``gt_labels`` arrives as ``(n, 1)`` while ``unique_labels`` stays ``(u,)``,
    # so both need an explicit ``(rows, label_dim)`` shape for the comparison to
    # broadcast over labels instead of over the trailing label dimension.
    query_labels = gt_labels.reshape(len(gt_labels), -1)
    unique_labels = unique_labels.reshape(len(unique_labels), -1)
    matches = (query_labels.unsqueeze(1) == unique_labels.unsqueeze(0)).all(dim=2)
    # ``argmax`` selects the first matching class. Rows whose label is absent
    # from ``unique_labels`` keep the zero count the upstream loop leaves them.
    relevant_count = match_counts[matches.to(torch.uint8).argmax(dim=1)] - int(
        ref_includes_query
    )
    count_per_query = torch.where(
        matches.any(dim=1),
        relevant_count,
        torch.zeros_like(relevant_count),
    )
    columns = torch.arange(shape[1], device=gt_labels.device)
    relevance_mask = columns.unsqueeze(0) < count_per_query.unsqueeze(1)
    return relevance_mask, count_per_query


#: Kept so the equivalence test compares against the real upstream loop rather
#: than a copy of it that could drift as pytorch-metric-learning changes.
_UPSTREAM_GET_RELEVANCE_MASK = accuracy_calculator.get_relevance_mask


def _patch_accuracy_calculator_relevance_mask():
    """Install the vectorised mask that ``r_precision``/MAP@R look up by name."""

    if accuracy_calculator.get_relevance_mask is not _vectorized_get_relevance_mask:
        accuracy_calculator.get_relevance_mask = _vectorized_get_relevance_mask


_patch_accuracy_calculator_relevance_mask()


MEASUREMENT_PRECISION_AT_1 = "precision_at_1"
MEASUREMENT_MAP_AT_R = "mean_average_precision_at_r"
MEASUREMENT_NMI = "NMI"
MEASUREMENT_R_PRECISION = "r_precision"
MEASUREMENT_AMI = "AMI"
MEASUREMENT_MRR = "mean_reciprocal_rank"
MEASUREMENT_MAP = "mean_average_precision"
# The measurements that read the whole retrieved neighbour list rather than
# only its top R, so a truncated list changes them.
FULL_DEPTH_MEASUREMENTS = (MEASUREMENT_MAP, MEASUREMENT_MRR)

DEFAULT_MEASUREMENTS = (
    MEASUREMENT_PRECISION_AT_1,
    MEASUREMENT_MAP_AT_R
)
# The whole vocabulary, in documented order. It is spelled out rather than
# derived from DEFAULT_MEASUREMENTS so changing what a run reports by default
# cannot reorder or duplicate the accepted choices.
AVAILABLE_MEASUREMENTS = (
    MEASUREMENT_PRECISION_AT_1,
    MEASUREMENT_MAP_AT_R,
    MEASUREMENT_NMI,
    MEASUREMENT_R_PRECISION,
    MEASUREMENT_AMI,
    MEASUREMENT_MRR,
    MEASUREMENT_MAP,
)
# The two metrics every run computes whatever --measurements asks for, because
# selection, early stopping, and the legacy result fields are defined in terms
# of them. They own dedicated columns; every other measurement is "additional"
# and gets a column of its own, so this set must stay at those two.
BASE_RETRIEVAL_METRICS = (
    MEASUREMENT_PRECISION_AT_1,
    MEASUREMENT_MAP_AT_R,
)
# Query-level scores the metric library can split by class. MRR and MAP belong
# here: with return_per_class each arrives as one value per class, exactly like
# RP, so reading it as a scalar would fail. NMI and AMI stay out as global scores.
PER_CLASS_RETRIEVAL_METRICS = (
    *BASE_RETRIEVAL_METRICS,
    MEASUREMENT_R_PRECISION,
    MEASUREMENT_MRR,
    MEASUREMENT_MAP,
)

_MEASUREMENT_ALIASES = {
    "precision_at_1": MEASUREMENT_PRECISION_AT_1,
    "precision@1": MEASUREMENT_PRECISION_AT_1,
    "p@1": MEASUREMENT_PRECISION_AT_1,
    "p1": MEASUREMENT_PRECISION_AT_1,
    "mean_average_precision_at_r": MEASUREMENT_MAP_AT_R,
    "map_at_r": MEASUREMENT_MAP_AT_R,
    "map@r": MEASUREMENT_MAP_AT_R,
    "nmi": MEASUREMENT_NMI,
    "normalized_mutual_information": MEASUREMENT_NMI,
    "r_precision": MEASUREMENT_R_PRECISION,
    "r-precision": MEASUREMENT_R_PRECISION,
    "rprecision": MEASUREMENT_R_PRECISION,
    "rp": MEASUREMENT_R_PRECISION,
    "ami": MEASUREMENT_AMI,
    "adjusted_mutual_information": MEASUREMENT_AMI,
    "mean_reciprocal_rank": MEASUREMENT_MRR,
    "mrr": MEASUREMENT_MRR,
    "mean_average_precision": MEASUREMENT_MAP,
    "map": MEASUREMENT_MAP,
}


def normalize_measurement_name(measurement):
    """Return the metric-library name for one CLI measurement spelling."""

    key = str(measurement).strip().lower()
    try:
        return _MEASUREMENT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown measurement {measurement!r}; choose from {AVAILABLE_MEASUREMENTS}"
        ) from exc


def normalize_measurements(measurements=None):
    """Validate and de-duplicate measurements while preserving their order."""

    if measurements is None:
        measurements = DEFAULT_MEASUREMENTS
    if isinstance(measurements, str):
        measurements = [measurements]
    normalized = []
    for measurement in measurements:
        canonical = normalize_measurement_name(measurement)
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("measurements must include at least one measurement")
    return tuple(normalized)


# The reported D_test table, following Vasileva & D'yakonov (2023): P@1 with the
# Recall@K ladder above it, the rank-aware RP, MAP@R, MAP and MRR, and the
# clustering scores NMI and AMI. Validation keeps its own cheaper set, so this is
# applied to the test evaluation only. The clustering scores share one k-means
# over the query embeddings; on a 14218-query In-Shop fold that costs about
# 1.4 s per evaluation, which is why they stay off validation.
#
# MAP and MRR are the metric library's definitions at its default depth, the
# whole reference set.
REPORT_TEST_MEASUREMENTS = (
    MEASUREMENT_PRECISION_AT_1,
    MEASUREMENT_R_PRECISION,
    MEASUREMENT_MAP_AT_R,
    MEASUREMENT_MAP,
    MEASUREMENT_MRR,
    MEASUREMENT_NMI,
    MEASUREMENT_AMI,
)
REPORT_TEST_RECALL_AT_K = (1, 10, 20, 40)


def resolve_test_measurements(args):
    """Return the measurements the D_test evaluation reports.

    Test reporting may be wider than validation: early stopping needs one
    metric per epoch, while the reported table wants every headline number
    once. An unset test list means "whatever validation reports".
    """

    explicit = getattr(args, "test_measurements", None)
    if explicit is not None:
        return normalize_measurements(explicit)
    validation = normalize_measurements(getattr(args, "measurements", None))
    if getattr(args, "report_test_metrics", True):
        # The preset widens the test table; anything --measurements asked for
        # is still reported rather than replaced.
        return normalize_measurements((*REPORT_TEST_MEASUREMENTS, *validation))
    return validation


def resolve_test_recall_at_k(args):
    """Return the Recall@K ladder the D_test evaluation reports.

    An explicitly empty ``--test_recall_at_k`` means no ladder on test, which
    is why an unset list is ``None`` rather than ``()``.
    """

    explicit = getattr(args, "test_recall_at_k", None)
    if explicit is not None:
        return normalize_recall_at_k(explicit)
    validation = normalize_recall_at_k(getattr(args, "recall_at_k", ()) or ())
    if getattr(args, "report_test_metrics", True):
        return normalize_recall_at_k((*REPORT_TEST_RECALL_AT_K, *validation))
    return validation


def resolve_validation_measurements(args):
    """Return the measurements every validation evaluation reports."""

    return normalize_measurements(getattr(args, "measurements", None))


def resolve_validation_recall_at_k(args):
    """Return the Recall@K ladder every validation evaluation reports."""

    return normalize_recall_at_k(getattr(args, "recall_at_k", ()) or ())


def logged_measurements(args):
    """Return the measurement columns a run's metrics.csv needs.

    Validation and test may report different sets, and one file holds both, so
    the columns are their union in validation-first order.
    """

    columns = list(resolve_validation_measurements(args))
    for measurement in resolve_test_measurements(args):
        if measurement not in columns:
            columns.append(measurement)
    return tuple(columns)


def logged_recall_at_k(args):
    """Return the Recall@K columns a run's metrics.csv needs."""

    return normalize_recall_at_k(
        (*resolve_validation_recall_at_k(args), *resolve_test_recall_at_k(args))
    )


def additional_measurements(measurements=None):
    """Return requested measurements beyond the two legacy headline metrics."""

    return tuple(
        measurement
        for measurement in normalize_measurements(measurements)
        if measurement not in BASE_RETRIEVAL_METRICS
    )


RECALL_AT_K_METRIC_PREFIX = "recall_at_"


def recall_at_k_metric_name(k):
    """Name the Recall@K metric for one K, as used in logs, CSV and JSON."""

    return f"{RECALL_AT_K_METRIC_PREFIX}{int(k)}"


def normalize_recall_at_k(recall_at_k):
    """Validate a requested Recall@K set and return it sorted and deduplicated."""

    if not recall_at_k:
        return ()
    values = []
    for value in recall_at_k:
        k = int(value)
        if k < 1:
            raise ValueError(f"recall_at_k values must be >= 1, got {value!r}")
        values.append(k)
    return tuple(sorted(set(values)))


class RecallAtKAccuracyCalculator(AccuracyCalculator):
    """``AccuracyCalculator`` plus Recall@K for a caller-chosen set of K.

    Recall@K in the retrieval sense is the fraction of queries with at least one
    correct neighbour among their K nearest, so Recall@1 is by definition equal
    to ``precision_at_1``. Queries whose label has no other reference sample are
    excluded exactly as upstream excludes them from ``precision_at_1``, which
    keeps the two comparable.

    ``pytorch-metric-learning`` discovers metrics by scanning for
    ``calculate_*`` attributes in ``__init__``, so the per-K methods are bound
    before delegating upwards.
    """

    def __init__(self, *args, recall_at_k=(), **kwargs):
        self.recall_at_k = normalize_recall_at_k(recall_at_k)
        for k in self.recall_at_k:
            setattr(
                self,
                f"calculate_{recall_at_k_metric_name(k)}",
                partial(self._calculate_recall_at_k, k),
            )
        super().__init__(*args, **kwargs)

    def requires_knn(self):
        return super().requires_knn() + [
            recall_at_k_metric_name(k) for k in getattr(self, "recall_at_k", ())
        ]

    def determine_k(self, bin_counts, num_reference_embeddings, ref_includes_query):
        num_k = super().determine_k(bin_counts, num_reference_embeddings, ref_includes_query)
        if not self.recall_at_k:
            return num_k
        # max_bin_count covers MAP@R and RP, and full depth covers everything.
        # Recall@K may ask for more than max_bin_count; retrieve more only then.
        available = num_reference_embeddings - int(bool(ref_includes_query))
        wanted = max(self.recall_at_k)
        if wanted > available:
            logger.warning(
                f"Recall@{wanted} was requested but only {available} reference "
                "embeddings are searchable; it is computed over all of them"
            )
        return max(num_k, min(wanted, available))

    def _calculate_recall_at_k(
        self,
        k,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = accuracy_calculator.try_getting_not_lone_labels(
            knn_labels, query_labels, not_lone_query_mask
        )
        if knn_labels is None:
            return accuracy_calculator.nan_accuracy(label_counts[0], self.return_per_class)
        query_labels = query_labels[:, None]
        same_label = self.label_comparison_fn(query_labels, knn_labels[:, :k])
        # Recall@K is a hit rate: one correct neighbour in the top K is enough,
        # which is what makes it differ from precision_at_k for K > 1.
        accuracy_per_sample = torch.any(same_label, dim=1).type(torch.float64)
        return accuracy_calculator.maybe_get_avg_of_avgs(
            accuracy_per_sample,
            query_labels,
            self.avg_of_avgs,
            self.return_per_class,
        )


def evaluate_embeddings(
    all_embeddings,
    all_labels,
    name="test set",
    return_per_class=False,
    dataset=None,
    device="cpu",
    allow_cpu_fallback=True,
    recall_at_k=(),
    measurements=None,
    return_measurements=False,
):
    """Compute the requested measurements from precomputed embeddings.

    ``recall_at_k`` optionally adds Recall@K for each requested K. It changes
    the return shape: the Recall@K mapping is appended as the last element, so
    the result is ``(precision_at_1, map_at_r)`` by default,
    ``(..., per_class_metrics)`` with ``return_per_class``,
    ``(..., recall_at_k)`` with ``recall_at_k``, and all four with both.

    ``measurements`` controls which headline measurements are reported. The two
    legacy metrics are still calculated internally because checkpoint selection
    and existing result fields depend on them. With ``return_measurements``, a
    mapping containing exactly the requested measurements is appended after the
    optional per-class and Recall@K values.
    """

    # AccuracyCalculator expects one matrix/vector spanning the full evaluation
    # dataset rather than a list of batches.
    device = torch.device(normalize_device_name(device))
    validate_finite_embeddings(all_embeddings, name)
    evaluation_sets = make_evaluation_embedding_sets(all_embeddings, all_labels, dataset=dataset)
    query_embeddings = evaluation_sets["query_embeddings"]
    query_labels = evaluation_sets["query_labels"]
    reference_embeddings = evaluation_sets["reference_embeddings"]
    reference_labels = evaluation_sets["reference_labels"]
    ref_includes_query = evaluation_sets["ref_includes_query"]
    measurements = normalize_measurements(measurements)
    recall_at_k = normalize_recall_at_k(recall_at_k)
    metric_names = tuple(
        dict.fromkeys(
            (
                *BASE_RETRIEVAL_METRICS,
                *measurements,
                *(recall_at_k_metric_name(k) for k in recall_at_k),
            )
        )
    )
    # MAP and MRR read every retrieved neighbour, so they get the library's
    # default depth: the whole reference set. Everything else reads at most the
    # top R <= max_bin_count (Recall@K raises that to its largest K), so for them
    # max_bin_count is exact and much cheaper. Measured on 12612 same-source
    # images: P@1 and MAP@R in 0.04 s against 0.63 s, identical to within 1e-7.
    retrieval_depth = (
        None
        if any(name in FULL_DEPTH_MEASUREMENTS for name in metric_names)
        else "max_bin_count"
    )
    calculator_kwargs = {
        "include": metric_names,
        "return_per_class": return_per_class,
        "k": retrieval_depth,
        "recall_at_k": recall_at_k,
    }
    cuda_backend = _cuda_retrieval_backend(all_embeddings, device)
    if (
        not allow_cpu_fallback
        and device.type == "cuda"
        and cuda_backend is None
    ):
        raise RuntimeError(
            f"{name}: CUDA retrieval was required but GPU FAISS is unavailable"
        )
    if cuda_backend is None:
        calculator_kwargs["device"] = torch.device("cpu")
    else:
        faiss_module, gpu_id = cuda_backend
        gpu_resident_embeddings = (
            torch.is_tensor(all_embeddings)
            and all_embeddings.device.type == "cuda"
        )
        # AccuracyCalculator first moves all input arrays to its device. Keep it
        # on CPU in low-memory mode; the custom KNN still builds and searches a
        # single-GPU FAISS index from those host tensors.
        calculator_kwargs["device"] = (
            device if gpu_resident_embeddings else torch.device("cpu")
        )
        calculator_kwargs["knn_func"] = _get_reusable_faiss_knn(
            faiss_module,
            gpu_id,
            low_memory=not gpu_resident_embeddings,
        )
        logger.debug(
            f"{name}: using single-GPU FAISS retrieval on cuda:{gpu_id} "
            f"for {len(all_embeddings)} embeddings "
            f"(embedding residency: "
            f"{'gpu' if gpu_resident_embeddings else 'cpu-low-memory'})"
        )

    def calculate_accuracy():
        accuracy_calculator = RecallAtKAccuracyCalculator(**calculator_kwargs)
        return accuracy_calculator.get_accuracy(
            query_embeddings,
            query_labels,
            reference=reference_embeddings,
            reference_labels=reference_labels,
            ref_includes_query=ref_includes_query,
        )

    try:
        accuracy = calculate_accuracy()
    except (AttributeError, RuntimeError) as exc:
        if cuda_backend is None or not allow_cpu_fallback:
            raise
        logger.warning(
            f"{name}: CUDA FAISS retrieval failed ({exc}); retrying on CPU"
        )
        calculator_kwargs = {
            "include": metric_names,
            "return_per_class": return_per_class,
            "k": retrieval_depth,
            "recall_at_k": recall_at_k,
            "device": torch.device("cpu"),
        }
        accuracy = calculate_accuracy()
    if return_per_class:
        per_class_metrics = make_per_class_retrieval_metrics(
            query_labels,
            accuracy,
            reference_labels=reference_labels,
            ref_includes_query=ref_includes_query,
        )
        precision_at_1 = weighted_per_class_metric(per_class_metrics, "precision_at_1")
        mean_average_precision_at_r = weighted_per_class_metric(
            per_class_metrics,
            "mean_average_precision_at_r",
        )
        recall_values = {
            k: weighted_per_class_metric(per_class_metrics, recall_at_k_metric_name(k))
            for k in recall_at_k
        }
        measurement_values = {
            measurement: (
                weighted_per_class_metric(per_class_metrics, measurement)
                if measurement in PER_CLASS_RETRIEVAL_METRICS
                else float(accuracy[measurement])
            )
            for measurement in measurements
        }
        logger.opt(lazy=True).debug(
            "{}",
            lambda: f"{name} per-class retrieval metrics: {json.dumps(per_class_metrics, sort_keys=True)}",
        )
    else:
        per_class_metrics = None
        precision_at_1 = accuracy["precision_at_1"]
        mean_average_precision_at_r = accuracy["mean_average_precision_at_r"]
        recall_values = {
            k: float(accuracy[recall_at_k_metric_name(k)]) for k in recall_at_k
        }
        measurement_values = {
            measurement: float(accuracy[measurement])
            for measurement in measurements
        }
    if evaluation_sets["mode"] == QUERY_GALLERY_EVALUATION:
        logger.info(
            f"{name}: query-gallery retrieval with {len(query_labels)} queries and "
            f"{len(reference_labels)} gallery images"
        )
    measurement_labels = {
        MEASUREMENT_PRECISION_AT_1: "Precision@1",
        MEASUREMENT_MAP_AT_R: "MAP@R",
        MEASUREMENT_NMI: "NMI",
        MEASUREMENT_R_PRECISION: "RP",
        MEASUREMENT_AMI: "AMI",
        MEASUREMENT_MRR: "MRR",
        MEASUREMENT_MAP: "MAP",
    }
    # An unlabeled measurement falls back to its own name: this line runs after
    # training, so a missing label must never cost the run its results.
    measurement_report = " , ".join(
        f"{measurement_labels.get(measurement, measurement)} = {value*100:.1f}"
        for measurement, value in measurement_values.items()
    )
    recall_report = "".join(
        f" , R@{k} = {value*100:.1f}" for k, value in recall_values.items()
    )
    logger.info(f"{name}: {measurement_report}{recall_report}")
    results = [precision_at_1, mean_average_precision_at_r]
    if return_per_class:
        results.append(per_class_metrics)
    if recall_at_k:
        results.append(recall_values)
    if return_measurements:
        results.append(measurement_values)
    return tuple(results)


def evaluate(
    model,
    eval_loader,
    name="test set",
    device="cuda",
    return_per_class=False,
    return_diagnostics=False,
    retrieval_device=None,
    allow_cpu_fallback=True,
    embedding_residency=EVALUATION_EMBEDDING_RESIDENCY_CPU,
    recall_at_k=(),
    measurements=None,
    return_measurements=False,
):
    """Embed a dataset and compute the requested evaluation measurements.

    ``recall_at_k`` is forwarded to :func:`evaluate_embeddings`; see there for
    how it changes the return shape, or use :func:`evaluate_split` to receive a
    named result instead.
    """

    total_started = time.perf_counter()
    dataset = getattr(eval_loader, "dataset", None)
    if retrieval_device is None:
        retrieval_device = resolve_evaluation_retrieval_device(dataset, device)
    else:
        retrieval_device = torch.device(normalize_device_name(retrieval_device))
    embedding_residency = str(embedding_residency).lower()
    if embedding_residency not in EVALUATION_EMBEDDING_RESIDENCIES:
        raise ValueError(
            "evaluation embedding residency must be one of "
            f"{EVALUATION_EMBEDDING_RESIDENCIES}: {embedding_residency!r}"
        )
    keep_on_device = (
        retrieval_device.type == "cuda"
        and embedding_residency == EVALUATION_EMBEDDING_RESIDENCY_GPU
    )
    embedding_started = time.perf_counter()
    all_embeddings, all_labels = extract_eval_embeddings(
        model,
        eval_loader,
        name=name,
        device=device,
        keep_on_device=keep_on_device,
        output_device=retrieval_device if keep_on_device else None,
    )
    embedding_seconds = time.perf_counter() - embedding_started
    retrieval_started = time.perf_counter()
    measurement_kwargs = {}
    if measurements is not None:
        measurement_kwargs["measurements"] = measurements
    if return_measurements:
        measurement_kwargs["return_measurements"] = True
    result = evaluate_embeddings(
        all_embeddings,
        all_labels,
        name=name,
        return_per_class=return_per_class,
        dataset=dataset,
        device=retrieval_device,
        allow_cpu_fallback=allow_cpu_fallback,
        recall_at_k=recall_at_k,
        **measurement_kwargs,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started
    if not return_diagnostics:
        return result

    diagnostics_started = time.perf_counter()
    if torch.is_tensor(all_embeddings):
        embedding_norms = torch.linalg.vector_norm(all_embeddings, dim=1)
        class_count = int(torch.unique(all_labels).numel())
        norm_min = float(embedding_norms.min().item())
        norm_mean = float(embedding_norms.mean().item())
        norm_std = float(embedding_norms.std(unbiased=False).item())
        norm_max = float(embedding_norms.max().item())
    else:
        embedding_norms = np.linalg.norm(all_embeddings, axis=1)
        class_count = len(np.unique(all_labels))
        norm_min = float(embedding_norms.min())
        norm_mean = float(embedding_norms.mean())
        norm_std = float(embedding_norms.std())
        norm_max = float(embedding_norms.max())
    diagnostics = {
        "timing/embedding_extraction_seconds": embedding_seconds,
        "timing/retrieval_metrics_seconds": retrieval_seconds,
        "data/sample_count": len(all_labels),
        "data/class_count": class_count,
        "data/embedding_dimension": all_embeddings.shape[1],
        "embedding_norm/min": norm_min,
        "embedding_norm/mean": norm_mean,
        "embedding_norm/std": norm_std,
        "embedding_norm/max": norm_max,
    }
    try:
        diagnostics["data/batch_count"] = len(eval_loader)
    except TypeError:
        pass
    diagnostics["timing/analysis_seconds"] = time.perf_counter() - diagnostics_started
    diagnostics["timing/total_seconds"] = time.perf_counter() - total_started
    return (*result, diagnostics)


class EvaluationOutcome(NamedTuple):
    """One evaluation's metrics, with the optional parts always addressable."""

    precision_at_1: float
    mean_average_precision_at_r: float
    per_class_metrics: dict | None = None
    recall_at_k: dict | None = None
    measurements: dict | None = None
    diagnostics: dict | None = None


def unpack_evaluation_result(
    result,
    *,
    return_per_class=False,
    recall_at_k=(),
    measurements=None,
    return_measurements=False,
    return_diagnostics=False,
):
    """Turn an ``evaluate``/``evaluate_embeddings`` tuple into a named result.

    Both functions append their optional outputs in a fixed order -- per-class
    metrics, Recall@K, requested measurements, then diagnostics -- so callers
    that toggle those options do not have to branch on tuple length themselves.
    """

    values = list(result)
    diagnostics = values.pop() if return_diagnostics else None
    measurement_values = values.pop() if return_measurements else None
    recall_values = values.pop() if normalize_recall_at_k(recall_at_k) else {}
    per_class_metrics = values.pop() if return_per_class else None
    if len(values) != 2:
        raise ValueError(f"Unexpected evaluation result shape: {result!r}")
    precision_at_1, mean_average_precision_at_r = values
    if measurement_values is None:
        base_values = {
            MEASUREMENT_PRECISION_AT_1: precision_at_1,
            MEASUREMENT_MAP_AT_R: mean_average_precision_at_r,
        }
        measurement_values = {
            measurement: base_values[measurement]
            for measurement in normalize_measurements(measurements)
            if measurement in base_values
        }
    return EvaluationOutcome(
        precision_at_1=precision_at_1,
        mean_average_precision_at_r=mean_average_precision_at_r,
        per_class_metrics=per_class_metrics,
        recall_at_k=recall_values,
        measurements=measurement_values,
        diagnostics=diagnostics,
    )


def evaluate_split(
    model,
    eval_loader,
    name="test set",
    *,
    recall_at_k=(),
    measurements=None,
    return_per_class=False,
    return_diagnostics=False,
    **kwargs,
):
    """``evaluate`` with a named result instead of a variable-length tuple."""

    result = evaluate(
        model,
        eval_loader,
        name=name,
        return_per_class=return_per_class,
        return_diagnostics=return_diagnostics,
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
        **kwargs,
    )
    return unpack_evaluation_result(
        result,
        return_per_class=return_per_class,
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
        return_diagnostics=return_diagnostics,
    )


def class_names_for_labels(dataset, labels):
    classes = dataset_classes(dataset)
    if classes is None:
        return [""] * len(labels)
    names = []
    for label in labels:
        label_index = int(label)
        if 0 <= label_index < len(classes):
            names.append(str(classes[label_index]))
        else:
            names.append("")
    return names


def cifar100_superclass_names_for_labels(superclass_labels):
    names = []
    for superclass_label in superclass_labels:
        superclass_index = int(superclass_label)
        if 0 <= superclass_index < len(CIFAR100_SUPERCLASS_NAMES):
            names.append(CIFAR100_SUPERCLASS_NAMES[superclass_index])
        else:
            names.append("")
    return names


def load_sop_superclass_metadata(data_root=None):
    data_root = Path("data") / "StanfordOnlineProducts" if data_root is None else Path(data_root)
    sop_root = data_root
    if sop_root.name != "Stanford_Online_Products":
        sop_root = sop_root / "Stanford_Online_Products"

    rows_by_file_class_id = {}
    for filename in ("Ebay_train.txt", "Ebay_test.txt"):
        path = sop_root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            columns = line.split()
            if len(columns) < 4:
                continue
            class_id = int(columns[1])
            super_class_id = int(columns[2])
            super_class_name = Path(columns[3]).parts[0].replace("_final", "")
            rows_by_file_class_id[class_id] = {
                "super_class_id": super_class_id,
                "super_class_name": super_class_name,
            }
    return rows_by_file_class_id


def align_sop_superclass_metadata(labels, rows_by_file_class_id):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not rows_by_file_class_id:
        return None

    unique_labels = set(int(label) for label in labels)
    file_class_ids = set(int(label) for label in rows_by_file_class_id)
    if unique_labels <= file_class_ids:
        label_to_file_class_id = {label: label for label in unique_labels}
    elif {label + 1 for label in unique_labels} <= file_class_ids:
        label_to_file_class_id = {label: label + 1 for label in unique_labels}
    else:
        return None

    superclass_labels = []
    superclass_names = []
    for label in labels:
        metadata = rows_by_file_class_id[label_to_file_class_id[int(label)]]
        superclass_labels.append(int(metadata["super_class_id"]))
        superclass_names.append(str(metadata["super_class_name"]))
    return np.asarray(superclass_labels, dtype=np.int64), superclass_names


def pacmap_plot_groups(labels, dataset=None, dataset_name=None):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    normalized_name = normalize_dataset_name(dataset_name) if dataset_name is not None else None

    if normalized_name == "CIFAR100":
        superclass_labels = cifar100_superclass_labels_for_fine_labels(labels)
        return {
            "labels": superclass_labels,
            "names": cifar100_superclass_names_for_labels(superclass_labels),
            "basis": "superclass",
            "legend_title": "superclass",
        }

    if normalized_name == "StanfordOnlineProducts":
        sop_groups = align_sop_superclass_metadata(labels, load_sop_superclass_metadata())
        if sop_groups is not None:
            superclass_labels, superclass_names = sop_groups
            return {
                "labels": superclass_labels,
                "names": superclass_names,
                "basis": "superclass",
                "legend_title": "superclass",
            }
        logger.warning(
            "SOP superclass metadata was not found or did not align with dataset labels; using class labels"
        )

    return {
        "labels": labels,
        "names": class_names_for_labels(dataset, labels) if dataset is not None else [""] * len(labels),
        "basis": "label",
        "legend_title": "label",
    }


def dataset_classes(dataset):
    if dataset is None:
        return None
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return classes
    nested_dataset = getattr(dataset, "dataset", None)
    if nested_dataset is not None:
        return dataset_classes(nested_dataset)
    return None


def _embedding_visualization_inputs(embeddings, labels, method_name):
    embeddings = np.ascontiguousarray(_as_numpy(embeddings), dtype=np.float32)
    labels = _as_numpy(labels).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError(f"{method_name} visualization requires an embedding matrix")
    if len(embeddings) != len(labels):
        raise ValueError(f"{method_name} embeddings and labels must have the same number of samples")
    if len(embeddings) < 2:
        raise ValueError(f"{method_name} visualization requires at least two test embeddings")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"{method_name} embeddings must be finite")
    return embeddings, labels


def project_tsne_embeddings(embeddings, seed=0, perplexity=None):
    """Project an embedding matrix to two reproducible t-SNE coordinates."""

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("t-SNE projection requires an embedding matrix")
    if len(embeddings) < 2:
        raise ValueError("t-SNE projection requires at least two embeddings")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("t-SNE embeddings must be finite")

    if perplexity is None:
        # sklearn requires perplexity < n_samples. Retain its normal default
        # whenever possible and shrink it only for smaller diagnostic sets.
        perplexity = min(30.0, float(len(embeddings) - 1))
    perplexity = float(perplexity)
    if not np.isfinite(perplexity) or not 0.0 < perplexity < len(embeddings):
        raise ValueError(
            "t-SNE perplexity must be finite, positive, and smaller than the number of embeddings"
        )

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    projection_input = embeddings
    if embeddings.shape[1] > 50:
        # sklearn recommends a preliminary reduction for high-dimensional
        # dense inputs to suppress noise and make pairwise distances cheaper.
        pca_components = min(50, len(embeddings), embeddings.shape[1])
        projection_input = PCA(
            n_components=pca_components,
            random_state=int(seed),
        ).fit_transform(embeddings)

    init = "pca" if projection_input.shape[1] >= 2 else "random"
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init=init,
        random_state=int(seed),
    ).fit_transform(projection_input)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (len(embeddings), 2) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            f"t-SNE returned invalid coordinates with shape {coordinates.shape}, expected [N, 2]"
        )
    return coordinates


def _write_embedding_visualization_artifacts(
    coordinates,
    labels,
    output_dir,
    stem,
    title,
    coordinate_prefix,
    axis_label,
    dataset=None,
    dataset_name=None,
):
    """Write shared CSV and scatter-plot artifacts for a 2-D projection."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    coordinates = np.asarray(coordinates, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if coordinates.shape != (len(labels), 2):
        raise ValueError(
            f"{axis_label} coordinates must have shape ({len(labels)}, 2); got {coordinates.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    class_names = class_names_for_labels(dataset, labels) if dataset is not None else [""] * len(labels)
    plot_groups = pacmap_plot_groups(labels, dataset=dataset, dataset_name=dataset_name)
    plot_group_labels = np.asarray(plot_groups["labels"], dtype=np.int64).reshape(-1)
    plot_group_names = list(plot_groups["names"])
    if len(plot_group_labels) != len(labels) or len(plot_group_names) != len(labels):
        raise ValueError("Visualization plot-group metadata must align with embeddings and labels")

    x_field = f"{coordinate_prefix}_x"
    y_field = f"{coordinate_prefix}_y"
    with coordinates_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_position",
                "label",
                "class_name",
                "plot_group_label",
                "plot_group_name",
                x_field,
                y_field,
            ],
        )
        writer.writeheader()
        for sample_position, (label, class_name, group_label, group_name, coordinate) in enumerate(
            zip(labels, class_names, plot_group_labels, plot_group_names, coordinates)
        ):
            writer.writerow(
                {
                    "sample_position": sample_position,
                    "label": int(label),
                    "class_name": class_name,
                    "plot_group_label": int(group_label),
                    "plot_group_name": group_name,
                    x_field: float(coordinate[0]),
                    y_field: float(coordinate[1]),
                }
            )

    unique_labels = np.unique(plot_group_labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(unique_labels) <= 20:
        color_map = plt.get_cmap("tab20", len(unique_labels))
        for color_index, label in enumerate(unique_labels):
            mask = plot_group_labels == label
            names_for_label = sorted(set(name for name in np.asarray(plot_group_names, dtype=object)[mask] if name))
            legend_label = names_for_label[0] if names_for_label else str(int(label))
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=9,
                alpha=0.78,
                linewidths=0,
                color=color_map(color_index),
                label=legend_label,
            )
        ax.legend(title=str(plot_groups["legend_title"]), loc="best", markerscale=1.8, fontsize="small")
    else:
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=plot_group_labels.astype(float),
            s=7,
            alpha=0.78,
            linewidths=0,
            cmap="turbo",
        )
        fig.colorbar(scatter, ax=ax, label=str(plot_groups["legend_title"]))

    ax.set_title(title)
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return {
        "coordinates": coordinates_path,
        "plot": plot_path,
        "sample_count": int(len(labels)),
        "color_basis": str(plot_groups["basis"]),
    }


def write_pacmap_visualization(
    embeddings,
    labels,
    output_dir,
    stem="test_pacmap",
    title="Test embeddings - PacMAP",
    dataset=None,
    dataset_name=None,
):
    """Write PacMAP 2D coordinates and a plot-group-colored scatter plot."""

    embeddings, labels = _embedding_visualization_inputs(embeddings, labels, "PacMAP")

    try:
        import pacmap
    except ImportError as exc:
        raise ImportError(
            "PacMAP visualization requires the pacmap package. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    coordinates = np.asarray(pacmap.PaCMAP().fit_transform(embeddings), dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"PacMAP returned coordinates with shape {coordinates.shape}, expected [N, 2]")
    coordinates = coordinates[:, :2]
    return _write_embedding_visualization_artifacts(
        coordinates=coordinates,
        labels=labels,
        output_dir=output_dir,
        stem=stem,
        title=title,
        coordinate_prefix="pacmap",
        axis_label="PacMAP",
        dataset=dataset,
        dataset_name=dataset_name,
    )


def write_tsne_visualization(
    embeddings,
    labels,
    output_dir,
    stem="test_tsne",
    title="Test embeddings - t-SNE",
    dataset=None,
    dataset_name=None,
    seed=0,
    perplexity=None,
):
    """Write t-SNE 2D coordinates and a plot-group-colored scatter plot."""

    embeddings, labels = _embedding_visualization_inputs(embeddings, labels, "t-SNE")
    coordinates = project_tsne_embeddings(
        embeddings,
        seed=seed,
        perplexity=perplexity,
    )
    return _write_embedding_visualization_artifacts(
        coordinates=coordinates,
        labels=labels,
        output_dir=output_dir,
        stem=stem,
        title=title,
        coordinate_prefix="tsne",
        axis_label="t-SNE",
        dataset=dataset,
        dataset_name=dataset_name,
    )


def make_per_class_retrieval_metrics(labels, accuracy, reference_labels=None, ref_includes_query=True):
    """Map AccuracyCalculator's sorted per-class values back to class labels."""

    labels = _as_numpy(labels).reshape(-1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if ref_includes_query:
        # Same-source retrieval excludes singleton classes because they have no
        # relevant reference after the query itself is removed.
        eligible = [(label, int(count)) for label, count in zip(unique_labels, counts) if count > 1]
    else:
        reference_labels = _as_numpy(reference_labels).reshape(-1)
        reference_label_set = set(reference_labels.tolist())
        eligible = [
            (label, int(count))
            for label, count in zip(unique_labels, counts)
            if label in reference_label_set
        ]
    # Optional measurements and Recall@K expand the per-class metric set. NMI
    # and AMI are deliberately absent because clustering agreement is one global
    # score, not a query-level value that can be split by class.
    metric_names = [
        name
        for name in accuracy
        if name in PER_CLASS_RETRIEVAL_METRICS
        or name.startswith(RECALL_AT_K_METRIC_PREFIX)
    ]
    mismatched = [name for name in metric_names if len(accuracy[name]) != len(eligible)]
    if mismatched:
        raise ValueError("Per-class retrieval metric count does not match eligible evaluation classes")
    return {
        str(int(label)): {
            "count": count,
            **{name: float(accuracy[name][index]) for name in metric_names},
        }
        for index, (label, count) in enumerate(eligible)
    }


def weighted_per_class_metric(per_class_metrics, metric):
    total_count = sum(values["count"] for values in per_class_metrics.values())
    if total_count == 0:
        return float("nan")
    return float(
        sum(values["count"] * values[metric] for values in per_class_metrics.values()) / total_count
    )
