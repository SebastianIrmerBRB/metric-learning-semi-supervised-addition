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
import multiprocessing as mp
import os
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_metric_learning.datasets as datasets
import pytorch_metric_learning.samplers as samplers
import torch
import torchvision.transforms as tfm
import torchvision.transforms.v2 as v2
from loguru import logger
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from . import local_datasets
from .dataset_constants import (
    CIFAR100_DEVELOPMENT_CLASSES,
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
    CIFAR_UNSEEN_CLASS_PROTOCOLS,
    CV_MODES,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
    DATASET_PROTOCOLS,
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_OFFICIAL,
    GROUPED_CV_MODES,
    POST_APPORTION_VAL_RATIO,
    QUERY_GALLERY_EVALUATION,
    SAME_SOURCE_EVALUATION,
    VAL_MODES,
    VAL_MODE_ALL,
    VAL_MODE_MATCH_TRAIN,
    VAL_MODE_SPLIT_AFTER_APPORTION,
)
from .dataset_composition import (
    CombinedDataset,
    DatasetBundle,
    EXTERNAL_UNLABELED_FILTERS,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER,
    EXTERNAL_UNLABELED_FILTER_NONE,
    append_external_unlabeled_dataset,
    get_nested_transform,
)
from .dataset_protocols import (
    apply_cifar_long_tail,
    get_dataset_class,
    is_dataset_ready,
    load_dataset_protocol_sources as _load_dataset_protocol_sources,
    make_cifar_long_tail_class_counts,
    normalize_dataset_name,
    validate_cifar_balanced_fraction_protocol,
    validate_cifar_imbalance_factor,
    validate_dataset_protocol,
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
    make_holdout_split_info,
    make_superclass_balanced_group_folds,
    make_train_valid_subsets,
    remap_positions,
    select_balanced_subset_indices,
    set_nested_transform,
    split_cifar_balanced_by_fraction,
    split_dataset_by_classes,
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

PREFERRED_TORCH_SHARING_STRATEGY = "file_descriptor"
TORCH_SHARING_STRATEGY_ENV = "METRIC_LEARNING_TORCH_SHARING_STRATEGY"

TENSORBOARD_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = exc

DATALOADER_START_METHODS = ("spawn", "forkserver", "fork", "default")


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

    # Different libraries maintain independent random-number generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        # Seed every visible CUDA device, not only the currently selected one.
        torch.cuda.manual_seed_all(seed)

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
    pin_memory=False,
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

    def __init__(self, log_dir, args):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the tensorboard package. "
                "Install it with `pip install -r requirements.txt`."
            ) from TENSORBOARD_IMPORT_ERROR

        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "metrics.csv"
        self.diagnostics_path = self.log_dir / "diagnostics.csv"
        # TensorBoard is convenient for visualization; CSV keeps metrics easy
        # to inspect and aggregate with ordinary tools.
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["step", "epoch", "split", "loss", "precision_at_1", "mean_average_precision_at_r"],
        )
        self.csv_writer.writeheader()
        self.diagnostics_file = self.diagnostics_path.open("w", newline="")
        self.diagnostics_writer = csv.DictWriter(
            self.diagnostics_file,
            fieldnames=["step", "epoch", "category", "name", "value", "details"],
        )
        self.diagnostics_writer.writeheader()
        self.writer.add_text("run/arguments", "\n".join(f"{key}: {value}" for key, value in vars(args).items()), 0)
        logger.info(f"TensorBoard logs are being saved in {self.log_dir / 'tensorboard'}")
        logger.info(f"CSV metrics are being saved in {self.csv_path}")
        logger.info(f"Diagnostic metrics are being saved in {self.diagnostics_path}")

    def log_train_batch(self, loss, epoch, step, diagnostics=None):
        # Batch loss uses global optimizer-step count on TensorBoard's x-axis.
        self.writer.add_scalar("train/batch_loss", loss, step)
        self._write_row(step=step, epoch=epoch, split="train_batch", loss=loss)
        self.log_diagnostics(diagnostics, step=step, epoch=epoch, category="train_batch")

    def log_train_epoch(self, loss, epoch, step, diagnostics=None):
        self.writer.add_scalar("train/epoch_loss", loss, step)
        self._write_row(step=step, epoch=epoch, split="train_epoch", loss=loss)
        self.log_diagnostics(diagnostics, step=step, epoch=epoch, category="train_epoch")

    def log_eval(
        self,
        split,
        precision_at_1,
        mean_average_precision_at_r,
        step,
        epoch=None,
        per_class_metrics=None,
    ):
        # The split prefix keeps validation and test curves separate.
        self.writer.add_scalar(f"{split}/precision_at_1", precision_at_1, step)
        self.writer.add_scalar(f"{split}/mean_average_precision_at_r", mean_average_precision_at_r, step)
        self._write_row(
            step=step,
            epoch=epoch,
            split=split,
            precision_at_1=precision_at_1,
            mean_average_precision_at_r=mean_average_precision_at_r,
        )
        for label, metrics in (per_class_metrics or {}).items():
            self.log_diagnostics(
                {
                    f"{split}/per_class/{label}/precision_at_1": metrics["precision_at_1"],
                    f"{split}/per_class/{label}/mean_average_precision_at_r": (
                        metrics["mean_average_precision_at_r"]
                    ),
                },
                step=step,
                epoch=epoch,
                category="eval_per_class",
                details={"class_label": label},
            )

    def close(self):
        self.csv_file.close()
        self.diagnostics_file.close()
        self.writer.close()

    def log_diagnostics(self, diagnostics, step, epoch, category, details=None):
        if not diagnostics:
            return
        for name, value in diagnostics.items():
            if value is None:
                continue
            scalar_value = float(value)
            self.writer.add_scalar(name, scalar_value, step)
            self.diagnostics_writer.writerow(
                {
                    "step": step,
                    "epoch": "" if epoch is None else epoch,
                    "category": category,
                    "name": name,
                    "value": scalar_value,
                    "details": "" if details is None else json.dumps(details, sort_keys=True),
                }
            )
        self.diagnostics_file.flush()

    def _write_row(
        self,
        step,
        epoch,
        split,
        loss=None,
        precision_at_1=None,
        mean_average_precision_at_r=None,
    ):
        # Empty strings produce clean sparse CSV columns for rows containing
        # either a loss or retrieval metrics.
        self.csv_writer.writerow(
            {
                "step": step,
                "epoch": "" if epoch is None else epoch,
                "split": split,
                "loss": "" if loss is None else loss,
                "precision_at_1": "" if precision_at_1 is None else precision_at_1,
                "mean_average_precision_at_r": ""
                if mean_average_precision_at_r is None
                else mean_average_precision_at_r,
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
):
    """Load source data and create the initial train/validation/test split.

    ``split_after_apportion`` is special: validation is left empty here and is
    carved from the selected support draw later.
    ``full_train`` uses every development sample and leaves validation empty for
    the one fixed-epoch model trained after HPO.
    """

    dataset_name = normalize_dataset_name(dataset_name)
    if data_split_seed is None:
        data_split_seed = seed
    # Training uses stochastic augmentation. Validation, test, and SSL feature
    # extraction use the deterministic test_transform below.
    # unchanged from initial setup
    train_transform = tfm.Compose(
        [
            v2.RGB(),
            tfm.Resize(size=(224, 224), antialias=True),
            # tfm.RandAugment(num_ops=3, interpolation=tfm.InterpolationMode.BILINEAR),
            tfm.ToTensor(),
            tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_transform = tfm.Compose(
        [
            v2.RGB(),
            tfm.Resize(size=(224, 224), antialias=True),
            tfm.ToTensor(),
            tfm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
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
            "post_apportion_val_ratio": float(POST_APPORTION_VAL_RATIO),
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
        if dataset_name == "CIFAR100" and cv_mode == CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
            train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes_superclass_balanced(
                train_val_dataset,
                seed=data_split_seed,
            )
            split_label = "superclass-balanced holdout"
        else:
            train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes(
                train_val_dataset,
                seed=data_split_seed,
            )
            split_label = "holdout"
        split_info = make_holdout_split_info(
            train_val_dataset=train_val_dataset,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            val_mode=val_mode,
        )
        if split_label == "superclass-balanced holdout":
            split_info["holdout_strategy"] = "superclass_balanced_by_cifar100_superclass"
    split_info["dataset_protocol"] = protocol_info
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


class TwoStreamMPerClassBatchSampler(torch.utils.data.Sampler):
    """Build M-per-class batches with fixed pseudo- and true-label quotas.

    The pseudo-labeled (originally unlabeled) stream is primary: without an
    explicit epoch length, its size determines the number of full batches.
    The true-labeled stream is sampled repeatedly for the same number of
    batches. Each stream independently follows ``MPerClassSampler`` semantics,
    including replacement when a selected class has fewer than ``m`` samples.
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
    ):
        labels = torch.as_tensor(labels, dtype=torch.long).cpu().numpy()
        if labels.ndim != 1:
            raise ValueError("TwoStreamMPerClassBatchSampler labels must be one-dimensional")

        self.batch_size = int(batch_size)
        self.labeled_batch_size = int(labeled_batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.m_per_class = int(m)
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

    def _sample_stream(self, grouped_indices, stream_batch_size):
        classes_per_batch = stream_batch_size // self.m_per_class
        labels = np.asarray(list(grouped_indices), dtype=np.int64)
        selected_labels = self.generator.permutation(labels)[:classes_per_batch]
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

    def __iter__(self):
        for _ in range(self.num_batches):
            # Keep the reference implementation's primary-then-secondary
            # ordering: pseudo-labeled samples first, true-labeled samples last.
            yield self._sample_stream(
                self._unlabeled_by_class,
                self.unlabeled_batch_size,
            ) + self._sample_stream(
                self._labeled_by_class,
                self.labeled_batch_size,
            )

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
    pin_memory=False,
    labeled_batch_size=None,
):
    """Create an M-per-class loader, optionally with labeled/pseudo streams."""

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
            )
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
                f"m={sampler.m_per_class} per stream, {sampler.num_samples} sampled examples/epoch, "
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

    def __init__(self, embeddings, batch_size, neighbors_per_query, seed):
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
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
    pin_memory=False,
):
    """Create the nearest-neighbor batch loader used by STML."""

    num_workers = dataloader_num_workers_for_dataset(train_dataset, num_workers)
    sampler = STMLNearestNeighborBatchSampler(
        embeddings=sampling_embeddings,
        batch_size=batch_size,
        neighbors_per_query=neighbors_per_query,
        seed=seed,
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


def make_eval_loader(dataset, batch_size=32, seed=0, num_workers=8, start_method="spawn", pin_memory=False):
    # Evaluation traverses every item exactly once in dataset order.
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
    ):
        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        if features.ndim not in {2, 3}:
            raise ValueError("precomputed backbone features must be a 2D or 3D tensor")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = features.contiguous()
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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        if features.ndim == 2:
            view_index = int(torch.randint(features.shape[0], ()).item())
            features = features[view_index]
        if self.sample_weights is None:
            return features, self.orig_labels[index]
        return features, self.orig_labels[index], self.sample_weights[index]


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


def dataset_has_precomputed_backbone_features(dataset):
    """Return True when a dataset is backed by an in-memory feature tensor."""

    if isinstance(dataset, PrecomputedBackboneFeatureDataset):
        return True
    if isinstance(dataset, Subset):
        return dataset_has_precomputed_backbone_features(dataset.dataset)
    if isinstance(dataset, CombinedDataset):
        return any(dataset_has_precomputed_backbone_features(child) for child in dataset.datasets)
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        return dataset_has_precomputed_backbone_features(child_dataset)
    return False


def dataloader_num_workers_for_dataset(dataset, num_workers):
    if dataset_has_precomputed_backbone_features(dataset):
        # Precomputed feature datasets already live as CPU tensors in this
        # process. Sending them through DataLoader workers can serialize or
        # duplicate large tensors and can hang shutdown/interrupt handling.
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


def precompute_backbone_feature_dataset(
    model,
    dataset,
    device,
    batch_size,
    seed,
    num_workers,
    start_method,
    desc,
    pin_memory=False,
    require_feature_transform=False,
    use_feature_transform=True,
    num_views=1,
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
    with torch.no_grad():
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
    )
    train_loader = make_train_loader(dataset_bundle.train_dataset, batch_size, sampler_m, seed, num_workers, start_method)
    valid_loader = make_eval_loader(dataset_bundle.valid_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    test_loader = make_eval_loader(dataset_bundle.test_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    return train_loader, valid_loader, test_loader, dataset_bundle.train_labels_mapper


def extract_eval_embeddings(model, eval_loader, name="test set", device="cuda"):
    """Embed a dataset once, returning NumPy embeddings and labels."""

    # eval() disables training-only behavior such as dropout and updates to
    # normalization statistics.
    model = model.eval()
    all_embeddings = []
    all_labels = []
    # Extract embeddings and labels
    progress = None
    try:
        with torch.no_grad():
            progress = tqdm(eval_loader, desc=name)
            for images, labels in progress:
                # Keep only CPU NumPy embeddings after each batch to free accelerator
                # memory before processing the next batch.
                forward_cached = getattr(model, "forward_cached", None)
                embeddings = forward_model_inputs(
                    model,
                    images,
                    device,
                    use_cache=forward_cached is not None,
                )
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
                all_labels.append(labels.cpu().numpy())
    except BaseException:
        shutdown_dataloader_workers(eval_loader)
        raise
    finally:
        if progress is not None:
            progress.close()
    # Concatenate all embeddings and labels
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    validate_finite_embeddings(all_embeddings, name)
    return all_embeddings, all_labels


def validate_finite_embeddings(all_embeddings, name="test set"):
    if not np.isfinite(all_embeddings).all():
        total_values = int(all_embeddings.size)
        nonfinite_values = int(total_values - np.isfinite(all_embeddings).sum())
        nan_values = int(np.isnan(all_embeddings).sum())
        inf_values = int(np.isinf(all_embeddings).sum())
        raise NonFiniteEmbeddingError(
            f"{name} produced non-finite embeddings before retrieval metric calculation: "
            f"{nonfinite_values}/{total_values} values are non-finite "
            f"({nan_values} NaN, {inf_values} +/-Inf). "
            "This usually indicates that the model diverged for the current hyperparameters."
        )


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


def make_evaluation_embedding_sets(all_embeddings, all_labels, dataset=None):
    """Build query/reference arrays for same-source or query-gallery retrieval."""

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
        "query_embeddings": all_embeddings[query_indices],
        "query_labels": all_labels[query_indices],
        "reference_embeddings": all_embeddings[gallery_indices],
        "reference_labels": all_labels[gallery_indices],
        "ref_includes_query": False,
    }


def evaluate_embeddings(all_embeddings, all_labels, name="test set", return_per_class=False, dataset=None):
    """Compute retrieval Precision@1 and MAP@R from precomputed embeddings."""

    # AccuracyCalculator expects one matrix/vector spanning the full evaluation
    # dataset rather than a list of batches.
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    all_labels = np.asarray(all_labels).reshape(-1)
    validate_finite_embeddings(all_embeddings, name)
    evaluation_sets = make_evaluation_embedding_sets(all_embeddings, all_labels, dataset=dataset)
    query_embeddings = evaluation_sets["query_embeddings"]
    query_labels = evaluation_sets["query_labels"]
    reference_embeddings = evaluation_sets["reference_embeddings"]
    reference_labels = evaluation_sets["reference_labels"]
    ref_includes_query = evaluation_sets["ref_includes_query"]
    # Retrieval metrics compare each embedding with the rest of this evaluation
    # set; no classifier head is used.
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "mean_average_precision_at_r"),
        return_per_class=return_per_class,
        k="max_bin_count",
        device=torch.device("cpu"),
    )
    accuracy = accuracy_calculator.get_accuracy(
        query_embeddings,
        query_labels,
        reference=reference_embeddings,
        reference_labels=reference_labels,
        ref_includes_query=ref_includes_query,
    )
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
        logger.debug(f"{name} per-class retrieval metrics: {json.dumps(per_class_metrics, sort_keys=True)}")
    else:
        per_class_metrics = None
        precision_at_1 = accuracy["precision_at_1"]
        mean_average_precision_at_r = accuracy["mean_average_precision_at_r"]
    if evaluation_sets["mode"] == QUERY_GALLERY_EVALUATION:
        logger.info(
            f"{name}: query-gallery retrieval with {len(query_labels)} queries and "
            f"{len(reference_labels)} gallery images"
        )
    logger.info(f"{name}: Precision@1 = {precision_at_1*100:.1f} , MAP@R = {mean_average_precision_at_r*100:.1f}")
    if return_per_class:
        return precision_at_1, mean_average_precision_at_r, per_class_metrics
    return precision_at_1, mean_average_precision_at_r


def evaluate(model, eval_loader, name="test set", device="cuda", return_per_class=False):
    """Embed a dataset and compute retrieval Precision@1 and MAP@R."""

    all_embeddings, all_labels = extract_eval_embeddings(model, eval_loader, name=name, device=device)
    return evaluate_embeddings(
        all_embeddings,
        all_labels,
        name=name,
        return_per_class=return_per_class,
        dataset=getattr(eval_loader, "dataset", None),
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
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
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

    labels = np.asarray(labels).reshape(-1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if ref_includes_query:
        # Same-source retrieval excludes singleton classes because they have no
        # relevant reference after the query itself is removed.
        eligible = [(label, int(count)) for label, count in zip(unique_labels, counts) if count > 1]
    else:
        reference_labels = np.asarray(reference_labels).reshape(-1)
        reference_label_set = set(reference_labels.tolist())
        eligible = [
            (label, int(count))
            for label, count in zip(unique_labels, counts)
            if label in reference_label_set
        ]
    precision_values = accuracy["precision_at_1"]
    map_values = accuracy["mean_average_precision_at_r"]
    if not (len(eligible) == len(precision_values) == len(map_values)):
        raise ValueError("Per-class retrieval metric count does not match eligible evaluation classes")
    return {
        str(int(label)): {
            "count": count,
            "precision_at_1": float(precision),
            "mean_average_precision_at_r": float(map_at_r),
        }
        for (label, count), precision, map_at_r in zip(eligible, precision_values, map_values)
    }


def weighted_per_class_metric(per_class_metrics, metric):
    total_count = sum(values["count"] for values in per_class_metrics.values())
    if total_count == 0:
        return float("nan")
    return float(
        sum(values["count"] * values[metric] for values in per_class_metrics.values()) / total_count
    )
