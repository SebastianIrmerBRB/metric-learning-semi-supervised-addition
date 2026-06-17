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

import local_datasets

TENSORBOARD_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = exc


DATALOADER_START_METHODS = ("spawn", "forkserver", "fork", "default")
CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD = "superclass_balanced_group_kfold"
CV_MODES = (
    "kfold",
    "group_kfold",
    "stratified_kfold",
    "stratified_group_kfold",
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
)
GROUPED_CV_MODES = ("group_kfold", "stratified_group_kfold", CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD)
DATASET_PROTOCOL_OFFICIAL = "official"
DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION = "cifar_balanced_fraction"
DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES = "cifar10_unseen_classes"
DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES = "cifar100_unseen_classes"
DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT = "cifar100_fine_class_disjoint"
DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT = "cifar100_superclass_disjoint"
DATASET_PROTOCOLS = (
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT,
)
CIFAR10_DEVELOPMENT_CLASSES = tuple(range(8))
CIFAR10_HELD_OUT_TEST_CLASSES = (8, 9)
# Legacy contiguous-ID split retained for reproducibility.
CIFAR100_DEVELOPMENT_CLASSES = tuple(range(50))
CIFAR100_HELD_OUT_TEST_CLASSES = tuple(range(50, 100))
CIFAR100_SUPERCLASS_NAMES = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)
CIFAR100_SUPERCLASS_FINE_CLASSES = (
    (4, 30, 55, 72, 95),
    (1, 32, 67, 73, 91),
    (54, 62, 70, 82, 92),
    (9, 10, 16, 28, 61),
    (0, 51, 53, 57, 83),
    (22, 39, 40, 86, 87),
    (5, 20, 25, 84, 94),
    (6, 7, 14, 18, 24),
    (3, 42, 43, 88, 97),
    (12, 17, 37, 68, 76),
    (23, 33, 49, 60, 71),
    (15, 19, 21, 31, 38),
    (34, 63, 64, 66, 75),
    (26, 45, 77, 79, 99),
    (2, 11, 35, 46, 98),
    (27, 29, 44, 78, 93),
    (36, 50, 65, 74, 80),
    (47, 52, 56, 59, 96),
    (8, 13, 48, 58, 90),
    (41, 69, 81, 85, 89),
)
CIFAR100_FINE_CLASS_TO_SUPERCLASS = {
    int(fine_class): int(superclass)
    for superclass, fine_classes in enumerate(CIFAR100_SUPERCLASS_FINE_CLASSES)
    for fine_class in fine_classes
}
CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES = tuple(
    sorted(
        fine_class
        for fine_classes in CIFAR100_SUPERCLASS_FINE_CLASSES
        for fine_class in fine_classes[:3]
    )
)
CIFAR100_FINE_CLASS_DISJOINT_TEST_CLASSES = tuple(
    sorted(set(range(100)) - set(CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES))
)
CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES = tuple(range(0, 20, 2))
CIFAR100_SUPERCLASS_DISJOINT_TEST_SUPERCLASSES = tuple(range(1, 20, 2))
CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES = tuple(
    sorted(
        fine_class
        for superclass_index in CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES
        for fine_class in CIFAR100_SUPERCLASS_FINE_CLASSES[superclass_index]
    )
)
CIFAR100_SUPERCLASS_DISJOINT_TEST_CLASSES = tuple(
    sorted(set(range(100)) - set(CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES))
)
CIFAR_UNSEEN_CLASS_PROTOCOLS = {
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES: {
        "dataset_name": "CIFAR10",
        "development_classes": CIFAR10_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR10_HELD_OUT_TEST_CLASSES,
        "split_basis": "fine_class_contiguous_ids",
    },
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_HELD_OUT_TEST_CLASSES,
        "split_basis": "fine_class_contiguous_ids_legacy",
        "superclass_disjoint_test": False,
    },
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_FINE_CLASS_DISJOINT_TEST_CLASSES,
        "split_basis": "fine_class_within_superclass",
        "development_superclasses": tuple(range(20)),
        "held_out_test_superclasses": tuple(range(20)),
        "superclass_disjoint_test": False,
    },
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_SUPERCLASS_DISJOINT_TEST_CLASSES,
        "split_basis": "superclass",
        "development_superclasses": CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES,
        "held_out_test_superclasses": CIFAR100_SUPERCLASS_DISJOINT_TEST_SUPERCLASSES,
        "superclass_disjoint_test": True,
    },
}
CIFAR_DATASETS = ("CIFAR10", "CIFAR100")
CIFAR_LONG_TAIL_SOURCE = "https://github.com/richardaecn/class-balanced-loss"
VAL_MODE_ALL = "all"
VAL_MODE_MATCH_TRAIN = "match_train"
VAL_MODE_SPLIT_AFTER_APPORTION = "split_after_apportion"
VAL_MODES = (VAL_MODE_ALL, VAL_MODE_MATCH_TRAIN, VAL_MODE_SPLIT_AFTER_APPORTION)
POST_APPORTION_VAL_RATIO = 0.2


class MPerClassSamplerCapacityError(ValueError):
    """Raised when the selected labels cannot fill one M-per-class batch."""

    pass


@dataclass
class DatasetBundle:
    """Datasets and metadata that must move together when splits are rebuilt."""

    train_dataset: Subset
    valid_dataset: Subset
    test_dataset: object
    train_labels_mapper: dict
    split_info: dict | None = None


class CombinedDataset(Dataset):
    """Present several datasets as one sequence while preserving their labels."""

    def __init__(self, datasets, transform=None):
        self.datasets = list(datasets)
        self.transform = transform
        # lengths and cumulative_sizes describe where each child dataset begins
        # and ends in the single combined index space.
        self.lengths = np.asarray([len(dataset) for dataset in self.datasets], dtype=np.int64)
        self.cumulative_sizes = np.cumsum(self.lengths)
        self.classes = getattr(self.datasets[0], "classes", None) if self.datasets else None
        # Flatten labels in exactly the same child-dataset order used by
        # __getitem__, allowing normal Subset/filter operations on this wrapper.
        self.labels = [
            int(label)
            for dataset in self.datasets
            for label in getattr(dataset, "labels", getattr(dataset, "targets", []))
        ]
        if len(self.labels) != len(self):
            raise ValueError("Every combined dataset must expose one label/target per sample")
        self.orig_labels = [
            int(label)
            for dataset in self.datasets
            for label in getattr(
                dataset,
                "orig_labels",
                getattr(dataset, "labels", getattr(dataset, "targets", [])),
            )
        ]
        if len(self.orig_labels) != len(self):
            raise ValueError("Every combined dataset must expose one original label/target per sample")

    def __len__(self):
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        # cumulative_sizes lets us translate a combined index into the source
        # dataset and the corresponding local index without copying samples.
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_size = 0 if dataset_index == 0 else int(self.cumulative_sizes[dataset_index - 1])
        # Subtract the preceding cumulative size to obtain the index expected by
        # the selected child dataset.
        image, label = self.datasets[dataset_index][index - previous_size]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_nested_transform(dataset):
    """Return the transform that currently produces images from a wrapped dataset."""

    if isinstance(dataset, Subset):
        return get_nested_transform(dataset.dataset)
    if isinstance(dataset, CombinedDataset):
        if dataset.transform is not None:
            return dataset.transform
        if not dataset.datasets:
            return None
        return get_nested_transform(dataset.datasets[0])
    return getattr(dataset, "transform", None)


def append_external_unlabeled_dataset(train_dataset, external_root):
    """Append recursively discovered external images to an existing train dataset."""

    train_transform = get_nested_transform(train_dataset)
    external_dataset = local_datasets.RecursiveUnlabeledImageDataset(
        root=external_root,
        transform=train_transform,
    )
    combined = CombinedDataset([train_dataset, external_dataset])
    feature_transform = getattr(train_dataset, "feature_transform", None)
    if feature_transform is not None:
        combined.feature_transform = feature_transform
    return combined, external_dataset


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
    """Create a timestamped run directory and configure console/file logging."""

    start_time = datetime.now()
    logger.remove()
    # Mutating args.log_dir makes the concrete timestamped path available to all
    # later artifact writers.
    args.log_dir = Path("logs") / args.save_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
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


def make_dataloader_kwargs(num_workers, seed, start_method, persistent_workers=False):
    """Build the shared deterministic multiprocessing options for DataLoaders."""

    num_workers = effective_num_workers(num_workers)
    # The generator controls DataLoader/sampler randomness. worker_init_fn then
    # transfers the derived worker seed to NumPy and Python's random module.
    kwargs = {
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": make_torch_generator(seed),
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
        for name, value in (diagnostics or {}).items():
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

    squared_norm = 0.0
    has_gradient = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        has_gradient = True
        gradient = parameter.grad.detach().float()
        squared_norm += float(torch.sum(gradient * gradient).item())
    return squared_norm**0.5 if has_gradient else None


def optimizer_learning_rates(optimizer, optimizer_name):
    return {
        f"train/learning_rate/{optimizer_name}/group_{index}": float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def normalize_dataset_name(dataset_name):
    return dataset_name


def get_dataset_class(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts
    if dataset_name == "CIFAR10":
        return local_datasets.CIFAR10
    if dataset_name == "CIFAR100":
        return local_datasets.CIFAR100

    return getattr(datasets, dataset_name)


def is_dataset_ready(dataset_name, data_root):
    dataset_name = normalize_dataset_name(dataset_name)
    data_root = Path(data_root)

    if dataset_name == "StanfordOnlineProducts":
        sop_root = data_root / "Stanford_Online_Products"
        return (
            (sop_root / "Ebay_train.txt").exists()
            and (sop_root / "Ebay_test.txt").exists()
        )
    if dataset_name == "CIFAR10":
        cifar_root = data_root / "cifar-10-batches-py"
        required_files = [f"data_batch_{index}" for index in range(1, 6)] + ["test_batch", "batches.meta"]
        return all((cifar_root / filename).exists() for filename in required_files)
    if dataset_name == "CIFAR100":
        cifar_root = data_root / "cifar-100-python"
        return all((cifar_root / filename).exists() for filename in ("train", "test", "meta"))

    return data_root.exists()

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
            tfm.RandAugment(num_ops=3, interpolation=tfm.InterpolationMode.BILINEAR),
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


def make_train_loader(
    train_dataset,
    batch_size,
    sampler_m,
    seed,
    num_workers=8,
    start_method="spawn",
    persistent_workers=True,
    length_before_new_iter=None,
):
    """Create a loader whose batches contain ``sampler_m`` samples per class."""

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
        **make_dataloader_kwargs(num_workers, seed, start_method, persistent_workers=persistent_workers),
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
):
    """Create the nearest-neighbor batch loader used by STML."""

    sampler = STMLNearestNeighborBatchSampler(
        embeddings=sampling_embeddings,
        batch_size=batch_size,
        neighbors_per_query=neighbors_per_query,
        seed=seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        **make_dataloader_kwargs(num_workers, seed, start_method, persistent_workers=False),
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


def make_eval_loader(dataset, batch_size=32, seed=0, num_workers=8, start_method="spawn"):
    # Evaluation traverses every item exactly once in dataset order.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_dataloader_kwargs(num_workers, seed, start_method, persistent_workers=True),
    )


def use_feature_transform_for_training(dataset):
    """Replace stochastic training augmentation with the deterministic feature transform."""

    feature_transform = getattr(dataset, "feature_transform", None)
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


def load_dataset_protocol_sources(
    dataset_name,
    data_root,
    train_transform,
    test_transform,
    dataset_protocol=DATASET_PROTOCOL_OFFICIAL,
    download=False,
    cifar_imbalance_factor=None,
    cifar_train_fraction=0.8,
    cifar_test_fraction=0.2,
    seed=0,
):
    """Load official splits or construct a custom CIFAR protocol.

    CIFAR's official train/test splits contain the same classes, so the unseen
    and balanced-fraction protocols recombine them before creating new splits.
    """

    validate_dataset_protocol(dataset_name, dataset_protocol)
    validate_cifar_imbalance_factor(dataset_name, cifar_imbalance_factor)
    validate_cifar_balanced_fraction_protocol(
        dataset_name=dataset_name,
        dataset_protocol=dataset_protocol,
        train_fraction=cifar_train_fraction,
        test_fraction=cifar_test_fraction,
        imbalance_factor=cifar_imbalance_factor,
    )

    dataset_cls = get_dataset_class(dataset_name)
    if dataset_protocol == DATASET_PROTOCOL_OFFICIAL:
        # Preserve the dataset provider's official train/test boundary.
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            transform=train_transform,
            download=download,
        )
        train_val_dataset, imbalance_info = apply_cifar_long_tail(
            train_val_dataset,
            imbalance_factor=cifar_imbalance_factor,
            seed=seed,
        )
        return (
            train_val_dataset,
            dataset_cls(root=str(data_root), split="test", transform=test_transform, download=False),
            {
                "name": DATASET_PROTOCOL_OFFICIAL,
                "source": "official_train_test_splits",
                "cifar_long_tail": imbalance_info,
            },
        )

    if dataset_protocol == DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        official_train = dataset_cls(root=str(data_root), split="train", transform=None, download=download)
        official_test = dataset_cls(root=str(data_root), split="test", transform=None, download=False)
        development_source = CombinedDataset([official_train, official_test], transform=train_transform)
        test_source = CombinedDataset([official_train, official_test], transform=test_transform)
        train_val_dataset, test_dataset, split_info = split_cifar_balanced_by_fraction(
            development_source=development_source,
            test_source=test_source,
            train_fraction=cifar_train_fraction,
            test_fraction=cifar_test_fraction,
            seed=seed,
        )
        return (
            train_val_dataset,
            test_dataset,
            {
                "name": dataset_protocol,
                "source": "combined_official_train_test_splits",
                "sample_disjoint_test": True,
                "class_disjoint_test": False,
                "cifar_long_tail": None,
                **split_info,
            },
        )

    # For unseen-class protocols, the official train/test boundary is
    # intentionally discarded to create a class-disjoint final test set.
    protocol_config = CIFAR_UNSEEN_CLASS_PROTOCOLS[dataset_protocol]
    development_classes = protocol_config["development_classes"]
    held_out_test_classes = protocol_config["held_out_test_classes"]
    official_train = dataset_cls(root=str(data_root), split="train", transform=None, download=download)
    official_test = dataset_cls(root=str(data_root), split="test", transform=None, download=False)
    # Build two views over the same combined samples because development data
    # needs augmentation while the held-out test view must be deterministic.
    development_source = CombinedDataset([official_train, official_test], transform=train_transform)
    test_source = CombinedDataset([official_train, official_test], transform=test_transform)
    train_val_dataset = subset_dataset_by_classes(development_source, development_classes)
    development_pool_size_before_long_tail = len(train_val_dataset)
    train_val_dataset, imbalance_info = apply_cifar_long_tail(
        train_val_dataset,
        imbalance_factor=cifar_imbalance_factor,
        seed=seed,
    )
    test_dataset = subset_dataset_by_classes(test_source, held_out_test_classes)
    assert_disjoint_dataset_classes(train_val_dataset, test_dataset, "development", "test")
    pooled_counts = np.unique(np.asarray(development_source.labels, dtype=np.int64), return_counts=True)[1]
    pooled_samples_per_fine_class = int(pooled_counts[0]) if np.all(pooled_counts == pooled_counts[0]) else None
    protocol_metadata = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in protocol_config.items()
        if key
        in {
            "split_basis",
            "development_superclasses",
            "held_out_test_superclasses",
            "superclass_disjoint_test",
        }
    }
    return (
        train_val_dataset,
        test_dataset,
        {
            "name": dataset_protocol,
            "source": "combined_official_train_test_splits",
            "development_classes": list(development_classes),
            "held_out_test_classes": list(held_out_test_classes),
            "development_pool_size_before_long_tail": development_pool_size_before_long_tail,
            "held_out_test_size": len(test_dataset),
            "pooled_samples_per_fine_class": pooled_samples_per_fine_class,
            "class_disjoint_test": True,
            "cifar_long_tail": imbalance_info,
            **protocol_metadata,
        },
    )


def validate_dataset_protocol(dataset_name, dataset_protocol):
    if dataset_protocol not in DATASET_PROTOCOLS:
        raise ValueError(f"dataset_protocol must be one of {DATASET_PROTOCOLS}: {dataset_protocol}")
    if dataset_protocol == DATASET_PROTOCOL_OFFICIAL:
        return
    if dataset_protocol == DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        if dataset_name not in CIFAR_DATASETS:
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for {CIFAR_DATASETS}"
            )
        return

    supported_dataset = CIFAR_UNSEEN_CLASS_PROTOCOLS[dataset_protocol]["dataset_name"]
    if dataset_name != supported_dataset:
        raise ValueError(f"dataset_protocol={dataset_protocol!r} is only supported for {supported_dataset}")


def validate_cifar_balanced_fraction_protocol(
    dataset_name,
    dataset_protocol,
    train_fraction,
    test_fraction,
    imbalance_factor=None,
):
    """Validate the optional balanced per-class CIFAR train/test split."""

    if dataset_protocol != DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        return
    if dataset_name not in CIFAR_DATASETS:
        raise ValueError(f"cifar_balanced_fraction is only supported for {CIFAR_DATASETS}")
    if imbalance_factor is not None:
        raise ValueError("cifar_imbalance_factor cannot be used with cifar_balanced_fraction")
    if train_fraction is None:
        raise ValueError("cifar_train_fraction must be set for cifar_balanced_fraction")
    if test_fraction is None:
        raise ValueError("cifar_test_fraction must be set for cifar_balanced_fraction")
    if not 0 < train_fraction <= 1:
        raise ValueError("cifar_train_fraction must be in (0, 1]")
    if not 0 < test_fraction <= 1:
        raise ValueError("cifar_test_fraction must be in (0, 1]")
    if train_fraction + test_fraction > 1 + 1e-12:
        raise ValueError("cifar_train_fraction + cifar_test_fraction must be less than or equal to 1")


def validate_cifar_imbalance_factor(dataset_name, imbalance_factor):
    if imbalance_factor is None:
        return
    if dataset_name not in CIFAR_DATASETS:
        raise ValueError(f"cifar_imbalance_factor is only supported for {CIFAR_DATASETS}")
    if not 0 < imbalance_factor <= 1:
        raise ValueError("cifar_imbalance_factor must be in (0, 1]")


def apply_cifar_long_tail(dataset, imbalance_factor, seed):
    """Subsample a CIFAR training source using Cui et al.'s long-tail schedule.

    Adapted from ``get_img_num_per_cls`` and ``get_imbalanced_data`` in
    https://github.com/richardaecn/class-balanced-loss (MIT License,
    Copyright (c) 2018 Yin Cui). The factor is ``img_min / img_max``.
    """

    if imbalance_factor is None:
        return dataset, None

    labels = np.asarray(getattr(dataset, "labels", getattr(dataset, "targets", [])), dtype=np.int64)
    if len(labels) != len(dataset):
        raise ValueError("CIFAR long-tail generation requires one label per dataset sample")

    class_labels, available_counts = np.unique(labels, return_counts=True)
    target_counts = make_cifar_long_tail_class_counts(
        class_labels=class_labels,
        available_counts=available_counts,
        imbalance_factor=imbalance_factor,
    )
    rng = random.Random(seed)
    selected_indices = []
    for class_label in class_labels:
        class_indices = np.flatnonzero(labels == class_label).astype(np.int64).tolist()
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[: target_counts[int(class_label)]])

    # Preserve source order so manifests and nested Subset indices remain easy
    # to inspect while the training sampler still controls iteration order.
    selected_indices = sorted(selected_indices)
    subset = Subset(dataset, selected_indices)
    subset.labels = labels[np.asarray(selected_indices, dtype=np.int64)].astype(np.int64).tolist()
    realized_labels, realized_counts = np.unique(np.asarray(subset.labels, dtype=np.int64), return_counts=True)
    realized_counts = {int(label): int(count) for label, count in zip(realized_labels, realized_counts)}
    min_count = min(realized_counts.values())
    max_count = max(realized_counts.values())
    return (
        subset,
        {
            "enabled": True,
            "factor_img_min_over_img_max": float(imbalance_factor),
            "realized_imbalance_ratio_max_to_min": float(max_count / min_count),
            "seed": int(seed),
            "class_counts": realized_counts,
            "source": CIFAR_LONG_TAIL_SOURCE,
            "attribution": "Class-Balanced Loss Based on Effective Number of Samples, Cui et al., CVPR 2019",
        },
    )


def make_cifar_long_tail_class_counts(class_labels, available_counts, imbalance_factor):
    """Return upstream-compatible exponentially decreasing per-class counts."""

    class_labels = np.asarray(class_labels, dtype=np.int64)
    available_counts = np.asarray(available_counts, dtype=np.int64)
    if len(class_labels) == 0 or len(class_labels) != len(available_counts):
        raise ValueError("CIFAR long-tail generation requires non-empty aligned class labels and counts")

    img_max = int(available_counts.max())
    class_count = len(class_labels)
    target_counts = {}
    for class_index, (class_label, available_count) in enumerate(zip(class_labels, available_counts)):
        exponent = 0.0 if class_count == 1 else class_index / (class_count - 1.0)
        target_count = int(img_max * (imbalance_factor**exponent))
        if target_count < 1:
            raise ValueError(
                "cifar_imbalance_factor produces a class with zero samples; "
                "increase the factor for this dataset"
            )
        target_counts[int(class_label)] = min(target_count, int(available_count))
    return target_counts


def subset_dataset_by_classes(dataset, selected_classes):
    # np.isin creates one mask over the full label array; flatnonzero converts
    # matching positions into the source indices expected by Subset.
    selected_classes = set(int(label) for label in selected_classes)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    indices = np.flatnonzero(np.isin(labels, list(selected_classes))).astype(np.int64).tolist()
    subset = Subset(dataset, indices)
    # torch.utils.data.Subset does not create a labels attribute itself, but the
    # later splitting/sampling code expects one aligned with subset positions.
    subset.labels = labels[np.asarray(indices, dtype=np.int64)].astype(np.int64).tolist()
    return subset


def split_cifar_balanced_by_fraction(
    development_source,
    test_source,
    train_fraction,
    test_fraction,
    seed,
):
    """Select disjoint balanced per-class development and test subsets."""

    labels = np.asarray(development_source.labels, dtype=np.int64)
    test_labels = np.asarray(test_source.labels, dtype=np.int64)
    if (
        len(labels) == 0
        or len(labels) != len(development_source)
        or len(test_source) != len(development_source)
        or not np.array_equal(labels, test_labels)
    ):
        raise ValueError("Balanced CIFAR splitting requires aligned non-empty source datasets and labels")

    class_labels, available_counts = np.unique(labels, return_counts=True)
    balanced_source_per_class = int(np.min(available_counts))
    train_per_class = int(np.floor(balanced_source_per_class * train_fraction))
    test_per_class = int(np.floor(balanced_source_per_class * test_fraction))
    if train_per_class < 1:
        raise ValueError("cifar_train_fraction is too small to select at least one sample per class")
    if test_per_class < 1:
        raise ValueError("cifar_test_fraction is too small to select at least one sample per class")
    if train_per_class + test_per_class > balanced_source_per_class:
        raise ValueError("CIFAR balanced train and test selections exceed the available samples per class")

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for class_label in class_labels:
        class_indices = np.flatnonzero(labels == class_label)
        class_indices = rng.permutation(class_indices)
        train_indices.extend(int(index) for index in class_indices[:train_per_class])
        test_indices.extend(
            int(index)
            for index in class_indices[train_per_class : train_per_class + test_per_class]
        )

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    if set(train_indices) & set(test_indices):
        raise RuntimeError("Balanced CIFAR train and test sample selections overlap")

    train_val_dataset = subset_dataset_by_indices(development_source, train_indices)
    test_dataset = subset_dataset_by_indices(test_source, test_indices)
    return (
        train_val_dataset,
        test_dataset,
        {
            "requested_train_fraction": float(train_fraction),
            "requested_test_fraction": float(test_fraction),
            "realized_train_fraction": float(len(train_val_dataset) / len(development_source)),
            "realized_test_fraction": float(len(test_dataset) / len(development_source)),
            "num_classes": int(len(class_labels)),
            "balanced_source_samples_per_class": balanced_source_per_class,
            "train_samples_per_class": train_per_class,
            "test_samples_per_class": test_per_class,
            "unused_size": int(len(development_source) - len(train_val_dataset) - len(test_dataset)),
            "seed": int(seed),
        },
    )


def subset_dataset_by_indices(dataset, indices):
    """Build a subset with a labels attribute aligned to subset positions."""

    indices = [int(index) for index in indices]
    labels = np.asarray(dataset.labels, dtype=np.int64)
    subset = Subset(dataset, indices)
    subset.labels = labels[np.asarray(indices, dtype=np.int64)].astype(np.int64).tolist()
    return subset


def assert_disjoint_dataset_classes(left_dataset, right_dataset, left_name, right_name):
    left_classes = set(int(label) for label in left_dataset.labels)
    right_classes = set(int(label) for label in right_dataset.labels)
    overlap = left_classes & right_classes
    if overlap:
        raise RuntimeError(f"{left_name} and {right_name} classes overlap: {sorted(overlap)}")


def split_dataset_by_classes(train_val_dataset, split_ratio=0.8, seed=0):
    """Create a holdout split whose train and validation classes are disjoint."""

    unique_classes = np.unique(train_val_dataset.labels)
    # Shuffle classes reproducibly before assigning the first split_ratio
    # fraction to training.
    unique_classes = np.random.default_rng(seed).permutation(unique_classes)
    # Split class IDs, not individual samples, so no class appears on both
    # sides of the holdout.
    split_point = int(len(unique_classes) * split_ratio)
    train_classes = set(unique_classes[:split_point])
    val_classes = set(unique_classes[split_point:])
    # Turn the class partition into sample indices for torch Subset.
    train_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in train_classes]
    val_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in val_classes]
    # Deep-copy the base dataset so changing validation to deterministic
    # transforms cannot also change training augmentation.
    train_dataset = Subset(copy.deepcopy(train_val_dataset), train_indices)
    val_dataset = Subset(copy.deepcopy(train_val_dataset), val_indices)
    # Keep original labels for reporting, but map training labels to a dense
    # zero-based range required by classification losses and samplers.
    train_dataset.orig_labels = [train_val_dataset.labels[i] for i in train_indices]
    train_labels_mapper = {label: i for i, label in enumerate(sorted(set(train_dataset.orig_labels)))}
    train_dataset.labels = [train_labels_mapper[label] for label in train_dataset.orig_labels]
    assert min(train_dataset.labels) == 0
    assert max(train_dataset.labels) == len(set(train_dataset.orig_labels)) - 1

    return train_dataset, val_dataset, train_labels_mapper


def apply_validation_mode(dataset_bundle, val_mode, target_train_size, target_train_num_classes, seed):
    """Optionally downsample validation to resemble the labeled training set."""

    if val_mode not in VAL_MODES:
        raise ValueError(f"val_mode must be one of {VAL_MODES}: {val_mode}")
    if val_mode == VAL_MODE_ALL:
        # Keep the original validation subset unchanged, but still record the
        # target labeled-training size/class count for comparison metadata.
        update_validation_mode_info(
            dataset_bundle,
            mode=val_mode,
            target_train_size=int(target_train_size),
            target_train_num_classes=int(target_train_num_classes),
            original_valid_size=len(dataset_bundle.valid_dataset),
            selected_valid_size=len(dataset_bundle.valid_dataset),
            selected_valid_num_classes=count_dataset_classes(dataset_bundle.valid_dataset),
        )
        return dataset_bundle
    if val_mode == VAL_MODE_SPLIT_AFTER_APPORTION:
        raise ValueError(
            "val_mode='split_after_apportion' must be applied after label apportioning with "
            "apply_post_apportion_validation_split"
        )
    if val_mode != VAL_MODE_MATCH_TRAIN:
        raise ValueError(f"Unknown val_mode: {val_mode}")
    if target_train_size <= 0:
        raise ValueError("target_train_size must be positive for val_mode='match_train'")

    # match_train reduces validation scale to approximately the amount of
    # labeled training data, while balancing the selected validation classes.
    original_valid_dataset = dataset_bundle.valid_dataset
    original_valid_size = len(original_valid_dataset)
    target_valid_size = min(int(target_train_size), original_valid_size)
    selected_indices = select_balanced_subset_indices(
        dataset=original_valid_dataset,
        target_size=target_valid_size,
        target_num_classes=target_train_num_classes,
        seed=seed,
    )
    # selected_indices are source-dataset indices, so wrap the same underlying
    # dataset rather than nesting another Subset around original_valid_dataset.
    dataset_bundle.valid_dataset = Subset(original_valid_dataset.dataset, selected_indices)
    update_validation_mode_info(
        dataset_bundle,
        mode=val_mode,
        target_train_size=int(target_train_size),
        target_train_num_classes=int(target_train_num_classes),
        original_valid_size=original_valid_size,
        selected_valid_size=len(dataset_bundle.valid_dataset),
        selected_valid_num_classes=count_dataset_classes(dataset_bundle.valid_dataset),
    )
    logger.info(
        "Validation mode match_train: "
        f"using {len(dataset_bundle.valid_dataset)} samples from "
        f"{count_dataset_classes(dataset_bundle.valid_dataset)} validation classes "
        f"out of {original_valid_size} validation samples to match "
        f"{target_train_size} labeled/fractioned training samples across {target_train_num_classes} classes"
    )
    return dataset_bundle


def apply_post_apportion_validation_split(
    dataset_bundle,
    labeled_positions=None,
    unlabeled_positions=None,
    seed=0,
    val_ratio=POST_APPORTION_VAL_RATIO,
):
    """Split validation from the selected labeled budget and remap positions.

    Validation samples are removed from the training subset.  Unlabeled
    positions remain eligible for SSL, except where they refer to a removed
    validation sample.
    """

    # Keep references to the old subset and its aligned labels/indices. Every
    # incoming position below addresses this old subset.
    original_train_dataset = dataset_bundle.train_dataset
    labels = np.asarray(original_train_dataset.labels, dtype=np.int64)
    old_indices = np.asarray(original_train_dataset.indices, dtype=np.int64)

    if labeled_positions is None:
        # A full-supervision run apportions every current training position.
        apportioned_positions = np.arange(len(original_train_dataset), dtype=np.int64)
    else:
        apportioned_positions = np.asarray(labeled_positions, dtype=np.int64)
    if unlabeled_positions is None:
        unlabeled_positions = np.array([], dtype=np.int64)
    else:
        unlabeled_positions = np.asarray(unlabeled_positions, dtype=np.int64)

    # Split only the apportioned/labeled pool. Unlabeled candidates are not
    # allowed to become validation examples because their labels are hidden.
    train_labeled_positions, valid_positions = split_positions_stratified_by_label(
        positions=apportioned_positions,
        labels=labels,
        val_ratio=val_ratio,
        seed=seed,
    )
    # Remove validation positions from the full old training subset. This keeps
    # all other samples, including SSL unlabeled candidates, in the new train set.
    valid_position_set = set(int(position) for position in valid_positions)
    remaining_train_positions = np.asarray(
        [
            int(position)
            for position in range(len(original_train_dataset))
            if int(position) not in valid_position_set
        ],
        dtype=np.int64,
    )
    # remaining_train_positions address the old subset.  Build a translation
    # table before replacing it so callers can keep using their split arrays.
    old_to_new_position = {
        int(old_position): int(new_position)
        for new_position, old_position in enumerate(remaining_train_positions)
    }

    # Convert old-subset positions to stable source-dataset indices before
    # constructing replacement train/validation subsets.
    train_indices = old_indices[remaining_train_positions].tolist()
    valid_indices = old_indices[valid_positions].tolist()
    train_dataset, valid_dataset, train_labels_mapper = make_train_valid_subsets(
        original_train_dataset.dataset,
        train_indices,
        valid_indices,
    )

    # Reattach deterministic feature extraction behavior that is not provided
    # automatically by torch Subset.
    feature_transform = getattr(original_train_dataset, "feature_transform", None)
    if feature_transform is not None:
        train_dataset.feature_transform = feature_transform
        set_nested_transform(valid_dataset, feature_transform)

    # Replace the bundle as one unit so its datasets and dense label mapping
    # remain consistent.
    dataset_bundle.train_dataset = train_dataset
    dataset_bundle.valid_dataset = valid_dataset
    dataset_bundle.train_labels_mapper = train_labels_mapper

    # The caller's SSL split must now address the rebuilt train subset, not the
    # old positions used to create validation.
    remapped_labeled_positions = remap_positions(train_labeled_positions, old_to_new_position)
    remapped_unlabeled_positions = remap_positions(unlabeled_positions, old_to_new_position)
    update_post_apportion_validation_info(
        dataset_bundle=dataset_bundle,
        val_ratio=val_ratio,
        original_train_size=len(original_train_dataset),
        apportioned_size=len(apportioned_positions),
        apportioned_num_classes=count_labels_at_positions(labels, apportioned_positions),
        selected_train_size=len(remapped_labeled_positions),
        selected_train_num_classes=count_labels_at_positions(labels, train_labeled_positions),
        selected_valid_size=len(valid_positions),
        selected_valid_num_classes=count_labels_at_positions(labels, valid_positions),
    )
    logger.info(
        "Validation mode split_after_apportion: "
        f"split {len(apportioned_positions)} apportioned labeled samples across "
        f"{count_labels_at_positions(labels, apportioned_positions)} classes into "
        f"{len(remapped_labeled_positions)} train samples across "
        f"{count_labels_at_positions(labels, train_labeled_positions)} classes and "
        f"{len(valid_positions)} validation samples across "
        f"{count_labels_at_positions(labels, valid_positions)} classes"
    )
    return dataset_bundle, remapped_labeled_positions, remapped_unlabeled_positions


def apply_apportioned_cross_validation_split(
    dataset_bundle,
    labeled_positions=None,
    unlabeled_positions=None,
    include_unlabeled=False,
    cv_k=1,
    cv_fold=None,
    cv_mode="group_kfold",
    seed=0,
):
    """Apply CV to the labeled budget, exclude leakage, and remap positions.

    Grouped modes hold out entire classes, so unlabeled samples from validation
    classes must also be excluded.  Non-grouped modes only exclude the exact
    validation positions.
    """

    # Incoming positions address this original/current training subset. The
    # function rebuilds it after selecting one CV fold.
    original_train_dataset = dataset_bundle.train_dataset
    labels = np.asarray(original_train_dataset.labels, dtype=np.int64)
    old_indices = np.asarray(original_train_dataset.indices, dtype=np.int64)
    if cv_mode == CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
        original_labels = getattr(original_train_dataset, "orig_labels", None)
        if original_labels is None:
            raise ValueError(
                f"{CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires original CIFAR-100 "
                "fine labels on the training subset"
            )
        superclass_labels = cifar100_superclass_labels_for_fine_labels(original_labels)
    else:
        superclass_labels = None

    if labeled_positions is None:
        # Without a label budget, every training position participates in CV.
        apportioned_positions = np.arange(len(original_train_dataset), dtype=np.int64)
    else:
        apportioned_positions = np.asarray(labeled_positions, dtype=np.int64)
    if unlabeled_positions is None:
        unlabeled_positions = np.array([], dtype=np.int64)
    else:
        unlabeled_positions = np.asarray(unlabeled_positions, dtype=np.int64)

    # CV sees only the apportioned labeled positions. Hidden labels on the
    # unlabeled pool are used below solely to enforce grouped leakage rules.
    train_labeled_positions, valid_positions = split_positions_cross_validation(
        positions=apportioned_positions,
        labels=labels,
        cv_k=cv_k,
        cv_fold=cv_fold,
        cv_mode=cv_mode,
        seed=seed,
        superclass_labels=superclass_labels,
    )
    valid_position_set = set(int(position) for position in valid_positions)
    valid_labels = set(int(label) for label in labels[valid_positions])
    if cv_mode in GROUPED_CV_MODES:
        # Keeping unlabeled samples from a held-out class would leak validation
        # class information into SSL training.
        train_unlabeled_positions = np.asarray(
            [
                int(position)
                for position in unlabeled_positions
                if int(labels[int(position)]) not in valid_labels
            ],
            dtype=np.int64,
        )
    else:
        # Sample-level CV permits other samples from a validation sample's class
        # in training, but the exact held-out positions must still be excluded.
        train_unlabeled_positions = np.asarray(
            [
                int(position)
                for position in unlabeled_positions
                if int(position) not in valid_position_set
            ],
            dtype=np.int64,
        )
    excluded_unlabeled_count = len(unlabeled_positions) - len(train_unlabeled_positions)
    # Supervised mode rebuilds train from labeled positions only. SSL mode also
    # retains eligible unlabeled candidates for later pseudo-labeling.
    train_position_groups = [train_labeled_positions]
    if include_unlabeled:
        train_position_groups.append(train_unlabeled_positions)
    remaining_train_positions = unique_sorted_positions(train_position_groups)
    if cv_mode in GROUPED_CV_MODES:
        # This assertion guards against future changes to filtering/splitting
        # accidentally reintroducing a held-out class.
        train_labels = set(int(label) for label in labels[remaining_train_positions])
        overlapping_labels = train_labels & valid_labels
        if overlapping_labels:
            raise RuntimeError(
                "Grouped cross-validation produced overlapping train/validation classes: "
                f"{sorted(overlapping_labels)}"
            )
    # Translate old-subset positions into positions in the rebuilt train set.
    old_to_new_position = {
        int(old_position): int(new_position)
        for new_position, old_position in enumerate(remaining_train_positions)
    }

    # Convert old-subset positions to source indices before making new subsets.
    train_indices = old_indices[remaining_train_positions].tolist()
    valid_indices = old_indices[valid_positions].tolist()
    train_dataset, valid_dataset, train_labels_mapper = make_train_valid_subsets(
        original_train_dataset.dataset,
        train_indices,
        valid_indices,
    )

    feature_transform = getattr(original_train_dataset, "feature_transform", None)
    if feature_transform is not None:
        train_dataset.feature_transform = feature_transform
        set_nested_transform(valid_dataset, feature_transform)

    dataset_bundle.train_dataset = train_dataset
    dataset_bundle.valid_dataset = valid_dataset
    dataset_bundle.train_labels_mapper = train_labels_mapper

    # Return positions aligned to the newly rebuilt training subset.
    remapped_labeled_positions = remap_positions(train_labeled_positions, old_to_new_position)
    remapped_unlabeled_positions = remap_positions(train_unlabeled_positions, old_to_new_position)
    update_apportioned_cross_validation_info(
        dataset_bundle=dataset_bundle,
        cv_k=cv_k,
        cv_fold=cv_fold,
        cv_mode=cv_mode,
        include_unlabeled=include_unlabeled,
        original_train_size=len(original_train_dataset),
        apportioned_size=len(apportioned_positions),
        apportioned_num_classes=count_labels_at_positions(labels, apportioned_positions),
        train_labeled_size=len(remapped_labeled_positions),
        train_labeled_num_classes=count_labels_at_positions(labels, train_labeled_positions),
        train_unlabeled_size=len(remapped_unlabeled_positions),
        excluded_unlabeled_size=excluded_unlabeled_count,
        valid_size=len(valid_positions),
        valid_num_classes=count_labels_at_positions(labels, valid_positions),
    )
    logger.info(
        "Validation mode split_after_apportion with CV: "
        f"split {len(apportioned_positions)} apportioned labeled samples across "
        f"{count_labels_at_positions(labels, apportioned_positions)} classes with "
        f"{cv_mode} fold {cv_fold + 1}/{cv_k}; train has "
        f"{len(remapped_labeled_positions)} labeled samples across "
        f"{count_labels_at_positions(labels, train_labeled_positions)} classes"
        f"{f' plus {len(remapped_unlabeled_positions)} unlabeled candidates' if include_unlabeled else ''}; "
        f"excluded {excluded_unlabeled_count} unlabeled candidates from validation "
        f"{'classes' if cv_mode in GROUPED_CV_MODES else 'positions'}; "
        f"validation has {len(valid_positions)} samples across "
        f"{count_labels_at_positions(labels, valid_positions)} classes"
    )
    return dataset_bundle, remapped_labeled_positions, remapped_unlabeled_positions


def unique_sorted_positions(position_groups):
    positions = []
    for group in position_groups:
        # Normalize each possible list/array and ignore empty groups before
        # merging them.
        group = np.asarray(group, dtype=np.int64)
        if len(group) > 0:
            positions.extend(int(position) for position in group)
    # Sorting makes rebuilt Subset order deterministic; set removes accidental
    # overlap between labeled and unlabeled groups.
    return np.asarray(sorted(set(positions)), dtype=np.int64)


def split_positions_stratified_by_label(positions, labels, val_ratio, seed):
    """Split each represented class while retaining samples on both sides."""

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) < 2:
        raise ValueError("val_mode='split_after_apportion' requires at least two apportioned samples")

    rng = np.random.default_rng(seed)
    # selected_labels is aligned one-to-one with positions, not with the full
    # source dataset.
    selected_labels = np.asarray(labels, dtype=np.int64)[positions]
    train_positions = []
    valid_positions = []

    for label in np.unique(selected_labels):
        # Boolean indexing retrieves original training-subset positions for this
        # class from the apportioned pool.
        class_positions = positions[selected_labels == label]
        if len(class_positions) < 2:
            raise ValueError(
                "val_mode='split_after_apportion' requires at least two apportioned samples per class; "
                f"class {int(label)} has {len(class_positions)}"
        )
        class_positions = rng.permutation(class_positions)
        # Clamp the count so every class contributes at least one validation
        # sample and still leaves at least one sample for training.
        valid_count = max(1, int(round(len(class_positions) * val_ratio)))
        valid_count = min(valid_count, len(class_positions) - 1)
        # The shuffled prefix goes to validation and the remainder stays in
        # labeled training.
        valid_positions.extend(class_positions[:valid_count])
        train_positions.extend(class_positions[valid_count:])

    return (
        np.asarray(sorted(int(position) for position in train_positions), dtype=np.int64),
        np.asarray(sorted(int(position) for position in valid_positions), dtype=np.int64),
    )


def cifar100_superclass_labels_for_fine_labels(fine_labels):
    fine_labels = np.asarray(fine_labels, dtype=np.int64)
    unknown_labels = sorted(set(int(label) for label in fine_labels) - set(CIFAR100_FINE_CLASS_TO_SUPERCLASS))
    if unknown_labels:
        raise ValueError(
            f"{CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires CIFAR-100 fine class labels; "
            f"got unknown labels {unknown_labels[:10]}"
        )
    return np.asarray(
        [CIFAR100_FINE_CLASS_TO_SUPERCLASS[int(label)] for label in fine_labels],
        dtype=np.int64,
    )


def make_superclass_balanced_group_folds(labels, superclass_labels, cv_k, seed=0, max_attempts=1000):
    """Grouped folds that preserve at least one fine class per superclass in train."""

    labels = np.asarray(labels, dtype=np.int64)
    superclass_labels = np.asarray(superclass_labels, dtype=np.int64)
    if len(labels) != len(superclass_labels):
        raise ValueError("labels and superclass_labels must have the same length")
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")

    group_positions = {}
    group_to_superclass = {}
    for position, (label, superclass) in enumerate(zip(labels, superclass_labels)):
        label = int(label)
        superclass = int(superclass)
        group_positions.setdefault(label, []).append(position)
        previous_superclass = group_to_superclass.setdefault(label, superclass)
        if previous_superclass != superclass:
            raise ValueError(
                f"{CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires each class group to belong "
                f"to exactly one superclass; class {label} maps to both "
                f"{previous_superclass} and {superclass}"
            )

    if cv_k > len(group_positions):
        raise ValueError(
            f"grouped cross-validation requires cv_k <= number of groups/classes ({len(group_positions)})"
        )

    groups_by_superclass = {}
    for group, superclass in group_to_superclass.items():
        groups_by_superclass.setdefault(int(superclass), []).append(int(group))

    max_val_groups_by_superclass = {}
    for superclass, groups in groups_by_superclass.items():
        if len(groups) < 2:
            raise ValueError(
                f"{CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires at least two selected "
                f"fine classes per superclass; superclass {superclass} has {len(groups)}"
            )
        max_val_groups_by_superclass[int(superclass)] = len(groups) - 1

    group_items = [
        (int(group), int(group_to_superclass[group]), len(group_positions[group]))
        for group in sorted(group_positions)
    ]
    total_groups = len(group_items)
    base_fold_groups = total_groups // cv_k
    fold_group_remainder = total_groups % cv_k
    target_group_counts = [
        base_fold_groups + (1 if fold < fold_group_remainder else 0)
        for fold in range(cv_k)
    ]

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        val_groups_by_fold = [set() for _ in range(cv_k)]
        fold_group_counts = [0 for _ in range(cv_k)]
        fold_sample_counts = [0 for _ in range(cv_k)]
        fold_superclass_counts = [dict() for _ in range(cv_k)]
        shuffled_items = [group_items[int(index)] for index in rng.permutation(len(group_items))]

        for group, superclass, group_size in shuffled_items:
            eligible_folds = []
            for fold in range(cv_k):
                superclass_count = fold_superclass_counts[fold].get(superclass, 0)
                if (
                    fold_group_counts[fold] < target_group_counts[fold]
                    and superclass_count < max_val_groups_by_superclass[superclass]
                ):
                    eligible_folds.append(fold)
            if not eligible_folds:
                break

            fold_tiebreakers = rng.random(len(eligible_folds))
            selected_fold = min(
                zip(eligible_folds, fold_tiebreakers),
                key=lambda item: (
                    fold_sample_counts[item[0]],
                    fold_group_counts[item[0]],
                    item[1],
                ),
            )[0]
            val_groups_by_fold[selected_fold].add(group)
            fold_group_counts[selected_fold] += 1
            fold_sample_counts[selected_fold] += group_size
            fold_superclass_counts[selected_fold][superclass] = (
                fold_superclass_counts[selected_fold].get(superclass, 0) + 1
            )
        else:
            if fold_group_counts == target_group_counts:
                break
    else:
        raise RuntimeError(
            f"Could not build {CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} folds after "
            f"{max_attempts} attempts"
        )

    all_positions = np.arange(len(labels), dtype=np.int64)
    folds = []
    for val_groups in val_groups_by_fold:
        val_positions = []
        for group in sorted(val_groups):
            val_positions.extend(group_positions[int(group)])
        val_positions = np.asarray(sorted(val_positions), dtype=np.int64)
        val_position_set = set(int(position) for position in val_positions)
        train_positions = np.asarray(
            [int(position) for position in all_positions if int(position) not in val_position_set],
            dtype=np.int64,
        )
        folds.append((train_positions, val_positions))
    return folds


def split_positions_cross_validation(positions, labels, cv_k, cv_fold, cv_mode, seed=0, superclass_labels=None):
    """Split only the supplied subset positions using the requested CV policy."""

    if cv_mode not in CV_MODES:
        raise ValueError(f"cv_mode must be one of {CV_MODES}: {cv_mode}")
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")
    if cv_fold is None:
        raise ValueError("cv_fold must be set for cross-validation")
    if not 0 <= cv_fold < cv_k:
        raise ValueError(f"cv_fold must be in [0, {cv_k - 1}], got {cv_fold}")

    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) < cv_k:
        raise ValueError(
            "apportioned cross-validation requires at least cv_k labeled samples; "
            f"got {len(positions)} samples for cv_k={cv_k}"
        )

    # sklearn splitters return offsets into this local positions array, not
    # positions in the original training subset.
    selected_labels = np.asarray(labels, dtype=np.int64)[positions]
    if superclass_labels is None:
        selected_superclass_labels = None
    else:
        superclass_labels = np.asarray(superclass_labels, dtype=np.int64)
        if len(superclass_labels) != len(labels):
            raise ValueError("superclass_labels must be aligned with labels")
        selected_superclass_labels = superclass_labels[positions]
    local_indices = np.arange(len(positions))
    # Using labels as groups makes GroupKFold variants hold out entire classes.
    groups = selected_labels

    if cv_mode == "kfold":
        # Plain KFold balances sample counts without considering class labels.
        splitter = KFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(local_indices)
    elif cv_mode == "group_kfold":
        # Every class appears in exactly one validation fold.
        validate_group_cv(selected_labels, cv_k)
        splitter = GroupKFold(n_splits=cv_k)
        folds = splitter.split(local_indices, selected_labels, groups=groups)
    elif cv_mode == "stratified_kfold":
        # Preserve class proportions, while allowing each class on both sides.
        validate_stratified_cv(selected_labels, cv_k)
        splitter = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(local_indices, selected_labels)
    elif cv_mode == "stratified_group_kfold":
        # Hold out entire classes while trying to balance fold sample counts.
        validate_group_cv(selected_labels, cv_k)
        splitter = StratifiedGroupKFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(local_indices, selected_labels, groups=groups)
    elif cv_mode == CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
        if selected_superclass_labels is None:
            raise ValueError(
                f"{CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires superclass_labels "
                "aligned with labels"
            )
        validate_group_cv(selected_labels, cv_k)
        folds = make_superclass_balanced_group_folds(
            labels=selected_labels,
            superclass_labels=selected_superclass_labels,
            cv_k=cv_k,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")

    # Select the requested fold, then map local splitter offsets back through
    # positions into the original training-subset coordinate system.
    train_local_positions, valid_local_positions = list(folds)[cv_fold]
    return (
        np.asarray(sorted(int(position) for position in positions[train_local_positions]), dtype=np.int64),
        np.asarray(sorted(int(position) for position in positions[valid_local_positions]), dtype=np.int64),
    )


def remap_positions(positions, old_to_new_position):
    """Translate positions after rebuilding a training ``Subset``."""

    # Positions omitted from old_to_new_position were removed when validation
    # was carved out and therefore disappear from the returned array.
    remapped = [old_to_new_position[int(position)] for position in positions if int(position) in old_to_new_position]
    return np.asarray(sorted(remapped), dtype=np.int64)


def count_labels_at_positions(labels, positions):
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return 0
    return int(len(np.unique(np.asarray(labels, dtype=np.int64)[positions])))


def set_nested_transform(dataset, transform):
    """Set the transform on the base dataset beneath any Subset wrappers."""

    if isinstance(dataset, Subset):
        set_nested_transform(dataset.dataset, transform)
        return
    if isinstance(dataset, CombinedDataset):
        if dataset.transform is not None:
            dataset.transform = transform
            return
        for child_dataset in dataset.datasets:
            set_nested_transform(child_dataset, transform)
        return
    if hasattr(dataset, "transform"):
        dataset.transform = transform


def select_balanced_subset_indices(dataset, target_size, target_num_classes, seed):
    """Select a roughly class-balanced validation subset by source index."""

    # indices are source-dataset indices; positions below are offsets into this
    # validation Subset and are converted back at the end.
    indices = np.asarray(dataset.indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    raw_labels = getattr(dataset.dataset, "labels", None)
    if raw_labels is None:
        # Without labels, fall back to a deterministic uniform sample.
        if target_size >= len(indices):
            return indices.tolist()
        return sorted(int(index) for index in rng.choice(indices, size=target_size, replace=False))

    # Align source labels with the current validation subset order.
    labels = np.asarray(raw_labels, dtype=np.int64)[indices]
    positions_by_label = {}
    for label in np.unique(labels):
        label_positions = np.flatnonzero(labels == label)
        positions_by_label[int(label)] = rng.permutation(label_positions)

    eligible_labels = [label for label, positions in positions_by_label.items() if len(positions) >= 2]
    # Requiring two examples per selected class avoids constructing a tiny
    # validation set with many single-example classes that cannot retrieve a
    # same-class neighbor for metric evaluation.
    if not eligible_labels:
        raise ValueError("val_mode='match_train' requires at least one validation class with two or more samples")

    target_size = min(int(target_size), sum(len(positions_by_label[label]) for label in eligible_labels))
    if target_size < 2:
        raise ValueError("val_mode='match_train' requires at least two validation samples")

    # With at least two samples per selected class, target_size // 2 is the
    # maximum number of classes that can fit in the requested subset.
    max_selected_labels = max(1, target_size // 2)
    num_selected_labels = min(int(target_num_classes), len(eligible_labels), max_selected_labels)
    if num_selected_labels <= 0:
        raise ValueError("target_num_classes must be positive for val_mode='match_train'")

    shuffled_labels = rng.permutation(np.asarray(eligible_labels, dtype=np.int64))
    selected_labels = shuffled_labels[:num_selected_labels]

    selected_positions = []
    selected_counts = {int(label): 0 for label in selected_labels}

    labels_order = np.asarray(selected_labels, dtype=np.int64)
    # Round-robin over classes instead of sampling globally, which prevents
    # large classes from dominating the matched validation subset.
    while len(selected_positions) < target_size:
        made_progress = False
        for label in rng.permutation(labels_order):
            label = int(label)
            selected_count = selected_counts[label]
            class_positions = positions_by_label[label]
            if selected_count >= len(class_positions):
                continue

            selected_positions.append(int(class_positions[selected_count]))
            selected_counts[label] = selected_count + 1
            made_progress = True
            if len(selected_positions) == target_size:
                break

        if not made_progress:
            break

    # Translate subset-local positions back to underlying source indices for the
    # new Subset constructor.
    selected_positions = np.asarray(sorted(selected_positions), dtype=np.int64)
    return indices[selected_positions].tolist()


def count_dataset_classes(dataset):
    indices = getattr(dataset, "indices", None)
    raw_labels = getattr(dataset.dataset, "labels", None) if indices is not None else getattr(dataset, "labels", None)
    if raw_labels is None:
        return None
    if indices is None:
        labels = np.asarray(raw_labels, dtype=np.int64)
    else:
        labels = np.asarray(raw_labels, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    return int(len(np.unique(labels)))


def update_validation_mode_info(
    dataset_bundle,
    mode,
    target_train_size,
    target_train_num_classes,
    original_valid_size,
    selected_valid_size,
    selected_valid_num_classes,
):
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["validation_mode"] = {
        "mode": mode,
        "target_train_size": int(target_train_size),
        "target_train_num_classes": int(target_train_num_classes),
        "original_valid_size": int(original_valid_size),
        "selected_valid_size": int(selected_valid_size),
        "selected_valid_num_classes": selected_valid_num_classes,
    }


def update_post_apportion_validation_info(
    dataset_bundle,
    val_ratio,
    original_train_size,
    apportioned_size,
    apportioned_num_classes,
    selected_train_size,
    selected_train_num_classes,
    selected_valid_size,
    selected_valid_num_classes,
):
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["validation_mode"] = {
        "mode": VAL_MODE_SPLIT_AFTER_APPORTION,
        "val_ratio": float(val_ratio),
        "original_train_size": int(original_train_size),
        "apportioned_size": int(apportioned_size),
        "apportioned_num_classes": int(apportioned_num_classes),
        "selected_train_size": int(selected_train_size),
        "selected_train_num_classes": int(selected_train_num_classes),
        "selected_valid_size": int(selected_valid_size),
        "selected_valid_num_classes": int(selected_valid_num_classes),
    }


def update_apportioned_cross_validation_info(
    dataset_bundle,
    cv_k,
    cv_fold,
    cv_mode,
    include_unlabeled,
    original_train_size,
    apportioned_size,
    apportioned_num_classes,
    train_labeled_size,
    train_labeled_num_classes,
    train_unlabeled_size,
    excluded_unlabeled_size,
    valid_size,
    valid_num_classes,
):
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info.update(
        {
            "split_kind": "apportioned_cross_validation",
            "cv_k": int(cv_k),
            "cv_fold": int(cv_fold),
            "cv_mode": cv_mode,
        }
    )
    dataset_bundle.split_info["validation_mode"] = {
        "mode": VAL_MODE_SPLIT_AFTER_APPORTION,
        "strategy": "cross_validation_after_label_apportion",
        "include_unlabeled_in_train": bool(include_unlabeled),
        "original_train_size": int(original_train_size),
        "apportioned_size": int(apportioned_size),
        "apportioned_num_classes": int(apportioned_num_classes),
        "train_labeled_size": int(train_labeled_size),
        "train_labeled_num_classes": int(train_labeled_num_classes),
        "train_unlabeled_size": int(train_unlabeled_size),
        "excluded_unlabeled_size": int(excluded_unlabeled_size),
        "unlabeled_exclusion_scope": "validation_classes" if cv_mode in GROUPED_CV_MODES else "validation_positions",
        "valid_size": int(valid_size),
        "valid_num_classes": int(valid_num_classes),
    }


def make_holdout_split_info(train_val_dataset, train_dataset, valid_dataset, val_mode):
    train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
    valid_indices = np.asarray(valid_dataset.indices, dtype=np.int64)
    labels = np.asarray(train_val_dataset.labels)
    train_classes = set(int(label) for label in labels[train_indices])
    valid_classes = set(int(label) for label in labels[valid_indices])
    return {
        "split_kind": "holdout",
        "val_mode": val_mode,
        "train_size": int(len(train_dataset)),
        "valid_size": int(len(valid_dataset)),
        "num_train_classes": int(len(train_classes)),
        "num_valid_classes": int(len(valid_classes)),
        "class_disjoint": not bool(train_classes & valid_classes),
    }


def split_dataset_cross_validation(train_val_dataset, cv_k, cv_fold, cv_mode, seed=0):
    """Create one train/validation fold from the complete development dataset."""

    if cv_mode not in CV_MODES:
        raise ValueError(f"cv_mode must be one of {CV_MODES}: {cv_mode}")
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")
    if cv_fold is None:
        raise ValueError("cv_fold must be set for cross-validation")
    if not 0 <= cv_fold < cv_k:
        raise ValueError(f"cv_fold must be in [0, {cv_k - 1}], got {cv_fold}")

    # Here indices and positions are identical because splitting begins from the
    # complete development dataset rather than an existing Subset.
    labels = np.asarray(train_val_dataset.labels)
    indices = np.arange(len(labels))
    # Treat class IDs as groups for class-disjoint CV modes.
    groups = labels

    if cv_mode == "kfold":
        splitter = KFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(indices)
    elif cv_mode == "group_kfold":
        validate_group_cv(labels, cv_k)
        splitter = GroupKFold(n_splits=cv_k)
        folds = splitter.split(indices, labels, groups=groups)
    elif cv_mode == "stratified_kfold":
        validate_stratified_cv(labels, cv_k)
        splitter = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(indices, labels)
    elif cv_mode == "stratified_group_kfold":
        validate_group_cv(labels, cv_k)
        splitter = StratifiedGroupKFold(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(indices, labels, groups=groups)
    elif cv_mode == CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
        validate_group_cv(labels, cv_k)
        superclass_labels = cifar100_superclass_labels_for_fine_labels(labels)
        folds = make_superclass_balanced_group_folds(
            labels=labels,
            superclass_labels=superclass_labels,
            cv_k=cv_k,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")

    # sklearn yields offsets; map them through indices for consistency with
    # helper functions that may eventually operate on nontrivial index arrays.
    train_positions, val_positions = list(folds)[cv_fold]
    train_indices = indices[train_positions].tolist()
    val_indices = indices[val_positions].tolist()
    return make_train_valid_subsets(train_val_dataset, train_indices, val_indices)


def validate_group_cv(labels, cv_k):
    num_groups = len(np.unique(labels))
    if cv_k > num_groups:
        raise ValueError(f"grouped cross-validation requires cv_k <= number of groups/classes ({num_groups})")


def validate_stratified_cv(labels, cv_k):
    _, class_counts = np.unique(labels, return_counts=True)
    min_class_count = int(class_counts.min())
    if cv_k > min_class_count:
        raise ValueError(
            f"stratified cross-validation requires cv_k <= samples in the smallest class ({min_class_count})"
        )


def make_train_valid_subsets(train_val_dataset, train_indices, val_indices):
    """Build independent subsets and densely remap the training labels."""

    # Deep copies allow training and validation to hold different transforms
    # without mutating a shared base dataset.
    train_dataset = Subset(copy.deepcopy(train_val_dataset), [int(index) for index in train_indices])
    val_dataset = Subset(copy.deepcopy(train_val_dataset), [int(index) for index in val_indices])
    # orig_labels retain source class IDs for dataset output/reporting.
    train_dataset.orig_labels = [train_val_dataset.labels[int(index)] for index in train_indices]
    # labels are dense IDs used internally by MPerClassSampler and losses.
    train_labels_mapper = {label: i for i, label in enumerate(sorted(set(train_dataset.orig_labels)))}
    train_dataset.labels = [train_labels_mapper[label] for label in train_dataset.orig_labels]
    assert min(train_dataset.labels) == 0
    assert max(train_dataset.labels) == len(set(train_dataset.orig_labels)) - 1

    return train_dataset, val_dataset, train_labels_mapper


def evaluate(model, eval_loader, name="test set", device="cuda", return_per_class=False):
    """Embed a dataset and compute retrieval Precision@1 and MAP@R."""

    # eval() disables training-only behavior such as dropout and updates to
    # normalization statistics.
    model = model.eval()
    all_embeddings = []
    all_labels = []
    # Extract embeddings and labels
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc=name):
            # Keep only CPU NumPy embeddings after each batch to free accelerator
            # memory before processing the next batch.
            forward_cached = getattr(model, "forward_cached", None)
            embeddings = model(images.to(device)) if forward_cached is None else forward_cached(images, device)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_labels.append(labels.cpu())
    # Concatenate all embeddings and labels
    # AccuracyCalculator expects one matrix/vector spanning the full evaluation
    # dataset rather than a list of batches.
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    # Retrieval metrics compare each embedding with the rest of this evaluation
    # set; no classifier head is used.
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "mean_average_precision_at_r"),
        return_per_class=return_per_class,
        k="max_bin_count",
        device=torch.device("cpu"),
    )
    accuracy = accuracy_calculator.get_accuracy(all_embeddings, all_labels)
    if return_per_class:
        per_class_metrics = make_per_class_retrieval_metrics(all_labels, accuracy)
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
    logger.info(f"{name}: Precision@1 = {precision_at_1*100:.1f} , MAP@R = {mean_average_precision_at_r*100:.1f}")
    if return_per_class:
        return precision_at_1, mean_average_precision_at_r, per_class_metrics
    return precision_at_1, mean_average_precision_at_r


def make_per_class_retrieval_metrics(labels, accuracy):
    """Map AccuracyCalculator's sorted per-class values back to class labels."""

    labels = np.asarray(labels).reshape(-1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Same-source retrieval excludes singleton classes because they have no
    # relevant reference after the query itself is removed.
    eligible = [(label, int(count)) for label, count in zip(unique_labels, counts) if count > 1]
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
