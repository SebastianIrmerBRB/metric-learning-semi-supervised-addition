import copy
import csv
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import local_datasets

TENSORBOARD_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = exc


DATALOADER_START_METHODS = ("spawn", "forkserver", "fork", "default")
CV_MODES = ("kfold", "group_kfold", "stratified_kfold", "stratified_group_kfold")
VAL_MODE_ALL = "all"
VAL_MODE_MATCH_TRAIN = "match_train"
VAL_MODES = (VAL_MODE_ALL, VAL_MODE_MATCH_TRAIN)
DATASET_NAME_ALIASES = {
    "CIFAR-10": "CIFAR10",
    "Cifar-10": "CIFAR10",
    "Cifar10": "CIFAR10",
    "cifar-10": "CIFAR10",
    "cifar10": "CIFAR10",
}


@dataclass
class DatasetBundle:
    train_dataset: Subset
    valid_dataset: Subset
    test_dataset: object
    train_labels_mapper: dict
    split_info: dict | None = None


def initialize_logger(args):
    start_time = datetime.now()
    logger.remove()
    args.log_dir = Path("logs") / args.save_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(args.log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(args.log_dir / "debug.log", level="DEBUG")
    sys.excepthook = lambda _, value, tb: logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"The outputs are being saved in {args.log_dir}")


def seed_everything(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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


def make_dataloader_kwargs(num_workers, seed, start_method):
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    kwargs = {
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": make_torch_generator(seed),
    }
    if num_workers > 0 and start_method != "default":
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"dataloader_start_method={start_method!r} is not available on this platform. "
                f"Available: {mp.get_all_start_methods()}"
            )
        kwargs["multiprocessing_context"] = start_method
    return kwargs


class MetricsLogger:
    def __init__(self, log_dir, args):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the tensorboard package. "
                "Install it with `pip install -r requirements.txt`."
            ) from TENSORBOARD_IMPORT_ERROR

        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "metrics.csv"
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=["step", "epoch", "split", "loss", "precision_at_1", "mean_average_precision_at_r"],
        )
        self.csv_writer.writeheader()
        self.writer.add_text("run/arguments", "\n".join(f"{key}: {value}" for key, value in vars(args).items()), 0)
        logger.info(f"TensorBoard logs are being saved in {self.log_dir / 'tensorboard'}")
        logger.info(f"CSV metrics are being saved in {self.csv_path}")

    def log_train_batch(self, loss, epoch, step):
        self.writer.add_scalar("train/batch_loss", loss, step)
        self._write_row(step=step, epoch=epoch, split="train_batch", loss=loss)

    def log_train_epoch(self, loss, epoch, step):
        self.writer.add_scalar("train/epoch_loss", loss, step)
        self._write_row(step=step, epoch=epoch, split="train_epoch", loss=loss)

    def log_eval(self, split, precision_at_1, mean_average_precision_at_r, step, epoch=None):
        self.writer.add_scalar(f"{split}/precision_at_1", precision_at_1, step)
        self.writer.add_scalar(f"{split}/mean_average_precision_at_r", mean_average_precision_at_r, step)
        self._write_row(
            step=step,
            epoch=epoch,
            split=split,
            precision_at_1=precision_at_1,
            mean_average_precision_at_r=mean_average_precision_at_r,
        )

    def close(self):
        self.csv_file.close()
        self.writer.close()

    def _write_row(
        self,
        step,
        epoch,
        split,
        loss=None,
        precision_at_1=None,
        mean_average_precision_at_r=None,
    ):
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

def normalize_dataset_name(dataset_name):
    return DATASET_NAME_ALIASES.get(dataset_name, dataset_name)


def get_dataset_class(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts
    if dataset_name == "CIFAR10":
        return local_datasets.CIFAR10

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

    return data_root.exists()

def setup_dataset_bundle(dataset_name, seed, cv_k=1, cv_fold=None, cv_mode="group_kfold", val_mode=VAL_MODE_ALL):
    dataset_name = normalize_dataset_name(dataset_name)
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
    dataset_cls = get_dataset_class(dataset_name)
    download = not is_dataset_ready(dataset_name, data_root)

    train_val_dataset = dataset_cls(
        root=str(data_root),
        split="train",
        transform=train_transform,
        download=download,
    )

    test_dataset = dataset_cls(
        root=str(data_root),
        split="test",
        transform=test_transform,
        download=False,
    )

    if val_mode not in VAL_MODES:
        raise ValueError(f"val_mode must be one of {VAL_MODES}: {val_mode}")

    if cv_k > 1:
        train_dataset, valid_dataset, train_labels_mapper = split_dataset_cross_validation(
            train_val_dataset,
            cv_k=cv_k,
            cv_fold=cv_fold,
            cv_mode=cv_mode,
            seed=seed,
        )
        split_label = f"{cv_mode} fold {cv_fold + 1}/{cv_k}"
        split_info = {
            "split_kind": "cross_validation",
            "cv_k": int(cv_k),
            "cv_fold": int(cv_fold),
            "cv_mode": cv_mode,
        }
    else:
        train_dataset, valid_dataset, train_labels_mapper = split_dataset_by_classes(
            train_val_dataset,
            seed=seed,
        )
        split_label = "holdout"
        split_info = make_holdout_split_info(
            train_val_dataset=train_val_dataset,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            val_mode=val_mode,
        )
    train_dataset.feature_transform = test_transform
    valid_dataset.dataset.transform = test_transform

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


def make_train_loader(train_dataset, batch_size, sampler_m, seed, num_workers=8, start_method="spawn"):
    sampler_length = make_sampler_epoch_length(len(train_dataset), batch_size)
    validate_m_per_class_sampler_capacity(train_dataset.labels, batch_size, sampler_m)
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
        **make_dataloader_kwargs(num_workers, seed, start_method),
    )
    logger.info(
        "Train loader: "
        f"{len(train_dataset)} samples, {len(set(train_dataset.labels))} labels, "
        f"{len(sampler)} sampled examples/epoch, {len(train_loader)} batches/epoch"
    )
    return train_loader


def validate_m_per_class_sampler_capacity(labels, batch_size, sampler_m):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sampler_m <= 0:
        raise ValueError("sampler_m must be positive")

    num_labels = len(set(int(label) for label in labels))
    max_samples_per_sampler_pass = sampler_m * num_labels
    min_required_labels = int(np.ceil(batch_size / sampler_m))
    if max_samples_per_sampler_pass < batch_size:
        raise ValueError(
            "MPerClassSampler cannot build one training batch from the selected labeled data: "
            f"batch_size={batch_size}, sampler_m={sampler_m}, labeled_classes={num_labels}, "
            f"sampler_m*labeled_classes={max_samples_per_sampler_pass}. "
            f"Need at least {min_required_labels} labeled classes. "
            "For k samples from every training class, use label_sampling_mode='per_class_min'. "
            "For class_subset_k_shot, increase labeled_fraction so the class subset contains enough classes, "
            "or reduce batch_size/sampler_m."
        )


def make_sampler_epoch_length(dataset_size, batch_size):
    if dataset_size <= 0:
        raise ValueError("training dataset must not be empty")
    return max(batch_size, int(np.ceil(dataset_size / batch_size) * batch_size))


def make_eval_loader(dataset, batch_size=32, seed=0, num_workers=8, start_method="spawn"):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **make_dataloader_kwargs(num_workers, seed, start_method),
    )


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
):
    dataset_bundle = setup_dataset_bundle(dataset_name, seed, cv_k, cv_fold, cv_mode, val_mode)
    train_loader = make_train_loader(dataset_bundle.train_dataset, batch_size, sampler_m, seed, num_workers, start_method)
    valid_loader = make_eval_loader(dataset_bundle.valid_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    test_loader = make_eval_loader(dataset_bundle.test_dataset, seed=seed, num_workers=num_workers, start_method=start_method)
    return train_loader, valid_loader, test_loader, dataset_bundle.train_labels_mapper


def split_dataset_by_classes(train_val_dataset, split_ratio=0.8, seed=0):
    unique_classes = np.unique(train_val_dataset.labels)
    unique_classes = np.random.default_rng(seed).permutation(unique_classes)
    # Determine split point
    split_point = int(len(unique_classes) * split_ratio)
    train_classes = set(unique_classes[:split_point])
    val_classes = set(unique_classes[split_point:])
    # Split indices by class
    train_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in train_classes]
    val_indices = [i for i, label in enumerate(train_val_dataset.labels) if label in val_classes]
    # We need to deepcopy the dataset so that the two subsets can use different transforms
    train_dataset = Subset(copy.deepcopy(train_val_dataset), train_indices)
    val_dataset = Subset(copy.deepcopy(train_val_dataset), val_indices)
    # Assign labels to train_dataset. Necessary for the sampler
    train_dataset.orig_labels = [train_val_dataset.labels[i] for i in train_indices]
    # Remap indexes so that they start from 0
    train_labels_mapper = {label: i for i, label in enumerate(sorted(set(train_dataset.orig_labels)))}
    train_dataset.labels = [train_labels_mapper[label] for label in train_dataset.orig_labels]
    assert min(train_dataset.labels) == 0
    assert max(train_dataset.labels) == len(set(train_dataset.orig_labels)) - 1

    return train_dataset, val_dataset, train_labels_mapper


def apply_validation_mode(dataset_bundle, val_mode, target_train_size, target_train_num_classes, seed):
    if val_mode not in VAL_MODES:
        raise ValueError(f"val_mode must be one of {VAL_MODES}: {val_mode}")
    if val_mode == VAL_MODE_ALL:
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
    if val_mode != VAL_MODE_MATCH_TRAIN:
        raise ValueError(f"Unknown val_mode: {val_mode}")
    if target_train_size <= 0:
        raise ValueError("target_train_size must be positive for val_mode='match_train'")

    original_valid_dataset = dataset_bundle.valid_dataset
    original_valid_size = len(original_valid_dataset)
    target_valid_size = min(int(target_train_size), original_valid_size)
    selected_indices = select_balanced_subset_indices(
        dataset=original_valid_dataset,
        target_size=target_valid_size,
        target_num_classes=target_train_num_classes,
        seed=seed,
    )
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


def select_balanced_subset_indices(dataset, target_size, target_num_classes, seed):
    indices = np.asarray(dataset.indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    raw_labels = getattr(dataset.dataset, "labels", None)
    if raw_labels is None:
        if target_size >= len(indices):
            return indices.tolist()
        return sorted(int(index) for index in rng.choice(indices, size=target_size, replace=False))

    labels = np.asarray(raw_labels, dtype=np.int64)[indices]
    positions_by_label = {}
    for label in np.unique(labels):
        label_positions = np.flatnonzero(labels == label)
        positions_by_label[int(label)] = rng.permutation(label_positions)

    eligible_labels = [label for label, positions in positions_by_label.items() if len(positions) >= 2]
    if not eligible_labels:
        raise ValueError("val_mode='match_train' requires at least one validation class with two or more samples")

    target_size = min(int(target_size), sum(len(positions_by_label[label]) for label in eligible_labels))
    if target_size < 2:
        raise ValueError("val_mode='match_train' requires at least two validation samples")

    max_selected_labels = max(1, target_size // 2)
    num_selected_labels = min(int(target_num_classes), len(eligible_labels), max_selected_labels)
    if num_selected_labels <= 0:
        raise ValueError("target_num_classes must be positive for val_mode='match_train'")

    shuffled_labels = rng.permutation(np.asarray(eligible_labels, dtype=np.int64))
    selected_labels = shuffled_labels[:num_selected_labels]

    selected_positions = []
    selected_counts = {int(label): 0 for label in selected_labels}

    labels_order = np.asarray(selected_labels, dtype=np.int64)
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
    if cv_mode not in CV_MODES:
        raise ValueError(f"cv_mode must be one of {CV_MODES}: {cv_mode}")
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")
    if cv_fold is None:
        raise ValueError("cv_fold must be set for cross-validation")
    if not 0 <= cv_fold < cv_k:
        raise ValueError(f"cv_fold must be in [0, {cv_k - 1}], got {cv_fold}")

    labels = np.asarray(train_val_dataset.labels)
    indices = np.arange(len(labels))
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
    else:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")

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
    train_dataset = Subset(copy.deepcopy(train_val_dataset), [int(index) for index in train_indices])
    val_dataset = Subset(copy.deepcopy(train_val_dataset), [int(index) for index in val_indices])
    train_dataset.orig_labels = [train_val_dataset.labels[int(index)] for index in train_indices]
    train_labels_mapper = {label: i for i, label in enumerate(sorted(set(train_dataset.orig_labels)))}
    train_dataset.labels = [train_labels_mapper[label] for label in train_dataset.orig_labels]
    assert min(train_dataset.labels) == 0
    assert max(train_dataset.labels) == len(set(train_dataset.orig_labels)) - 1

    return train_dataset, val_dataset, train_labels_mapper


def evaluate(model, eval_loader, name="test set", device="cuda"):
    model = model.eval()
    all_embeddings = []
    all_labels = []
    # Extract embeddings and labels
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc=name):
            embeddings = model(images.to(device))
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_labels.append(labels.cpu())
    # Concatenate all embeddings and labels
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    # Use AccuracyCalculator to compute metrics
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "mean_average_precision_at_r"), k="max_bin_count", device=torch.device("cpu")
    )
    accuracy = accuracy_calculator.get_accuracy(all_embeddings, all_labels)
    precision_at_1, mean_average_precision_at_r = accuracy["precision_at_1"], accuracy["mean_average_precision_at_r"]
    logger.info(f"{name}: Precision@1 = {precision_at_1*100:.1f} , MAP@R = {mean_average_precision_at_r*100:.1f}")
    return precision_at_1, mean_average_precision_at_r
