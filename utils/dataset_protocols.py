"""Dataset discovery and source-protocol construction."""

import random
from pathlib import Path

import numpy as np
import pytorch_metric_learning.datasets as datasets
from torch.utils.data import Subset

from . import local_datasets
from .dataset_composition import CombinedDataset
from .dataset_constants import (
    CIFAR_DATASETS,
    CIFAR_LONG_TAIL_SOURCE,
    CIFAR_UNSEEN_CLASS_PROTOCOLS,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOLS,
    QUERY_GALLERY_EVALUATION,
)
from .dataset_splits import (
    assert_disjoint_dataset_classes,
    split_cifar_balanced_by_fraction,
    subset_dataset_by_classes,
)


def normalize_dataset_name(dataset_name):
    aliases = {
        "DeepFashionInShopRetrieval": "DeepFashionInShop",
        "InShop": "DeepFashionInShop",
        "InShopRetrieval": "DeepFashionInShop",
    }
    return aliases.get(dataset_name, dataset_name)


def get_dataset_class(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts
    if dataset_name == "CIFAR10":
        return local_datasets.CIFAR10
    if dataset_name == "CIFAR100":
        return local_datasets.CIFAR100
    if dataset_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop

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
    if dataset_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop.find_metadata_file(data_root) is not None

    return data_root.exists()


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
    dataset_class_resolver=None,
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

    if dataset_class_resolver is None:
        dataset_class_resolver = get_dataset_class
    dataset_cls = dataset_class_resolver(dataset_name)
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
        test_dataset = dataset_cls(root=str(data_root), split="test", transform=test_transform, download=False)
        protocol_info = {
            "name": DATASET_PROTOCOL_OFFICIAL,
            "source": "official_train_test_splits",
            "cifar_long_tail": imbalance_info,
        }
        query_indices = getattr(test_dataset, "query_indices", None)
        gallery_indices = getattr(test_dataset, "gallery_indices", None)
        if query_indices is not None and gallery_indices is not None:
            protocol_info.update(
                {
                    "source": "official_train_query_gallery_splits",
                    "test_retrieval_mode": QUERY_GALLERY_EVALUATION,
                    "num_test_queries": int(len(query_indices)),
                    "num_test_gallery": int(len(gallery_indices)),
                }
            )
        return (
            train_val_dataset,
            test_dataset,
            protocol_info,
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
