"""Dataset containers and external-unlabeled-pool composition."""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset, Subset

from . import local_datasets


EXTERNAL_UNLABELED_FILTER_NONE = "none"


EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT = "compcars_model_min_count"


EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER = "compcars_stml_paper"


EXTERNAL_UNLABELED_FILTER_NABIRDS = "nabirds"


EXTERNAL_UNLABELED_FILTERS = (
    EXTERNAL_UNLABELED_FILTER_NONE,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER,
    EXTERNAL_UNLABELED_FILTER_NABIRDS,
)


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


def append_external_unlabeled_dataset(
    train_dataset,
    external_root,
    external_filter=EXTERNAL_UNLABELED_FILTER_NONE,
    compcars_min_model_images=100,
    compcars_strict_paper_counts=False,
):
    """Append recursively discovered external images to an existing train dataset."""

    train_transform = get_nested_transform(train_dataset)
    if external_filter == EXTERNAL_UNLABELED_FILTER_NONE:
        external_dataset = local_datasets.RecursiveUnlabeledImageDataset(
            root=external_root,
            transform=train_transform,
        )
    elif external_filter == EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT:
        external_dataset = local_datasets.CompCarsModelFilteredUnlabeledImageDataset(
            root=external_root,
            transform=train_transform,
            min_images_per_model=compcars_min_model_images,
        )
    elif external_filter == EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER:
        external_dataset = local_datasets.CompCarsSTMLPaperUnlabeledImageDataset(
            root=external_root,
            transform=train_transform,
            min_images_per_model=compcars_min_model_images,
            strict_paper_counts=compcars_strict_paper_counts,
        )
    elif external_filter == EXTERNAL_UNLABELED_FILTER_NABIRDS:
        external_dataset = local_datasets.NABirdsUnlabeledImageDataset(
            root=external_root,
            transform=train_transform,
        )
    else:
        raise ValueError(f"Unknown external_unlabeled_filter: {external_filter}")
    combined = CombinedDataset([train_dataset, external_dataset])
    feature_transform = getattr(train_dataset, "feature_transform", None)
    if feature_transform is not None:
        combined.feature_transform = feature_transform
    return combined, external_dataset
