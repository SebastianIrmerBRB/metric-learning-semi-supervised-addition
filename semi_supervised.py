import copy
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from scipy import stats

import numpy as np
import torch
from loguru import logger

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from tqdm import tqdm

import utils


UNLABELED_TARGET = -1
UPDATE_MODES = {"once", "every_epoch"}
LABEL_SAMPLING_MODES = {"per_class_min", "global_budget", "class_subset", "class_subset_k_shot"}


@dataclass(frozen=True)
class SemiSupervisedConfig:
    method: str = "none"
    update_mode: str = "once"
    warmup_epochs: int = 0
    label_sampling_mode: str = "global_budget"
    labeled_fraction: float = 1.0
    labeled_per_class: int | None = None
    seed: int | None = None
    confidence_threshold: float = 0.0
    max_unlabeled_samples: int | None = None
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
    labeled_positions: np.ndarray
    unlabeled_positions: np.ndarray


@dataclass(frozen=True)
class PseudoLabelResult:
    positions: np.ndarray
    mapped_labels: np.ndarray
    confidences: np.ndarray | None = None


class RelabeledSubset(Dataset):
    def __init__(self, dataset, positions, orig_labels, mapped_labels):
        if not (len(positions) == len(orig_labels) == len(mapped_labels)):
            raise ValueError("positions, orig_labels, and mapped_labels must have the same length")
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)
        self.orig_labels = [int(label) for label in orig_labels]
        self.labels = [int(label) for label in mapped_labels]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        image, _ = self.dataset[int(self.positions[index])]
        return image, self.orig_labels[index]


class BaseSemiSupervisedMethod:
    name = None

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
        raise NotImplementedError


class SklearnGraphSSLMethod(BaseSemiSupervisedMethod):
    def __init__(self, name, estimator_cls, default_params):
        self.name = name
        self.estimator_cls = estimator_cls
        self.default_params = default_params

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
        if len(split.unlabeled_positions) == 0:
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        ssl_targets = np.concatenate(
            [
                labels[split.labeled_positions],
                np.full(len(split.unlabeled_positions), UNLABELED_TARGET, dtype=np.int64),
            ]
        )

        features = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        params = dict(self.default_params)
        params.update(config.method_params)
        logger.info(f"Fitting {self.name} with params: {params}")
        estimator = self.estimator_cls(**params)
        estimator.fit(features, ssl_targets)

        unlabeled_start = len(split.labeled_positions)
        pseudo_labels = np.asarray(estimator.transduction_[unlabeled_start:], dtype=np.int64)
        distributions = getattr(estimator, "label_distributions_", None)
        confidences = None
        if distributions is not None:
            confidences = np.asarray(distributions[unlabeled_start:].max(axis=1), dtype=np.float32)

        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=confidences,
        )


class FaissKNNMajorityVotePseudoLabeler(BaseSemiSupervisedMethod):
    def __init__(self, name, n_neighbors=10):
        self.name = name
        self.n_neighbors = n_neighbors

    def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
        if len(split.unlabeled_positions) == 0:
            return PseudoLabelResult(
                positions=np.array([], dtype=np.int64),
                mapped_labels=np.array([], dtype=np.int64),
                confidences=np.array([], dtype=np.float32),
            )
        if len(split.labeled_positions) == 0:
            raise ValueError(f"{self.name} requires at least one labeled sample")

        params = dict(config.method_params)
        n_neighbors = int(params.pop("n_neighbors", self.n_neighbors))
        if params:
            raise ValueError(f"Unknown {self.name} params: {sorted(params)}")
        if n_neighbors <= 0:
            raise ValueError(f"{self.name} n_neighbors must be positive")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(f"{self.name} requires the faiss-cpu package") from exc

        k = min(n_neighbors, len(split.labeled_positions))
        if k < n_neighbors:
            logger.warning(
                f"{self.name} requested n_neighbors={n_neighbors}, but only "
                f"{len(split.labeled_positions)} labeled samples are available; using {k}"
            )

        ssl_positions = np.concatenate([split.labeled_positions, split.unlabeled_positions])
        embeddings = extract_embeddings(
            model=model,
            dataset=train_dataset,
            positions=ssl_positions,
            device=device,
            batch_size=config.embedding_batch_size,
            num_workers=config.embedding_num_workers,
            seed=config.seed if epoch is None else config.seed + epoch,
            start_method=start_method,
            desc=f"{self.name} embeddings",
        )

        num_labeled = len(split.labeled_positions)
        labeled_embeddings = np.ascontiguousarray(embeddings[:num_labeled], dtype=np.float32)
        unlabeled_embeddings = np.ascontiguousarray(embeddings[num_labeled:], dtype=np.float32)

        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        labeled_targets = labels[split.labeled_positions]

        faiss.normalize_L2(labeled_embeddings)
        faiss.normalize_L2(unlabeled_embeddings)

        index = faiss.IndexFlatIP(labeled_embeddings.shape[1])
        index.add(labeled_embeddings)
        similarities, neighbor_indices = index.search(unlabeled_embeddings, k)

        neighbor_labels = labeled_targets[neighbor_indices]

        if k == 1:
            pseudo_labels = neighbor_labels[:, 0]
            confidences = similarities[:, 0].astype(np.float32)
        else:
            pseudo_labels, vote_counts = majority_vote(neighbor_labels)
            confidences = (vote_counts / k).astype(np.float32)
        for confidence in confidences:
            print(confidence)

        return PseudoLabelResult(
            positions=split.unlabeled_positions,
            mapped_labels=pseudo_labels,
            confidences=confidences,
        )


def majority_vote(label_rows):
    pseudo_labels = np.empty(label_rows.shape[0], dtype=np.int64)
    vote_counts = np.empty(label_rows.shape[0], dtype=np.int64)

    for row_index, labels in enumerate(label_rows):
        unique_labels, counts = np.unique(labels, return_counts=True)
        best_index = int(np.argmax(counts))
        pseudo_labels[row_index] = unique_labels[best_index]
        vote_counts[row_index] = counts[best_index]

    return pseudo_labels, vote_counts


METHOD_REGISTRY = {
    # first implementation
    "faiss_majority_vote_knn": FaissKNNMajorityVotePseudoLabeler(name="faiss_knn", n_neighbors=10), # find reference anywhere in literature
    # implement distance based faiss implementation
    # these are basically unusable. Way to much memory usage. Interesting thought though <-_->.
    "sklearn_label_spreading": SklearnGraphSSLMethod(
        name="sklearn_label_spreading",
        estimator_cls=LabelSpreading,
        default_params={"kernel": "knn", "n_neighbors": 10, "alpha": 0.2, "max_iter": 30},
    ),
    "sklearn_label_propagation": SklearnGraphSSLMethod(
        name="sklearn_label_propagation",
        estimator_cls=LabelPropagation,
        default_params={"kernel": "knn", "n_neighbors": 10, "max_iter": 30},
    ),
}


def load_ssl_config(config_path, default_seed=0):
    if config_path is None:
        return SemiSupervisedConfig(seed=default_seed)

    path = Path(config_path)
    with path.open() as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"SSL config must be a JSON object: {path}")

    allowed_keys = set(SemiSupervisedConfig.__dataclass_fields__)
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown SSL config keys in {path}: {unknown_keys}. "
            "Put method-specific settings under method_params."
        )

    config = SemiSupervisedConfig(**raw_config)
    if config.seed is None:
        config = replace(config, seed=default_seed)
    validate_ssl_config(config, path)
    return config


def validate_ssl_config(config, path=None):
    source = f" in {path}" if path is not None else ""
    if config.method != "none" and config.method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown SSL method{source}: {config.method}. Available: {available_methods()}")
    if config.update_mode not in UPDATE_MODES:
        raise ValueError(f"Unknown SSL update_mode{source}: {config.update_mode}. Available: {sorted(UPDATE_MODES)}")
    if config.label_sampling_mode not in LABEL_SAMPLING_MODES:
        raise ValueError(
            f"Unknown label_sampling_mode{source}: {config.label_sampling_mode}. "
            f"Available: {sorted(LABEL_SAMPLING_MODES)}"
        )
    if config.method == "none" and config.update_mode != "once":
        raise ValueError(f"update_mode must be 'once' when method is 'none'{source}")
    if config.warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be non-negative{source}")
    if config.method == "none" and config.warmup_epochs != 0:
        raise ValueError(f"warmup_epochs must be 0 when method is 'none'{source}")
    if config.seed is None:
        raise ValueError(f"seed must be resolved before validation{source}")
    if config.labeled_per_class is None and not (0 < config.labeled_fraction <= 1):
        raise ValueError(f"labeled_fraction must be in (0, 1]{source}")
    if config.labeled_per_class is not None and config.labeled_per_class <= 0:
        raise ValueError(f"labeled_per_class must be positive{source}")
    if config.labeled_per_class is not None and config.label_sampling_mode not in {
        "per_class_min",
        "class_subset_k_shot",
    }:
        raise ValueError(
            f"labeled_per_class is only supported with label_sampling_mode='per_class_min' "
            f"or 'class_subset_k_shot'{source}"
        )
    if config.label_sampling_mode == "class_subset_k_shot" and config.labeled_per_class is None:
        raise ValueError(f"class_subset_k_shot requires labeled_per_class to set k-shot{source}")
    if not (0 <= config.confidence_threshold <= 1):
        raise ValueError(f"confidence_threshold must be in [0, 1]{source}")
    if config.max_unlabeled_samples is not None and config.max_unlabeled_samples <= 0:
        raise ValueError(f"max_unlabeled_samples must be positive{source}")
    if config.embedding_batch_size <= 0:
        raise ValueError(f"embedding_batch_size must be positive{source}")
    if config.embedding_num_workers < 0:
        raise ValueError(f"embedding_num_workers must be non-negative{source}")
    if not isinstance(config.method_params, dict):
        raise ValueError(f"method_params must be an object{source}")


def available_methods():
    return ["none", *sorted(METHOD_REGISTRY)]


def prepare_ssl_split(train_dataset, config):
    if not config.enabled:
        return None

    logger.info(f"Using semi-supervised config: {config.to_dict()}")
    return prepare_label_split(train_dataset, config)


def prepare_label_split(train_dataset, config):
    split = make_semi_supervised_split(
        labels=train_dataset.labels,
        label_sampling_mode=config.label_sampling_mode,
        labeled_fraction=config.labeled_fraction,
        labeled_per_class=config.labeled_per_class,
        max_unlabeled_samples=config.max_unlabeled_samples,
        seed=config.seed,
    )
    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    labeled_labels = labels[split.labeled_positions]
    num_labeled_classes = int(len(np.unique(labeled_labels))) if len(labeled_labels) > 0 else 0
    num_total_classes = int(len(np.unique(labels)))
    logger.info(
        "Semi-supervised split: "
        f"label_mode={config.label_sampling_mode}, "
        f"{len(split.labeled_positions)} labeled across {num_labeled_classes}/{num_total_classes} classes, "
        f"{len(split.unlabeled_positions)} unlabeled candidates"
    )
    return split


def build_ssl_training_dataset(
    model,
    train_dataset,
    train_labels_mapper,
    device,
    config,
    split=None,
    epoch=None,
    start_method="spawn",
):
    if not config.enabled:
        return train_dataset

    if split is None:
        split = prepare_ssl_split(train_dataset, config)

    epoch_label = "" if epoch is None else f" for epoch {epoch}"
    logger.info(f"Generating {config.method} pseudo-labels{epoch_label}")

    method = METHOD_REGISTRY[config.method]
    pseudo_labels = method.generate_pseudo_labels(
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=epoch,
        start_method=start_method,
    )
    pseudo_labels = filter_pseudo_labels(
        pseudo_labels=pseudo_labels,
        confidence_threshold=config.confidence_threshold,
        valid_mapped_labels=set(train_labels_mapper.values()),
    )

    if len(pseudo_labels.positions) == 0:
        logger.info("No pseudo-labels selected; training on labeled subset only")
    else:
        logger.info(f"Selected {len(pseudo_labels.positions)} pseudo-labeled samples")

    return make_relabeled_training_dataset(
        train_dataset=train_dataset,
        train_labels_mapper=train_labels_mapper,
        labeled_positions=split.labeled_positions,
        pseudo_labels=pseudo_labels,
    )


def build_labeled_training_dataset(train_dataset, train_labels_mapper, split):
    empty_pseudo_labels = PseudoLabelResult(
        positions=np.array([], dtype=np.int64),
        mapped_labels=np.array([], dtype=np.int64),
        confidences=np.array([], dtype=np.float32),
    )
    return make_relabeled_training_dataset(
        train_dataset=train_dataset,
        train_labels_mapper=train_labels_mapper,
        labeled_positions=split.labeled_positions,
        pseudo_labels=empty_pseudo_labels,
    )


def make_semi_supervised_split(
    labels,
    label_sampling_mode,
    labeled_fraction,
    labeled_per_class,
    max_unlabeled_samples,
    seed,
):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    labeled_positions, unlabeled_by_label = select_labeled_positions(
        labels=labels,
        label_sampling_mode=label_sampling_mode,
        labeled_fraction=labeled_fraction,
        labeled_per_class=labeled_per_class,
        rng=rng,
    )

    labeled_positions = np.asarray(sorted(labeled_positions), dtype=np.int64)
    unlabeled_positions = concatenate_position_groups(unlabeled_by_label.values())

    if max_unlabeled_samples is not None and len(unlabeled_positions) > max_unlabeled_samples:
        unlabeled_positions = rng.choice(unlabeled_positions, size=max_unlabeled_samples, replace=False)

    return SemiSupervisedSplit(
        labeled_positions=labeled_positions,
        unlabeled_positions=np.asarray(sorted(unlabeled_positions), dtype=np.int64),
    )


def select_labeled_positions(labels, label_sampling_mode, labeled_fraction, labeled_per_class, rng):
    if label_sampling_mode == "per_class_min":
        return select_per_class_min_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    if label_sampling_mode == "global_budget":
        return select_global_budget_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset":
        return select_class_subset_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset_k_shot":
        return select_class_subset_k_shot_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    raise ValueError(f"Unknown label_sampling_mode: {label_sampling_mode}")


def make_permuted_positions_by_label(labels, rng):
    positions_by_label = {}
    for label in np.unique(labels):
        class_positions = np.flatnonzero(labels == label)
        positions_by_label[int(label)] = rng.permutation(class_positions)
    return positions_by_label


def select_per_class_min_labeled_positions(labels, labeled_fraction, labeled_per_class, rng):
    positions_by_label = make_permuted_positions_by_label(labels, rng)
    labeled_positions = []
    unlabeled_by_label = {}

    for label, class_positions in positions_by_label.items():
        if labeled_per_class is None:
            num_labeled = max(1, int(round(len(class_positions) * labeled_fraction)))
        else:
            num_labeled = min(labeled_per_class, len(class_positions))

        labeled_positions.extend(class_positions[:num_labeled])
        unlabeled_by_label[int(label)] = class_positions[num_labeled:]

    return labeled_positions, unlabeled_by_label


def select_global_budget_labeled_positions(labels, labeled_fraction, rng):
    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    target_labeled = max(1, int(np.floor(len(labels) * labeled_fraction)))
    target_labeled = min(target_labeled, len(labels))
    selected_counts = {int(label): 0 for label in unique_labels}
    labeled_positions = []

    while len(labeled_positions) < target_labeled:
        made_progress = False
        for label in rng.permutation(unique_labels):
            label = int(label)
            selected_count = selected_counts[label]
            class_positions = positions_by_label[label]
            if selected_count >= len(class_positions):
                continue

            labeled_positions.append(class_positions[selected_count])
            selected_counts[label] = selected_count + 1
            made_progress = True
            if len(labeled_positions) == target_labeled:
                break

        if not made_progress:
            break

    unlabeled_by_label = {
        int(label): class_positions[selected_counts[int(label)] :]
        for label, class_positions in positions_by_label.items()
    }
    return labeled_positions, unlabeled_by_label


def select_class_subset_labeled_positions(labels, class_fraction, rng):
    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            labeled_positions.extend(class_positions)
            unlabeled_by_label[int(label)] = np.array([], dtype=np.int64)
        else:
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def select_class_subset_k_shot_labeled_positions(labels, class_fraction, labeled_per_class, rng):
    if labeled_per_class is None:
        raise ValueError("class_subset_k_shot requires labeled_per_class to set k-shot")

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            num_labeled = min(int(labeled_per_class), len(class_positions))
            labeled_positions.extend(class_positions[:num_labeled])
            unlabeled_by_label[int(label)] = class_positions[num_labeled:]
        else:
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def concatenate_position_groups(groups):
    arrays = [np.asarray(group, dtype=np.int64) for group in groups if len(group) > 0]
    if not arrays:
        return np.array([], dtype=np.int64)
    return np.concatenate(arrays).astype(np.int64, copy=False)


def filter_pseudo_labels(pseudo_labels, confidence_threshold, valid_mapped_labels):
    keep = np.isin(pseudo_labels.mapped_labels, list(valid_mapped_labels))

    if pseudo_labels.confidences is not None:
        keep = keep & (pseudo_labels.confidences >= confidence_threshold)

    dropped = int(len(keep) - keep.sum())
    if dropped > 0:
        logger.info(f"Dropped {dropped} pseudo-labels below confidence threshold or outside known classes")

    return PseudoLabelResult(
        positions=pseudo_labels.positions[keep],
        mapped_labels=pseudo_labels.mapped_labels[keep],
        confidences=None if pseudo_labels.confidences is None else pseudo_labels.confidences[keep],
    )


def make_relabeled_training_dataset(train_dataset, train_labels_mapper, labeled_positions, pseudo_labels):
    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    orig_labels = np.asarray(train_dataset.orig_labels, dtype=np.int64)
    inverse_labels_mapper = {mapped: original for original, mapped in train_labels_mapper.items()}

    all_positions = np.concatenate([labeled_positions, pseudo_labels.positions])
    all_mapped_labels = np.concatenate([labels[labeled_positions], pseudo_labels.mapped_labels])
    pseudo_orig_labels = np.asarray(
        [inverse_labels_mapper[int(label)] for label in pseudo_labels.mapped_labels],
        dtype=np.int64,
    )
    all_orig_labels = np.concatenate([orig_labels[labeled_positions], pseudo_orig_labels])

    return RelabeledSubset(
        dataset=train_dataset,
        positions=all_positions,
        orig_labels=all_orig_labels,
        mapped_labels=all_mapped_labels,
    )


def extract_embeddings(model, dataset, positions, device, batch_size, num_workers, seed, start_method, desc):
    feature_dataset = make_feature_dataset(dataset)
    loader = DataLoader(
        Subset(feature_dataset, [int(position) for position in positions]),
        batch_size=batch_size,
        shuffle=False,
        **utils.make_dataloader_kwargs(num_workers, seed, start_method),
    )

    was_training = model.training
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=desc):
            embeddings = model(images.to(device))
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
    if was_training:
        model.train()

    return np.concatenate(all_embeddings)


def make_feature_dataset(dataset):
    feature_dataset = copy.deepcopy(dataset)
    feature_transform = getattr(dataset, "feature_transform", None)
    if feature_transform is not None:
        set_nested_transform(feature_dataset, feature_transform)
    return feature_dataset


def set_nested_transform(dataset, transform):
    current_dataset = dataset
    while hasattr(current_dataset, "dataset"):
        current_dataset = current_dataset.dataset

    if not hasattr(current_dataset, "transform"):
        raise AttributeError("Could not find a transform attribute on the feature dataset")

    current_dataset.transform = transform
