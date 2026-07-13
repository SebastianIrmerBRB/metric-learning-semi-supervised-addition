"""Dataset holdout, cross-validation, and validation-budget splits."""

import copy

import numpy as np
from loguru import logger
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Subset

from .dataset_composition import CombinedDataset
from .dataset_constants import (
    CIFAR100_FINE_CLASS_TO_SUPERCLASS,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
    CV_MODES,
    GROUPED_CV_MODES,
    POST_APPORTION_VAL_RATIO,
    VAL_MODE_ALL,
    VAL_MODE_MATCH_TRAIN,
    VAL_MODE_SPLIT_AFTER_APPORTION,
    VAL_MODES,
)


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


def split_dataset_by_classes_superclass_balanced(train_val_dataset, split_ratio=0.8, seed=0):
    """Create a CIFAR-100 class holdout that keeps every superclass in train."""

    labels = np.asarray(train_val_dataset.labels, dtype=np.int64)
    superclass_labels = cifar100_superclass_labels_for_fine_labels(labels)
    train_indices, val_indices = split_positions_superclass_balanced_holdout(
        labels=labels,
        superclass_labels=superclass_labels,
        split_ratio=split_ratio,
        seed=seed,
    )
    return make_train_valid_subsets(train_val_dataset, train_indices, val_indices)


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
    """Create a class-disjoint validation holdout after label apportioning.

    Validation classes are chosen from the apportioned/labeled pool. Every
    source-train sample from those classes is moved to validation, and unlabeled
    candidates from those classes are removed from the SSL pool.
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

    # Choose validation classes using only the apportioned/labeled pool.
    # Validation itself is limited to those held-out apportioned samples, while
    # all other source-train samples from validation classes are excluded from
    # train and SSL.
    train_labeled_positions, valid_labeled_positions = split_positions_class_disjoint_by_label(
        positions=apportioned_positions,
        labels=labels,
        val_ratio=val_ratio,
        seed=seed,
    )
    valid_labels = set(int(label) for label in labels[valid_labeled_positions])
    valid_positions = np.asarray(sorted(int(position) for position in valid_labeled_positions), dtype=np.int64)
    excluded_validation_class_positions = np.asarray(
        [
            int(position)
            for position, label in enumerate(labels)
            if int(label) in valid_labels
        ],
        dtype=np.int64,
    )
    train_unlabeled_positions = np.asarray(
        [
            int(position)
            for position in unlabeled_positions
            if int(labels[int(position)]) not in valid_labels
        ],
        dtype=np.int64,
    )
    excluded_unlabeled_count = len(unlabeled_positions) - len(train_unlabeled_positions)
    excluded_validation_class_train_count = len(excluded_validation_class_positions) - len(valid_positions)
    # Remove every validation-class sample from the rebuilt training subset.
    valid_position_set = set(int(position) for position in excluded_validation_class_positions)
    remaining_train_positions = np.asarray(
        [
            int(position)
            for position in range(len(original_train_dataset))
            if int(position) not in valid_position_set
        ],
        dtype=np.int64,
    )
    train_labels = set(int(label) for label in labels[remaining_train_positions])
    overlapping_labels = train_labels & valid_labels
    if overlapping_labels:
        raise RuntimeError(
            "Post-apportion validation split produced overlapping train/validation classes: "
            f"{sorted(overlapping_labels)}"
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
    remapped_unlabeled_positions = remap_positions(train_unlabeled_positions, old_to_new_position)
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
        selected_valid_labeled_size=len(valid_labeled_positions),
        excluded_unlabeled_size=excluded_unlabeled_count,
        excluded_validation_class_train_size=excluded_validation_class_train_count,
        class_disjoint=True,
    )
    logger.info(
        "Validation mode split_after_apportion: "
        "class-disjoint holdout split "
        f"{len(apportioned_positions)} apportioned labeled samples across "
        f"{count_labels_at_positions(labels, apportioned_positions)} classes into "
        f"{len(remapped_labeled_positions)} train samples across "
        f"{count_labels_at_positions(labels, train_labeled_positions)} classes; "
        f"validation has {len(valid_positions)} apportioned samples across "
        f"{count_labels_at_positions(labels, valid_positions)} held-out classes; "
        f"excluded {excluded_unlabeled_count} unlabeled candidates and "
        f"{excluded_validation_class_train_count} additional source-train samples from validation classes"
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


def split_positions_class_disjoint_by_label(positions, labels, val_ratio, seed):
    """Split supplied positions by whole class labels."""

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) < 2:
        raise ValueError("val_mode='split_after_apportion' requires at least two apportioned samples")

    rng = np.random.default_rng(seed)
    # selected_labels is aligned one-to-one with positions, not with the full
    # source dataset.
    selected_labels = np.asarray(labels, dtype=np.int64)[positions]
    unique_labels = np.unique(selected_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "val_mode='split_after_apportion' requires at least two apportioned classes "
            "for a class-disjoint validation split"
        )
    shuffled_labels = rng.permutation(unique_labels)
    train_class_count = int(len(shuffled_labels) * (1.0 - val_ratio))
    train_class_count = min(max(train_class_count, 1), len(shuffled_labels) - 1)
    train_labels = set(int(label) for label in shuffled_labels[:train_class_count])
    valid_labels = set(int(label) for label in shuffled_labels[train_class_count:])

    train_positions = positions[np.isin(selected_labels, list(train_labels))]
    valid_positions = positions[np.isin(selected_labels, list(valid_labels))]

    return (
        np.asarray(sorted(int(position) for position in train_positions), dtype=np.int64),
        np.asarray(sorted(int(position) for position in valid_positions), dtype=np.int64),
    )


def split_positions_stratified_by_label(positions, labels, val_ratio, seed):
    """Legacy name for the post-apportion class-disjoint holdout splitter."""

    return split_positions_class_disjoint_by_label(
        positions=positions,
        labels=labels,
        val_ratio=val_ratio,
        seed=seed,
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


def build_class_groups_by_superclass(labels, superclass_labels):
    labels = np.asarray(labels, dtype=np.int64)
    superclass_labels = np.asarray(superclass_labels, dtype=np.int64)
    if len(labels) != len(superclass_labels):
        raise ValueError("labels and superclass_labels must have the same length")

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
    return group_positions, group_to_superclass


def split_positions_superclass_balanced_holdout(
    labels,
    superclass_labels,
    split_ratio=0.8,
    seed=0,
    max_attempts=1000,
):
    """Class-disjoint holdout that leaves each represented superclass in train."""

    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

    labels = np.asarray(labels, dtype=np.int64)
    group_positions, group_to_superclass = build_class_groups_by_superclass(labels, superclass_labels)
    total_groups = len(group_positions)
    target_train_groups = int(total_groups * split_ratio)
    target_val_groups = total_groups - target_train_groups
    if target_train_groups <= 0 or target_val_groups <= 0:
        raise ValueError(
            "class holdout requires at least one training class and one validation class; "
            f"got {target_train_groups} train and {target_val_groups} validation classes"
        )

    groups_by_superclass = {}
    for group, superclass in group_to_superclass.items():
        groups_by_superclass.setdefault(int(superclass), []).append(int(group))
    max_val_groups_by_superclass = {
        int(superclass): max(0, len(groups) - 1)
        for superclass, groups in groups_by_superclass.items()
    }
    max_total_val_groups = sum(max_val_groups_by_superclass.values())
    if target_val_groups > max_total_val_groups:
        raise ValueError(
            "Cannot build a superclass-balanced holdout with the requested split_ratio: "
            f"need {target_val_groups} validation classes but can hold out at most "
            f"{max_total_val_groups} while preserving every represented superclass in training"
        )

    group_items = [
        (int(group), int(group_to_superclass[group]), len(group_positions[group]))
        for group in sorted(group_positions)
    ]
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        val_groups = set()
        val_superclass_counts = {}
        shuffled_items = [group_items[int(index)] for index in rng.permutation(len(group_items))]

        for group, superclass, _group_size in shuffled_items:
            if len(val_groups) >= target_val_groups:
                break
            superclass_count = val_superclass_counts.get(superclass, 0)
            if superclass_count >= max_val_groups_by_superclass[superclass]:
                continue
            val_groups.add(group)
            val_superclass_counts[superclass] = superclass_count + 1

        if len(val_groups) == target_val_groups:
            break
    else:
        raise RuntimeError(
            "Could not build superclass-balanced class holdout after "
            f"{max_attempts} attempts"
        )

    val_positions = []
    for group in sorted(val_groups):
        val_positions.extend(group_positions[int(group)])
    val_positions = np.asarray(sorted(val_positions), dtype=np.int64)
    val_position_set = set(int(position) for position in val_positions)
    all_positions = np.arange(len(labels), dtype=np.int64)
    train_positions = np.asarray(
        [int(position) for position in all_positions if int(position) not in val_position_set],
        dtype=np.int64,
    )

    train_superclasses = set(int(superclass) for superclass in np.asarray(superclass_labels)[train_positions])
    missing_superclasses = set(groups_by_superclass) - train_superclasses
    if missing_superclasses:
        raise RuntimeError(
            "Superclass-balanced holdout failed to preserve training superclasses: "
            f"{sorted(missing_superclasses)}"
        )
    return train_positions, val_positions


def make_superclass_balanced_group_folds(labels, superclass_labels, cv_k, seed=0, max_attempts=1000):
    """Grouped folds that preserve at least one fine class per superclass in train."""

    labels = np.asarray(labels, dtype=np.int64)
    superclass_labels = np.asarray(superclass_labels, dtype=np.int64)
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")

    group_positions, group_to_superclass = build_class_groups_by_superclass(labels, superclass_labels)

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
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
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
    selected_valid_labeled_size=None,
    excluded_unlabeled_size=0,
    excluded_validation_class_train_size=0,
    class_disjoint=False,
):
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["validation_mode"] = {
        "mode": VAL_MODE_SPLIT_AFTER_APPORTION,
        "strategy": "class_disjoint_holdout_after_label_apportion"
        if class_disjoint
        else "sample_stratified_after_label_apportion",
        "val_ratio": float(val_ratio),
        "original_train_size": int(original_train_size),
        "apportioned_size": int(apportioned_size),
        "apportioned_num_classes": int(apportioned_num_classes),
        "selected_train_size": int(selected_train_size),
        "selected_train_num_classes": int(selected_train_num_classes),
        "selected_valid_size": int(selected_valid_size),
        "selected_valid_num_classes": int(selected_valid_num_classes),
        "selected_valid_labeled_size": int(
            selected_valid_size if selected_valid_labeled_size is None else selected_valid_labeled_size
        ),
        "excluded_unlabeled_size": int(excluded_unlabeled_size),
        "excluded_validation_class_train_size": int(excluded_validation_class_train_size),
        "unlabeled_exclusion_scope": "validation_classes" if class_disjoint else "validation_positions",
        "class_disjoint": bool(class_disjoint),
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
