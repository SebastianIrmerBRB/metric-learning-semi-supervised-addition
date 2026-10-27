"""Dataset holdout, cross-validation, and validation-budget splits."""

import copy

import numpy as np
from loguru import logger
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Subset

from .dataset_composition import CombinedDataset
from .dataset_constants import (
    CV_MODES,
    GROUPED_CV_MODES,
    POST_APPORTION_VAL_RATIO,
    QUERY_GALLERY_EVALUATION,
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


def derive_query_gallery_indices(labels, gallery_fraction, seed):
    """Split each class of one evaluation split into gallery and query rows.

    Datasets that define a query/gallery protocol define it only for their test
    split, so a validation fold that wants the same retrieval setup has to
    derive one. Every class contributes at least one gallery row and at least
    one query row; a class holding a single row cannot do both and becomes
    gallery-only, where it still acts as a distractor but is never asked to
    retrieve itself.
    """

    gallery_fraction = float(gallery_fraction)
    if not 0 < gallery_fraction < 1:
        raise ValueError(f"gallery_fraction must be in (0, 1), got {gallery_fraction}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) == 0:
        raise ValueError("a derived query/gallery split needs at least one sample")

    # One generator for the whole split, consumed in sorted class order, so the
    # partition depends only on the seed and the labels rather than on the
    # dictionary/iteration order of the classes.
    rng = np.random.default_rng(seed)
    query_parts = []
    gallery_parts = []
    gallery_only_classes = 0
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if len(positions) < 2:
            gallery_only_classes += 1
            gallery_parts.append(positions)
            continue
        positions = rng.permutation(positions)
        gallery_size = int(round(gallery_fraction * len(positions)))
        # Clamp rather than round-to-zero: both sides must stay non-empty or the
        # class drops out of the evaluation entirely.
        gallery_size = min(max(gallery_size, 1), len(positions) - 1)
        gallery_parts.append(positions[:gallery_size])
        query_parts.append(positions[gallery_size:])

    if not query_parts:
        raise ValueError(
            "a derived query/gallery split needs at least one class with two or "
            "more samples; every class in this split holds exactly one"
        )
    # Sorted so the partition reads as positions into the split rather than as
    # the shuffled order the classes were drawn in.
    query_indices = np.sort(np.concatenate(query_parts)).astype(np.int64)
    gallery_indices = np.sort(np.concatenate(gallery_parts)).astype(np.int64)
    info = {
        "mode": QUERY_GALLERY_EVALUATION,
        "source": "derived_per_class_split",
        "gallery_fraction": gallery_fraction,
        "seed": int(seed),
        "num_queries": int(len(query_indices)),
        "num_gallery": int(len(gallery_indices)),
        "num_query_classes": int(len(np.unique(labels[query_indices]))),
        "num_gallery_classes": int(len(np.unique(labels[gallery_indices]))),
        "gallery_only_classes": int(gallery_only_classes),
    }
    return query_indices, gallery_indices, info


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


def rebuild_train_valid_subsets(
    dataset_bundle,
    original_train_dataset,
    old_indices,
    remaining_train_positions,
    valid_positions,
    train_labeled_positions,
    train_unlabeled_positions,
):
    """Replace the bundle's train/validation subsets cut from old-subset positions.

    Returns the SSL split remapped onto the rebuilt train subset, which is what
    callers must address from now on.
    """

    old_to_new_position = {
        int(old_position): int(new_position)
        for new_position, old_position in enumerate(remaining_train_positions)
    }
    # Stable source-dataset indices, not old-subset positions, define the subsets.
    train_dataset, valid_dataset, train_labels_mapper = make_train_valid_subsets(
        original_train_dataset.dataset,
        old_indices[remaining_train_positions].tolist(),
        old_indices[valid_positions].tolist(),
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
    return (
        remap_positions(train_labeled_positions, old_to_new_position),
        remap_positions(train_unlabeled_positions, old_to_new_position),
    )


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
    remapped_labeled_positions, remapped_unlabeled_positions = rebuild_train_valid_subsets(
        dataset_bundle,
        original_train_dataset,
        old_indices,
        remaining_train_positions,
        valid_positions,
        train_labeled_positions,
        train_unlabeled_positions,
    )
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

    Grouped modes hold out entire classes, so matching unlabeled samples must
    also be excluded. Non-grouped modes only exclude the
    exact validation positions.
    """

    # Incoming positions address this original/current training subset. The
    # function rebuilds it after selecting one CV fold.
    original_train_dataset = dataset_bundle.train_dataset
    labels = np.asarray(original_train_dataset.labels, dtype=np.int64)
    old_indices = np.asarray(original_train_dataset.indices, dtype=np.int64)

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
    remapped_labeled_positions, remapped_unlabeled_positions = rebuild_train_valid_subsets(
        dataset_bundle,
        original_train_dataset,
        old_indices,
        remaining_train_positions,
        valid_positions,
        train_labeled_positions,
        train_unlabeled_positions,
    )
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
    validation_exclusion_unit = "classes" if cv_mode in GROUPED_CV_MODES else "positions"
    logger.info(
        "Validation mode split_after_apportion with CV: "
        f"split {len(apportioned_positions)} apportioned labeled samples across "
        f"{count_labels_at_positions(labels, apportioned_positions)} classes with "
        f"{cv_mode} fold {cv_fold + 1}/{cv_k}; train has "
        f"{len(remapped_labeled_positions)} labeled samples across "
        f"{count_labels_at_positions(labels, train_labeled_positions)} classes"
        f"{f' plus {len(remapped_unlabeled_positions)} unlabeled candidates' if include_unlabeled else ''}; "
        f"excluded {excluded_unlabeled_count} unlabeled candidates from validation "
        f"{validation_exclusion_unit}; "
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


def validate_cross_validation_request(cv_k, cv_fold, cv_mode):
    if cv_mode not in CV_MODES:
        raise ValueError(f"cv_mode must be one of {CV_MODES}: {cv_mode}")
    if cv_k <= 1:
        raise ValueError("cv_k must be greater than 1 for cross-validation")
    if cv_fold is None:
        raise ValueError("cv_fold must be set for cross-validation")
    if not 0 <= cv_fold < cv_k:
        raise ValueError(f"cv_fold must be in [0, {cv_k - 1}], got {cv_fold}")


def select_cross_validation_fold(indices, labels, cv_k, cv_fold, cv_mode, seed):
    """Return the ``cv_fold``-th (train, validation) offsets into ``indices``.

    Class IDs serve as the groups of the grouped modes, so those hold out entire
    classes. Every mode shuffles, so the class-to-fold assignment follows the
    seed rather than being identical for every seed.
    """

    if cv_mode == "kfold":
        # Plain KFold balances sample counts without considering class labels.
        folds = KFold(n_splits=cv_k, shuffle=True, random_state=seed).split(indices)
    elif cv_mode == "stratified_kfold":
        # Preserve class proportions, while allowing each class on both sides.
        validate_stratified_cv(labels, cv_k)
        folds = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed).split(indices, labels)
    elif cv_mode in ("group_kfold", "stratified_group_kfold"):
        # Every class appears in exactly one validation fold; the stratified
        # variant also tries to balance fold sample counts.
        validate_group_cv(labels, cv_k)
        splitter_class = GroupKFold if cv_mode == "group_kfold" else StratifiedGroupKFold
        splitter = splitter_class(n_splits=cv_k, shuffle=True, random_state=seed)
        folds = splitter.split(indices, labels, groups=labels)
    else:
        raise ValueError(f"Unsupported cv_mode: {cv_mode}")
    return list(folds)[cv_fold]


def split_positions_cross_validation(positions, labels, cv_k, cv_fold, cv_mode, seed=0):
    """Split only the supplied subset positions using the requested CV policy."""

    validate_cross_validation_request(cv_k, cv_fold, cv_mode)
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) < cv_k:
        raise ValueError(
            "apportioned cross-validation requires at least cv_k labeled samples; "
            f"got {len(positions)} samples for cv_k={cv_k}"
        )

    # sklearn splitters return offsets into this local positions array, not
    # positions in the original training subset; they are mapped back through
    # positions into the original training-subset coordinate system.
    selected_labels = np.asarray(labels, dtype=np.int64)[positions]
    train_local_positions, valid_local_positions = select_cross_validation_fold(
        np.arange(len(positions)), selected_labels, cv_k, cv_fold, cv_mode, seed
    )
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
        "unlabeled_exclusion_scope": (
            "validation_classes" if cv_mode in GROUPED_CV_MODES else "validation_positions"
        ),
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

    validate_cross_validation_request(cv_k, cv_fold, cv_mode)
    # Here indices and positions are identical because splitting begins from the
    # complete development dataset rather than an existing Subset.
    labels = np.asarray(train_val_dataset.labels)
    indices = np.arange(len(labels))
    train_positions, val_positions = select_cross_validation_fold(
        indices, labels, cv_k, cv_fold, cv_mode, seed
    )
    return make_train_valid_subsets(
        train_val_dataset,
        indices[train_positions].tolist(),
        indices[val_positions].tolist(),
    )


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
