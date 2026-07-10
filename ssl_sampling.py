"""Deterministic label-budget and class-sampling strategies."""

import numpy as np

from ssl_config import SemiSupervisedSplit


def make_semi_supervised_split(
    labels,
    label_sampling_mode,
    labeled_fraction,
    labeled_per_class,
    max_unlabeled_samples,
    seed,
):
    """Select labeled positions and optionally cap the unlabeled candidate pool."""

    # One RNG instance drives all choices in this split, making it reproducible
    # from a single seed.
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    labeled_positions, unlabeled_by_label = select_labeled_positions(
        labels=labels,
        label_sampling_mode=label_sampling_mode,
        labeled_fraction=labeled_fraction,
        labeled_per_class=labeled_per_class,
        rng=rng,
    )

    # Sort outputs so their order is stable and independent of dictionary/loop
    # traversal after random selection has finished.
    labeled_positions = np.asarray(sorted(labeled_positions), dtype=np.int64)
    unlabeled_positions = concatenate_position_groups(unlabeled_by_label.values())

    # Capping can make embedding extraction and graph methods tractable on
    # large datasets while leaving the labeled budget unchanged.
    if max_unlabeled_samples is not None and len(unlabeled_positions) > max_unlabeled_samples:
        # Sample without replacement, then sort below to restore deterministic
        # dataset order for embedding extraction.
        unlabeled_positions = rng.choice(unlabeled_positions, size=max_unlabeled_samples, replace=False)

    return SemiSupervisedSplit(
        labeled_positions=labeled_positions,
        unlabeled_positions=np.asarray(sorted(unlabeled_positions), dtype=np.int64),
    )


def select_labeled_positions(labels, label_sampling_mode, labeled_fraction, labeled_per_class, rng):
    """Dispatch to one of the supported label-budget semantics."""

    if label_sampling_mode == "per_class_min":
        # Every class receives at least one labeled example (or a fixed k).
        return select_per_class_min_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    if label_sampling_mode == "per_class_imbalanced":
        # Keep every class visible, but randomize how much of each class is
        # labeled while preserving the per_class_min total budget.
        return select_per_class_imbalanced_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    if label_sampling_mode == "global_budget":
        # labeled_fraction controls the total number of labeled samples.
        return select_global_budget_labeled_positions(
            labels=labels,
            labeled_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset":
        # labeled_fraction controls how many classes are fully labeled.
        return select_class_subset_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            rng=rng,
        )
    if label_sampling_mode == "class_subset_k_shot":
        # labeled_fraction selects classes; labeled_per_class selects k examples
        # from each chosen class.
        return select_class_subset_k_shot_labeled_positions(
            labels=labels,
            class_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
            rng=rng,
        )
    raise ValueError(f"Unknown label_sampling_mode: {label_sampling_mode}")


def make_permuted_positions_by_label(labels, rng):
    """Group positions by class and independently shuffle every class."""

    positions_by_label = {}
    for label in np.unique(labels):
        # flatnonzero returns positions in the current training subset. Shuffle
        # each class independently so prefixes form random labeled selections.
        class_positions = np.flatnonzero(labels == label)
        positions_by_label[int(label)] = rng.permutation(class_positions)
    return positions_by_label


def per_class_min_count(class_size, labeled_fraction, labeled_per_class):
    if labeled_per_class is None:
        return max(1, int(round(class_size * labeled_fraction)))
    return min(int(labeled_per_class), int(class_size))


def select_per_class_min_labeled_positions(labels, labeled_fraction, labeled_per_class, rng):
    """Label a fraction or fixed count within every represented class."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    labeled_positions = []
    unlabeled_by_label = {}

    for label, class_positions in positions_by_label.items():
        # round approximates the requested per-class fraction; max(1, ...)
        # guarantees every class remains represented in labeled training.  With
        # a fixed k, classes smaller than k contribute all available samples.
        num_labeled = per_class_min_count(
            class_size=len(class_positions),
            labeled_fraction=labeled_fraction,
            labeled_per_class=labeled_per_class,
        )

        # Because class_positions was shuffled, its prefix is a random labeled
        # subset and the suffix is the same class's unlabeled pool.
        labeled_positions.extend(class_positions[:num_labeled])
        unlabeled_by_label[int(label)] = class_positions[num_labeled:]

    return labeled_positions, unlabeled_by_label


def select_per_class_imbalanced_labeled_positions(labels, labeled_fraction, labeled_per_class, rng):
    """Label every class with bounded random skew around per_class_min counts."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    label_order = np.asarray(list(positions_by_label), dtype=np.int64)
    baseline_counts = np.asarray(
        [
            per_class_min_count(
                class_size=len(positions_by_label[int(label)]),
                labeled_fraction=labeled_fraction,
                labeled_per_class=labeled_per_class,
            )
            for label in label_order
        ],
        dtype=np.int64,
    )
    class_sizes = np.asarray(
        [len(positions_by_label[int(label)]) for label in label_order],
        dtype=np.int64,
    )
    total_labeled = int(baseline_counts.sum())
    lower_counts = np.maximum(1, np.floor(0.5 * baseline_counts).astype(np.int64))
    upper_counts = np.minimum(class_sizes, np.ceil(2.0 * baseline_counts).astype(np.int64))
    selected_counts = lower_counts.copy()
    capacities = upper_counts - selected_counts
    remaining = int(total_labeled - selected_counts.sum())
    random_weights = baseline_counts.astype(np.float64) * rng.lognormal(
        mean=0.0,
        sigma=0.6,
        size=len(label_order),
    )

    while remaining > 0 and np.any(capacities > 0):
        active_indices = np.flatnonzero(capacities > 0)
        active_weights = random_weights[active_indices]
        if not np.isfinite(active_weights).all() or active_weights.sum() <= 0:
            active_weights = np.ones(len(active_indices), dtype=np.float64)
        probabilities = active_weights / active_weights.sum()
        selected_index = int(rng.choice(active_indices, p=probabilities))
        selected_counts[selected_index] += 1
        capacities[selected_index] -= 1
        remaining -= 1

    labeled_positions = []
    unlabeled_by_label = {}
    counts_by_label = {
        int(label): int(count)
        for label, count in zip(label_order, selected_counts)
    }
    for label, class_positions in positions_by_label.items():
        label = int(label)
        num_labeled = counts_by_label[label]
        labeled_positions.extend(class_positions[:num_labeled])
        unlabeled_by_label[label] = class_positions[num_labeled:]

    return labeled_positions, unlabeled_by_label


def select_global_budget_labeled_positions(labels, labeled_fraction, rng):
    """Spend one dataset-wide label budget in a class-balanced round robin."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # Convert the fraction into one global integer budget, guaranteeing at least
    # one labeled sample and never exceeding the dataset.
    target_labeled = max(1, int(np.floor(len(labels) * labeled_fraction)))
    target_labeled = min(target_labeled, len(labels))
    selected_counts = {int(label): 0 for label in unique_labels}
    labeled_positions = []

    # Selecting one item per shuffled class per pass spreads a small global
    # budget across classes before assigning second examples to any class.
    while len(labeled_positions) < target_labeled:
        made_progress = False
        for label in rng.permutation(unique_labels):
            label = int(label)
            selected_count = selected_counts[label]
            class_positions = positions_by_label[label]
            if selected_count >= len(class_positions):
                # Small/exhausted classes stop contributing while larger classes
                # can continue filling the remaining global budget.
                continue

            # Take the next unused position from this class's shuffled list.
            labeled_positions.append(class_positions[selected_count])
            selected_counts[label] = selected_count + 1
            made_progress = True
            if len(labeled_positions) == target_labeled:
                break

        if not made_progress:
            break

    # Everything after each class's consumed prefix remains an unlabeled
    # candidate for pseudo-labeling.
    unlabeled_by_label = {
        int(label): class_positions[selected_counts[int(label)] :]
        for label, class_positions in positions_by_label.items()
    }
    return labeled_positions, unlabeled_by_label


def select_class_subset_labeled_positions(labels, class_fraction, rng):
    """Fully label a random fraction of classes and leave the rest unlabeled."""

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # Convert class_fraction to a count, retaining at least one selected class.
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            # Selected classes are completely labeled in this mode.
            labeled_positions.extend(class_positions)
            unlabeled_by_label[int(label)] = np.array([], dtype=np.int64)
        else:
            # Every sample from an unselected class becomes an unlabeled
            # candidate.
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def select_class_subset_k_shot_labeled_positions(labels, class_fraction, labeled_per_class, rng):
    """Select a class subset, then label at most k examples in each class."""

    if labeled_per_class is None:
        raise ValueError("class_subset_k_shot requires labeled_per_class to set k-shot")

    positions_by_label = make_permuted_positions_by_label(labels, rng)
    unique_labels = np.asarray(list(positions_by_label), dtype=np.int64)
    # First choose how many classes are visible to labeled training.
    num_selected_classes = max(1, int(np.floor(len(unique_labels) * class_fraction)))
    num_selected_classes = min(num_selected_classes, len(unique_labels))
    selected_labels = set(
        int(label) for label in rng.choice(unique_labels, size=num_selected_classes, replace=False)
    )

    labeled_positions = []
    unlabeled_by_label = {}
    for label, class_positions in positions_by_label.items():
        if int(label) in selected_labels:
            # Then spend the k-shot budget only inside selected classes.
            num_labeled = min(int(labeled_per_class), len(class_positions))
            labeled_positions.extend(class_positions[:num_labeled])
            unlabeled_by_label[int(label)] = class_positions[num_labeled:]
        else:
            # Classes outside the selected subset have no ground-truth examples
            # exposed to the training method.
            unlabeled_by_label[int(label)] = class_positions

    return labeled_positions, unlabeled_by_label


def concatenate_position_groups(groups):
    # Discard empty per-class arrays so np.concatenate always receives at least
    # one nonempty input when there are unlabeled candidates.
    arrays = [np.asarray(group, dtype=np.int64) for group in groups if len(group) > 0]
    if not arrays:
        return np.array([], dtype=np.int64)
    return np.concatenate(arrays).astype(np.int64, copy=False)
