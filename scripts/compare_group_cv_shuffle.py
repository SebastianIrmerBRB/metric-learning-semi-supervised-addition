"""Compare the grouped CV splitters used for class-disjoint folds.

Class labels are the grouping variable, so every class lands in exactly one
validation fold. The three variants differ in what they balance and whether the
seed changes anything:

    group_kfold  shuffle=False   balances sample counts, ignores the seed
    group_kfold  shuffle=True    balances class counts, seed-dependent
    stratified_group_kfold       balances sample counts, seed-dependent

sklearn implements the two shuffles differently. GroupKFold branches on
``shuffle`` and replaces the size-weighted greedy with ``np.array_split`` over a
permutation, so group sizes stop mattering. StratifiedGroupKFold permutes the
groups before a *stable* sort and always runs the greedy, so the permutation
only survives where the sort key ties -- which means its shuffle is a no-op when
every class has a distinct size. The final block demonstrates that.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

N_SPLITS = 4
SEEDS = (0, 1, 2, 3)


def splitters(seed):
    return {
        "group_kfold (shuffle=False)": GroupKFold(n_splits=N_SPLITS),
        "group_kfold (shuffle=True)": GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
        "stratified_group_kfold": StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
    }


def fold_summary(splitter, labels):
    """Return per-fold validation sizes, class counts, and the class partition."""

    indices = np.arange(len(labels))
    sizes, class_counts, partition = [], [], set()
    for _, val in splitter.split(indices, labels, groups=labels):
        val_classes = np.unique(labels[val])
        sizes.append(len(val))
        class_counts.append(len(val_classes))
        partition.add(frozenset(val_classes.tolist()))
    return sizes, class_counts, frozenset(partition)


def report(title, labels):
    _, counts = np.unique(labels, return_counts=True)
    tied = len(counts) - len(np.unique(counts))
    print(f"\n{title}")
    print(f"  {len(counts)} classes, sizes {counts.min()}-{counts.max()}, {tied} share a size with another class")

    for name in splitters(SEEDS[0]):
        partitions = set()
        for seed in SEEDS:
            sizes, class_counts, partition = fold_summary(splitters(seed)[name], labels)
            partitions.add(partition)
            if seed == SEEDS[0]:
                first = (sizes, class_counts)
        sizes, class_counts = first
        print(
            f"  {name:28s} val sizes {str(sizes):24s} spread {max(sizes) - min(sizes):4d}"
            f" | classes/fold {str(class_counts):16s}"
            f" | {len(partitions)}/{len(SEEDS)} distinct splits across seeds"
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Cars196-like: uneven class sizes with many ties, the common case.
    report("Uneven class sizes, many ties (Cars196-like)", np.repeat(np.arange(40), rng.integers(59, 98, size=40)))

    # Balanced classes: every size ties, so both shuffles vary the split freely.
    report("Balanced classes (CIFAR-like)", np.repeat(np.arange(40), 500))

    # Every class a distinct size: nothing for a stable sort to reorder, so
    # stratified_group_kfold collapses to a single split regardless of seed.
    report("All class sizes distinct (stable sort has no ties)", np.repeat(np.arange(40), np.arange(30, 70)))
