import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
from pathlib import Path

OUTPUT_DIR = Path("sop_cv_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EBAY_TRAIN = "Stanford_Online_Products/Ebay_train.txt"
# Sometimes it is under:
# EBAY_TRAIN = "Stanford_Online_Products/Info_Files/Ebay_train.txt"

K = 4
SEED = 2


# ---------------------------------------------------------------------
# Load exact SOP training class distribution
# ---------------------------------------------------------------------

df = pd.read_csv(EBAY_TRAIN, sep=r"\s+")

labels_real = df["class_id"].to_numpy()

classes, counts = np.unique(labels_real, return_counts=True)

print("REAL SOP TRAIN")
print(f"samples : {len(labels_real)}")
print(f"classes : {len(classes)}")
print(f"min     : {counts.min()}")
print(f"max     : {counts.max()}")
print(f"mean    : {counts.mean():.4f}")
print()


# ---------------------------------------------------------------------
# Emulate SOP without keeping the actual IDs / image paths.
#
# Crucially, this preserves the EXACT empirical class-size distribution.
# ---------------------------------------------------------------------

labels = np.repeat(
    np.arange(len(counts)),
    counts
)

X = np.zeros((len(labels), 1))


# ---------------------------------------------------------------------
# Powerful Benchmarker ClassDisjointSplitManager CV behavior
#
# Their implementation:
#
#   1. sorted(unique(labels))
#   2. KFold(...).split(class_list)
#   3. map classes -> samples using np.isin
#
# This implements the CV portion of that behavior.
# ---------------------------------------------------------------------

class PowerfulBenchmarkerClassDisjointKFold:

    def __init__(
        self,
        n_splits=4,
        shuffle=False,
        random_state=None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):

        y = np.asarray(y)

        # Matches:
        # sorted(list(get_label_set(dataset)))
        class_list = np.array(sorted(np.unique(y)))

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=(
                self.random_state if self.shuffle else None
            ),
        )

        for train_class_idx, val_class_idx in kfold.split(class_list):

            train_classes = class_list[train_class_idx]
            val_classes = class_list[val_class_idx]

            # Matches ClassDisjointSplitManager.convert_to_subset_idx:
            train_idx = np.where(np.isin(y, train_classes))[0]
            val_idx = np.where(np.isin(y, val_classes))[0]

            yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


# ---------------------------------------------------------------------
# Splitters to compare
# ---------------------------------------------------------------------

splitters = {
    "GroupKFold — greedy sample balance":
        GroupKFold(
            n_splits=K,
            shuffle=False,
        ),

    "GroupKFold — shuffled groups":
        GroupKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        ),

    "Powerful Benchmarker — class KFold":
        PowerfulBenchmarkerClassDisjointKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        ),
    "StratifiedKFold — NOT class-disjoint":
        StratifiedKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        ),

    # Keeps groups intact while attempting stratification.
    #
    # Here groups == labels, so every group contains exactly one
    # class and every class occurs in exactly one group.
    #
    # Therefore "stratification" cannot really do what it normally
    # means: a class cannot be represented in multiple validation
    # folds without breaking class-disjointness.
    "StratifiedGroupKFold — class groups":
        StratifiedGroupKFold(
            n_splits=K,
            shuffle=True,
            random_state=SEED,
        ),
}


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def get_splits(cv, labels):

    X_dummy = np.zeros((len(labels), 1))

    if isinstance(
        cv,
        (GroupKFold, StratifiedGroupKFold),
    ):
        return list(
            cv.split(
                X=X_dummy,
                y=labels,
                groups=labels,
            )
        )

    return list(
        cv.split(
            X=X_dummy,
            y=labels,
        )
    )


def print_stats(name, cv, labels):

    print("=" * 75)
    print(name)
    print("=" * 75)

    splits = get_splits(cv, labels)

    val_sizes = []
    val_class_counts = []
    overlaps = []

    for fold, (train_idx, val_idx) in enumerate(splits):

        train_classes = np.unique(labels[train_idx])
        val_classes = np.unique(labels[val_idx])

        overlap = np.intersect1d(
            train_classes,
            val_classes,
        )

        val_sizes.append(len(val_idx))
        val_class_counts.append(len(val_classes))
        overlaps.append(len(overlap))

        print(
            f"fold {fold}: "
            f"val samples={len(val_idx):5d} | "
            f"val classes={len(val_classes):5d} | "
            f"class overlap={len(overlap):5d}"
        )

    print()

    print(
        "sample range:",
        max(val_sizes) - min(val_sizes),
    )

    print(
        "class-count range:",
        max(val_class_counts) - min(val_class_counts),
    )

    print(
        "class-disjoint:",
        all(x == 0 for x in overlaps),
    )

    print()


for name, cv in splitters.items():
    print_stats(name, cv, labels)


# ---------------------------------------------------------------------
# Plot 1: exact SOP class-size distribution
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

bins = np.arange(
    counts.min() - 0.5,
    counts.max() + 1.5,
    1,
)

ax.hist(counts, bins=bins)

ax.set_xlabel("Images per product class")
ax.set_ylabel("Number of product classes")
ax.set_title(
    "Stanford Online Products training split\n"
    "Exact empirical class-size distribution"
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "sop_class_size_distribution.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ---------------------------------------------------------------------
# sklearn-style CV visualization
#
# We SORT samples by class so each product forms one contiguous block.
# That makes class-disjoint behavior very obvious.
# ---------------------------------------------------------------------

def plot_cv_indices(name, cv, labels, filename):

    order = np.argsort(labels, kind="stable")

    splits = get_splits(cv, labels)

    matrix = np.zeros(
        (len(splits), len(labels)),
        dtype=np.uint8,
    )

    for fold, (_, val_idx) in enumerate(splits):
        is_val = np.zeros(len(labels), dtype=np.uint8)
        is_val[val_idx] = 1
        matrix[fold] = is_val[order]

    fig, ax = plt.subplots(figsize=(14, 3.5))

    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(splits)))
    ax.set_yticklabels(
        [f"Fold {i}" for i in range(len(splits))]
    )

    ax.set_xlabel("Samples, ordered by product class")
    ax.set_ylabel("CV fold")
    ax.set_title(name)

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


plot_cv_indices(
    "GroupKFold — greedy sample balance",
    splitters["GroupKFold — greedy sample balance"],
    labels,
    "group_kfold_greedy_indices.png",
)

plot_cv_indices(
    "GroupKFold — shuffled groups",
    splitters["GroupKFold — shuffled groups"],
    labels,
    "group_kfold_shuffled_indices.png",
)

plot_cv_indices(
    "Powerful Benchmarker — class KFold",
    splitters["Powerful Benchmarker — class KFold"],
    labels,
    "powerful_benchmarker_indices.png",
)


# ---------------------------------------------------------------------
# Plot fold SAMPLE counts
# ---------------------------------------------------------------------

safe_names = {
    "GroupKFold — greedy sample balance":
        "group_kfold_greedy",

    "GroupKFold — shuffled groups":
        "group_kfold_shuffled",

    "Powerful Benchmarker — class KFold":
        "powerful_benchmarker",

    "StratifiedKFold — NOT class-disjoint":
        "stratified_kfold",

    "StratifiedGroupKFold — class groups":
        "stratified_group_kfold",
}

for name, cv in splitters.items():

    splits = get_splits(cv, labels)

    sample_counts = [
        len(val_idx)
        for _, val_idx in splits
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(np.arange(K), sample_counts)

    ax.set_xticks(np.arange(K))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation samples")
    ax.set_title(name + "\nSamples per fold")

    for i, n in enumerate(sample_counts):
        ax.text(i, n, str(n), ha="center", va="bottom")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{safe_names[name]}_samples_per_fold.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


for name, cv in splitters.items():

    splits = get_splits(cv, labels)

    class_counts = [
        len(np.unique(labels[val_idx]))
        for _, val_idx in splits
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(np.arange(K), class_counts)

    ax.set_xticks(np.arange(K))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation classes")
    ax.set_title(name + "\nClasses per fold")

    for i, n in enumerate(class_counts):
        ax.text(i, n, str(n), ha="center", va="bottom")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{safe_names[name]}_classes_per_fold.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

print(f"Plots saved to: {OUTPUT_DIR.resolve()}")

for name, cv in splitters.items():

    splits = get_splits(cv, labels)

    overlap_counts = []

    for train_idx, val_idx in splits:

        train_classes = np.unique(labels[train_idx])
        val_classes = np.unique(labels[val_idx])

        overlap = np.intersect1d(
            train_classes,
            val_classes,
        )

        overlap_counts.append(len(overlap))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        np.arange(K),
        overlap_counts,
    )

    ax.set_xticks(np.arange(K))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Classes in both train and validation")
    ax.set_title(
        name + "\nTrain/validation class overlap"
    )

    for i, n in enumerate(overlap_counts):
        ax.text(
            i,
            n,
            str(n),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"{safe_names[name]}_class_overlap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)