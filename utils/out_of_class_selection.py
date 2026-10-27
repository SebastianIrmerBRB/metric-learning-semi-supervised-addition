"""Deterministic, class-balanced subsetting of an out-of-class unlabeled pool.

Class-mismatch experiments vary how much out-of-class data reaches the
unlabeled pool. Both Semi-Aves and Semi-iNat draw that subset the same way, so
the rule lives here instead of being reimplemented per dataset.
"""

import hashlib
from pathlib import Path

from .image_dataset_base import ImageListDataset


def select_out_of_class_records(
    records,
    fraction,
    seed=0,
    *,
    selection_version,
    dataset_label,
):
    """Keep a class-balanced ``fraction`` of an out-of-class record list.

    The kept images are spread evenly over the out-of-class species instead of
    concentrated in whichever ones happen to be large. Selection is a pure
    function of the image path, the class, and ``seed``, so a fraction always
    yields the same images, and a larger fraction reuses a smaller one's images
    up to the per-class rounding.

    ``records`` are ``(relative_path, label, source)`` triples.
    """

    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{dataset_label} out-of-class fraction must be in [0, 1]: {fraction}"
        )
    records = list(records)
    if not records or fraction == 0.0:
        return []
    if fraction == 1.0:
        return records

    seed = int(seed)
    positions_by_class = {}
    for position, (relative_path, label, _) in enumerate(records):
        rank_key = hashlib.sha256(
            f"{selection_version}:{seed}:"
            f"{relative_path.as_posix()}".encode("utf-8")
        ).digest()
        positions_by_class.setdefault(int(label), []).append(
            (rank_key, position)
        )

    # Largest-remainder apportioning keeps every class's share proportional
    # while the kept total still matches the requested fraction exactly.
    target_total = int(round(len(records) * fraction))
    quotas = {}
    remainders = []
    for label, entries in positions_by_class.items():
        exact_quota = len(entries) * fraction
        quotas[label] = int(exact_quota)
        remainders.append((exact_quota - quotas[label], label))
    leftover = target_total - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, label in remainders[:leftover]:
        quotas[label] += 1

    kept_positions = []
    for label, entries in positions_by_class.items():
        entries.sort()
        kept_positions.extend(
            position for _, position in entries[: quotas[label]]
        )
    # Restore source order so manifests stay easy to compare across runs.
    kept_positions.sort()
    return [records[position] for position in kept_positions]


class OutOfClassUnlabeledPool(ImageListDataset):
    """A native out-of-class unlabeled pool with every oracle label hidden.

    ``out_of_class_fraction`` controls the class-mismatch level: it is the share
    of the released U-out pool that joins the unlabeled data, and the remaining
    out-of-class images are dropped rather than kept with a hidden label. 1.0
    reproduces the original protocol and 0.0 leaves no out-of-class data at all.

    Subclasses name the prepared ``DATASET`` they read from and implement
    :meth:`available_records`.
    """

    UNLABELED = True
    SOURCE = "u_train_out"
    DATASET = DATASET_NAME = LABEL = MODE = None

    def __init__(
        self,
        root,
        transform=None,
        out_of_class_fraction=1.0,
        out_of_class_seed=0,
    ):
        self.root = Path(root)
        if not self.DATASET.is_ready(self.root):
            raise ValueError(
                f"Prepared {self.DATASET_NAME} data was not found at {self.root}"
            )
        out_of_class_fraction = float(out_of_class_fraction)
        if not 0.0 <= out_of_class_fraction <= 1.0:
            raise ValueError(
                f"{self.LABEL} out_of_class_fraction must be in [0, 1]: "
                f"{out_of_class_fraction}"
            )
        available_records = self.available_records()
        selection_version = self.DATASET.OUT_OF_CLASS_SELECTION_VERSION
        records = select_out_of_class_records(
            available_records,
            out_of_class_fraction,
            out_of_class_seed,
            selection_version=selection_version,
            dataset_label=self.LABEL,
        )
        self.paths = [
            str(self.root / Path(*relative_path.parts))
            for relative_path, _, _ in records
        ]
        self.labels = [-1] * len(records)
        self.orig_labels = [int(label) for _, label, _ in records]
        self.oracle_labels = list(self.orig_labels)
        self.sample_sources = [self.SOURCE for _ in records]
        self.transform = transform
        self.out_of_class_fraction = out_of_class_fraction
        self.out_of_class_seed = int(out_of_class_seed)
        self.filter_info = {
            "mode": self.MODE,
            "candidate_source": "released_oracle_files_hidden_at_runtime",
            "labels_exposed_to_training": False,
            "kept_images": len(records),
            "kept_out_of_class_classes": len({label for _, label, _ in records}),
            "out_of_class_behavior": (
                "all_original_u_train_out_images_unlabeled"
                if out_of_class_fraction == 1.0
                else "class_balanced_out_of_class_subset_remainder_excluded"
            ),
            "out_of_class_fraction_requested": out_of_class_fraction,
            "out_of_class_fraction_realized": (
                len(records) / len(available_records)
                if available_records
                else 0.0
            ),
            "available_out_of_class_images": len(available_records),
            "excluded_out_of_class_images": (
                len(available_records) - len(records)
            ),
            "out_of_class_selection_seed": int(out_of_class_seed),
            "out_of_class_selection_version": selection_version,
        }

    def available_records(self):
        """Every released ``(relative_path, label, source)`` out-of-class record."""

        raise NotImplementedError
