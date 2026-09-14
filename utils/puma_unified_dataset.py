"""PUMA's unified metric-learning (UML) benchmark as a single dataset.

PUMA ("Learning Unified Distance Metric Across Diverse Data Distributions with
Parameter-Efficient Transfer Learning", WACV 2025, https://arxiv.org/abs/2309.08944)
trains one embedding model on the union of eight retrieval datasets instead of
one model per dataset. Four are the standard deep-metric-learning benchmarks --
CUB, Cars-196, Stanford Online Products and In-Shop -- and four are further
fine-grained sets: NABirds, Stanford Dogs, Oxford Flowers-102 and FGVC-Aircraft.

This module assembles that union. It owns no images and no split rules of its
own: every member is the dataset class this project already ships, asked for
the split PUMA asks it for. The only thing added here is the composition --
which members, in which order, and how their label spaces are kept apart.

Members and their data roots
----------------------------
``root`` for the union is ``data/PUMA``, which holds nothing. Each member is
read from its usual project location, ``data/<member name>``, found as a
sibling of the union root; a self-contained ``data/PUMA/<member name>`` layout
is used instead when it exists.

Label spaces
------------
PUMA offsets each dataset's labels by the number of distinct classes it has
already added, so a class id means one thing across the union. The same
arithmetic cannot run here, because this project builds the training and test
halves in two independent constructions and an offset derived from the classes
present in one half would not match the other, putting a training class and a
test class on the same label. Each member is therefore given a fixed block of
label ids, ``LABEL_SPACE_SIZE`` wide, whose position depends only on the member
list. The partition of images into classes is identical to PUMA's; only the
integers naming those classes differ, and every construction of the same member
list agrees on them.

Evaluation
----------
PUMA scores the benchmark two ways. *Dataset-specific* accuracy evaluates each
member separately and reports the per-dataset numbers of its Table 2 plus their
harmonic mean; :attr:`member_slices` gives the row ranges that partition needs.
*Unified* accuracy pools every member's test images into one gallery, which for
the seven same-source members means their evaluation images are both queries
and references while In-Shop keeps its query-to-gallery partition. That mixed
form has no equivalent in this project's evaluator, which takes either one
same-source set or two disjoint ones, so :attr:`query_indices` is deliberately
left unset on the pooled test split and the pooled split is scored same-source.
:attr:`unified_query_indices`, :attr:`unified_gallery_indices` and
:attr:`sample_ids` describe PUMA's pooled protocol exactly, for an evaluator
that can exclude self-matches by sample id.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_metric_learning.datasets as pml_datasets
from torch.utils.data import Dataset

from . import local_datasets


PAPER_REFERENCE = "https://arxiv.org/abs/2309.08944"


IMPLEMENTATION_REFERENCE = (
    "https://github.com/sung-yeon-kim/PUMA-WACV25/blob/main/dataset.py"
)


@dataclass(frozen=True)
class PUMAMember:
    """One dataset inside the union, and how PUMA refers to it.

    ``label_space_size`` is the width of the label block reserved for this
    member. It is the size of the member's own label space across both of its
    splits, not the class count of either half, so the block a member gets is
    the same whichever split is being built.
    """

    name: str
    puma_name: str
    dataset_id: int
    label_space_size: int
    dataset_kwargs: tuple = ()

    def build_kwargs(self):
        return dict(self.dataset_kwargs)


# PUMA's ``ds_ID`` numbering, from the ``ds_ID_set`` its evaluation uses. The
# ids are kept even when only some members are selected, so a per-sample id
# means the same thing in the four-dataset and eight-dataset benchmarks.
PUMA_MEMBERS = (
    PUMAMember("CUB", "CUB", 0, 201),
    PUMAMember("Cars196", "Cars", 1, 197),
    PUMAMember("StanfordOnlineProducts", "SOP", 2, 22_635),
    PUMAMember("DeepFashionInShop", "Inshop", 3, 7_983),
    # PUMA cuts NABirds at the midpoint of its sorted class ids and Stanford
    # Dogs at the midpoint of its breed directories. Neither is the split those
    # dataset classes use by default, so both are named explicitly here.
    PUMAMember(
        "NABirds",
        "NAbird",
        4,
        555,
        (("class_split", local_datasets.NABirds.CLASS_SPLIT_CLASS_ID_MIDPOINT),),
    ),
    PUMAMember(
        "StanfordDogs",
        "Dogs",
        5,
        120,
        (("class_split", local_datasets.StanfordDogs.CLASS_SPLIT_CLASS_ID_MIDPOINT),),
    ),
    PUMAMember("Flowers102", "Flowers", 6, 102),
    PUMAMember("FGVCAircraft", "Aircraft", 7, 100),
)


PUMA_MEMBERS_BY_NAME = {member.name: member for member in PUMA_MEMBERS}


PUMA_MEMBERS_BY_PUMA_NAME = {member.puma_name: member for member in PUMA_MEMBERS}


# PUMA's two named collections: ``All`` is the eight-dataset UML benchmark and
# ``Standard`` is the four classic retrieval benchmarks on their own.
PUMA_ALL_MEMBER_NAMES = tuple(member.name for member in PUMA_MEMBERS)


PUMA_STANDARD_MEMBER_NAMES = (
    "CUB",
    "Cars196",
    "StanfordOnlineProducts",
    "DeepFashionInShop",
)


# Table 1 of the paper, as ``(train images, train classes, test images, test
# classes)``. These are reported, never enforced: they are recorded next to the
# counts actually loaded so a mismatch is visible in run metadata. The In-Shop
# test row is the one that does not reconcile -- the official partition file
# holds 14,218 query and 12,612 gallery images, 26,830 together, against the
# 28.7K the table prints.
PUMA_PAPER_SPLIT_COUNTS = {
    "CUB": (5_800, 100, 5_900, 100),
    "Cars196": (8_000, 98, 8_100, 98),
    "StanfordOnlineProducts": (59_500, 11_300, 60_500, 11_300),
    "DeepFashionInShop": (25_800, 3_900, 28_700, 3_900),
    "NABirds": (22_900, 278, 25_600, 277),
    "StanfordDogs": (10_600, 60, 9_900, 60),
    "Flowers102": (3_500, 51, 4_700, 51),
    "FGVCAircraft": (5_000, 50, 5_000, 50),
}


def member_dataset_class(member_name):
    """Return the dataset class the union builds ``member_name`` from.

    The mapping is written out rather than routed through
    :func:`utils.dataset_protocols.get_dataset_class` so that this module has
    no import cycle with the protocol layer, and so the benchmark pins the
    exact eight classes it means.
    """

    if member_name == "CUB":
        return local_datasets.CUB
    if member_name == "Cars196":
        return pml_datasets.Cars196
    if member_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts
    if member_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop
    if member_name == "NABirds":
        return local_datasets.NABirds
    if member_name == "StanfordDogs":
        return local_datasets.StanfordDogs
    if member_name == "Flowers102":
        return local_datasets.Flowers102
    if member_name == "FGVCAircraft":
        return local_datasets.FGVCAircraft
    raise ValueError(f"{member_name!r} is not a PUMA benchmark member")


def normalize_member_names(member_names):
    """Accept either this project's dataset names or PUMA's own names."""

    normalized = []
    for member_name in member_names:
        member = PUMA_MEMBERS_BY_NAME.get(member_name)
        if member is None:
            member = PUMA_MEMBERS_BY_PUMA_NAME.get(member_name)
        if member is None:
            raise ValueError(
                f"{member_name!r} is not a PUMA benchmark member; expected one of "
                f"{PUMA_ALL_MEMBER_NAMES} or {tuple(PUMA_MEMBERS_BY_PUMA_NAME)}"
            )
        if member.name in normalized:
            raise ValueError(f"PUMA member {member.name!r} was listed twice")
        normalized.append(member.name)
    if not normalized:
        raise ValueError("A PUMA benchmark needs at least one member dataset")
    return tuple(normalized)


def ordered_members(member_names):
    """Order members the way PUMA's loader appends them.

    ``All_dataset`` walks its dataset list and then appends In-Shop after the
    loop, so In-Shop is always last. The rest keep ``ds_ID`` order.
    """

    members = [PUMA_MEMBERS_BY_NAME[name] for name in normalize_member_names(member_names)]
    return tuple(
        sorted(
            members,
            key=lambda member: (member.name == "DeepFashionInShop", member.dataset_id),
        )
    )


def member_root(root, member_name):
    """Locate one member's data directory relative to the union's root.

    The union's own root holds no images. ``data/PUMA/CUB`` is used when it
    exists, so a self-contained copy of the benchmark works; otherwise the
    member is read from ``data/CUB``, its usual place in this project.
    """

    root = Path(root)
    nested = root / member_name
    if nested.is_dir():
        return nested
    return root.parent / member_name


class PUMAUnified(Dataset):
    """The eight datasets of PUMA's UML benchmark presented as one dataset.

    ``split="train"`` is the union of every member's training half and
    ``split="test"`` the union of their held-out halves; the two are
    class-disjoint because each member's own halves are and because members
    occupy separate label blocks. ``split="query"`` and ``split="gallery"``
    build PUMA's pooled retrieval sides, which differ only in what In-Shop
    contributes.
    """

    DATASET_PAGE = PAPER_REFERENCE
    PROTOCOL_REFERENCE = PAPER_REFERENCE
    IMPLEMENTATION_REFERENCE = IMPLEMENTATION_REFERENCE
    AVAILABLE_SPLITS = ("train", "test", "query", "gallery", "train+test")
    DEFAULT_MEMBER_NAMES = PUMA_ALL_MEMBER_NAMES
    CLASS_SPLIT_VERSION = "puma_unified_member_label_blocks_v1"
    # Which member split each union split asks for. In-Shop is the only member
    # that distinguishes query from gallery; for everyone else PUMA uses the
    # same evaluation classes for 'eval', 'query' and 'gallery' alike.
    MEMBER_SPLITS = {
        "train": "train",
        "test": "test",
        "query": "test",
        "gallery": "test",
        "train+test": "train+test",
    }
    QUERY_GALLERY_MEMBER_SPLITS = {
        "test": "test",
        "query": "query",
        "gallery": "gallery",
        "train": "train",
        "train+test": "train+test",
    }

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        member_names=None,
        dataset_class_resolver=None,
    ):
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        if dataset_class_resolver is None:
            dataset_class_resolver = member_dataset_class
        member_names = self.DEFAULT_MEMBER_NAMES if member_names is None else member_names
        self.members = ordered_members(member_names)
        self.member_names = tuple(member.name for member in self.members)

        self.label_offsets = self._build_label_offsets(self.members)
        self.datasets = []
        self.member_slices = {}
        self.member_query_gallery = {}
        member_counts = {}

        labels = []
        dataset_ids = []
        query_positions = []
        gallery_positions = []
        start = 0
        for member in self.members:
            member_dataset, member_split = self._build_member(
                member,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
                dataset_class_resolver=dataset_class_resolver,
            )
            member_labels = self._validated_member_labels(member, member_dataset)
            offset = self.label_offsets[member.name]

            self.datasets.append(member_dataset)
            stop = start + len(member_labels)
            self.member_slices[member.name] = (start, stop)
            member_counts[member.name] = {
                "sample_count": int(len(member_labels)),
                "class_count": int(len(set(member_labels))),
                "label_offset": int(offset),
                "dataset_id": int(member.dataset_id),
                "member_split": member_split,
                "dataset_root": str(self._member_root(member)),
            }
            labels.extend(int(label) + offset for label in member_labels)
            dataset_ids.extend([member.dataset_id] * len(member_labels))

            member_query, member_gallery = self._member_query_gallery(
                member_dataset, len(member_labels)
            )
            self.member_query_gallery[member.name] = (member_query, member_gallery)
            query_positions.extend(start + index for index in member_query)
            gallery_positions.extend(start + index for index in member_gallery)
            start = stop

        if not labels:
            raise ValueError(f"PUMA unified split {split!r} is empty")

        self.labels = labels
        self.orig_labels = list(labels)
        self.dataset_ids = np.asarray(dataset_ids, dtype=np.int64)
        self.lengths = np.asarray([len(dataset) for dataset in self.datasets], dtype=np.int64)
        self.cumulative_sizes = np.cumsum(self.lengths)
        self.classes = self._build_classes()
        # PUMA's ``self.I``: a per-sample id that is unique across the union and
        # differs between a query row and a gallery row even when they show the
        # same item, which is what lets its pooled metric drop self-matches
        # without dropping true In-Shop matches.
        self.sample_ids = list(range(len(labels)))
        self.unified_query_indices = query_positions
        self.unified_gallery_indices = gallery_positions
        if self._pooled_split_is_query_gallery():
            # Every member partitions its rows into queries and gallery, so the
            # two sides are disjoint and the split is a plain query-to-gallery
            # retrieval set that this project's evaluator handles directly.
            self.query_indices = query_positions
            self.gallery_indices = gallery_positions

        self.class_disjoint_split = True
        self.class_split_info = self._build_class_split_info(member_counts)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_offsets(members):
        """Give each member a label block wide enough for its whole label space."""

        offsets = {}
        offset = 0
        for member in members:
            offsets[member.name] = offset
            offset += member.label_space_size
        return offsets

    def _member_root(self, member):
        return member_root(self.root, member.name)

    def _member_split(self, dataset_class, split):
        """Name the split this member is asked for, PUMA's In-Shop rule included.

        In-Shop is the only member that separates queries from gallery, so it
        is the only one whose split name changes with the side being built.
        """

        if self._member_has_query_gallery_splits(dataset_class):
            return self.QUERY_GALLERY_MEMBER_SPLITS[split]
        return self.MEMBER_SPLITS[split]

    @staticmethod
    def _member_has_query_gallery_splits(dataset_class):
        """Does this member separate queries from gallery, as In-Shop does?

        ``AVAILABLE_SPLITS`` is this project's convention. The
        pytorch-metric-learning members announce their splits through an
        instance method instead, which cannot be called on the class; none of
        them offers a retrieval partition, so failing to reach it is the right
        answer rather than an error.
        """

        available = getattr(dataset_class, "AVAILABLE_SPLITS", None)
        if available is None:
            get_available_splits = getattr(dataset_class, "get_available_splits", None)
            try:
                available = get_available_splits() if callable(get_available_splits) else ()
            except TypeError:
                available = ()
        return "query" in available and "gallery" in available

    def _build_member(
        self,
        member,
        split,
        transform,
        target_transform,
        download,
        dataset_class_resolver,
    ):
        """Construct one member on the split the union needs from it."""

        dataset_class = dataset_class_resolver(member.name)
        member_split = self._member_split(dataset_class, split)
        member_dataset = dataset_class(
            root=str(self._member_root(member)),
            split=member_split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            **member.build_kwargs(),
        )
        return member_dataset, member_split

    def _validated_member_labels(self, member, member_dataset):
        """Read a member's labels and check they fit the block reserved for it.

        A member whose labels outgrew ``label_space_size`` would collide with
        the next member's block and silently merge two classes into one, so the
        width is checked against the labels actually loaded rather than
        assumed.
        """

        labels = getattr(member_dataset, "labels", None)
        if labels is None:
            labels = getattr(member_dataset, "targets", None)
        if labels is None:
            raise ValueError(
                f"PUMA member {member.name!r} exposes neither labels nor targets"
            )
        labels = [int(label) for label in labels]
        if not labels:
            raise ValueError(f"PUMA member {member.name!r} contributed no samples")
        if len(labels) != len(member_dataset):
            raise ValueError(
                f"PUMA member {member.name!r} reports {len(member_dataset)} samples "
                f"but {len(labels)} labels"
            )
        smallest = min(labels)
        largest = max(labels)
        if smallest < 0 or largest >= member.label_space_size:
            raise ValueError(
                f"PUMA member {member.name!r} produced labels in "
                f"[{smallest}, {largest}], which does not fit the "
                f"{member.label_space_size}-wide label block reserved for it"
            )
        return labels

    @staticmethod
    def _member_query_gallery(member_dataset, sample_count):
        """Return one member's query and gallery rows, as local indices.

        A member without a retrieval partition is every row on both sides,
        which is what PUMA does: its ``query`` and ``gallery`` modes select the
        same evaluation classes for all members but In-Shop.
        """

        query_indices = getattr(member_dataset, "query_indices", None)
        gallery_indices = getattr(member_dataset, "gallery_indices", None)
        if query_indices is None or gallery_indices is None:
            everything = list(range(sample_count))
            return everything, everything
        return [int(index) for index in query_indices], [
            int(index) for index in gallery_indices
        ]

    def _pooled_split_is_query_gallery(self):
        """Is every member contributing disjoint query and gallery rows?

        Only then do the pooled sides form a genuine query-to-gallery problem.
        With even one same-source member the two sides share rows, and handing
        them to an evaluator that assumes disjoint sides would let those rows
        retrieve themselves.
        """

        return all(
            set(query).isdisjoint(gallery)
            for query, gallery in self.member_query_gallery.values()
        )

    def _build_classes(self):
        """Name every label in the union as ``<member>/<local class>``."""

        total_width = sum(member.label_space_size for member in self.members)
        classes = [""] * total_width
        for member, member_dataset in zip(self.members, self.datasets):
            offset = self.label_offsets[member.name]
            member_classes = getattr(member_dataset, "classes", None) or []
            for local_label in set(
                int(label) for label in getattr(member_dataset, "labels", [])
            ):
                if local_label < len(member_classes) and member_classes[local_label]:
                    local_name = member_classes[local_label]
                else:
                    local_name = str(local_label)
                classes[offset + local_label] = f"{member.name}/{local_name}"
        return classes

    def _build_class_split_info(self, member_counts):
        paper_counts = {
            member.name: PUMA_PAPER_SPLIT_COUNTS[member.name]
            for member in self.members
            if member.name in PUMA_PAPER_SPLIT_COUNTS
        }
        return {
            "dataset_root": str(self.root),
            "split": self.split,
            "source": "puma_unified_member_union",
            "protocol_reference": PAPER_REFERENCE,
            "implementation_reference": IMPLEMENTATION_REFERENCE,
            "class_disjoint_test": True,
            "official_image_level_split_used": False,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "class_split_basis": "per_member_label_blocks",
            "member_datasets": list(self.member_names),
            "member_dataset_ids": {
                member.name: int(member.dataset_id) for member in self.members
            },
            "member_label_offsets": dict(self.label_offsets),
            "member_slices": {
                name: [int(start), int(stop)]
                for name, (start, stop) in self.member_slices.items()
            },
            "member_counts": member_counts,
            "paper_table_1_counts": paper_counts,
            "sample_count": int(len(self.labels)),
            "class_count": int(len(set(self.labels))),
            "pooled_retrieval_mode": (
                "query_gallery"
                if getattr(self, "query_indices", None) is not None
                else "same_source"
            ),
            "unified_query_count": int(len(self.unified_query_indices)),
            "unified_gallery_count": int(len(self.unified_gallery_indices)),
        }

    # ------------------------------------------------------------------
    # readiness
    # ------------------------------------------------------------------

    @classmethod
    def member_roots(cls, root, member_names=None):
        """Map each member to the directory the union will read it from."""

        member_names = cls.DEFAULT_MEMBER_NAMES if member_names is None else member_names
        return {
            member.name: member_root(root, member.name)
            for member in ordered_members(member_names)
        }

    # ------------------------------------------------------------------
    # per-member views
    # ------------------------------------------------------------------

    def member_indices(self, member_name):
        """Row indices belonging to one member, for per-dataset evaluation."""

        member = PUMA_MEMBERS_BY_NAME.get(member_name)
        if member is None:
            member = PUMA_MEMBERS_BY_PUMA_NAME.get(member_name)
        if member is None or member.name not in self.member_slices:
            raise ValueError(
                f"{member_name!r} is not a member of this benchmark; it holds "
                f"{self.member_names}"
            )
        start, stop = self.member_slices[member.name]
        return list(range(start, stop))

    def __len__(self):
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        # cumulative_sizes turns a union index into the member holding it and
        # the index that member expects, without copying any sample.
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_size = 0 if dataset_index == 0 else int(self.cumulative_sizes[dataset_index - 1])
        image, _ = self.datasets[dataset_index][index - previous_size]
        # The member returned its own label; the union's is the offset one, and
        # target_transform has already been applied to the member's label, so
        # it is applied here to the offset label instead.
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class PUMAStandard(PUMAUnified):
    """PUMA's ``Standard`` collection: the four classic retrieval benchmarks.

    CUB, Cars-196, SOP and In-Shop, unified the same way as the full
    benchmark. It is the ablation setting PUMA reports alongside the
    eight-dataset results.
    """

    DEFAULT_MEMBER_NAMES = PUMA_STANDARD_MEMBER_NAMES
