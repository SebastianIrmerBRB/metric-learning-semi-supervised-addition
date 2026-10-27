"""Dataset discovery and source-protocol construction."""

from pathlib import Path

import pytorch_metric_learning.datasets as datasets

from . import local_datasets
from .puma_unified_dataset import (
    PUMA_ALL_MEMBER_NAMES,
    PUMA_STANDARD_MEMBER_NAMES,
    PUMAStandard,
    PUMAUnified,
    member_root,
)
from .dataset_constants import (
    ANY_DATASET,
    CUB_ALTERNATING_CLASS_SPLIT_VERSION,
    CUB_ALTERNATING_DEVELOPMENT_CLASSES,
    CUB_ALTERNATING_HELD_OUT_TEST_CLASSES,
    CUB_CLASS_IDS,
    CUB_PROTOCOL_REFERENCE,
    CUB_PROTOCOLS,
    NABIRDS_PROTOCOLS,
    NABIRDS_PROTOCOL_CLASS_SPLITS,
    VEHICLEID_PROTOCOLS,
    VEHICLEID_PROTOCOL_TEST_GALLERY_SIZES,
    DATASET_PROTOCOL_INFO,
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOLS,
    DATASET_PROTOCOL_SEMI_INAT_KNOWN_810,
    SEMI_INAT_DEFAULT_OOD_FRACTION,
    QUERY_GALLERY_EVALUATION,
    SEMI_AVES_DEFAULT_OOD_FRACTION,
    NATIVE_UNLABELED_SEMI_AVES,
    NATIVE_UNLABELED_SEMI_INAT,
    SEMI_AVES_KNOWN_PROTOCOLS,
    SEMI_AVES_PROTOCOL_CLASS_SPLITS,
    SEMI_AVES_PROTOCOLS,
    SEMI_INAT_PROTOCOLS,
    STANFORD_DOGS_PROTOCOL_CLASS_SPLITS,
    STANFORD_DOGS_PROTOCOLS,
)
from .dataset_splits import (
    assert_disjoint_dataset_classes,
    subset_dataset_by_classes,
)


def describe_dataset_protocols():
    """Render every registered protocol as one help/reporting line each."""

    lines = []
    for protocol, info in DATASET_PROTOCOL_INFO.items():
        datasets = info["datasets"]
        scope = "all datasets" if datasets == ANY_DATASET else ", ".join(datasets)
        lines.append(f"{protocol} [{scope}]: {info['summary']}")
    return tuple(lines)


def format_dataset_protocol_help():
    """Build the --dataset_protocol help text from the protocol registry."""

    return (
        "dataset split protocol, documented in docs/dataset_protocols.md. "
        + " | ".join(describe_dataset_protocols())
    )


def normalize_dataset_name(dataset_name):
    aliases = {
        "DeepFashionInShopRetrieval": "DeepFashionInShop",
        "InShop": "DeepFashionInShop",
        "InShopRetrieval": "DeepFashionInShop",
        "INaturalist2018": "iNat2018",
        "INaturalist2019": "iNat2019",
        "iNaturalist2019": "iNat2019",
        "iNat19": "iNat2019",
        "MiniINaturalist2021": "iNat2021Mini",
        "Mini-iNaturalist-2021": "iNat2021Mini",
        "mini-iNaturalist-2021": "iNat2021Mini",
        "iNaturalist2021Mini": "iNat2021Mini",
        "iNat2021": "iNat2021Mini",
        "SemiINaturalist2021": "iNat",
        "SemiINat2021": "iNat",
        "Semi-Aves": "SemiAves",
        "Semi_Aves": "SemiAves",
        "SemiAves2020": "SemiAves",
        "Food-101": "Food101",
        "Food_101": "Food101",
        "Flowers": "Flowers102",
        "Flowers-102": "Flowers102",
        "Flowers_102": "Flowers102",
        "OxfordFlowers": "Flowers102",
        "OxfordFlowers102": "Flowers102",
        "Aircraft": "FGVCAircraft",
        "FGVC-Aircraft": "FGVCAircraft",
        "FGVC_Aircraft": "FGVCAircraft",
        "Fungi": "FGVCFungi",
        "FGVCxFungi": "FGVCFungi",
        "FGVC-Fungi": "FGVCFungi",
        "FGVC_Fungi": "FGVCFungi",
        "NaBirds": "NABirds",
        "NA-Birds": "NABirds",
        "NABirds2015": "NABirds",
        "Pitts30k": "Pittsburgh30k",
        "pitts30k": "Pittsburgh30k",
        "Pittsburgh-30k": "Pittsburgh30k",
        "Pittsburgh_30k": "Pittsburgh30k",
        "Pitts250k": "Pittsburgh250k",
        "pitts250k": "Pittsburgh250k",
        "Pittsburgh-250k": "Pittsburgh250k",
        "Pittsburgh_250k": "Pittsburgh250k",
        "VehicleId": "VehicleID",
        "vehicleid": "VehicleID",
        "vehicle_id": "VehicleID",
        "Vehicle-ID": "VehicleID",
        "PKUVehicleID": "VehicleID",
        "PKU-VehicleID": "VehicleID",
        "PKU_VehicleID": "VehicleID",
        "pku-vehicleid": "VehicleID",
        "pku_vehicleid": "VehicleID",
        "PKU VehicleID": "VehicleID",
        "PUMA-WACV25": "PUMA",
        "PUMAAll": "PUMA",
        "PUMAUnified": "PUMA",
        "UML": "PUMA",
        "UnifiedMetricLearning": "PUMA",
        "PUMA-Standard": "PUMAStandard",
        "PUMA_Standard": "PUMAStandard",
        "sped": "SPED",
        "Sped": "SPED",
        "Specific Places Dataset": "SPED",
        "SpecificPlaces": "SPED",
        "SpecificPlacesDataset": "SPED",
        "Specific-PlacEs-Dataset": "SPED",
        "Specific_Places_Dataset": "SPED",
    }
    return aliases.get(dataset_name, dataset_name)


# Dataset name -> the local_datasets class that loads it. Looked up by name at
# call time so a test can patch the class on the module.
LOCAL_DATASET_CLASSES = {
    "StanfordOnlineProducts": "StanfordOnlineProducts",
    "CUB": "CUB",
    "DeepFashionInShop": "DeepFashionInShop",
    "iNat": "SemiINaturalist2021",
    "iNat2018": "INaturalist2018",
    "iNat2019": "INaturalist2019",
    "iNat2021Mini": "MiniINaturalist2021",
    "SemiAves": "SemiAves",
    "StanfordDogs": "StanfordDogs",
    "Food101": "Food101",
    "Flowers102": "Flowers102",
    "FGVCAircraft": "FGVCAircraft",
    "FGVCFungi": "FGVCFungi",
    "NABirds": "NABirds",
    "Pittsburgh30k": "Pittsburgh30k",
    "Pittsburgh250k": "Pittsburgh250k",
    "SPED": "SPED",
    "VehicleID": "VehicleID",
}


def get_dataset_class(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name in LOCAL_DATASET_CLASSES:
        return getattr(local_datasets, LOCAL_DATASET_CLASSES[dataset_name])
    if dataset_name == "PUMA":
        return PUMAUnified
    if dataset_name == "PUMAStandard":
        return PUMAStandard
    return getattr(datasets, dataset_name)


def is_dataset_ready(dataset_name, data_root):
    dataset_name = normalize_dataset_name(dataset_name)
    data_root = Path(data_root)

    if dataset_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop.find_metadata_file(data_root) is not None
    if dataset_name in LOCAL_DATASET_CLASSES:
        return getattr(local_datasets, LOCAL_DATASET_CLASSES[dataset_name]).is_ready(data_root)
    if dataset_name in {"PUMA", "PUMAStandard"}:
        # The union's own root holds nothing; it is ready exactly when every
        # member it unions is ready in its own directory.
        member_names = (
            PUMA_ALL_MEMBER_NAMES
            if dataset_name == "PUMA"
            else PUMA_STANDARD_MEMBER_NAMES
        )
        return all(
            is_dataset_ready(member_name, member_root(data_root, member_name))
            for member_name in member_names
        )

    return data_root.exists()


def load_dataset_protocol_sources(
    dataset_name,
    data_root,
    train_transform,
    test_transform,
    dataset_protocol=DATASET_PROTOCOL_OFFICIAL,
    download=False,
    dataset_class_resolver=None,
):
    """Load the development and test sources a dataset protocol prescribes."""

    validate_dataset_protocol(dataset_name, dataset_protocol)
    dataset_cls = (dataset_class_resolver or get_dataset_class)(dataset_name)

    def load_class_split(names, info_prefix=(), **kwargs):
        """Build both halves of a class-disjoint split the dataset class owns."""

        train_val_dataset = dataset_cls(
            root=str(data_root), split="train", transform=train_transform, download=download, **kwargs
        )
        test_dataset = dataset_cls(
            root=str(data_root), split="test", transform=test_transform, download=False, **kwargs
        )
        assert_disjoint_dataset_classes(train_val_dataset, test_dataset, *names)
        protocol_info = {
            "name": dataset_protocol,
            **dict(info_prefix),
            **getattr(train_val_dataset, "class_split_info", {}),
        }
        return train_val_dataset, test_dataset, protocol_info

    if dataset_protocol in SEMI_AVES_PROTOCOLS or dataset_protocol in SEMI_INAT_PROTOCOLS:
        # Semi-Aves protocols name a label pool and a rule for halving its
        # classes; Semi-iNat ones alternate species by class id inside each
        # kingdom. Both assign complete classes to their two halves, and only
        # the known protocols keep an out-of-class pool that no split ever
        # labels, so only they attach the native unlabeled data.
        if dataset_protocol in SEMI_AVES_PROTOCOLS:
            mode = (
                local_datasets.SemiAves.MODE_KNOWN
                if dataset_protocol in SEMI_AVES_KNOWN_PROTOCOLS
                else local_datasets.SemiAves.MODE_ORACLE
            )
            label = "Semi-Aves"
            kwargs = {"class_split": SEMI_AVES_PROTOCOL_CLASS_SPLITS[dataset_protocol]}
            native_pool = NATIVE_UNLABELED_SEMI_AVES if dataset_protocol in SEMI_AVES_KNOWN_PROTOCOLS else None
        else:
            known = dataset_protocol == DATASET_PROTOCOL_SEMI_INAT_KNOWN_810
            mode = (
                local_datasets.SemiINaturalist2021.MODE_KNOWN
                if known
                else local_datasets.SemiINaturalist2021.MODE_ORACLE
            )
            label = "Semi-iNat"
            kwargs = {}
            native_pool = NATIVE_UNLABELED_SEMI_INAT if known else None
        train_val_dataset, test_dataset, protocol_info = load_class_split(
            (f"{label} {mode} development", f"{label} {mode} test"),
            info_prefix={"dataset_root": str(data_root)},
            mode=mode,
            **kwargs,
        )
        if native_pool is not None:
            protocol_info["auto_native_unlabeled_pool"] = True
            protocol_info["native_unlabeled_dataset"] = native_pool
        return train_val_dataset, test_dataset, protocol_info
    if dataset_protocol in CUB_PROTOCOLS:
        # CUB numbers its 200 classes alphabetically by common name, so the
        # official 1-100/101-200 cut keeps whole genera on one side of the
        # split. This protocol pools both official halves and alternates class
        # ids instead -- development takes 1, 3, 5, ... and the held-out test
        # set takes 2, 4, 6, ... -- so every genus is represented on both sides.
        # It is the same rule the Semi-Aves known protocol uses, and it makes
        # the final test classes visually closer to the development classes
        # than the official contiguous cut does.
        development_classes = CUB_ALTERNATING_DEVELOPMENT_CLASSES
        held_out_test_classes = CUB_ALTERNATING_HELD_OUT_TEST_CLASSES
        # Build two views over the same pooled samples because development data
        # needs augmentation while the held-out test view must be deterministic.
        development_source = dataset_cls(
            root=str(data_root),
            split="train+test",
            transform=train_transform,
            download=download,
        )
        test_source = dataset_cls(
            root=str(data_root),
            split="train+test",
            transform=test_transform,
            download=False,
        )
        pooled_classes = set(int(label) for label in development_source.labels)
        if pooled_classes != set(CUB_CLASS_IDS):
            raise ValueError(
                f"CUB pooled sources cover {len(pooled_classes)} classes; "
                f"expected the {len(CUB_CLASS_IDS)} contiguous ids "
                f"{CUB_CLASS_IDS[0]}..{CUB_CLASS_IDS[-1]}"
            )
        train_val_dataset = subset_dataset_by_classes(development_source, development_classes)
        test_dataset = subset_dataset_by_classes(test_source, held_out_test_classes)
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            "CUB development",
            "CUB test",
        )
        pooled_sample_count = len(development_source)
        return (
            train_val_dataset,
            test_dataset,
            {
                "name": dataset_protocol,
                "source": "pooled_official_train_test_class_disjoint_100_100",
                "dataset_root": str(data_root),
                "protocol_reference": CUB_PROTOCOL_REFERENCE,
                "pooled_sources": ["official_train", "official_test"],
                "pooled_sample_count": pooled_sample_count,
                "class_disjoint_test": True,
                "class_split_version": CUB_ALTERNATING_CLASS_SPLIT_VERSION,
                "split_basis": "alternating_class_ids",
                "development_class_count": len(development_classes),
                "test_class_count": len(held_out_test_classes),
                "development_classes": list(development_classes),
                "held_out_test_classes": list(held_out_test_classes),
                "development_sample_count": len(train_val_dataset),
                "test_sample_count": len(test_dataset),
                "development_sample_fraction": len(train_val_dataset) / pooled_sample_count,
                "test_sample_fraction": len(test_dataset) / pooled_sample_count,
                "official_image_level_split_used": False,
            },
        )
    # NABirds and Stanford Dogs already repartition by class under `official`;
    # their protocols only swap which categories each half gets, a rule the
    # dataset class owns. VehicleID's identity partition is fixed by the
    # released lists; its protocols only choose which held-out list to test on.
    if dataset_protocol in NABIRDS_PROTOCOLS:
        return load_class_split(
            ("NABirds development", "NABirds test"),
            class_split=NABIRDS_PROTOCOL_CLASS_SPLITS[dataset_protocol],
        )
    if dataset_protocol in STANFORD_DOGS_PROTOCOLS:
        return load_class_split(
            ("Stanford Dogs development", "Stanford Dogs test"),
            class_split=STANFORD_DOGS_PROTOCOL_CLASS_SPLITS[dataset_protocol],
        )
    if dataset_protocol in VEHICLEID_PROTOCOLS:
        test_gallery_size = VEHICLEID_PROTOCOL_TEST_GALLERY_SIZES[dataset_protocol]
        return load_class_split(
            ("VehicleID development", f"VehicleID test_{test_gallery_size}"),
            test_gallery_size=test_gallery_size,
        )

    # Preserve the dataset provider's official train/test boundary. A dataset
    # that ships a validation split of its own -- Pittsburgh does -- names the
    # pooled development split it wants here, because this project carves
    # validation out of the development pool rather than reading a provider's;
    # without it that split would go unused.
    development_split = getattr(dataset_cls, "OFFICIAL_DEVELOPMENT_SPLIT", "train")
    train_val_dataset = dataset_cls(
        root=str(data_root),
        split=development_split,
        transform=train_transform,
        download=download,
    )
    test_dataset = dataset_cls(root=str(data_root), split="test", transform=test_transform, download=False)
    protocol_info = {
        "name": DATASET_PROTOCOL_OFFICIAL,
        "source": "official_train_test_splits",
    }
    if getattr(train_val_dataset, "class_disjoint_split", False):
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            "development",
            "test",
        )
        protocol_info.update(getattr(train_val_dataset, "class_split_info", {}))
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
    return train_val_dataset, test_dataset, protocol_info


# Non-official protocol families and the one dataset each belongs to.
PROTOCOL_DATASETS = (
    (SEMI_AVES_PROTOCOLS, "SemiAves", "SemiAves"),
    (SEMI_INAT_PROTOCOLS, "iNat", "SemiINaturalist2021"),
    (CUB_PROTOCOLS, "CUB", "CUB"),
    (NABIRDS_PROTOCOLS, "NABirds", "NABirds"),
    (STANFORD_DOGS_PROTOCOLS, "StanfordDogs", "StanfordDogs"),
    (VEHICLEID_PROTOCOLS, "VehicleID", "VehicleID"),
)


def validate_dataset_protocol(dataset_name, dataset_protocol):
    if dataset_protocol not in DATASET_PROTOCOLS:
        raise ValueError(f"dataset_protocol must be one of {DATASET_PROTOCOLS}: {dataset_protocol}")
    for protocols, required_dataset, display_name in PROTOCOL_DATASETS:
        if dataset_protocol in protocols and normalize_dataset_name(dataset_name) != required_dataset:
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for {display_name}"
            )


def validate_semi_inat_ood_fraction(dataset_protocol, ood_fraction):
    """Validate the Semi-iNat class-mismatch level for the unlabeled pool."""

    if ood_fraction is None:
        return
    ood_fraction = float(ood_fraction)
    if not 0 <= ood_fraction <= 1:
        raise ValueError("semi_inat_ood_fraction must be in [0, 1]")
    if (
        ood_fraction != SEMI_INAT_DEFAULT_OOD_FRACTION
        and dataset_protocol != DATASET_PROTOCOL_SEMI_INAT_KNOWN_810
    ):
        raise ValueError(
            "semi_inat_ood_fraction only applies to "
            f"dataset_protocol={DATASET_PROTOCOL_SEMI_INAT_KNOWN_810!r}, "
            "which owns the original out-of-class unlabeled pool"
        )


def validate_semi_aves_ood_fraction(dataset_protocol, ood_fraction):
    """Validate the Semi-Aves class-mismatch level for the unlabeled pool."""

    if ood_fraction is None:
        return
    ood_fraction = float(ood_fraction)
    if not 0 <= ood_fraction <= 1:
        raise ValueError("semi_aves_ood_fraction must be in [0, 1]")
    if (
        ood_fraction != SEMI_AVES_DEFAULT_OOD_FRACTION
        and dataset_protocol not in SEMI_AVES_KNOWN_PROTOCOLS
    ):
        raise ValueError(
            "semi_aves_ood_fraction only applies to "
            f"dataset_protocol in {SEMI_AVES_KNOWN_PROTOCOLS}, "
            "which own the original out-of-class unlabeled pool"
        )


# Out-of-class unlabeled pool fractions, paired with the protocols that own the
# pool each one resizes and the value that leaves the pool alone.
NATIVE_UNLABELED_POOL_FRACTION_ARGS = (
    (
        "semi_aves_ood_fraction",
        SEMI_AVES_KNOWN_PROTOCOLS,
        SEMI_AVES_DEFAULT_OOD_FRACTION,
    ),
    (
        "semi_inat_ood_fraction",
        (DATASET_PROTOCOL_SEMI_INAT_KNOWN_810,),
        SEMI_INAT_DEFAULT_OOD_FRACTION,
    ),
)


def sanitize_native_unlabeled_pool_fractions(args):
    """Reset out-of-class pool fractions the saved dataset_protocol cannot consume.

    Both defaults were 1.0 before they became 0, so studies from that era saved a
    non-zero fraction into base_args on every dataset, not just the two that own a
    native unlabeled pool. The value is inert for the others -- only a protocol
    with auto_native_unlabeled_pool ever reads it -- but the request validators
    still reject it, which breaks study-directory replays. Returns the names that
    were reset.
    """

    dataset_protocol = getattr(args, "dataset_protocol", None)
    reset_names = []
    for name, owning_protocols, default_fraction in NATIVE_UNLABELED_POOL_FRACTION_ARGS:
        fraction = getattr(args, name, None)
        if fraction is None or float(fraction) == float(default_fraction):
            continue
        if dataset_protocol in owning_protocols:
            continue
        setattr(args, name, default_fraction)
        reset_names.append(name)
    return reset_names
