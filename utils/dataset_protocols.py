"""Dataset discovery and source-protocol construction."""

import random
from pathlib import Path

import numpy as np
import pytorch_metric_learning.datasets as datasets
from torch.utils.data import Subset

from . import local_datasets
from .dataset_composition import CombinedDataset
from .puma_unified_dataset import (
    PUMA_ALL_MEMBER_NAMES,
    PUMA_STANDARD_MEMBER_NAMES,
    PUMAStandard,
    PUMAUnified,
    member_root,
)
from .dataset_constants import (
    ANY_DATASET,
    CIFAR_DATASETS,
    CIFAR_LONG_TAIL_SOURCE,
    CIFAR_UNSEEN_CLASS_PROTOCOLS,
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
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOLS,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100,
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500,
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
    split_cifar_balanced_by_fraction,
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


def get_dataset_class(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts
    if dataset_name == "CUB":
        return local_datasets.CUB
    if dataset_name == "CIFAR10":
        return local_datasets.CIFAR10
    if dataset_name == "CIFAR100":
        return local_datasets.CIFAR100
    if dataset_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop
    if dataset_name == "iNat":
        return local_datasets.SemiINaturalist2021
    if dataset_name == "iNat2018":
        return local_datasets.INaturalist2018
    if dataset_name == "SemiAves":
        return local_datasets.SemiAves
    if dataset_name == "StanfordDogs":
        return local_datasets.StanfordDogs
    if dataset_name == "Food101":
        return local_datasets.Food101
    if dataset_name == "Flowers102":
        return local_datasets.Flowers102
    if dataset_name == "FGVCAircraft":
        return local_datasets.FGVCAircraft
    if dataset_name == "FGVCFungi":
        return local_datasets.FGVCFungi
    if dataset_name == "NABirds":
        return local_datasets.NABirds
    if dataset_name == "Pittsburgh30k":
        return local_datasets.Pittsburgh30k
    if dataset_name == "Pittsburgh250k":
        return local_datasets.Pittsburgh250k
    if dataset_name == "SPED":
        return local_datasets.SPED
    if dataset_name == "VehicleID":
        return local_datasets.VehicleID
    if dataset_name == "PUMA":
        return PUMAUnified
    if dataset_name == "PUMAStandard":
        return PUMAStandard

    return getattr(datasets, dataset_name)


def is_dataset_ready(dataset_name, data_root):
    dataset_name = normalize_dataset_name(dataset_name)
    data_root = Path(data_root)

    if dataset_name == "StanfordOnlineProducts":
        return local_datasets.StanfordOnlineProducts.is_ready(data_root)
    if dataset_name == "CUB":
        return local_datasets.CUB.is_ready(data_root)
    if dataset_name == "CIFAR10":
        cifar_root = data_root / "cifar-10-batches-py"
        required_files = [f"data_batch_{index}" for index in range(1, 6)] + ["test_batch", "batches.meta"]
        return all((cifar_root / filename).exists() for filename in required_files)
    if dataset_name == "CIFAR100":
        cifar_root = data_root / "cifar-100-python"
        return all((cifar_root / filename).exists() for filename in ("train", "test", "meta"))
    if dataset_name == "DeepFashionInShop":
        return local_datasets.DeepFashionInShop.find_metadata_file(data_root) is not None
    if dataset_name == "iNat":
        return local_datasets.SemiINaturalist2021.is_ready(data_root)
    if dataset_name == "iNat2018":
        return local_datasets.INaturalist2018.is_ready(data_root)
    if dataset_name == "SemiAves":
        return local_datasets.SemiAves.is_ready(data_root)
    if dataset_name == "StanfordDogs":
        return local_datasets.StanfordDogs.is_ready(data_root)
    if dataset_name == "Food101":
        return local_datasets.Food101.is_ready(data_root)
    if dataset_name == "Flowers102":
        return local_datasets.Flowers102.is_ready(data_root)
    if dataset_name == "FGVCAircraft":
        return local_datasets.FGVCAircraft.is_ready(data_root)
    if dataset_name == "FGVCFungi":
        return local_datasets.FGVCFungi.is_ready(data_root)
    if dataset_name == "NABirds":
        return local_datasets.NABirds.is_ready(data_root)
    if dataset_name == "Pittsburgh30k":
        return local_datasets.Pittsburgh30k.is_ready(data_root)
    if dataset_name == "Pittsburgh250k":
        return local_datasets.Pittsburgh250k.is_ready(data_root)
    if dataset_name == "SPED":
        return local_datasets.SPED.is_ready(data_root)
    if dataset_name == "VehicleID":
        return local_datasets.VehicleID.is_ready(data_root)
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
    cifar_imbalance_factor=None,
    cifar_train_fraction=0.8,
    cifar_test_fraction=0.2,
    seed=0,
    dataset_class_resolver=None,
):
    """Load official splits or construct a custom CIFAR protocol.

    CIFAR's official train/test splits contain the same classes, so the unseen
    and balanced-fraction protocols recombine them before creating new splits.
    """

    validate_dataset_protocol(dataset_name, dataset_protocol)
    validate_cifar_imbalance_factor(dataset_name, cifar_imbalance_factor)
    validate_cifar_balanced_fraction_protocol(
        dataset_name=dataset_name,
        dataset_protocol=dataset_protocol,
        train_fraction=cifar_train_fraction,
        test_fraction=cifar_test_fraction,
        imbalance_factor=cifar_imbalance_factor,
    )

    if dataset_class_resolver is None:
        dataset_class_resolver = get_dataset_class
    dataset_cls = dataset_class_resolver(dataset_name)
    if dataset_protocol in SEMI_AVES_PROTOCOLS:
        # Each protocol names a label pool and a rule for halving its classes.
        class_split = SEMI_AVES_PROTOCOL_CLASS_SPLITS[dataset_protocol]
        mode = (
            local_datasets.SemiAves.MODE_KNOWN
            if dataset_protocol in SEMI_AVES_KNOWN_PROTOCOLS
            else local_datasets.SemiAves.MODE_ORACLE
        )
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            mode=mode,
            transform=train_transform,
            download=download,
            class_split=class_split,
        )
        test_dataset = dataset_cls(
            root=str(data_root),
            split="test",
            mode=mode,
            transform=test_transform,
            download=False,
            class_split=class_split,
        )
        protocol_info = {
            "name": dataset_protocol,
            "dataset_root": str(data_root),
            **getattr(train_val_dataset, "class_split_info", {}),
            "cifar_long_tail": None,
        }
        # Both Semi-Aves protocols pool released labels and then assign complete
        # bird classes to their development and test halves.
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            f"Semi-Aves {mode} development",
            f"Semi-Aves {mode} test",
        )
        if dataset_protocol in SEMI_AVES_KNOWN_PROTOCOLS:
            # Only the known protocols keep an out-of-class pool that no split
            # ever labels, so only they attach the native unlabeled data.
            protocol_info["auto_native_unlabeled_pool"] = True
            protocol_info["native_unlabeled_dataset"] = NATIVE_UNLABELED_SEMI_AVES
        return train_val_dataset, test_dataset, protocol_info
    if dataset_protocol in SEMI_INAT_PROTOCOLS:
        mode = (
            local_datasets.SemiINaturalist2021.MODE_KNOWN
            if dataset_protocol == DATASET_PROTOCOL_SEMI_INAT_KNOWN_810
            else local_datasets.SemiINaturalist2021.MODE_ORACLE
        )
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            mode=mode,
            transform=train_transform,
            download=download,
        )
        test_dataset = dataset_cls(
            root=str(data_root),
            split="test",
            mode=mode,
            transform=test_transform,
            download=False,
        )
        protocol_info = {
            "name": dataset_protocol,
            "dataset_root": str(data_root),
            **getattr(train_val_dataset, "class_split_info", {}),
            "cifar_long_tail": None,
        }
        # Both Semi-iNat protocols assign complete species to their development
        # and test halves, alternating by class id inside each kingdom.
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            f"Semi-iNat {mode} development",
            f"Semi-iNat {mode} test",
        )
        if dataset_protocol == DATASET_PROTOCOL_SEMI_INAT_KNOWN_810:
            # Only the known protocol leaves the 1,629 out-of-class species
            # unlabeled, so only it attaches the native unlabeled data.
            protocol_info["auto_native_unlabeled_pool"] = True
            protocol_info["native_unlabeled_dataset"] = NATIVE_UNLABELED_SEMI_INAT
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
                "cifar_long_tail": None,
            },
        )
    if dataset_protocol in NABIRDS_PROTOCOLS:
        # NABirds already repartitions by class under `official`; these
        # protocols only swap which categories each half gets. The dataset class
        # owns those rules because they read hierarchy.txt, so the protocol
        # layer just names one and reads back the split it recorded.
        class_split = NABIRDS_PROTOCOL_CLASS_SPLITS[dataset_protocol]
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            transform=train_transform,
            download=download,
            class_split=class_split,
        )
        test_dataset = dataset_cls(
            root=str(data_root),
            split="test",
            transform=test_transform,
            download=False,
            class_split=class_split,
        )
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            "NABirds development",
            "NABirds test",
        )
        protocol_info = {
            "name": dataset_protocol,
            "cifar_long_tail": None,
            **getattr(train_val_dataset, "class_split_info", {}),
        }
        return train_val_dataset, test_dataset, protocol_info

    if dataset_protocol in STANFORD_DOGS_PROTOCOLS:
        # Stanford Dogs already repartitions by breed under `official`; this
        # protocol only swaps which breeds each half gets. The dataset class
        # owns both rules, so the protocol layer just names one.
        class_split = STANFORD_DOGS_PROTOCOL_CLASS_SPLITS[dataset_protocol]
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            transform=train_transform,
            download=download,
            class_split=class_split,
        )
        test_dataset = dataset_cls(
            root=str(data_root),
            split="test",
            transform=test_transform,
            download=False,
            class_split=class_split,
        )
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            "Stanford Dogs development",
            "Stanford Dogs test",
        )
        protocol_info = {
            "name": dataset_protocol,
            "cifar_long_tail": None,
            **getattr(train_val_dataset, "class_split_info", {}),
        }
        return train_val_dataset, test_dataset, protocol_info

    if dataset_protocol in VEHICLEID_PROTOCOLS:
        # VehicleID's identity partition is fixed by the released lists; these
        # protocols only choose which of the three held-out lists is tested
        # against. The training half is identical for all three.
        test_gallery_size = VEHICLEID_PROTOCOL_TEST_GALLERY_SIZES[dataset_protocol]
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split="train",
            transform=train_transform,
            download=download,
            test_gallery_size=test_gallery_size,
        )
        test_dataset = dataset_cls(
            root=str(data_root),
            split="test",
            transform=test_transform,
            download=False,
            test_gallery_size=test_gallery_size,
        )
        assert_disjoint_dataset_classes(
            train_val_dataset,
            test_dataset,
            "VehicleID development",
            f"VehicleID test_{test_gallery_size}",
        )
        protocol_info = {
            "name": dataset_protocol,
            "cifar_long_tail": None,
            **getattr(train_val_dataset, "class_split_info", {}),
        }
        return train_val_dataset, test_dataset, protocol_info

    if dataset_protocol == DATASET_PROTOCOL_OFFICIAL:
        # Preserve the dataset provider's official train/test boundary. A
        # dataset that ships a validation split of its own -- Pittsburgh does --
        # names the pooled development split it wants here, because this project
        # carves validation out of the development pool rather than reading a
        # provider's; without it that split would go unused.
        development_split = getattr(dataset_cls, "OFFICIAL_DEVELOPMENT_SPLIT", "train")
        train_val_dataset = dataset_cls(
            root=str(data_root),
            split=development_split,
            transform=train_transform,
            download=download,
        )
        train_val_dataset, imbalance_info = apply_cifar_long_tail(
            train_val_dataset,
            imbalance_factor=cifar_imbalance_factor,
            seed=seed,
        )
        test_dataset = dataset_cls(root=str(data_root), split="test", transform=test_transform, download=False)
        protocol_info = {
            "name": DATASET_PROTOCOL_OFFICIAL,
            "source": "official_train_test_splits",
            "cifar_long_tail": imbalance_info,
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
        return (
            train_val_dataset,
            test_dataset,
            protocol_info,
        )

    if dataset_protocol == DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        official_train = dataset_cls(root=str(data_root), split="train", transform=None, download=download)
        official_test = dataset_cls(root=str(data_root), split="test", transform=None, download=False)
        development_source = CombinedDataset([official_train, official_test], transform=train_transform)
        test_source = CombinedDataset([official_train, official_test], transform=test_transform)
        train_val_dataset, test_dataset, split_info = split_cifar_balanced_by_fraction(
            development_source=development_source,
            test_source=test_source,
            train_fraction=cifar_train_fraction,
            test_fraction=cifar_test_fraction,
            seed=seed,
        )
        return (
            train_val_dataset,
            test_dataset,
            {
                "name": dataset_protocol,
                "source": "combined_official_train_test_splits",
                "sample_disjoint_test": True,
                "class_disjoint_test": False,
                "cifar_long_tail": None,
                **split_info,
            },
        )

    # For unseen-class protocols, the official train/test boundary is
    # intentionally discarded to create a class-disjoint final test set.
    protocol_config = CIFAR_UNSEEN_CLASS_PROTOCOLS[dataset_protocol]
    development_classes = protocol_config["development_classes"]
    held_out_test_classes = protocol_config["held_out_test_classes"]
    official_train = dataset_cls(root=str(data_root), split="train", transform=None, download=download)
    official_test = dataset_cls(root=str(data_root), split="test", transform=None, download=False)
    # Build two views over the same combined samples because development data
    # needs augmentation while the held-out test view must be deterministic.
    development_source = CombinedDataset([official_train, official_test], transform=train_transform)
    test_source = CombinedDataset([official_train, official_test], transform=test_transform)
    train_val_dataset = subset_dataset_by_classes(development_source, development_classes)
    development_pool_size_before_long_tail = len(train_val_dataset)
    train_val_dataset, imbalance_info = apply_cifar_long_tail(
        train_val_dataset,
        imbalance_factor=cifar_imbalance_factor,
        seed=seed,
    )
    test_dataset = subset_dataset_by_classes(test_source, held_out_test_classes)
    assert_disjoint_dataset_classes(train_val_dataset, test_dataset, "development", "test")
    pooled_counts = np.unique(np.asarray(development_source.labels, dtype=np.int64), return_counts=True)[1]
    pooled_samples_per_fine_class = int(pooled_counts[0]) if np.all(pooled_counts == pooled_counts[0]) else None
    protocol_metadata = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in protocol_config.items()
        if key
        in {
            "split_basis",
            "development_superclasses",
            "held_out_test_superclasses",
            "canonical_train_classes",
            "canonical_validation_classes",
            "canonical_train_superclasses",
            "canonical_validation_superclasses",
            "superclass_disjoint_test",
        }
    }
    return (
        train_val_dataset,
        test_dataset,
        {
            "name": dataset_protocol,
            "source": "combined_official_train_test_splits",
            "development_classes": list(development_classes),
            "held_out_test_classes": list(held_out_test_classes),
            "development_pool_size_before_long_tail": development_pool_size_before_long_tail,
            "held_out_test_size": len(test_dataset),
            "pooled_samples_per_fine_class": pooled_samples_per_fine_class,
            "class_disjoint_test": True,
            "cifar_long_tail": imbalance_info,
            **protocol_metadata,
        },
    )


def validate_dataset_protocol(dataset_name, dataset_protocol):
    if dataset_protocol not in DATASET_PROTOCOLS:
        raise ValueError(f"dataset_protocol must be one of {DATASET_PROTOCOLS}: {dataset_protocol}")
    if dataset_protocol == DATASET_PROTOCOL_OFFICIAL:
        return
    if dataset_protocol == DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        if dataset_name not in CIFAR_DATASETS:
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for {CIFAR_DATASETS}"
            )
        return
    if dataset_protocol in SEMI_AVES_PROTOCOLS:
        if dataset_name != "SemiAves":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported "
                "for SemiAves"
            )
        return
    if dataset_protocol in SEMI_INAT_PROTOCOLS:
        if normalize_dataset_name(dataset_name) != "iNat":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported "
                "for SemiINaturalist2021"
            )
        return

    if dataset_protocol in CUB_PROTOCOLS:
        if normalize_dataset_name(dataset_name) != "CUB":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for CUB"
            )
        return

    if dataset_protocol in NABIRDS_PROTOCOLS:
        if normalize_dataset_name(dataset_name) != "NABirds":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for NABirds"
            )
        return

    if dataset_protocol in STANFORD_DOGS_PROTOCOLS:
        if normalize_dataset_name(dataset_name) != "StanfordDogs":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported "
                "for StanfordDogs"
            )
        return

    if dataset_protocol in VEHICLEID_PROTOCOLS:
        if normalize_dataset_name(dataset_name) != "VehicleID":
            raise ValueError(
                f"dataset_protocol={dataset_protocol!r} is only supported for VehicleID"
            )
        return

    supported_dataset = CIFAR_UNSEEN_CLASS_PROTOCOLS[dataset_protocol]["dataset_name"]
    if dataset_name != supported_dataset:
        raise ValueError(f"dataset_protocol={dataset_protocol!r} is only supported for {supported_dataset}")


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


def validate_cifar_balanced_fraction_protocol(
    dataset_name,
    dataset_protocol,
    train_fraction,
    test_fraction,
    imbalance_factor=None,
):
    """Validate the optional balanced per-class CIFAR train/test split."""

    if dataset_protocol != DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION:
        return
    if dataset_name not in CIFAR_DATASETS:
        raise ValueError(f"cifar_balanced_fraction is only supported for {CIFAR_DATASETS}")
    if imbalance_factor is not None:
        raise ValueError("cifar_imbalance_factor cannot be used with cifar_balanced_fraction")
    if train_fraction is None:
        raise ValueError("cifar_train_fraction must be set for cifar_balanced_fraction")
    if test_fraction is None:
        raise ValueError("cifar_test_fraction must be set for cifar_balanced_fraction")
    if not 0 < train_fraction <= 1:
        raise ValueError("cifar_train_fraction must be in (0, 1]")
    if not 0 < test_fraction <= 1:
        raise ValueError("cifar_test_fraction must be in (0, 1]")
    if train_fraction + test_fraction > 1 + 1e-12:
        raise ValueError("cifar_train_fraction + cifar_test_fraction must be less than or equal to 1")


def validate_cifar_imbalance_factor(dataset_name, imbalance_factor):
    if imbalance_factor is None:
        return
    if dataset_name not in CIFAR_DATASETS:
        raise ValueError(f"cifar_imbalance_factor is only supported for {CIFAR_DATASETS}")
    if not 0 < imbalance_factor <= 1:
        raise ValueError("cifar_imbalance_factor must be in (0, 1]")


def apply_cifar_long_tail(dataset, imbalance_factor, seed):
    """Subsample a CIFAR training source using Cui et al.'s long-tail schedule.

    Adapted from ``get_img_num_per_cls`` and ``get_imbalanced_data`` in
    https://github.com/richardaecn/class-balanced-loss (MIT License,
    Copyright (c) 2018 Yin Cui). The factor is ``img_min / img_max``.
    """

    if imbalance_factor is None:
        return dataset, None

    labels = np.asarray(getattr(dataset, "labels", getattr(dataset, "targets", [])), dtype=np.int64)
    if len(labels) != len(dataset):
        raise ValueError("CIFAR long-tail generation requires one label per dataset sample")

    class_labels, available_counts = np.unique(labels, return_counts=True)
    target_counts = make_cifar_long_tail_class_counts(
        class_labels=class_labels,
        available_counts=available_counts,
        imbalance_factor=imbalance_factor,
    )
    rng = random.Random(seed)
    selected_indices = []
    for class_label in class_labels:
        class_indices = np.flatnonzero(labels == class_label).astype(np.int64).tolist()
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[: target_counts[int(class_label)]])

    # Preserve source order so manifests and nested Subset indices remain easy
    # to inspect while the training sampler still controls iteration order.
    selected_indices = sorted(selected_indices)
    subset = Subset(dataset, selected_indices)
    subset.labels = labels[np.asarray(selected_indices, dtype=np.int64)].astype(np.int64).tolist()
    realized_labels, realized_counts = np.unique(np.asarray(subset.labels, dtype=np.int64), return_counts=True)
    realized_counts = {int(label): int(count) for label, count in zip(realized_labels, realized_counts)}
    min_count = min(realized_counts.values())
    max_count = max(realized_counts.values())
    return (
        subset,
        {
            "enabled": True,
            "factor_img_min_over_img_max": float(imbalance_factor),
            "realized_imbalance_ratio_max_to_min": float(max_count / min_count),
            "seed": int(seed),
            "class_counts": realized_counts,
            "source": CIFAR_LONG_TAIL_SOURCE,
            "attribution": "Class-Balanced Loss Based on Effective Number of Samples, Cui et al., CVPR 2019",
        },
    )


def make_cifar_long_tail_class_counts(class_labels, available_counts, imbalance_factor):
    """Return upstream-compatible exponentially decreasing per-class counts."""

    class_labels = np.asarray(class_labels, dtype=np.int64)
    available_counts = np.asarray(available_counts, dtype=np.int64)
    if len(class_labels) == 0 or len(class_labels) != len(available_counts):
        raise ValueError("CIFAR long-tail generation requires non-empty aligned class labels and counts")

    img_max = int(available_counts.max())
    class_count = len(class_labels)
    target_counts = {}
    for class_index, (class_label, available_count) in enumerate(zip(class_labels, available_counts)):
        exponent = 0.0 if class_count == 1 else class_index / (class_count - 1.0)
        target_count = int(img_max * (imbalance_factor**exponent))
        if target_count < 1:
            raise ValueError(
                "cifar_imbalance_factor produces a class with zero samples; "
                "increase the factor for this dataset"
            )
        target_counts[int(class_label)] = min(target_count, int(available_count))
    return target_counts
