"""Constants shared by dataset protocols, splitting, and evaluation."""


CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD = "superclass_balanced_group_kfold"


CV_MODE_SUPERCLASS_GROUP_KFOLD = "superclass_group_kfold"


CV_MODES = (
    "kfold",
    "group_kfold",
    "stratified_kfold",
    "stratified_group_kfold",
    CV_MODE_SUPERCLASS_GROUP_KFOLD,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
)


GROUPED_CV_MODES = (
    "group_kfold",
    "stratified_group_kfold",
    CV_MODE_SUPERCLASS_GROUP_KFOLD,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
)


SUPERCLASS_AWARE_CV_MODES = (
    CV_MODE_SUPERCLASS_GROUP_KFOLD,
    CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD,
)


DATASET_PROTOCOL_OFFICIAL = "official"


DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION = "cifar_balanced_fraction"


DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES = "cifar10_unseen_classes"


DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES = "cifar100_unseen_classes"


DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT = "cifar100_fine_class_disjoint"


DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT = "cifar100_superclass_disjoint"


DATASET_PROTOCOL_CIFAR100_FC100 = "cifar100_fc100"


DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500 = "semi_aves_oracle_500_500"


DATASET_PROTOCOL_SEMI_AVES_ORACLE_HASH_500_500 = "semi_aves_oracle_hash_500_500"


DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100 = "semi_aves_known_100_100"


DATASET_PROTOCOL_SEMI_AVES_KNOWN_HASH_100_100 = "semi_aves_known_hash_100_100"


# Semi-Aves classes carry no taxonomy, only ids numbered from most to least
# frequent, so the two oracle protocols differ in how they halve that ordering.
# Alternating splits the frequency ordering evenly and is the default; hash
# ranking is arbitrary but reproducible, and is worth running only as a check
# that a result does not depend on the id ordering.
SEMI_AVES_ALTERNATING_CLASS_SPLIT = "alternating_class_ids"


SEMI_AVES_SHA256_CLASS_SPLIT = "sha256_rank"


# Which label pool each protocol reads. The known pool is the one that leaves
# out-of-class species unlabeled, so only these protocols attach the native
# unlabeled data and accept --semi_aves_ood_fraction.
SEMI_AVES_KNOWN_PROTOCOLS = (
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_HASH_100_100,
)


SEMI_AVES_ORACLE_PROTOCOLS = (
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500,
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_HASH_500_500,
)


# Which class partition each protocol selects on the dataset class.
SEMI_AVES_PROTOCOL_CLASS_SPLITS = {
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500: SEMI_AVES_ALTERNATING_CLASS_SPLIT,
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_HASH_500_500: SEMI_AVES_SHA256_CLASS_SPLIT,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100: SEMI_AVES_ALTERNATING_CLASS_SPLIT,
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_HASH_100_100: SEMI_AVES_SHA256_CLASS_SPLIT,
}


SEMI_AVES_PROTOCOLS = (
    *SEMI_AVES_ORACLE_PROTOCOLS,
    *SEMI_AVES_KNOWN_PROTOCOLS,
)


# Fraction of the original Semi-Aves out-of-class unlabeled pool that joins the
# unlabeled data. The pool is opt-in, so the default of 0.0 trains on in-class
# data alone; 1.0 keeps the whole released U-out pool and values in between
# reproduce a class-mismatch sweep by dropping the remaining out-of-class
# images entirely.
SEMI_AVES_DEFAULT_OOD_FRACTION = 0.0


SEMI_AVES_DEFAULT_OOD_SEED = 0


DATASET_PROTOCOL_SEMI_INAT_ORACLE_50_50 = "semi_inat_oracle_50_50"


DATASET_PROTOCOL_SEMI_INAT_KNOWN_810 = "semi_inat_known_810"


SEMI_INAT_PROTOCOLS = (
    DATASET_PROTOCOL_SEMI_INAT_ORACLE_50_50,
    DATASET_PROTOCOL_SEMI_INAT_KNOWN_810,
)


# Fraction of the original Semi-iNat out-of-class unlabeled pool that joins the
# unlabeled data, mirroring SEMI_AVES_DEFAULT_OOD_FRACTION.
SEMI_INAT_DEFAULT_OOD_FRACTION = 0


SEMI_INAT_DEFAULT_OOD_SEED = 0


DATASET_PROTOCOL_CUB_ALTERNATING_100_100 = "cub_alternating_100_100"


CUB_PROTOCOLS = (DATASET_PROTOCOL_CUB_ALTERNATING_100_100,)


# CUB-200-2011 labels its classes 1..200 in alphabetical order of the common
# species name, so genera occupy contiguous id blocks (001-003 Albatross,
# 005-008 Auklet, ...). The official protocol cuts that ordering in half, which
# hands complete genera to one side of the split; the alternating protocol takes
# every other class id instead, exactly as the Semi-Aves known protocol does.
CUB_EXPECTED_CLASS_COUNT = 200


CUB_CLASS_IDS = tuple(range(1, CUB_EXPECTED_CLASS_COUNT + 1))


CUB_ALTERNATING_DEVELOPMENT_CLASSES = CUB_CLASS_IDS[0::2]


CUB_ALTERNATING_HELD_OUT_TEST_CLASSES = CUB_CLASS_IDS[1::2]


CUB_ALTERNATING_CLASS_SPLIT_VERSION = "cub_alternating_class_ids_100_100_v1"


CUB_PROTOCOL_REFERENCE = (
    "https://github.com/KevinMusgrave/pytorch-metric-learning/"
    "blob/master/src/pytorch_metric_learning/datasets/cub.py"
)


DATASET_PROTOCOL_NABIRDS_ALTERNATING_277_278 = "nabirds_alternating_277_278"


DATASET_PROTOCOL_NABIRDS_PUMA_278_277 = "nabirds_puma_278_277"


# NABirds cannot alternate single class ids the way CUB does, because it lists
# the plumages of one species as separate categories under a shared hierarchy
# node; striding them would put one plumage in development and its sibling in
# the held-out test set. Both NABirds partitions therefore deal out complete
# parent groups, and this protocol strides those groups instead of taking a
# prefix of them. The category counts are 277/278 either way, but the low class
# ids are the sparser taxa, so the prefix that `official` uses leaves
# development holding 22,534 of the 48,562 images (46.4%) while the stride
# holds 24,548 (50.5%).
NABIRDS_ALTERNATING_CLASS_SPLIT = "alternating_parent_groups"


# PUMA (https://arxiv.org/abs/2309.08944, WACV 2025) compiles a unified metric
# learning benchmark over eight datasets and cuts NABirds at the midpoint of its
# sorted class ids, giving the first 278 categories (22,909 images) to training
# and holding out the last 277 (25,653). Its Table 1 reports exactly those
# counts. The cut ignores hierarchy.txt, and NABirds does not number the
# plumages of a species consecutively, so 12 species end up with categories on
# both sides -- the Red-tailed Hawk trains on its light-morph adult and is
# tested on its light-morph immature. Reproducing that is the point: it is what
# makes the published PUMA NABirds numbers comparable.
NABIRDS_PUMA_CLASS_SPLIT = "class_id_midpoint"


# Which NABirds class partition each protocol selects on the dataset class.
NABIRDS_PROTOCOL_CLASS_SPLITS = {
    DATASET_PROTOCOL_NABIRDS_ALTERNATING_277_278: NABIRDS_ALTERNATING_CLASS_SPLIT,
    DATASET_PROTOCOL_NABIRDS_PUMA_278_277: NABIRDS_PUMA_CLASS_SPLIT,
}


NABIRDS_PROTOCOLS = tuple(NABIRDS_PROTOCOL_CLASS_SPLITS)


DATASET_PROTOCOL_STANFORD_DOGS_PUMA_60_60 = "stanford_dogs_puma_60_60"


# The default Stanford Dogs split hash-ranks the breed directories and takes
# 72 of them for development. PUMA instead cuts the sorted directories at the
# midpoint, 60 breeds against 60, which is what its Table 1 counts (10.6K
# training and 9.9K test images). Reproducing the cut is what makes its
# published Dogs numbers comparable, and it is the split the unified benchmark
# uses for this member.
STANFORD_DOGS_PUMA_CLASS_SPLIT = "class_id_midpoint"


# Which Stanford Dogs class partition each protocol selects on the dataset class.
STANFORD_DOGS_PROTOCOL_CLASS_SPLITS = {
    DATASET_PROTOCOL_STANFORD_DOGS_PUMA_60_60: STANFORD_DOGS_PUMA_CLASS_SPLIT,
}


STANFORD_DOGS_PROTOCOLS = tuple(STANFORD_DOGS_PROTOCOL_CLASS_SPLITS)


DATASET_PROTOCOL_VEHICLEID_TEST_800 = "vehicleid_test_800"


DATASET_PROTOCOL_VEHICLEID_TEST_1600 = "vehicleid_test_1600"


DATASET_PROTOCOL_VEHICLEID_TEST_2400 = "vehicleid_test_2400"


# PKU VehicleID ships one training list and three nested held-out lists of 800,
# 1,600 and 2,400 identities. All three are published evaluation sizes and the
# literature reports each of them, so which one a run tests against is a
# protocol choice rather than a dataset property. `official` uses the 2,400
# identity list, the largest and hardest of the three.
VEHICLEID_PROTOCOL_TEST_GALLERY_SIZES = {
    DATASET_PROTOCOL_VEHICLEID_TEST_800: 800,
    DATASET_PROTOCOL_VEHICLEID_TEST_1600: 1600,
    DATASET_PROTOCOL_VEHICLEID_TEST_2400: 2400,
}


VEHICLEID_PROTOCOLS = tuple(VEHICLEID_PROTOCOL_TEST_GALLERY_SIZES)


# Which native out-of-class pool a protocol attaches, when it attaches one.
NATIVE_UNLABELED_SEMI_AVES = "semi_aves"


NATIVE_UNLABELED_SEMI_INAT = "semi_inat"


DATASET_PROTOCOLS = (
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_FC100,
    *SEMI_AVES_PROTOCOLS,
    *SEMI_INAT_PROTOCOLS,
    *CUB_PROTOCOLS,
    *NABIRDS_PROTOCOLS,
    *STANFORD_DOGS_PROTOCOLS,
    *VEHICLEID_PROTOCOLS,
)


CIFAR10_DEVELOPMENT_CLASSES = tuple(range(8))


CIFAR10_HELD_OUT_TEST_CLASSES = (8, 9)


CIFAR100_DEVELOPMENT_CLASSES = tuple(range(50))


CIFAR100_HELD_OUT_TEST_CLASSES = tuple(range(50, 100))


CIFAR100_SUPERCLASS_NAMES = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)


CIFAR100_SUPERCLASS_FINE_CLASSES = (
    (4, 30, 55, 72, 95),
    (1, 32, 67, 73, 91),
    (54, 62, 70, 82, 92),
    (9, 10, 16, 28, 61),
    (0, 51, 53, 57, 83),
    (22, 39, 40, 86, 87),
    (5, 20, 25, 84, 94),
    (6, 7, 14, 18, 24),
    (3, 42, 43, 88, 97),
    (12, 17, 37, 68, 76),
    (23, 33, 49, 60, 71),
    (15, 19, 21, 31, 38),
    (34, 63, 64, 66, 75),
    (26, 45, 77, 79, 99),
    (2, 11, 35, 46, 98),
    (27, 29, 44, 78, 93),
    (36, 50, 65, 74, 80),
    (47, 52, 56, 59, 96),
    (8, 13, 48, 58, 90),
    (41, 69, 81, 85, 89),
)


CIFAR100_FINE_CLASS_TO_SUPERCLASS = {
    int(fine_class): int(superclass)
    for superclass, fine_classes in enumerate(CIFAR100_SUPERCLASS_FINE_CLASSES)
    for fine_class in fine_classes
}


CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES = tuple(
    sorted(
        fine_class
        for fine_classes in CIFAR100_SUPERCLASS_FINE_CLASSES
        for fine_class in fine_classes[:3]
    )
)


CIFAR100_FINE_CLASS_DISJOINT_TEST_CLASSES = tuple(
    sorted(set(range(100)) - set(CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES))
)


CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES = tuple(range(0, 20, 2))


CIFAR100_SUPERCLASS_DISJOINT_TEST_SUPERCLASSES = tuple(range(1, 20, 2))


CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES = tuple(
    sorted(
        fine_class
        for superclass_index in CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES
        for fine_class in CIFAR100_SUPERCLASS_FINE_CLASSES[superclass_index]
    )
)


CIFAR100_SUPERCLASS_DISJOINT_TEST_CLASSES = tuple(
    sorted(set(range(100)) - set(CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES))
)


# FC100 split from the TADAM supplementary material.  The validation
# superclasses join the training superclasses in the development pool so they
# can either reproduce the canonical holdout or participate in cross-validation.
CIFAR100_FC100_TRAIN_SUPERCLASSES = (1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19)


CIFAR100_FC100_VALIDATION_SUPERCLASSES = (8, 11, 13, 16)


CIFAR100_FC100_TEST_SUPERCLASSES = (0, 7, 12, 14)


CIFAR100_FC100_DEVELOPMENT_SUPERCLASSES = tuple(
    sorted(CIFAR100_FC100_TRAIN_SUPERCLASSES + CIFAR100_FC100_VALIDATION_SUPERCLASSES)
)


def _cifar100_fine_classes_for_superclasses(superclasses):
    return tuple(
        sorted(
            fine_class
            for superclass_index in superclasses
            for fine_class in CIFAR100_SUPERCLASS_FINE_CLASSES[superclass_index]
        )
    )


CIFAR100_FC100_TRAIN_CLASSES = _cifar100_fine_classes_for_superclasses(
    CIFAR100_FC100_TRAIN_SUPERCLASSES
)


CIFAR100_FC100_VALIDATION_CLASSES = _cifar100_fine_classes_for_superclasses(
    CIFAR100_FC100_VALIDATION_SUPERCLASSES
)


CIFAR100_FC100_TEST_CLASSES = _cifar100_fine_classes_for_superclasses(
    CIFAR100_FC100_TEST_SUPERCLASSES
)


CIFAR100_FC100_DEVELOPMENT_CLASSES = tuple(
    sorted(CIFAR100_FC100_TRAIN_CLASSES + CIFAR100_FC100_VALIDATION_CLASSES)
)


CIFAR_UNSEEN_CLASS_PROTOCOLS = {
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES: {
        "dataset_name": "CIFAR10",
        "development_classes": CIFAR10_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR10_HELD_OUT_TEST_CLASSES,
        "split_basis": "fine_class_contiguous_ids",
    },
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_HELD_OUT_TEST_CLASSES,
        "split_basis": "fine_class_contiguous_ids_legacy",
        "superclass_disjoint_test": False,
    },
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_FINE_CLASS_DISJOINT_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_FINE_CLASS_DISJOINT_TEST_CLASSES,
        "split_basis": "fine_class_within_superclass",
        "development_superclasses": tuple(range(20)),
        "held_out_test_superclasses": tuple(range(20)),
        "superclass_disjoint_test": False,
    },
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_SUPERCLASS_DISJOINT_TEST_CLASSES,
        "split_basis": "superclass",
        "development_superclasses": CIFAR100_SUPERCLASS_DISJOINT_DEVELOPMENT_SUPERCLASSES,
        "held_out_test_superclasses": CIFAR100_SUPERCLASS_DISJOINT_TEST_SUPERCLASSES,
        "superclass_disjoint_test": True,
    },
    DATASET_PROTOCOL_CIFAR100_FC100: {
        "dataset_name": "CIFAR100",
        "development_classes": CIFAR100_FC100_DEVELOPMENT_CLASSES,
        "held_out_test_classes": CIFAR100_FC100_TEST_CLASSES,
        "split_basis": "fc100_superclass",
        "development_superclasses": CIFAR100_FC100_DEVELOPMENT_SUPERCLASSES,
        "held_out_test_superclasses": CIFAR100_FC100_TEST_SUPERCLASSES,
        "canonical_train_classes": CIFAR100_FC100_TRAIN_CLASSES,
        "canonical_validation_classes": CIFAR100_FC100_VALIDATION_CLASSES,
        "canonical_train_superclasses": CIFAR100_FC100_TRAIN_SUPERCLASSES,
        "canonical_validation_superclasses": CIFAR100_FC100_VALIDATION_SUPERCLASSES,
        "superclass_disjoint_test": True,
    },
}


CIFAR_DATASETS = ("CIFAR10", "CIFAR100")


CIFAR_LONG_TAIL_SOURCE = "https://github.com/richardaecn/class-balanced-loss"


# Every protocol accepted by --dataset_protocol, in the order they are offered.
# ``datasets`` is the tuple of dataset names the protocol may be combined with,
# or ANY_DATASET when the protocol works for all of them; validation in
# utils.dataset_protocols.validate_dataset_protocol enforces the same pairing.
# docs/dataset_protocols.md documents each entry in full.
ANY_DATASET = "any"


DATASET_PROTOCOL_INFO = {
    DATASET_PROTOCOL_OFFICIAL: {
        "datasets": ANY_DATASET,
        "class_disjoint_test": None,
        "summary": (
            "keeps each dataset class's own train/test boundary, which is the "
            "provider's official split for Cars196, CUB, SOP and In-Shop and a "
            "fixed class-disjoint repartition for the datasets whose official "
            "split shares classes"
        ),
    },
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION: {
        "datasets": CIFAR_DATASETS,
        "class_disjoint_test": False,
        "summary": (
            "pools the official CIFAR splits and draws sample-disjoint, "
            "class-balanced development and test subsets sized by "
            "cifar_train_fraction and cifar_test_fraction"
        ),
    },
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES: {
        "datasets": ("CIFAR10",),
        "class_disjoint_test": True,
        "summary": "develops on CIFAR-10 classes 0-7 and holds out classes 8-9",
    },
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES: {
        "datasets": ("CIFAR100",),
        "class_disjoint_test": True,
        "summary": (
            "develops on CIFAR-100 fine classes 0-49 and holds out 50-99; the "
            "contiguous cut splits superclasses, so both halves share coarse labels"
        ),
    },
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT: {
        "datasets": ("CIFAR100",),
        "class_disjoint_test": True,
        "summary": (
            "takes 3 of the 5 fine classes of every CIFAR-100 superclass for "
            "development and holds out the other 2, so all 20 superclasses appear "
            "on both sides"
        ),
    },
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT: {
        "datasets": ("CIFAR100",),
        "class_disjoint_test": True,
        "summary": (
            "develops on the 10 even CIFAR-100 superclasses and holds out the 10 "
            "odd ones, so no held-out class has a coarse relative in development"
        ),
    },
    DATASET_PROTOCOL_CIFAR100_FC100: {
        "datasets": ("CIFAR100",),
        "class_disjoint_test": True,
        "summary": (
            "reproduces the TADAM FC100 12/4/4 superclass split; requires "
            "cv_mode=superclass_group_kfold when cv_k > 1"
        ),
    },
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_500_500: {
        "datasets": ("SemiAves",),
        "class_disjoint_test": True,
        "summary": (
            "pools every released Semi-Aves label and alternates all 1,000 classes "
            "into a fixed 500/500 split, so the most-to-least-frequent id ordering "
            "is halved evenly"
        ),
    },
    DATASET_PROTOCOL_SEMI_AVES_ORACLE_HASH_500_500: {
        "datasets": ("SemiAves",),
        "class_disjoint_test": True,
        "summary": (
            "the oracle 500/500 split with classes ranked by a SHA-256 hash instead "
            "of alternated, as a check that a result does not depend on the "
            "most-to-least-frequent id ordering"
        ),
    },
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_100_100: {
        "datasets": ("SemiAves",),
        "class_disjoint_test": True,
        "summary": (
            "splits the 200 known Semi-Aves classes 100/100 by alternating class "
            "id and attaches the out-of-class pool as native unlabeled data "
            "(see semi_aves_ood_fraction)"
        ),
    },
    DATASET_PROTOCOL_SEMI_AVES_KNOWN_HASH_100_100: {
        "datasets": ("SemiAves",),
        "class_disjoint_test": True,
        "summary": (
            "the known 100/100 split with classes ranked by a SHA-256 hash instead "
            "of alternated; same out-of-class unlabeled pool, but the image balance "
            "is left to the hash rather than fixed by the id ordering"
        ),
    },
    DATASET_PROTOCOL_SEMI_INAT_ORACLE_50_50: {
        "datasets": ("iNat",),
        "class_disjoint_test": True,
        "summary": (
            "pools every released Semi-iNat label and alternates all 2,439 species "
            "into two halves of roughly equal size"
        ),
    },
    DATASET_PROTOCOL_SEMI_INAT_KNOWN_810: {
        "datasets": ("iNat",),
        "class_disjoint_test": True,
        "summary": (
            "splits the 810 labeled Semi-iNat species and attaches the 1,629 "
            "out-of-class species as native unlabeled data "
            "(see semi_inat_ood_fraction)"
        ),
    },
    DATASET_PROTOCOL_CUB_ALTERNATING_100_100: {
        "datasets": ("CUB",),
        "class_disjoint_test": True,
        "summary": (
            "pools CUB's official halves and alternates class ids 100/100, so each "
            "genus appears on both sides instead of only one"
        ),
    },
    DATASET_PROTOCOL_NABIRDS_ALTERNATING_277_278: {
        "datasets": ("NABirds",),
        "class_disjoint_test": True,
        "summary": (
            "alternates whole hierarchy-parent groups 277/278 rather than taking a "
            "prefix of them, which evens the images at 50.5/49.5 instead of "
            "46.4/53.6 while keeping every plumage of a species on one side"
        ),
    },
    DATASET_PROTOCOL_NABIRDS_PUMA_278_277: {
        "datasets": ("NABirds",),
        "class_disjoint_test": True,
        "summary": (
            "reproduces the PUMA benchmark's cut at the class id midpoint, 278/277, "
            "for comparability with its published numbers; ignores the hierarchy, so "
            "12 species keep categories on both sides (see straddling_parent_count)"
        ),
    },
    DATASET_PROTOCOL_STANFORD_DOGS_PUMA_60_60: {
        "datasets": ("StanfordDogs",),
        "class_disjoint_test": True,
        "summary": (
            "reproduces the PUMA benchmark's Stanford Dogs cut, the first 60 breed "
            "directories against the last 60, instead of the default hash-ranked "
            "72/48 split"
        ),
    },
    DATASET_PROTOCOL_VEHICLEID_TEST_800: {
        "datasets": ("VehicleID",),
        "class_disjoint_test": True,
        "summary": (
            "tests against VehicleID's 800-identity list, the small published "
            "evaluation set; training identities are unchanged"
        ),
    },
    DATASET_PROTOCOL_VEHICLEID_TEST_1600: {
        "datasets": ("VehicleID",),
        "class_disjoint_test": True,
        "summary": (
            "tests against VehicleID's 1,600-identity list, the medium published "
            "evaluation set; training identities are unchanged"
        ),
    },
    DATASET_PROTOCOL_VEHICLEID_TEST_2400: {
        "datasets": ("VehicleID",),
        "class_disjoint_test": True,
        "summary": (
            "tests against VehicleID's 2,400-identity list, the large published "
            "evaluation set and what `official` already uses"
        ),
    },
}


VAL_MODE_ALL = "all"


VAL_MODE_MATCH_TRAIN = "match_train"


VAL_MODE_SPLIT_AFTER_APPORTION = "split_after_apportion"


VAL_MODES = (VAL_MODE_ALL, VAL_MODE_MATCH_TRAIN, VAL_MODE_SPLIT_AFTER_APPORTION)


POST_APPORTION_VAL_RATIO = 0.2


QUERY_GALLERY_EVALUATION = "query_gallery"


SAME_SOURCE_EVALUATION = "same_source"


VALIDATION_RETRIEVAL_MODES = (SAME_SOURCE_EVALUATION, QUERY_GALLERY_EVALUATION)


DEFAULT_VALIDATION_GALLERY_FRACTION = 0.5
