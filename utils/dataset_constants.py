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


DATASET_PROTOCOLS = (
    DATASET_PROTOCOL_OFFICIAL,
    DATASET_PROTOCOL_CIFAR_BALANCED_FRACTION,
    DATASET_PROTOCOL_CIFAR10_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_UNSEEN_CLASSES,
    DATASET_PROTOCOL_CIFAR100_FINE_CLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_SUPERCLASS_DISJOINT,
    DATASET_PROTOCOL_CIFAR100_FC100,
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


VAL_MODE_ALL = "all"


VAL_MODE_MATCH_TRAIN = "match_train"


VAL_MODE_SPLIT_AFTER_APPORTION = "split_after_apportion"


VAL_MODES = (VAL_MODE_ALL, VAL_MODE_MATCH_TRAIN, VAL_MODE_SPLIT_AFTER_APPORTION)


POST_APPORTION_VAL_RATIO = 0.2


QUERY_GALLERY_EVALUATION = "query_gallery"


SAME_SOURCE_EVALUATION = "same_source"
