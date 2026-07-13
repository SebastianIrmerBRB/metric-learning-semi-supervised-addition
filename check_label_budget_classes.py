from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402
from training import semi_supervised  # noqa: E402
from training.types import DATASETS  # noqa: E402


@dataclass(frozen=True)
class ClassSelection:
    budget: float
    support_seed: int
    data_split_seed: int
    labeled_per_class: int | None
    labeled_samples: int
    total_train_samples: int
    selected_classes: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare which original classes are selected by different label "
            "budgets and support seeds."
        )
    )
    parser.add_argument("dataset", nargs="?", choices=DATASETS, help="Dataset to inspect.")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=r"C:\Users\Sebastian\PycharmProjects\metric-learning\configs\experiments\class\cifar100_config.json",
        help="Optional experiment JSON used for defaults such as dataset, protocol, budgets, and seeds.",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        #default=[10, 20],
        help="Label budgets / labeled_fraction values to compare, for example 0.1 0.2 0.5.",
    )
    parser.add_argument(
        "--seeds",
        "--support-seeds",
        dest="support_seeds",
        type=int,
        nargs="+",
        default=[0, 7],
        help="Support-selection seeds to compare.",
    )
    parser.add_argument(
        "--label-sampling-mode",
        choices=sorted(semi_supervised.LABEL_SAMPLING_MODES),
        default='class_subset_k_shot',
        help="Label sampling mode to check.",
    )
    parser.add_argument(
        "--labeled-per-class",
        "--k",
        dest="labeled_per_class",
        type=int,
        default=10,
        help="Fixed labeled examples per class, required for class_subset_k_shot.",
    )
    parser.add_argument(
        "--dataset-protocol",
        choices=utils.DATASET_PROTOCOLS,
        default=None,
        help="Dataset split protocol.",
    )
    parser.add_argument("--data-split-seed", type=int, default=None, help="Fixed dataset split seed.")
    parser.add_argument(
        "--vary-data-split-with-seed",
        action="store_true",
        help="Also use each support seed as the dataset split seed.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Runtime seed used while building the dataset bundle.")
    parser.add_argument("--cv-k", type=int, default=None, help="Number of CV folds used for dataset setup.")
    parser.add_argument("--cv-fold", type=int, default=0, help="CV fold to inspect when cv_k > 1.")
    parser.add_argument("--cv-mode", choices=utils.CV_MODES, default=None, help="CV split mode.")
    parser.add_argument("--val-mode", choices=utils.VAL_MODES, default=None, help="Validation mode.")
    parser.add_argument("--cifar-imbalance-factor", type=float, default=None)
    parser.add_argument("--cifar-train-fraction", type=float, default=None)
    parser.add_argument("--cifar-test-fraction", type=float, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow missing datasets to be downloaded by the normal dataset loader.",
    )
    parser.add_argument(
        "--show-classes",
        action="store_true",
        help="Print the full selected class list for every budget/seed pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for class_sets.csv and pairwise_class_overlap.csv.",
    )
    parser.add_argument(
        "--fail-on-difference",
        action="store_true",
        help="Exit with code 1 if any compared pair does not use exactly the same classes.",
    )
    return parser.parse_args()


def read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def first_config_value(config: dict, key: str):
    value = config.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def resolve_list(cli_value, config: dict, key: str, fallback):
    if cli_value is not None:
        return list(cli_value)
    value = config.get(key)
    if value is None:
        return list(fallback)
    if isinstance(value, list):
        return list(value)
    return [value]


def resolve_options(args: argparse.Namespace) -> argparse.Namespace:
    experiment = read_json(args.experiment_config)
    ssl_config = read_json(Path(experiment["ssl_config"])) if experiment.get("ssl_config") else {}

    dataset = args.dataset or experiment.get("dataset")
    if dataset is None:
        raise SystemExit("Provide a dataset or --experiment-config with a dataset.")

    label_modes = resolve_list(
        [args.label_sampling_mode] if args.label_sampling_mode is not None else None,
        experiment,
        "ssl_label_sampling_modes",
        [ssl_config.get("label_sampling_mode", "global_budget")],
    )
    if len(label_modes) != 1:
        raise SystemExit(
            "This checker expects one label sampling mode. Pass --label-sampling-mode to choose one."
        )

    budgets = [float(value) for value in resolve_list(
        args.budgets,
        experiment,
        "label_budget_grid",
        [ssl_config.get("labeled_fraction", 1.0)],
    )]
    for budget in budgets:
        if not (0 < budget <= 1):
            raise SystemExit(f"Budgets must be in (0, 1], got {budget}")

    support_seeds = [int(value) for value in resolve_list(
        args.support_seeds,
        experiment,
        "comparison_seeds",
        [experiment.get("support_seed", semi_supervised.DEFAULT_SUPPORT_SEED)],
    )]

    labeled_per_class = args.labeled_per_class
    if labeled_per_class is None:
        labeled_per_class = first_config_value(experiment, "k_shot_grid")
    if labeled_per_class is None:
        labeled_per_class = ssl_config.get("labeled_per_class")
    if labeled_per_class is not None:
        labeled_per_class = int(labeled_per_class)

    resolved = argparse.Namespace(**vars(args))
    resolved.dataset = utils.normalize_dataset_name(dataset)
    resolved.budgets = budgets
    resolved.support_seeds = support_seeds
    resolved.label_sampling_mode = label_modes[0]
    resolved.labeled_per_class = labeled_per_class
    resolved.dataset_protocol = args.dataset_protocol or experiment.get(
        "dataset_protocol",
        utils.DATASET_PROTOCOL_OFFICIAL,
    )
    resolved.data_split_seed = int(
        args.data_split_seed
        if args.data_split_seed is not None
        else experiment.get("data_split_seed", 7)
    )
    resolved.seed = int(args.seed if args.seed is not None else experiment.get("seed", 7))
    resolved.cv_k = int(args.cv_k if args.cv_k is not None else experiment.get("cv_k", 1))
    resolved.cv_mode = args.cv_mode or experiment.get("cv_mode", "group_kfold")
    resolved.val_mode = args.val_mode or experiment.get("val_mode", utils.VAL_MODE_ALL)
    resolved.cifar_imbalance_factor = (
        args.cifar_imbalance_factor
        if args.cifar_imbalance_factor is not None
        else experiment.get("cifar_imbalance_factor")
    )
    resolved.cifar_train_fraction = float(
        args.cifar_train_fraction
        if args.cifar_train_fraction is not None
        else experiment.get("cifar_train_fraction", 0.8)
    )
    resolved.cifar_test_fraction = float(
        args.cifar_test_fraction
        if args.cifar_test_fraction is not None
        else experiment.get("cifar_test_fraction", 0.2)
    )
    return resolved


def ensure_dataset_ready(dataset_name: str, allow_download: bool) -> None:
    data_root = Path("data") / dataset_name
    if allow_download or utils.is_dataset_ready(dataset_name, data_root):
        return
    raise SystemExit(
        f"{dataset_name} is not ready under {data_root}. "
        "Prepare the dataset first or rerun with --download."
    )


def build_dataset_bundle(args: argparse.Namespace, data_split_seed: int):
    return utils.setup_dataset_bundle(
        dataset_name=args.dataset,
        seed=args.seed,
        data_split_seed=data_split_seed,
        cv_k=args.cv_k,
        cv_fold=args.cv_fold,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
        dataset_protocol=args.dataset_protocol,
        cifar_imbalance_factor=args.cifar_imbalance_factor,
        cifar_train_fraction=args.cifar_train_fraction,
        cifar_test_fraction=args.cifar_test_fraction,
    )


def selected_original_classes(train_dataset, labeled_positions: np.ndarray) -> tuple[int, ...]:
    labels = np.asarray(getattr(train_dataset, "orig_labels", train_dataset.labels), dtype=np.int64)
    return tuple(sorted(int(label) for label in np.unique(labels[labeled_positions])))


def make_selection(
    args: argparse.Namespace,
    train_dataset,
    budget: float,
    support_seed: int,
    data_split_seed: int,
) -> ClassSelection:
    config = semi_supervised.SemiSupervisedConfig(
        method="none",
        label_sampling_mode=args.label_sampling_mode,
        labeled_fraction=float(budget),
        labeled_per_class=args.labeled_per_class,
        seed=int(support_seed),
        support_seed=int(support_seed),
    )
    semi_supervised.validate_ssl_config(config)
    split = semi_supervised.prepare_label_split(train_dataset, config)
    selected_classes = selected_original_classes(train_dataset, split.labeled_positions)
    return ClassSelection(
        budget=float(budget),
        support_seed=int(support_seed),
        data_split_seed=int(data_split_seed),
        labeled_per_class=args.labeled_per_class,
        labeled_samples=int(len(split.labeled_positions)),
        total_train_samples=int(len(train_dataset)),
        selected_classes=selected_classes,
    )


def format_classes(classes: tuple[int, ...], show_all: bool, preview: int = 20) -> str:
    if show_all or len(classes) <= preview:
        return "[" + ", ".join(str(label) for label in classes) + "]"
    head = ", ".join(str(label) for label in classes[:preview])
    return f"[{head}, ... +{len(classes) - preview} more]"


def print_selection_summary(selections: list[ClassSelection], show_classes: bool) -> None:
    print("\nSelected classes")
    for selection in selections:
        classes_text = format_classes(selection.selected_classes, show_classes)
        print(
            "  "
            f"budget={selection.budget:g}, "
            f"support_seed={selection.support_seed}, "
            f"data_split_seed={selection.data_split_seed}, "
            f"k={selection.labeled_per_class}, "
            f"labeled_samples={selection.labeled_samples}/{selection.total_train_samples}, "
            f"classes={len(selection.selected_classes)} {classes_text}"
        )


def pairwise_rows(selections: list[ClassSelection]) -> list[dict[str, object]]:
    rows = []
    for left, right in combinations(selections, 2):
        left_classes = set(left.selected_classes)
        right_classes = set(right.selected_classes)
        shared = left_classes & right_classes
        union = left_classes | right_classes
        only_left = tuple(sorted(left_classes - right_classes))
        only_right = tuple(sorted(right_classes - left_classes))
        rows.append(
            {
                "left_budget": left.budget,
                "left_support_seed": left.support_seed,
                "left_data_split_seed": left.data_split_seed,
                "right_budget": right.budget,
                "right_support_seed": right.support_seed,
                "right_data_split_seed": right.data_split_seed,
                "same_classes": left_classes == right_classes,
                "left_subset_of_right": left_classes <= right_classes,
                "right_subset_of_left": right_classes <= left_classes,
                "left_class_count": len(left_classes),
                "right_class_count": len(right_classes),
                "shared_class_count": len(shared),
                "union_class_count": len(union),
                "jaccard": 1.0 if not union else len(shared) / len(union),
                "only_left": " ".join(str(label) for label in only_left),
                "only_right": " ".join(str(label) for label in only_right),
            }
        )
    return rows


def print_pairwise_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("\nOnly one budget/seed pair was checked; no pairwise comparison to print.")
        return

    print("\nPairwise class-set comparison")
    for row in rows:
        print(
            "  "
            f"budget={row['left_budget']:g}, seed={row['left_support_seed']} "
            f"vs budget={row['right_budget']:g}, seed={row['right_support_seed']}: "
            f"same={yes_no(row['same_classes'])}, "
            f"left_subset={yes_no(row['left_subset_of_right'])}, "
            f"right_subset={yes_no(row['right_subset_of_left'])}, "
            f"shared={row['shared_class_count']}/{row['union_class_count']}, "
            f"jaccard={row['jaccard']:.3f}"
        )
        if row["only_left"] or row["only_right"]:
            print(f"    only_left: {row['only_left'] or '-'}")
            print(f"    only_right: {row['only_right'] or '-'}")


def yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def write_csv_outputs(output_dir: Path, selections: list[ClassSelection], rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_path = output_dir / "class_sets.csv"
    with class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "budget",
                "support_seed",
                "data_split_seed",
                "labeled_per_class",
                "labeled_samples",
                "total_train_samples",
                "selected_class_count",
                "selected_classes",
            ],
        )
        writer.writeheader()
        for selection in selections:
            writer.writerow(
                {
                    "budget": selection.budget,
                    "support_seed": selection.support_seed,
                    "data_split_seed": selection.data_split_seed,
                    "labeled_per_class": selection.labeled_per_class,
                    "labeled_samples": selection.labeled_samples,
                    "total_train_samples": selection.total_train_samples,
                    "selected_class_count": len(selection.selected_classes),
                    "selected_classes": " ".join(str(label) for label in selection.selected_classes),
                }
            )

    pairwise_path = output_dir / "pairwise_class_overlap.csv"
    with pairwise_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "left_budget",
            "left_support_seed",
            "left_data_split_seed",
            "right_budget",
            "right_support_seed",
            "right_data_split_seed",
            "same_classes",
            "left_subset_of_right",
            "right_subset_of_left",
            "left_class_count",
            "right_class_count",
            "shared_class_count",
            "union_class_count",
            "jaccard",
            "only_left",
            "only_right",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {class_path}")
    print(f"Wrote {pairwise_path}")


def main() -> int:
    args = resolve_options(parse_args())
    ensure_dataset_ready(args.dataset, allow_download=args.download)

    print(
        "Checking "
        f"dataset={args.dataset}, protocol={args.dataset_protocol}, "
        f"label_sampling_mode={args.label_sampling_mode}, budgets={args.budgets}, "
        f"support_seeds={args.support_seeds}"
    )

    bundle_cache = {}
    selections = []
    for support_seed in args.support_seeds:
        data_split_seed = support_seed if args.vary_data_split_with_seed else args.data_split_seed
        if data_split_seed not in bundle_cache:
            bundle_cache[data_split_seed] = build_dataset_bundle(args, data_split_seed)
        train_dataset = bundle_cache[data_split_seed].train_dataset
        for budget in args.budgets:
            selections.append(
                make_selection(
                    args=args,
                    train_dataset=train_dataset,
                    budget=budget,
                    support_seed=support_seed,
                    data_split_seed=data_split_seed,
                )
            )

    print_selection_summary(selections, show_classes=args.show_classes)
    rows = pairwise_rows(selections)
    print_pairwise_summary(rows)

    if args.output_dir is not None:
        write_csv_outputs(args.output_dir, selections, rows)

    if args.fail_on_difference and any(not row["same_classes"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
