from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import semi_supervised
import utils


SUPPORTED_DATASETS = ("Cars196", "StanfordOnlineProducts", "CIFAR10")
SUMMARY_COLUMNS = [
    "dataset",
    "scenario_id",
    "split",
    "num_samples",
    "num_classes",
    "min_samples_per_class",
    "max_samples_per_class",
    "avg_samples_per_class",
    "median_samples_per_class",
    "std_samples_per_class",
    "q25_samples_per_class",
    "q75_samples_per_class",
    "imbalance_ratio_max_to_min",
    "num_singleton_classes",
    "num_classes_lt_2",
    "num_classes_lt_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export exact class distributions, min/max/average samples per class, "
            "CSV tables, JSON, Markdown, and matplotlib diagrams for the project datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help=f"Dataset(s) to analyze. Use 'all' for {', '.join(SUPPORTED_DATASETS)}.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for train/validation and SSL splits.")
    parser.add_argument(
        "--dataset-protocol",
        choices=utils.DATASET_PROTOCOLS,
        default=utils.DATASET_PROTOCOL_OFFICIAL,
        help="Dataset split protocol, matching main.py.",
    )
    parser.add_argument("--cv-k", type=int, default=1, help="Number of CV folds. Use 1 for the default holdout split.")
    parser.add_argument(
        "--cv-fold",
        type=int,
        nargs="*",
        default=None,
        help="CV fold(s) to analyze. If omitted and --cv-k > 1, all folds are exported.",
    )
    parser.add_argument(
        "--cv-mode",
        choices=utils.CV_MODES,
        default="group_kfold",
        help="Cross-validation mode, matching main.py.",
    )
    parser.add_argument(
        "--val-mode",
        choices=utils.VAL_MODES,
        default=utils.VAL_MODE_ALL,
        help="Validation mode, matching main.py.",
    )
    parser.add_argument(
        "--ssl-config",
        type=Path,
        default=None,
        help="Optional SSL JSON config. Adds exact labeled/unlabeled split distributions.",
    )
    parser.add_argument(
        "--mode",
        choices=("supervised", "ssl"),
        default="supervised",
        help=(
            "How to name the effective training split when --ssl-config is given. "
            "This does not run pseudo-labeling; it reports true class distributions."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/reports/class_distributions"),
        help="Directory for generated CSV, JSON, Markdown, and chart files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow dataset download if metadata is missing. By default the script only uses local data.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Number of smallest/largest classes in detail charts.")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for PNG charts.")
    parser.add_argument(
        "--max-x-labels",
        type=int,
        default=80,
        help="Maximum class labels shown on count charts before labels are hidden.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)
    cv_folds = resolve_cv_folds(args.cv_k, args.cv_fold)
    out_dir = args.out_dir
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    ssl_config, ignored_ssl_keys = load_ssl_config(args.ssl_config, default_seed=args.seed)

    class_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    scenario_metadata: list[dict[str, Any]] = []

    for dataset_name, cv_fold in itertools.product(datasets, cv_folds):
        scenario = analyze_scenario(
            dataset_name=dataset_name,
            seed=args.seed,
            cv_k=args.cv_k,
            cv_fold=cv_fold,
            cv_mode=args.cv_mode,
            val_mode=args.val_mode,
            ssl_config=ssl_config,
            ssl_config_path=args.ssl_config,
            mode=args.mode,
            download=args.download,
            dataset_protocol=args.dataset_protocol,
        )
        scenario_metadata.append(scenario["metadata"])
        class_frames.extend(scenario["class_frames"])
        summary_rows.extend(scenario["summary_rows"])
        sample_frames.extend(scenario["sample_frames"])
        overlap_rows.extend(scenario["overlap_rows"])

    class_distribution = concat_or_empty(class_frames)
    split_summary = pd.DataFrame(summary_rows)
    sample_membership = concat_or_empty(sample_frames)
    class_overlap = pd.DataFrame(overlap_rows)

    class_distribution = sort_frame(class_distribution, ["dataset", "scenario_id", "split", "class_label"])
    split_summary = sort_frame(split_summary, ["dataset", "scenario_id", "split"])
    sample_membership = sort_frame(sample_membership, ["dataset", "scenario_id", "split", "source_index"])
    class_overlap = sort_frame(class_overlap, ["dataset", "scenario_id", "split_a", "split_b"])

    pivot = make_distribution_pivot(class_distribution)

    files = write_tables(
        out_dir=out_dir,
        class_distribution=class_distribution,
        split_summary=split_summary,
        sample_membership=sample_membership,
        class_overlap=class_overlap,
        pivot=pivot,
    )
    chart_files = write_charts(
        charts_dir=charts_dir,
        class_distribution=class_distribution,
        split_summary=split_summary,
        top_n=args.top_n,
        dpi=args.dpi,
        max_x_labels=args.max_x_labels,
    )
    files.extend(chart_files)

    metadata = {
        "parameters": {
            "datasets": datasets,
            "seed": args.seed,
            "cv_k": args.cv_k,
            "cv_folds": cv_folds,
            "cv_mode": args.cv_mode,
            "val_mode": args.val_mode,
            "ssl_config": None if args.ssl_config is None else str(args.ssl_config),
            "mode": args.mode,
            "download": args.download,
        },
        "ignored_ssl_config_keys": ignored_ssl_keys,
        "scenarios": scenario_metadata,
        "files": [str(path) for path in files],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files.append(summary_json)

    report_md = out_dir / "report.md"
    write_markdown_report(report_md, split_summary, class_overlap, files, metadata)
    files.append(report_md)

    print(f"Wrote class distribution report to: {out_dir}")
    print(f"Summary table: {out_dir / 'split_summary.csv'}")
    print(f"Exact per-class counts: {out_dir / 'class_distribution.csv'}")
    print(f"Charts: {charts_dir}")
    return 0


def resolve_datasets(raw_datasets: list[str]) -> list[str]:
    requested = [utils.normalize_dataset_name(value) for value in raw_datasets]
    if any(value.lower() == "all" for value in requested):
        return list(SUPPORTED_DATASETS)

    datasets: list[str] = []
    for dataset in requested:
        if dataset not in SUPPORTED_DATASETS:
            raise SystemExit(f"Unsupported dataset {dataset!r}. Available: all, {', '.join(SUPPORTED_DATASETS)}")
        if dataset not in datasets:
            datasets.append(dataset)
    return datasets


def resolve_cv_folds(cv_k: int, raw_cv_folds: list[int] | None) -> list[int | None]:
    if cv_k <= 0:
        raise SystemExit("--cv-k must be positive")
    if cv_k == 1:
        return [None]

    if raw_cv_folds is None or len(raw_cv_folds) == 0:
        return list(range(cv_k))

    invalid = [fold for fold in raw_cv_folds if fold < 0 or fold >= cv_k]
    if invalid:
        raise SystemExit(f"--cv-fold values must be in [0, {cv_k - 1}], got {invalid}")
    return sorted(set(raw_cv_folds))


def load_ssl_config(
    config_path: Path | None,
    default_seed: int,
) -> tuple[semi_supervised.SemiSupervisedConfig | None, list[str]]:
    if config_path is None:
        return None, []

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise SystemExit(f"SSL config must be a JSON object: {config_path}")

    allowed_keys = set(semi_supervised.SemiSupervisedConfig.__dataclass_fields__)
    ignored_keys = sorted(set(raw_config) - allowed_keys)
    filtered_config = {key: value for key, value in raw_config.items() if key in allowed_keys}
    config = semi_supervised.SemiSupervisedConfig(**filtered_config)
    if config.seed is None:
        config = replace(config, seed=default_seed)
    semi_supervised.validate_ssl_config(config, config_path)
    return config, ignored_keys


def analyze_scenario(
    dataset_name: str,
    seed: int,
    cv_k: int,
    cv_fold: int | None,
    cv_mode: str,
    val_mode: str,
    ssl_config: semi_supervised.SemiSupervisedConfig | None,
    ssl_config_path: Path | None,
    mode: str,
    download: bool,
    dataset_protocol: str = utils.DATASET_PROTOCOL_OFFICIAL,
) -> dict[str, Any]:
    train_val_dataset, test_dataset, protocol_info = load_datasets(
        dataset_name,
        download=download,
        dataset_protocol=dataset_protocol,
    )
    class_metadata = load_class_metadata(dataset_name, train_val_dataset, test_dataset)

    if val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        train_dataset, valid_dataset, train_labels_mapper = utils.make_train_valid_subsets(
            train_val_dataset,
            np.arange(len(train_val_dataset), dtype=np.int64).tolist(),
            [],
        )
        split_kind = "post_apportion_source"
    elif cv_k > 1:
        train_dataset, valid_dataset, train_labels_mapper = utils.split_dataset_cross_validation(
            train_val_dataset,
            cv_k=cv_k,
            cv_fold=cv_fold,
            cv_mode=cv_mode,
            seed=seed,
        )
        split_kind = "cross_validation"
    else:
        train_dataset, valid_dataset, train_labels_mapper = utils.split_dataset_by_classes(
            train_val_dataset,
            seed=seed,
        )
        split_kind = "holdout"

    bundle = utils.DatasetBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_labels_mapper=train_labels_mapper,
        split_info=None,
    )

    ssl_split = make_ssl_split(train_dataset, ssl_config)
    original_valid_dataset = bundle.valid_dataset
    if val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        labeled_positions = None if ssl_split is None else ssl_split.labeled_positions
        unlabeled_positions = None if ssl_split is None else ssl_split.unlabeled_positions
        if cv_k > 1:
            bundle, labeled_positions, unlabeled_positions = utils.apply_apportioned_cross_validation_split(
                dataset_bundle=bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                include_unlabeled=mode == "ssl" and ssl_config is not None and ssl_config.enabled,
                cv_k=cv_k,
                cv_fold=cv_fold,
                cv_mode=cv_mode,
                seed=seed,
            )
            split_kind = "apportioned_cross_validation"
        else:
            bundle, labeled_positions, unlabeled_positions = utils.apply_post_apportion_validation_split(
                dataset_bundle=bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                seed=seed,
            )
        if ssl_split is not None:
            ssl_split = semi_supervised.SemiSupervisedSplit(
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
            )
    else:
        target_train_size, target_train_num_classes = target_train_stats(train_dataset, ssl_split)
        bundle = utils.apply_validation_mode(
            dataset_bundle=bundle,
            val_mode=val_mode,
            target_train_size=target_train_size,
            target_train_num_classes=target_train_num_classes,
            seed=seed,
        )

    scenario_id = make_scenario_id(
        dataset_name=dataset_name,
        seed=seed,
        cv_k=cv_k,
        cv_fold=cv_fold,
        cv_mode=cv_mode,
        val_mode=val_mode,
        ssl_config=ssl_config,
        mode=mode,
        dataset_protocol=dataset_protocol,
    )
    common = {
        "dataset": dataset_name,
        "scenario_id": scenario_id,
        "seed": seed,
        "split_kind": split_kind,
        "cv_k": cv_k,
        "cv_fold": cv_fold,
        "cv_mode": cv_mode if cv_k > 1 else "",
        "val_mode": val_mode,
        "mode": mode,
        "dataset_protocol": dataset_protocol,
        "ssl_config_path": "" if ssl_config_path is None else str(ssl_config_path),
    }
    if ssl_config is not None:
        common.update(ssl_config_metadata(ssl_config))

    split_specs = make_split_specs(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        train_dataset=bundle.train_dataset,
        valid_dataset=bundle.valid_dataset,
        original_valid_dataset=original_valid_dataset,
        ssl_split=ssl_split,
        val_mode=val_mode,
        mode=mode,
    )

    class_frames = []
    summary_rows = []
    sample_frames = []
    class_sets: dict[str, set[int]] = {}

    for spec in split_specs:
        distribution = make_distribution_frame(spec, common, class_metadata)
        class_frames.append(distribution)
        summary_rows.append(make_summary_row(spec, common, distribution))
        sample_frames.append(make_sample_membership_frame(spec, common, class_metadata))
        class_sets[spec["split"]] = set(int(label) for label in spec["labels"])

    overlap_rows = make_overlap_rows(common, class_sets)
    metadata = {
        **common,
        "dataset_protocol_info": protocol_info,
        "num_source_train_samples": int(len(train_val_dataset)),
        "num_test_samples": int(len(test_dataset)),
        "num_train_samples": int(len(train_dataset)),
        "num_valid_samples": int(len(bundle.valid_dataset)),
        "num_train_classes": int(len(set(int(label) for label in original_labels(train_dataset)))),
        "num_valid_classes": int(len(set(int(label) for label in original_labels(bundle.valid_dataset)))),
        "num_test_classes": int(len(set(int(label) for label in original_labels(test_dataset)))),
    }
    if ssl_split is not None:
        metadata.update(
            {
                "num_labeled_samples": int(len(ssl_split.labeled_positions)),
                "num_unlabeled_samples": int(len(ssl_split.unlabeled_positions)),
            }
        )

    return {
        "metadata": metadata,
        "class_frames": class_frames,
        "summary_rows": summary_rows,
        "sample_frames": sample_frames,
        "overlap_rows": overlap_rows,
    }


def load_datasets(
    dataset_name: str,
    download: bool,
    dataset_protocol: str = utils.DATASET_PROTOCOL_OFFICIAL,
):
    data_root = Path("data") / dataset_name
    if not download and not utils.is_dataset_ready(dataset_name, data_root):
        raise SystemExit(
            f"{dataset_name} is not ready under {data_root}. "
            "Run with --download or prepare the dataset metadata first."
        )

    train_val_dataset, test_dataset, protocol_info = utils.load_dataset_protocol_sources(
        dataset_name=dataset_name,
        data_root=data_root,
        train_transform=None,
        test_transform=None,
        dataset_protocol=dataset_protocol,
        download=download,
    )
    require_labels(train_val_dataset, f"{dataset_name} train")
    require_labels(test_dataset, f"{dataset_name} test")
    return train_val_dataset, test_dataset, protocol_info


def require_labels(dataset: object, name: str) -> None:
    if not hasattr(dataset, "labels") and not hasattr(dataset, "targets"):
        raise SystemExit(f"{name} has no labels/targets attribute; cannot compute class distributions.")


def load_class_metadata(dataset_name: str, train_val_dataset: object, test_dataset: object) -> dict[int, dict[str, Any]]:
    if dataset_name == "CIFAR10":
        return load_cifar10_metadata(train_val_dataset)
    if dataset_name == "Cars196":
        return load_cars196_metadata(train_val_dataset, test_dataset)
    if dataset_name == "StanfordOnlineProducts":
        return load_sop_metadata(train_val_dataset, test_dataset)
    return {}


def load_cifar10_metadata(dataset: object) -> dict[int, dict[str, Any]]:
    current_dataset = dataset
    while hasattr(current_dataset, "dataset"):
        current_dataset = current_dataset.dataset
    classes = getattr(current_dataset, "classes", None)
    if not classes:
        return {}
    return {int(index): {"class_name": str(name)} for index, name in enumerate(classes)}


def load_cars196_metadata(train_val_dataset: object, test_dataset: object) -> dict[int, dict[str, Any]]:
    names_file = Path("data") / "Cars196" / "names.csv"
    if not names_file.exists():
        return {}

    names = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_labels = np.concatenate([original_labels(train_val_dataset), original_labels(test_dataset)]).astype(int)
    label_min = int(all_labels.min())
    label_max = int(all_labels.max())

    if label_min == 0 and label_max == len(names) - 1:
        offset = 0
    elif label_min == 1 and label_max == len(names):
        offset = 1
    else:
        offset = label_min

    return {int(index + offset): {"class_name": name} for index, name in enumerate(names)}


def load_sop_metadata(train_val_dataset: object, test_dataset: object) -> dict[int, dict[str, Any]]:
    sop_root = Path("data") / "StanfordOnlineProducts" / "Stanford_Online_Products"
    rows_by_file_class_id: dict[int, dict[str, Any]] = {}
    for filename in ("Ebay_train.txt", "Ebay_test.txt"):
        path = sop_root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            columns = line.split()
            if len(columns) < 4:
                continue
            class_id = int(columns[1])
            super_class_id = int(columns[2])
            super_class_name = Path(columns[3]).parts[0].replace("_final", "")
            rows_by_file_class_id[class_id] = {
                "class_name": f"{super_class_name} product {class_id}",
                "file_class_id": class_id,
                "super_class_id": super_class_id,
                "super_class_name": super_class_name,
            }

    if not rows_by_file_class_id:
        return {}

    dataset_labels = np.concatenate([original_labels(train_val_dataset), original_labels(test_dataset)]).astype(int)
    unique_dataset_labels = set(int(label) for label in dataset_labels)
    file_class_ids = set(rows_by_file_class_id)
    if unique_dataset_labels <= file_class_ids:
        return rows_by_file_class_id
    if {label + 1 for label in unique_dataset_labels} <= file_class_ids:
        return {
            int(label): {
                **rows_by_file_class_id[int(label) + 1],
                "file_class_id": int(label) + 1,
            }
            for label in unique_dataset_labels
        }
    return rows_by_file_class_id


def make_ssl_split(train_dataset: object, ssl_config: semi_supervised.SemiSupervisedConfig | None):
    if ssl_config is None:
        return None

    return semi_supervised.make_semi_supervised_split(
        labels=np.asarray(train_dataset.labels, dtype=np.int64),
        label_sampling_mode=ssl_config.label_sampling_mode,
        labeled_fraction=ssl_config.labeled_fraction,
        labeled_per_class=ssl_config.labeled_per_class,
        max_unlabeled_samples=ssl_config.max_unlabeled_samples,
        seed=ssl_config.seed,
    )


def target_train_stats(train_dataset: object, ssl_split) -> tuple[int, int]:
    if ssl_split is None:
        labels = np.asarray(train_dataset.labels, dtype=np.int64)
        return len(train_dataset), int(len(np.unique(labels)))

    labels = np.asarray(train_dataset.labels, dtype=np.int64)
    labeled_labels = labels[np.asarray(ssl_split.labeled_positions, dtype=np.int64)]
    return int(len(ssl_split.labeled_positions)), int(len(np.unique(labeled_labels)))


def ssl_config_metadata(ssl_config: semi_supervised.SemiSupervisedConfig) -> dict[str, Any]:
    return {
        "ssl_method": ssl_config.method,
        "ssl_update_mode": ssl_config.update_mode,
        "ssl_label_sampling_mode": ssl_config.label_sampling_mode,
        "ssl_labeled_fraction": ssl_config.labeled_fraction,
        "ssl_labeled_per_class": ssl_config.labeled_per_class,
        "ssl_seed": ssl_config.seed,
        "ssl_max_unlabeled_samples": ssl_config.max_unlabeled_samples,
    }


def make_scenario_id(
    dataset_name: str,
    seed: int,
    cv_k: int,
    cv_fold: int | None,
    cv_mode: str,
    val_mode: str,
    ssl_config: semi_supervised.SemiSupervisedConfig | None,
    mode: str,
    dataset_protocol: str = utils.DATASET_PROTOCOL_OFFICIAL,
) -> str:
    parts = [dataset_name, f"protocol_{dataset_protocol}", f"seed_{seed}"]
    if cv_k > 1:
        parts.extend([f"cv_{cv_mode}", f"k_{cv_k}", f"fold_{cv_fold}"])
    else:
        parts.append("holdout")
    parts.append(f"val_{val_mode}")
    if ssl_config is not None:
        parts.extend(
            [
                mode,
                ssl_config.label_sampling_mode,
                f"frac_{format_float_token(ssl_config.labeled_fraction)}",
            ]
        )
        if ssl_config.labeled_per_class is not None:
            parts.append(f"kshot_{ssl_config.labeled_per_class}")
        parts.append(f"sslseed_{ssl_config.seed}")
    return slugify("_".join(str(part) for part in parts))


def format_float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def make_split_specs(
    train_val_dataset: object,
    test_dataset: object,
    train_dataset: object,
    valid_dataset: object,
    original_valid_dataset: object,
    ssl_split,
    val_mode: str,
    mode: str,
) -> list[dict[str, Any]]:
    specs = [
        dataset_spec("source_train", train_val_dataset, source_split="train"),
        dataset_spec("train", train_dataset, source_split="train"),
    ]
    if val_mode == utils.VAL_MODE_MATCH_TRAIN:
        specs.append(dataset_spec("valid_before_match_train", original_valid_dataset, source_split="train"))
    specs.extend(
        [
            dataset_spec("valid", valid_dataset, source_split="train"),
            dataset_spec("test", test_dataset, source_split="test"),
        ]
    )

    if ssl_split is not None:
        labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)
        ssl_pool_positions = np.asarray(sorted(np.concatenate([labeled_positions, unlabeled_positions])), dtype=np.int64)
        unused_positions = np.setdiff1d(np.arange(len(train_dataset), dtype=np.int64), ssl_pool_positions)

        specs.extend(
            [
                positioned_train_spec("train_labeled", train_dataset, labeled_positions),
                positioned_train_spec("train_unlabeled_true", train_dataset, unlabeled_positions),
                positioned_train_spec("train_ssl_candidate_pool_true", train_dataset, ssl_pool_positions),
            ]
        )
        if len(unused_positions) > 0:
            specs.append(positioned_train_spec("train_unused_by_ssl", train_dataset, unused_positions))
        if mode == "supervised":
            specs.append(positioned_train_spec("effective_train_supervised", train_dataset, labeled_positions))
        else:
            specs.append(positioned_train_spec("effective_train_ssl_true_pool", train_dataset, ssl_pool_positions))

    return specs


def dataset_spec(split: str, dataset: object, source_split: str) -> dict[str, Any]:
    labels = original_labels(dataset)
    mapped = mapped_labels(dataset)
    source_indices = source_indices_for_dataset(dataset)
    return {
        "split": split,
        "source_split": source_split,
        "labels": labels,
        "mapped_labels": mapped,
        "source_indices": source_indices,
        "positions": np.arange(len(labels), dtype=np.int64),
    }


def positioned_train_spec(split: str, train_dataset: object, positions: np.ndarray) -> dict[str, Any]:
    all_labels = original_labels(train_dataset)
    all_mapped = mapped_labels(train_dataset)
    all_source_indices = source_indices_for_dataset(train_dataset)
    positions = np.asarray(positions, dtype=np.int64)
    return {
        "split": split,
        "source_split": "train",
        "labels": all_labels[positions],
        "mapped_labels": None if all_mapped is None else all_mapped[positions],
        "source_indices": all_source_indices[positions],
        "positions": positions,
    }


def original_labels(dataset: object) -> np.ndarray:
    if hasattr(dataset, "orig_labels"):
        return np.asarray(getattr(dataset, "orig_labels"), dtype=np.int64)

    indices = getattr(dataset, "indices", None)
    if indices is not None:
        parent_labels = base_labels(getattr(dataset, "dataset"))
        return parent_labels[np.asarray(indices, dtype=np.int64)]

    return base_labels(dataset)


def mapped_labels(dataset: object) -> np.ndarray | None:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        return None
    if len(labels) != len(dataset):
        return None
    return np.asarray(labels, dtype=np.int64)


def base_labels(dataset: object) -> np.ndarray:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        raise ValueError(f"Dataset {type(dataset).__name__} has no labels/targets.")
    return np.asarray(labels, dtype=np.int64)


def source_indices_for_dataset(dataset: object) -> np.ndarray:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return np.arange(len(dataset), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def make_distribution_frame(
    spec: dict[str, Any],
    common: dict[str, Any],
    class_metadata: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    labels = np.asarray(spec["labels"], dtype=np.int64)
    mapped = spec["mapped_labels"]
    if len(labels) == 0:
        return pd.DataFrame(columns=distribution_columns(class_metadata, common))

    frame = pd.DataFrame({"class_label": labels})
    distribution = (
        frame["class_label"]
        .value_counts(sort=False)
        .sort_index()
        .rename("sample_count")
        .reset_index()
    )
    distribution["percentage"] = distribution["sample_count"] / int(len(labels)) * 100.0

    if mapped is not None and len(mapped) == len(labels):
        mapped_frame = pd.DataFrame({"class_label": labels, "mapped_class_label": mapped})
        mapped_by_class = mapped_frame.drop_duplicates("class_label")
        distribution = distribution.merge(mapped_by_class, on="class_label", how="left")
    else:
        distribution["mapped_class_label"] = pd.NA

    distribution.insert(0, "source_split", spec["source_split"])
    distribution.insert(0, "split", spec["split"])
    for key, value in reversed(list(common.items())):
        distribution.insert(0, key, value)

    metadata_frame = metadata_frame_for_labels(class_metadata, distribution["class_label"])
    if not metadata_frame.empty:
        distribution = distribution.merge(metadata_frame, on="class_label", how="left")

    return distribution


def distribution_columns(class_metadata: dict[int, dict[str, Any]], common: dict[str, Any]) -> list[str]:
    metadata_keys = sorted({key for values in class_metadata.values() for key in values})
    return [
        *common.keys(),
        "split",
        "source_split",
        "class_label",
        "sample_count",
        "percentage",
        "mapped_class_label",
        *metadata_keys,
    ]


def metadata_frame_for_labels(class_metadata: dict[int, dict[str, Any]], labels: pd.Series) -> pd.DataFrame:
    rows = []
    for label in sorted(set(int(value) for value in labels.tolist())):
        row = {"class_label": label}
        row.update(class_metadata.get(label, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def make_summary_row(
    spec: dict[str, Any],
    common: dict[str, Any],
    distribution: pd.DataFrame,
) -> dict[str, Any]:
    counts = distribution["sample_count"].astype(int) if not distribution.empty else pd.Series(dtype=int)
    num_samples = int(counts.sum()) if len(counts) else 0
    num_classes = int(len(counts))
    row = {
        **common,
        "split": spec["split"],
        "source_split": spec["source_split"],
        "num_samples": num_samples,
        "num_classes": num_classes,
        "min_samples_per_class": int(counts.min()) if len(counts) else 0,
        "max_samples_per_class": int(counts.max()) if len(counts) else 0,
        "avg_samples_per_class": float(counts.mean()) if len(counts) else 0.0,
        "median_samples_per_class": float(counts.median()) if len(counts) else 0.0,
        "std_samples_per_class": float(counts.std(ddof=0)) if len(counts) else 0.0,
        "q25_samples_per_class": float(counts.quantile(0.25)) if len(counts) else 0.0,
        "q75_samples_per_class": float(counts.quantile(0.75)) if len(counts) else 0.0,
        "imbalance_ratio_max_to_min": float(counts.max() / counts.min()) if len(counts) and counts.min() else 0.0,
        "num_singleton_classes": int((counts == 1).sum()) if len(counts) else 0,
        "num_classes_lt_2": int((counts < 2).sum()) if len(counts) else 0,
        "num_classes_lt_4": int((counts < 4).sum()) if len(counts) else 0,
        "min_count_classes": "",
        "max_count_classes": "",
        "gini_samples_per_class": 0.0,
        "entropy_bits": 0.0,
        "normalized_entropy": 0.0,
    }

    if len(counts):
        min_count = int(counts.min())
        max_count = int(counts.max())
        row["min_count_classes"] = format_class_list(distribution.loc[counts == min_count, "class_label"].tolist())
        row["max_count_classes"] = format_class_list(distribution.loc[counts == max_count, "class_label"].tolist())
        row["gini_samples_per_class"] = gini(counts.to_numpy(dtype=float))
        row["entropy_bits"] = entropy_bits(counts.to_numpy(dtype=float))
        row["normalized_entropy"] = normalized_entropy(counts.to_numpy(dtype=float))
    return row


def format_class_list(values: list[Any], limit: int = 50) -> str:
    values = [str(int(value)) if pd.notna(value) else "" for value in values]
    if len(values) <= limit:
        return ";".join(values)
    return ";".join(values[:limit]) + f";...(+{len(values) - limit})"


def gini(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    sorted_values = np.sort(values)
    index = np.arange(1, len(values) + 1)
    return float((np.sum((2 * index - len(values) - 1) * sorted_values)) / (len(values) * np.sum(sorted_values)))


def entropy_bits(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def normalized_entropy(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 1.0 if len(values) == 1 else 0.0
    return float(entropy_bits(values) / math.log2(len(values)))


def make_sample_membership_frame(
    spec: dict[str, Any],
    common: dict[str, Any],
    class_metadata: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    labels = np.asarray(spec["labels"], dtype=np.int64)
    mapped = spec["mapped_labels"]
    frame = pd.DataFrame(
        {
            "split": spec["split"],
            "source_split": spec["source_split"],
            "position_in_split": np.arange(len(labels), dtype=np.int64),
            "position_in_train_subset": spec["positions"],
            "source_index": spec["source_indices"],
            "class_label": labels,
            "mapped_class_label": pd.NA if mapped is None else mapped,
        }
    )
    for key, value in reversed(list(common.items())):
        frame.insert(0, key, value)
    metadata_frame = metadata_frame_for_labels(class_metadata, frame["class_label"])
    if not metadata_frame.empty:
        frame = frame.merge(metadata_frame, on="class_label", how="left")
    return frame


def make_overlap_rows(common: dict[str, Any], class_sets: dict[str, set[int]]) -> list[dict[str, Any]]:
    rows = []
    split_names = sorted(class_sets)
    for split_a, split_b in itertools.combinations(split_names, 2):
        classes_a = class_sets[split_a]
        classes_b = class_sets[split_b]
        intersection = classes_a & classes_b
        union = classes_a | classes_b
        rows.append(
            {
                **common,
                "split_a": split_a,
                "split_b": split_b,
                "num_classes_a": len(classes_a),
                "num_classes_b": len(classes_b),
                "num_intersection_classes": len(intersection),
                "num_union_classes": len(union),
                "jaccard_classes": len(intersection) / len(union) if union else 0.0,
                "intersection_classes": format_class_list(sorted(intersection)),
            }
        )
    return rows


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    existing = [column for column in columns if column in frame.columns]
    return frame.sort_values(existing).reset_index(drop=True)


def make_distribution_pivot(class_distribution: pd.DataFrame) -> pd.DataFrame:
    if class_distribution.empty:
        return pd.DataFrame()
    pivot_source = class_distribution.copy()
    index_columns = [
        "dataset",
        "scenario_id",
        "class_label",
        "class_name",
        "super_class_id",
        "super_class_name",
    ]
    index_columns = [column for column in index_columns if column in pivot_source.columns]
    for column in index_columns:
        pivot_source[column] = pivot_source[column].fillna("")
    pivot = pivot_source.pivot_table(
        index=index_columns,
        columns="split",
        values="sample_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    return pivot


def write_tables(
    out_dir: Path,
    class_distribution: pd.DataFrame,
    split_summary: pd.DataFrame,
    sample_membership: pd.DataFrame,
    class_overlap: pd.DataFrame,
    pivot: pd.DataFrame,
) -> list[Path]:
    files = []
    table_specs = [
        ("class_distribution.csv", class_distribution),
        ("class_distribution_pivot.csv", pivot),
        ("split_summary.csv", split_summary),
        ("sample_membership.csv", sample_membership),
        ("class_overlap.csv", class_overlap),
    ]
    for filename, frame in table_specs:
        path = out_dir / filename
        frame.to_csv(path, index=False)
        files.append(path)

    if not class_distribution.empty:
        for (dataset, scenario_id, split), frame in class_distribution.groupby(["dataset", "scenario_id", "split"]):
            split_dir = out_dir / "per_split"
            split_dir.mkdir(parents=True, exist_ok=True)
            path = split_dir / f"{slugify(dataset)}__{slugify(scenario_id)}__{slugify(split)}.csv"
            frame.to_csv(path, index=False)
            files.append(path)
    return files


def write_charts(
    charts_dir: Path,
    class_distribution: pd.DataFrame,
    split_summary: pd.DataFrame,
    top_n: int,
    dpi: int,
    max_x_labels: int,
) -> list[Path]:
    if class_distribution.empty or split_summary.empty:
        return []

    files: list[Path] = []
    for (dataset, scenario_id), scenario_summary in split_summary.groupby(["dataset", "scenario_id"]):
        scenario_classes = class_distribution[
            (class_distribution["dataset"] == dataset) & (class_distribution["scenario_id"] == scenario_id)
        ]
        prefix = charts_dir / f"{slugify(dataset)}__{slugify(scenario_id)}"
        files.append(write_split_size_chart(prefix, scenario_summary, dpi=dpi))
        files.append(write_boxplot_chart(prefix, scenario_classes, dpi=dpi))

        for split, split_distribution in scenario_classes.groupby("split"):
            split_prefix = charts_dir / f"{slugify(dataset)}__{slugify(scenario_id)}__{slugify(split)}"
            files.append(
                write_class_count_chart(
                    split_prefix,
                    dataset,
                    scenario_id,
                    split,
                    split_distribution,
                    dpi=dpi,
                    max_x_labels=max_x_labels,
                )
            )
            files.append(write_histogram_chart(split_prefix, dataset, scenario_id, split, split_distribution, dpi=dpi))
            files.append(
                write_top_bottom_chart(
                    split_prefix,
                    dataset,
                    scenario_id,
                    split,
                    split_distribution,
                    top_n=top_n,
                    dpi=dpi,
                )
            )
    return [path for path in files if path is not None]


def write_split_size_chart(prefix: Path, summary: pd.DataFrame, dpi: int) -> Path:
    path = prefix.with_name(prefix.name + "__split_sizes.png")
    summary = summary.sort_values("split")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(summary["split"], summary["num_samples"], color="#3b6ea8")
    axes[0].set_title("Samples per split")
    axes[0].set_ylabel("samples")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(summary["split"], summary["num_classes"], color="#b65742")
    axes[1].set_title("Classes per split")
    axes[1].set_ylabel("classes")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle(summary["scenario_id"].iloc[0])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_boxplot_chart(prefix: Path, class_distribution: pd.DataFrame, dpi: int) -> Path:
    path = prefix.with_name(prefix.name + "__samples_per_class_boxplot.png")
    grouped = [(split, frame["sample_count"].astype(int).to_numpy()) for split, frame in class_distribution.groupby("split")]
    grouped = [(split, values) for split, values in grouped if len(values)]
    if not grouped:
        return path

    fig, ax = plt.subplots(figsize=(max(10, len(grouped) * 1.25), 6))
    ax.boxplot([values for _, values in grouped], tick_labels=[split for split, _ in grouped], showmeans=True)
    ax.set_title("Samples per class distribution")
    ax.set_ylabel("samples per class")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_class_count_chart(
    prefix: Path,
    dataset: str,
    scenario_id: str,
    split: str,
    distribution: pd.DataFrame,
    dpi: int,
    max_x_labels: int,
) -> Path:
    path = prefix.with_name(prefix.name + "__class_counts.png")
    counts = distribution.sort_values(["sample_count", "class_label"], ascending=[False, True]).reset_index(drop=True)
    num_classes = len(counts)
    width = min(28, max(10, num_classes * 0.08))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.bar(np.arange(num_classes), counts["sample_count"], color="#315c48")
    ax.axhline(counts["sample_count"].mean(), color="#c95830", linestyle="--", linewidth=1.25, label="average")
    ax.set_title(f"{dataset} | {split} | {scenario_id}")
    ax.set_xlabel("classes sorted by sample count")
    ax.set_ylabel("samples")
    if num_classes <= max_x_labels:
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels(counts["class_label"].astype(str), rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_histogram_chart(
    prefix: Path,
    dataset: str,
    scenario_id: str,
    split: str,
    distribution: pd.DataFrame,
    dpi: int,
) -> Path:
    path = prefix.with_name(prefix.name + "__samples_per_class_hist.png")
    counts = distribution["sample_count"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(50, max(1, int(counts.nunique())))
    ax.hist(counts, bins=bins, color="#6f7f3f", edgecolor="white")
    ax.set_title(f"Samples per class histogram | {dataset} | {split}")
    ax.set_xlabel("samples per class")
    ax.set_ylabel("number of classes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_top_bottom_chart(
    prefix: Path,
    dataset: str,
    scenario_id: str,
    split: str,
    distribution: pd.DataFrame,
    top_n: int,
    dpi: int,
) -> Path:
    path = prefix.with_name(prefix.name + "__smallest_largest_classes.png")
    if distribution.empty:
        return path
    smallest = distribution.nsmallest(top_n, ["sample_count", "class_label"]).assign(group="smallest")
    largest = distribution.nlargest(top_n, "sample_count").sort_values("sample_count").assign(group="largest")
    detail = pd.concat([smallest, largest], ignore_index=True).drop_duplicates(["group", "class_label"])
    labels = detail["group"] + " | " + detail["class_label"].astype(str)

    fig, ax = plt.subplots(figsize=(12, max(5, len(detail) * 0.24)))
    colors = detail["group"].map({"smallest": "#b65742", "largest": "#3b6ea8"}).fillna("#777777")
    ax.barh(labels, detail["sample_count"], color=colors)
    ax.set_title(f"Smallest/largest classes | {dataset} | {split}")
    ax.set_xlabel("samples")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_markdown_report(
    path: Path,
    split_summary: pd.DataFrame,
    class_overlap: pd.DataFrame,
    files: list[Path],
    metadata: dict[str, Any],
) -> None:
    summary_columns = [column for column in SUMMARY_COLUMNS if column in split_summary.columns]
    report_lines = [
        "# Class Distribution Report",
        "",
        "Generated by `scripts/class_distribution_report.py`.",
        "",
        "## Parameter",
        "",
        markdown_table(pd.DataFrame([metadata["parameters"]])),
        "",
        "## Tabelle: Moeglichkeiten Klassendistributionen",
        "",
        "Diese Tabelle enthaelt die geforderten min/max/avg Samples pro Klasse je Split.",
        "",
        markdown_table(split_summary[summary_columns] if summary_columns else split_summary),
        "",
        "## Class Overlap",
        "",
        markdown_table(
            class_overlap[
                [
                    column
                    for column in [
                        "dataset",
                        "scenario_id",
                        "split_a",
                        "split_b",
                        "num_intersection_classes",
                        "jaccard_classes",
                    ]
                    if column in class_overlap.columns
                ]
            ].head(200)
            if not class_overlap.empty
            else pd.DataFrame()
        ),
        "",
        "## Files",
        "",
    ]
    report_lines.extend(f"- `{file_path}`" for file_path in files)
    report_lines.append("")
    path.write_text("\n".join(report_lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(format_markdown_cell)
    columns = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in rendered.columns) + " |")
    return "\n".join(lines)


def format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def slugify(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


if __name__ == "__main__":
    raise SystemExit(main())
