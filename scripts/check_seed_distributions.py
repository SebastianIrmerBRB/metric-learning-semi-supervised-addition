from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402
from training import semi_supervised  # noqa: E402


DATASETS = (
    "Cars196",
    "StanfordOnlineProducts",
    "DeepFashionInShop",
    "FGVCAircraft",
)
SUMMARY_COLUMNS = (
    "dataset",
    "dataset_protocol",
    "seed",
    "split_seed",
    "label_seed",
    "cv_k",
    "cv_fold",
    "cv_mode",
    "val_mode",
    "split",
    "source_index_hash",
    "num_samples",
    "fraction_of_train_valid",
    "num_classes",
    "min_samples_per_class",
    "max_samples_per_class",
    "mean_samples_per_class",
    "median_samples_per_class",
    "std_samples_per_class",
    "singleton_classes",
    "train_valid_overlap_classes",
    "train_valid_overlap_labels",
)
COUNT_COLUMNS = (
    "dataset",
    "dataset_protocol",
    "seed",
    "split_seed",
    "label_seed",
    "cv_k",
    "cv_fold",
    "cv_mode",
    "val_mode",
    "split",
    "class_label",
    "sample_count",
    "percentage",
)
IDENTITY_COLUMNS = (
    "split",
    "left_scenario",
    "right_scenario",
    "same",
    "jaccard",
    "intersection_size",
    "left_size",
    "right_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare train/validation class sample distributions across split seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=utils.normalize_dataset_name, choices=DATASETS, default="Cars196")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Label/run seeds to inspect.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Fixed seed for dataset train/validation splitting. Omit to let each --seeds value drive the split.",
    )
    parser.add_argument(
        "--seed-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Add every seed in the inclusive range START..END.",
    )
    parser.add_argument("--cv-k", type=int, default=1, help="Use 1 for the normal holdout split.")
    parser.add_argument(
        "--cv-fold",
        type=int,
        nargs="*",
        default=None,
        help="Fold(s) to inspect when --cv-k > 1. Defaults to all folds.",
    )
    parser.add_argument("--cv-mode", choices=utils.CV_MODES, default="group_kfold")
    parser.add_argument("--val-mode", choices=utils.VAL_MODES, default=utils.VAL_MODE_ALL)
    parser.add_argument("--dataset-protocol", choices=utils.DATASET_PROTOCOLS, default=utils.DATASET_PROTOCOL_OFFICIAL)
    parser.add_argument("--ssl-config", type=Path, default=None, help="Optional SSL config to reuse label-budget fields.")
    parser.add_argument("--label-sampling-mode", choices=sorted(semi_supervised.LABEL_SAMPLING_MODES), default=None)
    parser.add_argument("--labeled-fraction", type=float, default=None)
    parser.add_argument("--labeled-per-class", type=int, default=None)
    parser.add_argument("--max-unlabeled-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true", help="Allow dataset download/preparation if missing.")
    parser.add_argument("--include-test", action="store_true", help="Also summarize the fixed test split.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional CSV path for split summaries.")
    parser.add_argument("--counts-csv", type=Path, default=None, help="Optional CSV path for per-class counts.")
    parser.add_argument("--identity-csv", type=Path, default=None, help="Optional CSV path for pairwise identity checks.")
    parser.add_argument(
        "--preview-classes",
        type=int,
        default=0,
        help="Print this many smallest and largest classes for each split.",
    )
    parser.add_argument("--verbose", action="store_true", help="Keep project INFO old_logs enabled.")
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    seeds = list(args.seeds)
    if args.seed_range is not None:
        start, end = args.seed_range
        step = 1 if end >= start else -1
        seeds.extend(range(start, end + step, step))
    return sorted(set(int(seed) for seed in seeds))


def resolve_folds(cv_k: int, raw_folds: list[int] | None) -> list[int | None]:
    if cv_k <= 0:
        raise SystemExit("--cv-k must be positive")
    if cv_k == 1:
        if raw_folds:
            raise SystemExit("--cv-fold is only valid when --cv-k > 1")
        return [None]
    if raw_folds is None or len(raw_folds) == 0:
        return list(range(cv_k))
    invalid = [fold for fold in raw_folds if fold < 0 or fold >= cv_k]
    if invalid:
        raise SystemExit(f"--cv-fold values must be in [0, {cv_k - 1}], got {invalid}")
    return sorted(set(raw_folds))


def load_sources(args: argparse.Namespace, seed: int):
    data_root = Path("data") / args.dataset
    if not args.download and not utils.is_dataset_ready(args.dataset, data_root):
        raise SystemExit(
            f"{args.dataset} is not ready under {data_root}. "
            "Run again with --download or prepare the dataset first."
        )
    return utils.load_dataset_protocol_sources(
        dataset_name=args.dataset,
        data_root=data_root,
        train_transform=None,
        test_transform=None,
        dataset_protocol=args.dataset_protocol,
        download=args.download,
    )


def make_bundle(args: argparse.Namespace, train_val_dataset, test_dataset, split_seed: int, fold: int | None):
    if args.val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        train_indices = np.arange(len(train_val_dataset), dtype=np.int64).tolist()
        train_dataset, valid_dataset, train_labels_mapper = utils.make_train_valid_subsets(
            train_val_dataset,
            train_indices,
            [],
        )
        bundle = utils.DatasetBundle(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            train_labels_mapper=train_labels_mapper,
            split_info=None,
        )
        if args.cv_k > 1:
            bundle, _, _ = utils.apply_apportioned_cross_validation_split(
                dataset_bundle=bundle,
                cv_k=args.cv_k,
                cv_fold=fold,
                cv_mode=args.cv_mode,
                seed=split_seed,
            )
        else:
            bundle, _, _ = utils.apply_post_apportion_validation_split(dataset_bundle=bundle, seed=split_seed)
        return bundle

    if args.cv_k > 1:
        train_dataset, valid_dataset, train_labels_mapper = utils.split_dataset_cross_validation(
            train_val_dataset,
            cv_k=args.cv_k,
            cv_fold=fold,
            cv_mode=args.cv_mode,
            seed=split_seed,
        )
    else:
        train_dataset, valid_dataset, train_labels_mapper = utils.split_dataset_by_classes(
            train_val_dataset,
            seed=split_seed,
        )

    bundle = utils.DatasetBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_labels_mapper=train_labels_mapper,
        split_info=None,
    )
    if args.val_mode == utils.VAL_MODE_MATCH_TRAIN:
        bundle = utils.apply_validation_mode(
            dataset_bundle=bundle,
            val_mode=args.val_mode,
            target_train_size=len(bundle.train_dataset),
            target_train_num_classes=count_classes(bundle.train_dataset),
            seed=split_seed,
        )
    return bundle


def load_label_config(args: argparse.Namespace) -> dict[str, object]:
    raw_config = {}
    if args.ssl_config is not None:
        raw_config = json.loads(args.ssl_config.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise SystemExit(f"SSL config must be a JSON object: {args.ssl_config}")

    return {
        "label_sampling_mode": args.label_sampling_mode or raw_config.get("label_sampling_mode", "global_budget"),
        "labeled_fraction": args.labeled_fraction
        if args.labeled_fraction is not None
        else raw_config.get("labeled_fraction", 1.0),
        "labeled_per_class": args.labeled_per_class
        if args.labeled_per_class is not None
        else raw_config.get("labeled_per_class"),
        "max_unlabeled_samples": args.max_unlabeled_samples
        if args.max_unlabeled_samples is not None
        else raw_config.get("max_unlabeled_samples"),
    }


def make_label_split(train_dataset, label_config: dict[str, object], label_seed: int):
    return semi_supervised.make_semi_supervised_split(
        labels=train_dataset.labels,
        label_sampling_mode=str(label_config["label_sampling_mode"]),
        labeled_fraction=float(label_config["labeled_fraction"]),
        labeled_per_class=label_config["labeled_per_class"],
        max_unlabeled_samples=label_config["max_unlabeled_samples"],
        seed=label_seed,
    )


def dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "orig_labels") and len(dataset.orig_labels) == len(dataset):
        return np.asarray(dataset.orig_labels, dtype=np.int64)
    if hasattr(dataset, "labels") and len(dataset.labels) == len(dataset):
        return np.asarray(dataset.labels, dtype=np.int64)
    if hasattr(dataset, "targets") and len(dataset.targets) == len(dataset):
        return np.asarray(dataset.targets, dtype=np.int64)
    if isinstance(dataset, Subset):
        base_labels = dataset_labels(dataset.dataset)
        return base_labels[np.asarray(dataset.indices, dtype=np.int64)]
    raise ValueError(f"Could not find labels for dataset object {type(dataset).__name__}")


def dataset_source_indices(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return np.asarray(dataset.indices, dtype=np.int64)
    return np.arange(len(dataset), dtype=np.int64)


def positions_to_source_indices(dataset, positions: np.ndarray) -> np.ndarray:
    return dataset_source_indices(dataset)[np.asarray(positions, dtype=np.int64)]


def count_classes(dataset) -> int:
    labels = dataset_labels(dataset)
    return int(len(np.unique(labels))) if len(labels) else 0


def fingerprint_indices(indices: np.ndarray) -> str:
    unique_sorted = np.asarray(sorted(set(int(index) for index in indices)), dtype=np.int64)
    return hashlib.blake2b(unique_sorted.tobytes(), digest_size=8).hexdigest()


def summarize_split(
    *,
    args: argparse.Namespace,
    seed: int,
    split_seed: int,
    label_seed: int,
    fold: int | None,
    split_name: str,
    labels: np.ndarray,
    source_indices: np.ndarray,
    train_valid_total: int,
    overlap_labels: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    counts = Counter(int(label) for label in labels)
    count_values = np.asarray(list(counts.values()), dtype=np.float64)
    num_samples = int(len(labels))
    summary = {
        "dataset": args.dataset,
        "dataset_protocol": args.dataset_protocol,
        "seed": seed,
        "split_seed": split_seed,
        "label_seed": label_seed,
        "cv_k": args.cv_k,
        "cv_fold": "" if fold is None else fold,
        "cv_mode": "" if args.cv_k == 1 else args.cv_mode,
        "val_mode": args.val_mode,
        "split": split_name,
        "source_index_hash": fingerprint_indices(source_indices),
        "num_samples": num_samples,
        "fraction_of_train_valid": round(num_samples / train_valid_total, 6) if train_valid_total else 0.0,
        "num_classes": int(len(counts)),
        "min_samples_per_class": int(count_values.min()) if len(count_values) else 0,
        "max_samples_per_class": int(count_values.max()) if len(count_values) else 0,
        "mean_samples_per_class": round(float(count_values.mean()), 6) if len(count_values) else 0.0,
        "median_samples_per_class": round(float(np.median(count_values)), 6) if len(count_values) else 0.0,
        "std_samples_per_class": round(float(count_values.std(ddof=0)), 6) if len(count_values) else 0.0,
        "singleton_classes": int(np.sum(count_values == 1)) if len(count_values) else 0,
        "train_valid_overlap_classes": len(overlap_labels),
        "train_valid_overlap_labels": " ".join(str(label) for label in overlap_labels),
    }
    count_rows = [
        {
            "dataset": args.dataset,
            "dataset_protocol": args.dataset_protocol,
            "seed": seed,
            "split_seed": split_seed,
            "label_seed": label_seed,
            "cv_k": args.cv_k,
            "cv_fold": "" if fold is None else fold,
            "cv_mode": "" if args.cv_k == 1 else args.cv_mode,
            "val_mode": args.val_mode,
            "split": split_name,
            "class_label": label,
            "sample_count": count,
            "percentage": round(count / num_samples * 100.0, 6) if num_samples else 0.0,
        }
        for label, count in sorted(counts.items())
    ]
    return summary, count_rows


def preview_counts(split_name: str, labels: np.ndarray, limit: int, seed: int, fold: int | None) -> None:
    if limit <= 0 or len(labels) == 0:
        return
    counts = sorted(Counter(int(label) for label in labels).items(), key=lambda item: (item[1], item[0]))
    smallest = counts[:limit]
    largest = list(reversed(counts[-limit:]))
    fold_text = "" if fold is None else f" fold={fold}"
    print(f"    seed={seed}{fold_text} {split_name} smallest: {format_count_preview(smallest)}")
    print(f"    seed={seed}{fold_text} {split_name} largest:  {format_count_preview(largest)}")


def format_count_preview(items: Iterable[tuple[int, int]]) -> str:
    return ", ".join(f"{label}:{count}" for label, count in items)


def write_csv(path: Path, rows: list[dict[str, object]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def scenario_name(row: dict[str, object]) -> str:
    fold = row["cv_fold"]
    fold_text = "" if fold == "" else f",fold={fold}"
    return f"seed={row['seed']},split_seed={row['split_seed']},label_seed={row['label_seed']}{fold_text}"


def make_identity_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for split in ("valid", "train_labeled"):
        split_records = [record for record in records if record["split"] == split]
        for left_index, left in enumerate(split_records):
            for right in split_records[left_index + 1 :]:
                left_indices = left["indices"]
                right_indices = right["indices"]
                intersection_size = len(left_indices & right_indices)
                union_size = len(left_indices | right_indices)
                rows.append(
                    {
                        "split": split,
                        "left_scenario": left["scenario"],
                        "right_scenario": right["scenario"],
                        "same": left_indices == right_indices,
                        "jaccard": round(intersection_size / union_size, 6) if union_size else 1.0,
                        "intersection_size": intersection_size,
                        "left_size": len(left_indices),
                        "right_size": len(right_indices),
                    }
                )
    return rows


def print_identity_report(records: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    print("\nIdentity checks")
    for split in ("valid", "train_labeled"):
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        unique_hashes = {record["hash"] for record in split_records}
        split_rows = [row for row in rows if row["split"] == split]
        print(f"  {split}: {len(unique_hashes)} unique subset(s) across {len(split_records)} scenario(s)")
        if split_rows:
            jaccards = np.asarray([float(row["jaccard"]) for row in split_rows], dtype=np.float64)
            print(
                "    pairwise Jaccard "
                f"min={jaccards.min():.6f}, mean={jaccards.mean():.6f}, max={jaccards.max():.6f}"
            )
            if len(split_rows) <= 15:
                for row in split_rows:
                    print(
                        f"    {row['left_scenario']} <-> {row['right_scenario']}: "
                        f"same={row['same']}, jaccard={row['jaccard']}"
                    )


def print_summary_table(rows: list[dict[str, object]]) -> None:
    headers = (
        "seed",
        "split_seed",
        "fold",
        "split",
        "hash",
        "samples",
        "frac",
        "classes",
        "min",
        "max",
        "mean",
        "std",
        "overlap",
    )
    print(
        f"{headers[0]:>6} {headers[1]:>10} {headers[2]:>4} {headers[3]:<13} "
        f"{headers[4]:>16} {headers[5]:>8} {headers[6]:>7} {headers[7]:>7} "
        f"{headers[8]:>5} {headers[9]:>5} {headers[10]:>9} {headers[11]:>9} {headers[12]:>7}"
    )
    for row in rows:
        print(
            f"{row['seed']:>6} {row['split_seed']:>10} {str(row['cv_fold']):>4} {row['split']:<13} "
            f"{row['source_index_hash']:>16} {row['num_samples']:>8} "
            f"{row['fraction_of_train_valid']:>7} {row['num_classes']:>7} "
            f"{row['min_samples_per_class']:>5} {row['max_samples_per_class']:>5} "
            f"{row['mean_samples_per_class']:>9} {row['std_samples_per_class']:>9} "
            f"{row['train_valid_overlap_classes']:>7}"
        )


def main() -> None:
    args = parse_args()
    if not args.verbose:
        logger.remove()
    seeds = resolve_seeds(args)
    folds = resolve_folds(args.cv_k, args.cv_fold)
    label_config = load_label_config(args)
    summary_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    identity_records: list[dict[str, object]] = []

    for seed in seeds:
        split_seed = seed if args.split_seed is None else args.split_seed
        label_seed = seed
        train_val_dataset, test_dataset, _ = load_sources(args, split_seed)
        for fold in folds:
            bundle = make_bundle(args, train_val_dataset, test_dataset, split_seed, fold)
            label_split = make_label_split(bundle.train_dataset, label_config, label_seed)
            split_labels = {
                "train": dataset_labels(bundle.train_dataset),
                "valid": dataset_labels(bundle.valid_dataset),
                "train_labeled": dataset_labels(bundle.train_dataset)[label_split.labeled_positions],
            }
            split_source_indices = {
                "train": dataset_source_indices(bundle.train_dataset),
                "valid": dataset_source_indices(bundle.valid_dataset),
                "train_labeled": positions_to_source_indices(bundle.train_dataset, label_split.labeled_positions),
            }
            if args.include_test:
                split_labels["test"] = dataset_labels(bundle.test_dataset)
                split_source_indices["test"] = dataset_source_indices(bundle.test_dataset)

            train_valid_total = len(split_labels["train"]) + len(split_labels["valid"])
            overlap_labels = sorted(set(split_labels["train"]) & set(split_labels["valid"]))
            for split_name, labels in split_labels.items():
                source_indices = split_source_indices[split_name]
                summary, counts = summarize_split(
                    args=args,
                    seed=seed,
                    split_seed=split_seed,
                    label_seed=label_seed,
                    fold=fold,
                    split_name=split_name,
                    labels=labels,
                    source_indices=source_indices,
                    train_valid_total=train_valid_total,
                    overlap_labels=overlap_labels,
                )
                summary_rows.append(summary)
                count_rows.extend(counts)
                if split_name in {"valid", "train_labeled"}:
                    identity_records.append(
                        {
                            "split": split_name,
                            "scenario": scenario_name(summary),
                            "hash": summary["source_index_hash"],
                            "indices": set(int(index) for index in source_indices),
                        }
                    )
                preview_counts(split_name, labels, args.preview_classes, seed, fold)

    print_summary_table(summary_rows)
    identity_rows = make_identity_rows(identity_records)
    print_identity_report(identity_records, identity_rows)
    if args.summary_csv is not None:
        write_csv(args.summary_csv, summary_rows, SUMMARY_COLUMNS)
        print(f"\nWrote summary CSV: {args.summary_csv}")
    if args.counts_csv is not None:
        write_csv(args.counts_csv, count_rows, COUNT_COLUMNS)
        print(f"Wrote per-class counts CSV: {args.counts_csv}")
    if args.identity_csv is not None:
        write_csv(args.identity_csv, identity_rows, IDENTITY_COLUMNS)
        print(f"Wrote identity CSV: {args.identity_csv}")


if __name__ == "__main__":
    main()
