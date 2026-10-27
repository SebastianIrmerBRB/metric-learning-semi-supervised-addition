from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ARRAYS = (
    "train_indices",
    "val_indices",
    "labeled_indices",
    "unlabeled_indices",
    "labeled_positions",
    "unlabeled_positions",
)

INDEX_ARRAYS = (
    "train_indices",
    "val_indices",
    "labeled_indices",
    "unlabeled_indices",
)

LABEL_COUNT_KEYS = {
    "labeled_label_counts",
    "unlabeled_label_counts",
    "label_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a saved split directory from old_logs. If SPLIT_DIR is omitted, "
            "the newest old_logs/**/split directory is used."
        )
    )
    parser.add_argument(
        "split_dir",
        nargs="?",
        type=Path,
        default=r"C:\Users\Sebastian\PycharmProjects\metric-learning\logs\cadas196\global_budget_label_0p1_ProxyAnchorLoss_no_miner_tune_seed_7\hpo\adada_supervised\trial_0000\2026-06-17_22-43-05",
        help="Path to a log split directory, or a log run directory containing split/.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of array values and mapped SOP rows to preview.",
    )
    parser.add_argument(
        "--arrays",
        nargs="*",
        default=list(DEFAULT_ARRAYS),
        help="Array basenames to read without the .npy suffix.",
    )
    parser.add_argument(
        "--show-label-counts",
        action="store_true",
        help="Preview the large per-label count dictionaries from the JSON files.",
    )
    parser.add_argument(
        "--map-sop",
        action="store_true",
        help="Map *_indices arrays back to Stanford Online Products class IDs and image paths.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/StanfordOnlineProducts/Stanford_Online_Products"),
        help="SOP directory containing Ebay_train.txt. Used with --map-sop or --csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for exporting mapped SOP rows from *_indices arrays.",
    )
    return parser.parse_args()


def find_latest_split_dir(log_root: Path = Path("old_logs")) -> Path:
    candidates = [path.parent for path in log_root.rglob("split_info.json")]
    if not candidates:
        raise SystemExit(f"No split_info.json files found under {log_root}")
    return max(candidates, key=lambda path: (path / "split_info.json").stat().st_mtime)


def resolve_split_dir(split_dir: Path | None) -> Path:
    if split_dir is None:
        return find_latest_split_dir()

    if (split_dir / "split").is_dir():
        split_dir = split_dir / "split"

    if not split_dir.is_dir():
        raise SystemExit(f"Split directory does not exist: {split_dir}")
    if not (split_dir / "split_info.json").exists():
        raise SystemExit(f"Expected split_info.json in: {split_dir}")
    return split_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_items_by_numeric_key(mapping: dict) -> list[tuple[str, object]]:
    def key_fn(item: tuple[str, object]) -> tuple[int, int | str]:
        key = item[0]
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    return sorted(mapping.items(), key=key_fn)


def print_json_file(path: Path, preview: int, show_label_counts: bool) -> None:
    if not path.exists():
        return

    data = load_json(path)
    print(f"\n{path.name}")
    for key, value in data.items():
        if key in LABEL_COUNT_KEYS and isinstance(value, dict):
            print(f"  {key}: {len(value)} classes")
            if show_label_counts:
                for label, count in sorted_items_by_numeric_key(value)[:preview]:
                    print(f"    {label}: {count}")
            continue

        if key == "ssl_config" and isinstance(value, dict):
            print("  ssl_config:")
            for config_key, config_value in value.items():
                print(f"    {config_key}: {config_value}")
            continue

        print(f"  {key}: {value}")


def print_array_file(path: Path, preview: int) -> None:
    if not path.exists():
        print(f"  {path.name}: missing")
        return

    values = np.load(path)
    preview_values = values[:preview].tolist()
    range_text = ""
    if values.size:
        range_text = f", min={int(values.min())}, max={int(values.max())}"
    print(
        f"  {path.stem}: shape={values.shape}, dtype={values.dtype}"
        f"{range_text}, preview={preview_values}"
    )


def read_sop_rows(dataset_root: Path) -> list[str]:
    train_file = dataset_root / "Ebay_train.txt"
    if not train_file.exists():
        raise SystemExit(f"Could not find SOP train file: {train_file}")
    return train_file.read_text(encoding="utf-8").splitlines()[1:]


def sop_record(rows: list[str], dataset_root: Path, index: int) -> dict[str, object]:
    if index < 0 or index >= len(rows):
        raise IndexError(f"SOP train index {index} is outside 0..{len(rows) - 1}")

    cols = rows[index].split()
    return {
        "index": index,
        "image_id": cols[0],
        "class_id": int(cols[1]),
        "super_class_id": int(cols[2]) if len(cols) > 2 else None,
        "path": str(dataset_root / cols[3]) if len(cols) > 3 else "",
    }


def print_sop_mapping(
    split_dir: Path,
    array_names: Iterable[str],
    dataset_root: Path,
    preview: int,
) -> None:
    rows = read_sop_rows(dataset_root)
    print("\nSOP mapping preview")
    for name in array_names:
        if name not in INDEX_ARRAYS:
            continue

        path = split_dir / f"{name}.npy"
        if not path.exists():
            continue

        indices = np.load(path)
        print(f"  {name}: {len(indices)} rows")
        for raw_index in indices[:preview]:
            record = sop_record(rows, dataset_root, int(raw_index))
            print(
                "    "
                f"{record['index']}: class={record['class_id']}, "
                f"image_id={record['image_id']}, path={record['path']}"
            )


def export_sop_csv(
    csv_path: Path,
    split_dir: Path,
    array_names: Iterable[str],
    dataset_root: Path,
) -> None:
    rows = read_sop_rows(dataset_root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "array",
                "index",
                "image_id",
                "class_id",
                "super_class_id",
                "path",
            ],
        )
        writer.writeheader()

        for name in array_names:
            if name not in INDEX_ARRAYS:
                continue

            path = split_dir / f"{name}.npy"
            if not path.exists():
                continue

            for raw_index in np.load(path):
                record = sop_record(rows, dataset_root, int(raw_index))
                record["array"] = name
                writer.writerow(record)

    print(f"\nWrote CSV: {csv_path}")


def main() -> int:
    args = parse_args()
    split_dir = resolve_split_dir(args.split_dir)

    print(f"Split directory: {split_dir}")
    print_json_file(split_dir / "split_info.json", args.preview, args.show_label_counts)
    print_json_file(split_dir / "test_info.json", args.preview, args.show_label_counts)

    print("\nArrays")
    for name in args.arrays:
        print_array_file(split_dir / f"{name}.npy", args.preview)

    if args.map_sop:
        print_sop_mapping(split_dir, args.arrays, args.dataset_root, args.preview)

    if args.csv is not None:
        export_sop_csv(args.csv, split_dir, args.arrays, args.dataset_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
