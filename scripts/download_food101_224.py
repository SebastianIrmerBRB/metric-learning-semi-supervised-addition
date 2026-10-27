"""Download pooled Food-101 with a fixed class-disjoint split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import Food101  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download Food-101 from Kaggle, pool its official train/test "
            "images, retain 224x224 JPEGs, and use a fixed class-disjoint "
            "71/30 development/test split."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "Food101",
        help="Destination dataset directory (default: data/Food101).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = Food101.is_ready(args.root)
    destination = Food101.download_224(args.root)
    if already_ready:
        print(f"Food-101 dataset is already complete: {destination}")
    else:
        print(f"Food-101 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
