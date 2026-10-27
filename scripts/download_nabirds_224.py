"""Download NABirds with 224x224 images and a class-disjoint split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import NABirds  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NABirds from its Kaggle mirror and store only 224x224 "
            "JPEGs in the official layout. The project pools the official "
            "image-level split and repartitions complete visual categories, "
            "so the class split is chosen at load time by dataset_protocol "
            "rather than baked in here. The images are Cornell's and remain "
            "under the terms at https://dl.allaboutbirds.org/nabirds."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "NABirds",
        help="Destination dataset directory (default: data/NABirds).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Use an already downloaded archive or extracted official "
            "directory instead of Kaggle. Without it, an archive.zip in or "
            "beside --root is used before anything is downloaded."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_complete = NABirds.is_224_complete(args.root)
    destination = NABirds.download_224(args.root, source=args.source)
    if already_complete:
        print(f"NABirds 224x224 dataset is already complete: {destination}")
    else:
        print(f"NABirds 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
