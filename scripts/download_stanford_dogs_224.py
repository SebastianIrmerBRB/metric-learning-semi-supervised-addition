"""Download Stanford Dogs with 224x224 images and a class-disjoint split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import StanfordDogs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream Stanford Dogs and store only 224x224 JPEGs. The project "
            "uses a fixed class-disjoint 60%/40% development/test split."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "StanfordDogs",
        help="Destination dataset directory (default: data/StanfordDogs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = StanfordDogs.is_ready(args.root)
    destination = StanfordDogs.download_224(args.root)
    if already_ready:
        print(f"Stanford Dogs 224x224 dataset is already complete: {destination}")
    else:
        print(f"Stanford Dogs 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
