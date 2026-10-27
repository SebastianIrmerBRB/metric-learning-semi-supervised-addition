"""Download pooled Semi-iNaturalist 2021 with a class-disjoint split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import SemiINaturalist2021  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download all Semi-iNaturalist 2021 image sources and oracle "
            "labels, retain 224x224 JPEGs, and use the fixed PML-style 70/30 "
            "class-disjoint development/test split."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "iNat",
        help="Destination dataset directory (default: data/iNat).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = SemiINaturalist2021.is_ready(args.root)
    destination = SemiINaturalist2021.download_224(args.root)
    if already_ready:
        print(f"Semi-iNaturalist 2021 dataset is already complete: {destination}")
    else:
        print(f"Semi-iNaturalist 2021 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
