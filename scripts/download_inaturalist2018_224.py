"""Download iNaturalist 2018 while storing only DINO-sized images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import INaturalist2018  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream the official iNaturalist 2018 train/validation archive and "
            "store only 224x224 JPEGs for this project's DINOv2 input pipeline."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "iNat2018",
        help="Destination dataset directory (default: data/iNat2018).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = INaturalist2018.is_ready(args.root)
    destination = INaturalist2018.download_224(args.root)
    if already_ready:
        print(f"iNaturalist 2018 224x224 dataset is already complete: {destination}")
    else:
        print(f"iNaturalist 2018 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
