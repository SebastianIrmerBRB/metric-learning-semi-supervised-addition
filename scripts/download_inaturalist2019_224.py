"""Download iNaturalist 2019 while storing only DINO-sized images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import INaturalist2019  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream the official iNaturalist 2019 train/validation archive, "
            "store only 224x224 JPEGs, and use the published DML split of 656 "
            "development species (48 genera) against 354 held-out species "
            "(24 genera)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "iNat2019",
        help="Destination dataset directory (default: data/iNat2019).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = INaturalist2019.is_ready(args.root)
    destination = INaturalist2019.download_224(args.root)
    if already_ready:
        print(f"iNaturalist 2019 224x224 dataset is complete: {destination}")
    else:
        print(f"iNaturalist 2019 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
