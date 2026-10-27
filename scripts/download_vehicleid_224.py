"""Download PKU VehicleID with 224x224 images and its official identity split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import VehicleID  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PKU VehicleID from its Kaggle mirror and store only 224x224 "
            "JPEGs in the official layout. A class is one vehicle identity, "
            "and the released train_test_split lists decide the partition: one "
            "training list and three held-out lists of 800, 1,600 and 2,400 "
            "identities. PKU distributes the dataset only against a signed "
            "agreement (https://www.pkuml.org/resources/pku-vehicleid.html); "
            "the mirror only removes that step, not the terms."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "VehicleID",
        help="Destination dataset directory (default: data/VehicleID).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Use an already downloaded archive or extracted directory instead "
            "of Kaggle. Without it, an archive in or beside --root is used "
            "before anything is downloaded."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_complete = VehicleID.is_224_complete(args.root)
    destination = VehicleID.download_224(args.root, source=args.source)
    if already_complete:
        print(f"VehicleID 224x224 dataset is already complete: {destination}")
    else:
        print(f"VehicleID 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
