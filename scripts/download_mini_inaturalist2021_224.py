"""Download mini-iNaturalist-2021 while storing only DINO-sized images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import MiniINaturalist2021  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream the official 500,000-image mini-iNaturalist-2021 archive, "
            "store only 224x224 JPEGs, and reproduce the Contextual Similarity "
            "authors' released 5,003/4,997 species-disjoint split."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "iNat2021Mini",
        help="Destination dataset directory (default: data/iNat2021Mini).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = MiniINaturalist2021.is_ready(args.root)
    destination = MiniINaturalist2021.download_224(args.root)
    if already_ready:
        print(f"mini-iNaturalist-2021 224x224 dataset is complete: {destination}")
    else:
        print(f"mini-iNaturalist-2021 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
