"""Download FGVC-Aircraft for the class-disjoint metric-learning protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import FGVCAircraft  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream FGVC-Aircraft and store only 224x224 JPEGs, with the "
            "required 20-pixel copyright banner removed. The project pools "
            "the official image splits and uses the first 50 variants for "
            "development and the remaining 50 for final testing. Source "
            "photographs are licensed for non-commercial research only."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "FGVCAircraft",
        help="Destination dataset directory (default: data/FGVCAircraft).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = FGVCAircraft.is_ready(args.root)
    destination = FGVCAircraft.download_224(args.root)
    if already_ready:
        print(
            "FGVC-Aircraft 224x224 dataset is already complete: "
            f"{destination}"
        )
    else:
        print(f"FGVC-Aircraft 224x224 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
