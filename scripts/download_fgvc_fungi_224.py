"""Download FGVCx Fungi 2018 with a class-disjoint 224px protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_datasets import FGVCFungi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download the labeled FGVCx Fungi 2018 train/validation images, "
            "retain only 224x224 JPEGs, pool Meta-Dataset's 994 training and "
            "200 validation species for cross-validation, and retain its 200 "
            "test species as a fixed class-disjoint test set. The "
            "official labs.gbif.org mirror answers 403, so the Kaggle copy of "
            "the competition is tried first: accept the terms at "
            "https://www.kaggle.com/c/fungi-challenge-fgvc-2018 and configure "
            "a Kaggle API token, or place fungi_train_val.tgz in --root."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data") / "FGVCFungi",
        help="Destination dataset directory (default: data/FGVCFungi).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = FGVCFungi.is_ready(args.root)
    destination = FGVCFungi.download_224(args.root)
    if already_ready:
        print(f"FGVCx Fungi 2018 dataset is already complete: {destination}")
    else:
        print(f"FGVCx Fungi 2018 dataset is ready: {destination}")


if __name__ == "__main__":
    main()
