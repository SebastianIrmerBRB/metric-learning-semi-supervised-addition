from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402
from training.types import DATASETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a few train and test images from a dataset into show/."
    )
    parser.add_argument("--dataset", choices=DATASETS, default="Cars196", help="Dataset to sample.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("show"),
        help="Directory where sampled images are written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of images to save from each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to choose sample positions.",
    )
    parser.add_argument(
        "--dataset-protocol",
        choices=utils.DATASET_PROTOCOLS,
        default=utils.DATASET_PROTOCOL_OFFICIAL,
        help="Dataset protocol to use for train/test sources.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to data/<dataset>.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision/pytorch-metric-learning to download missing data.",
    )
    return parser.parse_args()


def choose_positions(dataset_size: int, count: int, seed: int) -> list[int]:
    if count < 1:
        raise ValueError("--count must be at least 1")
    if dataset_size < 1:
        raise ValueError("Cannot sample from an empty dataset")
    if dataset_size <= count:
        return list(range(dataset_size))

    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), count))


def filename_label(label: object) -> str:
    text = str(label)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def as_pil_image(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    if hasattr(image, "detach"):
        return to_pil_image(image.detach().cpu())

    return Image.fromarray(np.asarray(image))


def save_samples(dataset, split_name: str, count: int, seed: int, output_dir: Path) -> None:
    positions = choose_positions(len(dataset), count, seed)
    for sample_number, position in enumerate(positions, start=1):
        image, label = dataset[position]
        pil_image = as_pil_image(image)
        if pil_image.mode not in {"RGB", "L"}:
            pil_image = pil_image.convert("RGB")

        output_path = output_dir / (
            f"{split_name}_{sample_number:02d}_pos_{position:06d}"
            f"_label_{filename_label(label)}.png"
        )
        pil_image.save(output_path)
        print(f"Wrote {output_path}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root if args.data_root is not None else Path("data") / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, _ = utils.load_dataset_protocol_sources(
        dataset_name=args.dataset,
        data_root=data_root,
        train_transform=None,
        test_transform=None,
        download=args.download,
        seed=args.seed,
        dataset_protocol=args.dataset_protocol,
    )

    save_samples(train_dataset, "train", args.count, args.seed, args.output_dir)
    save_samples(test_dataset, "test", args.count, args.seed + 1, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
