from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import local_datasets  # noqa: E402

CompCarsSLADE = local_datasets.CompCarsSLADEPaperUnlabeledImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a local CompCars copy reproduces the SLADE unlabeled pool of "
            "16,537 images across 145 make/model/year classes. Runs the same discovery, "
            "category inference, and threshold calibration as the "
            "external_unlabeled_filter='compcars_slade_paper' training path, without "
            "reading any image data."
        )
    )
    parser.add_argument("root", type=Path, help="CompCars root, the same path used for external_unlabeled_dir.")
    parser.add_argument(
        "--min-images-per-model",
        type=int,
        default=CompCarsSLADE.PAPER_MIN_IMAGES_PER_MODEL,
        help="Count threshold below which a make/model/year category is dropped as unbalanced.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Use --min-images-per-model verbatim instead of searching for the published counts.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        default=10,
        help="How many count thresholds to tabulate, closest to the published counts first.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for writing the full report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        print(f"CompCars root does not exist: {root}")
        return 2

    candidate_paths, candidate_source = CompCarsSLADE.discover_paper_candidate_paths(root)
    if not candidate_paths:
        print(f"No CompCars images found below {root}")
        return 2
    model_counts = Counter(CompCarsSLADE.infer_model_key(root, path) for path in candidate_paths)

    requested_threshold = args.min_images_per_model
    threshold = requested_threshold
    if not args.no_calibration:
        threshold = CompCarsSLADE.calibrate_min_images_per_model(
            model_counts,
            requested_threshold,
            CompCarsSLADE.PAPER_TARGET_IMAGES,
            CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES,
        )
    kept_counts = [count for count in model_counts.values() if count >= threshold]
    kept_images = sum(kept_counts)
    kept_model_classes = len(kept_counts)
    matches_paper = (
        kept_images == CompCarsSLADE.PAPER_TARGET_IMAGES
        and kept_model_classes == CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES
    )

    unit = CompCarsSLADE.CATEGORY_UNIT
    print(f"root:             {root}")
    print(f"candidate source: {candidate_source}")
    print(
        "candidate pool:   "
        f"{len(candidate_paths)} images / {len(model_counts)} {unit} "
        f"(full release: {CompCarsSLADE.WEB_NATURE_IMAGES} images / "
        f"{CompCarsSLADE.WEB_NATURE_MODEL_YEAR_CLASSES} {unit})"
    )
    calibration_note = f" (calibrated from {requested_threshold})" if threshold != requested_threshold else ""
    print(f"threshold:        >= {threshold} images per category{calibration_note}")
    print(
        "kept pool:        "
        f"{kept_images} images / {kept_model_classes} {unit} "
        f"(SLADE: {CompCarsSLADE.PAPER_TARGET_IMAGES} images / "
        f"{CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES} {unit})"
    )
    if kept_counts:
        summary = CompCarsSLADE.summarize_model_counts(
            {key: count for key, count in model_counts.items() if count >= threshold}
        )
        print(
            "kept per class:   "
            f"min={summary['min']}, median={summary['median']}, "
            f"mean={summary['mean']:.1f}, max={summary['max']}"
        )
    print(f"matches paper:    {matches_paper}")

    rows = CompCarsSLADE.nearest_threshold_diagnostics(
        model_counts,
        CompCarsSLADE.PAPER_TARGET_IMAGES,
        CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES,
        limit=args.thresholds,
    )
    print("\nnearest thresholds:")
    print(f"  {'threshold':>9}  {'images':>7}  {'classes':>7}")
    for row in rows:
        print(f"  {row['threshold']:>9}  {row['images']:>7}  {row['model_classes']:>7}")

    if args.json is not None:
        report = {
            "root": str(root),
            "candidate_source": candidate_source,
            "candidate_images": len(candidate_paths),
            "candidate_model_classes": len(model_counts),
            "requested_min_images_per_model": requested_threshold,
            "min_images_per_model": threshold,
            "kept_images": kept_images,
            "kept_model_classes": kept_model_classes,
            "matches_expected_counts": matches_paper,
            "nearest_count_thresholds": rows,
            "model_counts": dict(sorted(model_counts.items())),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if matches_paper else 1


if __name__ == "__main__":
    raise SystemExit(main())
