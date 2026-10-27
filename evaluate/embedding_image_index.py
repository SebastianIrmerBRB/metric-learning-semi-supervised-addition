"""The image file behind every sample of a run, read from the data folder.

The embedding viewer's image server asks for ``(run, set, sample)``; this
rebuilds the run's datasets from its own ``run_config.json`` (no model, no
features) and lists each set's files in dataset order, so a view never has to
carry image paths and nothing is copied. Paths are relative to the repository
(``data/...``), or null where a dataset has no image files (CIFAR).

A CV directory (``cv_*``) has no run_config.json of its own; its folds share
the test set but not the train/validation split, so it resolves through fold 0.

    python -m evaluate.embedding_image_index logs/final_tracking/.../fold_00/2026-09-17_13-49-28
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

IMAGE_SETS = ("train", "validation", "test")


def run_config_dir(run_dir):
    run_dir = Path(run_dir)
    if (run_dir / "run_config.json").is_file():
        return run_dir
    folds = sorted(path.parent for path in run_dir.glob("fold_*/*/run_config.json"))
    if folds:
        return folds[0]
    raise FileNotFoundError(f"{run_dir} holds no run_config.json")


def image_index(run_dir):
    from evaluate.export_embedding_views import dataset_image_paths, load_run
    from training import head_evaluation

    bundle = head_evaluation.rebuild_dataset_bundle(load_run(run_config_dir(run_dir)))
    datasets = dict(zip(IMAGE_SETS, (bundle.train_dataset, bundle.valid_dataset, bundle.test_dataset)))
    return {
        name: [] if dataset is None else dataset_image_paths(dataset, range(len(dataset)))
        for name, dataset in datasets.items()
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m evaluate.embedding_image_index <run_dir>")
    # Loguru and dataset setup log to stderr; stdout carries only the JSON.
    json.dump(image_index(sys.argv[1]), sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
