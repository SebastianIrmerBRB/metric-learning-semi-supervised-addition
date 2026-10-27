"""Frozen backbone features from the run cache, in place of decoded images.

A debug run's cost is image decoding: the tiny model is a projection over a
random 16x16 encoder, so the pixels are transformed and thrown at a layer that
learns nothing from them. A frozen-backbone production run has already written
one DINOv2 row per dataset position under ``data/<dataset>/backbone_cache``.
Feeding those rows to the tiny model removes decoding from the step *and* puts
the losses on the geometry a frozen run actually optimizes, instead of on the
geometry of a random projection of raw pixels.

The rows are one deterministic view per sample, so augmentation is gone: this is
the ``use_cache=True`` frozen-backbone regime, not a shortcut around it.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

import utils as project_utils
from scripts.bench_seraph_gradient_contribution import (
    find_cached_feature_matrix,
    nearest_neighbor_label_agreement,
)


# Frozen DINOv2 features of a retrieval dataset land far above chance. A value
# near 1 / num_classes means the cache rows are not the dataset positions the
# labels were read from, which would silently invalidate every distance the
# debugger prints.
MIN_NN_LABEL_AGREEMENT_FACTOR = 5.0


def resolve_feature_cache_path(
    args: argparse.Namespace,
    source_dataset_size: int,
) -> Path:
    """Return the cache matrix for the dataset split, or explain why there is none."""

    if args.backbone_feature_cache is not None:
        path = Path(args.backbone_feature_cache).expanduser()
        if not path.is_file():
            raise SystemExit(f"--backbone-feature-cache {path} does not exist")
        return path
    path = find_cached_feature_matrix(args.dataset, source_dataset_size, "data")
    if path is None:
        raise SystemExit(
            f"No cached backbone features with {source_dataset_size} rows under "
            f"{Path('data') / args.dataset / 'backbone_cache'}. The debugger cannot "
            "compute them: run a --use_cache frozen-backbone training job on "
            f"{args.dataset} first, pass --backbone-feature-cache, or rerun with "
            "--backbone-features=pixels."
        )
    return path


def build_precomputed_pool_dataset(
    image_pool: Subset,
    cache_path: Path,
    source_dataset_size: int,
    num_classes: int,
) -> Any:
    """Swap a debug pool's images for its rows of a cached feature matrix.

    ``image_pool`` is the ``Subset`` selected by ``_select_real_debug_pool``; its
    ``source_positions`` are positions in the full split, which is exactly how the
    cache matrix is indexed.
    """

    matrix = torch.from_numpy(np.load(cache_path)).float()
    if len(matrix) != source_dataset_size:
        raise SystemExit(
            f"{cache_path} has {len(matrix)} rows, but the dataset split it must be "
            f"indexed by has {source_dataset_size} samples"
        )
    source_positions = torch.as_tensor(image_pool.source_positions, dtype=torch.long)
    features = matrix[source_positions].contiguous()
    labels = torch.as_tensor(image_pool.labels, dtype=torch.long)

    agreement = nearest_neighbor_label_agreement(features, labels)
    if agreement < MIN_NN_LABEL_AGREEMENT_FACTOR / max(1, num_classes):
        raise SystemExit(
            f"Cached features from {cache_path} have 1-NN label agreement "
            f"{agreement:.4f} on the debug pool, near the {1 / max(1, num_classes):.4f} "
            "expected from unrelated rows. The matrix is not aligned with this "
            "dataset's positions; pass the right --backbone-feature-cache or rerun "
            "with --backbone-features=pixels."
        )

    dataset = project_utils.PrecomputedBackboneFeatureDataset(
        features=features,
        labels=image_pool.orig_labels,
        dense_labels=image_pool.labels,
        source_dataset=image_pool,
    )
    dataset.source_positions = image_pool.source_positions
    dataset.source_dataset_size = image_pool.source_dataset_size
    dataset.short_unlabeled_classes = image_pool.short_unlabeled_classes
    dataset.feature_cache_path = str(cache_path)
    dataset.nn_label_agreement = float(agreement)
    # The contact sheets still show pictures. They come from the images the cached
    # rows were computed from, which is the deterministic feature transform.
    dataset.display_image_dataset = _feature_transform_pool(image_pool)
    return dataset


def _feature_transform_pool(image_pool: Subset) -> Subset:
    """Return the same pool, reading images through the deterministic transform."""

    display_source = copy.deepcopy(image_pool.dataset)
    feature_transform = project_utils.get_nested_feature_transform(image_pool)
    if feature_transform is not None:
        project_utils.set_nested_transform(display_source, feature_transform)
    return Subset(display_source, list(image_pool.indices))
