"""Shared state for one debugger scenario."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

import utils as project_utils

from scripts.ssl_loss_debugger.models import TinyRetrievalModel
from scripts.ssl_loss_debugger.training.cached_features import (
    build_precomputed_pool_dataset,
    resolve_feature_cache_path,
)
from scripts.ssl_loss_debugger.training.datasets import TinyImageDataset
from scripts.ssl_loss_debugger.utils.constants import BACKBONE_FEATURES_CACHED
from scripts.ssl_loss_debugger.utils.debugging import seed_everything
from training.ssl.config import SemiSupervisedSplit


@dataclass
class DebugContext:
    args: argparse.Namespace
    dataset: Dataset
    split: SemiSupervisedSplit
    label_mapper: dict[int, int]
    model: nn.Module
    input_image_size: int
    source_dataset_size: int

    @property
    def device(self) -> torch.device:
        return torch.device(self.args.device)

    @property
    def num_classes(self) -> int:
        return len(self.label_mapper)


def _make_synthetic_context(args: argparse.Namespace) -> DebugContext:
    samples_per_class = args.labeled_per_class + args.unlabeled_per_class
    dataset = TinyImageDataset(
        num_classes=args.num_classes,
        samples_per_class=samples_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )
    labeled_positions = []
    unlabeled_positions = []
    for class_id in range(args.num_classes):
        start = class_id * samples_per_class
        labeled_positions.extend(range(start, start + args.labeled_per_class))
        unlabeled_positions.extend(
            range(
                start + args.labeled_per_class,
                start + samples_per_class,
            )
        )
    split = SemiSupervisedSplit(
        labeled_positions=np.asarray(labeled_positions, dtype=np.int64),
        unlabeled_positions=np.asarray(unlabeled_positions, dtype=np.int64),
    )
    model = TinyRetrievalModel(
        image_size=args.image_size,
        feat_dim=args.embedding_dim,
        backbone_tuning=args.backbone_tuning,
    ).to(args.device)
    label_mapper = {100 + class_id: class_id for class_id in range(args.num_classes)}
    return DebugContext(
        args=args,
        dataset=dataset,
        split=split,
        label_mapper=label_mapper,
        model=model,
        input_image_size=args.image_size,
        source_dataset_size=len(dataset),
    )


def _select_real_debug_pool(dataset: Dataset, args: argparse.Namespace):
    labels = np.asarray(dataset.labels, dtype=np.int64)
    original_labels = np.asarray(dataset.orig_labels, dtype=np.int64)
    if len(labels) != len(dataset) or len(original_labels) != len(dataset):
        raise ValueError(
            "Real debug datasets must expose aligned labels and orig_labels"
        )

    rng = np.random.default_rng(args.seed)
    labeled_source_positions = []
    unlabeled_source_positions = []
    short_unlabeled_classes = {}
    for class_label in np.unique(labels):
        class_positions = rng.permutation(np.flatnonzero(labels == class_label))
        required = args.labeled_per_class + 1
        if len(class_positions) < required:
            raise ValueError(
                f"Class {int(class_label)} has {len(class_positions)} samples, but the debugger "
                f"needs at least {required} for --labeled-per-class={args.labeled_per_class} "
                "plus an unlabeled example."
            )
        labeled = class_positions[: args.labeled_per_class]
        remaining = class_positions[args.labeled_per_class :]
        unlabeled_count = min(args.unlabeled_per_class, len(remaining))
        if unlabeled_count < args.unlabeled_per_class:
            short_unlabeled_classes[int(class_label)] = int(unlabeled_count)
        labeled_source_positions.extend(int(position) for position in labeled)
        unlabeled_source_positions.extend(
            int(position) for position in remaining[:unlabeled_count]
        )

    selected_source_positions = sorted(
        set(labeled_source_positions) | set(unlabeled_source_positions)
    )
    source_to_debug = {
        source_position: debug_position
        for debug_position, source_position in enumerate(selected_source_positions)
    }
    debug_dataset = Subset(dataset, selected_source_positions)
    selected = np.asarray(selected_source_positions, dtype=np.int64)
    debug_dataset.labels = labels[selected].tolist()
    debug_dataset.orig_labels = original_labels[selected].tolist()
    debug_dataset.feature_transform = project_utils.get_nested_feature_transform(
        dataset
    )
    debug_dataset.source_positions = selected_source_positions
    debug_dataset.source_dataset_size = len(dataset)
    debug_dataset.short_unlabeled_classes = short_unlabeled_classes
    split = SemiSupervisedSplit(
        labeled_positions=np.asarray(
            sorted(source_to_debug[position] for position in labeled_source_positions),
            dtype=np.int64,
        ),
        unlabeled_positions=np.asarray(
            sorted(
                source_to_debug[position] for position in unlabeled_source_positions
            ),
            dtype=np.int64,
        ),
    )
    return debug_dataset, split


def _make_real_context(args: argparse.Namespace) -> DebugContext:
    data_root = Path("data") / args.dataset
    if not project_utils.is_dataset_ready(args.dataset, data_root):
        if not args.download_dataset:
            raise SystemExit(
                f"{args.dataset} is not available at {data_root}. Populate that directory or "
                "rerun with --download-dataset."
            )

    dataset_bundle = project_utils.setup_dataset_bundle(
        dataset_name=args.dataset,
        seed=args.seed,
        data_split_seed=args.seed,
        dataset_protocol=args.dataset_protocol,
        full_train=True,
    )
    source_dataset = dataset_bundle.train_dataset
    dataset, split = _select_real_debug_pool(source_dataset, args)

    precomputed_feature_dim = None
    if args.backbone_features == BACKBONE_FEATURES_CACHED:
        cache_path = resolve_feature_cache_path(args, len(source_dataset))
        dataset = build_precomputed_pool_dataset(
            image_pool=dataset,
            cache_path=cache_path,
            source_dataset_size=len(source_dataset),
            num_classes=len(dataset_bundle.train_labels_mapper),
        )
        precomputed_feature_dim = dataset.features.shape[1]
        print(
            f"backbone features: {cache_path} "
            f"({precomputed_feature_dim}-d, 1-NN label agreement "
            f"{dataset.nn_label_agreement:.4f} on the debug pool); images are read "
            "only for the comparison contact sheets"
        )

    # setup_dataset_bundle uses the same 224x224 production transform for each
    # supported real dataset. Keep the tiny debugger model while matching that
    # actual transformed input rather than replacing it with synthetic views.
    input_image_size = 224
    seed_everything(args.seed)
    model = TinyRetrievalModel(
        image_size=input_image_size,
        feat_dim=args.embedding_dim,
        backbone_tuning=args.backbone_tuning,
        precomputed_feature_dim=precomputed_feature_dim,
    ).to(args.device)
    return DebugContext(
        args=args,
        dataset=dataset,
        split=split,
        label_mapper=dataset_bundle.train_labels_mapper,
        model=model,
        input_image_size=input_image_size,
        source_dataset_size=len(source_dataset),
    )


def make_context(args: argparse.Namespace) -> DebugContext:
    seed_everything(args.seed)
    if args.dataset == "synthetic":
        return _make_synthetic_context(args)
    return _make_real_context(args)
