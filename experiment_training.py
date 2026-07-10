"""Single-fold training, cross-validation, and model component construction."""

import argparse
import copy
import csv
import gc
import os
from dataclasses import replace
from datetime import datetime
from numbers import Real
from pathlib import Path

import time
from collections import defaultdict

import numpy as np
import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.miners as miners
import torch
from loguru import logger
from tqdm import tqdm

import metric_losses
import semi_supervised
import utils
from experiment_cli import (
    DEFAULT_DATA_SPLIT_SEED,
    FINAL_TEST_VISUALIZATION_MODES,
    FINAL_TEST_VISUALIZATION_NONE,
    FINAL_TEST_VISUALIZATION_PACMAP,
    normalize_backbone_tuning_args,
)
from experiment_io import namespace_to_dict, result_to_dict, write_json
from experiment_types import (
    ALL_LOSSES,
    ALL_MINERS,
    CLASSIFICATION_LOSSES,
    DATASETS,
    SELECTION_METRIC_MAP_AT_R,
    SELECTION_METRIC_PRECISION_AT_1,
    SELECTION_METRICS,
    TrainingResult,
)
from retrieval_model import BACKBONE_TUNING_FROZEN, DinoWrapper

BATCH_EASY_HARD_MINER_STRATEGIES = {"all", "easy", "hard", "semihard"}
BATCH_EASY_HARD_DEFAULT_POS_STRATEGY = "easy"
BATCH_EASY_HARD_DEFAULT_NEG_STRATEGY = "semihard"
BATCH_EASY_HARD_RANGE_PARAMS = ("allowed_pos_range", "allowed_neg_range")

def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _now(device):
    _sync_if_cuda(device)
    return time.perf_counter()


def _timing_enabled(args):
    return bool(getattr(args, "debug_batch_timing", False))


def _timing_interval(args):
    return int(getattr(args, "debug_batch_timing_interval", 5))


def _batch_diagnostics_enabled(args):
    return bool(getattr(args, "log_batch_diagnostics", False))


def _add_time(timings, name, start, end):
    timings[name] += end - start


def use_cuda_pin_memory(device):
    return torch.device(device).type == "cuda"


def should_precompute_frozen_features(args, ssl_config):
    regularized_ssl = is_cacheable_regularized_ssl(ssl_config)
    return (
        bool(getattr(args, "use_cache", False))
        and args.backbone_tuning == BACKBONE_TUNING_FROZEN
        and (
            (is_supervised_mode(args) and not ssl_config.enabled)
            or (not is_supervised_mode(args) and regularized_ssl)
        )
    )


def is_cacheable_regularized_ssl(ssl_config):
    if not ssl_config.enabled:
        return False
    ssl_method = semi_supervised.get_method(ssl_config)
    if ssl_method is None or not ssl_method.is_regularization_method:
        return False
    regularizer = ssl_method.make_regularizer(ssl_config)
    return regularizer.name in {"hoffer_entropy", "lrml", "slrmml"}


def get_frozen_feature_batch_size(args):
    batch_size = getattr(args, "frozen_feature_batch_size", None)
    return int(args.batch_size if batch_size is None else batch_size)


def get_frozen_feature_train_views(args):
    return int(getattr(args, "frozen_feature_train_views", 1))


def make_label_lookup_tensor(train_labels_mapper, device):
    label_ids = [int(label) for label in train_labels_mapper]
    if not label_ids or min(label_ids) < 0:
        return None
    max_label = max(label_ids)
    if max_label > 10_000_000:
        return None
    lookup = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
    for original_label, mapped_label in train_labels_mapper.items():
        lookup[int(original_label)] = int(mapped_label)
    return lookup


def map_training_labels(labels, label_lookup, train_labels_mapper, device):
    if torch.is_tensor(labels) and label_lookup is not None:
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        return label_lookup[labels]
    return torch.tensor([train_labels_mapper[int(label)] for label in labels], device=device, dtype=torch.long)


def shutdown_epoch_train_loader(train_loader, warmup_train_loader=None, static_train_loader=None, *reusable_loaders):
    """Shutdown loaders that are rebuilt for a single epoch."""

    if train_loader is None:
        return
    if train_loader is warmup_train_loader or train_loader is static_train_loader:
        return
    if any(train_loader is loader for loader in reusable_loaders):
        return
    shutdown = getattr(train_loader, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    utils.shutdown_dataloaders(train_loader)


def run_experiment(args, ssl_config, optuna_trial=None, optuna_metric=None):
    """Run either one holdout training job or all requested CV folds."""

    args = resolve_loss_driven_supervised_args(args)
    args, ssl_config = resolve_platform_dataloader_workers(args, ssl_config)
    # Keep this normalization step at the common entry point so direct runs,
    # HPO trials, comparisons, and grid scenarios resolve validation equally.
    args = resolve_validation_mode_args(args)
    if args.cv_k > 1:
        # The Optuna trial is reported only after folds complete; individual
        # folds do not independently prune the same trial.
        return run_cross_validation(args, ssl_config, optuna_trial=optuna_trial, optuna_metric=optuna_metric)
    return run_training(args, ssl_config, optuna_trial=optuna_trial, optuna_metric=optuna_metric)

def resolve_platform_dataloader_workers(args, ssl_config, platform_name=None):
    """Apply platform-specific effective worker counts without changing source configs."""

    effective_train_workers = utils.effective_num_workers(args.num_workers, platform_name=platform_name)
    effective_embedding_workers = utils.effective_num_workers(
        ssl_config.embedding_num_workers,
        platform_name=platform_name,
    )
    if (
        effective_train_workers == args.num_workers
        and effective_embedding_workers == ssl_config.embedding_num_workers
    ):
        return args, ssl_config

    resolved_args = copy.deepcopy(args)
    resolved_args.configured_num_workers = args.num_workers
    resolved_args.configured_ssl_embedding_num_workers = ssl_config.embedding_num_workers
    resolved_args.num_workers = effective_train_workers
    resolved_args.windows_dataloader_workers_forced_to_zero = True
    resolved_ssl_config = replace(ssl_config, embedding_num_workers=effective_embedding_workers)
    return resolved_args, resolved_ssl_config

def is_supervised_mode(args):
    return getattr(args, "mode", "supervised") == "supervised"


def resolve_loss_driven_supervised_args(args):
    """Use the configured warm-up objective for an STML supervised baseline."""

    if not is_supervised_mode(args) or getattr(args, "loss", None) != "STMLLoss":
        return args
    resolved = copy.deepcopy(args)
    resolved.loss = resolved.warmup_loss
    resolved.loss_params = dict(resolved.warmup_loss_params)
    resolved.miner = resolved.warmup_miner
    resolved.miner_params = dict(resolved.warmup_miner_params)
    return resolved


def resolve_validation_mode_args(args):
    return args

def resolve_mode_ssl_config(args, ssl_config):
    if is_supervised_mode(args):
        return make_supervised_split_config(ssl_config)
    if not ssl_config.enabled:
        raise ValueError("--mode ssl requires an enabled --ssl_config")
    return ssl_config

def get_selection_metric_value(selection_metric, precision_at_1, mean_average_precision_at_r):
    if selection_metric == SELECTION_METRIC_PRECISION_AT_1:
        return precision_at_1
    if selection_metric == SELECTION_METRIC_MAP_AT_R:
        return mean_average_precision_at_r
    raise ValueError(f"Unknown selection metric: {selection_metric}")

def uses_ssl_warmup_objective(ssl_config):
    """Return whether this run has labeled-only SSL warmup epochs."""

    return ssl_config.enabled and ssl_config.warmup_epochs > 0

def is_ssl_warmup_epoch(ssl_config, num_epoch):
    """Return whether the current epoch should use the warmup objective."""

    return ssl_config.enabled and num_epoch < ssl_config.warmup_epochs

def write_split_manifest(log_dir, dataset_bundle, ssl_config, ssl_split):
    """Persist the exact split so an experiment can be audited or reproduced.

    ``*_positions`` refer to offsets inside the current training subset,
    whereas ``*_indices`` refer to samples in the underlying source dataset.
    """

    split_dir = Path(log_dir) / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    if ssl_split is None:
        # No explicit label budget means every position in the current training
        # subset is treated as labeled and there is no unlabeled candidate pool.
        labeled_positions = np.arange(len(dataset_bundle.train_dataset), dtype=np.int64)
        unlabeled_positions = np.array([], dtype=np.int64)
    else:
        # Copy into stable integer arrays before saving/indexing, regardless of
        # which selector produced the split.
        labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)

    # Subset.indices maps current positions back to the underlying source
    # dataset. Saving both forms makes later debugging much less ambiguous.
    train_indices = get_subset_indices(dataset_bundle.train_dataset)
    val_indices = get_subset_indices(dataset_bundle.valid_dataset)

    np.save(split_dir / "labeled_positions.npy", labeled_positions)
    np.save(split_dir / "unlabeled_positions.npy", unlabeled_positions)
    np.save(split_dir / "val_indices.npy", val_indices)
    np.save(split_dir / "train_indices.npy", train_indices)
    # Convert labeled/unlabeled positions through train_indices so these files
    # refer to source samples even if the training subset is later rebuilt.
    np.save(split_dir / "labeled_indices.npy", positions_to_indices(train_indices, labeled_positions))
    np.save(split_dir / "unlabeled_indices.npy", positions_to_indices(train_indices, unlabeled_positions))

    write_json(
        split_dir / "split_info.json",
        {
            "ssl_config": ssl_config.to_dict(),
            "dataset_split": dataset_bundle.split_info,
            "train_size": len(dataset_bundle.train_dataset),
            "valid_size": len(dataset_bundle.valid_dataset),
            "num_labeled": len(labeled_positions),
            "num_unlabeled": len(unlabeled_positions),
            "labeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, labeled_positions),
            "unlabeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, unlabeled_positions),
        },
    )
    write_json(split_dir / "test_info.json", make_test_info(dataset_bundle.test_dataset))


def resolve_unlabeled_source_args(args):
    """Return the generic external-unlabeled source and root, honoring old STML args."""

    source = getattr(args, "unlabeled_source", None)
    legacy_source = getattr(args, "stml_unlabeled_source", "split")
    if source is None:
        source = legacy_source
    elif legacy_source != "split" and legacy_source != source:
        raise ValueError(
            "unlabeled_source and stml_unlabeled_source disagree: "
            f"{source!r} != {legacy_source!r}"
        )

    external_dir = getattr(args, "external_unlabeled_dir", None)
    legacy_external_dir = getattr(args, "stml_external_unlabeled_dir", None)
    if external_dir is None:
        external_dir = legacy_external_dir
    elif legacy_external_dir is not None and Path(external_dir) != Path(legacy_external_dir):
        raise ValueError(
            "external_unlabeled_dir and stml_external_unlabeled_dir disagree: "
            f"{external_dir} != {legacy_external_dir}"
        )

    return source, external_dir


def configure_external_unlabeled_pool(args, dataset_bundle, ssl_split):
    """Optionally replace or extend split-derived unlabeled candidates."""

    source, external_dir = resolve_unlabeled_source_args(args)
    external_filter = getattr(args, "external_unlabeled_filter", utils.EXTERNAL_UNLABELED_FILTER_NONE)
    if source == "split":
        return dataset_bundle, ssl_split
    if ssl_split is None:
        raise ValueError(f"unlabeled_source={source!r} requires a semi-supervised split")

    internal_train_size = len(dataset_bundle.train_dataset)
    combined_dataset, external_dataset = utils.append_external_unlabeled_dataset(
        dataset_bundle.train_dataset,
        external_dir,
        external_filter=external_filter,
        compcars_min_model_images=getattr(args, "compcars_min_model_images", 100),
        compcars_strict_paper_counts=getattr(args, "compcars_strict_paper_counts", False),
    )
    external_positions = np.arange(
        internal_train_size,
        internal_train_size + len(external_dataset),
        dtype=np.int64,
    )
    internal_unlabeled_positions = (
        np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)
        if source == "split_and_external"
        else np.array([], dtype=np.int64)
    )
    dataset_bundle.train_dataset = combined_dataset
    ssl_split = semi_supervised.SemiSupervisedSplit(
        labeled_positions=np.asarray(ssl_split.labeled_positions, dtype=np.int64),
        unlabeled_positions=np.concatenate([internal_unlabeled_positions, external_positions]),
    )
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["external_unlabeled_pool"] = {
        "source": source,
        "external_root": str(external_dir),
        "internal_train_size": int(internal_train_size),
        "internal_unlabeled_size": int(len(internal_unlabeled_positions)),
        "external_unlabeled_size": int(len(external_dataset)),
        "external_dataset_type": type(external_dataset).__name__,
        "external_unlabeled_filter": external_filter,
    }
    filter_info = getattr(external_dataset, "filter_info", None)
    if filter_info is not None:
        dataset_bundle.split_info["external_unlabeled_pool"]["filter_info"] = filter_info
        logger.info(
            "External unlabeled filter: "
            f"mode={filter_info.get('mode')}, "
            f"candidate_source={filter_info.get('candidate_source')}, "
            f"kept={filter_info.get('kept_images')} images / "
            f"{filter_info.get('kept_model_classes')} model classes, "
            f"dropped={filter_info.get('dropped_images')} images / "
            f"{filter_info.get('dropped_model_classes')} model classes"
        )
        if filter_info.get("matches_expected_counts") is False:
            logger.warning(
                "External unlabeled filter did not match documented STML CompCars counts: "
                f"expected={filter_info.get('expected_images')} images / "
                f"{filter_info.get('expected_model_classes')} model classes, "
                f"got={filter_info.get('kept_images')} images / "
                f"{filter_info.get('kept_model_classes')} model classes. "
                f"Nearest thresholds={filter_info.get('nearest_count_thresholds')}"
            )
    logger.info(
        "External unlabeled pool: "
        f"source={source}, {len(internal_unlabeled_positions)} internal candidates, "
        f"{len(external_dataset)} external candidates from {external_dir}, "
        f"filter={external_filter}"
    )
    return dataset_bundle, ssl_split


def configure_stml_unlabeled_pool(args, dataset_bundle, ssl_split):
    """Compatibility wrapper for the old STML-specific external pool hook."""

    return configure_external_unlabeled_pool(args, dataset_bundle, ssl_split)


def get_subset_indices(dataset):
    # Plain datasets already use source indices 0..N-1; Subset instances expose
    # an explicit mapping through their indices attribute.
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return np.arange(len(dataset), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)

def positions_to_indices(indices, positions):
    # Array indexing performs the position -> source-index lookup in one step.
    if len(indices) == 0 or len(positions) == 0:
        return np.array([], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)[np.asarray(positions, dtype=np.int64)]

def label_counts(labels, positions=None):
    # Restrict counts to the requested subset positions when supplied.
    labels = np.asarray(labels, dtype=np.int64)
    if positions is not None:
        labels = labels[np.asarray(positions, dtype=np.int64)]
    if len(labels) == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}

def make_test_info(test_dataset):
    labels = getattr(test_dataset, "labels", None)
    info = {
        "size": len(test_dataset),
        "dataset_type": type(test_dataset).__name__,
    }
    if labels is not None:
        info["num_classes"] = int(len(set(int(label) for label in labels)))
        info["label_counts"] = label_counts(labels)
    query_indices = getattr(test_dataset, "query_indices", None)
    gallery_indices = getattr(test_dataset, "gallery_indices", None)
    if query_indices is not None and gallery_indices is not None:
        info["retrieval_mode"] = "query_gallery"
        info["num_queries"] = int(len(query_indices))
        info["num_gallery"] = int(len(gallery_indices))
    return info

def get_data_split_seed(args):
    """Return the fixed seed used for validation/test split construction."""

    data_split_seed = getattr(args, "data_split_seed", None)
    return DEFAULT_DATA_SPLIT_SEED if data_split_seed is None else int(data_split_seed)

def run_training(args, ssl_config, optuna_trial=None, optuna_metric=None, cv_fold=None):
    """Build the data/model, train one fold, and return its best metrics.

    Training has four possible loader phases:
    1. labeled-only warmup;
    2. a single static pseudo-labeled dataset; or
    3. a pseudo-labeled dataset regenerated on a configured epoch cadence; or
    4. STML nearest-neighbor batches regenerated from student g on the same cadence.
    Validation selects the checkpoint and can trigger early stopping. A final
    HPO fit instead trains for a fixed duration without validation. Test data is
    evaluated only when the caller explicitly enables it.
    """

    normalize_backbone_tuning_args(args)
    # Record the resolved dataclass, including defaults and HPO overrides, on
    # the namespace that will later be serialized into run_config.json.
    args.ssl_config_resolved = ssl_config.to_dict()
    final_full_train = bool(getattr(args, "final_full_train", False))
    if final_full_train and cv_fold is not None:
        raise ValueError("A final full-development fit must train one model, not an individual CV fold")

    # Validate before creating a model or downloading/loading a dataset so
    # configuration mistakes fail quickly and cheaply.
    validate_run_args(args, ssl_config)
    utils.seed_everything(args.seed, device=args.device)
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    ssl_method = semi_supervised.get_method(ssl_config)
    regularized_ssl = ssl_method is not None and ssl_method.is_regularization_method
    regularizer = ssl_method.make_regularizer(ssl_config) if regularized_ssl else None
    stml_params = dict(getattr(args, "loss_params", {})) if loss_driven_ssl else {}
    supervised_mode = is_supervised_mode(args)
    precompute_frozen_features = should_precompute_frozen_features(args, ssl_config)
    regularized_frozen_feature_precompute = bool(precompute_frozen_features and regularized_ssl)
    requested_frozen_feature_train_views = get_frozen_feature_train_views(args)
    frozen_feature_train_views = (
        1 if regularized_frozen_feature_precompute else requested_frozen_feature_train_views
    )
    augmented_frozen_feature_precompute = (
        precompute_frozen_features
        and not regularized_frozen_feature_precompute
        and frozen_feature_train_views > 1
    )
    model_use_cache = bool(args.use_cache)
    pin_memory = use_cuda_pin_memory(args.device)

    args.torch_sharing_strategy_resolved = utils.configure_torch_sharing_strategy()
    utils.initialize_logger(args)
    logger.info(f"Torch multiprocessing sharing strategy: {args.torch_sharing_strategy_resolved}")
    if regularized_frozen_feature_precompute and requested_frozen_feature_train_views != 1:
        logger.warning(
            "Regularized frozen feature precompute uses one deterministic view per sample; "
            f"ignoring frozen_feature_train_views={requested_frozen_feature_train_views}"
        )
    # DinoWrapper supplies a pretrained DINO backbone and, when feat_dim is
    # provided, a projection layer that becomes the learned embedding space.
    if loss_driven_ssl:
        regularizer_model_kwargs = {
            "stml": True,
            "stml_g_dim": getattr(args, "stml_g_dim", None),
            "stml_normalize_student": bool(stml_params.get("normalize_student", False)),
        }
    else:
        regularizer_model_kwargs = {} if regularizer is None else regularizer.model_kwargs(args)
    model = DinoWrapper(
        dino_size=args.dino_size,
        feat_dim=args.feat_dim,
        backbone_tuning=args.backbone_tuning,
        use_cache=model_use_cache,
        cache_dir=Path("data") / args.dataset / "backbone_cache",
        **regularizer_model_kwargs,
    )
    model = model.to(args.device)
    args.backbone_cache_dir = None if model.cache_dir is None else model.cache_dir
    args.model_uses_backbone_cache = model_use_cache
    args.frozen_feature_precompute = precompute_frozen_features
    args.frozen_feature_batch_size_resolved = get_frozen_feature_batch_size(args)
    args.frozen_feature_train_views_resolved = frozen_feature_train_views
    args.model_total_parameters = sum(parameter.numel() for parameter in model.parameters())
    args.model_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        "Model parameters: "
        f"{args.model_trainable_parameters:,} trainable / {args.model_total_parameters:,} total. "
        f"Backbone tuning: {args.backbone_tuning}. Cache: {model_use_cache}. "
        f"Frozen feature precompute: {precompute_frozen_features}."
    )
    write_run_config(args, ssl_config)
    pseudo_label_diagnostics = semi_supervised.make_pseudo_label_diagnostics_tracker(
        args.log_dir,
        ssl_config,
    )

    # For a normal run this creates one holdout split.  During CV, cv_fold tells
    # the utility which fold to materialize for this independent training run.
    data_split_seed = get_data_split_seed(args)
    dataset_bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        data_split_seed=data_split_seed,
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_fold=cv_fold,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
        dataset_protocol=args.dataset_protocol,
        cifar_imbalance_factor=args.cifar_imbalance_factor,
        cifar_train_fraction=args.cifar_train_fraction,
        cifar_test_fraction=args.cifar_test_fraction,
        full_train=final_full_train,
    )
    if args.use_cache and not augmented_frozen_feature_precompute:
        # A reusable per-sample DINO embedding requires stable image inputs.
        # Training therefore uses the same deterministic transform as feature
        # extraction instead of stochastic RandAugment.
        utils.use_feature_transform_for_training(dataset_bundle.train_dataset)
        if precompute_frozen_features:
            logger.info(
                "Frozen feature precompute enabled: training uses deterministic transforms and "
                "one in-memory backbone feature tensor per active dataset"
            )
        else:
            logger.info("Cache mode enabled: training uses deterministic transforms and cached DINO embeddings")
    elif augmented_frozen_feature_precompute:
        logger.info(
            "Augmented frozen feature precompute enabled: training keeps stochastic transforms while "
            f"precomputing {frozen_feature_train_views} backbone feature views per sample"
        )
    # Even the supervised baseline uses the SSL config's label-selection rules
    # so supervised and SSL runs receive exactly the same labeled examples.
    if ssl_config.enabled:
        # Enabled SSL produces both true-labeled positions and candidates whose
        # hidden labels may be replaced by pseudo-labels.
        ssl_split = semi_supervised.prepare_ssl_split(dataset_bundle.train_dataset, ssl_config)
    elif supervised_mode:
        logger.info("Training supervised baseline")
        # method="none" disables pseudo-labeling but prepare_label_split still
        # applies the same label budget used by the corresponding SSL method.
        ssl_split = semi_supervised.prepare_label_split(dataset_bundle.train_dataset, ssl_config)
    else:
        # A fully supervised run without a split config uses every train sample.
        ssl_split = None

    if final_full_train:
        logger.info("Final HPO fit uses the complete development set without validation or early stopping")
    elif args.val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        labeled_positions = None if ssl_split is None else ssl_split.labeled_positions
        unlabeled_positions = None if ssl_split is None else ssl_split.unlabeled_positions
        if cv_fold is None:
            # Validation is part of the support draw for this mode.
            dataset_bundle, labeled_positions, unlabeled_positions = utils.apply_post_apportion_validation_split(
                dataset_bundle=dataset_bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                seed=ssl_config.support_seed,
            )
        else:
            dataset_bundle, labeled_positions, unlabeled_positions = utils.apply_apportioned_cross_validation_split(
                dataset_bundle=dataset_bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                include_unlabeled=ssl_config.enabled,
                cv_k=args.cv_k,
                cv_fold=cv_fold,
                cv_mode=args.cv_mode,
                seed=ssl_config.support_seed,
            )
        if ssl_split is not None:
            ssl_split = semi_supervised.SemiSupervisedSplit(
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
            )
    else:
        if ssl_split is None:
            # With all samples labeled, validation matching targets the complete
            # current training subset.
            target_train_size = len(dataset_bundle.train_dataset)
            target_train_num_classes = len(set(int(label) for label in dataset_bundle.train_dataset.labels))
        else:
            # With a label budget, match_train uses only labeled count and class
            # coverage, not the larger unlabeled candidate pool.
            target_train_size = len(ssl_split.labeled_positions)
            train_labels = np.asarray(dataset_bundle.train_dataset.labels, dtype=np.int64)
            target_train_num_classes = int(len(np.unique(train_labels[np.asarray(ssl_split.labeled_positions)])))
        dataset_bundle = utils.apply_validation_mode(
            dataset_bundle=dataset_bundle,
            val_mode=args.val_mode,
            target_train_size=target_train_size,
            target_train_num_classes=target_train_num_classes,
            seed=ssl_config.support_seed,
        )
    if ssl_config.enabled:
        dataset_bundle, ssl_split = configure_external_unlabeled_pool(args, dataset_bundle, ssl_split)
    # Persist both subset-relative positions and source-dataset indices before
    # training so the exact experiment split is recoverable.
    write_split_manifest(args.log_dir, dataset_bundle, ssl_config, ssl_split)
    precomputed_train_samples = None
    if regularized_frozen_feature_precompute:
        dataset_bundle.train_dataset = utils.precompute_backbone_feature_dataset(
            model=model,
            dataset=dataset_bundle.train_dataset,
            device=args.device,
            batch_size=get_frozen_feature_batch_size(args),
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            desc="precompute frozen regularized train features",
            pin_memory=pin_memory,
            require_feature_transform=True,
            use_feature_transform=True,
        )
        precomputed_train_samples = len(dataset_bundle.train_dataset)
    # Build loaders that are known before training.  For every-epoch SSL, the
    # loader is deliberately deferred until the current model can pseudo-label.
    static_train_loader = None
    warmup_train_loader = None
    ssl_training_dataset_update_epoch = None
    if supervised_mode and not ssl_config.enabled:
        # The supervised comparison trains permanently on true-labeled samples
        # only.  Its loader never changes between epochs.
        train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
        )
        if precompute_frozen_features:
            train_dataset = utils.precompute_backbone_feature_dataset(
                model=model,
                dataset=train_dataset,
                device=args.device,
                batch_size=get_frozen_feature_batch_size(args),
                seed=args.seed,
                num_workers=args.num_workers,
                start_method=args.dataloader_start_method,
                desc="precompute frozen train features",
                pin_memory=pin_memory,
                require_feature_transform=not augmented_frozen_feature_precompute,
                use_feature_transform=not augmented_frozen_feature_precompute,
                num_views=frozen_feature_train_views,
            )
        static_train_loader = utils.make_train_loader(
            train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            length_before_new_iter=args.length_before_new_iter,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            pin_memory=pin_memory,
        )
    elif ssl_config.enabled and (ssl_config.warmup_epochs > 0 or regularized_ssl):
        # Regularization methods keep this labeled loader active after warm-up
        # and pair it with each unlabeled regularizer batch.
        warmup_train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
        )
        warmup_train_loader = utils.make_train_loader(
            warmup_train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            length_before_new_iter=args.length_before_new_iter,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            pin_memory=pin_memory,
        )
    if loss_driven_ssl:
        loss_driven_train_dataset = semi_supervised.build_loss_driven_training_dataset(
            dataset_bundle.train_dataset,
            ssl_split,
            num_views=int(stml_params.get("num_views", 2)),
        )
    elif regularized_ssl:
        regularizer.build_dataset(
            dataset_bundle.train_dataset,
            ssl_split,
            use_cache=model_use_cache,
        )
    elif static_train_loader is None and ssl_config.update_mode == "once" and ssl_config.warmup_epochs == 0:
        # With no warmup, "once" means pseudo-label immediately using the
        # off-the-shelf model and reuse those predictions for every epoch.
        train_dataset = semi_supervised.build_ssl_training_dataset(
            model=model,
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            device=args.device,
            config=ssl_config,
            split=ssl_split,
            start_method=args.dataloader_start_method,
            diagnostics_tracker=pseudo_label_diagnostics,
            log_dir=args.log_dir,
        )
        ssl_training_dataset_update_epoch = 0
        static_train_loader = utils.make_train_loader(
            train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            length_before_new_iter=args.length_before_new_iter,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            pin_memory=pin_memory,
        )
    # Evaluation loaders never shuffle because metric computation needs only a
    # deterministic pass over all embeddings and labels.
    valid_loader = None
    if not final_full_train:
        valid_dataset = dataset_bundle.valid_dataset
        if precompute_frozen_features:
            valid_dataset = utils.precompute_backbone_feature_dataset(
                model=model,
                dataset=valid_dataset,
                device=args.device,
                batch_size=get_frozen_feature_batch_size(args),
                seed=args.seed,
                num_workers=args.num_workers,
                start_method=args.dataloader_start_method,
                desc="precompute frozen valid features",
                pin_memory=pin_memory,
            )
        valid_loader = utils.make_eval_loader(
            valid_dataset,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            pin_memory=pin_memory,
        )
    evaluate_test = bool(getattr(args, "evaluate_test", False))
    test_loader = None
    if evaluate_test:
        # Test evaluation is opt-in so HPO and intermediate runs cannot select
        # models based on held-out test performance.
        test_dataset = dataset_bundle.test_dataset
        if precompute_frozen_features:
            test_dataset = utils.precompute_backbone_feature_dataset(
                model=model,
                dataset=test_dataset,
                device=args.device,
                batch_size=get_frozen_feature_batch_size(args),
                seed=args.seed,
                num_workers=args.num_workers,
                start_method=args.dataloader_start_method,
                desc="precompute frozen test features",
                pin_memory=pin_memory,
            )
        test_loader = utils.make_eval_loader(
            test_dataset,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            pin_memory=pin_memory,
        )
    # Source datasets can use sparse/non-zero-based class IDs.  Training losses
    # receive this dense mapping, while datasets continue returning original IDs.
    train_labels_mapper = dataset_bundle.train_labels_mapper
    train_label_lookup = make_label_lookup_tensor(train_labels_mapper, args.device)

    # Frozen backbone parameters are deliberately omitted from optimizer state.
    trainable_model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # First create the optimizer for model/backbone parameters.
    if args.optim == "adamw":
        optim = torch.optim.AdamW(
            trainable_model_parameters,
            lr=args.lr,
            weight_decay=0.0,
        )
    elif args.optim == "adam":
        optim = torch.optim.Adam(
            trainable_model_parameters,
            lr=args.lr,
        )
    elif args.optim == "rmsprop":
        optim = torch.optim.RMSprop(
            trainable_model_parameters,
            lr=args.lr,
        )
    num_train_classes = len(dataset_bundle.train_labels_mapper)
    criterion, is_classification, miner, classifier_optim = make_training_loss_components(
        args=args,
        loss_name=args.loss,
        loss_params=getattr(args, "loss_params", {}),
        miner_name=args.miner,
        miner_params=getattr(args, "miner_params", {}),
        num_classes=num_train_classes,
        embedding_size=model.feat_dim,
    )
    warmup_criterion = None
    warmup_is_classification = False
    warmup_miner = None
    warmup_classifier_optim = None
    if uses_ssl_warmup_objective(ssl_config):
        warmup_criterion, warmup_is_classification, warmup_miner, warmup_classifier_optim = (
            make_training_loss_components(
                args=args,
                loss_name=args.warmup_loss,
                loss_params=args.warmup_loss_params,
                miner_name=args.warmup_miner,
                miner_params=args.warmup_miner_params,
                num_classes=num_train_classes,
                embedding_size=model.feat_dim,
            )
        )
    teacher_model = None
    # Stateful regularizers such as STML initialize their teacher only when the
    # regularized phase begins.
    regularizer_state = None

    # MetricsLogger mirrors values to TensorBoard and a CSV in the run folder.
    metrics_logger = utils.MetricsLogger(args.log_dir, args)
    # The checkpoint is temporary: it is used to restore the selected epoch and
    # removed after final evaluation because the run currently returns metrics.
    best_model_path = args.log_dir / "best_model.pth"
    final_train_loss = None
    test_precision = None
    test_map = None
    test_pacmap_coordinates = None
    test_pacmap_plot = None
    epoch0_test_precision = None
    epoch0_test_map = None
    best_precision = None
    best_map = None
    selected_epoch = -1
    train_loader = None
    loss_driven_train_loader = None
    loss_driven_sampling_rebuild_epoch = None
    tqdm_bar = None

    try:
        global_step = 0
        last_epoch = -1
        if not final_full_train:
            # Epoch -1 measures the pretrained/off-the-shelf embedding before
            # task-specific updates. It is also a valid initial checkpoint.
            valid_precision, valid_map, valid_per_class = utils.evaluate(
                model,
                valid_loader,
                "valid",
                device=args.device,
                return_per_class=True,
            )
            metrics_logger.log_eval(
                "valid",
                valid_precision,
                valid_map,
                step=0,
                epoch=-1,
                per_class_metrics=valid_per_class,
            )

            # Track both metrics for reporting, but only selection_metric decides
            # which model state is checkpointed and when patience is reset.
            patience = args.patience
            best_precision = valid_precision
            best_map = valid_map
            best_selection_value = get_selection_metric_value(args.selection_metric, valid_precision, valid_map)
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"Model selection metric: {args.selection_metric}. "
                f"Initial selected value: {best_selection_value:.6f}"
            )
        else:
            logger.info(f"Training final full-development model for exactly {args.epochs} epochs")

        if evaluate_test and final_full_train:
            epoch0_test_precision, epoch0_test_map, epoch0_test_per_class = utils.evaluate(
                model,
                test_loader,
                "test epoch 0 - no optimization",
                device=args.device,
                return_per_class=True,
            )
            metrics_logger.log_eval(
                "epoch0_test",
                epoch0_test_precision,
                epoch0_test_map,
                step=0,
                epoch=0,
                per_class_metrics=epoch0_test_per_class,
            )

        for num_epoch in range(args.epochs):
            last_epoch = num_epoch
            # Choose the data source for this epoch. During warmup only true
            # labels are used; afterward pseudo-label or regularization methods
            # own the loader lifecycle.
            if ssl_config.enabled and num_epoch < ssl_config.warmup_epochs:
                # Warmup loader contains only ground-truth labeled examples.
                train_loader = warmup_train_loader
            elif loss_driven_ssl:
                if semi_supervised.should_rebuild_on_epoch(
                    ssl_config.update_mode,
                    ssl_config.update_interval_epochs,
                    num_epoch,
                    loss_driven_sampling_rebuild_epoch,
                ):
                    if loss_driven_train_loader is not None:
                        utils.shutdown_dataloaders(loss_driven_train_loader)
                    sampling_embeddings = semi_supervised.extract_embeddings(
                        model=model,
                        dataset=dataset_bundle.train_dataset,
                        positions=loss_driven_train_dataset.positions,
                        device=args.device,
                        batch_size=ssl_config.embedding_batch_size,
                        num_workers=ssl_config.embedding_num_workers,
                        seed=args.seed + num_epoch,
                        start_method=args.dataloader_start_method,
                        desc=f"STML sampling embeddings - epoch {num_epoch}",
                        embedding_kind="stml_g",
                    )
                    loss_driven_train_loader = utils.make_stml_train_loader(
                        train_dataset=loss_driven_train_dataset,
                        sampling_embeddings=sampling_embeddings,
                        batch_size=args.batch_size,
                        neighbors_per_query=criterion.num_neighbors,
                        seed=args.seed + num_epoch,
                        num_workers=args.num_workers,
                        start_method=args.dataloader_start_method,
                        pin_memory=pin_memory,
                    )
                    loss_driven_sampling_rebuild_epoch = num_epoch
                train_loader = loss_driven_train_loader
            elif regularized_ssl:
                train_loader = regularizer.make_loader(
                    model=model,
                    train_dataset=dataset_bundle.train_dataset,
                    supervised_loader=warmup_train_loader,
                    device=args.device,
                    config=ssl_config,
                    batch_size=args.batch_size,
                    seed=args.seed + num_epoch,
                    num_workers=args.num_workers,
                    start_method=args.dataloader_start_method,
                    epoch=num_epoch,
                    log_dir=args.log_dir,
                )
            elif semi_supervised.should_rebuild_on_epoch(
                ssl_config.update_mode,
                ssl_config.update_interval_epochs,
                num_epoch,
                ssl_training_dataset_update_epoch,
            ):
                # Re-embed and pseudo-label with the latest model.  A new loader
                # is required because accepted pseudo-labels may have changed.
                if static_train_loader is not None:
                    utils.shutdown_dataloaders(static_train_loader)
                train_dataset = semi_supervised.build_ssl_training_dataset(
                    model=model,
                    train_dataset=dataset_bundle.train_dataset,
                    train_labels_mapper=train_labels_mapper,
                    device=args.device,
                    config=ssl_config,
                    split=ssl_split,
                    epoch=num_epoch,
                    start_method=args.dataloader_start_method,
                    diagnostics_tracker=pseudo_label_diagnostics,
                    log_dir=args.log_dir,
                )
                train_loader = utils.make_train_loader(
                    train_dataset,
                    args.batch_size,
                    args.sampler_m,
                    seed=args.seed + num_epoch,
                    length_before_new_iter=args.length_before_new_iter,
                    num_workers=args.num_workers,
                    start_method=args.dataloader_start_method,
                    persistent_workers=ssl_config.update_mode != "every_epoch",
                    pin_memory=pin_memory,
                )
                ssl_training_dataset_update_epoch = num_epoch
                if ssl_config.update_mode == "every_epoch":
                    static_train_loader = None
                else:
                    static_train_loader = train_loader
            else:
                if static_train_loader is None:
                    raise RuntimeError("SSL training loader was not built before reuse")
                train_loader = static_train_loader

            warmup_active = is_ssl_warmup_epoch(ssl_config, num_epoch)
            legacy_stml_active = loss_driven_ssl and not warmup_active
            regularization_active = regularized_ssl and not warmup_active
            active_criterion = warmup_criterion if warmup_active else criterion
            active_is_classification = warmup_is_classification if warmup_active else is_classification
            active_miner = warmup_miner if warmup_active else miner
            active_classifier_optim = warmup_classifier_optim if warmup_active else classifier_optim
            loss_phase = f"warmup ({args.warmup_loss})" if warmup_active else args.loss
            if regularization_active:
                loss_phase = f"{args.loss} + {regularizer.name} regularization"
            logger.info(f"Epoch {num_epoch}: training with {loss_phase}")
            teacher_model = initialize_stml_teacher_for_phase(
                teacher_model=teacher_model,
                student_model=model,
                active_criterion=active_criterion,
                device=args.device,
            )
            if regularization_active and regularizer_state is None:
                regularizer_state = regularizer.initialize_state(model, args.device)
            model.train()
            if teacher_model is not None:
                teacher_model.eval()
            # Accumulate detached scalar losses for epoch-level reporting while
            # global_step counts optimizer updates across all epochs.
            epoch_loss = 0.0
            num_batches = 0
            zero_loss_batches = 0
            epoch_miner_totals = {}
            tqdm_bar = tqdm(total=len(train_loader))
            epoch_timings = defaultdict(float)
            last_timing_log_batch = 0
            last_timing_totals = {}
            next_batch_wait_start = _now(args.device) if _timing_enabled(args) else None

            for batch_idx, batch in enumerate(train_loader):
                timing = _timing_enabled(args)

                data_ready_time = _now(args.device) if timing else None
                if timing and next_batch_wait_start is not None:
                    _add_time(epoch_timings, "data_wait", next_batch_wait_start, data_ready_time)
                if legacy_stml_active:
                    images, _, instance_ids = batch
                    if not isinstance(images, (list, tuple)) or len(images) != active_criterion.num_views:
                        raise ValueError(
                            f"STML batches must contain {active_criterion.num_views} augmented views per sample"
                    )
                    images = torch.cat(list(images), dim=0)
                    instance_ids = instance_ids.repeat(active_criterion.num_views).to(args.device, non_blocking=True)
                else:
                    supervised_batch, regularizer_batch = batch if regularization_active else (batch, None)
                    images, labels, sample_weights = unpack_training_batch(supervised_batch)
                    if regularization_active and getattr(regularizer, "name", None) == "hoffer_entropy":
                        hoffer_roles = torch.as_tensor(regularizer_batch[1])
                        hoffer_reference_count = int(hoffer_roles.bool().sum().item())
                        hoffer_unlabeled_count = int(hoffer_roles.numel() - hoffer_reference_count)
                        logger.debug(
                            "Hoffer combined training batch: "
                            f"epoch={num_epoch}, "
                            f"batch={batch_idx}, "
                            f"supervised_labeled_count={len(labels)}, "
                            f"regularizer_unlabeled_count={hoffer_unlabeled_count}, "
                            f"regularizer_reference_count={hoffer_reference_count}, "
                            f"regularizer_total_count={int(hoffer_roles.numel())}"
                        )

                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "unpack_batch", data_ready_time, t1)
                # Autocast reduces memory/compute cost while leaving parameters
                # and the optimizer responsible for their normal precision.
                with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
                    miner_outputs = None
                    supervised_loss = None
                    regularization_loss = None

                    if legacy_stml_active:
                        t0 = _now(args.device) if timing else None
                        student_g, student_f = model.forward_stml_cached(images, args.device)
                        with torch.no_grad():
                            teacher_g = teacher_model.forward_stml_teacher_cached(images, args.device)
                        t1 = _now(args.device) if timing else None
                        if timing:
                            _add_time(epoch_timings, "forward", t0, t1)

                        t0 = _now(args.device) if timing else None
                        loss = active_criterion(student_f, student_g, teacher_g, instance_ids)
                        t1 = _now(args.device) if timing else None
                        if timing:
                            _add_time(epoch_timings, "loss", t0, t1)

                    else:
                        t0 = _now(args.device) if timing else None
                        embeddings = utils.forward_model_inputs(
                            model,
                            images,
                            args.device,
                            use_cache=model_use_cache,
                        )
                        t1 = _now(args.device) if timing else None
                        if timing:
                            _add_time(epoch_timings, "forward", t0, t1)

                        t0 = _now(args.device) if timing else None
                        labels = map_training_labels(
                            labels,
                            train_label_lookup,
                            train_labels_mapper,
                            args.device,
                        )
                        sample_weights = sample_weights.to(args.device, non_blocking=True)
                        t1 = _now(args.device) if timing else None
                        if timing:
                            _add_time(epoch_timings, "label_weight_prep", t0, t1)

                        t0 = _now(args.device) if timing else None
                        if getattr(active_criterion, "supports_sample_weights", False):
                            supervised_loss = active_criterion(embeddings, labels, sample_weights=sample_weights)
                        elif not active_is_classification and active_miner is not None:
                            miner_outputs = active_miner(embeddings, labels)
                            supervised_loss = active_criterion(embeddings, labels, miner_outputs)
                        else:
                            supervised_loss = active_criterion(embeddings, labels)
                        t1 = _now(args.device) if timing else None
                        if timing:
                            _add_time(epoch_timings, "miner_and_loss", t0, t1)

                        if regularization_active:
                            t0 = _now(args.device) if timing else None
                            regularization_loss = regularizer.compute_loss(
                                student_model=model,
                                state=regularizer_state,
                                batch=regularizer_batch,
                                device=args.device,
                                timings=epoch_timings if timing else None,
                            )
                            loss = regularizer.combine_losses(supervised_loss, regularization_loss)
                            t1 = _now(args.device) if timing else None
                            if timing:
                                _add_time(epoch_timings, "regularization_loss", t0, t1)
                        else:
                            loss = supervised_loss

                t0 = _now(args.device) if timing else None
                loss_value = loss.detach().item()
                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "loss_item", t0, t1)

                t0 = _now(args.device) if timing else None
                loss.backward()
                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "backward", t0, t1)

                log_batch_diagnostics = _batch_diagnostics_enabled(args)
                t0 = _now(args.device) if timing and log_batch_diagnostics else None
                miner_diagnostics = utils.summarize_miner_outputs(miner_outputs)
                batch_diagnostics = None
                if log_batch_diagnostics:
                    batch_diagnostics = {
                        "train/zero_loss_batch": float(loss_value == 0.0),
                        "train/gradient_norm/model": utils.gradient_l2_norm(model.parameters()),
                        **utils.optimizer_learning_rates(optim, "model"),
                        **miner_diagnostics,
                    }
                    if active_is_classification:
                        batch_diagnostics["train/gradient_norm/criterion"] = utils.gradient_l2_norm(
                            active_criterion.parameters()
                        )
                        batch_diagnostics.update(utils.optimizer_learning_rates(active_classifier_optim, "criterion"))
                    batch_diagnostics["train/stml_active"] = float(legacy_stml_active)
                    batch_diagnostics["train/regularization_active"] = float(regularization_active)
                    if supervised_loss is not None:
                        batch_diagnostics["train/supervised_loss"] = supervised_loss.detach().item()
                    if regularization_loss is not None:
                        batch_diagnostics["train/regularization_loss"] = regularization_loss.detach().item()
                t1 = _now(args.device) if timing and log_batch_diagnostics else None
                if timing and log_batch_diagnostics:
                    _add_time(epoch_timings, "diagnostics", t0, t1)

                t0 = _now(args.device) if timing else None
                optim.step()
                optim.zero_grad()
                if active_is_classification:
                    active_classifier_optim.step()
                    active_classifier_optim.zero_grad()
                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "optimizer_step", t0, t1)

                t0 = _now(args.device) if timing else None
                if legacy_stml_active:
                    update_ema_teacher(
                        teacher_model,
                        model,
                        momentum=active_criterion.teacher_momentum,
                        excluded_parameter_prefixes=("fc.",),
                    )
                if regularization_active:
                    regularizer.after_optimizer_step(model, regularizer_state)
                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "post_step_hooks", t0, t1)

                t0 = _now(args.device) if timing else None
                metrics_logger.log_train_batch(
                    loss_value,
                    num_epoch,
                    global_step,
                    diagnostics=batch_diagnostics,
                )
                t1 = _now(args.device) if timing else None
                if timing:
                    _add_time(epoch_timings, "metrics_logging", t0, t1)

                epoch_loss += loss_value
                num_batches += 1
                zero_loss_batches += int(loss_value == 0.0)
                for name, value in miner_diagnostics.items():
                    epoch_miner_totals[name] = epoch_miner_totals.get(name, 0) + value
                global_step += 1
                tqdm_bar.set_description(f"loss = {loss_value:.5f}")
                tqdm_bar.update(1)

                if timing and (batch_idx + 1) % _timing_interval(args) == 0:
                    current_batch_count = batch_idx + 1
                    interval_batches = current_batch_count - last_timing_log_batch
                    interval_timings = {
                        name: total - last_timing_totals.get(name, 0.0)
                        for name, total in epoch_timings.items()
                    }
                    logger.info(
                        "Batch timing interval "
                        f"epoch={num_epoch} batch={current_batch_count}/{len(train_loader)} "
                        f"last_batches={interval_batches} "
                        f"phase={loss_phase}: "
                        + ", ".join(
                            f"{name}={total / interval_batches:.4f}s"
                            for name, total in sorted(interval_timings.items())
                        )
                    )
                    last_timing_log_batch = current_batch_count
                    last_timing_totals = dict(epoch_timings)
                if timing:
                    next_batch_wait_start = _now(args.device)
            tqdm_bar.close()
            tqdm_bar = None
            shutdown_epoch_train_loader(
                train_loader,
                warmup_train_loader,
                static_train_loader,
                loss_driven_train_loader,
            )
            gc.collect()
            if num_batches > 0:
                # The epoch metric is an unweighted mean of batch loss values.
                final_train_loss = epoch_loss / num_batches
                epoch_diagnostics = {
                    "train/zero_loss_batches": zero_loss_batches,
                    "train/zero_loss_fraction": zero_loss_batches / num_batches,
                }
                for name, total in epoch_miner_totals.items():
                    epoch_diagnostics[f"{name}_total"] = total
                    epoch_diagnostics[f"{name}_mean_per_batch"] = total / num_batches
                metrics_logger.log_train_epoch(
                    final_train_loss,
                    num_epoch,
                    global_step,
                    diagnostics=epoch_diagnostics,
                )
            if final_full_train:
                # No validation or early stopping is permitted in the final
                # fit; the HPO-selected duration determines the resulting model.
                selected_epoch = num_epoch
                continue

            # Validation runs after every epoch and supplies both early-stopping
            # decisions and intermediate values for Optuna pruning.
            cur_precision, cur_map, cur_per_class = utils.evaluate(
                model,
                valid_loader,
                f"valid - epoch {num_epoch:>2}",
                device=args.device,
                return_per_class=True,
            )
            metrics_logger.log_eval(
                "valid",
                cur_precision,
                cur_map,
                step=global_step,
                epoch=num_epoch,
                per_class_metrics=cur_per_class,
            )
            # Report the running best rather than only the current epoch when
            # the HPO objective is a "best_valid_*" metric.
            best_precision_for_report = max(best_precision, cur_precision)
            best_map_for_report = max(best_map, cur_map)
            maybe_report_to_optuna(
                optuna_trial=optuna_trial,
                metric=optuna_metric,
                epoch=num_epoch,
                train_loss=final_train_loss,
                valid_precision=cur_precision,
                valid_map=cur_map,
                best_precision=best_precision_for_report,
                best_map=best_map_for_report,
            )

            # Warmup epochs do not consume early-stopping patience because the
            # SSL method has not started using pseudo-labels yet.
            is_after_warmup = num_epoch >= ssl_config.warmup_epochs

            cur_selection_value = get_selection_metric_value(args.selection_metric, cur_precision, cur_map)
            # Keep independent maxima for the final report even when the chosen
            # checkpoint is selected by only one of these metrics.
            if cur_map > best_map:
                best_map = cur_map
            if cur_precision > best_precision:
                best_precision = cur_precision

            if cur_selection_value > best_selection_value:
                # Strict improvement replaces the selected checkpoint and
                # restarts patience.
                best_selection_value = cur_selection_value
                selected_epoch = num_epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            elif is_after_warmup:
                # Equal or worse selected metric consumes one patience unit.
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    # Restore immediately before leaving the loop; the load
                    # below also covers runs that finish all requested epochs.
                    model.load_state_dict(torch.load(best_model_path, weights_only=True))
                    break

        if not final_full_train:
            # Evaluate/report only the checkpoint selected on validation, never
            # the last epoch's potentially overfit model.
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        if evaluate_test:
            test_embeddings, test_labels = utils.extract_eval_embeddings(
                model,
                test_loader,
                "test",
                device=args.device,
            )
            test_precision, test_map, test_per_class = utils.evaluate_embeddings(
                test_embeddings,
                test_labels,
                name="test",
                return_per_class=True,
                dataset=dataset_bundle.test_dataset,
            )
            metrics_logger.log_eval(
                "test",
                test_precision,
                test_map,
                step=global_step,
                epoch=last_epoch,
                per_class_metrics=test_per_class,
            )
            if (
                getattr(args, "final_test_visualization", FINAL_TEST_VISUALIZATION_NONE)
                == FINAL_TEST_VISUALIZATION_PACMAP
            ):
                pacmap_artifacts = utils.write_pacmap_visualization(
                    test_embeddings,
                    test_labels,
                    output_dir=args.log_dir,
                    stem="test_pacmap",
                    title=f"{args.dataset} final test embeddings - PacMAP",
                    dataset=dataset_bundle.test_dataset,
                    dataset_name=args.dataset,
                )
                test_pacmap_coordinates = pacmap_artifacts["coordinates"]
                test_pacmap_plot = pacmap_artifacts["plot"]
                logger.info(f"PacMAP final test visualization written to {test_pacmap_plot}")
        else:
            logger.info("Skipping test evaluation for this run")
    finally:
        # Ensure file handles and TensorBoard writers close even when training,
        # evaluation, or an Optuna pruning decision raises an exception.
        if tqdm_bar is not None:
            tqdm_bar.close()
        shutdown_epoch_train_loader(
            train_loader,
            warmup_train_loader,
            static_train_loader,
            loss_driven_train_loader,
        )
        utils.shutdown_dataloaders(
            train_loader,
            static_train_loader,
            loss_driven_train_loader,
            warmup_train_loader,
            valid_loader,
            test_loader,
            getattr(regularizer, "_regularizer_loader", None),
        )
        gc.collect()
        metrics_logger.close()
        if precompute_frozen_features:
            feature_stats = {
                "enabled": True,
                "batch_size": get_frozen_feature_batch_size(args),
                "train_views": frozen_feature_train_views,
                "train_samples": (
                    precomputed_train_samples
                    if precomputed_train_samples is not None
                    else None if static_train_loader is None else len(static_train_loader.dataset)
                ),
                "valid_samples": None if valid_loader is None else len(valid_loader.dataset),
                "test_samples": None if test_loader is None else len(test_loader.dataset),
                "backbone": f"dinov2_vit{args.dino_size}14",
                "persistent_cache_enabled": bool(args.use_cache),
                "persistent_cache_dir": None if model.cache_dir is None else str(model.cache_dir),
            }
            write_json(args.log_dir / "frozen_feature_precompute_stats.json", feature_stats)
            logger.info(f"Frozen feature precompute stats: {feature_stats}")
            if args.use_cache:
                cache_stats = model.cache_stats()
                write_json(args.log_dir / "backbone_cache_stats.json", cache_stats)
                logger.info(f"Backbone cache stats: {cache_stats}")
        elif args.use_cache:
            cache_stats = model.cache_stats()
            write_json(args.log_dir / "backbone_cache_stats.json", cache_stats)
            logger.info(f"Backbone cache stats: {cache_stats}")
        if best_model_path.exists():
            os.remove(best_model_path)

    return TrainingResult(
        log_dir=args.log_dir,
        metrics_csv=args.log_dir / "metrics.csv",
        best_valid_precision_at_1=None if best_precision is None else float(best_precision),
        best_valid_mean_average_precision_at_r=None if best_map is None else float(best_map),
        test_precision_at_1=None if test_precision is None else float(test_precision),
        test_mean_average_precision_at_r=None if test_map is None else float(test_map),
        final_train_loss=None if final_train_loss is None else float(final_train_loss),
        last_epoch=last_epoch,
        selected_epoch=selected_epoch,
        global_step=global_step,
        epoch0_test_precision_at_1=None if epoch0_test_precision is None else float(epoch0_test_precision),
        epoch0_test_mean_average_precision_at_r=None if epoch0_test_map is None else float(epoch0_test_map),
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_mode=args.cv_mode if cv_fold is not None else None,
        cv_fold=cv_fold,
        test_pacmap_coordinates=test_pacmap_coordinates,
        test_pacmap_plot=test_pacmap_plot,
    )

def run_cross_validation(args, ssl_config, optuna_trial=None, optuna_metric=None):
    """Train each CV fold independently and aggregate fold-level metrics."""

    validate_run_args(args, ssl_config)
    # All fold directories live below one timestamped CV directory so their
    # partial and final aggregate summaries can be updated in place.
    cv_run_name = f"cv_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    cv_relative_dir = Path(args.save_dir) / cv_run_name
    cv_dir = Path("logs") / cv_relative_dir
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_index in range(args.cv_k):
        # Each fold gets a fresh namespace and model/training run. Only the
        # source arguments and immutable SSL config are shared.
        fold_args = copy.deepcopy(args)
        fold_args.cv_fold = fold_index
        fold_args.save_dir = cv_relative_dir / f"fold_{fold_index:02d}"
        result = run_training(
            fold_args,
            ssl_config,
            optuna_trial=None,
            optuna_metric=None,
            cv_fold=fold_index,
        )
        fold_results.append(result)
        # Rewrite the summary after every fold so interrupted CV still leaves
        # useful completed-fold results.
        write_cross_validation_summary(cv_dir, args, fold_results)
        # Optuna sees the aggregate of completed folds as an intermediate value
        # and may prune the trial before remaining folds are trained.
        maybe_report_cv_to_optuna(optuna_trial, optuna_metric, fold_results, fold_index)

    # The returned TrainingResult uses mean metrics, total optimization steps,
    # and the maximum last epoch reached among folds.
    aggregate = aggregate_cross_validation_result(cv_dir, args, fold_results)
    write_cross_validation_summary(cv_dir, args, fold_results, aggregate)
    return aggregate

def aggregate_cross_validation_result(cv_dir, args, fold_results):
    # Keep full fold dictionaries inside the aggregate result for JSON metadata
    # while exposing arithmetic means through the normal TrainingResult fields.
    fold_dicts = [result_to_dict(result) for result in fold_results]
    return TrainingResult(
        log_dir=cv_dir,
        metrics_csv=cv_dir / "cv_results.csv",
        best_valid_precision_at_1=mean_metric(fold_results, "best_valid_precision_at_1"),
        best_valid_mean_average_precision_at_r=mean_metric(
            fold_results,
            "best_valid_mean_average_precision_at_r",
        ),
        test_precision_at_1=mean_optional_metric(fold_results, "test_precision_at_1"),
        test_mean_average_precision_at_r=mean_optional_metric(fold_results, "test_mean_average_precision_at_r"),
        final_train_loss=mean_optional_metric(fold_results, "final_train_loss"),
        last_epoch=max(result.last_epoch for result in fold_results),
        selected_epoch=round(mean_metric(fold_results, "selected_epoch")),
        global_step=sum(result.global_step for result in fold_results),
        cv_k=args.cv_k,
        cv_mode=args.cv_mode,
        fold_results=fold_dicts,
    )

def mean_metric(results, attr):
    return float(sum(getattr(result, attr) for result in results) / len(results))

def mean_optional_metric(results, attr):
    values = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))

def write_cross_validation_summary(cv_dir, args, fold_results, aggregate=None):
    # During execution only completed folds are written. The final call adds a
    # synthetic "mean" row once all folds have completed.
    rows = [make_cv_summary_row(result) for result in fold_results]
    if aggregate is not None:
        rows.append(make_cv_summary_row(aggregate, fold="mean"))

    csv_path = cv_dir / "cv_results.csv"
    fieldnames = [
        "fold",
        "cv_k",
        "cv_mode",
        "log_dir",
        "metrics_csv",
        "best_valid_precision_at_1",
        "best_valid_mean_average_precision_at_r",
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
        "test_pacmap_coordinates",
        "test_pacmap_plot",
        "final_train_loss",
        "last_epoch",
        "selected_epoch",
        "global_step",
    ]
    # Rewriting is intentional: the file always represents the latest complete
    # view instead of requiring readers to deduplicate appended rows.
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        cv_dir / "cv_summary.json",
        {
            "args": namespace_to_dict(args),
            "completed_folds": len(fold_results),
            "cv_k": args.cv_k,
            "cv_mode": args.cv_mode,
            "folds": [result_to_dict(result) for result in fold_results],
            "aggregate": None if aggregate is None else result_to_dict(aggregate),
        },
    )

def make_cv_summary_row(result, fold=None):
    return {
        "fold": result.cv_fold if fold is None else fold,
        "cv_k": result.cv_k,
        "cv_mode": "" if result.cv_mode is None else result.cv_mode,
        "log_dir": str(result.log_dir),
        "metrics_csv": str(result.metrics_csv),
        "best_valid_precision_at_1": result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
        "test_precision_at_1": "" if result.test_precision_at_1 is None else result.test_precision_at_1,
        "test_mean_average_precision_at_r": ""
        if result.test_mean_average_precision_at_r is None
        else result.test_mean_average_precision_at_r,
        "test_pacmap_coordinates": ""
        if result.test_pacmap_coordinates is None
        else str(result.test_pacmap_coordinates),
        "test_pacmap_plot": "" if result.test_pacmap_plot is None else str(result.test_pacmap_plot),
        "final_train_loss": "" if result.final_train_loss is None else result.final_train_loss,
        "last_epoch": result.last_epoch,
        "selected_epoch": result.selected_epoch,
        "global_step": result.global_step,
    }

def maybe_report_cv_to_optuna(optuna_trial, metric, fold_results, fold_index):
    if optuna_trial is None or metric is None:
        return
    # Aggregate only folds completed so far. This gives the pruner an
    # increasingly representative estimate as cross-validation progresses.
    partial_result = aggregate_cross_validation_result(Path("."), make_cv_args_stub(fold_results), fold_results)
    value = getattr(partial_result, metric)
    if value is None:
        # Optional metrics such as test performance may be unavailable when
        # test evaluation is disabled during HPO.
        return
    optuna_trial.report(float(value), step=fold_index)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()

def make_cv_args_stub(fold_results):
    stub = argparse.Namespace()
    stub.cv_k = fold_results[0].cv_k
    stub.cv_mode = fold_results[0].cv_mode
    return stub

def validate_run_args(args, ssl_config):
    """Fail early on invalid combinations before allocating model resources."""

    normalize_backbone_tuning_args(args)
    if args.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}: {args.dataset}")
    utils.validate_dataset_protocol(args.dataset, args.dataset_protocol)
    utils.validate_cifar_imbalance_factor(args.dataset, args.cifar_imbalance_factor)
    utils.validate_cifar_balanced_fraction_protocol(
        dataset_name=args.dataset,
        dataset_protocol=args.dataset_protocol,
        train_fraction=args.cifar_train_fraction,
        test_fraction=args.cifar_test_fraction,
        imbalance_factor=args.cifar_imbalance_factor,
    )
    if args.dino_size not in {"s", "b", "l", "g"}:
        raise ValueError(f"dino_size must be one of ['s', 'b', 'l', 'g']: {args.dino_size}")
    if args.loss not in ALL_LOSSES:
        raise ValueError(f"loss must be one of {ALL_LOSSES}: {args.loss}")
    if args.miner not in ALL_MINERS:
        raise ValueError(f"miner must be one of {ALL_MINERS}: {args.miner}")
    for name in ("loss_params", "miner_params", "warmup_loss_params", "warmup_miner_params"):
        if not isinstance(getattr(args, name, {}), dict):
            raise ValueError(f"{name} must be a JSON object")
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    ssl_method = semi_supervised.get_method(ssl_config)
    regularized_ssl = ssl_method is not None and ssl_method.is_regularization_method
    regularizer = ssl_method.make_regularizer(ssl_config) if regularized_ssl else None
    stml_regularization = regularizer is not None and regularizer.name == "stml"
    validate_effective_miner_params(args.loss, args.miner, args.miner_params, "miner")
    if uses_ssl_warmup_objective(ssl_config):
        validate_warmup_loss_args(args)
    unlabeled_source, external_unlabeled_dir = resolve_unlabeled_source_args(args)
    external_unlabeled_filter = getattr(args, "external_unlabeled_filter", utils.EXTERNAL_UNLABELED_FILTER_NONE)
    if external_unlabeled_filter not in utils.EXTERNAL_UNLABELED_FILTERS:
        raise ValueError(
            "external_unlabeled_filter must be one of "
            f"{utils.EXTERNAL_UNLABELED_FILTERS}: {external_unlabeled_filter}"
        )
    if getattr(args, "compcars_min_model_images", 100) <= 0:
        raise ValueError("compcars_min_model_images must be positive")
    if (
        getattr(args, "compcars_strict_paper_counts", False)
        and external_unlabeled_filter != utils.EXTERNAL_UNLABELED_FILTER_COMPCARS_STML_PAPER
    ):
        raise ValueError("compcars_strict_paper_counts requires external_unlabeled_filter='compcars_stml_paper'")
    if unlabeled_source == "split" and external_unlabeled_dir is not None:
        raise ValueError(
            "external_unlabeled_dir requires unlabeled_source='external' or 'split_and_external'"
        )
    if unlabeled_source == "split" and external_unlabeled_filter != utils.EXTERNAL_UNLABELED_FILTER_NONE:
        raise ValueError("external_unlabeled_filter requires unlabeled_source='external' or 'split_and_external'")
    if unlabeled_source != "split" and not is_supervised_mode(args):
        if external_unlabeled_dir is None:
            raise ValueError(f"unlabeled_source={unlabeled_source!r} requires external_unlabeled_dir")
        if not Path(external_unlabeled_dir).is_dir():
            raise ValueError(f"external_unlabeled_dir does not exist: {external_unlabeled_dir}")
        if not ssl_config.enabled:
            raise ValueError("External unlabeled data requires --mode ssl with an enabled SSL config")
    if args.loss == "STMLLoss":
        if not loss_driven_ssl:
            raise ValueError("STMLLoss requires an SSL config with method='stml'")
        if args.miner != "no_miner":
            raise ValueError("STMLLoss requires miner='no_miner' because it does not consume labels")
        if args.batch_size < 2:
            raise ValueError("STMLLoss requires batch_size >= 2")
        if args.use_cache:
            raise ValueError("STMLLoss requires stochastic multi-view augmentation and cannot use backbone caching")
        if args.stml_g_dim is not None and args.stml_g_dim <= 0:
            raise ValueError("stml_g_dim must be positive when set")
        try:
            stml_loss = metric_losses.STMLLoss(**args.loss_params)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameters for loss STMLLoss: {args.loss_params}") from exc
        if args.batch_size % stml_loss.num_neighbors != 0:
            raise ValueError("STMLLoss requires batch_size to be divisible by loss_params.num_neighbors")
    elif loss_driven_ssl:
        raise ValueError(f"SSL method {ssl_config.method!r} requires loss='STMLLoss'")
    if regularizer is not None:
        if args.loss == "STMLLoss":
            raise ValueError("regularized mode requires a supervised loss, not STMLLoss")
        regularizer.validate_run_args(args)
    try:
        utils.normalize_device_name(args.device)
    except ValueError as exc:
        raise ValueError(f"{exc}: {args.device}") from exc
    if args.optim not in {"adamw", "adam", "rmsprop"}:
        raise ValueError(f"optim must be 'adam' or 'rmsprop': {args.optim}")
    if args.mode not in {"supervised", "ssl"}:
        raise ValueError("mode must be 'supervised' or 'ssl'")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if getattr(args, "frozen_feature_batch_size", None) is not None and args.frozen_feature_batch_size <= 0:
        raise ValueError("frozen_feature_batch_size must be positive when set")
    if get_frozen_feature_train_views(args) <= 0:
        raise ValueError("frozen_feature_train_views must be positive")
    if getattr(args, "debug_batch_timing_interval", 5) <= 0:
        raise ValueError("debug_batch_timing_interval must be positive")
    if args.length_before_new_iter is not None and args.length_before_new_iter < args.batch_size:
        raise ValueError("length_before_new_iter must be at least batch_size when set")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.classifier_lr <= 0:
        raise ValueError("classifier_lr must be positive")
    if args.sampler_m <= 0:
        raise ValueError("sampler_m must be positive")
    if args.epochs < 0 or (args.epochs == 0 and not getattr(args, "final_full_train", False)):
        raise ValueError("epochs must be positive, except a final full-train run may use zero selected epochs")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if args.cv_k <= 0:
        raise ValueError("cv_k must be positive")
    if getattr(args, "final_full_train", False) and args.cv_k != 1:
        raise ValueError("A final full-development fit requires cv_k=1")
    if args.cv_mode not in utils.CV_MODES:
        raise ValueError(f"cv_mode must be one of {utils.CV_MODES}: {args.cv_mode}")
    if args.val_mode not in utils.VAL_MODES:
        raise ValueError(f"val_mode must be one of {utils.VAL_MODES}: {args.val_mode}")
    if args.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}: {args.selection_metric}")
    final_test_visualization = getattr(args, "final_test_visualization", FINAL_TEST_VISUALIZATION_NONE)
    if final_test_visualization not in FINAL_TEST_VISUALIZATION_MODES:
        raise ValueError(f"final_test_visualization must be one of {FINAL_TEST_VISUALIZATION_MODES}")
    if args.feat_dim is not None and args.feat_dim <= 0:
        raise ValueError("feat_dim must be positive when set")
    regularizer_provides_projection = (
        loss_driven_ssl
        or (regularizer is not None and regularizer.provides_trainable_projection_without_feat_dim)
    )
    if args.backbone_tuning == BACKBONE_TUNING_FROZEN and args.feat_dim is None and not regularizer_provides_projection:
        raise ValueError("backbone_tuning='frozen' requires feat_dim so a trainable projection head remains")
    if args.use_cache and args.backbone_tuning != BACKBONE_TUNING_FROZEN:
        raise ValueError(
            "use_cache requires backbone_tuning='frozen' because tuned backbone features are not stable"
        )
    utils.validate_dataloader_settings(
        device=args.device,
        num_workers=args.num_workers,
        ssl_embedding_num_workers=ssl_config.embedding_num_workers if ssl_config.enabled else 0,
        start_method=args.dataloader_start_method,
    )

def validate_effective_miner_params(loss_name, miner_name, miner_params, param_name):
    """Validate miner params only when the training loop will actually use the miner."""

    if miner_name == "no_miner" or loss_name in CLASSIFICATION_LOSSES or loss_name == "STMLLoss":
        return
    try:
        validate_named_miner_params(miner_name, miner_params)
    except ValueError as exc:
        raise ValueError(f"Invalid parameters for {param_name} {miner_name}: {exc}") from exc


def validate_warmup_loss_args(args):
    """Validate the supervised objective used during labeled-only SSL warmup."""

    if args.warmup_loss not in ALL_LOSSES or args.warmup_loss == "STMLLoss":
        raise ValueError("warmup_loss must be a standard supervised loss")
    if args.warmup_miner not in ALL_MINERS:
        raise ValueError(f"warmup_miner must be one of {ALL_MINERS}: {args.warmup_miner}")
    if args.warmup_loss in CLASSIFICATION_LOSSES and args.warmup_miner != "no_miner":
        raise ValueError("classification warmup_loss requires warmup_miner='no_miner'")
    validate_effective_miner_params(
        args.warmup_loss,
        args.warmup_miner,
        args.warmup_miner_params,
        "warmup_miner",
    )


def validate_named_miner_params(name, params=None):
    """Return validated/normalized miner constructor params."""

    params = dict(params or {})
    if name == "BatchEasyHardMiner":
        return validate_batch_easy_hard_miner_params(params)
    return params


def validate_batch_easy_hard_miner_params(params):
    params = dict(params or {})
    pos_strategy = validate_batch_easy_hard_strategy(
        params.get("pos_strategy", BATCH_EASY_HARD_DEFAULT_POS_STRATEGY),
        "pos_strategy",
    )
    neg_strategy = validate_batch_easy_hard_strategy(
        params.get("neg_strategy", BATCH_EASY_HARD_DEFAULT_NEG_STRATEGY),
        "neg_strategy",
    )

    if pos_strategy == "semihard" and neg_strategy == "semihard":
        raise ValueError("pos_strategy and neg_strategy cannot both be 'semihard'")
    if pos_strategy == "semihard" and neg_strategy == "all":
        raise ValueError("neg_strategy cannot be 'all' when pos_strategy is 'semihard'")
    if pos_strategy == "all" and neg_strategy == "semihard":
        raise ValueError("pos_strategy cannot be 'all' when neg_strategy is 'semihard'")

    for range_name in BATCH_EASY_HARD_RANGE_PARAMS:
        if range_name in params:
            params[range_name] = validate_batch_easy_hard_allowed_range(params[range_name], range_name)
    return params


def validate_batch_easy_hard_strategy(value, name):
    if value not in BATCH_EASY_HARD_MINER_STRATEGIES:
        allowed = sorted(BATCH_EASY_HARD_MINER_STRATEGIES)
        raise ValueError(f"{name} must be one of {allowed}: {value!r}")
    return value


def validate_batch_easy_hard_allowed_range(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be null or a two-value range")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, Real)
        or not isinstance(upper, Real)
    ):
        raise ValueError(f"{name} bounds must be numeric")
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return (lower, upper)


def make_loss(args, num_classes=None, embedding_size=None):
    """Construct the selected loss with HPO-resolved constructor parameters."""

    return make_named_loss(
        name=args.loss,
        params=getattr(args, "loss_params", {}),
        num_classes=num_classes,
        embedding_size=embedding_size,
    )


def make_named_loss(name, params=None, num_classes=None, embedding_size=None):
    """Construct a selected loss name with explicit constructor parameters."""

    params = dict(params or {})
    loss_class = get_loss_class(name)
    logger.info(f"Loss: {name}, params={params}")
    try:
        if name in CLASSIFICATION_LOSSES:
            if num_classes is None or embedding_size is None:
                raise ValueError(f"{name} requires num_classes and embedding_size")
            return loss_class(num_classes, embedding_size, **params)
        return loss_class(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for loss {name}: {params}") from exc


def make_training_loss_components(
    args,
    loss_name,
    loss_params,
    miner_name,
    miner_params,
    num_classes,
    embedding_size,
):
    """Build a criterion, optional miner, and optional criterion optimizer."""

    criterion = make_named_loss(
        name=loss_name,
        params=loss_params,
        num_classes=num_classes,
        embedding_size=embedding_size,
    ).to(args.device)
    is_classification = loss_name in CLASSIFICATION_LOSSES
    classifier_optim = None
    if is_classification:
        classifier_optim = make_optimizer(
            args,
            criterion.parameters(),
            lr=args.classifier_lr,
        )
        miner = None
    else:
        miner = make_named_miner(miner_name, miner_params)
    return criterion, is_classification, miner, classifier_optim


def make_optimizer(args, parameters, lr):
    if args.optim == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    if args.optim == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if args.optim == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    raise ValueError(f"Unknown optimizer: {args.optim}")


def make_stml_teacher(student_model):
    """Create STML's teacher with the student's current backbone and a fresh g head."""

    teacher_model = copy.deepcopy(student_model)
    torch.nn.init.orthogonal_(teacher_model.embedding_g.weight)
    torch.nn.init.zeros_(teacher_model.embedding_g.bias)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    return teacher_model


def initialize_stml_teacher_for_phase(teacher_model, student_model, active_criterion, device):
    """Create the EMA teacher when, and only when, the STML phase begins."""

    if teacher_model is not None or not getattr(active_criterion, "requires_stml_embeddings", False):
        return teacher_model
    teacher_model = make_stml_teacher(student_model).to(device)
    logger.info("Initialized EMA teacher from the fully warmed-up student")
    return teacher_model


@torch.no_grad()
def update_ema_teacher(teacher_model, student_model, momentum, excluded_parameter_prefixes=()):
    """Update teacher state by EMA, copying only non-floating counters."""

    if not 0 <= momentum < 1:
        raise ValueError("teacher momentum must be in [0, 1)")
    teacher_parameters = dict(teacher_model.named_parameters())
    for name, student_parameter in student_model.named_parameters():
        if name.startswith(tuple(excluded_parameter_prefixes)):
            continue
        teacher_parameter = teacher_parameters[name]
        teacher_parameter.lerp_(student_parameter.detach(), 1 - momentum)
    teacher_buffers = dict(teacher_model.named_buffers())
    for name, student_buffer in student_model.named_buffers():
        teacher_buffer = teacher_buffers[name]
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.lerp_(student_buffer.detach(), 1 - momentum)
        else:
            teacher_buffer.copy_(student_buffer.detach())


def get_loss_class(name):
    """Return a project-local or pytorch-metric-learning loss class."""

    if name in metric_losses.LOSS_REGISTRY:
        return metric_losses.LOSS_REGISTRY[name]
    return getattr(losses, name)

def unpack_training_batch(batch):
    """Normalize two- or three-item training batches to include confidence."""

    if len(batch) == 2:
        images, labels = batch
        sample_weights = torch.ones(len(labels), dtype=torch.float32)
        return images, labels, sample_weights
    if len(batch) == 3:
        images, labels, sample_weights = batch
        return images, labels, sample_weights.to(dtype=torch.float32)
    raise ValueError(f"Training batches must contain images, labels, and optional confidence; got {len(batch)} items")

def make_miner(args):
    """Construct the selected miner with HPO-resolved constructor parameters."""

    return make_named_miner(args.miner, getattr(args, "miner_params", {}))


def make_named_miner(name, params=None):
    """Construct a selected miner name with explicit constructor parameters."""

    if name == "no_miner":
        return None
    raw_params = dict(params or {})
    params = validate_named_miner_params(name, raw_params)
    logger.info(f"Miner: {name}, params={params}")
    try:
        return getattr(miners, name)(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for miner {name}: {raw_params}") from exc

def make_supervised_split_config(ssl_config):
    # Preserve label-selection settings and seed so the supervised baseline
    # sees the same labeled subset, while removing all pseudo-label behavior.
    config_dict = ssl_config.to_dict()
    config_dict.update(
        {
            "method": "none",
            "update_mode": "once",
            "update_interval_epochs": 1,
            "warmup_epochs": 0,
            "confidence_threshold": 0.0,
            "method_params": {},
        }
    )
    config = semi_supervised.SemiSupervisedConfig(**config_dict)
    semi_supervised.validate_ssl_config(config)
    return config

def maybe_report_to_optuna(
    optuna_trial,
    metric,
    epoch,
    train_loss,
    valid_precision,
    valid_map,
    best_precision,
    best_map,
):
    """Report an intermediate objective value and honor Optuna pruning."""

    if optuna_trial is None or metric is None:
        # Normal non-HPO training uses this same path but has nothing to report
        # and cannot be pruned.
        return
    value_by_metric = {
        "best_valid_precision_at_1": best_precision,
        "best_valid_mean_average_precision_at_r": best_map,
        "final_train_loss": train_loss,
    }
    # Test metrics are unavailable until training finishes, so only metrics
    # meaningful during epochs appear in this intermediate mapping.
    value = value_by_metric.get(metric)
    if value is None:
        return
    # Optuna compares this step/value pair with other trials according to the
    # configured pruning algorithm.
    optuna_trial.report(float(value), step=epoch)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()

def write_run_config(args, ssl_config):
    write_json(
        args.log_dir / "run_config.json",
        {
            "args": namespace_to_dict(args),
            "ssl_config": ssl_config.to_dict(),
        },
    )
