"""Coordinate source-indexed mmap and in-process frozen-feature dataset views."""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Subset


_CACHE_VERSION = 2
_SOURCE_ARG_NAMES = (
    "dataset",
    "dataset_protocol",
    "device",
    "cifar_imbalance_factor",
    "cifar_train_fraction",
    "cifar_test_fraction",
    "seed",
    "data_split_seed",
    "cv_k",
    "cv_mode",
    "val_mode",
    "final_full_train",
    "unlabeled_source",
    "external_unlabeled_dir",
    "external_unlabeled_filter",
    "compcars_min_model_images",
    "compcars_strict_paper_counts",
    "compcars_paper_threshold_calibration",
)
# Split-affecting arguments that only join the key once they are set, so adding
# one of them does not invalidate every cache written before it existed.
_OPTIONAL_SOURCE_ARG_NAMES = ("holdout_val_ratio", "ablation_config", "ablation_variant")
_DATASET_SIGNATURE_ATTRS = (
    "indices",
    "positions",
    "orig_labels",
    "labels",
    "targets",
    "sample_weights",
    "confidences",
    "return_indices",
    "paths",
    "image_names",
    "samples",
    "imgs",
    "model_keys",
    "root",
    "dataset_root",
    "split",
    "train",
    "classes",
    "query_indices",
    "gallery_indices",
)


def _update_digest(digest, value):
    """Add a stable, type-aware representation of metadata to ``digest``."""

    digest.update(f"{type(value).__module__}.{type(value).__qualname__}:".encode("utf-8"))
    if value is None:
        digest.update(b"none;")
        return
    if isinstance(value, (str, bytes, Path, bool, int, float)):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b";")
        return
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        try:
            digest.update(tensor.numpy().tobytes())
        except TypeError:
            digest.update(repr(tensor.tolist()).encode("utf-8"))
        digest.update(b";")
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        if array.dtype.hasobject:
            _update_digest(digest, array.tolist())
        else:
            digest.update(array.tobytes())
        digest.update(b";")
        return
    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        digest.update(b";")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _update_digest(digest, item)
        digest.update(b";")
        return
    if isinstance(value, (set, frozenset)):
        for item in sorted(value, key=repr):
            _update_digest(digest, item)
        digest.update(b";")
        return
    digest.update(repr(value).encode("utf-8"))
    digest.update(b";")


def _update_dataset_digest(digest, dataset, seen):
    """Fingerprint sample selection and source identity without loading images."""

    object_id = id(dataset)
    if object_id in seen:
        _update_digest(digest, ("dataset_ref", seen[object_id]))
        return
    seen[object_id] = len(seen)
    _update_digest(
        digest,
        ("dataset_type", type(dataset).__module__, type(dataset).__qualname__, len(dataset)),
    )
    for attr_name in _DATASET_SIGNATURE_ATTRS:
        if hasattr(dataset, attr_name):
            _update_digest(digest, (attr_name, getattr(dataset, attr_name)))

    child_datasets = getattr(dataset, "datasets", None)
    if child_datasets is not None:
        for child_dataset in child_datasets:
            _update_dataset_digest(digest, child_dataset, seen)
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        _update_dataset_digest(digest, child_dataset, seen)


def _find_input_transform(dataset, use_feature_transform, seen=None):
    """Return the transform that produces the images sent to the backbone."""

    if seen is None:
        seen = set()
    object_id = id(dataset)
    if object_id in seen:
        return None
    seen.add(object_id)

    if use_feature_transform:
        feature_transform = getattr(dataset, "feature_transform", None)
        if feature_transform is not None:
            return feature_transform
    else:
        transform = getattr(dataset, "transform", None)
        if transform is not None:
            return transform

    child_datasets = getattr(dataset, "datasets", None)
    if child_datasets is not None:
        for child_dataset in child_datasets:
            transform = _find_input_transform(
                child_dataset,
                use_feature_transform,
                seen,
            )
            if transform is not None:
                return transform
    child_dataset = getattr(dataset, "dataset", None)
    if child_dataset is not None and child_dataset is not dataset:
        return _find_input_transform(
            child_dataset,
            use_feature_transform,
            seen,
        )
    return None


def _feature_batch_size(args):
    configured = getattr(args, "frozen_feature_batch_size", None)
    return int(args.batch_size if configured is None else configured)


def _model_cache_identity(model):
    explicit_identity = getattr(model, "frozen_feature_cache_identity", None)
    if callable(explicit_identity):
        explicit_identity = explicit_identity()
    if explicit_identity is not None:
        return explicit_identity

    dino_size = getattr(model, "dino_size", None)
    if dino_size is None:
        # Unknown model weights cannot be assumed equivalent across trials.
        return None
    return (
        type(model).__module__,
        type(model).__qualname__,
        f"dinov2_vit{dino_size}14",
        getattr(model, "backbone_tuning", None),
        str(getattr(model, "cache_dir", None)),
    )


def make_frozen_feature_cache_key(
    args,
    model,
    dataset,
    *,
    require_feature_transform,
    use_feature_transform,
    num_views,
):
    """Return a stable key for raw frozen-backbone features, when supported."""

    explicit_identity = _model_cache_identity(model)
    if explicit_identity is None:
        return None

    digest = hashlib.sha256()
    _update_digest(digest, ("frozen_feature_cache_version", _CACHE_VERSION))
    _update_digest(digest, ("backbone", explicit_identity))
    _update_digest(
        digest,
        tuple((name, getattr(args, name, None)) for name in _SOURCE_ARG_NAMES),
    )
    optional_source_args = tuple(
        (name, getattr(args, name, None))
        for name in _OPTIONAL_SOURCE_ARG_NAMES
        if getattr(args, name, None) is not None
    )
    if optional_source_args:
        _update_digest(digest, optional_source_args)
    _update_digest(
        digest,
        (
            "precompute_options",
            bool(require_feature_transform),
            bool(use_feature_transform),
            int(num_views),
        ),
    )
    input_transform = _find_input_transform(
        dataset,
        use_feature_transform=use_feature_transform,
    )
    if input_transform is None and use_feature_transform:
        # Evaluation datasets generally expose only their active deterministic
        # transform rather than a separate feature_transform.
        input_transform = _find_input_transform(
            dataset,
            use_feature_transform=False,
        )
    _update_digest(digest, ("input_transform", repr(input_transform)))
    if not use_feature_transform:
        # Stochastic augmented views also depend on the extraction RNG/worker
        # schedule. Deterministic feature transforms deliberately omit these so
        # HPO batch-size changes can still reuse the exact same feature tensor.
        _update_digest(
            digest,
            (
                "stochastic_precompute",
                getattr(args, "seed", None),
                _feature_batch_size(args),
                getattr(args, "num_workers", None),
                getattr(args, "dataloader_start_method", None),
            ),
        )
    _update_dataset_digest(digest, dataset, seen={})
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenFeatureIndexSpec:
    """Persistent matrix namespace and row mapping for one dataset view."""

    key: str
    row_indices: np.ndarray
    capacity: int


def _unwrap_indexed_dataset(dataset):
    """Map a selection view to stable row indices in its root dataset."""

    rows = np.arange(len(dataset), dtype=np.int64)
    current = dataset
    seen = set()
    while True:
        object_id = id(current)
        if object_id in seen:
            return None
        seen.add(object_id)

        child = getattr(current, "dataset", None)
        if child is None or child is current:
            return current, rows

        if isinstance(current, Subset):
            selector = np.asarray(current.indices, dtype=np.int64)
        else:
            positions = getattr(current, "positions", None)
            if positions is not None and len(positions) == len(current):
                selector = np.asarray(positions, dtype=np.int64)
            else:
                # An unknown wrapper may alter pixels even when it preserves
                # length. Fall back to an exact view-local matrix rather than
                # aliasing its rows with the child dataset.
                return None

        if selector.ndim != 1 or len(selector) != len(current):
            return None
        if np.any((rows < 0) | (rows >= len(selector))):
            return None
        rows = selector[rows]
        current = child


def make_frozen_feature_index_spec(
    args,
    model,
    dataset,
    *,
    require_feature_transform,
    use_feature_transform,
    num_views,
):
    """Build a source-indexed mmap key without reading or hashing image pixels."""

    model_identity = _model_cache_identity(model)
    if model_identity is None:
        return None

    num_views = int(num_views)
    deterministic = bool(use_feature_transform) and num_views == 1
    unwrapped = _unwrap_indexed_dataset(dataset) if deterministic else None
    if unwrapped is None:
        # Stochastic views depend on extraction order. Keep them in an exact
        # dataset-local matrix rather than sharing rows across different views.
        exact_key = make_frozen_feature_cache_key(
            args,
            model,
            dataset,
            require_feature_transform=require_feature_transform,
            use_feature_transform=use_feature_transform,
            num_views=num_views,
        )
        if exact_key is None:
            return None
        row_count = len(dataset) * num_views
        return FrozenFeatureIndexSpec(
            key=f"exact:{exact_key}",
            row_indices=np.arange(row_count, dtype=np.int64),
            capacity=row_count,
        )

    source_dataset, source_rows = unwrapped
    input_transform = _find_input_transform(
        dataset,
        use_feature_transform=use_feature_transform,
    )
    if input_transform is None:
        input_transform = _find_input_transform(
            dataset,
            use_feature_transform=False,
        )

    digest = hashlib.sha256()
    _update_digest(digest, ("indexed_frozen_feature_cache_version", _CACHE_VERSION))
    _update_digest(digest, ("backbone", model_identity))
    _update_digest(
        digest,
        (
            "precompute_options",
            bool(use_feature_transform),
            num_views,
        ),
    )
    _update_digest(digest, ("input_transform", repr(input_transform)))
    _update_dataset_digest(digest, source_dataset, seen={})
    return FrozenFeatureIndexSpec(
        key=f"source:{digest.hexdigest()}",
        row_indices=source_rows,
        capacity=len(source_dataset),
    )


def _precomputed_dataset_nbytes(dataset):
    total = 0
    storage_nbytes = getattr(dataset, "storage_nbytes", None)
    if callable(storage_nbytes):
        return int(storage_nbytes())
    for attr_name in ("features", "sample_weights"):
        value = getattr(dataset, attr_name, None)
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return int(total)


class FrozenFeatureDatasetCache:
    """Thread-safe CPU feature cache shared by all trials in one HPO study.

    A study that varies seed, fold, label budget, or any other key component
    materializes a separate resident view per combination, so an unbounded cache
    grows for the lifetime of the process. ``max_bytes`` caps the total and
    evicts least-recently-used entries; ``None`` keeps every entry, which is the
    historical behavior.
    """

    def __init__(self, max_bytes=None):
        if max_bytes is not None and int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive when set")
        self._datasets = OrderedDict()
        self._entry_bytes = {}
        self._inflight = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._evictions = 0
        self._bytes = 0
        self._max_bytes = None if max_bytes is None else int(max_bytes)

    def _evict_to_budget(self, protected_key):
        """Drop least-recently-used entries until the cache fits its budget.

        The entry just stored is protected so a view larger than the whole
        budget is still returned to its caller rather than evicted immediately.
        """

        if self._max_bytes is None:
            return
        for key in list(self._datasets):
            if self._bytes <= self._max_bytes:
                return
            if key == protected_key:
                continue
            self._datasets.pop(key)
            evicted_bytes = self._entry_bytes.pop(key, 0)
            self._bytes -= evicted_bytes
            self._evictions += 1
            logger.info(
                f"Evicted in-memory frozen features (key={key[:12]}, "
                f"{evicted_bytes / 1e9:.2f} GB) to stay within the "
                f"{self._max_bytes / 1e9:.2f} GB cache budget"
            )

    def get_or_compute(self, key, compute, desc):
        while True:
            with self._lock:
                cached = self._datasets.get(key)
                if cached is not None:
                    self._datasets.move_to_end(key)
                    self._hits += 1
                    logger.info(
                        f"Reusing in-memory frozen features for {desc}: "
                        f"{len(cached)} samples (key={key[:12]})"
                    )
                    return cached
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    self._misses += 1
                    owner = True
                else:
                    self._waits += 1
                    owner = False
            if owner:
                break
            logger.info(
                f"Waiting for another HPO worker to materialize {desc} "
                f"(key={key[:12]})"
            )
            event.wait()

        try:
            dataset = compute()
        except BaseException:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()
            raise

        with self._lock:
            self._datasets[key] = dataset
            self._entry_bytes[key] = _precomputed_dataset_nbytes(dataset)
            self._bytes += self._entry_bytes[key]
            self._evict_to_budget(protected_key=key)
            self._inflight.pop(key, None)
            event.set()
        logger.info(
            f"Stored in-memory frozen features for HPO reuse: {desc}, "
            f"{len(dataset)} samples (key={key[:12]})"
        )
        return dataset

    def stats(self):
        with self._lock:
            return {
                "entries": len(self._datasets),
                "hits": self._hits,
                "misses": self._misses,
                "waits": self._waits,
                "evictions": self._evictions,
                "bytes": self._bytes,
                "max_bytes": self._max_bytes,
            }
