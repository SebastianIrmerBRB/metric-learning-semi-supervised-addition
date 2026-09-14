"""GPU-resident feature batching for frozen-backbone training.

Frozen-backbone runs train a small projection head on precomputed features, so a
step costs far more in Python and host-to-device traffic than it does in GPU
math. Holding the whole feature view in device memory lets each step build its
batch from two ``index_select`` kernels, with no DataLoader, no collate workers,
and no per-step transfer.

The sampler is unchanged: it still runs on the CPU, but only once per epoch. Its
entire index stream is uploaded in a single transfer and then sliced per step, so
batch composition matches the DataLoader path exactly.

Besides the plain supervised view, the SSL wrapper datasets that are pure
gathers (``RelabeledSubset``, single-view ``UnlabeledSubset`` and
``HofferReferenceDataset``) and the sequential evaluation pass resolve to a
resident view too. Each one keeps the item tuple its DataLoader path produced,
so the consumers downstream are untouched.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

# Item layouts a resident view can be asked to produce.
RESIDENT_MODE_TRAIN = "train"
RESIDENT_MODE_EVAL = "eval"


@dataclass(frozen=True)
class ResidentBatchView:
    """A device-resident feature view plus how to rebuild its item tuple.

    ``features`` is ordered so row ``i`` is dataset item ``i``. ``columns`` are
    per-row tensors gathered with the same indices, and ``assemble`` turns the
    gathered features, gathered columns, and the batch's own indices back into
    the exact tuple the dataset's ``__getitem__``/collate path yielded.
    """

    features: torch.Tensor
    columns: tuple
    assemble: Callable
    desc: str = ""


def resolve_precomputed_feature_view(dataset):
    """Return ``(features, orig_labels, sample_weights)`` for a feature dataset.

    Returns ``None`` when the dataset is not backed by precomputed features, or
    is wrapped in something whose per-item work cannot be replayed as a gather
    (a stochastic view wrapper, for example). Callers fall back to the ordinary
    DataLoader in that case.
    """

    # Imported here because utils imports this module.
    from utils import PrecomputedBackboneFeatureDataset

    indices = None
    while isinstance(dataset, Subset):
        subset_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        indices = subset_indices if indices is None else indices.index_select(0, subset_indices)
        dataset = dataset.dataset

    if not isinstance(dataset, PrecomputedBackboneFeatureDataset):
        return None

    features = dataset.features
    orig_labels = torch.as_tensor(dataset.orig_labels, dtype=torch.long)
    sample_weights = dataset.sample_weights
    if indices is not None:
        features = features.index_select(0, indices)
        orig_labels = orig_labels.index_select(0, indices)
        if sample_weights is not None:
            sample_weights = sample_weights.index_select(0, indices)
    return features, orig_labels, sample_weights


def _ones_like_rows(features):
    return torch.ones(len(features), dtype=torch.float32)


def _resolve_plain_view(dataset, mode):
    """The bare precomputed view, in either the training or evaluation layout."""

    base = resolve_precomputed_feature_view(dataset)
    if base is None:
        return None
    features, orig_labels, sample_weights = base
    if mode == RESIDENT_MODE_EVAL:
        # extract_eval_embeddings unpacks exactly two fields per batch.
        return ResidentBatchView(
            features=features,
            columns=(orig_labels,),
            assemble=lambda gathered, columns, indices: (gathered, columns[0]),
            desc="precomputed features",
        )
    if sample_weights is None:
        sample_weights = _ones_like_rows(features)
    return ResidentBatchView(
        features=features,
        columns=(orig_labels, sample_weights.to(dtype=torch.float32)),
        assemble=lambda gathered, columns, indices: (gathered, columns[0], columns[1]),
        desc="precomputed features",
    )


def _resolve_relabeled_subset(dataset, mode):
    """``RelabeledSubset`` is a pure gather: positions, labels, confidences."""

    base = resolve_precomputed_feature_view(dataset.dataset)
    if base is None:
        return None
    features = base[0]
    positions = torch.as_tensor(dataset.positions, dtype=torch.long)
    features = features.index_select(0, positions)
    # The wrapped dataset's own label is deliberately ignored: pseudo-labeled
    # rows must expose their predicted label, exactly as __getitem__ does.
    orig_labels = torch.as_tensor(dataset.orig_labels, dtype=torch.long)
    confidences = torch.as_tensor(dataset.confidences, dtype=torch.float32)
    if dataset.return_indices:
        assemble = lambda gathered, columns, indices: (
            gathered,
            columns[0],
            columns[1],
            indices,
        )
    else:
        assemble = lambda gathered, columns, indices: (gathered, columns[0], columns[1])
    return ResidentBatchView(
        features=features,
        columns=(orig_labels, confidences),
        assemble=assemble,
        desc="relabeled subset",
    )


def _resolve_unlabeled_subset(dataset, mode):
    """Single-view ``UnlabeledSubset``; multi-view draws stay on the item path."""

    if int(dataset.num_views) != 1:
        return None
    from .ssl.config import UNLABELED_TARGET

    base = resolve_precomputed_feature_view(dataset.dataset)
    if base is None:
        return None
    positions = torch.as_tensor(dataset.positions, dtype=torch.long)
    features = base[0].index_select(0, positions)

    def assemble(gathered, columns, indices):
        targets = torch.full(
            (len(indices),),
            int(UNLABELED_TARGET),
            dtype=torch.long,
            device=gathered.device,
        )
        return gathered, targets, columns[0]

    return ResidentBatchView(
        features=features,
        columns=(positions,),
        assemble=assemble,
        desc="unlabeled subset",
    )


def _resolve_hoffer_reference(dataset, mode):
    """Joint unlabeled/reference index space flattens into one resident view."""

    base = resolve_precomputed_feature_view(dataset.dataset)
    if base is None:
        return None
    unlabeled = torch.as_tensor(dataset.unlabeled_positions, dtype=torch.long)
    labeled = torch.as_tensor(dataset.labeled_positions, dtype=torch.long)
    # Item i < num_unlabeled maps to unlabeled_positions[i], the rest follow in
    # order, so concatenating in that order makes row i item i.
    features = torch.cat(
        (
            base[0].index_select(0, unlabeled),
            base[0].index_select(0, labeled),
        )
    )
    roles = torch.cat(
        (
            torch.full((len(unlabeled),), int(dataset.UNLABELED_ROLE), dtype=torch.long),
            torch.full((len(labeled),), int(dataset.REFERENCE_ROLE), dtype=torch.long),
        )
    )
    return ResidentBatchView(
        features=features,
        columns=(roles,),
        assemble=lambda gathered, columns, indices: (gathered, columns[0]),
        desc="hoffer reference",
    )


def resolve_resident_batch_view(dataset, mode=RESIDENT_MODE_TRAIN):
    """Return a ``ResidentBatchView`` for ``dataset``, or ``None`` to keep the loader."""

    from .ssl.data import (
        HofferReferenceDataset,
        RelabeledSubset,
        UnlabeledSubset,
    )

    if isinstance(dataset, RelabeledSubset):
        return _resolve_relabeled_subset(dataset, mode)
    if isinstance(dataset, UnlabeledSubset):
        return _resolve_unlabeled_subset(dataset, mode)
    if isinstance(dataset, HofferReferenceDataset):
        return _resolve_hoffer_reference(dataset, mode)
    return _resolve_plain_view(dataset, mode)


def feature_view_nbytes(features, sample_weights):
    total = features.numel() * features.element_size()
    if sample_weights is not None:
        total += sample_weights.numel() * sample_weights.element_size()
    return int(total)


class GpuResidentFeatureLoader:
    """Yield precomputed-feature batches straight from device memory.

    Implements the part of the ``DataLoader`` contract that ``train_epoch`` and
    ``extract_eval_embeddings`` depend on: ``len`` in batches, and iteration
    over batches shaped like the dataset's items. Every tensor is already on the
    training device, so the step does no host synchronization to build inputs.

    The item layout comes from the ``ResidentBatchView``; passing the legacy
    ``(features, orig_labels, sample_weights)`` triple builds the plain
    supervised layout instead.
    """

    def __init__(
        self,
        features,
        orig_labels=None,
        sample_weights=None,
        device="cpu",
        batch_size=1,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        seed=None,
        dataset=None,
    ):
        if isinstance(features, ResidentBatchView):
            view = features
        else:
            if sample_weights is None:
                sample_weights = _ones_like_rows(features)
            view = ResidentBatchView(
                features=features,
                columns=(orig_labels, sample_weights),
                assemble=lambda gathered, columns, indices: (
                    gathered,
                    columns[0],
                    columns[1],
                ),
                desc="precomputed features",
            )

        if (sampler is None) == (batch_sampler is None):
            raise ValueError("provide exactly one of sampler or batch_sampler")
        if view.features.ndim not in {2, 3}:
            raise ValueError("precomputed features must be a 2D or 3D tensor")
        for column in view.columns:
            if len(column) != len(view.features):
                raise ValueError("resident view columns must align with the feature rows")

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.drop_last = bool(drop_last)
        # evaluate() and the retrieval-device resolver read the loader's dataset
        # for query/gallery partitions and class names.
        self.dataset = dataset
        self.assemble = view.assemble
        self.desc = view.desc
        # Sibling streams (the SSL regularizer loaders) reuse this budget
        # instead of threading the residency args through every regularizer.
        self.residency_max_bytes = None

        self.features = view.features.to(self.device, dtype=torch.float32).contiguous()
        self.columns = tuple(
            column.to(self.device).contiguous() for column in view.columns
        )
        # Retained for the plain supervised layout's stats and for callers that
        # still read these two directly.
        self.orig_labels = self.columns[0] if self.columns else None
        self.sample_weights = self.columns[1] if len(self.columns) > 1 else None

        self.num_views = self.features.shape[1] if self.features.ndim == 3 else 1
        self.generator = None
        if self.num_views > 1:
            self.generator = torch.Generator(device=self.device)
            if seed is not None:
                self.generator.manual_seed(int(seed))
        self._length = self._count_batches()

    def _index_stream(self):
        """Materialize this epoch's sampled indices as one CPU tensor of batches."""

        if self.batch_sampler is not None:
            batches = [torch.as_tensor(batch, dtype=torch.long) for batch in self.batch_sampler]
            batches = [batch for batch in batches if len(batch) == self.batch_size]
            if not batches:
                return torch.empty((0, self.batch_size), dtype=torch.long)
            return torch.stack(batches)

        flat = torch.as_tensor(list(self.sampler), dtype=torch.long)
        usable = (len(flat) // self.batch_size) * self.batch_size
        batches = flat[:usable].view(-1, self.batch_size)
        if self.drop_last or usable == len(flat):
            return batches
        # The ragged tail is kept in its own list entry so __iter__ can yield it
        # without padding, matching DataLoader(drop_last=False).
        return [batches, flat[usable:]]

    def _count_batches(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        sampled = len(self.sampler)
        if self.drop_last:
            return sampled // self.batch_size
        return (sampled + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self._length

    @property
    def num_samples(self):
        """Samples in the resident view; there is no ``dataset`` to measure."""

        return len(self.features)

    def _gather(self, indices):
        features = self.features.index_select(0, indices)
        if self.num_views > 1:
            view = torch.randint(
                self.num_views,
                (len(indices),),
                device=self.device,
                generator=self.generator,
            )
            features = features[torch.arange(len(indices), device=self.device), view]
        columns = tuple(column.index_select(0, indices) for column in self.columns)
        return self.assemble(features, columns, indices)

    def __iter__(self):
        stream = self._index_stream()
        tail = None
        if isinstance(stream, list):
            stream, tail = stream
        # One transfer per epoch instead of one per batch.
        stream = stream.to(self.device, non_blocking=True)
        for row in range(len(stream)):
            yield self._gather(stream[row])
        if tail is not None and len(tail):
            yield self._gather(tail.to(self.device, non_blocking=True))

    def shutdown(self):
        """No worker processes to tear down; present for loader-shutdown paths."""

    def storage_nbytes(self):
        total = self.features.numel() * self.features.element_size()
        for column in self.columns:
            total += column.numel() * column.element_size()
        return int(total)


def resident_loader_device(loader):
    """The CUDA device a resident loader batches on, or ``None`` for a DataLoader.

    SSL regularizer streams follow whatever the supervised loader resolved to,
    so opting into GPU residency once covers every stream in the step.
    """

    if not isinstance(loader, GpuResidentFeatureLoader):
        return None
    return loader.device if loader.device.type == "cuda" else None


def resident_loader_max_bytes(loader):
    """The device budget the supervised resident loader was built under."""

    return getattr(loader, "residency_max_bytes", None)


def loader_sample_count(loader):
    """Samples a loader draws from, for DataLoader and GPU-resident loaders alike."""

    if loader is None:
        return None
    if isinstance(loader, GpuResidentFeatureLoader):
        return loader.num_samples
    return len(loader.dataset)


def loader_features_are_resident(loader):
    """Whether a loader serves features from memory instead of the feature mmap."""

    if loader is None:
        return False
    if isinstance(loader, GpuResidentFeatureLoader):
        # Stronger than host residency: the whole view sits in device memory.
        return True
    return bool(getattr(loader.dataset, "is_resident", False))


def try_make_gpu_resident_loader(
    dataset,
    device,
    batch_size,
    sampler=None,
    batch_sampler=None,
    drop_last=False,
    seed=None,
    max_bytes=None,
    desc="train",
    mode=RESIDENT_MODE_TRAIN,
):
    """Build a GPU-resident loader, or return ``None`` to keep the DataLoader.

    Returning ``None`` is the normal outcome for image datasets, for wrappers
    whose per-item work is not a plain gather, and for feature views too large
    for the device budget; the caller keeps its existing loader.
    """

    if torch.device(device).type != "cuda":
        logger.info(f"GPU-resident {desc} batching skipped: device is {device}")
        return None

    view = resolve_resident_batch_view(dataset, mode)
    if view is None:
        logger.info(
            f"GPU-resident {desc} batching skipped: dataset is not backed by "
            "precomputed features, or its items are not a plain gather"
        )
        return None

    required = view.features.numel() * view.features.element_size()
    for column in view.columns:
        required += column.numel() * column.element_size()
    if max_bytes is not None and required > max_bytes:
        logger.warning(
            f"GPU-resident {desc} batching skipped: view needs {required / 1e9:.2f} GB, "
            f"above the {max_bytes / 1e9:.2f} GB device budget"
        )
        return None

    free_bytes, _total = torch.cuda.mem_get_info(torch.device(device))
    # Leave headroom for activations, gradients, and the optimizer state.
    if required > free_bytes * 0.5:
        logger.warning(
            f"GPU-resident {desc} batching skipped: view needs {required / 1e9:.2f} GB "
            f"but only {free_bytes / 1e9:.2f} GB is free on {device}"
        )
        return None

    loader = GpuResidentFeatureLoader(
        view,
        device=device,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        drop_last=drop_last,
        seed=seed,
        dataset=dataset,
    )
    loader.residency_max_bytes = max_bytes
    logger.info(
        f"GPU-resident {desc} batching: {len(view.features)} samples "
        f"({required / 1e9:.3f} GB, {view.desc}) held on {device}, "
        f"{len(loader)} batches/epoch, no DataLoader in the step"
    )
    return loader
