"""Dataset views, combined loaders, and graph-aware SSL samplers."""

import json
import math

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

import utils
from ssl_config import UNLABELED_TARGET


class RelabeledSubset(Dataset):
    """Dataset view containing true-labeled and accepted pseudo-labeled samples.

    ``orig_labels`` are returned by ``__getitem__`` because the main training
    loop applies the shared original-to-mapped label dictionary.  ``labels``
    stores the dense mapped labels needed by MPerClassSampler. The third item
    returned by ``__getitem__`` is an optional confidence consumed only by
    confidence-aware losses.
    """

    def __init__(self, dataset, positions, orig_labels, mapped_labels, confidences=None):
        if confidences is None:
            confidences = np.ones(len(positions), dtype=np.float32)
        if not (len(positions) == len(orig_labels) == len(mapped_labels) == len(confidences)):
            raise ValueError("positions, orig_labels, mapped_labels, and confidences must have the same length")
        self.dataset = dataset
        # positions chooses which samples are visible through this view. The two
        # label arrays stay aligned with that exact order.
        self.positions = np.asarray(positions, dtype=np.int64)
        self.orig_labels = [int(label) for label in orig_labels]
        # MPerClassSampler reads this attribute directly without calling
        # __getitem__, so it receives dense true/pseudo labels here.
        self.labels = [int(label) for label in mapped_labels]
        self.confidences = np.asarray(confidences, dtype=np.float32)
        if not np.all(np.isfinite(self.confidences)):
            raise ValueError("confidences must be finite")
        if np.any((self.confidences < 0) | (self.confidences > 1)):
            raise ValueError("confidences must be in [0, 1]")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        # Ignore the label returned by the wrapped dataset because pseudo-labeled
        # samples must expose their predicted label instead of the hidden truth.
        image = self.dataset[int(self.positions[index])][0]
        return image, self.orig_labels[index], self.confidences[index]


class UnlabeledSubset(Dataset):
    """Dataset view that intentionally hides all labels from loss-driven SSL."""

    def __init__(self, dataset, positions, num_views=1):
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)
        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = int(self.positions[index])
        views = [self.dataset[position][0] for _ in range(self.num_views)]
        image_or_views = views[0] if self.num_views == 1 else views
        return image_or_views, UNLABELED_TARGET, position


class LRMLGraphDataset(Dataset):
    """Expose graph nodes by their graph index for LRML Laplacian regularization.

    Item ``i`` corresponds to graph node ``i`` so the indices produced by
    ``LRMLGraphBatchSampler`` line up with the rows of the precomputed neighbor
    graph and the symmetric adjacency. With cached frozen backbones, the source
    dataset uses the deterministic feature transform or precomputed backbone
    features; otherwise it follows the active training transform. Only the node
    index is returned because labels stay hidden from the regularizer.
    """

    def __init__(self, dataset, positions):
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        image = self.dataset[int(self.positions[index])][0]
        return image, index


class HofferReferenceDataset(Dataset):
    """Joint index space over unlabeled samples and labeled reference candidates.

    Items ``[0, num_unlabeled)`` map to unlabeled training positions, items
    ``[num_unlabeled, num_unlabeled + num_labeled)`` to labeled reference
    candidates for the Hoffer & Ailon entropy regularizer. ``__getitem__``
    returns the transformed image plus a 0/1 flag marking reference items so
    ``compute_loss`` can split the collated batch. Class labels stay hidden:
    the entropy term never reads them, only the per-class candidate grouping
    inside the batch sampler does.
    """

    UNLABELED_ROLE = 0
    REFERENCE_ROLE = 1
    # Backward-compatible aliases for older direct tests/diagnostics. The
    # current entropy-only regularizer treats all labeled anchors as references.
    LABELED_QUERY_ROLE = REFERENCE_ROLE
    COMPARISON_ROLE = REFERENCE_ROLE

    def __init__(self, dataset, unlabeled_positions, labeled_positions):
        self.dataset = dataset
        self.unlabeled_positions = np.asarray(unlabeled_positions, dtype=np.int64)
        self.labeled_positions = np.asarray(labeled_positions, dtype=np.int64)
        self.num_unlabeled = len(self.unlabeled_positions)
        self.num_labeled = len(self.labeled_positions)

    def __len__(self):
        return self.num_unlabeled + self.num_labeled

    def __getitem__(self, index):
        index = int(index)
        if index < self.num_unlabeled:
            image = self.dataset[int(self.unlabeled_positions[index])][0]
            return image, self.UNLABELED_ROLE
        image = self.dataset[int(self.labeled_positions[index - self.num_unlabeled])][0]
        return image, self.REFERENCE_ROLE


class CombinedTrainingLoader:
    """Pair every supervised batch with one regularizer batch."""

    def __init__(self, supervised_loader, regularizer_loader):
        if len(supervised_loader) == 0:
            raise ValueError("regularized training requires at least one supervised batch")
        if len(regularizer_loader) == 0:
            raise ValueError("regularized training requires at least one regularizer batch")
        self.supervised_loader = supervised_loader
        self.regularizer_loader = regularizer_loader
        self.cycles_per_epoch = int(math.ceil(len(supervised_loader) / len(regularizer_loader)))
        self._regularizer_iterator = None

    def __len__(self):
        return len(self.supervised_loader)

    def shutdown(self, include_persistent=False):
        if include_persistent or not getattr(self.regularizer_loader, "persistent_workers", False):
            utils.shutdown_dataloaders(self._regularizer_iterator)
            utils.shutdown_dataloaders(self.regularizer_loader)

    def __iter__(self):
        regularizer_iterator = None
        try:
            regularizer_iterator = iter(self.regularizer_loader)
            self._regularizer_iterator = regularizer_iterator

            for supervised_batch in self.supervised_loader:
                try:
                    regularizer_batch = next(regularizer_iterator)
                except StopIteration:
                    regularizer_iterator = iter(self.regularizer_loader)
                    self._regularizer_iterator = regularizer_iterator
                    regularizer_batch = next(regularizer_iterator)

                yield supervised_batch, regularizer_batch
        finally:
            if (
                regularizer_iterator is not None
                and not getattr(self.regularizer_loader, "persistent_workers", False)
            ):
                utils.shutdown_dataloaders(regularizer_iterator)
            self._regularizer_iterator = None


class LRMLGraphBatchSampler(torch.utils.data.Sampler):
    """Build batches that keep each sampled graph node close to its neighbors."""

    def __init__(self, neighbor_indices, batch_size, seed):
        if batch_size < 2:
            raise ValueError("LRML batch_size must be at least 2")
        self.batch_size = int(batch_size)
        self.generator = utils.make_torch_generator(seed)
        self.set_neighbors(neighbor_indices)

    def set_neighbors(self, neighbor_indices):
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        if neighbor_indices.ndim != 2:
            raise ValueError("LRML neighbor_indices must be a matrix")
        if len(neighbor_indices) < 2:
            raise ValueError("LRML regularization requires at least two graph samples")
        if neighbor_indices.shape[1] == 0:
            raise ValueError("LRML graph must contain at least one neighbor per node")

        self.neighbor_indices = torch.as_tensor(neighbor_indices, dtype=torch.long)
        self.num_samples = int(len(neighbor_indices))
        self.neighbors_per_query = min(self.batch_size - 1, int(neighbor_indices.shape[1]))
        self.queries_per_batch = max(1, self.batch_size // (self.neighbors_per_query + 1))

    def __iter__(self):
        query_order = torch.randperm(self.num_samples, generator=self.generator)
        for start in range(0, self.num_samples, self.queries_per_batch):
            batch_indices = []
            seen = set()
            query_indices = query_order[start : start + self.queries_per_batch]
            for query_index in query_indices.tolist():
                candidates = [query_index]
                candidates.extend(self.neighbor_indices[query_index, : self.neighbors_per_query].tolist())
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    batch_indices.append(candidate)
            yield batch_indices

    def __len__(self):
        return int(math.ceil(self.num_samples / self.queries_per_batch))


class WeightedGraphBatchSampler(torch.utils.data.Sampler):
    """Sample graph-centered node batches from a weighted symmetric adjacency."""

    def __init__(self, adjacency, batch_size, seed):
        if batch_size < 2:
            raise ValueError("graph regularization requires batch_size >= 2")
        self.batch_size = int(batch_size)
        self.generator = utils.make_torch_generator(seed)
        self.set_graph(adjacency)

    def set_graph(self, adjacency):
        adjacency = adjacency.tocsr()
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("weighted graph adjacency must be square")
        if adjacency.shape[0] < 2:
            raise ValueError("graph regularization requires at least two graph samples")

        self.adjacency = adjacency
        self.num_samples = int(adjacency.shape[0])
        row_counts = np.diff(adjacency.indptr)
        self.query_nodes = np.flatnonzero(row_counts > 0).astype(np.int64)
        if len(self.query_nodes) == 0:
            raise ValueError("graph regularization requires at least one edge")

    def __iter__(self):
        query_order = torch.randperm(len(self.query_nodes), generator=self.generator)
        max_neighbors = self.batch_size - 1
        for order_index in query_order.tolist():
            query = int(self.query_nodes[order_index])
            start = int(self.adjacency.indptr[query])
            end = int(self.adjacency.indptr[query + 1])
            neighbors = self.adjacency.indices[start:end]
            if len(neighbors) > max_neighbors:
                selected_offsets = torch.randperm(len(neighbors), generator=self.generator)[:max_neighbors].numpy()
                selected_neighbors = neighbors[selected_offsets]
            else:
                selected_offsets = torch.randperm(len(neighbors), generator=self.generator).numpy()
                selected_neighbors = neighbors[selected_offsets]

            batch_indices = [query]
            seen = {query}
            for neighbor in selected_neighbors.tolist():
                neighbor = int(neighbor)
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                batch_indices.append(neighbor)
            yield batch_indices

    def __len__(self):
        return int(len(self.query_nodes))


class HofferReferenceBatchSampler(torch.utils.data.Sampler):
    """Emit batches of unlabeled indices plus labeled references per class.

    References are drawn uniformly within each class and resampled for every
    batch, as in Hoffer & Ailon (2018). Unlabeled samples are visited once per
    epoch in a shuffled order; each batch appends ``num_compared + 1`` freshly
    drawn reference indices per class (or per sampled class subset when
    ``max_reference_classes`` caps the count). Indices refer to a
    ``HofferReferenceDataset``, whose reference candidates start at
    ``num_unlabeled``.
    """

    def __init__(
        self,
        num_unlabeled,
        class_candidates,
        unlabeled_per_batch,
        seed,
        max_reference_classes=None,
        num_compared=0,
        class_labels=None,
        unlabeled_positions=None,
        labeled_positions=None,
    ):
        if num_unlabeled <= 0:
            raise ValueError("hoffer_entropy requires at least one unlabeled sample")
        if unlabeled_per_batch <= 0:
            raise ValueError("hoffer_entropy unlabeled_per_batch must be positive")
        if len(class_candidates) < 2:
            raise ValueError("hoffer_entropy requires labeled candidates from at least two classes")
        self.class_candidates = []
        for candidates in class_candidates:
            candidates = np.asarray(candidates, dtype=np.int64)
            if len(candidates) == 0:
                raise ValueError("every class needs at least one labeled reference candidate")
            self.class_candidates.append(candidates)
        self.num_unlabeled = int(num_unlabeled)
        self.unlabeled_per_batch = int(unlabeled_per_batch)
        self.num_classes = len(self.class_candidates)
        self.num_compared = int(num_compared)
        if self.num_compared < 0:
            raise ValueError("hoffer_entropy num_compared must be non-negative")
        self.references_per_class = self.num_compared + 1
        self.class_labels = None if class_labels is None else np.asarray(class_labels, dtype=np.int64)
        if self.class_labels is not None and len(self.class_labels) != self.num_classes:
            raise ValueError("hoffer_entropy class_labels must align with class_candidates")
        self.unlabeled_positions = (
            None if unlabeled_positions is None else np.asarray(unlabeled_positions, dtype=np.int64)
        )
        if self.unlabeled_positions is not None and len(self.unlabeled_positions) != self.num_unlabeled:
            raise ValueError("hoffer_entropy unlabeled_positions must align with num_unlabeled")
        self.labeled_positions = None if labeled_positions is None else np.asarray(labeled_positions, dtype=np.int64)
        if max_reference_classes is None:
            self.reference_classes_per_batch = self.num_classes
        else:
            self.reference_classes_per_batch = int(max_reference_classes)
            if not 2 <= self.reference_classes_per_batch <= self.num_classes:
                raise ValueError(
                    "hoffer_entropy max_reference_classes must be in "
                    f"[2, {self.num_classes}], got {self.reference_classes_per_batch}"
                )
        self.references_per_batch = self.reference_classes_per_batch * self.references_per_class
        self.generator = utils.make_torch_generator(seed)

    def __iter__(self):
        unlabeled_order = torch.randperm(self.num_unlabeled, generator=self.generator)
        for batch_number, start in enumerate(range(0, self.num_unlabeled, self.unlabeled_per_batch)):
            unlabeled_indices = unlabeled_order[start : start + self.unlabeled_per_batch].tolist()
            batch_indices = list(unlabeled_indices)
            if self.reference_classes_per_batch < self.num_classes:
                class_order = torch.randperm(self.num_classes, generator=self.generator)
                class_ids = class_order[: self.reference_classes_per_batch].tolist()
            else:
                class_ids = list(range(self.num_classes))
            reference_records = []
            for class_id in class_ids:
                candidates = self.class_candidates[class_id]
                for draw_index in range(self.references_per_class):
                    choice = int(torch.randint(len(candidates), (1,), generator=self.generator))
                    reference_index = int(candidates[choice])
                    batch_indices.append(reference_index)
                    reference_records.append(
                        self._make_reference_debug_record(reference_index, class_id, draw_index)
                    )
            unlabeled_positions = self._debug_unlabeled_positions(unlabeled_indices)
            reference_classes = [
                int(self.class_labels[int(class_id)]) if self.class_labels is not None else int(class_id)
                for class_id in class_ids
            ]
            logger.debug(
                "Hoffer reference batch: "
                f"batch={batch_number}, "
                f"unlabeled_count={len(batch_indices) - len(reference_records)}, "
                f"reference_count={len(reference_records)}, "
                f"reference_classes={reference_classes}, "
                f"unlabeled_joint_indices={unlabeled_indices}, "
                f"unlabeled_train_positions={unlabeled_positions}, "
                f"references={json.dumps(reference_records, sort_keys=True)}"
            )
            yield batch_indices

    def _debug_unlabeled_positions(self, unlabeled_indices):
        if self.unlabeled_positions is None:
            return None
        return [int(self.unlabeled_positions[int(index)]) for index in unlabeled_indices]

    def _make_reference_debug_record(self, reference_index, class_id, draw_index):
        labeled_offset = int(reference_index) - self.num_unlabeled
        record = {
            "class_slot": int(class_id),
            "draw": int(draw_index),
            "joint_index": int(reference_index),
            "labeled_offset": labeled_offset,
        }
        if self.class_labels is not None:
            record["class_label"] = int(self.class_labels[int(class_id)])
        if self.labeled_positions is not None and 0 <= labeled_offset < len(self.labeled_positions):
            record["train_position"] = int(self.labeled_positions[labeled_offset])
        return record

    def __len__(self):
        return int(math.ceil(self.num_unlabeled / self.unlabeled_per_batch))
