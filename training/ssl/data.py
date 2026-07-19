"""Dataset views, combined loaders, and graph-aware SSL samplers."""

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import utils
from .config import UNLABELED_TARGET


@dataclass(frozen=True)
class GraphBatchNodeIndex:
    """Dataset index carrying one graph batch's local edge map."""

    node_id: int
    edge_indices: tuple = ()


class RelabeledSubset(Dataset):
    """Dataset view containing true-labeled and accepted pseudo-labeled samples.

    ``orig_labels`` are returned by ``__getitem__`` because the main training
    loop applies the shared original-to-mapped label dictionary.  ``labels``
    stores the dense mapped labels needed by MPerClassSampler. The third item
    returned by ``__getitem__`` is an optional confidence consumed only by
    confidence-aware losses.
    """

    def __init__(
        self,
        dataset,
        positions,
        orig_labels,
        mapped_labels,
        confidences=None,
        return_indices=False,
        labeled_count=None,
    ):
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
        self.return_indices = bool(return_indices)
        if labeled_count is None:
            labeled_count = len(positions)
        self.labeled_count = int(labeled_count)
        if not 0 <= self.labeled_count <= len(self.positions):
            raise ValueError("labeled_count must be between zero and the dataset size")
        # Relabeling keeps known-label samples first and accepted pseudo-labels
        # second. Expose that provenance explicitly so two-stream samplers do
        # not have to infer it from confidence values or predicted labels.
        self.labeled_indices = np.arange(self.labeled_count, dtype=np.int64)
        self.unlabeled_indices = np.arange(
            self.labeled_count,
            len(self.positions),
            dtype=np.int64,
        )
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
        item = (image, self.orig_labels[index], self.confidences[index])
        if self.return_indices:
            # This is the stable row in the labeled memory bank, not a source-
            # dataset index. MPerClassSampler supplies this same view index.
            return (*item, int(index))
        return item


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
    ``GraphEdgeBatchSampler`` line up with the rows of the precomputed neighbor
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
        if isinstance(index, GraphBatchNodeIndex):
            node_id = int(index.node_id)
            image = self.dataset[int(self.positions[node_id])][0]
            return image, node_id, index.edge_indices
        image = self.dataset[int(self.positions[index])][0]
        return image, index


class LRMLEdgeDataset(Dataset):
    """Expose each undirected LRML edge once for ordinary shuffled loading.

    ``edge_index`` follows PyG's ``[2, num_directed_edges]`` convention and is
    expected to contain both directions. Dataset items are canonical global
    graph-node pairs; endpoint images are fetched exactly once later by
    :func:`collate_lrml_edge_batch`.
    """

    def __init__(self, edge_index, num_nodes):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("lrml edge_index must have shape [2, num_directed_edges]")
        if edge_index.shape[1] == 0:
            raise ValueError("lrml graph regularization requires at least one edge")
        self.num_nodes = int(num_nodes)
        if self.num_nodes < 2:
            raise ValueError("lrml graph regularization requires at least two nodes")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes:
            raise ValueError("lrml edge_index refers to a node outside the graph")

        canonical_edges = torch.sort(edge_index.t().contiguous(), dim=1).values
        if torch.any(canonical_edges[:, 0] == canonical_edges[:, 1]):
            raise ValueError("lrml edge_index must not contain self loops")
        edges, direction_counts = torch.unique(
            canonical_edges,
            dim=0,
            sorted=True,
            return_counts=True,
        )
        if torch.any(direction_counts != 2):
            raise ValueError(
                "lrml edge_index must contain each undirected edge in both directions once"
            )
        self.edge_index = edge_index.contiguous()
        self.edges = edges.contiguous()

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, index):
        return self.edges[int(index)]


def collate_lrml_edge_batch(batch, node_dataset):
    """Fetch unique endpoints and build a local PyG edge index for an edge batch."""

    if len(batch) == 0:
        raise ValueError("lrml edge collation requires at least one edge")
    edges = torch.stack(
        [torch.as_tensor(edge, dtype=torch.long) for edge in batch],
        dim=0,
    )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("lrml edge dataset items must have shape [2]")

    unique_node_ids = []
    local_index_by_node = {}
    local_edges = []
    for left_node, right_node in edges.tolist():
        local_pair = []
        for node_id in (int(left_node), int(right_node)):
            if node_id < 0 or node_id >= len(node_dataset):
                raise ValueError("lrml edge refers to a node outside the node dataset")
            local_index = local_index_by_node.get(node_id)
            if local_index is None:
                local_index = len(unique_node_ids)
                local_index_by_node[node_id] = local_index
                unique_node_ids.append(node_id)
            local_pair.append(local_index)
        local_edges.append(local_pair)

    node_items = [node_dataset[node_id] for node_id in unique_node_ids]
    images = [item[0] for item in node_items]
    return (
        default_collate(images),
        torch.as_tensor(unique_node_ids, dtype=torch.long),
        torch.as_tensor(local_edges, dtype=torch.long).t().contiguous(),
    )


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
    """Pair a full supervised epoch with a cycling regularizer stream.

    Recreating the regularizer iterator on each wrap also gives shuffled
    samplers a fresh permutation.
    """

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


def collate_graph_edge_batch(batch):
    """Deduplicate sampled edge endpoints and retain their local edge map.

    ``GraphEdgeBatchSampler`` normally emits unique ``GraphBatchNodeIndex``
    values and attaches the local edge map to the first one, so shared endpoints
    are fetched only once. The two-item fallback keeps direct unit-test and
    legacy batches working by deduplicating endpoint occurrences here.
    """

    if len(batch) == 0:
        raise ValueError("graph edge collation requires at least one graph node")

    item_width = len(batch[0])
    if item_width == 3:
        # Metadata batches are already endpoint-deduplicated by the sampler.
        # Their node count need not be even: for example, two adjacent edges
        # have three unique endpoints. The explicit edge map defines pairing.
        edge_maps = [item[2] for item in batch if len(item[2]) > 0]
        if len(edge_maps) != 1:
            raise ValueError("graph batch must carry exactly one local edge map")
        return (
            default_collate([item[0] for item in batch]),
            torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long),
            torch.as_tensor(edge_maps[0], dtype=torch.long),
        )
    if item_width != 2:
        raise ValueError("graph dataset items must have two or three fields")
    if len(batch) % 2 != 0:
        raise ValueError("legacy graph edge batches require adjacent endpoint pairs")

    unique_images = []
    unique_node_ids = []
    local_index_by_node = {}
    endpoint_indices = []
    for image, node_id in batch:
        node_id = int(node_id)
        local_index = local_index_by_node.get(node_id)
        if local_index is None:
            local_index = len(unique_node_ids)
            local_index_by_node[node_id] = local_index
            unique_node_ids.append(node_id)
            unique_images.append(image)
        endpoint_indices.append(local_index)

    return (
        default_collate(unique_images),
        torch.as_tensor(unique_node_ids, dtype=torch.long),
        torch.as_tensor(endpoint_indices, dtype=torch.long).reshape(-1, 2),
    )


def graph_upper_triangle_edges(adjacency):
    """Validate an undirected adjacency and return each edge exactly once."""

    adjacency = adjacency.tocsr()

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph adjacency must be square")

    if adjacency.shape[0] < 2:
        raise ValueError(
            "graph regularization requires at least two graph samples"
        )

    difference = (adjacency - adjacency.T).tocsr()
    if difference.nnz and not np.allclose(difference.data, 0.0):
        raise ValueError("graph adjacency must be symmetric")

    diagonal = adjacency.diagonal()
    assert diagonal.sum() == 0, (
        "graph adjacency must not contain self-loops"
    )
    if np.any(diagonal):
        raise ValueError("graph adjacency must not contain self-loops")

    coo = adjacency.tocoo()
    upper = coo.row < coo.col
    edge_rows = np.asarray(coo.row[upper], dtype=np.int64)
    edge_cols = np.asarray(coo.col[upper], dtype=np.int64)
    edge_weights = np.asarray(coo.data[upper], dtype=np.float64)

    if len(edge_rows) == 0:
        raise ValueError(
            "graph regularization requires at least one edge"
        )

    if (
        not np.all(np.isfinite(edge_weights))
        or np.any(edge_weights <= 0)
    ):
        raise ValueError(
            "graph edge weights must be finite and positive"
        )

    return edge_rows, edge_cols, edge_weights


class GraphEdgeBatchSampler(torch.utils.data.Sampler):
    """Uniformly batch the undirected graph edges themselves.

    ``graph_batch_size`` is a number of graph edges, independent of the
    supervised example batch size. The selected endpoints are deduplicated into
    at most ``2 * graph_batch_size`` ``GraphBatchNodeIndex`` values, with an
    explicit local edge map attached to the batch.

    When ``debug=True``, information about the first ``debug_max_batches``
    batches is printed each time a new iterator is created.
    """

    def __init__(
        self,
        adjacency,
        graph_batch_size,
        seed,
        num_batches=None,
        *,
        debug=False,
        debug_max_batches=1,
        debug_fn=print,
    ):
        if graph_batch_size <= 0:
            raise ValueError("graph_batch_size must be positive")

        self.graph_batch_size = int(graph_batch_size)
        self.edges_per_batch = self.graph_batch_size

        self.num_batches = None if num_batches is None else int(num_batches)
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError(
                "graph sampler num_batches must be positive when set"
            )

        if debug_max_batches is not None and debug_max_batches < 0:
            raise ValueError("debug_max_batches must be non-negative or None")

        self.debug = False
        self.debug_max_batches = debug_max_batches
        self.debug_fn = debug_fn

        self.generator = utils.make_torch_generator(seed)
        self.set_graph(adjacency)

    def set_graph(self, adjacency):
        edge_rows, edge_cols, edge_weights = graph_upper_triangle_edges(
            adjacency
        )

        self.edge_rows = torch.as_tensor(
            edge_rows,
            dtype=torch.long,
        )
        self.edge_cols = torch.as_tensor(
            edge_cols,
            dtype=torch.long,
        )
        self.edge_weights = torch.as_tensor(
            edge_weights,
            dtype=torch.float64,
        )

        self.num_edges = int(len(edge_rows))

    def _should_debug_batch(self, batch_number):
        if not self.debug:
            return False

        if self.debug_max_batches is None:
            return True

        return batch_number < self.debug_max_batches

    def _debug_batch(
        self,
        *,
        batch_number,
        selected_edge_ids,
        endpoint_pairs,
        selected_weights,
        unique_nodes,
        edge_indices,
    ):
        reconstructed_pairs = [
            (
                unique_nodes[left_local_index],
                unique_nodes[right_local_index],
            )
            for left_local_index, right_local_index in edge_indices
        ]

        expected_pairs = [
            (int(left_node), int(right_node))
            for left_node, right_node in endpoint_pairs
        ]

        if reconstructed_pairs != expected_pairs:
            raise RuntimeError(
                "Local graph edge map does not reconstruct the selected "
                "global graph edges.\n"
                f"Expected:      {expected_pairs}\n"
                f"Reconstructed: {reconstructed_pairs}"
            )

        lines = [
            "",
            f"[GraphEdgeBatchSampler] batch={batch_number}",
            f"  selected edge IDs:  {selected_edge_ids}",
            f"  global node pairs:  {expected_pairs}",
            f"  edge weights:       {selected_weights}",
            f"  unique global nodes:{unique_nodes}",
            f"  local edge map:     {list(edge_indices)}",
            f"  reconstructed pairs:{reconstructed_pairs}",
            (
                "  counts: "
                f"{len(expected_pairs)} edges, "
                f"{len(unique_nodes)} unique nodes"
            ),
        ]

        self.debug_fn("\n".join(lines))

    def __iter__(self):
        edge_order = torch.randperm(
            self.num_edges,
            generator=self.generator,
        )

        yielded = 0
        start = 0
        target_batches = len(self)

        while yielded < target_batches:
            if start >= self.num_edges:
                edge_order = torch.randperm(
                    self.num_edges,
                    generator=self.generator,
                )
                start = 0

            selected = edge_order[
                start : start + self.edges_per_batch
            ]

            endpoint_pairs = torch.stack(
                [
                    self.edge_rows[selected],
                    self.edge_cols[selected],
                ],
                dim=1,
            ).tolist()

            unique_nodes = []
            local_index_by_node = {}
            edge_indices = []

            for left_node, right_node in endpoint_pairs:
                local_pair = []

                for node_id in (int(left_node), int(right_node)):
                    local_index = local_index_by_node.get(node_id)

                    if local_index is None:
                        local_index = len(unique_nodes)
                        local_index_by_node[node_id] = local_index
                        unique_nodes.append(node_id)

                    local_pair.append(local_index)

                edge_indices.append(tuple(local_pair))

            edge_indices = tuple(edge_indices)

            if self._should_debug_batch(yielded):
                self._debug_batch(
                    batch_number=yielded,
                    selected_edge_ids=selected.tolist(),
                    endpoint_pairs=endpoint_pairs,
                    selected_weights=self.edge_weights[selected].tolist(),
                    unique_nodes=unique_nodes,
                    edge_indices=edge_indices,
                )

            yield [
                GraphBatchNodeIndex(
                    node_id=node_id,
                    edge_indices=(
                        edge_indices
                        if local_index == 0
                        else ()
                    ),
                )
                for local_index, node_id in enumerate(unique_nodes)
            ]

            yielded += 1
            start += self.edges_per_batch

    def __len__(self):
        if self.num_batches is not None:
            return self.num_batches

        return int(
            math.ceil(
                self.num_edges / self.edges_per_batch
            )
        )


class HofferReferenceBatchSampler(torch.utils.data.Sampler):
    """Emit batches of unlabeled indices plus labeled references per class.

    References are drawn uniformly within each class and resampled for every
    batch, as in Hoffer & Ailon (2018). Unlabeled samples are visited once per
    epoch in a shuffled order; each batch appends ``reference_sets`` freshly
    drawn reference indices for every class. Indices refer to a
    ``HofferReferenceDataset``, whose reference candidates start at
    ``num_unlabeled``.
    """

    def __init__(
        self,
        num_unlabeled,
        class_candidates,
        unlabeled_per_batch,
        seed,
        reference_sets=1,
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
        self.reference_sets = int(reference_sets)
        if self.reference_sets <= 0:
            raise ValueError("hoffer_entropy reference_sets must be positive")
        self.references_per_class = self.reference_sets
        self.class_labels = None if class_labels is None else np.asarray(class_labels, dtype=np.int64)
        if self.class_labels is not None and len(self.class_labels) != self.num_classes:
            raise ValueError("hoffer_entropy class_labels must align with class_candidates")
        self.unlabeled_positions = (
            None if unlabeled_positions is None else np.asarray(unlabeled_positions, dtype=np.int64)
        )
        if self.unlabeled_positions is not None and len(self.unlabeled_positions) != self.num_unlabeled:
            raise ValueError("hoffer_entropy unlabeled_positions must align with num_unlabeled")
        self.labeled_positions = None if labeled_positions is None else np.asarray(labeled_positions, dtype=np.int64)
        self.reference_classes_per_batch = self.num_classes
        self.references_per_batch = self.reference_classes_per_batch * self.references_per_class
        self.generator = utils.make_torch_generator(seed)

    def __iter__(self):
        unlabeled_order = torch.randperm(self.num_unlabeled, generator=self.generator)
        for batch_number, start in enumerate(range(0, self.num_unlabeled, self.unlabeled_per_batch)):
            unlabeled_indices = unlabeled_order[start : start + self.unlabeled_per_batch].tolist()
            batch_indices = list(unlabeled_indices)
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
