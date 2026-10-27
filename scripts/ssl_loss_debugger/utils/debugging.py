"""Small debugger and console-reporting helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from scripts.ssl_loss_debugger.training.seraph_trace import (
    DISTANCE_SPREAD_SELECTION,
    SAME_LABEL_SELECTION,
)

if TYPE_CHECKING:
    from scripts.ssl_loss_debugger.training.context import DebugContext


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_break(
    args: argparse.Namespace, phase: str, **locals_for_debugger: Any
) -> None:
    if args.break_at != phase:
        return
    print(f"\n[breakpoint] phase={phase}; useful locals={sorted(locals_for_debugger)}")
    # Assign the mapping to a named local so it is easy to expand in PyCharm/pdb.
    debug_locals = locals_for_debugger
    breakpoint()  # noqa: T100
    del debug_locals


def section(title: str) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def tensor_summary(value: torch.Tensor, max_values: int = 10) -> str:
    detached = value.detach().cpu()
    flat = detached.reshape(-1)
    shown = flat[:max_values].tolist()
    suffix = "..." if flat.numel() > max_values else ""
    finite = bool(torch.isfinite(flat).all()) if torch.is_floating_point(flat) else True
    stats = ""
    if flat.numel() and torch.is_floating_point(flat):
        stats = (
            f", min={float(flat.min()):.4g}, max={float(flat.max()):.4g}, "
            f"mean={float(flat.float().mean()):.4g}, finite={finite}"
        )
    return (
        f"Tensor(shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={value.device}{stats}, values={shown}{suffix})"
    )


def describe_tree(name: str, value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if torch.is_tensor(value):
        print(f"{prefix}{name}: {tensor_summary(value)}")
        return
    if isinstance(value, np.ndarray):
        print(
            f"{prefix}{name}: ndarray(shape={value.shape}, dtype={value.dtype}, "
            f"values={value.reshape(-1)[:10].tolist()}{'...' if value.size > 10 else ''})"
        )
        return
    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}(len={len(value)})")
        for index, item in enumerate(value):
            describe_tree(f"[{index}]", item, indent + 2)
        return
    print(f"{prefix}{name}: {value!r}")


def print_gradients(
    *named_modules: tuple[str, nn.Module],
    heading: str = "gradient norms",
    show_missing: bool = False,
    quiet: bool = False,
) -> float:
    """Print per-parameter norms and return the norm of the full gradient vector.

    ``quiet`` keeps the returned norm while printing nothing, so a long run can
    follow the gradient series across every epoch without the per-parameter dump.
    """

    if not quiet:
        print(f"{heading}:")
    any_gradient = False
    squared_global_norm = 0.0
    for module_prefix, module in named_modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                if show_missing and not quiet:
                    print(f"  {module_prefix}.{name}: <no gradient>")
                continue
            norm = float(
                torch.linalg.vector_norm(parameter.grad.detach().float()).cpu()
            )
            if not quiet:
                print(f"  {module_prefix}.{name}: {norm:.6g}")
            squared_global_norm += norm**2
            any_gradient = True
    global_norm = squared_global_norm**0.5
    if not quiet:
        if not any_gradient:
            print("  <none>")
        print(f"  global: {global_norm:.6g}")
    return global_norm


def print_split(context: DebugContext) -> None:
    labels = np.asarray(context.dataset.labels, dtype=np.int64)
    labeled_positions = np.asarray(context.split.labeled_positions, dtype=np.int64)
    unlabeled_positions = np.asarray(context.split.unlabeled_positions, dtype=np.int64)

    def compact_positions(positions: np.ndarray, limit: int = 16) -> str:
        values = positions.tolist()
        if len(values) <= limit:
            return str(values)
        half = limit // 2
        return f"{values[:half]} ... {values[-half:]} (count={len(values)})"

    def class_count_summary(positions: np.ndarray) -> str:
        _, counts = np.unique(labels[positions], return_counts=True)
        return (
            f"classes={len(counts)}, per_class_min={int(counts.min())}, "
            f"median={float(np.median(counts)):.4g}, max={int(counts.max())}"
        )

    print(
        f"dataset={context.args.dataset}, active_samples={len(context.dataset)}, "
        f"source_train_samples={context.source_dataset_size}, classes={context.num_classes}, "
        f"input_image_size={context.input_image_size}, "
        f"embedding_dim={context.args.embedding_dim}, protocol={context.args.dataset_protocol}"
    )
    print(f"labeled positions:   {compact_positions(labeled_positions)}")
    print(f"  labeled counts:    {class_count_summary(labeled_positions)}")
    print(f"unlabeled positions: {compact_positions(unlabeled_positions)}")
    print(f"  unlabeled counts:  {class_count_summary(unlabeled_positions)}")
    short_classes = getattr(context.dataset, "short_unlabeled_classes", {})
    if short_classes:
        realized_counts = list(short_classes.values())
        print(
            "  unlabeled cap note: "
            f"{len(short_classes)} classes had fewer than "
            f"{context.args.unlabeled_per_class} remaining samples; "
            f"realized_min={min(realized_counts)}, realized_max={max(realized_counts)}"
        )


def print_graph(regularizer: Any) -> None:
    adjacency = getattr(regularizer, "adjacency", None)
    if adjacency is None:
        return
    upper = adjacency.tocoo()
    keep = (upper.row < upper.col) & (upper.data != 0)
    edges = [
        (int(left), int(right), float(weight))
        for left, right, weight in zip(
            upper.row[keep],
            upper.col[keep],
            upper.data[keep],
        )
    ]
    print(
        f"graph: nodes={adjacency.shape[0]}, undirected_edges={len(edges)}, "
        f"weight_sum={sum(weight for _, _, weight in edges):.6g}"
    )
    print(f"graph positions: {getattr(regularizer, 'graph_positions', None)}")
    print(f"first graph edges (node_i, node_j, weight): {edges[:20]}")
    neighbor_indices = getattr(regularizer, "neighbor_indices", None)
    if neighbor_indices is not None:
        describe_tree("neighbor_indices", neighbor_indices)
    degrees = getattr(regularizer, "degrees", None)
    if degrees is not None:
        describe_tree("degrees", np.asarray(degrees))


def _undirected_edge_set(adjacency: Any) -> frozenset[tuple[int, int]]:
    """Return graph edges once each, independent of sparse storage order."""

    coordinates = adjacency.tocoo()
    return frozenset(
        (int(left), int(right))
        for left, right, weight in zip(
            coordinates.row,
            coordinates.col,
            coordinates.data,
        )
        if left < right and weight != 0
    )


@dataclass
class GraphEpochTracker:
    """Report whether a regularizer graph changes between epoch rebuilds."""

    snapshots: list[tuple[int, frozenset[tuple[int, int]]]] = field(
        default_factory=list
    )

    def record(self, epoch: int, regularizer: Any) -> None:
        adjacency = getattr(regularizer, "adjacency", None)
        if adjacency is None:
            return

        edges = _undirected_edge_set(adjacency)
        reported_count = getattr(regularizer, "graph_edge_count", None)
        if reported_count is not None and int(reported_count) != len(edges):
            raise RuntimeError(
                "regularizer graph_edge_count does not match its adjacency: "
                f"reported={int(reported_count)}, observed={len(edges)}"
            )

        if not self.snapshots:
            print(
                f"graph epoch check: epoch={epoch}, undirected_edges={len(edges)}, "
                "baseline"
            )
        else:
            previous_epoch, previous_edges = self.snapshots[-1]
            added = len(edges - previous_edges)
            removed = len(previous_edges - edges)
            delta = len(edges) - len(previous_edges)
            print(
                f"graph epoch check: epoch={epoch}, undirected_edges={len(edges)}, "
                f"previous_epoch={previous_epoch}, delta={delta:+d}, "
                f"count_changed={'yes' if delta else 'no'}, "
                f"topology_changed={'yes' if added or removed else 'no'}, "
                f"added={added}, removed={removed}"
            )
        self.snapshots.append((int(epoch), edges))

    def print_summary(self) -> None:
        if not self.snapshots:
            return

        counts = [len(edges) for _, edges in self.snapshots]
        history = ", ".join(
            f"epoch {epoch}={len(edges)}" for epoch, edges in self.snapshots
        )
        print(f"graph edge-count history: {history}")
        if len(self.snapshots) == 1:
            print(
                "graph edge-count change across epochs: not checked "
                "(run with --epochs 2 or greater)"
            )
            return

        baseline_edges = self.snapshots[0][1]
        count_changed = any(count != counts[0] for count in counts[1:])
        topology_changed = any(
            edges != baseline_edges for _, edges in self.snapshots[1:]
        )
        print(
            "graph edge-count changed across epochs: "
            f"{'YES' if count_changed else 'NO'} "
            f"(min={min(counts)}, max={max(counts)})"
        )
        print(
            "graph topology changed across epochs: "
            f"{'YES' if topology_changed else 'NO'}"
        )


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.6g}" for name, value in metrics.items())


@dataclass
class RegularizerEpochTracker:
    """Follow the loss and gradient balance of a regularizer across epochs.

    One line per step keeps a long run readable when ``--report-every`` silences
    the per-step dumps: the series is the reason to run 200 epochs, while the
    dumps are only useful on the epochs you stop to inspect.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        epoch: int,
        step: int,
        supervised_loss: float,
        regularizer_loss: float,
        combined_loss: float,
        supervised_gradient_norm: float,
        regularizer_gradient_norm: float,
        extra: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        ratio = (
            float("nan")
            if supervised_gradient_norm == 0.0
            else regularizer_gradient_norm / supervised_gradient_norm
        )
        row = {
            "epoch": int(epoch),
            "step": int(step),
            "supervised_loss": float(supervised_loss),
            "regularizer_loss": float(regularizer_loss),
            "combined_loss": float(combined_loss),
            "supervised_gradient_norm": float(supervised_gradient_norm),
            "regularizer_gradient_norm": float(regularizer_gradient_norm),
            "regularizer_to_supervised": ratio,
            "extra": dict(extra or {}),
        }
        self.rows.append(row)
        return row

    def print_row(self, row: dict[str, Any]) -> None:
        extra = f", {_format_metrics(row['extra'])}" if row["extra"] else ""
        print(
            f"epoch {row['epoch']:>4} step {row['step']:>3} | "
            f"sup_loss={row['supervised_loss']:.6g}, "
            f"reg_loss={row['regularizer_loss']:.6g}, "
            f"combined={row['combined_loss']:.6g}, "
            f"reg/sup_grad={row['regularizer_to_supervised']:.6g}{extra}"
        )

    def print_summary(self) -> None:
        if len(self.rows) < 2:
            return

        first, last = self.rows[0], self.rows[-1]
        print(f"\nregularizer trend over {len(self.rows)} recorded steps:")
        for label, row in (("first", first), ("last", last)):
            print(f"  {label:<5} ", end="")
            self.print_row(row)
        for name in (
            "supervised_loss",
            "regularizer_loss",
            "regularizer_to_supervised",
        ):
            values = [row[name] for row in self.rows]
            finite = [value for value in values if np.isfinite(value)]
            if not finite:
                continue
            print(
                f"  {name}: first={values[0]:.6g}, last={values[-1]:.6g}, "
                f"min={min(finite):.6g}, max={max(finite):.6g}, "
                f"mean={float(np.mean(finite)):.6g}"
            )
        for name in sorted(first["extra"]):
            values = [
                row["extra"][name] for row in self.rows if name in row["extra"]
            ]
            finite = [value for value in values if np.isfinite(value)]
            if not finite:
                continue
            print(
                f"  {name}: first={values[0]:.6g}, last={values[-1]:.6g}, "
                f"min={min(finite):.6g}, max={max(finite):.6g}"
            )


def print_regularizer_batch_details(
    context: DebugContext,
    regularizer: Any,
    batch: tuple[Any, ...],
) -> None:
    if regularizer.name in {"lrml", "slrml"} and len(batch) == 3:
        _, node_ids, edge_indices = batch
        pairs = []
        for left, right in edge_indices.tolist():
            left_node = int(node_ids[left])
            right_node = int(node_ids[right])
            weight = float(regularizer.adjacency[left_node, right_node])
            left_position = int(regularizer.graph_positions[left_node])
            right_position = int(regularizer.graph_positions[right_node])
            pairs.append(
                {
                    "local": (int(left), int(right)),
                    "graph_nodes": (left_node, right_node),
                    "train_positions": (left_position, right_position),
                    "weight": weight,
                }
            )
        print(f"decoded sampled graph edges: {pairs}")
    elif regularizer.name == "seraph":
        positions = [int(position) for position in batch[2].tolist()]
        source_positions = getattr(context.dataset, "source_positions", None)
        if source_positions is None:
            print(f"SERAPH unlabeled debug positions: {positions}")
        else:
            print(
                f"SERAPH unlabeled debug positions: {positions}; "
                f"production-train positions: {[source_positions[position] for position in positions]}"
            )
    elif regularizer.name == "hoffer_entropy":
        roles = batch[1].bool()
        print(
            f"Hoffer roles: unlabeled={(~roles).sum().item()}, "
            f"references={roles.sum().item()}, sequence={roles.int().tolist()}"
        )


def _seraph_sample_lookup(
    comparisons: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (sample["role"], int(sample["batch_index"])): sample
        for samples in comparisons.get("samples", {}).values()
        for sample in samples
    }


def _seraph_endpoint_description(
    context: DebugContext,
    sample: dict[str, Any],
) -> str:
    role = str(sample["role"])
    role_prefix = "L" if role == "labeled" else "U"
    batch_index = int(sample["batch_index"])
    position = sample.get("source_position")
    mapped_label = sample.get("mapped_label")
    fields = [f"{role_prefix}{batch_index}", f"pos={position}"]
    if mapped_label is not None:
        fields.append(f"mapped={mapped_label}")
    if position is not None:
        source_positions = getattr(context.dataset, "source_positions", None)
        if source_positions is not None:
            fields.append(f"train_pos={int(source_positions[int(position)])}")
        fields.append(f"true={int(context.dataset.labels[int(position)])}")
        fields.append(f"orig={int(context.dataset.orig_labels[int(position)])}")
    return "[" + ", ".join(fields) + "]"


def _seraph_debug_relation(
    context: DebugContext,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    left_position = left.get("source_position")
    right_position = right.get("source_position")
    if left_position is None or right_position is None:
        return "unknown"
    left_label = int(context.dataset.labels[int(left_position)])
    right_label = int(context.dataset.labels[int(right_position)])
    return "same" if left_label == right_label else "different"


def _print_seraph_true_label_split(block: dict[str, Any]) -> None:
    """Print how the block's pairs split by debugger-only true label.

    The block averages above are dominated by the true negatives, which outnumber
    the positives by roughly num_classes to one. These lines say whether the
    posterior tells the two apart: with no separation the two mean distances
    coincide and both positive rates match the block-wide one.
    """

    same_label_count = block.get("same_label_pair_count")
    if same_label_count is None:
        return
    pair_count = block["batch_pair_count"]
    print(
        f"    debug true labels: {same_label_count} same-label pairs "
        f"({same_label_count / pair_count:.4%} of the block), "
        f"{pair_count - same_label_count} different-label"
    )
    for relation in ("same", "different"):
        mean_distance = block[f"{relation}_label_mean_squared_distance"]
        if mean_distance is None:
            continue
        print(
            f"      {relation}-label: "
            f"mean_d2={mean_distance:.8g}, "
            f"mean_entropy={block[f'{relation}_label_mean_entropy']:.8g}, "
            f"positive_pair_rate={block[f'{relation}_label_positive_rate']:.6g}"
        )


def print_seraph_comparison_diagnostics(
    context: DebugContext,
    comparisons: dict[str, Any],
) -> None:
    """Print the pair comparisons behind one SERAPH batch's regularizer loss."""

    if not comparisons:
        print("SERAPH comparison trace: <not captured>")
        return

    print(
        "SERAPH comparison trace "
        "(true labels are debugger-only diagnostics; the regularizer never sees them):"
    )
    print(
        f"  objective={comparisons['objective']}"
        + (
            f" (em_sharpening={comparisons['em_sharpening']:g})"
            if comparisons["objective"] == "em"
            else ""
        )
        + f", metric={comparisons['metric']}, "
        f"kappa={comparisons['kappa']:.6g} ({comparisons['kappa_mode']}), "
        f"threshold_mode={comparisons['threshold_mode']}"
        # Only quantile mode reads either; fixed mode returns the configured
        # threshold without consulting them.
        + (
            ""
            if comparisons["threshold_mode"] != "quantile"
            else (
                f", positive_pair_prior={comparisons['positive_pair_prior']:.6g}"
                f", threshold_momentum={comparisons['threshold_momentum']:.6g}"
            )
        )
        + f", include_labeled_unlabeled={comparisons['include_labeled_unlabeled']}"
    )

    lookup = _seraph_sample_lookup(comparisons)
    print("  batch samples and compared embeddings:")
    for role in ("labeled", "unlabeled"):
        for sample in comparisons["samples"].get(role, []):
            endpoint = _seraph_endpoint_description(context, sample)
            embedding = ", ".join(f"{value:.5g}" for value in sample["embedding"])
            suffix = ", ..." if sample["embedding_truncated"] else ""
            print(f"    {endpoint} z=[{embedding}{suffix}]")

    for block in comparisons["loss_blocks"]:
        kind = block["kind"]
        population = block["population_pair_count"]
        print(
            f"  {kind.upper()} block: "
            f"eta={block['eta']:.8g}, kappa={block['kappa']:.8g}, "
            f"batch_pairs={block['batch_pair_count']}, "
            f"population_pairs={'unknown' if population is None else f'{population:.0f}'}, "
            f"block_weight={block['block_weight']:.8g}"
        )
        print(
            f"    mean_d2={block['mean_squared_distance']:.8g}, "
            # docs/seraph.md: the gradient peaks near mean |logit| = 2.4 and
            # vanishes at both ends, so this is the number to read when tuning.
            f"mean_abs_logit={block['mean_absolute_logit']:.8g}, "
            f"mean_entropy={block['mean_entropy']:.8g} of 0.693 max, "
            f"positive_pair_rate={block['positive_pair_rate']:.6g}"
        )
        _print_seraph_true_label_split(block)
        listed = block["listed_pairs"]
        if not listed:
            continue
        for selection, heading in (
            (
                DISTANCE_SPREAD_SELECTION,
                "spread evenly across the block's sorted distances",
            ),
            (
                SAME_LABEL_SELECTION,
                "spread evenly across the sorted distances of the true positives",
            ),
        ):
            selected = sorted(
                (
                    pair
                    for pair in comparisons["loss_pairs"]
                    if pair["kind"] == kind and selection in pair["selection"]
                ),
                key=lambda pair: pair["distance_rank"],
            )
            if not selected:
                continue
            print(
                f"    {len(selected)} of {block['batch_pair_count']} pairs, "
                f"{heading}:"
            )
            for pair in selected:
                left = lookup[(pair["left_role"], pair["left_index"])]
                right = lookup[(pair["right_role"], pair["right_index"])]
                target = (
                    "" if pair["target"] is None else f" | q_same={pair['target']:.6g}"
                )
                same_label_rank = (
                    ""
                    if pair["same_label_rank"] is None
                    else (
                        f" (same_label_rank {pair['same_label_rank']}"
                        f"/{pair['same_label_pair_count']})"
                    )
                )
                print(
                    f"      d2_rank {pair['distance_rank']:>5}/"
                    f"{pair['block_pair_count']}{same_label_rank} "
                    f"{_seraph_endpoint_description(context, left)} <-> "
                    f"{_seraph_endpoint_description(context, right)} | "
                    f"debug_relation={_seraph_debug_relation(context, left, right)} | "
                    f"d2={pair['squared_distance']:.8g} | logit={pair['logit']:.8g} | "
                    f"p_same={pair['same_probability']:.8g}{target} | "
                    f"entropy={pair['entropy']:.8g} | "
                    f"loss_contribution={pair['loss_contribution']:.8g}"
                )
    print(
        "  regularizer loss reconstructed from every scored pair="
        f"{comparisons['reconstructed_loss']:.8g}"
    )
