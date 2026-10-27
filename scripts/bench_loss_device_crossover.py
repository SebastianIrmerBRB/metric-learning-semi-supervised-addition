"""Find the batch size at which CUDA overtakes CPU for the training step.

This measures only what the *training* device owns — projection, loss, backward,
optimizer, plus the per-step regularizer term of the SSL method — for each loss
at each batch size on both devices, and reports the smallest batch size where
CUDA wins. That crossover is the number ``LOSS_GPU_BATCH_SIZE_THRESHOLDS`` in
``launch_runs.py`` wants. Graph and pseudo-label *construction* is excluded: it
belongs to ``ssl_device``, which always wants CUDA.

Objectives:

``supervised``
    Projection, loss, backward, optimizer. This is also the Iscen step: a
    pseudo-label method swaps the labels but changes nothing about the step, and
    only ``MixedLabelPropagationProxyLoss`` consumes its certainty weights.
``lrml``
    Adds the production per-step Laplacian term: one extra projection of the
    sampled edge endpoints, degree scaling, and edge energy, folded into the same
    backward. ``--graph-batch-sizes`` is independent of the supervised batch
    size, because ``GraphPairBatchSampler`` samples edges, not examples.
``hoffer``
    Adds the production entropy term: one forward over the unlabeled rows plus
    ``num_classes * reference_sets`` reference rows, then the
    ``[sets, unlabeled, classes, dim]`` broadcast that dominates its cost.

Steps run under the same ``torch.autocast(bfloat16)`` the engine uses, on the
same frozen-feature path (host features, per-step host-to-device copy), so a
result transfers to a scheduled run. ``--no-autocast`` measures what that
autocast costs, which matters on a CPU without native bfloat16.

Examples::

    python scripts/bench_loss_device_crossover.py --objective supervised
    python scripts/bench_loss_device_crossover.py --objective lrml \\
        --batch-sampler 32:8 --graph-batch-sizes 32 128 512
    python scripts/bench_loss_device_crossover.py --objective hoffer \\
        --hoffer-reference-sets 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The launcher gives every child four BLAS/OMP threads; the CPU side of this
# comparison is meaningless at any other setting.
DEFAULT_THREADS = 4
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, str(DEFAULT_THREADS))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.bench_train_step import (  # noqa: E402
    DataLoaderSource,
    Head,
    make_criterion,
    make_features,
)
from training.hpo import parse_batch_sampler_choice  # noqa: E402
from training.semi_supervised import _graph_edge_batch  # noqa: E402
from training.ssl.algorithms import build_lrml_knn_graph, suppress_ssl_timing_logs  # noqa: E402
from training.ssl.data import GraphPairBatchSampler  # noqa: E402

DEFAULT_LOSSES = (
    "MultiSimilarityLoss",
    "NTXentLoss",
    "CircleLoss",
    "ArcFaceLoss",
    "ProxyAnchorLoss",
)
# MPerClassSampler needs sampler_m * classes >= batch_size, so batch size and m
# travel together. These are the pairs the HPO batch_sampler spaces search.
DEFAULT_BATCH_SAMPLERS = ("16:8", "32:8", "64:8", "128:8", "256:8")
OBJECTIVES = ("supervised", "lrml", "hoffer")
MATCH_BATCH = "match"

_GRAPH_CACHE: dict[tuple, tuple] = {}


def build_graph(features, n_neighbors, cache_key):
    """Build the LRML kNN graph once and share it across cells."""

    if cache_key not in _GRAPH_CACHE:
        graph_features = np.ascontiguousarray(features.numpy(), dtype=np.float32)
        with suppress_ssl_timing_logs():
            _, adjacency, degrees = build_lrml_knn_graph(
                graph_features,
                n_neighbors=n_neighbors,
                prefer_gpu=torch.cuda.is_available(),
            )
        _GRAPH_CACHE[cache_key] = (adjacency, degrees)
    return _GRAPH_CACHE[cache_key]


class LrmlEdgeTerm:
    """The per-step half of LRML: project sampled edge endpoints, sum energy.

    Mirrors ``LRMLRegularizer.compute_loss`` in global graph mode. The endpoints
    of ``graph_batch_size`` sampled edges are deduplicated, projected once,
    scaled by 1/sqrt(deg) so the unnormalized energy equals the
    normalized-Laplacian energy, and reduced over the sampled edges.
    """

    def __init__(self, features, graph_batch_size, device, seed, n_neighbors, cache_key):
        adjacency, degrees = build_graph(features, n_neighbors, cache_key)
        self.adjacency = adjacency
        self.features = features
        self.device = device
        self.node_scale = torch.as_tensor(
            1.0 / np.sqrt(np.maximum(degrees, 1.0)), dtype=torch.float32
        )
        self.sampler = GraphPairBatchSampler(
            adjacency=adjacency, graph_batch_size=graph_batch_size, seed=seed
        )
        self._iterator = iter(self.sampler)

    def next_batch(self):
        """Return one sampled edge batch as (features, node ids, edge map)."""

        try:
            items = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.sampler)
            items = next(self._iterator)
        # The sampler emits deduplicated GraphBatchNodeIndex values and attaches
        # the batch's local edge map to one of them.
        node_ids = torch.as_tensor([int(item.node_id) for item in items], dtype=torch.long)
        edge_indices = next(
            (item.edge_indices for item in items if len(item.edge_indices) > 0), None
        )
        images = self.features[node_ids].to(self.device, non_blocking=True)
        return images, node_ids, edge_indices

    def loss(self, model, batch):
        images, node_ids, edge_indices = batch
        embeddings = model.project_features(images)
        scaled = embeddings * self.node_scale[node_ids].to(self.device)[:, None]
        left, right, weights = _graph_edge_batch(self.adjacency, node_ids, edge_indices)
        differences = scaled[left.to(self.device)] - scaled[right.to(self.device)]
        return (differences * differences).sum(dim=1).sum() / max(len(weights), 1)


class HofferEntropyTerm:
    """The per-step half of Hoffer: entropy of unlabeled-to-reference distances.

    Mirrors ``HofferEntropyRegularizer.compute_loss``: one forward over the
    unlabeled rows plus ``num_classes * reference_sets`` reference rows, then the
    ``[sets, unlabeled, classes, dim]`` broadcast whose materialization is the
    method's real per-step cost.
    """

    def __init__(self, features, unlabeled_batch_size, reference_sets, num_classes, device, seed, distance_scale=1.0):
        self.features = features
        self.device = device
        self.unlabeled_batch_size = int(unlabeled_batch_size)
        self.reference_sets = int(reference_sets)
        self.num_classes = int(num_classes)
        self.distance_scale = float(distance_scale)
        self.generator = torch.Generator().manual_seed(seed)

    def next_batch(self):
        """Return the concatenated (unlabeled + reference) rows for one step."""

        reference_count = self.num_classes * self.reference_sets
        total = self.unlabeled_batch_size + reference_count
        indices = torch.randint(
            0, len(self.features), (total,), generator=self.generator, dtype=torch.long
        )
        return self.features[indices].to(self.device, non_blocking=True)

    def loss(self, model, batch):
        embeddings = model.project_features(batch)
        unlabeled = embeddings[: self.unlabeled_batch_size]
        references = embeddings[self.unlabeled_batch_size :]
        reference_sets = references.reshape(self.num_classes, self.reference_sets, -1).permute(1, 0, 2)
        squared_distances = (
            unlabeled[None, :, None, :] - reference_sets[:, None, :, :]
        ).pow(2).sum(dim=-1)
        log_probabilities = F.log_softmax(-self.distance_scale * squared_distances, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
        return entropy.mean()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_term(objective, features, batch_size, extra_size, args, device, cache_key):
    """Return the per-step regularizer term for one objective, or None."""

    if objective == "supervised":
        return None
    size = batch_size if extra_size == MATCH_BATCH else int(extra_size)
    if objective == "lrml":
        return LrmlEdgeTerm(
            features, size, device, args.seed, args.lrml_neighbors, cache_key
        )
    return HofferEntropyTerm(
        features,
        size,
        args.reference_sets,
        args.num_classes,
        device,
        args.seed,
        args.hoffer_distance_scale,
    )


def measure_cell(loss_name, batch_size, sampler_m, device_name, objective, extra_size, args) -> dict:
    """Time one (loss, batch size, device, objective, term size) cell."""

    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    features, labels = make_features(
        args.num_samples, args.num_classes, args.feat_dim, args.seed, device
    )
    cache_key = (args.num_samples, args.num_classes, args.feat_dim, args.seed, args.lrml_neighbors)
    model = Head(args.feat_dim, args.feat_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion, criterion_optimizer = make_criterion(
        loss_name, args.num_classes, args.feat_dim, args.lr, device
    )
    source = DataLoaderSource(
        features, labels, batch_size, sampler_m, device, args.num_workers
    )
    term = make_term(objective, features, batch_size, extra_size, args, device, cache_key)

    step_times: list[float] = []
    steps_done = 0
    total_steps = args.warmup_steps + args.steps
    while steps_done < total_steps:
        for batch_features, batch_labels in source:
            term_batch = None if term is None else term.next_batch()
            synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=False)
            if criterion_optimizer is not None:
                criterion_optimizer.zero_grad(set_to_none=False)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=args.autocast
            ):
                loss = criterion(model.project_features(batch_features), batch_labels)
                if term is not None:
                    loss = loss + term.loss(model, term_batch)
            loss.backward()
            optimizer.step()
            if criterion_optimizer is not None:
                criterion_optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - started
            if steps_done >= args.warmup_steps:
                step_times.append(elapsed)
            steps_done += 1
            if steps_done >= total_steps:
                break

    return {
        "loss": loss_name,
        "batch_size": batch_size,
        "sampler_m": sampler_m,
        "device": device_name,
        "objective": objective,
        "term_size": batch_size if extra_size == MATCH_BATCH else int(extra_size),
        "term_size_mode": str(extra_size),
        "autocast": bool(args.autocast),
        "steps": len(step_times),
        "median_step_ms": statistics.median(step_times) * 1e3,
        "mean_step_ms": statistics.mean(step_times) * 1e3,
        "p90_step_ms": sorted(step_times)[int(0.9 * len(step_times)) - 1] * 1e3,
    }


def select(cells, **filters):
    """Return the first cell matching every filter, or None."""

    for cell in cells:
        if all(cell[key] == value for key, value in filters.items()):
            return cell
    return None


def crossover_batch_size(cells, loss_name, objective, term_mode, batch_sizes):
    """Return the smallest measured batch size where CUDA beats CPU."""

    for batch_size in sorted(batch_sizes):
        common = {
            "loss": loss_name,
            "objective": objective,
            "term_size_mode": term_mode,
            "batch_size": batch_size,
        }
        cpu = select(cells, device="cpu", **common)
        cuda = select(cells, device="cuda", **common)
        if cpu and cuda and cuda["median_step_ms"] < cpu["median_step_ms"]:
            return batch_size
    return None


def format_report(cells, losses, batch_sizes, objectives, term_modes) -> str:
    lines: list[str] = []
    for objective in objectives:
        for term_mode in term_modes.get(objective, [MATCH_BATCH]):
            present = [
                cell
                for cell in cells
                if cell["objective"] == objective and cell["term_size_mode"] == term_mode
            ]
            if not present:
                continue
            suffix = ""
            if objective == "lrml":
                suffix = f", graph_batch_size={term_mode}"
            elif objective == "hoffer":
                suffix = f", unlabeled_batch_size={term_mode}"
            lines.append("")
            lines.append(f"=== {objective} step, median ms cpu/cuda{suffix} ===")
            lines.append(f"{'loss':22}" + "".join(f"{f'b={size}':>18}" for size in batch_sizes))
            for loss_name in losses:
                row = f"{loss_name:22}"
                for size in batch_sizes:
                    common = {
                        "loss": loss_name,
                        "objective": objective,
                        "term_size_mode": term_mode,
                        "batch_size": size,
                    }
                    cpu = select(present, device="cpu", **common)
                    cuda = select(present, device="cuda", **common)
                    if cpu is None or cuda is None:
                        row += f"{'-':>18}"
                        continue
                    cpu_ms = cpu["median_step_ms"]
                    cuda_ms = cuda["median_step_ms"]
                    marker = "*" if cuda_ms < cpu_ms else " "
                    row += f"{f'{cpu_ms:.2f}/{cuda_ms:.2f}{marker}':>18}"
                lines.append(row)
            lines.append("* marks cells where CUDA is faster. Crossover batch size:")
            for loss_name in losses:
                crossover = crossover_batch_size(
                    present, loss_name, objective, term_mode, batch_sizes
                )
                verdict = (
                    f"> {max(batch_sizes)} (CPU wins at every measured size)"
                    if crossover is None
                    else str(crossover)
                )
                lines.append(f"  {loss_name:22} {verdict}")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--losses", nargs="+", default=list(DEFAULT_LOSSES))
    parser.add_argument(
        "--batch-sampler",
        nargs="+",
        default=list(DEFAULT_BATCH_SAMPLERS),
        help="'batch_size:sampler_m' pairs, as in the HPO batch_sampler space",
    )
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--objective", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument(
        "--graph-batch-sizes",
        nargs="+",
        default=[MATCH_BATCH],
        help=f"LRML sampled edges per step; '{MATCH_BATCH}' follows the supervised batch size",
    )
    parser.add_argument(
        "--hoffer-unlabeled-batch-sizes",
        nargs="+",
        default=[MATCH_BATCH],
        help=f"Hoffer unlabeled rows per step; '{MATCH_BATCH}' follows the supervised batch size",
    )
    parser.add_argument("--reference-sets", type=int, default=1)
    parser.add_argument("--hoffer-distance-scale", type=float, default=1.0)
    parser.add_argument("--lrml-neighbors", type=int, default=6)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--num_samples", type=int, default=28800, help="CIFAR100 train pool by default")
    parser.add_argument("--num_classes", type=int, default=48)
    parser.add_argument("--steps", type=int, default=60, help="measured steps per cell")
    parser.add_argument("--warmup_steps", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the engine's bfloat16 autocast; --no-autocast prices what it costs",
    )
    parser.add_argument("--json_out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    devices = [name for name in args.devices if name != "cuda" or torch.cuda.is_available()]
    if "cuda" in args.devices and "cuda" not in devices:
        print("CUDA is unavailable; measuring CPU only.", file=sys.stderr)

    pairs = [parse_batch_sampler_choice(choice) for choice in args.batch_sampler]
    batch_sizes = [batch_size for batch_size, _ in pairs]
    term_modes = {
        "supervised": [MATCH_BATCH],
        "lrml": list(args.graph_batch_sizes),
        "hoffer": list(args.hoffer_unlabeled_batch_sizes),
    }

    cells: list[dict] = []
    for objective in args.objective:
        for term_mode in term_modes[objective]:
            for loss_name in args.losses:
                for batch_size, sampler_m in pairs:
                    for device_name in devices:
                        cell = measure_cell(
                            loss_name,
                            batch_size,
                            sampler_m,
                            device_name,
                            objective,
                            term_mode,
                            args,
                        )
                        cells.append(cell)
                        term_note = "" if objective == "supervised" else f" term={cell['term_size']}"
                        print(
                            f"{objective:10} {loss_name:22} b={batch_size:<4} m={sampler_m:<3}"
                            f"{term_note:11} {device_name:5} "
                            f"median={cell['median_step_ms']:9.3f} ms",
                            flush=True,
                        )

    report = format_report(cells, args.losses, batch_sizes, args.objective, term_modes)
    print(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cells": cells, "args": {**vars(args), "json_out": str(args.json_out)}}
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
