"""Benchmark the frozen-backbone training step across batching/launch modes.

Three modes are compared on the same model, loss, sampler, and feature cache:

``dataloader``
    The current path: a ``DataLoader`` over a precomputed-feature dataset, with
    the batch copied to the device each step.
``gpu_resident``
    Mode 1. The feature view lives in device memory and batches are gathered
    on-device, so the step does no host transfer and runs no DataLoader.
``compile``
    GPU-resident batching plus ``torch.compile`` over the
    projection and loss, which fuses the elementwise chains that produce most of
    the step's kernel launches.
``dataloader_compile``
    ``torch.compile`` on top of the unchanged DataLoader path, to separate what
    compilation buys from what GPU-resident batching buys.

Run several of these at once, one per process, to see the numbers that matter
for concurrent HPO trials; a single-process measurement understates contention.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_metric_learning.losses as losses  # noqa: E402
import pytorch_metric_learning.samplers as samplers  # noqa: E402

import utils  # noqa: E402
from training.gpu_feature_loader import GpuResidentFeatureLoader  # noqa: E402

MODES = ("dataloader", "gpu_resident", "compile", "dataloader_compile")


class Head(torch.nn.Module):
    """The trainable part of a frozen-backbone run: one projection + L2 norm.

    Mirrors ``DinoWrapper.project_features`` so the benchmark measures the same
    kernels the real step launches, without loading a DINO backbone.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def project_features(self, features):
        return torch.nn.functional.normalize(self.fc(features), p=2.0, dim=1)

    def forward(self, features):
        return self.project_features(features)


def make_features(num_samples, num_classes, feat_dim, seed, device):
    """Build a class-structured feature matrix standing in for a cached backbone."""

    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, feat_dim, generator=generator) * 4.0
    labels = torch.arange(num_samples) % num_classes
    features = centers[labels] + torch.randn(num_samples, feat_dim, generator=generator)
    return features.contiguous(), labels.contiguous()


def make_criterion(loss_name, num_classes, feat_dim, learning_rate, device):
    """Return ``(criterion, optimizer_or_none)`` for a loss used by the real runs."""

    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5), None
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(temperature=0.07), None
    if loss_name == "CircleLoss":
        return losses.CircleLoss(m=0.4, gamma=80), None
    if loss_name == "ArcFaceLoss":
        criterion = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=feat_dim,
        ).to(device)
        return criterion, torch.optim.Adam(criterion.parameters(), lr=learning_rate)
    if loss_name == "ProxyAnchorLoss":
        criterion = losses.ProxyAnchorLoss(
            num_classes=num_classes,
            embedding_size=feat_dim,
        ).to(device)
        return criterion, torch.optim.Adam(criterion.parameters(), lr=learning_rate)
    raise ValueError(f"unsupported loss for this benchmark: {loss_name}")


class DataLoaderSource:
    """Baseline batching: sampler, DataLoader, and a host-to-device copy."""

    name = "dataloader"

    def __init__(self, features, labels, batch_size, sampler_m, device, num_workers):
        self.device = device
        dataset = utils.PrecomputedBackboneFeatureDataset(features, labels.tolist())
        sampler = samplers.MPerClassSampler(
            dataset.labels,
            m=sampler_m,
            batch_size=batch_size,
            length_before_new_iter=len(dataset),
        )
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch_features, batch_labels in self.loader:
            yield (
                batch_features.to(self.device, non_blocking=True),
                batch_labels.to(self.device, non_blocking=True),
            )


class GpuResidentSource:
    """Mode 1 batching: the feature view stays on the device."""

    name = "gpu_resident"

    def __init__(self, features, labels, batch_size, sampler_m, device, seed):
        dataset = utils.PrecomputedBackboneFeatureDataset(features, labels.tolist())
        sampler = samplers.MPerClassSampler(
            dataset.labels,
            m=sampler_m,
            batch_size=batch_size,
            length_before_new_iter=len(dataset),
        )
        self.loader = GpuResidentFeatureLoader(
            features,
            labels,
            None,
            device=device,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            seed=seed,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch_features, batch_labels, _weights in self.loader:
            yield batch_features, batch_labels


def make_forward_loss(model, criterion, compiled):
    """Return the projection-plus-loss callable, optionally ``torch.compile``d.

    Only forward and loss are compiled. Backward follows from the compiled
    forward's autograd graph, and the optimizer stays eager, which is what the
    real step would do if this mode won.
    """

    def forward_loss(features, labels):
        return criterion(model.project_features(features), labels)

    if not compiled:
        return forward_loss
    # Losses here mine pairs, so batch-dependent shapes flow into the graph.
    # Automatic dynamic shapes let the second distinct shape generalize the
    # kernel instead of recompiling once per shape until the cache limit trips.
    return torch.compile(forward_loss)


def eager_step(forward_loss, optimizer, criterion_optimizer, features, labels, device_type):
    optimizer.zero_grad(set_to_none=False)
    if criterion_optimizer is not None:
        criterion_optimizer.zero_grad(set_to_none=False)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, cache_enabled=False):
        loss = forward_loss(features, labels)
    loss.backward()
    optimizer.step()
    if criterion_optimizer is not None:
        criterion_optimizer.step()
    return loss


def run_mode(mode, args, device):
    """Time one mode end to end and return its per-step statistics."""

    torch.manual_seed(args.seed)
    features, labels = make_features(
        args.num_samples,
        args.num_classes,
        args.feat_dim,
        args.seed,
        device,
    )
    model = Head(args.feat_dim, args.feat_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion, criterion_optimizer = make_criterion(
        args.loss,
        args.num_classes,
        args.feat_dim,
        args.lr,
        device,
    )
    device_type = torch.device(device).type

    compiled = mode in ("compile", "dataloader_compile")
    if mode in ("dataloader", "dataloader_compile"):
        source = DataLoaderSource(
            features,
            labels,
            args.batch_size,
            args.sampler_m,
            device,
            args.num_workers,
        )
    else:
        source = GpuResidentSource(
            features,
            labels,
            args.batch_size,
            args.sampler_m,
            device,
            args.seed,
        )

    if compiled:
        # Each mode is timed independently, so no mode may inherit another's
        # compiled kernels; that would hide its own compile cost.
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    forward_loss = make_forward_loss(model, criterion, compiled)

    step_times = []
    losses_seen = []
    steps_done = 0
    total_steps = args.warmup_steps + args.steps
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    wall_measured_start = None

    while steps_done < total_steps:
        for batch_features, batch_labels in source:
            if steps_done == args.warmup_steps:
                torch.cuda.synchronize(device)
                wall_measured_start = time.perf_counter()
            started = time.perf_counter()
            loss = eager_step(
                forward_loss,
                optimizer,
                criterion_optimizer,
                batch_features,
                batch_labels,
                device_type,
            )
            if args.sync_each_step:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if steps_done >= args.warmup_steps:
                step_times.append(elapsed)
                if len(losses_seen) < 8:
                    losses_seen.append(loss.detach())
            steps_done += 1
            if steps_done >= total_steps:
                break

    torch.cuda.synchronize(device)
    wall_total = time.perf_counter() - wall_measured_start
    peak_bytes = torch.cuda.max_memory_allocated(device)

    compile_stats = None
    if compiled:
        counters = torch._dynamo.utils.counters
        # A loss that recompiles per shape spends its speedup on compilation, so
        # these counts are what make a compile result interpretable.
        compile_stats = {
            "unique_graphs": counters["stats"].get("unique_graphs", 0),
            "graph_breaks": sum(counters["graph_break"].values()),
            "recompiles": sum(counters["recompiles"].values()) if "recompiles" in counters else 0,
        }

    return {
        "mode": mode,
        "steps": len(step_times),
        "wall_s": wall_total,
        "steps_per_s": len(step_times) / wall_total,
        "mean_step_ms": statistics.mean(step_times) * 1e3,
        "median_step_ms": statistics.median(step_times) * 1e3,
        "p90_step_ms": sorted(step_times)[int(0.9 * len(step_times)) - 1] * 1e3,
        "peak_mem_mb": peak_bytes / 1e6,
        "first_losses": [round(float(value), 5) for value in losses_seen[:4]],
        "setup_s": wall_measured_start - wall_start,
        "compile_stats": compile_stats,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--loss", default="MultiSimilarityLoss")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sampler_m", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--num_samples", type=int, default=8054, help="Cars196 train size by default")
    parser.add_argument("--num_classes", type=int, default=98)
    parser.add_argument("--steps", type=int, default=600, help="measured steps per mode")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for the baseline mode")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sync_each_step",
        action="store_true",
        help="synchronize after every step; measures true per-step latency instead of queued throughput",
    )
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--label", default="", help="tag recorded in the JSON output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this benchmark measures CUDA launch behavior; use --device cuda")

    results = []
    for mode in args.modes:
        torch.cuda.reset_peak_memory_stats(device)
        result = run_mode(mode, args, device)
        result["pid"] = os.getpid()
        result["label"] = args.label
        results.append(result)
        if result.get("skipped"):
            print(f"{mode:>19}: skipped - {result['skipped']}", flush=True)
        else:
            print(
                f"{result['mode']:>19}: {result['steps_per_s']:9.1f} steps/s   "
                f"mean {result['mean_step_ms']:6.3f} ms   "
                f"median {result['median_step_ms']:6.3f} ms   "
                f"p90 {result['p90_step_ms']:6.3f} ms   "
                f"peak {result['peak_mem_mb']:7.1f} MB",
                flush=True,
            )

    baseline = next(
        (item for item in results if item["mode"] == "dataloader" and not item.get("skipped")),
        None,
    )
    if baseline is not None:
        for result in results:
            if result.get("skipped") or result["mode"] == "dataloader":
                continue
            result["speedup_vs_dataloader"] = result["steps_per_s"] / baseline["steps_per_s"]
            print(
                f"{result['mode']:>19}: {result['speedup_vs_dataloader']:.2f}x vs dataloader",
                flush=True,
            )

    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
