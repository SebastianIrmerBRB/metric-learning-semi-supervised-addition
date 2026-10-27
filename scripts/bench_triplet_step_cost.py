"""Measure the training-step cost of TripletMarginLoss against the five core losses.

Checks the thesis claim that the Triplet Loss trains much longer than the other
losses. The Phase-0 logs cannot answer it cleanly: trials ran concurrently, on
two hosts, and the scheduler rule ``TripletMarginLoss:all=99999`` put most
triplet trials on the CPU while the other losses ran on CUDA. This script
removes those confounders: one process, cells measured strictly one after
another, identical head, optimizer, data, and batch composition.

Each cell times one full frozen-backbone step (projection + L2 norm, miner,
loss, backward, Adam step) with the production loss classes from
``losses.metric_losses.LOSS_REGISTRY`` and the Phase-0 miners
(``MultiSimilarityMiner`` for MS, ``DistanceWeightedMiner`` for Triplet, none
for the rest). Features are already on the device, so data loading is
excluded. fp32, as in the Phase-0 runs (``train_amp=False``).

Example::

    python scripts/bench_triplet_step_cost.py --devices cpu:1 cpu:8 cuda \\
        --json_out scripts/reports/bench_triplet_step_cost/run1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pytorch_metric_learning import miners  # noqa: E402

from losses.metric_losses import LOSS_REGISTRY  # noqa: E402

# name -> (loss class, loss kwargs, miner factory or None, needs class count)
VARIANTS = {
    "ArcFace": ("ArcFaceLoss", {}, None),
    "Circle": ("CircleLoss", {"m": 0.4, "gamma": 80}, None),
    "MultiSimilarity": (
        "MultiSimilarityLoss",
        {"alpha": 2.0, "beta": 50.0, "base": 0.5},
        lambda: miners.MultiSimilarityMiner(epsilon=0.1),
    ),
    "NTXent": ("NTXentLoss", {"temperature": 0.07}, None),
    "ProxyAnchor": ("ProxyAnchorLoss", {}, None),
    # Phase-0 configuration.
    "Triplet+DWM": (
        "TripletMarginLoss",
        {"margin": 0.1},
        lambda: miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4),
    ),
    # Controls: every valid triplet, and the fused semihard miner path.
    "Triplet(all)": ("TripletMarginLoss", {"margin": 0.1}, None),
    "Triplet+TMM": (
        "TripletMarginLoss",
        {"margin": 0.1},
        lambda: miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard"),
    ),
}
PROXY_LOSSES = {"ArcFaceLoss", "ProxyAnchorLoss"}


class Head(torch.nn.Module):
    """The trainable part of a frozen-backbone run: one projection + L2 norm."""

    def __init__(self, dim):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, features):
        return F.normalize(self.fc(features), p=2.0, dim=1)


def make_data(num_classes, per_class, dim, seed):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=generator) * 4.0
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    features = centers[labels] + torch.randn(len(labels), dim, generator=generator)
    return features, labels


def make_batches(labels, num_classes, per_class, batch_size, m, count, seed):
    """Index batches with batch_size/m classes and m samples each (MPerClass)."""

    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        classes = torch.randperm(num_classes, generator=generator)[: batch_size // m]
        offsets = torch.stack(
            [torch.randperm(per_class, generator=generator)[:m] for _ in classes]
        )
        batches.append((classes[:, None] * per_class + offsets).reshape(-1))
    return batches


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(variant, batch_size, device_spec, features, labels, args):
    loss_name, loss_kwargs, miner_factory = VARIANTS[variant]
    if device_spec.startswith("cpu"):
        device = torch.device("cpu")
        torch.set_num_threads(int(device_spec.split(":")[1]) if ":" in device_spec else 1)
    else:
        device = torch.device(device_spec)
    torch.manual_seed(args.seed)
    head = Head(args.dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    kwargs = dict(loss_kwargs)
    if loss_name in PROXY_LOSSES:
        kwargs.update(num_classes=args.num_classes, embedding_size=args.dim)
    criterion = LOSS_REGISTRY[loss_name](**kwargs).to(device)
    criterion_optimizer = (
        torch.optim.Adam(criterion.parameters(), lr=1e-4) if loss_name in PROXY_LOSSES else None
    )
    miner = miner_factory() if miner_factory else None
    fused = (
        miner is not None
        and hasattr(criterion, "forward_with_miner")
        and criterion.supports_fused_miner(miner)
    )
    features = features.to(device)
    labels = labels.to(device)
    total = args.warmup + args.steps
    batches = make_batches(
        labels.cpu(), args.num_classes, args.per_class, batch_size, args.m, total, args.seed
    )

    times, mined = [], []
    for step, index in enumerate(batches):
        index = index.to(device)
        batch_features, batch_labels = features[index], labels[index]
        sync(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        if criterion_optimizer is not None:
            criterion_optimizer.zero_grad(set_to_none=False)
        embeddings = head(batch_features)
        if miner is None:
            loss = criterion(embeddings, batch_labels)
        elif fused:
            loss = criterion.forward_with_miner(embeddings, batch_labels, miner)
        else:
            tuples = miner(embeddings, batch_labels)
            loss = criterion(embeddings, batch_labels, tuples)
        loss.backward()
        optimizer.step()
        if criterion_optimizer is not None:
            criterion_optimizer.step()
        sync(device)
        if step >= args.warmup:
            times.append(time.perf_counter() - started)
            if miner is not None and not fused:
                mined.append(len(tuples[0]))
    return {
        "variant": variant,
        "loss": loss_name,
        "batch_size": batch_size,
        "device": device_spec,
        "fused_miner": fused,
        "steps": len(times),
        "median_ms": statistics.median(times) * 1e3,
        "p10_ms": sorted(times)[len(times) // 10] * 1e3,
        "p90_ms": sorted(times)[(9 * len(times)) // 10] * 1e3,
        "median_mined_tuples": statistics.median(mined) if mined else None,
    }


def report(cells, variants, batch_sizes, devices):
    lines = []
    for device in devices:
        lines.append(f"\n=== {device}: median ms per step (x = factor vs. median of the five core losses) ===")
        lines.append(f"{'variant':16}" + "".join(f"{f'b={b}':>16}" for b in batch_sizes))
        core = {}
        for b in batch_sizes:
            values = [
                c["median_ms"]
                for c in cells
                if c["device"] == device and c["batch_size"] == b and c["variant"] in list(VARIANTS)[:5]
            ]
            core[b] = statistics.median(values) if values else None
        for variant in variants:
            row = f"{variant:16}"
            for b in batch_sizes:
                cell = next(
                    (c for c in cells if c["device"] == device and c["batch_size"] == b and c["variant"] == variant),
                    None,
                )
                if cell is None:
                    row += f"{'-':>16}"
                    continue
                factor = cell["median_ms"] / core[b] if core[b] else float("nan")
                row += f"{f'{cell['median_ms']:.2f} ({factor:.1f}x)':>16}"
            lines.append(row)
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument(
        "--devices", nargs="+", default=["cpu:1", "cuda"],
        help="'cpu:<threads>' or a CUDA device; Phase 0 trained with one CPU thread",
    )
    parser.add_argument("--m", type=int, default=4, help="samples per class, as in the Phase-0 batch_sampler space")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json_out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    features, labels = make_data(args.num_classes, args.per_class, args.dim, args.seed)
    cells = []
    for device in args.devices:
        for batch_size in args.batch_sizes:
            for variant in args.variants:
                cell = measure(variant, batch_size, device, features, labels, args)
                cells.append(cell)
                print(
                    f"{device:6} b={batch_size:<4} {variant:16} median={cell['median_ms']:9.3f} ms"
                    f"  p10/p90={cell['p10_ms']:.3f}/{cell['p90_ms']:.3f}"
                    + (f"  mined={cell['median_mined_tuples']:.0f}" if cell["median_mined_tuples"] else ""),
                    flush=True,
                )
    print(report(cells, args.variants, args.batch_sizes, args.devices))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        args.json_out.write_text(
            json.dumps({"meta": meta, "args": {**vars(args), "json_out": str(args.json_out)}, "cells": cells}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
