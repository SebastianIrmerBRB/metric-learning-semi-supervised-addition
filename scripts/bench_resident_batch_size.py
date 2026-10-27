"""Batch-size sweep for GPU-resident vs DataLoader feature batching.

Answers two separate questions:

* training -- how the resident gather scales with the (searched) batch size;
* evaluation -- which batch size the sequential embedding pass should use,
  which is free to choose because it changes no result.

Run: venv/bin/python scripts/bench_resident_batch_size.py --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

import utils


class ProjectionHead(nn.Module):
    """One 768->768 projection + L2 norm, the shape a frozen-backbone run trains."""

    def __init__(self, feature_dim, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, features):
        return nn.functional.normalize(self.projection(features), dim=1)


def make_dataset(rows, dim, classes, seed):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, dim, generator=generator)
    labels = torch.arange(rows) % classes
    features = centers[labels] + 0.35 * torch.randn(rows, dim, generator=generator)
    return utils.PrecomputedBackboneFeatureDataset(features, labels.tolist())


def timed(iterate, warmup_batches, measure_batches, device):
    """Run the loader/step loop and return (seconds, batches, samples)."""

    seen = 0
    samples = 0
    started = None
    for batch in iterate():
        if seen == warmup_batches:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            samples = 0
        seen += 1
        samples += len(batch[1])
        if started is not None and seen - warmup_batches >= measure_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if started is None:
        return None, 0, 0
    return time.perf_counter() - started, seen - warmup_batches, samples


def train_case(dataset, args, batch_size, resident):
    device = torch.device(args.device)
    from pytorch_metric_learning import losses

    torch.manual_seed(91)
    model = ProjectionHead(args.feature_dim, args.embedding_dim).to(device)
    criterion = losses.MultiSimilarityLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    loader = utils.make_train_loader(
        dataset,
        batch_size,
        args.sampler_m,
        seed=0,
        num_workers=0,
        start_method="default",
        persistent_workers=False,
        pin_memory=torch.device(args.device).type == "cuda",
        gpu_resident_device=args.device if resident else None,
        gpu_resident_max_bytes=None,
    )
    is_resident = type(loader).__name__ == "GpuResidentFeatureLoader"
    if resident and not is_resident:
        return None

    def iterate():
        while True:
            for batch in loader:
                features, labels = batch[0], batch[1]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = criterion(
                        model(features.to(device, non_blocking=True)),
                        labels.to(device, non_blocking=True),
                    )
                loss.backward()
                optimizer.step()
                yield batch

    seconds, batches, samples = timed(iterate, args.warmup, args.steps, device)
    return {
        "batch_size": batch_size,
        "mode": "resident" if is_resident else "dataloader",
        "batches": batches,
        "steps_per_s": batches / seconds,
        "samples_per_s": samples / seconds,
        "ms_per_step": 1000 * seconds / batches,
    }


def eval_case(dataset, args, batch_size, resident):
    device = torch.device(args.device)
    torch.manual_seed(91)
    model = ProjectionHead(args.feature_dim, args.embedding_dim).to(device).eval()
    loader = utils.make_eval_loader(
        dataset,
        batch_size=batch_size,
        seed=0,
        num_workers=0,
        start_method="default",
        pin_memory=torch.device(args.device).type == "cuda",
        gpu_resident_device=args.device if resident else None,
    )
    is_resident = type(loader).__name__ == "GpuResidentFeatureLoader"
    if resident and not is_resident:
        return None

    def iterate():
        while True:
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                for features, labels in loader:
                    model(features.to(device, non_blocking=True))
                    yield features, labels

    passes = max(1, args.eval_samples // max(1, len(dataset)))
    measure = max(1, passes * len(loader))
    seconds, batches, samples = timed(iterate, len(loader), measure, device)
    return {
        "batch_size": batch_size,
        "mode": "resident" if is_resident else "dataloader",
        "batches": batches,
        "samples_per_s": samples / seconds,
        "ms_per_batch": 1000 * seconds / batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--classes", type=int, default=98)
    parser.add_argument("--sampler_m", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval_samples", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--train_batch_sizes", type=int, nargs="+",
        default=[32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--eval_batch_sizes", type=int, nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    dataset = make_dataset(args.rows, args.feature_dim, args.classes, seed=0)
    results = {"train": [], "eval": [], "config": vars(args)}

    # One throwaway pass warms the CUDA context and the caching allocator, so
    # the first measured case is not charged for work every later case reuses.
    train_case(dataset, args, args.train_batch_sizes[0], False)
    train_case(dataset, args, args.train_batch_sizes[0], True)

    def sweep(case, batch_sizes, key):
        for batch_size in batch_sizes:
            if batch_size > args.rows:
                continue
            if key == "train" and batch_size > args.sampler_m * args.classes:
                # MPerClassSampler needs batch_size / m distinct classes.
                print(f"skipping train batch_size={batch_size}: needs "
                      f"{batch_size // args.sampler_m} classes, have {args.classes}",
                      flush=True)
                continue
            runs = {"dataloader": [], "resident": []}
            # Repeats interleave the two modes so any drift over the sweep hits
            # both arms equally instead of only the one measured second.
            for _ in range(args.repeats):
                for resident in (False, True):
                    run = case(dataset, args, batch_size, resident)
                    if run is not None:
                        runs[run["mode"]].append(run)
            for mode in ("dataloader", "resident"):
                if not runs[mode]:
                    continue
                ordered = sorted(runs[mode], key=lambda run: run["samples_per_s"])
                median = ordered[len(ordered) // 2]
                median["samples_per_s_spread"] = (
                    ordered[-1]["samples_per_s"] / ordered[0]["samples_per_s"]
                )
                results[key].append(median)
                print(json.dumps(median), flush=True)

    sweep(train_case, args.train_batch_sizes, "train")
    sweep(eval_case, args.eval_batch_sizes, "eval")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
