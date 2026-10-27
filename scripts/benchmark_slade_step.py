"""Measure SLADE's frozen-head step with different gradient-probe cadences.

Uses the real losses, shared-parameter norm probes and Adam, but fixed synthetic
features/pseudo labels and lr=0 for a stable workload. Probe results do not change
the weights: this isolates their cost, not their effect on learning. Excludes
loading, feature extraction, clustering and retrieval. CUDA timings synchronize
only at measurement boundaries, not between operations inside each step.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_metric_learning import losses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses.metric_losses import FusedMultiSimilarityLoss, SyncFreeCircleLoss
from training.engine import measure_gradient_component_norms
from training.ssl.slade import SladeBasisLayer, SladeRegularizer


def benchmark(args, every):
    torch.manual_seed(7)
    device = torch.device(args.device)
    head = torch.nn.Linear(args.dim, args.dim).to(device)
    model = torch.nn.Module()
    model.add_module("head", head)
    model.add_module("slade_basis", SladeBasisLayer(args.dim, args.classes).to(device))
    labels = torch.arange(args.labeled_batch, device=device) // 2 % args.classes
    clusters = torch.arange(args.unlabeled_batch, device=device) // 4 % args.classes
    centers = torch.randn(args.classes, args.dim, device=device)
    features = torch.cat((centers[labels], centers[clusters]))
    features = features + 0.4 * torch.randn_like(features)
    positions = torch.arange(args.unlabeled_batch)
    batch = (None, None, positions)
    regularizer = SladeRegularizer(basis_warmup_steps=0, lambda_basis=0.25)
    regularizer._cluster_by_position = clusters.cpu().numpy()
    criterion = {
        "ms": lambda: FusedMultiSimilarityLoss(alpha=0.1545, beta=11.8265, base=0.79),
        "ntxent": lambda: losses.NTXentLoss(temperature=0.07),
        "circle": SyncFreeCircleLoss,
    }[args.loss]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    shared = tuple(head.parameters())

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        embeddings = F.normalize(head(features), dim=1)
        supervised = criterion(embeddings[:args.labeled_batch], labels)
        ranking = regularizer.compute_loss(
            model, None, batch, device,
            supervised_embeddings=embeddings[:args.labeled_batch],
            supervised_labels=labels,
            regularizer_embeddings=embeddings[args.labeled_batch:],
            supervised_criterion=criterion,
        )
        if every and index % every == 0:
            measure_gradient_component_norms(
                supervised, ranking, shared,
                supervised_weight=1.0, regularizer_weight=1.0,
                extra_components={"slade_basis": (regularizer._basis_loss, 1.0)},
                statistic=args.statistic, include_pair_norm=False,
            )
        if args.diagnostics_every and index % args.diagnostics_every == 0:
            measure_gradient_component_norms(
                supervised, ranking, shared,
                supervised_weight=1.0, regularizer_weight=1.0,
                extra_components=regularizer.extra_loss_components(),
            )
        combined = regularizer.combine_losses(supervised, ranking)
        combined.backward()
        optimizer.step()
        return combined.detach()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for index in range(4):
        step(index)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for index in range(args.steps):
        loss = step(index)
    sync()
    result = {
        "probe_every_steps": every,
        "probe_batches": sum(every > 0 and i % every == 0 for i in range(args.steps)),
        "mean_step_ms": (time.perf_counter() - start) * 1000 / args.steps,
        "final_loss": float(loss),
        "mining_active": regularizer._mining_active,
        "finite_gradients": all(bool(p.grad.isfinite().all()) for p in model.parameters()),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loss", choices=("ms", "ntxent", "circle"), default="ms")
    parser.add_argument("--labeled-batch", type=int, default=128)
    parser.add_argument("--unlabeled-batch", type=int, default=256)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--classes", type=int, default=2997)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--probe-every", nargs="+", type=int, default=[1, 4, 8, 0])
    parser.add_argument("--statistic", choices=("l2", "max_mean"), default="max_mean")
    parser.add_argument("--diagnostics-every", type=int, default=30)
    args = parser.parse_args()
    if min(args.dim, args.steps, args.labeled_batch) < 1 or min(args.classes, args.unlabeled_batch) < 2:
        parser.error("use positive sizes, at least two classes and two unlabeled samples")
    if min(args.probe_every) < 0 or args.diagnostics_every < 0:
        parser.error("probe/logging intervals must be nonnegative (0 disables)")
    torch.set_num_threads(1)
    logger.remove()
    print(json.dumps({"configuration": vars(args), "torch": torch.__version__}), flush=True)
    for every in args.probe_every:
        print(json.dumps(benchmark(args, every)), flush=True)


if __name__ == "__main__":
    main()
