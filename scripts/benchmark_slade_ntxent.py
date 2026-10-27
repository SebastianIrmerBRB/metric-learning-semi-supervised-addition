"""Profile SLADE's NT-Xent ranking loss on a dense synthetic mined pair set.

Run each implementation in a fresh process for comparable CUDA reserved memory:
    python scripts/benchmark_slade_ntxent.py --device cuda --batch-size 256
    python scripts/benchmark_slade_ntxent.py --device cuda --batch-size 64 --implementation upstream

The default is the bounded implementation. Upstream at 256 can require tens of
GB; use a smaller batch when validating it on a shared server. Measurements
exclude the backbone, basis loss, clustering and retrieval evaluation.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.loss_and_miner_utils import get_all_pairs_indices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.ssl.slade import SladeRegularizer, _both_directions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--gradient-probes", type=int, default=1)
    parser.add_argument("--implementation", choices=("bounded", "upstream"), default="bounded")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.batch_size < 4 or not 2 <= args.clusters <= args.batch_size // 2:
        parser.error("use batch-size >= 4 and 2 <= clusters <= batch-size/2")
    if args.repeats < 1 or args.dim < 1 or args.gradient_probes < 0:
        parser.error("repeats/dim must be positive and gradient-probes non-negative")
    if args.temperature <= 0:
        parser.error("temperature must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = torch.device(args.device)
    embeddings = torch.randn(args.batch_size, args.dim, device=device, requires_grad=True)
    labels = torch.arange(args.batch_size, device=device) % args.clusters
    a, p, b, n = get_all_pairs_indices(labels)
    positive = (a[a < p], p[a < p])
    negative = (b[b < n], n[b < n])
    pairs = (*_both_directions(positive), *_both_directions(negative))
    criterion = losses.NTXentLoss(temperature=args.temperature).to(device)
    regularizer = SladeRegularizer()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        embeddings.grad = None
        if args.implementation == "bounded":
            loss = regularizer._unlabeled_ranking_loss(
                embeddings, labels, positive, negative, criterion, False,
            )
        else:
            loss = criterion(embeddings, labels, pairs)
        for _ in range(args.gradient_probes):
            torch.autograd.grad(loss, embeddings, retain_graph=True)
        loss.backward()
        return loss.detach()

    saved_sizes = []
    saved_storages = {}

    def pack(tensor):
        saved_sizes.append(tensor.numel())
        storage = tensor.untyped_storage()
        saved_storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        step()
    for _ in range(2):
        step()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        loss = step()
        sync()
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "implementation": args.implementation,
        "device": str(device),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "clusters": args.clusters,
        "gradient_probes": args.gradient_probes,
        "directed_positive_pairs": len(pairs[0]),
        "directed_negative_pairs": len(pairs[2]),
        "upstream_one_fp32_pair_product_gib": len(pairs[0]) * len(pairs[2]) * 4 / 1024 ** 3,
        "largest_saved_tensor_elements": max(saved_sizes, default=0),
        "unique_saved_storage_mib": sum(saved_storages.values()) / 1024 ** 2,
        "median_step_ms": statistics.median(durations),
        "loss": float(loss),
        "finite_gradient": bool(embeddings.grad.isfinite().all()),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024 ** 2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024 ** 2,
        )
    rendered = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
