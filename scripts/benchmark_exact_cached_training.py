"""Synthetic cached-head benchmarks; no images, AMP, or changed batch sizes.

Run: venv/bin/python scripts/benchmark_exact_cached_training.py --device cuda
Original eager operations are retained in the exactness tests as references.
Timings include sampling, fetching, transfers, forward, backward, and Adam.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from loguru import logger
from torch.utils.benchmark import Timer
from torch.utils.data._utils.collate import default_collate

import utils
from losses.metric_losses import SyncFreeArcFaceLoss, SyncFreeCircleLoss
from tests.test_exact_cached_speedups import (
    ProjectionHead, original_hoffer_loss, original_mined_pairs,
    original_mining_state, original_reference_batches, original_update_distributions,
)
from training.semi_supervised import HofferEntropyRegularizer
from training.ssl.data import (
    HofferReferenceBatchSampler, HofferReferenceDataset, LRMLGraphDataset,
    RelabeledSubset, UnlabeledSubset,
)
from training.ssl.slade import SladeBasisLayer, SladeRegularizer


class OriginalBasis(SladeBasisLayer):
    update_distributions = original_update_distributions
    mining_state = original_mining_state


class OriginalSlade(SladeRegularizer):
    _mined_pairs = staticmethod(original_mined_pairs)


def measure(name, original, optimized, seconds):
    old = Timer("run()", globals={"run": original}).blocked_autorange(min_run_time=seconds).median
    new = Timer("run()", globals={"run": optimized}).blocked_autorange(min_run_time=seconds).median
    result = dict(name=name, original_ms=old * 1000, optimized_ms=new * 1000, speedup=old / new)
    print(json.dumps(result), flush=True)
    return result


def setup_step(method, optimized, args, source):
    torch.manual_seed(91)
    device, batch_size = args.device, args.batch_size
    model = ProjectionHead(args.feature_dim, args.embedding_dim).to(device)
    criterion = (SyncFreeCircleLoss() if method == "slade" else
                 SyncFreeArcFaceLoss(args.classes, args.embedding_dim)).to(device)
    regularizer = None
    supervised = RelabeledSubset(source, range(batch_size),
                                 [i // 4 for i in range(batch_size)], [i // 4 for i in range(batch_size)])
    indices = list(range(batch_size))
    if method == "hoffer":
        unlabeled_count = batch_size * 8
        dataset = HofferReferenceDataset(source, range(unlabeled_count),
                                        range(unlabeled_count, unlabeled_count + args.classes * 4))
        sampler = HofferReferenceBatchSampler(
            unlabeled_count, [np.arange(4) + unlabeled_count + i * 4 for i in range(args.classes)],
            batch_size, 91,
        )
        regularizer = HofferEntropyRegularizer()
        def batches():
            while True:
                yield from sampler if optimized else original_reference_batches(sampler)
        batch_iterator = iter(batches())
    elif method == "slade":
        basis_class = SladeBasisLayer if optimized else OriginalBasis
        regularizer_class = SladeRegularizer if optimized else OriginalSlade
        model.slade_basis = basis_class(args.embedding_dim, args.classes).to(device)
        regularizer = regularizer_class(basis_warmup_steps=0)
        regularizer._cluster_by_position = np.arange(len(source)) // 4
        dataset = UnlabeledSubset(source, range(batch_size, 2 * batch_size))
    parameters = list(model.parameters()) + list(criterion.parameters())
    # Keep the workload stable while still measuring the complete Adam step.
    optimizer = torch.optim.Adam(parameters, lr=0.0)
    def fetch(dataset, ids):
        return default_collate(dataset.__getitems__(ids) if optimized else [dataset[i] for i in ids])
    def step():
        optimizer.zero_grad(set_to_none=True)
        batch = fetch(supervised, indices)
        features, labels = batch[:2]
        if method == "slade":
            unlabeled = fetch(dataset, indices)
            embeddings = model.project_features(torch.cat((features, unlabeled[0])).to(device))
            labels = labels.to(device)
            supervised_loss = criterion(embeddings[:batch_size], labels)
            ranking = regularizer.compute_loss(
                model, None, unlabeled, device,
                supervised_embeddings=embeddings[:batch_size], supervised_labels=labels,
                regularizer_embeddings=embeddings[batch_size:], supervised_criterion=criterion,
            )
            loss = regularizer.combine_losses(supervised_loss, ranking)
        else:
            loss = criterion(model.project_features(features.to(device)), labels.to(device))
            if method == "hoffer":
                unlabeled = fetch(dataset, next(batch_iterator))
                entropy = (regularizer.compute_loss(model, None, unlabeled, device) if optimized else
                           original_hoffer_loss(regularizer, model, unlabeled, device))
                loss = loss + entropy
        loss.backward()
        optimizer.step()
        return loss.detach()
    return step, model, criterion


def benchmark_capture(args):
    """Measure the already-supported CUDA graph path independently of loading."""
    features = torch.randn(args.batch_size, args.feature_dim, device=args.device)
    labels = torch.arange(args.batch_size, device=args.device) // 4
    def setup(capture):
        torch.manual_seed(67)
        model = ProjectionHead(args.feature_dim, args.embedding_dim).to(args.device)
        criterion = SyncFreeArcFaceLoss(args.classes, args.embedding_dim).to(args.device)
        parameters = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.0)
        def forward_loss(inputs, targets):
            return criterion(model.project_features(inputs), targets)
        forward = torch.compile(forward_loss, backend="cudagraphs", fullgraph=True) if capture else forward_loss
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = forward(features, labels)
            loss.backward()
            optimizer.step()
            return loss.detach().clone()
        return step, parameters
    old, old_parameters = setup(False)
    captured, captured_parameters = setup(True)
    for _ in range(4):
        assert torch.equal(old(), captured()), "captured ArcFace loss mismatch"
        for left, right in zip(old_parameters, captured_parameters):
            assert torch.equal(left.grad, right.grad), "captured ArcFace gradient mismatch"
    measure("existing_arcface_cudagraph_step", old, captured, args.seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--classes", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    logger.remove()
    torch.set_num_threads(1)
    torch.manual_seed(39)
    print(json.dumps(dict(vars(args), torch=torch.__version__,
                          gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)), flush=True)
    if args.capture_only:
        if args.device != "cuda":
            parser.error("--capture-only requires CUDA")
        benchmark_capture(args)
        return
    count = args.batch_size * 8 + args.classes * 4
    source = utils.PrecomputedBackboneFeatureDataset(
        torch.randn(count * 2, args.feature_dim), np.arange(count) % args.classes,
        feature_indices=torch.randperm(count * 2)[:count],
    )
    ids = list(range(args.batch_size))
    for name, dataset in (
        ("supervised", source),
        ("propagation", RelabeledSubset(source, range(count), list(range(count)), list(range(count)))),
        ("slade", UnlabeledSubset(source, range(count))),
        ("hoffer", HofferReferenceDataset(source, range(count // 2), range(count // 2, count))),
        ("lrml", LRMLGraphDataset(source, range(count))),
    ):
        measure(name + "_fetch", lambda: default_collate([dataset[i] for i in ids]),
                lambda: default_collate(dataset.__getitems__(ids)), args.seconds)
    sampler = HofferReferenceBatchSampler(args.batch_size * 8,
        [np.arange(4) + 100000 + i * 4 for i in range(args.classes)], args.batch_size, 91)
    measure("hoffer_sample_epoch", lambda: list(original_reference_batches(sampler)), lambda: list(sampler), args.seconds)
    for method in ("supervised", "hoffer", "slade"):
        old, old_model, old_criterion = setup_step(method, False, args, source)
        new, new_model, new_criterion = setup_step(method, True, args, source)
        assert torch.equal(old(), new()), method + " loss mismatch"
        for left, right in zip(list(old_model.parameters()) + list(old_criterion.parameters()),
                               list(new_model.parameters()) + list(new_criterion.parameters())):
            assert torch.equal(left.grad, right.grad), method + " gradient mismatch"
        measure(method + "_training_step", old, new, args.seconds)


if __name__ == "__main__":
    main()
