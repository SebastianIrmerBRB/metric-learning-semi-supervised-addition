#!/usr/bin/env python3
"""Held-out test retrieval before any optimization, per CV fold and per seed.

This is ``epoch0_test``: the model as constructed, evaluated on D_test with no
training step taken. ``scripts/epoch0_test_baseline.py`` reports it once for a
full-development fit; this reports the whole seed x fold grid.

The grid costs one backbone pass, not one per cell. With ``backbone_tuning
=frozen`` the DINOv2 weights are pretrained, not seeded, so the test set's
backbone features are identical for every seed and every fold. They are
computed once and every cell is then just the projection head applied to them,
which is a matmul. (Note that ``--use_cache`` would not have helped here:
``utils.evaluate_split`` calls ``forward_cached`` without a cache key, so
``models/retrieval_model.py`` recomputes the backbone anyway.)

What actually moves the number:

* **the seed** -- with ``--feat_dim 768`` the head is a randomly initialized
  projection, so each seed is a different one;
* **not the fold** -- ``training/engine.py`` seeds every fold from ``args.seed``
  and D_test is fixed by the protocol, so an untrained model cannot differ
  across folds. The fold rows are measured rather than asserted, so the printed
  spread is evidence: it should be exactly zero.

With ``--feat_dim none`` the head is an Identity, the seed stops mattering too,
and every cell collapses to the raw DINOv2 CLS embedding's score.

Examples::

    python scripts/epoch0_test_folds.py --dataset CUB --seeds 0 1 2
    python scripts/epoch0_test_folds.py --dataset NABirds \
        --dataset_protocol nabirds_puma_278_277 --cv_k 5 --seeds 0 1 2
    python scripts/epoch0_test_folds.py --feat_dim none --json_out floor.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from training.engine import make_run_model
from training import semi_supervised


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="NABirds")
    parser.add_argument(
        "--dataset_protocol",
        default=utils.DATASET_PROTOCOL_OFFICIAL,
        help="split protocol to evaluate (default: official)",
    )
    parser.add_argument("--cv_k", type=int, default=5, help="number of folds")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        metavar="FOLD",
        help="evaluate only these fold indices (default: all of them)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], metavar="SEED")
    parser.add_argument(
        "--image_resize_mode",
        "--image-resize-mode",
        choices=utils.IMAGE_RESIZE_MODES,
        default=utils.DEFAULT_IMAGE_RESIZE_MODE,
    )
    parser.add_argument("--dino_size", default="b", choices=["s", "b", "l", "g"])
    parser.add_argument(
        "--feat_dim",
        default="768",
        help="projection width, or 'none' for the raw backbone CLS embedding",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recall_at_k", type=int, nargs="*", default=[1, 2, 4, 8], metavar="K")
    parser.add_argument("--json_out", default=None)
    return parser.parse_args(argv)


def build_model(args, seed):
    """Build the model exactly as engine.run_fold does for epoch 0."""

    # Seed before construction, so the random projection head matches what a
    # real run of this seed would hold before its first optimizer step.
    utils.seed_everything(seed, device=args.device)
    feat_dim = (
        None if str(args.feat_dim).lower() in {"none", "null", ""} else int(args.feat_dim)
    )
    model_args = argparse.Namespace(
        dataset=args.dataset,
        dataset_protocol=args.dataset_protocol,
        dino_size=args.dino_size,
        feat_dim=feat_dim,
        backbone_tuning="frozen",
        use_cache=False,
        projection_layers=1,
        projection_hidden_dim=None,
        device=args.device,
    )
    ssl_config = semi_supervised.load_ssl_config(None, default_seed=seed)
    model = make_run_model(model_args, ssl_config).to(args.device)
    model.eval()
    return model


def extract_backbone_features(args, model, test_dataset):
    """One frozen-backbone pass over D_test, shared by every cell."""

    batch_size = args.batch_size or utils.frozen_feature_eval_batch_size(
        test_dataset, args.device
    )
    loader = utils.make_eval_loader(
        test_dataset,
        batch_size=batch_size,
        seed=0,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    features, labels = [], []
    with torch.no_grad(), utils.eval_autocast(args.device):
        for images, batch_labels in tqdm(
            loader, desc="frozen backbone pass (once for the whole grid)"
        ):
            batch = model.forward_backbone(images.to(args.device, non_blocking=True))
            features.append(batch.detach().float().cpu())
            labels.append(torch.as_tensor(batch_labels).cpu())
    return torch.cat(features), torch.cat(labels)


def score_from_features(args, model, features, labels, name):
    """Project cached features through this seed's head and score retrieval."""

    # evaluate_split routes 2-D inputs to model.project_features, so this is
    # the head alone -- no backbone work.
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=args.batch_size or 1024,
        shuffle=False,
    )
    outcome = utils.evaluate_split(
        model,
        loader,
        name,
        device=args.device,
        return_per_class=False,
        recall_at_k=utils.normalize_recall_at_k(args.recall_at_k),
    )
    return {
        "precision_at_1": float(outcome.precision_at_1),
        "mean_average_precision_at_r": float(outcome.mean_average_precision_at_r),
        "recall_at_k": {int(k): float(v) for k, v in dict(outcome.recall_at_k).items()},
    }


def spread(values):
    return min(values), max(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main(argv=None):
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    utils.validate_dataset_protocol(args.dataset, args.dataset_protocol)

    folds = [None] if args.cv_k < 2 else (
        list(range(args.cv_k)) if args.folds is None else list(args.folds)
    )

    # D_test is fixed by the protocol: it depends on neither the seed nor the
    # fold, so it is built once.
    bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seeds[0],
        data_split_seed=args.seeds[0],
        cv_k=1,
        cv_fold=None,
        dataset_protocol=args.dataset_protocol,
        full_train=True,
        image_resize_mode=args.image_resize_mode,
    )
    test_dataset = bundle.test_dataset
    test_classes = len(set(test_dataset.labels))

    first_model = build_model(args, args.seeds[0])
    features, labels = extract_backbone_features(args, first_model, test_dataset)

    results = []
    for seed in args.seeds:
        model = first_model if seed == args.seeds[0] else build_model(args, seed)
        for fold in folds:
            scores = score_from_features(
                args, model, features, labels, f"seed={seed} fold={fold}"
            )
            results.append(
                {
                    "seed": seed,
                    "cv_fold": fold,
                    "cv_k": args.cv_k,
                    "test_classes": test_classes,
                    "test_images": len(test_dataset),
                    **scores,
                }
            )

    print()
    print(
        f"Untrained test retrieval - {args.dataset} / {args.dataset_protocol}, "
        f"DINOv2-{args.dino_size}, feat_dim={args.feat_dim}, "
        f"cv_k={args.cv_k}, resize={args.image_resize_mode}"
    )
    print(f"D_test: {test_classes} classes, {len(test_dataset)} images")
    ks = sorted(results[0]["recall_at_k"]) if results else []
    header = f"{'seed':>5} {'fold':>5}  {'P@1':>7} {'MAP@R':>7}" + "".join(
        f"  {'R@' + str(k):>7}" for k in ks
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        line = (
            f"{row['seed']:>5} {str(row['cv_fold']):>5}  "
            f"{row['precision_at_1']:>6.2%} {row['mean_average_precision_at_r']:>6.2%}"
        )
        line += "".join(f"  {row['recall_at_k'][k]:>6.2%}" for k in ks)
        print(line)
    print("-" * len(header))

    # Two different questions: across folds at a fixed seed this must be zero,
    # across seeds it is the projection head's initialization noise.
    for seed in args.seeds:
        per_fold = [r["precision_at_1"] for r in results if r["seed"] == seed]
        if len(per_fold) > 1:
            low, high, sd = spread(per_fold)
            note = "  (identical, as expected)" if high - low < 1e-12 else "  (UNEXPECTED)"
            print(
                f"seed {seed}: P@1 over {len(per_fold)} folds  "
                f"min {low:.4%}  max {high:.4%}  sd {sd:.4%}{note}"
            )
    if len(args.seeds) > 1:
        per_seed = [
            statistics.mean([r["precision_at_1"] for r in results if r["seed"] == seed])
            for seed in args.seeds
        ]
        low, high, sd = spread(per_seed)
        print(
            f"over {len(args.seeds)} seeds: P@1  min {low:.4%}  max {high:.4%}  sd {sd:.4%}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
