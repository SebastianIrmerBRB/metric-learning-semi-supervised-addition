#!/usr/bin/env python3
"""Held-out test retrieval before any optimization, per CV fold and per seed.

This is ``epoch0_test``: the model as constructed, evaluated on D_test with no
training step taken. ``scripts/epoch0_test_baseline.py`` reports it once for a
full-development fit; this reports it across the CV grid so the floor can be
compared with the per-fold numbers a real run logs.

What actually moves the number:

* **the seed** -- with ``--feat_dim 768`` the head is a randomly initialized
  projection of the frozen backbone, so each seed is a different projection;
* **not the fold** -- ``training/engine.py`` seeds every fold from ``args.seed``
  and D_test is fixed by the protocol, so an untrained model is identical
  across folds. The fold column is here to show that spread is exactly zero,
  which is the point: any fold-to-fold variance a trained run reports is
  produced by training, not by the split.

With ``--feat_dim none`` the head is an Identity, the seed stops mattering too,
and every cell collapses to the raw DINOv2 CLS embedding's score.

Examples::

    python scripts/epoch0_test_folds.py --dataset NABirds --seeds 0 1 2
    python scripts/epoch0_test_folds.py --dataset VehicleID \
        --dataset_protocol vehicleid_test_2400 --cv_k 5 --seeds 0 1
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
    parser.add_argument("--cv_k", type=int, default=4, help="number of folds")
    parser.add_argument("--cv_mode", default="group_kfold", help="fold assignment mode")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        metavar="FOLD",
        help="evaluate only these fold indices (default: all of them)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2], metavar="SEED"
    )
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
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="reuse the run-time backbone feature cache; worth it across a grid",
    )
    parser.add_argument("--json_out", default=None)
    return parser.parse_args(argv)


def evaluate_cell(args, seed, cv_fold):
    """Score D_test for one (seed, fold) cell with no training step taken."""

    bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=seed,
        data_split_seed=seed,
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_fold=cv_fold,
        cv_mode=args.cv_mode,
        dataset_protocol=args.dataset_protocol,
        full_train=cv_fold is None,
        image_resize_mode=args.image_resize_mode,
    )
    test_dataset = bundle.test_dataset

    # Seed before construction, exactly as engine.run_fold does, so the random
    # projection head matches what epoch 0 of a real run would hold.
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
        use_cache=bool(args.use_cache),
        projection_layers=1,
        projection_hidden_dim=None,
        device=args.device,
    )
    ssl_config = semi_supervised.load_ssl_config(None, default_seed=seed)
    model = make_run_model(model_args, ssl_config).to(args.device)
    model.eval()

    batch_size = args.batch_size or utils.frozen_feature_eval_batch_size(
        test_dataset, args.device
    )
    loader = utils.make_eval_loader(
        test_dataset,
        batch_size=batch_size,
        seed=seed,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    outcome = utils.evaluate_split(
        model,
        loader,
        f"epoch0 test seed={seed} fold={cv_fold}",
        device=args.device,
        return_per_class=False,
        recall_at_k=utils.normalize_recall_at_k(args.recall_at_k),
    )
    return {
        "seed": seed,
        "cv_fold": cv_fold,
        "train_images": len(bundle.train_dataset),
        "test_classes": len(set(test_dataset.labels)),
        "test_images": len(test_dataset),
        "precision_at_1": float(outcome.precision_at_1),
        "mean_average_precision_at_r": float(outcome.mean_average_precision_at_r),
        "recall_at_k": {int(k): float(v) for k, v in dict(outcome.recall_at_k).items()},
    }


def spread(values):
    """Return (min, max, stdev) for a list of scores; stdev is 0 for one value."""

    return min(values), max(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main(argv=None):
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    utils.validate_dataset_protocol(args.dataset, args.dataset_protocol)

    folds = list(range(args.cv_k)) if args.folds is None else list(args.folds)
    if args.cv_k < 2:
        folds = [None]

    results = []
    for seed in args.seeds:
        for fold in folds:
            print(f"seed {seed} fold {fold} ...", file=sys.stderr)
            results.append(evaluate_cell(args, seed, fold))

    print()
    print(
        f"Untrained test retrieval - {args.dataset} / {args.dataset_protocol}, "
        f"DINOv2-{args.dino_size}, feat_dim={args.feat_dim}, "
        f"cv_k={args.cv_k}, resize={args.image_resize_mode}"
    )
    header = f"{'seed':>5} {'fold':>5} {'train img':>10} {'test cls':>9} {'test img':>9}  {'P@1':>7} {'MAP@R':>7}"
    ks = sorted(results[0]["recall_at_k"]) if results else []
    header += "".join(f"  {'R@'+str(k):>7}" for k in ks)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        line = (
            f"{row['seed']:>5} {str(row['cv_fold']):>5} {row['train_images']:>10} "
            f"{row['test_classes']:>9} {row['test_images']:>9}  "
            f"{row['precision_at_1']:>6.2%} {row['mean_average_precision_at_r']:>6.2%}"
        )
        line += "".join(f"  {row['recall_at_k'][k]:>6.2%}" for k in ks)
        print(line)
    print("-" * len(header))

    # The two spreads answer different questions: across folds at a fixed seed
    # it should be exactly zero, across seeds it is the head's init noise.
    for seed in args.seeds:
        per_fold = [r["precision_at_1"] for r in results if r["seed"] == seed]
        if len(per_fold) > 1:
            low, high, sd = spread(per_fold)
            print(
                f"seed {seed}: P@1 across {len(per_fold)} folds "
                f"min {low:.4%} max {high:.4%} sd {sd:.4%}"
                + ("  (identical, as expected)" if high - low < 1e-12 else "")
            )
    per_seed = [
        statistics.mean([r["precision_at_1"] for r in results if r["seed"] == seed])
        for seed in args.seeds
    ]
    if len(per_seed) > 1:
        low, high, sd = spread(per_seed)
        print(
            f"across {len(args.seeds)} seeds: P@1 min {low:.4%} max {high:.4%} sd {sd:.4%}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
