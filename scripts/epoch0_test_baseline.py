#!/usr/bin/env python3
"""Measure held-out test retrieval before any optimization.

This is the number ``training/engine.py`` logs as ``epoch0_test`` for a final
full-development fit: the model as constructed, evaluated on D_test with no
training step taken. It is the floor every trained run has to beat, and it is
worth having per protocol because a protocol that leaks classes between the
halves starts from a higher floor.

The model is built through ``make_run_model`` and the split through
``setup_dataset_bundle``, so the score matches what a run would log rather than
approximating it. With ``--feat_dim 768`` (the default, matching the phase_1
configs) the embedding is a randomly initialized projection of the frozen
backbone, which is exactly what epoch 0 of such a run sees; ``--feat_dim none``
drops the projection and measures the raw DINOv2 CLS embedding instead.

Examples::

    python scripts/epoch0_test_baseline.py
    python scripts/epoch0_test_baseline.py --all_protocols --recall_at_k 1 2 4 8
    python scripts/epoch0_test_baseline.py --feat_dim none --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger

import utils
from training.engine import make_run_model
from training import semi_supervised


NABIRDS_PROTOCOLS = (
    utils.DATASET_PROTOCOL_OFFICIAL,
    utils.DATASET_PROTOCOL_NABIRDS_ALTERNATING_277_278,
    utils.DATASET_PROTOCOL_NABIRDS_PUMA_278_277,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="NABirds", help="dataset name")
    parser.add_argument(
        "--dataset_protocol",
        default=utils.DATASET_PROTOCOL_NABIRDS_PUMA_278_277,
        help="split protocol to evaluate",
    )
    parser.add_argument(
        "--all_protocols",
        action="store_true",
        help="evaluate every NABirds protocol so the floors are directly comparable",
    )
    parser.add_argument(
        "--image_resize_mode",
        "--image-resize-mode",
        choices=utils.IMAGE_RESIZE_MODES,
        default=utils.DEFAULT_IMAGE_RESIZE_MODE,
        help=utils.format_image_resize_mode_help(),
    )
    parser.add_argument("--dino_size", default="b", choices=["s", "b", "l", "g"])
    parser.add_argument(
        "--feat_dim",
        default="768",
        help="projection width, or 'none' for the raw backbone CLS embedding",
    )
    parser.add_argument("--seed", type=int, default=0, help="seeds the random projection head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None, help="default: the run-time frozen-feature size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recall_at_k", type=int, nargs="*", default=[1, 2, 4, 8], metavar="K")
    parser.add_argument("--use_cache", action="store_true", help="reuse the run-time backbone feature cache")
    parser.add_argument("--json_out", default=None, help="also write the results to this JSON file")
    return parser.parse_args(argv)


def make_model_args(args, dataset_protocol):
    """Build the argparse-shaped namespace make_run_model expects."""

    feat_dim = None if str(args.feat_dim).lower() in {"none", "null", ""} else int(args.feat_dim)
    return argparse.Namespace(
        dataset=args.dataset,
        dataset_protocol=dataset_protocol,
        dino_size=args.dino_size,
        feat_dim=feat_dim,
        backbone_tuning="frozen",
        use_cache=bool(args.use_cache),
        projection_layers=1,
        projection_hidden_dim=None,
        device=args.device,
    )


def evaluate_protocol(args, dataset_protocol):
    """Return the untrained test scores and the split those scores describe."""

    bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        data_split_seed=args.seed,
        cv_k=1,
        cv_fold=None,
        dataset_protocol=dataset_protocol,
        full_train=True,
        image_resize_mode=args.image_resize_mode,
    )
    test_dataset = bundle.test_dataset

    # Seed before construction so the random projection head is reproducible;
    # with --feat_dim none the head is an Identity and the seed does not matter.
    utils.seed_everything(args.seed, device=args.device)
    model_args = make_model_args(args, dataset_protocol)
    ssl_config = semi_supervised.load_ssl_config(None, default_seed=args.seed)
    model = make_run_model(model_args, ssl_config).to(args.device)
    model.eval()

    batch_size = args.batch_size or utils.frozen_feature_eval_batch_size(test_dataset, args.device)
    loader = utils.make_eval_loader(
        test_dataset,
        batch_size=batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )

    recall_at_k = utils.normalize_recall_at_k(args.recall_at_k)
    outcome = utils.evaluate_split(
        model,
        loader,
        f"{dataset_protocol} test epoch 0 - no optimization",
        device=args.device,
        return_per_class=False,
        recall_at_k=recall_at_k,
    )

    split_info = dict(getattr(bundle, "split_info", {}) or {})
    protocol_info = dict(split_info.get("dataset_protocol", {}) or {})
    return {
        "dataset_protocol": dataset_protocol,
        "image_resize_mode": split_info.get("image_resize_mode"),
        "class_split_version": protocol_info.get("class_split_version"),
        "test_classes": len(set(test_dataset.labels)),
        "test_images": len(test_dataset),
        "straddling_parent_count": protocol_info.get("straddling_parent_count"),
        "precision_at_1": float(outcome.precision_at_1),
        "mean_average_precision_at_r": float(outcome.mean_average_precision_at_r),
        "recall_at_k": {int(k): float(v) for k, v in dict(outcome.recall_at_k).items()},
    }


def format_row(result):
    recalls = " ".join(f"R@{k} {value:6.2%}" for k, value in sorted(result["recall_at_k"].items()))
    return (
        f"{result['dataset_protocol']:<28} "
        f"{result['test_classes']:>4} cls {result['test_images']:>6} img  "
        f"straddling {str(result['straddling_parent_count']):>3}  "
        f"P@1 {result['precision_at_1']:6.2%}  "
        f"MAP@R {result['mean_average_precision_at_r']:6.2%}  {recalls}"
    )


def main(argv=None):
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    protocols = NABIRDS_PROTOCOLS if args.all_protocols else (args.dataset_protocol,)
    for protocol in protocols:
        utils.validate_dataset_protocol(args.dataset, protocol)

    results = []
    for protocol in protocols:
        print(f"evaluating {protocol} ...", file=sys.stderr)
        results.append(evaluate_protocol(args, protocol))

    print()
    print(
        f"Untrained test retrieval - {args.dataset}, DINOv2-{args.dino_size}, "
        f"feat_dim={args.feat_dim}, resize={args.image_resize_mode}"
    )
    print("-" * 118)
    for result in results:
        print(format_row(result))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
