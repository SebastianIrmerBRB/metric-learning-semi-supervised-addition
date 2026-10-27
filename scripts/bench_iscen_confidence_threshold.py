"""Find which part of Iscen's ``confidence_threshold`` range actually does anything.

The threshold is a hard cutoff: ``filter_pseudo_labels`` keeps a prediction when
its entropy-certainty is ``>= confidence_threshold``. Whether a searched range is
useful therefore depends entirely on where Iscen's certainty values actually land,
and those are not spread over [0, 1] - LP-DeepSSL's certainty is
``1 - H(p) / log(num_classes)``, which concentrates low when the class count is
large and the diffusion is broad. Any part of the searched range above the
observed maximum certainty is a dead zone: every value in it produces the same
run.

Two things make "keeps nothing" the wrong description of that dead zone, and both
are measured here rather than assumed:

``class-capacity rescue``
    With a two-stream sampler the filter is given ``required_pseudo_label_classes``.
    When the threshold leaves fewer distinct classes than that, ``filter_pseudo_labels``
    re-admits the top ``sampler_m`` predictions of missing classes at
    ``pseudo_label_rescue_confidence_floor`` (0.0 by default). Above the rescue
    trigger point the threshold stops selecting confident pseudo-labels and starts
    selecting *arbitrary* ones, so the run is not "supervised-only" - it is
    training on rescue picks.
``the accuracy/volume trade``
    Coverage falls faster than accuracy rises, so the quantity that matters is
    correct pseudo-labels per unlabeled node, reported alongside both halves.

Everything runs through the real ``iscen_label_spreading`` and the real
``filter_pseudo_labels``, on cached DINOv2 rows, the runs' ``class_subset_k_shot``
split and the class-disjoint CV training fold. Per-dataset defaults are the best
trial of that dataset's tuned Iscen study, so the sweep is read at an operating
point the HPO actually chose.

Examples::

    python scripts/bench_iscen_confidence_threshold.py --datasets Cars196
    python scripts/bench_iscen_confidence_threshold.py --search-range 0.2 0.99
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bench_graph_neighborhood_size import (  # noqa: E402
    DATASET_K_SHOT,
    apply_head,
    select_training_fold,
    subsample_nodes,
)
from scripts.bench_seraph_gradient_contribution import (  # noqa: E402
    load_feature_source,
    make_k_shot_split,
)
from training.ssl.algorithms import iscen_label_spreading  # noqa: E402
from training.ssl.config import UNLABELED_TARGET, PseudoLabelResult  # noqa: E402
from training.ssl.pseudo_labels import filter_pseudo_labels  # noqa: E402


# Best trial of each dataset's tuned Iscen study, the same operating points the
# n_neighbors ablation bundles pin:
#   Cars196   scheduled/iscen/prx_loss/.../faiss_label_spreading_search (0.6420)
#   In-Shop   ssl/in_shop_all_gpu/iscen_prx/.../in-shop                 (0.6804)
# ``batch_size``/``sampler_m``/``labeled_batch_size`` are what set the two-stream
# class requirement, which is what makes high thresholds behave the way they do.
DATASET_OPERATING_POINT = {
    "Cars196": {
        "n_neighbors": 15,
        "gamma": 5.0,
        "alpha": 0.6364550947151395,
        "confidence_threshold": 0.7813009632573198,
        "batch_size": 128,
        "sampler_m": 4,
        "labeled_batch_size": 32,
    },
    "DeepFashionInShop": {
        "n_neighbors": 3,
        "gamma": 8.0,
        "alpha": 0.2088868676631643,
        "confidence_threshold": 0.5518935732637209,
        "batch_size": 128,
        "sampler_m": 2,
        "labeled_batch_size": 64,
    },
}

DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
DEFAULT_CG_RTOL = 1e-5
DEFAULT_CG_MAX_ITER = 2000
DEFAULT_LINEAR_SOLVER = "torch_block_cg"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cars196", "DeepFashionInShop"],
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--search-range",
        nargs=2,
        type=float,
        default=[0.2, 0.99],
        help="Searched interval to report dead-zone coverage for",
    )
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--no-rescue",
        action="store_true",
        help="Ignore the two-stream class requirement, as a single-stream run would",
    )
    parser.add_argument("--cg-rtol", type=float, default=DEFAULT_CG_RTOL)
    parser.add_argument("--cg-max-iter", type=int, default=DEFAULT_CG_MAX_ITER)
    parser.add_argument("--linear-solver", default=DEFAULT_LINEAR_SOLVER)
    parser.add_argument(
        "--head",
        choices=["identity", "random", "trained"],
        default="trained",
    )
    parser.add_argument("--feat-dim", type=int, default=768)
    parser.add_argument("--head-steps", type=int, default=500)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--cv-k", type=int, default=4)
    parser.add_argument("--data-split-seed", type=int, default=7)
    parser.add_argument("--support-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "scripts/reports/bench_graph_neighborhood_size/confidence_threshold.json"
        ),
    )
    return parser.parse_args()


def confidence_quantiles(confidences):
    levels = [0, 1, 10, 25, 50, 75, 90, 99, 100]
    values = np.percentile(confidences, levels)
    return {f"p{level}": float(value) for level, value in zip(levels, values)}


def coverage_at(confidences, threshold):
    return float((confidences >= threshold).mean())


def dead_zone(confidences, search_range):
    """Describe how much of a searched interval cannot change the run.

    A threshold only does something between the smallest and the largest observed
    certainty. Below the minimum every value keeps the whole pool; above the
    maximum every value keeps nothing (before rescue). Both tails of the searched
    interval are spent on trials that differ in the recorded hyperparameter and
    nowhere else.
    """

    low, high = float(search_range[0]), float(search_range[1])
    minimum = float(confidences.min())
    maximum = float(confidences.max())
    span = high - low
    dead_below = max(0.0, min(minimum, high) - low)
    dead_above = max(0.0, high - max(maximum, low))
    return {
        "search_low": low,
        "search_high": high,
        "min_confidence": minimum,
        "max_confidence": maximum,
        "dead_below_fraction": float(dead_below / span) if span > 0 else 0.0,
        "dead_above_fraction": float(dead_above / span) if span > 0 else 0.0,
        "dead_fraction_of_range": float((dead_below + dead_above) / span) if span > 0 else 0.0,
        "coverage_at_search_low": coverage_at(confidences, low),
        "coverage_at_search_high": coverage_at(confidences, high),
    }


def run_dataset(dataset_name, args, device):
    operating_point = DATASET_OPERATING_POINT[dataset_name]
    n_neighbors = args.n_neighbors or operating_point["n_neighbors"]
    gamma = args.gamma if args.gamma is not None else operating_point["gamma"]
    alpha = args.alpha if args.alpha is not None else operating_point["alpha"]

    source = load_feature_source(dataset_name, args.split, args.data_root, "cpu")
    labels_all = source.labels.numpy().astype(np.int64)
    fold_positions = select_training_fold(labels_all, args.cv_fold, args.cv_k, args.data_split_seed)
    fold_positions = subsample_nodes(fold_positions, labels_all, args.max_nodes, args.seed)
    labels = labels_all[fold_positions]
    features_cpu = source.features[torch.as_tensor(fold_positions)]

    labeled_local, _ = make_k_shot_split(
        torch.as_tensor(labels), DATASET_K_SHOT[dataset_name], args.support_seed
    )
    labeled_mask = np.zeros(len(labels), dtype=bool)
    labeled_mask[labeled_local] = True
    targets = np.where(labeled_mask, labels, UNLABELED_TARGET).astype(np.int64)

    embeddings = apply_head(features_cpu, labels, labeled_mask, args, device)
    features = embeddings.detach().cpu().numpy().astype(np.float32)

    probabilities, confidences = iscen_label_spreading(
        features=features,
        targets=targets,
        num_classes=int(labels.max()) + 1,
        n_neighbors=int(n_neighbors),
        gamma=float(gamma),
        alpha=float(alpha),
        cg_rtol=args.cg_rtol,
        cg_max_iter=args.cg_max_iter,
        linear_solver=args.linear_solver,
    )

    unlabeled_mask = ~labeled_mask
    unlabeled_positions = np.flatnonzero(unlabeled_mask)
    unlabeled_probabilities = probabilities[unlabeled_mask]
    propagated = unlabeled_probabilities.sum(axis=1) > 0.0
    # The adapter drops unpropagatable rows before the filter ever sees them.
    raw = PseudoLabelResult(
        positions=unlabeled_positions[propagated],
        mapped_labels=np.argmax(unlabeled_probabilities[propagated], axis=1).astype(np.int64),
        confidences=confidences[unlabeled_mask][propagated].astype(np.float32),
    )
    truth_by_position = dict(zip(unlabeled_positions, labels[unlabeled_mask]))

    # ``_pseudo_label_capacity_filter_kwargs``: a two-stream batch needs one
    # M-per-class group per pseudo class, and rescue may take up to sampler_m.
    sampler_m = int(operating_point["sampler_m"])
    pseudo_batch_size = int(operating_point["batch_size"]) - int(operating_point["labeled_batch_size"])
    required_classes = None if args.no_rescue else (pseudo_batch_size + sampler_m - 1) // sampler_m
    valid_mapped_labels = set(int(value) for value in np.unique(labels))

    summary = {
        "nodes": int(len(labels)),
        "classes": int(len(np.unique(labels))),
        "labeled": int(labeled_mask.sum()),
        "unlabeled": int(unlabeled_mask.sum()),
        "propagated": int(propagated.sum()),
        "n_neighbors": int(n_neighbors),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "tuned_confidence_threshold": float(operating_point["confidence_threshold"]),
        "required_pseudo_label_classes": required_classes,
        "rescue_top_k": sampler_m,
        "head": args.head,
        "confidence_quantiles": confidence_quantiles(raw.confidences),
        "dead_zone": dead_zone(raw.confidences, args.search_range),
    }
    logger.info(f"{dataset_name}: {json.dumps(summary)}")

    records = []
    previous_kept = None
    for threshold in args.thresholds:
        accepted = filter_pseudo_labels(
            pseudo_labels=raw,
            confidence_threshold=threshold,
            valid_mapped_labels=valid_mapped_labels,
            required_classes=required_classes,
            rescue_confidence_floor=0.0,
            rescue_top_k=sampler_m,
        )
        # The same filter without the sampler requirement isolates how much of the
        # accepted set exists only because rescue put it back.
        unrescued = filter_pseudo_labels(
            pseudo_labels=raw,
            confidence_threshold=threshold,
            valid_mapped_labels=valid_mapped_labels,
        )
        truth = np.array([truth_by_position[position] for position in accepted.positions])
        correct = accepted.mapped_labels == truth if len(truth) else np.array([], dtype=bool)
        record = {
            "threshold": float(threshold),
            "kept": int(len(accepted.positions)),
            "coverage": float(len(accepted.positions) / max(1, int(unlabeled_mask.sum()))),
            "accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "correct_per_node": (
                float(correct.sum() / max(1, int(unlabeled_mask.sum()))) if len(correct) else 0.0
            ),
            "classes_kept": int(len(np.unique(accepted.mapped_labels))) if len(accepted.positions) else 0,
            "kept_before_rescue": int(len(unrescued.positions)),
            "rescued": int(len(accepted.positions) - len(unrescued.positions)),
            "mean_confidence": (
                float(np.mean(accepted.confidences)) if len(accepted.positions) else float("nan")
            ),
        }
        record["rescue_active"] = record["rescued"] > 0
        record["sampler_satisfied"] = (
            None if required_classes is None else record["classes_kept"] >= required_classes
        )
        record["same_as_previous"] = previous_kept is not None and record["kept"] == previous_kept
        previous_kept = record["kept"]
        logger.info(f"threshold={threshold}: {json.dumps(record)}")
        records.append(record)

    return {"summary": summary, "records": records}


COLUMNS = [
    ("thresh", "threshold", "{:>6.2f}"),
    ("kept", "kept", "{:>7}"),
    ("coverage", "coverage", "{:>8.4f}"),
    ("acc", "accuracy", "{:>6.3f}"),
    ("corr/node", "correct_per_node", "{:>9.4f}"),
    ("classes", "classes_kept", "{:>7}"),
    ("pre-rescue", "kept_before_rescue", "{:>10}"),
    ("rescued", "rescued", "{:>7}"),
    ("mean_conf", "mean_confidence", "{:>9.4f}"),
]


def print_table(title, records, required_classes):
    from scripts.bench_graph_neighborhood_size import _column_width

    print(f"\n{title}")
    header = "  ".join(
        f"{name:>{max(len(name), _column_width(fmt))}}" for name, _, fmt in COLUMNS
    )
    header += "  flags"
    print(header)
    print("-" * len(header))
    for record in records:
        cells = []
        for name, key, fmt in COLUMNS:
            value = record.get(key)
            width = max(len(name), _column_width(fmt))
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append(f"{'-':>{width}}")
            else:
                cells.append(f"{fmt.format(value):>{width}}")
        flags = []
        if record["rescue_active"]:
            flags.append("RESCUE")
        if required_classes is not None and not record["sampler_satisfied"]:
            flags.append("UNDER-CLASSED")
        if record["same_as_previous"]:
            flags.append("=prev")
        print("  ".join(cells) + "  " + ",".join(flags))


def main():
    args = parse_args()
    args.thresholds = sorted(set(float(value) for value in args.thresholds))
    device = torch.device(args.device)
    results = {"config": vars(args), "datasets": {}}

    for dataset_name in args.datasets:
        block = run_dataset(dataset_name, args, device)
        summary = block["summary"]
        quantiles = summary["confidence_quantiles"]
        zone = summary["dead_zone"]
        print(
            f"\n=== {dataset_name} | {summary['nodes']} nodes, {summary['classes']} classes, "
            f"{summary['unlabeled']} unlabeled, k={summary['n_neighbors']}, "
            f"gamma={summary['gamma']}, alpha={summary['alpha']:.3f}, head={summary['head']} ==="
        )
        print(
            "certainty distribution: "
            + ", ".join(f"{name}={value:.4f}" for name, value in quantiles.items())
        )
        print(
            f"searched range [{zone['search_low']}, {zone['search_high']}]: certainty spans "
            f"[{zone['min_confidence']:.4f}, {zone['max_confidence']:.4f}], so the range is "
            f"{zone['dead_fraction_of_range'] * 100:.1f}% dead "
            f"({zone['dead_below_fraction'] * 100:.1f}% below keeps everything, "
            f"{zone['dead_above_fraction'] * 100:.1f}% above keeps nothing). "
            f"Coverage at the ends: {zone['coverage_at_search_low'] * 100:.2f}% -> "
            f"{zone['coverage_at_search_high'] * 100:.2f}%."
        )
        print_table(
            f"Iscen confidence threshold - {dataset_name} "
            f"(tuned value {summary['tuned_confidence_threshold']:.3f}, "
            f"sampler needs {summary['required_pseudo_label_classes']} classes)",
            block["records"],
            summary["required_pseudo_label_classes"],
        )
        results["datasets"][dataset_name] = block

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
