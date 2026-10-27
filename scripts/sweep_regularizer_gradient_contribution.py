"""Short training runs that measure the regularizer's gradient share directly.

``scripts/audit_regularizer_gradient_contribution.py`` reads the ratio out of
finished HPO studies, which means waiting for a study before learning whether a
weight range was sensible. This script runs the real training path for a couple
of epochs per condition with per-step gradient-contribution logging, so a weight
range (or a ``regularizer_target_ratio``) can be checked in minutes.

What it sweeps, as a full cross product:

* regularizer: ``lrml``, ``hoffer_entropy``, ``seraph``
* scale knob: ``--weights`` (fixed ``regularizer_weight``) and/or
  ``--target-ratios`` (calibrated weight, ``method_params.regularizer_target_ratio``)
* loss and loss parameters: the ``--loss-grid`` entries, because the supervised
  gradient norm - the denominator of the ratio - is mostly set by them.

Every condition trains from the same seed and split, so the numbers differ only
by what the sweep varied. Warmup is forced to 0 and the epoch length is capped so
the regularized phase starts at step 0.

Examples::

    python scripts/sweep_regularizer_gradient_contribution.py --regularizers lrml
    python scripts/sweep_regularizer_gradient_contribution.py \\
        --weights 10 100 500 --losses ArcFaceLoss NTXentLoss
    python scripts/sweep_regularizer_gradient_contribution.py \\
        --target-ratios 0.05 0.2 1.0 --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_regularizer_gradient_contribution import (  # noqa: E402
    RATIO_NAME,
    percentile,
    read_gradient_series,
    summarize,
)


# Regularizer parameters follow the most recent Cars196 studies for each method,
# so a sweep starts from a setting the tuner actually selected rather than from
# library defaults.
REGULARIZER_DEFAULTS = {
    "lrml": {
        "method": "regularized",
        "regularizer_params": {
            "graph_batch_size": 512,
            "graph_on": "all",
            "n_neighbors": 15,
            "normalized_laplacian": True,
            "reduction": "mean",
        },
        # Bracketing the range the finished lrml studies searched.
        "weights": [10.0, 100.0, 500.0],
    },
    "hoffer_entropy": {
        "method": "regularized",
        "regularizer_params": {
            "distance_scale": 11.28,
            "reference_sets": 2,
        },
        "weights": [1.0, 10.0, 100.0],
    },
    "seraph": {
        "method": "seraph",
        "regularizer_params": {
            "include_labeled_unlabeled": True,
            "kappa": 1.32,
            "kappa_mode": "batch_std",
            "metric": "identity",
            "objective": "entropy",
            "positive_pair_prior": 0.04,
            "threshold": 1.13,
            "threshold_mode": "quantile",
            "threshold_momentum": 0.14,
            "unlabeled_batch_size": 128,
        },
        "weights": [0.5, 2.0, 10.0],
    },
}


# Two settings per loss, chosen at the ends of the searched ranges, because the
# loss parameters move the supervised gradient norm more than the weight does.
DEFAULT_LOSS_GRID = {
    "ArcFaceLoss": [
        {"margin": 28.6, "scale": 16.0},
        {"margin": 28.6, "scale": 200.0},
    ],
    "NTXentLoss": [
        {"temperature": 0.05},
        {"temperature": 0.3},
    ],
    "ProxyAnchorLoss": [
        {"alpha": 8.0, "margin": 0.1},
        {"alpha": 64.0, "margin": 0.1},
    ],
    "CircleLoss": [
        {"gamma": 80.0, "m": 0.25},
        {"gamma": 400.0, "m": 0.25},
    ],
    "MultiSimilarityLoss": [
        {"alpha": 2.0, "base": 0.5, "beta": 40.0},
        {"alpha": 12.0, "base": 0.5, "beta": 120.0},
    ],
}

GRAD_NORM_SUPERVISED_NAME = "train/grad_norm/supervised_weight"
GRAD_NORM_REGULARIZER_NAME = "train/grad_norm/regularizer_weight"
GRAD_NORM_WEIGHT_NAMES = (GRAD_NORM_SUPERVISED_NAME, GRAD_NORM_REGULARIZER_NAME)

LOSS_MINERS = {"MultiSimilarityLoss": "MultiSimilarityMiner"}

CLASSIFICATION_LOSSES = {"ArcFaceLoss", "ProxyAnchorLoss"}


def build_ssl_config(regularizer, weight, target_ratio, grad_norm_alpha, args):
    """Compose the SSL config for one condition."""

    defaults = REGULARIZER_DEFAULTS[regularizer]
    method_params = {
        "supervised_weight": 1.0,
        "regularizer_params": dict(defaults["regularizer_params"]),
    }
    if defaults["method"] == "regularized":
        method_params["regularizer"] = regularizer
    if weight is not None:
        method_params["regularizer_weight"] = weight
    if target_ratio is not None:
        method_params["regularizer_target_ratio"] = target_ratio
        method_params["regularizer_target_ratio_batches"] = args.calibration_batches
        method_params["regularizer_target_ratio_stride"] = args.calibration_stride
    if grad_norm_alpha is not None:
        method_params["grad_norm_alpha"] = grad_norm_alpha
        method_params["grad_norm_renormalize"] = args.grad_norm_renormalize
        method_params["grad_norm_lr"] = args.grad_norm_lr
    return {
        "method": defaults["method"],
        "method_params": method_params,
        # The regularizer has to be active from the first step, or a two-epoch
        # probe would spend its budget on supervised warmup.
        "warmup_epochs": 0,
        "update_mode": "once",
        "label_sampling_mode": "class_subset_k_shot",
        "labeled_per_class": args.labeled_per_class,
        "labeled_fraction": 1.0,
        "embedding_batch_size": 32,
        "embedding_num_workers": 0,
        "pseudo_label_diagnostics_mode": "off",
        "seed": args.seed,
        "support_seed": args.seed,
    }


def scale_label(condition, separator="_"):
    """Name the knob that sets the regularizer's share of the gradient."""

    if condition["grad_norm_alpha"] is not None:
        return f"gradnorm{separator}{condition['grad_norm_alpha']:g}"
    if condition["target_ratio"] is not None:
        return f"ratio{separator}{condition['target_ratio']:g}"
    return f"weight{separator}{condition['weight']:g}"


def condition_slug(condition):
    scale = scale_label(condition)
    params = "_".join(
        f"{name}_{value:g}" if isinstance(value, float) else f"{name}_{value}"
        for name, value in sorted(condition["loss_params"].items())
    )
    return f"{condition['regularizer']}__{condition['loss']}__{params}__{scale}"


def build_conditions(args, loss_grid):
    scales = [(weight, None, None) for weight in args.weights or []]
    scales += [(None, ratio, None) for ratio in args.target_ratios or []]
    scales += [(None, None, alpha) for alpha in args.grad_norm_alphas or []]
    conditions = []
    for regularizer in args.regularizers:
        regularizer_scales = scales or [
            (weight, None, None) for weight in REGULARIZER_DEFAULTS[regularizer]["weights"]
        ]
        for loss in args.losses:
            for loss_params, (weight, target_ratio, alpha) in itertools.product(
                loss_grid[loss],
                regularizer_scales,
            ):
                conditions.append(
                    {
                        "regularizer": regularizer,
                        "loss": loss,
                        "miner": LOSS_MINERS.get(loss, "no_miner"),
                        "loss_params": loss_params,
                        "weight": weight,
                        "target_ratio": target_ratio,
                        "grad_norm_alpha": alpha,
                    }
                )
    return conditions


def run_condition(condition, args, sweep_dir):
    """Train one condition and return its per-epoch gradient-contribution rows."""

    # Imported lazily: the training package pulls in torch and the whole data
    # stack, which the argument parsing above should not pay for.
    from training import semi_supervised
    from training.cli import get_support_seed, parse_args_with_experiment_config
    from training.engine import run_training

    slug = condition_slug(condition)
    condition_dir = sweep_dir / slug
    condition_dir.mkdir(parents=True, exist_ok=True)
    ssl_config_path = condition_dir / "ssl_config.json"
    ssl_config_path.write_text(
        json.dumps(
            build_ssl_config(
                condition["regularizer"],
                condition["weight"],
                condition["target_ratio"],
                condition["grad_norm_alpha"],
                args,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    argv = [
        "--mode", "ssl",
        "--dataset", args.dataset,
        "--device", args.device,
        "--loss", condition["loss"],
        "--loss_params", json.dumps(condition["loss_params"]),
        "--miner", condition["miner"],
        "--ssl_config", str(ssl_config_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--sampler_m", str(args.sampler_m),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--cv_k", "1",
        "--feat_dim", str(args.feat_dim),
        "--backbone_tuning", "frozen",
        "--num_workers", "0",
        "--save_dir", str(condition_dir.relative_to("logs")),
        # Every step is measured: a two-epoch probe has few steps to spare.
        "--ssl_gradient_contribution_log_interval", "1",
        # The epoch length is what bounds the runtime of a condition. The plain
        # --length_before_new_iter is replaced by the fold pool size at run time;
        # only the override survives that.
        "--length_before_new_iter_override", str(args.steps_per_epoch * args.batch_size),
        "--patience", str(args.epochs + 1),
    ]
    if condition["loss"] in CLASSIFICATION_LOSSES:
        argv += ["--classifier_lr", str(args.classifier_lr)]
    if args.use_cache:
        argv += ["--use_cache"]

    run_args = parse_args_with_experiment_config(argv)
    ssl_config = semi_supervised.load_ssl_config(
        run_args.ssl_config,
        default_seed=run_args.seed,
        default_support_seed=get_support_seed(run_args),
    )
    started = time.perf_counter()
    run_training(run_args, ssl_config)
    duration = time.perf_counter() - started

    return collect_rows(condition, run_args.log_dir, duration, slug)


def collect_rows(condition, log_dir, duration, slug):
    """Summarize one finished condition per epoch and once overall."""

    diagnostics_path = Path(log_dir) / "diagnostics.csv"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"no diagnostics.csv under {log_dir}")

    per_epoch = {}
    learned_weights = {}
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name")
            try:
                value = float(row["value"])
                epoch = int(row["epoch"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if name == RATIO_NAME:
                per_epoch.setdefault(epoch, []).append(value)
            elif name in GRAD_NORM_WEIGHT_NAMES:
                # GradNorm keeps moving the weights, so the last value is the one
                # that describes where the run ended up.
                learned_weights[name] = value
    series, _ = read_gradient_series(diagnostics_path)

    calibrated_weight = None
    run_config_path = Path(log_dir) / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        calibrated_weight = run_config.get("args", {}).get("regularizer_weight_calibrated")

    rows = []
    for epoch in sorted(per_epoch) + ["all"]:
        values = (
            [value for epoch_values in per_epoch.values() for value in epoch_values]
            if epoch == "all"
            else per_epoch[epoch]
        )
        ratio = summarize(values)
        row = {
            "condition": slug,
            "regularizer": condition["regularizer"],
            "loss": condition["loss"],
            "miner": condition["miner"],
            "epoch": epoch,
            "regularizer_weight": condition["weight"],
            "target_ratio": condition["target_ratio"],
            "grad_norm_alpha": condition["grad_norm_alpha"],
            "calibrated_weight": calibrated_weight,
            "final_supervised_weight": learned_weights.get(GRAD_NORM_SUPERVISED_NAME),
            "final_regularizer_weight": learned_weights.get(GRAD_NORM_REGULARIZER_NAME),
            "steps": ratio["count"],
            "ratio_mean": ratio["mean"],
            "ratio_median": ratio["median"],
            "ratio_p10": ratio["p10"],
            "ratio_p90": ratio["p90"],
            "ratio_min": ratio["min"],
            "ratio_max": ratio["max"],
        }
        row.update(
            {f"loss.{name}": value for name, value in condition["loss_params"].items()}
        )
        if epoch == "all":
            for name, values in series.items():
                if name == RATIO_NAME:
                    continue
                row[f"{name.rsplit('/', 1)[-1]}_mean"] = summarize(values).get("mean")
            row["seconds"] = duration
        rows.append(row)
    return rows


def write_rows(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows, target_low, target_high):
    overall = [row for row in rows if row["epoch"] == "all"]
    if not overall:
        return
    header = (
        f"{'regularizer':<15}{'loss':<21}{'loss params':<26}"
        f"{'scale':>16}{'steps':>7}{'ratio med':>11}{'p10..p90':>20}{'in band':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in sorted(overall, key=lambda item: item["condition"]):
        scale = scale_label(row, separator="=")
        params = ", ".join(
            f"{name.split('.', 1)[1]}={value:g}"
            for name, value in row.items()
            if name.startswith("loss.") and isinstance(value, (int, float))
        )
        in_band = target_low <= row["ratio_median"] <= target_high
        print(
            f"{row['regularizer']:<15}{row['loss']:<21}{params[:25]:<26}"
            f"{scale:>16}{row['steps']:>7}{row['ratio_median']:>11.3g}"
            f"{_fmt_range(row['ratio_p10'], row['ratio_p90']):>20}"
            f"{('yes' if in_band else 'no'):>9}"
        )
    print(
        f"\nBand {target_low:g}..{target_high:g} is the regularizer/supervised "
        "gradient-norm ratio the weight is meant to reach."
    )
    calibrated = [row for row in overall if row["target_ratio"] is not None]
    if calibrated:
        print(
            "\nCalibrated conditions (target_ratio -> weight the trainer resolved):"
        )
        for row in sorted(calibrated, key=lambda item: item["condition"]):
            print(
                f"  {row['condition']}: target={row['target_ratio']:g} "
                f"weight={row['calibrated_weight']!r} "
                f"measured median={row['ratio_median']:.3g}"
            )
    learned = [row for row in overall if row["grad_norm_alpha"] is not None]
    if learned:
        print("\nGradNorm conditions (alpha -> weights the run ended on):")
        for row in sorted(learned, key=lambda item: item["condition"]):
            supervised = row["final_supervised_weight"]
            regularizer = row["final_regularizer_weight"]
            print(
                f"  {row['condition']}: alpha={row['grad_norm_alpha']:g} "
                f"w_sup={supervised if supervised is None else f'{supervised:.4g}'} "
                f"w_reg={regularizer if regularizer is None else f'{regularizer:.4g}'} "
                f"final ratio median={row['ratio_median']:.3g}"
            )


def _fmt_range(low, high):
    if low is None or high is None:
        return "n/a"
    return f"{low:.3g}..{high:.3g}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--regularizers",
        nargs="+",
        default=sorted(REGULARIZER_DEFAULTS),
        choices=sorted(REGULARIZER_DEFAULTS),
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["ArcFaceLoss", "NTXentLoss"],
        choices=sorted(DEFAULT_LOSS_GRID),
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Fixed regularizer_weight values. Defaults to a per-regularizer bracket.",
    )
    parser.add_argument(
        "--target-ratios",
        nargs="*",
        type=float,
        help="Calibrated regularizer_target_ratio values, swept instead of/next to --weights.",
    )
    parser.add_argument(
        "--grad-norm-alphas",
        nargs="*",
        type=float,
        help="GradNorm alpha values, swept instead of/next to the other scale knobs.",
    )
    parser.add_argument(
        "--grad-norm-renormalize",
        default="sum",
        choices=["sum", "supervised", "none"],
        help="Weight renormalization for GradNorm conditions.",
    )
    parser.add_argument("--grad-norm-lr", type=float, default=0.025)
    parser.add_argument(
        "--loss-grid",
        type=Path,
        help="JSON file mapping a loss name to a list of loss_params dicts.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=20,
        help="Optimizer steps per epoch; the main runtime knob.",
    )
    parser.add_argument("--calibration-batches", type=int, default=10)
    parser.add_argument(
        "--calibration-stride",
        type=int,
        default=1,
        help=(
            "Eligible batches between calibration measurements. The training "
            "default spreads them; a sweep this short has no transient to "
            "outrun, so it measures back to back unless asked otherwise."
        ),
    )
    parser.add_argument("--dataset", default="Cars196")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sampler_m", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-2)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--labeled_per_class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--target-ratio-low", type=float, default=0.1)
    parser.add_argument("--target-ratio-high", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/sweeps/gradient_contribution"),
        help="Run directory; the summary CSV is written at its root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the conditions without training.",
    )
    args = parser.parse_args()

    loss_grid = dict(DEFAULT_LOSS_GRID)
    if args.loss_grid is not None:
        loss_grid.update(json.loads(args.loss_grid.read_text(encoding="utf-8")))
    missing = [loss for loss in args.losses if loss not in loss_grid]
    if missing:
        print(f"No loss_params grid for: {missing}", file=sys.stderr)
        return 2

    conditions = build_conditions(args, loss_grid)
    print(
        f"{len(conditions)} conditions x {args.epochs} epochs x "
        f"{args.steps_per_epoch} steps"
    )
    if args.dry_run:
        for condition in conditions:
            print(f"  {condition_slug(condition)}")
        return 0

    sweep_dir = args.out
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, condition in enumerate(conditions, start=1):
        slug = condition_slug(condition)
        print(f"[{index}/{len(conditions)}] {slug}", flush=True)
        try:
            rows.extend(run_condition(condition, args, sweep_dir))
        except Exception as exc:  # one broken condition must not lose the sweep
            failures.append((slug, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)

    if not rows:
        print("No condition produced gradient-contribution rows", file=sys.stderr)
        return 1

    summary_path = sweep_dir / "sweep_summary.csv"
    write_rows(summary_path, rows)
    print(f"\nWrote {summary_path}")
    print_report(rows, args.target_ratio_low, args.target_ratio_high)
    if failures:
        print(f"\n{len(failures)} conditions failed:", file=sys.stderr)
        for slug, exc in failures:
            print(f"  {slug}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
