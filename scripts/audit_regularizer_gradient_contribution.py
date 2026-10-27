"""Audit how much gradient the SSL regularizer contributes across HPO trials.

Walks a log root (e.g. ``logs/ssl``), takes one cross-validation fold per trial,
and pairs the tuned parameters from ``run_config.json`` with the
``train/gradient_contribution/*`` diagnostics recorded in ``diagnostics.csv``.
One CSV is written per regularizer+loss combination, plus a summary that turns
the observed ratios into a suggested ``regularizer_weight`` search range.

The suggestion assumes the gradient ratio scales linearly with the weight, which
holds for every ``loss + weight * regularizer`` objective as long as the other
regularizer parameters stay fixed. The per-trial ``ratio_per_weight`` spread in
the group file shows how badly the remaining parameters break that assumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

GRADIENT_PREFIX = "train/gradient_contribution/"
RATIO_NAME = f"{GRADIENT_PREFIX}regularizer_to_supervised_ratio"
FRACTION_NAME = f"{GRADIENT_PREFIX}regularizer_fraction"
COSINE_NAME = f"{GRADIENT_PREFIX}cosine_similarity"
SUPERVISED_NAME = f"{GRADIENT_PREFIX}supervised_norm"
REGULARIZER_NAME = f"{GRADIENT_PREFIX}regularizer_norm"

TRIAL_DIR_PATTERN = re.compile(r"^trial_\d+$")


def percentile(sorted_values, fraction):
    """Linear-interpolated percentile over an already sorted list."""

    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summarize(values):
    """Mean/median/geometric-mean/quantiles for one diagnostic series."""

    if not values:
        return {}
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    positive = [value for value in ordered if value > 0.0]
    geomean = (
        math.exp(sum(math.log(value) for value in positive) / len(positive))
        if len(positive) == count
        else None
    )
    return {
        "mean": mean,
        "median": percentile(ordered, 0.5),
        "geomean": geomean,
        "p10": percentile(ordered, 0.1),
        "p90": percentile(ordered, 0.9),
        "min": ordered[0],
        "max": ordered[-1],
        "count": count,
    }


def read_gradient_series(diagnostics_path):
    """Collect every gradient-contribution series logged in one fold."""

    series = defaultdict(list)
    epochs = []
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name") or ""
            if not name.startswith(GRADIENT_PREFIX):
                continue
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            series[name].append(value)
            if name == RATIO_NAME:
                try:
                    epochs.append(int(row["epoch"]))
                except (TypeError, ValueError):
                    pass
    return series, epochs


def describe_method(ssl_config):
    """Name the regularizer being audited, falling back to the SSL method."""

    method = ssl_config.get("method") or "unknown"
    method_params = ssl_config.get("method_params") or {}
    regularizer = method_params.get("regularizer")
    name = regularizer if method == "regularized" and regularizer else method
    return name, method, method_params


def flatten(prefix, params):
    """Flatten one level of scalar parameters into prefixed columns."""

    flat = {}
    for key, value in (params or {}).items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        flat[f"{prefix}{key}"] = value
    return flat


def run_branch(fold_dir, root):
    """Name the log branch a fold belongs to, e.g. ``old_ser/seraph_cir``.

    Studies are re-run under new branches while keeping the same study name, and
    those generations usually search different weight ranges, so the branch has to
    be part of the grouping key or the ranges get averaged together.
    """

    try:
        parts = fold_dir.relative_to(root).parts
    except ValueError:
        parts = fold_dir.parts
    marker = next((i for i, part in enumerate(parts) if part in ("hpo", "final")), None)
    if marker is None:
        return "/".join(parts[:-1]) or "."
    # The component just before the marker is the loss/seed experiment name, which
    # is already broken out into its own columns.
    return "/".join(parts[: max(0, marker - 1)]) or "."


def collect_fold(fold_dir, run_config_path, diagnostics_path, root):
    """Build one output row from a single fold, or None if it logged no ratios."""

    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_args = config.get("args") or {}
    ssl_config = config.get("ssl_config") or run_args.get("ssl_config_resolved") or {}
    if not ssl_config:
        return None

    series, epochs = read_gradient_series(diagnostics_path)
    ratios = series.get(RATIO_NAME)
    if not ratios:
        return None

    name, method, method_params = describe_method(ssl_config)
    weight = method_params.get("regularizer_weight")
    ratio = summarize(ratios)
    fraction = summarize(series.get(FRACTION_NAME, []))
    cosine = summarize(series.get(COSINE_NAME, []))
    supervised = summarize(series.get(SUPERVISED_NAME, []))
    regularizer_norm = summarize(series.get(REGULARIZER_NAME, []))

    row = {
        "regularizer": name,
        "ssl_method": method,
        "loss": run_args.get("loss"),
        "miner": run_args.get("miner"),
        "dataset": run_args.get("dataset"),
        "study": run_args.get("hparam_study_name") or "final_runs",
        "branch": run_branch(fold_dir, root),
        "trial": run_args.get("trial_number"),
        "fold": run_args.get("cv_fold"),
        "cv_k": run_args.get("cv_k"),
        "regularizer_weight": weight,
        "supervised_weight": method_params.get("supervised_weight"),
        "warmup_epochs": ssl_config.get("warmup_epochs"),
        "batch_size": run_args.get("batch_size"),
        "sampler_m": run_args.get("sampler_m"),
        "lr": run_args.get("lr"),
        "classifier_lr": run_args.get("classifier_lr"),
        "labeled_per_class": ssl_config.get("labeled_per_class"),
        "label_sampling_mode": ssl_config.get("label_sampling_mode"),
    }
    row.update(flatten("loss.", run_args.get("loss_params")))
    row.update(flatten("reg.", method_params.get("regularizer_params")))
    row.update(
        {
            "n_logged_batches": ratio["count"],
            "epoch_first": min(epochs) if epochs else None,
            "epoch_last": max(epochs) if epochs else None,
            "ratio_mean": ratio["mean"],
            "ratio_median": ratio["median"],
            "ratio_geomean": ratio["geomean"],
            "ratio_p10": ratio["p10"],
            "ratio_p90": ratio["p90"],
            "ratio_min": ratio["min"],
            "ratio_max": ratio["max"],
            "fraction_mean": fraction.get("mean"),
            "cosine_mean": cosine.get("mean"),
            "supervised_norm_median": supervised.get("median"),
            "regularizer_norm_median": regularizer_norm.get("median"),
        }
    )
    # The weight is the only knob that scales the ratio exactly, so normalising by
    # it is what makes trials with different weights comparable.
    if isinstance(weight, (int, float)) and weight > 0:
        row["ratio_per_weight"] = ratio["mean"] / weight
    else:
        row["ratio_per_weight"] = None
    row["log_dir"] = str(fold_dir)
    return row


def iter_run_dirs(root):
    """Yield (trial_key, ordering_key, fold_dir) for every fold below root."""

    for diagnostics_path in root.rglob("diagnostics.csv"):
        fold_dir = diagnostics_path.parent
        if not (fold_dir / "run_config.json").exists():
            continue
        parts = fold_dir.parts
        trial_index = next(
            (i for i in range(len(parts) - 1, -1, -1) if TRIAL_DIR_PATTERN.match(parts[i])),
            None,
        )
        if trial_index is None:
            # A standalone (non-HPO) run: it is its own group of one.
            yield str(fold_dir), ("", fold_dir.name), fold_dir
            continue
        trial_key = str(Path(*parts[: trial_index + 1]))
        rest = parts[trial_index + 1 :]
        cv_name = rest[0] if rest else ""
        fold_name = rest[1] if len(rest) > 1 else ""
        # Newest cross-validation run first, then the lowest fold index.
        yield trial_key, (_invert(cv_name), fold_name), fold_dir


def _invert(name):
    """Sort key that orders timestamp-like names newest first."""

    return tuple(-ord(char) for char in name)


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)) if value is not None else "unknown"


def write_group(path, rows):
    """Write one group CSV with a stable, union-of-keys header."""

    preferred = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in preferred:
                preferred.insert(preferred.index("n_logged_batches"), key)
    rows.sort(key=lambda row: (row.get("regularizer_weight") is None, row.get("regularizer_weight") or 0.0))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(key, rows, target_low, target_high):
    """Turn per-trial ratios into an observed range and a suggested weight range."""

    regularizer, loss, miner, dataset, study, branch = key
    weights = sorted(
        row["regularizer_weight"]
        for row in rows
        if isinstance(row.get("regularizer_weight"), (int, float))
    )
    ratios = sorted(row["ratio_mean"] for row in rows)
    per_weight = sorted(row["ratio_per_weight"] for row in rows if row.get("ratio_per_weight"))
    in_band = [
        row
        for row in rows
        if target_low <= row["ratio_mean"] <= target_high
        and isinstance(row.get("regularizer_weight"), (int, float))
    ]
    in_band_weights = sorted(row["regularizer_weight"] for row in in_band)
    median_per_weight = percentile(per_weight, 0.5)
    # Invert ratio = k * weight for the band edges. The median k gives the weights
    # that hit the band for a typical trial; the p10/p90 of k widen that into a
    # range that still covers the band once the other parameters move k around.
    typical_low = target_low / median_per_weight if per_weight else None
    typical_high = target_high / median_per_weight if per_weight else None
    suggested_low = target_low / percentile(per_weight, 0.9) if per_weight else None
    suggested_high = target_high / percentile(per_weight, 0.1) if per_weight else None

    return {
        "regularizer": regularizer,
        "loss": loss,
        "miner": miner,
        "dataset": dataset,
        "study": study,
        "branch": branch,
        "n_trials": len(rows),
        "weight_min": weights[0] if weights else None,
        "weight_max": weights[-1] if weights else None,
        "ratio_min": ratios[0] if ratios else None,
        "ratio_p10": percentile(ratios, 0.1),
        "ratio_median": percentile(ratios, 0.5),
        "ratio_p90": percentile(ratios, 0.9),
        "ratio_max": ratios[-1] if ratios else None,
        "frac_below_target": sum(1 for value in ratios if value < target_low) / len(ratios),
        "frac_in_target": len(in_band) / len(rows),
        "frac_above_target": sum(1 for value in ratios if value > target_high) / len(ratios),
        "ratio_per_weight_p10": percentile(per_weight, 0.1),
        "ratio_per_weight_median": median_per_weight,
        "ratio_per_weight_p90": percentile(per_weight, 0.9),
        "observed_in_band_weight_min": in_band_weights[0] if in_band_weights else None,
        "observed_in_band_weight_max": in_band_weights[-1] if in_band_weights else None,
        "typical_weight_low": typical_low,
        "typical_weight_high": typical_high,
        "suggested_weight_low": suggested_low,
        "suggested_weight_high": suggested_high,
        "cosine_mean": (
            sum(row["cosine_mean"] for row in rows if row.get("cosine_mean") is not None)
            / max(1, sum(1 for row in rows if row.get("cosine_mean") is not None))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, default=Path("logs/ssl"), help="Log root to scan.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/reports/gradient_contribution"),
        help="Directory for the per-combination CSV files.",
    )
    parser.add_argument(
        "--target-ratio-low",
        type=float,
        default=0.1,
        help="Lower edge of the regularizer/supervised gradient ratio you want to reach.",
    )
    parser.add_argument(
        "--target-ratio-high",
        type=float,
        default=1.0,
        help="Upper edge of the wanted gradient ratio band.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Aggregate every fold instead of one fold per trial.",
    )
    parser.add_argument(
        "--trials-only",
        action="store_true",
        help="Skip standalone runs that live outside a trial_XXXX directory.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"No such log root: {args.root}", file=sys.stderr)
        return 2

    candidates = defaultdict(list)
    for trial_key, ordering, fold_dir in iter_run_dirs(args.root):
        if args.trials_only and trial_key == str(fold_dir):
            continue
        candidates[trial_key].append((ordering, fold_dir))

    groups = defaultdict(list)
    skipped_no_ratio = 0
    unreadable = []
    for trial_key, folds in sorted(candidates.items()):
        for _, fold_dir in sorted(folds):
            try:
                row = collect_fold(
                    fold_dir, fold_dir / "run_config.json", fold_dir / "diagnostics.csv", args.root
                )
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                unreadable.append((fold_dir, exc))
                continue
            if row is None:
                skipped_no_ratio += 1
                continue
            key = (
                row["regularizer"],
                row["loss"],
                row["miner"],
                row["dataset"],
                row["study"],
                row["branch"],
            )
            groups[key].append(row)
            if not args.all_folds:
                break

    if not groups:
        print(f"No gradient-contribution diagnostics found below {args.root}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        rows = groups[key]
        stem = "__".join(sanitize(part) for part in key if part is not None)
        write_group(args.out / f"{stem}.csv", rows)
        summaries.append(summarize_group(key, rows, args.target_ratio_low, args.target_ratio_high))

    summary_path = args.out / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Scanned {len(candidates)} trials below {args.root}")
    print(f"Wrote {len(summaries)} combination files + summary.csv to {args.out}")
    if skipped_no_ratio:
        print(f"Skipped {skipped_no_ratio} folds without gradient-contribution rows (no regularizer or all warmup)")
    for path, exc in unreadable[:5]:
        print(f"UNREADABLE {path}: {exc}", file=sys.stderr)

    header = (
        f"{'regularizer':<16}{'loss':<22}{'branch':<26}{'n':>4}"
        f"{'searched weight':>20}{'ratio p10..p90':>22}{'in band':>9}"
        f"{'typical weight':>20}{'covering weight':>22}"
    )
    print()
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item['regularizer']:<16}{str(item['loss']):<22}{str(item['branch'])[:25]:<26}"
            f"{item['n_trials']:>4}"
            f"{_fmt_range(item['weight_min'], item['weight_max']):>20}"
            f"{_fmt_range(item['ratio_p10'], item['ratio_p90']):>22}"
            f"{item['frac_in_target'] * 100:>8.0f}%"
            f"{_fmt_range(item['typical_weight_low'], item['typical_weight_high']):>20}"
            f"{_fmt_range(item['suggested_weight_low'], item['suggested_weight_high']):>22}"
        )
    print(
        f"\nTarget gradient ratio band: {args.target_ratio_low:g}..{args.target_ratio_high:g}"
        " (regularizer grad norm / supervised grad norm)."
        "\n'typical weight' hits that band for a median trial, 'covering weight' still"
        " reaches it at the p10/p90 of the other regularizer parameters."
    )
    return 0


def _fmt_range(low, high):
    if low is None or high is None:
        return "n/a"
    return f"{low:.3g}..{high:.3g}"


if __name__ == "__main__":
    raise SystemExit(main())
