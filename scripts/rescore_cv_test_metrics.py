#!/usr/bin/env python3
"""Recompute the D_test table of finished CV runs from their saved fold embeddings.

Runs recorded before the full test table existed only stored P@1 and MAP@R.
Every fold's ``test_embeddings.npz`` still holds the exact embeddings those
numbers came from, so the missing measurements can be added without training
again. For each ``cv_*`` directory this

1. backs up ``cv_summary.json`` and ``cv_results.csv`` as
   ``cv_summary.pre_rescore.json`` / ``cv_results.pre_rescore.csv`` (an existing
   backup is kept, so the backup always holds the original run's files),
2. evaluates every fold and the concatenated fold embeddings with the same
   ``utils.evaluate_embeddings`` a training run uses,
3. checks the stored P@1 and MAP@R (per fold, fold mean, concatenated) against
   the recomputed ones, and
4. rewrites both files through ``write_cross_validation_summary``, so they have
   exactly the fields and columns a fresh run writes.

A directory whose stored metrics do not reproduce within ``--tolerance`` is
reported and left untouched unless ``--allow_mismatch`` is given. Runs from
2026-09-26 on save their fold embeddings as float16, which reproduces P@1 only
to ~2.5e-4 and MAP@R to ~1e-5; those runs already write the full test table and
need no rescoring. Incomplete
runs (no aggregate) and runs with a missing fold embedding file are skipped.

Examples::

    python scripts/rescore_cv_test_metrics.py /mnt/f/server_logs/seeded_runs_cars --dry_run
    python scripts/rescore_cv_test_metrics.py /mnt/f/server_logs/seeded_runs_cars \
        --report rescore_cars.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from argparse import Namespace
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

import utils
from training.engine import (
    ConcatenatedEvaluationDataset,
    aggregate_cross_validation_result,
    evaluate_concatenated_fold_test_embeddings,
    write_cross_validation_summary,
)
from training.io import read_json, result_to_dict
from training.types import TrainingResult

SUMMARY_NAME = "cv_summary.json"
RESULTS_NAME = "cv_results.csv"
SUMMARY_BACKUP_NAME = "cv_summary.pre_rescore.json"
RESULTS_BACKUP_NAME = "cv_results.pre_rescore.csv"
CHECKED_METRICS = ("precision_at_1", "mean_average_precision_at_r")
REPORT_FIELDS = ("cv_dir", "scope", "metric", "stored", "recomputed", "abs_diff")
RESULT_FIELDS = {field.name for field in fields(TrainingResult)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="cv_* directories, or directories searched recursively for cv_summary.json",
    )
    parser.add_argument(
        "--measurements",
        nargs="+",
        default=list(utils.REPORT_TEST_MEASUREMENTS),
        help="test measurements to report (default: utils.REPORT_TEST_MEASUREMENTS)",
    )
    parser.add_argument(
        "--recall_at_k",
        nargs="*",
        type=int,
        default=[1, 2, 4, 8],
        help="test Recall@K ladder (default: 1 2 4 8)",
    )
    parser.add_argument("--device", default="cuda", help="retrieval device (default: cuda)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help=(
            "largest accepted |stored - recomputed| for P@1 and MAP@R. P@1 reproduces "
            "exactly; MAP@R moves by ~1e-6 when FAISS orders float32 near-ties differently"
        ),
    )
    parser.add_argument(
        "--allow_mismatch",
        action="store_true",
        help="rewrite a run even when its stored metrics do not reproduce",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="recompute and compare only; write no backup and change no file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("rescore_cv_test_metrics_report.csv"),
        help="CSV of every stored-vs-recomputed comparison",
    )
    return parser.parse_args(argv)


def find_cv_dirs(roots):
    cv_dirs = set()
    for root in roots:
        if (root / SUMMARY_NAME).is_file() or (root / SUMMARY_BACKUP_NAME).is_file():
            cv_dirs.add(root)
            continue
        cv_dirs.update(path.parent for path in root.rglob(SUMMARY_NAME))
    return sorted(cv_dirs)


def to_result(values):
    return TrainingResult(**{key: value for key, value in values.items() if key in RESULT_FIELDS})


def local_embeddings_path(cv_dir, recorded_path):
    """Map the path a run recorded (possibly on another machine) into ``cv_dir``."""

    if recorded_path is None:
        return None
    parts = Path(recorded_path).parts
    if cv_dir.name not in parts:
        return None
    return cv_dir.joinpath(*parts[parts.index(cv_dir.name) + 1 :])


def evaluate_fold(path, name, device, recall_at_k, measurements):
    with np.load(path) as arrays:
        embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
        labels = np.asarray(arrays["labels"]).reshape(-1)
        dataset = ConcatenatedEvaluationDataset(
            arrays["query_indices"] if "query_indices" in arrays else None,
            arrays["gallery_indices"] if "gallery_indices" in arrays else None,
        )
    return utils.unpack_evaluation_result(
        utils.evaluate_embeddings(
            embeddings,
            labels,
            name=name,
            dataset=dataset,
            device=device,
            allow_cpu_fallback=False,
            recall_at_k=recall_at_k,
            measurements=measurements,
            return_measurements=True,
        ),
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
    )


def compare(rows, cv_dir, scope, stored, recomputed):
    """Record stored vs recomputed P@1/MAP@R; return the largest difference."""

    largest = 0.0
    for metric in CHECKED_METRICS:
        old = stored.get(f"{scope[1]}{metric}")
        new = recomputed.get(f"{scope[1]}{metric}")
        if old is None:
            continue
        diff = abs(float(old) - float(new))
        largest = max(largest, diff)
        rows.append(
            {
                "cv_dir": str(cv_dir),
                "scope": scope[0],
                "metric": metric,
                "stored": old,
                "recomputed": new,
                "abs_diff": diff,
            }
        )
    return largest


def rescore_cv_dir(cv_dir, options, report_rows):
    """Rescore one run; return a short status string."""

    summary_backup = cv_dir / SUMMARY_BACKUP_NAME
    # The backup is the original run's summary, so a rerun compares against it
    # rather than against a summary this script already rewrote.
    summary = read_json(summary_backup if summary_backup.is_file() else cv_dir / SUMMARY_NAME)
    if summary.get("aggregate") is None:
        return "skipped: incomplete run (no aggregate)"

    measurements = utils.normalize_measurements(options.measurements)
    recall_at_k = utils.normalize_recall_at_k(options.recall_at_k)
    stored_folds = [to_result(values) for values in summary["folds"]]
    paths = [local_embeddings_path(cv_dir, fold.test_embeddings_path) for fold in stored_folds]
    missing = [fold.cv_fold for fold, path in zip(stored_folds, paths) if path is None or not path.is_file()]
    if missing:
        return f"skipped: fold(s) {missing} have no test_embeddings.npz"

    largest_diff = 0.0
    folds = []
    for fold, path in zip(stored_folds, paths):
        outcome = evaluate_fold(
            path, f"{cv_dir.name} fold {fold.cv_fold} test", options.device, recall_at_k, measurements
        )
        rescored = replace(
            fold,
            test_precision_at_1=float(outcome.precision_at_1),
            test_mean_average_precision_at_r=float(outcome.mean_average_precision_at_r),
            test_recall_at_k=dict(outcome.recall_at_k) or None,
            test_measurements=dict(outcome.measurements) or None,
        )
        largest_diff = max(
            largest_diff,
            compare(
                report_rows,
                cv_dir,
                (f"fold_{fold.cv_fold:02d}", "test_"),
                result_to_dict(fold),
                result_to_dict(rescored),
            ),
        )
        folds.append(rescored)

    args = Namespace(**summary["args"])
    args.report_test_metrics = True
    args.test_measurements = list(measurements)
    args.test_recall_at_k = list(recall_at_k)

    # Evaluate the concatenation from the local files, but keep the recorded
    # paths in the written summary.
    concatenated = evaluate_concatenated_fold_test_embeddings(
        [replace(fold, test_embeddings_path=path) for fold, path in zip(folds, paths)],
        device=options.device,
        recall_at_k=recall_at_k,
        measurements=measurements,
    )
    fresh = aggregate_cross_validation_result(cv_dir, args, folds)
    stored_aggregate = to_result(summary["aggregate"])
    aggregate = replace(
        stored_aggregate,
        test_precision_at_1=fresh.test_precision_at_1,
        test_mean_average_precision_at_r=fresh.test_mean_average_precision_at_r,
        test_precision_at_1_std=fresh.test_precision_at_1_std,
        test_mean_average_precision_at_r_std=fresh.test_mean_average_precision_at_r_std,
        test_recall_at_k=fresh.test_recall_at_k,
        test_measurements=fresh.test_measurements,
        test_measurements_std=fresh.test_measurements_std,
        fold_results=[result_to_dict(fold) for fold in folds],
        **concatenated,
    )
    stored_values = result_to_dict(stored_aggregate)
    new_values = result_to_dict(aggregate)
    largest_diff = max(
        largest_diff,
        compare(report_rows, cv_dir, ("mean", "test_"), stored_values, new_values),
        compare(report_rows, cv_dir, ("concatenated", "concatenated_test_"), stored_values, new_values),
    )

    if largest_diff > options.tolerance and not options.allow_mismatch:
        return f"MISMATCH: max |stored - recomputed| = {largest_diff:.3g}; not written"
    if options.dry_run:
        return f"dry run: max |stored - recomputed| = {largest_diff:.3g}"

    for name, backup_name in ((SUMMARY_NAME, SUMMARY_BACKUP_NAME), (RESULTS_NAME, RESULTS_BACKUP_NAME)):
        source, backup = cv_dir / name, cv_dir / backup_name
        if source.is_file() and not backup.is_file():
            shutil.copy2(source, backup)
    write_cross_validation_summary(cv_dir, args, folds, aggregate)
    return f"rewritten: max |stored - recomputed| = {largest_diff:.3g}"


def main(argv=None):
    options = parse_args(argv)
    cv_dirs = find_cv_dirs(options.roots)
    logger.info(f"Found {len(cv_dirs)} cross-validation run(s)")
    report_rows = []
    statuses = {}
    for index, cv_dir in enumerate(cv_dirs, start=1):
        try:
            status = rescore_cv_dir(cv_dir, options, report_rows)
        except Exception as exc:  # one broken run must not stop the sweep
            logger.exception(f"{cv_dir}: failed")
            status = f"FAILED: {type(exc).__name__}: {exc}"
        statuses[cv_dir] = status
        logger.info(f"[{index}/{len(cv_dirs)}] {cv_dir}: {status}")

    with options.report.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    counts = {}
    for status in statuses.values():
        key = status.split(":", 1)[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"\nComparison report: {options.report}")
    print("Status counts:", counts)
    for cv_dir, status in statuses.items():
        if not status.startswith(("rewritten", "dry run")):
            print(f"  {cv_dir}: {status}")
    return 1 if any(status.startswith(("MISMATCH", "FAILED")) for status in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
