"""Measure the wall-clock cost of the optional logging artifacts.

Each section times the same workload with the artifact enabled and disabled, so
the difference is the cost a run pays purely to produce diagnostics nothing
reads back. Sizes default to a Cars196-shaped SSL fold.

Examples:
    python scripts/bench_logging_overhead.py
    python scripts/bench_logging_overhead.py --classes 196 --epochs 60
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

import utils
from training.ssl.config import PseudoLabelResult
from training.ssl.data import HofferReferenceBatchSampler
from training.ssl.pseudo_labels import PseudoLabelDiagnosticsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classes", type=int, default=98, help="labeled classes (Cars196 train half).")
    parser.add_argument("--unlabeled", type=int, default=4096, help="unlabeled pool size.")
    parser.add_argument("--batch", type=int, default=32, help="unlabeled samples per batch.")
    parser.add_argument("--epochs", type=int, default=30, help="epochs simulated per section.")
    parser.add_argument("--generations", type=int, default=30, help="pseudo-label generations.")
    return parser.parse_args()


def timed(function):
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def measure_debug_log(args, work_dir):
    """Hoffer batch sampling, whose per-batch DEBUG record is the largest one."""

    sampler = HofferReferenceBatchSampler(
        num_unlabeled=args.unlabeled,
        class_candidates=[list(range(args.unlabeled + i * 8, args.unlabeled + i * 8 + 8)) for i in range(args.classes)],
        unlabeled_per_batch=args.batch,
        seed=7,
        reference_sets=1,
        class_labels=list(range(args.classes)),
        unlabeled_positions=list(range(args.unlabeled)),
        labeled_positions=list(range(args.classes * 8)),
    )

    def drain():
        for _ in range(args.epochs):
            for _ in sampler:
                pass

    logger.remove()
    path = work_dir / "debug.log"
    logger.add(path, level="DEBUG")
    on = timed(drain)
    on_bytes = path.stat().st_size
    logger.remove()

    logger.add(work_dir / "info.log", level="INFO")
    off = timed(drain)
    logger.remove()
    return on, off, on_bytes


def measure_metrics_logger(args, work_dir):
    """Epoch/eval diagnostics and their TensorBoard scalars."""

    per_class = {
        str(label): {"precision_at_1": 0.5, "mean_average_precision_at_r": 0.4, "count": 20}
        for label in range(args.classes)
    }
    eval_diagnostics = {f"timing/{name}_seconds": 1.0 for name in ("embedding_extraction", "retrieval_metrics", "total")}
    eval_diagnostics.update({f"embedding_norm/{stat}": 1.0 for stat in ("min", "mean", "std", "max")})

    def run(tensorboard, epoch_diagnostics):
        target = work_dir / f"ml_{tensorboard}_{epoch_diagnostics}"
        target.mkdir(parents=True, exist_ok=True)
        metrics_logger = utils.MetricsLogger(
            target,
            SimpleNamespace(tensorboard=tensorboard, log_epoch_diagnostics=epoch_diagnostics),
        )

        def workload():
            for epoch in range(args.epochs):
                metrics_logger.log_train_epoch(0.5, epoch=epoch, step=epoch * 100, diagnostics={"train/lr": 1e-4})
                metrics_logger.log_eval(
                    "valid", 0.6, 0.5, step=epoch * 100, epoch=epoch,
                    per_class_metrics=per_class, diagnostics=eval_diagnostics,
                )

        elapsed = timed(workload)
        metrics_logger.close()
        written = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        return elapsed, written

    return {
        "full (tb + diagnostics)": run(True, True),
        "no epoch diagnostics": run(True, False),
        "no tensorboard": run(False, True),
        "neither": run(False, False),
    }


def measure_pseudo_label_diagnostics(args, work_dir):
    """The hidden-label audit: per-class tallies plus the confidence-vs-correctness AUC."""

    rng = np.random.default_rng(7)
    count = args.unlabeled
    true_labels = rng.integers(0, args.classes, size=count)
    predictions = PseudoLabelResult(
        positions=np.arange(count),
        mapped_labels=rng.integers(0, args.classes, size=count),
        confidences=rng.random(count).astype(np.float32),
    )
    config = SimpleNamespace(
        method="iscen_label_spreading",
        confidence_threshold=0.5,
        pseudo_label_rescue_confidence_floor=0.0,
        pseudo_label_rescue_top_k=None,
    )
    dataset = SimpleNamespace(labels=true_labels)

    def run(mode):
        if mode == "off":
            return 0.0, 0
        target = work_dir / f"pl_{mode}"
        target.mkdir(parents=True, exist_ok=True)
        tracker = PseudoLabelDiagnosticsTracker(target, mode=mode)

        def workload():
            for generation in range(args.generations):
                tracker.log(predictions, predictions, dataset, config, epoch=generation)

        elapsed = timed(workload)
        path = target / "pseudo_label_diagnostics.jsonl"
        return elapsed, path.stat().st_size if path.exists() else 0

    return {mode: run(mode) for mode in ("save", "log", "off")}


def human_bytes(count):
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main():
    args = parse_args()
    work_dir = Path(tempfile.mkdtemp(prefix="bench_logging_"))
    # Console output would itself dominate the timings being measured.
    logger.remove()
    try:
        print(
            f"config: {args.classes} classes, {args.unlabeled} unlabeled, batch {args.batch}, "
            f"{args.epochs} epochs, {args.generations} pseudo-label generations\n"
        )

        on, off, written = measure_debug_log(args, work_dir)
        print("1. debug.log (Hoffer per-batch DEBUG record)")
        print(f"   sink on  : {on:7.2f} s   wrote {human_bytes(written)}")
        print(f"   sink off : {off:7.2f} s")
        print(f"   -> {on - off:.2f} s saved per {args.epochs} epochs ({100 * (on - off) / on:.1f}% of the logged path)\n")

        print("2. MetricsLogger (diagnostics.csv + TensorBoard scalars)")
        results = measure_metrics_logger(args, work_dir)
        baseline = results["full (tb + diagnostics)"][0]
        for label, (elapsed, written) in results.items():
            saved = "" if label == "full (tb + diagnostics)" else f"  -> {baseline - elapsed:6.2f} s saved"
            print(f"   {label:24s} {elapsed:7.2f} s   {human_bytes(written):>9s}{saved}")
        print()

        print("3. pseudo-label diagnostics (hidden-label audit)")
        pseudo = measure_pseudo_label_diagnostics(args, work_dir)
        save_time = pseudo["save"][0]
        for mode, (elapsed, written) in pseudo.items():
            saved = "" if mode == "save" else f"  -> {save_time - elapsed:6.2f} s saved"
            print(f"   mode={mode:5s} {elapsed:7.2f} s   {human_bytes(written):>9s}{saved}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
