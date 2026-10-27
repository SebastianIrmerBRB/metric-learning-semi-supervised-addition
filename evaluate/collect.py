"""Gather every tracked fold under a root into one tidy data directory.

This is the step the other readers build on, and it is deliberately the only one
that touches the run tree. It writes three files:

``data/epochs.csv``
    One row per run, fold and epoch, with whatever columns that run tracked.
    Long enough to pivot, wide enough to read.
``data/tracked_test.csv``
    The full D_test table from each run's own ``metrics.csv`` -- every
    measurement and the whole Recall@K ladder, which ``epoch_metrics.csv`` does
    not carry.
``data/runs.csv``
    One row per discovered fold: dataset, method, loss, budget, seed, ablation
    variant, and the directory it came from, so any number in a figure can be
    traced back to the run that produced it.

Examples::

    python evaluate/collect.py --root logs/phase_1 --out evaluate_out
    python evaluate/collect.py --root final/phase_1 --out ~/thesis/tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import common


def collect(root: Path, out: Path) -> dict:
    records = common.discover(root)
    if not records:
        raise SystemExit(
            f"No epoch-tracking output under {root}. Runs write it only with "
            "--track_epoch_artifacts and --track_epoch_interval set; see "
            "docs/epoch_tracking.md"
        )
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    epochs = common.load_epochs(records)
    tracked_test = common.load_tracked_test(records)
    index = common.runs_index(records)

    epochs.to_csv(data_dir / "epochs.csv", index=False)
    index.to_csv(data_dir / "runs.csv", index=False)
    if not tracked_test.empty:
        tracked_test.to_csv(data_dir / "tracked_test.csv", index=False)

    metrics = common.available_metrics(epochs)
    summary = common.fold_summary(epochs, metrics)
    if not summary.empty:
        summary.to_csv(data_dir / "summary.csv", index=False)

    return {
        "records": records,
        "epochs": epochs,
        "tracked_test": tracked_test,
        "runs": index,
        "summary": summary,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="directory to search for tracked runs")
    parser.add_argument("--out", type=Path, default=Path("evaluate_out"), help="output directory")
    args = parser.parse_args()

    result = collect(args.root, args.out)
    folds = len(result["records"])
    runs = result["runs"]["run_id"].nunique() if not result["runs"].empty else 0
    print(f"Collected {folds} tracked fold(s) across {runs} run(s) from {args.root}")
    print(f"  metrics tracked: {', '.join(result['metrics']) or 'none'}")
    print(f"  wrote {args.out / 'data'}")


if __name__ == "__main__":
    main()
