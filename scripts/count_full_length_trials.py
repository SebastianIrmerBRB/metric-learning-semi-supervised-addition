"""Count HPO trials that trained for the full epoch budget.

Searches the given directories recursively for Optuna study summaries
(``trials.jsonl``, or ``trials.csv`` where a study has no JSONL) and reports,
per study and in total, how many trials reached ``--epochs`` epochs.

``last_epoch`` is 0-based, so a trial that ran all 600 epochs records 599. For
a cross-validation trial the recorded ``last_epoch`` is the maximum over its
folds; the JSONL also keeps every fold, so "all folds" counts trials where no
fold stopped early. The CSV has no per-fold values, so that column is blank for
CSV-only studies.

Examples::

    python scripts/count_full_length_trials.py logs/ssl/in_shop/slade
    python scripts/count_full_length_trials.py logs/ssl --epochs 600 --list
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="Directories to search, e.g. logs/ssl/in_shop/slade")
    parser.add_argument("--epochs", type=int, default=600, help="Epoch budget a trial must reach (default 600)")
    parser.add_argument("--list", action="store_true", help="Also print the numbers of the trials that reached it")
    return parser.parse_args()


def json_value(text):
    if text in (None, ""):
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def read_jsonl(path):
    trials = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        attrs = record.get("user_attrs") or {}
        folds = attrs.get("fold_results") or []
        trials.append(
            {
                "number": record.get("number"),
                "state": record.get("state"),
                "last_epoch": attrs.get("last_epoch"),
                "fold_last_epochs": [fold.get("last_epoch") for fold in folds if isinstance(fold, dict)],
                "epochs": (attrs.get("resolved_args") or {}).get("epochs"),
            }
        )
    return trials


def read_csv(path):
    with path.open(newline="") as handle:
        return [
            {
                "number": json_value(row.get("number")),
                "state": row.get("state"),
                "last_epoch": json_value(row.get("attr:last_epoch")),
                "fold_last_epochs": None,
                "epochs": None,
            }
            for row in csv.DictReader(handle)
        ]


def study_files(roots):
    """One summary per study directory, preferring the JSONL."""

    directories = {}
    for root in roots:
        for path in sorted(Path(root).rglob("trials.jsonl")):
            directories[path.parent] = path
        for path in sorted(Path(root).rglob("trials.csv")):
            directories.setdefault(path.parent, path)
    return [directories[directory] for directory in sorted(directories)]


def main():
    options = parse_args()
    last_index = options.epochs - 1
    files = study_files(options.roots)
    if not files:
        raise SystemExit(f"No trials.jsonl or trials.csv below {options.roots}")

    totals = Counter()
    rows = []
    for path in files:
        trials = read_jsonl(path) if path.suffix == ".jsonl" else read_csv(path)
        counts = Counter(str(trial["state"]) for trial in trials)
        measured = [trial for trial in trials if isinstance(trial["last_epoch"], int)]
        reached = [trial for trial in measured if trial["last_epoch"] >= last_index]
        with_folds = [trial for trial in measured if trial["fold_last_epochs"]]
        all_folds = [
            trial
            for trial in with_folds
            if all(isinstance(epoch, int) and epoch >= last_index for epoch in trial["fold_last_epochs"])
        ]
        budgets = sorted({trial["epochs"] for trial in trials if trial["epochs"] is not None})
        rows.append((path, trials, counts, measured, reached, with_folds, all_folds, budgets))
        totals["trials"] += len(trials)
        totals["complete"] += counts.get("COMPLETE", 0)
        totals["measured"] += len(measured)
        totals["reached"] += len(reached)
        totals["all_folds"] += len(all_folds)
        totals["with_folds"] += len(with_folds)

    for path, trials, counts, measured, reached, with_folds, all_folds, budgets in rows:
        states = ", ".join(f"{state.lower()} {count}" for state, count in sorted(counts.items()))
        folds = f", all folds {len(all_folds)}/{len(with_folds)}" if with_folds else ""
        budget = f"  (configured epochs: {', '.join(map(str, budgets))})" if budgets else ""
        print(f"{path.parent}")
        print(
            f"  {len(trials)} trials ({states}); reached {options.epochs} epochs: "
            f"{len(reached)}/{len(measured)} with a recorded last epoch{folds}{budget}"
        )
        if options.list and reached:
            print(f"  trials: {', '.join(str(trial['number']) for trial in reached)}")

    print()
    print(
        f"{len(files)} studies, {totals['trials']} trials ({totals['complete']} complete): "
        f"{totals['reached']}/{totals['measured']} reached {options.epochs} epochs"
        + (
            f"; in {totals['all_folds']}/{totals['with_folds']} CV trials every fold did"
            if totals["with_folds"]
            else ""
        )
    )


if __name__ == "__main__":
    main()
