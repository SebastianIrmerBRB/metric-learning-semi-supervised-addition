#!/usr/bin/env python3
"""Remove exported trial CSVs and Optuna studies that hold too few runs.

The exporter produces two mirrored views of the same HPO runs:

  trials-files/<top>/<flattened path>.csv   one CSV per study, one row per trial
  optuna-studies/<preserved path>/optuna_study.db   plus derived artifacts

A study that never accumulated a meaningful number of trials is noise in the
visualizer, so this script counts the runs behind every CSV and every database
and deletes the ones below a threshold. Nothing is removed unless --apply is
passed; the default is a dry run.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIN_RUNS = 30
DEFAULT_STATES = ("COMPLETE",)
TRIALS_DIRECTORY = "trials-files"
STUDIES_DIRECTORY = "optuna-studies"
STUDY_DATABASE_NAME = "optuna_study.db"

# Aggregates that describe the whole export rather than a single study.
PROTECTED_NAMES = frozenset(
    {
        "test-results.csv",
        "parameter_importance_summary.csv",
        "parameter_importance_summary.json",
    }
)


@dataclass
class Candidate:
    """One trial CSV or one Optuna study directory."""

    kind: str
    path: Path
    counted_runs: int
    total_runs: int
    size_bytes: int
    note: str = ""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Exporter directory holding trials-files/ and optuna-studies/.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=DEFAULT_MIN_RUNS,
        help=f"Keep only studies with at least this many runs (default: {DEFAULT_MIN_RUNS}).",
    )
    parser.add_argument(
        "--states",
        default=",".join(DEFAULT_STATES),
        help=(
            "Comma-separated Optuna trial states that count as a run, "
            "or ALL to count every recorded trial (default: COMPLETE)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Move candidates into this directory instead of deleting them.",
    )
    parser.add_argument(
        "--skip-trials",
        action="store_true",
        help=f"Leave {TRIALS_DIRECTORY}/ untouched.",
    )
    parser.add_argument(
        "--skip-studies",
        action="store_true",
        help=f"Leave {STUDIES_DIRECTORY}/ untouched.",
    )
    parser.add_argument(
        "--keep-empty-directories",
        action="store_true",
        help="Do not clean up directories left empty by the removals.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also list the studies that are kept.",
    )
    return parser.parse_args(argv)


def resolve_states(raw: str) -> frozenset[str] | None:
    """Return the accepted state names, or None when every state counts."""
    if raw.strip().upper() == "ALL":
        return None

    states = frozenset(part.strip().upper() for part in raw.split(",") if part.strip())
    if not states:
        raise SystemExit("Error: --states must name at least one state, or be ALL.")
    return states


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    total = 0
    for directory, _, file_names in os.walk(path):
        for name in file_names:
            file_path = Path(directory) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def count_csv_runs(path: Path, states: frozenset[str] | None) -> tuple[int, int, str]:
    """Count data rows in a trials CSV, honouring quoted fields."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return 0, 0, "empty file"

            try:
                state_index = header.index("state")
            except ValueError:
                state_index = -1

            total = 0
            counted = 0
            for row in reader:
                if not any(field.strip() for field in row):
                    continue
                total += 1
                if states is None or state_index < 0:
                    counted += 1
                elif state_index < len(row):
                    if row[state_index].strip().strip('"').upper() in states:
                        counted += 1

            note = "" if state_index >= 0 else "no state column; counted every row"
            return counted, total, note
    except OSError as error:
        return -1, -1, f"unreadable: {error}"


def count_database_runs(
    path: Path, states: frozenset[str] | None
) -> tuple[int, int, str]:
    """Count trials in an optuna_study.db across all studies it contains."""
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return -1, -1, f"unreadable: {error}"

    try:
        rows = connection.execute(
            "SELECT state, COUNT(*) FROM trials GROUP BY state"
        ).fetchall()
    except sqlite3.Error as error:
        return -1, -1, f"unreadable: {error}"
    finally:
        connection.close()

    total = sum(count for _, count in rows)
    if states is None:
        return total, total, ""

    counted = sum(count for state, count in rows if str(state).upper() in states)
    return counted, total, ""


def collect_trial_candidates(
    trials_root: Path, states: frozenset[str] | None, min_runs: int
) -> tuple[list[Candidate], list[Candidate]]:
    remove: list[Candidate] = []
    keep: list[Candidate] = []

    if not trials_root.is_dir():
        return remove, keep

    for path in sorted(trials_root.rglob("*.csv")):
        if path.name in PROTECTED_NAMES:
            continue
        # Top-level files are export-wide aggregates, not per-study exports.
        if path.parent == trials_root:
            continue

        counted, total, note = count_csv_runs(path, states)
        candidate = Candidate(
            kind="trials-csv",
            path=path,
            counted_runs=counted,
            total_runs=total,
            size_bytes=path.stat().st_size,
            note=note,
        )

        if counted < 0:
            candidate.note = f"SKIPPED ({candidate.note})"
            keep.append(candidate)
        elif counted < min_runs:
            remove.append(candidate)
        else:
            keep.append(candidate)

    return remove, keep


def collect_study_candidates(
    studies_root: Path, states: frozenset[str] | None, min_runs: int
) -> tuple[list[Candidate], list[Candidate]]:
    remove: list[Candidate] = []
    keep: list[Candidate] = []

    if not studies_root.is_dir():
        return remove, keep

    for database in sorted(studies_root.rglob(STUDY_DATABASE_NAME)):
        study_directory = database.parent
        counted, total, note = count_database_runs(database, states)
        candidate = Candidate(
            kind="optuna-study",
            path=study_directory,
            counted_runs=counted,
            total_runs=total,
            size_bytes=directory_size(study_directory),
            note=note,
        )

        if counted < 0:
            candidate.note = f"SKIPPED ({candidate.note})"
            keep.append(candidate)
        elif counted < min_runs:
            remove.append(candidate)
        else:
            keep.append(candidate)

    return remove, keep


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def remove_candidate(candidate: Candidate, root: Path, archive: Path | None) -> None:
    if archive is not None:
        destination = archive / candidate.path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.move(str(candidate.path), str(destination))
        return

    if candidate.path.is_dir():
        shutil.rmtree(candidate.path)
    else:
        candidate.path.unlink()


def prune_empty_directories(root: Path, boundary: Path) -> list[Path]:
    """Delete directories under boundary that the removals left empty."""
    removed: list[Path] = []
    if not boundary.is_dir():
        return removed

    # os.walk with topdown=False yields children before parents, so a directory
    # that only becomes empty once its children are gone is still caught. Do not
    # sort this: sorting would reorder the walk and break that guarantee.
    for directory, _, _ in os.walk(boundary, topdown=False):
        path = Path(directory)
        if path == boundary or path == root:
            continue
        try:
            if not any(path.iterdir()):
                path.rmdir()
                removed.append(path)
        except OSError:
            pass
    return sorted(removed)


def report(
    title: str, candidates: list[Candidate], root: Path, states: frozenset[str] | None
) -> None:
    print(f"\n{title} ({len(candidates)})")
    print("-" * len(f"{title} ({len(candidates)})"))
    if not candidates:
        print("  none")
        return

    for candidate in sorted(candidates, key=lambda item: item.counted_runs):
        relative = candidate.path.relative_to(root)
        counts = f"{candidate.counted_runs:>4} run(s)"
        if states is not None and candidate.total_runs != candidate.counted_runs:
            counts += f" of {candidate.total_runs} recorded"
        extra = f"  [{candidate.note}]" if candidate.note else ""
        print(f"  {counts}  {format_size(candidate.size_bytes):>9}  {relative}{extra}")


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    root = arguments.root.resolve()

    if not root.is_dir():
        print(f"Error: root directory does not exist: {root}", file=sys.stderr)
        return 1

    states = resolve_states(arguments.states)
    archive = arguments.archive.resolve() if arguments.archive else None
    if archive is not None and (archive == root or root in archive.parents):
        print(
            "Error: --archive must point outside the exporter root.",
            file=sys.stderr,
        )
        return 1

    trials_root = root / TRIALS_DIRECTORY
    studies_root = root / STUDIES_DIRECTORY

    trial_remove: list[Candidate] = []
    trial_keep: list[Candidate] = []
    study_remove: list[Candidate] = []
    study_keep: list[Candidate] = []

    if not arguments.skip_trials:
        trial_remove, trial_keep = collect_trial_candidates(
            trials_root, states, arguments.min_runs
        )
    if not arguments.skip_studies:
        study_remove, study_keep = collect_study_candidates(
            studies_root, states, arguments.min_runs
        )

    state_label = "any state" if states is None else ", ".join(sorted(states))
    mode = "APPLY" if arguments.apply else "DRY RUN"
    print(f"Root:      {root}")
    print(f"Threshold: fewer than {arguments.min_runs} run(s) counted as [{state_label}]")
    print(f"Mode:      {mode}" + (f" (archive to {archive})" if archive else ""))

    report(f"{TRIALS_DIRECTORY}: to remove", trial_remove, root, states)
    report(f"{STUDIES_DIRECTORY}: to remove", study_remove, root, states)

    if arguments.verbose:
        report(f"{TRIALS_DIRECTORY}: kept", trial_keep, root, states)
        report(f"{STUDIES_DIRECTORY}: kept", study_keep, root, states)

    candidates = trial_remove + study_remove
    freed = sum(candidate.size_bytes for candidate in candidates)

    print(
        f"\nSummary: {len(trial_remove)}/{len(trial_remove) + len(trial_keep)} trial CSV(s) "
        f"and {len(study_remove)}/{len(study_remove) + len(study_keep)} study "
        f"director{'y' if len(study_remove) + len(study_keep) == 1 else 'ies'} below the "
        f"threshold, {format_size(freed)} total."
    )

    if not arguments.apply:
        print("Dry run: nothing was removed. Re-run with --apply to remove.")
        return 0

    verb = "Moved" if archive else "Deleted"
    failures = 0
    for candidate in candidates:
        try:
            remove_candidate(candidate, root, archive)
        except OSError as error:
            failures += 1
            print(f"Error: could not remove {candidate.path}: {error}", file=sys.stderr)

    print(f"{verb} {len(candidates) - failures} of {len(candidates)} candidate(s).")

    if not arguments.keep_empty_directories:
        emptied: list[Path] = []
        for boundary in (trials_root, studies_root):
            emptied.extend(prune_empty_directories(root, boundary))
        if emptied:
            print(f"Removed {len(emptied)} empty director{'y' if len(emptied) == 1 else 'ies'}:")
            for path in emptied:
                print(f"  {path.relative_to(root)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
