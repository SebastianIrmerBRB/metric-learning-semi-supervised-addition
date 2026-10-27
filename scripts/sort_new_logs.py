"""Moves phase 0 (random search) studies from logs/ and server_logs/ into sorted_logs/.

Follows the convention SORTING.md documents:

    sorted_logs/<bucket>/<mode>/<dataset>/[<ssl_method>[__<campaign>]/]<dml_loss>

Only `hparam_config.sampler == "random"` studies are considered; TPE studies are
left where they are. A study already present in sorted_logs is skipped rather
than duplicated, matched on (save_dir, study_name, first trial start), which
together name the campaign, the run, the Optuna study and the wall-clock start —
enough to tell two sweeps apart even where an old save_dir was reused.

Directories are moved with os.rename on the same filesystem, so nothing is
copied and nothing is duplicated, exactly as the first two imports did.

    python scripts/sort_new_logs.py             # plan only
    python scripts/sort_new_logs.py --apply     # move, and extend sorting_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from survey_unsorted_logs import (
    DB_NAME,
    REPO,
    Study,
    collect,
    count_trials,
    ssl_method,
)

UNFINISHED_TRIALS = 30
SORTED = REPO / "sorted_logs"
MAP_PATH = SORTED / "sorting_map.csv"

# Campaign folder names whose on-disk spelling differs from the name the archive
# already uses for them. sorting_map.csv and SORTING.md both refer to the
# contrastive rerun with the corrected spelling.
CAMPAIGN_RENAMES = {"intial_tr_contrastive": "initial_tr_contrastive"}


def first_trial_start(directory: Path) -> str:
    try:
        with sqlite3.connect(f"file:{directory / DB_NAME}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT min(datetime_start) FROM trials WHERE datetime_start IS NOT NULL"
            ).fetchone()
            return str(row[0]) if row and row[0] else ""
    except Exception:
        return ""


def complete_trials(directory: Path) -> int:
    try:
        with sqlite3.connect(f"file:{directory / DB_NAME}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT count(*) FROM trials WHERE state = 'COMPLETE'"
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def identity(study: Study) -> tuple[str, str, str]:
    return (study.save_dir, study.study_name, first_trial_start(study.directory))


def campaign_of(study: Study, root: Path, method: str | None) -> str:
    """The campaign folder a study was launched from, as a bare tag."""
    relative = study.directory.relative_to(root).parts
    name = None
    if method:
        for part in relative:
            if part == method or part.startswith(f"{method}_"):
                name = part
                break
    if name is None:
        name = relative[0] if relative else ""

    tag = name
    if method and tag.startswith(f"{method}_"):
        tag = tag[len(method) + 1 :]
    elif method and tag == method:
        tag = ""
    return CAMPAIGN_RENAMES.get(tag, tag)


@dataclass
class Move:
    study: Study
    root_label: str
    source: Path
    target: Path
    method: str | None
    campaign: str
    trials: int
    complete: int

    @property
    def bucket(self) -> str:
        return "unfinished" if self.trials < UNFINISHED_TRIALS else "phase_0"


def plan(sorted_index: dict[tuple[str, str, str], Path]) -> tuple[list[Move], list[str]]:
    moves: list[Move] = []
    notes: list[str] = []

    for label in ("logs", "server_logs"):
        root = REPO / label
        if not root.exists():
            continue
        for study in collect(root, label):
            if study.sampler != "random":
                continue
            if identity(study) in sorted_index:
                continue

            trials = count_trials(study.directory)
            method = ssl_method(study.directory, study.base_args) if study.mode == "ssl" else None
            if study.mode == "ssl" and not method:
                notes.append(f"no SSL method resolved, skipped: {study.directory.relative_to(REPO)}")
                continue

            campaign = campaign_of(study, root, method)
            bucket = "unfinished" if trials < UNFINISHED_TRIALS else "phase_0"
            parts = [bucket]
            if bucket == "unfinished":
                parts.append("phase_0")
            parts += [study.mode, study.dataset_dir]
            # Every import is tagged with its campaign, so a method that ran
            # more than once stays greppable and no bare directory silently
            # becomes ambiguous later.
            if study.mode == "ssl":
                parts.append(f"{method}__{campaign}" if campaign else str(method))
                parts.append(study.loss_dir)
            else:
                parts.append(f"{study.loss_dir}__{campaign}" if campaign else study.loss_dir)

            moves.append(
                Move(
                    study=study,
                    root_label=label,
                    source=study.directory,
                    target=SORTED.joinpath(*parts),
                    method=method,
                    campaign=campaign,
                    trials=trials,
                    complete=complete_trials(study.directory),
                )
            )

    seen: dict[Path, Move] = {}
    conflicts: list[Move] = []
    for move in moves:
        if move.target.exists():
            conflicts.append(move)
            notes.append(f"target already on disk, skipped: {move.target.relative_to(REPO)}")
            continue
        if move.target in seen:
            conflicts.append(move)
            notes.append(
                f"two studies claim {move.target.relative_to(REPO)}:\n"
                f"      {seen[move.target].source.relative_to(REPO)}\n"
                f"      {move.source.relative_to(REPO)}"
            )
            continue
        seen[move.target] = move

    return [move for move in moves if move not in conflicts], notes


def append_map_rows(moves: list[Move], stamp: str) -> int:
    """Adds a row per move, replacing any stale row that claimed the same path.

    Returns how many existing rows were replaced. The first sort recorded 15
    `lrml__initial_tr_contrastive` rows for directories that never landed on
    disk; the ones this import actually creates take those rows over instead of
    sitting beside them.
    """
    with MAP_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        existing = list(reader)

    rows = []
    for move in moves:
        base = move.study.base_args
        # <campaign>/<loss dir>/<run name>/hpo/<study dir>
        run_name = move.source.parent.parent.name
        rows.append(
            {
                "new_path": str(move.target.relative_to(REPO)),
                "old_path": str(move.source.relative_to(REPO)),
                "import": stamp,
                "bucket": move.bucket,
                "phase": "phase_0",
                "sampler": "random",
                "n_trials": move.trials,
                "n_complete": move.complete,
                "n_trials_cfg": move.study.hparam.get("n_trials"),
                "mode": move.study.mode,
                "dataset": base.get("dataset"),
                "ssl_method": move.method or "",
                "loss": base.get("loss"),
                "miner": base.get("miner"),
                "study_name": move.study.study_name,
                "run_name": run_name,
                "old_campaign_dir": str(move.source.relative_to(REPO)).split("/hpo/")[0],
                "experiment_config": base.get("experiment_config"),
                "hparam_config": (base.get("experiment_config_resolved") or {}).get(
                    "hparam_config"
                ),
            }
        )

    claimed = {row["new_path"] for row in rows}
    kept = [row for row in existing if row.get("new_path") not in claimed]
    replaced = len(existing) - len(kept)

    with MAP_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in kept + rows:
            writer.writerow(row)
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually move the directories")
    arguments = parser.parse_args()

    sorted_index = {identity(study): study.directory for study in collect(SORTED, "sorted_logs")}
    print(f"sorted_logs: {len(sorted_index)} studies indexed")

    moves, notes = plan(sorted_index)
    print(f"phase 0 studies to import: {len(moves)}\n")

    by_campaign: dict[str, list[Move]] = {}
    for move in moves:
        key = f"{move.root_label}: {move.method}__{move.campaign}"
        by_campaign.setdefault(key, []).append(move)

    for key, group in sorted(by_campaign.items()):
        print(f"  {key}  ({len(group)} studies)")
        for move in sorted(group, key=lambda item: str(item.target)):
            flag = "  [unfinished]" if move.bucket == "unfinished" else ""
            print(
                f"    {move.trials:>3} trials -> {move.target.relative_to(SORTED)}{flag}"
            )
        print()

    if notes:
        print("notes:")
        for note in notes:
            print(f"  - {note}")
        print()

    if not arguments.apply:
        print("Dry run. Re-run with --apply to move them.")
        return

    stamp = date.today().isoformat()
    for move in moves:
        move.target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(move.source, move.target)
    replaced = append_map_rows(moves, stamp)
    print(
        f"Moved {len(moves)} studies and wrote their rows to {MAP_PATH.relative_to(REPO)}"
        + (f", replacing {replaced} stale row(s)." if replaced else ".")
    )


if __name__ == "__main__":
    main()
