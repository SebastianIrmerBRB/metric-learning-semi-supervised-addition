"""Remove selected Optuna trials and their local artifact folders.

Examples:
    python scripts/remove_optuna_trials.py --study-dir old_logs/run/hpo/study --trials 26
    python scripts/remove_optuna_trials.py --study-dir old_logs/run/hpo/study --from-trial 26 --yes
    python scripts/remove_optuna_trials.py --study-dir old_logs/run/hpo/study --keep-up-to 25 --yes
    python scripts/remove_optuna_trials.py --study-dir old_logs/run/hpo/study --rewind-to-tpe-start
    python scripts/remove_optuna_trials.py --study-dir old_logs/run/hpo/study --extend-tpe-startup-to 60 --yes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote


SUMMARY_CSV = "trials.csv"
SUMMARY_JSONL = "trials.jsonl"
OPTUNA_DB = "optuna_study.db"
SAMPLER_STATE = "sampler.pkl"
TPE_TRANSITION_SAMPLER_STATE = "sampler_before_tpe.pkl"
TPE_BUDGET_STATES = {"COMPLETE", "PRUNED"}


@dataclass(frozen=True)
class TpeSamplerMigrationPlan:
    mode: str
    old_startup_trials: int
    target_startup_trials: int
    boundary_trial_number: int
    trial_numbers: list[int]
    transition_path: Path
    rolling_sampler_path: Path
    preserved_transition_path: Path | None
    transition_bytes: bytes
    rolling_sampler_bytes: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete Optuna trials from SQLite storage, remove matching trial_XXXX "
            "artifact folders, and rewrite trials.csv/trials.jsonl."
        )
    )
    parser.add_argument(
        "--study-dir",
        "--study",
        type=Path,
        default=None,
        help="Study artifact directory containing optuna_study.db and optional study_config.json.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Explicit SQLite Optuna storage URL, for example sqlite:///old_logs/run/study/optuna_study.db.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Defaults to study_config.json's resolved_study_name or 'optuna'.",
    )
    parser.add_argument(
        "--list-studies",
        action="store_true",
        help="List studies in the resolved SQLite DB and exit.",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--trials",
        "--trial",
        type=int,
        nargs="+",
        default=None,
        help="Exact trial number(s) to delete.",
    )
    selection.add_argument(
        "--from-trial",
        type=int,
        default=None,
        help="Delete this trial number and every larger trial number.",
    )
    selection.add_argument(
        "--after-trial",
        type=int,
        default=None,
        help="Delete every trial number larger than this value.",
    )
    selection.add_argument(
        "--keep-up-to",
        type=int,
        default=None,
        help="Keep trials 0..N and delete all larger trial numbers.",
    )
    selection.add_argument(
        "--rewind-to-tpe-start",
        action="store_true",
        help=(
            "Keep the original random-startup prefix, delete every later trial, "
            "and restore sampler_before_tpe.pkl as sampler.pkl."
        ),
    )
    selection.add_argument(
        "--extend-tpe-startup-to",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Rewind to sampler_before_tpe.pkl, raise its random-startup threshold "
            "to N, and preserve the old boundary checkpoint under a numbered name."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply changes. Without this flag the script only prints what would be removed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak.TIMESTAMP copies before modifying files.",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip SQLite VACUUM after deleting trials.",
    )
    parser.add_argument(
        "--keep-folders",
        action="store_true",
        help="Delete only Optuna DB rows and summaries; keep trial_XXXX artifact folders.",
    )
    return parser.parse_args()


def load_study_config(study_dir: Path | None) -> dict[str, Any]:
    if study_dir is None:
        return {}
    config_path = study_dir / "study_config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} must contain a JSON object.")
    return config


def sqlite_url_for_path(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def sqlite_path_from_url(storage: str) -> Path:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        raise SystemExit(f"Only sqlite:/// Optuna storage is supported: {storage}")
    raw_path = unquote(storage[len(prefix) :])
    if raw_path == ":memory:":
        raise SystemExit("In-memory SQLite studies cannot be edited by this script.")
    return Path(raw_path)


def resolve_study_location(args: argparse.Namespace) -> tuple[str, str, Path]:
    study_dir = None if args.study_dir is None else args.study_dir.resolve()
    config = load_study_config(study_dir)
    study_name = (
        args.study_name
        or config.get("resolved_study_name")
        or config.get("hparam_config", {}).get("study_name")
        or "optuna"
    )

    if args.storage is not None:
        storage = args.storage
    elif study_dir is not None and (study_dir / OPTUNA_DB).exists():
        # Prefer the DB next to the artifacts. Copied cluster old_logs can contain
        # absolute paths in study_config.json that are invalid on this machine.
        storage = sqlite_url_for_path(study_dir / OPTUNA_DB)
    elif config.get("resolved_storage") is not None:
        storage = str(config["resolved_storage"])
    elif study_dir is not None:
        storage = sqlite_url_for_path(study_dir / OPTUNA_DB)
    else:
        raise SystemExit("Provide --study-dir or --storage.")

    return str(study_name), storage, sqlite_path_from_url(storage)


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"Optuna SQLite DB does not exist: {db_path}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def list_studies(db_path: Path) -> None:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT studies.study_name, COUNT(trials.trial_id) AS trial_count
            FROM studies
            LEFT JOIN trials ON trials.study_id = studies.study_id
            GROUP BY studies.study_id, studies.study_name
            ORDER BY studies.study_name
            """
        ).fetchall()

    if not rows:
        print("No studies found.")
        return
    width = max(len("study_name"), *(len(str(row["study_name"])) for row in rows))
    print(f"{'study_name':<{width}}  trials")
    print(f"{'-' * width}  {'-' * 6}")
    for row in rows:
        print(f"{row['study_name']:<{width}}  {row['trial_count']}")


def get_study_id(connection: sqlite3.Connection, study_name: str) -> int:
    row = connection.execute(
        "SELECT study_id FROM studies WHERE study_name = ?",
        (study_name,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Study {study_name!r} was not found in the Optuna DB.")
    return int(row["study_id"])


def all_trial_numbers(connection: sqlite3.Connection, study_id: int) -> list[int]:
    rows = connection.execute(
        "SELECT number FROM trials WHERE study_id = ? ORDER BY number",
        (study_id,),
    ).fetchall()
    return [int(row["number"]) for row in rows]


def all_trial_state_rows(
    connection: sqlite3.Connection,
    study_id: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT trial_id, number, state
        FROM trials
        WHERE study_id = ?
        ORDER BY number
        """,
        (study_id,),
    ).fetchall()
    return [
        {
            "trial_id": int(row["trial_id"]),
            "number": int(row["number"]),
            "state": str(row["state"]),
        }
        for row in rows
    ]


def selected_trial_numbers(args: argparse.Namespace, existing_numbers: list[int]) -> list[int]:
    existing = set(existing_numbers)
    if args.trials is not None:
        selected = sorted(set(int(number) for number in args.trials))
    elif args.from_trial is not None:
        selected = [number for number in existing_numbers if number >= int(args.from_trial)]
    elif args.after_trial is not None:
        selected = [number for number in existing_numbers if number > int(args.after_trial)]
    elif args.keep_up_to is not None:
        selected = [number for number in existing_numbers if number > int(args.keep_up_to)]
    else:
        raise SystemExit("Select trials with --trials, --from-trial, --after-trial, or --keep-up-to.")

    missing = [number for number in selected if number not in existing]
    if missing:
        raise SystemExit(f"Selected trial number(s) not found in study: {missing}")
    if not selected:
        raise SystemExit("The selection matched no trials.")
    return selected


def find_tpe_startup_boundary_trial(
    trial_rows: list[dict[str, Any]],
    startup_trials: int,
) -> int:
    if startup_trials == 0:
        return -1
    budget_rows = [
        row
        for row in trial_rows
        if row["state"] in TPE_BUDGET_STATES
    ]
    if len(budget_rows) < startup_trials:
        raise SystemExit(
            f"The study has only {len(budget_rows)} completed/pruned trial(s), "
            f"but the transition sampler requires {startup_trials}."
        )
    return int(budget_rows[startup_trials - 1]["number"])


def validate_tpe_study_config(
    study_config: dict[str, Any],
    sampler_startup_trials: int,
) -> None:
    hparam_config = study_config.get("hparam_config", {})
    configured_sampler = hparam_config.get("sampler")
    if configured_sampler not in {None, "tpe"}:
        raise SystemExit(
            f"study_config.json identifies sampler={configured_sampler!r}, not 'tpe'."
        )
    configured_n_jobs = hparam_config.get("n_jobs", 1)
    if configured_n_jobs != 1:
        raise SystemExit(
            "Exact TPE sampler rewind/extension requires a checkpoint created "
            f"with n_jobs=1, but study_config.json records n_jobs={configured_n_jobs!r}. "
            "Parallel startup can include in-flight RNG advances beyond the visible boundary."
        )
    recorded_startup_trials = study_config.get("effective_tpe_startup_trials")
    if (
        isinstance(recorded_startup_trials, int)
        and not isinstance(recorded_startup_trials, bool)
        and recorded_startup_trials != sampler_startup_trials
    ):
        raise SystemExit(
            "sampler_before_tpe.pkl disagrees with study_config.json: "
            f"sampler threshold={sampler_startup_trials}, "
            f"recorded effective threshold={recorded_startup_trials}."
        )


def make_tpe_sampler_migration_plan(
    args: argparse.Namespace,
    study_dir: Path | None,
    study_config: dict[str, Any],
    trial_rows: list[dict[str, Any]],
) -> TpeSamplerMigrationPlan | None:
    if not args.rewind_to_tpe_start and args.extend_tpe_startup_to is None:
        return None
    if study_dir is None:
        raise SystemExit(
            "--rewind-to-tpe-start and --extend-tpe-startup-to require --study-dir."
        )

    active_trials = [
        int(row["number"])
        for row in trial_rows
        if row["state"] in {"RUNNING", "WAITING"}
    ]
    if active_trials:
        raise SystemExit(
            "Refusing to migrate a study with RUNNING/WAITING trials. "
            f"Stop all HPO workers and resolve these trials first: {active_trials}"
        )

    transition_path = study_dir / TPE_TRANSITION_SAMPLER_STATE
    if not transition_path.is_file():
        raise SystemExit(
            f"TPE transition sampler does not exist: {transition_path}"
        )
    try:
        transition_bytes = transition_path.read_bytes()
        transition_sampler = pickle.loads(transition_bytes)
    except Exception as exc:
        raise SystemExit(
            f"Could not load TPE transition sampler from {transition_path}: {exc}"
        ) from exc
    if transition_sampler.__class__.__name__ != "TPESampler":
        raise SystemExit(
            f"{transition_path} contains {transition_sampler.__class__.__name__}, "
            "not TPESampler."
        )

    old_startup_trials = getattr(transition_sampler, "_n_startup_trials", None)
    if (
        isinstance(old_startup_trials, bool)
        or not isinstance(old_startup_trials, int)
        or old_startup_trials < 0
    ):
        raise SystemExit(
            f"Could not read a valid TPE startup threshold from {transition_path}."
        )
    validate_tpe_study_config(study_config, old_startup_trials)

    boundary_trial_number = find_tpe_startup_boundary_trial(
        trial_rows,
        old_startup_trials,
    )
    trial_numbers = [
        int(row["number"])
        for row in trial_rows
        if int(row["number"]) > boundary_trial_number
    ]
    rolling_sampler_path = study_dir / SAMPLER_STATE

    if args.extend_tpe_startup_to is None:
        mode = "rewind"
        target_startup_trials = old_startup_trials
        preserved_transition_path = None
        rolling_sampler_bytes = transition_bytes
    else:
        mode = "extend"
        target_startup_trials = int(args.extend_tpe_startup_to)
        if target_startup_trials <= old_startup_trials:
            raise SystemExit(
                "--extend-tpe-startup-to must be larger than the checkpoint's "
                f"current threshold ({old_startup_trials}), got {target_startup_trials}."
            )
        preserved_transition_path = transition_path.with_name(
            f"{transition_path.stem}_{old_startup_trials}{transition_path.suffix}"
        )
        if preserved_transition_path.exists():
            if (
                not preserved_transition_path.is_file()
                or preserved_transition_path.read_bytes() != transition_bytes
            ):
                raise SystemExit(
                    f"Refusing to overwrite a different preserved checkpoint: "
                    f"{preserved_transition_path}"
                )
        transition_sampler._n_startup_trials = target_startup_trials
        try:
            rolling_sampler_bytes = pickle.dumps(transition_sampler)
        except Exception as exc:
            raise SystemExit(
                f"Could not serialize the updated TPE sampler: {exc}"
            ) from exc

    return TpeSamplerMigrationPlan(
        mode=mode,
        old_startup_trials=old_startup_trials,
        target_startup_trials=target_startup_trials,
        boundary_trial_number=boundary_trial_number,
        trial_numbers=trial_numbers,
        transition_path=transition_path,
        rolling_sampler_path=rolling_sampler_path,
        preserved_transition_path=preserved_transition_path,
        transition_bytes=transition_bytes,
        rolling_sampler_bytes=rolling_sampler_bytes,
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def tables_with_trial_id(connection: sqlite3.Connection) -> list[str]:
    tables = [
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    ]
    result = []
    for table in tables:
        columns = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
        if any(str(column["name"]) == "trial_id" for column in columns):
            result.append(table)
    return result


def trial_rows(
    connection: sqlite3.Connection,
    study_id: int,
    trial_numbers: list[int],
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in trial_numbers)
    return connection.execute(
        f"""
        SELECT trial_id, number, state
        FROM trials
        WHERE study_id = ? AND number IN ({placeholders})
        ORDER BY number
        """,
        (study_id, *trial_numbers),
    ).fetchall()


def backup_file(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def backup_files(paths: list[Path], enabled: bool) -> list[Path]:
    if not enabled:
        return []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    for path in paths:
        backup = backup_file(path, timestamp)
        if backup is not None:
            backups.append(backup)
    return backups


def write_bytes_atomically(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        temporary_path.write_bytes(contents)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def preserve_tpe_transition_checkpoint(
    plan: TpeSamplerMigrationPlan | None,
    dry_run: bool,
) -> None:
    if (
        plan is None
        or plan.preserved_transition_path is None
        or dry_run
        or plan.preserved_transition_path.exists()
    ):
        return
    if plan.transition_path.read_bytes() != plan.transition_bytes:
        raise SystemExit(
            f"{plan.transition_path} changed while the migration was being prepared. "
            "Stop all HPO workers and retry."
        )
    write_bytes_atomically(
        plan.preserved_transition_path,
        plan.transition_bytes,
    )


def apply_tpe_sampler_migration(
    plan: TpeSamplerMigrationPlan | None,
    dry_run: bool,
) -> None:
    if plan is None or dry_run:
        return
    if plan.transition_path.read_bytes() != plan.transition_bytes:
        raise SystemExit(
            f"{plan.transition_path} changed while trials were being removed. "
            "Restore the database backup before retrying."
        )
    write_bytes_atomically(
        plan.rolling_sampler_path,
        plan.rolling_sampler_bytes,
    )
    if plan.mode == "extend":
        plan.transition_path.unlink()


def delete_trials_from_sqlite(
    db_path: Path,
    study_name: str,
    trial_numbers: list[int],
    dry_run: bool,
    vacuum: bool,
) -> list[dict[str, Any]]:
    with connect(db_path) as connection:
        study_id = get_study_id(connection, study_name)
        rows = trial_rows(connection, study_id, trial_numbers)
        if len(rows) != len(trial_numbers):
            found = {int(row["number"]) for row in rows}
            missing = sorted(set(trial_numbers) - found)
            raise SystemExit(f"Selected trial number(s) not found in study: {missing}")

        selected = [
            {
                "trial_id": int(row["trial_id"]),
                "number": int(row["number"]),
                "state": str(row["state"]),
            }
            for row in rows
        ]
        if dry_run:
            return selected

        trial_ids = [row["trial_id"] for row in selected]
        placeholders = ",".join("?" for _ in trial_ids)
        child_tables = [table for table in tables_with_trial_id(connection) if table != "trials"]

        connection.execute("BEGIN")
        for table in child_tables:
            connection.execute(
                f"DELETE FROM {quote_identifier(table)} WHERE trial_id IN ({placeholders})",
                trial_ids,
            )
        connection.execute(
            f"DELETE FROM trials WHERE study_id = ? AND trial_id IN ({placeholders})",
            (study_id, *trial_ids),
        )
        connection.commit()

    if not dry_run and vacuum:
        with connect(db_path) as connection:
            connection.execute("VACUUM")
    return selected


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def remove_trial_folders(
    study_dir: Path | None,
    trial_numbers: list[int],
    dry_run: bool,
    keep_folders: bool,
) -> list[Path]:
    if study_dir is None or keep_folders:
        return []

    removed_or_selected = []
    for number in trial_numbers:
        trial_dir = study_dir / f"trial_{number:04d}"
        if not trial_dir.exists():
            continue
        if not trial_dir.is_dir():
            raise SystemExit(f"Expected a trial directory but found a file: {trial_dir}")
        if not is_relative_to(trial_dir, study_dir):
            raise SystemExit(f"Refusing to delete a path outside the study directory: {trial_dir}")
        removed_or_selected.append(trial_dir)
        if not dry_run:
            shutil.rmtree(trial_dir)
    return removed_or_selected


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def to_jsonable(value: Any) -> Any:
    if is_scalar(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def serialize_trial(trial: Any) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": to_jsonable(dict(trial.params)),
        "user_attrs": to_jsonable(trial.user_attrs),
        "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
        "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
        "duration_seconds": None if trial.duration is None else trial.duration.total_seconds(),
    }


def write_trials_summary(study: Any, csv_path: Path, jsonl_path: Path) -> None:
    trials = sorted(study.get_trials(deepcopy=False), key=lambda trial: trial.number)
    params_by_trial = [dict(trial.params) for trial in trials]
    param_names = sorted({name for params in params_by_trial for name in params})
    scalar_attr_names = sorted(
        {
            name
            for trial in trials
            for name, value in trial.user_attrs.items()
            if is_scalar(value) and name != "params"
        }
    )
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        *[f"param:{name}" for name in param_names],
        *[f"attr:{name}" for name in scalar_attr_names],
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for trial, params in zip(trials, params_by_trial):
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "datetime_start": "" if trial.datetime_start is None else trial.datetime_start.isoformat(),
                "datetime_complete": ""
                if trial.datetime_complete is None
                else trial.datetime_complete.isoformat(),
                "duration_seconds": "" if trial.duration is None else trial.duration.total_seconds(),
            }
            for name in param_names:
                row[f"param:{name}"] = json.dumps(to_jsonable(params.get(name)))
            for name in scalar_attr_names:
                row[f"attr:{name}"] = json.dumps(to_jsonable(trial.user_attrs.get(name)))
            writer.writerow(row)

    with jsonl_path.open("w") as jsonl_file:
        for trial in trials:
            jsonl_file.write(json.dumps(serialize_trial(trial), sort_keys=True) + "\n")


def rewrite_summaries(
    study_dir: Path | None,
    storage: str,
    study_name: str,
    dry_run: bool,
) -> list[Path]:
    if study_dir is None:
        return []
    summary_paths = [study_dir / SUMMARY_CSV, study_dir / SUMMARY_JSONL]
    if dry_run:
        return summary_paths

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is required to rewrite trials.csv/trials.jsonl.") from exc

    study = optuna.load_study(study_name=study_name, storage=storage)
    write_trials_summary(study, summary_paths[0], summary_paths[1])
    return summary_paths


def print_plan(
    study_name: str,
    db_path: Path,
    storage: str,
    selected_rows: list[dict[str, Any]],
    folders: list[Path],
    summaries: list[Path],
    backups: list[Path],
    dry_run: bool,
    sampler_migration: TpeSamplerMigrationPlan | None = None,
) -> None:
    action = "Would remove" if dry_run else "Removed"
    print(f"Study: {study_name}")
    print(f"Storage: {storage}")
    print(f"SQLite DB: {db_path}")
    print(f"{action} {len(selected_rows)} trial row(s):")
    for row in selected_rows:
        print(f"  trial_{row['number']:04d}  state={row['state']}  trial_id={row['trial_id']}")
    if folders:
        print(f"{'Would remove' if dry_run else 'Removed'} {len(folders)} artifact folder(s):")
        for folder in folders:
            print(f"  {folder}")
    if summaries:
        print(f"{'Would rewrite' if dry_run else 'Rewrote'} summary file(s):")
        for path in summaries:
            print(f"  {path}")
    if backups:
        print("Backup file(s):")
        for path in backups:
            print(f"  {path}")
    if sampler_migration is not None:
        migration_action = "Would restore" if dry_run else "Restored"
        boundary = (
            "before trial 0"
            if sampler_migration.boundary_trial_number < 0
            else f"after trial {sampler_migration.boundary_trial_number}"
        )
        print("TPE sampler migration:")
        print(
            f"  Original random-startup threshold: "
            f"{sampler_migration.old_startup_trials} ({boundary})"
        )
        if sampler_migration.mode == "extend":
            print(
                f"  New random-startup threshold: "
                f"{sampler_migration.target_startup_trials}"
            )
            preserve_action = "Would preserve" if dry_run else "Preserved"
            print(
                f"  {preserve_action} original boundary checkpoint as "
                f"{sampler_migration.preserved_transition_path}"
            )
            clear_action = "Would clear" if dry_run else "Cleared"
            print(
                f"  {clear_action} {sampler_migration.transition_path.name} so the "
                "new boundary can be checkpointed"
            )
        print(
            f"  {migration_action} boundary sampler as "
            f"{sampler_migration.rolling_sampler_path}"
        )
        print(
            f"  Resume with tpe_startup_trials="
            f"{sampler_migration.target_startup_trials}; set n_trials above that "
            "value to run model-based TPE trials."
        )
    if dry_run:
        print("Dry run only. Re-run with --yes to apply these changes.")


def main() -> None:
    args = parse_args()
    study_dir = None if args.study_dir is None else args.study_dir.resolve()
    study_config = load_study_config(study_dir)
    study_name, storage, db_path = resolve_study_location(args)

    if args.list_studies:
        list_studies(db_path)
        return

    with connect(db_path) as connection:
        study_id = get_study_id(connection, study_name)
        existing_numbers = all_trial_numbers(connection, study_id)
        trial_state_rows = all_trial_state_rows(connection, study_id)

    sampler_migration = make_tpe_sampler_migration_plan(
        args,
        study_dir,
        study_config,
        trial_state_rows,
    )
    if sampler_migration is None:
        trial_numbers = selected_trial_numbers(args, existing_numbers)
    else:
        trial_numbers = sampler_migration.trial_numbers
    dry_run = not args.yes
    backup_enabled = args.yes and not args.no_backup
    backup_paths = [db_path]
    if study_dir is not None:
        backup_paths.extend([study_dir / SUMMARY_CSV, study_dir / SUMMARY_JSONL])
    if sampler_migration is not None:
        backup_paths.extend(
            [
                sampler_migration.rolling_sampler_path,
                sampler_migration.transition_path,
            ]
        )
    backups = backup_files(backup_paths, backup_enabled)
    preserve_tpe_transition_checkpoint(sampler_migration, dry_run)

    if trial_numbers:
        selected_rows = delete_trials_from_sqlite(
            db_path=db_path,
            study_name=study_name,
            trial_numbers=trial_numbers,
            dry_run=dry_run,
            vacuum=not args.no_vacuum,
        )
    else:
        selected_rows = []
    apply_tpe_sampler_migration(sampler_migration, dry_run)
    folders = remove_trial_folders(
        study_dir=study_dir,
        trial_numbers=trial_numbers,
        dry_run=dry_run,
        keep_folders=args.keep_folders,
    )
    summaries = rewrite_summaries(
        study_dir=study_dir,
        storage=storage,
        study_name=study_name,
        dry_run=dry_run,
    )
    print_plan(
        study_name=study_name,
        db_path=db_path,
        storage=storage,
        selected_rows=selected_rows,
        folders=folders,
        summaries=summaries,
        backups=backups,
        dry_run=dry_run,
        sampler_migration=sampler_migration,
    )


if __name__ == "__main__":
    main()
