#!/usr/bin/env python3
"""Re-run recorded trials of one Optuna HPO study as fresh training runs.

The script never touches the study it reads.  It loads ``study_config.json``
for the study's base arguments, replays each selected trial's recorded
parameters through the same ``make_args_and_ssl_config_from_params`` path that
HPO itself uses, and writes every re-run below a new output directory.  The
original Optuna database, sampler state, and trial directories stay untouched,
so a re-run can be compared against the value the study recorded.

Choosing inputs:

* ``--study`` selects the study directory; omitting it prints a numbered menu
  of every directory below ``--log-root`` that contains ``study_config.json``.
* ``--n_jobs`` sets how many trials train at once; omitting it prompts.  Each
  trial runs in its own process, so trials cannot share model or cache state.
* ``--device`` overrides the device recorded by the study; it defaults to
  ``cpu``.  Pass ``study`` to keep whatever device the study was tuned on.
* ``--batch_sizes`` keeps only trials whose effective batch size is one of the
  given values (32 and 64 by default).  The batch size comes from the trial's
  own ``batch_sampler``/``batch_size`` parameter and falls back to the study's
  base arguments when neither was tuned.

Because CPU re-runs are slow, ``--num_workers`` and ``--torch_threads`` are
available to keep concurrent workers from oversubscribing the machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    # Worker processes import main.py from the workspace root, and spawn
    # re-imports this module before running the worker function.
    sys.path.insert(0, str(WORKSPACE_ROOT))

LOG_ROOT_NAME = "logs"
STUDY_CONFIG_FILENAME = "study_config.json"
TRIALS_JSONL_FILENAME = "trials.jsonl"
BATCH_SAMPLER_PARAM = "batch_sampler"
BATCH_SIZE_PARAM = "batch_size"
JOINT_PARAM_PREFIXES = ("__joint_component__.", "__joint_hparam__.")
DEFAULT_BATCH_SIZES = (32, 64)
TRIAL_STATES = ("COMPLETE", "PRUNED", "FAIL")
DEFAULT_TRIAL_STATES = ("COMPLETE",)
RERUN_DIR_NAME = "reruns"
KEEP_STUDY_DEVICE = "study"
DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")
# main.py sets these with setdefault, so a value exported here wins.
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    value: float | None
    params: dict
    batch_size: int | None
    sampler_m: int | None


@dataclass(frozen=True)
class Study:
    study_dir: Path
    study_name: str
    metric: str
    direction: str
    base_args: dict
    trials: tuple[TrialRecord, ...]
    trial_source: str


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def expand_joint_params(params: dict) -> dict:
    """Unpack Optuna joint parameters the way training/hpo.py does."""

    expanded = {}
    for name, value in params.items():
        if name.startswith(JOINT_PARAM_PREFIXES):
            expanded.update(json.loads(value))
            continue
        expanded[name] = value
    return expanded


def parse_batch_sampler_choice(choice) -> tuple[int, int] | tuple[None, None]:
    """Split a 'batch_size:sampler_m' choice, tolerating unexpected values."""

    if not isinstance(choice, str) or ":" not in choice:
        return None, None
    raw_batch_size, raw_sampler_m = choice.split(":", 1)
    try:
        return int(raw_batch_size), int(raw_sampler_m)
    except ValueError:
        return None, None


def resolve_trial_batch_size(params: dict, base_args: dict) -> tuple[int | None, int | None]:
    """Return the batch size and sampler_m a trial actually trained with."""

    if BATCH_SAMPLER_PARAM in params:
        return parse_batch_sampler_choice(params[BATCH_SAMPLER_PARAM])

    batch_size = params.get(BATCH_SIZE_PARAM, base_args.get(BATCH_SIZE_PARAM))
    sampler_m = params.get("sampler_m", base_args.get("sampler_m"))
    try:
        batch_size = None if batch_size is None else int(batch_size)
        sampler_m = None if sampler_m is None else int(sampler_m)
    except (TypeError, ValueError):
        return None, None
    return batch_size, sampler_m


def discover_studies(log_root: Path) -> list[Path]:
    return sorted({path.parent for path in Path(log_root).rglob(STUDY_CONFIG_FILENAME)})


def load_trials_from_jsonl(trials_jsonl: Path, base_args: dict) -> list[TrialRecord]:
    trials = []
    with Path(trials_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trial = json.loads(line)
            params = expand_joint_params(trial.get("params") or {})
            batch_size, sampler_m = resolve_trial_batch_size(params, base_args)
            trials.append(
                TrialRecord(
                    number=int(trial["number"]),
                    state=str(trial.get("state")),
                    value=None if trial.get("value") is None else float(trial["value"]),
                    params=params,
                    batch_size=batch_size,
                    sampler_m=sampler_m,
                )
            )
    return trials


def load_trials_from_storage(study_config: dict, study_dir: Path, base_args: dict) -> list[TrialRecord]:
    """Read trials straight from the Optuna database when no summary exists."""

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            f"{study_dir / TRIALS_JSONL_FILENAME} is missing and Optuna is not installed "
            "for the storage fallback."
        ) from exc

    storage = study_config.get("resolved_storage")
    if not storage:
        storage_path = (Path(study_dir) / "optuna_study.db").resolve()
        storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = study_config.get("resolved_study_name") or Path(study_dir).name
    study = optuna.load_study(study_name=study_name, storage=storage)

    trials = []
    for trial in study.get_trials(deepcopy=False):
        params = expand_joint_params(dict(trial.params))
        batch_size, sampler_m = resolve_trial_batch_size(params, base_args)
        trials.append(
            TrialRecord(
                number=int(trial.number),
                state=trial.state.name,
                value=None if trial.value is None else float(trial.value),
                params=params,
                batch_size=batch_size,
                sampler_m=sampler_m,
            )
        )
    return trials


def load_study(study_dir: Path) -> Study:
    study_config_path = Path(study_dir) / STUDY_CONFIG_FILENAME
    if not study_config_path.exists():
        raise SystemExit(f"Not an HPO study directory (no {STUDY_CONFIG_FILENAME}): {study_dir}")

    study_config = read_json(study_config_path)
    base_args = study_config.get("base_args")
    if not isinstance(base_args, dict):
        raise SystemExit(f"Study config must contain a base_args object: {study_config_path}")
    hparam_config = study_config.get("hparam_config") or {}

    trials_jsonl = Path(study_dir) / TRIALS_JSONL_FILENAME
    if trials_jsonl.exists():
        trials = load_trials_from_jsonl(trials_jsonl, base_args)
        trial_source = str(trials_jsonl)
    else:
        trials = load_trials_from_storage(study_config, study_dir, base_args)
        trial_source = "optuna storage"

    return Study(
        study_dir=Path(study_dir),
        study_name=(
            study_config.get("resolved_study_name") or hparam_config.get("study_name") or Path(study_dir).name
        ),
        metric=hparam_config.get("metric") or "best_valid_mean_average_precision_at_r",
        direction=hparam_config.get("direction") or "maximize",
        base_args=base_args,
        trials=tuple(sorted(trials, key=lambda trial: trial.number)),
        trial_source=trial_source,
    )


def select_trials(
    trials: tuple[TrialRecord, ...],
    batch_sizes: list[int] | None,
    states: list[str],
    trial_numbers: list[int] | None,
) -> tuple[list[TrialRecord], list[str]]:
    """Filter the study's trials and explain every requested trial that drops out."""

    notes = []
    by_number = {trial.number: trial for trial in trials}
    if trial_numbers is not None:
        missing = sorted(set(trial_numbers) - set(by_number))
        if missing:
            raise SystemExit(f"Trial(s) not present in this study: {missing}")
        candidates = [by_number[number] for number in sorted(set(trial_numbers))]
    else:
        candidates = list(trials)

    selected = []
    for trial in candidates:
        if trial.state not in states:
            if trial_numbers is not None:
                notes.append(f"trial {trial.number}: state {trial.state} is not in {states}")
            continue
        if trial.batch_size is None:
            notes.append(f"trial {trial.number}: batch size could not be determined; skipped")
            continue
        if batch_sizes is not None and trial.batch_size not in batch_sizes:
            if trial_numbers is not None:
                notes.append(f"trial {trial.number}: batch size {trial.batch_size} is not in {batch_sizes}")
            continue
        selected.append(trial)
    return selected, notes


def count_matching_trials(study_dir: Path, batch_sizes: list[int] | None, states: list[str]) -> str:
    """Describe one study for the selection menu without loading Optuna."""

    trials_jsonl = Path(study_dir) / TRIALS_JSONL_FILENAME
    if not trials_jsonl.exists():
        return "trial counts unknown (no trials.jsonl)"
    try:
        base_args = read_json(Path(study_dir) / STUDY_CONFIG_FILENAME).get("base_args") or {}
        trials = load_trials_from_jsonl(trials_jsonl, base_args)
    except (OSError, ValueError) as exc:
        return f"unreadable: {exc}"
    selected, _ = select_trials(tuple(trials), batch_sizes, states, None)
    return f"{len(trials)} trials, {len(selected)} selectable"


def read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit("\nNo input available; pass --study and --n_jobs explicitly.")


def choose_study(studies: list[Path], log_root: Path, batch_sizes: list[int] | None, states: list[str]) -> Path:
    print(f"Studies below {relative_to_workspace(log_root)}:\n")
    for index, study_dir in enumerate(studies):
        relative = relative_to_workspace(study_dir)
        print(f"  [{index:>2}] {relative}")
        print(f"       {count_matching_trials(study_dir, batch_sizes, states)}")
    print()

    while True:
        answer = read_line(f"Select a study [0-{len(studies) - 1}]: ")
        try:
            index = int(answer)
        except ValueError:
            print("Enter the number shown in brackets.")
            continue
        if 0 <= index < len(studies):
            return studies[index]
        print(f"Enter a number between 0 and {len(studies) - 1}.")


def choose_n_jobs(default: int = 1) -> int:
    while True:
        answer = read_line(f"Number of parallel trials (n_jobs) [{default}]: ")
        if not answer:
            return default
        try:
            n_jobs = int(answer)
        except ValueError:
            print("Enter a positive integer.")
            continue
        if n_jobs > 0:
            return n_jobs
        print("Enter a positive integer.")


def confirm(prompt: str) -> bool:
    return read_line(f"{prompt} [y/N]: ").lower() in {"y", "yes"}


def relative_to_workspace(path: Path) -> Path:
    try:
        return Path(path).relative_to(WORKSPACE_ROOT)
    except ValueError:
        return Path(path)


def make_save_dir(study_dir: Path, run_name: str) -> Path:
    """Return the re-run output directory as a path below logs/."""

    log_root = WORKSPACE_ROOT / LOG_ROOT_NAME
    try:
        study_relative = Path(study_dir).relative_to(log_root)
    except ValueError:
        # A study configured outside logs/ still needs a logs-relative save_dir
        # because initialize_logger always prepends logs/.
        return Path(RERUN_DIR_NAME) / Path(study_dir).name / run_name
    return study_relative / RERUN_DIR_NAME / run_name


def resolve_input_path(path: Path, original_cwd: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    bases = (original_cwd, WORKSPACE_ROOT, WORKSPACE_ROOT / LOG_ROOT_NAME)
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    return (original_cwd / candidate).resolve()


def make_trial_request(
    study: Study,
    trial: TrialRecord,
    save_dir: Path,
    args: argparse.Namespace,
) -> dict:
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "study_dir": str(study.study_dir),
        "study_name": study.study_name,
        "metric": study.metric,
        "trial_number": trial.number,
        "params": trial.params,
        "save_dir": str(save_dir / f"trial_{trial.number:04d}"),
        "device": None if args.device == KEEP_STUDY_DEVICE else args.device,
        "num_workers": args.num_workers,
        "torch_threads": args.torch_threads,
    }


def run_trial_request(request: dict) -> dict:
    """Train one trial's recorded parameters in this worker process."""

    record = {
        "trial_number": request["trial_number"],
        "status": "failed",
        "value": None,
        "log_dir": None,
        "error": None,
        "duration_seconds": None,
        "metrics": None,
    }
    started = time.perf_counter()
    try:
        os.chdir(request["workspace_root"])
        if request["torch_threads"] is not None:
            for name in THREAD_ENVIRONMENT_VARIABLES:
                os.environ[name] = str(request["torch_threads"])

        import torch

        import main

        if request["torch_threads"] is not None:
            # main.py pins one thread per process for HPO; CPU re-runs are
            # usually thread-bound instead.
            torch.set_num_threads(request["torch_threads"])

        study_config = read_json(Path(request["study_dir"]) / STUDY_CONFIG_FILENAME)
        base_args = main.make_base_args_from_study_config(study_config["base_args"])
        # Overrides are applied before the trial args are derived so that the
        # same validation HPO performs sees the values this re-run will use.
        if request["device"] is not None:
            base_args.device = request["device"]
        if request["num_workers"] is not None:
            base_args.num_workers = request["num_workers"]

        params = main.expand_joint_component_params(dict(request["params"]))
        trial_args, ssl_config = main.make_args_and_ssl_config_from_params(base_args, params)
        trial_args.save_dir = Path(request["save_dir"])
        # trial_number keeps the run directory stable instead of adding a
        # timestamp child, so re-running the same trial overwrites its output.
        trial_args.trial_number = request["trial_number"]
        trial_args.hparam_study_dir = Path(request["study_dir"])
        trial_args.hparam_study_name = request["study_name"]
        trial_args.hparam_params = params
        trial_args.hparam_config_resolved = study_config.get("hparam_config")
        trial_args.evaluate_test = not bool(getattr(base_args, "skip_test_during_hpo", False))
        trial_args.final_test_visualization = main.FINAL_TEST_VISUALIZATION_NONE

        result = main.run_experiment(trial_args, ssl_config)
        metrics = main.result_to_dict(result)
        record.update(
            status="completed",
            value=metrics.get(request["metric"]),
            log_dir=metrics.get("log_dir"),
            metrics=metrics,
        )
    except BaseException as exc:  # noqa: BLE001 - the summary must record every failure
        record.update(
            # Numerical divergence is an expected outcome of some sampled
            # parameters; HPO prunes such trials instead of failing the study.
            status="diverged" if type(exc).__name__ == "NonFiniteEmbeddingError" else "failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
        )
    finally:
        record["duration_seconds"] = time.perf_counter() - started
    return record


def format_value(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def print_plan(study: Study, selected: list[TrialRecord], notes: list[str], args: argparse.Namespace, save_dir: Path) -> None:
    print()
    print(f"Study:      {relative_to_workspace(study.study_dir)}")
    print(f"Study name: {study.study_name}")
    print(f"Metric:     {study.metric} ({study.direction}) from {study.trial_source}")
    print(f"Device:     {'study default' if args.device == KEEP_STUDY_DEVICE else args.device}")
    print(f"Output:     {Path(LOG_ROOT_NAME) / save_dir}")
    print(f"Trials:     {len(selected)} selected of {len(study.trials)}")
    print()
    print(f"  {'trial':>5}  {'state':<9} {'batch':>6} {'m':>4}  {'recorded':>10}")
    for trial in selected:
        sampler_m = "-" if trial.sampler_m is None else str(trial.sampler_m)
        print(
            f"  {trial.number:>5}  {trial.state:<9} {trial.batch_size:>6} {sampler_m:>4}  "
            f"{format_value(trial.value):>10}"
        )
    for note in notes:
        print(f"  note: {note}")
    print()


def print_results(results: list[dict], selected_by_number: dict[int, TrialRecord], metric: str) -> None:
    print()
    print(f"  {'trial':>5}  {'status':<10} {'recorded':>10} {'rerun':>10} {'delta':>10}  {'minutes':>8}")
    for record in sorted(results, key=lambda item: item["trial_number"]):
        trial = selected_by_number[record["trial_number"]]
        rerun_value = record["value"]
        delta = "-"
        if rerun_value is not None and trial.value is not None:
            delta = f"{float(rerun_value) - float(trial.value):+.6f}"
        minutes = "-" if record["duration_seconds"] is None else f"{record['duration_seconds'] / 60:.1f}"
        print(
            f"  {record['trial_number']:>5}  {record['status']:<10} {format_value(trial.value):>10} "
            f"{format_value(rerun_value):>10} {delta:>10}  {minutes:>8}"
        )
        if record["error"]:
            print(f"         {record['error']}")
    print(f"\nMetric: {metric}")


def write_summary(
    output_dir: Path,
    study: Study,
    selected: list[TrialRecord],
    results: list[dict],
    args: argparse.Namespace,
    wall_seconds: float,
) -> tuple[Path, Path]:
    selected_by_number = {trial.number: trial for trial in selected}
    rows = []
    for record in sorted(results, key=lambda item: item["trial_number"]):
        trial = selected_by_number[record["trial_number"]]
        rerun_value = record["value"]
        rows.append(
            {
                "trial_number": trial.number,
                "state": trial.state,
                "batch_size": trial.batch_size,
                "sampler_m": trial.sampler_m,
                "recorded_value": trial.value,
                "rerun_value": rerun_value,
                "delta": (
                    None
                    if rerun_value is None or trial.value is None
                    else float(rerun_value) - float(trial.value)
                ),
                "status": record["status"],
                "duration_seconds": record["duration_seconds"],
                "log_dir": record["log_dir"],
                "error": record["error"],
            }
        )

    summary_json = output_dir / "rerun_summary.json"
    write_json(
        summary_json,
        {
            "study_dir": str(study.study_dir),
            "study_name": study.study_name,
            "metric": study.metric,
            "direction": study.direction,
            "device": "study default" if args.device == KEEP_STUDY_DEVICE else args.device,
            "n_jobs": args.n_jobs,
            "num_workers": args.num_workers,
            "torch_threads": args.torch_threads,
            "batch_sizes": args.batch_sizes,
            "states": args.states,
            "wall_seconds": wall_seconds,
            "trials": rows,
            "trial_params": {str(trial.number): trial.params for trial in selected},
            "trial_results": {str(record["trial_number"]): record for record in results},
        },
    )

    summary_csv = output_dir / "rerun_summary.csv"
    fieldnames = [
        "trial_number",
        "state",
        "batch_size",
        "sampler_m",
        "recorded_value",
        "rerun_value",
        "delta",
        "status",
        "duration_seconds",
        "log_dir",
        "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_json, summary_csv


def run_trials(requests: list[dict], n_jobs: int) -> list[dict]:
    # Even a single worker runs out of process, so a crashed trial cannot take
    # down the whole re-run and the summary still records what happened.
    context = multiprocessing.get_context("spawn")
    results = []
    # maxtasksperchild=1 gives each trial a fresh interpreter, matching the
    # isolation of separate HPO processes.
    with context.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
        try:
            for index, record in enumerate(pool.imap_unordered(run_trial_request, requests), start=1):
                results.append(record)
                minutes = "-" if record["duration_seconds"] is None else f"{record['duration_seconds'] / 60:.1f}"
                print(
                    f"[{index}/{len(requests)}] trial {record['trial_number']}: "
                    f"{record['status']} in {minutes} min",
                    flush=True,
                )
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print("\nInterrupted; writing a summary for the trials that finished.", flush=True)
    return results


def parse_device(value: str) -> str:
    if value == KEEP_STUDY_DEVICE or DEVICE_PATTERN.match(value):
        return value
    raise argparse.ArgumentTypeError(f"device must be cpu, cuda, cuda:N, or {KEEP_STUDY_DEVICE!r}: {value}")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run recorded trials of one Optuna HPO study as fresh training runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=None,
        help="Study directory containing study_config.json. Omit to choose from a menu.",
    )
    parser.add_argument(
        "--log_root",
        "--log-root",
        type=Path,
        default=Path(LOG_ROOT_NAME),
        help="Directory searched for studies when --study is omitted.",
    )
    parser.add_argument(
        "--n_jobs",
        "--n-jobs",
        type=positive_int,
        default=None,
        help="Trials to train at once, each in its own process. Omit to be prompted.",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default="cpu",
        help=f"Device for every re-run: cpu, cuda, cuda:N, or {KEEP_STUDY_DEVICE!r} to keep the study's device.",
    )
    parser.add_argument(
        "--batch_sizes",
        "--batch-sizes",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Only re-run trials whose effective batch size is one of these values.",
    )
    parser.add_argument(
        "--all_batch_sizes",
        "--all-batch-sizes",
        action="store_true",
        help="Re-run trials of every batch size instead of filtering.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=TRIAL_STATES,
        default=list(DEFAULT_TRIAL_STATES),
        help="Trial states eligible for re-running.",
    )
    parser.add_argument(
        "--trials",
        type=non_negative_int,
        nargs="+",
        default=None,
        help="Restrict the re-run to these trial numbers; the other filters still apply.",
    )
    parser.add_argument(
        "--num_workers",
        "--num-workers",
        type=non_negative_int,
        default=None,
        help="Override the study's DataLoader worker count per trial.",
    )
    parser.add_argument(
        "--torch_threads",
        "--torch-threads",
        type=positive_int,
        default=None,
        help="Torch/BLAS threads per trial process. main.py pins 1 thread when unset.",
    )
    parser.add_argument(
        "--run_name",
        "--run-name",
        default=None,
        help="Name of the output directory below the study's reruns/ folder.",
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        type=Path,
        default=None,
        help="Output directory below logs/, replacing the default reruns/ location.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    original_cwd = Path.cwd()
    batch_sizes = None if args.all_batch_sizes else sorted(set(args.batch_sizes))
    args.batch_sizes = batch_sizes

    if args.study is not None:
        study_dir = resolve_input_path(args.study, original_cwd)
    else:
        log_root = resolve_input_path(args.log_root, original_cwd)
        studies = discover_studies(log_root)
        if not studies:
            print(f"No HPO studies found below {log_root}", file=sys.stderr)
            return 2
        study_dir = choose_study(studies, log_root, batch_sizes, args.states)

    # Runs resolve logs/ and data/ relative to the workspace root.
    os.chdir(WORKSPACE_ROOT)
    study = load_study(study_dir)
    selected, notes = select_trials(study.trials, batch_sizes, args.states, args.trials)

    run_name = args.run_name or f"rerun_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_dir = Path(args.save_dir) if args.save_dir is not None else make_save_dir(study.study_dir, run_name)
    print_plan(study, selected, notes, args, save_dir)

    if not selected:
        print("No trial matches the requested filters.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    if args.n_jobs is None:
        args.n_jobs = choose_n_jobs()
    if not args.yes and not confirm(f"Re-run {len(selected)} trial(s) with n_jobs={args.n_jobs}?"):
        print("Aborted.")
        return 1

    output_dir = WORKSPACE_ROOT / LOG_ROOT_NAME / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "rerun_plan.json",
        {
            "study_dir": str(study.study_dir),
            "study_name": study.study_name,
            "metric": study.metric,
            "direction": study.direction,
            "device": args.device,
            "n_jobs": args.n_jobs,
            "num_workers": args.num_workers,
            "torch_threads": args.torch_threads,
            "batch_sizes": batch_sizes,
            "states": args.states,
            "notes": notes,
            "trials": [
                {
                    "trial_number": trial.number,
                    "state": trial.state,
                    "batch_size": trial.batch_size,
                    "sampler_m": trial.sampler_m,
                    "recorded_value": trial.value,
                    "params": trial.params,
                }
                for trial in selected
            ],
        },
    )

    requests = [make_trial_request(study, trial, save_dir, args) for trial in selected]
    started = time.perf_counter()
    results = run_trials(requests, args.n_jobs)
    wall_seconds = time.perf_counter() - started

    finished_numbers = {record["trial_number"] for record in results}
    summary_json, summary_csv = write_summary(
        output_dir,
        study,
        [trial for trial in selected if trial.number in finished_numbers],
        results,
        args,
        wall_seconds,
    )
    print_results(results, {trial.number: trial for trial in selected}, study.metric)
    print(f"Wall time: {wall_seconds / 60:.1f} min")
    print(f"Summary:   {relative_to_workspace(summary_json)}")
    print(f"           {relative_to_workspace(summary_csv)}")

    failed = [record for record in results if record["status"] == "failed"]
    if failed or len(results) < len(requests):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
