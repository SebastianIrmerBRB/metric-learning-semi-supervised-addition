"""Report Optuna hyperparameter importances for one study or a study tree."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any


class StudyNotAnalyzable(ValueError):
    """A valid study artifact that cannot produce parameter importances."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an Optuna study and rank parameter importances. "
            "Use --study-dir for one study or --studies-root to recursively "
            "analyze every directory containing optuna_study.db."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--study-dir",
        type=Path,
        help="One study artifact directory containing study_config.json or optuna_study.db.",
    )
    source.add_argument(
        "--studies-root",
        type=Path,
        default=r"exporter/optuna-studies",
        help="Recursively analyze every optuna_study.db below this directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Single-study directory for interactive slice plots. Defaults to "
            "<study-dir>/parameter_importance_plots. Not valid with --studies-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Batch output root. Results mirror each study's relative path below this directory. "
            "By default, each study writes to its own parameter_importance directory."
        ),
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Single-study Optuna storage URL, for example sqlite:///path/to/optuna_study.db. "
            "Not valid with --studies-root."
        ),
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help=(
            "Single-study Optuna study name. By default it is read from study_config.json "
            "or discovered from the database. Not valid with --studies-root."
        ),
    )
    parser.add_argument(
        "--evaluator",
        choices=("default", "fanova", "mdi"),
        default="fanova",
        help="Importance evaluator. 'default' passes no evaluator to Optuna.",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Seed for supported evaluators."
    )
    parser.add_argument(
        "--n-trees", type=int, default=64, help="Tree count for fanova/mdi evaluators."
    )
    parser.add_argument(
        "--max-depth", type=int, default=64, help="Maximum tree depth for fanova."
    )
    parser.add_argument(
        "--target-attr",
        default=None,
        help=(
            "Use this completed-trial user_attr as the target instead of the study objective, "
            "for example best_valid_mean_average_precision_at_r."
        ),
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Optional parameter names to include. Defaults to all parameters Optuna can evaluate.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Return raw evaluator importances instead of normalizing them to sum to 1.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Only print/write the top K parameters."
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Optional JSON output path."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None, help="Optional CSV output path."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip interactive slice plots. This can substantially speed up a batch run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In batch mode, skip studies whose importances.json output is at least "
            "as new as their optuna_study.db. Studies that collected trials since "
            "their last ranking are re-analyzed."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="In batch mode, stop after the first failed study instead of continuing.",
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
        raise ValueError(f"{config_path} must contain a JSON object")
    return config


def discover_study_dirs(studies_root: Path) -> list[Path]:
    """Return directories containing a physical Optuna SQLite database."""

    studies_root = studies_root.resolve()
    if not studies_root.is_dir():
        raise ValueError(
            f"Studies root does not exist or is not a directory: {studies_root}"
        )
    return sorted(
        {
            database.parent
            for database in studies_root.rglob("optuna_study.db")
            if database.is_file()
        },
        key=lambda path: str(path.relative_to(studies_root)).lower(),
    )


def discover_database_study_name(optuna, storage: str) -> str:
    """Resolve an omitted study name when a database contains exactly one study."""

    study_names = optuna.study.get_all_study_names(storage=storage)
    if not study_names:
        raise StudyNotAnalyzable(f"No Optuna studies were found in {storage}.")
    if len(study_names) != 1:
        raise ValueError(
            f"Database contains multiple studies {study_names}; provide --study-name "
            "or add resolved_study_name to study_config.json."
        )
    return str(study_names[0])


def resolve_study_location(
    args: argparse.Namespace,
    optuna,
    study_dir: Path | None = None,
) -> tuple[str, str]:
    study_dir = (
        study_dir.resolve()
        if study_dir is not None
        else args.study_dir.resolve()
        if args.study_dir is not None
        else None
    )
    config = load_study_config(study_dir)

    if args.storage is not None:
        storage = args.storage
    elif study_dir is not None:
        # Prefer the database physically located in --study-dir.
        db_path = study_dir / "optuna_study.db"

        if not db_path.is_file():
            configured_storage = config.get("resolved_storage")
            if configured_storage is None:
                raise ValueError(f"Optuna database does not exist: {db_path}")
            storage = str(configured_storage)
        else:
            storage = f"sqlite:///{db_path.as_posix()}"
    else:
        configured_storage = config.get("resolved_storage")
        if configured_storage is None:
            raise ValueError("Provide --study-dir or --storage.")
        storage = str(configured_storage)

    study_name = args.study_name or config.get("resolved_study_name")
    if study_name is None:
        study_name = discover_database_study_name(optuna, str(storage))

    return str(study_name), str(storage)


def make_evaluator(args: argparse.Namespace):
    if args.evaluator == "default":
        return None

    import optuna.importance

    if args.evaluator == "fanova":
        return optuna.importance.FanovaImportanceEvaluator(
            n_trees=args.n_trees,
            max_depth=args.max_depth,
            seed=args.seed,
        )
    if args.evaluator == "mdi":
        return optuna.importance.MeanDecreaseImpurityImportanceEvaluator(
            n_trees=args.n_trees,
            seed=args.seed,
        )
    raise ValueError(f"Unknown evaluator: {args.evaluator}")


def make_target(study, target_attr: str | None):
    if target_attr is None:
        return None

    missing = []
    non_numeric = []
    for trial in study.get_trials(deepcopy=False, states=None):
        if trial.state.name != "COMPLETE":
            continue
        if target_attr not in trial.user_attrs:
            missing.append(trial.number)
            continue
        try:
            value = float(trial.user_attrs[target_attr])
        except (TypeError, ValueError):
            non_numeric.append(trial.number)
            continue
        if not math.isfinite(value):
            non_numeric.append(trial.number)
    if missing:
        raise ValueError(
            f"Completed trials missing user_attr {target_attr!r}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    if non_numeric:
        raise ValueError(
            f"Completed trials have non-finite/non-numeric user_attr {target_attr!r}: {non_numeric[:10]}"
            + (" ..." if len(non_numeric) > 10 else "")
        )

    def target(trial):
        return float(trial.user_attrs[target_attr])

    return target


def complete_trial_count(study) -> int:
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.state.name == "COMPLETE"
    )


def ranked_rows(
    importances: dict[str, float], top_k: int | None
) -> list[dict[str, Any]]:
    rows = [
        {"rank": rank, "param": name, "importance": float(value)}
        for rank, (name, value) in enumerate(importances.items(), start=1)
    ]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("--top-k must be positive when supplied.")
        rows = rows[:top_k]
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No parameter importances returned.")
        return
    name_width = max(len("param"), *(len(row["param"]) for row in rows))
    print(f"{'rank':>4}  {'param':<{name_width}}  importance")
    print(f"{'-' * 4}  {'-' * name_width}  {'-' * 10}")
    for row in rows:
        print(
            f"{row['rank']:>4}  {row['param']:<{name_width}}  {row['importance']:.8f}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["rank", "param", "importance"])
        writer.writeheader()
        writer.writerows(rows)


def select_plot_params(study) -> list[str]:
    """Select batch sampler, SSL, loss, miner, and regularizer parameters."""
    parameter_names: set[str] = set()

    for trial in study.get_trials(deepcopy=False):
        if trial.state.name != "COMPLETE":
            continue
        parameter_names.update(trial.params)

    def should_plot(name: str) -> bool:
        normalized = name.lower()

        return (
            normalized in {"batch_sampler", "lr", "classifier_lr"}
            or normalized.startswith("ssl_config.")
            or "loss" in normalized
            or "miner" in normalized
            or "regularizer" in normalized
        )

    return sorted(name for name in parameter_names if should_plot(name))


def safe_filename(value: str) -> str:
    """Convert a parameter name into a safe filename."""
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def write_slice_plots(
    study,
    params: list[str],
    target,
    target_name: str,
    output_dir: Path,
    *,
    verbose: bool = True,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Write one interactive slice plot per parameter."""
    import optuna.visualization

    output_dir.mkdir(parents=True, exist_ok=True)

    if not params:
        if verbose:
            print("No selected parameters found for slice plots.")
        return [], []

    if verbose:
        print("\nWriting slice plots:")

    output_paths = []
    errors = []
    for param in params:
        try:
            figure = optuna.visualization.plot_slice(
                study,
                params=[param],
                target=target,
                target_name=target_name,
            )

            output_path = output_dir / f"slice_{safe_filename(param)}.html"
            figure.write_html(str(output_path))
            output_paths.append(output_path)

            if verbose:
                print(f"  {param}")
                print(f"    -> {output_path}")
        except Exception as exc:
            errors.append(
                {
                    "param": param,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"  {param}: {type(exc).__name__}: {exc}")
    return output_paths, errors


def close_study_storage(study) -> None:
    """Release Optuna/SQLAlchemy SQLite handles, especially on Windows."""

    storage = getattr(study, "_storage", None)
    backend = getattr(storage, "_backend", None)
    candidates = [
        candidate for candidate in (storage, backend) if candidate is not None
    ]
    seen = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        remove_session = getattr(candidate, "remove_session", None)
        if callable(remove_session):
            try:
                remove_session()
            except Exception:
                pass
        engine = getattr(candidate, "engine", None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


def analyze_study(
    optuna,
    args: argparse.Namespace,
    study_dir: Path,
    *,
    plots_dir: Path | None,
    verbose: bool,
) -> dict[str, Any]:
    """Analyze one study and return a JSON-serializable result."""

    study_name, storage = resolve_study_location(args, optuna, study_dir)
    study = optuna.load_study(study_name=study_name, storage=storage)
    try:
        num_complete = complete_trial_count(study)
        if num_complete < 2:
            raise StudyNotAnalyzable(
                f"Study {study_name!r} has {num_complete} completed trial(s); "
                "parameter importance requires at least two."
            )

        evaluator = make_evaluator(args)
        target = make_target(study, args.target_attr)
        importances = optuna.importance.get_param_importances(
            study,
            evaluator=evaluator,
            params=args.params,
            target=target,
            normalize=not args.no_normalize,
        )
        rows = ranked_rows(importances, args.top_k)
        plot_params = select_plot_params(study)
        plot_paths: list[Path] = []
        plot_errors: list[dict[str, str]] = []

        if not args.no_plots and plots_dir is not None:
            try:
                plot_paths, plot_errors = write_slice_plots(
                    study=study,
                    params=plot_params,
                    target=target,
                    target_name="Objective"
                    if args.target_attr is None
                    else args.target_attr,
                    output_dir=plots_dir,
                    verbose=verbose,
                )
            except Exception as exc:
                plot_errors.append(
                    {
                        "param": "*",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        return {
            "study_name": study_name,
            "study_dir": str(study_dir.resolve()),
            "storage": storage,
            "evaluator": args.evaluator,
            "target": "objective"
            if args.target_attr is None
            else f"user_attrs.{args.target_attr}",
            "normalized": not args.no_normalize,
            "num_complete_trials": num_complete,
            "importances": rows,
            "plot_params": plot_params,
            "plot_files": [str(path.resolve()) for path in plot_paths],
            "plot_errors": plot_errors,
        }
    finally:
        close_study_storage(study)
        del study
        gc.collect()


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.studies_root is None:
        if args.output_root is not None:
            raise ValueError("--output-root only applies with --studies-root.")
        if args.skip_existing:
            raise ValueError("--skip-existing only applies with --studies-root.")
        if args.fail_fast:
            raise ValueError("--fail-fast only applies with --studies-root.")
        return

    incompatible = []
    for option, value in (
        ("--plots-dir", args.plots_dir),
        ("--storage", args.storage),
        ("--study-name", args.study_name),
        ("--output-json", args.output_json),
        ("--output-csv", args.output_csv),
    ):
        if value is not None:
            incompatible.append(option)
    if incompatible:
        raise ValueError(
            f"{', '.join(incompatible)} cannot be used with --studies-root. "
            "Use --output-root to control batch output placement."
        )


def require_plotly_when_needed(args: argparse.Namespace) -> None:
    if args.no_plots:
        return
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "Plot generation requires Plotly in the Python environment running this script. "
            "Install it with `python -m pip install plotly`, or pass --no-plots."
        ) from exc


def run_single_study(optuna, args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    plots_dir = None
    if not args.no_plots:
        plots_dir = (
            args.plots_dir.resolve()
            if args.plots_dir is not None
            else study_dir / "parameter_importance_plots"
        )
    payload = analyze_study(
        optuna,
        args,
        study_dir,
        plots_dir=plots_dir,
        verbose=True,
    )
    print_table(payload["importances"])
    if args.output_json is not None:
        write_json(args.output_json, payload)
    if args.output_csv is not None:
        write_csv(args.output_csv, payload["importances"])


def resolve_batch_result_dir(
    args: argparse.Namespace,
    studies_root: Path,
    study_dir: Path,
) -> Path:
    if args.output_root is None:
        return study_dir / "parameter_importance"
    return args.output_root.resolve() / study_dir.relative_to(studies_root)


def importance_is_current(result_json: Path, study_dir: Path) -> bool:
    """Whether a previous run already ranked every trial the database holds.

    Existence of importances.json is not enough: a study that keeps running
    collects trials after its ranking was written, and skipping it would pin
    the report to a stale trial count for good.
    """

    if not result_json.is_file():
        return False
    db_path = study_dir / "optuna_study.db"
    if not db_path.is_file():
        return True
    return result_json.stat().st_mtime >= db_path.stat().st_mtime


def batch_summary_paths(
    args: argparse.Namespace, studies_root: Path
) -> tuple[Path, Path]:
    if args.output_root is None:
        return (
            studies_root / "parameter_importance_summary.json",
            studies_root / "parameter_importance_summary.csv",
        )
    output_root = args.output_root.resolve()
    return output_root / "summary.json", output_root / "summary.csv"


def make_batch_summary(
    args: argparse.Namespace,
    studies_root: Path,
    entries: list[dict[str, Any]],
    num_discovered: int,
) -> dict[str, Any]:
    return {
        "studies_root": str(studies_root),
        "evaluator": args.evaluator,
        "target": "objective"
        if args.target_attr is None
        else f"user_attrs.{args.target_attr}",
        "normalized": not args.no_normalize,
        "num_discovered": num_discovered,
        "num_processed": len(entries),
        "num_succeeded": sum(entry["status"] == "ok" for entry in entries),
        "num_skipped": sum(entry["status"] == "skipped" for entry in entries),
        "num_failed": sum(entry["status"] == "error" for entry in entries),
        "studies": entries,
    }


def make_batch_csv_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        importances = entry.get("importances") or [None]
        for importance in importances:
            rows.append(
                {
                    "status": entry["status"],
                    "relative_study_dir": entry["relative_study_dir"],
                    "study_name": entry.get("study_name", ""),
                    "num_complete_trials": entry.get("num_complete_trials", ""),
                    "rank": "" if importance is None else importance["rank"],
                    "param": "" if importance is None else importance["param"],
                    "importance": ""
                    if importance is None
                    else importance["importance"],
                    "skip_reason": entry.get("skip_reason", ""),
                    "error_type": entry.get("error_type", ""),
                    "error": entry.get("error", ""),
                }
            )
    return rows


def write_batch_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "relative_study_dir",
        "study_name",
        "num_complete_trials",
        "rank",
        "param",
        "importance",
        "skip_reason",
        "error_type",
        "error",
    ]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(make_batch_csv_rows(entries))


def write_batch_summary(
    args: argparse.Namespace,
    studies_root: Path,
    entries: list[dict[str, Any]],
    num_discovered: int,
) -> tuple[Path, Path]:
    json_path, csv_path = batch_summary_paths(args, studies_root)
    write_json(
        json_path,
        make_batch_summary(args, studies_root, entries, num_discovered),
    )
    write_batch_csv(csv_path, entries)
    return json_path, csv_path


def load_existing_batch_result(path: Path) -> dict[str, Any]:
    with path.open() as result_file:
        payload = json.load(result_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def run_study_batch(optuna, args: argparse.Namespace) -> int:
    studies_root = args.studies_root.resolve()
    study_dirs = discover_study_dirs(studies_root)
    if not study_dirs:
        raise ValueError(f"No optuna_study.db files found below {studies_root}.")

    print(f"Discovered {len(study_dirs)} Optuna studies below {studies_root}.")
    entries: list[dict[str, Any]] = []
    for index, study_dir in enumerate(study_dirs, start=1):
        relative_study_dir = str(study_dir.relative_to(studies_root))
        result_dir = resolve_batch_result_dir(args, studies_root, study_dir)
        result_json = result_dir / "importances.json"
        result_csv = result_dir / "importances.csv"
        plots_dir = None if args.no_plots else result_dir / "plots"

        try:
            if args.skip_existing and importance_is_current(result_json, study_dir):
                payload = load_existing_batch_result(result_json)
                status = "skipped"
            else:
                payload = analyze_study(
                    optuna,
                    args,
                    study_dir,
                    plots_dir=plots_dir,
                    verbose=False,
                )
                write_json(result_json, payload)
                write_csv(result_csv, payload["importances"])
                status = "ok"

            entry = {
                "status": status,
                "relative_study_dir": relative_study_dir,
                "result_dir": str(result_dir.resolve()),
                **payload,
            }
            entries.append(entry)
            print(
                f"[{index}/{len(study_dirs)}] {status.upper():7} "
                f"{relative_study_dir} "
                f"({entry.get('num_complete_trials', '?')} complete, "
                f"{len(entry.get('importances') or [])} importances)"
            )
        except StudyNotAnalyzable as exc:
            entry = {
                "status": "skipped",
                "relative_study_dir": relative_study_dir,
                "result_dir": str(result_dir.resolve()),
                "skip_reason": str(exc),
            }
            entries.append(entry)
            print(f"[{index}/{len(study_dirs)}] SKIPPED {relative_study_dir}: {exc}")
            write_batch_summary(args, studies_root, entries, len(study_dirs))
            continue
        except Exception as exc:
            entry = {
                "status": "error",
                "relative_study_dir": relative_study_dir,
                "result_dir": str(result_dir.resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            entries.append(entry)
            print(
                f"[{index}/{len(study_dirs)}] ERROR   {relative_study_dir}: "
                f"{type(exc).__name__}: {exc}"
            )
            write_batch_summary(args, studies_root, entries, len(study_dirs))
            if args.fail_fast:
                raise
            continue

        write_batch_summary(args, studies_root, entries, len(study_dirs))

    json_path, csv_path = write_batch_summary(
        args,
        studies_root,
        entries,
        len(study_dirs),
    )
    failed = sum(entry["status"] == "error" for entry in entries)
    skipped = sum(entry["status"] == "skipped" for entry in entries)
    succeeded = sum(entry["status"] == "ok" for entry in entries)
    print(
        f"Finished: {succeeded} analyzed, {skipped} skipped, {failed} failed. "
        f"Summary: {json_path} and {csv_path}"
    )
    return failed


def main() -> None:
    args = parse_args()

    try:
        import optuna
        import optuna.importance
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required. Install it with `pip install optuna`."
        ) from exc

    try:
        validate_mode_args(args)
        require_plotly_when_needed(args)
        if args.studies_root is None:
            run_single_study(optuna, args)
            return
        failed = run_study_batch(optuna, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
