#!/usr/bin/env python3
"""Estimate early-stopping patience across complete log directory trees.

The calculation mirrors ``training/engine.py``:

* the model-selection metric must improve strictly;
* equal or worse values consume one patience unit;
* validation checks during SSL warmup do not consume patience; and
* an improvement resets the non-improvement counter.

For each run, the reported minimum is the smallest positive integer patience
that would have allowed training to reach the best checkpoint present in the
recorded trace.  Non-improving checks after that best checkpoint do not affect
this counterfactual minimum.  Each input is a log root; every nested run with a
``metrics.csv`` or ``info.log`` artifact is discovered recursively.  The
``--non-verbose`` mode instead prints only run directories whose recorded trace
reaches the configured early-stopping patience.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAP_AT_R = "mean_average_precision_at_r"
PRECISION_AT_1 = "precision_at_1"
METRIC_ALIASES = {
    "map_at_r": MAP_AT_R,
    MAP_AT_R: MAP_AT_R,
    "best_valid_mean_average_precision_at_r": MAP_AT_R,
    PRECISION_AT_1: PRECISION_AT_1,
    "best_valid_precision_at_1": PRECISION_AT_1,
}
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
INFO_VALIDATION_PATTERN = re.compile(
    rf"valid(?:\s*-\s*epoch\s+(-?\d+))?\s*:\s*"
    rf"Precision@1\s*=\s*({FLOAT_PATTERN})\s*,\s*"
    rf"MAP@R\s*=\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
INFO_SELECTION_METRIC_PATTERN = re.compile(
    r"Model selection metric:\s*([A-Za-z0-9_]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationPoint:
    epoch: int
    value: float


@dataclass(frozen=True)
class PatienceAnalysis:
    source: Path
    metric: str
    warmup_epochs: int
    validation_checks: int
    best_epoch: int
    best_value: float
    minimum_patience: int
    largest_recovered_streak: int
    recovered_streak_epochs: tuple[int, ...]
    recovery_epoch: int | None
    trailing_non_improvements: int
    configured_patience: int | None
    early_stopping_epoch: int | None
    trace_status: str
    rounded_info_values: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the minimum early-stopping patience needed to reach the best "
            "recorded validation checkpoint across complete log directory trees."
        )
    )
    parser.add_argument(
        "log_roots",
        nargs="+",
        type=Path,
        help=(
            "One or more log directories, such as logs/iter_test. Every nested "
            "run is searched recursively; metrics.csv is preferred when a run "
            "also contains info.log."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("auto", "map_at_r", MAP_AT_R, PRECISION_AT_1),
        default="auto",
        help=(
            "Selection metric. 'auto' reads run_config.json or info.log and "
            f"falls back to {MAP_AT_R}."
        ),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Override SSL warmup for every input (default: read run_config.json, else 0).",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress the per-run report and only print the aggregate values.",
    )
    output_mode.add_argument(
        "--non-verbose",
        action="store_true",
        help=(
            "Print only run directories whose recorded validation trace reaches "
            "the patience configured in run_config.json; suppress the summary."
        ),
    )
    return parser.parse_args()


def normalize_metric(metric: str) -> str:
    try:
        return METRIC_ALIASES[metric.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported selection metric: {metric!r}") from exc


def discover_sources(log_roots: Iterable[Path]) -> list[Path]:
    """Return one preferred metric source per run below the supplied log roots."""

    by_directory: dict[Path, Path] = {}
    for requested_root in log_roots:
        root = requested_root.expanduser()
        if not root.exists():
            raise ValueError(f"Log root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(
                f"Expected a log directory, got file: {root}. "
                "Pass the directory containing this run instead."
            )

        candidates = [
            *root.rglob("metrics.csv"),
            *root.rglob("info.log"),
        ]

        for candidate in candidates:
            directory = candidate.parent.resolve()
            current = by_directory.get(directory)
            if current is None or (
                candidate.name == "metrics.csv" and current.name == "info.log"
            ):
                by_directory[directory] = candidate

    return sorted(by_directory.values(), key=lambda item: str(item))


def load_run_config(source: Path) -> dict[str, Any]:
    config_path = source.parent / "run_config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return config


def config_args(config: dict[str, Any]) -> dict[str, Any]:
    args = config.get("args", {})
    return args if isinstance(args, dict) else {}


def info_log_path(source: Path) -> Path | None:
    path = source if source.name == "info.log" else source.parent / "info.log"
    return path if path.is_file() else None


def read_info_log(source: Path) -> str:
    path = info_log_path(source)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def resolve_metric(
    requested_metric: str,
    config: dict[str, Any],
    info_text: str,
) -> str:
    if requested_metric != "auto":
        return normalize_metric(requested_metric)

    configured_metric = config_args(config).get("selection_metric")
    if isinstance(configured_metric, str):
        return normalize_metric(configured_metric)

    match = INFO_SELECTION_METRIC_PATTERN.search(info_text)
    if match:
        return normalize_metric(match.group(1))
    return MAP_AT_R


def resolve_warmup_epochs(
    override: int | None,
    config: dict[str, Any],
) -> int:
    if override is not None:
        warmup_epochs = override
    else:
        args = config_args(config)
        ssl_config = args.get("ssl_config_resolved")
        if not isinstance(ssl_config, dict):
            ssl_config = config.get("ssl_config", {})
        if not isinstance(ssl_config, dict):
            ssl_config = {}
        warmup_epochs = ssl_config.get("warmup_epochs", 0)

    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
        raise ValueError(f"warmup_epochs must be an integer, got {warmup_epochs!r}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
    return warmup_epochs


def configured_patience(config: dict[str, Any]) -> int | None:
    value = config_args(config).get("patience")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def load_csv_points(path: Path, metric: str) -> list[ValidationPoint]:
    points: list[ValidationPoint] = []
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            required_columns = {"epoch", "split", metric}
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing column(s): {', '.join(sorted(missing))}"
                )

            for line_number, row in enumerate(reader, start=2):
                if (row.get("split") or "").strip().lower() != "valid":
                    continue
                raw_epoch = (row.get("epoch") or "").strip()
                raw_value = (row.get(metric) or "").strip()
                if not raw_epoch or not raw_value:
                    continue
                try:
                    epoch = int(raw_epoch)
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number} has an invalid epoch or {metric}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}:{line_number} has non-finite {metric}={raw_value!r}"
                    )
                points.append(ValidationPoint(epoch=epoch, value=value))
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return points


def load_info_points(
    path: Path,
    metric: str,
    info_text: str,
) -> list[ValidationPoint]:
    points: list[ValidationPoint] = []
    metric_group = 2 if metric == PRECISION_AT_1 else 3
    for match in INFO_VALIDATION_PATTERN.finditer(info_text):
        epoch = int(match.group(1)) if match.group(1) is not None else -1
        # info.log displays percentages, while metrics.csv stores fractions.
        value = float(match.group(metric_group)) / 100.0
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite validation value")
        points.append(ValidationPoint(epoch=epoch, value=value))
    return points


def validate_points(path: Path, points: list[ValidationPoint]) -> None:
    if not points:
        raise ValueError(f"No usable validation rows found in {path}")
    for previous, current in zip(points, points[1:]):
        if current.epoch <= previous.epoch:
            raise ValueError(
                f"Validation epochs in {path} are not strictly increasing: "
                f"{previous.epoch}, {current.epoch}"
            )


def classify_trace(info_text: str) -> str:
    if "Traceback (most recent call last):" in info_text or "KeyboardInterrupt" in info_text:
        return "interrupted"
    if (
        "Skipping test evaluation for this run" in info_text
        or "Backbone cache stats:" in info_text
    ):
        return "completed"
    return "unknown"


def analyze_points(
    source: Path,
    points: list[ValidationPoint],
    metric: str,
    warmup_epochs: int,
    recorded_patience: int | None,
    trace_status: str,
    rounded_info_values: bool,
) -> PatienceAnalysis:
    """Find the smallest patience that reaches the trace's strict maximum."""

    validate_points(source, points)
    best_value = points[0].value
    best_epoch = points[0].epoch
    non_improving_epochs: list[int] = []
    minimum_patience = 1
    largest_recovered_streak = 0
    recovered_streak_epochs: tuple[int, ...] = ()
    recovery_epoch: int | None = None
    early_stopping_epoch: int | None = None

    for point in points[1:]:
        if point.value > best_value:
            recovered_count = len(non_improving_epochs)
            required_here = recovered_count + 1
            if required_here > minimum_patience:
                minimum_patience = required_here
                largest_recovered_streak = recovered_count
                recovered_streak_epochs = tuple(non_improving_epochs)
                recovery_epoch = point.epoch
            best_value = point.value
            best_epoch = point.epoch
            non_improving_epochs.clear()
        elif point.epoch >= warmup_epochs:
            non_improving_epochs.append(point.epoch)
            if (
                recorded_patience is not None
                and early_stopping_epoch is None
                and len(non_improving_epochs) >= recorded_patience
            ):
                early_stopping_epoch = point.epoch

    return PatienceAnalysis(
        source=source,
        metric=metric,
        warmup_epochs=warmup_epochs,
        validation_checks=len(points),
        best_epoch=best_epoch,
        best_value=best_value,
        minimum_patience=minimum_patience,
        largest_recovered_streak=largest_recovered_streak,
        recovered_streak_epochs=recovered_streak_epochs,
        recovery_epoch=recovery_epoch,
        trailing_non_improvements=len(non_improving_epochs),
        configured_patience=recorded_patience,
        early_stopping_epoch=early_stopping_epoch,
        trace_status=trace_status,
        rounded_info_values=rounded_info_values,
    )


def analyze_source(
    source: Path,
    requested_metric: str,
    warmup_override: int | None,
) -> PatienceAnalysis:
    config = load_run_config(source)
    info_text = read_info_log(source)
    metric = resolve_metric(requested_metric, config, info_text)
    warmup_epochs = resolve_warmup_epochs(warmup_override, config)

    if source.name == "metrics.csv":
        points = load_csv_points(source, metric)
        rounded_info_values = False
    else:
        points = load_info_points(source, metric, info_text)
        rounded_info_values = True

    return analyze_points(
        source=source,
        points=points,
        metric=metric,
        warmup_epochs=warmup_epochs,
        recorded_patience=configured_patience(config),
        trace_status=classify_trace(info_text),
        rounded_info_values=rounded_info_values,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def format_epoch_list(epochs: tuple[int, ...]) -> str:
    if not epochs:
        return "none"
    if len(epochs) == 1:
        return str(epochs[0])
    if epochs == tuple(range(epochs[0], epochs[-1] + 1)):
        return f"{epochs[0]}..{epochs[-1]}"
    return ",".join(str(epoch) for epoch in epochs)


def print_analysis(analysis: PatienceAnalysis) -> None:
    print(display_path(analysis.source))
    print(
        f"  metric={analysis.metric}, checks={analysis.validation_checks}, "
        f"warmup_epochs={analysis.warmup_epochs}, trace={analysis.trace_status}"
    )
    print(
        f"  best: epoch={analysis.best_epoch}, value={analysis.best_value:.10g}"
    )
    print(
        "  observed sufficient patience: "
        f">= {analysis.minimum_patience}"
    )
    if analysis.largest_recovered_streak:
        print(
            "  limiting recovered streak: "
            f"{analysis.largest_recovered_streak} non-improvement(s) at epoch(s) "
            f"{format_epoch_list(analysis.recovered_streak_epochs)}, "
            f"then improvement at epoch {analysis.recovery_epoch}"
        )
    else:
        print("  limiting recovered streak: none")
    print(
        "  non-improvements after recorded best: "
        f"{analysis.trailing_non_improvements}"
    )
    if analysis.configured_patience is not None:
        verdict = (
            "sufficient for the recorded best"
            if analysis.configured_patience >= analysis.minimum_patience
            else "too small for the recorded best"
        )
        print(
            f"  configured patience: {analysis.configured_patience} ({verdict})"
        )
    if analysis.rounded_info_values:
        print(
            "  note: info.log values are rounded; use metrics.csv for an exact result"
        )
    if analysis.trace_status != "completed":
        print(
            "  note: this only covers recorded epochs; unseen later improvements "
            "cannot be assessed"
        )


def print_summary(analyses: list[PatienceAnalysis]) -> None:
    minimum_values = [analysis.minimum_patience for analysis in analyses]
    mean_minimum = statistics.fmean(minimum_values)
    ceiling_mean = math.ceil(mean_minimum)
    shared_safe_patience = max(minimum_values)

    print("Summary")
    print(f"  analyzed runs: {len(analyses)}")
    print(
        "  ceiling(mean per-run minimum): "
        f"{ceiling_mean} (mean={mean_minimum:.3f})"
    )
    print(
        "  sufficient for every recorded trace: "
        f">= {shared_safe_patience}"
    )
    if ceiling_mean < shared_safe_patience:
        print(
            "  note: the ceiling mean is not sufficient for every run; "
            "use the shared value above when one patience must cover all traces"
        )


def main() -> int:
    args = parse_args()
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        print("--warmup-epochs must be non-negative", file=sys.stderr)
        return 2

    try:
        sources = discover_sources(args.log_roots)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not sources:
        print(
            "error: no nested runs with metrics.csv or info.log found",
            file=sys.stderr,
        )
        return 2

    analyses: list[PatienceAnalysis] = []
    failures = 0
    for source in sources:
        try:
            analysis = analyze_source(
                source,
                requested_metric=args.metric,
                warmup_override=args.warmup_epochs,
            )
        except ValueError as exc:
            failures += 1
            print(f"error: {exc}", file=sys.stderr)
            continue
        analyses.append(analysis)
        if args.non_verbose:
            if analysis.early_stopping_epoch is not None:
                print(display_path(analysis.source.parent))
        elif not args.summary_only:
            if len(analyses) > 1:
                print()
            print_analysis(analysis)

    if not analyses:
        return 2
    if not args.non_verbose:
        if not args.summary_only:
            print()
        print_summary(analyses)
    if failures:
        print(f"warning: skipped {failures} unreadable run(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
