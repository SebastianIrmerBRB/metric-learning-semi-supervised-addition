"""Compare CPU and GPU execution of the SSL methods, end to end on CIFAR100.

Every condition is one real ``main.py`` process on the same fold, the same
frozen-feature cache, and the same number of epochs; only the SSL placement
changes. The training device stays on CPU so the measured difference is the SSL
work itself: embedding extraction, FAISS kNN search, affinity/graph
construction, and the label-spreading solve. Hoffer is the exception and says so
in its condition names, because its regularizer runs per batch on the training
device and never touches ``ssl_device``.

Iscen additionally sweeps ``linear_solver``. ``cg`` (scipy) and ``cholmod`` are
CPU solvers whatever ``ssl_device`` says, so a GPU condition that keeps ``cg``
only moves the kNN search; ``cupy_cg`` and ``torch_block_cg`` move the solve as
well.

Each run's ``info.log`` already carries the production timing records, so the
report separates SSL seconds from total wall clock without extra instrumentation
and confirms which FAISS backend actually ran.

Examples::

    python scripts/bench_ssl_devices.py --epochs 3
    python scripts/bench_ssl_devices.py --methods iscen --epochs 5
    python scripts/bench_ssl_devices.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "configs/test_bed/cifar.json"
METHOD_CONFIGS = {
    "iscen": PROJECT_ROOT / "configs/ssl_methods/iscen.json",
    "lrml": PROJECT_ROOT / "configs/ssl_methods/lrml.json",
    "hoffer": PROJECT_ROOT / "configs/ssl_methods/hoffer.json",
}
# The launcher gives every child four BLAS/OMP threads, so the CPU conditions
# must too: FAISS and scipy are the CPU-side cost being measured here.
DEFAULT_THREADS = 4
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

TIMING_PATTERN = re.compile(
    r"SSL timing \| operation='(?P<operation>[^']+)' seconds=(?P<seconds>[0-9.]+)(?P<details>.*)"
)
BACKEND_PATTERN = re.compile(r"backend='(?P<backend>[^']+)'")
SLRML_GRAPH_PATTERN = re.compile(
    r"timing_embeddings=(?P<embeddings>[0-9.]+)s, "
    r"timing_graph=(?P<graph>[0-9.]+)s, "
    r"timing_loader=(?P<loader>[0-9.]+)s, "
    r"timing_total=(?P<total>[0-9.]+)s"
)
BATCH_TIMING_PATTERN = re.compile(
    r"Batch timing interval epoch=(?P<epoch>\d+) batch=\S+ last_batches=\d+ "
    r"phase=(?P<phase>[^:]+): (?P<timings>.*)"
)
# The top-level timed calls that cover one method's whole out-of-batch update.
# Hoffer has none: its regularizer runs per batch on the training device.
SSL_TOTAL_OPERATIONS = {
    "iscen": ("iscen_label_spreading",),
    "lrml": ("build_lrml_knn_graph", "build_slrml_graph"),
    "hoffer": (),
}


@dataclass(frozen=True)
class Condition:
    """One measured configuration of a single SSL method."""

    method: str
    name: str
    device: str
    ssl_device: str
    linear_solver: str | None = None
    note: str = ""


@dataclass
class RunResult:
    """Everything one condition produced."""

    condition: Condition
    returncode: int
    wall_clock_seconds: float
    log_dir: Path | None = None
    epochs_seen: int = 0
    ssl_updates: int = 0
    ssl_seconds: float = 0.0
    operations: dict[str, dict[str, float]] = field(default_factory=dict)
    faiss_backends: tuple[str, ...] = ()
    batch_timings: dict[str, float] = field(default_factory=dict)
    error: str = ""


def make_conditions(methods: list[str]) -> list[Condition]:
    """Return the measured matrix, skipping methods that were not requested."""

    conditions: list[Condition] = []
    if "iscen" in methods:
        conditions.extend(
            [
                Condition("iscen", "cpu/cg", "cpu", "cpu", "cg", "scipy solve, CPU FAISS"),
                Condition(
                    "iscen",
                    "cpu/torch_block_cg",
                    "cpu",
                    "cpu",
                    "torch_block_cg",
                    "torch solve on CPU, CPU FAISS",
                ),
                Condition("iscen", "cuda/cg", "cpu", "cuda", "cg", "GPU FAISS, scipy solve on CPU"),
                Condition(
                    "iscen",
                    "cuda/torch_block_cg",
                    "cpu",
                    "cuda",
                    "torch_block_cg",
                    "GPU FAISS and GPU solve",
                ),
                Condition(
                    "iscen",
                    "cuda/cupy_cg",
                    "cpu",
                    "cuda",
                    "cupy_cg",
                    "GPU FAISS and CuPy solve",
                ),
            ]
        )
    if "lrml" in methods:
        conditions.extend(
            [
                Condition("lrml", "cpu", "cpu", "cpu", None, "CPU graph rebuild"),
                Condition("lrml", "cuda", "cpu", "cuda", None, "GPU graph rebuild"),
            ]
        )
    if "hoffer" in methods:
        conditions.extend(
            [
                Condition("hoffer", "cpu", "cpu", "cpu", None, "entropy term on CPU"),
                Condition(
                    "hoffer",
                    "cuda-ssl-only",
                    "cpu",
                    "cuda",
                    None,
                    "ssl_device only; the entropy term stays on the CPU training device",
                ),
                Condition(
                    "hoffer",
                    "cuda",
                    "cuda",
                    "cuda",
                    None,
                    "training device moved too, which is what this method responds to",
                ),
            ]
        )
    return conditions


def make_ssl_config(condition: Condition) -> dict:
    """Return the method config for one condition, with warmup removed."""

    values = json.loads(METHOD_CONFIGS[condition.method].read_text(encoding="utf-8"))
    # Warmup epochs train without the SSL machinery, which would leave a short
    # benchmark measuring nothing.
    values["warmup_epochs"] = 0
    if condition.linear_solver is not None:
        values.setdefault("method_params", {})["linear_solver"] = condition.linear_solver
    return values


def make_experiment_config(
    base_values: dict,
    condition: Condition,
    ssl_config_path: Path,
    save_dir: str,
    epochs: int,
) -> dict:
    """Return a short single-fold SSL run of one method."""

    values = copy.deepcopy(base_values)
    values.update(
        {
            "save_dir": save_dir,
            "mode": "ssl",
            "ssl_config": str(ssl_config_path),
            "epochs": epochs,
            # One fold, no HPO, no test pass: the SSL update is the subject.
            "cv_k": 1,
            "hparam_config": None,
            "final_test_after_hpo": False,
            # Early stopping must not end a run before its epoch budget.
            "patience": epochs + 1,
        }
    )
    return values


def build_command(experiment_config: Path, condition: Condition, args) -> list[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "main.py"),
        "--experiment-config",
        str(experiment_config),
        "--device",
        condition.device,
        "--ssl-device",
        condition.ssl_device,
    ]
    if args.batch_timing:
        command.extend(
            [
                "--debug_batch_timing",
                "--debug_batch_timing_interval",
                str(args.batch_timing_interval),
            ]
        )
    return command


def child_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in THREAD_ENV_VARS:
        environment[name] = str(threads)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def find_info_log(save_dir: str) -> Path | None:
    """Return the newest run log written under one benchmark save directory."""

    root = PROJECT_ROOT / "logs" / save_dir
    if not root.is_dir():
        return None
    logs = sorted(root.rglob("info.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def parse_info_log(path: Path, method: str) -> dict:
    """Aggregate the production timing records one run already wrote."""

    operations: dict[str, dict[str, float]] = {}
    backends: list[str] = []
    batch_totals: dict[str, list[float]] = {}
    epochs_seen = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "pseudo-labels for epoch" in line or "graph embeddings - epoch" in line:
            epochs_seen += 1
        timing = TIMING_PATTERN.search(line)
        if timing is not None:
            operation = timing.group("operation")
            seconds = float(timing.group("seconds"))
            entry = operations.setdefault(operation, {"calls": 0.0, "seconds": 0.0})
            entry["calls"] += 1
            entry["seconds"] += seconds
            backend = BACKEND_PATTERN.search(timing.group("details"))
            if backend is not None:
                backends.append(backend.group("backend"))
            continue
        graph = SLRML_GRAPH_PATTERN.search(line)
        if graph is not None:
            for name in ("embeddings", "graph", "loader", "total"):
                entry = operations.setdefault(
                    f"slrml_{name}", {"calls": 0.0, "seconds": 0.0}
                )
                entry["calls"] += 1
                entry["seconds"] += float(graph.group(name))
            continue
        batch = BATCH_TIMING_PATTERN.search(line)
        if batch is not None:
            for item in batch.group("timings").split(", "):
                name, _, value = item.partition("=")
                if value.endswith("s"):
                    batch_totals.setdefault(name.strip(), []).append(float(value[:-1]))

    ssl_seconds = sum(
        operations.get(name, {}).get("seconds", 0.0)
        for name in SSL_TOTAL_OPERATIONS[method]
    )
    # Each top-level call is one out-of-batch update, which is the unit the
    # per-update column divides by. Falling back to the epoch markers keeps a
    # method without such a call from reporting a nonsense rate.
    ssl_updates = int(
        sum(
            operations.get(name, {}).get("calls", 0.0)
            for name in SSL_TOTAL_OPERATIONS[method]
        )
    )
    return {
        "operations": operations,
        "faiss_backends": tuple(sorted(set(backends))),
        "batch_timings": {
            name: sum(values) / len(values) for name, values in batch_totals.items()
        },
        "epochs_seen": epochs_seen,
        "ssl_updates": ssl_updates,
        "ssl_seconds": ssl_seconds,
    }


def run_condition(condition: Condition, base_values: dict, args, session_dir: Path) -> RunResult:
    """Run one condition to completion and collect its timings."""

    label = f"{condition.method}:{condition.name}"
    save_dir = f"{args.save_dir_root}/{condition.method}_{condition.name.replace('/', '_')}"
    config_dir = session_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{condition.method}_{condition.name.replace('/', '_')}"
    ssl_config_path = config_dir / f"{stem}_ssl.json"
    ssl_config_path.write_text(
        json.dumps(make_ssl_config(condition), indent=2), encoding="utf-8"
    )
    experiment_config_path = config_dir / f"{stem}.json"
    experiment_config_path.write_text(
        json.dumps(
            make_experiment_config(
                base_values, condition, ssl_config_path, save_dir, args.epochs
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    command = build_command(experiment_config_path, condition, args)
    if args.dry_run:
        print(f"DRY RUN  {label}: {' '.join(command)}")
        return RunResult(condition=condition, returncode=0, wall_clock_seconds=0.0)

    stdout_path = session_dir / f"{stem}.out"
    print(f"RUN      {label} ({condition.note})", flush=True)
    started_at = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment(args.threads),
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_clock = time.perf_counter() - started_at

    result = RunResult(
        condition=condition,
        returncode=process.returncode,
        wall_clock_seconds=wall_clock,
    )
    if process.returncode != 0:
        tail = stdout_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        result.error = "\n".join(tail[-8:])
        print(f"FAILED   {label} (exit {process.returncode}); see {stdout_path}", flush=True)
        return result

    info_log = find_info_log(save_dir)
    if info_log is None:
        result.error = f"no info.log under logs/{save_dir}"
        return result
    parsed = parse_info_log(info_log, condition.method)
    result.log_dir = info_log.parent
    result.epochs_seen = parsed["epochs_seen"]
    result.ssl_updates = parsed["ssl_updates"]
    result.ssl_seconds = parsed["ssl_seconds"]
    result.operations = parsed["operations"]
    result.faiss_backends = parsed["faiss_backends"]
    result.batch_timings = parsed["batch_timings"]
    print(
        f"DONE     {label}: wall={wall_clock:.1f}s "
        f"ssl={result.ssl_seconds:.1f}s faiss={'/'.join(result.faiss_backends) or 'n/a'}",
        flush=True,
    )
    return result


def format_report(results: list[RunResult], epochs: int) -> str:
    """Return a per-method table with the CPU baseline as the reference."""

    lines = []
    header = (
        f"{'method':8} {'condition':22} {'wall s':>9} {'ssl s':>9} "
        f"{'s/update':>10} {'vs cpu':>8}  faiss"
    )
    for method in ("iscen", "lrml", "hoffer"):
        method_results = [item for item in results if item.condition.method == method]
        if not method_results:
            continue
        baseline = next(
            (
                item
                for item in method_results
                if item.condition.ssl_device == "cpu" and item.condition.device == "cpu"
            ),
            None,
        )
        lines.append("")
        lines.append(f"{method.upper()}  (epoch budget {epochs})")
        lines.append(header)
        for item in method_results:
            if item.returncode != 0:
                lines.append(
                    f"{method:8} {item.condition.name:22} {'FAILED':>9} "
                    f"{'':>9} {'':>10} {'':>8}  {item.error.splitlines()[-1][:60] if item.error else ''}"
                )
                continue
            per_epoch = item.ssl_seconds / item.ssl_updates if item.ssl_updates else 0.0
            ratio = ""
            if baseline is not None and baseline.wall_clock_seconds > 0 and item is not baseline:
                ratio = f"{baseline.wall_clock_seconds / item.wall_clock_seconds:.2f}x"
            lines.append(
                f"{method:8} {item.condition.name:22} "
                f"{item.wall_clock_seconds:9.1f} {item.ssl_seconds:9.1f} "
                f"{per_epoch:10.2f} {ratio:>8}  {'/'.join(item.faiss_backends) or 'n/a'}"
            )
            if item.batch_timings:
                top = sorted(item.batch_timings.items(), key=lambda pair: -pair[1])[:4]
                detail = ", ".join(f"{name}={value:.4f}s" for name, value in top)
                lines.append(f"{'':8} {'  per-batch:':22} {detail}")
    lines.append("")
    lines.append("'vs cpu' > 1 means the condition finished faster than its CPU baseline.")
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="base experiment config supplying dataset, protocol, and backbone settings",
    )
    parser.add_argument(
        "--methods",
        default="iscen,lrml,hoffer",
        help="comma-separated subset of iscen,lrml,hoffer",
    )
    parser.add_argument("--epochs", type=int, default=3, help="epochs per condition")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="BLAS/OMP threads per child, matching launch_runs.py",
    )
    parser.add_argument(
        "--batch-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="log per-batch phase timings, which is where LRML and Hoffer spend their time",
    )
    parser.add_argument("--debug-batch-timing-interval", dest="batch_timing_interval", type=int, default=20)
    parser.add_argument(
        "--save-dir-root",
        dest="save_dir_root",
        default="bench_ssl_devices",
        help="logs/ subdirectory that receives the benchmark runs",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_CONFIGS))
    if unknown:
        parser.error(f"unknown method(s) {unknown}; expected {sorted(METHOD_CONFIGS)}")

    base_values = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    session_dir = (
        PROJECT_ROOT
        / "logs"
        / args.save_dir_root
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session: {session_dir}")
    print(f"Base config: {args.experiment_config}, epochs={args.epochs}, threads={args.threads}")

    results = [
        run_condition(condition, base_values, args, session_dir)
        for condition in make_conditions(methods)
    ]
    if args.dry_run:
        return 0

    report = format_report(results, args.epochs)
    print(report)
    (session_dir / "report.txt").write_text(report, encoding="utf-8")
    (session_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "method": item.condition.method,
                    "condition": item.condition.name,
                    "device": item.condition.device,
                    "ssl_device": item.condition.ssl_device,
                    "linear_solver": item.condition.linear_solver,
                    "returncode": item.returncode,
                    "wall_clock_seconds": item.wall_clock_seconds,
                    "ssl_seconds": item.ssl_seconds,
                    "epochs_seen": item.epochs_seen,
                    "ssl_updates": item.ssl_updates,
                    "faiss_backends": list(item.faiss_backends),
                    "operations": item.operations,
                    "batch_timings": item.batch_timings,
                    "log_dir": None if item.log_dir is None else str(item.log_dir),
                    "error": item.error,
                }
                for item in results
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {session_dir / 'results.json'}")
    return 1 if any(item.returncode != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
