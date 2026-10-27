#!/usr/bin/env python3
"""Benchmark complete HPO run processes sharing one GPU or the host CPUs.

Unlike a kernel or training-loop microbenchmark, this script launches the real
``main.py`` entry point once per worker.  Every worker owns an isolated Optuna
study containing one fixed ArcFace trial, so the timed region includes process
startup, dataset/split construction, frozen-feature loading, all CV folds,
validation, logging, checkpoint handling, Optuna/SQLite writes, and teardown.

The benchmark fixes the number of epochs and disables early stopping so every
process performs the same work.  It reports whether N simultaneous processes
finish sooner than N sequential single-process runs, as well as aggregate run
throughput and its incremental gain over the preceding concurrency level.

``--device cuda`` (the default) shares one GPU between the workers and samples
utilization and memory with ``nvidia-smi``.  ``--device cpu`` hides every GPU
from the workers and samples host-wide CPU utilization and used memory from
``/proc`` instead.  CPU runs are far slower per process, so shrink ``--epochs``
and ``--concurrency`` accordingly, and use ``--cpu-threads`` to keep the BLAS
thread pools of concurrent workers from oversubscribing the machine.

CUDA runs default to ``--evaluation-embedding-residency cpu``. Validation still
uses FAISS-GPU, but each worker keeps projected embeddings in host memory and
releases the FAISS index after each evaluation. Use ``gpu`` to benchmark the
higher-memory, device-resident evaluation path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_EXPERIMENT_CONFIG = Path("configs/test_bed/cars.json")
DEFAULT_HPARAM_CONFIG = Path("configs/hpo/cars/supervised_pair.json")
PROC_STAT = Path("/proc/stat")
PROC_MEMINFO = Path("/proc/meminfo")
# main.py only sets these with setdefault, so an explicit value wins. Setting
# OMP_NUM_THREADS also steers torch's intra-op pool, which would otherwise size
# itself from the host's physical cores; METRIC_LEARNING_NUM_THREADS overrides both.
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class ResourceSamples:
    average_utilization_percent: float | None
    peak_memory_mib: int | None


@dataclass(frozen=True)
class ResourceLabels:
    """Names for the sampled columns, so each device reports what it measured."""

    utilization_column: str
    memory_column: str
    utilization_key: str
    memory_key: str


RESOURCE_LABELS = {
    "cuda": ResourceLabels(
        utilization_column="average_gpu_percent",
        memory_column="peak_gpu_memory_mib",
        utilization_key="average_gpu_utilization_percent",
        memory_key="peak_gpu_memory_mib",
    ),
    "cpu": ResourceLabels(
        utilization_column="average_cpu_percent",
        memory_column="peak_system_memory_mib",
        utilization_key="average_cpu_utilization_percent",
        memory_key="peak_system_memory_mib",
    ),
}


@dataclass(frozen=True)
class GroupResult:
    concurrency: int
    repeat: int
    wall_seconds: float
    process_seconds: list[float]
    resources: ResourceSamples

    @property
    def mean_process_seconds(self) -> float:
        return statistics.mean(self.process_seconds)

    @property
    def runs_per_hour(self) -> float:
        return 3600 * self.concurrency / self.wall_seconds


class ResourceMonitor:
    """Sample device utilization and used memory until the group finishes."""

    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None
        self._utilization = []
        self._memory_mib = []

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self._stop.set()
        self._thread.join(timeout=max(2.0, 2 * self.interval_seconds))

    def _sample_until_stopped(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def _sample(self) -> None:
        raise NotImplementedError

    def result(self) -> ResourceSamples:
        return ResourceSamples(
            average_utilization_percent=(
                statistics.mean(self._utilization) if self._utilization else None
            ),
            peak_memory_mib=max(self._memory_mib) if self._memory_mib else None,
        )


class GpuMonitor(ResourceMonitor):
    def __init__(self, gpu: str, interval_seconds: float):
        super().__init__(interval_seconds)
        self.gpu = gpu

    def _sample(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                self.gpu,
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        utilization, memory = result.stdout.strip().split(",", maxsplit=1)
        self._utilization.append(float(utilization.strip()))
        self._memory_mib.append(int(memory.strip()))


class CpuMonitor(ResourceMonitor):
    """Host-wide counterpart of the nvidia-smi sampler.

    Utilization is the busy share of all logical CPUs between two samples,
    scaled to 0-100 like ``utilization.gpu``.  Memory is host-wide used memory,
    matching the machine-wide reading of ``memory.used`` rather than the
    resident set of the benchmarked processes alone.
    """

    def __init__(self, interval_seconds: float):
        super().__init__(interval_seconds)
        self._previous_cpu_times = None

    def _sample(self) -> None:
        busy, total = read_cpu_times()
        previous = self._previous_cpu_times
        self._previous_cpu_times = (busy, total)
        if previous is not None and total > previous[1]:
            self._utilization.append(100 * (busy - previous[0]) / (total - previous[1]))
        self._memory_mib.append(read_used_memory_mib())


def read_cpu_times() -> tuple[int, int]:
    """Return cumulative (busy, total) jiffies across all logical CPUs."""

    with PROC_STAT.open(encoding="utf-8") as handle:
        fields = handle.readline().split()
    # user, nice, system, idle, iowait, irq, softirq, steal; the trailing guest
    # counters are already included in user/nice, so summing them double counts.
    if len(fields) < 9 or fields[0] != "cpu":
        raise ValueError("unexpected /proc/stat layout")
    values = [int(field) for field in fields[1:9]]
    total = sum(values)
    idle = values[3] + values[4]
    return total - idle, total


def read_used_memory_mib() -> int:
    wanted = ("MemTotal:", "MemAvailable:")
    kibibytes = {}
    with PROC_MEMINFO.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] in wanted:
                kibibytes[fields[0]] = int(fields[1])
                if len(kibibytes) == len(wanted):
                    break
    if len(kibibytes) != len(wanted):
        raise ValueError("unexpected /proc/meminfo layout")
    return (kibibytes["MemTotal:"] - kibibytes["MemAvailable:"]) // 1024


def make_monitor(args: argparse.Namespace) -> ResourceMonitor:
    if args.device == "cuda":
        return GpuMonitor(args.gpu, args.sample_interval)
    return CpuMonitor(args.sample_interval)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG
    )
    parser.add_argument("--hparam-config", type=Path, default=DEFAULT_HPARAM_CONFIG)
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=positive_int,
        default=list(range(1, 9)),
        help="simultaneous complete main.py processes to test",
    )
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=30,
        help="fixed epochs per CV fold; early stopping is disabled",
    )
    parser.add_argument("--cv-k", type=positive_int, default=4)
    parser.add_argument(
        "--warmup-epochs",
        type=non_negative_int,
        default=2,
        help="epochs for one untimed real-process warmup; 0 disables it",
    )
    parser.add_argument("--batch-sampler", default="32:4")
    parser.add_argument("--lr", type=positive_float, default=7.10536811996787e-6)
    parser.add_argument(
        "--classifier-lr", type=positive_float, default=4.780934055777077
    )
    parser.add_argument("--arc-scale", type=positive_float, default=41.008114816674876)
    parser.add_argument("--arc-margin", type=positive_float, default=47.55593300239236)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="device the worker processes train on; cpu hides every GPU from them",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help=(
            "physical GPU selector passed through CUDA_VISIBLE_DEVICES and "
            "nvidia-smi; ignored when --device is cpu"
        ),
    )
    parser.add_argument(
        "--evaluation-embedding-residency",
        choices=("cpu", "gpu"),
        default="cpu",
        help=(
            "projected validation/test embedding residency passed to each real "
            "worker; cpu keeps CUDA retrieval but enables the low-memory FAISS path"
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=positive_int,
        default=None,
        metavar="THREADS",
        help=(
            "BLAS/OpenMP threads per worker process, which torch's intra-op pool "
            "also follows; the default lets each worker size itself from the host's "
            "physical cores, which oversubscribes at high concurrency"
        ),
    )
    parser.add_argument(
        "--sample-interval",
        "--gpu-sample-interval",
        dest="sample_interval",
        type=positive_float,
        default=0.5,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="keep configs, process logs, and trial artifacts here; default uses a temporary directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path for the machine-readable result summary",
    )
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def make_configs(args: argparse.Namespace, root: Path, *, epochs: int) -> tuple[Path, Path]:
    experiment = read_json(resolve_workspace_path(args.experiment_config))
    experiment.update(
        {
            "save_dir": str(root / "unused_default_save_dir"),
            "epochs": epochs,
            "patience": epochs + 1,
            "cv_k": args.cv_k,
            "final_test_after_hpo": False,
            "batch_loss_log_interval": 0,
            "log_batch_diagnostics": False,
            "ssl_gradient_contribution_log_interval": 0,
            "debug_batch_timing": False,
            "num_workers": 0,
            "evaluation_embedding_residency": args.evaluation_embedding_residency,
        }
    )
    experiment_path = root / f"experiment_epochs_{epochs}.json"
    write_json(experiment_path, experiment)

    hparams = read_json(resolve_workspace_path(args.hparam_config))
    hparams.update(
        {
            "n_trials": 1,
            "n_jobs": 1,
            "timeout": None,
            "study_name": "actual_run_concurrency",
            "study_dir": None,
            "storage": None,
            "load_if_exists": False,
            "sampler": "grid",
            "sampler_params": {},
            "tpe_startup_trials": None,
            "pruner": "none",
            "pruner_params": {},
            "retry_failed_trials": False,
            "spaces": {
                "lr": {"type": "categorical", "choices": [args.lr]},
                "classifier_lr": {
                    "type": "categorical",
                    "choices": [args.classifier_lr],
                },
                "loss.ArcFaceLoss.scale": {
                    "type": "categorical",
                    "choices": [args.arc_scale],
                },
                "loss.ArcFaceLoss.margin": {
                    "type": "categorical",
                    "choices": [args.arc_margin],
                },
                "batch_sampler": {
                    "type": "categorical",
                    "choices": [args.batch_sampler],
                },
            },
        }
    )
    hparam_path = root / "hparams_one_fixed_trial.json"
    write_json(hparam_path, hparams)
    return experiment_path, hparam_path


def build_command(
    args: argparse.Namespace,
    experiment_path: Path,
    hparam_path: Path,
    save_dir: Path,
) -> list[str]:
    command = [
        str(resolve_workspace_path(args.python)),
        str(WORKSPACE_ROOT / "main.py"),
        "--experiment-config",
        str(experiment_path),
        "--hparam_config",
        str(hparam_path),
        "--save_dir",
        str(save_dir),
        "--loss_miner_grid",
        "ArcFaceLoss:no_miner",
        "--device",
        args.device,
        "--evaluation-embedding-residency",
        args.evaluation_embedding_residency,
    ]
    if args.cpu_threads is not None:
        # THREAD_ENVIRONMENT_VARIABLES above caps the child; this is what asks
        # it to actually spend the cap. The child treats the environment as a
        # ceiling rather than a request, because every numeric wheel it loads
        # sizes a separate pool from those variables.
        command += ["--num_threads", str(args.cpu_threads)]
    return command


def log_tail(path: Path, line_count: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<process log unavailable>"
    return "\n".join(lines[-line_count:])


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 15
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def run_group(
    args: argparse.Namespace,
    root: Path,
    experiment_path: Path,
    hparam_path: Path,
    concurrency: int,
    repeat: int,
    *,
    label: str = "measure",
) -> GroupResult:
    group_dir = root / f"{label}_n{concurrency:02d}_r{repeat:02d}"
    group_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_handles = []
    log_paths = []
    start_times = []
    finish_times: list[float | None] = [None] * concurrency
    environment = os.environ.copy()
    # An empty selector leaves torch.cuda unavailable, so a cpu group cannot
    # silently fall back to the GPU.
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "cuda" else ""
    environment["PYTHONUNBUFFERED"] = "1"
    if args.cpu_threads is not None:
        for variable in THREAD_ENVIRONMENT_VARIABLES:
            environment[variable] = str(args.cpu_threads)
    group_start = time.monotonic()
    try:
        with make_monitor(args) as monitor:
            for worker_id in range(concurrency):
                run_dir = group_dir / f"run_{worker_id:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "process.log"
                log_handle = log_path.open("w", encoding="utf-8")
                process_start = time.monotonic()
                process = subprocess.Popen(
                    build_command(args, experiment_path, hparam_path, run_dir),
                    cwd=WORKSPACE_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append(process)
                log_handles.append(log_handle)
                log_paths.append(log_path)
                start_times.append(process_start)

            unfinished = set(range(concurrency))
            while unfinished:
                now = time.monotonic()
                for worker_id in tuple(unfinished):
                    if processes[worker_id].poll() is not None:
                        finish_times[worker_id] = now
                        unfinished.remove(worker_id)
                if unfinished:
                    time.sleep(0.05)
            group_end = time.monotonic()
        resource_result = monitor.result()
    except BaseException:
        terminate_processes(processes)
        raise
    finally:
        for handle in log_handles:
            handle.close()

    failures = [
        worker_id
        for worker_id, process in enumerate(processes)
        if process.returncode != 0
    ]
    if failures:
        details = []
        for worker_id in failures:
            details.append(
                f"worker {worker_id} exited {processes[worker_id].returncode}:\n"
                f"{log_tail(log_paths[worker_id])}"
            )
        raise RuntimeError("\n\n".join(details))

    return GroupResult(
        concurrency=concurrency,
        repeat=repeat,
        wall_seconds=group_end - group_start,
        process_seconds=[
            finish_times[index] - start_times[index]  # type: ignore[operator]
            for index in range(concurrency)
        ],
        resources=resource_result,
    )


def median_group(results: list[GroupResult]) -> GroupResult:
    first = results[0]
    return GroupResult(
        concurrency=first.concurrency,
        repeat=-1,
        wall_seconds=statistics.median(result.wall_seconds for result in results),
        process_seconds=[
            statistics.median(result.mean_process_seconds for result in results)
        ]
        * first.concurrency,
        resources=ResourceSamples(
            average_utilization_percent=(
                statistics.median(
                    result.resources.average_utilization_percent
                    for result in results
                    if result.resources.average_utilization_percent is not None
                )
                if any(
                    result.resources.average_utilization_percent is not None
                    for result in results
                )
                else None
            ),
            peak_memory_mib=(
                max(
                    result.resources.peak_memory_mib
                    for result in results
                    if result.resources.peak_memory_mib is not None
                )
                if any(
                    result.resources.peak_memory_mib is not None for result in results
                )
                else None
            ),
        ),
    )


def format_optional(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "nan"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def run_benchmark(args: argparse.Namespace, root: Path) -> list[GroupResult]:
    experiment_path, hparam_path = make_configs(args, root, epochs=args.epochs)
    if args.warmup_epochs:
        warm_experiment_path, _ = make_configs(
            args, root / "warmup_configs", epochs=args.warmup_epochs
        )
        print(f"warming with one real run ({args.warmup_epochs} epochs/fold)...", flush=True)
        run_group(
            args,
            root,
            warm_experiment_path,
            hparam_path,
            1,
            0,
            label="warmup",
        )

    labels = RESOURCE_LABELS[args.device]
    medians = []
    for concurrency in args.concurrency:
        repeated = []
        for repeat in range(args.repeats):
            print(
                f"running {concurrency} simultaneous real trial process(es), "
                f"repeat {repeat + 1}/{args.repeats}...",
                flush=True,
            )
            result = run_group(
                args,
                root,
                experiment_path,
                hparam_path,
                concurrency,
                repeat,
            )
            repeated.append(result)
            print(
                f"  wall={result.wall_seconds:.2f}s, "
                f"mean_process={result.mean_process_seconds:.2f}s, "
                f"{labels.utilization_column}="
                f"{format_optional(result.resources.average_utilization_percent)}%, "
                f"{labels.memory_column}="
                f"{format_optional(result.resources.peak_memory_mib)} MiB",
                flush=True,
            )
        medians.append(median_group(repeated))
    return medians


def print_summary(results: list[GroupResult], labels: ResourceLabels) -> list[dict]:
    baseline = results[0]
    previous_throughput = None
    rows = []
    print("\nActual-process concurrency summary")
    print(
        "runs,group_wall_seconds,mean_run_seconds,runs_per_hour,"
        "throughput_speedup_vs_one,time_saved_vs_sequential_percent,"
        f"incremental_throughput_gain_percent,{labels.utilization_column},"
        f"{labels.memory_column}"
    )
    for result in results:
        speedup = result.runs_per_hour / baseline.runs_per_hour
        time_saved = 100 * (1 - result.wall_seconds / (result.concurrency * baseline.wall_seconds))
        incremental_gain = (
            None
            if previous_throughput is None
            else 100 * (result.runs_per_hour / previous_throughput - 1)
        )
        row = {
            "concurrency": result.concurrency,
            "group_wall_seconds": result.wall_seconds,
            "mean_run_seconds": result.mean_process_seconds,
            "runs_per_hour": result.runs_per_hour,
            "throughput_speedup_vs_one": speedup,
            "time_saved_vs_sequential_percent": time_saved,
            "incremental_throughput_gain_percent": incremental_gain,
            labels.utilization_key: result.resources.average_utilization_percent,
            labels.memory_key: result.resources.peak_memory_mib,
        }
        rows.append(row)
        print(
            f"{result.concurrency},{result.wall_seconds:.3f},"
            f"{result.mean_process_seconds:.3f},{result.runs_per_hour:.2f},"
            f"{speedup:.3f},{time_saved:.1f},"
            f"{format_optional(incremental_gain)},"
            f"{format_optional(result.resources.average_utilization_percent)},"
            f"{format_optional(result.resources.peak_memory_mib)}"
        )
        previous_throughput = result.runs_per_hour

    best = max(results, key=lambda result: result.runs_per_hour)
    print(
        f"highest measured completed-run throughput: {best.concurrency} simultaneous "
        f"runs ({best.runs_per_hour:.2f} runs/hour)"
    )
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency[0] != 1:
        raise ValueError("--concurrency must start with 1 for the sequential baseline")
    if args.concurrency != sorted(set(args.concurrency)):
        raise ValueError("--concurrency must be unique and sorted ascending")
    if not resolve_workspace_path(args.python).is_file():
        raise FileNotFoundError(resolve_workspace_path(args.python))
    if not resolve_workspace_path(args.experiment_config).is_file():
        raise FileNotFoundError(resolve_workspace_path(args.experiment_config))
    if not resolve_workspace_path(args.hparam_config).is_file():
        raise FileNotFoundError(resolve_workspace_path(args.hparam_config))


def main() -> None:
    args = parse_args()
    validate_args(args)
    print(f"python={resolve_workspace_path(args.python)}")
    print(f"workspace={WORKSPACE_ROOT}")
    print(f"device={args.device}")
    print(f"evaluation_embedding_residency={args.evaluation_embedding_residency}")
    if args.device == "cuda":
        print(f"gpu={args.gpu}")
    else:
        print(f"logical_cpus={os.cpu_count()}")
        if not PROC_STAT.exists():
            print("note: /proc is unavailable, so cpu sampling reports nan")
    if args.cpu_threads is not None:
        print(f"threads_per_process={args.cpu_threads}")
    print(f"batch_sampler={args.batch_sampler}")
    print(f"epochs_per_fold={args.epochs}")
    print(f"cv_folds={args.cv_k}")

    temporary = None
    if args.artifact_dir is None:
        temporary = tempfile.mkdtemp(prefix="actual_hpo_run_concurrency_")
        root = Path(temporary)
    else:
        root = resolve_workspace_path(args.artifact_dir)
        root.mkdir(parents=True, exist_ok=False)

    try:
        results = run_benchmark(args, root)
        rows = print_summary(results, RESOURCE_LABELS[args.device])
        if args.output_json is not None:
            output_path = resolve_workspace_path(args.output_json)
            write_json(
                output_path,
                {
                    "python": str(resolve_workspace_path(args.python)),
                    "device": args.device,
                    "evaluation_embedding_residency": args.evaluation_embedding_residency,
                    "gpu": args.gpu if args.device == "cuda" else None,
                    "cpu_threads_per_process": args.cpu_threads,
                    "batch_sampler": args.batch_sampler,
                    "epochs_per_fold": args.epochs,
                    "cv_folds": args.cv_k,
                    "repeats": args.repeats,
                    "results": rows,
                },
            )
            print(f"summary_json={output_path}")
        if args.artifact_dir is not None:
            print(f"artifacts={root}")
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
