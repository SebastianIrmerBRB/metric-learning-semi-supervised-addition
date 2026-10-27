#!/usr/bin/env python3
"""Benchmark frozen-feature logging overhead and single-GPU HPO concurrency.

The worker uses the project's real ``_EpochTrainer`` with the same small CUDA
workload as the cached Cars196 ArcFace path: pinned 768-dimensional backbone
features, a trainable 768 x 768 projection, BF16 autocast, 73 ArcFace classes,
and separate Adam optimizers for the projection and classifier weights.

Two measurements are reported:

* default-gated logging versus the previous every-batch loss logging behavior
  and opt-in full batch diagnostics, including process CPU time and real
  temporary TensorBoard/CSV writers; and
* aggregate training throughput for several independent spawned workers sharing
  one GPU, which approximates concurrent HPO trials after feature precompute.

Validation, dataset sampling, DINO feature extraction, and process startup are
deliberately outside the timed region. Use aggregate samples/second, rather than
individual-trial latency, when choosing a concurrency level.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import random
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch_metric_learning.losses import ArcFaceLoss

import training.engine as engine
import utils


FEATURE_DIM = 768
NUM_CLASSES = 73
DEFAULT_SAMPLED_EXAMPLES_PER_EPOCH = 5_984
WORKER_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class WorkerSpec:
    batch_size: int
    batches_per_epoch: int
    measured_epochs: int
    warmup_batches: int
    batch_loss_log_interval: int
    log_batch_diagnostics: bool
    reserve_gpu_mib: int
    seed: int


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    start_time: float
    end_time: float
    cpu_seconds: float
    steps: int
    samples: int
    gpu_name: str
    allocated_mib: float


@dataclass(frozen=True)
class GroupResult:
    concurrency: int
    wall_seconds: float
    cpu_seconds: float
    steps: int
    samples: int
    mean_trial_seconds: float
    gpu_name: str
    allocated_mib_per_worker: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.wall_seconds

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.wall_seconds

    @property
    def average_cpu_cores(self) -> float:
        return self.cpu_seconds / self.wall_seconds


class FrozenProjectionHead(nn.Module):
    """Projection-only stand-in for a DINO model receiving cached features."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURE_DIM, FEATURE_DIM)

    def project_features(self, features):
        return F.normalize(self.fc(features), p=2.0, dim=1)

    def forward(self, features):
        return self.project_features(features)


class SilentProgress:
    """Minimal tqdm replacement so terminal rendering is not benchmarked."""

    def __init__(self, iterable, total=None, **kwargs):
        del total, kwargs
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--batch-sizes", nargs="+", type=parse_positive_int, default=[32, 256])
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=parse_positive_int,
        default=list(range(1, 9)),
        help="numbers of simultaneous spawned training workers to measure",
    )
    parser.add_argument(
        "--measured-steps",
        type=parse_positive_int,
        default=600,
        help="minimum measured optimizer steps per worker and scenario",
    )
    parser.add_argument(
        "--warmup-steps",
        type=parse_non_negative_int,
        default=30,
        help="untimed warmup optimizer steps per worker",
    )
    parser.add_argument(
        "--sampled-examples-per-epoch",
        type=parse_positive_int,
        default=DEFAULT_SAMPLED_EXAMPLES_PER_EPOCH,
        help="used to reproduce the cached Cars196 batches-per-epoch counts",
    )
    parser.add_argument(
        "--cpu-repeats",
        type=parse_positive_int,
        default=4,
        help="single-worker repeats for the before/after CPU comparison",
    )
    parser.add_argument(
        "--concurrency-repeats",
        type=parse_positive_int,
        default=1,
        help="repeats for every batch-size/concurrency point",
    )
    parser.add_argument(
        "--reserve-gpu-mib",
        type=parse_non_negative_int,
        default=350,
        help="extra allocation per worker approximating the resident frozen DINO-B weights",
    )
    parser.add_argument(
        "--skip-cpu-comparison",
        action="store_true",
        help="run only the concurrency sweep",
    )
    parser.add_argument(
        "--skip-concurrency",
        action="store_true",
        help="run only the before/after CPU comparison",
    )
    return parser.parse_args()


def make_worker_spec(
    args: argparse.Namespace,
    batch_size: int,
    *,
    seed: int,
    batch_loss_log_interval: int = 0,
    log_batch_diagnostics: bool = False,
) -> WorkerSpec:
    batches_per_epoch = args.sampled_examples_per_epoch // batch_size
    if batches_per_epoch <= 0:
        raise ValueError(
            "sampled_examples_per_epoch must contain at least one complete batch: "
            f"{args.sampled_examples_per_epoch} < {batch_size}"
        )
    measured_epochs = math.ceil(args.measured_steps / batches_per_epoch)
    warmup_batches = min(args.warmup_steps, batches_per_epoch)
    return WorkerSpec(
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        measured_epochs=measured_epochs,
        warmup_batches=warmup_batches,
        batch_loss_log_interval=batch_loss_log_interval,
        log_batch_diagnostics=log_batch_diagnostics,
        reserve_gpu_mib=args.reserve_gpu_mib,
        seed=seed,
    )


def make_training_args(spec: WorkerSpec):
    return argparse.Namespace(
        device="cuda",
        debug_batch_timing=False,
        debug_batch_timing_interval=5,
        batch_loss_log_interval=spec.batch_loss_log_interval,
        log_batch_diagnostics=spec.log_batch_diagnostics,
        ssl_gradient_contribution_log_interval=0,
        lr_scheduler=engine.LR_SCHEDULER_NONE,
        lr_scheduler_params={},
        loss="ArcFaceLoss",
    )


def run_worker(
    worker_id: int,
    spec: WorkerSpec,
    start_event,
    ready_queue,
    result_queue,
    log_dir: str,
) -> None:
    """Initialize one independent CUDA trial, then await a shared start event."""

    ready_sent = False
    metrics = None
    try:
        logger.remove()
        engine.tqdm = SilentProgress
        torch.manual_seed(spec.seed + worker_id)
        torch.cuda.manual_seed_all(spec.seed + worker_id)
        device = torch.device("cuda")

        reserve = None
        if spec.reserve_gpu_mib:
            reserve = torch.empty(
                spec.reserve_gpu_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
            reserve.zero_()

        training_args = make_training_args(spec)
        ssl_config = argparse.Namespace(enabled=False, warmup_epochs=0, method="none")
        model = FrozenProjectionHead().to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = ArcFaceLoss(
            num_classes=NUM_CLASSES,
            embedding_size=FEATURE_DIM,
            scale=41.008114816674876,
            margin=58.40791066985647,
        ).to(device)
        criterion_optimizer = torch.optim.Adam(criterion.parameters(), lr=1e-3)
        objective = engine.TrainingLossComponents(
            criterion=criterion,
            is_classification=True,
            miner=None,
            optimizer=criterion_optimizer,
        )

        features = torch.randn(spec.batch_size, FEATURE_DIM, pin_memory=True)
        labels = (torch.arange(spec.batch_size) % NUM_CLASSES).pin_memory()
        batches = [(features, labels)] * spec.batches_per_epoch
        label_lookup = torch.arange(NUM_CLASSES, device=device)
        worker_log_dir = Path(log_dir)
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        metrics = utils.MetricsLogger(worker_log_dir, training_args)
        trainer = engine._EpochTrainer(
            args=training_args,
            ssl_config=ssl_config,
            model=model,
            model_optimizer=model_optimizer,
            main_objective=objective,
            warmup_objective=None,
            regularizer=None,
            metrics_logger=metrics,
            train_labels_mapper={label: label for label in range(NUM_CLASSES)},
            train_label_lookup=label_lookup,
            model_use_cache=False,
        )

        if spec.warmup_batches:
            trainer.train_epoch(batches[: spec.warmup_batches], epoch=-1)
        metrics.writer.flush()
        torch.cuda.synchronize(device)
        allocated_mib = torch.cuda.memory_allocated(device) / (1024 * 1024)
        ready_queue.put({"status": "ready", "worker_id": worker_id})
        ready_sent = True
        if not start_event.wait(WORKER_TIMEOUT_SECONDS):
            raise TimeoutError("timed out waiting for the shared benchmark start")

        start_time = time.monotonic()
        start_cpu = time.process_time()
        for epoch in range(spec.measured_epochs):
            trainer.train_epoch(batches, epoch=epoch)
        torch.cuda.synchronize(device)
        metrics.close()
        metrics = None
        cpu_seconds = time.process_time() - start_cpu
        end_time = time.monotonic()
        steps = spec.measured_epochs * spec.batches_per_epoch
        result_queue.put(
            {
                "status": "ok",
                "result": asdict(
                    WorkerResult(
                        worker_id=worker_id,
                        start_time=start_time,
                        end_time=end_time,
                        cpu_seconds=cpu_seconds,
                        steps=steps,
                        samples=steps * spec.batch_size,
                        gpu_name=torch.cuda.get_device_name(device),
                        allocated_mib=allocated_mib,
                    )
                ),
            }
        )
        del reserve
    except Exception:
        payload = {
            "status": "error",
            "worker_id": worker_id,
            "traceback": traceback.format_exc(),
        }
        (result_queue if ready_sent else ready_queue).put(payload)
    finally:
        if metrics is not None:
            metrics.close()


def queue_get(result_queue, description: str):
    try:
        return result_queue.get(timeout=WORKER_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError(f"timed out waiting for {description}") from exc


def terminate_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)


def run_group(
    context,
    concurrency: int,
    spec: WorkerSpec,
    log_root: Path,
) -> GroupResult:
    start_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    for worker_id in range(concurrency):
        worker_log_dir = log_root / f"worker_{worker_id:02d}"
        process = context.Process(
            target=run_worker,
            args=(
                worker_id,
                spec,
                start_event,
                ready_queue,
                result_queue,
                str(worker_log_dir),
            ),
        )
        process.start()
        processes.append(process)

    try:
        for _ in range(concurrency):
            payload = queue_get(ready_queue, "worker initialization")
            if payload["status"] != "ready":
                raise RuntimeError(payload["traceback"])
        start_event.set()
        worker_results = []
        for _ in range(concurrency):
            payload = queue_get(result_queue, "worker result")
            if payload["status"] != "ok":
                raise RuntimeError(payload["traceback"])
            worker_results.append(WorkerResult(**payload["result"]))
        for process in processes:
            process.join(timeout=WORKER_TIMEOUT_SECONDS)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"benchmark worker {process.pid} exited with code {process.exitcode}"
                )
    except Exception:
        start_event.set()
        terminate_processes(processes)
        raise
    finally:
        ready_queue.close()
        result_queue.close()

    return GroupResult(
        concurrency=concurrency,
        wall_seconds=max(result.end_time for result in worker_results)
        - min(result.start_time for result in worker_results),
        cpu_seconds=sum(result.cpu_seconds for result in worker_results),
        steps=sum(result.steps for result in worker_results),
        samples=sum(result.samples for result in worker_results),
        mean_trial_seconds=statistics.mean(
            result.end_time - result.start_time for result in worker_results
        ),
        gpu_name=worker_results[0].gpu_name,
        allocated_mib_per_worker=statistics.mean(
            result.allocated_mib for result in worker_results
        ),
    )


def median_group(results: list[GroupResult]) -> GroupResult:
    first = results[0]
    return GroupResult(
        concurrency=first.concurrency,
        wall_seconds=statistics.median(result.wall_seconds for result in results),
        cpu_seconds=statistics.median(result.cpu_seconds for result in results),
        steps=first.steps,
        samples=first.samples,
        mean_trial_seconds=statistics.median(
            result.mean_trial_seconds for result in results
        ),
        gpu_name=first.gpu_name,
        allocated_mib_per_worker=statistics.median(
            result.allocated_mib_per_worker for result in results
        ),
    )


def run_cpu_comparison(context, args, temp_root: Path) -> None:
    modes = {
        "default_gated": (0, False),
        "previous_every_batch": (1, False),
        "full_batch_diagnostics": (1, True),
    }
    rng = random.Random(7)
    print("\nCPU comparison (one worker)")
    print(
        "batch,mode,wall_ms_per_step,cpu_ms_per_step,average_cpu_cores,"
        "samples_per_second"
    )
    for batch_size in args.batch_sizes:
        results_by_mode = {mode: [] for mode in modes}
        for repeat in range(args.cpu_repeats):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                batch_loss_log_interval, log_batch_diagnostics = modes[mode]
                spec = make_worker_spec(
                    args,
                    batch_size,
                    seed=10_000 + repeat,
                    batch_loss_log_interval=batch_loss_log_interval,
                    log_batch_diagnostics=log_batch_diagnostics,
                )
                result = run_group(
                    context,
                    1,
                    spec,
                    temp_root / f"cpu_b{batch_size}_{mode}_r{repeat}",
                )
                results_by_mode[mode].append(result)

        medians = {mode: median_group(results) for mode, results in results_by_mode.items()}
        for mode, result in medians.items():
            print(
                f"{batch_size},{mode},"
                f"{result.wall_seconds * 1000 / result.steps:.4f},"
                f"{result.cpu_seconds * 1000 / result.steps:.4f},"
                f"{result.average_cpu_cores:.3f},"
                f"{result.samples_per_second:.1f}"
            )
        previous = medians["previous_every_batch"]
        current = medians["default_gated"]
        cpu_reduction = 100 * (1 - current.cpu_seconds / previous.cpu_seconds)
        wall_reduction = 100 * (1 - current.wall_seconds / previous.wall_seconds)
        print(
            f"batch {batch_size}: gated logging reduced process CPU time by "
            f"{cpu_reduction:.1f}% and wall time by {wall_reduction:.1f}%"
        )
        diagnostics = medians["full_batch_diagnostics"]
        diagnostics_cpu_reduction = 100 * (
            1 - current.cpu_seconds / diagnostics.cpu_seconds
        )
        diagnostics_wall_reduction = 100 * (
            1 - current.wall_seconds / diagnostics.wall_seconds
        )
        print(
            f"batch {batch_size}: compared with full batch diagnostics, the "
            f"default reduced process CPU time by {diagnostics_cpu_reduction:.1f}% "
            f"and wall time by {diagnostics_wall_reduction:.1f}%"
        )


def run_concurrency_sweep(context, args, temp_root: Path) -> None:
    print("\nConcurrency sweep (default-gated logging)")
    print(
        "batch,runs,aggregate_samples_per_second,speedup_vs_one,"
        "incremental_gain_percent,efficiency_percent,mean_trial_seconds,"
        "aggregate_cpu_cores"
    )
    for batch_size in args.batch_sizes:
        medians = []
        for concurrency in args.concurrency:
            repeated_results = []
            for repeat in range(args.concurrency_repeats):
                spec = make_worker_spec(
                    args,
                    batch_size,
                    seed=20_000 + repeat,
                )
                repeated_results.append(
                    run_group(
                        context,
                        concurrency,
                        spec,
                        temp_root / f"scale_b{batch_size}_n{concurrency}_r{repeat}",
                    )
                )
            medians.append(median_group(repeated_results))

        baseline = medians[0]
        previous = None
        for result in medians:
            speedup = result.samples_per_second / baseline.samples_per_second
            incremental_gain = (
                float("nan")
                if previous is None
                else 100
                * (result.samples_per_second / previous.samples_per_second - 1)
            )
            print(
                f"{batch_size},{result.concurrency},{result.samples_per_second:.1f},"
                f"{speedup:.3f},{incremental_gain:.1f},"
                f"{100 * speedup / result.concurrency:.1f},"
                f"{result.mean_trial_seconds:.3f},{result.average_cpu_cores:.3f}"
            )
            previous = result
        best = max(medians, key=lambda result: result.samples_per_second)
        print(
            f"batch {batch_size}: highest measured aggregate throughput used "
            f"{best.concurrency} concurrent runs ({best.samples_per_second:.1f} samples/s)"
        )


def main() -> None:
    args = parse_args()
    if args.skip_cpu_comparison and args.skip_concurrency:
        raise ValueError("cannot skip both benchmark sections")
    if sorted(args.concurrency) != args.concurrency or len(set(args.concurrency)) != len(
        args.concurrency
    ):
        raise ValueError("--concurrency values must be unique and sorted ascending")
    if args.concurrency[0] != 1:
        raise ValueError("--concurrency must start with 1 so speedups have a baseline")

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"platform={sys.platform}")
    print(f"logical_cpus={mp.cpu_count()}")
    print(f"torch_num_threads_per_process={torch.get_num_threads()}")
    print(f"reserved_gpu_mib_per_worker={args.reserve_gpu_mib}")
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="frozen_hpo_concurrency_") as temp_dir:
        temp_root = Path(temp_dir)
        if not args.skip_cpu_comparison:
            run_cpu_comparison(context, args, temp_root)
        if not args.skip_concurrency:
            run_concurrency_sweep(context, args, temp_root)


if __name__ == "__main__":
    main()
