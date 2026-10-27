#!/usr/bin/env python3
"""Benchmark DataLoader multiprocessing start methods (spawn vs forkserver).

``main.py`` forces the ``spawn`` start method process-wide and ``--dataloader_start_method``
defaults to ``spawn``. Both are valid with CUDA, but they pay very different
startup costs: a ``spawn`` worker boots a fresh interpreter and re-imports
``__main__`` (which pulls in torch and the whole ``training`` package), while a
``forkserver`` worker is forked from a server process that imported those
modules once.

Three numbers are reported per configuration:

* ``first_loader``: DataLoader construction to first batch for the first loader
  in the process. For forkserver this includes the one-time server bootstrap.
* ``loader_restart``: the same measurement for later loaders, which is what an
  evaluation loader, a non-persistent epoch restart, or a new HPO trial pays.
* ``throughput``: images/second once workers are warm, to confirm the start
  method does not change steady-state cost.

Resident and proportional set size of the whole worker tree are sampled
throughout, because forkserver workers share the imported modules copy-on-write
while spawn workers hold a private copy each.

Each configuration runs in its own subprocess so that the forkserver bootstrap
is charged exactly once, the way a training run charges it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Mirror the thread caps main.py sets before importing torch so worker startup
# and steady-state throughput see the same BLAS configuration as a real run.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import utils

# Importing the training entry point is the point of the benchmark, not an
# accident: spawn re-imports __main__ in every worker and forkserver preloads it
# once, so the module weight of a real run has to be present here to compare
# them. Keep this import at module scope for that reason.
import main as _training_entry_point  # noqa: F401


DEFAULT_IMAGE_ROOT = WORKSPACE_ROOT / "data" / "CUB" / "CUB_200_2011" / "images"
DEFAULT_BATCH_SIZE = 64
DEFAULT_RESTARTS = 4
DEFAULT_THROUGHPUT_BATCHES = 30
DEFAULT_REPEATS = 3
MEMORY_SAMPLE_INTERVAL = 0.05
CONFIG_TIMEOUT_SECONDS = 1_800


@dataclass(frozen=True)
class ConfigResult:
    start_method: str
    num_workers: int
    cuda: bool
    first_loader_s: float
    loader_restart_s: float
    loader_restart_samples: list
    throughput_img_s: float
    peak_tree_rss_mb: float
    peak_tree_pss_mb: float
    worker_process_count: int


class ImageFolderScanDataset(Dataset):
    """Minimal image dataset over a directory tree of JPEGs.

    The project's own datasets need protocol files and split metadata; the start
    method only interacts with worker startup and pickling, so a plain image
    reader with the standard eval transform reproduces the relevant worker cost.
    """

    def __init__(self, paths, image_size=224):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), 0


def collect_image_paths(image_root, limit):
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"image root {root} does not exist; pass --image-root")
    paths = sorted(str(path) for path in root.rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg files found under {root}")
    return paths[:limit]


def read_child_pids(pid):
    """Collect the transitive children of ``pid`` from procfs."""

    pids = []
    task_dir = Path(f"/proc/{pid}/task")
    try:
        task_ids = list(task_dir.iterdir())
    except OSError:
        return pids
    for task in task_ids:
        try:
            children = (task / "children").read_text().split()
        except OSError:
            continue
        for child in children:
            child_pid = int(child)
            pids.append(child_pid)
            pids.extend(read_child_pids(child_pid))
    return pids


def read_memory_kb(pid):
    """Return (rss_kb, pss_kb) for one process, or zeros if it already exited."""

    try:
        rollup = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return 0, 0
    rss_kb = 0
    pss_kb = 0
    for line in rollup.splitlines():
        if line.startswith("Rss:"):
            rss_kb = int(line.split()[1])
        elif line.startswith("Pss:"):
            pss_kb = int(line.split()[1])
    return rss_kb, pss_kb


class MemorySampler(threading.Thread):
    """Sample the RSS/PSS of this process tree while the workers are alive."""

    def __init__(self, interval=MEMORY_SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss_kb = 0
        self.peak_pss_kb = 0
        self.peak_process_count = 0
        self._stop_requested = threading.Event()

    def run(self):
        own_pid = os.getpid()
        while not self._stop_requested.is_set():
            child_pids = read_child_pids(own_pid)
            rss_kb = 0
            pss_kb = 0
            for pid in [own_pid, *child_pids]:
                process_rss_kb, process_pss_kb = read_memory_kb(pid)
                rss_kb += process_rss_kb
                pss_kb += process_pss_kb
            self.peak_rss_kb = max(self.peak_rss_kb, rss_kb)
            self.peak_pss_kb = max(self.peak_pss_kb, pss_kb)
            self.peak_process_count = max(self.peak_process_count, len(child_pids))
            self._stop_requested.wait(self.interval)

    def stop(self):
        self._stop_requested.set()
        self.join(timeout=5)


def time_to_first_batch(dataset, batch_size, num_workers, start_method, seed):
    """Construct a loader and time the wait for its first batch."""

    start = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **utils.make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=False,
            pin_memory=False,
        ),
    )
    iterator = iter(loader)
    next(iterator)
    elapsed = time.perf_counter() - start
    del iterator
    utils.shutdown_dataloader_workers(loader)
    return elapsed


def measure_throughput(dataset, batch_size, num_workers, start_method, seed, batches):
    """Images/second on a warm persistent loader, excluding worker startup."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **utils.make_dataloader_kwargs(
            num_workers,
            seed,
            start_method,
            persistent_workers=True,
            pin_memory=False,
        ),
    )
    iterator = iter(loader)
    warmup = max(2, num_workers)
    images = 0
    start = None
    for index, (batch, _) in enumerate(iterator):
        if index == warmup:
            start = time.perf_counter()
            images = 0
        if start is not None:
            images += batch.shape[0]
        if index >= warmup + batches:
            break
    elapsed = time.perf_counter() - start
    del iterator
    utils.shutdown_dataloader_workers(loader)
    return images / elapsed


def run_single_config(args):
    """Measure one (start method, worker count) pair; called in a fresh process."""

    paths = collect_image_paths(args.image_root, args.image_limit)
    dataset = ImageFolderScanDataset(paths, image_size=args.image_size)

    if args.cuda:
        # A real run holds a CUDA context and model weights before any loader is
        # built. Reproduce that so forkserver is exercised from the same state.
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda requested but torch.cuda.is_available() is False")
        torch.cuda.init()
        resident_model = torch.nn.Linear(2048, 2048).cuda()
        torch.zeros(1, 2048, device="cuda").mul_(1.0)
        del resident_model

    sampler = MemorySampler()
    sampler.start()
    try:
        first_loader_s = time_to_first_batch(
            dataset, args.batch_size, args.num_workers, args.start_method, args.seed
        )
        restarts = [
            time_to_first_batch(
                dataset, args.batch_size, args.num_workers, args.start_method, args.seed + index + 1
            )
            for index in range(args.restarts)
        ]
        throughput = measure_throughput(
            dataset,
            args.batch_size,
            args.num_workers,
            args.start_method,
            args.seed,
            args.throughput_batches,
        )
    finally:
        sampler.stop()

    return ConfigResult(
        start_method=args.start_method,
        num_workers=args.num_workers,
        cuda=bool(args.cuda),
        first_loader_s=first_loader_s,
        loader_restart_s=statistics.median(restarts),
        loader_restart_samples=[round(value, 4) for value in restarts],
        throughput_img_s=throughput,
        peak_tree_rss_mb=sampler.peak_rss_kb / 1024,
        peak_tree_pss_mb=sampler.peak_pss_kb / 1024,
        worker_process_count=sampler.peak_process_count,
    )


def run_config_in_subprocess(args, start_method, num_workers, repeat):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-single-config",
        "--start-method",
        start_method,
        "--num-workers",
        str(num_workers),
        "--batch-size",
        str(args.batch_size),
        "--image-root",
        str(args.image_root),
        "--image-limit",
        str(args.image_limit),
        "--image-size",
        str(args.image_size),
        "--restarts",
        str(args.restarts),
        "--throughput-batches",
        str(args.throughput_batches),
        "--seed",
        str(args.seed + repeat),
    ]
    if args.cuda:
        command.append("--cuda")
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=CONFIG_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{start_method}/{num_workers} workers failed with code {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    payload = completed.stdout.strip().splitlines()[-1]
    return ConfigResult(**json.loads(payload))


def format_table(results):
    header = (
        f"{'method':<11}{'workers':>8}{'first loader':>14}{'restart':>10}"
        f"{'img/s':>10}{'peak RSS':>11}{'peak PSS':>11}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.start_method:<11}{result.num_workers:>8}"
            f"{result.first_loader_s:>13.2f}s{result.loader_restart_s:>9.2f}s"
            f"{result.throughput_img_s:>10.0f}"
            f"{result.peak_tree_rss_mb:>9.0f}MB{result.peak_tree_pss_mb:>9.0f}MB"
        )
    return "\n".join(lines)


def aggregate(results):
    """Median-reduce repeats of the same configuration."""

    grouped = {}
    for result in results:
        grouped.setdefault((result.start_method, result.num_workers), []).append(result)
    reduced = []
    for (start_method, num_workers), group in grouped.items():
        reduced.append(
            ConfigResult(
                start_method=start_method,
                num_workers=num_workers,
                cuda=group[0].cuda,
                first_loader_s=statistics.median(item.first_loader_s for item in group),
                loader_restart_s=statistics.median(item.loader_restart_s for item in group),
                loader_restart_samples=[
                    value for item in group for value in item.loader_restart_samples
                ],
                throughput_img_s=statistics.median(item.throughput_img_s for item in group),
                peak_tree_rss_mb=statistics.median(item.peak_tree_rss_mb for item in group),
                peak_tree_pss_mb=statistics.median(item.peak_tree_pss_mb for item in group),
                worker_process_count=max(item.worker_process_count for item in group),
            )
        )
    return reduced


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default="spawn,forkserver", help="comma-separated start methods to compare")
    parser.add_argument("--worker-counts", default="4,8", help="comma-separated DataLoader worker counts")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--image-limit", type=int, default=4_096, help="cap on scanned images")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS, help="loader restarts timed per config")
    parser.add_argument("--throughput-batches", type=int, default=DEFAULT_THROUGHPUT_BATCHES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="subprocess repeats per config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="hold a CUDA context before building loaders")
    parser.add_argument("--json-out", default=None, help="write raw per-repeat results to this path")
    parser.add_argument("--run-single-config", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-method", default="spawn", help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=4, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.run_single_config:
        print(json.dumps(asdict(run_single_config(args))))
        return 0

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    worker_counts = [int(count) for count in args.worker_counts.split(",") if count.strip()]

    results = []
    for repeat in range(args.repeats):
        for num_workers in worker_counts:
            for method in methods:
                print(
                    f"[{repeat + 1}/{args.repeats}] {method} with {num_workers} workers ...",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(run_config_in_subprocess(args, method, num_workers, repeat))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([asdict(result) for result in results], indent=2))

    print()
    print(format_table(sorted(aggregate(results), key=lambda item: (item.num_workers, item.start_method))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
