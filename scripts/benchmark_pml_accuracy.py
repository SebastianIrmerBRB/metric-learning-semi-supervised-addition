#!/usr/bin/env python3
"""Benchmark PyTorch Metric Learning accuracy-calculation strategies.

The benchmark compares:

1. CPU with NumPy inputs and a rebuilt FAISS index.
2. The default PML CUDA path with NumPy inputs and all visible FAISS GPUs.
3. One FAISS GPU with CUDA-resident inputs and a rebuilt index.
4. One FAISS GPU with CUDA-resident inputs and a persistent reference index.

Run from the repository root with:

    python scripts/benchmark_pml_accuracy.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import faiss
import numpy as np
import pytorch_metric_learning
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import FaissKNN, return_results


METRICS = ("precision_at_1", "mean_average_precision_at_r")


class PersistentFaissKNN:
    """Exact FAISS k-NN that keeps one reference index between searches."""

    def __init__(self, gpu_id: int | None) -> None:
        self.gpu_id = gpu_id
        self.resources = None
        self.index = None
        self.reference = None
        self.reference_key = None

    @staticmethod
    def tensor_key(tensor: torch.Tensor) -> tuple:
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.device,
            tensor.dtype,
            tensor._version,
        )

    def build(self, reference: torch.Tensor) -> None:
        if reference.dtype != torch.float32 or not reference.is_contiguous():
            raise ValueError("The persistent FAISS reference must be contiguous float32.")
        dimension = int(reference.shape[1])
        cpu_index = faiss.IndexFlatL2(dimension)
        if self.gpu_id is None:
            if reference.device.type != "cpu":
                raise ValueError("A CPU FAISS index requires a CPU reference tensor.")
            self.index = cpu_index
        else:
            expected_device = torch.device("cuda", self.gpu_id)
            if reference.device != expected_device:
                raise ValueError(
                    f"FAISS GPU {self.gpu_id} requires a reference on {expected_device}; "
                    f"got {reference.device}."
                )
            self.resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(
                self.resources,
                self.gpu_id,
                cpu_index,
            )
        self.index.add(reference)
        self.reference = reference
        self.reference_key = self.tensor_key(reference)

    def __call__(
        self,
        query: torch.Tensor,
        k: int,
        reference: torch.Tensor | None = None,
        ref_includes_query: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.index is None:
            raise RuntimeError("Call build(reference) before using PersistentFaissKNN.")
        reference = query if reference is None else reference
        if self.tensor_key(reference) != self.reference_key:
            raise ValueError(
                "The reference tensor changed. Rebuild the persistent FAISS index first."
            )
        search_k = k + int(ref_includes_query)
        distances, indices = self.index.search(query, search_k)
        if not torch.is_tensor(distances):
            distances = torch.as_tensor(distances, device=query.device)
            indices = torch.as_tensor(indices, device=query.device)
        return return_results(distances, indices, ref_includes_query)


def make_embeddings(
    sample_count: int,
    embedding_dim: int,
    samples_per_class: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, normalized class clusters."""
    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) // samples_per_class
    class_count = int(labels[-1]) + 1
    centers = rng.standard_normal((class_count, embedding_dim), dtype=np.float32)
    noise = rng.standard_normal((sample_count, embedding_dim), dtype=np.float32)
    embeddings = centers[labels] + noise_std * noise
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    return np.ascontiguousarray(embeddings), labels


def make_calculator(
    device: torch.device,
    knn_func: Callable | None = None,
) -> AccuracyCalculator:
    kwargs = {
        "include": METRICS,
        "k": "max_bin_count",
        "device": device,
    }
    if knn_func is not None:
        kwargs["knn_func"] = knn_func
    return AccuracyCalculator(**kwargs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_accuracy(
    calculator: AccuracyCalculator,
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    device: torch.device,
    repeats: int,
) -> tuple[dict[str, float], float, float]:
    durations = []
    peak_memory_mib = []
    accuracy = {}

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        accuracy = calculator.get_accuracy(embeddings, labels)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        if device.type == "cuda":
            peak_memory_mib.append(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            )

    return (
        {metric: float(accuracy[metric]) for metric in METRICS},
        statistics.median(durations),
        statistics.median(peak_memory_mib) if peak_memory_mib else 0.0,
    )


def timed_persistent_build(
    knn: PersistentFaissKNN,
    reference: torch.Tensor,
    device: torch.device,
) -> float:
    synchronize(device)
    started = time.perf_counter()
    knn.build(reference)
    synchronize(device)
    return time.perf_counter() - started


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*(("-" * width) for width in widths)))
    for row in rows:
        print(template.format(*row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 5_000, 10_000],
        help="sample counts to benchmark (default: 1000 5000 10000)",
    )
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=3.0,
        help="within-class noise (default: 3.0)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ordinal for the single-GPU tests (default: current CUDA device)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(size < 2 for size in args.sizes):
        raise SystemExit("Every sample count must be at least 2.")
    if args.embedding_dim < 1:
        raise SystemExit("--embedding-dim must be at least 1.")
    if args.samples_per_class < 2:
        raise SystemExit("--samples-per-class must be at least 2.")
    if not np.isfinite(args.noise_std) or args.noise_std < 0:
        raise SystemExit("--noise-std must be finite and non-negative.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, so the CPU/CUDA comparison cannot run.")
    if args.gpu is not None and not 0 <= args.gpu < torch.cuda.device_count():
        raise SystemExit(
            f"--gpu must be between 0 and {torch.cuda.device_count() - 1}."
        )


def result_row(
    sample_count: int,
    mode: str,
    accuracy: dict[str, float],
    seconds: float,
    cpu_seconds: float,
    cpu_accuracy: dict[str, float],
    cuda_peak_mib: float,
    build_seconds: float | None = None,
) -> tuple[list[str], float]:
    max_delta = max(
        abs(cpu_accuracy[metric] - accuracy[metric]) for metric in METRICS
    )
    return (
        [
            f"{sample_count:,}",
            mode,
            f"{seconds:.4f}",
            "-" if build_seconds is None else f"{build_seconds:.4f}",
            f"{cpu_seconds / seconds:.2f}x",
            f"{accuracy['precision_at_1']:.6f}",
            f"{accuracy['mean_average_precision_at_r']:.6f}",
            f"{max_delta:.2e}",
            "-" if not cuda_peak_mib else f"{cuda_peak_mib:.1f}",
        ],
        max_delta,
    )


def warm_up(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cuda_device: torch.device,
    gpu_id: int,
    faiss_gpu_available: bool,
) -> None:
    cpu_device = torch.device("cpu")
    benchmark_accuracy(
        make_calculator(cpu_device), embeddings, labels, cpu_device, repeats=1
    )
    benchmark_accuracy(
        make_calculator(cuda_device), embeddings, labels, cuda_device, repeats=1
    )
    if not faiss_gpu_available:
        return

    cuda_embeddings = torch.as_tensor(
        embeddings, dtype=torch.float32, device=cuda_device
    )
    cuda_labels = torch.as_tensor(labels, dtype=torch.float32, device=cuda_device)
    single_gpu_calculator = make_calculator(
        cuda_device,
        FaissKNN(gpus=[gpu_id]),
    )
    benchmark_accuracy(
        single_gpu_calculator,
        cuda_embeddings,
        cuda_labels,
        cuda_device,
        repeats=1,
    )
    persistent_knn = PersistentFaissKNN(gpu_id)
    persistent_knn.build(cuda_embeddings)
    benchmark_accuracy(
        make_calculator(cuda_device, persistent_knn),
        cuda_embeddings,
        cuda_labels,
        cuda_device,
        repeats=1,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    gpu_id = torch.cuda.current_device() if args.gpu is None else args.gpu
    torch.cuda.set_device(gpu_id)
    cuda_device = torch.device("cuda", gpu_id)
    cpu_device = torch.device("cpu")
    faiss_gpu_count = int(faiss.get_num_gpus())
    faiss_gpu_available = faiss_gpu_count > gpu_id

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Metric Learning: {pytorch_metric_learning.__version__}")
    print(f"CUDA device {gpu_id}: {torch.cuda.get_device_name(cuda_device)}")
    print(
        f"FAISS: {getattr(faiss, '__version__', 'unknown')} "
        f"({faiss_gpu_count} visible GPU(s), "
        f"{faiss.omp_get_max_threads()} CPU threads)"
    )
    if not faiss_gpu_available:
        print(
            "Note: FAISS-GPU is unavailable for the selected device. "
            "Single-GPU and persistent-index rows will be skipped."
        )
    print(
        f"Configuration: dim={args.embedding_dim}, "
        f"samples/class={args.samples_per_class}, noise_std={args.noise_std}, "
        f"repeats={args.repeats}\n"
    )

    warmup_size = min(max(args.samples_per_class * 2, 64), min(args.sizes))
    warmup_embeddings, warmup_labels = make_embeddings(
        warmup_size,
        args.embedding_dim,
        args.samples_per_class,
        args.noise_std,
        args.seed,
    )
    warm_up(
        warmup_embeddings,
        warmup_labels,
        cuda_device,
        gpu_id,
        faiss_gpu_available,
    )

    rows = []
    largest_delta = 0.0
    for sample_count in args.sizes:
        embeddings, labels = make_embeddings(
            sample_count,
            args.embedding_dim,
            args.samples_per_class,
            args.noise_std,
            args.seed + sample_count,
        )

        cpu_accuracy, cpu_seconds, _ = benchmark_accuracy(
            make_calculator(cpu_device),
            embeddings,
            labels,
            cpu_device,
            args.repeats,
        )
        row, delta = result_row(
            sample_count,
            "CPU rebuild",
            cpu_accuracy,
            cpu_seconds,
            cpu_seconds,
            cpu_accuracy,
            0.0,
        )
        rows.append(row)
        largest_delta = max(largest_delta, delta)

        default_accuracy, default_seconds, default_peak_mib = benchmark_accuracy(
            make_calculator(cuda_device),
            embeddings,
            labels,
            cuda_device,
            args.repeats,
        )
        row, delta = result_row(
            sample_count,
            f"CUDA default ({faiss_gpu_count}G)",
            default_accuracy,
            default_seconds,
            cpu_seconds,
            cpu_accuracy,
            default_peak_mib,
        )
        rows.append(row)
        largest_delta = max(largest_delta, delta)

        if not faiss_gpu_available:
            continue

        # This transfer is intentionally outside the timed region. In a real
        # evaluation path, retain model outputs on CUDA instead of round-tripping
        # through NumPy.
        cuda_embeddings = torch.as_tensor(
            embeddings,
            dtype=torch.float32,
            device=cuda_device,
        )
        cuda_labels = torch.as_tensor(
            labels,
            dtype=torch.float32,
            device=cuda_device,
        )
        synchronize(cuda_device)

        single_accuracy, single_seconds, single_peak_mib = benchmark_accuracy(
            make_calculator(cuda_device, FaissKNN(gpus=[gpu_id])),
            cuda_embeddings,
            cuda_labels,
            cuda_device,
            args.repeats,
        )
        row, delta = result_row(
            sample_count,
            "CUDA 1G rebuild",
            single_accuracy,
            single_seconds,
            cpu_seconds,
            cpu_accuracy,
            single_peak_mib,
        )
        rows.append(row)
        largest_delta = max(largest_delta, delta)

        persistent_knn = PersistentFaissKNN(gpu_id)
        build_seconds = timed_persistent_build(
            persistent_knn,
            cuda_embeddings,
            cuda_device,
        )
        reuse_accuracy, reuse_seconds, reuse_peak_mib = benchmark_accuracy(
            make_calculator(cuda_device, persistent_knn),
            cuda_embeddings,
            cuda_labels,
            cuda_device,
            args.repeats,
        )
        row, delta = result_row(
            sample_count,
            "CUDA 1G reuse",
            reuse_accuracy,
            reuse_seconds,
            cpu_seconds,
            cpu_accuracy,
            reuse_peak_mib,
            build_seconds,
        )
        rows.append(row)
        largest_delta = max(largest_delta, delta)

    print_table(
        [
            "Samples",
            "Mode",
            "Median s",
            "Build s*",
            "CPU/mode",
            "P@1",
            "MAP@R",
            "Max |delta|",
            "CUDA MiB**",
        ],
        rows,
    )
    print(
        "\n* Persistent-index build is a one-time cost excluded from its median "
        "query/metric time."
    )
    print("** PyTorch peak memory excludes FAISS allocations.")
    print("CPU/mode greater than 1.0x means that mode was faster than CPU.")
    print(
        "CUDA 1G reuse is valid only while the reference embeddings remain unchanged."
    )
    if largest_delta > 1e-6:
        print(
            f"Warning: largest CPU/GPU metric difference was {largest_delta:.2e}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
