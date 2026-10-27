#!/usr/bin/env python3
"""Compare the production CPU and CUDA validation-retrieval backends.

The model/projection device is deliberately CPU.  A deterministic embedding
matrix is staged in ``--batch-size`` chunks exactly as CPU model outputs are
handled by the evaluation path: NumPy arrays for CPU retrieval and CUDA tensors
for GPU retrieval.  The shared CPU forward pass is excluded so the reported
difference is attributable to staging plus retrieval, not model inference.

By default, the benchmark uses the exact test-set label distributions of CUB
and Cars196, 768-dimensional embeddings, a batch size of 128, and 100 repeated
"epochs" per backend.  The embeddings are identical for both backends.

Run from the repository root with a FAISS-GPU environment:

    python scripts/benchmark_validation_retrieval.py

For a short CPU-only smoke run:

    python scripts/benchmark_validation_retrieval.py --datasets CUB \
        --epochs 2 --limit-samples 256 --embedding-dim 32 --backends cpu

To benchmark embeddings exported from a real checkpoint, provide an ``.npz``
file containing two arrays named ``embeddings`` and ``labels``:

    python scripts/benchmark_validation_retrieval.py \
        --embeddings-npz path/to/test_embeddings.npz

An indexed DINO backbone cache contains only a feature matrix.  Point
``--features-npy`` at it and the script will infer the matching dataset split
by row count and attach its real labels:

    python scripts/benchmark_validation_retrieval.py \
        --features-npy data/Cars196/backbone_cache/v3/dinov2_vitb14/HASH.features.npy
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402


BACKEND_CPU = "cpu"
BACKEND_CUDA = "cuda"
BACKENDS = (BACKEND_CPU, BACKEND_CUDA)
DATASETS = ("CUB", "Cars196")
METRIC_NAMES = ("precision_at_1", "mean_average_precision_at_r")


@dataclass(frozen=True)
class DurationSummary:
    total: float
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class BackendBenchmark:
    backend: str
    device: str
    epochs: int
    metrics: dict[str, float]
    maximum_metric_span: dict[str, float]
    staging_seconds: DurationSummary
    retrieval_seconds: DurationSummary
    total_seconds: DurationSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="test label distributions to benchmark (default: CUB Cars196)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="dataset root containing CUB/ and Cars196/ (default: data)",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=None,
        help="optional real embeddings/labels NPZ; replaces --datasets",
    )
    parser.add_argument(
        "--features-npy",
        type=Path,
        default=None,
        help=(
            "optional backbone-cache feature matrix; real dataset labels are "
            "inferred or selected with --labels-dataset/--labels-split"
        ),
    )
    parser.add_argument(
        "--labels-dataset",
        choices=DATASETS,
        default=None,
        help="dataset supplying labels for --features-npy (default: infer)",
    )
    parser.add_argument(
        "--labels-split",
        choices=("train", "test", "train+test"),
        default=None,
        help="dataset split supplying labels for --features-npy (default: infer)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="validation repetitions per backend (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="CPU-output staging batch size (default: 128)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="synthetic embedding dimension (default: 768)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=3.0,
        help="synthetic within-class noise standard deviation (default: 3.0)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="optional prefix limit, mainly for smoke benchmarks",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=list(BACKENDS),
        help="backends to run (default: cpu cuda)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA ordinal (default: current CUDA device)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-6,
        help="maximum accepted CPU/CUDA metric delta (default: 1e-6)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="print progress every N epochs; 0 disables it (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for the complete JSON report",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.embeddings_npz is not None and args.features_npy is not None:
        raise SystemExit("Use only one of --embeddings-npz and --features-npy")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.embedding_dim < 1:
        raise SystemExit("--embedding-dim must be at least 1")
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        raise SystemExit("--noise-std must be finite and non-negative")
    if args.limit_samples is not None and args.limit_samples < 2:
        raise SystemExit("--limit-samples must be at least 2")
    if not math.isfinite(args.metric_tolerance) or args.metric_tolerance < 0:
        raise SystemExit("--metric-tolerance must be finite and non-negative")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must not be negative")


def resolve_cuda_device(gpu: int | None) -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the CPU/CUDA comparison cannot run")
    gpu_count = torch.cuda.device_count()
    gpu_id = torch.cuda.current_device() if gpu is None else int(gpu)
    if not 0 <= gpu_id < gpu_count:
        raise SystemExit(f"--gpu must be between 0 and {gpu_count - 1}")
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    if utils.available_cuda_retrieval_device(device) is None:
        raise SystemExit(
            "CUDA is available, but this environment has no FAISS-GPU backend. "
            "Install/use faiss-gpu before running the CPU/CUDA comparison."
        )
    return device


def load_dataset_labels(
    dataset_name: str,
    data_root: Path,
    split: str,
) -> np.ndarray:
    dataset_root = data_root / dataset_name
    dataset_class = utils.get_dataset_class(dataset_name)
    dataset = dataset_class(
        root=str(dataset_root),
        split=split,
        transform=None,
        download=False,
    )
    return np.asarray(dataset.labels, dtype=np.int64)


def validate_labels(labels: np.ndarray, profile_name: str) -> None:
    if labels.ndim != 1:
        raise SystemExit(f"{profile_name}: labels must be one-dimensional")
    if len(labels) < 2:
        raise SystemExit(f"{profile_name}: at least two samples are required")
    _, counts = np.unique(labels, return_counts=True)
    if int(counts.min()) < 2:
        raise SystemExit(
            f"{profile_name}: every included class needs at least two samples; "
            "increase --limit-samples"
        )


def make_embeddings(
    labels: np.ndarray,
    embedding_dim: int,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    """Create deterministic normalized clusters for an exact label distribution."""

    _, dense_labels = np.unique(labels, return_inverse=True)
    class_count = int(dense_labels.max()) + 1
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal(
        (class_count, embedding_dim),
        dtype=np.float32,
    )
    noise = rng.standard_normal(
        (len(labels), embedding_dim),
        dtype=np.float32,
    )
    embeddings = centers[dense_labels] + float(noise_std) * noise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.clip(norms, 1e-12, None)
    return torch.from_numpy(np.ascontiguousarray(embeddings))


def load_npz_profile(path: Path) -> tuple[str, torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        missing = {"embeddings", "labels"} - set(payload.files)
        if missing:
            raise SystemExit(
                f"{path} is missing required array(s): {', '.join(sorted(missing))}"
            )
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if embeddings.ndim != 2:
        raise SystemExit(f"{path}: embeddings must be a two-dimensional matrix")
    if len(embeddings) != len(labels):
        raise SystemExit(
            f"{path}: embeddings and labels contain different sample counts"
        )
    validate_labels(labels, path.stem)
    return (
        path.stem,
        torch.from_numpy(np.ascontiguousarray(embeddings)),
        torch.from_numpy(np.ascontiguousarray(labels)),
    )


def resolve_feature_matrix_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"Backbone feature path does not exist: {path}")

    candidates = sorted(path.glob("*.features.npy"))
    if not candidates:
        raise SystemExit(f"No *.features.npy files were found under {path}")
    complete = []
    diagnostics = []
    for candidate in candidates:
        try:
            matrix = np.load(candidate, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            diagnostics.append(f"{candidate.name}: unreadable ({exc})")
            continue
        if matrix.ndim != 2:
            diagnostics.append(f"{candidate.name}: shape={matrix.shape}")
            continue
        valid_rows = int(np.isfinite(matrix[:, 0]).sum())
        diagnostics.append(
            f"{candidate.name}: {valid_rows}/{len(matrix)} populated rows"
        )
        if valid_rows == len(matrix):
            complete.append(candidate)
    if len(complete) == 1:
        return complete[0]
    detail = "\n  ".join(diagnostics)
    if not complete:
        raise SystemExit(
            f"No complete backbone feature matrix was found under {path}:\n  {detail}"
        )
    choices = "\n  ".join(str(candidate) for candidate in complete)
    raise SystemExit(
        "Multiple complete backbone feature matrices were found; pass one exact "
        f"file path with --features-npy:\n  {choices}"
    )


def infer_label_dataset(path: Path, configured_dataset: str | None) -> str:
    if configured_dataset is not None:
        return configured_dataset
    path_parts = {part.lower() for part in path.parts}
    matches = [
        dataset_name
        for dataset_name in DATASETS
        if dataset_name.lower() in path_parts
    ]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        "Could not infer the label dataset from the feature path; provide "
        "--labels-dataset CUB or --labels-dataset Cars196"
    )


def resolve_feature_labels(
    feature_path: Path,
    row_count: int,
    args: argparse.Namespace,
) -> tuple[str, str, np.ndarray]:
    dataset_name = infer_label_dataset(feature_path, args.labels_dataset)
    candidate_splits = (
        (args.labels_split,)
        if args.labels_split is not None
        else ("train", "test", "train+test")
    )
    matches = []
    lengths = {}
    for split in candidate_splits:
        labels = load_dataset_labels(dataset_name, args.data_root, split)
        lengths[split] = len(labels)
        if len(labels) == row_count:
            matches.append((split, labels))
    if len(matches) == 1:
        split, labels = matches[0]
        return dataset_name, split, labels
    if not matches:
        sizes = ", ".join(
            f"{split}={length}" for split, length in lengths.items()
        )
        raise SystemExit(
            f"{feature_path}: {row_count} feature rows do not match the "
            f"{dataset_name} label counts ({sizes})"
        )
    matching_splits = ", ".join(split for split, _ in matches)
    raise SystemExit(
        f"{feature_path}: row count matches multiple splits ({matching_splits}); "
        "select one with --labels-split"
    )


def load_feature_matrix_profile(
    path: Path,
    args: argparse.Namespace,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    feature_path = resolve_feature_matrix_path(path)
    matrix = np.load(feature_path, mmap_mode="c", allow_pickle=False)
    if matrix.ndim != 2:
        raise SystemExit(
            f"{feature_path}: feature matrix must be two-dimensional, got {matrix.shape}"
        )
    if matrix.dtype != np.float32:
        raise SystemExit(
            f"{feature_path}: feature matrix must use float32, got {matrix.dtype}"
        )
    valid_rows = np.isfinite(matrix[:, 0])
    if not bool(valid_rows.all()):
        raise SystemExit(
            f"{feature_path}: only {int(valid_rows.sum())}/{len(matrix)} cache "
            "rows are populated"
        )
    if not bool(np.isfinite(matrix).all()):
        raise SystemExit(f"{feature_path}: feature matrix contains non-finite values")

    dataset_name, split, labels = resolve_feature_labels(
        feature_path,
        len(matrix),
        args,
    )
    if args.limit_samples is not None:
        matrix = matrix[: args.limit_samples]
        labels = labels[: args.limit_samples]
    profile_name = f"{dataset_name}-{split}-{feature_path.stem[:12]}"
    validate_labels(labels, profile_name)
    return (
        profile_name,
        torch.from_numpy(matrix),
        torch.from_numpy(np.ascontiguousarray(labels)),
    )


def make_dataset_profile(
    dataset_name: str,
    args: argparse.Namespace,
    profile_index: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    labels = load_dataset_labels(dataset_name, args.data_root, "test")
    if args.limit_samples is not None:
        labels = labels[: args.limit_samples]
    validate_labels(labels, dataset_name)
    embeddings = make_embeddings(
        labels,
        args.embedding_dim,
        args.noise_std,
        args.seed + profile_index,
    )
    return dataset_name, embeddings, torch.from_numpy(np.ascontiguousarray(labels))


def stage_cpu(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_batches = []
    label_batches = []
    for start in range(0, len(embeddings), batch_size):
        stop = start + batch_size
        embedding_batches.append(
            embeddings[start:stop].detach().cpu().numpy().astype(np.float32)
        )
        label_batches.append(labels[start:stop].detach().cpu().numpy())
    return np.concatenate(embedding_batches), np.concatenate(label_batches)


def stage_cuda(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_batches = []
    label_batches = []
    for start in range(0, len(embeddings), batch_size):
        stop = start + batch_size
        embedding_batches.append(
            embeddings[start:stop].to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )
        label_batches.append(labels[start:stop].to(device, non_blocking=True))
    return torch.cat(embedding_batches), torch.cat(label_batches)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    backend: str,
    device: torch.device,
) -> tuple[dict[str, float], float, float, float]:
    synchronize(device)
    total_started = time.perf_counter()
    staging_started = total_started
    if backend == BACKEND_CPU:
        staged_embeddings, staged_labels = stage_cpu(
            embeddings,
            labels,
            batch_size,
        )
    else:
        staged_embeddings, staged_labels = stage_cuda(
            embeddings,
            labels,
            batch_size,
            device,
        )
    synchronize(device)
    staging_seconds = time.perf_counter() - staging_started

    retrieval_started = time.perf_counter()
    precision_at_1, mean_average_precision_at_r = utils.evaluate_embeddings(
        staged_embeddings,
        staged_labels,
        name=f"benchmark {backend}",
        return_per_class=False,
        dataset=None,
        device=device,
        allow_cpu_fallback=False,
    )
    synchronize(device)
    retrieval_seconds = time.perf_counter() - retrieval_started
    total_seconds = time.perf_counter() - total_started
    return (
        {
            "precision_at_1": float(precision_at_1),
            "mean_average_precision_at_r": float(
                mean_average_precision_at_r
            ),
        },
        staging_seconds,
        retrieval_seconds,
        total_seconds,
    )


def summarize_durations(durations: list[float]) -> DurationSummary:
    ordered = sorted(float(duration) for duration in durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return DurationSummary(
        total=float(sum(ordered)),
        mean=float(statistics.fmean(ordered)),
        median=float(statistics.median(ordered)),
        p95=ordered[p95_index],
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def benchmark_backend(
    profile_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    backend: str,
    device: torch.device,
    args: argparse.Namespace,
) -> BackendBenchmark:
    # Warm up imports, allocator state, and the reusable FAISS index outside the
    # measured repetitions.
    run_once(embeddings, labels, args.batch_size, backend, device)

    metric_history = {metric: [] for metric in METRIC_NAMES}
    staging_durations = []
    retrieval_durations = []
    total_durations = []
    for epoch in range(1, args.epochs + 1):
        metrics, staging_seconds, retrieval_seconds, total_seconds = run_once(
            embeddings,
            labels,
            args.batch_size,
            backend,
            device,
        )
        for metric in METRIC_NAMES:
            metric_history[metric].append(metrics[metric])
        staging_durations.append(staging_seconds)
        retrieval_durations.append(retrieval_seconds)
        total_durations.append(total_seconds)
        if args.progress_every and (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.progress_every == 0
        ):
            print(
                f"  {profile_name} {backend}: epoch {epoch}/{args.epochs} "
                f"({total_seconds:.4f}s)",
                flush=True,
            )

    metric_snapshot = {
        metric: float(metric_history[metric][-1]) for metric in METRIC_NAMES
    }
    maximum_metric_span = {
        metric: float(max(values) - min(values))
        for metric, values in metric_history.items()
    }
    return BackendBenchmark(
        backend=backend,
        device=str(device),
        epochs=args.epochs,
        metrics=metric_snapshot,
        maximum_metric_span=maximum_metric_span,
        staging_seconds=summarize_durations(staging_durations),
        retrieval_seconds=summarize_durations(retrieval_durations),
        total_seconds=summarize_durations(total_durations),
    )


def make_comparison(
    cpu_result: BackendBenchmark,
    cuda_result: BackendBenchmark,
    tolerance: float,
) -> dict[str, object]:
    metric_deltas = {
        metric: abs(cpu_result.metrics[metric] - cuda_result.metrics[metric])
        for metric in METRIC_NAMES
    }
    maximum_metric_delta = max(metric_deltas.values())
    maximum_within_backend_span = max(
        *cpu_result.maximum_metric_span.values(),
        *cuda_result.maximum_metric_span.values(),
    )
    maximum_observed_difference = max(
        maximum_metric_delta,
        maximum_within_backend_span,
    )
    return {
        "metric_absolute_deltas": metric_deltas,
        "maximum_metric_absolute_delta": maximum_metric_delta,
        "maximum_within_backend_metric_span": maximum_within_backend_span,
        "maximum_observed_metric_difference": maximum_observed_difference,
        "metric_tolerance": tolerance,
        "metrics_match_within_tolerance": (
            maximum_observed_difference <= tolerance
        ),
        "retrieval_median_speedup_cpu_over_cuda": (
            cpu_result.retrieval_seconds.median
            / cuda_result.retrieval_seconds.median
        ),
        "total_median_speedup_cpu_over_cuda": (
            cpu_result.total_seconds.median / cuda_result.total_seconds.median
        ),
        "total_time_speedup_cpu_over_cuda": (
            cpu_result.total_seconds.total / cuda_result.total_seconds.total
        ),
    }


def print_result_table(
    profile_name: str,
    sample_count: int,
    class_count: int,
    results: dict[str, BackendBenchmark],
    comparison: dict[str, object] | None,
) -> None:
    print(
        f"\n{profile_name}: {sample_count:,} samples across {class_count:,} classes"
    )
    print(
        "Backend  Stage median  Retrieval median  Total median  Total (all)  "
        "P@1         MAP@R"
    )
    print(
        "-------  ------------  ----------------  ------------  -----------  "
        "----------  ----------"
    )
    for backend in BACKENDS:
        result = results.get(backend)
        if result is None:
            continue
        print(
            f"{backend:<7}  {result.staging_seconds.median:>10.4f}s  "
            f"{result.retrieval_seconds.median:>14.4f}s  "
            f"{result.total_seconds.median:>10.4f}s  "
            f"{result.total_seconds.total:>9.2f}s  "
            f"{result.metrics['precision_at_1']:>10.8f}  "
            f"{result.metrics['mean_average_precision_at_r']:>10.8f}"
        )
    if comparison is None:
        return
    print(
        "CPU/GPU median speedup: "
        f"retrieval={comparison['retrieval_median_speedup_cpu_over_cuda']:.2f}x, "
        f"staging+retrieval={comparison['total_median_speedup_cpu_over_cuda']:.2f}x"
    )
    print(
        "Maximum observed metric difference: "
        f"{comparison['maximum_observed_metric_difference']:.3e} "
        f"(tolerance {comparison['metric_tolerance']:.3e}, "
        f"match={comparison['metrics_match_within_tolerance']})"
    )


def system_info(cuda_device: torch.device | None) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": None if cuda_device is None else str(cuda_device),
        "cuda_device_name": (
            None
            if cuda_device is None
            else torch.cuda.get_device_name(cuda_device)
        ),
        "faiss": getattr(faiss, "__version__", "unknown"),
        "faiss_gpu_count": int(faiss.get_num_gpus()),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    cuda_device = (
        resolve_cuda_device(args.gpu)
        if BACKEND_CUDA in args.backends
        else None
    )
    devices = {
        BACKEND_CPU: torch.device("cpu"),
        BACKEND_CUDA: cuda_device,
    }

    if args.embeddings_npz is not None:
        profiles = [load_npz_profile(args.embeddings_npz)]
        if args.limit_samples is not None:
            profile_name, embeddings, labels = profiles[0]
            embeddings = embeddings[: args.limit_samples]
            labels = labels[: args.limit_samples]
            validate_labels(labels.numpy(), profile_name)
            profiles = [(profile_name, embeddings, labels)]
    elif args.features_npy is not None:
        profiles = [load_feature_matrix_profile(args.features_npy, args)]
    else:
        profiles = [
            make_dataset_profile(dataset_name, args, profile_index)
            for profile_index, dataset_name in enumerate(args.datasets)
        ]

    info = system_info(cuda_device)
    print(json.dumps(info, indent=2))
    print(
        f"Configuration: CPU model outputs, batch_size={args.batch_size}, "
        f"epochs/backend={args.epochs}, backends={args.backends}"
    )
    print(
        "Timing includes output staging and exact retrieval; the identical CPU "
        "model forward pass is intentionally excluded."
    )

    report = {
        "system": info,
        "configuration": {
            "model_device": "cpu",
            "batch_size": args.batch_size,
            "epochs_per_backend": args.epochs,
            "backends": args.backends,
            "embedding_dim": args.embedding_dim,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "metric_tolerance": args.metric_tolerance,
            "embeddings_npz": (
                None
                if args.embeddings_npz is None
                else str(args.embeddings_npz)
            ),
            "features_npy": (
                None if args.features_npy is None else str(args.features_npy)
            ),
        },
        "profiles": [],
    }
    all_metrics_match = True
    logger.disable("utils")
    try:
        for profile_name, embeddings, labels in profiles:
            unique_labels = torch.unique(labels)
            results = {}
            for backend in args.backends:
                device = devices[backend]
                if device is None:
                    raise RuntimeError(f"No device was resolved for {backend}")
                results[backend] = benchmark_backend(
                    profile_name,
                    embeddings,
                    labels,
                    backend,
                    device,
                    args,
                )
            comparison = None
            if BACKEND_CPU in results and BACKEND_CUDA in results:
                comparison = make_comparison(
                    results[BACKEND_CPU],
                    results[BACKEND_CUDA],
                    args.metric_tolerance,
                )
                all_metrics_match = all_metrics_match and bool(
                    comparison["metrics_match_within_tolerance"]
                )
            print_result_table(
                profile_name,
                len(labels),
                len(unique_labels),
                results,
                comparison,
            )
            report["profiles"].append(
                {
                    "name": profile_name,
                    "sample_count": len(labels),
                    "class_count": len(unique_labels),
                    "embedding_dim": int(embeddings.shape[1]),
                    "results": {
                        backend: asdict(result)
                        for backend, result in results.items()
                    },
                    "comparison": comparison,
                }
            )
    finally:
        logger.enable("utils")

    report["all_metrics_match_within_tolerance"] = all_metrics_match
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.output}")
    if not all_metrics_match:
        print("\nCPU and CUDA retrieval metrics exceeded the requested tolerance.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
