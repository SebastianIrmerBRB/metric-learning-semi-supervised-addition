#!/usr/bin/env python3
"""Measure VRAM and latency of ``--evaluation_embedding_residency gpu`` policies.

Every variant keeps the projected embeddings resident on the GPU and runs the
production ``utils.evaluate_embeddings`` path.  Only the memory policy of the
reusable single-GPU FAISS wrapper changes, so the reported deltas isolate the
cost of the FAISS scratch arena and of retaining the gallery index between
evaluations:

``baseline``
    What ``residency=gpu`` does today: ``StandardGpuResources`` keeps its
    default scratch arena and the flat index survives every evaluation.
``arena-<MiB>``
    Same, but the scratch arena is capped via ``setTempMemory``.
``arena-<MiB>-discard``
    Capped arena plus ``_discard_index()`` after each evaluation.

Each variant runs in a fresh subprocess because the FAISS arena and the torch
caching allocator are process-lifetime state.  Per-process VRAM comes from
``nvidia-smi`` (the FAISS arena is allocated outside the torch allocator and is
invisible to ``torch.cuda.memory_reserved``).

Run from the repository root with a FAISS-GPU environment:

    python scripts/benchmark_gpu_residency_vram.py

To benchmark one split only, with more repeats:

    python scripts/benchmark_gpu_residency_vram.py --profiles CUB/test --repeats 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402

MIB = 1024 * 1024
DEFAULT_PROFILES = ("CUB/test", "Food101/test", "iNat/test")
DEFAULT_VARIANTS = ("baseline", "arena-0", "arena-256", "arena-256-discard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        help="dataset/split pairs supplying the real label distribution",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="memory policies to compare; all keep embeddings on the GPU",
    )
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=5)
    # Loose enough that the clusters overlap: identical metrics across variants
    # then actually constrain the comparison instead of all reading 1.0.
    parser.add_argument("--noise-std", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    # Internal: one worker process measures exactly one (profile, variant) pair.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_variant(variant: str) -> tuple[int | None, bool]:
    """Return ``(temp_memory_bytes, discard_index)`` for a variant name."""

    if variant == "baseline":
        return None, False
    parts = variant.split("-")
    if parts[0] != "arena" or len(parts) not in (2, 3):
        raise SystemExit(f"unrecognized variant: {variant}")
    if len(parts) == 3 and parts[2] != "discard":
        raise SystemExit(f"unrecognized variant: {variant}")
    return int(parts[1]) * MIB, len(parts) == 3


def load_profile_labels(profile: str, data_root: Path) -> np.ndarray:
    dataset_name, _, split = profile.partition("/")
    if not split:
        raise SystemExit(f"profile must be DATASET/SPLIT: {profile}")
    dataset_class = utils.get_dataset_class(dataset_name)
    dataset = dataset_class(
        root=str(data_root / dataset_name),
        split=split,
        transform=None,
        download=False,
    )
    return np.asarray(dataset.labels, dtype=np.int64)


def make_embeddings(
    labels: np.ndarray,
    embedding_dim: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Create deterministic normalized clusters for an exact label distribution."""

    _, dense_labels = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal(
        (int(dense_labels.max()) + 1, embedding_dim),
        dtype=np.float32,
    )
    embeddings = centers[dense_labels]
    embeddings += float(noise_std) * rng.standard_normal(
        embeddings.shape,
        dtype=np.float32,
    )
    embeddings /= np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
        None,
    )
    return np.ascontiguousarray(embeddings)


def device_vram_bytes(gpu: int) -> int:
    """Read total in-use GPU memory, including allocations outside torch.

    Per-process accounting (``--query-compute-apps``) reports nothing under
    WSL2, so the worker measures device-wide usage and subtracts the ambient
    reading taken before it created a CUDA context.
    """

    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(query.stdout.strip().splitlines()[0]) * MIB


def install_knn_policy(
    device: torch.device,
    temp_memory_bytes: int | None,
    discard_index: bool,
) -> None:
    """Seed the production KNN cache with an instance using this policy.

    ``evaluate_embeddings`` looks the wrapper up under ``low_memory=False`` for
    GPU-resident embeddings, so the instance is registered under that key no
    matter which policy it carries.
    """

    backend = utils._selected_faiss_gpu(device)
    if backend is None:
        raise SystemExit("GPU FAISS is unavailable; this benchmark requires it")
    faiss_module, gpu_id = backend
    knn = utils._ReusableSingleGpuFaissKNN(faiss_module, gpu_id, low_memory=False)
    if temp_memory_bytes is not None:
        knn.resources.setTempMemory(temp_memory_bytes)
    # ``low_memory`` additionally drives index disposal and the empty_cache in
    # the high-k torch fallback; the arena is already sized above.
    knn.low_memory = discard_index
    utils._REUSABLE_FAISS_KNN[(id(faiss_module), int(gpu_id), False)] = knn


def run_worker(args: argparse.Namespace) -> int:
    temp_memory_bytes, discard_index = parse_variant(args.variant)
    device = torch.device("cuda", args.gpu)
    # Sampled before the CUDA context exists so the delta covers this process
    # in full: context, torch allocator, and the FAISS scratch arena.
    ambient_vram = device_vram_bytes(args.gpu)
    torch.cuda.init()

    labels = load_profile_labels(args.profile, args.data_root)
    embeddings = make_embeddings(
        labels,
        args.embedding_dim,
        args.noise_std,
        args.seed,
    )
    _, counts = np.unique(labels, return_counts=True)

    install_knn_policy(device, temp_memory_bytes, discard_index)

    # Staged exactly like extract_eval_embeddings(keep_on_device=True).
    gpu_embeddings = torch.from_numpy(embeddings).to(device, dtype=torch.float32)
    gpu_labels = torch.from_numpy(labels).to(device)
    embedding_bytes = gpu_embeddings.numel() * gpu_embeddings.element_size()

    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    metrics: tuple[float, float] | None = None
    peak_vram = 0
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        precision_at_1, mean_average_precision_at_r = utils.evaluate_embeddings(
            gpu_embeddings,
            gpu_labels,
            name=args.profile,
            return_per_class=False,
            dataset=None,
            device=device,
            allow_cpu_fallback=False,
        )
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        peak_vram = max(peak_vram, device_vram_bytes(args.gpu) - ambient_vram)
        metrics = (float(precision_at_1), float(mean_average_precision_at_r))

    print(
        json.dumps(
            {
                "profile": args.profile,
                "variant": args.variant,
                "samples": int(len(labels)),
                "classes": int(len(counts)),
                "k": int(counts.max()) - 1,
                "embedding_bytes": int(embedding_bytes),
                "ambient_vram_bytes": int(ambient_vram),
                "peak_vram_bytes": int(peak_vram),
                "torch_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "median_seconds": statistics.median(durations),
                "min_seconds": min(durations),
                "precision_at_1": metrics[0],
                "mean_average_precision_at_r": metrics[1],
            }
        ),
        flush=True,
    )
    return 0


def run_variant(args: argparse.Namespace, profile: str, variant: str) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--profile",
        profile,
        "--variant",
        variant,
        "--embedding-dim",
        str(args.embedding_dim),
        "--repeats",
        str(args.repeats),
        "--noise-std",
        str(args.noise_std),
        "--seed",
        str(args.seed),
        "--data-root",
        str(args.data_root),
        "--gpu",
        str(args.gpu),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{profile} / {variant} failed:\n{completed.stdout}\n{completed.stderr}"
        )
    payload = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not payload:
        raise SystemExit(
            f"{profile} / {variant} produced no result:\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(payload[-1])


def print_profile_table(profile: str, results: list[dict]) -> None:
    header = results[0]
    print(
        f"\n{profile}: {header['samples']} samples, {header['classes']} classes, "
        f"k={header['k']}, embeddings={header['embedding_bytes'] / MIB:.0f} MiB on GPU"
    )
    print(
        f"  {'variant':<20} {'process VRAM':>13} {'ctx+faiss':>11} "
        f"{'torch peak':>11} {'median s':>10} {'vs base':>9}  metrics"
    )
    baseline_vram = results[0]["peak_vram_bytes"]
    baseline_seconds = results[0]["median_seconds"]
    for result in results:
        peak = result["peak_vram_bytes"]
        non_torch = peak - result["torch_peak_reserved_bytes"]
        saved = (baseline_vram - peak) / MIB
        slowdown = result["median_seconds"] / baseline_seconds
        print(
            f"  {result['variant']:<20} {peak / MIB:>10.0f} MiB "
            f"{non_torch / MIB:>8.0f} MiB "
            f"{result['torch_peak_reserved_bytes'] / MIB:>8.0f} MiB "
            f"{result['median_seconds']:>10.3f} "
            f"{slowdown:>8.2f}x  "
            f"P@1={result['precision_at_1']:.6f} "
            f"MAP@R={result['mean_average_precision_at_r']:.6f}"
            + ("" if saved <= 0 else f"   (-{saved:.0f} MiB)")
        )


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark requires a GPU")

    all_results = []
    for profile in args.profiles:
        results = [run_variant(args, profile, variant) for variant in args.variants]
        baseline_metrics = (
            results[0]["precision_at_1"],
            results[0]["mean_average_precision_at_r"],
        )
        for result in results[1:]:
            current = (
                result["precision_at_1"],
                result["mean_average_precision_at_r"],
            )
            if not np.allclose(current, baseline_metrics, rtol=0, atol=1e-9):
                raise SystemExit(
                    f"{profile} / {result['variant']} changed the metrics: "
                    f"{current} != {baseline_metrics}"
                )
        print_profile_table(profile, results)
        all_results.extend(results)

    if args.json is not None:
        args.json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
