"""Compare ``torch_block_cg`` and ``cupy_cg`` on the real Iscen label system.

Both backends implement the same freezing block conjugate gradient over
independent right-hand sides, so any difference between them is floating-point
rounding, not a different algorithm. This script quantifies that difference on
the actual linear system ``iscen_label_spreading`` builds, using the parameters
from ``configs/ssl_methods/iscen.json``.

The system is captured from production code rather than rebuilt here: the
script calls ``iscen_label_spreading`` with a ``_dependencies`` override that
intercepts ``solve_sparse_label_system``, so the matrix, the class-balanced
seed matrix, and the tolerances are exactly what a training run would solve.

Three layers are measured:

1. Solver level. Each backend solves the captured system. Reported per column:
   the relative residual ``||b - Ax|| / ||b||``, the error against a tight
   reference solve, the CG iteration count, and wall time.
2. Backend delta. ``torch_block_cg`` against ``cupy_cg`` directly: max and
   median absolute difference, difference relative to the column scale, and
   how much of that survives the ``float32`` cast the method applies to its
   output.
3. End to end. A full ``iscen_label_spreading`` call per backend, compared on
   the values training actually consumes: the argmax pseudo-label, the
   certainty weight, and the set of candidates passing ``confidence_threshold``.

Two targeted checks cover the structural differences between the two code
paths: right-hand-side batching (the backends use different batch caps) and
non-canonical CSR input (only the Torch path canonicalizes the matrix).

Features are the cached DINOv2 CIFAR100 embeddings under ``data/CIFAR100``;
``--synthetic`` falls back to clustered unit-norm vectors when that cache is
absent.

Examples::

    python scripts/bench_cg_solver_numerics.py
    python scripts/bench_cg_solver_numerics.py --samples 40000 --repeats 5
    python scripts/bench_cg_solver_numerics.py --cases config,reference
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from training.ssl import algorithms  # noqa: E402
from training.ssl.algorithms import (  # noqa: E402
    iscen_label_spreading,
    ssl_algorithm_device,
)
from training.ssl.config import UNLABELED_TARGET  # noqa: E402

ISCEN_CONFIG = PROJECT_ROOT / "configs/ssl_methods/iscen.json"
FEATURE_CACHE = PROJECT_ROOT / "data/CIFAR100/backbone_cache/v3/dinov2_vitb14"
CIFAR_LABELS = PROJECT_ROOT / "data/CIFAR100/cifar-100-python"
BACKENDS = ("torch_block_cg", "cupy_cg")


@dataclass(frozen=True)
class Case:
    """One (alpha, rtol, max_iter) operating point of the same graph."""

    name: str
    alpha: float
    cg_rtol: float
    cg_max_iter: int
    note: str


@dataclass
class BackendResult:
    """One backend's solution of one captured system."""

    backend: str
    solution: np.ndarray
    iterations: np.ndarray
    seconds: list[float] = field(default_factory=list)
    device: str = ""
    batch_size: int = 0
    repeat_bitwise_identical: bool = True
    max_repeat_difference: float = 0.0
    error: str = ""


def load_features(args) -> tuple[np.ndarray, np.ndarray, str]:
    """Return embeddings and their ground-truth classes."""

    if not args.synthetic:
        cached = sorted(FEATURE_CACHE.glob("*.features.npy"))
        if cached and CIFAR_LABELS.is_dir():
            features = np.load(cached[0], mmap_mode="r")
            labels = []
            for split in ("train", "test"):
                with (CIFAR_LABELS / split).open("rb") as handle:
                    payload = pickle.load(handle, encoding="bytes")
                labels.append(np.asarray(payload[b"fine_labels"], dtype=np.int64))
            labels = np.concatenate(labels)
            if len(labels) == len(features):
                return features, labels, f"dinov2_vitb14 CIFAR100 ({cached[0].name[:12]})"
        if not args.synthetic:
            print("Cached CIFAR100 features unavailable; using synthetic clusters")

    # Unit-norm class clusters with the tightness real embeddings show, so the
    # kNN graph has comparable connectivity and the system a comparable
    # spectrum.
    rng = np.random.default_rng(args.seed)
    class_count = max(args.classes, 2)
    per_class = max(args.samples // class_count, 2)
    centers = rng.normal(size=(class_count, 768))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    features = np.repeat(centers, per_class, axis=0)
    features = features + 0.55 * rng.normal(size=features.shape)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.repeat(np.arange(class_count, dtype=np.int64), per_class)
    return features.astype(np.float32), labels, "synthetic clusters"


def subsample(features, labels, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return a class-limited subset plus its k-shot labeled targets."""

    rng = np.random.default_rng(args.seed)
    classes = np.unique(labels)
    if args.classes < len(classes):
        classes = rng.choice(classes, size=args.classes, replace=False)
        classes.sort()
    in_scope = np.flatnonzero(np.isin(labels, classes))
    if args.samples < len(in_scope):
        in_scope = rng.choice(in_scope, size=args.samples, replace=False)
        in_scope.sort()

    subset_features = np.ascontiguousarray(features[in_scope], dtype=np.float32)
    subset_labels = labels[in_scope]
    remap = {int(value): index for index, value in enumerate(classes)}
    subset_labels = np.array([remap[int(value)] for value in subset_labels], dtype=np.int64)

    targets = np.full(len(subset_labels), UNLABELED_TARGET, dtype=np.int64)
    for class_index in range(len(classes)):
        members = np.flatnonzero(subset_labels == class_index)
        if len(members) == 0:
            continue
        seeds = rng.choice(members, size=min(args.k_shot, len(members)), replace=False)
        targets[seeds] = class_index
    return subset_features, subset_labels, targets, len(classes)


class _SystemCaptured(Exception):
    """Stop ``iscen_label_spreading`` once its linear system exists."""


def capture_system(features, targets, num_classes, params, case) -> dict:
    """Return the exact system a training run would hand to the solver."""

    captured: dict = {}

    def intercept(matrix, right_hand_side, rtol, max_iter, name, **kwargs):
        captured.update(
            matrix=matrix.tocsr().astype(np.float64, copy=False),
            right_hand_side=np.asarray(right_hand_side, dtype=np.float64),
            rtol=float(rtol),
            max_iter=int(max_iter),
            name=name,
        )
        raise _SystemCaptured

    started_at = time.perf_counter()
    try:
        iscen_label_spreading(
            features,
            targets,
            num_classes,
            n_neighbors=params["n_neighbors"],
            gamma=params["gamma"],
            alpha=case.alpha,
            cg_rtol=case.cg_rtol,
            cg_max_iter=case.cg_max_iter,
            linear_solver="cg",
            _dependencies={"solve_sparse_label_system": intercept},
        )
    except _SystemCaptured:
        pass
    else:  # pragma: no cover - the interceptor always raises
        raise RuntimeError("iscen_label_spreading did not reach the linear solve")
    captured["build_seconds"] = time.perf_counter() - started_at
    return captured


def relative_residuals(matrix, right_hand_side, solution) -> np.ndarray:
    """Return ``||b - Ax|| / ||b||`` per right-hand side, in float64 on CPU."""

    residual = right_hand_side - matrix @ solution
    rhs_norm = np.linalg.norm(right_hand_side, axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    scale = np.where(rhs_norm > 0.0, rhs_norm, 1.0)
    return residual_norm / scale


def reference_solution(matrix, right_hand_side, args) -> tuple[np.ndarray, dict]:
    """Return a solve tight enough to serve as ground truth for both backends."""

    started_at = time.perf_counter()
    if args.reference == "spsolve":
        solution = sparse_linalg.spsolve(matrix.tocsc(), right_hand_side)
        solution = np.asarray(solution).reshape(right_hand_side.shape)
    else:
        solution = np.zeros_like(right_hand_side)
        for column in range(right_hand_side.shape[1]):
            rhs = right_hand_side[:, column]
            if not np.any(rhs):
                continue
            values, _ = sparse_linalg.cg(
                matrix,
                rhs,
                rtol=args.reference_rtol,
                atol=0.0,
                maxiter=args.reference_max_iter,
            )
            solution[:, column] = values
    seconds = time.perf_counter() - started_at
    residual = relative_residuals(matrix, right_hand_side, solution)
    return solution, {
        "method": args.reference,
        "seconds": seconds,
        "max_relative_residual": float(residual.max()),
    }


def run_backend(backend, captured, args) -> BackendResult:
    """Time and collect one backend's solve of the captured system."""

    matrix = captured["matrix"]
    right_hand_side = captured["right_hand_side"]
    result = BackendResult(backend=backend, solution=None, iterations=None)
    for repeat in range(args.repeats + 1):
        metadata: dict = {}
        started_at = time.perf_counter()
        try:
            with ssl_algorithm_device(args.device):
                if backend == "cupy_cg":
                    (
                        solution,
                        _info,
                        iterations,
                        device,
                    ) = algorithms._solve_sparse_label_system_cupy_cg(
                        matrix,
                        right_hand_side,
                        rtol=captured["rtol"],
                        max_iter=captured["max_iter"],
                        warm_start=None,
                        name="bench",
                        collect_iterations=True,
                        metadata=metadata,
                    )
                else:
                    (
                        solution,
                        _live,
                        _converged,
                        _breakdown,
                        iterations,
                        device,
                    ) = algorithms._solve_sparse_label_system_torch_block_cg(
                        matrix,
                        right_hand_side,
                        rtol=captured["rtol"],
                        max_iter=captured["max_iter"],
                        warm_start=None,
                        name="bench",
                        metadata=metadata,
                    )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        elapsed = time.perf_counter() - started_at
        # The first pass pays for CUDA context setup, kernel compilation, and
        # the sparse handle; it is not part of the steady-state cost.
        if repeat > 0:
            result.seconds.append(elapsed)
        if result.solution is not None:
            # Identical inputs on identical hardware should give identical
            # bits. Anything else makes a training run irreproducible for
            # reasons unrelated to which backend it selected.
            difference = float(np.abs(result.solution - solution).max())
            result.max_repeat_difference = max(result.max_repeat_difference, difference)
            result.repeat_bitwise_identical &= bool(
                np.array_equal(result.solution, solution)
            )
        result.solution = solution
        result.iterations = np.asarray(iterations, dtype=np.int64)
        result.device = str(device)
        result.batch_size = int(metadata.get("right_hand_side_batch_size", 0))
    return result


def difference_summary(left, right) -> dict:
    """Return how far apart two solutions of the same system are."""

    delta = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    column_scale = np.maximum(np.abs(left).max(axis=0), np.abs(right).max(axis=0))
    column_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    nonzero = scale > 0.0
    elementwise_relative = np.zeros_like(delta)
    elementwise_relative[nonzero] = delta[nonzero] / scale[nonzero]
    left_norm = np.linalg.norm(left, axis=0)
    left_norm = np.where(left_norm > 0.0, left_norm, 1.0)
    return {
        "max_absolute": float(delta.max()),
        "median_absolute": float(np.median(delta)),
        "max_relative_to_column_max": float((delta / column_scale).max()),
        "max_elementwise_relative": float(elementwise_relative.max()),
        "max_column_l2_relative": float(
            (np.linalg.norm(left - right, axis=0) / left_norm).max()
        ),
        "identical_fraction": float(np.mean(left == right)),
        "identical_fraction_float32": float(
            np.mean(left.astype(np.float32) == right.astype(np.float32))
        ),
    }


def run_end_to_end(features, targets, num_classes, params, case, args) -> dict:
    """Run the whole method once per backend and compare training inputs."""

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    timings: dict[str, float] = {}
    for backend in ("cg", *BACKENDS):
        started_at = time.perf_counter()
        try:
            with ssl_algorithm_device(args.device):
                probabilities, confidences = iscen_label_spreading(
                    features,
                    targets,
                    num_classes,
                    n_neighbors=params["n_neighbors"],
                    gamma=params["gamma"],
                    alpha=case.alpha,
                    cg_rtol=case.cg_rtol,
                    cg_max_iter=case.cg_max_iter,
                    linear_solver=backend,
                )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            outputs[backend] = None
            timings[backend] = float("nan")
            print(f"  end-to-end {backend} FAILED: {type(exc).__name__}: {exc}")
            continue
        timings[backend] = time.perf_counter() - started_at
        outputs[backend] = (probabilities, confidences)

    threshold = params["confidence_threshold"]
    summary: dict = {"seconds": timings, "confidence_threshold": threshold}
    unlabeled = targets == UNLABELED_TARGET
    for backend, values in outputs.items():
        if values is None:
            continue
        probabilities, confidences = values
        mass = probabilities.sum(axis=1)
        labels = np.where(mass > 0.0, probabilities.argmax(axis=1), UNLABELED_TARGET)
        selected = unlabeled & (mass > 0.0) & (confidences >= threshold)
        summary[backend] = {
            "labels": labels,
            "confidences": np.asarray(confidences, dtype=np.float64),
            "selected": selected,
            "selected_count": int(selected.sum()),
        }

    for left, right in (("cg", "torch_block_cg"), ("cg", "cupy_cg"), BACKENDS):
        if left not in summary or right not in summary:
            continue
        left_values = summary[left]
        right_values = summary[right]
        compared = unlabeled
        label_mismatch = int(
            np.sum(left_values["labels"][compared] != right_values["labels"][compared])
        )
        selection_mismatch = int(
            np.sum(left_values["selected"] != right_values["selected"])
        )
        summary[f"{left}_vs_{right}"] = {
            "pseudo_label_mismatches": label_mismatch,
            "pseudo_label_mismatch_fraction": float(label_mismatch / max(int(compared.sum()), 1)),
            "max_confidence_difference": float(
                np.abs(left_values["confidences"] - right_values["confidences"]).max()
            ),
            "selection_mismatches": selection_mismatch,
            "selected_counts": [
                left_values["selected_count"],
                right_values["selected_count"],
            ],
        }
    for backend in ("cg", *BACKENDS):
        if backend in summary:
            summary[backend] = {
                "selected_count": summary[backend]["selected_count"],
            }
    return summary


def check_batch_invariance(captured, args) -> dict:
    """Confirm the RHS batch cap does not change either backend's answer."""

    findings = {}
    caps = {
        "torch_block_cg": "TORCH_BLOCK_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH",
        "cupy_cg": "CUPY_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH",
    }
    for backend, attribute in caps.items():
        original = getattr(algorithms, attribute)
        baseline = run_backend(backend, captured, args)
        if baseline.error:
            findings[backend] = {"error": baseline.error}
            continue
        try:
            setattr(algorithms, attribute, 7)
            narrow = run_backend(backend, captured, args)
        finally:
            setattr(algorithms, attribute, original)
        if narrow.error:
            findings[backend] = {"error": narrow.error}
            continue
        findings[backend] = {
            "default_cap": int(original),
            "narrow_cap": 7,
            "bitwise_identical": bool(np.array_equal(baseline.solution, narrow.solution)),
            "max_absolute_difference": float(
                np.abs(baseline.solution - narrow.solution).max()
            ),
        }
    return findings


def check_non_canonical_matrix(captured, args) -> dict:
    """Solve the same system given as an unsorted, duplicated-entry CSR."""

    matrix = captured["matrix"]
    # Split every stored value into two halves that sum back to it and reverse
    # each row's column order. The CSR arrays are built directly because every
    # SciPy conversion path would sum the duplicates and sort the indices,
    # which is exactly the normalization under test.
    indptr = matrix.indptr
    indices = []
    values = []
    for row in range(matrix.shape[0]):
        row_columns = matrix.indices[indptr[row]:indptr[row + 1]][::-1]
        row_values = matrix.data[indptr[row]:indptr[row + 1]][::-1]
        indices.append(np.repeat(row_columns, 2))
        values.append(np.repeat(row_values, 2) * np.tile([0.25, 0.75], len(row_values)))
    scrambled = sparse.csr_matrix(
        (
            np.concatenate(values),
            np.concatenate(indices).astype(matrix.indices.dtype),
            (2 * indptr).astype(matrix.indptr.dtype),
        ),
        shape=matrix.shape,
    )
    scrambled_capture = dict(captured)
    scrambled_capture["matrix"] = scrambled

    findings = {
        "duplicate_entries": int(scrambled.nnz - matrix.nnz),
        "has_canonical_format": bool(scrambled.has_canonical_format),
        "has_sorted_indices": bool(scrambled.has_sorted_indices),
        "reconstruction_max_error": float(
            abs(scrambled - matrix).max() if scrambled.nnz else 0.0
        ),
    }
    for backend in BACKENDS:
        clean = run_backend(backend, captured, args)
        dirty = run_backend(backend, scrambled_capture, args)
        if clean.error or dirty.error:
            findings[backend] = {"error": clean.error or dirty.error}
            continue
        findings[backend] = {
            "max_absolute_difference": float(
                np.abs(clean.solution - dirty.solution).max()
            ),
            "max_relative_residual": float(
                relative_residuals(
                    captured["matrix"],
                    captured["right_hand_side"],
                    dirty.solution,
                ).max()
            ),
        }
    return findings


def format_case_report(case, captured, reference_info, results, comparisons, end_to_end) -> str:
    """Return the human-readable block for one operating point."""

    lines = [
        "",
        f"CASE {case.name}: alpha={case.alpha} rtol={case.cg_rtol:g} "
        f"max_iter={case.cg_max_iter}  ({case.note})",
        f"  system: {captured['matrix'].shape[0]} rows, "
        f"{captured['matrix'].nnz} nonzeros, "
        f"{captured['right_hand_side'].shape[1]} right-hand sides, "
        f"built in {captured['build_seconds']:.2f}s",
        f"  reference: {reference_info['method']} "
        f"({reference_info['seconds']:.2f}s, "
        f"max relative residual {reference_info['max_relative_residual']:.3e})",
        "",
        f"  {'backend':16} {'median s':>9} {'rhs/batch':>10} {'iters med':>10} "
        f"{'iters max':>10} {'max rel resid':>14} {'max err vs ref':>15} "
        f"{'run-to-run':>12}",
    ]
    for backend, result in results.items():
        if result.error:
            lines.append(f"  {backend:16} FAILED  {result.error[:70]}")
            continue
        metrics = comparisons["per_backend"][backend]
        median_seconds = float(np.median(result.seconds)) if result.seconds else float("nan")
        lines.append(
            f"  {backend:16} {median_seconds:9.3f} {result.batch_size:10d} "
            f"{int(np.median(result.iterations)):10d} {int(result.iterations.max()):10d} "
            f"{metrics['max_relative_residual']:14.3e} "
            f"{metrics['max_absolute_error_vs_reference']:15.3e} "
            f"{result.max_repeat_difference:12.3e}"
        )
    lines.append(
        "  'run-to-run' is one backend against itself on repeated identical calls."
    )

    delta = comparisons.get("torch_vs_cupy")
    if delta is not None:
        lines.extend(
            [
                "",
                "  torch_block_cg vs cupy_cg",
                f"    max absolute difference       {delta['max_absolute']:.3e}",
                f"    median absolute difference    {delta['median_absolute']:.3e}",
                f"    max difference / column max   {delta['max_relative_to_column_max']:.3e}",
                f"    max column L2 relative        {delta['max_column_l2_relative']:.3e}",
                f"    bitwise identical entries     {delta['identical_fraction'] * 100:.2f}%",
                f"    identical after float32 cast  "
                f"{delta['identical_fraction_float32'] * 100:.2f}%",
                f"    iteration counts identical    {comparisons['iterations_identical']}"
                f" (max difference {comparisons['max_iteration_difference']})",
                f"    vs worst self-disagreement    "
                f"{comparisons['cross_backend_over_self']:.2f}x",
            ]
        )

    if end_to_end:
        lines.extend(["", "  end to end (pseudo-labels handed to training)"])
        for backend in ("cg", *BACKENDS):
            if backend in end_to_end:
                lines.append(
                    f"    {backend:16} {end_to_end['seconds'][backend]:7.2f}s  "
                    f"selected {end_to_end[backend]['selected_count']} candidates"
                )
        for key in (
            "cg_vs_torch_block_cg",
            "cg_vs_cupy_cg",
            "torch_block_cg_vs_cupy_cg",
        ):
            if key not in end_to_end:
                continue
            values = end_to_end[key]
            lines.append(
                f"    {key:26} labels differ {values['pseudo_label_mismatches']} "
                f"({values['pseudo_label_mismatch_fraction'] * 100:.4f}%), "
                f"selection differs {values['selection_mismatches']}, "
                f"max certainty delta {values['max_confidence_difference']:.3e}"
            )
    return "\n".join(lines)


def make_cases(params) -> dict[str, Case]:
    return {
        "config": Case(
            "config",
            float(params["alpha"]),
            float(params["cg_rtol"]),
            int(params["cg_max_iter"]),
            "configs/ssl_methods/iscen.json, solved to tolerance",
        ),
        "reference": Case(
            "reference",
            0.99,
            1e-6,
            20,
            "LP-DeepSSL defaults: stiff system, truncated at 20 iterations",
        ),
        "tight": Case(
            "tight",
            0.99,
            1e-10,
            5000,
            "stiff system driven to a far tighter tolerance",
        ),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=20000, help="graph nodes")
    parser.add_argument("--classes", type=int, default=100, help="classes, one RHS each")
    parser.add_argument("--k-shot", type=int, default=16, help="labeled seeds per class")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats after warmup")
    parser.add_argument(
        "--cases",
        default="config,reference",
        help="comma-separated subset of config,reference,tight",
    )
    parser.add_argument("--device", default="cuda", help="ssl_device both backends run on")
    parser.add_argument(
        "--reference",
        choices=("scipy_cg", "spsolve"),
        default="scipy_cg",
        help="ground-truth solve the backends are scored against",
    )
    parser.add_argument("--reference-rtol", type=float, default=1e-13)
    parser.add_argument("--reference-max-iter", type=int, default=100000)
    parser.add_argument(
        "--end-to-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run whole iscen_label_spreading calls per backend",
    )
    parser.add_argument(
        "--checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the batching and non-canonical-CSR structural checks",
    )
    parser.add_argument(
        "--cupy-batch-cap",
        type=int,
        default=None,
        help="override CUPY_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH (default 32)",
    )
    parser.add_argument(
        "--torch-batch-cap",
        type=int,
        default=None,
        help="override TORCH_BLOCK_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH (default 512)",
    )
    parser.add_argument("--synthetic", action="store_true", help="skip the CIFAR100 cache")
    parser.add_argument("--verbose", action="store_true", help="keep the library log output")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the full result record to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if not args.verbose:
        logger.remove()

    if args.cupy_batch_cap is not None:
        algorithms.CUPY_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH = args.cupy_batch_cap
    if args.torch_batch_cap is not None:
        algorithms.TORCH_BLOCK_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH = args.torch_batch_cap

    config = json.loads(ISCEN_CONFIG.read_text(encoding="utf-8"))
    params = dict(config["method_params"])
    params["confidence_threshold"] = float(config.get("confidence_threshold", 0.8))
    all_cases = make_cases(params)
    requested = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(all_cases))
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; expected {sorted(all_cases)}")

    features, labels, source = load_features(args)
    features, _, targets, num_classes = subsample(features, labels, args)
    print(f"Features: {source}")
    print(
        f"Graph: {len(features)} samples, {num_classes} classes, "
        f"{int(np.sum(targets != UNLABELED_TARGET))} labeled seeds, "
        f"n_neighbors={params['n_neighbors']}, gamma={params['gamma']}"
    )
    print(f"Device: {args.device}  torch={torch.__version__}")
    try:
        import cupy

        print(f"CuPy: {cupy.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"CuPy unavailable: {type(exc).__name__}: {exc}")

    record: dict = {
        "features": source,
        "samples": int(len(features)),
        "classes": int(num_classes),
        "k_shot": args.k_shot,
        "method_params": params,
        "cases": {},
    }

    for name in requested:
        case = all_cases[name]
        captured = capture_system(features, targets, num_classes, params, case)
        reference, reference_info = reference_solution(
            captured["matrix"], captured["right_hand_side"], args
        )

        results = {backend: run_backend(backend, captured, args) for backend in BACKENDS}
        comparisons: dict = {"per_backend": {}}
        for backend, result in results.items():
            if result.error:
                comparisons["per_backend"][backend] = {"error": result.error}
                continue
            residual = relative_residuals(
                captured["matrix"], captured["right_hand_side"], result.solution
            )
            comparisons["per_backend"][backend] = {
                "device": result.device,
                "right_hand_side_batch_size": result.batch_size,
                "repeat_bitwise_identical": result.repeat_bitwise_identical,
                "max_repeat_difference": result.max_repeat_difference,
                "seconds": result.seconds,
                "median_seconds": float(np.median(result.seconds)) if result.seconds else None,
                "max_relative_residual": float(residual.max()),
                "median_relative_residual": float(np.median(residual)),
                "median_iterations": int(np.median(result.iterations)),
                "max_iterations": int(result.iterations.max()),
                "max_absolute_error_vs_reference": float(
                    np.abs(result.solution - reference).max()
                ),
                "max_column_l2_error_vs_reference": float(
                    difference_summary(reference, result.solution)["max_column_l2_relative"]
                ),
            }

        left, right = results["torch_block_cg"], results["cupy_cg"]
        if not left.error and not right.error:
            comparisons["torch_vs_cupy"] = difference_summary(left.solution, right.solution)
            iteration_delta = np.abs(left.iterations - right.iterations)
            comparisons["iterations_identical"] = bool(np.array_equal(
                left.iterations, right.iterations
            ))
            comparisons["max_iteration_difference"] = int(iteration_delta.max())
            # Each backend already disagrees with itself across identical calls,
            # because cuSPARSE picks a reduction order at run time. A ratio near
            # or below one means the two backends differ no more than one
            # backend differs from itself.
            self_disagreement = max(
                left.max_repeat_difference,
                right.max_repeat_difference,
            )
            comparisons["worst_self_disagreement"] = self_disagreement
            comparisons["cross_backend_over_self"] = (
                comparisons["torch_vs_cupy"]["max_absolute"] / self_disagreement
                if self_disagreement > 0.0
                else float("inf")
            )

        end_to_end = (
            run_end_to_end(features, targets, num_classes, params, case, args)
            if args.end_to_end
            else {}
        )
        print(
            format_case_report(case, captured, reference_info, results, comparisons, end_to_end)
        )
        record["cases"][name] = {
            "case": {
                "alpha": case.alpha,
                "cg_rtol": case.cg_rtol,
                "cg_max_iter": case.cg_max_iter,
                "note": case.note,
            },
            "system": {
                "rows": int(captured["matrix"].shape[0]),
                "nnz": int(captured["matrix"].nnz),
                "right_hand_sides": int(captured["right_hand_side"].shape[1]),
                "build_seconds": captured["build_seconds"],
            },
            "reference": reference_info,
            "comparisons": comparisons,
            "end_to_end": {
                key: value
                for key, value in end_to_end.items()
                if key not in BACKENDS or isinstance(value, dict)
            },
        }

        if args.checks and name == requested[0]:
            batching = check_batch_invariance(captured, args)
            canonical = check_non_canonical_matrix(captured, args)
            record["checks"] = {"batching": batching, "non_canonical_csr": canonical}
            print("\n  structural checks (this case's system)")
            for backend, values in batching.items():
                if "error" in values:
                    print(f"    batching {backend:16} FAILED {values['error'][:60]}")
                    continue
                print(
                    f"    batching {backend:16} cap {values['default_cap']} vs 7: "
                    f"bitwise identical={values['bitwise_identical']}, "
                    f"max difference {values['max_absolute_difference']:.3e}"
                )
            print(
                f"    non-canonical CSR ({canonical['duplicate_entries']} duplicate entries, "
                f"sorted_indices={canonical['has_sorted_indices']}, same matrix to "
                f"{canonical['reconstruction_max_error']:.1e})"
            )
            for backend in BACKENDS:
                values = canonical.get(backend, {})
                if "error" in values:
                    print(f"      {backend:16} FAILED {values['error'][:60]}")
                    continue
                print(
                    f"      {backend:16} difference vs canonical input "
                    f"{values['max_absolute_difference']:.3e}, "
                    f"max relative residual {values['max_relative_residual']:.3e}"
                )

    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
