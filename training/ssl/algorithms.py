"""Pure graph construction and label-propagation algorithms."""

import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch
from loguru import logger
from scipy import sparse, special
from scipy.sparse import linalg as sparse_linalg

from .config import GRAPH_EDGE_ABLATION_MODES, UNLABELED_TARGET
from .graph_diagnostics import (
    maybe_save_graph_diagnostics,
    maybe_update_graph_propagation_diagnostics,
    numeric_diagnostic_summary,
)


FAISS_GPU_MAX_K = 2048
SOLVER_DIAGNOSTICS_MAX_RIGHT_HAND_SIDES = 500
CUPY_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH = 32
# B, X, R, P, and A@P account for five dense N x batch arrays. Reserve a
# sixth array's worth of memory for sparse-matmul workspaces and transient
# allocations, then use only half of currently free VRAM as a further guard
# against competing allocations from the training process.
CUPY_CG_DENSE_WORK_ARRAYS = 6
CUPY_CG_FREE_MEMORY_FRACTION = 0.5
TORCH_BLOCK_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH = 512
TORCH_BLOCK_CG_DENSE_WORK_ARRAYS = 8
TORCH_BLOCK_CG_FREE_MEMORY_FRACTION = 0.5
ENTROPY_CONFIDENCE_CPU_CHUNK_BYTES = 64 * 1024 * 1024
ENTROPY_CONFIDENCE_CUDA_CHUNK_BYTES = 256 * 1024 * 1024
ENTROPY_CONFIDENCE_CUDA_MIN_BYTES = 128 * 1024 * 1024
LINEAR_SOLVERS = ("cg", "cupy_cg", "torch_block_cg", "cholmod")
LINEAR_SOLVER_ALIASES = {
    "cupy": "cupy_cg",
    "block_cg": "torch_block_cg",
}
LINEAR_SOLVER_OPTIONS = frozenset(
    (*LINEAR_SOLVERS, *LINEAR_SOLVER_ALIASES)
)


def apply_graph_edge_ablation(
    adjacency,
    labels,
    *,
    mode="off",
    seed=0,
):
    """Audit or prune a symmetric graph using hidden true labels.

    This helper exists only for oracle ablations. The pruning modes never add
    or reweight an edge. ``random_pruned`` removes exactly as many undirected
    edges as ``oracle_pruned`` would remove, making it the graph-density
    control. ``degree_rewired`` keeps every edge and its weight but randomizes
    which nodes it joins (:func:`degree_preserving_rewire`), so the pool, the
    edge count, the weight multiset and every node's degree stay while the
    neighborhood structure the embedding produced is destroyed.
    """

    mode = str(mode)
    if mode not in GRAPH_EDGE_ABLATION_MODES:
        raise ValueError(
            "graph_edge_ablation must be one of "
            f"{sorted(GRAPH_EDGE_ABLATION_MODES)}"
        )
    if mode == "off":
        return adjacency, None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError("graph edge ablation seed must be an integer")
    if int(seed) < 0:
        raise ValueError("graph edge ablation seed must be non-negative")

    adjacency = sparse.csr_matrix(adjacency)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph edge ablation requires a square adjacency")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != adjacency.shape[0]:
        raise ValueError("graph edge ablation labels must align with graph nodes")
    if np.any(labels < 0):
        raise ValueError("graph edge ablation requires hidden true labels for every node")

    asymmetry = adjacency - adjacency.T
    if asymmetry.nnz and float(np.max(np.abs(asymmetry.data))) > 1e-12:
        raise ValueError("graph edge ablation requires a symmetric adjacency")

    upper = sparse.triu(adjacency, k=1, format="coo")
    edge_count = int(upper.nnz)
    if edge_count == 0:
        raise ValueError("graph edge ablation requires at least one edge")
    same_class = labels[upper.row] == labels[upper.col]
    same_class_count = int(same_class.sum())
    cross_class_count = edge_count - same_class_count

    keep = np.ones(edge_count, dtype=bool)
    if mode == "oracle_pruned":
        keep = same_class
    elif mode == "random_pruned" and cross_class_count:
        generator = np.random.default_rng(int(seed))
        removed = generator.choice(
            edge_count,
            size=cross_class_count,
            replace=False,
        )
        keep[removed] = False

    retained_count = int(keep.sum())
    if retained_count == 0:
        raise ValueError(
            f"graph edge ablation mode={mode!r} removed every graph edge"
        )

    retained_rows = upper.row[keep]
    retained_cols = upper.col[keep]
    retained_weights = upper.data[keep]
    rewire_stats = None
    if mode == "degree_rewired":
        retained_rows, retained_cols, rewire_stats = degree_preserving_rewire(
            retained_rows,
            retained_cols,
            adjacency.shape[0],
            seed=int(seed),
        )
    filtered = sparse.coo_matrix(
        (
            np.concatenate([retained_weights, retained_weights]),
            (
                np.concatenate([retained_rows, retained_cols]),
                np.concatenate([retained_cols, retained_rows]),
            ),
        ),
        shape=adjacency.shape,
        dtype=adjacency.dtype,
    ).tocsr()
    filtered.eliminate_zeros()

    edge_weight = np.asarray(upper.data, dtype=np.float64)
    original_weight = float(edge_weight.sum())
    same_class_weight = float(edge_weight[same_class].sum())
    # Audit the edges the graph ends up with: for the pruning modes these are
    # the kept originals, for ``degree_rewired`` the rewired ones.
    retained_same = labels[retained_rows] == labels[retained_cols]
    retained_edge_weight = np.asarray(retained_weights, dtype=np.float64)
    retained_same_class = int(retained_same.sum())
    retained_cross_class = retained_count - retained_same_class
    retained_weight = float(retained_edge_weight.sum())
    retained_same_class_weight = float(retained_edge_weight[retained_same].sum())
    degrees = np.asarray(filtered.sum(axis=1), dtype=np.float64).ravel()
    stats = {
        "mode": mode,
        "seed": int(seed) if mode in ("random_pruned", "degree_rewired") else None,
        "original_undirected_edges": edge_count,
        "same_class_undirected_edges": same_class_count,
        "cross_class_undirected_edges": cross_class_count,
        "edge_purity": same_class_count / float(edge_count),
        "cross_class_edge_rate": cross_class_count / float(edge_count),
        "original_undirected_weight": original_weight,
        "weighted_edge_purity": (
            same_class_weight / original_weight if original_weight > 0.0 else 0.0
        ),
        "retained_undirected_edges": retained_count,
        "removed_undirected_edges": edge_count - retained_count,
        "retained_edge_fraction": retained_count / float(edge_count),
        "retained_same_class_undirected_edges": retained_same_class,
        "retained_cross_class_undirected_edges": retained_cross_class,
        "retained_edge_purity": retained_same_class / float(retained_count),
        "retained_undirected_weight": retained_weight,
        "retained_weight_fraction": (
            retained_weight / original_weight if original_weight > 0.0 else 0.0
        ),
        "retained_weighted_edge_purity": (
            retained_same_class_weight / retained_weight
            if retained_weight > 0.0
            else 0.0
        ),
        "isolated_nodes": int(np.sum(degrees == 0.0)),
    }
    if rewire_stats is not None:
        stats.update(rewire_stats)
    return filtered, stats


# Double-edge swaps per undirected edge. A few times the edge count is the usual
# budget for a degree-preserving null model; the achieved count and the share of
# original edges that survived are reported so an under-mixed graph is visible.
DEGREE_REWIRE_SWAPS_PER_EDGE = 10
DEGREE_REWIRE_MAX_ROUNDS = 400
DEGREE_REWIRE_MAX_OCCUPANCY_BYTES = 64 * 1024 * 1024


def degree_preserving_rewire(
    rows,
    cols,
    num_nodes,
    *,
    seed=0,
    swaps_per_edge=DEGREE_REWIRE_SWAPS_PER_EDGE,
    max_rounds=DEGREE_REWIRE_MAX_ROUNDS,
):
    """Randomize a simple undirected graph by double-edge swaps.

    Each swap takes two edges ``(a, b)`` and ``(c, d)`` and replaces them with
    ``(a, d)`` and ``(c, b)`` (Maslov & Sneppen, 2002), rejecting self-loops and
    multi-edges, so every node keeps its exact degree and the graph stays
    simple. Edge ``i`` keeps its weight through every swap, so the weight
    multiset is unchanged too; a weighted degree is not, unless the weights are
    uniform (LRML's binary graph).

    Swaps run in vectorized rounds: each round pairs every edge with at most one
    partner and applies the proposals that collide neither with the current
    edge set nor with each other.

    Returns ``(rows, cols, stats)`` with ``rows[i], cols[i]`` the new endpoints
    of input edge ``i``.
    """

    u = np.asarray(rows, dtype=np.int64).copy()
    v = np.asarray(cols, dtype=np.int64).copy()
    edge_count = len(u)
    num_nodes = int(num_nodes)

    def keys(a, b):
        return np.minimum(a, b) * num_nodes + np.maximum(a, b)

    original_keys = keys(u, v)
    original_degrees = np.bincount(np.concatenate([u, v]), minlength=num_nodes)
    # Membership of a proposed edge in the current graph is the hot path. A flat
    # occupancy map answers it in O(1) (one byte per node pair, ~36 MB for the
    # 6k-node Cars pool); a larger graph falls back to a sorted search.
    occupied = None
    if num_nodes * num_nodes <= DEGREE_REWIRE_MAX_OCCUPANCY_BYTES:
        occupied = np.zeros(num_nodes * num_nodes, dtype=bool)
        occupied[original_keys] = True
    generator = np.random.default_rng(int(seed))
    target = int(swaps_per_edge) * edge_count
    swaps = 0
    rounds = 0
    idle_rounds = 0
    half = edge_count // 2
    while half and swaps < target and rounds < int(max_rounds) and idle_rounds < 3:
        rounds += 1
        order = generator.permutation(edge_count)
        first, second = order[:half], order[half : 2 * half]
        a, b = u[first], v[first]
        # A random orientation of the partner edge makes both rewirings of a
        # pair reachable: (a, d)(c, b) and (a, c)(d, b).
        flip = generator.random(half) < 0.5
        c = np.where(flip, v[second], u[second])
        d = np.where(flip, u[second], v[second])
        new_first, new_second = keys(a, d), keys(c, b)
        if occupied is not None:
            taken_first, taken_second = occupied[new_first], occupied[new_second]
        else:
            current = keys(u, v)
            taken_first, taken_second = np.isin(new_first, current), np.isin(new_second, current)
        accept = (a != d) & (c != b) & (new_first != new_second) & ~taken_first & ~taken_second
        proposed = np.concatenate([new_first[accept], new_second[accept]])
        values, counts = np.unique(proposed, return_counts=True)
        clashing = values[counts > 1]
        if clashing.size:
            accept &= ~np.isin(new_first, clashing) & ~np.isin(new_second, clashing)
        accepted = int(accept.sum())
        if accepted == 0:
            idle_rounds += 1
            continue
        idle_rounds = 0
        if occupied is not None:
            occupied[keys(a[accept], b[accept])] = False
            occupied[keys(c[accept], d[accept])] = False
            occupied[new_first[accept]] = True
            occupied[new_second[accept]] = True
        u[first[accept]], v[first[accept]] = a[accept], d[accept]
        u[second[accept]], v[second[accept]] = c[accept], b[accept]
        swaps += accepted

    degrees = np.bincount(np.concatenate([u, v]), minlength=num_nodes)
    if not np.array_equal(degrees, original_degrees):
        raise RuntimeError("degree-preserving rewiring changed a node degree")
    final_keys = keys(u, v)
    if len(np.unique(final_keys)) != edge_count or np.any(u == v):
        raise RuntimeError("degree-preserving rewiring produced a non-simple graph")
    stats = {
        "rewire_swaps": int(swaps),
        "rewire_swaps_per_edge": swaps / float(edge_count) if edge_count else 0.0,
        "rewire_target_swaps_per_edge": int(swaps_per_edge),
        "rewire_rounds": int(rounds),
        "rewire_original_edge_share": (
            float(np.isin(final_keys, original_keys).mean()) if edge_count else 0.0
        ),
    }
    return u, v, stats


def normalize_linear_solver(linear_solver):
    """Return the canonical configured sparse-solver name."""

    linear_solver = str(linear_solver)
    linear_solver = LINEAR_SOLVER_ALIASES.get(linear_solver, linear_solver)
    if linear_solver not in LINEAR_SOLVERS:
        choices = sorted(LINEAR_SOLVER_OPTIONS)
        raise ValueError(f"linear_solver must be one of {choices}")
    return linear_solver


_SSL_TIMING_LOGS_ENABLED = ContextVar(
    "ssl_timing_logs_enabled",
    default=True,
)
_SSL_COMPUTE_DEVICE = ContextVar(
    "ssl_compute_device",
    default=None,
)
#: Set once from ``--low_memory_mode``. Unlike evaluation retrieval, this module
#: builds a fresh ``StandardGpuResources`` per graph search, so the default
#: ~1.5 GiB device arena and 256 MiB pinned host buffer are re-reserved on every
#: pseudo-label refresh. Kept module-local so this file stays dependency-light;
#: ``engine.apply_low_memory_mode`` sets it alongside the evaluation policy.
_SSL_FAISS_LOW_MEMORY = ContextVar(
    "ssl_faiss_low_memory",
    default=None,
)


def set_ssl_faiss_low_memory(temp_memory_bytes):
    """Cap the per-search FAISS resource buffers; ``None`` keeps FAISS defaults."""

    if temp_memory_bytes is not None and int(temp_memory_bytes) < 0:
        raise ValueError("FAISS temp memory must be non-negative")
    _SSL_FAISS_LOW_MEMORY.set(
        None if temp_memory_bytes is None else int(temp_memory_bytes)
    )


def _apply_ssl_faiss_resource_policy(resources):
    """Size a freshly built ``StandardGpuResources`` for the active policy."""

    temp_memory_bytes = _SSL_FAISS_LOW_MEMORY.get()
    if temp_memory_bytes is None:
        return
    for method_name, size_bytes in (
        ("setTempMemory", temp_memory_bytes),
        # A per-search index build plus one search does not stream enough to
        # earn a persistent pinned staging area.
        ("setPinnedMemory", 0),
    ):
        setter = getattr(resources, method_name, None)
        if callable(setter):
            setter(int(size_bytes))


@contextmanager
def ssl_algorithm_device(device):
    """Route implicit FAISS/CuPy/Torch backends to one SSL device."""

    token = _SSL_COMPUTE_DEVICE.set(torch.device(device))
    try:
        yield
    finally:
        _SSL_COMPUTE_DEVICE.reset(token)


def _configured_ssl_device():
    return _SSL_COMPUTE_DEVICE.get()


def _log_debug_timing(operation, started_at, **details):
    """Emit one timing record at the application's console-visible level."""

    if not _SSL_TIMING_LOGS_ENABLED.get():
        return
    elapsed_seconds = time.perf_counter() - started_at
    detail_text = " ".join(
        f"{key}={value!r}" for key, value in details.items()
    )
    suffix = f" {detail_text}" if detail_text else ""
    # The application configures stdout and info.log at INFO; DEBUG is written
    # only to debug.log. Keep timing records visible in live server output.
    logger.info(
        f"SSL timing | operation={operation!r} "
        f"seconds={elapsed_seconds:.6f}{suffix}"
    )


@contextmanager
def suppress_ssl_timing_logs():
    """Avoid emitting whole-graph timing records for every local training graph."""

    token = _SSL_TIMING_LOGS_ENABLED.set(False)
    try:
        yield
    finally:
        _SSL_TIMING_LOGS_ENABLED.reset(token)


def _dependency(overrides, name, default):
    """Resolve a helper supplied by the orchestration façade, if any."""

    if overrides is None:
        return default
    return overrides.get(name, default)


def _make_affinity_with_graph_diagnostics(
    *,
    features,
    n_neighbors,
    gamma,
    graph_diagnostics,
    dependencies,
    graph_edge_ablation="off",
    graph_edge_ablation_labels=None,
    graph_edge_ablation_seed=0,
    graph_edge_ablation_name="graph",
):
    """Build an affinity and retain kNN/ablation details when enabled."""

    request = (
        None
        if graph_diagnostics is None
        else graph_diagnostics.get("request")
    )
    builder = _dependency(
        dependencies,
        "make_mixed_label_affinity",
        make_mixed_label_affinity,
    )
    graph_edge_ablation = str(graph_edge_ablation)
    if graph_edge_ablation not in GRAPH_EDGE_ABLATION_MODES:
        raise ValueError(
            "graph_edge_ablation must be one of "
            f"{sorted(GRAPH_EDGE_ABLATION_MODES)}"
        )
    if request is None and graph_edge_ablation == "off":
        return (
            builder(
                features,
                n_neighbors=n_neighbors,
                gamma=gamma,
            ),
            None,
            None,
        )

    result = builder(
        features,
        n_neighbors=n_neighbors,
        gamma=gamma,
        return_diagnostics=True,
    )
    if isinstance(result, tuple) and len(result) == 2:
        affinity, construction_metadata = result
    else:
        # Keep diagnostics compatible with a custom/facade affinity builder
        # that still implements the pre-diagnostics return contract.
        affinity = result
        construction_metadata = {
            "graph_kind": "positive_part_cosine_knn",
            "requested_n_neighbors": int(n_neighbors),
            "search_n_neighbors": min(
                int(n_neighbors),
                max(len(features) - 1, 0),
            ),
            "gamma": float(gamma),
        }
    affinity, ablation_stats = apply_graph_edge_ablation(
        affinity,
        graph_edge_ablation_labels,
        mode=graph_edge_ablation,
        seed=graph_edge_ablation_seed,
    )
    if ablation_stats is not None:
        construction_metadata = dict(construction_metadata)
        construction_metadata["graph_edge_ablation"] = ablation_stats
        logger.warning(
            f"{graph_edge_ablation_name} ORACLE ABLATION uses hidden true labels: "
            f"mode={graph_edge_ablation}, "
            f"raw_edge_purity={ablation_stats['edge_purity']:.3%}, "
            f"retained={ablation_stats['retained_undirected_edges']} / "
            f"{ablation_stats['original_undirected_edges']} edges, "
            f"retained_edge_purity={ablation_stats['retained_edge_purity']:.3%}, "
            f"isolated_nodes={ablation_stats['isolated_nodes']}"
        )
    if request is not None:
        _dependency(
            dependencies,
            "maybe_save_graph_diagnostics",
            maybe_save_graph_diagnostics,
        )(
            request=request,
            embeddings=features,
            adjacency=affinity,
            positions=graph_diagnostics.get("positions"),
            labels=graph_diagnostics.get("labels"),
            known_mask=graph_diagnostics.get("known_mask"),
            graph_metadata=construction_metadata,
        )
    return affinity, construction_metadata, request


def _record_graph_propagation_diagnostics(
    *,
    request,
    graph_diagnostics,
    scores,
    confidences,
    method,
    dependencies,
    solver_diagnostics=None,
    extra=None,
):
    if request is None:
        return
    _dependency(
        dependencies,
        "maybe_update_graph_propagation_diagnostics",
        maybe_update_graph_propagation_diagnostics,
    )(
        request=request,
        scores=scores,
        confidences=confidences,
        score_class_labels=graph_diagnostics.get("score_class_labels"),
        labels=graph_diagnostics.get("labels"),
        known_mask=graph_diagnostics.get("known_mask"),
        method=method,
        confidence_threshold=graph_diagnostics.get("confidence_threshold"),
        solver_diagnostics=solver_diagnostics,
        extra=extra,
    )


def _initialize_solver_diagnostics(
    diagnostics,
    *,
    matrix,
    right_hand_side,
    name,
    linear_solver,
    rtol,
    max_iter,
    warm_start,
    allow_nonconvergence,
):
    if diagnostics is None:
        return
    diagnostics.clear()
    diagonal = matrix.diagonal().astype(np.float64, copy=False)
    asymmetry = (matrix - matrix.T).tocsr()
    diagnostics.update(
        {
            "status": "running",
            "name": name,
            "backend": str(linear_solver),
            "matrix_shape": [int(value) for value in matrix.shape],
            "matrix_nnz": int(matrix.nnz),
            "matrix_memory_bytes": int(
                matrix.data.nbytes
                + matrix.indices.nbytes
                + matrix.indptr.nbytes
            ),
            "matrix_diagonal": numeric_diagnostic_summary(diagonal),
            "matrix_max_absolute_asymmetry": (
                0.0
                if asymmetry.nnz == 0
                else float(np.max(np.abs(asymmetry.data)))
            ),
            "right_hand_side_count": int(right_hand_side.shape[1]),
            "zero_right_hand_side_count": int(
                np.sum(~np.any(right_hand_side != 0.0, axis=0))
            ),
            "right_hand_side_l2_norm": numeric_diagnostic_summary(
                np.linalg.norm(right_hand_side, axis=0)
            ),
            "rtol": float(rtol),
            "max_iter": int(max_iter),
            "warm_start": warm_start is not None,
            "allow_nonconvergence": bool(allow_nonconvergence),
        }
    )


def _finish_solver_diagnostics(
    diagnostics,
    *,
    matrix,
    right_hand_side,
    solution,
    started_at,
):
    if diagnostics is None:
        return
    solution = np.asarray(solution, dtype=np.float64)
    residual = np.asarray(
        matrix @ solution - right_hand_side,
        dtype=np.float64,
    )
    absolute_residual = np.linalg.norm(residual, axis=0)
    rhs_norm = np.linalg.norm(right_hand_side, axis=0)
    relative_residual = np.divide(
        absolute_residual,
        rhs_norm,
        out=np.where(
            absolute_residual == 0.0,
            0.0,
            np.nan,
        ),
        where=rhs_norm > 0.0,
    )
    diagnostics.update(
        {
            "status": "computed",
            "elapsed_seconds": float(time.perf_counter() - started_at),
            "solution_finite": bool(np.all(np.isfinite(solution))),
            "solution_l2_norm": numeric_diagnostic_summary(
                np.linalg.norm(solution, axis=0)
            ),
            "absolute_residual_l2": numeric_diagnostic_summary(
                absolute_residual
            ),
            "relative_residual_l2": numeric_diagnostic_summary(
                relative_residual
            ),
            "max_absolute_residual_entry": (
                0.0
                if residual.size == 0
                else float(np.max(np.abs(residual)))
            ),
        }
    )


def require_faiss(purpose):
    """Import a FAISS build with an actionable package hint on failure."""

    try:
        import faiss
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"{purpose} requires FAISS; install faiss-gpu-cu12 on a supported "
            "CUDA 12 host or faiss-cpu on a CPU-only host"
        ) from exc
    return faiss


def _faiss_gpu_device_id(faiss, purpose):
    """Return the active FAISS GPU ID, or ``None`` for a CPU-only build/host."""

    configured_device = _configured_ssl_device()
    if configured_device is not None and configured_device.type != "cuda":
        return None

    gpu_api = ("get_num_gpus", "StandardGpuResources", "index_cpu_to_gpu")
    if not all(hasattr(faiss, name) for name in gpu_api):
        return None
    try:
        gpu_count = int(faiss.get_num_gpus())
    except Exception as exc:
        logger.warning(f"{purpose}: FAISS GPU discovery failed ({exc}); using CPU")
        return None
    if gpu_count <= 0:
        return None

    # Follow the CUDA device selected by the training process. CUDA_VISIBLE_DEVICES
    # remaps both PyTorch and FAISS device IDs, so this also behaves correctly in
    # the usual one-process-per-GPU distributed setup.
    try:
        if torch.cuda.is_available():
            device_id = (
                int(torch.cuda.current_device())
                if configured_device is None or configured_device.index is None
                else int(configured_device.index)
            )
            if 0 <= device_id < gpu_count:
                return device_id
    except Exception as exc:
        logger.debug(f"{purpose}: could not read PyTorch's active CUDA device ({exc})")
    return 0


def faiss_gpu_flat_index(faiss, dim, purpose, *, spherical=False):
    """Return ``(gpu_index, resources)`` for a flat index, or ``None`` for CPU.

    ``resources`` must outlive the index: a FAISS GPU index does not own its
    ``StandardGpuResources``, and releasing them first crashes the process. The
    caller therefore keeps both, which is also what lets the SSL low-memory
    policy size the scratch arena -- ``faiss.Kmeans(gpu=True)`` builds its own
    resources through ``index_cpu_to_all_gpus`` and cannot be capped, which
    measured ~1.25 GiB of extra device memory per caller.
    """

    device_id = _faiss_gpu_device_id(faiss, purpose)
    if device_id is None:
        return None
    try:
        resources = faiss.StandardGpuResources()
        _apply_ssl_faiss_resource_policy(resources)
        cpu_index = faiss.IndexFlatIP(dim) if spherical else faiss.IndexFlatL2(dim)
        return faiss.index_cpu_to_gpu(resources, device_id, cpu_index), resources
    except Exception as exc:
        logger.warning(f"{purpose}: FAISS GPU index unavailable ({exc}); using CPU")
        return None


def _faiss_flat_search(database, queries, k, purpose, faiss_module, prefer_gpu, index_type):
    """Exact flat-index search on GPU whenever FAISS supports it, else on CPU.

    ``faiss-gpu-cu12`` contains the CPU indexes too, which makes the CPU retry
    usable when CUDA is unavailable or a GPU allocation/search fails. GPU FAISS
    only supports ``k <= 2048``, so larger exact searches stay on the CPU.
    """

    database = np.ascontiguousarray(database, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if database.ndim != 2 or database.shape[1] == 0 or len(database) == 0:
        raise ValueError("FAISS database must be a non-empty feature matrix")
    if queries.ndim != 2 or queries.shape[1] != database.shape[1]:
        raise ValueError("FAISS queries must be a feature matrix matching the database")
    k = int(k)
    if k <= 0 or k > len(database):
        raise ValueError("FAISS k must be positive and no larger than the database")

    faiss = require_faiss(purpose) if faiss_module is None else faiss_module
    cpu_index = getattr(faiss, index_type)(database.shape[1])
    gpu_device_id = _faiss_gpu_device_id(faiss, purpose) if prefer_gpu else None
    if gpu_device_id is not None and k <= FAISS_GPU_MAX_K:
        gpu_started_at = time.perf_counter()
        try:
            # Keep resources alive until search has copied its results back to
            # host memory. GPU indexes do not own StandardGpuResources.
            gpu_resources = faiss.StandardGpuResources()
            _apply_ssl_faiss_resource_policy(gpu_resources)
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_device_id, cpu_index)
            gpu_index.add(database)
            return gpu_index.search(queries, k)
        except Exception as exc:
            _log_debug_timing(
                "faiss_flat_ip_search.backend",
                gpu_started_at,
                purpose=purpose,
                backend=f"cuda:{gpu_device_id}",
                database_size=len(database),
                query_size=len(queries),
                k=k,
                outcome="failed",
            )
            logger.warning(
                f"{purpose}: FAISS GPU search failed on CUDA device {gpu_device_id} "
                f"({exc}); retrying on CPU"
            )
    elif gpu_device_id is not None:
        logger.debug(
            f"{purpose}: requested k={k} exceeds the FAISS GPU limit "
            f"of {FAISS_GPU_MAX_K}; using CPU"
        )

    cpu_index.add(database)
    return cpu_index.search(queries, k)


def faiss_flat_ip_search(database, queries, k, purpose, faiss_module=None, *, prefer_gpu=True):
    """Run exact inner-product search, on GPU whenever FAISS supports it."""

    return _faiss_flat_search(database, queries, k, purpose, faiss_module, prefer_gpu, "IndexFlatIP")


def select_self_training_candidates(
    distances,
    predicted_labels,
    selection_fraction,
    *,
    strategy="per_predicted_class",
    eligible=None,
):
    """Select the closest 1-NN predictions for one self-training iteration.

    ``global`` implements the literal top-distance step in Algorithm 1 of
    Sahito et al. ``per_predicted_class`` implements the stratified variant in
    the authors' reference code: the integer selection budget is divided
    equally over the classes predicted in the current unlabeled pool.

    Returned indices address the original ``distances``/``predicted_labels``
    arrays. Stable distance ordering makes equal-distance selections
    deterministic.
    """

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    predicted_labels = np.asarray(predicted_labels).reshape(-1)
    if len(distances) != len(predicted_labels):
        raise ValueError("distances and predicted_labels must be aligned")
    if np.any(~np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("self-training distances must be finite and non-negative")

    selection_fraction = float(selection_fraction)
    if not np.isfinite(selection_fraction) or not (0.0 < selection_fraction <= 1.0):
        raise ValueError("selection_fraction must be in (0, 1]")
    if strategy not in {"global", "per_predicted_class"}:
        raise ValueError(
            "selection_strategy must be one of ['global', 'per_predicted_class']"
        )

    if eligible is None:
        eligible = np.ones(len(distances), dtype=bool)
    else:
        eligible = np.asarray(eligible, dtype=bool).reshape(-1)
        if len(eligible) != len(distances):
            raise ValueError("eligible must be aligned with distances")

    selection_budget = int(len(distances) * selection_fraction)
    if selection_budget == 0 or not np.any(eligible):
        return np.empty(0, dtype=np.int64)

    distance_order = np.argsort(distances, kind="stable")
    if strategy == "global":
        return distance_order[eligible[distance_order]][:selection_budget].astype(
            np.int64,
            copy=False,
        )

    predicted_classes = np.unique(predicted_labels)
    per_class = selection_budget // len(predicted_classes)
    if per_class == 0:
        return np.empty(0, dtype=np.int64)

    selected = []
    for predicted_class in predicted_classes:
        class_order = distance_order[
            eligible[distance_order]
            & (predicted_labels[distance_order] == predicted_class)
        ]
        selected.append(class_order[:per_class])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected).astype(np.int64, copy=False)


def symmetric_normalized_affinity(affinity, degrees):
    """``D^-1/2 W D^-1/2``; an isolated node keeps an all-zero row and column."""

    inverse_sqrt_degrees = np.zeros_like(degrees, dtype=np.float64)
    positive_degree = degrees > 0.0
    inverse_sqrt_degrees[positive_degree] = 1.0 / np.sqrt(degrees[positive_degree])
    degree_scaling = sparse.diags(inverse_sqrt_degrees)
    return (degree_scaling @ affinity @ degree_scaling).tocsr()


def faiss_label_spreading(
    features,
    targets,
    num_classes,
    n_neighbors=10,
    gamma=1.0,
    alpha=0.2,
    cg_rtol=1e-5,
    cg_max_iter=1000,
    linear_solver="cg",
    graph_diagnostics=None,
    graph_edge_ablation="off",
    graph_edge_ablation_labels=None,
    graph_edge_ablation_seed=0,
    _dependencies=None,
):
    """Solve Zhou et al.'s label-spreading fixed point on a FAISS kNN graph."""

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be a matrix aligned with targets")
    if len(features) < 2:
        raise ValueError("faiss_label_spreading requires at least two samples")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    labeled = targets != UNLABELED_TARGET
    if not np.any(labeled):
        raise ValueError("faiss_label_spreading requires at least one labeled target")
    if np.any((targets[labeled] < 0) | (targets[labeled] >= num_classes)):
        raise ValueError("labeled targets must be in [0, num_classes)")

    affinity, _, diagnostic_request = _make_affinity_with_graph_diagnostics(
        features=features,
        n_neighbors=n_neighbors,
        gamma=gamma,
        graph_diagnostics=graph_diagnostics,
        dependencies=_dependencies,
        graph_edge_ablation=graph_edge_ablation,
        graph_edge_ablation_labels=graph_edge_ablation_labels,
        graph_edge_ablation_seed=graph_edge_ablation_seed,
        graph_edge_ablation_name="faiss_label_spreading",
    )

    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    normalized_affinity = symmetric_normalized_affinity(affinity, degrees)

    alpha = float(alpha)
    system = (
        sparse.eye(len(features), format="csr", dtype=np.float64)
        - alpha * normalized_affinity
    ).tocsr()
    one_hot_targets = np.zeros((len(features), num_classes), dtype=np.float64)
    one_hot_targets[np.flatnonzero(labeled), targets[labeled]] = 1.0

    solver_diagnostics = {} if diagnostic_request is not None else None
    solve_kwargs = {}
    if solver_diagnostics is not None:
        solve_kwargs["diagnostics"] = solver_diagnostics
    scores = _dependency(
        _dependencies,
        "solve_sparse_label_system",
        solve_sparse_label_system,
    )(
        system,
        (1.0 - alpha) * one_hot_targets,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="faiss label spreading",
        linear_solver=str(linear_solver),
        **solve_kwargs,
    )
    probabilities = _dependency(
        _dependencies,
        "normalize_label_spreading_rows",
        normalize_label_spreading_rows,
    )(scores)
    confidences = probabilities.max(axis=1)
    _record_graph_propagation_diagnostics(
        request=diagnostic_request,
        graph_diagnostics=graph_diagnostics,
        scores=probabilities,
        confidences=confidences,
        method="faiss_label_spreading",
        dependencies=_dependencies,
        solver_diagnostics=solver_diagnostics,
        extra={
            "alpha": alpha,
            "normalized_affinity_nnz": int(normalized_affinity.nnz),
        },
    )
    return probabilities.astype(np.float32), confidences.astype(np.float32)


def iscen_label_spreading(
    features,
    targets,
    num_classes,
    n_neighbors=50,
    gamma=3.0,
    alpha=0.99,
    cg_rtol=1e-6,
    cg_max_iter=20,
    linear_solver="cg",
    graph_diagnostics=None,
    graph_edge_ablation="off",
    graph_edge_ablation_labels=None,
    graph_edge_ablation_seed=0,
    _dependencies=None,
):
    """Run the LP-DeepSSL diffusion and entropy-certainty calculation.

    This follows Iscen et al. (CVPR 2019) and their reference implementation:
    cosine kNN affinities are symmetrized and degree-normalized, class-balanced
    label seeds are diffused with the reference truncated conjugate-gradient
    solve (or optional exact CHOLMOD solve), and the propagated rows are
    converted to entropy-based certainty weights.
    """

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if features.ndim != 2 or features.shape[1] == 0 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features must be a matrix aligned with targets")
    if len(features) < 2:
        raise ValueError("iscen_label_spreading requires at least two samples")
    if not np.all(np.isfinite(features)):
        raise ValueError("iscen_label_spreading features must be finite")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    if int(n_neighbors) <= 0:
        raise ValueError("iscen_label_spreading n_neighbors must be positive")
    if not np.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("iscen_label_spreading gamma must be finite and positive")
    if not np.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError("iscen_label_spreading alpha must be in (0, 1)")
    if not np.isfinite(float(cg_rtol)) or float(cg_rtol) <= 0.0:
        raise ValueError("iscen_label_spreading cg_rtol must be finite and positive")
    if int(cg_max_iter) <= 0:
        raise ValueError("iscen_label_spreading cg_max_iter must be positive")
    try:
        linear_solver = normalize_linear_solver(linear_solver)
    except ValueError as exc:
        raise ValueError(f"iscen_label_spreading {exc}") from exc
    labeled = targets != UNLABELED_TARGET
    if not np.any(labeled):
        raise ValueError("iscen_label_spreading requires at least one labeled target")
    if np.any((targets[labeled] < 0) | (targets[labeled] >= int(num_classes))):
        raise ValueError("labeled targets must be in [0, num_classes)")

    affinity, _, diagnostic_request = _make_affinity_with_graph_diagnostics(
        features=features,
        n_neighbors=int(n_neighbors),
        gamma=float(gamma),
        graph_diagnostics=graph_diagnostics,
        dependencies=_dependencies,
        graph_edge_ablation=graph_edge_ablation,
        graph_edge_ablation_labels=graph_edge_ablation_labels,
        graph_edge_ablation_seed=graph_edge_ablation_seed,
        graph_edge_ablation_name="iscen_label_spreading",
    )

    # A disconnected component without a labeled node has no source term in
    # any class-specific linear system. Its exact propagated score is therefore
    # identically zero. Exclude those components from the solve and preserve
    # all-zero output rows as an explicit "no pseudo-label distribution"
    # sentinel for the orchestration adapter.
    component_count, component_ids = sparse.csgraph.connected_components(
        affinity,
        directed=False,
        return_labels=True,
    )
    component_has_labeled = np.zeros(component_count, dtype=bool)
    component_has_labeled[np.unique(component_ids[labeled])] = True
    seed_reachable = component_has_labeled[component_ids]
    active_indices = np.flatnonzero(seed_reachable)
    seedless_unlabeled = (~labeled) & (~seed_reachable)

    degrees = np.asarray(affinity.sum(axis=1), dtype=np.float64).ravel()
    seedless_count = int(seedless_unlabeled.sum())
    if seedless_count > 0:
        seedless_components = int(np.unique(component_ids[seedless_unlabeled]).size)
        zero_degree_count = int(np.sum(seedless_unlabeled & (degrees == 0.0)))
        logger.warning(
            "Iscen label spreading found "
            f"{seedless_count} unlabeled candidates in {seedless_components} "
            "graph components without a labeled target; marking them "
            "unpropagatable so pseudo-label training omits them "
            f"(zero_degree={zero_degree_count})"
        )

    active_affinity = affinity[active_indices][:, active_indices].tocsr()
    normalized_affinity = symmetric_normalized_affinity(active_affinity, degrees[active_indices])
    system = (
        sparse.eye(len(active_indices), format="csr", dtype=np.float64)
        - float(alpha) * normalized_affinity
    ).tocsr()

    # The public LP-DeepSSL implementation gives every class unit total seed
    # mass. This is a deliberate reference-code detail beyond paper equation
    # (5), and prevents classes with more labeled examples from dominating the
    # diffusion before the later class-balanced training sampler is applied.
    active_targets = targets[active_indices]
    active_labeled = active_targets != UNLABELED_TARGET
    labeled_indices = np.flatnonzero(active_labeled)
    labeled_targets = active_targets[active_labeled]
    class_seed_counts = np.bincount(labeled_targets, minlength=int(num_classes)).astype(np.float64)
    one_hot_targets = np.zeros((len(active_indices), int(num_classes)), dtype=np.float64)
    one_hot_targets[labeled_indices, labeled_targets] = 1.0 / class_seed_counts[labeled_targets]


    solver_diagnostics = {} if diagnostic_request is not None else None
    solve_kwargs = {}
    if solver_diagnostics is not None:
        solve_kwargs["diagnostics"] = solver_diagnostics
    scores = _dependency(
        _dependencies,
        "solve_sparse_label_system",
        solve_sparse_label_system,
    )(
        system,
        one_hot_targets,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="Iscen label spreading",
        linear_solver=linear_solver,
        # The public LP-DeepSSL implementation uses the final iterate when its
        # reference limit of 20 CG iterations is reached.
        allow_nonconvergence=linear_solver != "cholmod",
        **solve_kwargs,
    )

    # A finite truncated CG solve can contain negative numerical overshoot.
    # Clamp either solver's output consistently before row normalization.
    active_nonnegative_scores = np.asarray(scores, dtype=np.float64)
    if not active_nonnegative_scores.flags.writeable:
        active_nonnegative_scores = active_nonnegative_scores.copy()
    np.maximum(
        active_nonnegative_scores,
        0.0,
        out=active_nonnegative_scores,
    )
    active_row_masses = active_nonnegative_scores.sum(axis=1)
    positive_mass = active_row_masses > 0.0
    zero_mass_active_indices = active_indices[~positive_mass]
    if len(zero_mass_active_indices) > 0:
        logger.warning(
            "Iscen label spreading produced "
            f"{len(zero_mass_active_indices)} additional zero-mass rows inside "
            "seed-reachable graph components after the sparse solve; marking "
            "them unpropagatable. Increase cg_max_iter or use "
            "linear_solver='cholmod' "
            "if this warning recurs"
        )

    # Rows without positive mass remain exactly zero. The adapter recognizes
    # that sentinel and removes those candidates before argmax, including when
    # the configured confidence threshold is zero.
    positive_mass_indices = active_indices[positive_mass]
    normalized_positive_probabilities = None
    if len(positive_mass_indices) > 0:
        positive_scores = (
            active_nonnegative_scores
            if np.all(positive_mass)
            else active_nonnegative_scores[positive_mass]
        )
        normalized_positive_probabilities = _dependency(
            _dependencies,
            "normalize_label_spreading_rows",
            normalize_label_spreading_rows,
        )(positive_scores)

    if len(positive_mass_indices) == len(features):
        probabilities = normalized_positive_probabilities
    else:
        probabilities = np.zeros(
            (len(features), int(num_classes)),
            dtype=np.float64,
        )
        if len(positive_mass_indices) > 0:
            probabilities[positive_mass_indices] = (
                normalized_positive_probabilities
            )

    confidences = np.zeros(len(features), dtype=np.float64)
    if len(positive_mass_indices) > 0:
        entropy_helper = _dependency(
            _dependencies,
            "entropy_confidence",
            entropy_confidence,
        )
        entropy_kwargs = (
            {"validate": False}
            if entropy_helper is entropy_confidence
            else {}
        )
        confidences[positive_mass_indices] = entropy_helper(
            normalized_positive_probabilities,
            **entropy_kwargs,
        )
    max_confidence = float(np.max(confidences))
    if max_confidence > 0.0:
        # TODO: CONSIDER THIS.
        # we might not want to use this???   / max_confidence
        confidences = confidences / max_confidence # consider:  / max_confidence, otherwise this is basically graded on a curve, no? It barely makes a difference. like +- 0.2 map@r.
    else:
        # Uniform predictions carry no information. Keep their certainty at
        # zero rather than reproducing the reference implementation's 0 / 0.
        confidences = np.zeros_like(confidences)
    confidences[labeled] = 1.0
    _record_graph_propagation_diagnostics(
        request=diagnostic_request,
        graph_diagnostics=graph_diagnostics,
        scores=probabilities,
        confidences=confidences,
        method="iscen_label_spreading",
        dependencies=_dependencies,
        solver_diagnostics=solver_diagnostics,
        extra={
            "alpha": float(alpha),
            "component_count": int(component_count),
            "seed_reachable_node_count": int(seed_reachable.sum()),
            "seedless_unlabeled_node_count": seedless_count,
            "active_zero_mass_node_count": int(len(zero_mass_active_indices)),
            "normalized_affinity_nnz": int(normalized_affinity.nnz),
        },
    )
    return probabilities.astype(np.float32), confidences.astype(np.float32)


def _find_lrml_knn_neighbors(
    embeddings,
    n_neighbors,
    *,
    prefer_gpu=True,
    return_similarities=False,
):
    """Return exact non-self cosine neighbors for each LRML graph node.

    The rows are L2-normalized here, the way ``make_mixed_label_affinity``
    normalizes for Iscen, so ``IndexFlatIP`` ranks by cosine rather than by raw
    inner product. Callers that already hand over unit-norm embeddings -- the
    supervised path does, because the projection head ends in ``F.normalize`` --
    are unaffected. Callers that do not, such as the in-batch graph and the
    benchmark scripts, used to get an inner-product ranking here while the Iscen
    arm they were being compared against got a cosine one. Measured on raw
    Cars196 backbone rows, whose norms span only 45 to 49, that disagreement
    already moved 13% of the edges.
    """

    features = np.ascontiguousarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("lrml embeddings must be a non-empty feature matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("lrml embeddings must be finite")
    # normalize_L2 works in place and ascontiguousarray can alias its argument,
    # so copy rather than rescaling the caller's matrix underneath it.
    features = features.copy()
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms <= 1e-12)
        # normalize_L2 leaves these at zero rather than raising, and a zero row
        # ties at inner product 0 against every other node, so its whole
        # neighborhood would be arbitrary.
        raise ValueError(
            "lrml embeddings contain zero-norm vectors at indices "
            f"{bad[:10].tolist()}"
        )
    require_faiss("lrml regularization").normalize_L2(features)
    num_nodes = len(features)
    k = min(int(n_neighbors), num_nodes - 1)
    if k <= 0:
        raise ValueError("lrml graph needs at least two samples and one neighbor")

    # Query k + 1 because the indexed database contains each query itself. Filter
    # by node ID below instead of assuming ties always leave self in column zero.
    search_kwargs = {} if prefer_gpu else {"prefer_gpu": False}
    similarities, neighbors = faiss_flat_ip_search(
        database=features,
        queries=features,
        k=k + 1,
        purpose="lrml regularization",
        **search_kwargs,
    )

    neighbor_indices = np.empty((num_nodes, k), dtype=np.int64)
    neighbor_similarities = (
        np.empty((num_nodes, k), dtype=np.float32)
        if return_similarities
        else None
    )
    for node, (neighbor_row, similarity_row) in enumerate(
        zip(neighbors, similarities)
    ):
        kept = 0
        for neighbor, similarity in zip(neighbor_row, similarity_row):
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            neighbor_indices[node, kept] = neighbor
            if neighbor_similarities is not None:
                neighbor_similarities[node, kept] = similarity
            kept += 1
            if kept == k:
                break
        if kept != k:
            raise RuntimeError(
                f"FAISS returned only {kept} non-self LRML neighbors for node {node}; "
                f"expected {k}"
            )
    if neighbor_similarities is not None:
        return neighbor_indices, neighbor_similarities
    return neighbor_indices


def build_lrml_knn_graph(embeddings, n_neighbors, *, prefer_gpu=True):
    """Build the binary symmetric cosine-kNN graph of Hoi et al.

    The rows are L2-normalized during the search, so this and
    ``make_mixed_label_affinity`` select the same neighbors from the same
    embeddings; the two differ only in what they write on the resulting edges.
    """

    neighbor_kwargs = {} if prefer_gpu else {"prefer_gpu": False}
    neighbor_indices = _find_lrml_knn_neighbors(
        embeddings,
        n_neighbors,
        **neighbor_kwargs,
    )
    num_nodes, k = neighbor_indices.shape
    assert not np.any(
        neighbor_indices == np.arange(num_nodes, dtype=np.int64)[:, None]
    ), "LRML neighbor search must exclude self-matches"
    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), k)
    cols = neighbor_indices.reshape(-1)

    directed = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    # W_ij = 1 if x_i in N(x_j) OR x_j in N(x_i): union of the directed graph and
    # its transpose, clipped back to a binary adjacency.
    symmetric = (directed + directed.T).tocsr()
    symmetric.data[:] = 1.0
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    assert symmetric.diagonal().sum() == 0, (
        "LRML adjacency must not contain self-loops"
    )
    degrees = np.asarray(symmetric.sum(axis=1), dtype=np.float64).ravel()
    return neighbor_indices, symmetric, degrees


def build_manifold_preserving_graph(
    embeddings,
    n_neighbors,
    similarity_v=10.0,
    density_bandwidth=None,
    *,
    prefer_gpu=True,
):
    """Build Ying et al.'s density-weighted Gaussian kNN graph.

    This is the regularity term in Equations (2), (7), and (8) of
    *Manifold Preserving: An Intrinsic Approach for Semisupervised Distance
    Metric Learning* (Ying et al., 2018):

        Reg = sum_i beta_i sum_{j in N(i)} S_ij ||z_i - z_j||^2,
        S_ij = exp(-||z_i-z_j||^2 / (2 sigma^2)).

    ``beta_i`` is the paper's Parzen-window density estimate, normalized by
    the largest estimate.  Constants common to every density cancel in that
    normalization, so it is evaluated stably as a Gaussian log-mean.  The
    paper leaves its density bandwidth ``h`` unspecified; ``None`` uses the
    graph similarity bandwidth so both kernels operate on one local scale.

    The paper's W = diag(beta) S * N is directed.  Downstream LRML training
    samples undirected edges, so this function stores ``W + W.T``.  This is an
    exact representation rather than a graph change because squared distance
    is symmetric:

        sum_ij W_ij d_ij^2 = sum_{i<j} (W_ij + W_ji) d_ij^2.

    Graph features are L2-normalized before Euclidean distances are measured,
    matching the cosine-neighbor convention used by the deep LRML adapter.
    For unit rows, nearest cosine and nearest Euclidean neighbors coincide.
    """

    similarity_v = float(similarity_v)
    if not np.isfinite(similarity_v) or similarity_v <= 0:
        raise ValueError(
            "manifold-preserving similarity_v must be finite and positive"
        )
    if density_bandwidth is not None:
        density_bandwidth = float(density_bandwidth)
        if not np.isfinite(density_bandwidth) or density_bandwidth <= 0:
            raise ValueError(
                "manifold-preserving density_bandwidth must be finite and positive"
            )

    features = np.ascontiguousarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(
            "manifold-preserving embeddings must be a non-empty feature matrix"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("manifold-preserving embeddings must be finite")
    features = features.copy()
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        bad = np.flatnonzero(norms <= 1e-12)
        raise ValueError(
            "manifold-preserving embeddings contain zero-norm vectors at indices "
            f"{bad[:10].tolist()}"
        )
    faiss = require_faiss("manifold-preserving LRML regularization")
    faiss.normalize_L2(features)

    neighbor_indices, neighbor_cosines = _find_lrml_knn_neighbors(
        features,
        n_neighbors,
        prefer_gpu=prefer_gpu,
        return_similarities=True,
    )
    # On the unit sphere, ||x-y||^2 = 2 - 2<x,y>.  Clamp the tiny numerical
    # excursions FAISS can return around cosine +/-1.
    neighbor_squared_distances = np.clip(
        2.0 - 2.0 * neighbor_cosines.astype(np.float64),
        0.0,
        4.0,
    )

    # The closest distinct pair is present in every kNN graph with k >= 1.
    minimum_distance = float(np.sqrt(neighbor_squared_distances.min()))
    # Find the farthest pair exactly without materializing an O(n^2) distance
    # matrix: max_y <-x,y> is -min_y <x,y>.
    search_kwargs = {} if prefer_gpu else {"prefer_gpu": False}
    opposite_similarities, _ = faiss_flat_ip_search(
        database=features,
        queries=np.ascontiguousarray(-features),
        k=1,
        purpose="manifold-preserving LRML bandwidth",
        faiss_module=faiss,
        **search_kwargs,
    )
    farthest_squared_distances = np.clip(
        2.0 + 2.0 * opposite_similarities[:, 0].astype(np.float64),
        0.0,
        4.0,
    )
    maximum_distance = float(np.sqrt(farthest_squared_distances.max()))

    # Section IV: sigma = min(D) + (max(D) - min(D)) / v.  Identical rows make
    # both extrema zero; an arbitrary positive scale then correctly gives all
    # Gaussian similarities the limiting value one.
    similarity_sigma = minimum_distance + (
        maximum_distance - minimum_distance
    ) / similarity_v
    effective_sigma = max(float(similarity_sigma), np.finfo(np.float64).eps)
    effective_density_bandwidth = (
        effective_sigma
        if density_bandwidth is None
        else float(density_bandwidth)
    )

    density_log_kernel = -neighbor_squared_distances / (
        2.0 * effective_density_bandwidth**2
    )
    log_density = special.logsumexp(density_log_kernel, axis=1) - np.log(
        neighbor_squared_distances.shape[1]
    )
    density_weights = np.exp(log_density - log_density.max())

    similarities = np.exp(
        -neighbor_squared_distances / (2.0 * effective_sigma**2)
    )
    directed_values = density_weights[:, None] * similarities
    num_nodes, k = neighbor_indices.shape
    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), k)
    cols = neighbor_indices.reshape(-1)
    directed = sparse.coo_matrix(
        (directed_values.reshape(-1), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    adjacency = (directed + directed.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    if adjacency.nnz == 0:
        raise ValueError(
            "manifold-preserving graph has no positive Gaussian-weighted edges"
        )
    degrees = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()

    # beta_i and S_ij are both Gaussians, so their product spans twice the
    # decades either one does, and a sigma well below the typical neighbor
    # distance drives the tail past what float32 can hold. The training energy
    # is evaluated in float32, so those edges contribute exactly nothing there;
    # if a whole sampled edge batch lands in the tail, the regularizer gradient
    # is exactly zero and any ratio measured against it has no denominator.
    # Report it here, where sigma is still in scope, rather than letting it
    # surface as a silently inert regularizer.
    edge_weights = adjacency.tocoo().data
    underflowed = int(np.count_nonzero(edge_weights.astype(np.float32) == 0.0))
    underflow_fraction = underflowed / float(len(edge_weights))
    if underflowed:
        logger.warning(
            "manifold-preserving graph: "
            f"{underflow_fraction:.2%} of edge weights ({underflowed} of "
            f"{len(edge_weights)}) underflow to zero in float32 at "
            f"similarity_v={similarity_v:g} (sigma={effective_sigma:.4g}, "
            f"h={effective_density_bandwidth:.4g}). Those edges cannot "
            "contribute to the training energy; lower similarity_v or raise "
            "density_bandwidth to keep the graph live"
        )
    metadata = {
        "distance_min": minimum_distance,
        "distance_max": maximum_distance,
        "similarity_v": similarity_v,
        "similarity_sigma": effective_sigma,
        "density_bandwidth": effective_density_bandwidth,
        "density_weight_min": float(density_weights.min()),
        "density_weight_mean": float(density_weights.mean()),
        "density_weight_max": float(density_weights.max()),
        "edge_weight_min": float(edge_weights.min()),
        "edge_weight_max": float(edge_weights.max()),
        "float32_underflow_edge_fraction": underflow_fraction,
    }
    return neighbor_indices, adjacency, degrees, metadata


# --------------------------------------------------------------------------
# Locally linear reconstruction, Soleymani Baghshah and Bagheri Shouraki
# (IJCAI 2009), "Semi-Supervised Metric Learning Using Pairwise Constraints".
# --------------------------------------------------------------------------

LLE_RECONSTRUCTION_METRICS = ("euclidean", "cosine")

# Roweis and Saul's conditioner for eq. (5)'s local Gram matrix: C <- C +
# delta^2 * trace(C) * I / k. It only binds when the k local difference vectors
# do not span their own space, which for k > d is always, and it is the
# regularization the LLE paper prescribes rather than a numerical afterthought.
DEFAULT_LLE_RECONSTRUCTION_REGULARIZATION = 1e-3


def faiss_flat_l2_search(database, queries, k, purpose, faiss_module=None, *, prefer_gpu=True):
    """Exact squared-Euclidean search, mirroring :func:`faiss_flat_ip_search`.

    Locally linear reconstruction fits each point by least squares in the input
    space, so the neighborhood eq. (5) selects is the Euclidean one. On
    unit-norm rows this ranks identically to the inner-product search, because
    ``||a - b||^2 = 2 - 2 cos`` is monotone in the cosine; it exists for the
    rows that are *not* normalized, which is exactly the case the paper's ``X``
    is in and which raw backbone features are in.
    """

    return _faiss_flat_search(database, queries, k, purpose, faiss_module, prefer_gpu, "IndexFlatL2")


def find_reconstruction_neighbors(
    features,
    n_neighbors,
    *,
    metric="euclidean",
    prefer_gpu=True,
):
    """Return the ``k`` non-self neighbors of eq. (5)'s ``N_k(x_i)``.

    ``euclidean`` is LLE's own metric and the default. ``cosine`` reuses the
    ranking :func:`_find_lrml_knn_neighbors` builds, so a study can hold the
    neighborhoods fixed across this regularizer and the Laplacian ones and vary
    only what is written on them.
    """

    metric = str(metric)
    if metric not in LLE_RECONSTRUCTION_METRICS:
        raise ValueError(
            "reconstruction neighbor metric must be one of "
            f"{list(LLE_RECONSTRUCTION_METRICS)}"
        )
    if metric == "cosine":
        return _find_lrml_knn_neighbors(
            features,
            n_neighbors,
            prefer_gpu=prefer_gpu,
        )

    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("reconstruction features must be a non-empty matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("reconstruction features must be finite")
    num_nodes = len(features)
    k = min(int(n_neighbors), num_nodes - 1)
    if k <= 0:
        raise ValueError(
            "locally linear reconstruction needs at least two samples "
            "and one neighbor"
        )
    # Query k + 1 because the database contains each query itself, and filter by
    # node id rather than assuming ties leave self in column zero.
    _, neighbors = faiss_flat_l2_search(
        database=features,
        queries=features,
        k=k + 1,
        purpose="locally linear reconstruction",
        prefer_gpu=prefer_gpu,
    )
    neighbors = np.asarray(neighbors, dtype=np.int64)
    self_ids = np.arange(num_nodes, dtype=np.int64)[:, None]
    keep = neighbors != self_ids
    # Every row has at most one self-match, so dropping it leaves at least k
    # columns; the extra column of a row that never matched itself is trimmed.
    ordered = np.argsort(~keep, axis=1, kind="stable")
    neighbors = np.take_along_axis(neighbors, ordered, axis=1)[:, :k]
    if np.any(neighbors == self_ids):
        raise RuntimeError(
            "locally linear reconstruction neighbor search kept a self-match"
        )
    return np.ascontiguousarray(neighbors, dtype=np.int64)


def solve_lle_reconstruction_weights(
    features,
    neighbor_indices,
    *,
    neighbor_mask=None,
    regularization=DEFAULT_LLE_RECONSTRUCTION_REGULARIZATION,
    chunk_size=4096,
):
    """Equation (5): the weights that reconstruct each row from its neighbors.

        min_S sum_i || x_i - sum_{j in N_k(x_i)} s_ij x_j ||^2
        s.t.  s_ij = 0 for x_j not in N_k(x_i),  sum_j s_ij = 1

    The sum-to-one constraint makes the residual a difference of differences:
    ``x_i - sum_j s_ij x_j = sum_j s_ij (x_i - x_j)``, so the problem is the
    constrained least squares of Roweis and Saul (2000) with local Gram matrix
    ``C_jl = (x_i - x_j) . (x_i - x_l)``, solved as ``C w = 1`` followed by a
    rescaling to ``sum_j w_j = 1``.

    ``neighbor_mask`` marks which columns of ``neighbor_indices`` are real. A
    masked-out column is returned with weight zero, which keeps the weights
    rectangular while leaving the constraint satisfied over the columns that
    remain. It is what lets the oracle ablation drop neighbors and still hand
    back a valid eq. (5) solution, rather than one whose weights no longer sum
    to one.

    Returns ``(weights, metadata)`` where ``weights`` has the shape of
    ``neighbor_indices``. Rows with no usable neighbor get all-zero weights and
    are reported in ``metadata['isolated_nodes']``; the caller is expected to
    keep them out of the anchor pool, because an all-zero row's residual is
    ``z_i`` itself and would pull that embedding toward the origin instead of
    toward its neighborhood.
    """

    features = np.ascontiguousarray(features, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("reconstruction features must be a non-empty matrix")
    if neighbor_indices.ndim != 2 or len(neighbor_indices) != len(features):
        raise ValueError(
            "neighbor_indices must have one row of neighbors per feature row"
        )
    num_nodes, k = neighbor_indices.shape
    if k == 0:
        raise ValueError("locally linear reconstruction needs at least one neighbor")
    if neighbor_indices.min() < 0 or neighbor_indices.max() >= num_nodes:
        raise ValueError("neighbor_indices refer to a row outside the feature matrix")
    if neighbor_mask is None:
        neighbor_mask = np.ones((num_nodes, k), dtype=bool)
    else:
        neighbor_mask = np.asarray(neighbor_mask, dtype=bool)
        if neighbor_mask.shape != neighbor_indices.shape:
            raise ValueError("neighbor_mask must match neighbor_indices")
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("reconstruction regularization must be finite and non-negative")
    chunk_size = max(1, int(chunk_size))

    weights = np.zeros((num_nodes, k), dtype=np.float64)
    identity = np.eye(k, dtype=np.float64)
    singular_rows = 0
    degenerate_rows = 0
    for start in range(0, num_nodes, chunk_size):
        stop = min(start + chunk_size, num_nodes)
        rows = np.arange(start, stop)
        mask = neighbor_mask[start:stop]
        # (m, k, d): x_i - x_j for each of row i's neighbors j.
        differences = features[rows][:, None, :] - features[neighbor_indices[start:stop]]
        # A masked column must not reach the Gram matrix at all; zeroing its
        # difference makes its row and column vanish, and the conditioner below
        # puts a positive value on its diagonal so the solve stays well posed.
        differences = differences * mask[:, :, None]
        gram = differences @ np.swapaxes(differences, 1, 2)
        trace = np.trace(gram, axis1=1, axis2=2)
        # trace == 0 means every usable neighbor coincides with the anchor, so
        # the fit is exact for any weights and only the constraint decides.
        conditioner = np.where(trace > 0.0, regularization * trace / k, regularization)
        gram = gram + conditioner[:, None, None] * identity
        targets = mask.astype(np.float64)
        try:
            solved = np.linalg.solve(gram, targets[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            # One singular block would fail the whole batch, so fall back per
            # row rather than losing the chunk.
            singular_rows += len(rows)
            solved = np.empty_like(targets)
            for offset in range(len(rows)):
                solved[offset] = np.linalg.lstsq(
                    gram[offset],
                    targets[offset],
                    rcond=None,
                )[0]
        solved = solved * mask
        totals = solved.sum(axis=1)
        usable = mask.any(axis=1)
        # A vanishing total cannot be rescaled to one. Uniform weights over the
        # usable columns satisfy the constraint and are the least-committal fit.
        degenerate = usable & (np.abs(totals) <= 1e-12)
        degenerate_rows += int(degenerate.sum())
        counts = mask.sum(axis=1)
        uniform = np.divide(
            mask.astype(np.float64),
            np.maximum(counts, 1)[:, None],
            dtype=np.float64,
        )
        normalized = np.divide(
            solved,
            np.where(np.abs(totals) <= 1e-12, 1.0, totals)[:, None],
        )
        normalized = np.where(degenerate[:, None], uniform, normalized)
        weights[start:stop] = np.where(usable[:, None], normalized, 0.0)

    usable_rows = neighbor_mask.any(axis=1)
    sums = weights.sum(axis=1)
    if usable_rows.any():
        worst_deviation = float(np.max(np.abs(sums[usable_rows] - 1.0)))
    else:
        worst_deviation = 0.0
    if worst_deviation > 1e-6:
        raise RuntimeError(
            "locally linear reconstruction weights violate the sum-to-one "
            f"constraint of eq. (5) by {worst_deviation:.3e}"
        )
    metadata = {
        "nodes": int(num_nodes),
        "n_neighbors": int(k),
        "regularization": regularization,
        "isolated_nodes": int((~usable_rows).sum()),
        "mean_used_neighbors": float(neighbor_mask.sum(axis=1).mean()),
        "singular_gram_rows": int(singular_rows),
        "degenerate_weight_rows": int(degenerate_rows),
        "weight_min": float(weights[usable_rows].min()) if usable_rows.any() else 0.0,
        "weight_max": float(weights[usable_rows].max()) if usable_rows.any() else 0.0,
        "negative_weight_fraction": (
            float((weights[usable_rows] < 0.0).mean()) if usable_rows.any() else 0.0
        ),
        "worst_weight_sum_deviation": worst_deviation,
    }
    return weights, metadata


def apply_reconstruction_neighbor_ablation(
    neighbor_indices,
    labels,
    *,
    mode="off",
    seed=0,
):
    """Audit or prune reconstruction neighborhoods using hidden true labels.

    The counterpart of :func:`apply_graph_edge_ablation` for eq. (5)'s
    neighborhoods. It cannot be that helper: the shared one prunes a symmetric
    adjacency whose weights carry no constraint, whereas eq. (5)'s weights must
    sum to one over the neighbors that remain. So this returns a *mask* and the
    weights are solved afterwards on the retained columns, rather than a pruned
    weight matrix whose rows no longer sum to one.

    ``normal`` only measures purity. ``oracle_pruned`` keeps same-class
    neighbors. ``random_pruned`` drops the same total number of neighbors
    uniformly at random, which makes it the neighborhood-size control for the
    oracle arm. A row is never emptied by ``random_pruned``: it keeps at least
    the neighbor the oracle would have kept it, or its first column, so the
    anchor pool stays comparable between the two arms.

    Returns ``(neighbor_mask, stats)``.
    """

    mode = str(mode)
    if mode not in GRAPH_EDGE_ABLATION_MODES:
        raise ValueError(
            "graph_edge_ablation must be one of "
            f"{sorted(GRAPH_EDGE_ABLATION_MODES)}"
        )
    if mode == "degree_rewired":
        raise ValueError(
            "graph_edge_ablation='degree_rewired' rewires a symmetric adjacency; "
            "eq. (5) reconstruction neighborhoods have no such graph to rewire"
        )
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    keep_all = np.ones(neighbor_indices.shape, dtype=bool)
    if mode == "off":
        return keep_all, None

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(neighbor_indices):
        raise ValueError(
            "reconstruction neighbor ablation labels must align with graph nodes"
        )
    if np.any(labels < 0):
        raise ValueError(
            "reconstruction neighbor ablation requires a hidden true label for "
            "every node"
        )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("reconstruction neighbor ablation seed must be an integer")
    if int(seed) < 0:
        raise ValueError("reconstruction neighbor ablation seed must be non-negative")

    same_class = labels[neighbor_indices] == labels[:, None]
    purity = float(same_class.mean())
    total = int(same_class.size)
    oracle_removed = int((~same_class).sum())

    if mode == "normal":
        neighbor_mask = keep_all
    elif mode == "oracle_pruned":
        neighbor_mask = same_class
    else:
        # Remove the same count the oracle would, chosen uniformly, but never
        # the column that keeps a row non-empty.
        generator = np.random.default_rng(int(seed))
        neighbor_mask = keep_all.copy()
        protected = np.argmax(same_class, axis=1)
        protected = np.where(same_class.any(axis=1), protected, 0)
        removable = keep_all.copy()
        removable[np.arange(len(neighbor_indices)), protected] = False
        candidates = np.flatnonzero(removable.reshape(-1))
        drop = min(oracle_removed, len(candidates))
        if drop:
            chosen = generator.choice(candidates, size=drop, replace=False)
            flat = neighbor_mask.reshape(-1)
            flat[chosen] = False
            neighbor_mask = flat.reshape(neighbor_indices.shape)

    if not neighbor_mask.any():
        raise ValueError(
            f"graph_edge_ablation={mode!r} removed every reconstruction neighbor"
        )
    retained = int(neighbor_mask.sum())
    retained_purity = (
        float(same_class[neighbor_mask].mean()) if retained else 0.0
    )
    stats = {
        "mode": mode,
        "neighbor_purity": purity,
        "original_neighbors": total,
        "retained_neighbors": retained,
        "removed_neighbors": total - retained,
        "retained_neighbor_fraction": retained / float(total),
        "retained_neighbor_purity": retained_purity,
        "isolated_nodes": int((~neighbor_mask.any(axis=1)).sum()),
    }
    return neighbor_mask, stats


def build_lle_reconstruction_graph(
    features,
    n_neighbors,
    *,
    metric="euclidean",
    regularization=DEFAULT_LLE_RECONSTRUCTION_REGULARIZATION,
    ablation_mode="off",
    ablation_labels=None,
    ablation_seed=0,
    prefer_gpu=True,
):
    """Build eq. (5)'s neighborhoods and weights in one call.

    ``ablation_mode`` runs :func:`apply_reconstruction_neighbor_ablation` before
    the weights are solved, so a pruned neighborhood is re-fitted and its
    weights still satisfy eq. (5)'s sum-to-one constraint.

    Returns ``(neighbor_indices, neighbor_mask, weights, metadata)``.
    """

    neighbor_indices = find_reconstruction_neighbors(
        features,
        n_neighbors,
        metric=metric,
        prefer_gpu=prefer_gpu,
    )
    if str(ablation_mode) != "off" and ablation_labels is None:
        raise ValueError(
            f"graph_edge_ablation={ablation_mode!r} needs hidden true labels"
        )
    neighbor_mask, ablation_stats = apply_reconstruction_neighbor_ablation(
        neighbor_indices,
        ablation_labels,
        mode=ablation_mode,
        seed=ablation_seed,
    )
    weights, metadata = solve_lle_reconstruction_weights(
        features,
        neighbor_indices,
        neighbor_mask=neighbor_mask,
        regularization=regularization,
    )
    metadata = {
        **metadata,
        "neighbor_metric": str(metric),
        "graph_edge_ablation": str(ablation_mode),
        "ablation": ablation_stats,
        "raw_neighbor_purity": (
            None if ablation_stats is None else ablation_stats["neighbor_purity"]
        ),
    }
    return neighbor_indices, neighbor_mask, weights, metadata


def build_slrml_supervised_graph(labels):
    """Build SLRML's supervised same-class adjacency W^l.

    Labels equal to ``UNLABELED_TARGET`` are treated as unknown and receive no
    supervised edges.  ``N_S`` is the number of unordered positive pairs among
    known labeled samples.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("slrml labels must be a vector")

    row_parts = []
    col_parts = []
    positive_pair_count = 0
    known_labels = labels[labels != UNLABELED_TARGET]
    for label in np.unique(known_labels):
        class_indices = np.flatnonzero(labels == int(label)).astype(np.int64)
        if len(class_indices) < 2:
            continue
        local_rows, local_cols = np.triu_indices(len(class_indices), k=1)
        rows = class_indices[local_rows]
        cols = class_indices[local_cols]
        row_parts.append(rows)
        col_parts.append(cols)
        positive_pair_count += int(len(rows))

    if positive_pair_count == 0:
        empty = sparse.csr_matrix((len(labels), len(labels)), dtype=np.float64)
        return empty, 0

    rows = np.concatenate(row_parts).astype(np.int64)
    cols = np.concatenate(col_parts).astype(np.int64)
    weight = 1.0 / (2.0 * float(positive_pair_count))
    data = np.full(2 * positive_pair_count, weight, dtype=np.float64)
    supervised = sparse.coo_matrix(
        (data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(len(labels), len(labels)),
        dtype=np.float64,
    ).tocsr()
    return supervised, positive_pair_count


def build_slrml_graph(
    embeddings,
    n_neighbors,
    labels=None,
    include_supervised_graph=False,
    *,
    prefer_gpu=True,
):
    """Build the single-stream SLRML graph.

    The default is the requested label-free regularizer ``W^u``.  Setting
    ``include_supervised_graph`` explicitly adds the paper's same-class
    labeled component, producing ``W^s = W^u + W^l``.
    """

    graph_kwargs = {} if prefer_gpu else {"prefer_gpu": False}
    neighbor_indices, unsupervised, _ = build_lrml_knn_graph(
        embeddings=embeddings,
        n_neighbors=n_neighbors,
        **graph_kwargs,
    )
    actual_neighbors = int(neighbor_indices.shape[1])
    if actual_neighbors <= 0:
        raise ValueError("slrml graph needs at least one neighbor per sample")
    unsupervised = unsupervised.copy().tocsr()
    unsupervised.data[:] = 1.0 / float(actual_neighbors)

    graph = unsupervised
    positive_pair_count = 0
    if include_supervised_graph:
        if labels is None:
            raise ValueError("slrml include_supervised_graph requires graph labels")
        supervised, positive_pair_count = build_slrml_supervised_graph(labels)
        if supervised.shape != unsupervised.shape:
            raise ValueError("slrml labels must be aligned with embeddings")
        graph = (unsupervised + supervised).tocsr()

    graph.setdiag(0)
    graph.eliminate_zeros()
    degrees = np.asarray(graph.sum(axis=1), dtype=np.float64).ravel()
    return neighbor_indices, graph, degrees, positive_pair_count


def make_mixed_label_affinity(
    features,
    n_neighbors,
    gamma,
    return_diagnostics=False,
    *,
    prefer_gpu=True,
):
    """Build equation (15)'s sparse symmetric cosine-affinity graph.

    When requested, return the retained directed kNN candidates as compact
    construction metadata. This lets graph diagnostics report how many of the
    requested neighbors survived positive-part clipping before symmetrization.
    """

    faiss = require_faiss("mixed label propagation")
    normalized = np.ascontiguousarray(features, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    k = min(int(n_neighbors), len(normalized) - 1)
    search_started_at = time.perf_counter()
    search_kwargs = {} if prefer_gpu else {"prefer_gpu": False}
    similarities, neighbors = faiss_flat_ip_search(
        database=normalized,
        queries=normalized,
        k=k + 1,
        purpose="mixed label propagation",
        faiss_module=faiss,
        **search_kwargs,
    )
    search_seconds = time.perf_counter() - search_started_at

    construction_started_at = time.perf_counter()
    num_samples, retrieved = neighbors.shape
    query_indices = np.repeat(np.arange(num_samples, dtype=np.int64), retrieved)
    neighbor_indices = neighbors.ravel().astype(np.int64)
    # Drop self matches, then keep only the first k survivors per row --
    # identical to the loop's `continue` on self and `break` at kept == k.
    keep = neighbor_indices != query_indices
    survivor_rank = keep.reshape(num_samples, retrieved).cumsum(axis=1).ravel()
    keep &= survivor_rank <= k
    selected_neighbors = neighbor_indices[keep].reshape(num_samples, k)
    selected_similarities = similarities.ravel()[keep].reshape(
        num_samples,
        k,
    ).astype(np.float64, copy=False)
    # Power in float64 to match the original loop, which converted each float32
    # similarity to a Python float before ** gamma.
    selected_values = (
        np.clip(selected_similarities, 0.0, None) ** float(gamma)
    )
    selected_queries = np.repeat(
        np.arange(num_samples, dtype=np.int64),
        k,
    )

    directed = sparse.coo_matrix(
        (
            selected_values.ravel(),
            (selected_neighbors.ravel(), selected_queries),
        ),
        shape=(num_samples, num_samples),
        dtype=np.float64,
    ).tocsr()
    affinity = (directed + directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    construction_seconds = time.perf_counter() - construction_started_at
    if not return_diagnostics:
        return affinity
    return affinity, {
        "graph_kind": "positive_part_cosine_knn",
        "requested_n_neighbors": int(n_neighbors),
        "search_n_neighbors": int(k),
        "gamma": float(gamma),
        "neighbor_indices": selected_neighbors,
        "neighbor_similarities": selected_similarities,
        "search_seconds": float(search_seconds),
        "construction_seconds": float(construction_seconds),
    }


@torch.no_grad()
def _block_cg_with_iterations(A, B, rtol, max_iter, X=None):
    """Run independent CG solves together, freezing completed columns."""

    tiny = torch.finfo(B.dtype).tiny
    b_norm = B.norm(dim=0, keepdim=True)
    live = b_norm > 0
    target = (rtol * b_norm) ** 2

    if X is None:
        X, R = torch.zeros_like(B), B.clone()
    else:
        X = torch.where(live, X, 0.0)
        R = B - A @ X
    X = torch.where(live, X, 0.0)
    R = torch.where(live, R, 0.0)
    rs = (R * R).sum(0, keepdim=True)

    breakdown = live & ~torch.isfinite(rs)
    active = live & ~breakdown & (rs > target)
    P = torch.where(active, R, 0.0)
    counts = torch.zeros_like(b_norm, dtype=torch.int64)

    for _ in range(max_iter):
        if not bool(active.any()):
            break
        AP = A @ P
        curvature = (P * AP).sum(0, keepdim=True)
        valid = active & torch.isfinite(curvature) & (curvature > 0)
        breakdown |= active & ~valid
        counts += valid
        alpha = torch.where(
            valid,
            rs / torch.where(valid, curvature, 1.0),
            0.0,
        )
        X.addcmul_(P, alpha)
        R.addcmul_(AP, alpha, value=-1.0)
        rs_new = (R * R).sum(0, keepdim=True)
        finite = torch.isfinite(rs_new)
        breakdown |= valid & ~finite
        active = valid & finite & (rs_new > target)
        beta = torch.where(
            active,
            rs_new / rs.clamp_min(tiny),
            0.0,
        )
        R = torch.where(active, R, 0.0)
        P = R + beta * P
        rs = rs_new

    converged = (~live | ~(active | breakdown)).squeeze(0)
    return (
        X,
        counts.squeeze(0),
        live.squeeze(0),
        converged,
        breakdown.squeeze(0),
    )


@torch.no_grad()
def block_cg(A, B, rtol, max_iter, X=None):
    """CG on all right-hand sides at once.

    Each column keeps its own alpha and beta, so this is numerically the same
    algorithm SciPy runs per column. Only the memory traffic changes.
    """

    solution, _, _, _, _ = _block_cg_with_iterations(
        A,
        B,
        rtol=rtol,
        max_iter=max_iter,
        X=X,
    )
    return solution


def _require_cupy_cg(name):
    """Import the optional CuPy sparse-CG stack with an actionable error."""

    try:
        import cupy
        from cupyx.scipy import sparse as cupy_sparse
        from cupyx.scipy.sparse import linalg as cupy_sparse_linalg
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"{name}: linear_solver='cupy_cg' requires CuPy; install "
            "cupy-cuda12x for this CUDA 12 environment"
        ) from exc
    return cupy, cupy_sparse, cupy_sparse_linalg


def _cupy_device_id(cupy, name):
    """Choose the CUDA device already selected by the training process."""

    configured_device = _configured_ssl_device()
    if configured_device is not None and configured_device.type != "cuda":
        raise RuntimeError(
            f"{name}: linear_solver='cupy_cg' requires a CUDA SSL device"
        )

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            f"{name}: CuPy could not query the available CUDA devices"
        ) from exc
    if device_count <= 0:
        raise RuntimeError(
            f"{name}: linear_solver='cupy_cg' requires an available CUDA GPU"
        )

    if configured_device is not None and configured_device.index is not None:
        device_id = int(configured_device.index)
    elif torch.cuda.is_available():
        device_id = int(torch.cuda.current_device())
    else:
        device_id = int(cupy.cuda.runtime.getDevice())
    if not 0 <= device_id < device_count:
        raise RuntimeError(
            f"{name}: selected CUDA device {device_id} is unavailable to CuPy "
            f"(visible device count: {device_count})"
        )
    return device_id


def _cupy_cg_right_hand_side_batch_size(
    cupy,
    *,
    row_count,
    right_hand_side_count,
    itemsize,
):
    """Choose a bounded multi-RHS batch that leaves ample free CUDA memory."""

    right_hand_side_count = int(right_hand_side_count)
    if right_hand_side_count <= 0:
        return 0

    maximum = min(
        right_hand_side_count,
        CUPY_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH,
    )
    try:
        free_bytes, _ = cupy.cuda.runtime.memGetInfo()
        bytes_per_right_hand_side = max(
            int(row_count)
            * int(itemsize)
            * CUPY_CG_DENSE_WORK_ARRAYS,
            1,
        )
        memory_limited = max(
            1,
            int(
                int(free_bytes)
                * CUPY_CG_FREE_MEMORY_FRACTION
                // bytes_per_right_hand_side
            ),
        )
        selected = min(maximum, memory_limited)
    except Exception:
        # Memory introspection is an optimization only. The hard cap remains
        # conservative, and the caller also retries OOMs with smaller batches.
        selected = maximum

    # Multiples of a warp make the common large-batch path predictable. Do not
    # round tiny memory-limited batches down to zero.
    if selected >= 32:
        selected = (selected // 32) * 32
    return max(1, selected)


def _solve_cupy_cg_right_hand_side_batch(
    cupy,
    gpu_matrix,
    right_hand_side,
    *,
    rtol,
    max_iter,
    warm_start,
    collect_iterations,
    column_dot,
):
    """Advance independent CG solves together using sparse-dense products."""

    # CuPy's CSR-dense matmul requires Fortran-contiguous dense operands.
    # Keeping every work array column-major avoids an implicit full-size
    # C-to-F copy before every CG sparse-matmul.
    gpu_rhs = cupy.asarray(right_hand_side, order="F")
    right_hand_side_norm_squared = column_dot(
        gpu_rhs,
        gpu_rhs,
        axis=0,
    )
    nonzero_rhs = right_hand_side_norm_squared > 0.0
    target_residual_squared = (
        float(rtol) * float(rtol) * right_hand_side_norm_squared
    )

    if warm_start is None:
        gpu_solution = cupy.zeros_like(gpu_rhs)
        residual = gpu_rhs.copy()
    else:
        gpu_solution = cupy.asarray(warm_start, order="F").copy()
        # Match the scalar solver's zero-RHS special case: its exact solution
        # is zero even if a nonzero warm start was supplied.
        cupy.multiply(
            gpu_solution,
            nonzero_rhs[None, :],
            out=gpu_solution,
        )
        residual = gpu_matrix @ gpu_solution
        cupy.negative(residual, out=residual)
        cupy.add(residual, gpu_rhs, out=residual)

    residual_norm_squared = column_dot(
        residual,
        residual,
        axis=0,
    )
    breakdown = nonzero_rhs & (
        ~cupy.isfinite(right_hand_side_norm_squared)
        | ~cupy.isfinite(residual_norm_squared)
    )
    active = (
        nonzero_rhs
        & ~breakdown
        & (residual_norm_squared > target_residual_squared)
    )
    search_direction = residual.copy()
    cupy.multiply(
        search_direction,
        active[None, :],
        out=search_direction,
    )
    gpu_iteration_counts = cupy.zeros(
        right_hand_side.shape[1],
        dtype=cupy.int64,
    )

    for _ in range(int(max_iter)):
        # This is the only device-to-host convergence synchronization per
        # iteration for the entire batch, rather than one per class.
        if not bool(cupy.any(active)):
            break

        matrix_times_direction = gpu_matrix @ search_direction
        direction_curvature = column_dot(
            search_direction,
            matrix_times_direction,
            axis=0,
        )
        valid_step = (
            active
            & cupy.isfinite(direction_curvature)
            & (direction_curvature > 0.0)
        )
        breakdown |= active & ~valid_step
        gpu_iteration_counts += valid_step

        step_size = cupy.divide(
            residual_norm_squared,
            cupy.where(valid_step, direction_curvature, 1.0),
        )
        cupy.multiply(step_size, valid_step, out=step_size)

        # Reuse A@P as the dense scratch array for both axpy updates. This
        # avoids allocating another N x batch temporary on every iteration.
        cupy.multiply(
            matrix_times_direction,
            step_size[None, :],
            out=matrix_times_direction,
        )
        cupy.subtract(
            residual,
            matrix_times_direction,
            out=residual,
        )
        cupy.multiply(
            search_direction,
            step_size[None, :],
            out=matrix_times_direction,
        )
        cupy.add(
            gpu_solution,
            matrix_times_direction,
            out=gpu_solution,
        )

        new_residual_norm_squared = column_dot(
            residual,
            residual,
            axis=0,
        )
        finite_residual = cupy.isfinite(new_residual_norm_squared)
        breakdown |= valid_step & ~finite_residual
        active = (
            valid_step
            & finite_residual
            & (new_residual_norm_squared > target_residual_squared)
        )

        conjugate_scale = cupy.divide(
            new_residual_norm_squared,
            cupy.where(active, residual_norm_squared, 1.0),
        )
        cupy.multiply(conjugate_scale, active, out=conjugate_scale)
        # Freeze converged and failed columns while the remaining columns in
        # this batch continue. They still occupy the dense batch but generate
        # no further numerical work.
        cupy.multiply(residual, active[None, :], out=residual)
        cupy.multiply(
            search_direction,
            conjugate_scale[None, :],
            out=search_direction,
        )
        cupy.add(search_direction, residual, out=search_direction)
        residual_norm_squared = new_residual_norm_squared

    gpu_solver_info = cupy.where(
        breakdown,
        -1,
        cupy.where(active, gpu_iteration_counts, 0),
    )
    solution = cupy.asnumpy(gpu_solution)
    solver_info = cupy.asnumpy(gpu_solver_info).astype(
        np.int64,
        copy=False,
    )
    if collect_iterations:
        iteration_counts = cupy.asnumpy(gpu_iteration_counts).astype(
            np.int64,
            copy=False,
        )
    else:
        iteration_counts = np.zeros(
            right_hand_side.shape[1],
            dtype=np.int64,
        )
    return solution, solver_info, iteration_counts


def _solve_in_adaptive_batches(column_count, batch_size, solve_batch, store, *, oom_error, can_retry, on_oom, label):
    """Solve ``[start, stop)`` column batches, halving the batch on out-of-memory.

    Sparse SpMM workspace needs vary by CUDA/cuSPARSE version, so the
    free-memory estimate behind ``batch_size`` can still be too optimistic;
    the same columns are then retried at half width. Returns the batching
    metadata the solver diagnostics report.
    """

    batch_count = 0
    oom_retry_count = 0
    largest_batch_size = 0
    smallest_batch_size = None
    batch_start = 0
    while batch_start < column_count:
        batch_stop = min(batch_start + batch_size, column_count)
        try:
            result = solve_batch(batch_start, batch_stop)
        except oom_error:
            if not can_retry or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            oom_retry_count += 1
            on_oom()
            logger.warning(
                f"{label} ran out of memory; retrying "
                f"with {batch_size} right-hand sides per batch"
            )
            continue
        store(batch_start, batch_stop, result)
        completed_batch_size = batch_stop - batch_start
        largest_batch_size = max(largest_batch_size, completed_batch_size)
        smallest_batch_size = (
            completed_batch_size
            if smallest_batch_size is None
            else min(smallest_batch_size, completed_batch_size)
        )
        batch_count += 1
        batch_start = batch_stop
    return {
        "right_hand_side_batch_size": int(largest_batch_size),
        "smallest_right_hand_side_batch_size": int(smallest_batch_size or 0),
        "right_hand_side_batch_count": int(batch_count),
        "oom_retry_count": int(oom_retry_count),
    }


def _solve_sparse_label_system_cupy_cg(
    matrix,
    right_hand_side,
    *,
    rtol,
    max_iter,
    warm_start,
    name,
    collect_iterations,
    metadata=None,
):
    """Run memory-bounded, batched multi-RHS CG on one GPU CSR matrix."""

    cupy, cupy_sparse, _ = _require_cupy_cg(name)
    device_id = _cupy_device_id(cupy, name)
    if torch.cuda.is_available():
        # PyTorch and CuPy have independent caching allocators. Return
        # PyTorch's unused blocks before CuPy allocates the sparse system.
        torch.cuda.synchronize(device_id)
        torch.cuda.empty_cache()

    def solve_on_device():
        solutions = np.zeros_like(right_hand_side)
        iteration_counts = np.zeros(
            right_hand_side.shape[1],
            dtype=np.int64,
        )
        solver_info = np.zeros(
            right_hand_side.shape[1],
            dtype=np.int64,
        )

        gpu_matrix = cupy_sparse.csr_matrix(
            (
                cupy.asarray(matrix.data),
                cupy.asarray(matrix.indices),
                cupy.asarray(matrix.indptr),
            ),
            shape=matrix.shape,
        )
        column_dot = cupy.ReductionKernel(
            "T left, T right",
            "T result",
            "left * right",
            "a + b",
            "result = a",
            "0",
            "ssl_cupy_cg_column_dot",
        )
        batch_size = _cupy_cg_right_hand_side_batch_size(
            cupy,
            row_count=right_hand_side.shape[0],
            right_hand_side_count=right_hand_side.shape[1],
            itemsize=right_hand_side.dtype.itemsize,
        )
        estimated_dense_working_set_bytes = int(
            right_hand_side.shape[0]
            * batch_size
            * right_hand_side.dtype.itemsize
            * CUPY_CG_DENSE_WORK_ARRAYS
        )
        logger.info(
            f"{name}: CuPy batched CG using at most {batch_size} "
            "right-hand sides per batch "
            f"(estimated dense CUDA working set "
            f"{estimated_dense_working_set_bytes / (1024 ** 3):.2f} GiB)"
        )
        def solve_batch(batch_start, batch_stop):
            return _solve_cupy_cg_right_hand_side_batch(
                cupy,
                gpu_matrix,
                right_hand_side[:, batch_start:batch_stop],
                rtol=rtol,
                max_iter=max_iter,
                warm_start=(
                    None
                    if warm_start is None
                    else warm_start[:, batch_start:batch_stop]
                ),
                collect_iterations=collect_iterations,
                column_dot=column_dot,
            )

        def store(batch_start, batch_stop, result):
            batch_solutions, batch_solver_info, batch_iteration_counts = result
            solutions[:, batch_start:batch_stop] = batch_solutions
            solver_info[batch_start:batch_stop] = batch_solver_info
            iteration_counts[batch_start:batch_stop] = batch_iteration_counts
            # All CUDA arrays owned by the completed helper call are now out
            # of scope. Do not let allocator fragmentation or version-specific
            # sparse-matmul workspaces accumulate across thousands of RHS.
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()

        batching = _solve_in_adaptive_batches(
            right_hand_side.shape[1],
            batch_size,
            solve_batch,
            store,
            oom_error=cupy.cuda.memory.OutOfMemoryError,
            can_retry=True,
            on_oom=lambda: cupy.get_default_memory_pool().free_all_blocks(),
            label=f"{name}: CuPy batched CG",
        )
        cupy.cuda.get_current_stream().synchronize()
        if metadata is not None:
            metadata.update(
                {
                    "algorithm": "batched_independent_cg",
                    **batching,
                    "estimated_dense_working_set_bytes": estimated_dense_working_set_bytes,
                }
            )
        return solutions, solver_info, iteration_counts

    with cupy.cuda.Device(device_id):
        try:
            result = solve_on_device()
        finally:
            # CuPy and PyTorch use separate caching allocators. Release CuPy's
            # cached blocks so the training model can reclaim all free VRAM.
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
    return (*result, f"cuda:{device_id}")


def _torch_solver_device():
    """Use the active CUDA device, with a CPU fallback for portability."""

    configured_device = _configured_ssl_device()
    if configured_device is not None:
        if configured_device.type != "cuda":
            return torch.device("cpu")
        if configured_device.index is not None:
            return configured_device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _torch_block_cg_right_hand_side_batch_size(
    *,
    device,
    row_count,
    right_hand_side_count,
    itemsize,
):
    """Choose a bounded Torch CG batch from the currently free device memory."""

    right_hand_side_count = int(right_hand_side_count)
    if right_hand_side_count <= 0:
        return 0
    maximum = min(
        right_hand_side_count,
        TORCH_BLOCK_CG_MAX_RIGHT_HAND_SIDES_PER_BATCH,
    )
    if device.type != "cuda":
        return maximum

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        bytes_per_right_hand_side = max(
            int(row_count)
            * int(itemsize)
            * TORCH_BLOCK_CG_DENSE_WORK_ARRAYS,
            1,
        )
        memory_limited = max(
            1,
            int(
                int(free_bytes)
                * TORCH_BLOCK_CG_FREE_MEMORY_FRACTION
                // bytes_per_right_hand_side
            ),
        )
        selected = min(maximum, memory_limited)
    except (RuntimeError, TypeError):
        selected = maximum

    if selected >= 32 and selected < right_hand_side_count:
        selected = (selected // 32) * 32
    return max(1, selected)


def _solve_sparse_label_system_torch_block_cg(
    matrix,
    right_hand_side,
    *,
    rtol,
    max_iter,
    warm_start,
    name="sparse label system",
    metadata=None,
):
    """Convert one CSR matrix and solve bounded RHS chunks with Torch."""

    device = _torch_solver_device()
    if device.type == "cuda":
        # Discard unrelated cached blocks before sizing the bounded solve.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()

    crow_indices = torch.as_tensor(matrix.indptr, device=device)
    column_indices = torch.as_tensor(matrix.indices, device=device)
    values = torch.as_tensor(matrix.data, device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        torch_matrix = torch.sparse_csr_tensor(
            crow_indices,
            column_indices,
            values,
            size=matrix.shape,
            dtype=values.dtype,
            device=device,
            check_invariants=False,
        )
    batch_size = _torch_block_cg_right_hand_side_batch_size(
        device=device,
        row_count=right_hand_side.shape[0],
        right_hand_side_count=right_hand_side.shape[1],
        itemsize=right_hand_side.dtype.itemsize,
    )
    estimated_dense_working_set_bytes = int(
        right_hand_side.shape[0]
        * batch_size
        * right_hand_side.dtype.itemsize
        * TORCH_BLOCK_CG_DENSE_WORK_ARRAYS
    )
    logger.info(
        f"{name}: Torch block CG using at most {batch_size} "
        "right-hand sides per batch "
        f"(estimated dense {device} working set "
        f"{estimated_dense_working_set_bytes / (1024 ** 3):.2f} GiB)"
    )

    solutions = np.zeros_like(right_hand_side)
    active = np.zeros(right_hand_side.shape[1], dtype=bool)
    converged = np.ones(right_hand_side.shape[1], dtype=bool)
    breakdown = np.zeros(right_hand_side.shape[1], dtype=bool)
    iteration_counts = np.zeros(
        right_hand_side.shape[1],
        dtype=np.int64,
    )

    def solve_batch(batch_start, batch_stop):
        torch_rhs = torch.as_tensor(
            right_hand_side[:, batch_start:batch_stop],
            device=device,
        )
        torch_warm_start = (
            None
            if warm_start is None
            else torch.as_tensor(
                warm_start[:, batch_start:batch_stop],
                device=device,
            )
        )
        (
            torch_solution,
            torch_iteration_counts,
            torch_active,
            torch_converged,
            torch_breakdown,
        ) = _block_cg_with_iterations(
            torch_matrix,
            torch_rhs,
            rtol=float(rtol),
            max_iter=int(max_iter),
            X=torch_warm_start,
        )
        return (
            torch_solution.cpu().numpy(),
            torch_active.cpu().numpy(),
            torch_converged.cpu().numpy(),
            torch_breakdown.cpu().numpy(),
            torch_iteration_counts.cpu().numpy(),
        )

    def store(batch_start, batch_stop, result):
        batch_solution, batch_active, batch_converged, batch_breakdown, batch_iteration_counts = result
        solutions[:, batch_start:batch_stop] = batch_solution
        active[batch_start:batch_stop] = batch_active
        converged[batch_start:batch_stop] = batch_converged
        breakdown[batch_start:batch_stop] = batch_breakdown
        iteration_counts[batch_start:batch_stop] = batch_iteration_counts

    # Freed blocks from a completed batch intentionally stay in PyTorch's
    # CUDA cache. Every full batch has the same shapes, so the next batch can
    # reuse the scratch pool instead of returning it to CUDA and immediately
    # allocating it again. OOM recovery and final cleanup still purge it.
    batching = _solve_in_adaptive_batches(
        right_hand_side.shape[1],
        batch_size,
        solve_batch,
        store,
        oom_error=torch.OutOfMemoryError,
        can_retry=device.type == "cuda",
        on_oom=torch.cuda.empty_cache,
        label=f"{name}: Torch block CG",
    )
    if metadata is not None:
        metadata.update(
            {
                "algorithm": "chunked_block_cg",
                **batching,
                "estimated_dense_working_set_bytes": estimated_dense_working_set_bytes,
            }
        )

    device_name = str(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        del torch_matrix, crow_indices, column_indices, values
        torch.cuda.empty_cache()
    return (
        solutions,
        active,
        converged,
        breakdown,
        iteration_counts,
        device_name,
    )


def solve_sparse_label_system(
        matrix,
        right_hand_side,
        rtol,
        max_iter,
        name,
        linear_solver="cg",
        warm_start=None,
        allow_nonconvergence=False,
        cholmod_factor_cache=None,
        diagnostics=None,
):
    """Solve a sparse SPD system with CG or CHOLMOD.

    SciPy CG runs independently for each right-hand-side column. CuPy CG and
    Torch block CG advance bounded batches of columns together, and CHOLMOD
    factors the matrix once before solving all columns.
    """

    solve_started_at = time.perf_counter()
    matrix = matrix.tocsr().astype(np.float64, copy=False)
    right_hand_side = np.asarray(right_hand_side, dtype=np.float64)

    # Normalize a single RHS to shape (N, 1), then restore it before returning.
    single_rhs = right_hand_side.ndim == 1
    if single_rhs:
        right_hand_side = right_hand_side[:, None]
    elif right_hand_side.ndim != 2:
        raise ValueError(
            f"{name}: right_hand_side must be one- or two-dimensional, "
            f"got shape {right_hand_side.shape}"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name}: matrix must be square, got shape {matrix.shape}")

    if matrix.shape[0] != right_hand_side.shape[0]:
        raise ValueError(
            f"{name}: incompatible shapes: matrix={matrix.shape}, "
            f"right_hand_side={right_hand_side.shape}"
        )

    try:
        linear_solver = normalize_linear_solver(linear_solver)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc

    if not np.isfinite(float(rtol)) or float(rtol) <= 0.0:
        raise ValueError(f"{name}: rtol must be finite and positive")
    if int(max_iter) <= 0:
        raise ValueError(f"{name}: max_iter must be positive")

    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=np.float64)
        if single_rhs and warm_start.ndim == 1:
            warm_start = warm_start[:, None]

        if warm_start.shape != right_hand_side.shape:
            raise ValueError(
                f"{name}: warm_start has shape {warm_start.shape}, "
                f"expected {right_hand_side.shape}"
            )

    _initialize_solver_diagnostics(
        diagnostics,
        matrix=matrix,
        right_hand_side=right_hand_side,
        name=name,
        linear_solver=linear_solver,
        rtol=rtol,
        max_iter=max_iter,
        warm_start=warm_start,
        allow_nonconvergence=allow_nonconvergence,
    )

    def restore_shape(solution):
        return solution[:, 0] if single_rhs else solution

    if linear_solver == "cholmod":
        cholmod_diagnostics = {} if diagnostics is not None else None
        cholmod_kwargs = {}
        if cholmod_diagnostics is not None:
            cholmod_kwargs["diagnostics"] = cholmod_diagnostics
        solution = solve_sparse_label_system_cholmod(
            matrix,
            right_hand_side,
            name=name,
            factor_cache=cholmod_factor_cache,
            **cholmod_kwargs,
        )
        solution = np.asarray(solution)
        if diagnostics is not None:
            diagnostics["cholmod"] = cholmod_diagnostics
        _finish_solver_diagnostics(
            diagnostics,
            matrix=matrix,
            right_hand_side=right_hand_side,
            solution=solution,
            started_at=solve_started_at,
        )
        return restore_shape(solution)

    solutions = None
    iteration_counts = np.zeros(right_hand_side.shape[1], dtype=np.int64)
    solver_info = np.zeros(right_hand_side.shape[1], dtype=np.int64)
    solver_device = None
    solver_backend_metadata = {}

    if linear_solver == "cupy_cg":
        (
            solutions,
            solver_info,
            iteration_counts,
            solver_device,
        ) = _solve_sparse_label_system_cupy_cg(
            matrix,
            right_hand_side,
            rtol=rtol,
            max_iter=max_iter,
            warm_start=warm_start,
            name=name,
            collect_iterations=diagnostics is not None,
            metadata=solver_backend_metadata,
        )
        solver_description = "CuPy batched conjugate gradient"
    elif linear_solver == "torch_block_cg":
        (
            solutions,
            nonzero_rhs,
            block_converged,
            block_breakdown,
            block_iteration_counts,
            solver_device,
        ) = _solve_sparse_label_system_torch_block_cg(
            matrix,
            right_hand_side,
            rtol=rtol,
            max_iter=max_iter,
            warm_start=warm_start,
            name=name,
            metadata=solver_backend_metadata,
        )
        iteration_counts = block_iteration_counts
        block_unconverged = nonzero_rhs & ~block_converged
        solver_info[block_unconverged] = iteration_counts[
            block_unconverged
        ]
        solver_info[block_breakdown] = -1
        solver_description = "Torch block conjugate gradient"
    else:
        solutions = np.zeros_like(right_hand_side)
        for class_index in range(right_hand_side.shape[1]):
            rhs = right_hand_side[:, class_index]

            # The exact solution for a zero RHS is zero. Special-casing it
            # avoids the purely relative stopping tolerance becoming zero.
            if not np.any(rhs):
                continue

            x0 = (
                None
                if warm_start is None
                else warm_start[:, class_index]
            )

            cg_kwargs = {}
            if diagnostics is not None:
                def count_iteration(_iterate, index=class_index):
                    iteration_counts[index] += 1

                cg_kwargs["callback"] = count_iteration
            solution, info = sparse_linalg.cg(
                matrix,
                rhs,
                x0=x0,
                rtol=float(rtol),
                atol=0.0,
                maxiter=int(max_iter),
                **cg_kwargs,
            )

            solutions[:, class_index] = solution
            solver_info[class_index] = int(info)

        solver_description = "SciPy conjugate gradient"

    failed = np.flatnonzero(solver_info < 0).tolist()
    if failed:
        raise RuntimeError(
            f"{name}: {solver_description} failed due to numerical "
            f"breakdown for classes {failed[:10]}"
        )
    unconverged = np.flatnonzero(solver_info > 0).tolist()
    timing_details = {
        "system": name,
        "rows": matrix.shape[0],
        "matrix_nnz": matrix.nnz,
        "right_hand_sides": right_hand_side.shape[1],
        "warm_start": warm_start is not None,
        "unconverged": len(unconverged),
    }
    if solver_device is not None:
        timing_details["device"] = solver_device
    timing_details.update(solver_backend_metadata)

    if diagnostics is not None:
        reported_rhs_count = min(
            right_hand_side.shape[1],
            SOLVER_DIAGNOSTICS_MAX_RIGHT_HAND_SIDES,
        )
        cg_diagnostics = {
            "implementation": linear_solver,
            "iterations_per_right_hand_side": iteration_counts[
                :reported_rhs_count
            ].tolist(),
            "iterations": numeric_diagnostic_summary(iteration_counts),
            "solver_info_per_right_hand_side": solver_info[
                :reported_rhs_count
            ].tolist(),
            "right_hand_side_details_truncated": (
                right_hand_side.shape[1]
                > SOLVER_DIAGNOSTICS_MAX_RIGHT_HAND_SIDES
            ),
            "converged_right_hand_side_count": int(
                np.sum(solver_info == 0)
            ),
            "unconverged_right_hand_side_count": int(len(unconverged)),
            "unconverged_right_hand_side_indices": [
                int(index) for index in unconverged[:100]
            ],
            "unconverged_indices_truncated": len(unconverged) > 100,
        }
        if linear_solver == "cg":
            cg_diagnostics["scipy_info_per_right_hand_side"] = (
                solver_info[:reported_rhs_count].tolist()
            )
        cg_diagnostics.update(solver_backend_metadata)
        diagnostics["cg"] = cg_diagnostics
    _finish_solver_diagnostics(
        diagnostics,
        matrix=matrix,
        right_hand_side=right_hand_side,
        solution=solutions,
        started_at=solve_started_at,
    )
    if diagnostics is not None and unconverged:
        diagnostics["status"] = (
            "computed_truncated"
            if allow_nonconvergence
            else "failed_nonconvergence"
        )

    if unconverged:
        message = (
            f"{name}: {solver_description} did not converge within "
            f"{max_iter} iterations for classes {unconverged[:10]}"
        )
        if allow_nonconvergence:
            logger.warning(f"{message}; using the truncated iterates")
        else:
            raise RuntimeError(message)

    return restore_shape(solutions)


def solve_sparse_label_system_cholmod(
    matrix,
    right_hand_side,
    name,
    cholmod_module=None,
    factor_cache=None,
    diagnostics=None,
):
    """Factor once with CHOLMOD and solve all class columns together.

    scikit-sparse 0.5 replaced the callable ``Factor`` returned by
    ``cholesky`` with ``cho_factor`` and ``CholeskyFactor.solve``. It also
    renamed same-pattern numeric refactorization to ``factorize``. Support
    both APIs because scikit-sparse is intentionally not version-pinned.
    """

    if cholmod_module is None:
        try:
            from sksparse import cholmod as cholmod_module
        except ImportError as exc:
            raise ImportError(
                f"{name}: linear_solver='cholmod' requires scikit-sparse"
            ) from exc

    # Retain a reference to the caller's canonical CSR matrix for the pattern
    # comparison. Mixed LP already keeps both systems alive, so this avoids
    # copying their potentially large index arrays into the factor cache.
    pattern_matrix = matrix.tocsr(copy=False)
    pattern_matrix.sum_duplicates()
    pattern_matrix.sort_indices()
    cached_pattern = (
        factor_cache.get("pattern_matrix")
        if factor_cache is not None
        else None
    )
    cached_pattern_matches = (
        cached_pattern is not None
        and cached_pattern.shape == pattern_matrix.shape
        and np.array_equal(cached_pattern.indptr, pattern_matrix.indptr)
        and np.array_equal(cached_pattern.indices, pattern_matrix.indices)
    )

    matrix = pattern_matrix.tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    right_hand_side = np.asarray(right_hand_side, dtype=np.float64)
    cached_pattern_matches = (
        cached_pattern_matches
        and factor_cache.get("factor") is not None
    )

    factor_started_at = time.perf_counter()
    factor = factor_cache.get("factor") if cached_pattern_matches else None
    symbolic_reused = factor is not None
    if factor is not None and hasattr(factor, "factorize"):
        # scikit-sparse >= 0.5: repeat only the numeric factorization.
        factor.factorize(matrix)
        factorization_api = "factor.factorize"
    elif factor is not None and hasattr(factor, "cholesky_inplace"):
        # scikit-sparse 0.4: in-place numeric refactorization.
        factor.cholesky_inplace(matrix)
        factorization_api = "factor.cholesky_inplace"
    elif factor is not None and hasattr(factor, "cholesky"):
        # Older Factor fallback that preserves the cached symbolic analysis.
        factor = factor.cholesky(matrix)
        factorization_api = "factor.cholesky"
    elif hasattr(cholmod_module, "cho_factor"):
        factor = cholmod_module.cho_factor(matrix)
        factorization_api = "cho_factor"
        symbolic_reused = False
    else:
        factor = cholmod_module.cholesky(matrix)
        factorization_api = "cholesky"
        symbolic_reused = False

    if factor_cache is not None:
        factor_cache["factor"] = factor
        factor_cache["pattern_matrix"] = pattern_matrix

    factor_seconds = time.perf_counter() - factor_started_at

    solve_started_at = time.perf_counter()
    if hasattr(factor, "solve"):
        solution = factor.solve(right_hand_side)
        solve_api = "factor.solve"
    else:
        solution = factor(right_hand_side)
        solve_api = "factor.__call__"
    solve_seconds = time.perf_counter() - solve_started_at

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "status": "computed",
                "factorization_api": factorization_api,
                "solve_api": solve_api,
                "symbolic_analysis_reused": bool(symbolic_reused),
                "factorization_seconds": float(factor_seconds),
                "solve_seconds": float(solve_seconds),
            }
        )
    return solution


def normalize_label_spreading_rows(values):
    """Convert nonnegative fixed-point scores into per-row class probabilities."""

    probabilities = np.asarray(values, dtype=np.float64).copy()
    if probabilities.ndim != 2:
        raise ValueError("label spreading scores must be a matrix")
    if probabilities.shape[1] <= 1:
        return np.ones_like(probabilities, dtype=np.float64)
    row_chunk_size = _dense_row_chunk_size(
        probabilities,
        ENTROPY_CONFIDENCE_CPU_CHUNK_BYTES,
    )
    for start in range(0, len(probabilities), row_chunk_size):
        chunk = probabilities[start:start + row_chunk_size]
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError(
                "faiss_label_spreading produced non-finite class scores"
            )

    negative_tolerance = 1e-12
    for start in range(0, len(probabilities), row_chunk_size):
        chunk = probabilities[start:start + row_chunk_size]
        if np.any(chunk < -negative_tolerance):
            raise RuntimeError(
                "faiss_label_spreading produced negative class scores"
            )

    for start in range(0, len(probabilities), row_chunk_size):
        stop = min(start + row_chunk_size, len(probabilities))
        chunk = probabilities[start:stop]
        np.maximum(chunk, 0.0, out=chunk)
        row_sums = chunk.sum(axis=1, keepdims=True)
        zero_rows = np.flatnonzero(row_sums.ravel() == 0.0)
        if len(zero_rows) > 0:
            zero_rows = zero_rows + start
            raise RuntimeError(
                "faiss_label_spreading produced zero-mass rows, usually "
                "because a graph component has no labeled target: "
                f"{zero_rows[:10].tolist()}"
            )
        np.divide(chunk, row_sums, out=chunk)
    return probabilities


def _dense_row_chunk_size(values, target_bytes):
    """Return a positive row count near the requested dense-memory budget."""

    bytes_per_row = max(
        int(values.shape[1]) * int(values.dtype.itemsize),
        1,
    )
    return max(1, int(target_bytes) // bytes_per_row)


def _validate_probability_rows_chunked(probabilities, row_chunk_size):
    """Validate a large probability matrix without full-size boolean arrays."""

    for start in range(0, len(probabilities), row_chunk_size):
        chunk = probabilities[start:start + row_chunk_size]
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError(
                "Equation (21) received non-finite normalized class values"
            )

    for start in range(0, len(probabilities), row_chunk_size):
        chunk = probabilities[start:start + row_chunk_size]
        if np.any(chunk < 0):
            raise RuntimeError(
                "Mixed label propagation produced negative normalized class "
                "values; equation (21) is undefined because they are not "
                "probabilities"
            )

    for start in range(0, len(probabilities), row_chunk_size):
        chunk = probabilities[start:start + row_chunk_size]
        if not np.allclose(
            chunk.sum(axis=1),
            1.0,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Equation (21) received class values that do not sum to one"
            )


def _entropy_confidence_cpu_chunked(
    probabilities,
    *,
    row_chunk_size,
    log_num_classes,
):
    """Evaluate p*log(p) in bounded CPU chunks using SciPy's fused ufunc."""

    confidences = np.empty(len(probabilities), dtype=np.float64)
    for start in range(0, len(probabilities), row_chunk_size):
        stop = min(start + row_chunk_size, len(probabilities))
        chunk = probabilities[start:stop]
        entropy_terms = special.xlogy(chunk, chunk)
        confidences[start:stop] = (
            1.0
            + entropy_terms.sum(axis=1) / log_num_classes
        )
    return confidences


def _entropy_confidence_cuda_chunked(
    probabilities,
    *,
    row_chunk_size,
    log_num_classes,
):
    """Evaluate entropy in bounded CUDA chunks and return a NumPy vector."""

    device = _torch_solver_device()
    confidences = np.empty(len(probabilities), dtype=np.float64)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    for start in range(0, len(probabilities), row_chunk_size):
        stop = min(start + row_chunk_size, len(probabilities))
        torch_probabilities = torch.as_tensor(
            probabilities[start:stop],
            device=device,
        )
        torch_confidences = (
            1.0
            + torch.xlogy(
                torch_probabilities,
                torch_probabilities,
            ).sum(dim=1)
            / log_num_classes
        )
        confidences[start:stop] = torch_confidences.cpu().numpy()
        del torch_probabilities, torch_confidences
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return confidences


def entropy_confidence(probabilities, *, validate=True):
    """Equation (21): one minus entropy normalized by log(number of classes)."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a matrix")
    if probabilities.shape[1] <= 1:
        return np.ones(probabilities.shape[0], dtype=np.float64)
    cpu_row_chunk_size = _dense_row_chunk_size(
        probabilities,
        ENTROPY_CONFIDENCE_CPU_CHUNK_BYTES,
    )
    if validate:
        _validate_probability_rows_chunked(
            probabilities,
            cpu_row_chunk_size,
        )

    log_num_classes = float(np.log(probabilities.shape[1]))
    use_cuda = (
        probabilities.nbytes >= ENTROPY_CONFIDENCE_CUDA_MIN_BYTES
        and torch.cuda.is_available()
    )
    if use_cuda:
        cuda_row_chunk_size = _dense_row_chunk_size(
            probabilities,
            ENTROPY_CONFIDENCE_CUDA_CHUNK_BYTES,
        )
        logger.info(
            "entropy_confidence using chunked CUDA evaluation "
            f"(rows_per_chunk={cuda_row_chunk_size}, "
            f"device={_torch_solver_device()})"
        )
        return _entropy_confidence_cuda_chunked(
            probabilities,
            row_chunk_size=cuda_row_chunk_size,
            log_num_classes=log_num_classes,
        )
    return _entropy_confidence_cpu_chunked(
        probabilities,
        row_chunk_size=cpu_row_chunk_size,
        log_num_classes=log_num_classes,
    )


def majority_vote(label_rows):
    """Return the most frequent label and its vote count for every row."""

    pseudo_labels = np.empty(label_rows.shape[0], dtype=np.int64)
    vote_counts = np.empty(label_rows.shape[0], dtype=np.int64)

    for row_index, labels in enumerate(label_rows):
        # np.unique sorts labels and returns an aligned occurrence count. In a
        # tie, argmax chooses the first/smallest label deterministically.
        unique_labels, counts = np.unique(labels, return_counts=True)
        best_index = int(np.argmax(counts))
        pseudo_labels[row_index] = unique_labels[best_index]
        vote_counts[row_index] = counts[best_index]

    return pseudo_labels, vote_counts
