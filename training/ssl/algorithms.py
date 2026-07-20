"""Pure graph construction and label-propagation algorithms."""

import time
from functools import wraps

import numpy as np
import torch
from loguru import logger
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .config import UNLABELED_TARGET
from .graph_diagnostics import maybe_save_graph_diagnostics


FAISS_GPU_MAX_K = 2048


def _log_debug_timing(operation, started_at, **details):
    """Emit one timing record at the application's console-visible level."""

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


def _debug_timed(function):
    """Log total wall time for an algorithm helper, including failed calls."""

    @wraps(function)
    def timed_function(*args, **kwargs):
        started_at = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _log_debug_timing(function.__name__, started_at)

    return timed_function


def _dependency(overrides, name, default):
    """Resolve a helper supplied by the orchestration façade, if any."""

    if overrides is None:
        return default
    return overrides.get(name, default)


@_debug_timed
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


@_debug_timed
def _faiss_gpu_device_id(faiss, purpose):
    """Return the active FAISS GPU ID, or ``None`` for a CPU-only build/host."""

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
            device_id = int(torch.cuda.current_device())
            if 0 <= device_id < gpu_count:
                return device_id
    except Exception as exc:
        logger.debug(f"{purpose}: could not read PyTorch's active CUDA device ({exc})")
    return 0


@_debug_timed
def faiss_flat_ip_search(database, queries, k, purpose, faiss_module=None):
    """Run exact inner-product search on GPU whenever FAISS supports it.

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
    cpu_index = faiss.IndexFlatIP(database.shape[1])
    gpu_device_id = _faiss_gpu_device_id(faiss, purpose)
    if gpu_device_id is not None and k <= FAISS_GPU_MAX_K:
        gpu_started_at = time.perf_counter()
        try:
            # Keep resources alive until search has copied its results back to
            # host memory. GPU indexes do not own StandardGpuResources.
            gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_device_id, cpu_index)
            gpu_index.add(database)
            results = gpu_index.search(queries, k)
            logger.debug(
                f"{purpose}: used FAISS GPU IndexFlatIP on CUDA device {gpu_device_id}"
            )
            _log_debug_timing(
                "faiss_flat_ip_search.backend",
                gpu_started_at,
                purpose=purpose,
                backend=f"cuda:{gpu_device_id}",
                database_size=len(database),
                query_size=len(queries),
                k=k,
                outcome="success",
            )
            return results
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

    cpu_started_at = time.perf_counter()
    cpu_index.add(database)
    results = cpu_index.search(queries, k)
    _log_debug_timing(
        "faiss_flat_ip_search.backend",
        cpu_started_at,
        purpose=purpose,
        backend="cpu",
        database_size=len(database),
        query_size=len(queries),
        k=k,
        outcome="success",
    )
    return results


@_debug_timed
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

    affinity = _dependency(
        _dependencies,
        "make_mixed_label_affinity",
        make_mixed_label_affinity,
    )(features, n_neighbors=n_neighbors, gamma=gamma)
    if graph_diagnostics is not None:
        _dependency(
            _dependencies,
            "maybe_save_graph_diagnostics",
            maybe_save_graph_diagnostics,
        )(
            request=graph_diagnostics.get("request"),
            embeddings=features,
            adjacency=affinity,
            positions=graph_diagnostics.get("positions"),
            labels=graph_diagnostics.get("labels"),
            known_mask=graph_diagnostics.get("known_mask"),
        )

    system_started_at = time.perf_counter()
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    inverse_sqrt_degrees = np.zeros_like(degrees, dtype=np.float64)
    positive_degree = degrees > 0.0
    inverse_sqrt_degrees[positive_degree] = 1.0 / np.sqrt(degrees[positive_degree])
    degree_scaling = sparse.diags(inverse_sqrt_degrees)
    normalized_affinity = (degree_scaling @ affinity @ degree_scaling).tocsr()

    alpha = float(alpha)
    system = (
        sparse.eye(len(features), format="csr", dtype=np.float64)
        - alpha * normalized_affinity
    ).tocsr()
    one_hot_targets = np.zeros((len(features), num_classes), dtype=np.float64)
    one_hot_targets[np.flatnonzero(labeled), targets[labeled]] = 1.0
    _log_debug_timing(
        "faiss_label_spreading.system_construction",
        system_started_at,
        samples=len(features),
        classes=num_classes,
        matrix_nnz=system.nnz,
    )
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
    )
    probabilities = _dependency(
        _dependencies,
        "normalize_label_spreading_rows",
        normalize_label_spreading_rows,
    )(scores)
    confidences = probabilities.max(axis=1)
    return probabilities.astype(np.float32), confidences.astype(np.float32)


@_debug_timed
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
    linear_solver = str(linear_solver)
    if linear_solver not in {"cg", "cholmod"}:
        raise ValueError(
            "iscen_label_spreading linear_solver must be one of ['cg', 'cholmod']"
        )
    labeled = targets != UNLABELED_TARGET
    if not np.any(labeled):
        raise ValueError("iscen_label_spreading requires at least one labeled target")
    if np.any((targets[labeled] < 0) | (targets[labeled] >= int(num_classes))):
        raise ValueError("labeled targets must be in [0, num_classes)")

    affinity = _dependency(
        _dependencies,
        "make_mixed_label_affinity",
        make_mixed_label_affinity,
    )(features, n_neighbors=int(n_neighbors), gamma=float(gamma))
    if graph_diagnostics is not None:
        _dependency(
            _dependencies,
            "maybe_save_graph_diagnostics",
            maybe_save_graph_diagnostics,
        )(
            request=graph_diagnostics.get("request"),
            embeddings=features,
            adjacency=affinity,
            positions=graph_diagnostics.get("positions"),
            labels=graph_diagnostics.get("labels"),
            known_mask=graph_diagnostics.get("known_mask"),
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

    system_started_at = time.perf_counter()
    active_affinity = affinity[active_indices][:, active_indices].tocsr()
    active_degrees = degrees[active_indices]
    inverse_sqrt_degrees = np.zeros_like(active_degrees)
    positive_degree = active_degrees > 0.0
    inverse_sqrt_degrees[positive_degree] = 1.0 / np.sqrt(active_degrees[positive_degree])
    degree_scaling = sparse.diags(inverse_sqrt_degrees)
    normalized_affinity = (degree_scaling @ active_affinity @ degree_scaling).tocsr()
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
    _log_debug_timing(
        "iscen_label_spreading.system_construction",
        system_started_at,
        samples=len(active_indices),
        omitted_seedless=seedless_count,
        classes=int(num_classes),
        matrix_nnz=system.nnz,
    )

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
        # The public LP-DeepSSL implementation uses SciPy's final iterate when
        # its reference limit of 20 CG iterations is reached.
        allow_nonconvergence=linear_solver == "cg",
    )

    # A finite truncated CG solve can contain negative numerical overshoot.
    # Clamp either solver's output consistently before row normalization.
    active_nonnegative_scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    active_row_masses = active_nonnegative_scores.sum(axis=1)
    positive_mass = active_row_masses > 0.0
    zero_mass_active_indices = active_indices[~positive_mass]
    if len(zero_mass_active_indices) > 0:
        logger.warning(
            "Iscen label spreading produced "
            f"{len(zero_mass_active_indices)} additional zero-mass rows inside "
            "seed-reachable graph components after the sparse solve; marking "
            "them unpropagatable. Increase cg_max_iter or use linear_solver='cholmod' "
            "if this warning recurs"
        )

    # Rows without positive mass remain exactly zero. The adapter recognizes
    # that sentinel and removes those candidates before argmax, including when
    # the configured confidence threshold is zero.
    probabilities = np.zeros((len(features), int(num_classes)), dtype=np.float64)
    positive_mass_indices = active_indices[positive_mass]
    if len(positive_mass_indices) > 0:
        probabilities[positive_mass_indices] = _dependency(
            _dependencies,
            "normalize_label_spreading_rows",
            normalize_label_spreading_rows,
        )(active_nonnegative_scores[positive_mass])

    confidences = np.zeros(len(features), dtype=np.float64)
    if len(positive_mass_indices) > 0:
        confidences[positive_mass_indices] = _dependency(
            _dependencies,
            "entropy_confidence",
            entropy_confidence,
        )(probabilities[positive_mass_indices])
    max_confidence = float(np.max(confidences))
    if max_confidence > 0.0:
        confidences = confidences / max_confidence
    else:
        # Uniform predictions carry no information. Keep their certainty at
        # zero rather than reproducing the reference implementation's 0 / 0.
        confidences = np.zeros_like(confidences)
    confidences[labeled] = 1.0
    return probabilities.astype(np.float32), confidences.astype(np.float32)


@_debug_timed
def mixed_label_propagation(
    features,
    targets,
    num_classes,
    n_neighbors=50,
    gamma=3.0,
    temperature=4.0,
    beta=1.0,
    mu=1.0 / 99.0,
    cg_rtol=1e-5,
    cg_max_iter=1000,
    edge_batch_size=65536,
    linear_solver="cg",
    graph_diagnostics=None,
    _dependencies=None,
):
    """Run equations (14)-(24) and return mixed-LP scores/confidences."""

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be a matrix aligned with targets")
    if len(features) < 2:
        raise ValueError("mixed_label_propagation requires at least two samples")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    labeled = targets != UNLABELED_TARGET
    if not np.any(labeled):
        raise ValueError("mixed_label_propagation requires at least one labeled target")
    if np.any((targets[labeled] < 0) | (targets[labeled] >= num_classes)):
        raise ValueError("labeled targets must be in [0, num_classes)")

    affinity = _dependency(
        _dependencies,
        "make_mixed_label_affinity",
        make_mixed_label_affinity,
    )(features, n_neighbors=n_neighbors, gamma=gamma)
    if graph_diagnostics is not None:
        _dependency(
            _dependencies,
            "maybe_save_graph_diagnostics",
            maybe_save_graph_diagnostics,
        )(
            request=graph_diagnostics.get("request"),
            embeddings=features,
            adjacency=affinity,
            positions=graph_diagnostics.get("positions"),
            labels=graph_diagnostics.get("labels"),
            known_mask=graph_diagnostics.get("known_mask"),
        )
    initial_system_started_at = time.perf_counter()
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees) - affinity
    anchors = sparse.diags(np.where(labeled, float(mu), 0.0))

    one_hot_targets = np.zeros((len(features), num_classes), dtype=np.float64)
    one_hot_targets[np.flatnonzero(labeled), targets[labeled]] = 1.0
    right_hand_side = anchors @ one_hot_targets
    initial_system = (laplacian + anchors).tocsr()
    _log_debug_timing(
        "mixed_label_propagation.initial_system_construction",
        initial_system_started_at,
        samples=len(features),
        classes=num_classes,
        matrix_nnz=initial_system.nnz,
    )
    solve_system = _dependency(
        _dependencies,
        "solve_sparse_label_system",
        solve_sparse_label_system,
    )
    linear_solver = str(linear_solver)
    # Both mixed-LP matrices have the affinity graph's sparsity pattern. Keep
    # one CHOLMOD factor cache so the second numeric factorization can reuse
    # the first solve's symbolic analysis and fill-reducing permutation.
    cholmod_solve_kwargs = (
        {"cholmod_factor_cache": {}}
        if linear_solver == "cholmod"
        else {}
    )
    initial_labels = solve_system(
        initial_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="initial label propagation",
        linear_solver=linear_solver,
        **cholmod_solve_kwargs,
    )

    dissimilarity = _dependency(
        _dependencies,
        "make_dissimilarity_affinity",
        make_dissimilarity_affinity,
    )(
        affinity=affinity,
        degrees=degrees,
        propagated_labels=initial_labels,
        temperature=float(temperature),
        edge_batch_size=int(edge_batch_size),
    )
    mixed_system_started_at = time.perf_counter()
    dissimilarity_degrees = np.asarray(dissimilarity.sum(axis=1)).ravel()
    signless_laplacian = sparse.diags(dissimilarity_degrees) + dissimilarity
    # Equation (24) sums both directions of each symmetric edge, yielding the
    # factor 2 in the derivative of beta/2 * D(G).
    mixed_system = (
        laplacian
        + anchors
        + 2.0 * float(beta) * signless_laplacian
    ).tocsr()
    _log_debug_timing(
        "mixed_label_propagation.mixed_system_construction",
        mixed_system_started_at,
        samples=len(features),
        classes=num_classes,
        matrix_nnz=mixed_system.nnz,
    )
    mixed_labels = solve_system(
        mixed_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="mixed label propagation",
        linear_solver=linear_solver,
        # Warm start from the initial-LP solution: the mixed system differs only
        # by the signless-Laplacian term, so CG typically converges in a
        # handful of iterations from here.
        warm_start=initial_labels,
        **cholmod_solve_kwargs,
    )
    normalized_scores = _dependency(
        _dependencies,
        "normalize_mixed_label_rows",
        normalize_mixed_label_rows,
    )(mixed_labels)
    # Section 3.3 applies Eq. (21) directly to G*_i / ||G*_i||_1.  The
    # temperature-scaled softmax in Eq. (20) is only for the earlier
    # leave-one-edge scores used to construct dissimilarity weights.
    confidences = _dependency(
        _dependencies,
        "entropy_confidence",
        entropy_confidence,
    )(normalized_scores)
    return normalized_scores.astype(np.float32), confidences.astype(np.float32)


@_debug_timed
def _find_lrml_knn_neighbors(embeddings, n_neighbors):
    """Return exact non-self FAISS neighbors for each LRML graph node."""

    features = np.ascontiguousarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("lrml embeddings must be a non-empty feature matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("lrml embeddings must be finite")
    num_nodes = len(features)
    k = min(int(n_neighbors), num_nodes - 1)
    if k <= 0:
        raise ValueError("lrml graph needs at least two samples and one neighbor")

    # Query k + 1 because the indexed database contains each query itself. Filter
    # by node ID below instead of assuming ties always leave self in column zero.
    _, neighbors = faiss_flat_ip_search(
        database=features,
        queries=features,
        k=k + 1,
        purpose="lrml regularization",
    )

    neighbor_indices = np.empty((num_nodes, k), dtype=np.int64)
    for node, neighbor_row in enumerate(neighbors):
        kept = 0
        for neighbor in neighbor_row:
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            neighbor_indices[node, kept] = neighbor
            kept += 1
            if kept == k:
                break
        if kept != k:
            raise RuntimeError(
                f"FAISS returned only {kept} non-self LRML neighbors for node {node}; "
                f"expected {k}"
            )
    return neighbor_indices


@_debug_timed
def _lrml_pyg_utils():
    """Load the PyG graph utilities only when the LRML edge path is used."""

    try:
        from torch_geometric.utils import degree, get_laplacian, to_undirected
    except ImportError as exc:
        raise ImportError(
            "lrml edge-index regularization requires torch-geometric; "
            "install the project requirements"
        ) from exc
    return to_undirected, degree, get_laplacian


@_debug_timed
def build_lrml_knn_edge_index(embeddings, n_neighbors):
    """Build LRML's binary symmetric kNN graph as a PyG ``edge_index``.

    The returned ``edge_index`` has shape ``[2, 2M]`` and contains both
    directions of every one of the ``M`` undirected edges. Degrees are computed
    directly from its source row with :func:`torch_geometric.utils.degree`.
    """

    to_undirected, pyg_degree, _ = _lrml_pyg_utils()
    neighbor_indices = _find_lrml_knn_neighbors(embeddings, n_neighbors)
    num_nodes, k = neighbor_indices.shape
    source = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(k)
    target = torch.from_numpy(neighbor_indices.reshape(-1))
    directed_edge_index = torch.stack((source, target), dim=0)
    edge_index = to_undirected(directed_edge_index, num_nodes=num_nodes).contiguous()
    degrees = pyg_degree(
        edge_index[0],
        num_nodes=num_nodes,
        dtype=torch.float64,
    )
    if torch.any(degrees <= 0):
        raise RuntimeError("lrml kNN graph contains an isolated node")
    return neighbor_indices, edge_index, degrees


@_debug_timed
def validate_lrml_laplacian(edge_index, embeddings, normalized_laplacian):
    """Materialize a PyG Laplacian and verify its quadratic graph energy.

    This is intended for opt-in graph-build validation, not for the stochastic
    training hot path. The returned tensors make the validated sparse
    Laplacian available to callers for debugging and inspection.
    """

    _, pyg_degree, get_laplacian = _lrml_pyg_utils()
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    features = torch.as_tensor(embeddings, dtype=torch.float64, device="cpu")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("lrml edge_index must have shape [2, num_directed_edges]")
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("lrml validation embeddings must be a non-empty matrix")
    if edge_index.numel() == 0:
        raise ValueError("lrml edge_index must contain at least one edge")
    num_nodes = len(features)
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("lrml edge_index refers to a node outside the embeddings")
    if not torch.isfinite(features).all():
        raise ValueError("lrml validation embeddings must be finite")

    normalization = "sym" if normalized_laplacian else None
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(
        edge_index,
        normalization=normalization,
        dtype=features.dtype,
        num_nodes=num_nodes,
    )
    laplacian_source, laplacian_target = laplacian_edge_index
    laplacian_energy = (
        laplacian_edge_weight[:, None]
        * features[laplacian_source]
        * features[laplacian_target]
    ).sum()

    degrees = pyg_degree(
        edge_index[0],
        num_nodes=num_nodes,
        dtype=features.dtype,
    )
    if torch.any(degrees <= 0):
        raise ValueError("lrml Laplacian validation requires positive node degrees")
    scaled = features / degrees.sqrt()[:, None] if normalized_laplacian else features
    unique_edge_mask = edge_index[0] < edge_index[1]
    left = edge_index[0, unique_edge_mask]
    right = edge_index[1, unique_edge_mask]
    pairwise_energy = ((scaled[left] - scaled[right]) ** 2).sum()
    try:
        torch.testing.assert_close(
            laplacian_energy,
            pairwise_energy,
            rtol=1e-6,
            atol=1e-8,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "LRML pairwise energy does not match the PyG Laplacian quadratic form"
        ) from exc
    return laplacian_edge_index, laplacian_edge_weight


@_debug_timed
def build_lrml_knn_graph(embeddings, n_neighbors):
    """Build the legacy SciPy LRML graph used by the weighted SLRML path."""

    neighbor_indices = _find_lrml_knn_neighbors(embeddings, n_neighbors)
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


@_debug_timed
def make_slrml_graph_labels(train_dataset, graph_positions, labeled_positions):
    """Return SLRML graph-node labels with unlabeled nodes masked as unknown."""

    dataset_labels = np.asarray(train_dataset.labels, dtype=np.int64)
    graph_positions = np.asarray(graph_positions, dtype=np.int64)
    labeled_positions = np.asarray(labeled_positions, dtype=np.int64)
    if len(dataset_labels) < len(train_dataset):
        raise ValueError("SLRML requires train_dataset.labels to align with train_dataset")
    graph_labels = np.full(len(graph_positions), UNLABELED_TARGET, dtype=np.int64)
    if len(labeled_positions) == 0:
        return graph_labels

    graph_order = {int(position): index for index, position in enumerate(graph_positions.tolist())}
    for position in labeled_positions.tolist():
        graph_index = graph_order.get(int(position))
        if graph_index is None:
            continue
        graph_labels[graph_index] = int(dataset_labels[int(position)])
    return graph_labels


@_debug_timed
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


@_debug_timed
def build_slrml_graph(
    embeddings,
    n_neighbors,
    labels=None,
    include_supervised_graph=False,
):
    """Build the single-stream SLRML graph.

    The default is the requested label-free regularizer ``W^u``.  Setting
    ``include_supervised_graph`` explicitly adds the paper's same-class
    labeled component, producing ``W^s = W^u + W^l``.
    """

    neighbor_indices, unsupervised, _ = build_lrml_knn_graph(
        embeddings=embeddings,
        n_neighbors=n_neighbors,
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


@_debug_timed
def induced_subgraph_edges(adjacency, node_ids):
    """Upper-triangular edges of the sub-graph induced on ``node_ids``.

    Row/column indices are local (into the batch order given by ``node_ids``) so
    they index straight into the batch embedding matrix. Taking only the upper
    triangle counts each undirected edge of the symmetric graph exactly once.
    """

    sub = adjacency[node_ids][:, node_ids].tocoo()
    upper = sub.row < sub.col
    return sub.row[upper], sub.col[upper], sub.data[upper]


@_debug_timed
def make_mixed_label_affinity(features, n_neighbors, gamma):
    """Build equation (15)'s sparse symmetric cosine-affinity graph (vectorized)."""

    faiss = require_faiss("mixed label propagation")
    normalized = np.ascontiguousarray(features, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    k = min(int(n_neighbors), len(normalized) - 1)
    similarities, neighbors = faiss_flat_ip_search(
        database=normalized,
        queries=normalized,
        k=k + 1,
        purpose="mixed label propagation",
        faiss_module=faiss,
    )

    num_samples, retrieved = neighbors.shape
    query_indices = np.repeat(np.arange(num_samples, dtype=np.int64), retrieved)
    neighbor_indices = neighbors.ravel().astype(np.int64)
    # Power in float64 to match the original loop, which converted each float32
    # similarity to a Python float before ** gamma.
    values = np.clip(similarities.ravel().astype(np.float64), 0.0, None) ** float(gamma)

    # Drop self matches, then keep only the first k survivors per row --
    # identical to the loop's `continue` on self and `break` at kept == k.
    keep = neighbor_indices != query_indices
    survivor_rank = keep.reshape(num_samples, retrieved).cumsum(axis=1).ravel()
    keep &= survivor_rank <= k

    directed = sparse.coo_matrix(
        (values[keep], (neighbor_indices[keep], query_indices[keep])),
        shape=(num_samples, num_samples),
        dtype=np.float64,
    ).tocsr()
    affinity = (directed + directed.T).tocsr()
    affinity.setdiag(0)
    affinity.eliminate_zeros()
    return affinity


@_debug_timed
def make_dissimilarity_affinity(
    affinity,
    degrees,
    propagated_labels,
    temperature,
    edge_batch_size,
):
    """Compute first-order-neighbor hard-negative weights from equations (20)-(22)."""

    upper = sparse.triu(affinity, k=1).tocoo()
    if upper.nnz == 0:
        return sparse.csr_matrix(affinity.shape, dtype=np.float64)

    left = upper.row
    right = upper.col
    edge_weights = upper.data
    weights = np.empty(upper.nnz, dtype=np.float64)
    # Chunk edge/class computations to retain the paper's O(Nk + NC) memory
    # bound instead of materializing one O(NkC) tensor for the whole graph.
    for start in range(0, upper.nnz, int(edge_batch_size)):
        stop = min(start + int(edge_batch_size), upper.nnz)
        chunk_left = left[start:stop]
        chunk_right = right[start:stop]
        chunk_edge_weights = edge_weights[start:stop]
        left_logits = float(temperature) * (
            degrees[chunk_left, None] * propagated_labels[chunk_left]
            - chunk_edge_weights[:, None] * propagated_labels[chunk_right]
        )
        right_logits = float(temperature) * (
            degrees[chunk_right, None] * propagated_labels[chunk_right]
            - chunk_edge_weights[:, None] * propagated_labels[chunk_left]
        )
        left_probabilities = stable_softmax(left_logits)
        right_probabilities = stable_softmax(right_logits)
        dissimilarity_probability = 1.0 - np.sum(left_probabilities * right_probabilities, axis=1)
        weights[start:stop] = (
            entropy_confidence(left_probabilities)
            * entropy_confidence(right_probabilities)
            * dissimilarity_probability
        )

    dissimilarity = sparse.coo_matrix(
        (
            np.concatenate([weights, weights]),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=affinity.shape,
        dtype=np.float64,
    ).tocsr()
    dissimilarity.eliminate_zeros()
    return dissimilarity


@_debug_timed
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
):
    """Solve a sparse SPD system with SciPy CG or CHOLMOD.

    SciPy CG is run independently for each right-hand-side column, matching
    the formulation used by both label-propagation papers.  CHOLMOD factors
    the matrix once and solves all columns together.
    """

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

    linear_solver = str(linear_solver)
    if linear_solver not in {"cholmod", "cg"}:
        raise ValueError(
            f"{name}: unsupported linear_solver={linear_solver!r}; "
            "expected 'cholmod' or 'cg'"
        )

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

    def restore_shape(solution):
        return solution[:, 0] if single_rhs else solution

    if linear_solver == "cholmod":
        solution = solve_sparse_label_system_cholmod(
            matrix,
            right_hand_side,
            name=name,
            factor_cache=cholmod_factor_cache,
        )
        return restore_shape(np.asarray(solution))

    solutions = np.zeros_like(right_hand_side)
    unconverged = []
    cg_started_at = time.perf_counter()

    for class_index in range(right_hand_side.shape[1]):
        rhs = right_hand_side[:, class_index]

        # The exact solution for a zero RHS is zero. Special-casing it avoids
        # the purely relative stopping tolerance becoming zero.
        if not np.any(rhs):
            solutions[:, class_index] = 0.0
            continue

        x0 = (
            None
            if warm_start is None
            else warm_start[:, class_index]
        )

        solution, info = sparse_linalg.cg(
            matrix,
            rhs,
            x0=x0,
            rtol=float(rtol),
            atol=0.0,
            maxiter=int(max_iter),
        )

        solutions[:, class_index] = solution

        if info < 0:
            raise RuntimeError(
                f"{name}: scipy conjugate gradient failed due to numerical "
                f"breakdown for class {class_index}"
            )
        if info > 0:
            unconverged.append(class_index)

    _log_debug_timing(
        "scipy.sparse.linalg.cg",
        cg_started_at,
        system=name,
        rows=matrix.shape[0],
        matrix_nnz=matrix.nnz,
        right_hand_sides=right_hand_side.shape[1],
        warm_start=warm_start is not None,
        unconverged=len(unconverged),
    )

    if unconverged:
        message = (
            f"{name}: scipy conjugate gradient did not converge within "
            f"{max_iter} iterations for classes {unconverged[:10]}"
        )
        if allow_nonconvergence:
            logger.warning(f"{message}; using the truncated iterates")
        else:
            raise RuntimeError(message)

    return restore_shape(solutions)


@_debug_timed
def solve_sparse_label_system_cholmod(
    matrix,
    right_hand_side,
    name,
    cholmod_module=None,
    factor_cache=None,
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

    _log_debug_timing(
        "cholmod.factorization",
        factor_started_at,
        system=name,
        api=factorization_api,
        symbolic_reused=symbolic_reused,
        rows=matrix.shape[0],
        matrix_nnz=matrix.nnz,
    )

    solve_started_at = time.perf_counter()
    if hasattr(factor, "solve"):
        solution = factor.solve(right_hand_side)
        solve_api = "factor.solve"
    else:
        solution = factor(right_hand_side)
        solve_api = "factor.__call__"
    _log_debug_timing(
        "cholmod.solve",
        solve_started_at,
        system=name,
        api=solve_api,
        rows=matrix.shape[0],
        right_hand_sides=(
            1 if right_hand_side.ndim == 1 else right_hand_side.shape[1]
        ),
    )
    return solution


@_debug_timed
def stable_softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@_debug_timed
def normalize_label_spreading_rows(values):
    """Convert nonnegative fixed-point scores into per-row class probabilities."""

    probabilities = np.asarray(values, dtype=np.float64).copy()
    if probabilities.ndim != 2:
        raise ValueError("label spreading scores must be a matrix")
    if probabilities.shape[1] <= 1:
        return np.ones_like(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("faiss_label_spreading produced non-finite class scores")
    negative_tolerance = 1e-12
    if np.any(probabilities < -negative_tolerance):
        raise RuntimeError("faiss_label_spreading produced negative class scores")
    probabilities[probabilities < 0.0] = 0.0
    row_sums = probabilities.sum(axis=1, keepdims=True)
    zero_rows = np.flatnonzero(row_sums.ravel() == 0.0)
    if len(zero_rows) > 0:
        raise RuntimeError(
            "faiss_label_spreading produced zero-mass rows, usually because a graph component "
            f"has no labeled target: {zero_rows[:10].tolist()}"
        )
    return probabilities / row_sums


@_debug_timed
def normalize_mixed_label_rows(values):
    """Convert propagated scores to the paper's L1-normalized class scores."""

    l1_norms = np.linalg.norm(values, ord=1, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(l1_norms.ravel() == 0)
    if len(zero_rows) > 0:
        raise RuntimeError(
            "Mixed label propagation produced zero-L1-norm rows, which cannot be normalized "
            f"as specified by the paper: {zero_rows[:10].tolist()}"
        )
    return values / l1_norms


@_debug_timed
def entropy_confidence(probabilities):
    """Equation (21): one minus entropy normalized by log(number of classes)."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a matrix")
    if probabilities.shape[1] <= 1:
        return np.ones(probabilities.shape[0], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Equation (21) received non-finite normalized class values")
    if np.any(probabilities < 0):
        raise RuntimeError(
            "Mixed label propagation produced negative normalized class values; "
            "equation (21) is undefined because they are not probabilities"
        )
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise RuntimeError("Equation (21) received class values that do not sum to one")
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    entropy = -np.sum(entropy_terms, axis=1)
    return 1.0 - entropy / np.log(probabilities.shape[1])


@_debug_timed
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
