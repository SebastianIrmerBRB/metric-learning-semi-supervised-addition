"""Pure graph construction and label-propagation algorithms."""

import numpy as np
import torch
from loguru import logger
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .config import UNLABELED_TARGET
from .graph_diagnostics import maybe_save_graph_diagnostics


def _dependency(overrides, name, default):
    """Resolve a helper supplied by the orchestration façade, if any."""

    if overrides is None:
        return default
    return overrides.get(name, default)


def faiss_label_spreading(
    features,
    targets,
    num_classes,
    n_neighbors=10,
    gamma=1.0,
    alpha=0.2,
    cg_rtol=1e-5,
    cg_max_iter=1000,
    linear_solver="auto",
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
    linear_solver="auto",
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
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees) - affinity
    anchors = sparse.diags(np.where(labeled, float(mu), 0.0))

    one_hot_targets = np.zeros((len(features), num_classes), dtype=np.float64)
    one_hot_targets[np.flatnonzero(labeled), targets[labeled]] = 1.0
    right_hand_side = anchors @ one_hot_targets
    initial_system = (laplacian + anchors).tocsr()
    solve_system = _dependency(
        _dependencies,
        "solve_sparse_label_system",
        solve_sparse_label_system,
    )
    initial_labels = solve_system(
        initial_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="initial label propagation",
        linear_solver=str(linear_solver),
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
    dissimilarity_degrees = np.asarray(dissimilarity.sum(axis=1)).ravel()
    signless_laplacian = sparse.diags(dissimilarity_degrees) + dissimilarity
    # Equation (24) sums both directions of each symmetric edge, yielding the
    # factor 2 in the derivative of beta/2 * D(G).
    mixed_system = (
        laplacian
        + anchors
        + 2.0 * float(beta) * signless_laplacian
    ).tocsr()
    mixed_labels = solve_system(
        mixed_system,
        right_hand_side,
        rtol=float(cg_rtol),
        max_iter=int(cg_max_iter),
        name="mixed label propagation",
        linear_solver=str(linear_solver),
        # Warm start from the initial-LP solution: the mixed system differs only
        # by the signless-Laplacian term, so CG typically converges in a
        # handful of iterations from here.
        warm_start=initial_labels,
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


def _find_lrml_knn_neighbors(embeddings, n_neighbors):
    """Return exact non-self FAISS neighbors for each LRML graph node."""

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("lrml regularization requires the faiss-cpu package") from exc

    features = np.ascontiguousarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("lrml embeddings must be a non-empty feature matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("lrml embeddings must be finite")
    num_nodes = len(features)
    k = min(int(n_neighbors), num_nodes - 1)
    if k <= 0:
        raise ValueError("lrml graph needs at least two samples and one neighbor")

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    # Query k + 1 because the first hit of each row is the node itself.
    _, neighbors = index.search(features, k + 1)

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


def build_lrml_knn_graph(embeddings, n_neighbors):
    """Build the legacy SciPy LRML graph used by the weighted SLRML path."""

    neighbor_indices = _find_lrml_knn_neighbors(embeddings, n_neighbors)
    num_nodes, k = neighbor_indices.shape
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
    degrees = np.asarray(symmetric.sum(axis=1), dtype=np.float64).ravel()
    return neighbor_indices, symmetric, degrees


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


def induced_subgraph_edges(adjacency, node_ids):
    """Upper-triangular edges of the sub-graph induced on ``node_ids``.

    Row/column indices are local (into the batch order given by ``node_ids``) so
    they index straight into the batch embedding matrix. Taking only the upper
    triangle counts each undirected edge of the symmetric graph exactly once.
    """

    sub = adjacency[node_ids][:, node_ids].tocoo()
    upper = sub.row < sub.col
    return sub.row[upper], sub.col[upper], sub.data[upper]


def make_mixed_label_affinity(features, n_neighbors, gamma):
    """Build equation (15)'s sparse symmetric cosine-affinity graph (vectorized)."""

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("mixed_label_propagation requires the faiss-cpu package") from exc

    normalized = np.ascontiguousarray(features, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    k = min(int(n_neighbors), len(normalized) - 1)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    similarities, neighbors = index.search(normalized, k + 1)

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


def solve_sparse_label_system(
        matrix,
        right_hand_side,
        rtol,
        max_iter,
        name,
        linear_solver="auto",
        warm_start=None,
):
    """Solve the sparse SPD system for all classes at once when possible.

    linear_solver:
      - "cholmod": sparse Cholesky via scikit-sparse (fails loudly if missing).
      - "cg": batched conjugate gradient (all class columns at once) with a
        Jacobi preconditioner and an optional warm start (e.g. the initial-LP
        solution for the mixed system).
      - "auto": CHOLMOD if importable, else scipy splu, else preconditioned CG.
        Direct factorizations solve all C right-hand sides after one
        factorization, so their advantage grows linearly with the number of
        classes. For very large graphs (N >> 1e5) where factorization fill-in
        exhausts memory, fall back to "cg".
    """

    right_hand_side = np.asarray(right_hand_side, dtype=np.float64)

    if linear_solver in ("auto", "cholmod"):
        try:
            return solve_sparse_label_system_cholmod(
                matrix,
                right_hand_side,
                name=name,
            )
        except ImportError:
            if linear_solver == "cholmod":
                raise ImportError(f"{name}: linear_solver='cholmod' requires scikit-sparse")

    if linear_solver == "auto":
        try:
            lu = sparse_linalg.splu(matrix.tocsc())
            return lu.solve(right_hand_side)
        except (MemoryError, RuntimeError) as exc:
            logger.warning(f"{name}: direct factorization failed ({exc}); falling back to CG")

    # Batched Jacobi-preconditioned CG: solve every class column simultaneously
    # so each iteration is one sparse @ dense matmul instead of C separate
    # sparse @ vector products. Per-column alpha/beta keep the iterates
    # identical to running scipy's preconditioned CG independently per class
    # (same Jacobi preconditioner, same rtol * ||b|| stopping rule), but the
    # matmul form is far more cache-friendly and removes C solver-call
    # overheads. The diagonal is strictly positive (degrees + mu anchors
    # [+ dissimilarity degrees]) so the Jacobi preconditioner is well-defined.
    matrix = matrix.tocsr()
    inverse_diagonal = (1.0 / matrix.diagonal())[:, None]
    if warm_start is None:
        solutions = np.zeros_like(right_hand_side)
        residuals = right_hand_side.copy()
    else:
        solutions = np.array(warm_start, dtype=np.float64, copy=True)
        residuals = right_hand_side - matrix @ solutions
    rhs_norms = np.linalg.norm(right_hand_side, axis=0)
    # Zero columns are already solved by x=0; avoid dividing by zero below.
    rhs_norms[rhs_norms == 0.0] = 1.0
    preconditioned = inverse_diagonal * residuals
    directions = preconditioned.copy()
    residual_dots = np.einsum("ij,ij->j", residuals, preconditioned)
    for _ in range(int(max_iter)):
        if np.all(np.linalg.norm(residuals, axis=0) <= rtol * rhs_norms):
            return solutions
        matrix_directions = matrix @ directions
        curvature = np.einsum("ij,ij->j", directions, matrix_directions)
        # Converged columns can have ~zero curvature; freeze them instead of
        # producing NaN steps.
        safe_curvature = np.where(curvature > 0.0, curvature, 1.0)
        step = np.where(curvature > 0.0, residual_dots / safe_curvature, 0.0)
        solutions += step * directions
        residuals -= step * matrix_directions
        preconditioned = inverse_diagonal * residuals
        new_residual_dots = np.einsum("ij,ij->j", residuals, preconditioned)
        safe_dots = np.where(residual_dots > 0.0, residual_dots, 1.0)
        directions = preconditioned + (new_residual_dots / safe_dots) * directions
        residual_dots = new_residual_dots
    if np.all(np.linalg.norm(residuals, axis=0) <= rtol * rhs_norms):
        return solutions
    unconverged = np.flatnonzero(np.linalg.norm(residuals, axis=0) > rtol * rhs_norms)
    raise RuntimeError(
        f"{name} conjugate gradient did not converge within {max_iter} iterations "
        f"for classes {unconverged[:10].tolist()}"
    )


def solve_sparse_label_system_cholmod(
    matrix,
    right_hand_side,
    name,
    cholmod_module=None,
):
    """Factor once with CHOLMOD and solve all class columns together."""

    if cholmod_module is None:
        from sksparse import cholmod as cholmod_module
    try:
        factor = cholmod_module.cholesky(matrix.tocsc())
        return factor(np.asarray(right_hand_side, dtype=np.float64))
    except ImportError as exc:
        raise ImportError(f"{name}: linear_solver='cholmod' requires scikit-sparse") from exc


def stable_softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


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
