"""Optional graph plots and sampled-edge diagnostics for SSL methods."""

import csv
import json
import math
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import sparse

import utils

from .config import GraphDiagnosticsRequest


GRAPH_DIAGNOSTICS_ARTIFACT_VERSION = 2
GRAPH_DIAGNOSTICS_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
GRAPH_DIAGNOSTICS_MAX_CLASS_PAIRS = 500
GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS = 500
GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NODES = 5_000
GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NNZ = 500_000
GRAPH_DIAGNOSTICS_MAX_CORRECT_ANCHOR_CLASSES = 200
GRAPH_DIAGNOSTICS_MAX_CORRECT_ANCHOR_WORK = 200_000_000
GRAPH_DIAGNOSTICS_MAX_CLASS_FOCUS_EDGE_ROWS = 20_000
# The class-scoped plot keeps most of its node budget for the selected classes
# themselves and spends the rest on the out-of-selection nodes they attach to.
GRAPH_DIAGNOSTICS_CLASS_FOCUS_NODE_SHARE = 0.7
# Edge colors for the class-scoped plot: the edges the Laplacian term should be
# pulling on, the ones fusing two selected classes, and the ones leaving.
GRAPH_EDGE_SAME_CLASS_COLOR = "#2f7d4f"
GRAPH_EDGE_BETWEEN_CLASSES_COLOR = "#c1442e"
GRAPH_EDGE_LEAVING_COLOR = "#a9aeb6"

# Diagnostics are generated in one process during normal epoch-by-epoch graph
# rebuilding. Keeping only the immediately preceding compact graph state makes
# temporal comparisons possible without writing a full adjacency artifact.
_GRAPH_DIAGNOSTIC_HISTORY = {}


def _safe_diagnostic_slug(value):
    text = str(value).strip().lower()
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text]
    slug = "".join(cleaned).strip("_")
    return slug or "graph"


def make_graph_diagnostics_request(config, log_dir, name, epoch=None, title=None):
    if config.graph_diagnostics_mode != "save" or log_dir is None:
        return None

    epoch_slug = "initial" if epoch is None else f"epoch_{int(epoch):04d}"
    seed = int(config.seed if config.seed is not None else 0)
    if epoch is not None:
        seed += int(epoch)
    series_slug = _safe_diagnostic_slug(name)
    return GraphDiagnosticsRequest(
        output_dir=Path(log_dir) / "graph_diagnostics",
        slug=_safe_diagnostic_slug(f"{series_slug}_{epoch_slug}"),
        title=title or str(name),
        max_nodes=int(config.graph_diagnostics_max_nodes),
        max_edges=int(config.graph_diagnostics_max_edges),
        max_labels=int(config.graph_diagnostics_max_labels),
        seed=seed,
        layout=str(config.graph_diagnostics_layout),
        series_slug=series_slug,
        epoch=None if epoch is None else int(epoch),
        class_focus=str(config.graph_diagnostics_class_focus),
        class_count=int(config.graph_diagnostics_class_count),
        classes=(
            None
            if config.graph_diagnostics_classes is None
            else tuple(int(label) for label in config.graph_diagnostics_classes)
        ),
        class_context=bool(config.graph_diagnostics_class_context),
    )


def dataset_labels_for_positions(train_dataset, positions):
    labels = getattr(train_dataset, "labels", None)
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return np.array([], dtype=np.int64)
    if int(positions.max()) >= len(labels):
        return None
    return labels[positions]


def maybe_save_graph_diagnostics(
    request,
    embeddings,
    adjacency,
    positions,
    labels=None,
    known_mask=None,
    graph_metadata=None,
):
    """Write graph diagnostic artifacts without affecting training."""

    if request is None:
        return None
    try:
        return save_graph_diagnostics(
            request=request,
            embeddings=embeddings,
            adjacency=adjacency,
            positions=positions,
            labels=labels,
            known_mask=known_mask,
            graph_metadata=graph_metadata,
        )
    except Exception as exc:  # pragma: no cover - diagnostics must not stop training
        logger.warning(f"Could not save graph diagnostics {request.slug}: {exc}")
        return None


def save_graph_diagnostics(
    request,
    embeddings,
    adjacency,
    positions,
    labels=None,
    known_mask=None,
    graph_metadata=None,
):
    """Write bounded visual artifacts plus complete graph-level/node-level data."""

    plt = None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed; writing graph diagnostic data without PNGs"
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.int64)
    adjacency = adjacency.tocsr(copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    num_nodes = adjacency.shape[0]
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph diagnostics adjacency must be square")
    if len(embeddings) != num_nodes or len(positions) != num_nodes:
        raise ValueError("graph diagnostics embeddings, positions, and adjacency must align")
    if np.any(adjacency.diagonal() != 0):
        raise ValueError("graph diagnostics adjacency must not contain self-loops")
    asymmetry = (adjacency - adjacency.T).tocsr()
    if asymmetry.nnz and not np.allclose(asymmetry.data, 0.0):
        raise ValueError("graph diagnostics adjacency must be symmetric")
    if adjacency.nnz and (
        not np.all(np.isfinite(adjacency.data)) or np.any(adjacency.data <= 0.0)
    ):
        raise ValueError("graph diagnostics adjacency weights must be finite and positive")

    labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    if labels is not None and len(labels) != num_nodes:
        labels = None
    known_mask = None if known_mask is None else np.asarray(known_mask, dtype=bool)
    if known_mask is not None and len(known_mask) != num_nodes:
        known_mask = None

    request.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = graph_diagnostic_artifact_paths(request)
    analysis_started_at = time.perf_counter()
    analysis = analyze_graph_diagnostics(
        request=request,
        adjacency=adjacency,
        positions=positions,
        labels=labels,
        known_mask=known_mask,
        graph_metadata=graph_metadata,
    )

    rng = np.random.default_rng(request.seed)
    focus_classes = analysis["focus_classes"]
    focus_mask = None
    if len(focus_classes):
        node_indices, focus_mask = choose_class_focus_nodes(
            adjacency=adjacency,
            labels=labels,
            focus=focus_classes,
            max_nodes=request.max_nodes,
            include_context=bool(request.class_context),
            rng=rng,
        )
    else:
        node_indices = choose_graph_diagnostic_nodes(
            adjacency=adjacency,
            max_nodes=request.max_nodes,
            rng=rng,
        )
    sampled_adjacency = adjacency[node_indices][:, node_indices]
    drawn_adjacency = sampled_adjacency
    if focus_mask is not None:
        # Context nodes are drawn because the selected classes attach to them.
        # The edges *between* two context nodes say nothing about the selection
        # and would bury the ones that do.
        drawn_adjacency = _restrict_to_incident_edges(sampled_adjacency, focus_mask)
    edge_rows, edge_cols, edge_weights, full_edge_count = sample_graph_diagnostic_edges(
        adjacency=drawn_adjacency,
        max_edges=request.max_edges,
        rng=rng,
    )

    projection_name = None
    if plt is not None:
        coords, projection_name = project_graph_embeddings_2d(
            embeddings[node_indices],
            layout=request.layout,
            seed=request.seed,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        edge_styles = _graph_edge_styles(
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            labels=None if labels is None else labels[node_indices],
            focus_mask=focus_mask,
        )
        for index, (row, col, weight) in enumerate(
            zip(edge_rows, edge_cols, edge_weights)
        ):
            color, width, base_alpha, edge_zorder = edge_styles[index]
            alpha = base_alpha + 0.35 * min(abs(float(weight)), 1.0)
            ax.plot(
                [coords[row, 0], coords[col, 0]],
                [coords[row, 1], coords[col, 1]],
                color=color,
                linewidth=width,
                alpha=min(alpha, 1.0),
                zorder=edge_zorder,
            )
        if focus_mask is not None:
            for color, width, style_label in (
                (GRAPH_EDGE_SAME_CLASS_COLOR, 0.9, "edge within a selected class"),
                (
                    GRAPH_EDGE_BETWEEN_CLASSES_COLOR,
                    0.9,
                    "edge between selected classes",
                ),
                (GRAPH_EDGE_LEAVING_COLOR, 0.5, "edge leaving the selection"),
            ):
                ax.plot([], [], color=color, linewidth=width, label=style_label)

        sampled_labels = None if labels is None else labels[node_indices]
        sampled_known = None if known_mask is None else known_mask[node_indices]
        scatter_graph_nodes(
            ax,
            coords,
            sampled_labels,
            sampled_known,
            focus_mask=focus_mask,
        )
        if len(node_indices) <= request.max_labels:
            label_offsets = (
                (4, 4),
                (-10, 4),
                (4, -11),
                (-10, -11),
                (10, 0),
                (-14, 0),
                (0, 10),
                (0, -14),
            )
            for local_index, position in enumerate(positions[node_indices]):
                ax.annotate(
                    str(int(position)),
                    xy=(coords[local_index, 0], coords[local_index, 1]),
                    xytext=label_offsets[local_index % len(label_offsets)],
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.82,
                    zorder=5,
                )

        graph_summary = analysis["summary"]["graph"]
        degree_summary = analysis["summary"]["degree"]["unweighted"]
        connectivity = analysis["summary"]["connectivity"]
        class_focus_line = ""
        if len(focus_classes):
            focus_summary = analysis["summary"]["class_focus"]
            shown_classes = ", ".join(str(int(label)) for label in focus_classes[:8])
            if len(focus_classes) > 8:
                shown_classes += f", +{len(focus_classes) - 8} more"
            class_focus_line = (
                f"\nclasses {shown_classes} "
                f"({request.class_focus}): "
                f"{focus_summary['same_class_edge_count']} same-class, "
                f"{focus_summary['between_selected_classes_edge_count']} between selected, "
                f"{focus_summary['edge_count_to_classes_outside_selection']} leaving edges"
            )
        ax.set_title(
            f"{request.title}\n"
            f"showing {len(node_indices)}/{num_nodes} samples and "
            f"{len(edge_rows)}/{graph_summary['undirected_edge_count']} undirected edges\n"
            f"full graph: mean degree={degree_summary.get('mean', 0.0):.2f}, "
            f"components={connectivity['component_count']}, "
            f"isolated={connectivity['isolated_node_count']}"
            f"{class_focus_line}",
            fontsize=10,
        )
        ax.set_xlabel(f"{projection_name} 1")
        ax.set_ylabel(f"{projection_name} 2")
        ax.tick_params(labelsize=8)
        ax.legend(
            loc="best",
            fontsize=7 if focus_mask is not None else 9,
            ncol=2 if focus_mask is not None else 1,
        )
        fig.tight_layout()
        fig.savefig(artifact_paths["graph_png"], dpi=160)
        plt.close(fig)

        write_graph_distribution_png(
            artifact_paths["distributions_png"],
            analysis["plot_data"],
            title=request.title,
        )

    write_graph_edge_csv(
        csv_path=artifact_paths["sampled_edges_csv"],
        node_indices=node_indices,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        edge_weights=edge_weights,
        positions=positions,
        labels=labels,
        known_mask=known_mask,
    )
    write_graph_node_csv(
        csv_path=artifact_paths["nodes_csv"],
        node_data=analysis["node_data"],
    )
    if analysis["rank_rows"]:
        write_dict_rows_csv(
            artifact_paths["similarity_by_rank_csv"],
            analysis["rank_rows"],
        )
    if analysis["class_pair_rows"]:
        write_dict_rows_csv(
            artifact_paths["class_pair_edges_csv"],
            analysis["class_pair_rows"],
        )
    if analysis["class_focus_rows"]:
        write_dict_rows_csv(
            artifact_paths["class_focus_edges_csv"],
            analysis["class_focus_rows"],
        )

    created_artifacts = {
        "summary_json": artifact_paths["summary_json"].name,
        "nodes_csv": artifact_paths["nodes_csv"].name,
        "sampled_edges_csv": artifact_paths["sampled_edges_csv"].name,
    }
    if plt is not None:
        created_artifacts.update(
            {
                "graph_png": artifact_paths["graph_png"].name,
                "distributions_png": artifact_paths["distributions_png"].name,
            }
        )
    if analysis["rank_rows"]:
        created_artifacts["similarity_by_rank_csv"] = artifact_paths[
            "similarity_by_rank_csv"
        ].name
    if analysis["class_pair_rows"]:
        created_artifacts["class_pair_edges_csv"] = artifact_paths[
            "class_pair_edges_csv"
        ].name
    if analysis["class_focus_rows"]:
        created_artifacts["class_focus_edges_csv"] = artifact_paths[
            "class_focus_edges_csv"
        ].name
    analysis["summary"]["artifacts"] = created_artifacts
    analysis["summary"]["visualization"] = {
        "statistics_scope": "full_graph",
        "graph_png_scope": (
            "class_focus_induced_subgraph"
            if focus_mask is not None
            else "sampled_induced_subgraph"
        ),
        "class_focus_node_count": (
            None if focus_mask is None else int(focus_mask.sum())
        ),
        "class_focus_context_node_count": (
            None if focus_mask is None else int((~focus_mask).sum())
        ),
        "drawn_edge_scope": (
            "edges_incident_to_the_selected_classes"
            if focus_mask is not None
            else "all_edges_among_sampled_nodes"
        ),
        "sampled_node_count": int(len(node_indices)),
        "sampled_edge_count": int(len(edge_rows)),
        "available_sampled_induced_edge_count": int(full_edge_count),
        "projection": projection_name,
        "requested_layout": request.layout,
        "max_nodes": int(request.max_nodes),
        "max_edges": int(request.max_edges),
        "max_labels": int(request.max_labels),
        "seed": int(request.seed),
    }
    analysis["summary"]["diagnostics_generation_seconds"] = (
        time.perf_counter() - analysis_started_at
    )
    write_graph_summary_json(
        artifact_paths["summary_json"],
        analysis["summary"],
    )

    logger.info(
        "Saved graph diagnostics: "
        f"summary={artifact_paths['summary_json']}, "
        f"nodes={artifact_paths['nodes_csv']}, "
        f"png={artifact_paths['graph_png'] if plt is not None else 'unavailable'}, "
        f"sampled_edges={artifact_paths['sampled_edges_csv']}, "
        f"sampled_nodes={len(node_indices)}, sampled_edges={len(edge_rows)}, "
        f"available_sampled_edges={full_edge_count}, projection={projection_name}, "
        f"mean_degree={analysis['summary']['degree']['unweighted'].get('mean', 0.0):.4f}, "
        f"mean_weighted_degree="
        f"{analysis['summary']['degree']['weighted'].get('mean', 0.0):.6f}"
    )
    return (
        artifact_paths["graph_png"]
        if plt is not None
        else artifact_paths["summary_json"]
    )


def graph_diagnostic_artifact_paths(request):
    base = request.output_dir / request.slug
    return {
        "graph_png": base.with_suffix(".png"),
        "summary_json": request.output_dir / f"{request.slug}_summary.json",
        "nodes_csv": request.output_dir / f"{request.slug}_nodes.csv",
        "sampled_edges_csv": request.output_dir / f"{request.slug}_sampled_edges.csv",
        "distributions_png": request.output_dir / f"{request.slug}_distributions.png",
        "similarity_by_rank_csv": request.output_dir
        / f"{request.slug}_similarity_by_rank.csv",
        "class_pair_edges_csv": request.output_dir
        / f"{request.slug}_class_pair_edges.csv",
        "class_focus_edges_csv": request.output_dir
        / f"{request.slug}_class_focus_edges.csv",
    }


def _graph_diagnostic_series_slug(request):
    if request.series_slug:
        return str(request.series_slug)
    return re.sub(r"_(?:initial|epoch_\d+)$", "", str(request.slug))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def numeric_diagnostic_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    summary = {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(values) - len(finite)),
    }
    if len(finite) == 0:
        return summary
    summary.update(
        {
            "min": float(finite.min()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "max": float(finite.max()),
        }
    )
    percentile_values = np.percentile(finite, GRAPH_DIAGNOSTICS_PERCENTILES)
    for percentile, value in zip(
        GRAPH_DIAGNOSTICS_PERCENTILES,
        percentile_values,
    ):
        summary[f"p{int(percentile):02d}"] = float(value)
    return summary


def _summary_csv_columns(prefix, summary):
    columns = {}
    for name in ("min", "mean", "std", "p05", "p25", "p50", "p75", "p95", "max"):
        columns[f"{prefix}_{name}"] = summary.get(name, "")
    return columns


def _csr_memory_bytes(matrix):
    matrix = matrix.tocsr()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _safe_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _multi_source_hop_distances(adjacency, source_mask):
    """Return unweighted shortest-hop distance from the closest selected node."""

    source_mask = np.asarray(source_mask, dtype=bool)
    distances = np.full(adjacency.shape[0], np.inf, dtype=np.float64)
    sources = np.flatnonzero(source_mask)
    if len(sources) == 0:
        return distances

    queue = deque(int(source) for source in sources)
    distances[sources] = 0.0
    indptr = adjacency.indptr
    indices = adjacency.indices
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1.0
        for neighbor in indices[indptr[node] : indptr[node + 1]]:
            neighbor = int(neighbor)
            if not np.isfinite(distances[neighbor]):
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def _correct_label_anchor_distances(adjacency, labels, known_mask):
    """Return oracle hop distances to a labeled node of the node's true class."""

    num_nodes = adjacency.shape[0]
    distances = np.full(num_nodes, np.nan, dtype=np.float64)
    if labels is None or known_mask is None:
        return distances, {"status": "unavailable", "reason": "labels_or_known_mask_missing"}

    valid = labels >= 0
    classes = np.unique(labels[valid])
    estimated_work = int(adjacency.nnz) * int(len(classes))
    if (
        len(classes) > GRAPH_DIAGNOSTICS_MAX_CORRECT_ANCHOR_CLASSES
        or estimated_work > GRAPH_DIAGNOSTICS_MAX_CORRECT_ANCHOR_WORK
    ):
        return distances, {
            "status": "skipped",
            "reason": "graph_or_class_count_too_large",
            "class_count": int(len(classes)),
            "estimated_edge_visits": estimated_work,
        }

    computed_classes = 0
    for label in classes:
        targets = labels == label
        sources = np.flatnonzero(targets & known_mask)
        if len(sources) == 0:
            distances[targets] = np.inf
            continue
        class_distances = sparse.csgraph.dijkstra(
            adjacency,
            directed=False,
            indices=sources,
            unweighted=True,
            min_only=True,
        )
        distances[targets] = np.asarray(
            class_distances,
            dtype=np.float64,
        )[targets]
        computed_classes += 1
    return distances, {
        "status": "computed",
        "class_count": int(len(classes)),
        "classes_with_labeled_anchor": int(computed_classes),
    }


def _spectral_graph_summary(adjacency, component_count):
    num_nodes = adjacency.shape[0]
    if num_nodes == 0:
        return {"status": "unavailable", "reason": "empty_graph"}
    if num_nodes == 1:
        return {
            "status": "computed",
            "normalized_laplacian_smallest_eigenvalues": [0.0],
            "algebraic_connectivity": 0.0,
            "smallest_positive_eigenvalue": None,
        }
    if (
        num_nodes > GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NODES
        or adjacency.nnz > GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NNZ
    ):
        return {
            "status": "skipped",
            "reason": "graph_too_large",
            "node_limit": GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NODES,
            "nnz_limit": GRAPH_DIAGNOSTICS_MAX_SPECTRAL_NNZ,
        }

    try:
        laplacian = sparse.csgraph.laplacian(
            adjacency,
            normed=True,
        )
        if num_nodes <= 512:
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            eigenvalues = eigenvalues[: min(8, len(eigenvalues))]
            backend = "dense_eigvalsh"
        else:
            k = min(8, num_nodes - 1)
            eigenvalues = sparse.linalg.eigsh(
                laplacian,
                k=k,
                which="SM",
                return_eigenvectors=False,
                tol=1e-3,
                maxiter=2000,
            )
            eigenvalues = np.sort(eigenvalues)
            backend = "sparse_eigsh"
        eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
        positive = eigenvalues[eigenvalues > 1e-8]
        return {
            "status": "computed",
            "backend": backend,
            "normalized_laplacian_smallest_eigenvalues": eigenvalues.tolist(),
            "algebraic_connectivity": (
                0.0
                if component_count > 1
                else (
                    float(eigenvalues[1])
                    if len(eigenvalues) > 1
                    else 0.0
                )
            ),
            "smallest_positive_eigenvalue": (
                None if len(positive) == 0 else float(positive[0])
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
        }


def _row_weight_concentration(adjacency):
    num_nodes = adjacency.shape[0]
    strongest_neighbors = np.full(num_nodes, -1, dtype=np.int64)
    strongest_weights = np.zeros(num_nodes, dtype=np.float64)
    top_share_arrays = {
        top_k: np.zeros(num_nodes, dtype=np.float64)
        for top_k in (1, 5, 10, 20)
    }

    for node in range(num_nodes):
        start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
        row_weights = adjacency.data[start:end]
        row_neighbors = adjacency.indices[start:end]
        if len(row_weights) == 0:
            continue
        strongest_offset = int(np.argmax(row_weights))
        strongest_neighbors[node] = int(row_neighbors[strongest_offset])
        strongest_weights[node] = float(row_weights[strongest_offset])
        total_weight = float(row_weights.sum())
        sorted_weights = np.sort(row_weights)[::-1]
        for top_k in top_share_arrays:
            top_share_arrays[top_k][node] = (
                float(sorted_weights[:top_k].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            )
    return strongest_neighbors, strongest_weights, top_share_arrays


def _edge_label_quality(
    upper,
    labels,
    known_mask,
    mutual_edge_mask=None,
    candidate_edge_mask=None,
):
    if labels is None:
        return {
            "status": "unavailable",
            "reason": "labels_missing",
        }, [], None

    valid = (labels[upper.row] >= 0) & (labels[upper.col] >= 0)
    same = valid & (labels[upper.row] == labels[upper.col])
    valid_count = int(valid.sum())
    same_count = int(same.sum())
    valid_weight = float(upper.data[valid].sum())
    same_weight = float(upper.data[same].sum())
    quality = {
        "status": "computed",
        "uses_oracle_labels": True,
        "valid_labeled_edge_count": valid_count,
        "same_label_edge_count": same_count,
        "edge_purity": _safe_ratio(same_count, valid_count),
        "oracle_edge_purity": _safe_ratio(same_count, valid_count),
        "valid_labeled_edge_weight": valid_weight,
        "same_label_edge_weight": same_weight,
        "weighted_edge_purity": _safe_ratio(same_weight, valid_weight),
        "oracle_weighted_edge_purity": _safe_ratio(
            same_weight,
            valid_weight,
        ),
    }

    if known_mask is not None:
        edge_kinds = {
            "labeled_labeled": known_mask[upper.row] & known_mask[upper.col],
            "labeled_unlabeled": known_mask[upper.row] ^ known_mask[upper.col],
            "unlabeled_unlabeled": (~known_mask[upper.row]) & (~known_mask[upper.col]),
        }
        quality["by_endpoint_kind"] = {}
        for name, kind_mask in edge_kinds.items():
            kind_valid = valid & kind_mask
            kind_same = same & kind_mask
            kind_valid_count = int(kind_valid.sum())
            kind_valid_weight = float(upper.data[kind_valid].sum())
            quality["by_endpoint_kind"][name] = {
                "edge_count": int(kind_mask.sum()),
                "valid_labeled_edge_count": kind_valid_count,
                "edge_purity": _safe_ratio(
                    int(kind_same.sum()),
                    kind_valid_count,
                ),
                "total_weight": float(upper.data[kind_mask].sum()),
                "weighted_edge_purity": _safe_ratio(
                    float(upper.data[kind_same].sum()),
                    kind_valid_weight,
                ),
            }

    if mutual_edge_mask is not None:
        if candidate_edge_mask is None:
            candidate_edge_mask = np.ones(len(upper.data), dtype=bool)
        direction_masks = {
            "mutual": mutual_edge_mask,
            "one_way": candidate_edge_mask & (~mutual_edge_mask),
        }
        if np.any(~candidate_edge_mask):
            direction_masks["outside_directed_knn"] = ~candidate_edge_mask
        quality["by_knn_directionality"] = {}
        for name, direction_mask in direction_masks.items():
            direction_valid = valid & direction_mask
            direction_same = same & direction_mask
            valid_direction_count = int(direction_valid.sum())
            valid_direction_weight = float(upper.data[direction_valid].sum())
            quality["by_knn_directionality"][name] = {
                "edge_count": int(direction_mask.sum()),
                "valid_labeled_edge_count": valid_direction_count,
                "edge_purity": _safe_ratio(
                    int(direction_same.sum()),
                    valid_direction_count,
                ),
                "total_weight": float(upper.data[direction_mask].sum()),
                "weighted_edge_purity": _safe_ratio(
                    float(upper.data[direction_same].sum()),
                    valid_direction_weight,
                ),
            }

    per_class = defaultdict(lambda: {"edge_count": 0, "same_count": 0, "weight": 0.0, "same_weight": 0.0})
    class_pairs = defaultdict(
        lambda: {
            "edge_count": 0,
            "total_weight": 0.0,
            "max_weight": 0.0,
            "mutual_edge_count": 0,
        }
    )
    valid_indices = np.flatnonzero(valid)
    for edge_index in valid_indices:
        left_label = int(labels[int(upper.row[edge_index])])
        right_label = int(labels[int(upper.col[edge_index])])
        weight = float(upper.data[edge_index])
        is_same = left_label == right_label
        for label in {left_label, right_label}:
            record = per_class[label]
            record["edge_count"] += 1
            record["weight"] += weight
            if is_same:
                record["same_count"] += 1
                record["same_weight"] += weight
        if is_same:
            continue
        pair = tuple(sorted((left_label, right_label)))
        pair_record = class_pairs[pair]
        pair_record["edge_count"] += 1
        pair_record["total_weight"] += weight
        pair_record["max_weight"] = max(pair_record["max_weight"], weight)
        if mutual_edge_mask is not None and bool(mutual_edge_mask[edge_index]):
            pair_record["mutual_edge_count"] += 1

    per_class_rows = []
    for label, record in per_class.items():
        per_class_rows.append(
            {
                "label": int(label),
                "incident_valid_edge_count": int(record["edge_count"]),
                "edge_purity": _safe_ratio(
                    record["same_count"],
                    record["edge_count"],
                ),
                "incident_edge_weight": float(record["weight"]),
                "weighted_edge_purity": _safe_ratio(
                    record["same_weight"],
                    record["weight"],
                ),
            }
        )
    per_class_rows.sort(
        key=lambda row: (
            1.0 if row["weighted_edge_purity"] is None else row["weighted_edge_purity"],
            row["label"],
        )
    )
    quality["per_class_edge_quality"] = per_class_rows[
        :GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS
    ]
    quality["per_class_edge_quality_truncated"] = (
        len(per_class_rows) > GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS
    )

    class_pair_rows = []
    for (left_label, right_label), record in class_pairs.items():
        class_pair_rows.append(
            {
                "class_a": left_label,
                "class_b": right_label,
                "edge_count": int(record["edge_count"]),
                "total_weight": float(record["total_weight"]),
                "mean_weight": _safe_ratio(
                    record["total_weight"],
                    record["edge_count"],
                ),
                "max_weight": float(record["max_weight"]),
                "mutual_edge_count": int(record["mutual_edge_count"]),
            }
        )
    class_pair_rows.sort(
        key=lambda row: (row["total_weight"], row["edge_count"]),
        reverse=True,
    )
    quality["cross_class_pair_count"] = int(len(class_pair_rows))
    quality["class_pair_file_row_limit"] = GRAPH_DIAGNOSTICS_MAX_CLASS_PAIRS
    quality["class_pair_file_truncated"] = (
        len(class_pair_rows) > GRAPH_DIAGNOSTICS_MAX_CLASS_PAIRS
    )
    return (
        quality,
        class_pair_rows[:GRAPH_DIAGNOSTICS_MAX_CLASS_PAIRS],
        same,
    )


def _directed_construction_diagnostics(graph_metadata, num_nodes, labels):
    graph_metadata = {} if graph_metadata is None else dict(graph_metadata)
    neighbors = graph_metadata.get("neighbor_indices")
    similarities = graph_metadata.get("neighbor_similarities")
    query_indices = graph_metadata.get("query_indices")
    gamma = graph_metadata.get("gamma")
    requested_k = graph_metadata.get("requested_n_neighbors")
    search_k = graph_metadata.get("search_n_neighbors")

    construction = {
        "graph_kind": graph_metadata.get("graph_kind", "symmetric_affinity"),
        "requested_k_definition": (
            "maximum directed non-self candidates requested per query node"
        ),
        "final_degree_note": (
            "final degree can differ from k after positive-part clipping and "
            "undirected union/symmetrization"
        ),
        "requested_n_neighbors": (
            None if requested_k is None else int(requested_k)
        ),
        "search_n_neighbors": None if search_k is None else int(search_k),
        "gamma": None if gamma is None else float(gamma),
    }
    for name in (
        "include_supervised_graph",
        "supervised_positive_pair_count",
        "construction_seconds",
        "search_seconds",
    ):
        if name in graph_metadata:
            construction[name] = graph_metadata[name]
    rank_rows = []
    mutual_edge_keys = None
    candidate_edge_keys = None
    positive_out_degree = np.full(num_nodes, np.nan, dtype=np.float64)
    incoming_positive_degree = np.full(num_nodes, np.nan, dtype=np.float64)

    if neighbors is None:
        construction["directed_candidate_diagnostics"] = {
            "status": "unavailable",
            "reason": "neighbor_indices_missing",
        }
        return (
            construction,
            rank_rows,
            mutual_edge_keys,
            candidate_edge_keys,
            positive_out_degree,
            incoming_positive_degree,
        )

    neighbors = np.asarray(neighbors, dtype=np.int64)
    if query_indices is None:
        query_indices = np.arange(len(neighbors), dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    similarities_available = similarities is not None
    if similarities_available:
        similarities = np.asarray(similarities, dtype=np.float64)
    if (
        neighbors.ndim != 2
        or (
            similarities_available
            and similarities.shape != neighbors.shape
        )
        or len(query_indices) != len(neighbors)
        or len(np.unique(query_indices)) != len(query_indices)
        or np.any((query_indices < 0) | (query_indices >= num_nodes))
    ):
        construction["directed_candidate_diagnostics"] = {
            "status": "unavailable",
            "reason": "neighbor_metadata_shape_mismatch",
        }
        return (
            construction,
            rank_rows,
            mutual_edge_keys,
            candidate_edge_keys,
            positive_out_degree,
            incoming_positive_degree,
        )

    if np.any((neighbors < 0) | (neighbors >= num_nodes)):
        construction["directed_candidate_diagnostics"] = {
            "status": "unavailable",
            "reason": "neighbor_index_out_of_range",
        }
        return (
            construction,
            rank_rows,
            mutual_edge_keys,
            candidate_edge_keys,
            positive_out_degree,
            incoming_positive_degree,
        )

    positive = (
        similarities > 0.0
        if similarities_available
        else np.ones(neighbors.shape, dtype=bool)
    )
    positive_out_degree[query_indices] = positive.sum(axis=1).astype(np.float64)
    positive_queries = np.repeat(query_indices, neighbors.shape[1])[
        positive.ravel()
    ]
    positive_neighbors = neighbors.ravel()[positive.ravel()]
    incoming_positive_degree = np.bincount(
        positive_neighbors,
        minlength=num_nodes,
    ).astype(np.float64)
    directed = sparse.coo_matrix(
        (
            np.ones(len(positive_queries), dtype=np.int8),
            (positive_queries, positive_neighbors),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    directed.sum_duplicates()
    directed.data[:] = 1
    mutual = directed.multiply(directed.T).tocsr()
    mutual_upper = sparse.triu(mutual, k=1).tocoo()
    final_upper = sparse.triu((directed + directed.T).astype(bool), k=1).tocoo()
    mutual_keys = (
        mutual_upper.row.astype(np.int64) * int(num_nodes)
        + mutual_upper.col.astype(np.int64)
    )
    final_keys = (
        final_upper.row.astype(np.int64) * int(num_nodes)
        + final_upper.col.astype(np.int64)
    )
    mutual_edge_keys = np.unique(mutual_keys)
    candidate_edge_keys = np.unique(final_keys)

    directed_candidate_count = int(neighbors.size)
    positive_candidate_count = int(positive.sum())
    mutual_edge_count = int(mutual_upper.nnz)
    one_way_edge_count = int(final_upper.nnz - mutual_edge_count)
    construction["directed_candidate_diagnostics"] = {
        "status": "computed",
        "similarities_available": bool(similarities_available),
        "query_node_count": int(len(query_indices)),
        "candidate_count": directed_candidate_count,
        "positive_candidate_count": positive_candidate_count,
        "nonpositive_candidate_count": int(
            directed_candidate_count - positive_candidate_count
        ),
        "positive_candidate_fraction": _safe_ratio(
            positive_candidate_count,
            directed_candidate_count,
        ),
        "nodes_losing_at_least_one_candidate": int(
            np.sum(positive_out_degree[query_indices] < neighbors.shape[1])
        ),
        "positive_out_degree": numeric_diagnostic_summary(positive_out_degree),
        "incoming_positive_degree": numeric_diagnostic_summary(
            incoming_positive_degree
        ),
        "mutual_undirected_edge_count": mutual_edge_count,
        "one_way_undirected_edge_count": one_way_edge_count,
        "mutual_edge_fraction": _safe_ratio(
            mutual_edge_count,
            final_upper.nnz,
        ),
    }

    if not similarities_available:
        affinity_values = np.ones(neighbors.shape, dtype=np.float64)
    elif gamma is None:
        affinity_values = np.where(positive, 1.0, 0.0)
    else:
        affinity_values = np.clip(similarities, 0.0, None) ** float(gamma)
    for rank in range(neighbors.shape[1]):
        rank_similarities = (
            similarities[:, rank]
            if similarities_available
            else np.asarray([], dtype=np.float64)
        )
        rank_positive = positive[:, rank]
        similarity_summary = numeric_diagnostic_summary(rank_similarities)
        positive_similarity_summary = numeric_diagnostic_summary(
            rank_similarities[rank_positive]
            if similarities_available
            else rank_similarities
        )
        affinity_summary = numeric_diagnostic_summary(
            affinity_values[:, rank][rank_positive]
        )
        row = {
            "neighbor_rank": int(rank + 1),
            "candidate_count": int(len(query_indices)),
            "positive_count": int(rank_positive.sum()),
            "positive_fraction": _safe_ratio(
                int(rank_positive.sum()),
                len(query_indices),
            ),
            **_summary_csv_columns("similarity", similarity_summary),
            **_summary_csv_columns(
                "positive_similarity",
                positive_similarity_summary,
            ),
            **_summary_csv_columns("affinity", affinity_summary),
        }
        if labels is not None:
            rank_neighbors = neighbors[:, rank]
            valid = (
                (labels[query_indices] >= 0)
                & (labels[rank_neighbors] >= 0)
            )
            same = valid & (
                labels[query_indices] == labels[rank_neighbors]
            )
            positive_valid = valid & rank_positive
            row.update(
                {
                    "valid_label_count": int(valid.sum()),
                    "same_label_fraction": _safe_ratio(
                        int(same.sum()),
                        int(valid.sum()),
                    ),
                    "positive_valid_label_count": int(positive_valid.sum()),
                    "positive_same_label_fraction": _safe_ratio(
                        int((same & rank_positive).sum()),
                        int(positive_valid.sum()),
                    ),
                }
            )
        rank_rows.append(row)

    return (
        construction,
        rank_rows,
        mutual_edge_keys,
        candidate_edge_keys,
        positive_out_degree,
        incoming_positive_degree,
    )


def _temporal_graph_diagnostics(
    request,
    positions,
    upper,
    degree,
    weighted_degree,
    component_count,
    edge_purity,
):
    series_key = (
        str(Path(request.output_dir).resolve()),
        _graph_diagnostic_series_slug(request),
    )
    num_nodes = len(positions)
    edge_keys = (
        upper.row.astype(np.int64) * int(max(num_nodes, 1))
        + upper.col.astype(np.int64)
    )
    order = np.argsort(edge_keys)
    edge_keys = edge_keys[order]
    edge_weights = upper.data[order].astype(np.float64, copy=True)
    previous_entry = _GRAPH_DIAGNOSTIC_HISTORY.get(series_key)
    previous = None if previous_entry is None else previous_entry["current"]

    current = {
        "slug": request.slug,
        "epoch": request.epoch,
        "positions": positions.copy(),
        "edge_keys": edge_keys,
        "edge_weights": edge_weights,
        "degree": degree.copy(),
        "weighted_degree": weighted_degree.copy(),
        "component_count": int(component_count),
        "edge_purity": edge_purity,
        "predicted_labels": None,
        "confidences": None,
        "propagated_mask": None,
    }
    temporal = {
        "status": "unavailable",
        "reason": "first_graph_in_process_or_process_restarted",
    }
    if previous is not None:
        if (
            len(previous["positions"]) != len(positions)
            or not np.array_equal(previous["positions"], positions)
        ):
            temporal = {
                "status": "unavailable",
                "reason": "graph_node_order_or_membership_changed",
                "previous_slug": previous["slug"],
            }
        else:
            common, previous_indices, current_indices = np.intersect1d(
                previous["edge_keys"],
                edge_keys,
                assume_unique=True,
                return_indices=True,
            )
            union_count = (
                len(previous["edge_keys"]) + len(edge_keys) - len(common)
            )
            retained_degree = np.zeros(num_nodes, dtype=np.float64)
            if len(common) > 0:
                common_rows = common // max(num_nodes, 1)
                common_cols = common % max(num_nodes, 1)
                retained_degree += np.bincount(
                    common_rows,
                    minlength=num_nodes,
                )
                retained_degree += np.bincount(
                    common_cols,
                    minlength=num_nodes,
                )
            previous_degree = previous["degree"]
            retention_fraction = np.divide(
                retained_degree,
                previous_degree,
                out=np.zeros_like(retained_degree),
                where=previous_degree > 0,
            )
            common_weight_delta = (
                edge_weights[current_indices]
                - previous["edge_weights"][previous_indices]
            )
            temporal = {
                "status": "computed",
                "previous_slug": previous["slug"],
                "previous_epoch": previous["epoch"],
                "common_edge_count": int(len(common)),
                "edge_union_count": int(union_count),
                "edge_jaccard": _safe_ratio(len(common), union_count),
                "previous_edges_retained_fraction": _safe_ratio(
                    len(common),
                    len(previous["edge_keys"]),
                ),
                "current_edges_already_present_fraction": _safe_ratio(
                    len(common),
                    len(edge_keys),
                ),
                "per_node_previous_neighbor_retention": (
                    numeric_diagnostic_summary(retention_fraction)
                ),
                "degree_delta": numeric_diagnostic_summary(
                    degree - previous_degree
                ),
                "weighted_degree_delta": numeric_diagnostic_summary(
                    weighted_degree - previous["weighted_degree"]
                ),
                "common_edge_weight_delta": numeric_diagnostic_summary(
                    common_weight_delta
                ),
                "component_count_delta": int(
                    component_count - previous["component_count"]
                ),
                "edge_purity_delta": (
                    None
                    if edge_purity is None or previous["edge_purity"] is None
                    else float(edge_purity - previous["edge_purity"])
                ),
            }

    _GRAPH_DIAGNOSTIC_HISTORY[series_key] = {
        "previous": previous,
        "current": current,
    }
    return temporal


def select_focus_classes(request, labels, per_class_rows, rng):
    """Choose the classes the class-scoped view reports on.

    Selection runs on the true labels of every graph node, unlabeled ones
    included, which is the whole point of the view: the run withheld those
    labels from the model, so they are available here to score what the graph
    did with the samples the model had to place on its own.
    """

    present, counts = np.unique(labels[labels >= 0], return_counts=True)
    # A single-node class has no internal structure to report on and would
    # otherwise dominate a purity ranking with a degenerate 0.0.
    eligible = present[counts >= 2]
    if len(eligible) == 0:
        return np.array([], dtype=np.int64), {
            "status": "unavailable",
            "reason": "no_class_has_at_least_two_graph_nodes",
        }

    count = int(request.class_count)
    mode = str(request.class_focus)
    if mode == "explicit":
        requested = np.asarray(
            () if request.classes is None else request.classes,
            dtype=np.int64,
        )
        chosen = requested[np.isin(requested, eligible)]
        missing = requested[~np.isin(requested, eligible)]
        if len(missing):
            logger.warning(
                "graph_diagnostics_classes lists classes that are absent from this "
                "graph or have fewer than two nodes in it, skipping them: "
                f"{missing[:10].tolist()}"
            )
        if len(chosen) == 0:
            return np.array([], dtype=np.int64), {
                "status": "unavailable",
                "reason": "no_requested_class_is_present_in_this_graph",
                "requested_classes": requested.tolist(),
            }
        return chosen.astype(np.int64), None
    if mode == "largest":
        eligible_counts = counts[counts >= 2]
        order = np.argsort(-eligible_counts, kind="stable")
        return eligible[order[:count]].astype(np.int64), None
    if mode == "random":
        size = min(count, len(eligible))
        return np.sort(
            rng.choice(eligible, size=size, replace=False)
        ).astype(np.int64), None
    if mode == "lowest_purity":
        # ``per_class_rows`` is already sorted worst-purity first.
        eligible_labels = set(eligible.tolist())
        ranked = [
            int(row["label"])
            for row in per_class_rows
            if int(row["label"]) in eligible_labels
        ]
        if not ranked:
            return np.array([], dtype=np.int64), {
                "status": "unavailable",
                "reason": "per_class_edge_quality_unavailable",
            }
        return np.asarray(ranked[:count], dtype=np.int64), None
    return np.array([], dtype=np.int64), {
        "status": "unavailable",
        "reason": f"unknown_class_focus_mode_{mode}",
    }


def _class_focus_diagnostics(
    request,
    adjacency,
    upper,
    labels,
    known_mask,
    component_ids,
    correct_anchor_distances,
    per_class_rows,
    rng,
):
    """Report how the selected classes' nodes ended up connected.

    Every count here is full-graph scoped. The plot's node budget bounds only
    what gets drawn; the numbers below describe every node of every selected
    class.
    """

    if str(request.class_focus) == "off":
        return {"status": "disabled"}, np.array([], dtype=np.int64)
    if labels is None:
        return {
            "status": "unavailable",
            "reason": "labels_missing",
        }, np.array([], dtype=np.int64)

    focus, unavailable = select_focus_classes(request, labels, per_class_rows, rng)
    if unavailable is not None:
        unavailable["selection_mode"] = str(request.class_focus)
        return unavailable, np.array([], dtype=np.int64)

    num_nodes = adjacency.shape[0]
    focus_size = len(focus)
    outside_code = focus_size
    unknown_code = focus_size + 1
    lookup = np.full(int(labels.max()) + 2, outside_code, dtype=np.int64)
    lookup[focus] = np.arange(focus_size, dtype=np.int64)
    code = lookup[np.maximum(labels, 0)]
    code[labels < 0] = unknown_code

    # One pass over the upper triangle classifies every edge of the graph into
    # the (class, class) cell it belongs to, so no per-class edge scan is needed.
    pair_counts = np.zeros((focus_size + 2, focus_size + 2), dtype=np.int64)
    pair_weights = np.zeros((focus_size + 2, focus_size + 2), dtype=np.float64)
    np.add.at(pair_counts, (code[upper.row], code[upper.col]), 1)
    np.add.at(pair_weights, (code[upper.row], code[upper.col]), upper.data)
    diagonal_counts = np.diag(pair_counts).copy()
    diagonal_weights = np.diag(pair_weights).copy()
    pair_counts = pair_counts + pair_counts.T
    pair_weights = pair_weights + pair_weights.T
    np.fill_diagonal(pair_counts, diagonal_counts)
    np.fill_diagonal(pair_weights, diagonal_weights)

    binary = adjacency.copy()
    binary.data[:] = 1.0
    member_masks = np.stack(
        [(labels == int(label)) for label in focus],
        axis=1,
    ).astype(np.float64)
    labeled_member_masks = (
        member_masks
        if known_mask is None
        else member_masks * known_mask.astype(np.float64)[:, None]
    )
    # (N, F): how many labeled anchors of each focus class each node touches.
    anchor_reach = np.asarray(binary @ labeled_member_masks, dtype=np.float64)

    class_rows = []
    for index, label in enumerate(focus):
        members = np.flatnonzero(labels == int(label))
        member_count = int(len(members))
        internal_count = int(pair_counts[index, index])
        to_focus_count = int(pair_counts[index, :focus_size].sum() - internal_count)
        to_outside_count = int(pair_counts[index, outside_code])
        to_unknown_count = int(pair_counts[index, unknown_code])
        incident_count = (
            internal_count + to_focus_count + to_outside_count + to_unknown_count
        )
        internal_weight = float(pair_weights[index, index])
        to_focus_weight = float(
            pair_weights[index, :focus_size].sum() - internal_weight
        )
        to_outside_weight = float(pair_weights[index, outside_code])
        to_unknown_weight = float(pair_weights[index, unknown_code])
        incident_weight = (
            internal_weight + to_focus_weight + to_outside_weight + to_unknown_weight
        )

        # Connectivity of the class taken on its own: does it form one blob, or
        # did the graph shatter it into pieces that can never exchange signal?
        internal_adjacency = adjacency[members][:, members]
        internal_components, internal_component_ids = sparse.csgraph.connected_components(
            internal_adjacency,
            directed=False,
            return_labels=True,
        )
        internal_component_sizes = np.bincount(
            internal_component_ids,
            minlength=internal_components,
        )
        internal_degree = np.diff(internal_adjacency.indptr).astype(np.float64)
        total_degree = np.diff(adjacency.indptr).astype(np.float64)[members]

        labeled_members = (
            None if known_mask is None else known_mask[members]
        )
        unlabeled_members = (
            None if labeled_members is None else ~labeled_members
        )
        unlabeled_member_count = (
            None if unlabeled_members is None else int(unlabeled_members.sum())
        )
        attached_unlabeled = (
            None
            if unlabeled_members is None
            else int((anchor_reach[members, index][unlabeled_members] > 0).sum())
        )

        anchor_distances = (
            None
            if correct_anchor_distances is None
            else correct_anchor_distances[members]
        )
        row = {
            "label": int(label),
            "node_count": member_count,
            "labeled_node_count": (
                None if labeled_members is None else int(labeled_members.sum())
            ),
            "unlabeled_node_count": unlabeled_member_count,
            "same_class_edge_count": internal_count,
            "edge_count_to_other_focus_classes": to_focus_count,
            "edge_count_to_classes_outside_selection": to_outside_count,
            "edge_count_to_unlabeled_class_nodes": to_unknown_count,
            "incident_edge_count": incident_count,
            "edge_purity": _safe_ratio(internal_count, incident_count),
            "same_class_edge_weight": internal_weight,
            "edge_weight_to_other_focus_classes": to_focus_weight,
            "edge_weight_to_classes_outside_selection": to_outside_weight,
            "incident_edge_weight": incident_weight,
            "weighted_edge_purity": _safe_ratio(internal_weight, incident_weight),
            "mean_same_class_degree": float(internal_degree.mean()),
            "mean_total_degree": float(total_degree.mean()),
            "isolated_within_class_node_count": int((internal_degree == 0).sum()),
            "same_class_component_count": int(internal_components),
            "largest_same_class_component_size": int(
                internal_component_sizes.max() if internal_components else 0
            ),
            "largest_same_class_component_fraction": _safe_ratio(
                int(internal_component_sizes.max()) if internal_components else 0,
                member_count,
            ),
            "spans_full_graph_component_count": int(
                len(np.unique(component_ids[members]))
            ),
            "unlabeled_nodes_touching_a_same_class_labeled_node": attached_unlabeled,
            "unlabeled_nodes_touching_a_same_class_labeled_node_fraction": (
                None
                if attached_unlabeled is None
                else _safe_ratio(attached_unlabeled, unlabeled_member_count)
            ),
            "distance_to_same_class_labeled_node": (
                None
                if anchor_distances is None
                else numeric_diagnostic_summary(
                    anchor_distances[np.isfinite(anchor_distances)]
                )
            ),
            "unreachable_from_same_class_labeled_node_count": (
                None
                if anchor_distances is None
                else int(np.isinf(anchor_distances).sum())
            ),
        }
        class_rows.append(row)

    # The between-class block of the same matrix, reported only where the two
    # selected classes actually touch.
    pair_rows = []
    for left in range(focus_size):
        for right in range(left + 1, focus_size):
            edge_count = int(pair_counts[left, right])
            if edge_count == 0:
                continue
            pair_rows.append(
                {
                    "class_a": int(focus[left]),
                    "class_b": int(focus[right]),
                    "edge_count": edge_count,
                    "total_weight": float(pair_weights[left, right]),
                    "share_of_class_a_incident_edges": _safe_ratio(
                        edge_count,
                        int(pair_counts[left].sum()),
                    ),
                }
            )
    pair_rows.sort(key=lambda row: (row["edge_count"], row["total_weight"]), reverse=True)

    member_mask = np.isin(labels, focus)
    focus_nodes = np.flatnonzero(member_mask)
    summary = {
        "status": "computed",
        "uses_oracle_labels": True,
        "scope": "full_graph",
        "selection_mode": str(request.class_focus),
        "requested_class_count": int(request.class_count),
        "classes": [int(label) for label in focus],
        "node_count": int(len(focus_nodes)),
        "labeled_node_count": (
            None if known_mask is None else int(known_mask[focus_nodes].sum())
        ),
        "unlabeled_node_count": (
            None if known_mask is None else int((~known_mask[focus_nodes]).sum())
        ),
        "same_class_edge_count": int(
            np.trace(pair_counts[:focus_size, :focus_size])
        ),
        "between_selected_classes_edge_count": int(
            (
                pair_counts[:focus_size, :focus_size].sum()
                - np.trace(pair_counts[:focus_size, :focus_size])
            )
            // 2
        ),
        "edge_count_to_classes_outside_selection": int(
            pair_counts[:focus_size, outside_code].sum()
        ),
        "edge_count_to_unlabeled_class_nodes": int(
            pair_counts[:focus_size, unknown_code].sum()
        ),
        "per_class": class_rows,
        "selected_class_pairs": pair_rows,
    }
    return summary, focus


def _class_focus_edge_rows(adjacency, upper, labels, known_mask, positions, focus):
    """Every edge incident to a selected class, labeled by what it connects."""

    if len(focus) == 0:
        return []
    member_mask = np.isin(labels, focus)
    incident = member_mask[upper.row] | member_mask[upper.col]
    edge_indices = np.flatnonzero(incident)
    truncated = len(edge_indices) > GRAPH_DIAGNOSTICS_MAX_CLASS_FOCUS_EDGE_ROWS
    if truncated:
        edge_indices = edge_indices[:GRAPH_DIAGNOSTICS_MAX_CLASS_FOCUS_EDGE_ROWS]
    rows = []
    for edge_index in edge_indices:
        left = int(upper.row[edge_index])
        right = int(upper.col[edge_index])
        left_label = int(labels[left])
        right_label = int(labels[right])
        left_in = bool(member_mask[left])
        right_in = bool(member_mask[right])
        if left_label == right_label:
            edge_kind = "same_class"
        elif left_in and right_in:
            edge_kind = "between_selected_classes"
        else:
            edge_kind = "to_class_outside_selection"
        rows.append(
            {
                "edge_kind": edge_kind,
                "source_position": int(positions[left]),
                "target_position": int(positions[right]),
                "source_label": left_label,
                "target_label": right_label,
                "source_in_selection": left_in,
                "target_in_selection": right_in,
                "source_kind": graph_node_kind(known_mask, left),
                "target_kind": graph_node_kind(known_mask, right),
                "weight": float(upper.data[edge_index]),
                "source_graph_node": left,
                "target_graph_node": right,
            }
        )
    rows.sort(key=lambda row: (row["edge_kind"], row["source_label"], row["target_label"]))
    return rows


def _restrict_to_incident_edges(adjacency, keep_mask):
    """Drop the edges joining two nodes that are both outside ``keep_mask``."""

    coo = sparse.triu(adjacency, k=1).tocoo()
    incident = keep_mask[coo.row] | keep_mask[coo.col]
    rows = np.concatenate([coo.row[incident], coo.col[incident]])
    cols = np.concatenate([coo.col[incident], coo.row[incident]])
    data = np.concatenate([coo.data[incident], coo.data[incident]])
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=adjacency.shape,
    )


def _graph_edge_styles(edge_rows, edge_cols, labels, focus_mask):
    """Style every drawn edge by what it connects.

    Outside the class-scoped view every edge is drawn alike, because there is no
    selection to be inside or outside of.
    """

    default = ("#8a8f98", 0.45, 0.12, 1)
    if focus_mask is None or labels is None:
        return [default] * len(edge_rows)

    styles = []
    for row, col in zip(edge_rows, edge_cols):
        left = int(row)
        right = int(col)
        if focus_mask[left] and focus_mask[right]:
            if labels[left] == labels[right]:
                styles.append((GRAPH_EDGE_SAME_CLASS_COLOR, 0.9, 0.30, 2))
            else:
                styles.append((GRAPH_EDGE_BETWEEN_CLASSES_COLOR, 0.9, 0.35, 2))
        else:
            styles.append((GRAPH_EDGE_LEAVING_COLOR, 0.5, 0.10, 1))
    return styles


def choose_class_focus_nodes(adjacency, labels, focus, max_nodes, include_context, rng):
    """Pick the nodes the class-scoped plot draws.

    Labeled members come first because they are the anchors the class is
    supposed to be organized around, then unlabeled members, then -- when
    context is on -- the out-of-selection nodes those members attach to most,
    since an induced subgraph would hide the leaked edges entirely.
    """

    member_mask = np.isin(labels, focus)
    focus_nodes = np.flatnonzero(member_mask)
    focus_budget = (
        max_nodes
        if not include_context
        else max(1, int(round(GRAPH_DIAGNOSTICS_CLASS_FOCUS_NODE_SHARE * max_nodes)))
    )
    if len(focus_nodes) > focus_budget:
        per_class = max(1, focus_budget // max(len(focus), 1))
        kept = []
        for label in focus:
            members = np.flatnonzero(labels == int(label))
            if len(members) <= per_class:
                kept.append(members)
                continue
            kept.append(rng.choice(members, size=per_class, replace=False))
        focus_nodes = np.unique(np.concatenate(kept)) if kept else focus_nodes
        if len(focus_nodes) > focus_budget:
            focus_nodes = np.sort(
                rng.choice(focus_nodes, size=focus_budget, replace=False)
            )

    context_nodes = np.array([], dtype=np.int64)
    remaining = int(max_nodes) - len(focus_nodes)
    if include_context and remaining > 0 and len(focus_nodes):
        neighbors = adjacency[focus_nodes].indices
        if len(neighbors):
            outside = neighbors[~member_mask[neighbors]]
            if len(outside):
                candidates, edge_counts = np.unique(outside, return_counts=True)
                # The out-of-selection nodes with the most edges into the
                # selection are the ones a fused class is fusing with.
                order = np.argsort(-edge_counts, kind="stable")
                context_nodes = np.sort(candidates[order[:remaining]])

    node_indices = np.sort(
        np.unique(np.concatenate([focus_nodes, context_nodes])).astype(np.int64)
    )
    focus_mask = member_mask[node_indices]
    return node_indices, focus_mask


def analyze_graph_diagnostics(
    request,
    adjacency,
    positions,
    labels,
    known_mask,
    graph_metadata,
):
    """Compute complete graph statistics while keeping output artifacts bounded."""

    num_nodes = adjacency.shape[0]
    upper = sparse.triu(adjacency, k=1).tocoo()
    degree = np.diff(adjacency.indptr).astype(np.float64)
    weighted_degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    squared_weight_sum = np.asarray(
        adjacency.multiply(adjacency).sum(axis=1),
        dtype=np.float64,
    ).ravel()
    effective_degree = np.divide(
        weighted_degree * weighted_degree,
        squared_weight_sum,
        out=np.zeros_like(weighted_degree),
        where=squared_weight_sum > 0.0,
    )
    strongest_neighbors, strongest_weights, top_share_arrays = (
        _row_weight_concentration(adjacency)
    )

    component_count, component_ids = sparse.csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(component_ids, minlength=component_count)
    isolated = degree == 0

    if known_mask is None:
        component_labeled_counts = np.zeros(component_count, dtype=np.int64)
        component_has_labeled = np.zeros(component_count, dtype=bool)
        labeled_neighbor_count = np.full(num_nodes, np.nan)
        distance_to_labeled = np.full(num_nodes, np.nan)
    else:
        component_labeled_counts = np.bincount(
            component_ids,
            weights=known_mask.astype(np.int64),
            minlength=component_count,
        ).astype(np.int64)
        component_has_labeled = component_labeled_counts > 0
        binary = adjacency.copy()
        binary.data[:] = 1.0
        labeled_neighbor_count = np.asarray(
            binary @ known_mask.astype(np.float64),
            dtype=np.float64,
        ).ravel()
        distance_to_labeled = _multi_source_hop_distances(
            binary,
            known_mask,
        )

    component_label_class_counts = np.zeros(component_count, dtype=np.int64)
    if labels is not None and known_mask is not None:
        for component in range(component_count):
            component_labels = labels[
                (component_ids == component) & known_mask & (labels >= 0)
            ]
            component_label_class_counts[component] = len(
                np.unique(component_labels)
            )

    (
        construction,
        rank_rows,
        mutual_edge_keys,
        candidate_edge_keys,
        positive_out_degree,
        incoming_positive_degree,
    ) = _directed_construction_diagnostics(
        graph_metadata,
        num_nodes,
        labels,
    )
    requested_k = construction.get("requested_n_neighbors")
    search_k = construction.get("search_n_neighbors")

    actual_edge_keys = (
        upper.row.astype(np.int64) * int(max(num_nodes, 1))
        + upper.col.astype(np.int64)
    )
    mutual_edge_mask = (
        None
        if mutual_edge_keys is None
        else np.isin(
            actual_edge_keys,
            mutual_edge_keys,
            assume_unique=False,
        )
    )
    candidate_edge_mask = (
        None
        if candidate_edge_keys is None
        else np.isin(
            actual_edge_keys,
            candidate_edge_keys,
            assume_unique=False,
        )
    )
    if candidate_edge_mask is not None:
        construction["directed_candidate_diagnostics"].update(
            {
                "final_graph_candidate_edge_count": int(
                    candidate_edge_mask.sum()
                ),
                "final_graph_edge_outside_directed_candidates": int(
                    (~candidate_edge_mask).sum()
                ),
            }
        )

    label_quality, class_pair_rows, _ = _edge_label_quality(
        upper,
        labels,
        known_mask,
        mutual_edge_mask=mutual_edge_mask,
        candidate_edge_mask=candidate_edge_mask,
    )
    edge_purity = (
        None
        if label_quality.get("status") != "computed"
        else label_quality.get("edge_purity")
    )

    same_label_neighbor_count = np.full(num_nodes, np.nan, dtype=np.float64)
    weighted_same_label_degree = np.full(num_nodes, np.nan, dtype=np.float64)
    same_label_labeled_neighbor_count = np.full(
        num_nodes,
        np.nan,
        dtype=np.float64,
    )
    strongest_neighbor_same_label = np.full(num_nodes, None, dtype=object)
    strongest_neighbor_labels = np.full(num_nodes, -1, dtype=np.int64)
    if labels is not None:
        coo = adjacency.tocoo()
        valid_directed = (labels[coo.row] >= 0) & (labels[coo.col] >= 0)
        same_directed = valid_directed & (labels[coo.row] == labels[coo.col])
        same_label_neighbor_count = np.bincount(
            coo.row[same_directed],
            minlength=num_nodes,
        ).astype(np.float64)
        weighted_same_label_degree = np.bincount(
            coo.row[same_directed],
            weights=coo.data[same_directed],
            minlength=num_nodes,
        ).astype(np.float64)
        if known_mask is not None:
            same_labeled = same_directed & known_mask[coo.col]
            same_label_labeled_neighbor_count = np.bincount(
                coo.row[same_labeled],
                minlength=num_nodes,
            ).astype(np.float64)
        valid_strongest = strongest_neighbors >= 0
        strongest_neighbor_labels[valid_strongest] = labels[
            strongest_neighbors[valid_strongest]
        ]
        comparable = (
            valid_strongest
            & (labels >= 0)
            & (strongest_neighbor_labels >= 0)
        )
        strongest_neighbor_same_label[comparable] = (
            labels[comparable] == strongest_neighbor_labels[comparable]
        )

    same_label_neighbor_fraction = np.divide(
        same_label_neighbor_count,
        degree,
        out=np.full(num_nodes, np.nan, dtype=np.float64),
        where=degree > 0,
    )
    weighted_same_label_fraction = np.divide(
        weighted_same_label_degree,
        weighted_degree,
        out=np.full(num_nodes, np.nan, dtype=np.float64),
        where=weighted_degree > 0,
    )
    correct_anchor_distances, correct_anchor_status = (
        _correct_label_anchor_distances(
            adjacency,
            labels,
            known_mask,
        )
    )

    edge_weights = upper.data.astype(np.float64, copy=False)
    total_edge_weight = float(edge_weights.sum())
    sorted_weights = np.sort(edge_weights)
    weakest_half_count = int(math.ceil(len(sorted_weights) / 2))
    weight_concentration = {
        "weakest_50_percent_weight_share": _safe_ratio(
            float(sorted_weights[:weakest_half_count].sum()),
            total_edge_weight,
        ),
    }
    for fraction in (0.01, 0.05, 0.10):
        top_count = int(math.ceil(len(sorted_weights) * fraction))
        weight_concentration[f"strongest_{int(fraction * 100)}_percent_weight_share"] = (
            _safe_ratio(
                float(sorted_weights[-top_count:].sum()),
                total_edge_weight,
            )
            if top_count > 0
            else None
        )

    components_without_labeled = (
        int((~component_has_labeled).sum())
        if known_mask is not None
        else None
    )
    nodes_without_labeled_component = (
        int(component_sizes[~component_has_labeled].sum())
        if known_mask is not None
        else None
    )
    connectivity = {
        "component_count": int(component_count),
        "component_size": numeric_diagnostic_summary(component_sizes),
        "largest_component_size": (
            0 if len(component_sizes) == 0 else int(component_sizes.max())
        ),
        "largest_component_fraction": (
            0.0
            if num_nodes == 0 or len(component_sizes) == 0
            else float(component_sizes.max()) / float(num_nodes)
        ),
        "largest_component_sizes": sorted(
            (int(size) for size in component_sizes),
            reverse=True,
        )[:20],
        "isolated_node_count": int(isolated.sum()),
        "components_without_labeled_node": components_without_labeled,
        "nodes_in_components_without_labeled_node": nodes_without_labeled_component,
        "unlabeled_nodes_with_direct_labeled_neighbor": (
            None
            if known_mask is None
            else int(((~known_mask) & (labeled_neighbor_count > 0)).sum())
        ),
        "unlabeled_fraction_with_direct_labeled_neighbor": (
            None
            if known_mask is None
            else _safe_ratio(
                int(((~known_mask) & (labeled_neighbor_count > 0)).sum()),
                int((~known_mask).sum()),
            )
        ),
        "distance_to_any_labeled_node": numeric_diagnostic_summary(
            distance_to_labeled
        ),
        "distance_to_correct_class_labeled_node": {
            **correct_anchor_status,
            "distribution": numeric_diagnostic_summary(
                correct_anchor_distances
            ),
            "unreachable_count": int(
                np.isinf(correct_anchor_distances).sum()
            ),
        },
    }

    class_focus, focus_classes = _class_focus_diagnostics(
        request=request,
        adjacency=adjacency,
        upper=upper,
        labels=labels,
        known_mask=known_mask,
        component_ids=component_ids,
        correct_anchor_distances=(
            None
            if correct_anchor_status.get("status") != "computed"
            else correct_anchor_distances
        ),
        per_class_rows=label_quality.get("per_class_edge_quality", []),
        rng=np.random.default_rng(request.seed),
    )
    class_focus_rows = (
        []
        if labels is None
        else _class_focus_edge_rows(
            adjacency=adjacency,
            upper=upper,
            labels=labels,
            known_mask=known_mask,
            positions=positions,
            focus=focus_classes,
        )
    )

    spectral = _spectral_graph_summary(adjacency, component_count)
    temporal = _temporal_graph_diagnostics(
        request=request,
        positions=positions,
        upper=upper,
        degree=degree,
        weighted_degree=weighted_degree,
        component_count=component_count,
        edge_purity=edge_purity,
    )

    graph_summary = {
        "node_count": int(num_nodes),
        "directed_nonzero_count": int(adjacency.nnz),
        "undirected_edge_count": int(upper.nnz),
        "edge_count_definition": (
            "nonzero undirected edges in the final symmetrized graph"
        ),
        "density": (
            0.0
            if num_nodes < 2
            else float(2 * upper.nnz) / float(num_nodes * (num_nodes - 1))
        ),
        "total_edge_weight": total_edge_weight,
        "adjacency_memory_bytes": _csr_memory_bytes(adjacency),
    }
    requested_k_scope = np.isfinite(positive_out_degree)
    if not np.any(requested_k_scope):
        requested_k_scope = np.ones(num_nodes, dtype=bool)
    degree_summary = {
        "unweighted_definition": (
            "nonzero neighbors per node in the final symmetrized graph"
        ),
        "weighted_definition": "sum of final incident edge weights per node",
        "effective_definition": (
            "(sum incident weights)^2 / sum squared incident weights"
        ),
        "unweighted": numeric_diagnostic_summary(degree),
        "weighted": numeric_diagnostic_summary(weighted_degree),
        "effective": numeric_diagnostic_summary(effective_degree),
        "zero_degree_node_count": int((degree == 0).sum()),
        "requested_k_scope_node_count": int(requested_k_scope.sum()),
        "nodes_below_requested_k": (
            None
            if requested_k is None
            else int(
                (
                    degree[requested_k_scope]
                    < int(min(requested_k, max(num_nodes - 1, 0)))
                ).sum()
            )
        ),
        "nodes_below_search_k": (
            None
            if search_k is None
            else int(
                (degree[requested_k_scope] < int(search_k)).sum()
            )
        ),
        "nodes_equal_search_k": (
            None
            if search_k is None
            else int(
                (degree[requested_k_scope] == int(search_k)).sum()
            )
        ),
        "nodes_above_search_k": (
            None
            if search_k is None
            else int(
                (degree[requested_k_scope] > int(search_k)).sum()
            )
        ),
        "per_node_top_weight_share": {
            f"top_{top_k}": numeric_diagnostic_summary(values)
            for top_k, values in top_share_arrays.items()
        },
    }
    weight_summary = {
        "edge_weight": numeric_diagnostic_summary(edge_weights),
        "total_edge_weight": total_edge_weight,
        "concentration": weight_concentration,
    }

    node_data = {
        "graph_node": np.arange(num_nodes, dtype=np.int64),
        "position": positions,
        "label": (
            np.full(num_nodes, -1, dtype=np.int64)
            if labels is None
            else labels
        ),
        "kind": np.asarray(
            [graph_node_kind(known_mask, node) for node in range(num_nodes)],
            dtype=object,
        ),
        "degree": degree.astype(np.int64),
        "weighted_degree": weighted_degree,
        "effective_degree": effective_degree,
        "positive_out_degree": positive_out_degree,
        "incoming_positive_degree": incoming_positive_degree,
        "component_id": component_ids,
        "component_size": component_sizes[component_ids],
        "component_has_labeled_node": (
            np.full(num_nodes, None, dtype=object)
            if known_mask is None
            else component_has_labeled[component_ids]
        ),
        "component_labeled_node_count": component_labeled_counts[component_ids],
        "component_labeled_class_count": component_label_class_counts[component_ids],
        "labeled_neighbor_count": labeled_neighbor_count,
        "same_label_labeled_neighbor_count": same_label_labeled_neighbor_count,
        "distance_to_labeled_node": distance_to_labeled,
        "distance_to_correct_label_labeled_node": correct_anchor_distances,
        "same_label_neighbor_fraction": same_label_neighbor_fraction,
        "weighted_same_label_fraction": weighted_same_label_fraction,
        "strongest_neighbor_graph_node": strongest_neighbors,
        "strongest_neighbor_position": np.where(
            strongest_neighbors >= 0,
            positions[np.maximum(strongest_neighbors, 0)],
            -1,
        ),
        "strongest_neighbor_label": strongest_neighbor_labels,
        "strongest_neighbor_weight": strongest_weights,
        "strongest_neighbor_same_label": strongest_neighbor_same_label,
        **{
            f"top_{top_k}_weight_share": values
            for top_k, values in top_share_arrays.items()
        },
    }

    summary = {
        "artifact_version": GRAPH_DIAGNOSTICS_ARTIFACT_VERSION,
        "slug": request.slug,
        "series_slug": _graph_diagnostic_series_slug(request),
        "title": request.title,
        "epoch": request.epoch,
        "overview": {
            "node_count": int(num_nodes),
            "undirected_edge_count": int(upper.nnz),
            "requested_n_neighbors": requested_k,
            "search_n_neighbors": search_k,
            "mean_degree": degree_summary["unweighted"].get("mean"),
            "median_degree": degree_summary["unweighted"].get("p50"),
            "min_degree": degree_summary["unweighted"].get("min"),
            "max_degree": degree_summary["unweighted"].get("max"),
            "mean_weighted_degree": degree_summary["weighted"].get("mean"),
            "mean_effective_degree": degree_summary["effective"].get("mean"),
            "mean_positive_out_degree": construction[
                "directed_candidate_diagnostics"
            ].get("positive_out_degree", {}).get("mean"),
            "nodes_below_requested_k": degree_summary[
                "nodes_below_requested_k"
            ],
            "isolated_node_count": int(isolated.sum()),
            "component_count": int(component_count),
            "nodes_in_components_without_labeled_node": (
                nodes_without_labeled_component
            ),
            "oracle_edge_purity": label_quality.get("edge_purity"),
            "oracle_weighted_edge_purity": label_quality.get(
                "weighted_edge_purity"
            ),
        },
        "graph": graph_summary,
        "construction": construction,
        "degree": degree_summary,
        "weights": weight_summary,
        "connectivity": connectivity,
        "label_quality": label_quality,
        "class_focus": class_focus,
        "numerical": {
            "adjacency_shape": list(adjacency.shape),
            "adjacency_nnz": int(adjacency.nnz),
            "adjacency_memory_bytes": _csr_memory_bytes(adjacency),
            "spectral": spectral,
        },
        "temporal": temporal,
        "propagation": {
            "status": "not_applicable_or_not_yet_recorded",
        },
    }
    return {
        "summary": summary,
        "node_data": node_data,
        "rank_rows": rank_rows,
        "class_pair_rows": class_pair_rows,
        "focus_classes": focus_classes,
        "class_focus_rows": class_focus_rows,
        "plot_data": {
            "degree": degree,
            "weighted_degree": weighted_degree,
            "effective_degree": effective_degree,
            "edge_weight": edge_weights,
        },
    }


def _format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return "" if not math.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_graph_summary_json(path, summary):
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            _json_ready(summary),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")


def write_graph_node_csv(csv_path, node_data):
    fieldnames = list(node_data)
    row_count = 0 if not fieldnames else len(node_data[fieldnames[0]])
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(row_count):
            writer.writerow(
                {
                    name: _format_csv_value(values[row_index])
                    for name, values in node_data.items()
                }
            )


def write_dict_rows_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: _format_csv_value(row.get(name))
                    for name in fieldnames
                }
            )


def write_graph_distribution_png(path, plot_data, title):
    import matplotlib.pyplot as plt

    panels = (
        ("degree", "Nonzero neighbors per node", "degree"),
        ("weighted_degree", "Weighted degree per node", "weighted degree"),
        ("effective_degree", "Effective degree per node", "effective degree"),
        ("edge_weight", "Nonzero edge weights", "edge weight"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, panel_title, xlabel) in zip(axes.ravel(), panels):
        values = np.asarray(plot_data[key], dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) > 0:
            value_span = float(np.ptp(values))
            value_scale = max(1.0, float(np.max(np.abs(values))))
            bins = (
                np.asarray(
                    [
                        float(np.mean(values)) - 0.05 * value_scale,
                        float(np.mean(values)) + 0.05 * value_scale,
                    ]
                )
                if value_span <= np.finfo(np.float64).eps * value_scale * 16
                else min(60, max(10, int(np.sqrt(len(values)))))
            )
            ax.hist(values, bins=bins, color="#4e79a7", alpha=0.82)
            ax.axvline(
                float(np.mean(values)),
                color="#e15759",
                linewidth=1.2,
                label=f"mean={np.mean(values):.3g}",
            )
            ax.axvline(
                float(np.median(values)),
                color="#59a14f",
                linewidth=1.2,
                label=f"median={np.median(values):.3g}",
            )
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(alpha=0.18)
    fig.suptitle(f"{title} - full graph distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def maybe_update_graph_propagation_diagnostics(
    request,
    scores,
    confidences,
    labels=None,
    known_mask=None,
    method=None,
    confidence_threshold=None,
    initial_scores=None,
    dissimilarity=None,
    solver_diagnostics=None,
    extra=None,
    score_class_labels=None,
):
    """Append propagation results to graph artifacts without affecting training.

    ``score_class_labels`` maps compact score columns back to the dataset-wide
    class IDs displayed in artifacts.
    """

    if request is None:
        return None
    try:
        return update_graph_propagation_diagnostics(
            request=request,
            scores=scores,
            confidences=confidences,
            score_class_labels=score_class_labels,
            labels=labels,
            known_mask=known_mask,
            method=method,
            confidence_threshold=confidence_threshold,
            initial_scores=initial_scores,
            dissimilarity=dissimilarity,
            solver_diagnostics=solver_diagnostics,
            extra=extra,
        )
    except Exception as exc:  # pragma: no cover - diagnostics must not stop training
        logger.warning(
            f"Could not update graph propagation diagnostics {request.slug}: {exc}"
        )
        return None


def _normalized_diagnostic_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("propagation diagnostic scores must be a matrix")
    finite_rows = np.all(np.isfinite(scores), axis=1)
    nonnegative = np.maximum(scores, 0.0)
    row_mass = nonnegative.sum(axis=1)
    propagated = finite_rows & (row_mass > 0.0)
    probabilities = np.zeros_like(nonnegative)
    probabilities[propagated] = (
        nonnegative[propagated] / row_mass[propagated, None]
    )
    return probabilities, propagated, row_mass


def _prediction_entropy(probabilities, propagated):
    entropies = np.full(len(probabilities), np.nan, dtype=np.float64)
    if probabilities.shape[1] <= 1:
        entropies[propagated] = 0.0
        return entropies
    active = probabilities[propagated]
    entropy_terms = np.zeros_like(active)
    positive = active > 0.0
    entropy_terms[positive] = active[positive] * np.log(active[positive])
    active_entropy = -np.sum(entropy_terms, axis=1)
    entropies[propagated] = active_entropy / math.log(probabilities.shape[1])
    return entropies


def _threshold_diagnostics(
    confidences,
    predicted_labels,
    propagated,
    evaluation_mask,
    labels,
    configured_threshold,
):
    thresholds = {0.0, 0.25, 0.5, 0.75, 0.9, 0.95}
    if configured_threshold is not None:
        thresholds.add(float(configured_threshold))
    rows = []
    for threshold in sorted(thresholds):
        selected = propagated & (confidences >= threshold)
        selected_evaluation = selected & evaluation_mask
        rows.append(
            {
                "threshold": float(threshold),
                "selected_count": int(selected.sum()),
                "selected_fraction": _safe_ratio(
                    int(selected.sum()),
                    int(propagated.sum()),
                ),
                "oracle_evaluation_count": int(selected_evaluation.sum()),
                "oracle_accuracy": (
                    None
                    if labels is None or not np.any(selected_evaluation)
                    else float(
                        np.mean(
                            predicted_labels[selected_evaluation]
                            == labels[selected_evaluation]
                        )
                    )
                ),
            }
        )
    return rows


def _confidence_calibration_rows(
    confidences,
    predicted_labels,
    evaluation_mask,
    labels,
):
    if labels is None:
        return {"status": "unavailable", "reason": "labels_missing", "bins": []}
    valid_confidence = (
        np.isfinite(confidences)
        & (confidences >= 0.0)
        & (confidences <= 1.0)
    )
    calibration_mask = evaluation_mask & valid_confidence
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = calibration_mask & (confidences >= low)
        if bin_index == len(edges) - 2:
            in_bin &= confidences <= high
        else:
            in_bin &= confidences < high
        count = int(in_bin.sum())
        rows.append(
            {
                "lower": float(low),
                "upper": float(high),
                "count": count,
                "mean_confidence": (
                    None if count == 0 else float(confidences[in_bin].mean())
                ),
                "oracle_accuracy": (
                    None
                    if count == 0
                    else float(
                        np.mean(
                            predicted_labels[in_bin] == labels[in_bin]
                        )
                    )
                ),
            }
        )
    calibration_count = int(calibration_mask.sum())
    weighted_gaps = [
        row["count"]
        * abs(row["mean_confidence"] - row["oracle_accuracy"])
        for row in rows
        if row["count"] > 0
    ]
    bin_gaps = [
        abs(row["mean_confidence"] - row["oracle_accuracy"])
        for row in rows
        if row["count"] > 0
    ]
    correct = (
        predicted_labels[calibration_mask] == labels[calibration_mask]
    ).astype(np.float64)
    return {
        "status": "computed" if calibration_count > 0 else "unavailable",
        "evaluation_count": calibration_count,
        "excluded_invalid_or_out_of_range_confidence_count": int(
            (evaluation_mask & (~valid_confidence)).sum()
        ),
        "expected_calibration_error": (
            None
            if calibration_count == 0
            else float(sum(weighted_gaps) / calibration_count)
        ),
        "maximum_calibration_error": (
            None if not bin_gaps else float(max(bin_gaps))
        ),
        "confidence_brier_score": (
            None
            if calibration_count == 0
            else float(
                np.mean(
                    (
                        confidences[calibration_mask]
                        - correct
                    )
                    ** 2
                )
            )
        ),
        "bins": rows,
    }


def _per_class_prediction_rows(
    probabilities,
    confidences,
    predicted_labels,
    propagated,
    evaluation_mask,
    labels,
    score_class_labels,
):
    class_count = probabilities.shape[1]
    probability_mass = probabilities.sum(axis=0)
    if class_count > GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS:
        selected_columns = np.argsort(
            -probability_mass,
            kind="stable",
        )[:GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS]
        selected_columns = np.sort(selected_columns)
    else:
        selected_columns = np.arange(class_count)
    rows = []
    for column, label in zip(
        selected_columns,
        score_class_labels[selected_columns],
    ):
        column = int(column)
        label = int(label)
        predicted = propagated & (predicted_labels == label)
        row = {
            "label": label,
            "predicted_count": int(predicted.sum()),
            "probability_mass": float(probability_mass[column]),
            "mean_prediction_confidence": (
                None if not np.any(predicted) else float(confidences[predicted].mean())
            ),
        }
        if labels is not None:
            predicted_eval = predicted & evaluation_mask
            true_eval = evaluation_mask & (labels == label)
            true_positive = int((predicted_eval & (labels == label)).sum())
            row.update(
                {
                    "oracle_true_count": int(true_eval.sum()),
                    "oracle_true_positive_count": true_positive,
                    "oracle_precision": _safe_ratio(
                        true_positive,
                        int(predicted_eval.sum()),
                    ),
                    "oracle_recall": _safe_ratio(
                        true_positive,
                        int(true_eval.sum()),
                    ),
                }
            )
        rows.append(row)
    return rows, class_count > GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS


def _predicted_label_direct_anchor_support(
    request,
    predicted_labels,
    propagated,
    labels,
    known_mask,
):
    if labels is None or known_mask is None:
        return None
    series_key = (
        str(Path(request.output_dir).resolve()),
        _graph_diagnostic_series_slug(request),
    )
    entry = _GRAPH_DIAGNOSTIC_HISTORY.get(series_key)
    if entry is None:
        return None
    state = entry["current"]
    num_nodes = len(predicted_labels)
    support = np.zeros(num_nodes, dtype=bool)
    edge_keys = state["edge_keys"]
    rows = edge_keys // max(num_nodes, 1)
    cols = edge_keys % max(num_nodes, 1)
    row_supported = (
        propagated[rows]
        & known_mask[cols]
        & (predicted_labels[rows] == labels[cols])
    )
    col_supported = (
        propagated[cols]
        & known_mask[rows]
        & (predicted_labels[cols] == labels[rows])
    )
    support[rows[row_supported]] = True
    support[cols[col_supported]] = True
    return support


def _dissimilarity_diagnostics(dissimilarity):
    if dissimilarity is None:
        return {"status": "unavailable"}
    dissimilarity = dissimilarity.tocsr()
    upper = sparse.triu(dissimilarity, k=1).tocoo()
    degree = np.diff(dissimilarity.indptr)
    weighted_degree = np.asarray(
        dissimilarity.sum(axis=1),
        dtype=np.float64,
    ).ravel()
    return {
        "status": "computed",
        "undirected_edge_count": int(upper.nnz),
        "total_weight": float(upper.data.sum()),
        "edge_weight": numeric_diagnostic_summary(upper.data),
        "degree": numeric_diagnostic_summary(degree),
        "weighted_degree": numeric_diagnostic_summary(weighted_degree),
        "zero_degree_node_count": int((degree == 0).sum()),
        "matrix_nnz": int(dissimilarity.nnz),
        "matrix_memory_bytes": _csr_memory_bytes(dissimilarity),
    }


def _update_temporal_predictions(
    request,
    predicted_labels,
    confidences,
    propagated,
):
    series_key = (
        str(Path(request.output_dir).resolve()),
        _graph_diagnostic_series_slug(request),
    )
    entry = _GRAPH_DIAGNOSTIC_HISTORY.get(series_key)
    if entry is None:
        return {
            "status": "unavailable",
            "reason": "graph_history_missing",
        }
    current = entry["current"]
    previous = entry["previous"]
    temporal = {
        "status": "unavailable",
        "reason": "previous_propagation_missing",
    }
    if (
        previous is not None
        and previous.get("predicted_labels") is not None
        and len(previous["predicted_labels"]) == len(predicted_labels)
        and np.array_equal(previous["positions"], current["positions"])
    ):
        comparable = propagated & previous["propagated_mask"]
        changed = comparable & (
            predicted_labels != previous["predicted_labels"]
        )
        temporal = {
            "status": "computed",
            "previous_slug": previous["slug"],
            "comparable_propagated_node_count": int(comparable.sum()),
            "prediction_flip_count": int(changed.sum()),
            "prediction_flip_fraction": _safe_ratio(
                int(changed.sum()),
                int(comparable.sum()),
            ),
            "confidence_delta": numeric_diagnostic_summary(
                confidences[comparable] - previous["confidences"][comparable]
            ),
        }
    current["predicted_labels"] = predicted_labels.copy()
    current["confidences"] = confidences.copy()
    current["propagated_mask"] = propagated.copy()
    return temporal


def _append_propagation_columns_to_node_csv(
    csv_path,
    predicted_labels,
    confidences,
    entropies,
    prediction_margins,
    propagated,
    accepted_at_threshold,
    direct_anchor_support,
    labels,
    known_mask,
    initial_predictions,
    previous_predictions,
    previous_confidences,
    previous_propagated,
):
    with csv_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    extra_fieldnames = [
        "propagated",
        "predicted_label",
        "prediction_confidence",
        "prediction_entropy",
        "prediction_margin",
        "prediction_correct",
        "accepted_at_configured_threshold",
        "direct_labeled_neighbor_support",
        "initial_predicted_label",
        "changed_from_initial_prediction",
        "previous_predicted_label",
        "changed_from_previous_prediction",
        "confidence_delta_from_previous",
    ]
    temp_path = csv_path.with_suffix(".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=original_fieldnames + extra_fieldnames,
        )
        writer.writeheader()
        for row_index, row in enumerate(rows):
            is_propagated = bool(propagated[row_index])
            comparable_label = (
                labels is not None
                and labels[row_index] >= 0
                and (known_mask is None or not known_mask[row_index])
                and is_propagated
            )
            row.update(
                {
                    "propagated": is_propagated,
                    "predicted_label": (
                        int(predicted_labels[row_index])
                        if is_propagated
                        else ""
                    ),
                    "prediction_confidence": (
                        _format_csv_value(confidences[row_index])
                        if is_propagated
                        else ""
                    ),
                    "prediction_entropy": (
                        _format_csv_value(entropies[row_index])
                        if is_propagated
                        else ""
                    ),
                    "prediction_margin": (
                        _format_csv_value(prediction_margins[row_index])
                        if is_propagated
                        else ""
                    ),
                    "prediction_correct": (
                        bool(predicted_labels[row_index] == labels[row_index])
                        if comparable_label
                        else ""
                    ),
                    "accepted_at_configured_threshold": (
                        ""
                        if accepted_at_threshold is None
                        else bool(accepted_at_threshold[row_index])
                    ),
                    "direct_labeled_neighbor_support": (
                        ""
                        if direct_anchor_support is None or not is_propagated
                        else bool(direct_anchor_support[row_index])
                    ),
                    "initial_predicted_label": (
                        ""
                        if initial_predictions is None or not is_propagated
                        else int(initial_predictions[row_index])
                    ),
                    "changed_from_initial_prediction": (
                        ""
                        if initial_predictions is None or not is_propagated
                        else bool(
                            predicted_labels[row_index]
                            != initial_predictions[row_index]
                        )
                    ),
                    "previous_predicted_label": (
                        ""
                        if (
                            previous_predictions is None
                            or previous_propagated is None
                            or not previous_propagated[row_index]
                        )
                        else int(previous_predictions[row_index])
                    ),
                    "changed_from_previous_prediction": (
                        ""
                        if (
                            previous_predictions is None
                            or previous_propagated is None
                            or not is_propagated
                            or not previous_propagated[row_index]
                        )
                        else bool(
                            predicted_labels[row_index]
                            != previous_predictions[row_index]
                        )
                    ),
                    "confidence_delta_from_previous": (
                        ""
                        if (
                            previous_confidences is None
                            or previous_propagated is None
                            or not is_propagated
                            or not previous_propagated[row_index]
                        )
                        else _format_csv_value(
                            confidences[row_index]
                            - previous_confidences[row_index]
                        )
                    ),
                }
            )
            writer.writerow(row)
    temp_path.replace(csv_path)


def update_graph_propagation_diagnostics(
    request,
    scores,
    confidences,
    labels=None,
    known_mask=None,
    method=None,
    confidence_threshold=None,
    initial_scores=None,
    dissimilarity=None,
    solver_diagnostics=None,
    extra=None,
    score_class_labels=None,
):
    paths = graph_diagnostic_artifact_paths(request)
    if not paths["summary_json"].exists() or not paths["nodes_csv"].exists():
        raise FileNotFoundError(
            "base graph diagnostics must be written before propagation diagnostics"
        )

    with paths["summary_json"].open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    num_nodes = int(summary["graph"]["node_count"])
    probabilities, propagated, row_mass = _normalized_diagnostic_probabilities(
        scores
    )
    if len(probabilities) != num_nodes:
        raise ValueError(
            "propagation scores are not aligned with graph diagnostic nodes"
        )
    if score_class_labels is None:
        score_class_labels = np.arange(
            probabilities.shape[1],
            dtype=np.int64,
        )
    else:
        score_class_labels = np.asarray(score_class_labels, dtype=np.int64)
        if (
            score_class_labels.ndim != 1
            or len(score_class_labels) != probabilities.shape[1]
        ):
            raise ValueError(
                "score_class_labels must identify every propagation score column"
            )
        if np.any(score_class_labels < 0) or len(np.unique(score_class_labels)) != len(
            score_class_labels
        ):
            raise ValueError("score_class_labels must be unique non-negative labels")

    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if len(confidences) != num_nodes:
        raise ValueError(
            "propagation confidences are not aligned with graph diagnostic nodes"
        )
    labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    if labels is not None and len(labels) != num_nodes:
        labels = None
    known_mask = (
        None if known_mask is None else np.asarray(known_mask, dtype=bool)
    )
    if known_mask is not None and len(known_mask) != num_nodes:
        known_mask = None

    predicted_columns = np.argmax(probabilities, axis=1).astype(np.int64)
    predicted_labels = score_class_labels[predicted_columns]
    predicted_labels[~propagated] = -1
    entropies = _prediction_entropy(probabilities, propagated)
    if probabilities.shape[1] <= 1:
        prediction_margins = probabilities[:, 0].copy()
    else:
        top_two = np.partition(
            probabilities,
            kth=probabilities.shape[1] - 2,
            axis=1,
        )[:, -2:]
        prediction_margins = top_two[:, 1] - top_two[:, 0]
    prediction_margins[~propagated] = np.nan
    unlabeled = (
        np.ones(num_nodes, dtype=bool)
        if known_mask is None
        else ~known_mask
    )
    evaluation_mask = (
        unlabeled & propagated
        if labels is None
        else unlabeled & propagated & (labels >= 0)
    )
    correct = (
        np.zeros(num_nodes, dtype=bool)
        if labels is None
        else evaluation_mask & (predicted_labels == labels)
    )

    initial_predictions = None
    initial_change = None
    if initial_scores is not None:
        initial_probabilities, initial_propagated, _ = (
            _normalized_diagnostic_probabilities(initial_scores)
        )
        if (
            len(initial_probabilities) == num_nodes
            and initial_probabilities.shape[1] == len(score_class_labels)
        ):
            initial_prediction_columns = np.argmax(
                initial_probabilities,
                axis=1,
            ).astype(np.int64)
            initial_predictions = score_class_labels[
                initial_prediction_columns
            ]
            initial_predictions[~initial_propagated] = -1
            comparable_initial = propagated & initial_propagated
            initial_changed = comparable_initial & (
                predicted_labels != initial_predictions
            )
            initial_change = {
                "comparable_node_count": int(comparable_initial.sum()),
                "changed_prediction_count": int(initial_changed.sum()),
                "changed_prediction_fraction": _safe_ratio(
                    int(initial_changed.sum()),
                    int(comparable_initial.sum()),
                ),
            }

    series_key = (
        str(Path(request.output_dir).resolve()),
        _graph_diagnostic_series_slug(request),
    )
    history_entry = _GRAPH_DIAGNOSTIC_HISTORY.get(series_key)
    previous_predictions = None
    previous_confidences = None
    previous_propagated = None
    if history_entry is not None and history_entry["previous"] is not None:
        previous_predictions = history_entry["previous"].get(
            "predicted_labels"
        )
        previous_confidences = history_entry["previous"].get(
            "confidences"
        )
        previous_propagated = history_entry["previous"].get(
            "propagated_mask"
        )

    direct_anchor_support = _predicted_label_direct_anchor_support(
        request=request,
        predicted_labels=predicted_labels,
        propagated=propagated,
        labels=labels,
        known_mask=known_mask,
    )
    temporal_predictions = _update_temporal_predictions(
        request=request,
        predicted_labels=predicted_labels,
        confidences=confidences,
        propagated=propagated,
    )
    per_class_rows, per_class_truncated = _per_class_prediction_rows(
        probabilities=probabilities,
        confidences=confidences,
        predicted_labels=predicted_labels,
        propagated=propagated,
        evaluation_mask=evaluation_mask,
        labels=labels,
        score_class_labels=score_class_labels,
    )

    nonzero_prediction_labels, prediction_counts = np.unique(
        predicted_labels[propagated],
        return_counts=True,
    )
    prediction_count_order = nonzero_prediction_labels[
        np.argsort(
            -prediction_counts,
            kind="stable",
        )
    ]
    reported_prediction_labels = prediction_count_order[
        :GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS
    ]
    prediction_count_by_label = {
        int(label): int(count)
        for label, count in zip(nonzero_prediction_labels, prediction_counts)
    }
    accepted_at_threshold = (
        None
        if confidence_threshold is None
        else (
            propagated
            & unlabeled
            & (confidences >= float(confidence_threshold))
        )
    )
    accepted_evaluation = (
        np.zeros(num_nodes, dtype=bool)
        if accepted_at_threshold is None
        else accepted_at_threshold & evaluation_mask
    )
    propagation = {
        "status": "computed",
        "method": method,
        "class_count": int(probabilities.shape[1]),
        "class_labels": score_class_labels.tolist(),
        "node_count": int(num_nodes),
        "propagated_node_count": int(propagated.sum()),
        "unpropagated_node_count": int((~propagated).sum()),
        "unlabeled_propagated_count": int((unlabeled & propagated).sum()),
        "unlabeled_coverage": _safe_ratio(
            int((unlabeled & propagated).sum()),
            int(unlabeled.sum()),
        ),
        "zero_or_invalid_mass_row_count": int((~propagated).sum()),
        "row_mass": numeric_diagnostic_summary(row_mass),
        "confidence": numeric_diagnostic_summary(confidences[propagated]),
        "nonfinite_confidence_count": int(
            np.sum(propagated & (~np.isfinite(confidences)))
        ),
        "confidence_below_zero_count": int(
            np.sum(propagated & (confidences < 0.0))
        ),
        "confidence_above_one_count": int(
            np.sum(propagated & (confidences > 1.0))
        ),
        "prediction_entropy": numeric_diagnostic_summary(
            entropies[propagated]
        ),
        "prediction_margin": numeric_diagnostic_summary(
            prediction_margins[propagated]
        ),
        "configured_confidence_threshold": (
            None
            if confidence_threshold is None
            else float(confidence_threshold)
        ),
        "accepted_at_configured_threshold_count": (
            None
            if accepted_at_threshold is None
            else int(accepted_at_threshold.sum())
        ),
        "accepted_at_configured_threshold_fraction": (
            None
            if accepted_at_threshold is None
            else _safe_ratio(
                int(accepted_at_threshold.sum()),
                int((propagated & unlabeled).sum()),
            )
        ),
        "accepted_at_configured_threshold_oracle_accuracy": (
            None
            if (
                labels is None
                or accepted_at_threshold is None
                or not np.any(accepted_evaluation)
            )
            else float(
                np.mean(
                    predicted_labels[accepted_evaluation]
                    == labels[accepted_evaluation]
                )
            )
        ),
        "oracle_evaluation_count": int(evaluation_mask.sum()),
        "oracle_accuracy": (
            None
            if labels is None or not np.any(evaluation_mask)
            else float(correct[evaluation_mask].mean())
        ),
        "predicted_class_count": int(len(nonzero_prediction_labels)),
        "predicted_class_counts": {
            str(label): prediction_count_by_label[int(label)]
            for label in reported_prediction_labels
        },
        "predicted_class_counts_truncated": (
            len(nonzero_prediction_labels)
            > GRAPH_DIAGNOSTICS_MAX_CLASS_ROWS
        ),
        "thresholds": _threshold_diagnostics(
            confidences=confidences,
            predicted_labels=predicted_labels,
            propagated=propagated & unlabeled,
            evaluation_mask=evaluation_mask,
            labels=labels,
            configured_threshold=confidence_threshold,
        ),
        "confidence_calibration": _confidence_calibration_rows(
            confidences=confidences,
            predicted_labels=predicted_labels,
            evaluation_mask=evaluation_mask,
            labels=labels,
        ),
        "per_class": per_class_rows,
        "per_class_truncated": per_class_truncated,
        "direct_labeled_neighbor_support": (
            {"status": "unavailable"}
            if direct_anchor_support is None
            else {
                "status": "computed",
                "supported_prediction_count": int(
                    (direct_anchor_support & unlabeled & propagated).sum()
                ),
                "supported_prediction_fraction": _safe_ratio(
                    int(
                        (
                            direct_anchor_support
                            & unlabeled
                            & propagated
                        ).sum()
                    ),
                    int((unlabeled & propagated).sum()),
                ),
            }
        ),
        "initial_to_final_change": initial_change,
        "dissimilarity_graph": _dissimilarity_diagnostics(dissimilarity),
        "solver": solver_diagnostics,
        "temporal": temporal_predictions,
        "extra": {} if extra is None else extra,
    }
    summary["propagation"] = propagation
    summary.setdefault("overview", {}).update(
        {
            "propagation_method": method,
            "unlabeled_propagation_coverage": propagation[
                "unlabeled_coverage"
            ],
            "oracle_pseudo_label_accuracy": propagation[
                "oracle_accuracy"
            ],
            "accepted_at_configured_threshold_count": propagation[
                "accepted_at_configured_threshold_count"
            ],
            "accepted_at_configured_threshold_fraction": propagation[
                "accepted_at_configured_threshold_fraction"
            ],
        }
    )
    if isinstance(summary.get("temporal"), dict):
        summary["temporal"]["propagation"] = temporal_predictions
    write_graph_summary_json(paths["summary_json"], summary)

    _append_propagation_columns_to_node_csv(
        csv_path=paths["nodes_csv"],
        predicted_labels=predicted_labels,
        confidences=confidences,
        entropies=entropies,
        prediction_margins=prediction_margins,
        propagated=propagated,
        accepted_at_threshold=accepted_at_threshold,
        direct_anchor_support=direct_anchor_support,
        labels=labels,
        known_mask=known_mask,
        initial_predictions=initial_predictions,
        previous_predictions=previous_predictions,
        previous_confidences=previous_confidences,
        previous_propagated=previous_propagated,
    )
    logger.info(
        "Updated graph propagation diagnostics: "
        f"summary={paths['summary_json']}, nodes={paths['nodes_csv']}, "
        f"coverage={propagation['unlabeled_coverage']}, "
        f"oracle_accuracy={propagation['oracle_accuracy']}"
    )
    return paths["summary_json"]


def choose_graph_diagnostic_nodes(adjacency, max_nodes, rng):
    num_nodes = adjacency.shape[0]
    if num_nodes <= max_nodes:
        return np.arange(num_nodes, dtype=np.int64)

    upper = sparse.triu(adjacency, k=1).tocoo()
    if upper.nnz == 0:
        return np.sort(rng.choice(num_nodes, size=max_nodes, replace=False)).astype(np.int64)

    sampled_edge_count = min(max_nodes, upper.nnz)
    edge_indices = rng.choice(upper.nnz, size=sampled_edge_count, replace=False)
    endpoints = np.unique(np.concatenate([upper.row[edge_indices], upper.col[edge_indices]]))
    if len(endpoints) > max_nodes:
        endpoints = rng.choice(endpoints, size=max_nodes, replace=False)
    elif len(endpoints) < max_nodes:
        remaining = np.setdiff1d(np.arange(num_nodes, dtype=np.int64), endpoints, assume_unique=False)
        fill = rng.choice(remaining, size=max_nodes - len(endpoints), replace=False)
        endpoints = np.concatenate([endpoints, fill])
    return np.sort(endpoints.astype(np.int64))


def sample_graph_diagnostic_edges(adjacency, max_edges, rng):
    upper = sparse.triu(adjacency, k=1).tocoo()
    if upper.nnz == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            0,
        )
    if upper.nnz <= max_edges:
        chosen = np.arange(upper.nnz, dtype=np.int64)
    else:
        chosen = np.sort(rng.choice(upper.nnz, size=max_edges, replace=False))
    return (
        upper.row[chosen].astype(np.int64),
        upper.col[chosen].astype(np.int64),
        upper.data[chosen].astype(np.float64),
        int(upper.nnz),
    )


def project_graph_embeddings_2d(embeddings, layout="pacmap", seed=0):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("graph diagnostic embeddings must be a matrix")
    if len(embeddings) == 0:
        return np.zeros((0, 2), dtype=np.float64), "PCA"
    if layout == "pacmap" and len(embeddings) >= 20:
        try:
            import pacmap

            n_neighbors = min(10, len(embeddings) - 1)
            coordinates = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=n_neighbors,
            ).fit_transform(np.ascontiguousarray(embeddings, dtype=np.float32))
            coordinates = np.asarray(coordinates, dtype=np.float64)
            if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
                return coordinates[:, :2], "PaCMAP"
            logger.warning(
                f"PaCMAP returned coordinates with shape {coordinates.shape}; falling back to PCA"
            )
        except Exception as exc:
            logger.warning(f"PaCMAP graph projection failed; falling back to PCA: {exc}")
    elif layout == "tsne" and len(embeddings) >= 2:
        try:
            coordinates = utils.project_tsne_embeddings(
                embeddings,
                seed=seed,
            ).astype(np.float64, copy=False)
            return coordinates, "t-SNE"
        except Exception as exc:
            logger.warning(f"t-SNE graph projection failed; falling back to PCA: {exc}")

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if embeddings.shape[1] == 1:
        return (
            np.column_stack([centered[:, 0], np.zeros(len(centered), dtype=np.float64)]),
            "PCA",
        )
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords), dtype=np.float64)])
    return coords, "PCA"


def _scatter_class_focus_nodes(ax, coords, labels, known_mask, focus_mask):
    """Draw the selected classes in their own colors over a muted context."""

    import matplotlib.pyplot as plt

    context = ~focus_mask
    if np.any(context):
        ax.scatter(
            coords[context, 0],
            coords[context, 1],
            s=14,
            color="#c2c7ce",
            alpha=0.55,
            linewidths=0,
            label="other classes",
            zorder=2,
        )

    focus_labels = np.unique(labels[focus_mask])
    colormap = plt.get_cmap("tab10" if len(focus_labels) <= 10 else "tab20")
    for index, label in enumerate(focus_labels):
        color = colormap(index % colormap.N)
        members = focus_mask & (labels == label)
        unlabeled = members if known_mask is None else members & (~known_mask)
        labeled = np.zeros_like(members) if known_mask is None else members & known_mask
        if np.any(unlabeled):
            ax.scatter(
                coords[unlabeled, 0],
                coords[unlabeled, 1],
                color=[color],
                marker="o",
                s=20,
                alpha=0.62,
                linewidths=0,
                label=f"class {int(label)} ({int(unlabeled.sum())} unlabeled)",
                zorder=3,
            )
        if np.any(labeled):
            ax.scatter(
                coords[labeled, 0],
                coords[labeled, 1],
                color=[color],
                marker="D",
                s=34,
                alpha=0.95,
                edgecolors="#111111",
                linewidths=0.35,
                label=f"class {int(label)} ({int(labeled.sum())} labeled)",
                zorder=4,
            )


def scatter_graph_nodes(ax, coords, labels, known_mask, focus_mask=None):
    if focus_mask is not None and labels is not None:
        _scatter_class_focus_nodes(ax, coords, labels, known_mask, focus_mask)
        return
    if labels is None:
        if known_mask is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=18, color="#4e79a7", alpha=0.78, label="samples", zorder=3)
            return
        labels = np.zeros(len(coords), dtype=np.int64)

    if known_mask is None:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="tab20",
            s=18,
            alpha=0.78,
            linewidths=0,
            label="samples",
            zorder=3,
        )
        return

    unlabeled = ~known_mask
    if np.any(unlabeled):
        ax.scatter(
            coords[unlabeled, 0],
            coords[unlabeled, 1],
            c=labels[unlabeled],
            cmap="tab20",
            marker="o",
            s=18,
            alpha=0.48,
            linewidths=0,
            label="unlabeled",
            zorder=3,
        )
    if np.any(known_mask):
        ax.scatter(
            coords[known_mask, 0],
            coords[known_mask, 1],
            c=labels[known_mask],
            cmap="tab20",
            marker="D",
            s=32,
            alpha=0.92,
            edgecolors="#111111",
            linewidths=0.35,
            label="labeled",
            zorder=4,
        )


def write_graph_edge_csv(
    csv_path,
    node_indices,
    edge_rows,
    edge_cols,
    edge_weights,
    positions,
    labels,
    known_mask,
):
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_graph_node",
                "target_graph_node",
                "source_position",
                "target_position",
                "weight",
                "source_label",
                "target_label",
                "source_kind",
                "target_kind",
            ],
        )
        writer.writeheader()
        for row, col, weight in zip(edge_rows, edge_cols, edge_weights):
            source = int(node_indices[int(row)])
            target = int(node_indices[int(col)])
            writer.writerow(
                {
                    "source_graph_node": source,
                    "target_graph_node": target,
                    "source_position": int(positions[source]),
                    "target_position": int(positions[target]),
                    "weight": float(weight),
                    "source_label": "" if labels is None else int(labels[source]),
                    "target_label": "" if labels is None else int(labels[target]),
                    "source_kind": graph_node_kind(known_mask, source),
                    "target_kind": graph_node_kind(known_mask, target),
                }
            )


def graph_node_kind(known_mask, node):
    if known_mask is None:
        return ""
    return "labeled" if bool(known_mask[int(node)]) else "unlabeled"
