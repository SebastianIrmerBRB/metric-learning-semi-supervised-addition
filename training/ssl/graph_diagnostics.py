"""Optional graph plots and sampled-edge diagnostics for SSL methods."""

import csv
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import sparse

from .config import GraphDiagnosticsRequest


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
    return GraphDiagnosticsRequest(
        output_dir=Path(log_dir) / "graph_diagnostics",
        slug=_safe_diagnostic_slug(f"{name}_{epoch_slug}"),
        title=title or str(name),
        max_nodes=int(config.graph_diagnostics_max_nodes),
        max_edges=int(config.graph_diagnostics_max_edges),
        max_labels=int(config.graph_diagnostics_max_labels),
        seed=seed,
        layout=str(config.graph_diagnostics_layout),
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
):
    """Write a graph PNG and sampled edge CSV without affecting training."""

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
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping graph diagnostics PNG")
        return None

    embeddings = np.asarray(embeddings, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.int64)
    adjacency = adjacency.tocsr()
    num_nodes = adjacency.shape[0]
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph diagnostics adjacency must be square")
    if len(embeddings) != num_nodes or len(positions) != num_nodes:
        raise ValueError("graph diagnostics embeddings, positions, and adjacency must align")

    labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    if labels is not None and len(labels) != num_nodes:
        labels = None
    known_mask = None if known_mask is None else np.asarray(known_mask, dtype=bool)
    if known_mask is not None and len(known_mask) != num_nodes:
        known_mask = None

    rng = np.random.default_rng(request.seed)
    node_indices = choose_graph_diagnostic_nodes(
        adjacency=adjacency,
        max_nodes=request.max_nodes,
        rng=rng,
    )
    sampled_adjacency = adjacency[node_indices][:, node_indices]
    edge_rows, edge_cols, edge_weights, full_edge_count = sample_graph_diagnostic_edges(
        adjacency=sampled_adjacency,
        max_edges=request.max_edges,
        rng=rng,
    )

    request.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = request.output_dir / f"{request.slug}.png"
    csv_path = request.output_dir / f"{request.slug}_sampled_edges.csv"

    coords, projection_name = project_graph_embeddings_2d(
        embeddings[node_indices],
        layout=request.layout,
        seed=request.seed,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    for row, col, weight in zip(edge_rows, edge_cols, edge_weights):
        alpha = 0.12 + 0.35 * min(abs(float(weight)), 1.0)
        ax.plot(
            [coords[row, 0], coords[col, 0]],
            [coords[row, 1], coords[col, 1]],
            color="#8a8f98",
            linewidth=0.45,
            alpha=alpha,
            zorder=1,
        )

    sampled_labels = None if labels is None else labels[node_indices]
    sampled_known = None if known_mask is None else known_mask[node_indices]
    scatter_graph_nodes(ax, coords, sampled_labels, sampled_known)
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

    total_edges = sparse.triu(adjacency, k=1).nnz
    ax.set_title(
        f"{request.title}\n"
        f"showing {len(node_indices)}/{num_nodes} samples and "
        f"{len(edge_rows)}/{total_edges} undirected edges"
    )
    ax.set_xlabel(f"{projection_name} 1")
    ax.set_ylabel(f"{projection_name} 2")
    ax.tick_params(labelsize=8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    write_graph_edge_csv(
        csv_path=csv_path,
        node_indices=node_indices,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        edge_weights=edge_weights,
        positions=positions,
        labels=labels,
        known_mask=known_mask,
    )
    logger.info(
        "Saved graph diagnostics: "
        f"png={png_path}, sampled_edges={csv_path}, "
        f"sampled_nodes={len(node_indices)}, sampled_edges={len(edge_rows)}, "
        f"available_sampled_edges={full_edge_count}, projection={projection_name}"
    )
    return png_path


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


def scatter_graph_nodes(ax, coords, labels, known_mask):
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
