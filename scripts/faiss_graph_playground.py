"""Small FAISS graph-index playground.

Examples:
    python scripts/faiss_graph_playground.py
    python scripts/faiss_graph_playground.py --index hnsw --hnsw-m 4 --ef-search 8
    python scripts/faiss_graph_playground.py --index all --points 500 --k 10 --no-plot

The script builds a tiny synthetic embedding set, compares graph indexes against
an exact FlatL2 baseline, and writes optional 2D graph plots under show/.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


MIN_NNDESCENT_POINTS = 101


@dataclass(frozen=True)
class BuiltIndex:
    name: str
    explanation: str
    params: dict[str, object]
    index: object
    build_seconds: float


@dataclass(frozen=True)
class SearchReport:
    built: BuiltIndex
    distances: np.ndarray
    ids: np.ndarray
    search_seconds: float
    edges: list[tuple[int, int]]
    graph_notes: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build small FAISS indexes and compare graph constructions against "
            "an exact FlatL2 search baseline."
        )
    )
    parser.add_argument(
        "--index",
        choices=("all", "flat", "hnsw", "nsg", "nndescent"),
        default="all",
        help="Which FAISS index to build.",
    )
    parser.add_argument("--points", type=int, default=3000, help="Database vectors.")
    parser.add_argument("--queries", type=int, default=5, help="Query vectors.")
    parser.add_argument("--dim", type=int, default=2, help="Embedding dimension.")
    parser.add_argument("--clusters", type=int, default=4, help="Synthetic clusters.")
    parser.add_argument("--cluster-std", type=float, default=0.22, help="Cluster noise.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--k", type=int, default=50, help="Nearest neighbors to retrieve.")
    parser.add_argument(
        "--show-query",
        type=int,
        default=0,
        help="Query row to print and draw in detail.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("show") / "faiss_graph_playground",
        help="Directory for plots.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib plots.")
    parser.add_argument(
        "--max-plot-edges",
        type=int,
            default=1400,
        help="Maximum graph edges to draw per plot.",
    )

    parser.add_argument("--hnsw-m", type=int, default=8, help="HNSW neighbor budget M.")
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=40,
        help="HNSW candidate list size during construction.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=16,
        help="HNSW candidate list size during search.",
    )

    parser.add_argument("--nsg-r", type=int, default=16, help="NSG max out-degree R.")
    parser.add_argument(
        "--nsg-search-l",
        type=int,
        default=32,
        help="NSG search candidate list size.",
    )

    parser.add_argument(
        "--nnd-k",
        type=int,
        default=16,
        help="NNDescent stored neighbor count K.",
    )
    parser.add_argument(
        "--nnd-iter",
        type=int,
        default=10,
        help="NNDescent refinement iterations.",
    )
    parser.add_argument(
        "--nnd-search-l",
        type=int,
        default=32,
        help="NNDescent search candidate list size.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.points < 2:
        raise SystemExit("--points must be at least 2")
    if args.queries < 1:
        raise SystemExit("--queries must be at least 1")
    if args.dim < 1:
        raise SystemExit("--dim must be at least 1")
    if args.clusters < 1:
        raise SystemExit("--clusters must be at least 1")
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    if args.k > args.points:
        raise SystemExit("--k cannot be larger than --points")
    if args.show_query < 0 or args.show_query >= args.queries:
        raise SystemExit("--show-query must refer to an existing query row")
    for name in (
        "hnsw_m",
        "ef_construction",
        "ef_search",
        "nsg_r",
        "nsg_search_l",
        "nnd_k",
        "nnd_iter",
        "nnd_search_l",
        "max_plot_edges",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")


def make_clustered_vectors(
    points: int,
    queries: int,
    dim: int,
    clusters: int,
    cluster_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if dim == 2:
        angles = np.linspace(0, 2 * math.pi, clusters, endpoint=False)
        centers = np.column_stack([np.cos(angles), np.sin(angles)]).astype("float32")
        centers *= 4.0
    else:
        centers = rng.normal(size=(clusters, dim)).astype("float32")
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
        centers *= 4.0

    labels = np.arange(points) % clusters
    rng.shuffle(labels)
    query_labels = np.arange(queries) % clusters

    xb = centers[labels] + rng.normal(scale=cluster_std, size=(points, dim))
    xq = centers[query_labels] + rng.normal(scale=cluster_std, size=(queries, dim))
    return (
        np.ascontiguousarray(xb.astype("float32")),
        np.ascontiguousarray(xq.astype("float32")),
        labels.astype("int32"),
        query_labels.astype("int32"),
    )


def timed(call: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def build_flat(faiss, xb: np.ndarray, args: argparse.Namespace) -> BuiltIndex:
    def construct():
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)
        return index

    index, seconds = timed(construct)
    return BuiltIndex(
        name="FlatL2 exact",
        explanation=(
            "No graph: FAISS scans every vector and returns the true L2 nearest "
            "neighbors. Use this as the recall baseline."
        ),
        params={"metric": "squared L2"},
        index=index,
        build_seconds=seconds,
    )


def build_hnsw(faiss, xb: np.ndarray, args: argparse.Namespace) -> BuiltIndex:
    def construct():
        index = faiss.IndexHNSWFlat(xb.shape[1], args.hnsw_m)
        index.hnsw.efConstruction = args.ef_construction
        index.hnsw.efSearch = args.ef_search
        index.add(xb)
        return index

    index, seconds = timed(construct)
    return BuiltIndex(
        name="HNSW",
        explanation=(
            "Hierarchical Navigable Small World graph. Points are inserted one at "
            "a time into layered proximity graphs; search walks from sparse upper "
            "layers down to the dense base layer."
        ),
        params={
            "M": args.hnsw_m,
            "efConstruction": args.ef_construction,
            "efSearch": args.ef_search,
        },
        index=index,
        build_seconds=seconds,
    )


def build_nsg(faiss, xb: np.ndarray, args: argparse.Namespace) -> BuiltIndex:
    if xb.shape[0] < MIN_NNDESCENT_POINTS:
        raise ValueError(
            "FAISS builds NSG from an NNDescent candidate graph in this path, "
            f"which requires at least {MIN_NNDESCENT_POINTS} vectors."
        )

    def construct():
        index = faiss.IndexNSGFlat(xb.shape[1], args.nsg_r)
        index.nsg.search_L = max(args.nsg_search_l, args.k)
        index.add(xb)
        return index

    index, seconds = timed(construct)
    return BuiltIndex(
        name="NSG",
        explanation=(
            "Navigating Spreading-out Graph. FAISS first creates an approximate "
            "kNN candidate graph, then prunes it into a sparse graph that tries "
            "to keep short paths and angularly diverse outgoing links."
        ),
        params={
            "R": args.nsg_r,
            "search_L": max(args.nsg_search_l, args.k),
            "internal_L": index.nsg.L,
            "internal_C": index.nsg.C,
        },
        index=index,
        build_seconds=seconds,
    )


def build_nndescent(faiss, xb: np.ndarray, args: argparse.Namespace) -> BuiltIndex:
    if xb.shape[0] < MIN_NNDESCENT_POINTS:
        raise ValueError(
            "FAISS NNDescent requires more than 100 vectors for its built-in "
            "graph evaluation sample."
        )

    def construct():
        index = faiss.IndexNNDescentFlat(xb.shape[1], args.nnd_k)
        index.nndescent.iter = args.nnd_iter
        index.nndescent.search_L = max(args.nnd_search_l, args.k)
        index.add(xb)
        return index

    index, seconds = timed(construct)
    return BuiltIndex(
        name="NNDescent",
        explanation=(
            "Approximate kNN graph construction. It repeatedly tests neighbors "
            "of neighbors, because nearby points often share nearby points."
        ),
        params={
            "K": args.nnd_k,
            "iter": args.nnd_iter,
            "search_L": max(args.nnd_search_l, args.k),
            "internal_R": index.nndescent.R,
            "internal_L": index.nndescent.L,
            "internal_S": index.nndescent.S,
        },
        index=index,
        build_seconds=seconds,
    )


def selected_builders(args: argparse.Namespace):
    builders = {
        "flat": build_flat,
        "hnsw": build_hnsw,
        "nsg": build_nsg,
        "nndescent": build_nndescent,
    }
    if args.index == "all":
        return [builders[name] for name in ("flat", "hnsw", "nsg", "nndescent")]
    return [builders[args.index]]


def extract_hnsw_edges(faiss, index) -> tuple[list[tuple[int, int]], dict[str, object]]:
    offsets = faiss.vector_to_array(index.hnsw.offsets)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    levels = faiss.vector_to_array(index.hnsw.levels)
    base_slots = int(index.hnsw.nb_neighbors(0))
    edges: list[tuple[int, int]] = []

    for source in range(index.ntotal):
        start = int(offsets[source])
        stop = min(start + base_slots, int(offsets[source + 1]))
        for target in neighbors[start:stop]:
            target = int(target)
            if target >= 0 and target != source:
                edges.append((source, target))

    unique_levels, counts = np.unique(levels, return_counts=True)
    level_counts = {int(level): int(count) for level, count in zip(unique_levels, counts)}
    return edges, {
        "base_slots_per_node": base_slots,
        "entry_point": int(index.hnsw.entry_point),
        "max_level": int(index.hnsw.max_level),
        "stored_level_counts": level_counts,
        "note": "Plotted edges are only the dense base layer; HNSW also has sparse upper layers.",
    }


def extract_nsg_edges(index) -> tuple[list[tuple[int, int]], dict[str, object]]:
    graph = index.nsg.get_final_graph()
    edges: list[tuple[int, int]] = []
    for source in range(graph.N):
        for slot in range(graph.K):
            target = int(graph.at(source, slot))
            if target >= 0 and target != source:
                edges.append((source, target))
    return edges, {
        "slots_per_node": int(graph.K),
        "enterpoint": int(index.nsg.enterpoint),
        "note": "NSG rows may contain -1 slots after pruning, so out-degree can be below R.",
    }


def extract_nndescent_edges(faiss, index) -> tuple[list[tuple[int, int]], dict[str, object]]:
    graph = faiss.vector_to_array(index.nndescent.final_graph)
    slots = int(index.nndescent.K)
    if graph.size % index.ntotal != 0:
        slots = graph.size // index.ntotal

    edges: list[tuple[int, int]] = []
    for source in range(index.ntotal):
        row = graph[source * slots : (source + 1) * slots]
        for target in row:
            target = int(target)
            if target >= 0 and target != source:
                edges.append((source, target))
    return edges, {
        "slots_per_node": slots,
        "has_built": bool(index.nndescent.has_built),
        "note": "This is the final approximate kNN graph after NNDescent refinement.",
    }


def extract_edges(faiss, built: BuiltIndex) -> tuple[list[tuple[int, int]], dict[str, object]]:
    if built.name == "HNSW":
        return extract_hnsw_edges(faiss, built.index)
    if built.name == "NSG":
        return extract_nsg_edges(built.index)
    if built.name == "NNDescent":
        return extract_nndescent_edges(faiss, built.index)
    return [], {"note": "FlatL2 has no graph."}


def graph_summary(nodes: int, edges: list[tuple[int, int]]) -> dict[str, object]:
    if not edges:
        return {
            "directed_edges": 0,
            "undirected_edges": 0,
            "mean_out_degree": 0.0,
            "min_out_degree": 0,
            "max_out_degree": 0,
            "reciprocal_fraction": 0.0,
            "components": nodes,
            "largest_component": 1 if nodes else 0,
            "isolated_nodes": nodes,
        }

    out_degree = np.zeros(nodes, dtype=np.int32)
    adjacency: list[set[int]] = [set() for _ in range(nodes)]
    edge_set = set(edges)
    undirected = set()
    for source, target in edges:
        out_degree[source] += 1
        adjacency[source].add(target)
        adjacency[target].add(source)
        undirected.add((min(source, target), max(source, target)))

    seen = np.zeros(nodes, dtype=bool)
    component_sizes: list[int] = []
    for start in range(nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        component_sizes.append(size)

    reciprocal = sum(1 for source, target in edges if (target, source) in edge_set)
    return {
        "directed_edges": len(edges),
        "undirected_edges": len(undirected),
        "mean_out_degree": float(out_degree.mean()),
        "min_out_degree": int(out_degree.min()),
        "max_out_degree": int(out_degree.max()),
        "reciprocal_fraction": reciprocal / len(edges),
        "components": len(component_sizes),
        "largest_component": max(component_sizes),
        "isolated_nodes": int(sum(1 for neighbors in adjacency if not neighbors)),
    }


def search_index(built: BuiltIndex, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    def run_search():
        return built.index.search(xq, k)

    result, seconds = timed(run_search)
    distances, ids = result
    return distances, ids, seconds


def recall_at_k(exact_ids: np.ndarray, candidate_ids: np.ndarray, k: int) -> float:
    recalls = []
    for exact_row, candidate_row in zip(exact_ids[:, :k], candidate_ids[:, :k]):
        exact = set(int(value) for value in exact_row if value >= 0)
        candidate = set(int(value) for value in candidate_row if value >= 0)
        recalls.append(len(exact & candidate) / max(1, len(exact)))
    return float(np.mean(recalls))


def format_ids(values: np.ndarray) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def format_distances(values: np.ndarray) -> str:
    formatted = []
    missing_distance = np.finfo(np.float32).max / 2.0
    for value in values:
        value = float(value)
        if math.isfinite(value) and value < missing_distance:
            formatted.append(f"{value:.4f}".rstrip("0").rstrip("."))
        else:
            formatted.append("inf")
    return "[" + ", ".join(formatted) + "]"


def valid_ids(values: np.ndarray) -> list[int]:
    return [int(value) for value in values if int(value) >= 0]


def print_report(
    args: argparse.Namespace,
    xb: np.ndarray,
    xq: np.ndarray,
    query_labels: np.ndarray,
    reports: list[SearchReport],
    exact_report: SearchReport,
) -> None:
    print("\nFAISS graph playground")
    print(
        f"  data: {xb.shape[0]} database vectors, {xq.shape[0]} queries, "
        f"dim={xb.shape[1]}, k={args.k}"
    )
    print(
        f"  query {args.show_query}: synthetic_cluster="
        f"{int(query_labels[args.show_query])}, vector={xq[args.show_query].round(3).tolist()}"
    )

    exact_ids = exact_report.ids
    exact_distances = exact_report.distances
    print("\nExact baseline")
    print(f"  ids:       {format_ids(exact_ids[args.show_query])}")
    print(f"  distances: {format_distances(exact_distances[args.show_query])}")

    for report in reports:
        summary = graph_summary(xb.shape[0], report.edges)
        recall = recall_at_k(exact_ids, report.ids, args.k)
        print(f"\n{report.built.name}")
        print(f"  idea: {report.built.explanation}")
        print(f"  params: {report.built.params}")
        print(
            f"  build={report.built.build_seconds * 1000:.2f} ms, "
            f"search={report.search_seconds * 1000:.2f} ms, "
            f"recall@{args.k} vs FlatL2={recall:.3f}"
        )
        print(
            "  graph: "
            f"directed_edges={summary['directed_edges']}, "
            f"mean_out_degree={summary['mean_out_degree']:.2f}, "
            f"min/max_out_degree={summary['min_out_degree']}/{summary['max_out_degree']}, "
            f"components={summary['components']}, "
            f"largest_component={summary['largest_component']}, "
            f"reciprocal_fraction={summary['reciprocal_fraction']:.2f}"
        )
        print(f"  graph notes: {report.graph_notes}")
        exact_set = set(valid_ids(exact_ids[args.show_query]))
        result_set = set(valid_ids(report.ids[args.show_query]))
        missed = sorted(exact_set - result_set)
        unfilled = int(np.sum(report.ids[args.show_query] < 0))
        print(f"  ids:       {format_ids(report.ids[args.show_query])}")
        print(f"  distances: {format_distances(report.distances[args.show_query])}")
        if unfilled:
            print(f"  unfilled result slots for query {args.show_query}: {unfilled}")
        print(f"  missed exact ids for query {args.show_query}: {missed}")


def plot_report(
    args: argparse.Namespace,
    xb: np.ndarray,
    labels: np.ndarray,
    xq: np.ndarray,
    report: SearchReport,
    exact_report: SearchReport,
) -> Path | None:
    if args.no_plot or xb.shape[1] != 2:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.", file=sys.stderr)
        return None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{report.built.name.lower().replace(' ', '_')}.png"

    rng = np.random.default_rng(args.seed)
    edges = report.edges
    if len(edges) > args.max_plot_edges:
        edge_indices = rng.choice(len(edges), size=args.max_plot_edges, replace=False)
        edges = [edges[int(index)] for index in edge_indices]

    fig, ax = plt.subplots(figsize=(8, 7))
    for source, target in edges:
        ax.plot(
            [xb[source, 0], xb[target, 0]],
            [xb[source, 1], xb[target, 1]],
            color="#9aa0a6",
            linewidth=0.45,
            alpha=0.25,
            zorder=1,
        )

    scatter = ax.scatter(
        xb[:, 0],
        xb[:, 1],
        c=labels,
        cmap="tab10",
        s=18,
        alpha=0.82,
        linewidths=0,
        zorder=2,
    )
    scatter.set_rasterized(True)

    query_index = args.show_query
    exact_neighbors = valid_ids(exact_report.ids[query_index])
    result_neighbors = valid_ids(report.ids[query_index])
    if exact_neighbors:
        ax.scatter(
            xb[exact_neighbors, 0],
            xb[exact_neighbors, 1],
            s=150,
            facecolors="none",
            edgecolors="#188038",
            linewidths=1.8,
            label="exact neighbors",
            zorder=4,
        )
    if result_neighbors:
        ax.scatter(
            xb[result_neighbors, 0],
            xb[result_neighbors, 1],
            s=70,
            marker="s",
            facecolors="none",
            edgecolors="#f29900",
            linewidths=1.6,
            label=f"{report.built.name} neighbors",
            zorder=5,
        )
    ax.scatter(
        xq[query_index, 0],
        xq[query_index, 1],
        s=180,
        marker="x",
        color="#d93025",
        linewidths=2.4,
        label="query",
        zorder=6,
    )

    ax.set_title(f"{report.built.name}: graph edges and query {query_index}")
    ax.set_xlabel("synthetic embedding dimension 0")
    ax.set_ylabel("synthetic embedding dimension 1")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    validate_args(args)

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "This playground requires faiss-cpu. Install the project requirements first."
        ) from exc

    xb, xq, labels, query_labels = make_clustered_vectors(
        points=args.points,
        queries=args.queries,
        dim=args.dim,
        clusters=args.clusters,
        cluster_std=args.cluster_std,
        seed=args.seed,
    )

    flat = build_flat(faiss, xb, args)
    flat_distances, flat_ids, flat_search_seconds = search_index(flat, xq, args.k)
    flat_edges, flat_notes = extract_edges(faiss, flat)
    exact_report = SearchReport(
        built=flat,
        distances=flat_distances,
        ids=flat_ids,
        search_seconds=flat_search_seconds,
        edges=flat_edges,
        graph_notes=flat_notes,
    )

    reports: list[SearchReport] = []
    for builder in selected_builders(args):
        if builder is build_flat:
            reports.append(exact_report)
            continue

        try:
            built = builder(faiss, xb, args)
        except ValueError as exc:
            print(f"\nSkipping {builder.__name__.removeprefix('build_')}: {exc}")
            continue

        distances, ids, search_seconds = search_index(built, xq, args.k)
        edges, notes = extract_edges(faiss, built)
        reports.append(
            SearchReport(
                built=built,
                distances=distances,
                ids=ids,
                search_seconds=search_seconds,
                edges=edges,
                graph_notes=notes,
            )
        )

    if not reports:
        reports.append(exact_report)

    print_report(args, xb, xq, query_labels, reports, exact_report)

    plot_paths = []
    for report in reports:
        path = plot_report(args, xb, labels, xq, report, exact_report)
        if path is not None:
            plot_paths.append(path)

    if plot_paths:
        print("\nPlots")
        for path in plot_paths:
            print(f"  {path}")
    elif args.no_plot:
        print("\nPlots disabled by --no-plot.")
    elif xb.shape[1] != 2:
        print("\nPlots skipped because --dim is not 2.")

    print("\nTry changing --ef-search, --hnsw-m, --nsg-r, --nnd-k, or --points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
