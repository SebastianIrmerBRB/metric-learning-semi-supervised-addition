"""Write ``label_propagation.npz`` for epochs that were tracked before it existed.

Epoch tracking saves the trace of the label diffusion from the epoch it was
added onwards. Every earlier tracked epoch still holds the two things the
diffusion needs -- its graph (``graph_edges.npz``) and which nodes carried a
label (``graph_nodes.csv``) -- but not the embeddings, so the graph cannot be
rebuilt with label propagation's own settings. This runs the same series over
the graph the epoch saved instead, and records that in ``graph_source``:

``label_propagation``
    written during the run, on label propagation's own cosine kNN graph
    (Iscen's settings for an Iscen run), the graph ``class_evidence.npz``
    diffuses over. The trace ends where that file's answer is.

``tracked_graph``
    written here, on the run's own tracked graph, whose k, weighting and gamma
    are the method's. For an Iscen run with matching settings this is the same
    matrix; for a method with a binary kNN graph (LRML) it is a different one,
    so the propagation is "how labels would spread over this run's graph", not
    a replay of Iscen's numbers.

Examples::

    python scripts/backfill_label_propagation.py logs/final_tracking
    python scripts/backfill_label_propagation.py --force logs/.../fold_00/2026-09-17_12-00-00
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.epoch_tracking import (  # noqa: E402
    GRAPH_EDGES_NAME,
    OUTPUT_DIR_NAME,
    PROPAGATION_NAME,
    PROPAGATION_STEPS,
    graph_parameters,
    label_propagation_params,
    write_label_propagation_trace,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Run directories, tracking directories, or roots to search")
    parser.add_argument("--force", action="store_true", help="Rewrite epochs that already have the file")
    parser.add_argument("--steps", type=int, default=PROPAGATION_STEPS, help="Steps of the series to record")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Diffusion alpha, overriding the run's own (default: the run's, else 0.99)",
    )
    return parser.parse_args()


def find_tracking_dirs(paths):
    """Every ``epoch_tracking`` directory under the given paths, in order."""

    found = []
    for path in paths:
        path = Path(path)
        if path.name == OUTPUT_DIR_NAME and path.is_dir():
            found.append(path)
            continue
        if (path / OUTPUT_DIR_NAME).is_dir():
            found.append(path / OUTPUT_DIR_NAME)
            continue
        found.extend(sorted(candidate for candidate in path.rglob(OUTPUT_DIR_NAME) if candidate.is_dir()))
    return sorted(dict.fromkeys(found))


def read_graph(edges_path):
    """The saved edges as a symmetric CSR affinity over the graph's nodes."""

    with np.load(edges_path) as saved:
        positions = saved["positions"].astype(np.int64)
        source = saved["source"].astype(np.int64)
        target = saved["target"].astype(np.int64)
        weight = saved["weight"].astype(np.float64)
        symmetric = bool(saved["symmetric"])
    count = len(positions)
    affinity = sparse.coo_matrix((weight, (source, target)), shape=(count, count)).tocsr()
    if symmetric:
        # Only the upper triangle was stored; the graph itself is undirected.
        affinity = affinity + affinity.T
    else:
        affinity = affinity.maximum(affinity.T)
    return positions, affinity.tocsr()


def read_nodes(nodes_path, positions):
    """``(labels, known_mask)`` in the edge file's node order."""

    with Path(nodes_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(positions):
        raise ValueError(f"{nodes_path} has {len(rows)} nodes, the graph has {len(positions)}")
    if [int(row["position"]) for row in rows] != positions.tolist():
        raise ValueError(f"{nodes_path} is not in the same node order as the saved edges")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    known_mask = np.asarray([row["kind"] == "labeled" for row in rows], dtype=bool)
    return labels, known_mask


def run_params(tracking_dir, alpha=None):
    """The diffusion settings of the run that wrote this tracking directory.

    Only ``alpha`` can still change the answer here: the graph is the one the
    epoch saved, so its k and gamma are already fixed in the matrix -- they are
    passed through as metadata so the saved trace says which graph it ran on.
    An Iscen run tuned alpha, and reading a run's diffusion back with someone
    else's alpha would be a different diffusion.
    """

    config_path = Path(tracking_dir).parent / "run_config.json"
    params = {}
    if config_path.is_file():
        ssl_config = json.loads(config_path.read_text()).get("ssl_config") or {}
        params = {key: value for key, value in label_propagation_params(ssl_config).items() if key == "alpha"}
        # k and gamma are recorded, not applied: they describe the graph the
        # epoch saved, which is the run's tracked graph rather than one built here.
        n_neighbors, _, gamma = graph_parameters(ssl_config)
        params.update({"n_neighbors": n_neighbors, "gamma": gamma})
    if alpha is not None:
        params["alpha"] = float(alpha)
    return params


def backfill_epoch(epoch_dir, steps, params=None):
    positions, affinity = read_graph(epoch_dir / GRAPH_EDGES_NAME)
    labels, known_mask = read_nodes(epoch_dir / "graph_nodes.csv", positions)
    if not known_mask.any():
        raise ValueError("the epoch has no labeled node to propagate from")
    return write_label_propagation_trace(
        epoch_dir / PROPAGATION_NAME,
        None,
        positions,
        labels,
        known_mask,
        dict(params or {}),
        graph_source="tracked_graph",
        steps=steps,
        affinity=affinity,
    )


def main():
    options = parse_args()
    tracking_dirs = find_tracking_dirs(options.paths)
    if not tracking_dirs:
        logger.error(f"No {OUTPUT_DIR_NAME} directory under {[str(path) for path in options.paths]}")
        return 1
    written = skipped = failed = 0
    for tracking_dir in tracking_dirs:
        params = run_params(tracking_dir, options.alpha)
        logger.info(f"{tracking_dir}: alpha={params.get('alpha', 0.99)}")
        for epoch_dir in sorted(path for path in tracking_dir.glob("epoch_*") if path.is_dir()):
            if not (epoch_dir / GRAPH_EDGES_NAME).is_file() or not (epoch_dir / "graph_nodes.csv").is_file():
                continue
            if (epoch_dir / PROPAGATION_NAME).is_file() and not options.force:
                skipped += 1
                continue
            started_at = time.perf_counter()
            try:
                summary = backfill_epoch(epoch_dir, options.steps, params)
            except Exception as exc:
                logger.warning(f"{epoch_dir}: {exc}")
                failed += 1
                continue
            written += 1
            reached = summary.get("lp_reached_share")
            purity = summary.get("lp_tree_edge_purity")
            logger.info(
                f"{epoch_dir.relative_to(tracking_dir.parent)}: {time.perf_counter() - started_at:.1f}s"
                + (f", reached {reached:.3f}" if reached is not None else "")
                + (f", first-arrival edge purity {purity:.3f}" if purity is not None else "")
            )
    logger.info(
        f"{written} epochs written, {skipped} already had {PROPAGATION_NAME} (--force rewrites), {failed} failed, "
        f"over {len(tracking_dirs)} tracking directories"
    )
    return 0 if written or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
