"""Epoch tracking read one epoch at a time, while the run is still training.

``export_embedding_views.tracking_view`` turns a finished fold into one file:
every epoch, aligned and normalized together, written into the app's
``public/``. That is the right shape for the static site and the wrong one for
watching a run: every new epoch would mean rebuilding and resending all the
epochs before it, and a fold is only exported once it has finished.

This module splits the same view into the parts that never change during a run
and the parts one epoch adds, so the dev server can hand the Embeddings page
exactly what is new:

``base``
    Everything shared by all epochs: which pool samples are tracked, their
    classes and labeled/unlabeled subsets, and the run's description. It is
    read from the earliest epoch that has a layout, so it is available as soon
    as the epoch -1 projection lands.

``frames``
    One entry per requested epoch: its raw 2-D (and, once written, 3-D)
    projection, the per-node graph metrics, the graph's edges, class evidence,
    the propagation series, and the run-level pseudo labels, SLADE clusters and
    mined pairs that epoch was trained on. The encodings are the exporter's own,
    so the viewer decodes a live frame exactly as it decodes an exported one.

The layouts are left unaligned. Procrustes alignment runs in the browser over
the frames it holds, because each epoch is aligned to the one before it and the
common scale is set by all of them together, both of which change as epochs
arrive.

Each part a run writes is read on its own and may still be missing or half
written: an epoch whose 2-D projection is not complete yet is reported as
``pending`` and the rest of a frame lists what it could not read under
``incomplete``, so the next poll simply asks again. Nothing here writes into a
run directory.

Class indices: a view numbers classes in first-seen order over the tracked
rows. A predicted class that no tracked row carries (a pseudo label on a
labeled class outside the tracked sample) is appended after them. Frames are
built in separate processes, so the caller passes the labels already appended
(``--extra-labels``) and gets the extended list back; frames built against a
shorter list stay valid under a longer one.

Command line, used by ``react_trial_visualizer/app/api/live-tracking/``::

    python -m evaluate.live_tracking base   <tracking_dir>
    python -m evaluate.live_tracking frames <tracking_dir> --epochs -1,0,5 [--extra-labels 12,40]

Both print one JSON document to stdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import export_embedding_views as exporter  # noqa: E402

FORMAT_VERSION = 1
LAYOUT_PREFIXES = ("tsne", "pacmap", "pca")


def epoch_slug(epoch):
    """The directory name the tracker gives an epoch: ``epoch_m001``, ``epoch_0015``."""

    epoch = int(epoch)
    return f"epoch_m{abs(epoch):03d}" if epoch < 0 else f"epoch_{epoch:04d}"


def epoch_dirs(tracking_dir):
    """``[(epoch, directory)]`` in epoch order."""

    found = []
    for path in Path(tracking_dir).glob("epoch_*"):
        if not path.is_dir():
            continue
        try:
            found.append((exporter.epoch_from_slug(path.name), path))
        except ValueError:
            continue
    return sorted(found, key=lambda item: item[0])


def layout_csv(epoch_dir, prefix=None):
    """The 2-D layout CSV of one epoch (``embeddings_<prefix>.csv``), or None."""

    names = [f"embeddings_{prefix}.csv"] if prefix else [f"embeddings_{name}.csv" for name in LAYOUT_PREFIXES]
    for name in names:
        path = Path(epoch_dir) / name
        if path.is_file():
            return path
    return None


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_layout(path, prefix, axes, count=None, labels=None):
    """Coordinates of a layout CSV, or ``None`` while it is incomplete.

    A CSV being written is a prefix of the final one; it is recognized by its
    row count, and by labels that no longer line up with the tracked rows.
    """

    try:
        rows = read_csv_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None, None
    if not rows or (count is not None and len(rows) != count):
        return None, rows
    if labels is not None and [row.get("label") for row in rows] != labels:
        return None, rows
    try:
        coordinates = np.asarray([[float(row[f"{prefix}_{axis}"]) for axis in axes] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None, rows
    if not np.all(np.isfinite(coordinates)):
        return None, rows
    return coordinates, rows


class TrackingContext:
    """What every frame of one tracking directory needs from the base."""

    def __init__(self, tracking_dir):
        self.tracking_dir = Path(tracking_dir)
        self.run_dir = self.tracking_dir.parent
        self.epochs = epoch_dirs(self.tracking_dir)
        self.reference_epoch = None
        self.projection = None
        self.base_rows = None
        for epoch, directory in self.epochs:
            path = layout_csv(directory)
            if path is None:
                continue
            prefix = path.stem.removeprefix("embeddings_")
            coordinates, rows = read_layout(path, prefix, ("x", "y"))
            if coordinates is None:
                continue
            self.reference_epoch, self.projection, self.base_rows = epoch, prefix, rows
            break
        if self.base_rows is None:
            raise LookupError(f"{self.tracking_dir} has no complete tracked embedding layout yet")

        self.count = len(self.base_rows)
        self.labels = [row["label"] for row in self.base_rows]
        # The labeled/unlabeled split comes from the first graph_nodes.csv that
        # lines up with the tracked rows; until one is written every point is "train".
        self.node_rows = []
        self.node_epoch = None
        for epoch, directory in self.epochs:
            path = directory / "graph_nodes.csv"
            if not path.is_file():
                continue
            try:
                nodes = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            if len(nodes) == self.count and [node.get("label") for node in nodes] == self.labels:
                self.node_rows, self.node_epoch = nodes, epoch
                break
        if self.node_rows:
            self.subsets = [
                exporter.SUBSET_LABELED if node["kind"] == "labeled" else exporter.SUBSET_UNLABELED
                for node in self.node_rows
            ]
            self.position_to_view = {int(node["position"]): index for index, node in enumerate(self.node_rows)}
            self.samples = [int(node["position"]) for node in self.node_rows]
        else:
            self.subsets = [exporter.SUBSET_TRAIN] * self.count
            self.position_to_view = {}
            self.samples = [int(row.get("sample_position") or index) for index, row in enumerate(self.base_rows)]
        names = [row.get("class_name") or f"class {row['label']}" for row in self.base_rows]
        self.class_names, self.class_index = exporter.encode_classes(names)
        self.base_class_count = len(self.class_names)
        self.view_class_of_label = {
            int(row["label"]): int(index) for row, index in zip(self.base_rows, self.class_index)
        }
        self.extra_labels = []

    @property
    def base_key(self):
        """Changes whenever the rows, subsets or classes of the base would."""

        digest = hashlib.sha1()
        digest.update(json.dumps([self.reference_epoch, self.projection, self.node_epoch, self.count]).encode())
        digest.update("|".join(self.labels).encode())
        digest.update("|".join(self.subsets).encode())
        digest.update("|".join(map(str, self.samples)).encode())
        return digest.hexdigest()[:16]

    def seed_extra_labels(self, labels):
        for label in labels:
            self.class_index_of_label(int(label))

    def class_index_of_label(self, label):
        # A predicted class no tracked row carries still gets a name.
        if label not in self.view_class_of_label:
            self.view_class_of_label[label] = len(self.class_names)
            self.class_names.append(f"class {label}")
            self.extra_labels.append(int(label))
        return self.view_class_of_label[label]

    def base(self):
        args, ssl_config = exporter.read_run_metadata(self.run_dir)
        subset_names, subset_index = exporter.encode_subsets(self.subsets)
        return {
            "version": FORMAT_VERSION,
            "id": exporter.view_id(self.tracking_dir, "tracking"),
            "baseKey": self.base_key,
            "source": "epoch_tracking",
            "sourcePath": exporter.repository_relative(self.tracking_dir),
            "projection": self.projection,
            "referenceEpoch": self.reference_epoch,
            "nodeEpoch": self.node_epoch,
            **exporter.describe_run(self.run_dir, args, ssl_config),
            "subsets": subset_names,
            "classNames": self.class_names[: self.base_class_count],
            "points": {"subset": subset_index, "class": self.class_index.tolist(), "sample": self.samples},
        }


def node_metrics(nodes, position_to_view):
    metrics = {}
    for name in exporter.TRACKING_NODE_METRICS:
        if name not in nodes[0]:
            continue
        values = []
        for node in nodes:
            raw = node[name]
            value = 1.0 if raw == "True" else 0.0 if raw == "False" else exporter.float_or_none(raw)
            values.append(None if value is None else round(value, 4))
        metrics[name] = values
    if "strongest_neighbor_position" in nodes[0]:
        metrics["strongest_neighbor"] = [
            position_to_view.get(int(float(node["strongest_neighbor_position"])), -1)
            if exporter.float_or_none(node["strongest_neighbor_position"]) is not None
            else -1
            for node in nodes
        ]
    return metrics


class RunLevel:
    """Run-level artifacts shared by every frame: pseudo labels and SLADE refreshes."""

    def __init__(self, context):
        run_dir = context.run_dir
        self.generations = []
        self.refreshes = []
        self.center_refreshes = []
        self.incomplete = []
        if not context.node_rows:
            return
        for name, loader, target in (
            ("pseudoLabels", exporter.load_pseudo_label_generations, "generations"),
            ("clusters", exporter.load_saved_clusters, "refreshes"),
            ("centerSimilarity", exporter.load_saved_center_similarity, "center_refreshes"),
        ):
            try:
                setattr(self, target, loader(run_dir))
            except Exception:  # noqa: BLE001 - a file being written; the next poll reads it
                self.incomplete.append(name)


def frame(context, run_level, epoch, directory):
    """One epoch's frame, or ``{"epoch", "pending"}`` while its 2-D layout is incomplete."""

    epoch = int(epoch)
    path = layout_csv(directory, context.projection)
    if path is None:
        return {"epoch": epoch, "pending": "no layout yet"}
    coordinates, _ = read_layout(path, context.projection, ("x", "y"), context.count, context.labels)
    if coordinates is None:
        return {"epoch": epoch, "pending": "layout still being written"}

    incomplete = []
    result = {"epoch": epoch, "layout": exporter.pack(coordinates, "<f4"), "layout3d": None}
    layout3_path = directory / f"embeddings_{context.projection}_3d.csv"
    if layout3_path.is_file():
        coordinates3, _ = read_layout(layout3_path, context.projection, ("x", "y", "z"), context.count, context.labels)
        if coordinates3 is None:
            incomplete.append("layout3d")
        else:
            result["layout3d"] = exporter.pack(coordinates3, "<f4")

    has_nodes = bool(context.node_rows)
    result["nodeMetrics"] = None
    nodes_path = directory / "graph_nodes.csv"
    if has_nodes and nodes_path.is_file():
        try:
            nodes = read_csv_rows(nodes_path)
            if len(nodes) != context.count or [node.get("label") for node in nodes] != context.labels:
                raise ValueError("graph_nodes.csv does not line up with the tracked rows yet")
            result["nodeMetrics"] = node_metrics(nodes, context.position_to_view)
        except Exception:  # noqa: BLE001
            incomplete.append("nodeMetrics")

    def optional(name, file_name, encode):
        target = directory / file_name
        if not has_nodes or not target.is_file():
            result[name] = None
            return
        try:
            result[name] = encode(target)
        except Exception:  # noqa: BLE001 - a partially written npz
            result[name] = None
            incomplete.append(name)

    optional("edges", "graph_edges.npz", lambda target: exporter.encode_graph_edges(target, context.position_to_view, context.count))
    optional(
        "evidence",
        "class_evidence.npz",
        lambda target: exporter.encode_class_evidence(
            target, context.position_to_view, context.count, context.class_index_of_label, len(context.class_names)
        ),
    )
    optional(
        "propagation",
        "label_propagation.npz",
        lambda target: exporter.encode_propagation(
            target, context.position_to_view, context.count, context.class_index_of_label, len(context.class_names)
        ),
    )

    result["pseudoLabels"] = None
    active = [arrays for generation_epoch, arrays in run_level.generations if generation_epoch <= epoch]
    if active:
        result["pseudoLabels"] = exporter.encode_pseudo_labels(
            active[-1], context.samples, context.count, context.class_index_of_label, len(context.class_names)
        )

    result["clusters"] = result["clusterK"] = result["clusterRefreshEpoch"] = None
    active = [refresh for refresh in run_level.refreshes if refresh[0] <= epoch]
    if active:
        refresh_epoch, by_position, cluster_k = active[-1]
        result["clusters"] = [by_position.get(position, -1) for position in context.samples]
        result["clusterK"] = cluster_k
        result["clusterRefreshEpoch"] = refresh_epoch

    result["centerSimilarity"] = None
    active = [similarity for refresh_epoch, similarity in run_level.center_refreshes if refresh_epoch <= epoch]
    if active:
        values = np.asarray([active[-1].get(position, np.nan) for position in context.samples], dtype=np.float64)
        # A byte over [-1, 1]; 0 marks a row with no cluster.
        encoded = np.where(np.isfinite(values), np.round((np.clip(values, -1.0, 1.0) + 1.0) * 127.0) + 1, 0)
        result["centerSimilarity"] = exporter.pack(encoded, "<u1")

    result["minedPairs"] = None
    pairs_path = context.run_dir / "slade_pairs" / f"epoch_{epoch:04d}.npz"
    if has_nodes and epoch >= 0 and pairs_path.is_file():
        try:
            result["minedPairs"] = exporter.encode_mined_pairs(pairs_path, context.samples)
        except Exception:  # noqa: BLE001
            incomplete.append("minedPairs")

    if incomplete:
        result["incomplete"] = incomplete
    return result


def build_frames(tracking_dir, epochs, extra_labels=()):
    context = TrackingContext(tracking_dir)
    context.seed_extra_labels(extra_labels)
    run_level = RunLevel(context)
    directories = dict(context.epochs)
    frames = []
    for epoch in epochs:
        directory = directories.get(int(epoch))
        if directory is None:
            frames.append({"epoch": int(epoch), "pending": "no such epoch directory"})
            continue
        frames.append(frame(context, run_level, epoch, directory))
    return {
        "version": FORMAT_VERSION,
        "baseKey": context.base_key,
        "extraLabels": context.extra_labels,
        "extraClassNames": context.class_names[context.base_class_count :],
        "pseudoLabelGenerations": len(run_level.generations),
        "clusterRefreshes": len(run_level.refreshes),
        "incomplete": run_level.incomplete,
        "frames": frames,
    }


def build_base(tracking_dir):
    return TrackingContext(tracking_dir).base()


def parse_int_list(value):
    return [int(part) for part in str(value or "").split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("base", "frames"))
    parser.add_argument("tracking_dir")
    parser.add_argument("--epochs", default="", help="Comma-separated epochs, e.g. -1,0,5")
    parser.add_argument("--extra-labels", default="", help="Labels already appended to the class list, in order")
    options = parser.parse_args(argv)
    tracking_dir = Path(options.tracking_dir)
    try:
        if options.command == "base":
            document = build_base(tracking_dir)
        else:
            document = build_frames(tracking_dir, parse_int_list(options.epochs), parse_int_list(options.extra_labels))
    except LookupError as exc:
        document = {"pending": str(exc)}
    json.dump(document, sys.stdout, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
