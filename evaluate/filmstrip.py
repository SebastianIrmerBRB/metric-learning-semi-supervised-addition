"""Stitch each run's per-epoch pictures into one strip you can scan in a second.

The tracking output holds one graph and one embedding projection per tracked
epoch, in their own directories. Runs before 2026-09-26 saved them as PNGs;
later runs save only the data (the 2-D layout CSV and the sampled graph edges),
and the pictures are drawn here, in memory, when a strip needs them. Either way
the problem is that they are in different places: to see a representation unfold
you would open eight pictures in order and hold them in your head.

This lays them out left to right in epoch order with the epoch stamped on each
frame, so the baseline sits at the far left and the trained model at the far
right. One strip per run, per picture kind.

Only the first fold of a cross-validation run is used by default: the point is
to watch one model move, not to average pictures, which cannot be averaged.

Examples::

    python evaluate/filmstrip.py --root logs/phase_1 --out evaluate_out
    python evaluate/filmstrip.py --root logs/phase_1 --out evaluate_out --all-folds
    python evaluate/filmstrip.py --render path/to/epoch_tracking/epoch_0045
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import common

# The stems training/epoch_tracking.py wrote before it stopped saving pictures,
# by the layout that produced them.
PICTURE_STEMS = {
    "graph": ("graph.png",),
    "embeddings": ("embeddings_tsne.png", "embeddings_pacmap.png"),
}
# The layouts a picture is drawn from when its PNG was never saved, preferred
# first. A 3-D layout contributes its first two axes; "pca" tracking only has one.
LAYOUT_CSV_STEMS = (
    "embeddings_tsne.csv",
    "embeddings_pacmap.csv",
    "embeddings_tsne_3d.csv",
    "embeddings_pacmap_3d.csv",
    "embeddings_pca_3d.csv",
)
# The drawn edges are graph_sampled_edges.csv's, already capped by the run's
# graph_diagnostics_max_edges; this is the color graph.png gave them.
GRAPH_EDGE_COLOR = "#8a8f98"

FRAME_HEIGHT = 360
CAPTION_HEIGHT = 26
FRAME_GAP = 8
BACKGROUND = (255, 255, 255)
CAPTION_COLOR = (55, 65, 81)


def _layout_csv(directory: Path) -> Path | None:
    return next((directory / stem for stem in LAYOUT_CSV_STEMS if (directory / stem).is_file()), None)


def _epoch_caption(directory: Path) -> str:
    token = directory.name[len("epoch_"):] if directory.name.startswith("epoch_") else directory.name
    epoch = -int(token[1:]) if token.startswith("m") and token[1:].isdigit() else token.lstrip("0") or "0"
    return f"{directory.parent.parent.name} - epoch {epoch}"


def render_embeddings_png(directory: Path, target) -> bool:
    """Draw an epoch's embedding projection from its layout CSV into ``target``."""

    import utils

    layout_csv = _layout_csv(directory)
    if layout_csv is None:
        return False
    utils.plot_projection_csv(layout_csv, target, title=f"{_epoch_caption(directory)} - embeddings")
    return True


def render_graph_png(directory: Path, target) -> bool:
    """Draw an epoch's sampled graph over its embedding layout into ``target``.

    The saved graph.png of older runs projected its node sample with a layout of
    its own; this draws the same sampled edges over the epoch's embedding layout,
    so the graph and embedding frames of one epoch share their coordinates.
    """

    layout_csv = _layout_csv(directory)
    edges_csv = directory / "graph_sampled_edges.csv"
    if layout_csv is None or not edges_csv.is_file():
        return False

    import utils
    from matplotlib.collections import LineCollection
    from training.ssl.graph_diagnostics import scatter_graph_nodes

    coordinates = utils.read_projection_csv(layout_csv)["coordinates"]
    with edges_csv.open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    if not edges:
        return False
    # graph_node indexes the tracked rows, which are the layout CSV's rows.
    labels = {}
    kinds = {}
    for edge in edges:
        for side in ("source", "target"):
            node = int(edge[f"{side}_graph_node"])
            labels[node] = int(edge[f"{side}_label"]) if edge[f"{side}_label"] != "" else 0
            kinds[node] = edge[f"{side}_kind"]
    nodes = np.asarray(sorted(labels), dtype=np.int64)
    if nodes.max() >= len(coordinates):
        raise ValueError(f"{edges_csv} does not line up with {layout_csv.name}")
    segments = np.stack(
        [
            coordinates[[int(edge["source_graph_node"]) for edge in edges]],
            coordinates[[int(edge["target_graph_node"]) for edge in edges]],
        ],
        axis=1,
    )
    known_mask = None
    if all(kinds[node] for node in nodes):
        known_mask = np.asarray([kinds[node] == "labeled" for node in nodes], dtype=bool)

    with utils.PYPLOT_LOCK:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.add_collection(LineCollection(segments, colors=GRAPH_EDGE_COLOR, linewidths=0.45, alpha=0.3, zorder=1))
        scatter_graph_nodes(
            ax,
            coordinates[nodes],
            np.asarray([labels[node] for node in nodes], dtype=np.int64),
            known_mask,
        )
        ax.set_title(
            f"{_epoch_caption(directory)} - graph\n"
            f"{len(nodes)} sampled nodes, {len(edges)} sampled edges",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)
        ax.legend(loc="best", fontsize=9)
        ax.autoscale_view()
        fig.tight_layout()
        fig.savefig(target, dpi=160, format="png")
        plt.close(fig)
    return True


RENDERERS = {"graph": render_graph_png, "embeddings": render_embeddings_png}


def _frames(record: common.RunRecord, kind: str) -> list[tuple[int, Path]]:
    frames = []
    for epoch, directory in common.epoch_directories(record):
        for stem in PICTURE_STEMS[kind]:
            candidate = directory / stem
            if candidate.is_file():
                frames.append((epoch, candidate))
                break
        else:
            # No saved picture: the directory itself stands for the frame and
            # is drawn from its data when the strip is built.
            if kind == "embeddings" and _layout_csv(directory) is not None:
                frames.append((epoch, directory))
            elif kind == "graph" and (directory / "graph_sampled_edges.csv").is_file() and _layout_csv(directory):
                frames.append((epoch, directory))
    return frames


def _open_frame(source: Path, kind: str):
    from PIL import Image

    if source.is_file():
        return Image.open(source).convert("RGB")
    buffer = io.BytesIO()
    if not RENDERERS[kind](source, buffer):
        raise OSError(f"nothing to draw a {kind} picture from in {source}")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build_filmstrip(record: common.RunRecord, kind: str, path: Path) -> Path | None:
    from PIL import Image, ImageDraw

    frames = _frames(record, kind)
    if len(frames) < 2:
        # A single frame is not a strip; the picture itself is already readable.
        return None

    images = []
    for epoch, source in frames:
        try:
            image = _open_frame(source, kind)
        except (OSError, ValueError):
            continue
        scale = FRAME_HEIGHT / image.height
        images.append((epoch, image.resize((max(1, int(image.width * scale)), FRAME_HEIGHT))))
    if len(images) < 2:
        return None

    width = sum(image.width for _, image in images) + FRAME_GAP * (len(images) - 1)
    strip = Image.new("RGB", (width, FRAME_HEIGHT + CAPTION_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(strip)
    offset = 0
    for epoch, image in images:
        strip.paste(image, (offset, CAPTION_HEIGHT))
        caption = "epoch -1 (baseline)" if epoch == common.BASELINE_EPOCH else f"epoch {epoch}"
        draw.text((offset + 6, 7), caption, fill=CAPTION_COLOR)
        offset += image.width + FRAME_GAP

    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)
    return path


def write_filmstrips(collected: dict, out: Path, all_folds: bool = False) -> list[Path]:
    strips_dir = out / "filmstrips"
    written = []
    seen_runs = set()
    for record in collected["records"]:
        if not all_folds:
            if record.group_id in seen_runs:
                continue
            seen_runs.add(record.group_id)
        suffix = "" if record.fold in (None, -1) or not all_folds else f"_fold{record.fold:02d}"
        for kind in PICTURE_STEMS:
            path = strips_dir / f"{record.group_id}{suffix}_{kind}.png"
            built = build_filmstrip(record, kind, path)
            if built is not None:
                written.append(built)
    return written


def main() -> None:
    from evaluate.collect import collect

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("evaluate_out"))
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="build a strip for every fold instead of only the first of each run",
    )
    parser.add_argument(
        "--render",
        type=Path,
        nargs="+",
        metavar="EPOCH_DIR",
        help=(
            "draw the graph and embedding pictures of these tracked epoch directories "
            "into --out, instead of building strips"
        ),
    )
    args = parser.parse_args()

    if args.render:
        args.out.mkdir(parents=True, exist_ok=True)
        for directory in args.render:
            for kind, render in RENDERERS.items():
                path = args.out / f"{directory.parent.parent.name}_{directory.name}_{kind}.png"
                if render(directory, path):
                    print(f"wrote {path}")
                else:
                    print(f"no {kind} data in {directory}")
        return
    if args.root is None:
        parser.error("--root is required unless --render is given")

    written = write_filmstrips(collect(args.root, args.out), args.out, args.all_folds)
    if not written:
        print("No per-epoch pictures found; run with --track_epoch_artifacts graph embeddings")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
