"""Draw the tracked curves.

Four kinds of picture, each answering a question the table cannot:

``curve_<metric>.png``
    The metric against the epoch, one line per run, with a band over that run's
    folds. The baseline is drawn at epoch -1 behind a dashed rule, so "did
    training move this at all" is the first thing the eye lands on.
``overview_<group>.png``
    Every metric in one group as small multiples on a shared epoch axis, for
    reading a whole family at once.
``trajectory.png``
    Structure against score: effective rank (or graph purity) on x, test P@1 on
    y, with an arrow per run from its baseline to its final epoch. A method that
    gains score while collapsing its representation looks completely different
    here than one that gains score by spreading it out, and on a line chart the
    two are indistinguishable.
``delta_baseline.png``
    How far each run moved each metric from epoch -1, as a bar chart. The
    summary table in one glance.

Examples::

    python evaluate/figures.py --root logs/phase_1 --out evaluate_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import common

FIGURE_DPI = 140
# A qualitative cycle that stays distinguishable in grayscale print, which a
# thesis figure eventually becomes.
PALETTE = (
    "#3b6ea5", "#c1442e", "#2f7d4f", "#8a5fb0",
    "#d08a20", "#2a9d9c", "#a03a63", "#6b7280",
)


def _runs_in_order(frame: pd.DataFrame) -> list[tuple[str, str]]:
    seen = {}
    for run_id, run in zip(frame["run_id"], frame["run"]):
        seen.setdefault(run_id, run)
    return list(seen.items())


def _fold_band(group: pd.DataFrame, metric: str):
    """Mean over folds with a min/max envelope, on the epochs actually tracked."""

    usable = group.dropna(subset=[metric])
    if usable.empty:
        return None
    stats = usable.groupby("epoch")[metric].agg(["mean", "min", "max", "count"])
    return stats.sort_index()


def plot_metric(epochs: pd.DataFrame, metric: str, path: Path) -> bool:
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    drawn = False
    for index, (run_id, run) in enumerate(_runs_in_order(epochs)):
        group = epochs[epochs["run_id"] == run_id]
        stats = _fold_band(group, metric)
        if stats is None:
            continue
        color = PALETTE[index % len(PALETTE)]
        axis.plot(stats.index, stats["mean"], marker="o", ms=3.5, lw=1.7, color=color, label=run)
        if (stats["count"] > 1).any():
            axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.15, lw=0)
        drawn = True
    if not drawn:
        plt.close(figure)
        return False

    if epochs["epoch"].min() <= common.BASELINE_EPOCH:
        # Everything left of this line is the untrained model.
        axis.axvline(-0.5, color="#9aa0a6", ls="--", lw=1.0)
        axis.text(
            -0.45, 0.02, "baseline", rotation=90, va="bottom", ha="left",
            fontsize=8, color="#6b7280", transform=axis.get_xaxis_transform(),
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(common.METRIC_LABELS.get(metric, metric))
    axis.set_title(common.METRIC_LABELS.get(metric, metric))
    axis.grid(alpha=0.25, lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, frameon=False, ncols=1 if len(_runs_in_order(epochs)) < 5 else 2)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return True


def plot_group_overview(epochs: pd.DataFrame, group: str, metrics: list[str], path: Path) -> bool:
    present = [m for m in metrics if m in epochs.columns and epochs[m].notna().any()]
    if not present:
        return False
    columns = min(3, len(present))
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows), squeeze=False)
    runs = _runs_in_order(epochs)
    for position, metric in enumerate(present):
        axis = axes[position // columns][position % columns]
        for index, (run_id, run) in enumerate(runs):
            stats = _fold_band(epochs[epochs["run_id"] == run_id], metric)
            if stats is None:
                continue
            color = PALETTE[index % len(PALETTE)]
            axis.plot(stats.index, stats["mean"], marker="o", ms=2.5, lw=1.4, color=color, label=run)
            if (stats["count"] > 1).any():
                axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.13, lw=0)
        axis.set_title(common.METRIC_LABELS.get(metric, metric), fontsize=10)
        axis.grid(alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    for position in range(len(present), rows * columns):
        axes[position // columns][position % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", fontsize=8, frameon=False,
                      ncols=min(4, len(labels)), bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(f"{group.capitalize()} metrics across training", fontsize=12)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_trajectory(epochs: pd.DataFrame, path: Path) -> bool:
    """Structure on x, score on y, one arrow per run from baseline to final."""

    score = "test_precision_at_1"
    for structure in ("embedding_effective_rank", "graph_edge_purity", "embedding_class_separation"):
        if structure in epochs.columns and epochs[structure].notna().any():
            break
    else:
        return False
    if score not in epochs.columns or epochs[score].isna().all():
        return False

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    drawn = False
    for index, (run_id, run) in enumerate(_runs_in_order(epochs)):
        group = epochs[epochs["run_id"] == run_id].dropna(subset=[structure, score])
        if group.empty:
            continue
        averaged = group.groupby("epoch")[[structure, score]].mean().sort_index()
        if len(averaged) < 2:
            continue
        color = PALETTE[index % len(PALETTE)]
        axis.plot(averaged[structure], averaged[score], "-", color=color, alpha=0.55, lw=1.3)
        axis.scatter(averaged[structure], averaged[score], s=18, color=color, alpha=0.75, zorder=3)
        start = averaged.iloc[0]
        end = averaged.iloc[-1]
        axis.annotate(
            "", xy=(end[structure], end[score]), xytext=(start[structure], start[score]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8, "alpha": 0.9},
        )
        axis.scatter([start[structure]], [start[score]], s=70, facecolors="none",
                     edgecolors=color, lw=1.6, zorder=4)
        axis.annotate(run, xy=(end[structure], end[score]), fontsize=8, color=color,
                      xytext=(4, 4), textcoords="offset points")
        drawn = True
    if not drawn:
        plt.close(figure)
        return False
    axis.set_xlabel(common.METRIC_LABELS.get(structure, structure))
    axis.set_ylabel(common.METRIC_LABELS.get(score, score))
    axis.set_title("Where the score came from\n(hollow marker = epoch -1, arrow = direction of training)",
                   fontsize=11)
    axis.grid(alpha=0.25, lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return True


def plot_baseline_delta(summary: pd.DataFrame, path: Path) -> bool:
    if summary.empty:
        return False
    metrics = [m for m in common.ALL_METRICS if m in set(summary["metric"])]
    metrics = [m for m in metrics if summary[summary["metric"] == m]["delta"].notna().any()]
    if not metrics:
        return False
    columns = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 0.5 + 2.6 * rows), squeeze=False)
    for position, metric in enumerate(metrics):
        axis = axes[position // columns][position % columns]
        frame = summary[summary["metric"] == metric]
        labels = list(frame["run"])
        values = list(frame["delta"])
        better = common.HIGHER_IS_BETTER.get(metric)
        colors = []
        for value in values:
            if better is None or pd.isna(value):
                colors.append("#6b7280")
            else:
                improved = value > 0 if better else value < 0
                colors.append("#2f7d4f" if improved else "#c1442e")
        axis.barh(range(len(values)), values, color=colors, height=0.6)
        axis.set_yticks(range(len(values)), labels, fontsize=7)
        axis.axvline(0, color="#374151", lw=0.9)
        axis.set_title(common.METRIC_LABELS.get(metric, metric), fontsize=10)
        axis.grid(axis="x", alpha=0.25, lw=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.invert_yaxis()
    for position in range(len(metrics), rows * columns):
        axes[position // columns][position % columns].axis("off")
    figure.suptitle("Change from the epoch -1 baseline (green = improved)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return True


def write_figures(collected: dict, out: Path) -> list[Path]:
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = collected["epochs"]
    written = []
    if epochs.empty:
        return written

    for metric in collected["metrics"]:
        path = figures_dir / f"curve_{metric}.png"
        if plot_metric(epochs, metric, path):
            written.append(path)

    for group, metrics in common.METRIC_GROUPS.items():
        path = figures_dir / f"overview_{group}.png"
        if plot_group_overview(epochs, group, metrics, path):
            written.append(path)

    trajectory = figures_dir / "trajectory.png"
    if plot_trajectory(epochs, trajectory):
        written.append(trajectory)

    delta = figures_dir / "delta_baseline.png"
    if plot_baseline_delta(collected["summary"], delta):
        written.append(delta)
    return written


def main() -> None:
    from evaluate.collect import collect

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluate_out"))
    args = parser.parse_args()

    for path in write_figures(collect(args.root, args.out), args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
