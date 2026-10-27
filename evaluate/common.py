"""Finding tracked runs and reading them into one tidy frame.

One fold's tracking output is a directory. A comparison is forty of them, half
belonging to the same cross-validation run and half to different methods, and
the difference matters: folds of one run are repeats to be averaged, while
different methods are the rows of the table. So discovery keys on the ``cv_*``
ancestor a fold sits under, not on the leaf directory.

Run identity comes from ``run_config.json`` rather than from the directory name.
Directory names are built from a subset of the settings and collide across
datasets -- an In-Shop study and a Cars study can produce the same leaf name --
whereas the recorded ``args`` and ``ssl_config`` say exactly what ran.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRACKING_DIR_NAME = "epoch_tracking"
EPOCH_METRICS_NAME = "epoch_metrics.csv"
RUN_CONFIG_NAME = "run_config.json"
METRICS_NAME = "metrics.csv"
# The split name training/engine.py logs the off-selection test curve under.
TRACKED_TEST_SPLIT = "tracked_test"
BASELINE_EPOCH = -1

# Metrics are grouped because they answer different questions and belong on
# different axes: a purity is a fraction, a degree is a count, and a uniformity
# is a negative log. Plotting them on one axis would hide all but the largest.
METRIC_GROUPS = {
    "test": [
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
    ],
    "embedding": [
        "embedding_effective_rank",
        "embedding_class_separation",
        "embedding_alignment",
        "embedding_uniformity",
        "embedding_top1_variance_share",
        "embedding_norm_mean",
    ],
    "graph": [
        "graph_edge_purity",
        "graph_weighted_edge_purity",
        "graph_mean_degree",
        "graph_component_count",
        "graph_isolated_node_count",
        "graph_nodes_without_labeled_component",
    ],
}

METRIC_LABELS = {
    "test_precision_at_1": "Test P@1",
    "test_mean_average_precision_at_r": "Test MAP@R",
    "embedding_effective_rank": "Effective rank",
    "embedding_class_separation": "Class separation (cos)",
    "embedding_alignment": "Alignment (lower = tighter)",
    "embedding_uniformity": "Uniformity",
    "embedding_top1_variance_share": "Top-1 variance share",
    "embedding_norm_mean": "Mean embedding norm",
    "graph_edge_purity": "Graph edge purity",
    "graph_weighted_edge_purity": "Weighted edge purity",
    "graph_mean_degree": "Mean degree",
    "graph_component_count": "Components",
    "graph_isolated_node_count": "Isolated nodes",
    "graph_nodes_without_labeled_component": "Nodes w/o labeled component",
}

# Only used to orient a delta arrow. Metrics with no natural direction -- a
# degree, a norm, a component count -- are deliberately absent and are reported
# as plain numbers.
HIGHER_IS_BETTER = {
    "test_precision_at_1": True,
    "test_mean_average_precision_at_r": True,
    "embedding_effective_rank": True,
    "embedding_class_separation": True,
    "embedding_alignment": False,
    "graph_edge_purity": True,
    "graph_weighted_edge_purity": True,
}

ALL_METRICS = [metric for group in METRIC_GROUPS.values() for metric in group]

_FOLD_PATTERN = re.compile(r"^fold_(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    """One fold's tracking directory, with the run it belongs to."""

    tracking_dir: Path
    run_dir: Path
    group_dir: Path
    group_id: str
    label: str
    fold: int | None
    identity: dict

    @property
    def epoch_metrics_path(self) -> Path:
        return self.tracking_dir / EPOCH_METRICS_NAME


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _fold_of(run_dir: Path) -> tuple[int | None, Path | None]:
    """Return this leaf's fold index and the cross-validation directory above it."""

    for parent in run_dir.parents:
        match = _FOLD_PATTERN.match(parent.name)
        if match:
            return int(match.group(1)), parent.parent
    return None, None


def read_identity(run_dir: Path) -> dict:
    """Flatten the settings a reader needs out of ``run_config.json``."""

    config = _read_json(run_dir / RUN_CONFIG_NAME)
    args = config.get("args", {}) or {}
    ssl_config = config.get("ssl_config", {}) or {}
    method_params = ssl_config.get("method_params", {}) or {}
    ablation = args.get("ablation_config_resolved") or {}
    regularizer = method_params.get("regularizer")
    method = ssl_config.get("method") or "supervised"
    if method == "regularized" and regularizer:
        method = str(regularizer)
    elif method == "none":
        method = "supervised"
    return {
        "dataset": args.get("dataset"),
        "loss": args.get("loss"),
        "miner": args.get("miner"),
        "method": method,
        "seed": args.get("seed"),
        "epochs": args.get("epochs"),
        "cv_k": args.get("cv_k"),
        "warmup_epochs": ssl_config.get("warmup_epochs"),
        "label_sampling_mode": ssl_config.get("label_sampling_mode"),
        "labeled_per_class": ssl_config.get("labeled_per_class"),
        "labeled_fraction": ssl_config.get("labeled_fraction"),
        "ablation": ablation.get("name"),
        "variant": ablation.get("variant"),
        "track_interval": args.get("track_epoch_interval"),
        "track_artifacts": args.get("track_epoch_artifacts"),
        "track_epoch_layout": args.get("track_epoch_layout"),
        "save_dir": args.get("save_dir"),
    }


def make_label(identity: dict, fallback: str) -> str:
    """A short human name: what method, on what loss, under what budget."""

    parts = []
    if identity.get("method"):
        parts.append(str(identity["method"]))
    loss = identity.get("loss")
    if loss:
        parts.append(str(loss).replace("Loss", ""))
    per_class = identity.get("labeled_per_class")
    if per_class:
        parts.append(f"k{per_class}")
    elif identity.get("labeled_fraction") not in (None, 1.0):
        parts.append(f"f{identity['labeled_fraction']}")
    if identity.get("variant"):
        parts.append(str(identity["variant"]))
    if identity.get("seed") is not None:
        parts.append(f"s{identity['seed']}")
    return " · ".join(parts) if parts else fallback


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return slug or "run"


def discover(root: Path) -> list[RunRecord]:
    """Find tracked folds under ``root``, keeping the newest attempt per fold."""

    root = Path(root)
    records: list[RunRecord] = []
    for metrics_path in sorted(root.rglob(f"{TRACKING_DIR_NAME}/{EPOCH_METRICS_NAME}")):
        tracking_dir = metrics_path.parent
        run_dir = tracking_dir.parent
        fold, cv_dir = _fold_of(run_dir)
        group_dir = cv_dir if cv_dir is not None else run_dir
        identity = read_identity(run_dir)
        try:
            relative = group_dir.relative_to(root)
        except ValueError:
            relative = group_dir
        records.append(
            RunRecord(
                tracking_dir=tracking_dir,
                run_dir=run_dir,
                group_dir=group_dir,
                group_id=slugify(str(relative)),
                label=make_label(identity, fallback=group_dir.name),
                fold=fold,
                identity=identity,
            )
        )
    # A resumed/retried CV fold can leave more than one timestamped run
    # directory beneath the same ``fold_XX`` directory. The sorted path walk
    # puts the newest timestamp last, so assignment retains that attempt and
    # prevents its epochs and filmstrip from being counted twice.
    newest_by_fold = {}
    for record in records:
        newest_by_fold[(record.group_id, record.fold)] = record
    return list(newest_by_fold.values())


def load_epochs(records: list[RunRecord]) -> pd.DataFrame:
    """One tidy row per run, fold and epoch, with whatever columns were tracked."""

    frames = []
    for record in records:
        try:
            frame = pd.read_csv(record.epoch_metrics_path)
        except (OSError, ValueError):
            continue
        if frame.empty:
            continue
        frame.insert(0, "run", record.label)
        frame.insert(1, "run_id", record.group_id)
        frame.insert(2, "fold", -1 if record.fold is None else record.fold)
        for key in ("dataset", "method", "loss", "seed", "variant"):
            frame[key] = record.identity.get(key)
        frame["run_dir"] = str(record.run_dir)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["run_id", "fold", "epoch"], ignore_index=True
    )


def load_tracked_test(records: list[RunRecord]) -> pd.DataFrame:
    """The wide D_test table the engine logged under the tracked_test split.

    ``epoch_metrics.csv`` carries only the two headline numbers; the full table
    -- every requested measurement and the whole Recall@K ladder -- is in the
    run's own ``metrics.csv``, which is where this reads it from.
    """

    frames = []
    for record in records:
        path = record.run_dir / METRICS_NAME
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError):
            continue
        if "split" not in frame.columns:
            continue
        tracked = frame[frame["split"] == TRACKED_TEST_SPLIT].copy()
        if tracked.empty:
            continue
        tracked = tracked.drop(columns=[c for c in ("split", "loss") if c in tracked])
        tracked = tracked.dropna(axis=1, how="all")
        tracked.insert(0, "run", record.label)
        tracked.insert(1, "run_id", record.group_id)
        tracked.insert(2, "fold", -1 if record.fold is None else record.fold)
        frames.append(tracked)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["run_id", "fold", "epoch"], ignore_index=True
    )


def runs_index(records: list[RunRecord]) -> pd.DataFrame:
    """One row per discovered fold: who it is and where it lives."""

    rows = []
    for record in records:
        row = {
            "run_id": record.group_id,
            "run": record.label,
            "fold": -1 if record.fold is None else record.fold,
            **record.identity,
            "run_dir": str(record.run_dir),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def available_metrics(frame: pd.DataFrame) -> list[str]:
    """The tracked metrics actually present, in documented order."""

    if frame.empty:
        return []
    return [
        metric
        for metric in ALL_METRICS
        if metric in frame.columns and frame[metric].notna().any()
    ]


def epoch_directories(record: RunRecord) -> list[tuple[int, Path]]:
    """This fold's per-epoch directories, in epoch order, baseline first."""

    found = []
    for path in record.tracking_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("epoch_"):
            continue
        token = path.name[len("epoch_"):]
        try:
            epoch = -int(token[1:]) if token.startswith("m") else int(token)
        except ValueError:
            continue
        found.append((epoch, path))
    return sorted(found)


def fold_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Baseline, final and delta per run, averaged over that run's folds.

    The baseline is epoch -1 when it was tracked; a run resumed without one
    falls back to its first tracked epoch, which is reported in ``baseline_epoch``
    so a reader is never silently comparing against a different starting point.
    """

    rows = []
    for (run_id, run), group in frame.groupby(["run_id", "run"], sort=False):
        for metric in metrics:
            if metric not in group.columns or group[metric].isna().all():
                continue
            per_fold = []
            for _, fold_frame in group.groupby("fold", sort=True):
                values = fold_frame.dropna(subset=[metric]).sort_values("epoch")
                if values.empty:
                    continue
                per_fold.append(
                    (
                        int(values["epoch"].iloc[0]),
                        float(values[metric].iloc[0]),
                        float(values[metric].iloc[-1]),
                        float(values[metric].max()),
                        float(values[metric].min()),
                        int(values["epoch"].iloc[-1]),
                    )
                )
            if not per_fold:
                continue
            baseline_epochs = [item[0] for item in per_fold]
            final_epochs = [item[5] for item in per_fold]
            baselines = pd.Series([item[1] for item in per_fold])
            finals = pd.Series([item[2] for item in per_fold])
            rows.append(
                {
                    "run_id": run_id,
                    "run": run,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "folds": len(per_fold),
                    "baseline_epoch": max(set(baseline_epochs), key=baseline_epochs.count),
                    "baseline": baselines.mean(),
                    "baseline_std": baselines.std(ddof=0),
                    # A run killed mid-fold leaves its folds at different last
                    # epochs, and averaging those means comparing one fold's
                    # epoch 40 with another's epoch 0. The spread is carried so a
                    # reader is told rather than silently handed the mean.
                    "final_epoch": max(set(final_epochs), key=final_epochs.count),
                    "final_epoch_spread": max(final_epochs) - min(final_epochs),
                    "final": finals.mean(),
                    "final_std": finals.std(ddof=0),
                    "delta": finals.mean() - baselines.mean(),
                    "max": pd.Series([item[3] for item in per_fold]).mean(),
                    "min": pd.Series([item[4] for item in per_fold]).mean(),
                }
            )
    return pd.DataFrame(rows)
