"""Export tracked runs into the Trial Atlas app as a static snapshot.

The app builds statically (``next build`` then ``prepare-sites-dist.mjs``), so a
view cannot read the run tree at request time -- everything it shows has to be a
file under ``public/``. This writes that file, and copies the figures and
filmstrips next to it so the page can show real pictures rather than links into
a directory the browser cannot reach.

Discovery, run identity and the metric grouping are not reimplemented here. They
live in :mod:`evaluate.common` and are shared with the CSV and HTML readers, so
the app and the thesis tables can never disagree about what a run is called or
what its baseline was.

Series are pre-aggregated over folds (mean with a min/max envelope) because the
browser has no reason to carry per-fold rows it would only average again.

Examples::

    python evaluate/export_react.py --root logs/demo_epoch_tracking
    python evaluate/export_react.py --root final/phase_1 --app react_trial_visualizer
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import common, figures, filmstrip
from evaluate.collect import collect

ASSET_DIR_NAME = "epoch-tracking"
SNAPSHOT_NAME = "epoch-tracking.json"

FIGURE_CAPTIONS = {
    "trajectory": "Structure against score. Hollow marker is epoch -1; the arrow is the direction of training.",
    "delta_baseline": "How far each run moved each metric from its epoch -1 baseline.",
}


def _series(group, metric: str) -> list[list[float]]:
    """[epoch, mean, min, max] per tracked epoch, or [] when untracked."""

    usable = group.dropna(subset=[metric])
    if usable.empty:
        return []
    stats = usable.groupby("epoch")[metric].agg(["mean", "min", "max"]).sort_index()
    return [
        [int(epoch), float(row["mean"]), float(row["min"]), float(row["max"])]
        for epoch, row in stats.iterrows()
    ]


def _copy_assets(paths, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in paths:
        target = destination / path.name
        shutil.copyfile(path, target)
        copied[path.stem] = f"{ASSET_DIR_NAME}/{path.name}"
    return copied


def export(root: Path, app_dir: Path, out: Path | None = None) -> Path:
    work_dir = out or (app_dir / ".epoch-tracking-build")
    collected = collect(root, work_dir)
    figure_paths = figures.write_figures(collected, work_dir)
    # Unlike the standalone report's compact default, the interactive app lets
    # the reader choose a run and should retain every fold within that run.
    strip_paths = filmstrip.write_filmstrips(collected, work_dir, all_folds=True)

    public_dir = app_dir / "public"
    assets = _copy_assets(figure_paths + strip_paths, public_dir / ASSET_DIR_NAME)

    epochs = collected["epochs"]
    summary = collected["summary"]
    metrics = collected["metrics"]

    runs = []
    for run_id in epochs["run_id"].unique():
        group = epochs[epochs["run_id"] == run_id]
        first = group.iloc[0]
        run_records = [
            record for record in collected["records"] if record.group_id == run_id
        ]
        run_summary = []
        if not summary.empty:
            for _, row in summary[summary["run_id"] == run_id].iterrows():
                run_summary.append(
                    {
                        "metric": row["metric"],
                        "label": row["metric_label"],
                        "baseline": None if row["baseline"] != row["baseline"] else float(row["baseline"]),
                        "final": None if row["final"] != row["final"] else float(row["final"]),
                        "delta": None if row["delta"] != row["delta"] else float(row["delta"]),
                        "folds": int(row["folds"]),
                        "baselineEpoch": int(row["baseline_epoch"]),
                    }
                )
        run_filmstrips = []
        for record in run_records:
            fold = -1 if record.fold is None else int(record.fold)
            suffix = "" if fold == -1 else f"_fold{fold:02d}"
            entry = {"fold": fold}
            for kind in ("graph", "embeddings"):
                asset = assets.get(f"{run_id}{suffix}_{kind}")
                if asset is not None:
                    entry[kind] = asset
            if len(entry) > 1:
                run_filmstrips.append(entry)

        runs.append(
            {
                "id": run_id,
                "label": first["run"],
                "dataset": first.get("dataset"),
                "method": first.get("method"),
                "loss": first.get("loss"),
                "seed": None if first.get("seed") is None else int(first["seed"]),
                "variant": first.get("variant"),
                "layout": run_records[0].identity.get("track_epoch_layout"),
                "folds": sorted(int(f) for f in group["fold"].unique()),
                "epochs": sorted(int(e) for e in group["epoch"].unique()),
                "series": {metric: _series(group, metric) for metric in metrics
                           if metric in group.columns and group[metric].notna().any()},
                "summary": run_summary,
                "filmstrips": sorted(run_filmstrips, key=lambda item: item["fold"]),
            }
        )

    snapshot = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "runCount": len(runs),
        # Count only folds with metric rows, matching the runs actually present
        # in the snapshot. A just-created run can have a header-only CSV while
        # its baseline bundle is still being generated.
        "foldCount": sum(len(run["folds"]) for run in runs),
        "metrics": metrics,
        "metricLabels": {m: common.METRIC_LABELS.get(m, m) for m in metrics},
        "metricGroups": {
            group: [m for m in names if m in metrics]
            for group, names in common.METRIC_GROUPS.items()
        },
        "higherIsBetter": {m: common.HIGHER_IS_BETTER[m] for m in metrics if m in common.HIGHER_IS_BETTER},
        "baselineEpoch": common.BASELINE_EPOCH,
        "overviewFigures": [
            {"name": stem, "file": asset, "caption": FIGURE_CAPTIONS.get(stem, stem)}
            for stem, asset in sorted(assets.items())
            if stem in FIGURE_CAPTIONS
        ],
        "runs": runs,
    }

    snapshot_path = public_dir / SNAPSHOT_NAME
    snapshot_path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="directory to search for tracked runs")
    parser.add_argument(
        "--app",
        type=Path,
        default=PROJECT_ROOT / "react_trial_visualizer",
        help="the Next.js app directory whose public/ receives the snapshot",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="where the intermediate CSVs and PNGs are built (default: inside the app dir)",
    )
    args = parser.parse_args()

    path = export(args.root, args.app, args.work)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")
    print(f"  {snapshot['runCount']} run(s), {snapshot['foldCount']} fold(s)")
    print(f"  metrics: {', '.join(snapshot['metrics']) or 'none'}")


if __name__ == "__main__":
    main()
