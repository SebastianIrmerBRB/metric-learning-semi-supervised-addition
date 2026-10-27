#!/usr/bin/env python3
"""Geometry of the learned test embeddings: what did the SSL signal change in the space?

Reads the ``test_embeddings.npz`` every Phase-1 fold kept and measures, per fold
and for the Concat ensemble:

* the singular spectrum of the normalized embeddings, its effective rank, the
  top-1 variance share and Roth et al. (2020)'s spectral decay rho (with and
  without the dominant direction) -- a faster decay means a more compressed
  representation, which Roth et al. link to worse generalization;
* alignment / uniformity (Wang & Isola 2020) and the intra/inter-class cosine
  gap, via the same function epoch tracking uses
  (``training.epoch_tracking.embedding_structure_metrics``);
* intra-class distance, centroid distances and their ratio;
* Roth et al.'s rho / pi measures exactly as their reference code computes them
  (``*_roth``, see ``training.epoch_tracking.roth_rho_spectrum``). Roth measure
  them on the training data; on test classes they are a transfer-side reading.

Differences to the supervised run of the same loss are paired by seed. With
``--backbone-npz`` (written by ``frozen_feature_diagnostics.py``) the frozen
DINOv2 test features are added as the untrained reference point.

These are test classes (unseen during training), so the numbers describe how
the head generalizes, not how well it fits the training classes.

Outputs below ``--out/geometry``: metrics.csv (per fold / concat), paired.csv,
spectra_<dataset>_k<k>.png, metrics_<dataset>.png.

Examples::

    python main_eval/embedding_geometry.py --dataset cars --k 4 16 --losses ms
    python main_eval/embedding_geometry.py --dataset cub --combos sup_ms iscen_ms lrml_ms hoffer_ms \\
        --backbone-npz main_eval_out/frozen/backbone_test_CUB.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

SPECTRUM_POINTS = 128
METRICS = [
    "rho_roth", "rho_roth_all", "pi_intra_roth", "pi_inter_roth", "pi_ratio_roth",
    "embedding_effective_rank", "embedding_top1_variance_share", "spectral_decay", "spectral_decay_wo_first",
    "embedding_alignment", "embedding_uniformity", "embedding_class_separation",
    "intra_class_distance", "inter_centroid_distance", "intra_inter_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_run_args(parser)
    parser.add_argument("--backbone-npz", type=Path, nargs="*", default=[],
                        help="frozen-backbone test features (embeddings, labels) as reference rows")
    parser.add_argument("--no-concat", action="store_true", help="skip the Concat ensemble rows")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def measure(embeddings, labels, seed=0):
    from training.epoch_tracking import embedding_structure_metrics, roth_class_distances, roth_rho_spectrum

    singular = common.singular_spectrum(embeddings)
    row = embedding_structure_metrics(embeddings, labels, seed=seed)
    row["spectral_decay"] = common.spectral_decay(singular)
    row["spectral_decay_wo_first"] = common.spectral_decay(singular, drop_first=True)
    row.update(common.class_geometry(embeddings, labels))
    unit = common.l2_normalize(np.asarray(embeddings, dtype=np.float64))
    if len(unit) >= unit.shape[1]:
        row["rho_roth"] = roth_rho_spectrum(unit, 2)
        row["rho_roth_all"] = roth_rho_spectrum(unit, 1)
    row.update(roth_class_distances(unit, labels))
    power = singular**2
    shares = power / power.sum()
    row["_spectrum"] = shares[:SPECTRUM_POINTS]
    return row


def main():
    args = parse_args()
    out = args.out / "geometry"
    runs = common.runs_from_args(args)
    print(f"{len(runs)} runs")
    rows = []
    for run in runs:
        folds, labels = common.load_run_test_embeddings(run)
        if folds is None:
            print(f"[warn] no test_embeddings.npz for {run.combo} k={run.k} seed={run.seed}", file=sys.stderr)
            continue
        base = {"dataset": run.dataset, "combo": run.combo, "method": run.method, "loss": run.loss,
                "variant": run.variant, "k": run.k, "seed": run.seed}
        for fold, embeddings in enumerate(folds):
            rows.append({**base, "view": f"fold_{fold}", **measure(embeddings, labels, seed=fold)})
        if not args.no_concat:
            rows.append({**base, "view": "concat", **measure(common.concat_embeddings(folds), labels)})
        print(f"  {run.dataset:<18} {run.combo:<24} k={run.k:<3} seed={run.seed}")

    for path in args.backbone_npz:
        data = np.load(path)
        dataset = str(data["dataset"]) if "dataset" in data.files else path.stem
        rows.append({"dataset": dataset, "combo": "backbone", "method": "backbone", "loss": "-", "variant": "",
                     "k": -1, "seed": -1, "view": "backbone", **measure(data["embeddings"], data["labels"])})

    common.write_rows(out / "metrics.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])

    # Fold-averaged value per run, then paired against the supervised run of the same loss.
    per_run = defaultdict(dict)
    for row in rows:
        if row["view"].startswith("fold_"):
            per_run[(row["dataset"], row["combo"], row["k"], row["seed"])].setdefault("folds", []).append(row)
        elif row["view"] == "concat":
            per_run[(row["dataset"], row["combo"], row["k"], row["seed"])]["concat"] = row
    paired = []
    for (dataset, combo, k, seed), views in per_run.items():
        method, loss = views.get("concat", (views.get("folds") or [{}])[0]).get("method"), combo.split("_")[1]
        reference = per_run.get((dataset, f"sup_{loss}", k, seed))
        if method == "sup" or reference is None:
            continue
        for view in ("folds", "concat"):
            mine, theirs = views.get(view), reference.get(view)
            if not mine or not theirs:
                continue
            entry = {"dataset": dataset, "combo": combo, "reference": f"sup_{loss}", "k": k, "seed": seed, "view": view}
            for metric in METRICS:
                a = np.mean([r[metric] for r in mine]) if view == "folds" else mine.get(metric)
                b = np.mean([r[metric] for r in theirs]) if view == "folds" else theirs.get(metric)
                entry[f"d_{metric}"] = (a - b) if a is not None and b is not None else None
            paired.append(entry)
    common.write_rows(out / "paired.csv", paired)

    if not args.no_figures:
        draw(rows, out)
    print(f"wrote {out}")


def draw(rows, out):
    pyplot = common.plt()
    groups = defaultdict(list)
    for row in rows:
        if row["view"] == "fold_0" or row["view"] == "backbone":
            groups[(row["dataset"], row["k"])].append(row)
    backbone = {r["dataset"]: r for r in rows if r["view"] == "backbone"}
    for (dataset, k), items in groups.items():
        if k == -1:
            continue
        fig, ax = pyplot.subplots(figsize=(5, 3.4))
        by_combo = defaultdict(list)
        for row in items:
            by_combo[row["combo"]].append(row["_spectrum"])
        for i, (combo, spectra) in enumerate(sorted(by_combo.items())):
            length = min(len(s) for s in spectra)
            ax.plot(np.arange(1, length + 1), np.mean([s[:length] for s in spectra], axis=0), label=combo, lw=1.2, color=f"C{i}")
        for name, row in backbone.items():
            if name == dataset or name == common.DATASET_SHORT.get(dataset):
                ax.plot(np.arange(1, len(row["_spectrum"]) + 1), row["_spectrum"], "k--", lw=1, label="backbone")
        ax.set_yscale("log")
        ax.set_xlabel("singular direction")
        ax.set_ylabel("variance share")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"{dataset}, k={k}: test spectrum (fold 0, mean over seeds)", fontsize=8)
        common.savefig(fig, out / f"spectra_{common.DATASET_SHORT.get(dataset, dataset)}_k{k}.png")

    for dataset in sorted({r["dataset"] for r in rows if r["view"] != "backbone"}):
        items = [r for r in rows if r["dataset"] == dataset and r["view"].startswith("fold_")]
        combos = sorted({r["combo"] for r in items})
        ks = sorted({r["k"] for r in items})
        fig, axes = pyplot.subplots(2, 5, figsize=(17, 6.4))
        fig.subplots_adjust(hspace=0.9, wspace=0.35)
        for ax, metric in zip(axes.flat, METRICS):
            for j, k in enumerate(ks):
                means = []
                for combo in combos:
                    values = [r[metric] for r in items if r["combo"] == combo and r["k"] == k]
                    means.append(np.mean(values) if values else np.nan)
                ax.plot(range(len(combos)), means, "o-", label=f"k={k}", lw=0.8, ms=3)
            if dataset in backbone:
                ax.axhline(backbone[dataset][metric], color="k", ls="--", lw=0.7)
            ax.set_xticks(range(len(combos)), combos, rotation=70, fontsize=6)
            ax.set_title(metric.replace("embedding_", ""), fontsize=8)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f"{dataset}: test-embedding geometry (mean over folds and seeds; dashed = backbone)", fontsize=9)
        common.savefig(fig, out / f"metrics_{common.DATASET_SHORT.get(dataset, dataset)}.png")


if __name__ == "__main__":
    main()
