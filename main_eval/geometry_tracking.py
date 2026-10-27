#!/usr/bin/env python3
"""B6: geometry of the embedding space over training, per split, from epoch tracking's ``geometry``.

Reads every ``epoch_tracking/geometry.jsonl`` below ``--roots`` (runs launched with
``--track-artifacts geometry``, e.g. ``configs/run_manifests/phase2/cars_cir_geometry.json``).
Each file holds, per tracked epoch and per split (whole training pool, its
labeled / unlabeled halves, validation classes, test classes), the geometry of
the head's embeddings, plus one row per split for the frozen backbone features
the head receives:

* Roth et al. (2020)'s measures exactly as their reference code computes them
  (``*_roth``): rho = rho_spectrum@2 (uncentered SVD, top D-1 singular values,
  largest dropped, KL(uniform || spectrum)), rho_spectrum@1 (largest kept), and
  pi_intra / pi_inter / pi_ratio from cosine distances (dists.py). The paper
  measures them on the embedded training data, i.e. the ``train`` split here;
* Roth's class-conditioned spectrum (supp. E, Fig. 11): per class the first 35
  singular values of its uncentered embeddings, averaged over classes;
* our own variants: spectrum of the centered embeddings, effective rank, top-1
  variance share, centroid-based intra/inter distances and their ratio;
* alignment and uniformity (Wang & Isola 2020) and the intra/inter cosine gap.

``correlation.csv`` repeats Roth et al.'s Fig. 5 analysis: the Pearson
correlation (their ``np.corrcoef``; Spearman alongside) between a train-split
measure and the test score, one point per finished run -- MAP@R (the thesis'
primary metric) and Recall@1 (Roth's). Two deliberate deviations:
the measure is read at the validation-selected epoch, not at the epoch of the
best *test* R@1 as in their Result_Evaluations.py, and the points are
(method, k) runs rather than DML objectives. Roth's rho needs at least D
samples; rank-deficient rows (the labeled split at small k) are blanked.

The hypothesis "SSL acts as a regularizer against overfitting to the few labels"
predicts: the supervised head compresses more at small k (larger rho), the SSL
head less, and the gap shrinks as k grows. ``hypothesis.csv`` reports exactly
that gap, supervised minus SSL, at the selected epoch of each fold, per split
and k; positive means the supervised head is more compressed.

Comparisons are made within one split: the tracked samples are fixed per fold,
but splits differ in size, and the spectrum estimate depends on the sample
count (the labeled split has barely more samples than dimensions at small k).

Outputs below ``--out/geometry_tracking``: rows.csv, selected.csv,
hypothesis.csv, figures, and with ``--app`` the Trial Atlas snapshot
``public/geometry.json``.

Examples::

    python main_eval/geometry_tracking.py
    python main_eval/geometry_tracking.py --roots logs/final/phase_2/cars_geometry --app react_trial_visualizer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

ROTH_METRICS = ["rho_roth", "rho_roth_all", "pi_intra_roth", "pi_inter_roth", "pi_ratio_roth"]
METRICS = ROTH_METRICS + [
    "spectral_decay", "spectral_decay_wo_first", "embedding_effective_rank", "embedding_top1_variance_share",
    "intra_class_distance", "inter_centroid_distance", "intra_inter_ratio",
    "embedding_alignment", "embedding_uniformity", "embedding_class_separation",
]
METRIC_LABEL = {
    "rho_roth": "ρ (Roth et al.)", "rho_roth_all": "ρ inkl. größtem Singulärwert (Roth)",
    "pi_intra_roth": "π_intra (Roth)", "pi_inter_roth": "π_inter (Roth)", "pi_ratio_roth": "π_ratio (Roth)",
    "spectral_decay": "ρ zentriert (eigene Variante)", "spectral_decay_wo_first": "ρ zentriert ohne 1. Richtung",
    "embedding_effective_rank": "Effektiver Rang", "embedding_top1_variance_share": "Varianzanteil 1. Richtung",
    "intra_class_distance": "Intra-Klassen-Abstand", "inter_centroid_distance": "Inter-Zentroid-Abstand",
    "intra_inter_ratio": "Intra/Inter-Verhältnis", "embedding_alignment": "Alignment",
    "embedding_uniformity": "Uniformity", "embedding_class_separation": "Cosinus-Klassentrennung",
}
SPLITS = ["train", "labeled", "unlabeled", "validation", "test"]
SPLIT_LABEL = {"train": "Training gesamt (Roth)", "labeled": "Training, gelabelt", "unlabeled": "Training, ungelabelt",
               "validation": "Validierungsklassen", "test": "Testklassen"}
RUN_DIR = re.compile(r"class_subset_k_shot_label_[\d.]+_k_(?P<k>\d+)_comparison_seed_(?P<seed>\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+",
                        default=[common.REPO / "logs" / "final" / "phase_2" / "cars_geometry"])
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    parser.add_argument("--app", type=Path, default=None, help="Trial Atlas folder; writes public/geometry.json")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--stdout", action="store_true",
                        help="print the app snapshot as JSON to stdout (null if nothing is tracked yet), write no files")
    return parser.parse_args()


def selected_epochs(cv_dir: Path) -> dict:
    """fold log_dir name -> selected epoch, from the CV summary when the run finished."""
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.is_file():
        return {}
    out = {}
    for fold in common.read_json(summary_path).get("folds") or []:
        log_dir, epoch = fold.get("log_dir"), fold.get("selected_epoch")
        if log_dir and epoch is not None:
            out[Path(log_dir).name] = int(epoch)
    return out


def load_rows(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).glob("*/*/final/*/cv_*/fold_*/*/epoch_tracking/geometry.jsonl")):
            fold_log = path.parent.parent
            cv_dir = fold_log.parent.parent
            run_dir = cv_dir.parent.parent.parent
            combo_dir = run_dir.parent
            match, combo = RUN_DIR.fullmatch(run_dir.name), common.COMBO.match(combo_dir.name)
            if not match or not combo:
                continue
            selected = selected_epochs(cv_dir).get(fold_log.name)
            base = {"root": Path(root).name, "combo": combo_dir.name, "method": common.COMBO_METHOD[combo["prefix"]],
                    "loss": combo["loss"], "variant": combo["variant"] or "", "k": int(match["k"]),
                    "seed": int(match["seed"]), "fold": fold_log.parent.name, "selected_epoch": selected,
                    **{f"test_{t}": v for t, v in run_test_scores(cv_dir).items()}}
            for line in path.read_text().splitlines():
                if line.strip():
                    row = {**base, **json.loads(line)}
                    if row.get("embedding_count", 0) < row.get("embedding_dim", 0):
                        # Roth's rho is undefined below D samples (their SVD pads zeros).
                        row["rho_roth"] = row["rho_roth_all"] = None
                    rows.append(row)
    return rows


TARGETS = {"map_r": ("mean_average_precision_at_r", "MAP@R"), "r1": ("precision_at_1", "R@1")}


def run_test_scores(cv_dir: Path) -> dict:
    """The run's test MAP@R and R@1 (concatenated folds, else the fold mean); empty until every fold is done."""
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.is_file():
        return {}
    summary = common.read_json(summary_path)
    completed = summary.get("completed_folds")
    completed = len(completed) if isinstance(completed, (list, tuple)) else completed
    if completed is None or int(completed) < int(summary.get("cv_k") or 0):
        return {}
    aggregate = summary.get("aggregate") or {}
    scores = {}
    for target, (key, _) in TARGETS.items():
        value = aggregate.get(f"concatenated_test_{key}")
        if value is None:
            folds = [f.get(f"test_{key}") for f in aggregate.get("fold_results") or []]
            folds = [f for f in folds if f is not None]
            value = float(np.mean(folds)) if folds else None
        scores[target] = value
    return scores


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None


def correlation_points(selected, scores):
    """One point per (root, combo, k, seed, split): fold-mean measures at the selected epoch and the run's test scores."""
    points = defaultdict(list)
    for row in selected:
        if row["representation"] == "head":
            points[(row["root"], row["combo"], row["k"], row["seed"], row["split"])].append(row)
    table = []
    for (root, combo, k, seed, split), items in sorted(points.items()):
        run = scores.get((root, combo, k, seed)) or {}
        if any(run.get(t) is not None for t in TARGETS):
            table.append({"root": root, "combo": combo, "method": items[0]["method"], "k": k, "seed": seed,
                          "split": split, **{t: run.get(t) for t in TARGETS}, **{m: fold_mean(items, m) for m in METRICS}})
    return table


def correlation(table):
    """Roth et al. Fig. 5: per root, split, target and metric, corr(measure, test score) over runs; pooled and per k."""
    out = []
    for root in sorted({t["root"] for t in table}):
        for split in SPLITS:
            rows = [t for t in table if t["root"] == root and t["split"] == split]
            for scope in ["all"] + sorted({t["k"] for t in rows}):
                scoped = [t for t in rows if scope == "all" or t["k"] == scope]
                for target in TARGETS:
                    for metric in METRICS:
                        pairs = [(t[metric], t[target]) for t in scoped if t[metric] is not None and t[target] is not None]
                        if len(pairs) < 3:
                            continue
                        x, y = map(np.asarray, zip(*pairs))
                        out.append({"root": root, "split": split, "k": scope, "target": target, "metric": metric,
                                    "n": len(pairs), "pearson": _pearson(x, y),
                                    "spearman": _pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))})
    return out


def at_selected(rows):
    """Per fold, split and representation: the head row at the last tracked epoch <= selected epoch."""
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[(row["root"], row["combo"], row["k"], row["seed"], row["fold"], row["split"], row["representation"])].append(row)
    out = []
    for key, items in by_fold.items():
        if key[-1] == "backbone":
            out.append({**items[0], "tracked_epoch": None})
            continue
        items = sorted(items, key=lambda r: r["epoch"])
        target = items[0]["selected_epoch"]
        eligible = [r for r in items if target is None or r["epoch"] <= target]
        chosen = (eligible or items)[-1]
        out.append({**chosen, "tracked_epoch": chosen["epoch"]})
    return out


def fold_mean(items, metric):
    values = [r[metric] for r in items if r.get(metric) is not None and np.isfinite(r[metric])]
    return float(np.mean(values)) if values else None


def hypothesis(selected):
    """Supervised minus SSL head, per (root, loss, k, split), averaged over folds and seeds."""
    groups = defaultdict(list)
    for row in selected:
        if row["representation"] == "head":
            groups[(row["root"], row["loss"], row["k"], row["split"], row["method"])].append(row)
    out = []
    for (root, loss, k, split, method), items in sorted(groups.items()):
        if method == "sup":
            continue
        reference = groups.get((root, loss, k, split, "sup"))
        if not reference:
            continue
        entry = {"root": root, "loss": loss, "k": k, "split": split, "method": method,
                 "folds": len(items), "reference_folds": len(reference)}
        for metric in METRICS:
            a, b = fold_mean(reference, metric), fold_mean(items, metric)
            entry[f"sup_minus_ssl_{metric}"] = (a - b) if a is not None and b is not None else None
        out.append(entry)
    return out


def curves(rows):
    """Fold-mean metric per epoch for each (root, combo, k, seed, split); backbone as a flat reference."""
    groups = defaultdict(lambda: defaultdict(list))
    backbone = defaultdict(list)
    for row in rows:
        key = (row["root"], row["combo"], row["method"], row["loss"], row["k"], row["seed"], row["split"])
        if row["representation"] == "backbone":
            backbone[key].append(row)
        else:
            groups[key][row["epoch"]].append(row)
    out = []
    for key, by_epoch in sorted(groups.items()):
        root, combo, method, loss, k, seed, split = key
        epochs = sorted(by_epoch)
        series = {m: [fold_mean(by_epoch[e], m) for e in epochs] for m in METRICS}
        selected = [r["selected_epoch"] for e in epochs for r in by_epoch[e] if r["selected_epoch"] is not None]
        out.append({"root": root, "combo": combo, "method": method, "loss": loss, "k": k, "seed": seed,
                    "split": split, "epochs": epochs, "folds": max(len(v) for v in by_epoch.values()),
                    "selectedEpoch": float(np.mean(selected)) if selected else None, "series": series,
                    "backbone": {m: fold_mean(backbone.get(key, []), m) for m in METRICS}})
    return out


def spectra(selected, key="spectrum"):
    """Mean leading spectrum at the selected epoch, per (root, combo, k, split); backbone separately.

    ``spectrum``: variance shares of the centered embeddings (our variant).
    ``spectrum_roth``: Roth et al.'s rho_spectrum@0, the normalized singular
    values of the uncentered embeddings (their Fig. 7).
    """
    groups = defaultdict(list)
    for row in selected:
        if not row.get(key):
            continue
        name = "backbone" if row["representation"] == "backbone" else row["combo"]
        groups[(row["root"], name, row["k"] if name != "backbone" else -1, row["split"])].append(row[key])
    out = []
    for (root, name, k, split), items in sorted(groups.items()):
        length = min(len(s) for s in items)
        out.append({"root": root, "combo": name, "k": k, "split": split,
                    "shares": [round(float(v), 7) for v in np.mean([s[:length] for s in items], axis=0)]})
    return out


def draw(selected, hyp, curve_rows, out, metric="rho_roth"):
    pyplot = common.plt()
    # rho over training, one panel per split, per (root, k)
    for (root, k) in sorted({(c["root"], c["k"]) for c in curve_rows}):
        fig, axes = pyplot.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.2), sharey=False)
        for ax, split in zip(axes, SPLITS):
            for i, c in enumerate(c for c in curve_rows if c["root"] == root and c["k"] == k and c["split"] == split):
                ax.plot(c["epochs"], [v if v is not None else np.nan for v in c["series"][metric]],
                        lw=1.2, color=f"C{i}", label=c["combo"])
                if c["backbone"][metric] is not None and i == 0:
                    ax.axhline(c["backbone"][metric], color="k", ls="--", lw=0.8, label="backbone")
            ax.set_title(SPLIT_LABEL[split], fontsize=9)
            ax.set_xlabel("epoch")
        axes[0].set_ylabel("spectral decay ρ (Roth et al.)")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{root}, k={k}: ρ over training (fold mean)", fontsize=9)
        common.savefig(fig, out / f"rho_curves_{root}_k{k}.png")
    # the hypothesis: supervised minus SSL rho against k
    for root in sorted({h["root"] for h in hyp}):
        fig, axes = pyplot.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.2), sharey=True)
        for ax, split in zip(axes, SPLITS):
            for i, method in enumerate(sorted({h["method"] for h in hyp})):
                items = sorted((h for h in hyp if h["root"] == root and h["split"] == split and h["method"] == method),
                               key=lambda h: h["k"])
                if items:
                    ax.plot([h["k"] for h in items],
                            [h[f"sup_minus_ssl_{metric}"] if h[f"sup_minus_ssl_{metric}"] is not None else np.nan
                             for h in items], "o-",
                            color=f"C{i}", label=method)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_title(SPLIT_LABEL[split], fontsize=9)
            ax.set_xlabel("k")
        axes[0].set_ylabel("ρ(supervised) − ρ(SSL)")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{root}: > 0 means the supervised head is more compressed", fontsize=9)
        common.savefig(fig, out / f"hypothesis_{root}.png")


def main():
    args = parse_args()
    real_stdout = sys.stdout
    if args.stdout:
        sys.stdout = sys.stderr
    out = args.out / "geometry_tracking"
    rows = load_rows(args.roots)
    if not rows and args.stdout:
        real_stdout.write("null")
        return
    if not rows:
        raise SystemExit(f"no geometry.jsonl below {', '.join(map(str, args.roots))}")
    print(f"{len(rows)} geometry rows from {len({(r['root'], r['combo'], r['k'], r['seed'], r['fold']) for r in rows})} folds")
    selected = at_selected(rows)
    hyp = hypothesis(selected)
    curve_rows = curves(rows)
    points = correlation_points(selected, {(r["root"], r["combo"], r["k"], r["seed"]): {t: r.get(f"test_{t}") for t in TARGETS}
                                           for r in rows})
    corr = correlation(points)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roots": [str(r) for r in args.roots],
        "metrics": METRICS, "metricLabels": METRIC_LABEL, "splits": SPLITS, "splitLabels": SPLIT_LABEL,
        "curves": curve_rows, "hypothesis": hyp, "spectra": spectra(selected),
        "spectraRoth": spectra(selected, "spectrum_roth"), "classSpectraRoth": spectra(selected, "class_spectrum_roth"),
        "correlation": corr, "correlationPoints": points,
        "targets": {t: label for t, (_, label) in TARGETS.items()},
    }
    if args.stdout:
        real_stdout.write(json.dumps(payload, default=common._json_default))
        real_stdout.flush()
        return
    drop = {"spectrum", "spectrum_roth", "class_spectrum_roth"}
    common.write_rows(out / "rows.csv", [{k: v for k, v in r.items() if k not in drop} for r in rows])
    common.write_rows(out / "selected.csv", [{k: v for k, v in r.items() if k not in drop} for r in selected])
    common.write_rows(out / "hypothesis.csv", hyp)
    common.write_rows(out / "correlation.csv", corr)
    if not args.no_figures:
        draw(selected, hyp, curve_rows, out)
    def fmt(value):
        return f"{value:+.3f}" if value is not None else "   n/a"
    for h in hyp:
        print(f"  {h['root']:<14} {h['loss']:<4} k={h['k']:<3} {h['split']:<10} {h['method']:<7} "
              f"Δρ_Roth={fmt(h['sup_minus_ssl_rho_roth'])}  Δρ_centered={fmt(h['sup_minus_ssl_spectral_decay'])}")
    for c in corr:
        if c["split"] == "train" and c["k"] == "all" and c["metric"] in ROTH_METRICS:
            print(f"  corr {c['root']:<14} {c['target']:<6} {c['metric']:<14} n={c['n']:<3} pearson={fmt(c['pearson'])}  spearman={fmt(c['spearman'])}")
    if args.app:
        target = args.app / "public" / "geometry.json"
        common.write_json(target, payload)
        print(f"wrote {target}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
