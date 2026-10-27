#!/usr/bin/env python3
"""Pseudo-label quality of label propagation, read from runs that already logged it.

The Phase-1 seed runs had pseudo-label diagnostics switched off, but the Phase-0
random-search trials wrote ``pseudo_label_diagnostics.jsonl`` for every fold:
per pseudo-label generation the accepted count, the hidden-label accuracy of the
accepted and of all raw predictions, the confidence/correctness AUC and the
predicted class distribution. Because those trials are a random sample of the
search space, they answer three Phase-2 questions without a new run:

1. How correct and how complete were the pseudo-labels the model actually
   trained on (at the validation-selected epoch)?
2. Does retrieval quality follow pseudo-label *accuracy*, *coverage*, or the
   number of *correct* / *wrong* accepted labels? (quality vs. quantity)
3. Which LP hyperparameters move accuracy and coverage (HP -> mechanism), and
   does that explain their effect on MAP@R (mechanism -> performance)?

Hidden labels are only read here; they never reached training.

Outputs below ``--out/pseudo_labels``: generations.csv (one row per fold and
generation), trials.csv (one row per trial), correlations.csv and figures per study.

Examples::

    python main_eval/pseudo_label_quality.py
    python main_eval/pseudo_label_quality.py --roots sorted_logs/phase_0/ssl/cars196 --method lp
    python main_eval/pseudo_label_quality.py --roots logs/phase2_ablations   # new runs with diagnostics on
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

DIAGNOSTICS = "pseudo_label_diagnostics.jsonl"
PP = 100.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, default=[common.PHASE0_ROOT])
    parser.add_argument("--method", nargs="+", default=["lp"], help="methods whose studies are read (default: lp)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--min-trials", type=int, default=10)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    return parser.parse_args()


def class_entropy(counts: dict | None) -> float:
    """Normalized entropy of the predicted-class distribution: 1 = uniform, 0 = one class."""

    if not counts:
        return float("nan")
    values = np.array([float(v) for v in counts.values() if v], dtype=float)
    if len(values) < 2:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def changes(value):
    if isinstance(value, dict):
        for key in ("changed", "count", "n_changed", "changes"):
            if key in value:
                return value[key]
        return None
    return value


def generation_rows(fold_dir: Path):
    path = fold_dir / DIAGNOSTICS
    if not path.exists():
        return []
    rows = []
    for record in common.read_jsonl(path):
        accepted = record.get("accepted") or {}
        raw = record.get("raw") or {}
        accepted_count = accepted.get("count") or 0
        raw_count = raw.get("count") or 0
        accuracy = accepted.get("audit_accuracy")
        rows.append({
            "epoch": record.get("epoch"),
            "generation": record.get("generation_index"),
            "confidence_threshold": record.get("confidence_threshold"),
            "accepted": accepted_count,
            "raw": raw_count,
            "coverage": accepted_count / raw_count if raw_count else float("nan"),
            "accuracy": accuracy,
            "raw_accuracy": raw.get("audit_accuracy"),
            "correct_share": (accuracy * accepted_count / raw_count) if raw_count and accuracy is not None else float("nan"),
            "wrong_share": ((1 - accuracy) * accepted_count / raw_count) if raw_count and accuracy is not None else float("nan"),
            "auc": (accepted.get("confidence_correctness") or {}).get("auc"),
            "classes_predicted": len([v for v in (accepted.get("predicted_class_counts") or {}).values() if v]),
            "classes_true": len(accepted.get("true_class_counts") or {}),
            "class_entropy": class_entropy(accepted.get("predicted_class_counts")),
            "changes": changes(record.get("accepted_changes_from_previous_generation")),
            "_buckets": (accepted.get("confidence_correctness") or {}).get("buckets"),
        })
    return rows


def at_epoch(generations, epoch):
    """The generation the model trained on at ``epoch``: the last one generated at or before it."""

    if not generations:
        return None
    if epoch is None:
        return generations[-1]
    earlier = [g for g in generations if g["epoch"] is not None and g["epoch"] <= epoch]
    return earlier[-1] if earlier else None


def mean_of(rows, key):
    values = [r[key] for r in rows if r is not None and r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(values)) if values else float("nan")


def collect(study):
    trial_rows, generation_table = [], []
    for trial in study.completed():
        folds = (trial.get("user_attrs") or {}).get("fold_results") or []
        per_fold = []
        for fold in folds:
            fold_dir = common.local_trial_dir(study.path, fold.get("log_dir", ""))
            generations = generation_rows(fold_dir)
            if not generations:
                continue
            for g in generations:
                generation_table.append({
                    "study_id": study.study_id, "label": study.label, "number": trial["number"],
                    "fold": fold.get("cv_fold"), "selected_epoch": fold.get("selected_epoch"),
                    **{k: v for k, v in g.items() if not k.startswith("_")},
                })
            per_fold.append({
                "first": generations[0],
                "selected": at_epoch(generations, fold.get("selected_epoch")),
                "last": generations[-1],
                "n": len(generations),
            })
        if not per_fold:
            continue
        row = {
            "study_id": study.study_id,
            "label": study.label,
            "dataset": study.dataset,
            "loss": study.loss,
            "number": int(trial["number"]),
            "is_random": study.is_random(trial),
            "value": float(trial["value"]),
            "n_folds_with_diagnostics": len(per_fold),
            "generations_mean": float(np.mean([f["n"] for f in per_fold])),
            "selected_before_first_generation": int(sum(f["selected"] is None for f in per_fold)),
        }
        for when in ("first", "selected", "last"):
            chosen = [f[when] for f in per_fold]
            for key in ("accuracy", "raw_accuracy", "coverage", "correct_share", "wrong_share", "auc", "class_entropy", "classes_predicted"):
                row[f"{key}_{when}"] = mean_of(chosen, key)
        for name, value in (trial.get("params") or {}).items():
            row[f"p:{name}"] = value
        # The confidence/accuracy buckets of the selected generation, fold 0 only (for the calibration plot).
        selected = per_fold[0]["selected"]
        row["_buckets"] = selected.get("_buckets") if selected else None
        trial_rows.append(row)
    return trial_rows, generation_table


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 6 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def partial_spearman(x, y, controls):
    """Spearman of x and y after removing the rank-linear effect of ``controls`` from both."""

    data = np.column_stack([x, y] + list(controls)).astype(float)
    data = data[np.all(np.isfinite(data), axis=1)]
    if len(data) < 8:
        return float("nan"), len(data)
    ranks = np.apply_along_axis(stats.rankdata, 0, data)
    design = np.column_stack([np.ones(len(ranks)), ranks[:, 2:]])
    residual = []
    for column in (0, 1):
        beta, *_ = np.linalg.lstsq(design, ranks[:, column], rcond=None)
        residual.append(ranks[:, column] - design @ beta)
    return float(np.corrcoef(residual[0], residual[1])[0, 1]), len(data)


def correlations(study, rows):
    out = []
    random_rows = [r for r in rows if r["is_random"]]
    value = [r["value"] for r in random_rows]
    mechanisms = ["accuracy_selected", "coverage_selected", "correct_share_selected", "wrong_share_selected",
                  "auc_selected", "class_entropy_selected", "accuracy_first"]
    for mechanism in mechanisms:
        rho, p, n = spearman([r[mechanism] for r in random_rows], value)
        out.append({"study_id": study.study_id, "label": study.label, "x": mechanism, "y": "value",
                    "spearman": rho, "p": p, "n": n})
    # Quality vs quantity: accuracy after controlling coverage, and the reverse.
    acc = [r["accuracy_selected"] for r in random_rows]
    cov = [r["coverage_selected"] for r in random_rows]
    rho, n = partial_spearman(acc, value, [cov])
    out.append({"study_id": study.study_id, "label": study.label, "x": "accuracy_selected | coverage", "y": "value", "spearman": rho, "n": n})
    rho, n = partial_spearman(cov, value, [acc])
    out.append({"study_id": study.study_id, "label": study.label, "x": "coverage_selected | accuracy", "y": "value", "spearman": rho, "n": n})
    # HP -> mechanism and HP -> value.
    params = sorted({k[2:] for r in random_rows for k in r if k.startswith("p:")})
    for name in params:
        x = [r.get(f"p:{name}") for r in random_rows]
        if not all(isinstance(v, (int, float)) for v in x if v is not None):
            continue
        x = [float(v) if v is not None else np.nan for v in x]
        for target in ("accuracy_selected", "coverage_selected", "value"):
            rho, p, n = spearman(x, [r[target] for r in random_rows])
            out.append({"study_id": study.study_id, "label": study.label, "x": name, "y": target, "spearman": rho, "p": p, "n": n})
    return out


def draw(study, rows, generations, out_dir):
    pyplot = common.plt()
    value = np.array([r["value"] for r in rows]) * PP
    acc = np.array([r["accuracy_selected"] for r in rows]) * PP
    cov = np.array([r["coverage_selected"] for r in rows]) * PP

    fig, axes = pyplot.subplots(1, 3, figsize=(11, 3.2))
    sc = axes[0].scatter(acc, value, c=cov, cmap="viridis", s=16)
    fig.colorbar(sc, ax=axes[0], label="coverage [%]")
    axes[0].set_xlabel("pseudo-label accuracy at selected epoch [%]")
    axes[0].set_ylabel("validation MAP@R [%]")
    sc = axes[1].scatter(cov, value, c=acc, cmap="magma", s=16)
    fig.colorbar(sc, ax=axes[1], label="accuracy [%]")
    axes[1].set_xlabel("coverage of unlabeled pool [%]")
    correct = np.array([r["correct_share_selected"] for r in rows]) * PP
    wrong = np.array([r["wrong_share_selected"] for r in rows]) * PP
    axes[2].scatter(correct, value, s=16, label="correct / pool", c="tab:green")
    axes[2].scatter(wrong, value, s=16, label="wrong / pool", c="tab:red")
    axes[2].set_xlabel("accepted labels as share of pool [%]")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{study.label}: quality vs. quantity of pseudo-labels ({len(rows)} trials)", fontsize=9)
    common.savefig(fig, out_dir / "quality_vs_quantity.png")

    params = [k[2:] for k in rows[0] if k.startswith("p:") and isinstance(rows[0][k], (int, float))]
    if params:
        cols = min(4, len(params))
        nrows = math.ceil(len(params) / cols)
        fig, axes = pyplot.subplots(nrows, cols, figsize=(3.2 * cols, 2.8 * nrows), squeeze=False)
        fig.subplots_adjust(hspace=0.6, wspace=0.35)
        for ax, name in zip(axes.flat, params):
            x = np.array([float(r.get(f"p:{name}", np.nan)) for r in rows])
            space = study.spaces.get(name) or {}
            if space.get("log"):
                ax.set_xscale("log")
            ax.scatter(x, acc, c=value, cmap="viridis", s=12)
            ax.set_title(common.short_param(name), fontsize=8)
        for ax in axes.flat[len(params):]:
            ax.axis("off")
        axes.flat[0].set_ylabel("PL accuracy [%]")
        fig.suptitle(f"{study.label}: hyperparameter -> pseudo-label accuracy (colour: MAP@R)", fontsize=9)
        common.savefig(fig, out_dir / "hp_to_accuracy.png")

    order = np.argsort(-value)
    picks = {"top 5": order[:5], "bottom 5": order[-5:]}
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 3.2))
    for style, (name, indices) in zip(("-", ":"), picks.items()):
        for index in indices:
            number = rows[index]["number"]
            series = [g for g in generations if g["number"] == number and g["fold"] in (0, None)]
            if not series:
                continue
            epochs = [g["epoch"] for g in series]
            axes[0].plot(epochs, [g["accuracy"] * PP for g in series], style, lw=1)
            axes[1].plot(epochs, [g["coverage"] * PP for g in series], style, lw=1)
    axes[0].set_ylabel("accepted accuracy [%]")
    axes[1].set_ylabel("coverage [%]")
    for ax in axes:
        ax.set_xlabel("epoch (fold 0)")
    fig.suptitle(f"{study.label}: solid = 5 best trials, dotted = 5 worst", fontsize=9)
    common.savefig(fig, out_dir / "generations.png")

    buckets = rows[int(order[0])].get("_buckets")
    if buckets:
        fig, ax = pyplot.subplots(figsize=(4, 3))
        ax.plot([b["confidence_mean"] for b in buckets], [b["accuracy"] for b in buckets], "o-")
        ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.7)
        ax.set_xlabel("confidence (decile mean)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{study.label}: calibration, best trial, fold 0", fontsize=8)
        common.savefig(fig, out_dir / "calibration_best.png")


def main():
    args = parse_args()
    out = args.out / "pseudo_labels"
    studies = [
        s for s in common.load_studies(args.roots, min_trials=args.min_trials)
        if s.method in args.method and (args.dataset is None or s.dataset == args.dataset)
    ]
    print(f"{len(studies)} candidate studies")
    all_trials, all_generations, all_correlations = [], [], []
    for study in studies:
        rows, generations = collect(study)
        if len(rows) < 6:
            continue
        print(f"  {study.label:<22} {len(rows):>3} trials with diagnostics  ({study.study_id})")
        all_trials.extend(rows)
        all_generations.extend(generations)
        all_correlations.extend(correlations(study, rows))
        if not args.no_figures:
            draw(study, rows, generations, out / "figures" / re.sub(r"[^A-Za-z0-9_.-]+", "_", study.study_id))
    if not all_trials:
        print("no trial had pseudo_label_diagnostics.jsonl below these roots")
        return
    common.write_rows(out / "trials.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_trials])
    common.write_rows(out / "generations.csv", all_generations)
    common.write_rows(out / "correlations.csv", all_correlations)
    print("\naccuracy / coverage vs. MAP@R on random trials (Spearman):")
    for row in all_correlations:
        if row["y"] == "value" and row["x"].startswith(("accuracy_selected", "coverage_selected")):
            print(f"  {row['label']:<22} {row['x']:<32} rho={row['spearman']:+.2f} (n={row['n']})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
