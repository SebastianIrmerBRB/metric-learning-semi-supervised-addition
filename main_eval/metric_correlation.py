#!/usr/bin/env python3
"""Roth et al. (2020), Fig. 4: how strongly do the evaluation metrics agree?

Roth et al. correlate R@1, R@2, R@4, NMI, F1 and mAP@C over their DML
objectives: per dataset one point per objective (the metric averaged over
seeds), Pearson correlation between every pair of metrics (``np.corrcoef`` in
their Result_Evaluations.py). Here one point is one (method, loss) combination
at one label budget k, averaged over seeds, so the matrix answers the same
question for the Phase-1 runs: would ranking the runs by R@1 or NMI tell the
same story as MAP@R, the thesis' primary metric?

Two metric sets:

* stored (every run): the test metrics the runs recorded, fold mean -- MAP@R,
  R@1, R@2, R@4 (Cars/CUB), NMI, AMI, R-Precision, mAP, MRR.
* ``--roth-metrics``: Roth's own implementations (``main_eval/roth_metrics.py``,
  checked against their code) of R@1/2/4, NMI, F1, mAP@C and mAP@1000,
  computed from each fold's ``test_embeddings.npz`` and cached. Only runs whose
  embeddings are available locally take part, so this matrix can have fewer
  points than the stored one.

Pooled over k, the label budget moves every metric at once and inflates the
correlations; the per-k matrices are the comparison Roth's figure makes (a
fixed training setup, different objectives).

Outputs below ``--out/metric_correlation``: points.csv, correlation.csv (long
form, Pearson and Spearman), matrix_<dataset>_<scope>.{png,pdf} and a booktabs
matrix_<dataset>_<scope>.tex per metric set.

Examples::

    python main_eval/metric_correlation.py
    python main_eval/metric_correlation.py --dataset cars --roth-metrics
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

STORED = {
    "map_r": "MAP@R", "r1": "R@1", "r2": "R@2", "r4": "R@4", "nmi": "NMI", "ami": "AMI",
    "r_precision": "R-Precision", "map": "mAP", "mrr": "MRR",
}
# Roth's set, with the thesis' MAP@R (as the runs recorded it) put in front.
ROTH = {
    "map_r": "MAP@R", "roth_recall@1": "R@1", "roth_recall@2": "R@2", "roth_recall@4": "R@4", "roth_nmi": "NMI",
    "roth_f1": "F1", "roth_map_c": "mAP@C", "roth_map_1000": "mAP@1000",
}
MEASUREMENT_KEY = {"map_r": "mean_average_precision_at_r", "r1": "precision_at_1", "nmi": "NMI", "ami": "AMI",
                   "r_precision": "r_precision", "map": "mean_average_precision", "mrr": "mean_reciprocal_rank"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_run_args(parser)
    parser.add_argument("--roth-metrics", action="store_true",
                        help="also compute Roth's metric implementations from the fold test embeddings (cached)")
    parser.add_argument("--per-seed", action="store_true", help="one point per seed instead of the seed mean")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def stored_metrics(run) -> dict:
    aggregate = run.summary.get("aggregate") or {}
    measurements = aggregate.get("test_measurements") or {}
    out = {name: measurements.get(key) for name, key in MEASUREMENT_KEY.items()}
    recall = aggregate.get("test_recall_at_k") or {}
    for k in (2, 4):
        out[f"r{k}"] = recall.get(str(k), recall.get(k))
    return out


def roth_fold_metrics(run, cache: dict, cache_path: Path) -> dict | None:
    from main_eval.roth_metrics import roth_metrics

    per_fold = []
    for fold_dir in run.fold_dirs():
        key = str(fold_dir)
        if key not in cache:
            embeddings, labels = common.load_test_embeddings(fold_dir)
            if embeddings is None:
                return None
            cache[key] = {f"roth_{name}": value for name, value in roth_metrics(embeddings, labels).items()}
            common.write_rows(cache_path, [{"fold_dir": k, **v} for k, v in cache.items()])
        per_fold.append(cache[key])
    return {name: float(np.mean([fold[name] for fold in per_fold])) for name in per_fold[0]}


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    import csv

    with path.open() as handle:
        return {row.pop("fold_dir"): {k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(handle)}


def points(runs, args, out):
    cache_path = out / "roth_metrics_cache.csv"
    cache = load_cache(cache_path) if args.roth_metrics else {}
    rows = []
    for i, run in enumerate(runs):
        row = {"dataset": common.DATASET_SHORT.get(run.dataset, run.dataset), "combo": run.combo,
               "method": run.method, "loss": run.loss, "k": run.k, "seed": run.seed, **stored_metrics(run)}
        if args.roth_metrics:
            roth = roth_fold_metrics(run, cache, cache_path)
            if roth:
                row.update(roth)
            print(f"  [{i + 1}/{len(runs)}] {row['dataset']} {run.combo} k={run.k} seed={run.seed}: "
                  f"{'Roth metrics' if roth else 'no test embeddings'}", flush=True)
        rows.append(row)
    if args.per_seed:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["combo"], row["k"])].append(row)
    averaged = []
    for (dataset, combo, k), items in sorted(groups.items()):
        entry = {"dataset": dataset, "combo": combo, "method": items[0]["method"], "loss": items[0]["loss"],
                 "k": k, "seeds": len(items)}
        for name in list(STORED) + list(ROTH):
            values = [r[name] for r in items if r.get(name) is not None]
            entry[name] = float(np.mean(values)) if values else None
        averaged.append(entry)
    return averaged


def _rank(values):
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float)


def matrices(table, metric_set):
    """Per dataset and scope (all k, each k): Pearson and Spearman over the points that have every metric."""
    out = []
    for dataset in sorted({t["dataset"] for t in table}):
        rows = [t for t in table if t["dataset"] == dataset]
        for scope in ["all"] + sorted({t["k"] for t in rows}):
            scoped = [t for t in rows if scope == "all" or t["k"] == scope]
            names = [m for m in metric_set if sum(t.get(m) is not None for t in scoped) >= 3]
            complete = [t for t in scoped if all(t.get(m) is not None for m in names)]
            if len(complete) < 3 or len(names) < 2:
                continue
            values = np.array([[t[m] for m in names] for t in complete], dtype=float).T
            varying = values.std(axis=1) > 0
            names = [m for m, keep in zip(names, varying) if keep]
            values = values[varying]
            pearson = np.corrcoef(values)
            spearman = np.corrcoef(np.array([_rank(v) for v in values]))
            out.append({"dataset": dataset, "scope": scope, "names": names, "n": len(complete),
                        "pearson": pearson, "spearman": spearman})
    return out


def draw(entry, labels, path):
    pyplot = common.plt()
    names = entry["names"]
    size = 0.62 * len(names) + 1.6
    fig, ax = pyplot.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(entry["pearson"], vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), [labels[m] for m in names], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), [labels[m] for m in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            value = entry["pearson"][i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.6 else "black")
    scope = "alle k" if entry["scope"] == "all" else f"k={entry['scope']}"
    ax.set_title(f"{entry['dataset']}, {scope}: Pearson, n={entry['n']}", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    common.savefig(fig, path.with_suffix(".png"))
    return path


def latex(entry, labels):
    names = entry["names"]
    header = " & ".join([""] + [labels[m] for m in names]) + r" \\"
    body = [" & ".join([labels[a]] + [f"{entry['pearson'][i, j]:.2f}" for j in range(len(names))]) + r" \\"
            for i, a in enumerate(names)]
    scope = "über alle k" if entry["scope"] == "all" else f"bei k={entry['scope']}"
    return "\n".join([
        r"\begin{table}[htbp]", r"  \centering", r"  \small",
        rf"  \begin{{tabular}}{{l{'r' * len(names)}}}", r"    \toprule", f"    {header}", r"    \midrule",
        *[f"    {line}" for line in body], r"    \bottomrule", r"  \end{tabular}",
        rf"  \caption{{Pearson-Korrelation der Evaluationsmetriken ({entry['dataset']}, {scope}, "
        rf"n={entry['n']} Kombinationen, Mittel über Seeds), wie Roth et al.\ (2020), Abb.~4.}}",
        rf"  \label{{tab:metrik-korrelation-{entry['dataset']}-{entry['scope']}}}", r"\end{table}", "",
    ])


def main():
    args = parse_args()
    out = args.out / "metric_correlation"
    out.mkdir(parents=True, exist_ok=True)
    runs = common.runs_from_args(args)
    print(f"{len(runs)} runs")
    table = points(runs, args, out)
    common.write_rows(out / "points.csv", table)
    long = []
    for set_name, labels in (("stored", STORED), ("roth", ROTH)):
        if set_name == "roth" and not args.roth_metrics:
            continue
        for entry in matrices(table, list(labels)):
            names = entry["names"]
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    long.append({"set": set_name, "dataset": entry["dataset"], "k": entry["scope"], "n": entry["n"],
                                 "metric_a": labels[a], "metric_b": labels[b],
                                 "pearson": float(entry["pearson"][i, j]), "spearman": float(entry["spearman"][i, j])})
            stem = out / f"matrix_{set_name}_{entry['dataset']}_{entry['scope'] if entry['scope'] == 'all' else 'k' + str(entry['scope'])}"
            (stem.with_suffix(".tex")).write_text(latex(entry, labels))
            if not args.no_figures:
                draw(entry, labels, stem)
            if "map_r" in names:
                i = names.index("map_r")
                others = ", ".join(f"{labels[m]} {entry['pearson'][i, j]:+.2f}" for j, m in enumerate(names) if m != "map_r")
                print(f"  {set_name:6s} {entry['dataset']:8s} {str(entry['scope']):>4s} n={entry['n']:<3} MAP@R vs {others}")
    common.write_rows(out / "correlation.csv", long)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
