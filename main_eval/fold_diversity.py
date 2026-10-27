#!/usr/bin/env python3
"""Why does Concat help? Diversity between the fold models of one run.

Concat averages the fold models' cosine similarities before ranking, so it can
only beat the fold mean where the folds disagree. For every run this measures,
on the shared test set:

* linear CKA between every pair of fold embeddings (1 = the same geometry up to
  rotation and scale);
* the overlap of each query's top-``R`` neighbour lists between fold pairs
  (Jaccard, averaged over queries);
* how often two folds disagree on whether the first neighbour is correct.

and joins them with the run's Concat-minus-Mean gain from ``cv_summary.json``.
The hypothesis from the thesis discussion: methods that gain more in Concat than
in Mean (LRML at small k) do so because their fold models are *more different*,
not because each model is better. The scatter of diversity vs. Concat gain
tests that across all runs.

Outputs below ``--out/diversity``: runs.csv, diversity_vs_gain.png.

Examples::

    python main_eval/fold_diversity.py --dataset cars --k 4 16
    python main_eval/fold_diversity.py --dataset cub --losses ms nt --topk 10
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

PP = 100.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_run_args(parser)
    parser.add_argument("--topk", type=int, default=10, help="neighbour list length for the overlap")
    parser.add_argument("--max-queries", type=int, default=4000, help="queries sampled for the overlap measure")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def neighbour_overlap(a, b):
    """Mean Jaccard overlap of two neighbour-index matrices, row by row."""

    total = 0.0
    for row_a, row_b in zip(a, b):
        set_a, set_b = set(row_a.tolist()), set(row_b.tolist())
        total += len(set_a & set_b) / len(set_a | set_b)
    return total / len(a)


def first_hit(neighbours, labels, queries):
    return labels[neighbours[:, 0]] == labels[queries]


def main():
    args = parse_args()
    out = args.out / "diversity"
    runs = common.runs_from_args(args)
    print(f"{len(runs)} runs")
    rng = np.random.default_rng(0)
    rows = []
    for run in runs:
        folds, labels = common.load_run_test_embeddings(run)
        if folds is None or len(folds) < 2:
            continue
        queries = np.arange(len(labels))
        if len(queries) > args.max_queries:
            queries = np.sort(rng.choice(queries, size=args.max_queries, replace=False))
        neighbours = []
        for embeddings in folds:
            full = common.topk_neighbors(embeddings, args.topk)
            neighbours.append(full[queries])
        cka, overlap, disagreement = [], [], []
        for i, j in itertools.combinations(range(len(folds)), 2):
            cka.append(common.linear_cka(folds[i], folds[j]))
            overlap.append(neighbour_overlap(neighbours[i], neighbours[j]))
            disagreement.append(np.mean(first_hit(neighbours[i], labels, queries) != first_hit(neighbours[j], labels, queries)))
        mean_map = run.metric("map_at_r", "mean")
        concat_map = run.metric("map_at_r", "concat")
        rows.append({
            "dataset": run.dataset, "combo": run.combo, "method": run.method, "loss": run.loss, "variant": run.variant,
            "k": run.k, "seed": run.seed, "n_folds": len(folds),
            "cka_mean": float(np.mean(cka)), "cka_min": float(np.min(cka)),
            f"top{args.topk}_overlap": float(np.mean(overlap)),
            "p1_disagreement": float(np.mean(disagreement)),
            "map_at_r_mean": mean_map * PP if mean_map is not None else None,
            "map_at_r_concat": concat_map * PP if concat_map is not None else None,
            "concat_gain_pp": (concat_map - mean_map) * PP if mean_map is not None and concat_map is not None else None,
        })
        print(f"  {run.combo:<24} k={run.k:<3} seed={run.seed:<3} CKA={rows[-1]['cka_mean']:.3f} "
              f"overlap={rows[-1][f'top{args.topk}_overlap']:.3f} concat gain={rows[-1]['concat_gain_pp']:+.2f} PP")
    common.write_rows(out / "runs.csv", rows)
    if rows and not args.no_figures:
        draw(rows, out, args.topk)
    print(f"wrote {out}")


def draw(rows, out, topk):
    pyplot = common.plt()
    from scipy import stats

    fig, axes = pyplot.subplots(1, 2, figsize=(10, 3.6))
    methods = sorted({r["method"] for r in rows}, key=lambda m: common.METHOD_ORDER.index(m) if m in common.METHOD_ORDER else 99)
    for ax, key, label in ((axes[0], "cka_mean", "mean pairwise linear CKA"), (axes[1], f"top{topk}_overlap", f"top-{topk} neighbour overlap")):
        for i, method in enumerate(methods):
            items = [r for r in rows if r["method"] == method]
            ax.scatter([r[key] for r in items], [r["concat_gain_pp"] for r in items], s=14, label=method, color=f"C{i}")
        x = np.array([r[key] for r in rows], dtype=float)
        y = np.array([r["concat_gain_pp"] for r in rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 5:
            rho = stats.spearmanr(x[mask], y[mask]).statistic
            ax.set_title(f"Spearman rho = {rho:+.2f} (n={mask.sum()})", fontsize=8)
        ax.set_xlabel(label + " between folds")
    axes[0].set_ylabel("Concat − Mean MAP@R [PP]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fold diversity vs. ensemble gain (each point: one run = one seed and budget)", fontsize=9)
    common.savefig(fig, out / "diversity_vs_gain.png")


if __name__ == "__main__":
    main()
