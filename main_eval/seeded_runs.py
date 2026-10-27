#!/usr/bin/env python3
"""Collect every seeded CV run and pair it: method vs. supervised, variant vs. base.

This is the evaluation side of the ablations. A replay launched with
``--ablation_config`` lands in ``<save_dir_root>/<combo>_<variant>/`` (for
example ``hoffer_ms_pool_labeled``); this script finds it next to the unmodified
replay (``hoffer_ms``), pairs both by dataset, budget k and seed, and reports the
paired differences the thesis uses (mean, SD, min, max, positive seeds). Every
method is also paired with the supervised run of the same loss.

Pairs are formed within one root first (a variant belongs to the series it was
replayed from), and only then across roots.

Outputs below ``--out/seeded``: runs.csv (one row per run and aggregation),
paired.csv (one row per pair and seed), paired_summary.csv, and one figure per
dataset with the paired MAP@R differences.

Examples::

    python main_eval/seeded_runs.py
    python main_eval/seeded_runs.py --dataset cars --include-variants
    python main_eval/seeded_runs.py --seeded-roots /mnt/f/server_logs/seeded_runs_cars logs/phase2_ablations
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

PP = 100.0
AGGREGATIONS = ("mean", "concat")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_run_args(parser)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def run_row(run: common.SeededRun, aggregation: str) -> dict:
    aggregate = run.summary.get("aggregate") or {}
    folds = aggregate.get("fold_results") or []
    row = {
        "dataset": run.dataset,
        "root": str(run.root),
        "combo": run.combo,
        "method": run.method,
        "loss": run.loss,
        "variant": run.variant,
        "k": run.k,
        "seed": run.seed,
        "aggregation": aggregation,
        "n_folds": len(folds),
        "selected_epoch_mean": float(np.mean([f["selected_epoch"] for f in folds if f.get("selected_epoch") is not None]))
        if folds else float("nan"),
        "last_epoch_mean": float(np.mean([f["last_epoch"] for f in folds if f.get("last_epoch") is not None]))
        if folds else float("nan"),
        "best_valid_map_at_r": aggregate.get("best_valid_mean_average_precision_at_r"),
        "cv_summary": str(run.cv_summary_path),
    }
    for short in common.PRIMARY_METRICS:
        value = run.metric(short, aggregation)
        row[short] = value * PP if value is not None else None
    return row


def find_partner(run, index, combo):
    """Same dataset/k/seed under ``combo``; prefer the same root."""

    candidates = index.get((run.dataset, combo, run.k, run.seed), [])
    same_root = [c for c in candidates if c.root == run.root]
    return (same_root or candidates or [None])[0]


def main():
    args = parse_args()
    out = args.out / "seeded"
    everything = common.load_seeded_runs(args.seeded_roots)
    runs = common.select_runs(
        everything, dataset=args.dataset, k=args.k, combos=args.combos, methods=args.methods,
        losses=args.losses, include_variants=args.include_variants,
    )
    print(f"{len(runs)} runs selected ({len(everything)} found)")
    index = defaultdict(list)
    for run in everything:
        index[(run.dataset, run.combo, run.k, run.seed)].append(run)

    common.write_rows(out / "runs.csv", [run_row(r, a) for r in runs for a in AGGREGATIONS])

    paired = []
    for run in runs:
        references = []
        if run.method != "sup":
            references.append(("vs_sup", f"sup_{run.loss}"))
        if run.variant:
            references.append(("vs_base", run.base_combo))
        for kind, combo in references:
            partner = find_partner(run, index, combo)
            if partner is None:
                continue
            for aggregation in AGGREGATIONS:
                row = {
                    "dataset": run.dataset, "comparison": kind, "combo": run.combo, "reference": combo,
                    "method": run.method, "loss": run.loss, "variant": run.variant, "k": run.k, "seed": run.seed,
                    "aggregation": aggregation, "same_root": partner.root == run.root,
                }
                for short in common.PRIMARY_METRICS:
                    a, b = run.metric(short, aggregation), partner.metric(short, aggregation)
                    row[f"d_{short}"] = (a - b) * PP if a is not None and b is not None else None
                paired.append(row)
    common.write_rows(out / "paired.csv", paired)

    groups = defaultdict(list)
    for row in paired:
        groups[(row["dataset"], row["comparison"], row["combo"], row["reference"], row["k"], row["aggregation"])].append(row)
    summary = []
    for (dataset, comparison, combo, reference, k, aggregation), rows in sorted(groups.items()):
        entry = {"dataset": dataset, "comparison": comparison, "combo": combo, "reference": reference, "k": k,
                 "aggregation": aggregation, "seeds": sorted(r["seed"] for r in rows)}
        for short in common.PRIMARY_METRICS:
            stats = common.paired_summary(r[f"d_{short}"] for r in rows)
            entry.update({f"{short}_{key}": value for key, value in stats.items()})
        summary.append(entry)
    common.write_rows(out / "paired_summary.csv", summary)

    print("\npaired Concat-MAP@R differences [PP] (mean ± SD, positive seeds):")
    for entry in summary:
        if entry["aggregation"] == "concat":
            print(f"  {common.DATASET_SHORT.get(entry['dataset'], entry['dataset']):<8} k={entry['k']:<3} "
                  f"{entry['combo']:<28} vs {entry['reference']:<14} "
                  f"{entry.get('map_at_r_mean', float('nan')):+6.2f} ± {entry.get('map_at_r_sd', float('nan')):.2f} "
                  f"({entry.get('map_at_r_n_positive', 0)}/{entry.get('map_at_r_n', 0)})")

    if not args.no_figures:
        draw(summary, out)
    print(f"\nwrote {out}")


def draw(summary, out):
    pyplot = common.plt()
    by_dataset = defaultdict(list)
    for entry in summary:
        if entry["aggregation"] == "concat" and entry.get("map_at_r_n"):
            by_dataset[(entry["dataset"], entry["comparison"])].append(entry)
    for (dataset, comparison), entries in by_dataset.items():
        combos = sorted({e["combo"] for e in entries})
        ks = sorted({e["k"] for e in entries})
        fig, ax = pyplot.subplots(figsize=(max(5, 0.45 * len(combos) * len(ks)), 3.2))
        width = 0.8 / max(len(ks), 1)
        for j, k in enumerate(ks):
            xs, ys, es = [], [], []
            for i, combo in enumerate(combos):
                match = [e for e in entries if e["combo"] == combo and e["k"] == k]
                if match:
                    xs.append(i + (j - (len(ks) - 1) / 2) * width)
                    ys.append(match[0]["map_at_r_mean"])
                    es.append(match[0]["map_at_r_sd"] if np.isfinite(match[0]["map_at_r_sd"]) else 0)
            ax.bar(xs, ys, width=width, yerr=es, capsize=2, label=f"k={k}")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(combos)), combos, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Δ Concat-MAP@R [PP]")
        ax.legend(fontsize=7, ncol=len(ks))
        ax.set_title(f"{dataset}: paired differences ({comparison}), mean ± seed SD", fontsize=9)
        common.savefig(fig, out / f"paired_{common.DATASET_SHORT.get(dataset, dataset)}_{comparison}.png")


if __name__ == "__main__":
    main()
