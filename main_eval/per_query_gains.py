#!/usr/bin/env python3
"""Where does a method gain? Paired per-query and per-class differences to the supervised run.

Five seeds give little power for one headline number, but every run is scored
on thousands of test queries from ~100 unseen classes. This script recomputes
MAP@R per query from the stored test embeddings (Concat by default, or the mean
over fold models), pairs method and supervised run seed by seed, and reports:

* the distribution of per-query gains (how many queries improve / worsen);
* the gain per test class with a bootstrap interval over that class' queries;
* how the class gain relates to properties that need no training: the class
  size and -- with ``--backbone-npz`` -- how hard the class already is for the
  frozen DINOv2 features (their per-class MAP@R) and how compact it is there.

Test classes never appear in training, so the question answered is "on which
kind of *new* class does the method help", not which training class it fits.

Same-set retrieval only (Cars196, CUB). The recomputed mean is checked against
the MAP@R stored in cv_summary.json and the difference is printed.

Outputs below ``--out/per_query/<dataset>_k<k>_<combo>``: queries.csv,
classes.csv, summary.json, class_gains.png, gain_vs_difficulty.png.

Examples::

    python main_eval/per_query_gains.py --dataset cars --k 4 --combo iscen_ms
    python main_eval/per_query_gains.py --dataset cub --k 16 --combo lrml_ms --view folds \\
        --backbone-npz main_eval_out/frozen/backbone_test_CUB.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

PP = 100.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeded-roots", nargs="+", type=Path, default=common.SEEDED_ROOTS)
    parser.add_argument("--dataset", required=True, help="cars or cub")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--combo", required=True, help="e.g. iscen_ms, lrml_ms, hoffer_ms_pool_labeled")
    parser.add_argument("--reference", default=None, help="default: sup_<loss>")
    parser.add_argument("--view", choices=("concat", "folds"), default="concat",
                        help="concat: the Concat ensemble; folds: mean of the fold models' per-query AP")
    parser.add_argument("--backbone-npz", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    return parser.parse_args()


def per_query(run, view):
    folds, labels = common.load_run_test_embeddings(run)
    if folds is None:
        raise SystemExit(f"no test embeddings for {run.combo} seed {run.seed}")
    if view == "concat":
        result = common.per_query_retrieval(common.concat_embeddings(folds), labels)
        stored = run.metric("map_at_r", "concat")
    else:
        results = [common.per_query_retrieval(e, labels) for e in folds]
        result = {key: np.mean([r[key] for r in results], axis=0) for key in ("ap_at_r", "r_precision", "p_at_1")}
        result["r"] = results[0]["r"]
        stored = run.metric("map_at_r", "mean")
    recomputed = float(np.nanmean(result["ap_at_r"]))
    if stored is not None and abs(recomputed - stored) > 2e-3:
        print(f"[warn] {run.combo} seed {run.seed}: recomputed MAP@R {recomputed:.4f} vs stored {stored:.4f}", file=sys.stderr)
    return result, labels, recomputed, stored


def main():
    args = parse_args()
    loss = args.combo.split("_")[1]
    reference = args.reference or f"sup_{loss}"
    runs = common.load_seeded_runs(args.seeded_roots)
    mine = {r.seed: r for r in common.select_runs(runs, dataset=args.dataset, k=args.k, combos=[args.combo])}
    theirs = {r.seed: r for r in common.select_runs(runs, dataset=args.dataset, k=args.k, combos=[reference])}
    seeds = sorted(set(mine) & set(theirs))
    if not seeds:
        raise SystemExit(f"no paired seeds for {args.combo} vs {reference} at k={args.k}")
    dataset = mine[seeds[0]].dataset
    if dataset == "DeepFashionInShop":
        raise SystemExit("In-Shop uses a query/gallery protocol; per-query gains are only implemented for same-set retrieval")
    out = args.out / "per_query" / f"{common.DATASET_SHORT.get(dataset, dataset)}_k{args.k}_{args.combo}_{args.view}"

    deltas, method_ap, reference_ap, labels = [], [], [], None
    checks = []
    for seed in seeds:
        a, labels_a, rec_a, sto_a = per_query(mine[seed], args.view)
        b, labels_b, rec_b, sto_b = per_query(theirs[seed], args.view)
        if not np.array_equal(labels_a, labels_b):
            raise SystemExit("test label order differs between method and reference")
        labels = labels_a
        method_ap.append(a["ap_at_r"])
        reference_ap.append(b["ap_at_r"])
        deltas.append(a["ap_at_r"] - b["ap_at_r"])
        checks.append({"seed": seed, "method_recomputed": rec_a, "method_stored": sto_a,
                       "reference_recomputed": rec_b, "reference_stored": sto_b})
        print(f"  seed {seed}: {args.combo} {rec_a * PP:.2f} vs {reference} {rec_b * PP:.2f} -> {np.nanmean(deltas[-1]) * PP:+.2f} PP")
    delta = np.nanmean(deltas, axis=0)
    method_mean = np.nanmean(method_ap, axis=0)
    reference_mean = np.nanmean(reference_ap, axis=0)

    backbone_ap = backbone_labels = backbone_embeddings = None
    if args.backbone_npz:
        data = np.load(args.backbone_npz)
        backbone_embeddings, backbone_labels = data["embeddings"], data["labels"]
        if not np.array_equal(backbone_labels, labels):
            raise SystemExit("backbone test labels are in a different order than the run's")
        backbone_ap = common.per_query_retrieval(backbone_embeddings, backbone_labels)["ap_at_r"]

    common.write_rows(out / "queries.csv", [
        {"query": i, "label": int(labels[i]), "reference_ap_at_r": reference_mean[i] * PP, "method_ap_at_r": method_mean[i] * PP,
         "delta_pp": delta[i] * PP, "seeds_positive": int(sum(d[i] > 0 for d in deltas)),
         **({"backbone_ap_at_r": backbone_ap[i] * PP} if backbone_ap is not None else {})}
        for i in range(len(labels))
    ])

    classes = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        low, high = common.bootstrap_ci(delta[members] * PP)
        row = {"label": int(label), "n": len(members), "reference_ap_at_r": float(np.nanmean(reference_mean[members]) * PP),
               "method_ap_at_r": float(np.nanmean(method_mean[members]) * PP), "delta_pp": float(np.nanmean(delta[members]) * PP),
               "ci_low": low, "ci_high": high}
        if backbone_ap is not None:
            unit = common.l2_normalize(backbone_embeddings[members])
            centroid = unit.mean(axis=0)
            row["backbone_ap_at_r"] = float(np.nanmean(backbone_ap[members]) * PP)
            row["backbone_compactness"] = float(np.mean(unit @ (centroid / np.linalg.norm(centroid))))
        classes.append(row)
    common.write_rows(out / "classes.csv", classes)

    from scipy import stats

    summary = {
        "dataset": dataset, "k": args.k, "combo": args.combo, "reference": reference, "view": args.view, "seeds": seeds,
        "mean_delta_pp": float(np.nanmean(delta) * PP),
        "queries_improved_share": float(np.mean(delta > 0)), "queries_worsened_share": float(np.mean(delta < 0)),
        "classes_improved": int(sum(c["delta_pp"] > 0 for c in classes)), "classes": len(classes),
        "classes_ci_above_zero": int(sum(c["ci_low"] > 0 for c in classes)),
        "classes_ci_below_zero": int(sum(c["ci_high"] < 0 for c in classes)),
        "spearman_class_delta_vs_size": float(stats.spearmanr([c["n"] for c in classes], [c["delta_pp"] for c in classes]).statistic),
        "spearman_class_delta_vs_reference_ap": float(stats.spearmanr([c["reference_ap_at_r"] for c in classes], [c["delta_pp"] for c in classes]).statistic),
        "metric_check": checks,
    }
    if backbone_ap is not None:
        summary["spearman_class_delta_vs_backbone_ap"] = float(
            stats.spearmanr([c["backbone_ap_at_r"] for c in classes], [c["delta_pp"] for c in classes]).statistic)
        summary["spearman_class_delta_vs_backbone_compactness"] = float(
            stats.spearmanr([c["backbone_compactness"] for c in classes], [c["delta_pp"] for c in classes]).statistic)
    common.write_json(out / "summary.json", summary)
    for key, value in summary.items():
        if key != "metric_check":
            print(f"  {key}: {value}")

    if not args.no_figures:
        pyplot = common.plt()
        order = np.argsort([c["delta_pp"] for c in classes])
        fig, ax = pyplot.subplots(figsize=(max(6, 0.08 * len(classes)), 3.2))
        ys = np.array([classes[i]["delta_pp"] for i in order])
        err = np.array([[classes[i]["delta_pp"] - classes[i]["ci_low"], classes[i]["ci_high"] - classes[i]["delta_pp"]] for i in order]).T
        ax.bar(range(len(order)), ys, color=np.where(ys > 0, "tab:green", "tab:red"), yerr=err, error_kw={"lw": 0.4})
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("test class (sorted)")
        ax.set_ylabel("Δ MAP@R [PP]")
        ax.set_title(f"{dataset} k={args.k}: {args.combo} − {reference} per test class ({args.view}, {len(seeds)} seeds, 95% bootstrap)", fontsize=8)
        common.savefig(fig, out / "class_gains.png")

        x_key = "backbone_ap_at_r" if backbone_ap is not None else "reference_ap_at_r"
        fig, ax = pyplot.subplots(figsize=(4.4, 3.4))
        ax.scatter([c[x_key] for c in classes], [c["delta_pp"] for c in classes], s=[4 + c["n"] / 3 for c in classes], alpha=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(f"class {x_key.replace('_', ' ')} [%]")
        ax.set_ylabel("Δ MAP@R [PP]")
        ax.set_title("gain vs. how hard the class already is (size = #images)", fontsize=8)
        common.savefig(fig, out / "gain_vs_difficulty.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
