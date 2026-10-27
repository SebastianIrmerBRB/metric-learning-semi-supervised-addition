#!/usr/bin/env python3
"""What the SSL methods can see before any training: diagnostics on frozen DINOv2 features.

The backbone is frozen, so the geometry every method starts from is fixed. For
one fold of a seeded run this script rebuilds the fold's data exactly as
training did (recorded args, seeds and split), embeds the training pool and the
test set with the *identity* head (raw DINOv2 features) and measures, using the
hidden labels of the unlabeled pool for scoring only:

* **kNN graph purity** for several k: the share of same-class edges, and how
  often an unlabeled node has a correct labeled neighbour -- the ceiling of
  what graph methods (LP, LRML) can propagate at the start;
* **label propagation** (Iscen et al., with the run's own parameters when the
  run is an LP run, otherwise LP's defaults): unlabeled accuracy, and the
  accuracy/coverage trade-off over confidence thresholds on the method weight;
* **Hoffer's reference assignment**: accuracy of the argmax class and mean
  certainty, for the run's ``distance_scale`` and ``--hoffer-scales``;
* **k-means clustering** (the SLADE assumption) at several clusters-per-class
  ratios: NMI, ARI, pair precision/recall and cluster-size distribution;
* **embedding structure** of pool and test set (effective rank, alignment,
  uniformity, class separation), and the backbone's own test retrieval.

Comparing these across datasets (Cars196 / CUB / In-Shop) and budgets k is the
cheapest explanation of *why* the SSL gain differs between them.

It also writes ``backbone_test_<dataset>.npz`` (embeddings, labels), the
reference that ``embedding_geometry.py`` and ``per_query_gains.py`` accept via
``--backbone-npz``.

Needs the dataset images or the frozen-feature cache, and a GPU is recommended.

Outputs below ``--out/frozen``: <dataset>_k<k>_seed<s>_fold<f>.json,
summary.csv (all JSONs so far), lp_threshold_curve.png, backbone_test_<dataset>.npz.

Examples::

    python main_eval/frozen_feature_diagnostics.py --dataset cars --k 4 16 32 --folds 0
    python main_eval/frozen_feature_diagnostics.py --dataset in_shop --k 2 4 --combo iscen_ms --seed 2
    python main_eval/frozen_feature_diagnostics.py --fold-dir /mnt/f/server_logs/seeded_runs_cub/iscen_ms/.../fold_00/<ts>
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

PP = 100.0
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeded-roots", nargs="+", type=Path, default=common.SEEDED_ROOTS)
    parser.add_argument("--dataset", help="cars / cub / in_shop")
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--combo", default="iscen_ms",
                        help="whose recorded split and LP parameters to use (default: iscen_ms)")
    parser.add_argument("--seed", type=int, default=None, help="default: the first seed found")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="default: all folds")
    parser.add_argument("--fold-dir", type=Path, nargs="*", default=[], help="explicit fold directories instead")
    parser.add_argument("--knn", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--cluster-ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                        help="k-means clusters per class present in the pool")
    parser.add_argument("--hoffer-scales", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-test", action="store_true", help="skip the test-set embedding")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    return parser.parse_args()


# ------------------------------------------------------------- rebuild a fold


def rebuild_fold(fold_dir: Path, device: str, scratch: Path):
    """The fold's datasets and split, rebuilt from its run_config.json, plus an identity-head model."""

    from training import engine, semi_supervised
    from training.head_evaluation import make_args_from_run_config
    from training.ssl.config import SemiSupervisedConfig
    import utils

    run_config = common.read_json(fold_dir / "run_config.json")
    request = Namespace(device=device, explicit_cli_args=["device"], experiment_config_resolved={})
    args = make_args_from_run_config(run_config["args"], request)
    args.log_dir = args.save_dir = scratch
    ssl_fields = {f.name for f in dataclasses.fields(SemiSupervisedConfig)}
    ssl_config = SemiSupervisedConfig(**{k: v for k, v in run_config["ssl_config"].items() if k in ssl_fields})
    split_info = common.read_json(fold_dir / "split" / "split_info.json")
    cv_fold = split_info.get("dataset_split", {}).get("cv_fold", getattr(args, "cv_fold", 0))

    utils.seed_everything(args.seed, device=args.device)
    bundle, split = engine._prepare_datasets(
        args, ssl_config, cv_fold,
        precompute_frozen_features=engine.should_precompute_frozen_features(args, ssl_config),
        augmented_frozen_feature_precompute=False,
        frozen_feature_train_views=1,
    )
    labeled = np.load(fold_dir / "split" / "labeled_positions.npy")
    unlabeled = np.load(fold_dir / "split" / "unlabeled_positions.npy")
    if split is not None:
        same = np.array_equal(np.sort(labeled), np.sort(np.asarray(split.labeled_positions)))
        if not same:
            raise SystemExit(f"rebuilt labeled split differs from the recorded one in {fold_dir}")

    identity_args = copy.copy(args)
    identity_args.feat_dim = None
    none_config = dataclasses.replace(ssl_config, method="none")
    model = engine.make_run_model(identity_args, none_config).to(args.device)
    return args, ssl_config, bundle, labeled, unlabeled, model, cv_fold, run_config


def embed_pool(model, args, ssl_config, train_dataset, positions):
    from training import semi_supervised

    return np.asarray(semi_supervised.extract_embeddings(
        model=model, dataset=train_dataset, positions=positions,
        device=args.device, batch_size=ssl_config.embedding_batch_size or 256, num_workers=0,
        seed=int(args.seed), start_method=getattr(args, "dataloader_start_method", None),
        desc="frozen pool features",
    ), dtype=np.float32)


def embed_test(model, args, ssl_config, test_dataset):
    import torch
    from training import engine
    import utils

    loader = engine._make_eval_loader(
        args, model, test_dataset, "frozen test features",
        torch.device(utils.normalize_device_name(args.device)).type == "cuda",
        precompute_features=engine.should_precompute_frozen_features(args, ssl_config),
    )
    try:
        embeddings, labels = utils.extract_eval_embeddings(model, loader, "test", device=args.device)
    finally:
        utils.shutdown_dataloaders(loader)
    return to_numpy(embeddings, np.float32), to_numpy(labels, np.int64).reshape(-1)


def to_numpy(values, dtype):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


# ------------------------------------------------------------------ measures


def knn_measures(embeddings, labels, known, ks):
    neighbours = common.topk_neighbors(embeddings, max(ks))
    same = labels[neighbours] == labels[:, None]
    labeled_neighbour = known[neighbours]
    out = {}
    unlabeled = ~known
    for k in ks:
        out[f"knn{k}_purity"] = float(same[:, :k].mean())
        out[f"knn{k}_purity_unlabeled"] = float(same[unlabeled, :k].mean()) if unlabeled.any() else None
        has_correct = (same[:, :k] & labeled_neighbour[:, :k]).any(axis=1)
        out[f"knn{k}_unlabeled_with_correct_labeled_neighbour"] = float(has_correct[unlabeled].mean()) if unlabeled.any() else None
    return out


def lp_measures(embeddings, labels, known, params):
    from training.epoch_tracking import label_propagation_evidence

    probabilities, classes, certainty, weights = label_propagation_evidence(embeddings, labels, known, params)
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    reachable = probabilities.sum(axis=1) > 0
    correct = (predicted == labels) & reachable
    unlabeled = ~known
    curve = []
    for threshold in THRESHOLDS:
        accepted = unlabeled & reachable & (weights >= threshold)
        curve.append({
            "threshold": threshold,
            "coverage": float(accepted.sum() / max(unlabeled.sum(), 1)),
            "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        })
    return {
        "lp_params": params,
        "lp_unlabeled_accuracy": float(correct[unlabeled].mean()),
        "lp_unlabeled_reachable": float(reachable[unlabeled].mean()),
        "lp_certainty_mean": float(certainty[unlabeled].mean()),
        "lp_curve": curve,
    }


def hoffer_measures(embeddings, labels, known, scales):
    from training.epoch_tracking import hoffer_evidence

    out = {}
    unlabeled = ~known
    # Hoffer only assigns to labeled classes; score unlabeled samples of those classes.
    scored = unlabeled & np.isin(labels, np.unique(labels[known]))
    for scale in scales:
        probabilities, classes, certainty = hoffer_evidence(embeddings, labels, known, distance_scale=scale)
        predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
        tag = f"hoffer_s{scale:g}"
        out[f"{tag}_accuracy"] = float((predicted[scored] == labels[scored]).mean())
        out[f"{tag}_certainty_mean"] = float(certainty[scored].mean())
    return out


def cluster_measures(embeddings, labels, ratios, seed=0):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    unit = common.l2_normalize(embeddings)
    n_classes = len(np.unique(labels))
    out = {}
    for ratio in ratios:
        n_clusters = max(2, int(round(ratio * n_classes)))
        assignment = kmeans(unit, n_clusters, seed)
        contingency = {}
        for c, y in zip(assignment, labels):
            contingency[(c, y)] = contingency.get((c, y), 0) + 1
        pairs = lambda n: n * (n - 1) / 2  # noqa: E731
        same_pairs = sum(pairs(v) for v in contingency.values())
        cluster_sizes = np.bincount(assignment, minlength=n_clusters)
        class_sizes = np.unique(labels, return_counts=True)[1]
        tag = f"kmeans_r{ratio:g}"
        out[f"{tag}_clusters"] = n_clusters
        out[f"{tag}_nmi"] = float(normalized_mutual_info_score(labels, assignment))
        out[f"{tag}_ari"] = float(adjusted_rand_score(labels, assignment))
        out[f"{tag}_pair_precision"] = float(same_pairs / max(sum(pairs(n) for n in cluster_sizes), 1))
        out[f"{tag}_pair_recall"] = float(same_pairs / max(sum(pairs(n) for n in class_sizes), 1))
        out[f"{tag}_singleton_share"] = float(np.mean(cluster_sizes[cluster_sizes > 0] == 1))
        out[f"{tag}_largest_cluster_share"] = float(cluster_sizes.max() / len(labels))
    return out


def kmeans(unit, n_clusters, seed):
    try:
        import faiss

        model = faiss.Kmeans(unit.shape[1], n_clusters, niter=50, seed=seed, spherical=True, gpu=False, verbose=False)
        model.train(np.ascontiguousarray(unit))
        return model.index.search(np.ascontiguousarray(unit), 1)[1].reshape(-1)
    except ImportError:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, n_init=3, random_state=seed).fit_predict(unit)


# ---------------------------------------------------------------------- main


def fold_dirs_from_args(args):
    if args.fold_dir:
        return [(Path(d), None) for d in args.fold_dir]
    if not args.dataset or not args.k:
        raise SystemExit("give --fold-dir, or --dataset and --k")
    runs = common.select_runs(common.load_seeded_runs(args.seeded_roots), dataset=args.dataset, k=args.k, combos=[args.combo])
    if not runs:
        raise SystemExit(f"no {args.combo} runs for {args.dataset} k={args.k}")
    chosen = []
    for k in args.k:
        candidates = [r for r in runs if r.k == k and (args.seed is None or r.seed == args.seed)]
        if not candidates:
            print(f"[warn] no run for k={k}", file=sys.stderr)
            continue
        run = sorted(candidates, key=lambda r: r.seed)[0]
        dirs = run.fold_dirs()
        for index, fold_dir in enumerate(dirs):
            if args.folds is None or index in args.folds:
                chosen.append((fold_dir, run))
    return chosen


def main():
    args = parse_args()
    out = args.out / "frozen"
    from training.epoch_tracking import embedding_structure_metrics, label_propagation_params
    from training.ssl import graph_diagnostics

    targets = fold_dirs_from_args(args)
    print(f"{len(targets)} fold(s)")
    test_saved = set()
    for fold_dir, run in targets:
        with tempfile.TemporaryDirectory() as scratch:
            run_args, ssl_config, bundle, labeled, unlabeled, model, cv_fold, run_config = rebuild_fold(
                fold_dir, args.device, Path(scratch))
            positions = np.concatenate([labeled, unlabeled])
            labels = np.asarray(graph_diagnostics.dataset_labels_for_positions(bundle.train_dataset, positions), dtype=np.int64)
            known = np.arange(len(positions)) < len(labeled)
            recorded_counts = common.read_json(fold_dir / "split" / "split_info.json").get("labeled_label_counts") or {}
            if recorded_counts and sum(recorded_counts.values()) != int(known.sum()):
                raise SystemExit(f"labeled count {int(known.sum())} differs from split_info {sum(recorded_counts.values())}")
            dataset = run_args.dataset
            k = (run_config.get("ssl_config") or {}).get("labeled_per_class")
            seed = run_args.seed
            print(f"\n{dataset} k={k} seed={seed} fold={cv_fold}: {known.sum()} labeled + {(~known).sum()} unlabeled, "
                  f"{len(np.unique(labels))} classes")
            pool = embed_pool(model, run_args, ssl_config, bundle.train_dataset, positions)

            result = {
                "dataset": dataset, "k": k, "seed": seed, "fold": cv_fold, "source_fold_dir": str(fold_dir),
                "n_labeled": int(known.sum()), "n_unlabeled": int((~known).sum()), "n_classes": int(len(np.unique(labels))),
                "unlabeled_per_class_mean": float((~known).sum() / max(len(np.unique(labels)), 1)),
            }
            result.update(knn_measures(pool, labels, known, args.knn))
            lp_params = label_propagation_params(run_config["ssl_config"]) or {}
            result.update(lp_measures(pool, labels, known, lp_params))
            scales = list(args.hoffer_scales)
            own_scale = ((run_config["ssl_config"].get("method_params") or {}).get("regularizer_params") or {}).get("distance_scale")
            if own_scale is not None:
                scales.append(float(own_scale))
            result.update(hoffer_measures(pool, labels, known, sorted(set(scales))))
            result.update(cluster_measures(pool, labels, args.cluster_ratios))
            result.update({f"pool_{key}": value for key, value in embedding_structure_metrics(pool, labels).items()})

            if not args.no_test:
                test_embeddings, test_labels = embed_test(model, run_args, ssl_config, bundle.test_dataset)
                result.update({f"test_{key}": value for key, value in embedding_structure_metrics(test_embeddings, test_labels).items()})
                if dataset != "DeepFashionInShop":
                    retrieval = common.per_query_retrieval(test_embeddings, test_labels)
                    result["test_backbone_map_at_r"] = float(np.nanmean(retrieval["ap_at_r"]) * PP)
                    result["test_backbone_p_at_1"] = float(np.nanmean(retrieval["p_at_1"]) * PP)
                if dataset not in test_saved:
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / f"backbone_test_{dataset}.npz", embeddings=test_embeddings,
                                        labels=test_labels, dataset=np.array(dataset))
                    test_saved.add(dataset)

            name = f"{common.DATASET_SHORT.get(dataset, dataset)}_k{k}_seed{seed}_fold{cv_fold}.json"
            common.write_json(out / name, result)
            for key in ("knn10_purity", "knn10_unlabeled_with_correct_labeled_neighbour", "lp_unlabeled_accuracy",
                        "kmeans_r1_nmi", "kmeans_r1_pair_precision", "test_backbone_map_at_r"):
                if key in result:
                    print(f"  {key:<48} {result[key]:.4f}")
            del model
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    summarize(out)


def summarize(out):
    rows, curves = [], []
    for path in sorted(out.glob("*_k*_seed*_fold*.json")):
        result = common.read_json(path)
        for point in result.pop("lp_curve", []):
            curves.append({"dataset": result["dataset"], "k": result["k"], "fold": result["fold"], **point})
        result.pop("lp_params", None)
        rows.append(result)
    if not rows:
        return
    common.write_rows(out / "summary.csv", rows)
    common.write_rows(out / "lp_threshold_curve.csv", curves)
    pyplot = common.plt()
    fig, ax = pyplot.subplots(figsize=(4.6, 3.4))
    groups = {}
    for point in curves:
        groups.setdefault((point["dataset"], point["k"]), []).append(point)
    for (dataset, k), points in sorted(groups.items()):
        by_threshold = {}
        for p in points:
            if p["accuracy"] is not None:
                by_threshold.setdefault(p["threshold"], []).append((p["coverage"], p["accuracy"]))
        xs = [np.mean([c for c, _ in v]) * PP for _, v in sorted(by_threshold.items())]
        ys = [np.mean([a for _, a in v]) * PP for _, v in sorted(by_threshold.items())]
        ax.plot(xs, ys, "o-", ms=3, label=f"{common.DATASET_SHORT.get(dataset, dataset)} k={k}")
    ax.set_xlabel("coverage of unlabeled pool [%]")
    ax.set_ylabel("pseudo-label accuracy [%]")
    ax.set_title("LP on frozen features: accuracy vs. coverage\n(points: weight thresholds 0 … 0.95)", fontsize=8)
    ax.legend(fontsize=7)
    common.savefig(fig, out / "lp_threshold_curve.png")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
