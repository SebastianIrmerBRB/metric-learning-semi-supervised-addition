#!/usr/bin/env python3
"""Score D_test before any training, with each dataset's Phase-1 protocol.

Two reference points are reported per dataset:

* the pretrained DINOv2 features: the Phase-1 model with an identity
  projection. It is deterministic and independent of seed, fold and budget.
* the actual Phase-1 model at step 0: DINOv2 plus a linear projection head
  with PyTorch's default random initialisation (the orthogonal init applies
  to STML only). ``run_training`` seeds and then builds the model with no
  random draw in between, and every fold reuses the run's seed, so each seed
  has exactly one initial head. The script rebuilds it the same way and proves
  the reconstruction by replaying fold 0's logged epoch -1 validation score
  before it scores D_test with that head.

The replay must reproduce P@1 and MAP@R. MAP@R moves by up to ~1e-5 when the
retrieval orders float32 near-ties differently (In-Shop has many near-duplicate
images), so its tolerance is also tied to how far apart the seeds' own logged
scores are: a replay closer to a neighbouring seed than a quarter of that
distance would not identify the head and fails.

Inputs are supervised Phase-1 ``cv_summary.json`` files, one per seed and
dataset. The rescored aggregate fixes the reported measurements and Recall@K
ladder, so the baseline has exactly the columns of the thesis tables.

Example::

    python scripts/evaluate_untrained_baseline.py \
        /mnt/f/server_logs/seeded_runs_cars/sup_arc/*_k_16_*/final/supervised/cv_*/cv_summary.json \
        /mnt/f/server_logs/seeded_runs_in_shop/sup_arc/*_k_2_*/final/supervised/cv_*/cv_summary.json \
        --output scripts/reports/untrained_baseline
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import utils
from training import engine, semi_supervised
from training.head_evaluation import make_args_from_run_config
from training.io import read_json
from training.ssl.config import SemiSupervisedConfig

DISPLAY_NAMES = {"Cars196": "Cars196", "DeepFashionInShop": "In-Shop"}
PRIMARY = (("r_precision", "RP"), ("mean_average_precision_at_r", "MAP@R"), ("precision_at_1", "P@1"))
REPLAY_P1_TOLERANCE = 1e-3
REPLAY_NEIGHBOUR_FRACTION = 0.25


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold_dir(cv_summary_path, fold_index):
    """Map a fold's recorded log_dir (possibly from another machine) below this cv_* directory."""

    cv_dir = cv_summary_path.parent
    folds = read_json(cv_summary_path)["folds"]
    recorded = Path(next(f for f in folds if f["cv_fold"] == fold_index)["log_dir"]).parts
    return cv_dir.joinpath(*recorded[recorded.index(cv_dir.name) + 1 :])


def logged_initial_validation(directory):
    row = next(r for r in csv.DictReader(open(directory / "metrics.csv")) if r["epoch"] == "-1" and r["split"] == "valid")
    return float(row["mean_average_precision_at_r"]), float(row["precision_at_1"])


def load_run(cv_summary_path, device, scratch):
    summary = read_json(cv_summary_path)
    directory = fold_dir(cv_summary_path, 0)
    run_config = read_json(directory / "run_config.json")
    request = Namespace(device=device, explicit_cli_args=["device"], experiment_config_resolved={})
    args = make_args_from_run_config(run_config["args"], request)
    # Recorded paths belong to the training machine; nothing here may write there.
    args.log_dir = args.save_dir = scratch / f"seed_{args.seed}"
    aggregate = summary["aggregate"]
    # Report the rescored table of this dataset, not whatever the run originally recorded.
    args.test_measurements = sorted(aggregate["test_measurements"])
    args.test_recall_at_k = sorted(int(k) for k in aggregate["test_recall_at_k"])
    ssl_config = SemiSupervisedConfig(**run_config["ssl_config"])
    return args, ssl_config, directory


def build_initial_model(args, ssl_config):
    """The same draws, in the same order, as ``engine.run_training`` before its first update."""

    utils.seed_everything(args.seed, device=args.device)
    ssl_method = semi_supervised.create_method(ssl_config)
    regularizer = (
        ssl_method.make_regularizer(ssl_config)
        if ssl_method is not None and ssl_method.is_regularization_method
        else None
    )
    return engine.make_run_model(args, ssl_config, regularizer=regularizer).to(args.device)


def pin_memory(args):
    return torch.device(utils.normalize_device_name(args.device)).type == "cuda"


def replay_initial_validation(model, args, ssl_config):
    precompute = engine.should_precompute_frozen_features(args, ssl_config)
    bundle, _ = engine._prepare_datasets(
        args,
        ssl_config,
        0,
        precompute_frozen_features=precompute,
        augmented_frozen_feature_precompute=False,
        frozen_feature_train_views=1,
    )
    loader = engine._make_eval_loader(
        args, model, bundle.valid_dataset, "precompute frozen valid features", pin_memory(args), precompute_features=precompute
    )
    try:
        valid_device, _, _ = engine.resolve_run_evaluation_retrieval_devices(args, loader, None)
        outcome = utils.evaluate_split(
            model,
            loader,
            "valid",
            device=args.device,
            return_per_class=False,
            return_diagnostics=True,
            retrieval_device=valid_device,
            allow_cpu_fallback=False,
            embedding_residency=engine.get_evaluation_embedding_residency(args),
            recall_at_k=utils.normalize_recall_at_k(getattr(args, "recall_at_k", ()) or ()),
            measurements=utils.normalize_measurements(getattr(args, "measurements", None)),
        )
    finally:
        utils.shutdown_dataloaders(loader)
    return float(outcome.mean_average_precision_at_r), float(outcome.precision_at_1)


def score_test(model, test_loader, dataset, args, device):
    embeddings, labels = utils.extract_eval_embeddings(model, test_loader, "test", device=args.device)
    measurements = utils.resolve_test_measurements(args)
    recall_at_k = utils.resolve_test_recall_at_k(args)
    outcome = utils.unpack_evaluation_result(
        utils.evaluate_embeddings(
            embeddings,
            labels,
            name="test",
            return_per_class=False,
            dataset=dataset,
            device=device,
            allow_cpu_fallback=False,
            recall_at_k=recall_at_k,
            measurements=measurements,
            return_measurements=True,
        ),
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
    )
    metrics = {k: float(v) for k, v in outcome.measurements.items()}
    metrics.update({f"recall_at_{k}": float(v) for k, v in outcome.recall_at_k.items()})
    return metrics, len(embeddings)


def evaluate_dataset(cv_summaries, device, tolerance, scratch):
    runs = [(path, *load_run(path, device, scratch)) for path in cv_summaries]
    seeds = [args.seed for _, args, _, _ in runs]
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    logged_by_seed = {args.seed: logged_initial_validation(directory) for _, args, _, directory in runs}
    first_args, first_ssl = runs[0][1], runs[0][2]

    # One evaluation model and one D_test loader; D_test does not depend on seed or fold.
    eval_model = build_initial_model(first_args, first_ssl)
    bundle = utils.setup_dataset_bundle(
        first_args.dataset,
        seed=first_args.seed,
        data_split_seed=engine.get_data_split_seed(first_args),
        cv_k=first_args.cv_k,
        cv_fold=0,
        cv_mode=first_args.cv_mode,
        val_mode=first_args.val_mode,
        dataset_protocol=first_args.dataset_protocol,
        full_train=False,
        holdout_val_ratio=engine.get_holdout_val_ratio(first_args),
        image_resize_mode=engine.get_image_resize_mode(first_args),
    )
    test_loader = engine._make_eval_loader(
        first_args,
        eval_model,
        bundle.test_dataset,
        "precompute frozen test features",
        pin_memory(first_args),
        precompute_features=engine.should_precompute_frozen_features(first_args, first_ssl),
    )
    head = eval_model.projection_head()
    assert isinstance(head, torch.nn.Linear), f"expected one linear projection, got {head}"
    per_seed = []
    try:
        _, test_device, _ = engine.resolve_run_evaluation_retrieval_devices(first_args, None, test_loader)
        for path, args, ssl_config, directory in runs:
            seed_model = build_initial_model(args, ssl_config)
            logged = logged_by_seed[args.seed]
            replayed = replay_initial_validation(seed_model, args, ssl_config)
            map_difference, p1_difference = abs(logged[0] - replayed[0]), abs(logged[1] - replayed[1])
            nearest_other = min(abs(logged[0] - other[0]) for seed, other in logged_by_seed.items() if seed != args.seed)
            map_limit = min(tolerance, REPLAY_NEIGHBOUR_FRACTION * nearest_other)
            assert map_difference <= map_limit and p1_difference <= REPLAY_P1_TOLERANCE, (
                args.dataset, args.seed, logged, replayed, map_limit
            )
            eval_model.projection_head().load_state_dict(seed_model.projection_head().state_dict())
            metrics, n_test = score_test(eval_model, test_loader, bundle.test_dataset, first_args, test_device)
            per_seed.append({
                "seed": int(args.seed),
                "logged_epoch_minus1_valid_map_at_r_p1_fold0": list(logged),
                "replayed_epoch_minus1_valid_map_at_r_p1_fold0": list(replayed),
                "replay_map_at_r_abs_difference": map_difference,
                "replay_p1_abs_difference": p1_difference,
                "nearest_other_seed_logged_map_at_r_distance": nearest_other,
                "replay_map_at_r_limit": map_limit,
                "test": metrics,
                "source_cv_summary": str(path),
                "source_cv_summary_sha256": sha256(path),
                "source_run_config": str(directory / "run_config.json"),
                "source_run_config_sha256": sha256(directory / "run_config.json"),
            })
            del seed_model
            torch.cuda.empty_cache()
        with torch.no_grad():
            head.weight.copy_(torch.eye(head.weight.shape[0], dtype=head.weight.dtype, device=head.weight.device))
            head.bias.zero_()
        pretrained, n_test = score_test(eval_model, test_loader, bundle.test_dataset, first_args, test_device)
    finally:
        utils.shutdown_dataloaders(test_loader)

    query_gallery = utils.get_query_gallery_indices(bundle.test_dataset, n_test)
    names = sorted(pretrained)
    random_mean = {k: statistics.mean(s["test"][k] for s in per_seed) for k in names}
    random_std = {k: statistics.stdev(s["test"][k] for s in per_seed) for k in names}
    return {
        "dataset": first_args.dataset,
        "display_name": DISPLAY_NAMES.get(first_args.dataset, first_args.dataset),
        "dataset_protocol": first_args.dataset_protocol,
        "dino_size": first_args.dino_size,
        "feat_dim": first_args.feat_dim,
        "projection_layers": engine.get_projection_layers(first_args),
        "image_resize_mode": engine.get_image_resize_mode(first_args),
        "test_images": n_test,
        "query_images": None if query_gallery is None else int(len(query_gallery[0])),
        "gallery_images": None if query_gallery is None else int(len(query_gallery[1])),
        "test_retrieval_backend": engine._retrieval_backend_name(test_device),
        "pretrained_dinov2_test": pretrained,
        "initial_head_test_mean": random_mean,
        "initial_head_test_std_ddof1": random_std,
        "initial_head_per_seed": per_seed,
    }


def german(value):
    return f"{100 * value:.2f}".replace(".", "{,}")


def write_table(records, path):
    rows = []
    for index, r in enumerate(records):
        if index:
            rows.append(r"\addlinespace[3pt]")
        pretrained = [f"${german(r['pretrained_dinov2_test'][key])}$" for key, _ in PRIMARY]
        initial = [
            f"${german(r['initial_head_test_mean'][key])}\\pm{german(r['initial_head_test_std_ddof1'][key])}$" for key, _ in PRIMARY
        ]
        rows.append(" & ".join([r["display_name"], "DINOv2-Merkmale"] + pretrained) + r" \\")
        rows.append(" & ".join(["", "Initialer Kopf"] + initial) + r" \\")
    sizes = "; ".join(
        f"{r['display_name']}: {r['test_images']} Bilder" if r["query_images"] is None
        else f"{r['display_name']}: {r['query_images']} Anfragen, {r['gallery_images']} Galeriebilder"
        for r in records
    )
    caption = (
        r"Testleistung vor jedem Training [\%], gemessen mit dem Testprotokoll der Phase~1 ("
        + sizes
        + r"). DINOv2-Merkmale: vortrainierte ViT-B/14-Einbettungen ohne Projektion; unabhängig von Seed, Fold und Label-Budget. "
        r"Initialer Kopf: das Phase-1-Modell vor dem ersten Optimierungsschritt mit zufällig initialisiertem linearen Projektionskopf, "
        r"Mittelwert $\pm$ Stichprobenstandardabweichung über die fünf Seeds; innerhalb eines Seeds ist der Kopf für alle Folds, Verfahren und Budgets identisch."
    )
    header = ["Datensatz", "Ausgangsmodell"] + [label for _, label in PRIMARY]
    path.write_text("\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{tab:untrained-baseline}",
        r"\small", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cv_summaries", nargs="+", type=Path, help="supervised Phase-1 cv_summary.json files, one per seed and dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="largest accepted |logged - replayed| epoch -1 validation MAP@R; also capped at a quarter of the nearest other seed's distance",
    )
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)

    by_dataset = defaultdict(list)
    for path in options.cv_summaries:
        path = path.resolve()
        by_dataset[read_json(fold_dir(path, 0) / "run_config.json")["args"]["dataset"]].append(path)
    order = [name for name in DISPLAY_NAMES if name in by_dataset] + sorted(set(by_dataset) - set(DISPLAY_NAMES))
    with tempfile.TemporaryDirectory(prefix="untrained_baseline_") as scratch:
        records = [evaluate_dataset(by_dataset[name], options.device, options.tolerance, Path(scratch) / name) for name in order]

    payload = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "torch": torch.__version__, "records": records}
    (options.output / "untrained_baseline.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_table(records, options.output / "untrained_baseline.tex")
    for r in records:
        print(r["display_name"], "| DINOv2:", {k: round(100 * r["pretrained_dinov2_test"][k], 2) for k, _ in PRIMARY},
              "| initial head:", {k: f"{100 * r['initial_head_test_mean'][k]:.2f}±{100 * r['initial_head_test_std_ddof1'][k]:.2f}" for k, _ in PRIMARY},
              "| max replay MAP@R diff:", f"{max(s['replay_map_at_r_abs_difference'] for s in r['initial_head_per_seed']):.1e}")


if __name__ == "__main__":
    main()
