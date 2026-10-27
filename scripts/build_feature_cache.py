"""Fill the frozen-backbone feature cache a run manifest's replays will read, with a progress bar.

A frozen-backbone run extracts DINOv2 features for its training set into
data/<dataset>/backbone_cache/ before it trains: on the run's own dataloader
(num_workers is 0 in the studies), with no progress output, and committed only
once the whole set is done. Runs that train on the same images share one
matrix. For the external-pool replays (final_fit_mode=cross_validation_folds)
that is one matrix per CV fold and pool fraction, whatever the method, loss or
seed: the pool is appended to each fold's own training split.

This script builds each run's training sets with the replay's own code (the
study's args, the manifest's ablation variant, the first comparison scenario,
every fold), keeps one per distinct cache key, and fills the missing rows with
more loader workers. Rows are committed every --commit-rows images, so a killed
build resumes, and runs that start meanwhile reuse whatever is already there.
Run it from the directory the scheduler runs in (the repo root):

    python scripts/build_feature_cache.py configs/run_manifests/phase2/cars_full_external.json

Only study replays are handled. Validation and test sets are not touched: they
are the in-split sets the studies' own runs already cached.
"""

import argparse
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
import utils  # noqa: E402
from training import ablation, engine, run_scheduler, semi_supervised  # noqa: E402
from training.frozen_feature_cache import (  # noqa: E402
    _OPTIONAL_SOURCE_ARG_NAMES,
    _SOURCE_ARG_NAMES,
    make_frozen_feature_index_spec,
)

# Labels some ablation variant; what it changes shows up in the other arguments.
_SIGNATURE_IGNORED_ARGS = {"ablation_config", "ablation_variant"}
_SIGNATURE_MODEL_ARGS = ("dino_size", "backbone_tuning", "use_cache", "image_resize_mode", "image_size")
# SSL-config fields that decide which images are labeled; the method's hyperparameters do not.
_SIGNATURE_SSL_FIELD_MARKERS = ("label", "support", "unlabeled", "split", "seed", "fraction", "per_class")


def replay_final_args(run, config, scratch_dir):
    """The args and SSL config of ``run``'s first final_train replay, built like main.py builds them."""

    command = run_scheduler.build_run_command(run, config)
    request = main.parse_args_with_experiment_config(command[2:])
    base_args, hparam_config, study_result = main.load_hparam_study_for_final_test(request.final_test_study_dir)
    base_args = main.copy_final_test_request_args(request, base_args)
    ablation.prepare_ablation(base_args)
    overrides = set(getattr(base_args, "final_study_request_overrides", []))
    scenario_args = base_args
    if main.has_final_study_dir_replay_grid(overrides):
        main.reset_unrequested_final_study_grid_args(base_args, overrides)
        # The scenarios write their SSL configs below save_dir; keep them out of the results tree.
        base_args.save_dir = str(scratch_dir / run.name)
        scenarios = main.make_comparison_scenarios(
            base_args, main.load_args_ssl_config(base_args), grid_dir_name="study_replay_grid"
        )
        scenario_args = main.make_args_for_scenario(base_args, scenarios[0])
    if main.get_study_dir_mode(scenario_args) != main.STUDY_DIR_MODE_FINAL_TRAIN:
        return None, f"study_dir_mode={main.get_study_dir_mode(scenario_args)}"
    final_args, final_ssl_config = main.make_replay_args(scenario_args, hparam_config, study_result, "final")
    # The same settings run_single_final*_from_hparam gives each fit mode.
    fit_mode = main.get_final_fit_mode(scenario_args)
    if fit_mode == main.FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS:
        final_args.final_full_train = False
        folds = list(range(final_args.cv_k))
    else:
        final_args.final_full_train = fit_mode == main.FINAL_FIT_MODE_FULL_TRAIN
        final_args.cv_k = 1
        folds = [None]
    return (final_args, final_ssl_config, folds), None


def training_set_signature(args, ssl_config):
    names = [name for name in (*_SOURCE_ARG_NAMES, *_OPTIONAL_SOURCE_ARG_NAMES, *_SIGNATURE_MODEL_ARGS)
             if name not in _SIGNATURE_IGNORED_ARGS]
    ssl_fields = {
        key: value for key, value in ssl_config.to_dict().items()
        if any(marker in key for marker in _SIGNATURE_SSL_FIELD_MARKERS)
    }
    return repr(([(name, getattr(args, name, None)) for name in names], sorted(ssl_fields.items()),
                 bool(ssl_config.enabled)))


def frozen_training_set(args, ssl_config, cv_fold):
    """The dataset run_training precomputes features for, or None with the reason it has none."""

    if not (bool(args.use_cache) and engine.should_precompute_frozen_features(args, ssl_config)):
        return None, "no frozen-feature precompute"
    if not ssl_config.enabled and engine.get_frozen_feature_train_views(args) > 1:
        return None, "augmented views are keyed by extraction order"
    utils.seed_everything(args.seed, device=args.device)
    bundle, split = engine._prepare_datasets(
        args,
        ssl_config,
        cv_fold,
        precompute_frozen_features=True,
        augmented_frozen_feature_precompute=False,
        frozen_feature_train_views=1,
    )
    if ssl_config.enabled:
        return bundle.train_dataset, None
    if engine.is_supervised_mode(args):
        return semi_supervised.build_labeled_training_dataset(
            train_dataset=bundle.train_dataset,
            train_labels_mapper=bundle.train_labels_mapper,
            split=split,
        ), None
    return None, "no training-set precompute in this mode"


def missing_positions(model, spec, scheduled):
    """Dataset positions whose row is neither cached nor already ``scheduled`` by another view of the matrix."""

    rows = spec.row_indices
    valid = np.zeros(len(rows), dtype=bool)
    path = Path(model.cached_backbone_feature_matrix_path(cache_key=spec.key, cache_size=spec.capacity))
    if path.exists():
        try:
            matrix = np.load(path, mmap_mode="r")
            if matrix.shape[0] == spec.capacity:
                valid = np.isfinite(matrix[rows, 0])
        except (OSError, ValueError):
            pass
    first_occurrence = np.zeros(len(rows), dtype=bool)
    first_occurrence[np.unique(rows, return_index=True)[1]] = True
    positions = np.flatnonzero(first_occurrence & ~valid & ~scheduled[rows])
    scheduled[rows[positions]] = True
    return positions


def fill_cache(model, part, positions, args, progress, batch_size, num_workers, commit_rows):
    """Compute the rows at ``positions`` and commit them through the model's indexed cache."""

    spec = part["spec"]
    feature_dataset = utils.make_feature_transform_dataset(part["dataset"], require_feature_transform=True)
    device = torch.device(args.device)
    loader = DataLoader(
        Subset(feature_dataset, positions.tolist()),
        batch_size=batch_size,
        shuffle=False,
        **utils.make_dataloader_kwargs(
            num_workers, args.seed, args.dataloader_start_method,
            persistent_workers=False, pin_memory=device.type == "cuda",
        ),
    )
    pending_positions, pending_features = [], []

    def commit():
        if not pending_positions:
            return
        rows = spec.row_indices[np.concatenate(pending_positions)]
        features = torch.cat(pending_features)
        model.materialize_cached_backbone_features(
            cache_key=spec.key,
            cache_indices=rows,
            cache_size=spec.capacity,
            compute_missing=lambda local_positions: features[torch.as_tensor(local_positions)],
        )
        pending_positions.clear()
        pending_features.clear()

    start = 0
    try:
        # The same forward as utils._compute_backbone_feature_rows under the same autocast.
        with torch.no_grad(), utils.eval_autocast(device):
            for batch in loader:
                images = batch[0]
                pending_features.append(model.forward_backbone(images.to(device, non_blocking=True)).detach().float().cpu())
                pending_positions.append(positions[start:start + len(images)])
                start += len(images)
                progress.update(len(images))
                if start % commit_rows < len(images):
                    commit()
            commit()
    finally:
        utils.shutdown_dataloader_workers(loader)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path, help="Run manifest whose replays the cache is for.")
    parser.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="Loader processes decoding images (the runs use 0).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Backbone batch size (default: the runs' frozen_feature_batch_size).")
    parser.add_argument("--commit-rows", type=int, default=2048, help="Rows computed between two cache commits.")
    parser.add_argument("--device", default=None, help="Override the manifest's device, e.g. cuda:1.")
    parser.add_argument("--dry-run", action="store_true", help="List the caches and what is missing, compute nothing.")
    return parser.parse_args()


def main_cli():
    cli = parse_args()
    config = run_scheduler.load_scheduler_config(cli.manifest)
    # Every path the runs use (data/<dataset>/backbone_cache, data/CompCars, ...) is relative to here.
    os.chdir(config.working_dir)
    caches = OrderedDict()
    signatures = {}
    skipped = []
    models = {}
    first_args = None
    with tempfile.TemporaryDirectory(prefix="feature_cache_scenarios_") as scratch:
        scratch_dir = Path(scratch)
        runs = [run for run in config.runs if run.enabled]
        for run in tqdm(runs, desc="resolving runs", unit="run"):
            if run.study_dir is None:
                tqdm.write(f"skip {run.name}: not a study replay")
                continue
            resolved, reason = replay_final_args(run, config, scratch_dir)
            if resolved is None:
                tqdm.write(f"skip {run.name}: {reason}")
                continue
            args, ssl_config, folds = resolved
            if cli.dry_run:
                # A cache key never depends on the device, so listing needs no GPU.
                args.device = "cpu"
            elif cli.device is not None:
                args.device = cli.device
            for cv_fold in folds:
                signature = (cv_fold, training_set_signature(args, ssl_config))
                if signature in signatures:
                    if signatures[signature] is None:
                        skipped.append(run.name)
                    else:
                        caches[signatures[signature]]["runs"].append(run.name)
                    continue
                dataset, reason = frozen_training_set(args, ssl_config, cv_fold)
                if dataset is None:
                    signatures[signature] = None
                    tqdm.write(f"skip {run.name}: {reason}")
                    skipped.append(run.name)
                    continue
                if first_args is None:
                    first_args = args
                    main.configure_threads(args)
                    main.configure_float32_precision()
                    utils.set_eval_autocast_enabled(args.eval_amp)
                model_key = (args.dino_size, args.backbone_tuning, args.dataset)
                if model_key not in models:
                    models[model_key] = engine.make_run_model(args, ssl_config, regularizer=None).to(args.device).eval()
                spec = make_frozen_feature_index_spec(
                    args, models[model_key], dataset,
                    require_feature_transform=True, use_feature_transform=True, num_views=1,
                )
                signatures[signature] = spec.key
                # In-split runs share one matrix over the whole dataset, each fold filling its own rows.
                entry = caches.setdefault(spec.key, {"runs": [], "parts": {}, "args": args, "model_key": model_key})
                entry["parts"].setdefault(cv_fold, {"spec": spec, "dataset": dataset})
                entry["runs"].append(run.name)

    if skipped:
        print(f"{len(set(skipped))} runs need no training-set cache")
    if not caches:
        print("No frozen-feature cache to build.")
        return
    total_missing = 0
    print(f"\n{len(caches)} cache matrices:")
    for key, entry in caches.items():
        args = entry["args"]
        capacity = next(iter(entry["parts"].values()))["spec"].capacity
        scheduled = np.zeros(capacity, dtype=bool)
        rows = set()
        for part in entry["parts"].values():
            part["missing"] = missing_positions(models[entry["model_key"]], part["spec"], scheduled)
            rows.update(part["spec"].row_indices.tolist())
        missing = sum(len(part["missing"]) for part in entry["parts"].values())
        total_missing += missing
        if getattr(args, "unlabeled_source", "split") == "external":
            pool = f"{100 * (getattr(args, 'external_unlabeled_fraction', None) or 1.0):g} % pool"
        else:
            pool = "no pool"
        folds = ",".join("full" if fold is None else str(fold) for fold in entry["parts"])
        print(f"  {key[7:19]}  {args.dataset}  fold {folds:<7}  {pool:<11}  rows {len(rows):>7,}  "
              f"missing {missing:>7,}  ({len(set(entry['runs']))} runs)")
    if cli.dry_run or total_missing == 0:
        print("Nothing to compute." if total_missing == 0 else "Dry run: nothing computed.")
        return

    with tqdm(total=total_missing, desc="feature cache", unit="img", smoothing=0.05) as progress:
        for key, entry in caches.items():
            args = entry["args"]
            batch_size = cli.batch_size or engine.get_frozen_feature_batch_size(args)
            for part in entry["parts"].values():
                if len(part["missing"]) == 0:
                    continue
                progress.set_postfix_str(f"{key[7:19]} ({len(part['missing']):,} rows)")
                fill_cache(models[entry["model_key"]], part, part["missing"], args, progress,
                           batch_size, cli.num_workers, cli.commit_rows)
    print(f"done: {len(caches)} cache matrices filled")


if __name__ == "__main__":
    main_cli()
