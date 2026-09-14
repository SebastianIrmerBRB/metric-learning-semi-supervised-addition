"""Measure D_test again from saved projection heads, without training.

``--save_head_state`` writes the validation-selected head of every run next to
its logs. With a frozen backbone that head is the whole fitted model, so a
second test evaluation needs no optimizer, no training data, and no epochs:
rebuild the run's model, load its head, embed D_test, and score it. Pointed at
a cross-validation directory this reproduces the complete report of the
protocol from "A Metric Learning Reality Check" -- every fold's own score,
their mean and sample standard deviation, and the concatenated-embedding
evaluation over all folds.
"""

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import utils
from . import engine, semi_supervised
from .cli import parser
from .io import read_json, write_json
from .ssl.config import SemiSupervisedConfig

SAVED_HEAD_EVALUATION_STEM = "saved_head_evaluation"

# Where the work runs is a property of the current request, not of the finished
# run: everything else is replayed from the run's own recorded configuration so
# the reloaded head sees exactly the test set it was measured on.
RUNTIME_REQUEST_OVERRIDES = (
    "device",
    "ssl_device",
    "num_workers",
    "num_threads",
    "dataloader_start_method",
    "eval_amp",
    "use_cache",
    "evaluation_embedding_residency",
    "frozen_feature_residency",
    "frozen_feature_residency_max_gb",
    "frozen_feature_batch_size",
    "low_memory_mode",
    "faiss_temp_memory_mb",
    # What to report is a property of the request too: measuring a saved head
    # again is exactly when a wider test table is wanted than the original run
    # recorded.
    "measurements",
    "recall_at_k",
    "report_test_metrics",
    "test_measurements",
    "test_recall_at_k",
)


@dataclass(frozen=True)
class SavedHeadRun:
    """One finished run whose head can be reloaded and measured again."""

    log_dir: Path
    head_state_path: Path
    args: Any
    ssl_config: SemiSupervisedConfig
    cv_fold: int | None


def run_saved_head_evaluation(request_args):
    """Evaluate D_test from every saved head under the requested directory."""

    directory = resolve_saved_head_directory(request_args.evaluate_saved_heads)
    run_dirs = find_saved_head_runs(directory)
    logger.info(
        f"Evaluating {len(run_dirs)} saved projection head(s) under {directory} "
        "without retraining"
    )

    fold_records = []
    embedding_sets = []
    for run_dir in run_dirs:
        run = load_saved_head_run(run_dir, request_args)
        record, embedding_set = evaluate_saved_head_run(run)
        fold_records.append(record)
        embedding_sets.append(embedding_set)

    summary = summarize_saved_head_evaluation(fold_records, embedding_sets)
    summary["source_directory"] = str(directory)
    write_saved_head_evaluation(directory, summary)
    return summary


def find_saved_head_runs(directory):
    """Return the run directories the requested path refers to, in fold order.

    A cross-validation directory is described by its own ``cv_summary.json``,
    which fixes the fold order the concatenated evaluation depends on. A single
    run is identified by its saved head. Either may sit below the given path,
    which lets a replay's ``final/<role>`` directory be named directly.
    """

    directory = resolve_saved_head_directory(directory)

    cv_summaries = sorted(directory.rglob("cv_summary.json"))
    if len(cv_summaries) > 1:
        raise ValueError(
            f"{directory} contains {len(cv_summaries)} cross-validation runs; name one of "
            f"them directly: {[str(path.parent) for path in cv_summaries]}"
        )
    if cv_summaries:
        return read_cv_summary_run_dirs(cv_summaries[0])

    head_states = sorted(directory.rglob(engine.HEAD_STATE_FILENAME))
    if not head_states:
        raise FileNotFoundError(
            f"No {engine.HEAD_STATE_FILENAME} exists under {directory}. Only runs launched "
            "with --save_head_state write one"
        )
    if len(head_states) > 1:
        raise ValueError(
            f"{directory} contains {len(head_states)} saved heads that belong to no single "
            f"cross-validation run; name one of them directly: "
            f"{[str(path.parent) for path in head_states]}"
        )
    return [head_states[0].parent]


def resolve_saved_head_directory(directory):
    """Accept a path relative to the repository root or to ``logs/``."""

    directory = Path(directory)
    if directory.is_dir():
        return directory
    below_logs = Path("logs") / directory
    if below_logs.is_dir():
        return below_logs
    raise FileNotFoundError(
        f"Saved-head directory not found: {directory} or {below_logs}"
    )


def read_cv_summary_run_dirs(cv_summary_path):
    """Return one cross-validation run's fold directories in fold order."""

    summary = read_json(cv_summary_path)
    folds = summary.get("folds") or []
    if not folds:
        raise ValueError(f"{cv_summary_path} records no completed folds")
    run_dirs = []
    for fold in folds:
        log_dir = Path(fold["log_dir"])
        head_state_path = log_dir / engine.HEAD_STATE_FILENAME
        if not head_state_path.exists():
            raise FileNotFoundError(
                f"Fold {fold.get('cv_fold')} of {cv_summary_path.parent} saved no "
                f"{engine.HEAD_STATE_FILENAME}; only runs launched with --save_head_state can "
                "be evaluated again without training"
            )
        run_dirs.append(log_dir)
    return run_dirs


def load_saved_head_run(run_dir, request_args):
    """Rebuild one finished run's configuration from its own artifacts."""

    run_dir = Path(run_dir)
    run_config = read_json(run_dir / "run_config.json")
    args = make_args_from_run_config(run_config["args"], request_args)
    ssl_config = SemiSupervisedConfig(**run_config["ssl_config"])
    return SavedHeadRun(
        log_dir=run_dir,
        head_state_path=run_dir / engine.HEAD_STATE_FILENAME,
        args=args,
        ssl_config=ssl_config,
        cv_fold=run_config["args"].get("cv_fold"),
    )


def make_args_from_run_config(recorded_args, request_args):
    """Restore a run's namespace, letting the request redirect where it runs."""

    args = parser.parse_args([])
    for name, value in recorded_args.items():
        setattr(args, name, value)
    args.log_dir = Path(recorded_args["log_dir"])
    args.save_dir = Path(recorded_args["save_dir"])

    request_overrides = set(getattr(request_args, "explicit_cli_args", None) or []) | set(
        getattr(request_args, "experiment_config_resolved", None) or {}
    )
    for name in RUNTIME_REQUEST_OVERRIDES:
        if name in request_overrides and hasattr(request_args, name):
            setattr(args, name, getattr(request_args, name))
    if "device" in request_overrides and "ssl_device" not in request_overrides:
        args.ssl_device = args.device

    # Retrieval ran on the backend validation picked, and the recorded value is
    # the only way a reloaded head lands on the same one.
    args.validation_retrieval_backend = recorded_args.get(
        "validation_retrieval_backend_resolved"
    )
    return args


def evaluate_saved_head_run(run):
    """Embed and score D_test with one run's reloaded projection head."""

    args = run.args
    ssl_config = run.ssl_config
    payload = engine.load_head_state(run.head_state_path)
    metadata = payload.get("metadata") or {}
    logger.info(
        f"Loading the head {run.head_state_path} selected at epoch "
        f"{metadata.get('selected_epoch')} of fold {metadata.get('cv_fold')}"
    )

    ssl_method = semi_supervised.create_method(ssl_config)
    regularizer = (
        ssl_method.make_regularizer(ssl_config)
        if ssl_method is not None and ssl_method.is_regularization_method
        else None
    )
    model = engine.make_run_model(args, ssl_config, regularizer=regularizer)
    model = model.to(args.device)
    engine.restore_head_state(model, payload)

    dataset_bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        data_split_seed=engine.get_data_split_seed(args),
        cv_k=args.cv_k if run.cv_fold is not None else 1,
        cv_fold=run.cv_fold,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
        dataset_protocol=args.dataset_protocol,
        cifar_imbalance_factor=args.cifar_imbalance_factor,
        cifar_train_fraction=args.cifar_train_fraction,
        cifar_test_fraction=args.cifar_test_fraction,
        full_train=bool(getattr(args, "final_full_train", False)),
        holdout_val_ratio=engine.get_holdout_val_ratio(args),
        image_resize_mode=engine.get_image_resize_mode(args),
    )

    pin_memory = torch.device(utils.normalize_device_name(args.device)).type == "cuda"
    test_loader = engine._make_eval_loader(
        args,
        model,
        dataset_bundle.test_dataset,
        "precompute frozen test features",
        pin_memory,
        precompute_features=engine.should_precompute_frozen_features(args, ssl_config),
    )
    try:
        _, test_device, _ = engine.resolve_run_evaluation_retrieval_devices(
            args,
            None,
            test_loader,
        )
        test_embeddings, test_labels = utils.extract_eval_embeddings(
            model,
            test_loader,
            "test",
            device=args.device,
        )
        measurements = utils.resolve_test_measurements(args)
        recall_at_k = utils.resolve_test_recall_at_k(args)
        outcome = utils.unpack_evaluation_result(
            utils.evaluate_embeddings(
                test_embeddings,
                test_labels,
                name="test",
                return_per_class=False,
                dataset=dataset_bundle.test_dataset,
                device=test_device,
                allow_cpu_fallback=False,
                recall_at_k=recall_at_k,
                measurements=measurements,
                return_measurements=True,
            ),
            recall_at_k=recall_at_k,
            measurements=measurements,
            return_measurements=True,
        )
        precision_at_1 = outcome.precision_at_1
        mean_average_precision_at_r = outcome.mean_average_precision_at_r
    finally:
        utils.shutdown_dataloaders(test_loader)

    embeddings = engine.to_numpy_embeddings(test_embeddings)
    labels = engine.to_numpy_embeddings(test_labels).reshape(-1)
    query_gallery_indices = utils.get_query_gallery_indices(
        dataset_bundle.test_dataset,
        len(embeddings),
    )
    embedding_set = {
        "embeddings": embeddings,
        "labels": labels,
        "query_indices": None if query_gallery_indices is None else query_gallery_indices[0],
        "gallery_indices": None if query_gallery_indices is None else query_gallery_indices[1],
    }
    record = {
        "log_dir": str(run.log_dir),
        "head_state_path": str(run.head_state_path),
        "cv_fold": run.cv_fold,
        "cv_k": args.cv_k,
        "cv_mode": args.cv_mode,
        "selected_epoch": metadata.get("selected_epoch"),
        "seed": metadata.get("seed"),
        "support_seed": metadata.get("support_seed"),
        "test_precision_at_1": float(precision_at_1),
        "test_mean_average_precision_at_r": float(mean_average_precision_at_r),
        "test_measurements": dict(outcome.measurements),
        "test_recall_at_k": dict(outcome.recall_at_k),
        "test_retrieval_backend": engine._retrieval_backend_name(test_device),
        "embedding_dim": int(embeddings.shape[1]),
    }
    logger.info(
        f"Saved head {run.head_state_path}: precision_at_1={precision_at_1:.6f}, "
        f"mean_average_precision_at_r={mean_average_precision_at_r:.6f}"
        + "".join(
            f", {name}={value:.6f}"
            for name, value in outcome.measurements.items()
            if name not in utils.BASE_RETRIEVAL_METRICS
        )
        + "".join(
            f", recall_at_{k}={value:.6f}"
            for k, value in outcome.recall_at_k.items()
        )
    )
    return record, embedding_set


def summarize_saved_head_evaluation(fold_records, embedding_sets):
    """Aggregate the per-run scores the way the original protocol reports them."""

    test_measurements = mean_measurements(fold_records, "test_measurements")
    test_measurements_std = std_measurements(fold_records, "test_measurements")
    test_recall_at_k = mean_measurements(fold_records, "test_recall_at_k")
    test_recall_at_k_std = std_measurements(fold_records, "test_recall_at_k")
    summary = {
        "runs": fold_records,
        "test_precision_at_1": mean_metric(fold_records, "test_precision_at_1"),
        "test_mean_average_precision_at_r": mean_metric(
            fold_records,
            "test_mean_average_precision_at_r",
        ),
        "test_precision_at_1_std": std_metric(fold_records, "test_precision_at_1"),
        "test_mean_average_precision_at_r_std": std_metric(
            fold_records,
            "test_mean_average_precision_at_r",
        ),
        "concatenated_test_precision_at_1": None,
        "concatenated_test_mean_average_precision_at_r": None,
        "concatenated_test_embedding_dim": None,
        "test_measurements": test_measurements,
        "test_measurements_std": test_measurements_std,
        "test_recall_at_k": test_recall_at_k or None,
        "test_recall_at_k_std": test_recall_at_k_std or None,
        "concatenated_test_measurements": None,
        "concatenated_test_recall_at_k": None,
    }
    if len(embedding_sets) < 2:
        # One model has nothing to concatenate with; its own score is the result.
        return summary

    concatenated_device = torch.device(fold_records[0]["test_retrieval_backend"])
    concatenated, labels, query_indices, gallery_indices = (
        engine.concatenate_test_embedding_sets(embedding_sets)
    )
    evaluation_dataset = engine.ConcatenatedEvaluationDataset(query_indices, gallery_indices)
    measurements = tuple(test_measurements) or utils.DEFAULT_MEASUREMENTS
    recall_at_k = utils.normalize_recall_at_k(tuple(test_recall_at_k))
    outcome = utils.unpack_evaluation_result(
        utils.evaluate_embeddings(
            concatenated,
            labels,
            name="concatenated fold test",
            return_per_class=False,
            dataset=evaluation_dataset,
            device=concatenated_device,
            allow_cpu_fallback=False,
            recall_at_k=recall_at_k,
            measurements=measurements,
            return_measurements=True,
        ),
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
    )
    precision_at_1 = outcome.precision_at_1
    mean_average_precision_at_r = outcome.mean_average_precision_at_r
    summary["concatenated_test_precision_at_1"] = float(precision_at_1)
    summary["concatenated_test_mean_average_precision_at_r"] = float(
        mean_average_precision_at_r
    )
    summary["concatenated_test_embedding_dim"] = int(concatenated.shape[1])
    summary["concatenated_test_measurements"] = dict(outcome.measurements)
    summary["concatenated_test_recall_at_k"] = dict(outcome.recall_at_k) or None
    logger.info(
        f"Concatenated saved-head evaluation ({concatenated.shape[1]}-dim from "
        f"{len(embedding_sets)} runs): precision_at_1={precision_at_1:.6f}, "
        f"mean_average_precision_at_r={mean_average_precision_at_r:.6f}"
        + "".join(
            f", {name}={value:.6f}"
            for name, value in outcome.measurements.items()
            if name not in utils.BASE_RETRIEVAL_METRICS
        )
        + "".join(
            f", recall_at_{k}={value:.6f}"
            for k, value in outcome.recall_at_k.items()
        )
    )
    return summary


def mean_metric(records, name):
    values = [record[name] for record in records if record[name] is not None]
    return None if not values else float(statistics.fmean(values))


def std_metric(records, name):
    values = [record[name] for record in records if record[name] is not None]
    return None if len(values) < 2 else float(statistics.stdev(values))


def _record_measurements(record, name):
    recorded = record.get(name)
    values = dict(recorded or {})
    if name == "test_measurements" and recorded is None:
        legacy = {
            utils.MEASUREMENT_PRECISION_AT_1: record.get("test_precision_at_1"),
            utils.MEASUREMENT_MAP_AT_R: record.get(
                "test_mean_average_precision_at_r"
            ),
        }
        for measurement, value in legacy.items():
            if value is not None:
                values.setdefault(measurement, float(value))
    return values


def mean_measurements(records, name):
    mappings = [_record_measurements(record, name) for record in records]
    keys = sorted({key for mapping in mappings for key in mapping})
    return {
        key: float(statistics.fmean(mapping[key] for mapping in mappings if key in mapping))
        for key in keys
    }


def std_measurements(records, name):
    mappings = [_record_measurements(record, name) for record in records]
    spread = {}
    for key in sorted({key for mapping in mappings for key in mapping}):
        values = [mapping[key] for mapping in mappings if key in mapping]
        if len(values) >= 2:
            spread[key] = float(statistics.stdev(values))
    return spread


def _recall_at_k_values(mapping):
    """Return a Recall@K mapping keyed by int, however it was recorded.

    A summary that has been through JSON carries string keys, so the CSV
    lookups normalize instead of silently missing every column.
    """

    return {int(k): value for k, value in (mapping or {}).items()}


def write_saved_head_evaluation(directory, summary):
    """Write the reloaded-head report beside the runs it measured."""

    directory = Path(directory)
    write_json(directory / f"{SAVED_HEAD_EVALUATION_STEM}.json", summary)

    additional = utils.additional_measurements(
        tuple(summary.get("test_measurements") or utils.DEFAULT_MEASUREMENTS)
    )
    recall_at_k = utils.normalize_recall_at_k(
        tuple(summary.get("test_recall_at_k") or ())
    )
    fieldnames = [
        "cv_fold",
        "log_dir",
        "selected_epoch",
        "seed",
        "support_seed",
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
        *(f"test_{name}" for name in additional),
        *(f"test_recall_at_{k}" for k in recall_at_k),
        "test_retrieval_backend",
        "embedding_dim",
    ]
    csv_path = directory / f"{SAVED_HEAD_EVALUATION_STEM}.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary["runs"]:
            measurement_values = _record_measurements(record, "test_measurements")
            recall_values = _recall_at_k_values(record.get("test_recall_at_k"))
            writer.writerow(
                {
                    **{name: record.get(name) for name in fieldnames},
                    **{
                        f"test_{name}": measurement_values.get(name)
                        for name in additional
                    },
                    **{
                        f"test_recall_at_{k}": recall_values.get(k)
                        for k in recall_at_k
                    },
                }
            )
        writer.writerow(
            {
                "cv_fold": "mean",
                "test_precision_at_1": summary["test_precision_at_1"],
                "test_mean_average_precision_at_r": summary[
                    "test_mean_average_precision_at_r"
                ],
                **{
                    f"test_{name}": (summary.get("test_measurements") or {}).get(name)
                    for name in additional
                },
                **{
                    f"test_recall_at_{k}": _recall_at_k_values(
                        summary.get("test_recall_at_k")
                    ).get(k)
                    for k in recall_at_k
                },
            }
        )
        if summary["concatenated_test_embedding_dim"] is not None:
            writer.writerow(
                {
                    "cv_fold": "concatenated",
                    "test_precision_at_1": summary["concatenated_test_precision_at_1"],
                    "test_mean_average_precision_at_r": summary[
                        "concatenated_test_mean_average_precision_at_r"
                    ],
                    **{
                        f"test_recall_at_{k}": _recall_at_k_values(
                            summary.get("concatenated_test_recall_at_k")
                        ).get(k)
                        for k in recall_at_k
                    },
                    **{
                        f"test_{name}": (
                            summary.get("concatenated_test_measurements") or {}
                        ).get(name)
                        for name in additional
                    },
                    "embedding_dim": summary["concatenated_test_embedding_dim"],
                }
            )
    logger.info(f"Saved-head evaluation written to {csv_path}")
    return csv_path
