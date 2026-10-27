"""JSON and CSV reporting for HPO replay and comparison workflows."""

import json
import math

import numpy as np
from loguru import logger

from .cli import (
    FINAL_EPOCH_AGGREGATION_MAX,
    FINAL_EPOCH_AGGREGATION_MEAN,
    FINAL_EPOCH_AGGREGATION_MEDIAN,
    FINAL_EPOCH_AGGREGATIONS,
    FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS,
    FINAL_FIT_MODE_EARLY_STOP_HOLDOUT,
    FINAL_FIT_MODE_FULL_TRAIN,
    STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL,
    STUDY_DIR_MODE_TRAIN_VAL,
)
from .io import namespace_to_dict, result_to_dict, write_csv, write_json


def _best_trial_source_results(attrs):
    """The best trial's per-run results: validation replays, CV folds, or itself."""

    validation_replay_results = attrs.get("validation_replay_results") or []
    if validation_replay_results:
        return validation_replay_results, attrs.get("validation_replay_source", "validation_replays")
    fold_results = attrs.get("fold_results") or []
    if fold_results:
        return fold_results, "cross_validation_folds"
    return [attrs], "single_validation_run"


def get_hpo_validation_retrieval_backend(study_result):
    """Return the one validation backend a final test must inherit."""

    attrs = study_result.best_user_attrs or {}
    source_results, _ = _best_trial_source_results(attrs)
    backends = {
        str(result["validation_retrieval_backend"]).lower()
        for result in source_results
        if result.get("validation_retrieval_backend") is not None
    }
    if not backends and attrs.get("validation_retrieval_backend") is not None:
        backends.add(str(attrs["validation_retrieval_backend"]).lower())
    invalid = backends - {"cpu", "cuda"}
    if invalid:
        raise ValueError(
            f"HPO validation contains invalid retrieval backends: {sorted(invalid)}"
        )
    if len(backends) > 1:
        raise ValueError(
            "HPO validation mixed CPU and CUDA retrieval; final test cannot "
            "inherit a single comparable backend"
        )
    if not backends:
        logger.warning(
            "HPO study predates validation retrieval metadata; "
            "defaulting final test retrieval to CPU"
        )
        return "cpu"
    return next(iter(backends))


def make_final_epoch_plan(study_result, aggregation=FINAL_EPOCH_AGGREGATION_MEAN):
    """Choose a fixed final training duration from the best trial's checkpoints.

    Every fold stopped at its own selected epoch, so one duration has to stand in
    for all of them. ``mean`` follows the average fold, ``median`` ignores a
    single fold that stopped far earlier or later than the rest, and ``max``
    trains for the longest fold's duration.
    """

    if aggregation not in FINAL_EPOCH_AGGREGATIONS:
        raise ValueError(
            f"final epoch aggregation must be one of {FINAL_EPOCH_AGGREGATIONS}: {aggregation}"
        )
    source_results, source = _best_trial_source_results(study_result.best_user_attrs or {})
    selected_epoch_key = "selected_epoch"
    if not source_results or any(result.get(selected_epoch_key) is None for result in source_results):
        selected_epoch_key = "last_epoch"
    selected_epochs = [
        int(result[selected_epoch_key])
        for result in source_results
        if result.get(selected_epoch_key) is not None
    ]
    if not selected_epochs:
        raise ValueError("Best HPO trial does not contain epoch information for final retraining")

    training_epoch_counts = [max(0, epoch + 1) for epoch in selected_epochs]
    mean_training_epochs = float(np.mean(training_epoch_counts))
    if aggregation == FINAL_EPOCH_AGGREGATION_MEAN:
        aggregated_training_epochs = mean_training_epochs
    elif aggregation == FINAL_EPOCH_AGGREGATION_MEDIAN:
        aggregated_training_epochs = float(np.median(training_epoch_counts))
    else:
        aggregated_training_epochs = float(max(training_epoch_counts))
    # max is already integral; mean and an even-length median are not, so both
    # round half up rather than truncating a fold's worth of training away.
    if aggregation == FINAL_EPOCH_AGGREGATION_MAX:
        rounding = "none"
        final_training_epochs = int(aggregated_training_epochs)
    else:
        rounding = "nearest_integer_half_up"
        final_training_epochs = int(math.floor(aggregated_training_epochs + 0.5))
    return {
        "source": source,
        "fit_mode": FINAL_FIT_MODE_FULL_TRAIN,
        "epoch_field": selected_epoch_key,
        "aggregation": aggregation,
        "selected_epoch_indices": selected_epochs,
        "training_epoch_counts": training_epoch_counts,
        "mean_training_epochs": mean_training_epochs,
        "aggregated_training_epochs": aggregated_training_epochs,
        "rounding": rounding,
        "final_training_epochs": final_training_epochs,
    }


def _unplanned_duration_plan(args, source, fit_mode, **details):
    """A final evaluation whose duration is not transferred from the study.

    Early stopping picks each tested checkpoint during the run itself, so no
    count is fixed up front; the plan records the budget the run may not exceed
    and the validation it stops on.
    """

    return {
        "source": source,
        "fit_mode": fit_mode,
        "epoch_field": None,
        "aggregation": None,
        "selected_epoch_indices": [],
        "training_epoch_counts": [],
        "mean_training_epochs": None,
        "aggregated_training_epochs": None,
        "rounding": "none",
        "final_training_epochs": None,
        "max_training_epochs": int(args.epochs),
        "patience": int(args.patience),
        # Only recorded when set, so plans of runs that never used it stay unchanged.
        **(
            {"patience_delay_epochs": int(args.patience_delay_epochs)}
            if getattr(args, "patience_delay_epochs", 0)
            else {}
        ),
        "val_mode": args.val_mode,
        "selection_metric": getattr(args, "selection_metric", None),
        **details,
    }


def make_final_cross_validation_plan(args):
    """Describe a final evaluation that tests every fold instead of one model.

    No extra model is fitted: each fold already early-stopped on its own
    validation partition, and it is that checkpoint which is tested. The
    reported score is the mean over folds, optionally alongside one evaluation
    of the folds' concatenated embeddings.
    """

    return _unplanned_duration_plan(
        args,
        "cross_validation_fold_models",
        FINAL_FIT_MODE_CROSS_VALIDATION_FOLDS,
        cv_k=int(args.cv_k),
        cv_mode=args.cv_mode,
        concatenated_fold_test=bool(getattr(args, "concatenated_fold_test", False)),
    )


def make_final_early_stopping_plan(args):
    """Describe a final fit whose duration early stopping decides on its own."""

    holdout_val_ratio = getattr(args, "holdout_val_ratio", None)
    return _unplanned_duration_plan(
        args,
        "final_early_stopping_holdout",
        FINAL_FIT_MODE_EARLY_STOP_HOLDOUT,
        holdout_val_ratio=None if holdout_val_ratio is None else float(holdout_val_ratio),
    )


def _best_attrs_columns(study_result, prefix=""):
    attrs = study_result.best_user_attrs or {}
    return {
        f"{prefix}best_valid_precision_at_1": optional_number(attrs.get("best_valid_precision_at_1")),
        f"{prefix}best_valid_mean_average_precision_at_r": optional_number(
            attrs.get("best_valid_mean_average_precision_at_r")
        ),
    }


def _projection_columns(result, prefix=""):
    return {
        f"{prefix}test_{layout}_{kind}": optional_path(getattr(result, f"test_{layout}_{kind}"))
        for layout in ("pacmap", "tsne")
        for kind in ("coordinates", "plot")
    }


def _test_columns(result):
    return {
        "final_log_dir": str(result.log_dir),
        "final_metrics_csv": str(result.metrics_csv),
        "final_train_loss": optional_number(result.final_train_loss),
        "epoch0_test_precision_at_1": optional_number(result.epoch0_test_precision_at_1),
        "epoch0_test_mean_average_precision_at_r": optional_number(
            result.epoch0_test_mean_average_precision_at_r
        ),
        "test_precision_at_1": optional_number(result.test_precision_at_1),
        "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
    }


def write_hparam_final_evaluation_summary(study_result, final_result, epoch_plan, role, summary_stem="final_evaluation"):
    """Write a compact study-to-final-test audit artifact."""

    summary = {
        "role": role,
        "study": hparam_study_result_to_dict(study_result),
        "epoch_plan": epoch_plan,
        "final_result": result_to_dict(final_result),
    }
    write_json(study_result.study_dir / f"{summary_stem}.json", summary)
    test_columns = _test_columns(final_result)
    row = {
        "role": role,
        "study_name": study_result.study_name,
        "best_trial_number": optional_number(study_result.best_trial_number),
        "best_hpo_value": optional_number(study_result.best_value),
        "best_params": json.dumps(study_result.best_params, sort_keys=True),
        **_best_attrs_columns(study_result),
        "epoch_source": epoch_plan["source"],
        "final_fit_mode": epoch_plan.get("fit_mode", FINAL_FIT_MODE_FULL_TRAIN),
        "epoch_field": optional_number(epoch_plan["epoch_field"]),
        "epoch_aggregation": optional_number(epoch_plan.get("aggregation")),
        "selected_epoch_indices": json.dumps(epoch_plan["selected_epoch_indices"]),
        "fold_training_epoch_counts": json.dumps(epoch_plan["training_epoch_counts"]),
        "mean_training_epochs": epoch_plan["mean_training_epochs"],
        "aggregated_training_epochs": optional_number(epoch_plan.get("aggregated_training_epochs")),
        "final_training_epochs": realized_final_training_epochs(epoch_plan, final_result),
        **test_columns,
        # Populated only by cross_validation_folds, where the reported score is
        # a mean over fold models rather than one model's score.
        **{
            name: optional_number(getattr(final_result, name, None))
            for name in (
                "test_precision_at_1_std",
                "test_mean_average_precision_at_r_std",
                "concatenated_test_precision_at_1",
                "concatenated_test_mean_average_precision_at_r",
                "concatenated_test_embedding_dim",
            )
        },
        "validation_retrieval_backend": final_result.validation_retrieval_backend,
        "test_retrieval_backend": final_result.test_retrieval_backend,
        **_projection_columns(final_result),
        "selected_epoch": final_result.selected_epoch,
        "global_step": final_result.global_step,
    }
    write_csv(study_result.study_dir / f"{summary_stem}.csv", [row])
    logger.info(f"Final HPO test evaluation summary written to {study_result.study_dir}")


def write_hparam_train_val_evaluation_summary(study_result, train_result, role, summary_stem="train_val_evaluation"):
    """Write a compact study-to-train/validation replay audit artifact."""

    summary = {
        "role": role,
        "study_dir_mode": STUDY_DIR_MODE_TRAIN_VAL,
        "study": hparam_study_result_to_dict(study_result),
        "train_val_result": result_to_dict(train_result),
    }
    write_json(study_result.study_dir / f"{summary_stem}.json", summary)
    row = {
        "role": role,
        "study_name": study_result.study_name,
        "study_dir_mode": STUDY_DIR_MODE_TRAIN_VAL,
        "best_trial_number": optional_number(study_result.best_trial_number),
        "best_hpo_value": optional_number(study_result.best_value),
        "best_params": json.dumps(study_result.best_params, sort_keys=True),
        **_best_attrs_columns(study_result, prefix="hpo_"),
        "train_val_log_dir": str(train_result.log_dir),
        "train_val_metrics_csv": str(train_result.metrics_csv),
        "train_val_best_valid_precision_at_1": optional_number(train_result.best_valid_precision_at_1),
        "train_val_best_valid_mean_average_precision_at_r": optional_number(
            train_result.best_valid_mean_average_precision_at_r
        ),
        "train_val_final_train_loss": optional_number(train_result.final_train_loss),
        "train_val_selected_epoch": train_result.selected_epoch,
        "train_val_last_epoch": train_result.last_epoch,
        "train_val_global_step": train_result.global_step,
        "validation_retrieval_backend": train_result.validation_retrieval_backend,
        "test_precision_at_1": optional_number(train_result.test_precision_at_1),
        "test_mean_average_precision_at_r": optional_number(train_result.test_mean_average_precision_at_r),
    }
    write_csv(study_result.study_dir / f"{summary_stem}.csv", [row])
    logger.info(f"Train/validation HPO replay summary written to {study_result.study_dir}")


def write_cross_seed_train_val_evaluation_summary(
    output_dir,
    summary_stem,
    role,
    study_result,
    scenarios,
    selection_metric,
    candidates,
    winner,
    epoch_plan,
    final_result,
):
    """Write candidate-by-seed validation evidence and the one final result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{summary_stem}.json"
    csv_path = output_dir / f"{summary_stem}.csv"
    write_json(
        json_path,
        {
            "role": role,
            "study_dir_mode": STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL,
            "study": hparam_study_result_to_dict(study_result),
            "selection_metric": selection_metric,
            "scenarios": [comparison_scenario_to_dict(scenario) for scenario in scenarios],
            "candidates": candidates,
            "winner_trial_number": winner["trial"]["trial_number"],
            "winner_mean_selection_value": winner["mean_selection_value"],
            "epoch_plan": epoch_plan,
            "final_result": result_to_dict(final_result),
        },
    )
    rows = []
    for candidate in candidates:
        selected = bool(candidate["selected_for_final"])
        runs = candidate["validation_runs"]
        rows.append(
            {
                "role": role,
                "study_name": study_result.study_name,
                "trial_number": candidate["trial"]["trial_number"],
                "original_hpo_value": optional_number(candidate["trial"].get("value")),
                "params": json.dumps(candidate["trial"]["params"], sort_keys=True),
                "selection_metric": selection_metric,
                "mean_selection_value": candidate["mean_selection_value"],
                "mean_best_valid_precision_at_1": candidate["mean_best_valid_precision_at_1"],
                "mean_best_valid_mean_average_precision_at_r": candidate[
                    "mean_best_valid_mean_average_precision_at_r"
                ],
                "comparison_seeds": json.dumps([run["comparison_seed"] for run in runs]),
                "per_seed_selection_values": json.dumps([run["selection_value"] for run in runs]),
                "per_seed_selected_epochs": json.dumps([run["result"]["selected_epoch"] for run in runs]),
                "selected_for_final": selected,
                "final_training_epochs": (
                    realized_final_training_epochs(epoch_plan, final_result) if selected else ""
                ),
                "test_precision_at_1": optional_number(
                    final_result.test_precision_at_1 if selected else None
                ),
                "test_mean_average_precision_at_r": optional_number(
                    final_result.test_mean_average_precision_at_r if selected else None
                ),
                "final_log_dir": str(final_result.log_dir) if selected else "",
            }
        )
    write_csv(csv_path, rows)
    logger.info(f"Cross-seed train/validation selection summary written to {output_dir}")
    return {"json": json_path, "csv": csv_path}


def _write_trial_evaluations_summary(study_result, evaluated, role, stem, request, description):
    """Write one summary for the final-test evaluations of several trials."""

    rows = []
    for item in evaluated:
        trial = item["trial"]
        final_result = item["final_result"]
        test_columns = _test_columns(final_result)
        rows.append(
            {
                "role": role,
                "study_name": study_result.study_name,
                "trial_number": trial["trial_number"],
                "hpo_value": optional_number(trial["value"]),
                "params": json.dumps(trial["params"], sort_keys=True),
                "summary_stem": item["summary_stem"],
                "selected_for_test": item.get("selected_for_test", True),
                "validation_selection_metric": item.get("validation_selection_metric", ""),
                "validation_selection_value": optional_number(item.get("validation_selection_value")),
                **test_columns,
                **_projection_columns(final_result),
                "selected_epoch": final_result.selected_epoch,
                "global_step": final_result.global_step,
            }
        )

    write_json(
        study_result.study_dir / f"{stem}.json",
        {
            "role": role,
            "study": hparam_study_result_to_dict(study_result),
            **request,
            "evaluations": [
                {
                    "trial": item["trial"],
                    "summary_stem": item["summary_stem"],
                    "selected_for_test": item.get("selected_for_test", True),
                    "validation_selection_metric": item.get("validation_selection_metric"),
                    "validation_selection_value": item.get("validation_selection_value"),
                    "final_result": result_to_dict(item["final_result"]),
                }
                for item in evaluated
            ],
        },
    )
    write_csv(study_result.study_dir / f"{stem}.csv", rows)
    logger.info(f"{description} final HPO test evaluation summary written to {study_result.study_dir}")


def write_hparam_top_final_evaluation_summary(study_result, evaluated, role, requested_top_n, summary_stem):
    """Write one summary for all top-N final-test evaluations."""

    _write_trial_evaluations_summary(
        study_result,
        evaluated,
        role,
        f"{summary_stem}_top_{requested_top_n}",
        {"requested_top_n": requested_top_n},
        f"Top-{requested_top_n}",
    )


def write_hparam_selected_final_evaluation_summary(study_result, evaluated, role, trial_numbers, summary_stem):
    """Write one summary for explicitly selected final-test evaluations."""

    _write_trial_evaluations_summary(
        study_result,
        evaluated,
        role,
        f"{summary_stem}_selected_trials",
        {"trial_numbers": trial_numbers},
        "Selected-trial",
    )


def write_comparison_summary(
    comparison_dir,
    args,
    scenario,
    ssl_config,
    supervised_study,
    ssl_study,
    supervised_final,
    ssl_final,
):
    write_csv(
        comparison_dir / "comparison_summary.csv",
        [
            make_comparison_row("supervised", supervised_study, supervised_final),
            make_comparison_row("ssl", ssl_study, ssl_final),
        ],
    )
    write_json(
        comparison_dir / "comparison_summary.json",
        {
            "args": namespace_to_dict(args),
            "scenario": comparison_scenario_to_dict(scenario),
            "split_ssl_config": ssl_config.to_dict(),
            "supervised_study": hparam_study_result_to_dict(supervised_study),
            "ssl_study": hparam_study_result_to_dict(ssl_study),
            "supervised_final": result_to_dict(supervised_final),
            "ssl_final": result_to_dict(ssl_final),
            "delta_ssl_minus_supervised": make_comparison_deltas(supervised_final, ssl_final),
        },
    )
    logger.info(f"Comparison summary written to {comparison_dir}")


COMPARISON_DELTA_METRICS = (
    "epoch0_test_precision_at_1",
    "epoch0_test_mean_average_precision_at_r",
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
    "best_valid_precision_at_1",
    "best_valid_mean_average_precision_at_r",
)


def make_comparison_deltas(supervised_final, ssl_final):
    return {
        name: subtract_optional(getattr(ssl_final, name), getattr(supervised_final, name))
        for name in COMPARISON_DELTA_METRICS
    }


def make_comparison_row(method, study_result, final_result):
    return {
        "method": method,
        "study_name": study_result.study_name,
        "study_dir": str(study_result.study_dir),
        "best_trial_number": optional_number(study_result.best_trial_number),
        "best_hpo_value": optional_number(study_result.best_value),
        "final_log_dir": str(final_result.log_dir),
        "best_valid_precision_at_1": final_result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": final_result.best_valid_mean_average_precision_at_r,
        **{
            name: optional_number(getattr(final_result, name))
            for name in COMPARISON_DELTA_METRICS[:4]
        },
        **_projection_columns(final_result),
        "final_training_epochs": max(0, final_result.last_epoch + 1),
        "selected_epoch": final_result.selected_epoch,
        "last_epoch": final_result.last_epoch,
        "global_step": final_result.global_step,
    }


def subtract_optional(left, right):
    if left is None or right is None:
        return None
    return float(left - right)


def hparam_study_result_to_dict(result):
    return {
        "study_name": result.study_name,
        "study_dir": str(result.study_dir),
        "trials_csv": str(result.trials_csv),
        "trials_jsonl": str(result.trials_jsonl),
        "best_trial_number": result.best_trial_number,
        "best_value": result.best_value,
        "best_params": result.best_params,
        "best_user_attrs": result.best_user_attrs,
        "completed_trials": result.completed_trials,
    }


def comparison_scenario_to_dict(scenario):
    return {
        "name": scenario.name,
        "labeled_fraction": scenario.labeled_fraction,
        "labeled_per_class": scenario.labeled_per_class,
        "comparison_seed": scenario.seed,
        "comparison_seed_targets": list(scenario.comparison_seed_targets),
        "run_seed": scenario.run_seed,
        "data_split_seed": scenario.data_split_seed,
        "support_seed": scenario.support_seed,
        "hparam_seed": scenario.hparam_seed,
        "label_sampling_mode": scenario.label_sampling_mode,
        "loss": scenario.loss,
        "miner": scenario.miner,
        "ssl_config_path": str(scenario.ssl_config_path),
    }


def _write_grid_summary(grid_dir, rows, aggregate_rows, description):
    grid_dir.mkdir(parents=True, exist_ok=True)
    write_csv(grid_dir / "grid_summary.csv", rows)
    write_csv(grid_dir / "grid_aggregate.csv", aggregate_rows)
    write_json(grid_dir / "grid_summary.json", {"runs": rows, "aggregates": aggregate_rows})
    logger.info(f"{description} summary written to {grid_dir}")


def write_comparison_grid_summary(grid_dir, grid_results):
    rows = [make_grid_summary_row(result) for result in grid_results]
    _write_grid_summary(grid_dir, rows, make_grid_aggregate_rows(rows), "Comparison grid")


def grid_recall_at_k(grid_results):
    """Collect every K any run in the grid reported, so all rows share columns."""

    recall_at_k = set()
    for grid_result in grid_results:
        result = grid_result.get("result")
        if result is None:
            continue
        for attr in ("best_valid_recall_at_k", "test_recall_at_k"):
            recall_at_k.update(getattr(result, attr, None) or {})
    return tuple(sorted(int(k) for k in recall_at_k))


def write_single_method_grid_summary(grid_dir, grid_results):
    recall_at_k = grid_recall_at_k(grid_results)
    rows = [
        make_single_method_grid_summary_row(result, recall_at_k=recall_at_k)
        for result in grid_results
    ]
    _write_grid_summary(grid_dir, rows, make_single_method_grid_aggregate_rows(rows), "Single-method grid")


def recall_at_k_columns(result, recall_at_k):
    """One column per requested K for the validation and the test evaluation."""

    best_valid = getattr(result, "best_valid_recall_at_k", None) or {}
    test = getattr(result, "test_recall_at_k", None) or {}
    return {
        **{
            f"best_valid_recall_at_{k}": optional_number(best_valid.get(k))
            for k in recall_at_k
        },
        **{f"test_recall_at_{k}": optional_number(test.get(k)) for k in recall_at_k},
    }


def scenario_columns(scenario):
    """The outer-grid coordinates every grid summary row starts with."""

    return {
        "scenario": scenario.name,
        "label_sampling_mode": scenario.label_sampling_mode,
        "labeled_fraction": scenario.labeled_fraction,
        "labeled_per_class": optional_number(scenario.labeled_per_class),
        "loss": scenario.loss,
        "miner": scenario.miner,
        "comparison_seed": scenario.seed,
        "comparison_seed_targets": ",".join(scenario.comparison_seed_targets),
        "run_seed": optional_number(scenario.run_seed),
        "data_split_seed": optional_number(scenario.data_split_seed),
        "support_seed": optional_number(scenario.support_seed),
        "hparam_seed": optional_number(scenario.hparam_seed),
    }


SINGLE_METHOD_GRID_RESULT_COLUMNS = (
    "best_trial_number",
    "best_hpo_value",
    "log_dir",
    "metrics_csv",
    "best_valid_precision_at_1",
    "best_valid_mean_average_precision_at_r",
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
    "test_pacmap_coordinates",
    "test_pacmap_plot",
    "test_tsne_coordinates",
    "test_tsne_plot",
    "epoch0_test_precision_at_1",
    "epoch0_test_mean_average_precision_at_r",
    "final_train_loss",
    "final_training_epochs",
    "selected_epoch",
    "global_step",
)


def make_single_method_grid_summary_row(grid_result, recall_at_k=()):
    study = grid_result["study"]
    result = grid_result["result"]
    row = {
        "method": grid_result["method"],
        **scenario_columns(grid_result["scenario"]),
        **dict.fromkeys(SINGLE_METHOD_GRID_RESULT_COLUMNS, ""),
        # Recall@K columns exist only when some run in the grid reported them,
        # and every row carries the same set so the CSV header stays rectangular.
        **{f"best_valid_recall_at_{k}": "" for k in recall_at_k},
        **{f"test_recall_at_{k}": "" for k in recall_at_k},
    }
    if study is not None:
        attrs = study.best_user_attrs or {}
        row.update(
            {
                "best_trial_number": optional_number(study.best_trial_number),
                "best_hpo_value": optional_number(study.best_value),
                "log_dir": str(attrs.get("log_dir", "")),
                "metrics_csv": str(attrs.get("metrics_csv", "")),
                **_best_attrs_columns(study),
                "test_precision_at_1": optional_number(attrs.get("test_precision_at_1")),
                "test_mean_average_precision_at_r": optional_number(attrs.get("test_mean_average_precision_at_r")),
            }
        )
    if result is not None:
        if study is None:
            row["best_valid_precision_at_1"] = result.best_valid_precision_at_1
            row["best_valid_mean_average_precision_at_r"] = result.best_valid_mean_average_precision_at_r
        test_columns = _test_columns(result)
        row.update(
            {
                "log_dir": test_columns.pop("final_log_dir"),
                "metrics_csv": test_columns.pop("final_metrics_csv"),
                **test_columns,
                **_projection_columns(result),
                "final_training_epochs": max(0, result.last_epoch + 1),
                "selected_epoch": result.selected_epoch,
                "global_step": result.global_step,
                **recall_at_k_columns(result, recall_at_k),
            }
        )
    return row


def aggregate_grid_rows(rows, group_keys, metric_names):
    """Mean and sample std of each metric over the rows sharing ``group_keys``."""

    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[name] for name in group_keys), []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items()):
        aggregate = {**dict(zip(group_keys, key)), "num_seeds": len(group_rows)}
        for metric_name in metric_names:
            values = [row[metric_name] for row in group_rows if row[metric_name] != ""]
            mean_value, std_value = mean_std(values)
            aggregate[f"{metric_name}_mean"] = optional_number(mean_value)
            aggregate[f"{metric_name}_std"] = optional_number(std_value)
        aggregate_rows.append(aggregate)
    return aggregate_rows


GRID_GROUP_KEYS = ("label_sampling_mode", "labeled_fraction", "labeled_per_class", "loss", "miner")


def make_single_method_grid_aggregate_rows(rows):
    metric_names = [
        "best_hpo_value",
        "best_valid_precision_at_1",
        "best_valid_mean_average_precision_at_r",
        "epoch0_test_precision_at_1",
        "epoch0_test_mean_average_precision_at_r",
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
        "final_train_loss",
        "final_training_epochs",
        # Recall@K columns are present only when the grid reported any.
        *(
            name
            for name in rows[0]
            if name.startswith(("best_valid_recall_at_", "test_recall_at_"))
        ),
    ]
    return aggregate_grid_rows(rows, ("method", *GRID_GROUP_KEYS), metric_names)


def make_grid_summary_row(result):
    supervised_final = result["supervised_final"]
    ssl_final = result["ssl_final"]
    deltas = result["deltas"]
    row = {**scenario_columns(result["scenario"]), "comparison_dir": str(result["comparison_dir"])}

    def add_triplet(name, wrap):
        row[f"supervised_{name}"] = wrap(getattr(supervised_final, name))
        row[f"ssl_{name}"] = wrap(getattr(ssl_final, name))
        row[f"delta_{name}"] = optional_number(deltas[name])

    for name in COMPARISON_DELTA_METRICS[:4]:
        add_triplet(name, optional_number)
    for layout in ("pacmap", "tsne"):
        for prefix, final in (("supervised", supervised_final), ("ssl", ssl_final)):
            for kind in ("coordinates", "plot"):
                row[f"{prefix}_test_{layout}_{kind}"] = optional_path(getattr(final, f"test_{layout}_{kind}"))
    for name in COMPARISON_DELTA_METRICS[4:]:
        add_triplet(name, lambda value: value)
    return row


def make_grid_aggregate_rows(rows):
    metric_names = [
        f"{prefix}_{name}"
        for name in COMPARISON_DELTA_METRICS
        for prefix in ("supervised", "ssl", "delta")
    ]
    return aggregate_grid_rows(rows, GRID_GROUP_KEYS, metric_names)


def mean_std(values):
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    std = 0.0 if len(array) == 1 else float(np.std(array, ddof=1))
    return float(np.mean(array)), std


def optional_number(value):
    return "" if value is None else value


def realized_final_training_epochs(epoch_plan, final_result):
    """Report the duration the final fit actually ran for.

    ``full_train`` fixes that duration in advance, while ``early_stop_holdout``
    only learns it once early stopping fires, so the run itself supplies it.
    """

    planned_epochs = epoch_plan.get("final_training_epochs")
    if planned_epochs is not None:
        return int(planned_epochs)
    last_epoch = getattr(final_result, "last_epoch", None)
    return "" if last_epoch is None else max(0, int(last_epoch) + 1)


def optional_path(value):
    return "" if value is None else str(value)
