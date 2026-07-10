"""JSON and CSV reporting for HPO replay and comparison workflows."""

import csv
import json
import math

import numpy as np
from loguru import logger

from experiment_cli import STUDY_DIR_MODE_TRAIN_VAL
from experiment_io import namespace_to_dict, result_to_dict, write_json


def make_final_epoch_plan(study_result):
    """Choose a fixed final training duration from the best trial's checkpoints."""

    attrs = study_result.best_user_attrs or {}
    fold_results = attrs.get("fold_results") or []
    source_results = fold_results if fold_results else [attrs]
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
    final_training_epochs = int(math.floor(mean_training_epochs + 0.5))
    return {
        "source": "cross_validation_folds" if fold_results else "single_validation_run",
        "epoch_field": selected_epoch_key,
        "selected_epoch_indices": selected_epochs,
        "training_epoch_counts": training_epoch_counts,
        "mean_training_epochs": mean_training_epochs,
        "rounding": "nearest_integer_half_up",
        "final_training_epochs": final_training_epochs,
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
    csv_path = study_result.study_dir / f"{summary_stem}.csv"
    row = {
        "role": role,
        "study_name": study_result.study_name,
        "best_trial_number": optional_number(study_result.best_trial_number),
        "best_hpo_value": optional_number(study_result.best_value),
        "best_params": json.dumps(study_result.best_params, sort_keys=True),
        "best_valid_precision_at_1": optional_number(
            (study_result.best_user_attrs or {}).get("best_valid_precision_at_1")
        ),
        "best_valid_mean_average_precision_at_r": optional_number(
            (study_result.best_user_attrs or {}).get("best_valid_mean_average_precision_at_r")
        ),
        "epoch_source": epoch_plan["source"],
        "epoch_field": epoch_plan["epoch_field"],
        "selected_epoch_indices": json.dumps(epoch_plan["selected_epoch_indices"]),
        "fold_training_epoch_counts": json.dumps(epoch_plan["training_epoch_counts"]),
        "mean_training_epochs": epoch_plan["mean_training_epochs"],
        "final_training_epochs": epoch_plan["final_training_epochs"],
        "final_log_dir": str(final_result.log_dir),
        "final_metrics_csv": str(final_result.metrics_csv),
        "final_train_loss": optional_number(final_result.final_train_loss),
        "epoch0_test_precision_at_1": optional_number(final_result.epoch0_test_precision_at_1),
        "epoch0_test_mean_average_precision_at_r": optional_number(
            final_result.epoch0_test_mean_average_precision_at_r
        ),
        "test_precision_at_1": optional_number(final_result.test_precision_at_1),
        "test_mean_average_precision_at_r": optional_number(final_result.test_mean_average_precision_at_r),
        "test_pacmap_coordinates": optional_path(final_result.test_pacmap_coordinates),
        "test_pacmap_plot": optional_path(final_result.test_pacmap_plot),
        "selected_epoch": final_result.selected_epoch,
        "global_step": final_result.global_step,
    }
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
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
    csv_path = study_result.study_dir / f"{summary_stem}.csv"
    row = {
        "role": role,
        "study_name": study_result.study_name,
        "study_dir_mode": STUDY_DIR_MODE_TRAIN_VAL,
        "best_trial_number": optional_number(study_result.best_trial_number),
        "best_hpo_value": optional_number(study_result.best_value),
        "best_params": json.dumps(study_result.best_params, sort_keys=True),
        "hpo_best_valid_precision_at_1": optional_number(
            (study_result.best_user_attrs or {}).get("best_valid_precision_at_1")
        ),
        "hpo_best_valid_mean_average_precision_at_r": optional_number(
            (study_result.best_user_attrs or {}).get("best_valid_mean_average_precision_at_r")
        ),
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
        "test_precision_at_1": optional_number(train_result.test_precision_at_1),
        "test_mean_average_precision_at_r": optional_number(train_result.test_mean_average_precision_at_r),
    }
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Train/validation HPO replay summary written to {study_result.study_dir}")


def write_hparam_top_final_evaluation_summary(study_result, evaluated, role, requested_top_n, summary_stem):
    """Write one summary for all top-N final-test evaluations."""

    top_stem = f"{summary_stem}_top_{requested_top_n}"
    rows = []
    for item in evaluated:
        trial = item["trial"]
        final_result = item["final_result"]
        rows.append(
            {
                "role": role,
                "study_name": study_result.study_name,
                "trial_number": trial["trial_number"],
                "hpo_value": optional_number(trial["value"]),
                "params": json.dumps(trial["params"], sort_keys=True),
                "summary_stem": item["summary_stem"],
                "final_log_dir": str(final_result.log_dir),
                "final_metrics_csv": str(final_result.metrics_csv),
                "final_train_loss": optional_number(final_result.final_train_loss),
                "epoch0_test_precision_at_1": optional_number(final_result.epoch0_test_precision_at_1),
                "epoch0_test_mean_average_precision_at_r": optional_number(
                    final_result.epoch0_test_mean_average_precision_at_r
                ),
                "test_precision_at_1": optional_number(final_result.test_precision_at_1),
                "test_mean_average_precision_at_r": optional_number(final_result.test_mean_average_precision_at_r),
                "test_pacmap_coordinates": optional_path(final_result.test_pacmap_coordinates),
                "test_pacmap_plot": optional_path(final_result.test_pacmap_plot),
                "selected_epoch": final_result.selected_epoch,
                "global_step": final_result.global_step,
            }
        )

    write_json(
        study_result.study_dir / f"{top_stem}.json",
        {
            "role": role,
            "study": hparam_study_result_to_dict(study_result),
            "requested_top_n": requested_top_n,
            "evaluations": [
                {
                    "trial": item["trial"],
                    "summary_stem": item["summary_stem"],
                    "final_result": result_to_dict(item["final_result"]),
                }
                for item in evaluated
            ],
        },
    )
    csv_path = study_result.study_dir / f"{top_stem}.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Top-{requested_top_n} final HPO test evaluation summary written to {study_result.study_dir}")


def write_hparam_selected_final_evaluation_summary(study_result, evaluated, role, trial_numbers, summary_stem):
    """Write one summary for explicitly selected final-test evaluations."""

    selected_stem = f"{summary_stem}_selected_trials"
    rows = []
    for item in evaluated:
        trial = item["trial"]
        final_result = item["final_result"]
        rows.append(
            {
                "role": role,
                "study_name": study_result.study_name,
                "trial_number": trial["trial_number"],
                "hpo_value": optional_number(trial["value"]),
                "params": json.dumps(trial["params"], sort_keys=True),
                "summary_stem": item["summary_stem"],
                "final_log_dir": str(final_result.log_dir),
                "final_metrics_csv": str(final_result.metrics_csv),
                "final_train_loss": optional_number(final_result.final_train_loss),
                "epoch0_test_precision_at_1": optional_number(final_result.epoch0_test_precision_at_1),
                "epoch0_test_mean_average_precision_at_r": optional_number(
                    final_result.epoch0_test_mean_average_precision_at_r
                ),
                "test_precision_at_1": optional_number(final_result.test_precision_at_1),
                "test_mean_average_precision_at_r": optional_number(final_result.test_mean_average_precision_at_r),
                "test_pacmap_coordinates": optional_path(final_result.test_pacmap_coordinates),
                "test_pacmap_plot": optional_path(final_result.test_pacmap_plot),
                "selected_epoch": final_result.selected_epoch,
                "global_step": final_result.global_step,
            }
        )

    write_json(
        study_result.study_dir / f"{selected_stem}.json",
        {
            "role": role,
            "study": hparam_study_result_to_dict(study_result),
            "trial_numbers": trial_numbers,
            "evaluations": [
                {
                    "trial": item["trial"],
                    "summary_stem": item["summary_stem"],
                    "final_result": result_to_dict(item["final_result"]),
                }
                for item in evaluated
            ],
        },
    )
    csv_path = study_result.study_dir / f"{selected_stem}.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Selected-trial final HPO test evaluation summary written to {study_result.study_dir}")


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
    rows = [
        make_comparison_row("supervised", supervised_study, supervised_final),
        make_comparison_row("ssl", ssl_study, ssl_final),
    ]
    csv_path = comparison_dir / "comparison_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    deltas = make_comparison_deltas(supervised_final, ssl_final)
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
            "delta_ssl_minus_supervised": deltas,
        },
    )
    logger.info(f"Comparison summary written to {comparison_dir}")


def make_comparison_deltas(supervised_final, ssl_final):
    return {
        "epoch0_test_precision_at_1": subtract_optional(
            ssl_final.epoch0_test_precision_at_1,
            supervised_final.epoch0_test_precision_at_1,
        ),
        "epoch0_test_mean_average_precision_at_r": subtract_optional(
            ssl_final.epoch0_test_mean_average_precision_at_r,
            supervised_final.epoch0_test_mean_average_precision_at_r,
        ),
        "test_precision_at_1": subtract_optional(
            ssl_final.test_precision_at_1,
            supervised_final.test_precision_at_1,
        ),
        "test_mean_average_precision_at_r": subtract_optional(
            ssl_final.test_mean_average_precision_at_r,
            supervised_final.test_mean_average_precision_at_r,
        ),
        "best_valid_precision_at_1": subtract_optional(
            ssl_final.best_valid_precision_at_1,
            supervised_final.best_valid_precision_at_1,
        ),
        "best_valid_mean_average_precision_at_r": subtract_optional(
            ssl_final.best_valid_mean_average_precision_at_r,
            supervised_final.best_valid_mean_average_precision_at_r,
        ),
    }


def make_comparison_row(method, study_result, final_result):
    return {
        "method": method,
        "study_name": study_result.study_name,
        "study_dir": str(study_result.study_dir),
        "best_trial_number": "" if study_result.best_trial_number is None else study_result.best_trial_number,
        "best_hpo_value": "" if study_result.best_value is None else study_result.best_value,
        "final_log_dir": str(final_result.log_dir),
        "best_valid_precision_at_1": final_result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": final_result.best_valid_mean_average_precision_at_r,
        "epoch0_test_precision_at_1": ""
        if final_result.epoch0_test_precision_at_1 is None
        else final_result.epoch0_test_precision_at_1,
        "epoch0_test_mean_average_precision_at_r": ""
        if final_result.epoch0_test_mean_average_precision_at_r is None
        else final_result.epoch0_test_mean_average_precision_at_r,
        "test_precision_at_1": "" if final_result.test_precision_at_1 is None else final_result.test_precision_at_1,
        "test_mean_average_precision_at_r": ""
        if final_result.test_mean_average_precision_at_r is None
        else final_result.test_mean_average_precision_at_r,
        "test_pacmap_coordinates": optional_path(final_result.test_pacmap_coordinates),
        "test_pacmap_plot": optional_path(final_result.test_pacmap_plot),
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
        "run_seed": scenario.run_seed,
        "data_split_seed": scenario.data_split_seed,
        "support_seed": scenario.support_seed,
        "hparam_seed": scenario.hparam_seed,
        "label_sampling_mode": scenario.label_sampling_mode,
        "loss": scenario.loss,
        "miner": scenario.miner,
        "ssl_config_path": str(scenario.ssl_config_path),
    }


def write_comparison_grid_summary(grid_dir, grid_results):
    grid_dir.mkdir(parents=True, exist_ok=True)
    rows = [make_grid_summary_row(result) for result in grid_results]
    summary_csv = grid_dir / "grid_summary.csv"
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = make_grid_aggregate_rows(rows)
    aggregate_csv = grid_dir / "grid_aggregate.csv"
    with aggregate_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    write_json(
        grid_dir / "grid_summary.json",
        {
            "runs": rows,
            "aggregates": aggregate_rows,
        },
    )
    logger.info(f"Comparison grid summary written to {grid_dir}")


def write_single_method_grid_summary(grid_dir, grid_results):
    grid_dir.mkdir(parents=True, exist_ok=True)
    rows = [make_single_method_grid_summary_row(result) for result in grid_results]
    summary_csv = grid_dir / "grid_summary.csv"
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = make_single_method_grid_aggregate_rows(rows)
    aggregate_csv = grid_dir / "grid_aggregate.csv"
    with aggregate_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    write_json(
        grid_dir / "grid_summary.json",
        {
            "runs": rows,
            "aggregates": aggregate_rows,
        },
    )
    logger.info(f"Single-method grid summary written to {grid_dir}")


def make_single_method_grid_summary_row(grid_result):
    scenario = grid_result["scenario"]
    method = grid_result["method"]
    study = grid_result["study"]
    result = grid_result["result"]

    row = {
        "method": method,
        "scenario": scenario.name,
        "label_sampling_mode": scenario.label_sampling_mode,
        "labeled_fraction": scenario.labeled_fraction,
        "labeled_per_class": "" if scenario.labeled_per_class is None else scenario.labeled_per_class,
        "loss": scenario.loss,
        "miner": scenario.miner,
        "comparison_seed": scenario.seed,
        "run_seed": "" if scenario.run_seed is None else scenario.run_seed,
        "data_split_seed": "" if scenario.data_split_seed is None else scenario.data_split_seed,
        "support_seed": "" if scenario.support_seed is None else scenario.support_seed,
        "hparam_seed": "" if scenario.hparam_seed is None else scenario.hparam_seed,
        "best_trial_number": "",
        "best_hpo_value": "",
        "log_dir": "",
        "metrics_csv": "",
        "best_valid_precision_at_1": "",
        "best_valid_mean_average_precision_at_r": "",
        "test_precision_at_1": "",
        "test_mean_average_precision_at_r": "",
        "test_pacmap_coordinates": "",
        "test_pacmap_plot": "",
        "epoch0_test_precision_at_1": "",
        "epoch0_test_mean_average_precision_at_r": "",
        "final_train_loss": "",
        "final_training_epochs": "",
        "selected_epoch": "",
        "global_step": "",
    }
    if study is not None:
        attrs = study.best_user_attrs or {}
        row.update(
            {
                "best_trial_number": optional_number(study.best_trial_number),
                "best_hpo_value": optional_number(study.best_value),
                "log_dir": str(attrs.get("log_dir", "")),
                "metrics_csv": str(attrs.get("metrics_csv", "")),
                "best_valid_precision_at_1": optional_number(attrs.get("best_valid_precision_at_1")),
                "best_valid_mean_average_precision_at_r": optional_number(
                    attrs.get("best_valid_mean_average_precision_at_r")
                ),
                "test_precision_at_1": optional_number(attrs.get("test_precision_at_1")),
                "test_mean_average_precision_at_r": optional_number(attrs.get("test_mean_average_precision_at_r")),
            }
        )
        if result is not None:
            row.update(
                {
                    "log_dir": str(result.log_dir),
                    "metrics_csv": str(result.metrics_csv),
                    "epoch0_test_precision_at_1": optional_number(result.epoch0_test_precision_at_1),
                    "epoch0_test_mean_average_precision_at_r": optional_number(
                        result.epoch0_test_mean_average_precision_at_r
                    ),
                    "test_precision_at_1": optional_number(result.test_precision_at_1),
                    "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
                    "test_pacmap_coordinates": optional_path(result.test_pacmap_coordinates),
                    "test_pacmap_plot": optional_path(result.test_pacmap_plot),
                    "final_train_loss": optional_number(result.final_train_loss),
                    "final_training_epochs": max(0, result.last_epoch + 1),
                    "selected_epoch": result.selected_epoch,
                    "global_step": result.global_step,
                }
            )
    else:
        row.update(
            {
                "log_dir": str(result.log_dir),
                "metrics_csv": str(result.metrics_csv),
                "best_valid_precision_at_1": result.best_valid_precision_at_1,
                "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
                "epoch0_test_precision_at_1": optional_number(result.epoch0_test_precision_at_1),
                "epoch0_test_mean_average_precision_at_r": optional_number(
                    result.epoch0_test_mean_average_precision_at_r
                ),
                "test_precision_at_1": optional_number(result.test_precision_at_1),
                "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
                "test_pacmap_coordinates": optional_path(result.test_pacmap_coordinates),
                "test_pacmap_plot": optional_path(result.test_pacmap_plot),
                "final_train_loss": optional_number(result.final_train_loss),
                "final_training_epochs": max(0, result.last_epoch + 1),
                "selected_epoch": result.selected_epoch,
                "global_step": result.global_step,
            }
        )
    return row


def make_single_method_grid_aggregate_rows(rows):
    group_keys = [
        "method",
        "label_sampling_mode",
        "labeled_fraction",
        "labeled_per_class",
        "loss",
        "miner",
    ]
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
    ]
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items()):
        aggregate = {
            "method": key[0],
            "label_sampling_mode": key[1],
            "labeled_fraction": key[2],
            "labeled_per_class": key[3],
            "loss": key[4],
            "miner": key[5],
            "num_seeds": len(group_rows),
        }
        for metric_name in metric_names:
            values = [row[metric_name] for row in group_rows if row[metric_name] != ""]
            mean_value, std_value = mean_std(values)
            aggregate[f"{metric_name}_mean"] = optional_number(mean_value)
            aggregate[f"{metric_name}_std"] = optional_number(std_value)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def make_grid_summary_row(result):
    scenario = result["scenario"]
    supervised_final = result["supervised_final"]
    ssl_final = result["ssl_final"]
    deltas = result["deltas"]
    return {
        "scenario": scenario.name,
        "label_sampling_mode": scenario.label_sampling_mode,
        "labeled_fraction": scenario.labeled_fraction,
        "labeled_per_class": "" if scenario.labeled_per_class is None else scenario.labeled_per_class,
        "loss": scenario.loss,
        "miner": scenario.miner,
        "comparison_seed": scenario.seed,
        "run_seed": "" if scenario.run_seed is None else scenario.run_seed,
        "data_split_seed": "" if scenario.data_split_seed is None else scenario.data_split_seed,
        "support_seed": "" if scenario.support_seed is None else scenario.support_seed,
        "hparam_seed": "" if scenario.hparam_seed is None else scenario.hparam_seed,
        "comparison_dir": str(result["comparison_dir"]),
        "supervised_epoch0_test_precision_at_1": optional_number(supervised_final.epoch0_test_precision_at_1),
        "ssl_epoch0_test_precision_at_1": optional_number(ssl_final.epoch0_test_precision_at_1),
        "delta_epoch0_test_precision_at_1": optional_number(deltas["epoch0_test_precision_at_1"]),
        "supervised_epoch0_test_mean_average_precision_at_r": optional_number(
            supervised_final.epoch0_test_mean_average_precision_at_r
        ),
        "ssl_epoch0_test_mean_average_precision_at_r": optional_number(
            ssl_final.epoch0_test_mean_average_precision_at_r
        ),
        "delta_epoch0_test_mean_average_precision_at_r": optional_number(
            deltas["epoch0_test_mean_average_precision_at_r"]
        ),
        "supervised_test_precision_at_1": optional_number(supervised_final.test_precision_at_1),
        "ssl_test_precision_at_1": optional_number(ssl_final.test_precision_at_1),
        "delta_test_precision_at_1": optional_number(deltas["test_precision_at_1"]),
        "supervised_test_mean_average_precision_at_r": optional_number(
            supervised_final.test_mean_average_precision_at_r
        ),
        "ssl_test_mean_average_precision_at_r": optional_number(ssl_final.test_mean_average_precision_at_r),
        "delta_test_mean_average_precision_at_r": optional_number(deltas["test_mean_average_precision_at_r"]),
        "supervised_test_pacmap_coordinates": optional_path(supervised_final.test_pacmap_coordinates),
        "supervised_test_pacmap_plot": optional_path(supervised_final.test_pacmap_plot),
        "ssl_test_pacmap_coordinates": optional_path(ssl_final.test_pacmap_coordinates),
        "ssl_test_pacmap_plot": optional_path(ssl_final.test_pacmap_plot),
        "supervised_best_valid_precision_at_1": supervised_final.best_valid_precision_at_1,
        "ssl_best_valid_precision_at_1": ssl_final.best_valid_precision_at_1,
        "delta_best_valid_precision_at_1": optional_number(deltas["best_valid_precision_at_1"]),
        "supervised_best_valid_mean_average_precision_at_r": supervised_final.best_valid_mean_average_precision_at_r,
        "ssl_best_valid_mean_average_precision_at_r": ssl_final.best_valid_mean_average_precision_at_r,
        "delta_best_valid_mean_average_precision_at_r": optional_number(
            deltas["best_valid_mean_average_precision_at_r"]
        ),
    }


def make_grid_aggregate_rows(rows):
    group_keys = ["label_sampling_mode", "labeled_fraction", "labeled_per_class", "loss", "miner"]
    metric_names = [
        "supervised_epoch0_test_precision_at_1",
        "ssl_epoch0_test_precision_at_1",
        "delta_epoch0_test_precision_at_1",
        "supervised_epoch0_test_mean_average_precision_at_r",
        "ssl_epoch0_test_mean_average_precision_at_r",
        "delta_epoch0_test_mean_average_precision_at_r",
        "supervised_test_precision_at_1",
        "ssl_test_precision_at_1",
        "delta_test_precision_at_1",
        "supervised_test_mean_average_precision_at_r",
        "ssl_test_mean_average_precision_at_r",
        "delta_test_mean_average_precision_at_r",
        "supervised_best_valid_precision_at_1",
        "ssl_best_valid_precision_at_1",
        "delta_best_valid_precision_at_1",
        "supervised_best_valid_mean_average_precision_at_r",
        "ssl_best_valid_mean_average_precision_at_r",
        "delta_best_valid_mean_average_precision_at_r",
    ]
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items()):
        aggregate = {
            "label_sampling_mode": key[0],
            "labeled_fraction": key[1],
            "labeled_per_class": key[2],
            "loss": key[3],
            "miner": key[4],
            "num_seeds": len(group_rows),
        }
        for metric_name in metric_names:
            values = [row[metric_name] for row in group_rows if row[metric_name] != ""]
            mean_value, std_value = mean_std(values)
            aggregate[f"{metric_name}_mean"] = optional_number(mean_value)
            aggregate[f"{metric_name}_std"] = optional_number(std_value)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def mean_std(values):
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    std = 0.0 if len(array) == 1 else float(np.std(array, ddof=1))
    return float(np.mean(array)), std


def optional_number(value):
    return "" if value is None else value


def optional_path(value):
    return "" if value is None else str(value)
