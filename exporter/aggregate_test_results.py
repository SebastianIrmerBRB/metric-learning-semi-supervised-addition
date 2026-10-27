#!/usr/bin/env python3
"""Aggregate final test-set evaluation summaries below a logs directory.

The training code can write several audit summaries for one physical final run
(for example an individual summary plus a top-N or selected-trials summary).
This extractor treats the final run's ``log_dir`` as its identity so those
views do not inflate the result count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


TEST_METRIC_FIELDS = (
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
)

OUTPUT_FIELDS = (
    "result_id",
    "dataset",
    "dataset_protocol",
    "log_group",
    "scenario",
    "mode",
    "role",
    "loss",
    "miner",
    "ssl_method",
    "label_sampling_mode",
    "labeled_fraction",
    "labeled_per_class",
    "seed",
    "data_split_seed",
    "support_seed",
    "hparam_seed",
    "backbone_tuning",
    "feat_dim",
    "batch_size",
    "sampler_m",
    "lr",
    "classifier_lr",
    "weight_decay",
    "study_name",
    "trial_number",
    "hpo_value",
    "best_valid_precision_at_1",
    "best_valid_mean_average_precision_at_r",
    "epoch0_test_precision_at_1",
    "epoch0_test_mean_average_precision_at_r",
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
    "test_precision_at_1_std",
    "test_mean_average_precision_at_r_std",
    "concatenated_test_precision_at_1",
    "concatenated_test_mean_average_precision_at_r",
    "concatenated_test_embedding_dim",
    "final_train_loss",
    "final_training_epochs",
    "final_fit_mode",
    "epoch_aggregation",
    "final_full_train",
    "selected_epoch",
    "global_step",
    "params",
    "final_log_dir",
    "final_metrics_csv",
    "run_config",
    "summary_name",
    "source_file",
    "source_count",
    "recorded_at_utc",
)


def _present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if _present(value):
            return value
    return ""


def _plan_field(args: Mapping[str, Any], field: str) -> Any:
    """Read one field of the fit plan a final run recorded in its own args."""

    plan = args.get("hparam_final_epoch_plan")
    if not isinstance(plan, Mapping):
        return ""
    return _first(plan, field)


def _realized_training_epochs(result: Mapping[str, Any]) -> Any:
    """Recover how long a run trained when no plan fixed the duration up front."""

    last_epoch = result.get("last_epoch")
    if not _present(last_epoch):
        return ""
    try:
        return max(0, int(last_epoch) + 1)
    except (TypeError, ValueError):
        return ""


def _has_test_performance(mapping: Mapping[str, Any]) -> bool:
    return any(_present(mapping.get(field)) for field in TEST_METRIC_FIELDS)


def _json_text(value: Any) -> str:
    if not _present(value):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalise_log_reference(value: Any) -> str:
    """Return a stable logs-relative path where possible."""

    if not _present(value):
        return ""
    normalised = str(value).strip().replace("\\", "/")
    marker = "/logs/"
    if marker in normalised:
        normalised = normalised.split(marker, 1)[1]
    elif normalised.startswith("logs/"):
        normalised = normalised[len("logs/") :]
    return normalised.removeprefix("./").rstrip("/")


def _summary_layout(source_file: str, final_log_dir: str) -> dict[str, str]:
    source_parts = [part for part in source_file.replace("\\", "/").split("/") if part]
    final_parts = [part for part in _normalise_log_reference(final_log_dir).split("/") if part]
    log_group = source_parts[0] if source_parts else (final_parts[0] if final_parts else "")
    scenario = ""

    for parts in (source_parts, final_parts):
        boundary = next(
            (index for index, part in enumerate(parts) if part in {"hpo", "final"}),
            None,
        )
        if boundary is not None and boundary > 0:
            scenario = parts[boundary - 1]
            if not log_group:
                log_group = parts[0]
            break

    return {"log_group": log_group, "scenario": scenario}


def _candidate_from_csv(
    row: Mapping[str, Any],
    source_path: Path,
    log_root: Path,
) -> dict[str, Any]:
    source_file = source_path.relative_to(log_root).as_posix()
    final_log_dir = _first(row, "final_log_dir", "log_dir", "train_val_log_dir")
    candidate = {
        "role": _first(row, "role"),
        "study_name": _first(row, "study_name"),
        "trial_number": _first(row, "trial_number", "best_trial_number"),
        "hpo_value": _first(row, "hpo_value", "best_hpo_value", "original_hpo_value"),
        "best_valid_precision_at_1": _first(
            row,
            "best_valid_precision_at_1",
            "hpo_best_valid_precision_at_1",
            "mean_best_valid_precision_at_1",
        ),
        "best_valid_mean_average_precision_at_r": _first(
            row,
            "best_valid_mean_average_precision_at_r",
            "hpo_best_valid_mean_average_precision_at_r",
            "mean_best_valid_mean_average_precision_at_r",
        ),
        "epoch0_test_precision_at_1": _first(row, "epoch0_test_precision_at_1"),
        "epoch0_test_mean_average_precision_at_r": _first(
            row,
            "epoch0_test_mean_average_precision_at_r",
        ),
        "test_precision_at_1": _first(row, "test_precision_at_1"),
        "test_mean_average_precision_at_r": _first(
            row,
            "test_mean_average_precision_at_r",
        ),
        "test_precision_at_1_std": _first(row, "test_precision_at_1_std"),
        "test_mean_average_precision_at_r_std": _first(
            row,
            "test_mean_average_precision_at_r_std",
        ),
        "concatenated_test_precision_at_1": _first(row, "concatenated_test_precision_at_1"),
        "concatenated_test_mean_average_precision_at_r": _first(
            row,
            "concatenated_test_mean_average_precision_at_r",
        ),
        "concatenated_test_embedding_dim": _first(row, "concatenated_test_embedding_dim"),
        "final_train_loss": _first(row, "final_train_loss", "train_val_final_train_loss"),
        "final_training_epochs": _first(row, "final_training_epochs"),
        "final_fit_mode": _first(row, "final_fit_mode"),
        "epoch_aggregation": _first(row, "epoch_aggregation"),
        "selected_epoch": _first(row, "selected_epoch", "train_val_selected_epoch"),
        "global_step": _first(row, "global_step", "train_val_global_step"),
        "params": _json_text(_first(row, "params", "best_params")),
        "final_log_dir": str(final_log_dir),
        "final_metrics_csv": _first(
            row,
            "final_metrics_csv",
            "metrics_csv",
            "train_val_metrics_csv",
        ),
        "summary_name": source_path.stem,
        "_source_files": {source_file},
    }
    candidate.update(_summary_layout(source_file, str(final_log_dir)))
    return candidate


def _json_context(node: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    context = dict(inherited)
    for key in ("role", "study_name", "summary_stem"):
        if _present(node.get(key)):
            context[key] = node[key]

    study = node.get("study")
    if isinstance(study, Mapping):
        context["study_name"] = _first(study, "study_name") or context.get("study_name", "")
        context["trial_number"] = _first(study, "best_trial_number")
        context["hpo_value"] = _first(study, "best_value")
        context["params"] = _first(study, "best_params")
        best_attrs = study.get("best_user_attrs")
        if isinstance(best_attrs, Mapping):
            context["best_valid_precision_at_1"] = _first(
                best_attrs,
                "best_valid_precision_at_1",
            )
            context["best_valid_mean_average_precision_at_r"] = _first(
                best_attrs,
                "best_valid_mean_average_precision_at_r",
            )

    trial = node.get("trial")
    if isinstance(trial, Mapping):
        context["trial_number"] = _first(trial, "trial_number", "number")
        context["hpo_value"] = _first(trial, "value", "hpo_value")
        context["params"] = _first(trial, "params")

    epoch_plan = node.get("epoch_plan")
    if isinstance(epoch_plan, Mapping):
        # An early-stopped final fit leaves this empty on purpose; the run's own
        # last epoch supplies the realized duration in _candidate_from_json.
        context["final_training_epochs"] = _first(epoch_plan, "final_training_epochs")
        context["final_fit_mode"] = _first(epoch_plan, "fit_mode")
        context["epoch_aggregation"] = _first(epoch_plan, "aggregation")
    return context


def _walk_final_results(
    node: Any,
    inherited: Mapping[str, Any] | None = None,
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if isinstance(node, list):
        for item in node:
            yield from _walk_final_results(item, inherited)
        return
    if not isinstance(node, Mapping):
        return

    context = _json_context(node, inherited or {})
    final_result = node.get("final_result")
    if isinstance(final_result, Mapping) and _has_test_performance(final_result):
        yield final_result, context

    for key, value in node.items():
        if key != "final_result" and isinstance(value, (Mapping, list)):
            yield from _walk_final_results(value, context)


def _candidate_from_json(
    result: Mapping[str, Any],
    context: Mapping[str, Any],
    source_path: Path,
    log_root: Path,
) -> dict[str, Any]:
    source_file = source_path.relative_to(log_root).as_posix()
    final_log_dir = _first(result, "log_dir", "final_log_dir")
    candidate = {
        "role": _first(context, "role"),
        "study_name": _first(context, "study_name"),
        "trial_number": _first(context, "trial_number"),
        "hpo_value": _first(context, "hpo_value"),
        "best_valid_precision_at_1": _first(
            result,
            "best_valid_precision_at_1",
        )
        or _first(context, "best_valid_precision_at_1"),
        "best_valid_mean_average_precision_at_r": _first(
            result,
            "best_valid_mean_average_precision_at_r",
        )
        or _first(context, "best_valid_mean_average_precision_at_r"),
        "epoch0_test_precision_at_1": _first(result, "epoch0_test_precision_at_1"),
        "epoch0_test_mean_average_precision_at_r": _first(
            result,
            "epoch0_test_mean_average_precision_at_r",
        ),
        "test_precision_at_1": _first(result, "test_precision_at_1"),
        "test_mean_average_precision_at_r": _first(
            result,
            "test_mean_average_precision_at_r",
        ),
        "test_precision_at_1_std": _first(result, "test_precision_at_1_std"),
        "test_mean_average_precision_at_r_std": _first(
            result,
            "test_mean_average_precision_at_r_std",
        ),
        "concatenated_test_precision_at_1": _first(result, "concatenated_test_precision_at_1"),
        "concatenated_test_mean_average_precision_at_r": _first(
            result,
            "concatenated_test_mean_average_precision_at_r",
        ),
        "concatenated_test_embedding_dim": _first(result, "concatenated_test_embedding_dim"),
        "final_train_loss": _first(result, "final_train_loss"),
        "final_training_epochs": _first(context, "final_training_epochs")
        or _realized_training_epochs(result),
        "final_fit_mode": _first(context, "final_fit_mode"),
        "epoch_aggregation": _first(context, "epoch_aggregation"),
        "selected_epoch": _first(result, "selected_epoch"),
        "global_step": _first(result, "global_step"),
        "params": _json_text(_first(context, "params")),
        "final_log_dir": str(final_log_dir),
        "final_metrics_csv": _first(result, "metrics_csv", "final_metrics_csv"),
        "summary_name": _first(context, "summary_stem") or source_path.stem,
        "_source_files": {source_file},
    }
    candidate.update(_summary_layout(source_file, str(final_log_dir)))
    return candidate


def _candidate_from_metrics(
    metrics_path: Path,
    config: Mapping[str, Any],
    log_root: Path,
) -> dict[str, Any] | None:
    args = config.get("args") if isinstance(config.get("args"), Mapping) else {}
    final_relative_dir = metrics_path.parent.relative_to(log_root).as_posix()
    is_final_path = "final" in metrics_path.parent.relative_to(log_root).parts
    if not bool(args.get("evaluate_test")):
        return None
    if not bool(args.get("final_full_train")) and not is_final_path:
        return None

    test_row: Mapping[str, Any] | None = None
    epoch0_row: Mapping[str, Any] | None = None
    final_train_loss: Any = ""
    best_valid_precision: float | None = None
    best_valid_map: float | None = None
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", "")).strip().lower()
            if split == "test":
                test_row = row
            elif split == "epoch0_test":
                epoch0_row = row
            elif split == "train_epoch" and _present(row.get("loss")):
                final_train_loss = row["loss"]
            elif split == "valid":
                try:
                    precision = float(str(row.get("precision_at_1", "")))
                    best_valid_precision = (
                        precision
                        if best_valid_precision is None
                        else max(best_valid_precision, precision)
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    map_at_r = float(str(row.get("mean_average_precision_at_r", "")))
                    best_valid_map = (
                        map_at_r if best_valid_map is None else max(best_valid_map, map_at_r)
                    )
                except (TypeError, ValueError):
                    pass

    if test_row is None or not _has_test_performance(
        {
            "test_precision_at_1": test_row.get("precision_at_1"),
            "test_mean_average_precision_at_r": test_row.get(
                "mean_average_precision_at_r"
            ),
        }
    ):
        return None

    source_file = metrics_path.relative_to(log_root).as_posix()
    final_log_dir = _first(args, "log_dir") or f"logs/{final_relative_dir}"
    candidate = {
        "dataset": _first(args, "dataset"),
        "dataset_protocol": _first(args, "dataset_protocol"),
        "mode": _first(args, "mode"),
        "role": _first(args, "mode"),
        "loss": _first(args, "loss"),
        "miner": _first(args, "miner"),
        "seed": _first(args, "seed"),
        "data_split_seed": _first(args, "data_split_seed"),
        "support_seed": _first(args, "support_seed"),
        "hparam_seed": _first(args, "hparam_seed"),
        "backbone_tuning": _first(args, "backbone_tuning"),
        "feat_dim": _first(args, "feat_dim"),
        "batch_size": _first(args, "batch_size"),
        "sampler_m": _first(args, "sampler_m"),
        "lr": _first(args, "lr"),
        "classifier_lr": _first(args, "classifier_lr"),
        "weight_decay": _first(args, "weight_decay"),
        "study_name": _first(
            args,
            "hparam_study_name",
            "hparam_final_from_study",
        ),
        "trial_number": _first(
            args,
            "hparam_final_trial_number",
            "trial_number",
        ),
        "best_valid_precision_at_1": (
            "" if best_valid_precision is None else best_valid_precision
        ),
        "best_valid_mean_average_precision_at_r": (
            "" if best_valid_map is None else best_valid_map
        ),
        "epoch0_test_precision_at_1": (
            "" if epoch0_row is None else _first(epoch0_row, "precision_at_1")
        ),
        "epoch0_test_mean_average_precision_at_r": (
            ""
            if epoch0_row is None
            else _first(epoch0_row, "mean_average_precision_at_r")
        ),
        "test_precision_at_1": _first(test_row, "precision_at_1"),
        "test_mean_average_precision_at_r": _first(
            test_row,
            "mean_average_precision_at_r",
        ),
        "final_train_loss": final_train_loss,
        "final_training_epochs": _first(args, "epochs"),
        "final_fit_mode": _plan_field(args, "fit_mode"),
        "epoch_aggregation": _plan_field(args, "aggregation"),
        "final_full_train": bool(args.get("final_full_train")),
        "selected_epoch": _first(test_row, "epoch"),
        "global_step": _first(test_row, "step"),
        "params": _json_text(_first(args, "hparam_params")),
        "final_log_dir": str(final_log_dir),
        "final_metrics_csv": f"logs/{source_file}",
        "summary_name": "metrics",
        "_source_files": {source_file},
    }
    candidate.update(_summary_layout(source_file, str(final_log_dir)))
    return candidate


def _read_candidates(log_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in sorted(log_root.rglob("*evaluation*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if _has_test_performance(row):
                        candidates.append(_candidate_from_csv(row, path, log_root))
        except (OSError, csv.Error, UnicodeError) as exc:
            warnings.append(f"Skipped {path}: {exc}")

    for path in sorted(log_root.rglob("*evaluation*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            for result, context in _walk_final_results(document):
                candidates.append(
                    _candidate_from_json(result, context, path, log_root)
                )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped {path}: {exc}")

    for config_path in sorted(log_root.rglob("run_config.json")):
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            if not isinstance(config, Mapping):
                continue
            metrics_path = config_path.parent / "metrics.csv"
            if not metrics_path.is_file():
                continue
            candidate = _candidate_from_metrics(metrics_path, config, log_root)
            if candidate is not None:
                candidates.append(candidate)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped {config_path}: {exc}")

    return candidates, warnings


def _dedupe_key(candidate: Mapping[str, Any]) -> str:
    final_log_dir = _normalise_log_reference(candidate.get("final_log_dir"))
    if final_log_dir:
        return f"log:{final_log_dir}"
    metrics_csv = _normalise_log_reference(candidate.get("final_metrics_csv"))
    if metrics_csv:
        return f"metrics:{metrics_csv}"
    fingerprint = "|".join(
        str(candidate.get(field, ""))
        for field in (
            "study_name",
            "trial_number",
            "role",
            "test_precision_at_1",
            "test_mean_average_precision_at_r",
        )
    )
    sources = "|".join(sorted(candidate.get("_source_files", set())))
    return f"fallback:{fingerprint}|{sources}"


def _merge_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = _dedupe_key(candidate)
        current = by_key.get(key)
        if current is None:
            candidate["_dedupe_key"] = key
            by_key[key] = candidate
            continue

        current.setdefault("_source_files", set()).update(
            candidate.get("_source_files", set())
        )
        for field, value in candidate.items():
            if field.startswith("_"):
                continue
            if not _present(current.get(field)) and _present(value):
                current[field] = value
    return list(by_key.values())


def _resolve_log_path(log_root: Path, raw_path: Any) -> Path | None:
    reference = _normalise_log_reference(raw_path)
    if not reference:
        return None
    candidate = (log_root / Path(reference)).resolve()
    root = log_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _load_run_config(log_root: Path, candidate: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    final_dir = _resolve_log_path(log_root, candidate.get("final_log_dir"))
    if final_dir is None:
        metrics_path = _resolve_log_path(log_root, candidate.get("final_metrics_csv"))
        final_dir = metrics_path.parent if metrics_path is not None else None
    if final_dir is None:
        return {}, ""

    config_path = final_dir / "run_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, ""
    try:
        relative_path = config_path.resolve().relative_to(log_root.resolve()).as_posix()
    except ValueError:
        relative_path = ""
    return config if isinstance(config, dict) else {}, relative_path


def _config_value(args: Mapping[str, Any], ssl: Mapping[str, Any], field: str) -> Any:
    if field in {
        "label_sampling_mode",
        "labeled_fraction",
        "labeled_per_class",
    }:
        return _first(ssl, field)
    if field == "ssl_method":
        return _first(ssl, "method")
    return _first(args, field)


def _recorded_at(candidate: Mapping[str, Any], log_root: Path) -> str:
    timestamps = []
    for source_file in candidate.get("_source_files", set()):
        try:
            timestamps.append((log_root / source_file).stat().st_mtime)
        except OSError:
            pass
    if not timestamps:
        return ""
    return datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()


def _enrich(candidate: dict[str, Any], log_root: Path) -> dict[str, Any]:
    config, run_config_path = _load_run_config(log_root, candidate)
    args = config.get("args") if isinstance(config.get("args"), Mapping) else {}
    ssl = config.get("ssl_config")
    if not isinstance(ssl, Mapping):
        resolved_ssl = args.get("ssl_config_resolved")
        ssl = resolved_ssl if isinstance(resolved_ssl, Mapping) else {}

    for field in (
        "dataset",
        "dataset_protocol",
        "mode",
        "loss",
        "miner",
        "ssl_method",
        "label_sampling_mode",
        "labeled_fraction",
        "labeled_per_class",
        "seed",
        "data_split_seed",
        "support_seed",
        "hparam_seed",
        "backbone_tuning",
        "feat_dim",
        "batch_size",
        "sampler_m",
        "lr",
        "classifier_lr",
        "weight_decay",
    ):
        value = _config_value(args, ssl, field)
        if not _present(candidate.get(field)) and _present(value):
            candidate[field] = value

    if not _present(candidate.get("role")):
        candidate["role"] = _first(args, "mode")
    if not _present(candidate.get("study_name")):
        candidate["study_name"] = _first(
            args,
            "hparam_study_name",
            "hparam_final_from_study",
        )
    if not _present(candidate.get("trial_number")):
        candidate["trial_number"] = _first(
            args,
            "hparam_final_trial_number",
            "trial_number",
        )
    if not _present(candidate.get("params")):
        candidate["params"] = _json_text(_first(args, "hparam_params"))
    if not _present(candidate.get("final_training_epochs")):
        candidate["final_training_epochs"] = _first(args, "epochs")
    if not _present(candidate.get("final_fit_mode")):
        candidate["final_fit_mode"] = _plan_field(args, "fit_mode")
    if not _present(candidate.get("epoch_aggregation")):
        candidate["epoch_aggregation"] = _plan_field(args, "aggregation")
    if not _present(candidate.get("final_full_train")):
        candidate["final_full_train"] = _first(args, "final_full_train")

    sources = sorted(candidate.get("_source_files", set()))
    candidate["source_file"] = "|".join(sources)
    candidate["source_count"] = len(sources)
    candidate["run_config"] = run_config_path
    candidate["recorded_at_utc"] = _recorded_at(candidate, log_root)
    key = str(candidate.get("_dedupe_key", _dedupe_key(candidate)))
    candidate["result_id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    if not _present(candidate.get("dataset")):
        candidate["dataset"] = candidate.get("log_group", "")
    return {field: candidate.get(field, "") for field in OUTPUT_FIELDS}


def aggregate_test_results(log_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return one enriched row per physical final test run."""

    log_root = Path(log_root).resolve()
    candidates, warnings = _read_candidates(log_root)
    rows = [_enrich(candidate, log_root) for candidate in _merge_candidates(candidates)]
    rows.sort(
        key=lambda row: (
            str(row["dataset"]).casefold(),
            str(row["scenario"]).casefold(),
            str(row["mode"]).casefold(),
            str(row["loss"]).casefold(),
            str(row["seed"]),
            str(row["trial_number"]),
            str(row["result_id"]),
        )
    )
    return rows, warnings


def write_test_results_csv(output: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write the stable aggregate schema atomically."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate final test-set performance summaries under logs.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory to scan (default: ./logs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-results.csv"),
        help="Aggregate CSV to write (default: ./test-results.csv).",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Return a non-zero exit code when no completed test result is found.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.log_root.is_dir():
        print(f"Logs directory does not exist: {args.log_root}", file=sys.stderr)
        return 2

    rows, warnings = aggregate_test_results(args.log_root)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    write_test_results_csv(args.output, rows)
    print(f"Aggregated {len(rows)} final test result(s) into {args.output}", file=sys.stderr)
    if args.require_results and not rows:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
