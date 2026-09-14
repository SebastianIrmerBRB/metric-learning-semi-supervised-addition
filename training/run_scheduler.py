"""Device-aware scheduling for independent experiment/HPO processes.

This module deliberately sits above :mod:`main`.  Every scheduled item is a
complete experiment run (and therefore at most one Optuna study), while trials
inside that process keep their normal sequential ``n_jobs=1`` execution.

The scheduler has no PyTorch import. In CUDA-training mode, when CPU runs
request a dedicated ``ssl_gpus`` pool, or when CPU runs configure ``gpus`` for
continuous SSL acceleration plus large-batch training, it discovers physical
GPUs with ``nvidia-smi`` and starts each child with exactly one GPU UUID in
``CUDA_VISIBLE_DEVICES``. Inside the child that physical GPU is consequently
available as logical ``cuda:0``. Pure CPU mode skips NVIDIA discovery and uses
the host-wide run limit for admission.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from . import ablation, cpu_threads, device_thresholds
# Importable here because the SSL config module itself stays free of PyTorch.
from .ssl.config import LABEL_SAMPLING_MODES


MANIFEST_VERSION = 1
DEFAULT_GPU_MEMORY_MIB = 2048
DEFAULT_RESERVE_GPU_MEMORY_MIB = 1024
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
GPU_BATCH_SIZE_THRESHOLD = device_thresholds.DEFAULT_GPU_BATCH_SIZE_THRESHOLD
_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Mirrored from training.cli, which cannot be imported here because it reaches
# PyTorch. tests/test_run_scheduler.py asserts the two copies stay in sync.
STUDY_DIR_MODE_FINAL_TRAIN = "final_train"
STUDY_DIR_MODE_TRAIN_VAL = "train_val"
STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL = "cross_seed_train_val"
STUDY_DIR_MODES = (
    STUDY_DIR_MODE_FINAL_TRAIN,
    STUDY_DIR_MODE_TRAIN_VAL,
    STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL,
)
DEFAULT_STUDY_DIR_MODE = STUDY_DIR_MODE_TRAIN_VAL
COMPARISON_SEED_TARGETS = (
    "seed",
    "data_split_seed",
    "support_seed",
    "hparam_seed",
)
# hparam_seed alone leaves a fixed-parameter validation replay unchanged.
CROSS_SEED_VALIDATION_TARGETS = ("seed", "data_split_seed", "support_seed")
# The one sampling mode whose label budget has a per-class k in addition to the
# fraction of classes, so only it can vary over a k-shot grid.
LABEL_SAMPLING_MODE_CLASS_SUBSET_K_SHOT = "class_subset_k_shot"
FINAL_TEST_VISUALIZATION_NONE = "none"
FINAL_TEST_VISUALIZATION_MODES = (FINAL_TEST_VISUALIZATION_NONE, "pacmap", "tsne")
# Mirrors training.cli: keep the file, delete it once the concatenated fold
# evaluation has read it, or never write one.
FOLD_TEST_EMBEDDING_STORAGES = ("file", "temporary", "memory")
# Mirrors utils.AVAILABLE_MEASUREMENTS, in CLI spelling.
MEASUREMENTS = (
    "precision_at_1",
    "mean_average_precision_at_r",
    "NMI",
    "r_precision",
    "AMI",
    "mean_reciprocal_rank",
    "mean_average_precision",
)


class RunSchedulerError(RuntimeError):
    """Base class for actionable launcher failures."""


class ManifestError(ValueError):
    """Raised when a run manifest is malformed or unsafe."""


class NvidiaSmiError(RunSchedulerError):
    """Raised when GPU state cannot be read from ``nvidia-smi``."""


@dataclass(frozen=True)
class GpuProcess:
    """One CUDA compute process reported by NVIDIA SMI."""

    pid: int
    used_memory_mib: int | None


@dataclass(frozen=True)
class GpuSnapshot:
    """Scheduling-relevant state for one physical GPU."""

    index: int
    uuid: str
    name: str
    total_memory_mib: int
    free_memory_mib: int
    utilization_percent: int | None
    processes: tuple[GpuProcess, ...] = ()

    @property
    def used_memory_mib(self) -> int:
        return max(0, self.total_memory_mib - self.free_memory_mib)


@dataclass(frozen=True)
class StudyReplaySpec:
    """Trial selection, seeds, and final-fit mode for a study-directory replay.

    ``None`` means "not configured here", so a run-level block overrides the
    manifest-wide block field by field and everything left unset keeps the
    ``main.py`` default.
    """

    study_dir_mode: str | None = None
    final_test_top_n: int | None = None
    final_test_trial_numbers: tuple[int, ...] | None = None
    comparison_seeds: tuple[int, ...] | None = None
    comparison_seed_targets: tuple[str, ...] | None = None
    final_test_visualization: tuple[str, ...] | None = None
    # Where the replayed folds keep the test embeddings the concatenated fold
    # evaluation needs: on disk, on disk until it has read them, or in memory.
    fold_test_embedding_storage: str | None = None
    # The reported D_test table. Both the per-fold and the concatenated
    # evaluation happen while the final run still holds the embeddings, so this
    # is the only place the full table can be asked for.
    report_test_metrics: bool | None = None
    test_recall_at_k: tuple[int, ...] | None = None
    test_measurements: tuple[str, ...] | None = None
    # Label-budget dimensions replay the same trials under other data
    # conditions. Each combination becomes its own selection group with its own
    # winner and final fit, exactly as in an experiment-config grid.
    ssl_label_sampling_modes: tuple[str, ...] | None = None
    label_budget_grid: tuple[float, ...] | None = None
    k_shot_grid: tuple[int, ...] | None = None

    @property
    def is_configured(self) -> bool:
        return any(getattr(self, item.name) is not None for item in fields(self))

    @property
    def resolved_study_dir_mode(self) -> str:
        return self.study_dir_mode or DEFAULT_STUDY_DIR_MODE

    def to_arguments(self) -> list[str]:
        """Return the ``main.py`` options for these replay settings."""

        arguments: list[str] = []
        if self.study_dir_mode is not None:
            arguments.extend(["--study_dir_mode", self.study_dir_mode])
        if self.final_test_top_n is not None:
            arguments.extend(["--final_test_top_n", str(self.final_test_top_n)])
        if self.final_test_trial_numbers is not None:
            arguments.append("--final_test_trial_numbers")
            arguments.extend(str(number) for number in self.final_test_trial_numbers)
        if self.comparison_seeds is not None:
            arguments.append("--comparison_seeds")
            arguments.extend(str(seed) for seed in self.comparison_seeds)
        if self.comparison_seed_targets is not None:
            arguments.append("--comparison_seed_targets")
            arguments.extend(self.comparison_seed_targets)
        if self.final_test_visualization is not None:
            arguments.append("--final_test_visualization")
            arguments.extend(self.final_test_visualization)
        if self.fold_test_embedding_storage is not None:
            arguments.extend(
                ["--fold_test_embedding_storage", self.fold_test_embedding_storage]
            )
        if self.report_test_metrics is not None:
            arguments.append(
                "--report_test_metrics"
                if self.report_test_metrics
                else "--no-report_test_metrics"
            )
        if self.test_recall_at_k is not None:
            # An empty ladder is a deliberate "no Recall@K on test", so the
            # option is still passed with no values.
            arguments.append("--test_recall_at_k")
            arguments.extend(str(k) for k in self.test_recall_at_k)
        if self.test_measurements is not None:
            arguments.append("--test_measurements")
            arguments.extend(self.test_measurements)
        if self.ssl_label_sampling_modes is not None:
            arguments.append("--ssl_label_sampling_modes")
            arguments.extend(self.ssl_label_sampling_modes)
        if self.label_budget_grid is not None:
            arguments.append("--label_budget_grid")
            arguments.extend(str(budget) for budget in self.label_budget_grid)
        if self.k_shot_grid is not None:
            arguments.append("--k_shot_grid")
            arguments.extend(str(k_shot) for k_shot in self.k_shot_grid)
        return arguments


@dataclass(frozen=True)
class RunSpec:
    """One complete child process to schedule."""

    name: str
    experiment_config: str | None = None
    hparam_config: str | None = None
    study_dir: str | None = None
    study_replay: StudyReplaySpec = StudyReplaySpec()
    # Set on the manifest's run; every child expanded from it also names the
    # one variant it executes.
    ablation_config: str | None = None
    ablation_variant: str | None = None
    save_dir: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    gpu_memory_mib: int = DEFAULT_GPU_MEMORY_MIB
    slots: int = 1
    gpu: str | None = None
    priority: int = 0
    enabled: bool = True


@dataclass(frozen=True)
class SchedulerConfig:
    """Validated settings loaded from a run manifest."""

    manifest_path: Path
    working_dir: Path
    python: str
    entrypoint: str
    gpu_selectors: tuple[str, ...] | None
    max_concurrent_per_gpu: int
    max_total_runs: int | None
    default_gpu_memory_mib: int
    reserve_gpu_memory_mib: int
    poll_interval_seconds: float
    launch_delay_seconds: float
    fail_fast: bool
    allow_parallel_trials: bool
    scheduler_log_dir: Path
    save_dir_root: str | None
    runs: tuple[RunSpec, ...]
    device: str = "cuda"
    ssl_gpu_selectors: tuple[str, ...] | None = None
    gpu_batch_size_thresholds: tuple[
        device_thresholds.GpuBatchSizeThresholdRule, ...
    ] = ()
    study_replay: StudyReplaySpec = StudyReplaySpec()


@dataclass
class ActiveRun:
    """Mutable process state for a running child."""

    spec: RunSpec
    process: subprocess.Popen
    gpu: GpuSnapshot | None
    command: list[str]
    log_path: Path
    log_file: TextIO
    started_at: str


@dataclass(frozen=True)
class GpuLoad:
    """Owned reservations plus external load on one GPU."""

    snapshot: GpuSnapshot
    active_slots: int
    active_reserved_memory_mib: int
    active_observed_memory_mib: int
    external_processes: int

    @property
    def occupied_slots(self) -> int:
        return self.active_slots + self.external_processes

    @property
    def unallocated_reserved_memory_mib(self) -> int:
        return max(
            0,
            self.active_reserved_memory_mib - self.active_observed_memory_mib,
        )


_TOP_LEVEL_KEYS = {
    "version",
    "working_dir",
    "python",
    "entrypoint",
    "device",
    "gpus",
    "ssl_gpus",
    "gpu_batch_size_thresholds",
    "max_concurrent_per_gpu",
    "max_total_runs",
    "default_gpu_memory_mib",
    "reserve_gpu_memory_mib",
    "poll_interval_seconds",
    "launch_delay_seconds",
    "fail_fast",
    "allow_parallel_trials",
    "scheduler_log_dir",
    "save_dir_root",
    "study_replay",
    "runs",
}
_RUN_KEYS = {
    "name",
    "experiment_config",
    "hparam_config",
    "study_dir",
    "study_replay",
    "ablation_config",
    "save_dir",
    "args",
    "env",
    "gpu_memory_mib",
    "slots",
    "gpu",
    "priority",
    "enabled",
}
_STUDY_REPLAY_KEYS = {item.name for item in fields(StudyReplaySpec)}
# A run that sets ablation_config gets these from the launcher, one variant per child.
_ABLATION_OWNED_OPTIONS = {
    "--ablation_config",
    "--ablation-config",
    "--ablation_variant",
    "--ablation-variant",
}
_SCHEDULER_OWNED_OPTIONS = {
    "--device",
    "--ssl-device",
    "--ssl_device",
    "--experiment-config",
    "--experiment_config",
}
# Only study-replay runs reserve these; ordinary HPO runs may still pass them.
_STUDY_REPLAY_OWNED_OPTIONS = {
    "--final_test_study_dir",
    "--final-test-study-dir",
    "--study_dir_mode",
    "--study-dir-mode",
    "--final_test_study_dir_mode",
    "--final-test-study-dir-mode",
    "--final_test_top_n",
    "--final_test_trial_numbers",
    "--comparison_seeds",
    "--comparison_seed_targets",
    "--comparison-seed-targets",
    "--final_test_visualization",
    "--final-test-visualization",
    "--fold_test_embedding_storage",
    "--fold-test-embedding-storage",
    "--report_test_metrics",
    "--no-report_test_metrics",
    "--report-test-metrics",
    "--test_recall_at_k",
    "--test-recall-at-k",
    "--test_measurements",
    "--test-measurements",
    "--ssl_label_sampling_modes",
    "--label_budget_grid",
    "--k_shot_grid",
}


def _require_mapping(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{source} must be a JSON object")
    return value


def _require_bool(value: Any, source: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{source} must be true or false")
    return value


def _device(value: Any, source: str) -> str:
    if not isinstance(value, str) or value.lower() not in {"cpu", "cuda"}:
        raise ManifestError(f"{source} must be 'cpu' or 'cuda'")
    return value.lower()


def _positive_int(value: Any, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{source} must be a positive integer")
    return value


def _non_negative_int(value: Any, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{source} must be a non-negative integer")
    return value


def _non_negative_float(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ManifestError(f"{source} must be a non-negative number")
    return float(value)


def _positive_float(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ManifestError(f"{source} must be a positive number")
    return float(value)


def _optional_string(value: Any, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{source} must be a non-empty string or null")
    return value


def _string_list(value: Any, source: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{source} must be a JSON array")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise ManifestError(f"{source}[{index}] must be a string or number")
        result.append(str(item))
    return tuple(result)


def _unique_int_list(value: Any, source: str, *, allow_negative: bool = True) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{source} must be a JSON array")
    if not value:
        raise ManifestError(f"{source} must contain at least one integer")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ManifestError(f"{source}[{index}] must be an integer")
        if not allow_negative and item < 0:
            raise ManifestError(f"{source}[{index}] must be a non-negative integer")
        result.append(item)
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ManifestError(f"{source} must not repeat values; duplicates: {duplicates}")
    return tuple(result)


def _label_budget_list(value: Any, source: str) -> tuple[float, ...]:
    """Load labeled-fraction grid values, which are always in (0, 1]."""

    if not isinstance(value, list):
        raise ManifestError(f"{source} must be a JSON array")
    if not value:
        raise ManifestError(f"{source} must contain at least one label budget")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0 < item <= 1:
            raise ManifestError(f"{source}[{index}] must be a number in (0, 1]")
        result.append(float(item))
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ManifestError(f"{source} must not repeat values; duplicates: {duplicates}")
    return tuple(result)


def _k_shot_list(value: Any, source: str) -> tuple[int, ...]:
    """Load k-shot grid values, which count labeled examples per class."""

    k_shots = _unique_int_list(value, source, allow_negative=False)
    non_positive = [k_shot for k_shot in k_shots if k_shot <= 0]
    if non_positive:
        raise ManifestError(f"{source} must contain positive integers; got {non_positive}")
    return k_shots


def _recall_at_k_list(value: Any, source: str) -> tuple[int, ...]:
    """Load a Recall@K ladder, which may be empty to report no ladder at all."""

    if value == []:
        return ()
    ladder = _unique_int_list(value, source, allow_negative=False)
    non_positive = [k for k in ladder if k <= 0]
    if non_positive:
        raise ManifestError(f"{source} must contain positive integers; got {non_positive}")
    return ladder


def _choice(value: Any, source: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ManifestError(f"{source} must be one of {list(choices)}")
    return value


def _choice_list(value: Any, source: str, choices: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{source} must be a JSON array")
    if not value:
        raise ManifestError(f"{source} must contain at least one of {list(choices)}")
    result = [
        _choice(item, f"{source}[{index}]", choices) for index, item in enumerate(value)
    ]
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ManifestError(f"{source} must not repeat values; duplicates: {duplicates}")
    return tuple(result)


def _final_test_visualization_list(value: Any, source: str) -> tuple[str, ...]:
    """Load one or more final-test visualizations, one mode still being a bare string."""

    if isinstance(value, str):
        return (_choice(value, source, FINAL_TEST_VISUALIZATION_MODES),)
    modes = _choice_list(value, source, FINAL_TEST_VISUALIZATION_MODES)
    if FINAL_TEST_VISUALIZATION_NONE in modes and len(modes) > 1:
        raise ManifestError(
            f"{source} combines {FINAL_TEST_VISUALIZATION_NONE!r} with a visualization; "
            "request one or the other"
        )
    return modes


def _load_study_replay(raw: Any, source: str) -> StudyReplaySpec:
    """Load one ``study_replay`` block; unset fields stay ``None`` for merging."""

    values = _require_mapping(raw, source)
    unknown = sorted(set(values) - _STUDY_REPLAY_KEYS)
    if unknown:
        raise ManifestError(
            f"Unknown keys in {source}: {unknown}; known keys: {sorted(_STUDY_REPLAY_KEYS)}"
        )

    def parse(name: str, loader) -> Any:
        value = values.get(name)
        return None if value is None else loader(value, f"{source}.{name}")

    spec = StudyReplaySpec(
        study_dir_mode=parse(
            "study_dir_mode",
            lambda value, item: _choice(value, item, STUDY_DIR_MODES),
        ),
        final_test_top_n=parse("final_test_top_n", _positive_int),
        final_test_trial_numbers=parse(
            "final_test_trial_numbers",
            lambda value, item: _unique_int_list(value, item, allow_negative=False),
        ),
        comparison_seeds=parse("comparison_seeds", _unique_int_list),
        comparison_seed_targets=parse(
            "comparison_seed_targets",
            lambda value, item: _choice_list(value, item, COMPARISON_SEED_TARGETS),
        ),
        final_test_visualization=parse(
            "final_test_visualization",
            _final_test_visualization_list,
        ),
        fold_test_embedding_storage=parse(
            "fold_test_embedding_storage",
            lambda value, item: _choice(value, item, FOLD_TEST_EMBEDDING_STORAGES),
        ),
        report_test_metrics=parse("report_test_metrics", _require_bool),
        test_recall_at_k=parse("test_recall_at_k", _recall_at_k_list),
        test_measurements=parse(
            "test_measurements",
            lambda value, item: _choice_list(value, item, MEASUREMENTS),
        ),
        ssl_label_sampling_modes=parse(
            "ssl_label_sampling_modes",
            lambda value, item: _choice_list(value, item, sorted(LABEL_SAMPLING_MODES)),
        ),
        label_budget_grid=parse("label_budget_grid", _label_budget_list),
        k_shot_grid=parse("k_shot_grid", _k_shot_list),
    )
    if spec.final_test_top_n is not None and spec.final_test_trial_numbers is not None:
        raise ManifestError(
            f"{source} sets both final_test_top_n and final_test_trial_numbers; "
            "choose the best N trials or explicit trial numbers"
        )
    return spec


def merge_study_replay(
    defaults: StudyReplaySpec,
    overrides: StudyReplaySpec,
) -> StudyReplaySpec:
    """Overlay a run's replay block on the manifest-wide block."""

    selection_names = ("final_test_top_n", "final_test_trial_numbers")
    # Trial selection is one decision: an explicit run-level choice replaces the
    # manifest-wide one instead of colliding with it.
    overrides_selection = any(
        getattr(overrides, name) is not None for name in selection_names
    )
    merged = {}
    for item in fields(StudyReplaySpec):
        override_value = getattr(overrides, item.name)
        if item.name in selection_names and overrides_selection:
            merged[item.name] = override_value
            continue
        merged[item.name] = (
            override_value if override_value is not None else getattr(defaults, item.name)
        )
    return StudyReplaySpec(**merged)


def _gpu_batch_size_threshold_rules(
    value: Any,
    source: str,
) -> tuple[device_thresholds.GpuBatchSizeThresholdRule, ...]:
    values = _string_list(value, source)
    rules = []
    for index, item in enumerate(values):
        try:
            rule = device_thresholds.parse_gpu_batch_size_threshold_rule(item)
        except ValueError as exc:
            raise ManifestError(f"Invalid {source}[{index}]: {exc}") from exc
        rules.append(rule)
    return tuple(rules)


def _environment(value: Any, source: str) -> dict[str, str]:
    mapping = _require_mapping(value, source)
    result = {}
    for name, item in mapping.items():
        if not isinstance(name, str) or not name:
            raise ManifestError(f"{source} keys must be non-empty strings")
        if not isinstance(item, (str, int, float, bool)):
            raise ManifestError(f"{source}.{name} must be a scalar value")
        if name in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}:
            raise ManifestError(
                f"{source}.{name} is owned by the run scheduler and cannot be overridden"
            )
        result[name] = str(item)
    return result


def _validate_extra_args(
    args: tuple[str, ...],
    run_name: str,
    *,
    replays_study: bool,
    ablates: bool = False,
) -> None:
    for token in args:
        option = token.split("=", 1)[0]
        if ablates and option in _ABLATION_OWNED_OPTIONS:
            raise ManifestError(
                f"run {run_name!r} cannot set {option} in args; its ablation_config field "
                "already names the config, and the launcher selects one variant per child"
            )
        if option in _SCHEDULER_OWNED_OPTIONS:
            raise ManifestError(
                f"run {run_name!r} cannot set {option} in args; use the manifest's "
                "experiment_config field and let the scheduler assign training/SSL devices"
            )
        if replays_study and option in _STUDY_REPLAY_OWNED_OPTIONS:
            raise ManifestError(
                f"run {run_name!r} cannot set {option} in args; configure it in the "
                "manifest's study_dir/study_replay fields"
            )


def expand_ablation_runs(runs: Sequence[RunSpec], working_dir: Path) -> tuple[RunSpec, ...]:
    """Replace every run that names an ablation config with one run per variant.

    Children are named ``<run>_<variant>`` and, when the run pins its own
    ``save_dir``, write to ``<save_dir>/<variant>``, so variants never share
    outputs. The config is parsed here as well as in the child so a malformed
    ablation fails the manifest before anything launches. Disabled runs are
    never launched and are left unexpanded.
    """

    expanded: list[RunSpec] = []
    for run in runs:
        if run.ablation_config is None or not run.enabled:
            expanded.append(run)
            continue
        path = _workspace_path(run.ablation_config, working_dir).resolve()
        try:
            config = ablation.load_ablation_config(path)
        except (OSError, ValueError) as exc:
            raise ManifestError(f"Invalid ablation_config for run {run.name!r}: {exc}") from exc
        for variant in config["variants"]:
            expanded.append(
                replace(
                    run,
                    name=f"{run.name}_{variant}",
                    ablation_config=str(path),
                    ablation_variant=variant,
                    save_dir=None if run.save_dir is None else str(Path(run.save_dir) / variant),
                )
            )
    return tuple(expanded)


def _load_run_spec(
    raw: Any,
    *,
    index: int,
    default_gpu_memory_mib: int,
) -> RunSpec:
    source = f"runs[{index}]"
    values = _require_mapping(raw, source)
    unknown = sorted(set(values) - _RUN_KEYS)
    if unknown:
        raise ManifestError(f"Unknown keys in {source}: {unknown}")

    name = _optional_string(values.get("name"), f"{source}.name")
    if name is None:
        raise ManifestError(f"{source}.name is required")
    if not _RUN_NAME_PATTERN.fullmatch(name):
        raise ManifestError(
            f"{source}.name must match {_RUN_NAME_PATTERN.pattern!r}; got {name!r}"
        )

    study_dir = _optional_string(values.get("study_dir"), f"{source}.study_dir")
    study_replay = _load_study_replay(
        values.get("study_replay", {}),
        f"{source}.study_replay",
    )
    if study_dir is None and study_replay.is_configured:
        raise ManifestError(
            f"{source}.study_replay requires {source}.study_dir; replay settings only "
            "apply to a run that replays an existing HPO study directory"
        )
    if study_dir is not None and values.get("hparam_config") is not None:
        raise ManifestError(
            f"{source} sets both study_dir and hparam_config. A study-directory replay "
            "reuses the search configuration saved in study_config.json and never "
            "schedules new trials; remove hparam_config"
        )

    ablation_config = _optional_string(
        values.get("ablation_config"), f"{source}.ablation_config"
    )
    args = _string_list(values.get("args", []), f"{source}.args")
    _validate_extra_args(
        args,
        name,
        replays_study=study_dir is not None,
        ablates=ablation_config is not None,
    )
    gpu = values.get("gpu")
    if gpu is not None:
        if isinstance(gpu, bool) or not isinstance(gpu, (str, int)):
            raise ManifestError(f"{source}.gpu must be a GPU index/UUID string, integer, or null")
        gpu = str(gpu)

    priority = values.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ManifestError(f"{source}.priority must be an integer")

    return RunSpec(
        name=name,
        experiment_config=_optional_string(
            values.get("experiment_config"), f"{source}.experiment_config"
        ),
        hparam_config=_optional_string(
            values.get("hparam_config"), f"{source}.hparam_config"
        ),
        study_dir=study_dir,
        study_replay=study_replay,
        ablation_config=ablation_config,
        save_dir=_optional_string(values.get("save_dir"), f"{source}.save_dir"),
        args=args,
        env=_environment(values.get("env", {}), f"{source}.env"),
        gpu_memory_mib=_positive_int(
            values.get("gpu_memory_mib", default_gpu_memory_mib),
            f"{source}.gpu_memory_mib",
        ),
        slots=_positive_int(values.get("slots", 1), f"{source}.slots"),
        gpu=gpu,
        priority=priority,
        enabled=_require_bool(values.get("enabled", True), f"{source}.enabled"),
    )


def load_scheduler_config(path: str | Path) -> SchedulerConfig:
    """Load and strictly validate a scheduler manifest."""

    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Run manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in run manifest {manifest_path}: {exc}") from exc

    values = _require_mapping(raw, f"run manifest {manifest_path}")
    unknown = sorted(set(values) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ManifestError(f"Unknown run-manifest keys: {unknown}")

    version = values.get("version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"Unsupported run-manifest version {version!r}; expected {MANIFEST_VERSION}"
        )

    working_dir_value = values.get("working_dir")
    working_dir = (
        Path.cwd()
        if working_dir_value is None
        else Path(_optional_string(working_dir_value, "working_dir"))
    ).resolve()
    if not working_dir.is_dir():
        raise ManifestError(f"working_dir is not a directory: {working_dir}")

    default_gpu_memory_mib = _positive_int(
        values.get("default_gpu_memory_mib", DEFAULT_GPU_MEMORY_MIB),
        "default_gpu_memory_mib",
    )
    raw_runs = values.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ManifestError("runs must be a non-empty JSON array")
    runs = tuple(
        _load_run_spec(
            item,
            index=index,
            default_gpu_memory_mib=default_gpu_memory_mib,
        )
        for index, item in enumerate(raw_runs)
    )
    runs = expand_ablation_runs(runs, working_dir)
    enabled_names = [run.name for run in runs if run.enabled]
    duplicates = sorted({name for name in enabled_names if enabled_names.count(name) > 1})
    if duplicates:
        raise ManifestError(f"Enabled run names must be unique; duplicates: {duplicates}")
    if not enabled_names:
        raise ManifestError("The manifest does not contain any enabled runs")

    raw_gpus = values.get("gpus")
    gpu_selectors = None if raw_gpus is None else _string_list(raw_gpus, "gpus")
    if gpu_selectors == ():
        raise ManifestError("gpus must contain at least one index or UUID when provided")

    raw_ssl_gpus = values.get("ssl_gpus")
    ssl_gpu_selectors = (
        None
        if raw_ssl_gpus is None
        else _string_list(raw_ssl_gpus, "ssl_gpus")
    )
    if ssl_gpu_selectors == ():
        raise ManifestError(
            "ssl_gpus must contain at least one index or UUID when provided"
        )

    max_total_runs = values.get("max_total_runs")
    if max_total_runs is not None:
        max_total_runs = _positive_int(max_total_runs, "max_total_runs")

    scheduler_log_dir = Path(
        _optional_string(
            values.get("scheduler_log_dir", "logs/run_scheduler"),
            "scheduler_log_dir",
        )
    )
    if not scheduler_log_dir.is_absolute():
        scheduler_log_dir = working_dir / scheduler_log_dir

    config = SchedulerConfig(
        manifest_path=manifest_path,
        working_dir=working_dir,
        python=_optional_string(values.get("python", sys.executable), "python"),
        entrypoint=_optional_string(values.get("entrypoint", "main.py"), "entrypoint"),
        device=_device(values.get("device", "cuda"), "device"),
        gpu_selectors=gpu_selectors,
        max_concurrent_per_gpu=_positive_int(
            values.get("max_concurrent_per_gpu", 1),
            "max_concurrent_per_gpu",
        ),
        max_total_runs=max_total_runs,
        default_gpu_memory_mib=default_gpu_memory_mib,
        reserve_gpu_memory_mib=_non_negative_int(
            values.get("reserve_gpu_memory_mib", DEFAULT_RESERVE_GPU_MEMORY_MIB),
            "reserve_gpu_memory_mib",
        ),
        poll_interval_seconds=_positive_float(
            values.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS),
            "poll_interval_seconds",
        ),
        launch_delay_seconds=_non_negative_float(
            values.get("launch_delay_seconds", 0),
            "launch_delay_seconds",
        ),
        fail_fast=_require_bool(values.get("fail_fast", False), "fail_fast"),
        allow_parallel_trials=_require_bool(
            values.get("allow_parallel_trials", False),
            "allow_parallel_trials",
        ),
        scheduler_log_dir=scheduler_log_dir.resolve(),
        save_dir_root=_optional_string(values.get("save_dir_root"), "save_dir_root"),
        runs=runs,
        ssl_gpu_selectors=ssl_gpu_selectors,
        gpu_batch_size_thresholds=_gpu_batch_size_threshold_rules(
            values.get("gpu_batch_size_thresholds", []),
            "gpu_batch_size_thresholds",
        ),
        study_replay=_load_study_replay(
            values.get("study_replay", {}),
            "study_replay",
        ),
    )
    _validate_device_pools(config)
    validate_run_files_and_hpo(config)
    validate_study_replay_runs(config)
    return config


def _workspace_path(value: str, working_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else working_dir / path


def _read_json_object(path: Path, source: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{source} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {source} {path}: {exc}") from exc
    return _require_mapping(value, f"{source} {path}")


def _last_option_value(args: Sequence[str], names: set[str]) -> str | None:
    result = None
    index = 0
    while index < len(args):
        token = args[index]
        option, separator, inline_value = token.partition("=")
        if option in names:
            if separator:
                result = inline_value
            elif index + 1 < len(args):
                result = args[index + 1]
                index += 1
        index += 1
    return result


def _resolved_hparam_config_path(
    run: RunSpec,
    experiment_values: dict[str, Any],
    working_dir: Path,
) -> Path | None:
    value = experiment_values.get("hparam_config")
    if run.hparam_config is not None:
        value = run.hparam_config
    args_value = _last_option_value(run.args, {"--hparam_config", "--hparam-config"})
    if args_value is not None:
        value = args_value
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"run {run.name!r} resolves hparam_config to a non-string value")
    return _workspace_path(value, working_dir).resolve()


def _study_dir_candidates(value: str, working_dir: Path) -> tuple[Path, ...]:
    """Return the paths ``main.py`` searches for a study directory, in order."""

    path = Path(value)
    if path.is_absolute():
        return (path,)
    return (working_dir / path, working_dir / "logs" / path)


def resolve_study_dir(value: str, working_dir: Path) -> Path:
    """Return the existing study directory, mirroring main.py's logs/ fallback."""

    candidates = _study_dir_candidates(value, working_dir)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def effective_study_replay(run: RunSpec, config: SchedulerConfig) -> StudyReplaySpec:
    """Return one run's replay settings after the manifest-wide defaults."""

    return merge_study_replay(config.study_replay, run.study_replay)


def validate_study_replay_runs(config: SchedulerConfig) -> None:
    """Fail early for unusable study directories and replay-setting combinations."""

    replay_runs = [run for run in config.runs if run.enabled and run.study_dir is not None]
    if config.study_replay.is_configured and not replay_runs:
        raise ManifestError(
            "study_replay is configured, but no enabled run sets study_dir. Give each "
            "run the HPO study directory it replays, or remove study_replay"
        )

    for run in replay_runs:
        study_dir = resolve_study_dir(run.study_dir, config.working_dir)
        if not study_dir.is_dir():
            searched = [
                str(candidate)
                for candidate in _study_dir_candidates(run.study_dir, config.working_dir)
            ]
            raise ManifestError(
                f"study_dir for run {run.name!r} does not exist; searched: {searched}"
            )
        study_config_path = study_dir / "study_config.json"
        if not study_config_path.is_file():
            raise ManifestError(
                f"study_dir for run {run.name!r} is not an HPO study directory: "
                f"{study_config_path} is missing"
            )
        _validate_study_replay(
            effective_study_replay(run, config),
            f"run {run.name!r}",
        )


def _validate_study_replay(spec: StudyReplaySpec, source: str) -> None:
    if (
        spec.k_shot_grid is not None
        and spec.ssl_label_sampling_modes is not None
        and LABEL_SAMPLING_MODE_CLASS_SUBSET_K_SHOT not in spec.ssl_label_sampling_modes
    ):
        raise ManifestError(
            f"{source} sets k_shot_grid with "
            f"ssl_label_sampling_modes={list(spec.ssl_label_sampling_modes)}; a k-shot "
            f"grid needs {LABEL_SAMPLING_MODE_CLASS_SUBSET_K_SHOT!r}. Leave "
            "ssl_label_sampling_modes unset to keep the study's own sampling mode"
        )
    if spec.resolved_study_dir_mode != STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL:
        return
    seeds = spec.comparison_seeds
    if seeds is None or len(set(seeds)) < 2:
        raise ManifestError(
            f"{source} uses study_dir_mode='cross_seed_train_val', which needs at least "
            "two distinct comparison_seeds"
        )
    targets = spec.comparison_seed_targets or COMPARISON_SEED_TARGETS
    if not set(targets) & set(CROSS_SEED_VALIDATION_TARGETS):
        raise ManifestError(
            f"{source} uses study_dir_mode='cross_seed_train_val' with "
            f"comparison_seed_targets={list(targets)}; include at least one of "
            f"{list(CROSS_SEED_VALIDATION_TARGETS)} because hparam_seed alone does not "
            "change a fixed-parameter validation replay"
        )


def validate_run_files_and_hpo(config: SchedulerConfig) -> None:
    """Fail early for missing configs and accidental trial-level parallelism."""

    entrypoint_path = _workspace_path(config.entrypoint, config.working_dir)
    if not entrypoint_path.is_file():
        raise ManifestError(f"entrypoint does not exist: {entrypoint_path}")

    for run in config.runs:
        if not run.enabled:
            continue
        experiment_values: dict[str, Any] = {}
        if run.experiment_config is not None:
            experiment_path = _workspace_path(
                run.experiment_config, config.working_dir
            ).resolve()
            experiment_values = _read_json_object(
                experiment_path,
                f"experiment_config for run {run.name!r}",
            )
        if run.study_dir is not None:
            # A replay reuses the search configuration saved in the study
            # directory and never schedules new trials.
            continue
        hparam_path = _resolved_hparam_config_path(
            run,
            experiment_values,
            config.working_dir,
        )
        if hparam_path is None:
            continue
        hparam_values = _read_json_object(
            hparam_path,
            f"hparam_config for run {run.name!r}",
        )
        if not hparam_values.get("enabled", True):
            continue
        n_jobs = hparam_values.get("n_jobs", 1)
        if n_jobs != 1 and not config.allow_parallel_trials:
            raise ManifestError(
                f"run {run.name!r} has hparam_config n_jobs={n_jobs!r}. "
                "This launcher parallelizes complete runs; set n_jobs to 1 for sequential TPE, "
                "or set allow_parallel_trials=true only if nested trial parallelism is intentional."
            )


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported", "[not supported]"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_gpu_query_output(output: str) -> list[GpuSnapshot]:
    """Parse ``nvidia-smi --query-gpu`` CSV output."""

    snapshots = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 6:
            raise NvidiaSmiError(f"Unexpected nvidia-smi GPU row: {row!r}")
        index, uuid, name, total, free, utilization = (item.strip() for item in row)
        try:
            snapshots.append(
                GpuSnapshot(
                    index=int(index),
                    uuid=uuid,
                    name=name,
                    total_memory_mib=int(float(total)),
                    free_memory_mib=int(float(free)),
                    utilization_percent=_parse_optional_int(utilization),
                )
            )
        except ValueError as exc:
            raise NvidiaSmiError(f"Unexpected nvidia-smi GPU row: {row!r}") from exc
    if not snapshots:
        raise NvidiaSmiError("nvidia-smi did not report any GPUs")
    return snapshots


def parse_compute_process_query_output(output: str) -> dict[str, tuple[GpuProcess, ...]]:
    """Parse ``nvidia-smi --query-compute-apps`` CSV output by GPU UUID."""

    by_uuid: dict[str, list[GpuProcess]] = {}
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 3:
            raise NvidiaSmiError(f"Unexpected nvidia-smi compute-process row: {row!r}")
        uuid, pid, used_memory = (item.strip() for item in row)
        try:
            process = GpuProcess(
                pid=int(pid),
                used_memory_mib=_parse_optional_int(used_memory),
            )
        except ValueError as exc:
            raise NvidiaSmiError(
                f"Unexpected nvidia-smi compute-process row: {row!r}"
            ) from exc
        by_uuid.setdefault(uuid, []).append(process)
    return {uuid: tuple(processes) for uuid, processes in by_uuid.items()}


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise NvidiaSmiError(
            "nvidia-smi was not found. Run the scheduler on the NVIDIA GPU host."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise NvidiaSmiError(
            f"nvidia-smi failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stdout


def query_gpu_snapshots() -> list[GpuSnapshot]:
    """Return a current physical-GPU and compute-process snapshot."""

    gpu_output = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_output = _run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = parse_compute_process_query_output(process_output)
    return [
        replace(snapshot, processes=processes.get(snapshot.uuid, ()))
        for snapshot in parse_gpu_query_output(gpu_output)
    ]


def _resolve_selector(selector: str, snapshots: Sequence[GpuSnapshot]) -> GpuSnapshot:
    selector = selector.strip()
    index_matches = [snapshot for snapshot in snapshots if str(snapshot.index) == selector]
    uuid_matches = [
        snapshot
        for snapshot in snapshots
        if snapshot.uuid == selector or snapshot.uuid.startswith(selector)
    ]
    matches = index_matches or uuid_matches
    if len(matches) != 1:
        available = [f"{gpu.index} ({gpu.uuid})" for gpu in snapshots]
        raise ManifestError(
            f"GPU selector {selector!r} does not identify exactly one GPU. Available: {available}"
        )
    return matches[0]


def select_gpus(
    snapshots: Sequence[GpuSnapshot],
    selectors: Sequence[str] | None,
    inherited_cuda_visible_devices: str | None,
) -> list[GpuSnapshot]:
    """Select GPUs while respecting an inherited CUDA allocation."""

    inherited = None
    if inherited_cuda_visible_devices is not None:
        inherited = tuple(
            token.strip()
            for token in inherited_cuda_visible_devices.split(",")
            if token.strip()
        )
        if not inherited or inherited == ("-1",):
            raise ManifestError(
                "CUDA_VISIBLE_DEVICES hides all GPUs from the launcher environment"
            )

    effective_selectors = selectors if selectors is not None else inherited
    if effective_selectors is None:
        selected = list(snapshots)
    else:
        selected = [_resolve_selector(selector, snapshots) for selector in effective_selectors]

    if inherited is not None and selectors is not None:
        inherited_uuids = {
            _resolve_selector(selector, snapshots).uuid for selector in inherited
        }
        escaped = [gpu for gpu in selected if gpu.uuid not in inherited_uuids]
        if escaped:
            raise ManifestError(
                "Requested GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation: "
                f"{[gpu.index for gpu in escaped]}"
            )

    uuids = [gpu.uuid for gpu in selected]
    if len(set(uuids)) != len(uuids):
        raise ManifestError("The selected GPU list contains duplicates")
    return sorted(selected, key=lambda gpu: gpu.index)


def make_gpu_loads(
    snapshots: Sequence[GpuSnapshot],
    active_runs: Iterable[ActiveRun],
) -> dict[str, GpuLoad]:
    """Combine live NVIDIA state with reservations owned by this scheduler."""

    active_by_uuid: dict[str, list[ActiveRun]] = {gpu.uuid: [] for gpu in snapshots}
    for active in active_runs:
        if active.gpu is None:
            raise RunSchedulerError("A CPU run cannot be included in GPU load accounting")
        active_by_uuid.setdefault(active.gpu.uuid, []).append(active)

    loads = {}
    for snapshot in snapshots:
        owned = active_by_uuid.get(snapshot.uuid, [])
        owned_pids = {active.process.pid for active in owned}
        observed_owned_memory = sum(
            process.used_memory_mib or 0
            for process in snapshot.processes
            if process.pid in owned_pids
        )
        external_pids = {
            process.pid for process in snapshot.processes if process.pid not in owned_pids
        }
        loads[snapshot.uuid] = GpuLoad(
            snapshot=snapshot,
            active_slots=sum(active.spec.slots for active in owned),
            active_reserved_memory_mib=sum(
                active.spec.gpu_memory_mib for active in owned
            ),
            active_observed_memory_mib=observed_owned_memory,
            external_processes=len(external_pids),
        )
    return loads


def run_fits_gpu(run: RunSpec, load: GpuLoad, config: SchedulerConfig) -> bool:
    """Return whether slot and conservative memory reservations admit ``run``."""

    if load.occupied_slots + run.slots > config.max_concurrent_per_gpu:
        return False
    projected_used = (
        load.snapshot.used_memory_mib
        + load.unallocated_reserved_memory_mib
        + run.gpu_memory_mib
        + config.reserve_gpu_memory_mib
    )
    return projected_used <= load.snapshot.total_memory_mib


def choose_gpu(
    run: RunSpec,
    loads: dict[str, GpuLoad],
    config: SchedulerConfig,
) -> GpuSnapshot | None:
    """Choose the least-loaded admissible GPU, honoring optional affinity."""

    candidates = list(loads.values())
    if run.gpu is not None:
        selected = _resolve_selector(
            run.gpu,
            [load.snapshot for load in candidates],
        )
        candidates = [loads[selected.uuid]]
    candidates = [load for load in candidates if run_fits_gpu(run, load, config)]
    if not candidates:
        return None

    def score(load: GpuLoad):
        utilization = load.snapshot.utilization_percent
        return (
            load.occupied_slots / config.max_concurrent_per_gpu,
            101 if utilization is None else utilization,
            -load.snapshot.free_memory_mib,
            load.snapshot.index,
        )

    return min(candidates, key=score).snapshot


def _effective_save_dir(run: RunSpec, config: SchedulerConfig) -> str | None:
    if run.save_dir is not None:
        return run.save_dir
    if config.save_dir_root is not None:
        return str(Path(config.save_dir_root) / run.name)
    return None


def _uses_gpu_placement(config: SchedulerConfig) -> bool:
    """Return whether each child needs an assigned physical GPU."""

    return (
        config.device == "cuda"
        or config.ssl_gpu_selectors is not None
        or _uses_batch_gpu(config)
    )


def _uses_batch_gpu(config: SchedulerConfig) -> bool:
    """Return whether CPU runs may switch to their configured GPU by batch size."""

    return config.device == "cpu" and config.gpu_selectors is not None


def _child_ssl_device(config: SchedulerConfig) -> str:
    """Return the child-visible device for out-of-batch SSL computation."""

    return (
        "cuda"
        if (
            config.device == "cuda"
            or config.ssl_gpu_selectors is not None
            or _uses_batch_gpu(config)
        )
        else "cpu"
    )


def _placement_selectors(config: SchedulerConfig) -> tuple[str, ...] | None:
    return (
        config.gpu_selectors
        if config.device == "cuda" or _uses_batch_gpu(config)
        else config.ssl_gpu_selectors
    )


def _placement_description(
    config: SchedulerConfig,
    gpu: GpuSnapshot | None,
) -> str:
    if gpu is None:
        return "CPU"
    if config.device == "cuda":
        return f"GPU {gpu.index}"
    if _uses_batch_gpu(config):
        if config.gpu_batch_size_thresholds:
            return (
                f"CPU + SSL GPU {gpu.index} "
                "(loss/SSL-specific training GPU thresholds; "
                f"default effective batch size > {GPU_BATCH_SIZE_THRESHOLD})"
            )
        return (
            f"CPU + SSL GPU {gpu.index} "
            f"(training GPU for effective batch size > {GPU_BATCH_SIZE_THRESHOLD})"
        )
    return f"CPU + SSL GPU {gpu.index}"


def _validate_device_pools(config: SchedulerConfig) -> None:
    if config.device == "cuda" and config.ssl_gpu_selectors is not None:
        raise ManifestError(
            "ssl_gpus is for CPU-training runs. CUDA-training runs already use "
            "their assigned gpus for SSL computation; remove ssl_gpus or set device='cpu'."
        )
    if (
        config.device == "cpu"
        and config.gpu_selectors is not None
        and config.ssl_gpu_selectors is not None
    ):
        raise ManifestError(
            "CPU runs cannot configure both gpus and ssl_gpus: gpus already uses "
            "the assigned GPU for SSL plus batch-triggered training. Use ssl_gpus "
            "only when GPU acceleration must be limited to SSL."
        )


def build_run_command(run: RunSpec, config: SchedulerConfig) -> list[str]:
    """Build a shell-free argv vector for one child."""

    entrypoint = _workspace_path(config.entrypoint, config.working_dir).resolve()
    command = [config.python, str(entrypoint)]
    if run.experiment_config is not None:
        experiment_path = _workspace_path(
            run.experiment_config, config.working_dir
        ).resolve()
        command.extend(["--experiment-config", str(experiment_path)])
    if run.hparam_config is not None:
        hparam_path = _workspace_path(run.hparam_config, config.working_dir).resolve()
        command.extend(["--hparam_config", str(hparam_path)])
    if run.study_dir is not None:
        study_dir = resolve_study_dir(run.study_dir, config.working_dir)
        command.extend(["--final_test_study_dir", str(study_dir)])
        command.extend(effective_study_replay(run, config).to_arguments())
    if run.ablation_config is not None:
        ablation_path = _workspace_path(run.ablation_config, config.working_dir).resolve()
        command.extend(["--ablation_config", str(ablation_path)])
        if run.ablation_variant is not None:
            command.extend(["--ablation_variant", run.ablation_variant])
    save_dir = _effective_save_dir(run, config)
    if save_dir is not None:
        command.extend(["--save_dir", save_dir])
    command.extend(run.args)
    # These override experiment-config values. Batch-triggered training-device
    # selection happens in the child after its resolved batch size is known;
    # SSL continuously uses the assigned GPU for either GPU-pool mode.
    command.extend(
        [
            "--ssl-device",
            _child_ssl_device(config),
            "--device",
            config.device,
        ]
    )
    return command


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_directory(config: SchedulerConfig) -> Path:
    token = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.scheduler_log_dir / f"{token}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _child_environment(
    run: RunSpec,
    gpu: GpuSnapshot | None,
    config: SchedulerConfig,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(run.env)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    # Children size their own thread pools from this. setdefault keeps an
    # explicit export (shell or run.env) authoritative.
    environment.setdefault(
        cpu_threads.THREAD_BUDGET_ENV, str(_child_thread_budget(config))
    )
    environment["RUN_SCHEDULER_DEVICE"] = config.device
    environment["RUN_SCHEDULER_SSL_DEVICE"] = _child_ssl_device(config)
    environment["RUN_SCHEDULER_RUN_NAME"] = run.name
    if _uses_batch_gpu(config):
        environment[device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV] = str(
            GPU_BATCH_SIZE_THRESHOLD
        )
        if config.gpu_batch_size_thresholds:
            environment[
                device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV
            ] = device_thresholds.encode_gpu_batch_size_threshold_rules(
                config.gpu_batch_size_thresholds
            )
        else:
            environment.pop(
                device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV,
                None,
            )
    else:
        environment.pop(
            device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLD_ENV,
            None,
        )
        environment.pop(
            device_thresholds.RUN_SCHEDULER_GPU_BATCH_SIZE_THRESHOLDS_ENV,
            None,
        )
    if not _uses_gpu_placement(config):
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment.pop("RUN_SCHEDULER_PHYSICAL_GPU", None)
        environment.pop("RUN_SCHEDULER_GPU_UUID", None)
        environment.pop("RUN_SCHEDULER_PHYSICAL_SSL_GPU", None)
        environment.pop("RUN_SCHEDULER_SSL_GPU_UUID", None)
    else:
        if gpu is None:
            raise RunSchedulerError("GPU-accelerated runs require an assigned GPU")
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        uses_training_gpu = config.device == "cuda" or _uses_batch_gpu(config)
        uses_ssl_gpu = _child_ssl_device(config) == "cuda"
        if uses_training_gpu:
            environment["RUN_SCHEDULER_PHYSICAL_GPU"] = str(gpu.index)
            environment["RUN_SCHEDULER_GPU_UUID"] = gpu.uuid
        else:
            environment.pop("RUN_SCHEDULER_PHYSICAL_GPU", None)
            environment.pop("RUN_SCHEDULER_GPU_UUID", None)
        if uses_ssl_gpu:
            environment["RUN_SCHEDULER_PHYSICAL_SSL_GPU"] = str(gpu.index)
            environment["RUN_SCHEDULER_SSL_GPU_UUID"] = gpu.uuid
        else:
            environment.pop("RUN_SCHEDULER_PHYSICAL_SSL_GPU", None)
            environment.pop("RUN_SCHEDULER_SSL_GPU_UUID", None)
    return environment


def _child_thread_budget(config: SchedulerConfig) -> int:
    """Split the host's physical cores across the children admitted at once.

    CPU mode admits one child at a time by default, so the common case hands the
    whole host to that child. Raising ``max_total_runs`` divides the cores rather
    than letting every child spawn a host-wide pool and contend with the others.
    """

    concurrency = _max_active_runs(config) or 1
    return cpu_threads.resolve_thread_budget(concurrency=max(1, concurrency))


def _popen_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def launch_run(
    run: RunSpec,
    gpu: GpuSnapshot | None,
    config: SchedulerConfig,
    session_dir: Path,
) -> ActiveRun:
    """Start one isolated child and redirect its combined output to a log."""

    command = build_run_command(run, config)
    log_path = session_dir / f"{run.name}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    started_at = _utc_now()
    log_file.write(f"started_at={started_at}\n")
    log_file.write(f"device={config.device}\n")
    log_file.write(f"ssl_device={_child_ssl_device(config)}\n")
    if _uses_batch_gpu(config):
        log_file.write(
            f"device_when_effective_batch_size_gt_{GPU_BATCH_SIZE_THRESHOLD}=cuda\n"
        )
        if config.gpu_batch_size_thresholds:
            log_file.write(
                "gpu_batch_size_thresholds="
                + device_thresholds.encode_gpu_batch_size_threshold_rules(
                    config.gpu_batch_size_thresholds
                )
                + "\n"
            )
    if gpu is not None:
        if config.device == "cuda" or _uses_batch_gpu(config):
            log_file.write(f"physical_gpu={gpu.index}\n")
            log_file.write(f"gpu_uuid={gpu.uuid}\n")
        if _child_ssl_device(config) == "cuda":
            log_file.write(f"physical_ssl_gpu={gpu.index}\n")
            log_file.write(f"ssl_gpu_uuid={gpu.uuid}\n")
    log_file.write(f"command={_format_command(command)}\n\n")
    try:
        process = subprocess.Popen(
            command,
            cwd=config.working_dir,
            env=_child_environment(run, gpu, config),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **_popen_group_kwargs(),
        )
    except BaseException:
        log_file.close()
        raise
    return ActiveRun(
        spec=run,
        process=process,
        gpu=gpu,
        command=command,
        log_path=log_path,
        log_file=log_file,
        started_at=started_at,
    )


def _initial_statuses(runs: Sequence[RunSpec]) -> dict[str, dict[str, Any]]:
    return {
        run.name: {
            "name": run.name,
            "status": "disabled" if not run.enabled else "queued",
            "priority": run.priority,
            "slots": run.slots,
            "gpu_memory_mib": run.gpu_memory_mib,
        }
        for run in runs
    }


def _write_state(
    path: Path,
    config: SchedulerConfig,
    statuses: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "manifest": str(config.manifest_path),
        "updated_at": _utc_now(),
        "runs": [statuses[run.name] for run in config.runs],
    }
    temp_path = path.with_suffix(f".{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _terminate_active(active_runs: Sequence[ActiveRun]) -> None:
    for active in active_runs:
        if active.process.poll() is not None:
            continue
        try:
            if os.name != "nt":
                os.killpg(active.process.pid, signal.SIGTERM)
            else:
                active.process.terminate()
        except (OSError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 5.0
    for active in active_runs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            active.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(active.process.pid, signal.SIGKILL)
                else:
                    active.process.kill()
            except (OSError, ProcessLookupError):
                pass
        finally:
            active.log_file.close()


def _sorted_pending(runs: Iterable[RunSpec]) -> list[RunSpec]:
    # Larger reservations are placed first to reduce bin-packing fragmentation;
    # priority remains the strongest user-controlled ordering signal.
    return sorted(
        runs,
        key=lambda run: (-run.priority, -run.slots, -run.gpu_memory_mib, run.name),
    )


def _refresh_selected_snapshots(
    selected_uuids: set[str],
) -> list[GpuSnapshot]:
    snapshots = query_gpu_snapshots()
    selected = [gpu for gpu in snapshots if gpu.uuid in selected_uuids]
    missing = selected_uuids - {gpu.uuid for gpu in selected}
    if missing:
        raise NvidiaSmiError(f"Previously selected GPUs disappeared: {sorted(missing)}")
    return selected


def _validate_static_capacity(
    runs: Sequence[RunSpec],
    selected: Sequence[GpuSnapshot],
    config: SchedulerConfig,
) -> None:
    for run in runs:
        if run.slots > config.max_concurrent_per_gpu:
            raise ManifestError(
                f"run {run.name!r} requests {run.slots} slots, but "
                f"max_concurrent_per_gpu={config.max_concurrent_per_gpu}"
            )
        candidate_gpus = list(selected)
        if run.gpu is not None:
            candidate_gpus = [_resolve_selector(run.gpu, selected)]
        if not any(
            run.gpu_memory_mib + config.reserve_gpu_memory_mib <= gpu.total_memory_mib
            for gpu in candidate_gpus
        ):
            raise ManifestError(
                f"run {run.name!r} reserves {run.gpu_memory_mib} MiB plus "
                f"{config.reserve_gpu_memory_mib} MiB headroom, which cannot fit its eligible GPU(s)"
            )


def _print_gpu_summary(selected: Sequence[GpuSnapshot], config: SchedulerConfig) -> None:
    if config.device == "cuda":
        role = "training/SSL"
    elif _uses_batch_gpu(config):
        if config.gpu_batch_size_thresholds:
            role = (
                "SSL + loss/SSL-specific workload-threshold training "
                f"(default > {GPU_BATCH_SIZE_THRESHOLD})"
            )
        else:
            role = (
                f"SSL + effective batch size > {GPU_BATCH_SIZE_THRESHOLD} training"
            )
    else:
        role = "SSL"
    print(
        f"Selected {len(selected)} {role} GPU(s); max {config.max_concurrent_per_gpu} "
        "run slot(s) per GPU."
    )
    for gpu in selected:
        utilization = "n/a" if gpu.utilization_percent is None else f"{gpu.utilization_percent}%"
        print(
            f"  GPU {gpu.index}: {gpu.name}, free={gpu.free_memory_mib}/{gpu.total_memory_mib} MiB, "
            f"util={utilization}, existing_compute_processes={len(gpu.processes)}"
        )


def _validate_cpu_runs(runs: Sequence[RunSpec], config: SchedulerConfig) -> None:
    if config.ssl_gpu_selectors is not None or _uses_batch_gpu(config):
        return
    pinned = [run.name for run in runs if run.gpu is not None]
    if pinned:
        raise ManifestError(
            "Per-run GPU affinity is not valid when device='cpu'; remove 'gpu' from "
            f"these runs: {pinned}"
        )


def _max_active_runs(config: SchedulerConfig) -> int | None:
    if config.max_total_runs is not None:
        return config.max_total_runs
    # A pure CPU process may use all host cores, so one child is the safe
    # default. An assigned GPU pool intentionally opts into GPU-shaped
    # concurrency; max_total_runs remains available to cap CPU/RAM pressure.
    return 1 if not _uses_gpu_placement(config) else None


def dry_run_cpu_schedule(config: SchedulerConfig) -> int:
    """Print the first CPU wave without starting processes."""

    pending = _sorted_pending(run for run in config.runs if run.enabled)
    max_active_runs = _max_active_runs(config)
    launch_count = min(len(pending), max_active_runs or len(pending))
    for run in pending[:launch_count]:
        print(
            f"DRY RUN  {run.name} -> CPU; "
            f"command: {_format_command(build_run_command(run, config))}"
        )
    for run in pending[launch_count:]:
        print(f"QUEUED   {run.name} (waits for a CPU run slot from the first wave)")
    return 0


def dry_run_schedule(config: SchedulerConfig, selected: Sequence[GpuSnapshot]) -> int:
    """Print the immediately launchable wave without starting processes."""

    pending = _sorted_pending(run for run in config.runs if run.enabled)
    simulated: list[ActiveRun] = []
    loads = make_gpu_loads(selected, simulated)
    launched = 0
    while pending and (config.max_total_runs is None or launched < config.max_total_runs):
        placed = False
        for run in list(pending):
            gpu = choose_gpu(run, loads, config)
            if gpu is None:
                continue
            # A small stand-in carries only attributes consumed by make_gpu_loads.
            fake_process = type("DryRunProcess", (), {"pid": -(launched + 1)})()
            simulated.append(
                ActiveRun(
                    spec=run,
                    process=fake_process,
                    gpu=gpu,
                    command=[],
                    log_path=Path(),
                    log_file=None,
                    started_at="",
                )
            )
            loads = make_gpu_loads(selected, simulated)
            print(
                f"DRY RUN  {run.name} -> "
                f"{_placement_description(config, gpu)} ({gpu.uuid}); "
                f"command: {_format_command(build_run_command(run, config))}"
            )
            pending.remove(run)
            launched += 1
            placed = True
            break
        if not placed:
            break
    for run in pending:
        print(f"QUEUED   {run.name} (waits for a slot or memory from the first wave)")
    return 0


def run_scheduler(config: SchedulerConfig, *, dry_run: bool = False) -> int:
    """Run the scheduling loop and return a process-style exit code."""

    _validate_device_pools(config)
    enabled_runs = [run for run in config.runs if run.enabled]
    selected: list[GpuSnapshot] = []
    gpu_placement = _uses_gpu_placement(config)
    if not gpu_placement:
        _validate_cpu_runs(enabled_runs, config)
        max_active_runs = _max_active_runs(config)
        print(f"Selected CPU execution; max {max_active_runs} simultaneous run(s).")
        if dry_run:
            return dry_run_cpu_schedule(config)
    else:
        if config.device == "cpu":
            _validate_cpu_runs(enabled_runs, config)
        all_snapshots = query_gpu_snapshots()
        selected = select_gpus(
            all_snapshots,
            _placement_selectors(config),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        _validate_static_capacity(enabled_runs, selected, config)
        _print_gpu_summary(selected, config)
        max_active_runs = _max_active_runs(config)
        if dry_run:
            return dry_run_schedule(config, selected)

    session_dir = _session_directory(config)
    state_path = session_dir / "state.json"
    statuses = _initial_statuses(config.runs)
    _write_state(state_path, config, statuses)
    print(f"Scheduler logs: {session_dir}")

    pending = _sorted_pending(enabled_runs)
    active: list[ActiveRun] = []
    selected_uuids = {gpu.uuid for gpu in selected}
    saw_failure = False
    last_wait_message = 0.0

    try:
        while pending or active:
            for running in list(active):
                returncode = running.process.poll()
                if returncode is None:
                    continue
                running.log_file.close()
                active.remove(running)
                succeeded = returncode == 0
                saw_failure = saw_failure or not succeeded
                statuses[running.spec.name].update(
                    {
                        "status": "succeeded" if succeeded else "failed",
                        "returncode": returncode,
                        "finished_at": _utc_now(),
                    }
                )
                print(
                    f"{'DONE' if succeeded else 'FAILED'}  {running.spec.name} "
                    f"on {_placement_description(config, running.gpu)} "
                    f"(exit {returncode}); log={running.log_path}"
                )
                _write_state(state_path, config, statuses)

            if saw_failure and config.fail_fast and pending:
                for run in pending:
                    statuses[run.name].update(
                        {
                            "status": "skipped",
                            "reason": "fail_fast: an earlier run failed",
                            "finished_at": _utc_now(),
                        }
                    )
                pending.clear()
                _write_state(state_path, config, statuses)

            launched_any = False
            if pending and (max_active_runs is None or len(active) < max_active_runs):
                if not gpu_placement:
                    while pending and len(active) < max_active_runs:
                        run = pending.pop(0)
                        running = launch_run(run, None, config, session_dir)
                        active.append(running)
                        statuses[run.name].update(
                            {
                                "status": "running",
                                "pid": running.process.pid,
                                "device": "cpu",
                                "ssl_device": "cpu",
                                "started_at": running.started_at,
                                "log": str(running.log_path),
                                "command": running.command,
                            }
                        )
                        print(
                            f"START    {run.name} -> CPU, pid={running.process.pid}, "
                            f"log={running.log_path}"
                        )
                        _write_state(state_path, config, statuses)
                        launched_any = True
                        if config.launch_delay_seconds:
                            time.sleep(config.launch_delay_seconds)
                else:
                    selected = _refresh_selected_snapshots(selected_uuids)
                    loads = make_gpu_loads(selected, active)
                    while pending and (
                        max_active_runs is None or len(active) < max_active_runs
                    ):
                        placement = None
                        for run in pending:
                            gpu = choose_gpu(run, loads, config)
                            if gpu is not None:
                                placement = (run, gpu)
                                break
                        if placement is None:
                            break
                        run, gpu = placement
                        running = launch_run(run, gpu, config, session_dir)
                        active.append(running)
                        pending.remove(run)
                        statuses[run.name].update(
                            {
                                "status": "running",
                                "pid": running.process.pid,
                                "device": config.device,
                                "ssl_device": _child_ssl_device(config),
                                "started_at": running.started_at,
                                "log": str(running.log_path),
                                "command": running.command,
                            }
                        )
                        if config.device == "cuda" or _uses_batch_gpu(config):
                            statuses[run.name].update(
                                {
                                    "physical_gpu": gpu.index,
                                    "gpu_uuid": gpu.uuid,
                                }
                            )
                            if _uses_batch_gpu(config):
                                statuses[run.name].update(
                                    {
                                        "gpu_batch_size_threshold": GPU_BATCH_SIZE_THRESHOLD,
                                        "device_above_gpu_batch_size_threshold": "cuda",
                                    }
                                )
                                if config.gpu_batch_size_thresholds:
                                    statuses[run.name]["gpu_batch_size_thresholds"] = [
                                        rule.to_spec()
                                        for rule in config.gpu_batch_size_thresholds
                                    ]
                        if _child_ssl_device(config) == "cuda":
                            statuses[run.name].update(
                                {
                                    "physical_ssl_gpu": gpu.index,
                                    "ssl_gpu_uuid": gpu.uuid,
                                }
                            )
                        print(
                            f"START    {run.name} -> {_placement_description(config, gpu)}, "
                            f"pid={running.process.pid}, "
                            f"log={running.log_path}"
                        )
                        _write_state(state_path, config, statuses)
                        launched_any = True
                        loads = make_gpu_loads(selected, active)
                        if config.launch_delay_seconds:
                            time.sleep(config.launch_delay_seconds)

            if not pending and not active:
                break
            if pending and not active and not launched_any:
                now = time.monotonic()
                if now - last_wait_message >= 30:
                    if not gpu_placement:
                        print(f"WAIT     {len(pending)} run(s) queued for a CPU run slot.")
                    else:
                        print(
                            f"WAIT     {len(pending)} run(s) queued; selected GPUs are occupied "
                            "or lack the requested free-memory reservation."
                        )
                    last_wait_message = now
            time.sleep(config.poll_interval_seconds)
    except KeyboardInterrupt:
        print("INTERRUPT terminating scheduler-owned child processes...", file=sys.stderr)
        for running in active:
            statuses[running.spec.name].update(
                {"status": "interrupted", "finished_at": _utc_now()}
            )
        for run in pending:
            statuses[run.name].update(
                {"status": "canceled", "finished_at": _utc_now()}
            )
        _terminate_active(active)
        _write_state(state_path, config, statuses)
        return 130
    except BaseException:
        # Do not orphan long-running studies if NVIDIA polling, state writing,
        # or a later child launch fails after earlier children have started.
        print("ERROR     terminating scheduler-owned child processes...", file=sys.stderr)
        for running in active:
            statuses[running.spec.name].update(
                {
                    "status": "aborted",
                    "reason": "scheduler error",
                    "finished_at": _utc_now(),
                }
            )
        for run in pending:
            statuses[run.name].update(
                {
                    "status": "canceled",
                    "reason": "scheduler error",
                    "finished_at": _utc_now(),
                }
            )
        _terminate_active(active)
        try:
            _write_state(state_path, config, statuses)
        except OSError:
            pass
        raise

    succeeded = sum(status["status"] == "succeeded" for status in statuses.values())
    failed = sum(status["status"] == "failed" for status in statuses.values())
    skipped = sum(status["status"] == "skipped" for status in statuses.values())
    print(
        f"SUMMARY  succeeded={succeeded}, failed={failed}, skipped={skipped}; "
        f"state={state_path}"
    )
    return 1 if failed else 0


def _gpu_batch_size_threshold_rule_arg(
    value: str,
) -> device_thresholds.GpuBatchSizeThresholdRule:
    try:
        return device_thresholds.parse_gpu_batch_size_threshold_rule(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schedule independent metric-learning runs on CPU or across GPUs."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="run-manifest JSON path")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="execution device; overrides the manifest (default: cuda)",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help=(
            "physical GPU indexes or UUIDs for CUDA training/SSL, or for CPU runs "
            "whose SSL always uses CUDA and whose training uses CUDA when effective "
            f"batch size exceeds {GPU_BATCH_SIZE_THRESHOLD}; overrides the "
            "manifest and respects inherited CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument(
        "--ssl-gpus",
        "--ssl_gpus",
        nargs="+",
        help=(
            "physical GPU indexes or UUIDs used for SSL when --device cpu; "
            "overrides ssl_gpus in the manifest"
        ),
    )
    parser.add_argument(
        "--gpu-batch-size-thresholds",
        "--gpu-batch-size-threshold",
        "--gpu_batch_size_thresholds",
        "--gpu_batch_size_threshold",
        nargs="+",
        type=_gpu_batch_size_threshold_rule_arg,
        metavar="LOSS:SSL_METHOD=N",
        help=(
            "loss/SSL-specific workload cutoffs for CPU runs with --gpus; "
            "training uses CUDA when effective batch size is greater than the "
            "selected cutoff. LRML combines supervised and graph-edge batch "
            "sizes; for example NTXentLoss:all=0 ArcFace:lrml=128"
        ),
    )
    parser.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        help="override the manifest's per-GPU run-slot capacity",
    )
    parser.add_argument(
        "--max-total-runs",
        type=int,
        help="optional host-wide cap on simultaneously running children",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        help="override the process/GPU polling interval",
    )
    parser.add_argument(
        "--scheduler-log-dir",
        type=Path,
        help="override the directory for launcher stdout logs and state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the immediately launchable placement wave without starting children",
    )
    return parser


def apply_cli_overrides(config: SchedulerConfig, args: argparse.Namespace) -> SchedulerConfig:
    updates: dict[str, Any] = {}
    if getattr(args, "device", None) is not None:
        updates["device"] = args.device
    if args.gpus is not None:
        updates["gpu_selectors"] = tuple(args.gpus)
    if args.ssl_gpus is not None:
        updates["ssl_gpu_selectors"] = tuple(args.ssl_gpus)
    if args.gpu_batch_size_thresholds is not None:
        updates["gpu_batch_size_thresholds"] = tuple(
            args.gpu_batch_size_thresholds
        )
    if args.max_concurrent_per_gpu is not None:
        updates["max_concurrent_per_gpu"] = _positive_int(
            args.max_concurrent_per_gpu,
            "--max-concurrent-per-gpu",
        )
    if args.max_total_runs is not None:
        updates["max_total_runs"] = _positive_int(
            args.max_total_runs,
            "--max-total-runs",
        )
    if args.poll_interval_seconds is not None:
        updates["poll_interval_seconds"] = _positive_float(
            args.poll_interval_seconds,
            "--poll-interval-seconds",
        )
    if args.scheduler_log_dir is not None:
        log_dir = args.scheduler_log_dir
        if not log_dir.is_absolute():
            log_dir = config.working_dir / log_dir
        updates["scheduler_log_dir"] = log_dir.resolve()
    resolved = replace(config, **updates)
    _validate_device_pools(resolved)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config = apply_cli_overrides(load_scheduler_config(args.manifest), args)
        return run_scheduler(config, dry_run=args.dry_run)
    except (ManifestError, NvidiaSmiError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
