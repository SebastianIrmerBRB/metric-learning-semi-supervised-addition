"""Optuna configuration, study execution, and search-space constraints."""

import copy
import csv
import gc
import itertools
import json
import math
import multiprocessing as mp
import pickle
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytorch_metric_learning.miners as miners
import torch
from loguru import logger

import utils
from models.retrieval_model import BACKBONE_TUNING_FROZEN

from . import ablation, semi_supervised
from .cli import (
    FINAL_TEST_VISUALIZATION_NONE,
    get_hparam_seed,
    get_support_seed,
    normalize_backbone_tuning_args,
    resolve_scheduler_batch_device,
)
from .io import is_scalar, namespace_to_dict, result_to_dict, to_jsonable, write_json
from .engine import (
    get_data_split_seed,
    get_frozen_feature_cache_max_bytes,
    get_image_resize_mode,
    get_loss_class,
    resolve_loss_driven_supervised_args,
    resolve_mode_ssl_config,
    run_experiment,
    uses_ssl_warmup_objective,
    validate_named_miner_params,
    validate_run_args,
)
from .frozen_feature_cache import FrozenFeatureDatasetCache
from .types import (
    ALL_LOSSES,
    ALL_MINERS,
    BATCH_SAMPLER_HPARAM_KEY,
    CLASSIFICATION_LOSSES,
    HPARAM_CONFIG_EXTENDS_KEY,
    HPARAM_CONFIG_PER_LOSS_KEY,
    HPO_MODE_KEYS,
    HParamSearchConfig,
    HParamStudyResult,
    JOINT_COMPONENT_HPARAM_PREFIX,
    JOINT_HPARAM_PREFIX,
    JOINT_STML_IN_BATCH_HPARAM_KEY,
    JOINT_TWO_STREAM_HPARAM_KEY,
    LABELED_BATCH_FRACTION_HPARAM_ALIASES,
    LABELED_BATCH_FRACTION_HPARAM_KEY,
    LABELED_BATCH_SIZE_HPARAM_KEY,
    LABELED_BATCH_SIZE_HPARAM_KEYS,
    MAX_LABELED_BATCH_FRACTION,
    LOSS_HPARAM_PREFIX,
    STML_IN_BATCH_HPARAM_KEYS,
    STML_IN_BATCH_ONLY_HPARAM_KEYS,
    STML_MERGED_INERT_HPARAM_KEYS,
    MERGED_HPARAM_CONFIG_KEYS,
    MINER_HPARAM_PREFIX,
    OBJECTIVE_METRICS,
    SAMPLER_CAPACITY_HPARAM_KEYS,
    SELECTION_METRICS,
)


SAMPLER_STATE_FILENAME = "sampler.pkl"
TPE_TRANSITION_SAMPLER_STATE_FILENAME = "sampler_before_tpe.pkl"
IN_FLIGHT_SAMPLER_STATE_FILENAME = "sampler_in_flight.pkl"
_SAMPLER_STATE_LOCK = threading.Lock()

# ArcFace and ProxyAnchor are the two classification losses whose learned
# class weights benefit from searching the model/classifier learning rates as
# one overall scale plus their relative scale.  These names are HPO-only: the
# training engine continues to receive the ordinary ``lr`` and
# ``classifier_lr`` arguments.
BASE_LR_HPARAM_KEY = "base_lr"
LR_RATIO_HPARAM_KEY = "lr_ratio"
LR_RATIO_LOSSES = frozenset({"ArcFaceLoss", "ProxyAnchorLoss"})
LR_RATIO_MIN_OPTIMIZER_LR = 1e-5
LR_RATIO_MAX_MODEL_LR = 0.3
LR_RATIO_MAX_CLASSIFIER_LR = 0.5


class LrRatioOutOfBoundsError(ValueError):
    """A sampled LR-ridge point reconstructs an unsafe optimizer LR."""


class ParallelOptunaSampler:
    """Serialize access to one resumable sampler shared by Optuna worker threads.

    ``Study.optimize(n_jobs>1)`` calls ``reseed_rng`` once in every worker. That
    behavior is useful when workers own independent sampler copies, but this
    project deliberately keeps one sampler state and checkpoints it to disk.
    Sampling is therefore serialized through one shared sampler and reseeding
    is suppressed so loading ``sampler.pkl`` actually continues its RNG state.
    """

    def __init__(self, sampler):
        self._sampler = sampler
        self._state_lock = threading.RLock()

    @property
    def wrapped_sampler(self):
        return self._sampler

    @property
    def state_lock(self):
        return self._state_lock

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def __str__(self):
        return str(self._sampler)

    def before_trial(self, study, trial):
        with self._state_lock:
            return self._sampler.before_trial(study, trial)

    def infer_relative_search_space(self, study, trial):
        with self._state_lock:
            return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        with self._state_lock:
            return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        with self._state_lock:
            return self._sampler.sample_independent(
                study,
                trial,
                param_name,
                param_distribution,
            )

    def after_trial(self, study, trial, state, values):
        with self._state_lock:
            return self._sampler.after_trial(study, trial, state, values)

    def reseed_rng(self):
        # All Optuna jobs in this process share the wrapped sampler. Its calls
        # are serialized, so independent thread-local reseeding is unnecessary
        # and would discard a state restored from sampler.pkl.
        return None


def terminate_active_children(timeout=5.0):
    """Terminate DataLoader workers left alive after a failed single-job trial."""

    children = mp.active_children()
    for child in children:
        if child.is_alive():
            try:
                child.terminate()
            except OSError:
                pass
    for child in children:
        try:
            child.join(timeout=timeout)
        except OSError:
            pass
    for child in children:
        if child.is_alive():
            try:
                child.kill()
            except OSError:
                pass
            try:
                child.join(timeout=1.0)
            except OSError:
                pass


def cleanup_after_trial(terminate_children=False):
    """Release Python-owned handles after a trial."""

    gc.collect()
    if terminate_children:
        terminate_active_children()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def load_hparam_config(config_path):
    """Load and validate an Optuna JSON config, or return ``None``."""

    if config_path is None:
        return None

    path = Path(config_path)
    raw_config = resolve_hparam_config_inheritance(path)

    allowed_keys = set(HParamSearchConfig.__dataclass_fields__)
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown hyperparameter config keys in {path}: {unknown_keys}")

    config = HParamSearchConfig(**raw_config)
    validate_hparam_config(config, path)
    validate_per_loss_hparam_config(config, path)
    return config

def read_hparam_config_json(path):
    with path.open() as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Hyperparameter config must be a JSON object: {path}")
    return raw_config

def resolve_hparam_config_inheritance(path, _seen=None):
    """Merge a config onto the parent(s) named by its ``extends`` key."""

    path = Path(path).resolve()
    if _seen is None:
        _seen = []
    if path in _seen:
        chain = " -> ".join(str(item) for item in [*_seen, path])
        raise ValueError(f"Circular hyperparameter config inheritance: {chain}")

    raw_config = read_hparam_config_json(path)
    parent_reference = raw_config.pop(HPARAM_CONFIG_EXTENDS_KEY, None)
    if parent_reference is None:
        return raw_config

    inherited = {}
    for parent_path in resolve_parent_config_paths(parent_reference, path):
        parent_config = resolve_hparam_config_inheritance(parent_path, [*_seen, path])
        inherited = merge_hparam_config_dicts(inherited, parent_config)
    return merge_hparam_config_dicts(inherited, raw_config)

def resolve_parent_config_paths(parent_reference, path):
    """Resolve one config's ``extends`` value into parent paths, in merge order.

    A list lets a layer inherit from several parents at once -- a dataset's
    ranges plus a method's shared search space -- so neither has to restate the
    other. Parents merge left to right, so a later parent outranks an earlier
    one and the child outranks them all.
    """

    references = parent_reference if isinstance(parent_reference, list) else [parent_reference]
    if not references:
        raise ValueError(
            f"{HPARAM_CONFIG_EXTENDS_KEY!r} must name at least one parent config in {path}"
        )

    parent_paths = []
    for reference in references:
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                f"{HPARAM_CONFIG_EXTENDS_KEY!r} must be a non-empty path string, "
                f"or a list of them, in {path}"
            )
        # Parent paths are written relative to the child so a config tree can be
        # moved or copied as a unit.
        parent_path = Path(reference)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        if not parent_path.is_file():
            raise FileNotFoundError(
                f"Parent hyperparameter config not found: {parent_path} "
                f"(referenced by {HPARAM_CONFIG_EXTENDS_KEY!r} in {path})"
            )
        parent_paths.append(parent_path)
    return parent_paths

def merge_hparam_config_dicts(base, override):
    """Merge one config layer onto another.

    ``spaces`` and ``per_loss`` merge entry by entry so a layer can retune a
    single search space without restating the rest, and an individual space spec
    merges field by field so ``{"low": 1e-5}`` narrows a range in place. Every
    other key, including ``sampler_params`` and ``pruner_params``, is replaced
    outright so a layer can also reset one back to an empty object.
    """

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in MERGED_HPARAM_CONFIG_KEYS and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_hparam_config_entries(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def merge_hparam_config_entries(base, override):
    """Merge a ``spaces``/``per_loss`` object entry by entry."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_hparam_config_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def validate_per_loss_hparam_config(config, path=None):
    """Validate each per-loss block as the config it will become when applied."""

    source = f" in {path}" if path is not None else ""
    if not isinstance(config.per_loss, dict):
        raise ValueError(f"per_loss must be an object{source}")
    for loss_name, overrides in config.per_loss.items():
        if loss_name not in ALL_LOSSES:
            raise ValueError(f"Unknown loss class in per_loss: {loss_name!r}{source}")
        if not isinstance(overrides, dict):
            raise ValueError(f"per_loss entry {loss_name!r} must be an object{source}")
        unknown_keys = sorted(set(overrides) - set(HParamSearchConfig.__dataclass_fields__))
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in per_loss entry {loss_name!r}: {unknown_keys}{source}"
            )
        if HPARAM_CONFIG_PER_LOSS_KEY in overrides:
            raise ValueError(f"per_loss entries cannot nest per_loss: {loss_name!r}{source}")
        validate_hparam_config(apply_per_loss_overrides(config, loss_name), path)

def apply_per_loss_overrides(config, loss_name):
    """Merge one loss class's per-loss block and drop the per-loss mapping."""

    overrides = config.per_loss.get(loss_name)
    fields = config.to_dict()
    fields.pop(HPARAM_CONFIG_PER_LOSS_KEY, None)
    if overrides:
        fields = merge_hparam_config_dicts(fields, overrides)
        fields.pop(HPARAM_CONFIG_PER_LOSS_KEY, None)
    return HParamSearchConfig(**fields)

def apply_per_loss_hparam_config(args, config):
    """Specialize a shared dataset config for the loss this run has fixed."""

    if not config.per_loss:
        return config

    loss_name = getattr(args, "loss", None)
    specialized = apply_per_loss_overrides(config, loss_name)
    if loss_name in config.per_loss:
        logger.info(f"Applied per_loss hyperparameter overrides for loss {loss_name!r}.")
    return specialized

def validate_hparam_config(config, path=None):
    source = f" in {path}" if path is not None else ""
    if config.n_trials <= 0:
        raise ValueError(f"n_trials must be positive{source}")
    if not config.enabled:
        return
    if config.timeout is not None and config.timeout <= 0:
        raise ValueError(f"timeout must be positive when set{source}")
    if (
        isinstance(config.n_jobs, bool)
        or not isinstance(config.n_jobs, int)
        or config.n_jobs == 0
        or config.n_jobs < -1
    ):
        raise ValueError(f"n_jobs must be a positive integer or -1{source}")
    if config.direction not in {"maximize", "minimize"}:
        raise ValueError(f"direction must be 'maximize' or 'minimize'{source}")
    if config.metric not in OBJECTIVE_METRICS:
        raise ValueError(f"metric must be one of {sorted(OBJECTIVE_METRICS)}{source}")
    if config.sampler not in {"tpe", "random", "grid"}:
        raise ValueError(f"sampler must be one of ['tpe', 'random', 'grid']{source}")
    if config.tpe_startup_trials is not None:
        validate_tpe_startup_trials(config.tpe_startup_trials, source)
        if config.sampler != "tpe":
            raise ValueError(f"tpe_startup_trials only applies when sampler is 'tpe'{source}")
    if config.pruner not in {"none", "median", "successive_halving", "hyperband"}:
        raise ValueError(f"pruner must be one of ['none', 'median', 'successive_halving', 'hyperband']{source}")
    if not isinstance(config.sampler_params, dict):
        raise ValueError(f"sampler_params must be an object{source}")
    if not isinstance(config.pruner_params, dict):
        raise ValueError(f"pruner_params must be an object{source}")
    if not isinstance(config.retry_failed_trials, bool):
        raise ValueError(f"retry_failed_trials must be true or false{source}")
    if not isinstance(config.spaces, dict) or not config.spaces:
        raise ValueError(f"spaces must be a non-empty object{source}")
    if BATCH_SAMPLER_HPARAM_KEY in config.spaces and {"batch_size", "sampler_m"} & set(config.spaces):
        raise ValueError(
            f"Search space {BATCH_SAMPLER_HPARAM_KEY!r} sets both batch_size and sampler_m. "
            f"Do not also include 'batch_size' or 'sampler_m'{source}."
        )
    labeled_batch_size_keys = sorted(set(config.spaces) & LABELED_BATCH_SIZE_HPARAM_KEYS)
    if len(labeled_batch_size_keys) > 1:
        raise ValueError(
            "Use only one labeled-batch-size HPO key; "
            f"these keys are aliases for the same SSL setting: {labeled_batch_size_keys}{source}"
        )
    prefixed_fraction_keys = sorted(set(config.spaces) & LABELED_BATCH_FRACTION_HPARAM_ALIASES)
    if prefixed_fraction_keys:
        raise ValueError(
            f"{prefixed_fraction_keys} are not SSL config fields; the labeled batch share is "
            f"searched with the bare {LABELED_BATCH_FRACTION_HPARAM_KEY!r} key, which derives "
            f"{LABELED_BATCH_SIZE_HPARAM_KEY!r} from the sampled batch size{source}"
        )
    if LABELED_BATCH_FRACTION_HPARAM_KEY in config.spaces and labeled_batch_size_keys:
        raise ValueError(
            f"Search either {LABELED_BATCH_FRACTION_HPARAM_KEY!r} or {labeled_batch_size_keys}, "
            "not both; the fraction already determines the labeled batch size"
            f"{source}"
        )
    for name, spec in config.spaces.items():
        if name in {"loss", "miner", *HPO_MODE_KEYS}:
            instruction = (
                f"Set it with --{name.replace('_', '-')} or in the experiment config."
                if name in HPO_MODE_KEYS
                else f"Set it with --{name} or compare fixed pairs with --loss_miner_grid."
            )
            raise ValueError(
                f"{name!r} is not a valid HPO space key{source}. "
                f"{instruction}"
            )
        validate_component_override_name(name, source)
        validate_space_spec(name, spec, source)

def validate_tpe_startup_trials(value, source=""):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"tpe_startup_trials must be a non-negative integer{source}")

def validate_space_spec(name, spec, source=""):
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"Search space {name!r} choices must not be empty{source}")
        if name == BATCH_SAMPLER_HPARAM_KEY:
            validate_batch_sampler_choices(spec, source)
        if name in LABELED_BATCH_SIZE_HPARAM_KEYS:
            validate_labeled_batch_size_choices(spec, source)
        if name == LABELED_BATCH_FRACTION_HPARAM_KEY:
            validate_labeled_batch_fraction_choices(spec, source)
        if name == "selection_metric":
            validate_selection_metric_choices(spec, source)
        return
    if not isinstance(spec, dict):
        raise ValueError(f"Search space {name!r} must be an object or a list of categorical choices{source}")

    space_type = spec.get("type", "categorical" if "choices" in spec else None)
    if space_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical search space {name!r} requires a non-empty choices list{source}")
        if name == BATCH_SAMPLER_HPARAM_KEY:
            validate_batch_sampler_choices(choices, source)
        if name in LABELED_BATCH_SIZE_HPARAM_KEYS:
            validate_labeled_batch_size_choices(choices, source)
        if name == LABELED_BATCH_FRACTION_HPARAM_KEY:
            validate_labeled_batch_fraction_choices(choices, source)
        if name == "selection_metric":
            validate_selection_metric_choices(choices, source)
    elif space_type in {"float", "int"}:
        if name == BATCH_SAMPLER_HPARAM_KEY:
            raise ValueError(f"Search space {name!r} must be categorical choices like '32:8'{source}")
        if name in LABELED_BATCH_SIZE_HPARAM_KEYS:
            raise ValueError(
                f"Search space {name!r} must use categorical positive-integer choices so it can be "
                f"paired safely with {BATCH_SAMPLER_HPARAM_KEY!r}{source}"
            )
        if name == LABELED_BATCH_FRACTION_HPARAM_KEY:
            if space_type != "float":
                raise ValueError(
                    f"Search space {name!r} must be a float range or categorical fractions{source}"
                )
            validate_labeled_batch_fraction_choices([spec.get("low"), spec.get("high")], source)
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"{space_type} search space {name!r} requires low and high{source}")
        if spec["low"] > spec["high"]:
            raise ValueError(f"{space_type} search space {name!r} low must be <= high{source}")
        if space_type == "int" and (not isinstance(spec["low"], int) or not isinstance(spec["high"], int)):
            raise ValueError(f"int search space {name!r} low/high must be integers{source}")
        if spec.get("step") is not None and spec["step"] <= 0:
            raise ValueError(f"{space_type} search space {name!r} step must be positive{source}")
    else:
        raise ValueError(f"Unknown search space type for {name!r}: {space_type!r}{source}")

def validate_component_override_name(name, source=""):
    """Validate optional loss./miner. constructor-parameter search keys."""

    if not is_component_override(name):
        return
    parts = name.split(".")
    if len(parts) not in {2, 3} or not parts[-1]:
        raise ValueError(
            f"Search space {name!r} must use '<component>.<parameter>' or "
            f"'<component>.<ClassName>.<parameter>'{source}"
        )
    if len(parts) == 3:
        component, class_name, _ = parts
        valid_classes = ALL_LOSSES if component == "loss" else ALL_MINERS
        if class_name not in valid_classes:
            raise ValueError(f"Unknown {component} class in search space {name!r}: {class_name!r}{source}")
        if component == "miner" and class_name == "no_miner":
            raise ValueError(f"Search space {name!r} cannot target no_miner{source}")

def validate_batch_sampler_choices(choices, source=""):
    for choice in choices:
        parse_batch_sampler_choice(choice, source)

def validate_labeled_batch_size_choices(choices, source=""):
    for choice in choices:
        parse_labeled_batch_size_choice(choice, source)

def validate_selection_metric_choices(choices, source=""):
    for choice in choices:
        if choice not in SELECTION_METRICS:
            raise ValueError(f"selection_metric choice must be one of {SELECTION_METRICS}, got {choice!r}{source}")

def parse_batch_sampler_choice(choice, source=""):
    if not isinstance(choice, str) or ":" not in choice:
        raise ValueError(
            f"{BATCH_SAMPLER_HPARAM_KEY!r} choices must be strings formatted as 'batch_size:sampler_m', "
            f"got {choice!r}{source}"
        )

    raw_batch_size, raw_sampler_m = choice.split(":", 1)
    try:
        batch_size = int(raw_batch_size)
        sampler_m = int(raw_sampler_m)
    except ValueError as exc:
        raise ValueError(
            f"{BATCH_SAMPLER_HPARAM_KEY!r} choice must contain integer values, got {choice!r}{source}"
        ) from exc

    if batch_size <= 0 or sampler_m <= 0:
        raise ValueError(
            f"{BATCH_SAMPLER_HPARAM_KEY!r} choice must use positive integers, got {choice!r}{source}"
        )
    if batch_size % sampler_m != 0:
        raise ValueError(
            f"{BATCH_SAMPLER_HPARAM_KEY!r} choice must satisfy batch_size % sampler_m == 0, "
            f"got {choice!r}{source}"
        )
    return batch_size, sampler_m

def parse_labeled_batch_size_choice(choice, source=""):
    if isinstance(choice, bool) or not isinstance(choice, (str, int, np.integer)):
        raise ValueError(
            f"labeled_batch_size choices must be positive integers or integer strings, "
            f"got {choice!r}{source}"
        )
    try:
        labeled_batch_size = int(choice)
    except ValueError as exc:
        raise ValueError(
            f"labeled_batch_size choices must be positive integers or integer strings, "
            f"got {choice!r}{source}"
        ) from exc
    if labeled_batch_size <= 0:
        raise ValueError(
            f"labeled_batch_size choices must be positive, got {choice!r}{source}"
        )
    return labeled_batch_size

def validate_labeled_batch_fraction_choices(choices, source=""):
    for choice in choices:
        parse_labeled_batch_fraction_choice(choice, source)

def parse_labeled_batch_fraction_choice(choice, source=""):
    if isinstance(choice, bool) or not isinstance(choice, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} choices must be numbers in "
            f"(0, {MAX_LABELED_BATCH_FRACTION}], got {choice!r}{source}"
        )
    fraction = float(choice)
    if not 0 < fraction <= MAX_LABELED_BATCH_FRACTION:
        raise ValueError(
            f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} choices must be in "
            f"(0, {MAX_LABELED_BATCH_FRACTION}], got {choice!r}{source}"
        )
    return fraction

def get_labeled_batch_fraction_values(spec):
    """Return the fractions a space can draw, or the endpoints of its range."""

    if is_categorical_space(spec):
        return [parse_labeled_batch_fraction_choice(choice) for choice in get_categorical_choices(spec)]
    return [parse_labeled_batch_fraction_choice(spec["low"]), parse_labeled_batch_fraction_choice(spec["high"])]

def get_max_labeled_batch_size(batch_size, sampler_m):
    """Return the largest labeled stream this sampler choice can carry.

    ``labeled_batch_size <= unlabeled_batch_size`` caps the labeled stream at
    half the batch, and both streams have to be whole ``sampler_m`` groups.
    """

    return ((int(batch_size) // int(sampler_m)) // 2) * int(sampler_m)

def derive_labeled_batch_size_from_fraction(batch_size, sampler_m, fraction, source=""):
    """Snap a labeled batch share onto the sampler grid of this batch size.

    The fraction is the searched hyperparameter because it means the same thing
    at every ``batch_sampler`` choice; the absolute labeled size it lands on is
    whatever the sampler groups and the half-batch cap allow.
    """

    fraction = parse_labeled_batch_fraction_choice(fraction, source)
    batch_size = int(batch_size)
    sampler_m = int(sampler_m)
    max_labeled_batch_size = get_max_labeled_batch_size(batch_size, sampler_m)
    if max_labeled_batch_size < sampler_m:
        raise ValueError(
            f"batch_size={batch_size} cannot host both streams of an "
            f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} split at sampler_m={sampler_m}; "
            f"batch_size must be at least {2 * sampler_m}{source}"
        )
    # Round to the nearest whole group, then keep at least one group and stay
    # inside the half-batch cap that fractions above 0.5 would otherwise exceed.
    groups = math.floor(fraction * batch_size / sampler_m + 0.5)
    labeled_batch_size = min(max(groups, 1) * sampler_m, max_labeled_batch_size)
    return labeled_batch_size

def run_hparam_search(args, config):
    """Create or resume an Optuna study and execute its remaining trials."""

    args = resolve_loss_driven_supervised_args(args)
    config = apply_per_loss_hparam_config(args, config)
    config = filter_component_hparam_config(args, config)
    config = filter_loss_dependent_hparam_config(args, config)
    config = filter_slade_contrastive_spaces(args, config)
    config = filter_lrml_contrastive_spaces(args, config)
    config = make_backbone_tuning_spaces_aware(args, config)
    config = make_sampler_spaces_k_shot_aware(args, config)
    config = make_sampler_spaces_label_budget_aware(args, config)
    # TEMPORARILY DISABLED FOR BASELINE
    config = make_component_spaces_constraint_aware(args, config)
    validate_hparam_config(config)
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna hyperparameter search requires the optuna package. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    if getattr(args, "skip_test_during_hpo", False) and config.metric.startswith("test_"):
        raise ValueError("Cannot use a test metric as Optuna objective when --skip_test_during_hpo is set")

    # study_dir contains human-readable artifacts; storage is the durable
    # Optuna backend used to resume trials and sampler state.
    study_name = config.study_name or "optuna"
    study_dir, relative_study_dir = make_study_dir(args.save_dir, study_name, config.study_dir)
    storage = resolve_optuna_storage(config.storage, study_dir)
    sampler_state_path = study_dir / SAMPLER_STATE_FILENAME
    tpe_transition_sampler_state_path = study_dir / TPE_TRANSITION_SAMPLER_STATE_FILENAME
    in_flight_sampler_state_path = study_dir / IN_FLIGHT_SAMPLER_STATE_FILENAME
    resolved_tpe_startup_trials = resolve_tpe_startup_trials(args, config)

    # The sampler proposes values; the pruner can stop weak trials based on
    # intermediate reports from epochs or CV folds.
    sampler, sampler_state_loaded = load_or_make_optuna_sampler(
        optuna,
        config,
        get_hparam_seed(args),
        sampler_state_path,
        tpe_startup_trials=resolved_tpe_startup_trials,
    )
    configured_tpe_startup_trials = (
        resolved_tpe_startup_trials
        if resolved_tpe_startup_trials is not None
        else config.sampler_params.get("n_startup_trials")
    )
    effective_tpe_startup_trials = (
        get_effective_tpe_startup_trials(
            sampler,
            fallback=(
                configured_tpe_startup_trials
                if configured_tpe_startup_trials is not None
                else 10
            ),
        )
        if config.sampler == "tpe"
        else None
    )
    if config.sampler == "tpe" and configured_tpe_startup_trials is None:
        configured_tpe_startup_trials = effective_tpe_startup_trials
    if (
        sampler_state_loaded
        and config.sampler == "tpe"
        and effective_tpe_startup_trials != configured_tpe_startup_trials
    ):
        raise ValueError(
            f"Optuna sampler state at {sampler_state_path} uses "
            f"n_startup_trials={effective_tpe_startup_trials}, but the current "
            f"configuration requests {configured_tpe_startup_trials}. Restore the "
            "matching sampler/configuration or migrate the study with "
            "scripts/remove_optuna_trials.py --extend-tpe-startup-to N."
        )
    write_json(
        study_dir / "study_config.json",
        {
            "base_args": namespace_to_dict(args),
            "hparam_config": config.to_dict(),
            "resolved_tpe_startup_trials": resolved_tpe_startup_trials,
            "effective_tpe_startup_trials": effective_tpe_startup_trials,
            "resolved_study_name": study_name,
            "resolved_storage": storage,
            "sampler_state_path": str(sampler_state_path),
            "in_flight_sampler_state_path": str(in_flight_sampler_state_path),
            "tpe_transition_sampler_state_path": (
                str(tpe_transition_sampler_state_path)
                if config.sampler == "tpe"
                else None
            ),
        },
    )
    sampler = make_parallel_optuna_sampler(sampler, config.n_jobs)
    pruner = make_optuna_pruner(optuna, config)
    study = optuna.create_study(
        direction=config.direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=config.load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    # Reusing a study with changed parameter distributions would mix
    # incomparable trials, so compare current spaces with stored distributions.
    validate_study_distributions_compatible(optuna, study, config, storage)
    recovered_trials = recover_unfinished_trials(optuna, study)
    replayed_trials = replay_random_sampler_state_if_needed(
        optuna,
        study,
        config,
        sampler_state_loaded,
    )
    # Runs after the history replay so the recovered trial's draws are consumed
    # in the same order the interrupted process consumed them.
    resynced_trials = resync_sampler_with_recovered_trials(
        optuna,
        study,
        config,
        recovered_trials,
        in_flight_sampler_state_path,
    )
    if replayed_trials or resynced_trials:
        save_optuna_sampler(study.sampler, sampler_state_path)
    last_completed_sampler = clone_optuna_sampler(study.sampler)
    initial_budget_trials = count_hpo_budget_trials(optuna, study)
    tpe_transition_checkpoint_enabled = config.sampler == "tpe"
    if (
        tpe_transition_checkpoint_enabled
        and not tpe_transition_sampler_state_path.exists()
        and initial_budget_trials > effective_tpe_startup_trials
    ):
        # The exact boundary state cannot be reconstructed retrospectively from
        # a study that already advanced into TPE before this artifact existed.
        logger.warning(
            f"Did not create {tpe_transition_sampler_state_path}: the study already has "
            f"{initial_budget_trials} completed/pruned trials, past the TPE startup boundary "
            f"of {effective_tpe_startup_trials}. The checkpoint will be created for new studies."
        )
        tpe_transition_checkpoint_enabled = False
    elif tpe_transition_checkpoint_enabled:
        maybe_save_tpe_transition_sampler(
            optuna,
            study,
            last_completed_sampler,
            tpe_transition_sampler_state_path,
            effective_tpe_startup_trials,
        )
    trials_csv = study_dir / "trials.csv"
    trials_jsonl = study_dir / "trials.jsonl"
    retry_failed_trials_mode = should_retry_failed_hpo_trials(args, config)
    frozen_feature_cache = None
    if (
        bool(getattr(args, "use_cache", False))
        and args.backbone_tuning == BACKBONE_TUNING_FROZEN
    ):
        cache_max_bytes = get_frozen_feature_cache_max_bytes(args)
        frozen_feature_cache = FrozenFeatureDatasetCache(max_bytes=cache_max_bytes)
        budget = (
            "unbounded"
            if cache_max_bytes is None
            else f"{cache_max_bytes / 1e9:.2f} GB, least-recently-used eviction"
        )
        logger.info(
            "Enabled one source-indexed mmap frozen-feature cache shared by all "
            f"trials and cross-validation folds in this HPO study ({budget})"
        )

    def objective(trial):
        # Resolve trial suggestions into a fresh argparse namespace and SSL
        # config so trials cannot mutate one another's settings.
        if tpe_transition_checkpoint_enabled:
            # If a parallel worker starts the first model-based trial before
            # the preceding callback runs, capture the boundary before any TPE
            # suggestion advances the sampler.
            maybe_save_tpe_transition_sampler(
                optuna,
                study,
                study.sampler,
                tpe_transition_sampler_state_path,
                effective_tpe_startup_trials,
            )
        try:
            trial_args, ssl_config, suggested_params = run_with_sampler_checkpoint(
                study.sampler,
                sampler_state_path,
                lambda: make_trial_args_and_ssl_config(args, config, trial),
                in_flight_path=in_flight_sampler_state_path,
                trial=trial,
            )
        except LrRatioOutOfBoundsError as exc:
            # The complete rectangular base/ratio space deliberately covers
            # both observed inverse-LR modes. Its extreme corners reconstruct
            # optimizer rates outside the safe range and are conditional
            # invalid points, so count them as pruned rather than failed.
            pruning_reason = str(exc)
            suggested_params = expand_joint_component_params(dict(trial.params))
            trial.set_user_attr("params", suggested_params)
            trial.set_user_attr("pruning_reason", pruning_reason)
            logger.info(f"Pruning HPO trial {trial.number}: {pruning_reason}")
            raise optuna.TrialPruned(pruning_reason) from exc
        trial_args.hparam_config_resolved = config.to_dict()
        trial_args.hparam_params = suggested_params
        trial_args.hparam_study_dir = study_dir
        trial_args.hparam_study_name = study.study_name
        trial_args.trial_number = trial.number
        # Comparisons suppress test evaluation during HPO. Standalone studies
        # may opt in, but validation metrics remain the usual objective.
        trial_args.evaluate_test = not bool(getattr(args, "skip_test_during_hpo", False))
        trial_args.final_test_visualization = FINAL_TEST_VISUALIZATION_NONE
        trial_args.save_dir = relative_study_dir / f"trial_{trial.number:04d}"

        # User attributes make the persistent trial self-describing without
        # requiring the separate run_config.json file.
        trial.set_user_attr("params", suggested_params)
        trial.set_user_attr("resolved_args", namespace_to_dict(trial_args))
        trial.set_user_attr("resolved_ssl_config", ssl_config.to_dict())

        try:
            result = run_experiment(
                trial_args,
                ssl_config,
                optuna_trial=trial,
                optuna_metric=config.metric,
                frozen_feature_cache=frozen_feature_cache,
            )
        except (utils.NonFiniteEmbeddingError, FloatingPointError) as exc:
            # Numerical divergence is an expected outcome for some sampled
            # hyperparameters. Reject only this trial instead of aborting the
            # entire study, while preserving the diagnostic in study outputs.
            #
            # FloatingPointError is the same category, raised earlier: a
            # regularizer whose loss saturates to zero, or which goes non-finite,
            # is a property of the sampled point, not of the code. Letting it
            # propagate aborts the process, and because a failed trial does not
            # advance the saved sampler, the next launch redraws the identical
            # point and dies in the same place -- an unbreakable crash loop.
            pruning_reason = str(exc)
            cleanup_after_trial(terminate_children=config.n_jobs == 1)
            trial.set_user_attr("pruning_reason", pruning_reason)
            logger.warning(
                f"Pruning HPO trial {trial.number} after numerical divergence: "
                f"{pruning_reason}"
            )
            raise optuna.TrialPruned(pruning_reason) from exc
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial_error = str(exc)
            cleanup_after_trial(terminate_children=config.n_jobs == 1)
            trial.set_user_attr("failure_reason", trial_error)
            raise
        result_dict = result_to_dict(result)
        for key, value in result_dict.items():
            trial.set_user_attr(key, value)
        return get_objective_value(result, config.metric)

    record_trial_lock = threading.Lock()

    def record_trial(study, trial):
        nonlocal last_completed_sampler
        # Optuna invokes callbacks concurrently for n_jobs>1. Serialize the
        # sampler snapshot and summary rewrite so neither artifact can regress
        # to an older callback's view.
        with record_trial_lock:
            if config.n_jobs == 1:
                last_completed_sampler = update_optuna_sampler_after_trial(
                    optuna,
                    study,
                    trial,
                    sampler_state_path,
                    last_completed_sampler,
                )
            else:
                # A failed parallel trial cannot be rolled back independently
                # after other trials have sampled. The suggestion checkpoint
                # already includes it; persist sampler after_trial state too.
                save_optuna_sampler(study.sampler, sampler_state_path)
            if (
                tpe_transition_checkpoint_enabled
                and trial_counts_toward_hpo_budget(optuna, trial)
            ):
                transition_sampler = (
                    last_completed_sampler
                    if config.n_jobs == 1
                    else study.sampler
                )
                maybe_save_tpe_transition_sampler(
                    optuna,
                    study,
                    transition_sampler,
                    tpe_transition_sampler_state_path,
                    effective_tpe_startup_trials,
                )
            cleanup_after_trial(terminate_children=config.n_jobs == 1)
            # Refresh summaries after each finished trial so interrupted studies
            # still leave readable progress outside the Optuna database.
            write_trials_summary(study, trials_csv, trials_jsonl)

    if retry_failed_trials_mode:
        enqueued_retries = enqueue_failed_hpo_trial_retries(optuna, study)
        logger.info(
            f"HPO failed/pruned retry mode enabled for {study_dir}. "
            f"Enqueued retry trials: {enqueued_retries}. Parallel jobs: {config.n_jobs}."
        )
        if enqueued_retries <= 0:
            write_trials_summary(study, trials_csv, trials_jsonl)
            logger.info("No failed or pruned Optuna trials were found to retry.")
            return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

        study.optimize(
            objective,
            n_trials=enqueued_retries,
            timeout=config.timeout,
            n_jobs=config.n_jobs,
            callbacks=[record_trial],
            gc_after_trial=True,
        )
        write_trials_summary(study, trials_csv, trials_jsonl)
        if any(trial.state.name == "COMPLETE" for trial in study.trials):
            logger.info(
                f"Best trial: {study.best_trial.number}, value={study.best_value}, params={study.best_trial.params}"
            )
        return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

    # Completed and pruned trials consume the target HPO budget. Failed trials
    # stay in the study history but do not advance the saved sampler, so resume
    # appends replacements for them.
    budget_trials = count_hpo_budget_trials(optuna, study)
    remaining_trials = config.n_trials - budget_trials
    optimize_trials = remaining_trials
    logger.info(
        f"Starting Optuna study outputs in {study_dir}. "
        f"Budget trials: {budget_trials}/{config.n_trials}. "
        f"Recovered unfinished trials: {len(recovered_trials)}. "
        f"Replayed sampler trials: {replayed_trials}. "
        f"Resynced sampler trials: {resynced_trials}. "
        f"Remaining this run: {max(optimize_trials, 0)}. "
        f"Parallel jobs: {config.n_jobs}."
    )
    if optimize_trials <= 0:
        write_trials_summary(study, trials_csv, trials_jsonl)
        logger.info(f"Optuna study already has {budget_trials} budget-counting trials; no new trials requested.")
        return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

    study.optimize(
        objective,
        n_trials=optimize_trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
        callbacks=[record_trial],
        gc_after_trial=True,
    )
    write_trials_summary(study, trials_csv, trials_jsonl)
    if any(trial.state.name == "COMPLETE" for trial in study.trials):
        logger.info(f"Best trial: {study.best_trial.number}, value={study.best_value}, params={study.best_trial.params}")
    return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

def make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl):
    # A study can contain only pruned/failed trials, in which case there is no
    # valid parameter set to use for final retraining.
    complete_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    if not complete_trials:
        return HParamStudyResult(
            study_name=study_name,
            study_dir=study_dir,
            trials_csv=trials_csv,
            trials_jsonl=trials_jsonl,
            best_trial_number=None,
            best_value=None,
            best_params=None,
            best_user_attrs=None,
            completed_trials=[],
        )

    best_trial = study.best_trial
    completed_trial_dicts = make_completed_hparam_trial_dicts(complete_trials)
    return HParamStudyResult(
        study_name=study_name,
        study_dir=study_dir,
        trials_csv=trials_csv,
        trials_jsonl=trials_jsonl,
        best_trial_number=best_trial.number,
        best_value=float(best_trial.value),
        best_params=expand_joint_component_params(dict(best_trial.params)),
        best_user_attrs=dict(best_trial.user_attrs),
        completed_trials=completed_trial_dicts,
    )

def make_completed_hparam_trial_dicts(complete_trials):
    """Return completed trials sorted by highest objective value first."""

    return [
        {
            "trial_number": trial.number,
            "value": float(trial.value),
            "params": expand_joint_component_params(dict(trial.params)),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in sorted(complete_trials, key=lambda trial: (-float(trial.value), trial.number))
    ]

def make_study_dir(base_save_dir, study_name, configured_study_dir=None):
    # Return both the physical path and the path passed as save_dir. The latter
    # must stay relative because initialize_logger prepends logs/ itself.
    if configured_study_dir is None:
        relative_study_dir = Path(base_save_dir) / study_name
        study_dir = Path("logs") / relative_study_dir
    else:
        configured_path = Path(configured_study_dir)
        if configured_path.is_absolute():
            study_dir = configured_path
            relative_study_dir = configured_path
        else:
            relative_study_dir = configured_path
            study_dir = Path("logs") / relative_study_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir, relative_study_dir

def resolve_optuna_storage(configured_storage, study_dir):
    if configured_storage is not None:
        return configured_storage
    storage_path = (Path(study_dir) / "optuna_study.db").resolve()
    return f"sqlite:///{storage_path.as_posix()}"

def make_parallel_optuna_sampler(sampler, n_jobs):
    """Wrap the sampler when Optuna will share it across worker threads."""

    if n_jobs == 1 or isinstance(sampler, ParallelOptunaSampler):
        return sampler
    return ParallelOptunaSampler(sampler)

def run_with_sampler_checkpoint(sampler, path, operation, in_flight_path=None, trial=None):
    """Run trial suggestion atomically with a parallel sampler checkpoint.

    The outer sampler lock remains held until every ``trial.suggest_*`` call
    has also written its value to Optuna storage. The checkpoint therefore
    cannot contain an RNG advance whose corresponding parameter is still only
    in another thread's local state.

    Sequential runs keep ``sampler.pkl`` at the last finished trial so a failed
    trial can be replaced by an identical redraw. They instead record the state
    that already includes this trial's draws in ``in_flight_path``, which is
    what a killed and later recovered trial needs to avoid a repeated
    configuration.
    """

    if not isinstance(sampler, ParallelOptunaSampler):
        try:
            return operation()
        finally:
            if in_flight_path is not None and trial is not None:
                save_in_flight_sampler_state(sampler, in_flight_path, trial)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SAMPLER_STATE_LOCK:
        with sampler.state_lock:
            try:
                return operation()
            finally:
                _write_optuna_sampler_state(sampler.wrapped_sampler, path)

def load_or_make_optuna_sampler(optuna, config, seed, sampler_state_path, tpe_startup_trials=None):
    if config.load_if_exists and sampler_state_path.exists():
        sampler = load_optuna_sampler(sampler_state_path)
        validate_loaded_sampler_matches_config(sampler, config, sampler_state_path)
        logger.info(f"Loaded Optuna sampler state from {sampler_state_path}.")
        return sampler, True
    return (
        make_optuna_sampler(
            optuna,
            config,
            seed,
            tpe_startup_trials=tpe_startup_trials,
        ),
        False,
    )

def validate_loaded_sampler_matches_config(sampler, config, sampler_state_path):
    if isinstance(sampler, ParallelOptunaSampler):
        sampler = sampler.wrapped_sampler
    expected_class_names = {
        "tpe": "TPESampler",
        "random": "RandomSampler",
        "grid": "GridSampler",
    }
    expected_class_name = expected_class_names.get(config.sampler)
    actual_class_name = sampler.__class__.__name__
    if expected_class_name is not None and actual_class_name != expected_class_name:
        raise ValueError(
            f"Optuna sampler state at {sampler_state_path} contains {actual_class_name}, "
            f"but the current HPO config requests {expected_class_name}. "
            "Use the original sampler config, start a new study_dir/storage, or remove the stale sampler.pkl."
        )

def load_optuna_sampler(path):
    try:
        with path.open("rb") as sampler_file:
            return pickle.load(sampler_file)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Optuna sampler state from {path}. "
            "Restore this file, start a new study_dir/storage, or remove it if reproducible sampler resume "
            "is not required."
        ) from exc

def save_optuna_sampler(sampler, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SAMPLER_STATE_LOCK:
        if isinstance(sampler, ParallelOptunaSampler):
            with sampler.state_lock:
                _write_optuna_sampler_state(sampler.wrapped_sampler, path)
        else:
            _write_optuna_sampler_state(sampler, path)

def save_optuna_sampler_if_missing(sampler, path):
    """Atomically save a sampler without replacing an existing checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SAMPLER_STATE_LOCK:
        if path.exists():
            return False
        if isinstance(sampler, ParallelOptunaSampler):
            with sampler.state_lock:
                _write_optuna_sampler_state(sampler.wrapped_sampler, path)
        else:
            _write_optuna_sampler_state(sampler, path)
    return True

def _write_optuna_sampler_state(sampler, path):
    """Write a sampler while the caller holds the sampler-file lock."""

    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("wb") as sampler_file:
            pickle.dump(sampler, sampler_file)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def save_in_flight_sampler_state(sampler, path, trial):
    """Checkpoint the sampler state that includes one trial's parameter draws.

    The stored trial number and parameter values let a resumed run verify that
    the checkpoint really belongs to the interrupted trial before adopting it.
    """

    if isinstance(sampler, ParallelOptunaSampler):
        sampler = sampler.wrapped_sampler
    payload = {
        "trial_number": trial.number,
        "params": dict(trial.params),
        "sampler": sampler,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SAMPLER_STATE_LOCK:
        _write_optuna_sampler_state(payload, path)

def load_in_flight_sampler_state(path):
    """Return the in-flight checkpoint payload, or ``None`` when unusable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as sampler_file:
            payload = pickle.load(sampler_file)
    except Exception as exc:
        # This checkpoint is an optimization for interrupted trials, so a
        # damaged file must not prevent the study from resuming.
        logger.warning(f"Ignoring unreadable in-flight sampler state at {path}: {exc}")
        return None
    if not isinstance(payload, dict) or not {"trial_number", "params", "sampler"} <= set(payload):
        logger.warning(f"Ignoring in-flight sampler state at {path}: unexpected file contents")
        return None
    return payload

def in_flight_state_matches_trial(payload, trial):
    """Check that a checkpoint holds exactly the draws this trial recorded."""

    return (
        payload["trial_number"] == trial.number
        and payload["params"] == dict(trial.params)
    )

def clone_optuna_sampler(sampler):
    if isinstance(sampler, ParallelOptunaSampler):
        with sampler.state_lock:
            return pickle.loads(pickle.dumps(sampler.wrapped_sampler))
    return pickle.loads(pickle.dumps(sampler))

def update_optuna_sampler_after_trial(
    optuna,
    study,
    trial,
    path,
    last_completed_sampler,
    restore_after_incomplete=True,
):
    if not trial_counts_toward_hpo_budget(optuna, trial):
        if restore_after_incomplete:
            study.sampler = clone_optuna_sampler(last_completed_sampler)
        return last_completed_sampler
    save_optuna_sampler(study.sampler, path)
    return clone_optuna_sampler(study.sampler)

def replay_random_sampler_state_if_needed(optuna, study, config, sampler_state_loaded):
    if sampler_state_loaded or not config.load_if_exists or config.sampler != "random":
        return 0
    replay_plans = make_random_sampler_replay_plans(study, config)
    if not replay_plans:
        return 0

    replay_sampler_draws(optuna, study, config, replay_plans)
    logger.info(
        f"No Optuna sampler state file was found; advanced RandomSampler through "
        f"{len(replay_plans)} prior sampled trial(s) using an in-memory Optuna study."
    )
    return len(replay_plans)

def make_random_sampler_replay_plans(study, config):
    plans = []
    for trial in study.get_trials(deepcopy=False):
        if not trial_counts_toward_hpo_budget_name(trial):
            continue
        if trial.datetime_start is None:
            continue
        sampled_names = get_sampled_param_names(trial, config)
        if not sampled_names:
            continue
        plans.append(sampled_names)
    return plans

def get_fixed_param_names(trial):
    fixed_params = trial.system_attrs.get("fixed_params", {})
    if isinstance(fixed_params, dict):
        return set(fixed_params)
    return set()

def count_hpo_budget_trials(optuna, study):
    return sum(trial_counts_toward_hpo_budget(optuna, trial) for trial in study.trials)

def maybe_save_tpe_transition_sampler(
    optuna,
    study,
    sampler,
    path,
    startup_trials,
):
    """Save the one-time sampler state between random startup and TPE."""

    path = Path(path)
    if path.exists():
        return False
    budget_trials = count_hpo_budget_trials(optuna, study)
    if budget_trials < startup_trials:
        return False
    if not save_optuna_sampler_if_missing(sampler, path):
        return False

    if budget_trials == startup_trials:
        logger.info(
            f"Saved TPE transition sampler state to {path} after "
            f"{startup_trials} completed/pruned random startup trial(s)."
        )
    else:
        # Parallel workers can finish close together, so a callback may first
        # observe the study just after the exact count was crossed.
        logger.warning(
            f"Saved the first available TPE transition sampler state to {path} after "
            f"{budget_trials} completed/pruned trials; the configured startup boundary "
            f"was {startup_trials}."
        )
    return True

def trial_counts_toward_hpo_budget(optuna, trial):
    return trial.state in {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }

def trial_counts_toward_hpo_budget_name(trial):
    return trial.state.name in {"COMPLETE", "PRUNED"}


def should_retry_failed_hpo_trials(args, config):
    return bool(getattr(args, "retry_failed_hpo_trials", False) or config.retry_failed_trials)


def get_retryable_hpo_trials(optuna, study):
    retryable_states = {optuna.trial.TrialState.PRUNED}
    fail_state = getattr(optuna.trial.TrialState, "FAIL", None)
    if fail_state is not None:
        retryable_states.add(fail_state)
    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state in retryable_states
    ]


def enqueue_failed_hpo_trial_retries(optuna, study):
    retryable_trials = get_retryable_hpo_trials(optuna, study)
    for trial in retryable_trials:
        study.enqueue_trial(
            dict(trial.params),
            user_attrs={
                "retry_source_trial_number": trial.number,
                "retry_source_trial_state": trial.state.name,
            },
            skip_if_exists=False,
        )
    return len(retryable_trials)

def recover_unfinished_trials(optuna, study):
    # A killed process can leave persistent trials marked RUNNING forever.
    # Reset them to WAITING so Optuna can schedule the same trials again.
    recovered = []
    running_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in running_trials:
        try:
            if study._storage.set_trial_state_values(trial._trial_id, state=optuna.trial.TrialState.WAITING):
                recovered.append(trial)
        except optuna.exceptions.UpdateFinishedTrialError:
            continue

    if recovered:
        logger.warning(
            f"Recovered {len(recovered)} unfinished Optuna trial(s) in study {study.study_name!r}. "
            "They were reset from RUNNING to WAITING and will be rerun with the same trial numbers and parameters."
        )
    return recovered

def resync_sampler_with_recovered_trials(optuna, study, config, recovered_trials, in_flight_path):
    """Advance a resumed sampler past draws recovered trials already recorded.

    A trial recovered from ``RUNNING`` reruns with the parameters Optuna wrote
    to storage before the process was killed, so its ``suggest_*`` calls return
    those values without consuming the sampler again. Sequential runs refresh
    ``sampler.pkl`` only once a trial finishes, so without this resync the next
    fresh trial would resume from an RNG state that predates the recovered
    trial and redraw exactly its configuration, giving the study two trials
    with identical hyperparameters.
    """

    plans = []
    for trial in recovered_trials:
        sampled_names = get_sampled_param_names(trial, config)
        if not sampled_names:
            continue
        plans.append((trial, sampled_names))
    if not plans:
        return 0

    payload = load_in_flight_sampler_state(in_flight_path)
    resynced = 0
    replay_plans = []
    for trial, sampled_names in plans:
        if payload is not None and in_flight_state_matches_trial(payload, trial):
            validate_loaded_sampler_matches_config(payload["sampler"], config, in_flight_path)
            study.sampler = make_parallel_optuna_sampler(payload["sampler"], config.n_jobs)
            logger.info(
                f"Restored the sampler state saved after trial {trial.number} suggested its "
                f"parameters from {in_flight_path}, so its rerun does not repeat as a new trial."
            )
            resynced += 1
            continue
        if isinstance(study.sampler, ParallelOptunaSampler):
            # Parallel runs checkpoint sampler.pkl immediately after every
            # suggestion, so those draws are already part of the loaded state.
            continue
        replay_plans.append(sampled_names)

    if replay_plans:
        replay_sampler_draws(optuna, study, config, replay_plans)
        resynced += len(replay_plans)
        logger.warning(
            f"No matching in-flight sampler checkpoint was found for {len(replay_plans)} recovered "
            f"trial(s) in {in_flight_path}; replayed their recorded parameter draws so the next new "
            "trial does not repeat a recovered configuration. This reproduces the original draws "
            "exactly for random and TPE startup trials."
        )
    return resynced

def get_sampled_param_names(trial, config):
    """Return the configured space names this trial drew from the sampler."""

    if set(trial.params) - set(config.spaces):
        # An unknown parameter means the trial predates the current space, so
        # its draws cannot be reproduced from this configuration.
        return []
    fixed_names = get_fixed_param_names(trial)
    return [
        name
        for name in config.spaces
        if name in trial.params and name not in fixed_names
    ]

def replay_sampler_draws(optuna, study, config, plans):
    """Consume RNG draws for recorded trials through a throwaway study."""

    replay_study = optuna.create_study(direction=config.direction, sampler=study.sampler)
    for sampled_names in plans:
        replay_trial = replay_study.ask()
        for name in sampled_names:
            suggest_value(replay_trial, name, config.spaces[name])
        replay_study.tell(replay_trial, 0.0)

def resolve_tpe_startup_trials(args, config):
    value = getattr(args, "tpe_startup_trials", None)
    if value is None:
        value = config.tpe_startup_trials
    if value is None:
        return None
    validate_tpe_startup_trials(value)
    if config.sampler != "tpe":
        raise ValueError("tpe_startup_trials only applies when sampler is 'tpe'")
    return int(value)

def get_effective_tpe_startup_trials(sampler, fallback=None):
    """Read the startup count actually carried by an instantiated TPESampler."""

    if isinstance(sampler, ParallelOptunaSampler):
        sampler = sampler.wrapped_sampler
    startup_trials = getattr(sampler, "_n_startup_trials", None)
    if startup_trials is None:
        # Lightweight test doubles may not expose Optuna's private attribute.
        # The fallback follows TPESampler's public constructor inputs/default.
        startup_trials = fallback
    if (
        isinstance(startup_trials, bool)
        or not isinstance(startup_trials, (int, np.integer))
        or startup_trials < 0
    ):
        raise RuntimeError(
            "Could not determine the effective n_startup_trials from Optuna's TPESampler"
        )
    return int(startup_trials)

def make_optuna_sampler(optuna, config, seed, tpe_startup_trials=None):
    sampler_params = dict(config.sampler_params)
    if config.sampler in {"tpe", "random"}:
        sampler_params.setdefault("seed", seed)
    if config.sampler == "tpe":
        if tpe_startup_trials is not None:
            sampler_params["n_startup_trials"] = int(tpe_startup_trials)
        return optuna.samplers.TPESampler(**sampler_params)
    if config.sampler == "random":
        return optuna.samplers.RandomSampler(**sampler_params)
    if config.sampler == "grid":
        return optuna.samplers.GridSampler(search_space=make_grid_search_space(config.spaces), **sampler_params)
    raise ValueError(f"Unsupported Optuna sampler: {config.sampler}")

def make_optuna_pruner(optuna, config):
    pruner_params = dict(config.pruner_params)
    if config.pruner == "none":
        return optuna.pruners.NopPruner(**pruner_params)
    if config.pruner == "median":
        return optuna.pruners.MedianPruner(**pruner_params)
    if config.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
    if config.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(**pruner_params)
    raise ValueError(f"Unsupported Optuna pruner: {config.pruner}")

def validate_study_distributions_compatible(optuna, study, config, storage):
    # Optuna requires each parameter name to retain a compatible distribution
    # across a persistent study. Check explicitly to provide a clearer error.
    configured_distributions = make_optuna_distributions(optuna, config.spaces)
    previous_names = {
        name
        for trial in study.trials
        for name in trial.distributions
    }
    configured_names = set(configured_distributions)
    if previous_names and previous_names != configured_names:
        raise ValueError(
            "Existing Optuna study is incompatible with the current hyperparameter search space. "
            f"Study {study.study_name!r} in storage {storage!r} uses parameter names "
            f"{sorted(previous_names)}, but the current resolved config uses {sorted(configured_names)}. "
            "Use a new study_name/save_dir/study_dir/storage or remove the stale Optuna database."
        )
    for trial in study.trials:
        for name, previous_distribution in trial.distributions.items():
            if name not in configured_distributions:
                continue
            configured_distribution = configured_distributions[name]
            try:
                optuna.distributions.check_distribution_compatibility(
                    previous_distribution,
                    configured_distribution,
                )
            except ValueError as exc:
                raise ValueError(
                    "Existing Optuna study is incompatible with the current hyperparameter search space. "
                    f"Study {study.study_name!r} in storage {storage!r} already has parameter {name!r} "
                    f"with distribution {previous_distribution!r}, but the current config uses "
                    f"{configured_distribution!r}. Use a new study_name/save_dir/study_dir/storage, "
                    "restore the old search space, or remove the stale Optuna database."
                ) from exc

def make_optuna_distributions(optuna, spaces):
    distributions = {}
    for name, spec in spaces.items():
        if isinstance(spec, list):
            distributions[name] = optuna.distributions.CategoricalDistribution(spec)
            continue

        space_type = spec.get("type", "categorical" if "choices" in spec else None)
        if space_type == "categorical":
            distributions[name] = optuna.distributions.CategoricalDistribution(spec["choices"])
        elif space_type == "float":
            distributions[name] = optuna.distributions.FloatDistribution(
                low=spec["low"],
                high=spec["high"],
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
            )
        elif space_type == "int":
            # IntDistribution rejects step=None, so omit it exactly like
            # suggest_int does and keep Optuna's default step of 1.
            int_kwargs = {}
            if spec.get("step") is not None:
                int_kwargs["step"] = spec["step"]
            distributions[name] = optuna.distributions.IntDistribution(
                low=spec["low"],
                high=spec["high"],
                log=bool(spec.get("log", False)),
                **int_kwargs,
            )
        else:
            raise ValueError(f"Unsupported search space type for {name!r}: {space_type}")
    return distributions

def make_grid_search_space(spaces):
    grid = {}
    for name, spec in spaces.items():
        if isinstance(spec, list):
            grid[name] = spec
        elif spec.get("type", "categorical" if "choices" in spec else None) == "categorical":
            grid[name] = spec["choices"]
        else:
            raise ValueError(f"GridSampler only supports categorical spaces; {name!r} is {spec}")
    return grid

def make_trial_args_and_ssl_config(base_args, config, trial):
    raw_suggested_params = {}
    for name, spec in config.spaces.items():
        # Map the JSON search-space representation to the corresponding Optuna
        # suggest_* call for this trial.
        raw_suggested_params[name] = suggest_value(trial, name, spec)

    suggested_params = expand_joint_component_params(raw_suggested_params)
    trial_args, ssl_config = make_args_and_ssl_config_from_params(base_args, suggested_params)
    return trial_args, ssl_config, suggested_params

def make_args_and_ssl_config_from_params(base_args, params):
    """Apply flat Optuna parameters to CLI args and nested SSL settings."""

    params = resolve_lr_ratio_params(base_args, params)
    trial_args = copy.deepcopy(base_args)
    ssl_overrides = []
    labeled_batch_fraction = None

    # Names beginning with ssl./ssl_config. target the nested dataclass;
    # everything else is a direct command-line argument override.
    for name, value in params.items():
        if name == LABELED_BATCH_FRACTION_HPARAM_KEY:
            # Held back until batch_sampler has set batch_size/sampler_m below.
            labeled_batch_fraction = value
        elif is_ssl_override(name):
            ssl_overrides.append((name, value))
        elif is_component_override(name):
            set_component_param(trial_args, name, value)
        else:
            set_arg_value(trial_args, name, value)

    # A replayed ablation variant outranks the trial's own values, and has to land
    # before the labeled-batch share below is snapped against batch_size and sampler_m.
    ablation.apply_ablation_args(trial_args)

    if labeled_batch_fraction is not None:
        if getattr(trial_args, "mode", None) != "ssl":
            raise ValueError(
                f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} sizes the two-stream SSL sampler "
                "and requires --mode ssl"
            )
        ssl_overrides.append(
            (
                LABELED_BATCH_SIZE_HPARAM_KEY,
                derive_labeled_batch_size_from_fraction(
                    trial_args.batch_size,
                    trial_args.sampler_m,
                    labeled_batch_fraction,
                ),
            )
        )

    ssl_config = semi_supervised.load_ssl_config(
        trial_args.ssl_config,
        default_seed=trial_args.seed,
        default_support_seed=get_support_seed(trial_args),
    )
    if ssl_overrides:
        # Convert the immutable dataclass to a mutable dictionary, apply nested
        # overrides, and then rebuild/validate a new dataclass instance.
        ssl_dict = ssl_config.to_dict()
        for name, value in ssl_overrides:
            path_parts = get_ssl_override_path(name)
            if path_parts == [LABELED_BATCH_SIZE_HPARAM_KEY]:
                value = parse_labeled_batch_size_choice(value)
            set_nested_value(ssl_dict, path_parts, value)
        ssl_config = semi_supervised.SemiSupervisedConfig(**ssl_dict)
        semi_supervised.validate_ssl_config(ssl_config)

    ssl_config = resolve_mode_ssl_config(trial_args, ssl_config)
    trial_args = resolve_loss_driven_supervised_args(trial_args)
    resolve_scheduler_batch_device(trial_args, ssl_config)
    trial_args = resolve_hpo_warmup_objective(trial_args, ssl_config)

    if (
        (set(params) & LABELED_BATCH_SIZE_HPARAM_KEYS or labeled_batch_fraction is not None)
        and ssl_config.labeled_batch_size is not None
    ):
        validate_two_stream_hpo_batch_combination(
            trial_args.batch_size,
            trial_args.sampler_m,
            ssl_config.labeled_batch_size,
        )
    validate_run_args(trial_args, ssl_config)
    return trial_args, ssl_config


def resolve_lr_ratio_params(base_args, params):
    """Turn the ArcFace/ProxyAnchor LR ridge coordinates into optimizer LRs.

    ``base_lr`` is the geometric mean of the model/projection LR and the
    classification-loss LR. ``lr_ratio`` is model LR divided by classifier LR.
    The input mapping is copied so the raw Optuna coordinates remain available
    in trial summaries and can be replayed for final training.

    For compatibility with a config that adds ``lr_ratio`` beside an existing
    ``lr`` space, ``lr`` is accepted as a shorthand for ``base_lr``. Resolved
    study configs use the explicit ``base_lr`` name.
    """

    resolved = dict(params)
    has_base_lr = BASE_LR_HPARAM_KEY in resolved
    has_lr_ratio = LR_RATIO_HPARAM_KEY in resolved
    if not has_base_lr and not has_lr_ratio:
        return resolved

    loss_name = getattr(base_args, "loss", None)
    if loss_name not in LR_RATIO_LOSSES:
        raise ValueError(
            f"{BASE_LR_HPARAM_KEY!r}/{LR_RATIO_HPARAM_KEY!r} may only be used "
            f"with {sorted(LR_RATIO_LOSSES)}, got loss {loss_name!r}"
        )

    if not has_base_lr and has_lr_ratio and "lr" in resolved:
        resolved[BASE_LR_HPARAM_KEY] = resolved.pop("lr")
        has_base_lr = True
    if not has_base_lr or not has_lr_ratio:
        raise ValueError(
            f"{BASE_LR_HPARAM_KEY!r} and {LR_RATIO_HPARAM_KEY!r} must be searched together"
        )

    conflicts = sorted({"lr", "classifier_lr"} & set(resolved))
    if conflicts:
        raise ValueError(
            f"{BASE_LR_HPARAM_KEY!r}/{LR_RATIO_HPARAM_KEY!r} determine both optimizer "
            f"learning rates; remove the independent spaces {conflicts}"
        )

    base_lr = resolved.pop(BASE_LR_HPARAM_KEY)
    lr_ratio = resolved.pop(LR_RATIO_HPARAM_KEY)
    if (
        isinstance(base_lr, bool)
        or not isinstance(base_lr, (int, float))
        or not math.isfinite(base_lr)
        or base_lr <= 0
    ):
        raise ValueError(f"{BASE_LR_HPARAM_KEY!r} must be a finite positive number")
    if (
        isinstance(lr_ratio, bool)
        or not isinstance(lr_ratio, (int, float))
        or not math.isfinite(lr_ratio)
        or lr_ratio <= 0
    ):
        raise ValueError(f"{LR_RATIO_HPARAM_KEY!r} must be a finite positive number")

    sqrt_ratio = math.sqrt(lr_ratio)
    model_lr = base_lr * sqrt_ratio
    classifier_lr = base_lr / sqrt_ratio
    derived_lrs = {"lr": model_lr, "classifier_lr": classifier_lr}
    lr_bounds = {
        "lr": (LR_RATIO_MIN_OPTIMIZER_LR, LR_RATIO_MAX_MODEL_LR),
        "classifier_lr": (
            LR_RATIO_MIN_OPTIMIZER_LR,
            LR_RATIO_MAX_CLASSIFIER_LR,
        ),
    }
    invalid_lrs = {
        name: (value, lr_bounds[name])
        for name, value in derived_lrs.items()
        if not lr_bounds[name][0] <= value <= lr_bounds[name][1]
    }
    if invalid_lrs:
        values = ", ".join(
            f"{name}={value:.6g} outside [{low:g}, {high:g}]"
            for name, (value, (low, high)) in invalid_lrs.items()
        )
        raise LrRatioOutOfBoundsError(
            f"Derived {values} from "
            f"{BASE_LR_HPARAM_KEY}={base_lr:.6g}, {LR_RATIO_HPARAM_KEY}={lr_ratio:.6g}"
        )

    resolved.update(derived_lrs)
    return resolved


def resolve_hpo_warmup_objective(args, ssl_config):
    """Warm up with the fully resolved objective of this HPO trial.

    This runs after component hyperparameters have been applied, so the warmup
    receives the selected loss/miner names and the exact tuned constructor
    parameters. Loss-driven STML is the sole exception: its unsupervised
    multi-view objective cannot run on the labeled-only warmup batches.
    """

    if not uses_ssl_warmup_objective(ssl_config) or args.loss == "STMLLoss":
        return args

    resolved = copy.deepcopy(args)
    resolved.warmup_loss = resolved.loss
    resolved.warmup_loss_params = copy.deepcopy(resolved.loss_params)
    resolved.warmup_miner = resolved.miner
    resolved.warmup_miner_params = copy.deepcopy(resolved.miner_params)
    return resolved

def suggest_value(trial, name, spec):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    space_type = spec.get("type", "categorical" if "choices" in spec else None)
    if space_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if space_type == "float":
        kwargs = {
            "low": spec["low"],
            "high": spec["high"],
            "log": bool(spec.get("log", False)),
        }
        if spec.get("step") is not None:
            kwargs["step"] = spec["step"]
        return trial.suggest_float(name, **kwargs)
    if space_type == "int":
        kwargs = {
            "low": spec["low"],
            "high": spec["high"],
            "log": bool(spec.get("log", False)),
        }
        if spec.get("step") is not None:
            kwargs["step"] = spec["step"]
        return trial.suggest_int(name, **kwargs)
    raise ValueError(f"Unsupported search space type for {name!r}: {space_type}")

def is_ssl_override(name):
    return (
        name == LABELED_BATCH_SIZE_HPARAM_KEY
        or name.startswith("ssl_config.")
        or name.startswith("ssl.")
    )

def get_ssl_override_path(name):
    if name == LABELED_BATCH_SIZE_HPARAM_KEY:
        return [LABELED_BATCH_SIZE_HPARAM_KEY]
    return name.split(".")[1:]

def is_component_override(name):
    return name.startswith(LOSS_HPARAM_PREFIX) or name.startswith(MINER_HPARAM_PREFIX)

def filter_component_hparam_config(args, config):
    """Remove class-qualified spaces that do not target this fixed scenario."""

    spaces = {
        name: spec
        for name, spec in config.spaces.items()
        if component_override_applies(args, name)
    }
    return replace(config, spaces=spaces)

def filter_loss_dependent_hparam_config(args, config):
    """Remove HPO spaces for optimizer knobs unused by the fixed loss."""

    spaces = dict(config.spaces)
    loss_name = getattr(args, "loss", None)
    lr_ridge_keys = {BASE_LR_HPARAM_KEY, LR_RATIO_HPARAM_KEY}
    present_lr_ridge_keys = lr_ridge_keys & set(spaces)

    if loss_name in LR_RATIO_LOSSES and present_lr_ridge_keys:
        # A config may add lr_ratio alongside its inherited lr space. Treat
        # that lr distribution as the base/geometric-mean distribution and
        # expose the unambiguous name to Optuna and persisted study metadata.
        if BASE_LR_HPARAM_KEY not in spaces and LR_RATIO_HPARAM_KEY in spaces and "lr" in spaces:
            spaces[BASE_LR_HPARAM_KEY] = spaces.pop("lr")
            present_lr_ridge_keys.add(BASE_LR_HPARAM_KEY)
        if present_lr_ridge_keys != lr_ridge_keys:
            missing = sorted(lr_ridge_keys - present_lr_ridge_keys)
            raise ValueError(
                f"Loss {loss_name!r} has an incomplete LR-ratio search space; "
                f"missing {missing}. Search {BASE_LR_HPARAM_KEY!r} and "
                f"{LR_RATIO_HPARAM_KEY!r} together."
            )

        removed = sorted({"lr", "classifier_lr"} & set(spaces))
        for name in removed:
            del spaces[name]
        logger.info(
            f"Using {BASE_LR_HPARAM_KEY}/{LR_RATIO_HPARAM_KEY} for loss {loss_name!r}; "
            f"the derived trial values set lr and classifier_lr. Removed independent spaces: {removed}."
        )
    elif present_lr_ridge_keys:
        for name in present_lr_ridge_keys:
            del spaces[name]
        logger.info(
            f"Excluded {sorted(present_lr_ridge_keys)} from HPO because LR-ratio "
            f"parameterization only applies to {sorted(LR_RATIO_LOSSES)}, not loss {loss_name!r}."
        )

    if loss_name not in CLASSIFICATION_LOSSES and "classifier_lr" in spaces:
        del spaces["classifier_lr"]
        logger.info(
            f"Excluded classifier_lr from HPO because loss {loss_name!r} "
            "does not use a classifier optimizer."
        )

    return config if spaces == config.spaces else replace(config, spaces=spaces)

SLADE_UNLABELED_RANKING_LOSS_HPARAM_KEY = (
    "ssl_config.method_params.regularizer_params.unlabeled_ranking_loss"
)
SLADE_CONTRASTIVE_MARGIN_HPARAM_KEYS = (
    "ssl_config.method_params.regularizer_params.contrastive_pos_margin",
    "ssl_config.method_params.regularizer_params.contrastive_neg_margin",
)


def filter_slade_contrastive_spaces(args, config):
    """Remove Eq 1's margins unless SLADE's unlabeled term actually reads them.

    ``unlabeled_ranking_loss='supervised'`` -- the default -- hands Eq 9's
    unlabeled term the supervised criterion, which carries its own margins.
    ``contrastive_pos_margin`` and ``contrastive_neg_margin`` are then read by
    nothing, and ``make_trial_args_and_ssl_config`` suggests every space on every
    trial, so leaving them in would give the sampler two dimensions that cannot
    move the objective. TPE would still build a density estimate over them and
    split its observations across values that are all the same run.
    """

    present = [
        name for name in SLADE_CONTRASTIVE_MARGIN_HPARAM_KEYS if name in config.spaces
    ]
    if not present:
        return config

    if SLADE_UNLABELED_RANKING_LOSS_HPARAM_KEY in config.spaces:
        # The mode is itself being searched, so the margins are live in some
        # trials. Dropping them would break exactly the trials they exist for;
        # the inert ones only cost the warning the regularizer already logs.
        logger.info(
            f"Kept {sorted(present)} in HPO because "
            f"{SLADE_UNLABELED_RANKING_LOSS_HPARAM_KEY!r} is searched, so some "
            "trials use Eq 1's margins and some ignore them."
        )
        return config

    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    regularizer_params = dict(ssl_config.method_params.get("regularizer_params") or {})
    mode = str(regularizer_params.get("unlabeled_ranking_loss", "supervised"))
    if mode == "contrastive":
        return config

    spaces = {name: spec for name, spec in config.spaces.items() if name not in present}
    logger.info(
        f"Excluded {sorted(present)} from HPO because slade "
        f"unlabeled_ranking_loss={mode!r} reuses the supervised loss and never "
        "reads Eq 1's margins."
    )
    return replace(config, spaces=spaces)


LRML_CONTRASTIVE_HPARAM_KEY = (
    "ssl_config.method_params.regularizer_params.contrastive"
)
LRML_CONTRASTIVE_ONLY_HPARAM_KEYS = (
    "ssl_config.method_params.regularizer_params.contrastive_margin",
    "ssl_config.method_params.regularizer_params.contrastive_pairs",
)


def filter_lrml_contrastive_spaces(args, config):
    """Remove eq. (3)'s repulsion knobs unless lrml actually draws its pairs.

    ``contrastive=False`` -- the default -- makes ``_draw_contrastive_pairs``
    return before it reads ``contrastive_margin`` or ``contrastive_pairs``, so
    both describe a repulsion term that never runs. This is the lrml counterpart
    of :func:`filter_slade_contrastive_spaces`: every space is suggested on every
    trial, so leaving them in would hand TPE dimensions that cannot move the
    objective and split its observations across values that are all one run.
    """

    present = [
        name for name in LRML_CONTRASTIVE_ONLY_HPARAM_KEYS if name in config.spaces
    ]
    if not present:
        return config

    if LRML_CONTRASTIVE_HPARAM_KEY in config.spaces:
        # The switch is itself being searched, so the repulsion is live in some
        # trials and dropping its knobs would break exactly those.
        logger.info(
            f"Kept {sorted(present)} in HPO because "
            f"{LRML_CONTRASTIVE_HPARAM_KEY!r} is searched, so some trials draw "
            "eq. (3)'s repulsion pairs and some do not."
        )
        return config

    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    regularizer_params = dict(ssl_config.method_params.get("regularizer_params") or {})
    if bool(regularizer_params.get("contrastive", False)):
        return config

    spaces = {name: spec for name, spec in config.spaces.items() if name not in present}
    logger.info(
        f"Excluded {sorted(present)} from HPO because lrml contrastive=False "
        "draws no repulsion pairs, so nothing reads eq. (3)'s margin."
    )
    return replace(config, spaces=spaces)


def make_backbone_tuning_spaces_aware(args, config):
    """Remove projectionless choices that cannot train with a frozen backbone."""

    normalize_backbone_tuning_args(args)
    if args.backbone_tuning != BACKBONE_TUNING_FROZEN:
        return config
    if "feat_dim" not in config.spaces:
        if args.feat_dim is None:
            raise ValueError(
                "backbone_tuning='frozen' requires a fixed non-null feat_dim or a feat_dim HPO space "
                "with non-null choices"
            )
        return config

    feat_dim_spec = config.spaces["feat_dim"]
    if not is_categorical_space(feat_dim_spec):
        return config
    choices = [choice for choice in get_categorical_choices(feat_dim_spec) if choice is not None]
    if not choices:
        raise ValueError(
            "backbone_tuning='frozen' removed every feat_dim HPO choice; add at least one non-null dimension"
        )

    spaces = dict(config.spaces)
    if isinstance(feat_dim_spec, list):
        spaces["feat_dim"] = choices
    else:
        resolved_spec = dict(feat_dim_spec)
        resolved_spec["choices"] = choices
        spaces["feat_dim"] = resolved_spec
    if len(choices) != len(get_categorical_choices(feat_dim_spec)):
        logger.info(
            "Excluded feat_dim=None from HPO because backbone_tuning='frozen' requires a trainable projection head"
        )
    return replace(config, spaces=spaces)

def make_sampler_spaces_k_shot_aware(args, config):
    """Remove sampler choices that would repeat examples within a k-shot class."""

    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    if ssl_config.label_sampling_mode != "class_subset_k_shot" or ssl_config.labeled_per_class is None:
        return config

    k = int(ssl_config.labeled_per_class)
    spaces = dict(config.spaces)
    excluded_count = 0

    batch_sampler_spec = spaces.get(BATCH_SAMPLER_HPARAM_KEY)
    sampler_m_spec = spaces.get("sampler_m")
    if batch_sampler_spec is None and sampler_m_spec is None and args.sampler_m > k:
        raise ValueError(
            f"Fixed sampler_m={args.sampler_m} is invalid for class_subset_k_shot k={k}; "
            "sampler_m must be less than or equal to k."
        )

    if batch_sampler_spec is not None:
        choices = get_categorical_choices(batch_sampler_spec)
        valid_choices = [choice for choice in choices if parse_batch_sampler_choice(choice)[1] <= k]
        excluded_count += len(choices) - len(valid_choices)
        spaces[BATCH_SAMPLER_HPARAM_KEY] = replace_categorical_choices(
            BATCH_SAMPLER_HPARAM_KEY,
            batch_sampler_spec,
            valid_choices,
            k,
        )

    if sampler_m_spec is not None:
        if is_categorical_space(sampler_m_spec):
            choices = get_categorical_choices(sampler_m_spec)
            valid_choices = [choice for choice in choices if not isinstance(choice, int) or choice <= k]
            excluded_count += len(choices) - len(valid_choices)
            spaces["sampler_m"] = replace_categorical_choices("sampler_m", sampler_m_spec, valid_choices, k)
        elif isinstance(sampler_m_spec, dict) and sampler_m_spec.get("type") == "int":
            constrained_spec = dict(sampler_m_spec)
            old_high = constrained_spec["high"]
            constrained_spec["high"] = min(old_high, k)
            if constrained_spec["low"] > constrained_spec["high"]:
                raise ValueError(
                    f"No valid sampler_m values remain for class_subset_k_shot k={k}: "
                    f"configured range is [{constrained_spec['low']}, {old_high}]."
                )
            spaces["sampler_m"] = constrained_spec

    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} sampler hyperparameter choices with sampler_m > k={k} "
            "to prevent MPerClassSampler replacement."
        )
    return replace(config, spaces=spaces)

def make_sampler_spaces_label_budget_aware(args, config, training_label_sets_factory=None):
    """Constrain sampler spaces by two-stream quotas and labeled-data capacity."""

    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    config = make_two_stream_sampler_spaces_constraint_aware(args, config, ssl_config)
    config = make_stml_in_batch_spaces_constraint_aware(args, config, ssl_config)
    capacity_space = get_sampler_capacity_space(args, config, ssl_config)
    if capacity_space is None:
        return config

    space_name, sampler_spec, capacity_choices = capacity_space
    varying_split_keys = sorted(set(config.spaces) & SAMPLER_CAPACITY_HPARAM_KEYS)
    if varying_split_keys:
        logger.info(
            "Cannot prefilter sampler choices because these HPO dimensions change "
            f"the labeled training split: {varying_split_keys}"
        )
        return config

    if ssl_config.label_sampling_mode not in {"class_subset", "class_subset_k_shot"}:
        return config

    if training_label_sets_factory is None:
        training_label_sets_factory = make_label_budget_training_label_sets
    training_label_sets = training_label_sets_factory(args, ssl_config)
    valid_choices = filter_sampler_capacity_choices_for_training_labels(
        capacity_choices,
        training_label_sets,
    )
    excluded_count = len(capacity_choices) - len(valid_choices)
    if not valid_choices:
        fold_summaries = summarize_training_label_sets(training_label_sets)
        raise ValueError(
            "No valid sampler choices remain for the selected label budget and validation splits. "
            f"Fold labeled-data summaries: {fold_summaries}"
        )

    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} sampler hyperparameter choices that cannot form "
            "an MPerClassSampler batch in every labeled training split. "
            f"Remaining choices: {valid_choices}. "
            f"Fold labeled-data summaries: {summarize_training_label_sets(training_label_sets)}"
        )

    spaces = dict(config.spaces)
    spaces[space_name] = replace_categorical_choices_for_label_budget(
        sampler_spec,
        valid_choices,
    )
    return replace(config, spaces=spaces)

def make_two_stream_sampler_spaces_constraint_aware(args, config, ssl_config=None):
    """Represent only valid categorical two-stream batch-size combinations."""

    if getattr(args, "mode", None) != "ssl":
        return config
    if ssl_config is None:
        ssl_config = semi_supervised.load_ssl_config(
            args.ssl_config,
            default_seed=args.seed,
            default_support_seed=get_support_seed(args),
        )

    spaces = dict(config.spaces)
    labeled_keys = sorted(set(spaces) & LABELED_BATCH_SIZE_HPARAM_KEYS)
    if len(labeled_keys) > 1:
        raise ValueError(
            "Use only one labeled-batch-size HPO key; "
            f"these keys are aliases for the same SSL setting: {labeled_keys}"
        )
    labeled_key = labeled_keys[0] if labeled_keys else None
    batch_sampler_spec = spaces.get(BATCH_SAMPLER_HPARAM_KEY)
    independently_tuned_sampler_keys = sorted({"batch_size", "sampler_m"} & set(spaces))
    if LABELED_BATCH_FRACTION_HPARAM_KEY in spaces:
        return make_labeled_batch_fraction_spaces_constraint_aware(
            args,
            config,
            ssl_config,
            labeled_key,
            independently_tuned_sampler_keys,
        )
    if labeled_key is not None and batch_sampler_spec is None and independently_tuned_sampler_keys:
        raise ValueError(
            "labeled_batch_size HPO must pair changing batch_size/sampler_m values through "
            f"the {BATCH_SAMPLER_HPARAM_KEY!r} categorical space; remove separate spaces "
            f"{independently_tuned_sampler_keys}"
        )
    if batch_sampler_spec is None and labeled_key is None:
        return config

    fixed_labeled_batch_size = ssl_config.labeled_batch_size
    if labeled_key is None and fixed_labeled_batch_size is None:
        return config
    validate_two_stream_hpo_method(ssl_config, LABELED_BATCH_SIZE_HPARAM_KEY)

    if batch_sampler_spec is None:
        batch_choices = [f"{int(args.batch_size)}:{int(args.sampler_m)}"]
    else:
        batch_choices = get_categorical_choices(batch_sampler_spec)

    if labeled_key is None:
        labeled_batch_size_spec = None
        labeled_choices = [int(fixed_labeled_batch_size)]
    else:
        labeled_batch_size_spec = spaces[labeled_key]
        labeled_choices = [
            parse_labeled_batch_size_choice(choice)
            for choice in get_categorical_choices(labeled_batch_size_spec)
        ]

    valid_combinations = []
    for batch_choice, labeled_batch_size in itertools.product(batch_choices, labeled_choices):
        batch_size, sampler_m = parse_batch_sampler_choice(batch_choice)
        try:
            validate_two_stream_hpo_batch_combination(
                batch_size,
                sampler_m,
                labeled_batch_size,
            )
        except ValueError:
            continue
        valid_combinations.append((batch_choice, labeled_batch_size))

    combination_count = len(batch_choices) * len(labeled_choices)
    if not valid_combinations:
        raise ValueError(
            "No valid two-stream sampler HPO combinations remain. labeled_batch_size must be "
            "no larger than unlabeled_batch_size (batch_size - labeled_batch_size), "
            "and both stream sizes must be divisible by sampler_m."
        )

    excluded_count = combination_count - len(valid_combinations)
    if batch_sampler_spec is not None and labeled_key is not None:
        del spaces[BATCH_SAMPLER_HPARAM_KEY]
        del spaces[labeled_key]
        spaces[JOINT_TWO_STREAM_HPARAM_KEY] = [
            serialize_joint_component_params(
                {
                    BATCH_SAMPLER_HPARAM_KEY: batch_choice,
                    labeled_key: labeled_batch_size,
                }
            )
            for batch_choice, labeled_batch_size in valid_combinations
        ]
    elif batch_sampler_spec is not None:
        valid_batch_choices = [batch_choice for batch_choice, _ in valid_combinations]
        spaces[BATCH_SAMPLER_HPARAM_KEY] = replace_categorical_choices_for_label_budget(
            batch_sampler_spec,
            valid_batch_choices,
        )
    else:
        valid_labeled_choices = list(
            dict.fromkeys(labeled_batch_size for _, labeled_batch_size in valid_combinations)
        )
        spaces[labeled_key] = replace_categorical_choices_for_label_budget(
            labeled_batch_size_spec,
            valid_labeled_choices,
        )

    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} of {combination_count} two-stream sampler HPO combinations; "
            f"retained {len(valid_combinations)} with labeled_batch_size <= unlabeled_batch_size "
            "and both stream sizes divisible by sampler_m."
        )
    return replace(config, spaces=spaces)

def make_labeled_batch_fraction_spaces_constraint_aware(
    args,
    config,
    ssl_config,
    labeled_key,
    independently_tuned_sampler_keys,
):
    """Keep the labeled share and the batch size as independent dimensions.

    Enumerating valid ``(batch_sampler, labeled_batch_size)`` pairs makes the
    marginal over batch sizes proportional to how many labeled sizes each one
    can host, which starves the small batches: at
    ``labeled_batch_size in {16..256}`` a 32-image batch admits one partner and
    a 512-image batch admits five. A fraction stays valid at every batch size,
    so the two dimensions never have to be crossed and every ``batch_sampler``
    choice keeps the same sampling probability.
    """

    if labeled_key is not None:
        raise ValueError(
            f"Search either {LABELED_BATCH_FRACTION_HPARAM_KEY!r} or {labeled_key!r}, "
            "not both; the fraction already determines the labeled batch size"
        )
    if independently_tuned_sampler_keys:
        raise ValueError(
            f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} must read batch_size/sampler_m from the "
            f"{BATCH_SAMPLER_HPARAM_KEY!r} categorical space; remove separate spaces "
            f"{independently_tuned_sampler_keys}"
        )
    validate_two_stream_hpo_method(ssl_config, LABELED_BATCH_FRACTION_HPARAM_KEY)

    spaces = dict(config.spaces)
    batch_sampler_spec = spaces.get(BATCH_SAMPLER_HPARAM_KEY)
    if batch_sampler_spec is None:
        # A fixed sampler still has to leave room for both streams.
        if get_max_labeled_batch_size(args.batch_size, args.sampler_m) < args.sampler_m:
            raise ValueError(
                f"batch_size={args.batch_size} cannot host both streams of an "
                f"{LABELED_BATCH_FRACTION_HPARAM_KEY!r} split at sampler_m={args.sampler_m}; "
                f"batch_size must be at least {2 * int(args.sampler_m)}"
            )
        return config

    batch_choices = get_categorical_choices(batch_sampler_spec)
    valid_batch_choices = []
    for batch_choice in batch_choices:
        batch_size, sampler_m = parse_batch_sampler_choice(batch_choice)
        if get_max_labeled_batch_size(batch_size, sampler_m) >= sampler_m:
            valid_batch_choices.append(batch_choice)

    if not valid_batch_choices:
        raise ValueError(
            f"No {BATCH_SAMPLER_HPARAM_KEY!r} choice can host a two-stream split: every "
            "batch_size must fit at least two sampler_m groups so the labeled stream stays "
            f"no larger than the unlabeled one. Choices: {batch_choices}"
        )

    excluded_count = len(batch_choices) - len(valid_batch_choices)
    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} of {len(batch_choices)} {BATCH_SAMPLER_HPARAM_KEY!r} "
            "choices too small to split into a labeled and an unlabeled sampler_m group; "
            f"retained {valid_batch_choices}."
        )
        spaces[BATCH_SAMPLER_HPARAM_KEY] = replace_categorical_choices_for_label_budget(
            batch_sampler_spec,
            valid_batch_choices,
        )
        return replace(config, spaces=spaces)
    return config

def make_stml_in_batch_spaces_constraint_aware(args, config, ssl_config):
    """Search only sampler/graph combinations the in-batch STML mode can run.

    The per-step mode ties four knobs together: the graph's labeled rows come out
    of the supervised batch, and its unlabeled stream is filled with whole
    neighbor groups. Searching them independently would hand Optuna combinations
    that fail at ``validate_run_args`` and lose the trial, so the searched
    dimensions collapse into one joint categorical space of valid tuples -- the
    same treatment ``labeled_batch_size`` gets for the two-stream sampler.
    """

    if ssl_config.method != "stml_threshold":
        return config
    if ssl_config.graph_batch_mode == "in_batch_merged":
        return drop_merged_inert_stml_spaces(config)
    if ssl_config.graph_batch_mode != "in_batch":
        return drop_in_batch_only_stml_spaces(config)

    spaces = dict(config.spaces)
    labeled_spec = spaces.get("ssl_config.graph_labeled_batch_size")
    if labeled_spec is not None and None in get_categorical_choices(labeled_spec):
        # A null here reads as "do not subsample the labeled rows", which this
        # mode cannot express: it always adds its term on top of the engine's
        # supervised loss, so the labeled rows would still be counted twice.
        raise ValueError(
            "'ssl_config.graph_labeled_batch_size' cannot be null under "
            "graph_batch_mode='in_batch'. Use graph_batch_mode='in_batch_merged' to put the "
            "whole supervised batch in the graph and run a single loss over it."
        )
    searched_keys = [
        name
        for name in (BATCH_SAMPLER_HPARAM_KEY, *STML_IN_BATCH_HPARAM_KEYS)
        if name in spaces
    ]
    if not searched_keys:
        return config

    method_params = dict(ssl_config.method_params)
    fixed_values = {
        BATCH_SAMPLER_HPARAM_KEY: f"{int(args.batch_size)}:{int(args.sampler_m)}",
        "ssl_config.graph_labeled_batch_size": ssl_config.graph_labeled_batch_size,
        "ssl_config.graph_unlabeled_batch_size": ssl_config.graph_unlabeled_batch_size,
        "ssl_config.method_params.n_neighbors": method_params.get("n_neighbors"),
    }
    choices_by_key = {
        name: (
            list(get_categorical_choices(spaces[name]))
            if name in spaces
            else [fixed_values[name]]
        )
        for name in fixed_values
    }

    valid_combinations = [
        combination
        for combination in (
            dict(zip(choices_by_key, values))
            for values in itertools.product(*choices_by_key.values())
        )
        if stml_in_batch_combination_is_valid(args, ssl_config, combination)
    ]
    combination_count = int(np.prod([len(values) for values in choices_by_key.values()]))
    if not valid_combinations:
        raise ValueError(
            "No valid in-batch STML sampler/graph HPO combinations remain. "
            "graph_labeled_batch_size must fit inside batch_size and "
            "graph_unlabeled_batch_size must be divisible by method_params.n_neighbors; "
            f"searched {sorted(searched_keys)}."
        )

    if len(searched_keys) == 1:
        # One searched dimension needs no joint space, only a pruned one.
        searched_key = searched_keys[0]
        retained = list(
            dict.fromkeys(combination[searched_key] for combination in valid_combinations)
        )
        spaces[searched_key] = replace_categorical_choices_for_label_budget(
            spaces[searched_key],
            retained,
        )
    else:
        for name in searched_keys:
            del spaces[name]
        spaces[JOINT_STML_IN_BATCH_HPARAM_KEY] = [
            serialize_joint_component_params(
                {name: combination[name] for name in searched_keys}
            )
            for combination in valid_combinations
        ]

    excluded_count = combination_count - len(valid_combinations)
    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} of {combination_count} in-batch STML sampler/graph HPO "
            f"combinations; retained {len(valid_combinations)} where the graph fits the batch "
            "and the unlabeled stream divides into whole neighbor groups."
        )
    return replace(config, spaces=spaces)


def drop_merged_inert_stml_spaces(config):
    """Remove dimensions the merged in-batch mode ignores.

    Unlike the two-term mode, nothing here needs a joint space: the graph takes
    the supervised batch whole, so ``batch_sampler`` is unconstrained, and the
    uniform unlabeled stream has no group size to divide by. What is left is a
    plain independent search over ``batch_sampler`` and
    ``graph_unlabeled_batch_size``.
    """

    removed = sorted(set(config.spaces) & set(STML_MERGED_INERT_HPARAM_KEYS))
    if not removed:
        return config
    spaces = {name: spec for name, spec in config.spaces.items() if name not in removed}
    logger.info(
        f"Excluded {removed} from HPO because graph_batch_mode='in_batch_merged' puts the whole "
        "supervised batch in the graph, draws the unlabeled rows uniformly, and runs a single "
        "weighted loss."
    )
    return replace(config, spaces=spaces)


def drop_in_batch_only_stml_spaces(config):
    """Remove graph-size spaces the pool-wide STML mode never reads.

    One study config can then cover both modes: the in-batch mode constrains
    these against the batch, and the pool-wide mode drops them instead of
    spending trials on dimensions that change nothing.
    """

    removed = sorted(set(config.spaces) & set(STML_IN_BATCH_ONLY_HPARAM_KEYS))
    if not removed:
        return config
    spaces = {name: spec for name, spec in config.spaces.items() if name not in removed}
    logger.info(
        f"Excluded {removed} from HPO because graph_batch_mode='global' builds one graph over "
        "the whole pool and has no per-step graph to size."
    )
    return replace(config, spaces=spaces)


def stml_in_batch_combination_is_valid(args, ssl_config, combination):
    """Check one combination against the regularizer's own run-time rules."""

    batch_size, sampler_m = parse_batch_sampler_choice(combination[BATCH_SAMPLER_HPARAM_KEY])
    trial_ssl_config = replace(
        ssl_config,
        graph_labeled_batch_size=combination["ssl_config.graph_labeled_batch_size"],
        graph_unlabeled_batch_size=combination["ssl_config.graph_unlabeled_batch_size"],
        method_params={
            **ssl_config.method_params,
            "n_neighbors": combination["ssl_config.method_params.n_neighbors"],
        },
    )
    trial_args = copy.copy(args)
    trial_args.batch_size = batch_size
    trial_args.sampler_m = sampler_m
    try:
        # Asking the regularizer itself keeps the search space and the run-time
        # check from drifting apart.
        method = semi_supervised.get_method(trial_ssl_config)
        method.make_regularizer(trial_ssl_config).validate_run_args(trial_args)
    except ValueError:
        return False
    return True


def validate_two_stream_hpo_method(ssl_config, hparam_key):
    """Reject studies whose SSL config can never build a two-stream sampler.

    The sampler mixes whole-pool pseudo-labels back into the supervised batch, so
    it needs a method that relabels the pool offline. An in-batch graph never
    produces that dataset: it labels and consumes each batch in the same step.
    """

    if ssl_config.method not in semi_supervised.TWO_STREAM_SAMPLER_METHODS:
        raise ValueError(
            f"{hparam_key!r} HPO requires an SSL config with one of the two-stream methods "
            f"{sorted(semi_supervised.TWO_STREAM_SAMPLER_METHODS)}"
        )
    if ssl_config.graph_batch_mode in semi_supervised.IN_BATCH_GRAPH_MODES:
        raise ValueError(
            f"{hparam_key!r} HPO sizes the whole-pool post-propagation sampler and cannot be "
            f"combined with graph_batch_mode={ssl_config.graph_batch_mode!r}; search "
            "graph_unlabeled_batch_size instead"
        )


def validate_two_stream_hpo_batch_combination(batch_size, sampler_m, labeled_batch_size):
    labeled_batch_size = parse_labeled_batch_size_choice(labeled_batch_size)
    batch_size = int(batch_size)
    sampler_m = int(sampler_m)
    if batch_size <= 0 or sampler_m <= 0:
        raise ValueError("Two-stream HPO batch_size and sampler_m must be positive")
    unlabeled_batch_size = batch_size - labeled_batch_size
    if labeled_batch_size > unlabeled_batch_size:
        raise ValueError(
            "Two-stream HPO requires labeled_batch_size <= unlabeled_batch_size: "
            f"labeled_batch_size={labeled_batch_size}, "
            f"unlabeled_batch_size={unlabeled_batch_size}, batch_size={batch_size}"
        )
    for stream_name, stream_batch_size in (
        ("labeled", labeled_batch_size),
        ("unlabeled", unlabeled_batch_size),
    ):
        if stream_batch_size % sampler_m != 0:
            raise ValueError(
                f"Two-stream HPO {stream_name} batch size must be divisible by sampler_m: "
                f"{stream_name}_batch_size={stream_batch_size}, sampler_m={sampler_m}"
            )
    return unlabeled_batch_size

def get_sampler_capacity_space(args, config, ssl_config):
    """Return HPO choices paired with their actual true-labeled stream sizes."""

    spaces = config.spaces
    joint_spec = spaces.get(JOINT_TWO_STREAM_HPARAM_KEY)
    if joint_spec is not None:
        choices = get_categorical_choices(joint_spec)
        capacity_choices = []
        for choice in choices:
            params = expand_joint_component_params({JOINT_TWO_STREAM_HPARAM_KEY: choice})
            batch_size, sampler_m = parse_batch_sampler_choice(params[BATCH_SAMPLER_HPARAM_KEY])
            labeled_key = next(iter(set(params) & LABELED_BATCH_SIZE_HPARAM_KEYS))
            labeled_batch_size = parse_labeled_batch_size_choice(params[labeled_key])
            capacity_choices.append((choice, labeled_batch_size, sampler_m))
        return JOINT_TWO_STREAM_HPARAM_KEY, joint_spec, capacity_choices

    in_batch_spec = spaces.get(JOINT_STML_IN_BATCH_HPARAM_KEY)
    if in_batch_spec is not None:
        # The in-batch mode draws its graph rows from the full supervised batch,
        # so batch_size is what the M-per-class sampler has to satisfy.
        capacity_choices = []
        for choice in get_categorical_choices(in_batch_spec):
            params = expand_joint_component_params({JOINT_STML_IN_BATCH_HPARAM_KEY: choice})
            batch_choice = params.get(
                BATCH_SAMPLER_HPARAM_KEY,
                f"{int(args.batch_size)}:{int(args.sampler_m)}",
            )
            batch_size, sampler_m = parse_batch_sampler_choice(batch_choice)
            capacity_choices.append((choice, batch_size, sampler_m))
        return JOINT_STML_IN_BATCH_HPARAM_KEY, in_batch_spec, capacity_choices

    fraction_spec = spaces.get(LABELED_BATCH_FRACTION_HPARAM_KEY)
    batch_sampler_spec = spaces.get(BATCH_SAMPLER_HPARAM_KEY)
    if fraction_spec is not None:
        if batch_sampler_spec is None:
            if not is_categorical_space(fraction_spec):
                # A continuous share has no choices to drop; the derived labeled
                # size is checked against the labeled data when the trial runs.
                return None
            capacity_choices = [
                (
                    choice,
                    derive_labeled_batch_size_from_fraction(
                        args.batch_size,
                        args.sampler_m,
                        choice,
                    ),
                    int(args.sampler_m),
                )
                for choice in get_categorical_choices(fraction_spec)
            ]
            return LABELED_BATCH_FRACTION_HPARAM_KEY, fraction_spec, capacity_choices

        # The labeled stream grows with the fraction, so the largest one decides
        # whether a sampler choice can be served for every fraction it may draw.
        largest_fraction = max(get_labeled_batch_fraction_values(fraction_spec))
        capacity_choices = []
        for choice in get_categorical_choices(batch_sampler_spec):
            batch_size, sampler_m = parse_batch_sampler_choice(choice)
            labeled_batch_size = derive_labeled_batch_size_from_fraction(
                batch_size,
                sampler_m,
                largest_fraction,
            )
            capacity_choices.append((choice, labeled_batch_size, sampler_m))
        return BATCH_SAMPLER_HPARAM_KEY, batch_sampler_spec, capacity_choices

    if batch_sampler_spec is not None:
        capacity_choices = []
        for choice in get_categorical_choices(batch_sampler_spec):
            batch_size, sampler_m = parse_batch_sampler_choice(choice)
            labeled_batch_size = (
                batch_size
                if ssl_config.labeled_batch_size is None or getattr(args, "mode", None) != "ssl"
                else int(ssl_config.labeled_batch_size)
            )
            capacity_choices.append((choice, labeled_batch_size, sampler_m))
        return BATCH_SAMPLER_HPARAM_KEY, batch_sampler_spec, capacity_choices

    labeled_keys = sorted(set(spaces) & LABELED_BATCH_SIZE_HPARAM_KEYS)
    if not labeled_keys:
        return None
    labeled_key = labeled_keys[0]
    labeled_spec = spaces[labeled_key]
    capacity_choices = [
        (choice, parse_labeled_batch_size_choice(choice), int(args.sampler_m))
        for choice in get_categorical_choices(labeled_spec)
    ]
    return labeled_key, labeled_spec, capacity_choices

def filter_sampler_capacity_choices_for_training_labels(capacity_choices, training_label_sets):
    valid_choices = []
    for choice, labeled_batch_size, sampler_m in capacity_choices:
        try:
            for labels in training_label_sets:
                utils.validate_m_per_class_sampler_capacity(
                    labels,
                    labeled_batch_size,
                    sampler_m,
                )
        except utils.MPerClassSamplerCapacityError:
            continue
        valid_choices.append(choice)
    return valid_choices

def make_label_budget_training_label_sets(args, ssl_config):
    """Reproduce the deterministic labeled training data used by each fold."""

    data_split_seed = get_data_split_seed(args)
    if args.val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        dataset_bundle = utils.setup_dataset_bundle(
            args.dataset,
            seed=args.seed,
            data_split_seed=data_split_seed,
            cv_k=1,
            cv_fold=None,
            cv_mode=args.cv_mode,
            val_mode=args.val_mode,
            dataset_protocol=args.dataset_protocol,
            cifar_imbalance_factor=args.cifar_imbalance_factor,
            cifar_train_fraction=args.cifar_train_fraction,
            cifar_test_fraction=args.cifar_test_fraction,
            image_resize_mode=get_image_resize_mode(args),
        )
        split = semi_supervised.prepare_label_split(dataset_bundle.train_dataset, ssl_config)
        return make_post_apportion_training_label_sets(
            args,
            dataset_bundle.train_dataset.labels,
            split.labeled_positions,
            original_labels=getattr(dataset_bundle.train_dataset, "orig_labels", None),
            support_seed=ssl_config.support_seed,
        )

    training_label_sets = []
    fold_indices = range(args.cv_k) if args.cv_k > 1 else [None]
    for fold_index in fold_indices:
        dataset_bundle = utils.setup_dataset_bundle(
            args.dataset,
            seed=args.seed,
            data_split_seed=data_split_seed,
            cv_k=args.cv_k if fold_index is not None else 1,
            cv_fold=fold_index,
            cv_mode=args.cv_mode,
            val_mode=args.val_mode,
            dataset_protocol=args.dataset_protocol,
            cifar_imbalance_factor=args.cifar_imbalance_factor,
            cifar_train_fraction=args.cifar_train_fraction,
            cifar_test_fraction=args.cifar_test_fraction,
            image_resize_mode=get_image_resize_mode(args),
        )
        split = semi_supervised.prepare_label_split(dataset_bundle.train_dataset, ssl_config)
        labels = np.asarray(dataset_bundle.train_dataset.labels, dtype=np.int64)
        training_label_sets.append(labels[np.asarray(split.labeled_positions, dtype=np.int64)])
    return training_label_sets

def make_post_apportion_training_label_sets(args, labels, labeled_positions, original_labels=None, support_seed=None):
    """Return labeled training labels after each post-apportion validation split."""

    labels = np.asarray(labels, dtype=np.int64)
    labeled_positions = np.asarray(labeled_positions, dtype=np.int64)
    split_seed = get_support_seed(args) if support_seed is None else int(support_seed)
    if args.cv_mode in utils.SUPERCLASS_AWARE_CV_MODES:
        if original_labels is None:
            raise ValueError(
                f"{args.cv_mode} requires original CIFAR-100 "
                "fine labels for post-apportion CV"
            )
        superclass_labels = utils.cifar100_superclass_labels_for_fine_labels(original_labels)
    else:
        superclass_labels = None
    if args.cv_k > 1:
        training_label_sets = []
        for fold_index in range(args.cv_k):
            train_positions, _ = utils.split_positions_cross_validation(
                positions=labeled_positions,
                labels=labels,
                cv_k=args.cv_k,
                cv_fold=fold_index,
                cv_mode=args.cv_mode,
                seed=split_seed,
                superclass_labels=superclass_labels,
            )
            training_label_sets.append(labels[train_positions])
        return training_label_sets

    train_positions, _ = utils.split_positions_class_disjoint_by_label(
        positions=labeled_positions,
        labels=labels,
        val_ratio=utils.POST_APPORTION_VAL_RATIO,
        seed=split_seed,
    )
    return [labels[train_positions]]

def filter_batch_sampler_choices_for_training_labels(choices, training_label_sets):
    valid_choices = []
    for choice in choices:
        batch_size, sampler_m = parse_batch_sampler_choice(choice)
        try:
            for labels in training_label_sets:
                utils.validate_m_per_class_sampler_capacity(labels, batch_size, sampler_m)
        except utils.MPerClassSamplerCapacityError:
            continue
        valid_choices.append(choice)
    return valid_choices

def summarize_training_label_sets(training_label_sets):
    summaries = []
    for labels in training_label_sets:
        counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)[1]
        summaries.append(
            {
                "samples": int(np.sum(counts)),
                "classes": int(len(counts)),
                "min_samples_per_class": int(np.min(counts)),
            }
        )
    return summaries

def replace_categorical_choices_for_label_budget(spec, choices):
    if isinstance(spec, list):
        return choices
    constrained_spec = dict(spec)
    constrained_spec["choices"] = choices
    return constrained_spec

def replace_categorical_choices(name, spec, choices, k):
    if not choices:
        raise ValueError(
            f"No valid {name} choices remain for class_subset_k_shot k={k}; "
            "sampler_m must be less than or equal to k."
        )
    if isinstance(spec, list):
        return choices
    constrained_spec = dict(spec)
    constrained_spec["choices"] = choices
    return constrained_spec

def make_component_spaces_constraint_aware(args, config):
    """Collapse invalid categorical constructor combinations into valid joint spaces."""

    spaces = dict(config.spaces)
    for component in ("loss", "miner"):
        component_spaces = {
            name: spec
            for name, spec in spaces.items()
            if is_component_override(name)
            and name.startswith(f"{component}.")
            and is_categorical_space(spec)
        }
        if not component_spaces:
            continue

        names = sorted(component_spaces)
        choices_by_name = [get_categorical_choices(component_spaces[name]) for name in names]
        valid_combinations = []
        invalid_combinations = []
        for values in itertools.product(*choices_by_name):
            combination = dict(zip(names, values))
            error = validate_component_combination(args, component, combination)
            if error is None:
                valid_combinations.append(combination)
            else:
                invalid_combinations.append((combination, error))

        needs_joint_space = bool(invalid_combinations) or component_choices_require_joint_space(choices_by_name)
        if not needs_joint_space:
            continue
        if not valid_combinations:
            example_error = invalid_combinations[0][1]
            raise ValueError(
                f"No valid categorical {component} hyperparameter combinations remain for "
                f"{getattr(args, component)}. Example constructor error: {example_error}"
            )

        for name in names:
            del spaces[name]
        joint_name = f"{JOINT_COMPONENT_HPARAM_PREFIX}{component}.{getattr(args, component)}"
        spaces[joint_name] = [serialize_joint_component_params(params) for params in valid_combinations]
        logger.info(
            f"Collapsed {len(valid_combinations) + len(invalid_combinations)} categorical {component} "
            f"combinations for {getattr(args, component)} into {len(valid_combinations)} valid combinations; "
            f"excluded {len(invalid_combinations)} constructor-invalid combinations."
        )

    return replace(config, spaces=spaces)

def component_choices_require_joint_space(choices_by_name):
    """Optuna categorical distributions are scalar; joint JSON strings carry complex values."""

    return any(not is_scalar(choice) for choices in choices_by_name for choice in choices)

def is_categorical_space(spec):
    return isinstance(spec, list) or (
        isinstance(spec, dict)
        and spec.get("type", "categorical" if "choices" in spec else None) == "categorical"
    )

def get_categorical_choices(spec):
    return spec if isinstance(spec, list) else spec["choices"]

def validate_component_combination(args, component, combination):
    candidate_args = copy.deepcopy(args)
    for name, value in combination.items():
        set_component_param(candidate_args, name, value)
    try:
        if component == "loss":
            loss_class = get_loss_class(candidate_args.loss)
            params = dict(getattr(candidate_args, "loss_params", {}))
            if candidate_args.loss in CLASSIFICATION_LOSSES:
                loss_class(2, candidate_args.feat_dim or 128, **params)
            else:
                loss_class(**params)
        else:
            params = validate_named_miner_params(
                candidate_args.miner,
                dict(getattr(candidate_args, "miner_params", {})),
            )
            getattr(miners, candidate_args.miner)(**params)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None

def serialize_joint_component_params(params):
    return json.dumps(to_jsonable(params), sort_keys=True, separators=(",", ":"))

def expand_joint_component_params(params):
    expanded = {}
    for name, value in params.items():
        if not name.startswith((JOINT_COMPONENT_HPARAM_PREFIX, JOINT_HPARAM_PREFIX)):
            expanded[name] = value
            continue
        joint_params = json.loads(value)
        duplicate_names = sorted(set(expanded) & set(joint_params))
        if duplicate_names:
            raise ValueError(f"Joint component hyperparameters duplicate existing values: {duplicate_names}")
        expanded.update(joint_params)
    return expanded

def component_override_applies(args, name):
    if not is_component_override(name):
        return True
    parts = name.split(".")
    component = parts[0]
    if component == "miner" and (
        getattr(args, "miner") == "no_miner"
        or getattr(args, "loss") in CLASSIFICATION_LOSSES
        or getattr(args, "loss") == "STMLLoss"
    ):
        return False
    if len(parts) == 2:
        return True
    _, class_name, _ = parts
    return class_name == getattr(args, component)

def set_component_param(args, name, value):
    """Apply one loss/miner constructor kwarg to the selected component."""

    if not component_override_applies(args, name):
        return
    parts = name.split(".")
    component = parts[0]
    parameter = parts[-1]
    params_attr = f"{component}_params"
    params = dict(getattr(args, params_attr, {}))
    if parameter in params:
        raise ValueError(f"Duplicate {component} parameter in hyperparameter space: {parameter!r}")
    params[parameter] = value
    setattr(args, params_attr, params)

def set_arg_value(args, name, value):
    if name == BATCH_SAMPLER_HPARAM_KEY:
        args.batch_size, args.sampler_m = parse_batch_sampler_choice(value)
        return
    if not hasattr(args, name):
        raise ValueError(f"Unknown training argument in hyperparameter space: {name}")
    if name in {"ssl_config", "hparam_config", "save_dir"} and value is not None:
        value = Path(value)
    setattr(args, name, value)

def set_nested_value(config, path_parts, value):
    if not path_parts:
        raise ValueError("SSL override must include a nested config key, for example ssl_config.method_params.n_neighbors")
    current = config
    for part in path_parts[:-1]:
        # Create missing intermediate dictionaries so HPO may introduce a new
        # method_params key that was absent from the base JSON.
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested SSL config path: {'.'.join(path_parts)}")
        if part not in current:
            current[part] = {}
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"Cannot set nested SSL config path: {'.'.join(path_parts)}")
    current[path_parts[-1]] = value

def get_objective_value(result, metric):
    value = getattr(result, metric)
    if value is None:
        raise ValueError(f"Objective metric {metric!r} is None; choose a metric available for this run")
    return float(value)

def write_trials_summary(study, csv_path, jsonl_path):
    trials = list(study.trials)
    expanded_params = [expand_joint_component_params(dict(trial.params)) for trial in trials]
    param_names = sorted({name for params in expanded_params for name in params})
    scalar_attr_names = sorted(
        {
            name
            for trial in trials
            for name, value in trial.user_attrs.items()
            if is_scalar(value) and name not in {"params"}
        }
    )
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        *[f"param:{name}" for name in param_names],
        *[f"attr:{name}" for name in scalar_attr_names],
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for trial, params in zip(trials, expanded_params):
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "datetime_start": "" if trial.datetime_start is None else trial.datetime_start.isoformat(),
                "datetime_complete": "" if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
                "duration_seconds": "" if trial.duration is None else trial.duration.total_seconds(),
            }
            for name in param_names:
                row[f"param:{name}"] = json.dumps(to_jsonable(params.get(name)))
            for name in scalar_attr_names:
                row[f"attr:{name}"] = json.dumps(to_jsonable(trial.user_attrs.get(name)))
            writer.writerow(row)

    with jsonl_path.open("w") as jsonl_file:
        for trial in trials:
            jsonl_file.write(json.dumps(serialize_trial(trial), default=str) + "\n")

def serialize_trial(trial):
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": to_jsonable(expand_joint_component_params(dict(trial.params))),
        "user_attrs": to_jsonable(trial.user_attrs),
        "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
        "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
        "duration_seconds": None if trial.duration is None else trial.duration.total_seconds(),
    }
