"""Optuna configuration, study execution, and search-space constraints."""

import copy
import csv
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytorch_metric_learning.miners as miners
from loguru import logger

import semi_supervised
import utils
from experiment_cli import (
    FINAL_TEST_VISUALIZATION_NONE,
    get_hparam_seed,
    get_support_seed,
    normalize_backbone_tuning_args,
)
from experiment_io import is_scalar, namespace_to_dict, result_to_dict, to_jsonable, write_json
from experiment_training import (
    get_loss_class,
    get_data_split_seed,
    resolve_loss_driven_supervised_args,
    resolve_mode_ssl_config,
    run_experiment,
    validate_named_miner_params,
    validate_run_args,
)
from experiment_types import (
    ALL_LOSSES,
    ALL_MINERS,
    BATCH_SAMPLER_HPARAM_KEY,
    CLASSIFICATION_LOSSES,
    HPO_MODE_KEYS,
    HParamSearchConfig,
    HParamStudyResult,
    JOINT_COMPONENT_HPARAM_PREFIX,
    LOSS_HPARAM_PREFIX,
    MINER_HPARAM_PREFIX,
    OBJECTIVE_METRICS,
    SAMPLER_CAPACITY_HPARAM_KEYS,
    SELECTION_METRICS,
)
from retrieval_model import BACKBONE_TUNING_FROZEN

def load_hparam_config(config_path):
    """Load and validate an Optuna JSON config, or return ``None``."""

    if config_path is None:
        return None

    path = Path(config_path)
    with path.open() as config_file:
        raw_config = json.load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Hyperparameter config must be a JSON object: {path}")

    allowed_keys = set(HParamSearchConfig.__dataclass_fields__)
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown hyperparameter config keys in {path}: {unknown_keys}")

    config = HParamSearchConfig(**raw_config)
    validate_hparam_config(config, path)
    return config

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
    if not isinstance(config.spaces, dict) or not config.spaces:
        raise ValueError(f"spaces must be a non-empty object{source}")
    if BATCH_SAMPLER_HPARAM_KEY in config.spaces and {"batch_size", "sampler_m"} & set(config.spaces):
        raise ValueError(
            f"Search space {BATCH_SAMPLER_HPARAM_KEY!r} sets both batch_size and sampler_m. "
            f"Do not also include 'batch_size' or 'sampler_m'{source}."
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
        if name == "selection_metric":
            validate_selection_metric_choices(choices, source)
    elif space_type in {"float", "int"}:
        if name == BATCH_SAMPLER_HPARAM_KEY:
            raise ValueError(f"Search space {name!r} must be categorical choices like '32:8'{source}")
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

def run_hparam_search(args, config):
    """Create or resume an Optuna study and execute its remaining trials."""

    args = resolve_loss_driven_supervised_args(args)
    config = filter_component_hparam_config(args, config)
    config = filter_loss_dependent_hparam_config(args, config)
    config = make_backbone_tuning_spaces_aware(args, config)
    config = make_sampler_spaces_k_shot_aware(args, config)
    config = make_sampler_spaces_label_budget_aware(args, config)
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
    resolved_tpe_startup_trials = resolve_tpe_startup_trials(args, config)
    write_json(
        study_dir / "study_config.json",
        {
            "base_args": namespace_to_dict(args),
            "hparam_config": config.to_dict(),
            "resolved_tpe_startup_trials": resolved_tpe_startup_trials,
            "resolved_study_name": study_name,
            "resolved_storage": storage,
        },
    )

    # The sampler proposes values; the pruner can stop weak trials based on
    # intermediate reports from epochs or CV folds.
    sampler = make_optuna_sampler(
        optuna,
        config,
        get_hparam_seed(args),
        tpe_startup_trials=resolved_tpe_startup_trials,
    )
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
    trials_csv = study_dir / "trials.csv"
    trials_jsonl = study_dir / "trials.jsonl"

    def objective(trial):
        # Resolve trial suggestions into a fresh argparse namespace and SSL
        # config so trials cannot mutate one another's settings.
        trial_args, ssl_config, suggested_params = make_trial_args_and_ssl_config(args, config, trial)
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
            )
        except utils.MPerClassSamplerCapacityError as exc:
            # Some sampled batch/sampler/label-budget combinations cannot form
            # one M-per-class batch. Prune those combinations without aborting
            # the entire study.
            trial.set_user_attr("pruned_reason", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        except utils.NonFiniteEmbeddingError as exc:
            # Divergent hyperparameters can produce NaN/Inf embeddings during
            # validation. Treat them as invalid trials instead of aborting HPO.
            trial.set_user_attr("pruned_reason", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        # Store all artifacts and metrics on the trial, then return only the
        # configured scalar objective to Optuna.
        result_dict = result_to_dict(result)
        for key, value in result_dict.items():
            trial.set_user_attr(key, value)
        return get_objective_value(result, config.metric)

    def record_trial(study, trial):
        # Refresh summaries after each finished trial so interrupted studies
        # still leave readable progress outside the Optuna database.
        write_trials_summary(study, trials_csv, trials_jsonl)

    # Finished trials count toward n_trials when a persistent study is resumed.
    finished_trials = count_finished_trials(optuna, study)
    remaining_trials = config.n_trials - finished_trials
    logger.info(
        f"Starting Optuna study outputs in {study_dir}. "
        f"Finished trials: {finished_trials}/{config.n_trials}. "
        f"Recovered unfinished trials: {recovered_trials}. "
        f"Remaining this run: {max(remaining_trials, 0)}. "
        f"Parallel jobs: {config.n_jobs}."
    )
    if remaining_trials <= 0:
        write_trials_summary(study, trials_csv, trials_jsonl)
        logger.info(f"Optuna study already has {finished_trials} finished trials; no new trials requested.")
        return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

    study.optimize(
        objective,
        n_trials=remaining_trials,
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

def count_finished_trials(optuna, study):
    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    return sum(trial.state in finished_states for trial in study.trials)

def recover_unfinished_trials(optuna, study):
    # A killed process can leave persistent trials marked RUNNING forever.
    # Reset them to WAITING so Optuna can schedule the same trials again.
    recovered = 0
    running_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in running_trials:
        try:
            if study._storage.set_trial_state_values(trial._trial_id, state=optuna.trial.TrialState.WAITING):
                recovered += 1
        except optuna.exceptions.UpdateFinishedTrialError:
            continue

    if recovered:
        logger.warning(
            f"Recovered {recovered} unfinished Optuna trial(s) in study {study.study_name!r}. "
            "They were reset from RUNNING to WAITING and will be rerun with the same trial numbers and parameters."
        )
    return recovered

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
            distributions[name] = optuna.distributions.IntDistribution(
                low=spec["low"],
                high=spec["high"],
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
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

    trial_args = copy.deepcopy(base_args)
    ssl_overrides = []

    # Names beginning with ssl./ssl_config. target the nested dataclass;
    # everything else is a direct command-line argument override.
    for name, value in params.items():
        if is_ssl_override(name):
            ssl_overrides.append((name, value))
        elif is_component_override(name):
            set_component_param(trial_args, name, value)
        else:
            set_arg_value(trial_args, name, value)

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
            path_parts = name.split(".")[1:]
            set_nested_value(ssl_dict, path_parts, value)
        ssl_config = semi_supervised.SemiSupervisedConfig(**ssl_dict)
        semi_supervised.validate_ssl_config(ssl_config)

    ssl_config = resolve_mode_ssl_config(trial_args, ssl_config)
    trial_args = resolve_loss_driven_supervised_args(trial_args)

    validate_run_args(trial_args, ssl_config)
    return trial_args, ssl_config

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
    return name.startswith("ssl_config.") or name.startswith("ssl.")

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

    if getattr(args, "loss", None) in CLASSIFICATION_LOSSES or "classifier_lr" not in config.spaces:
        return config

    spaces = dict(config.spaces)
    del spaces["classifier_lr"]
    logger.info(
        f"Excluded classifier_lr from HPO because loss {getattr(args, 'loss', None)!r} "
        "does not use a classifier optimizer."
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
    """Remove joint batch/sampler choices infeasible for the fixed label split."""

    batch_sampler_spec = config.spaces.get(BATCH_SAMPLER_HPARAM_KEY)
    if batch_sampler_spec is None:
        return config

    varying_split_keys = sorted(set(config.spaces) & SAMPLER_CAPACITY_HPARAM_KEYS)
    if varying_split_keys:
        logger.info(
            "Cannot prefilter batch_sampler choices because these HPO dimensions change "
            f"the labeled training split: {varying_split_keys}"
        )
        return config

    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    if ssl_config.label_sampling_mode not in {"class_subset", "class_subset_k_shot"}:
        return config

    if training_label_sets_factory is None:
        training_label_sets_factory = make_label_budget_training_label_sets
    training_label_sets = training_label_sets_factory(args, ssl_config)
    choices = get_categorical_choices(batch_sampler_spec)
    valid_choices = filter_batch_sampler_choices_for_training_labels(choices, training_label_sets)
    excluded_count = len(choices) - len(valid_choices)
    if not valid_choices:
        fold_summaries = summarize_training_label_sets(training_label_sets)
        raise ValueError(
            "No valid batch_sampler choices remain for the selected label budget and validation splits. "
            f"Fold labeled-data summaries: {fold_summaries}"
        )

    if excluded_count:
        logger.info(
            f"Excluded {excluded_count} batch_sampler hyperparameter choices that cannot form "
            "an MPerClassSampler batch in every labeled training split. "
            f"Remaining choices: {valid_choices}. "
            f"Fold labeled-data summaries: {summarize_training_label_sets(training_label_sets)}"
        )

    spaces = dict(config.spaces)
    spaces[BATCH_SAMPLER_HPARAM_KEY] = replace_categorical_choices_for_label_budget(
        batch_sampler_spec,
        valid_choices,
    )
    return replace(config, spaces=spaces)

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
    if args.cv_mode == utils.CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD:
        if original_labels is None:
            raise ValueError(
                f"{utils.CV_MODE_SUPERCLASS_BALANCED_GROUP_KFOLD} requires original CIFAR-100 "
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
        if not name.startswith(JOINT_COMPONENT_HPARAM_PREFIX):
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
