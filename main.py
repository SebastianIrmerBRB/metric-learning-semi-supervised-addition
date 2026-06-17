# ruff: noqa: F401,F403,F405
"""Experiment orchestration and executable entry point.

Training, CLI parsing, shared types, and HPO implementation live in focused
modules. Imports below preserve the historical ``main`` module API.
"""


import math
import multiprocessing as mp

import experiment_hpo as _experiment_hpo
from experiment_cli import *  # noqa: F403
from experiment_hpo import *  # noqa: F403
from experiment_io import *  # noqa: F403
from experiment_training import *  # noqa: F403
from experiment_types import *  # noqa: F403


def make_sampler_spaces_label_budget_aware(args, config):
    """Preserve the historical patch point while delegating HPO constraints."""

    return _experiment_hpo.make_sampler_spaces_label_budget_aware(
        args,
        config,
        training_label_sets_factory=make_label_budget_training_label_sets,
    )


def main():
    """Dispatch the CLI request from the broadest orchestration mode inward."""

    # Experiment-config values replace parser defaults, while explicit CLI
    # arguments remain the highest-precedence source.
    args = parse_args_with_experiment_config()

    # A missing --hparam_config produces None.  A present but disabled config is
    # still loaded so its resolved contents can be written into run metadata.
    hparam_config = load_hparam_config(args.hparam_config)
    # Comparison and grid modes own their own HPO/training loops, so they must
    # be handled before the standalone HPO and single-run paths.
    if args.compare_supervised_ssl:
        run_supervised_ssl_comparison(args, hparam_config)
        return

    if has_outer_comparison_grid(args):
        # An outer grid varies experimental conditions such as label budget or
        # seed.  Each grid point may itself contain an Optuna study.
        run_single_method_grid(args, hparam_config)
        return

    if hparam_config is not None and hparam_config.enabled:
        # In supervised mode, remove SSL-only HPO dimensions while preserving
        # settings that define which examples belong to the labeled split.
        hparam_config = make_standalone_hparam_config(args, hparam_config)
        study_result = run_hparam_search(args, hparam_config)
        if args.final_test_after_hpo:
            run_final_from_best_hparam(
                args,
                hparam_config,
                study_result,
                role=args.mode,
            )
        return

    if hparam_config is not None:
        # Store disabled HPO configuration for provenance even though this run
        # will use the command-line/default parameters directly.
        args.hparam_config_resolved = hparam_config.to_dict()

    # The mode may turn an enabled SSL config into a split-only supervised
    # config.  This keeps label selection identical between comparison methods.
    ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    ssl_config = resolve_mode_ssl_config(args, ssl_config)
    run_experiment(args, ssl_config)

def run_supervised_ssl_comparison(args, hparam_config):
    """Run matched supervised and SSL studies for every outer-grid scenario."""

    # This base config defines both the SSL algorithm and the common label
    # apportioning rules used by the supervised baseline.
    base_ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    validate_comparison_setup(args, hparam_config, base_ssl_config)

    # Scenarios are concrete combinations of label budget, sampling mode,
    # loss/miner, and eval seed.  Their SSL configs are already written to disk.
    scenarios = make_comparison_scenarios(args, base_ssl_config)
    grid_results = []
    for scenario_group in group_scenarios_by_frozen_config(scenarios):
        grid_results.extend(
            run_supervised_ssl_frozen_hparam_group(
                args=args,
                hparam_config=hparam_config,
                base_ssl_config=base_ssl_config,
                scenario_group=scenario_group,
            )
        )

    if len(scenarios) > 1 or has_outer_comparison_grid(args):
        # The grid summary places all scenario-level metrics and deltas in one
        # CSV/JSON collection after every scenario has finished.
        write_comparison_grid_summary(Path("logs") / args.save_dir / "comparison_grid", grid_results)

def run_supervised_ssl_frozen_hparam_group(args, hparam_config, base_ssl_config, scenario_group):
    """Tune supervised and SSL once, then evaluate both with frozen params per seed."""

    reference_scenario = scenario_group[0]
    group_name, reference_ssl_config_path, reference_ssl_config = write_reference_ssl_config(
        args=args,
        base_ssl_config=base_ssl_config,
        scenario=reference_scenario,
        grid_dir_name="comparison_grid",
    )
    reference_args = make_reference_hpo_args(
        args=args,
        group_name=group_name,
        reference_ssl_config_path=reference_ssl_config_path,
        scenario=reference_scenario,
    )
    comparison_dir = Path("logs") / reference_args.save_dir / "supervised_ssl_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    supervised_args = copy.deepcopy(reference_args)
    supervised_args.mode = "supervised"
    supervised_args.skip_test_during_hpo = True

    ssl_args = copy.deepcopy(reference_args)
    ssl_args.mode = "ssl"
    ssl_args.skip_test_during_hpo = True

    supervised_hparam_config = make_comparison_hparam_config(hparam_config, role="supervised")
    ssl_hparam_config = make_comparison_hparam_config(hparam_config, role="ssl")

    write_json(
        comparison_dir / "comparison_setup.json",
        {
            "methodology": (
                "supervised and SSL tune once on a reference support draw; "
                "the selected hyperparameters are frozen and evaluated on each comparison seed. "
                "Both methods use the same fixed D_val, D_test, support draws, objective metric, and HPO budget."
            ),
            "base_args": namespace_to_dict(reference_args),
            "eval_seeds": [scenario.seed for scenario in scenario_group],
            "reference_group": group_name,
            "reference_ssl_config": reference_ssl_config.to_dict(),
            "supervised_hparam_config": supervised_hparam_config.to_dict(),
            "ssl_hparam_config": ssl_hparam_config.to_dict(),
        },
    )

    supervised_study = run_hparam_search(supervised_args, supervised_hparam_config)
    ssl_study = run_hparam_search(ssl_args, ssl_hparam_config)

    group_results = []
    for scenario in scenario_group:
        eval_args = make_args_for_scenario(args, scenario)
        scenario_ssl_config = semi_supervised.load_ssl_config(scenario.ssl_config_path, default_seed=scenario.seed)

        supervised_eval_args = copy.deepcopy(eval_args)
        supervised_eval_args.mode = "supervised"
        supervised_final = run_final_from_best_hparam(
            supervised_eval_args,
            supervised_hparam_config,
            supervised_study,
            role="supervised",
            summary_stem=f"final_evaluation_{scenario.name}_supervised",
        )

        ssl_eval_args = copy.deepcopy(eval_args)
        ssl_eval_args.mode = "ssl"
        ssl_final = run_final_from_best_hparam(
            ssl_eval_args,
            ssl_hparam_config,
            ssl_study,
            role="ssl",
            summary_stem=f"final_evaluation_{scenario.name}_ssl",
        )

        scenario_comparison_dir = Path("logs") / eval_args.save_dir / "supervised_ssl_comparison"
        scenario_comparison_dir.mkdir(parents=True, exist_ok=True)
        write_comparison_summary(
            comparison_dir=scenario_comparison_dir,
            args=eval_args,
            scenario=scenario,
            ssl_config=scenario_ssl_config,
            supervised_study=supervised_study,
            ssl_study=ssl_study,
            supervised_final=supervised_final,
            ssl_final=ssl_final,
        )
        group_results.append(
            {
                "scenario": scenario,
                "comparison_dir": scenario_comparison_dir,
                "ssl_config": scenario_ssl_config,
                "supervised_study": supervised_study,
                "ssl_study": ssl_study,
                "supervised_final": supervised_final,
                "ssl_final": ssl_final,
                "deltas": make_comparison_deltas(supervised_final, ssl_final),
            }
        )
    return group_results

def run_single_supervised_ssl_comparison(args, hparam_config, ssl_config, scenario):
    """Tune and retrain both methods while keeping their data split fixed."""

    comparison_dir = Path("logs") / args.save_dir / "supervised_ssl_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # The two namespaces begin with identical data, model, and HPO settings.
    # Only mode differs; resolve_mode_ssl_config later disables pseudo-labeling
    # for the supervised branch while retaining its label split.
    supervised_args = copy.deepcopy(args)
    supervised_args.mode = "supervised"
    supervised_args.skip_test_during_hpo = True

    ssl_args = copy.deepcopy(args)
    ssl_args.mode = "ssl"
    ssl_args.skip_test_during_hpo = True

    # SSL-specific search dimensions are removed from the supervised study.
    # Both studies otherwise retain the same trial budget and common spaces.
    supervised_hparam_config = make_comparison_hparam_config(hparam_config, role="supervised")
    ssl_hparam_config = make_comparison_hparam_config(hparam_config, role="ssl")

    # Write methodology and resolved inputs before doing expensive work.  This
    # leaves an audit trail even if a later study is interrupted.
    write_json(
        comparison_dir / "comparison_setup.json",
        {
            "methodology": (
                "supervised uses only D_train; SSL uses D_train + D_UL without D_UL labels; "
                "both use the same D_val, D_test, objective metric, and HPO budget"
            ),
            "base_args": namespace_to_dict(args),
            "scenario": comparison_scenario_to_dict(scenario),
            "split_ssl_config": ssl_config.to_dict(),
            "supervised_hparam_config": supervised_hparam_config.to_dict(),
            "ssl_hparam_config": ssl_hparam_config.to_dict(),
        },
    )

    # Each method gets an independent study with the same HPO budget.  Test
    # evaluation is disabled during HPO so it cannot influence model selection.
    supervised_study = run_hparam_search(supervised_args, supervised_hparam_config)
    ssl_study = run_hparam_search(ssl_args, ssl_hparam_config)

    # Retraining separates parameter selection from the final reported model.
    # The best parameters are applied to fresh runs after each study finishes.
    supervised_final = run_final_from_best_hparam(
        supervised_args,
        supervised_hparam_config,
        supervised_study,
        role="supervised",
    )
    ssl_final = run_final_from_best_hparam(
        ssl_args,
        ssl_hparam_config,
        ssl_study,
        role="ssl",
    )
    write_comparison_summary(
        comparison_dir=comparison_dir,
        args=args,
        scenario=scenario,
        ssl_config=ssl_config,
        supervised_study=supervised_study,
        ssl_study=ssl_study,
        supervised_final=supervised_final,
        ssl_final=ssl_final,
    )
    # Return rich Python objects for the outer grid, which later flattens the
    # relevant values into its aggregate CSV/JSON reports.
    return {
        "scenario": scenario,
        "comparison_dir": comparison_dir,
        "ssl_config": ssl_config,
        "supervised_study": supervised_study,
        "ssl_study": ssl_study,
        "supervised_final": supervised_final,
        "ssl_final": ssl_final,
        "deltas": make_comparison_deltas(supervised_final, ssl_final),
    }

def make_comparison_scenarios(args, base_ssl_config, grid_dir_name="comparison_grid"):
    """Expand CLI grid dimensions into concrete, reproducible SSL configs."""

    # None means "do not vary this dimension"; an explicitly empty CLI list is
    # considered an error because it would silently produce zero experiments.
    label_budgets = args.label_budget_grid
    has_label_budget_grid = label_budgets is not None
    if label_budgets is None:
        label_budgets = [base_ssl_config.labeled_fraction]
    else:
        if not label_budgets:
            raise ValueError("--label_budget_grid must include at least one value when provided")

    seeds = args.comparison_seeds
    if seeds is None:
        seeds = [base_ssl_config.seed]
    elif not seeds:
        raise ValueError("--comparison_seeds must include at least one value when provided")

    label_sampling_modes = get_effective_label_sampling_modes(args, base_ssl_config)
    validate_k_shot_grid_usage(args, label_sampling_modes)

    # A loss and miner are treated as a pair because many losses require a
    # compatible mining strategy (and classification losses ignore miners).
    loss_miner_pairs = get_loss_miner_pairs(args)
    include_loss_miner_in_name = args.loss_miner_grid is not None

    config_dir = Path("logs") / args.save_dir / grid_dir_name / "ssl_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    # Names become directory/config filenames, so duplicates would overwrite
    # outputs and make two distinct requests indistinguishable.
    scenario_names = set()
    # The nested loops form the Cartesian product of all requested outer-grid
    # dimensions.  Each generated config is persisted before it is executed.
    for label_sampling_mode in label_sampling_modes:
        for label_budget in label_budgets:
            labeled_fraction, default_labeled_per_class = resolve_scenario_label_budget(
                label_sampling_mode=label_sampling_mode,
                label_budget=label_budget,
                has_label_budget_grid=has_label_budget_grid,
                base_ssl_config=base_ssl_config,
            )
            for scenario_labeled_per_class in get_k_shot_values(
                label_sampling_mode=label_sampling_mode,
                default_labeled_per_class=default_labeled_per_class,
                args=args,
            ):
                for loss_name, miner_name in loss_miner_pairs:
                    for seed in seeds:
                        # dataclasses.replace creates a new immutable config,
                        # leaving the shared base config untouched.
                        scenario_ssl_config = replace(
                            base_ssl_config,
                            label_sampling_mode=label_sampling_mode,
                            labeled_fraction=float(labeled_fraction),
                            labeled_per_class=scenario_labeled_per_class,
                            seed=int(seed),
                        )
                        semi_supervised.validate_ssl_config(scenario_ssl_config)
                        # Include only dimensions needed to distinguish the
                        # requested scenarios; this keeps paths readable.
                        scenario_name = make_scenario_name(
                            scenario_ssl_config,
                            loss=loss_name if include_loss_miner_in_name else None,
                            miner=miner_name if include_loss_miner_in_name else None,
                        )
                        if scenario_name in scenario_names:
                            raise ValueError(
                                f"Duplicate outer-grid scenario name {scenario_name!r}. "
                                "Check for duplicate label budgets, k-shot counts, seeds, "
                                "label sampling modes, or loss/miner pairs."
                            )
                        scenario_names.add(scenario_name)
                        config_path = config_dir / f"{scenario_name}.json"
                        # Persist each expanded config so the scenario can be
                        # rerun independently without reconstructing the grid.
                        write_json(config_path, scenario_ssl_config.to_dict())
                        scenarios.append(
                            ComparisonScenario(
                                name=scenario_name,
                                labeled_fraction=scenario_ssl_config.labeled_fraction,
                                labeled_per_class=scenario_ssl_config.labeled_per_class,
                                seed=scenario_ssl_config.seed,
                                label_sampling_mode=scenario_ssl_config.label_sampling_mode,
                                loss=loss_name,
                                miner=miner_name,
                                ssl_config_path=config_path,
                            )
                        )
    return scenarios

def resolve_scenario_label_budget(label_sampling_mode, label_budget, has_label_budget_grid, base_ssl_config):
    # labeled_fraction has different semantics by mode: it can mean a fraction
    # of samples or a fraction of classes.  It is always constrained to (0, 1].
    labeled_fraction = float(label_budget)
    validate_labeled_fraction(labeled_fraction, "label budget")
    if label_sampling_mode == "class_subset_k_shot":
        # k controls examples per selected class; default to one-shot if the
        # base config omitted it.
        labeled_per_class = base_ssl_config.labeled_per_class or 1
    else:
        # When a fraction grid is explicitly supplied, do not let a fixed
        # per-class count override that grid dimension.
        labeled_per_class = None if has_label_budget_grid else base_ssl_config.labeled_per_class
    return labeled_fraction, labeled_per_class

def validate_labeled_fraction(value, name):
    if not math.isfinite(value) or not (0 < value <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {value}")

def validate_k_shot_grid_usage(args, label_sampling_modes):
    k_shot_grid = getattr(args, "k_shot_grid", None)
    if k_shot_grid is None:
        return
    if not k_shot_grid:
        raise ValueError("--k_shot_grid must include at least one positive integer when provided")
    if "class_subset_k_shot" not in label_sampling_modes:
        raise ValueError("--k_shot_grid is only valid when --ssl_label_sampling_modes includes class_subset_k_shot")

def get_k_shot_values(label_sampling_mode, default_labeled_per_class, args):
    # A k-shot grid is meaningful only for the mode that selects both a class
    # subset and a fixed number of examples within each selected class.
    if label_sampling_mode != "class_subset_k_shot":
        return [default_labeled_per_class]
    k_shot_grid = getattr(args, "k_shot_grid", None)
    if k_shot_grid is None:
        return [default_labeled_per_class]
    return [validate_k_shot_value(value) for value in k_shot_grid]

def validate_k_shot_value(value):
    if value <= 0:
        raise ValueError(f"--k_shot_grid values must be positive integers, got {value}")
    return int(value)

def get_effective_label_sampling_modes(args, base_ssl_config):
    label_sampling_modes = args.ssl_label_sampling_modes
    if label_sampling_modes is None:
        return [base_ssl_config.label_sampling_mode]
    if not label_sampling_modes:
        raise ValueError("--ssl_label_sampling_modes must include at least one value when provided")
    return label_sampling_modes

def has_outer_comparison_grid(args):
    return (
        args.label_budget_grid is not None
        or getattr(args, "k_shot_grid", None) is not None
        or args.loss_miner_grid is not None
        or args.comparison_seeds is not None
        or args.ssl_label_sampling_modes is not None
    )

def get_loss_miner_pairs(args):
    if args.loss_miner_grid is None:
        validate_loss_miner_pair(args.loss, args.miner, require_effective_miner=False)
        return [(args.loss, args.miner)]
    if not args.loss_miner_grid:
        raise ValueError("--loss_miner_grid must include at least one LOSS:MINER pair when provided")
    return [parse_loss_miner_pair(raw_pair) for raw_pair in args.loss_miner_grid]

def parse_loss_miner_pair(raw_pair):
    if ":" in raw_pair:
        loss_name, miner_name = raw_pair.split(":", 1)
    elif "," in raw_pair:
        loss_name, miner_name = raw_pair.split(",", 1)
    else:
        raise ValueError(f"Loss/miner grid entry must be formatted as LOSS:MINER, got {raw_pair!r}")

    loss_name = loss_name.strip()
    miner_name = miner_name.strip()
    validate_loss_miner_pair(loss_name, miner_name, source=f" in --loss_miner_grid entry {raw_pair!r}")
    return loss_name, miner_name

def validate_loss_miner_pair(loss_name, miner_name, source="", require_effective_miner=True):
    if loss_name not in ALL_LOSSES:
        raise ValueError(f"loss must be one of {ALL_LOSSES}{source}: {loss_name}")
    if miner_name not in ALL_MINERS:
        raise ValueError(f"miner must be one of {ALL_MINERS}{source}: {miner_name}")
    if require_effective_miner and loss_name in CLASSIFICATION_LOSSES and miner_name != "no_miner":
        raise ValueError(
            f"classification loss {loss_name} should be paired with no_miner because miners are ignored{source}"
        )
    if loss_name == "STMLLoss" and miner_name != "no_miner":
        raise ValueError(f"STMLLoss must be paired with no_miner because it does not consume labels{source}")

def make_label_budget_name(label_sampling_mode, labeled_fraction, labeled_per_class):
    if label_sampling_mode == "class_subset_k_shot":
        label_part = (
            f"label_{format_float_token(labeled_fraction)}"
            f"_k_{labeled_per_class}"
        )
    elif labeled_per_class is None:
        label_part = f"label_{format_float_token(labeled_fraction)}"
    else:
        label_part = f"per_class_{labeled_per_class}"
    return label_part

def make_scenario_name(ssl_config, loss=None, miner=None):
    # Encode all varied dimensions into a filesystem-safe, deterministic name.
    label_part = make_label_budget_name(
        ssl_config.label_sampling_mode,
        ssl_config.labeled_fraction,
        ssl_config.labeled_per_class,
    )
    parts = [ssl_config.label_sampling_mode, label_part]
    if loss is not None and miner is not None:
        # Loss/miner are omitted when they are fixed globally to avoid adding
        # redundant text to every scenario name.
        parts.extend([loss, miner])
    parts.extend(["seed", str(ssl_config.seed)])
    return "_".join(parts)

def make_frozen_hparam_group_name(scenario, reference_seed):
    label_part = make_label_budget_name(
        scenario.label_sampling_mode,
        scenario.labeled_fraction,
        scenario.labeled_per_class,
    )
    parts = [
        scenario.label_sampling_mode,
        label_part,
        scenario.loss,
        scenario.miner,
        "tune_seed",
        str(int(reference_seed)),
    ]
    return "_".join(parts)

def scenario_group_key(scenario):
    return (
        scenario.label_sampling_mode,
        scenario.labeled_fraction,
        scenario.labeled_per_class,
        scenario.loss,
        scenario.miner,
    )

def group_scenarios_by_frozen_config(scenarios):
    groups_by_key = {}
    ordered_keys = []
    for scenario in scenarios:
        key = scenario_group_key(scenario)
        if key not in groups_by_key:
            groups_by_key[key] = []
            ordered_keys.append(key)
        groups_by_key[key].append(scenario)
    return [groups_by_key[key] for key in ordered_keys]

def write_reference_ssl_config(args, base_ssl_config, scenario, grid_dir_name):
    """Persist the fixed support draw used for HPO for one non-seed scenario."""

    group_name = make_frozen_hparam_group_name(scenario, args.seed)
    reference_ssl_config = replace(
        base_ssl_config,
        label_sampling_mode=scenario.label_sampling_mode,
        labeled_fraction=float(scenario.labeled_fraction),
        labeled_per_class=scenario.labeled_per_class,
        seed=int(args.seed),
    )
    semi_supervised.validate_ssl_config(reference_ssl_config)
    config_dir = Path("logs") / args.save_dir / grid_dir_name / "ssl_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{group_name}.json"
    write_json(config_path, reference_ssl_config.to_dict())
    return group_name, config_path, reference_ssl_config

def make_args_for_scenario(args, scenario):
    scenario_args = copy.deepcopy(args)
    scenario_args.seed = scenario.seed
    scenario_args.loss = scenario.loss
    scenario_args.miner = scenario.miner
    scenario_args.ssl_config = scenario.ssl_config_path
    scenario_args.save_dir = Path(args.save_dir) / scenario.name
    return scenario_args

def make_reference_hpo_args(args, group_name, reference_ssl_config_path, scenario):
    reference_args = copy.deepcopy(args)
    reference_args.seed = int(args.seed)
    reference_args.loss = scenario.loss
    reference_args.miner = scenario.miner
    reference_args.ssl_config = reference_ssl_config_path
    reference_args.save_dir = Path(args.save_dir) / group_name / "hpo"
    return reference_args

def format_float_token(value):
    return f"{value:g}".replace(".", "p")

def run_single_method_grid(args, hparam_config):
    """Run an outer experiment grid for only the selected training method."""

    base_ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    validate_single_method_grid_setup(args, hparam_config, base_ssl_config)

    scenarios = make_comparison_scenarios(args, base_ssl_config, grid_dir_name="experiment_grid")
    if hparam_config is not None and hparam_config.enabled:
        grid_results = run_single_method_frozen_hparam_grid(
            args=args,
            hparam_config=hparam_config,
            base_ssl_config=base_ssl_config,
            scenarios=scenarios,
        )
        write_single_method_grid_summary(Path("logs") / args.save_dir / "experiment_grid", grid_results)
        return

    grid_results = []
    for scenario in scenarios:
        scenario_args = make_args_for_scenario(args, scenario)
        grid_results.append(run_single_method_scenario(scenario_args, hparam_config, scenario))

    write_single_method_grid_summary(Path("logs") / args.save_dir / "experiment_grid", grid_results)

def run_single_method_frozen_hparam_grid(args, hparam_config, base_ssl_config, scenarios):
    """Tune once per non-seed scenario, then evaluate frozen params for each seed."""

    grid_results = []
    for scenario_group in group_scenarios_by_frozen_config(scenarios):
        reference_scenario = scenario_group[0]
        group_name, reference_ssl_config_path, _ = write_reference_ssl_config(
            args=args,
            base_ssl_config=base_ssl_config,
            scenario=reference_scenario,
            grid_dir_name="experiment_grid",
        )
        reference_args = make_reference_hpo_args(
            args=args,
            group_name=group_name,
            reference_ssl_config_path=reference_ssl_config_path,
            scenario=reference_scenario,
        )
        scenario_hparam_config = make_standalone_hparam_config(reference_args, hparam_config)
        study_result = run_hparam_search(reference_args, scenario_hparam_config)
        for scenario in scenario_group:
            eval_args = make_args_for_scenario(args, scenario)
            final_result = run_final_from_best_hparam(
                eval_args,
                scenario_hparam_config,
                study_result,
                role=eval_args.mode,
                summary_stem=f"final_evaluation_{scenario.name}_{eval_args.mode}",
            )
            grid_results.append(
                {
                    "method": eval_args.mode,
                    "scenario": scenario,
                    "study": study_result,
                    "result": final_result,
                }
            )
    return grid_results

def validate_single_method_grid_setup(args, hparam_config, base_ssl_config):
    validate_k_shot_grid_hparam_setup(args, hparam_config, base_ssl_config)
    if hparam_config is not None and hparam_config.enabled:
        if hparam_config.study_dir is not None or hparam_config.storage is not None:
            raise ValueError(
                "The outer experiment grid needs separate Optuna storage per scenario. "
                "Leave hparam_config.study_dir and hparam_config.storage as null."
            )

def run_single_method_scenario(args, hparam_config, scenario):
    method = args.mode
    if hparam_config is not None and hparam_config.enabled:
        scenario_hparam_config = make_standalone_hparam_config(args, hparam_config)
        study_result = run_hparam_search(args, scenario_hparam_config)
        final_result = None
        if args.final_test_after_hpo:
            final_result = run_final_from_best_hparam(
                args,
                scenario_hparam_config,
                study_result,
                role=method,
            )
        return {
            "method": method,
            "scenario": scenario,
            "study": study_result,
            "result": final_result,
        }

    if hparam_config is not None:
        args.hparam_config_resolved = hparam_config.to_dict()
    ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    ssl_config = resolve_mode_ssl_config(args, ssl_config)
    result = run_experiment(args, ssl_config)
    return {
        "method": method,
        "scenario": scenario,
        "study": None,
        "result": result,
    }

def make_standalone_hparam_config(args, config):
    if is_supervised_mode(args):
        return make_comparison_hparam_config(config, role="supervised")
    return config

def validate_comparison_setup(args, hparam_config, ssl_config):
    if hparam_config is None or not hparam_config.enabled:
        raise ValueError("--compare_supervised_ssl requires an enabled --hparam_config")
    if not ssl_config.enabled:
        raise ValueError("--compare_supervised_ssl requires an enabled --ssl_config for the SSL method and split")
    if not hparam_config.metric.startswith("best_valid_"):
        raise ValueError(
            "Use a validation metric for comparison HPO; D_test and train loss are not valid model-selection targets"
        )
    if has_outer_comparison_grid(args) and (hparam_config.study_dir is not None or hparam_config.storage is not None):
        raise ValueError(
            "The outer comparison grid needs separate Optuna storage per scenario. "
            "Leave hparam_config.study_dir and hparam_config.storage as null."
        )
    validate_k_shot_grid_hparam_setup(args, hparam_config, ssl_config)

    forbidden_keys = sorted(set(hparam_config.spaces) & COMPARISON_FORBIDDEN_HPARAM_KEYS)
    if forbidden_keys:
        raise ValueError(
            "The comparison mode requires fixed dataset/split settings. "
            f"Remove these keys from the HPO spaces: {forbidden_keys}"
        )

    supervised_spaces = {
        name: spec
        for name, spec in hparam_config.spaces.items()
        if not is_ssl_override(name) or name in SUPERVISED_SPLIT_SSL_HPARAM_KEYS
    }
    if not supervised_spaces:
        raise ValueError(
            "The supervised HPO search space is empty after removing ssl_config.* entries. "
            "Add at least one non-SSL hyperparameter such as lr, batch_size, sampler_m, or classifier_lr."
        )

def validate_k_shot_grid_hparam_setup(args, hparam_config, base_ssl_config):
    if hparam_config is None or not hparam_config.enabled or getattr(args, "k_shot_grid", None) is None:
        return
    if "class_subset_k_shot" not in get_effective_label_sampling_modes(args, base_ssl_config):
        return

    forbidden_keys = sorted(set(hparam_config.spaces) & LABELED_PER_CLASS_HPARAM_KEYS)
    if forbidden_keys:
        raise ValueError(
            "For label_sampling_mode='class_subset_k_shot', k-shot settings are controlled by "
            f"--k_shot_grid. Remove these keys from the HPO spaces: {forbidden_keys}"
        )

def make_comparison_hparam_config(config, role):
    if role not in {"supervised", "ssl"}:
        raise ValueError(f"Unknown comparison role: {role}")

    base_study_name = config.study_name or "optuna"
    spaces = dict(config.spaces)
    if role == "supervised":
        spaces = {
            name: spec
            for name, spec in spaces.items()
            if not is_ssl_override(name) or name in SUPERVISED_SPLIT_SSL_HPARAM_KEYS
        }

    resolved = replace(
        config,
        study_name=f"{base_study_name}_{role}",
        study_dir=append_study_dir_role(config.study_dir, role),
        spaces=spaces,
    )
    validate_hparam_config(resolved)
    return resolved

def append_study_dir_role(study_dir, role):
    if study_dir is None:
        return None
    return str(Path(study_dir) / role)

def run_final_from_best_hparam(base_args, hparam_config, study_result, role, summary_stem="final_evaluation"):
    """Train the winning HPO configuration on all development data and test it."""

    role = role or "model"
    if study_result.best_params is None:
        raise ValueError(f"No completed {role} HPO trial is available for final retraining")

    epoch_plan = make_final_epoch_plan(study_result)
    final_args, final_ssl_config = make_args_and_ssl_config_from_params(base_args, study_result.best_params)
    final_args.hparam_config_resolved = hparam_config.to_dict()
    final_args.hparam_params = study_result.best_params
    final_args.hparam_final_from_study = study_result.study_name
    final_args.hparam_final_trial_number = study_result.best_trial_number
    final_args.hparam_final_epoch_plan = epoch_plan
    final_args.final_full_train = True
    final_args.cv_k = 1
    final_args.epochs = epoch_plan["final_training_epochs"]
    final_args.evaluate_test = True
    final_args.skip_test_during_hpo = False
    final_args.save_dir = Path(base_args.save_dir) / "final" / role

    final_result = run_experiment(final_args, final_ssl_config)
    best_attrs = study_result.best_user_attrs or {}
    final_result = replace(
        final_result,
        best_valid_precision_at_1=best_attrs.get("best_valid_precision_at_1"),
        best_valid_mean_average_precision_at_r=best_attrs.get("best_valid_mean_average_precision_at_r"),
    )
    write_hparam_final_evaluation_summary(study_result, final_result, epoch_plan, role, summary_stem=summary_stem)
    return final_result

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
        "test_precision_at_1": optional_number(final_result.test_precision_at_1),
        "test_mean_average_precision_at_r": optional_number(final_result.test_mean_average_precision_at_r),
        "selected_epoch": final_result.selected_epoch,
        "global_step": final_result.global_step,
    }
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Final HPO test evaluation summary written to {study_result.study_dir}")

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
        "test_precision_at_1": "" if final_result.test_precision_at_1 is None else final_result.test_precision_at_1,
        "test_mean_average_precision_at_r": ""
        if final_result.test_mean_average_precision_at_r is None
        else final_result.test_mean_average_precision_at_r,
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
    }

def comparison_scenario_to_dict(scenario):
    return {
        "name": scenario.name,
        "labeled_fraction": scenario.labeled_fraction,
        "labeled_per_class": scenario.labeled_per_class,
        "seed": scenario.seed,
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
        "seed": scenario.seed,
        "best_trial_number": "",
        "best_hpo_value": "",
        "log_dir": "",
        "metrics_csv": "",
        "best_valid_precision_at_1": "",
        "best_valid_mean_average_precision_at_r": "",
        "test_precision_at_1": "",
        "test_mean_average_precision_at_r": "",
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
                    "test_precision_at_1": optional_number(result.test_precision_at_1),
                    "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
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
                "test_precision_at_1": optional_number(result.test_precision_at_1),
                "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
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
        "seed": scenario.seed,
        "comparison_dir": str(result["comparison_dir"]),
        "supervised_test_precision_at_1": optional_number(supervised_final.test_precision_at_1),
        "ssl_test_precision_at_1": optional_number(ssl_final.test_precision_at_1),
        "delta_test_precision_at_1": optional_number(deltas["test_precision_at_1"]),
        "supervised_test_mean_average_precision_at_r": optional_number(
            supervised_final.test_mean_average_precision_at_r
        ),
        "ssl_test_mean_average_precision_at_r": optional_number(ssl_final.test_mean_average_precision_at_r),
        "delta_test_mean_average_precision_at_r": optional_number(deltas["test_mean_average_precision_at_r"]),
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()
