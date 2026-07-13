# ruff: noqa: F401,F403,F405
"""Experiment orchestration and executable entry point.

Training, CLI parsing, shared types, and HPO implementation live in focused
modules. Imports below preserve the historical ``main`` module API.
"""


import json
import math
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from training import hpo as _experiment_hpo
from training.cli import *  # noqa: F403
from training.hpo import *  # noqa: F403
from training.io import *  # noqa: F403
from training.reporting import (
    comparison_scenario_to_dict,
    make_comparison_deltas,
    make_final_epoch_plan,
    write_cross_seed_train_val_evaluation_summary,
    write_comparison_grid_summary,
    write_comparison_summary,
    write_hparam_final_evaluation_summary,
    write_hparam_selected_final_evaluation_summary,
    write_hparam_top_final_evaluation_summary,
    write_hparam_train_val_evaluation_summary,
    write_single_method_grid_summary,
)
from training.engine import *  # noqa: F403
from training.types import *  # noqa: F403

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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

    if args.final_test_study_dir is not None:
        run_final_from_study_dir_request(args)
        return

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
        if should_run_final_hparam_evaluation(args):
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
    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    ssl_config = resolve_mode_ssl_config(args, ssl_config)
    run_experiment(args, ssl_config)

def run_supervised_ssl_comparison(args, hparam_config):
    """Run matched supervised and SSL studies for every outer-grid scenario."""

    # This base config defines both the SSL algorithm and the common label
    # apportioning rules used by the supervised baseline.
    base_ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
    validate_comparison_setup(args, hparam_config, base_ssl_config)

    # Scenarios are concrete combinations of label budget, sampling mode,
    # loss/miner, and comparison seed.  Each seed owns a full supervised/SSL HPO
    # pair because it changes both the dataset split and the labeled support draw.
    scenarios = make_comparison_scenarios(args, base_ssl_config)
    grid_results = []
    for scenario in scenarios:
        scenario_args = make_args_for_scenario(args, scenario)
        scenario_ssl_config = semi_supervised.load_ssl_config(
            scenario_args.ssl_config,
            default_seed=scenario_args.seed,
            default_support_seed=get_support_seed(scenario_args),
        )
        grid_results.append(
            run_single_supervised_ssl_comparison(
                scenario_args,
                hparam_config,
                scenario_ssl_config,
                scenario,
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
        scenario_ssl_config = semi_supervised.load_ssl_config(
            scenario.ssl_config_path,
            default_seed=eval_args.seed,
            default_support_seed=get_support_seed(eval_args),
        )

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

def get_comparison_seed_targets(args):
    """Return validated seed channels varied by each comparison seed."""

    raw_targets = getattr(args, "comparison_seed_targets", None)
    if raw_targets is None:
        raw_targets = COMPARISON_SEED_TARGETS
    if isinstance(raw_targets, str) or not raw_targets:
        raise ValueError(
            "comparison_seed_targets must contain one or more of "
            f"{list(COMPARISON_SEED_TARGETS)}"
        )
    targets = list(raw_targets)
    invalid_targets = sorted(set(targets) - set(COMPARISON_SEED_TARGETS))
    if invalid_targets:
        raise ValueError(
            f"Unknown comparison_seed_targets {invalid_targets}; choose from {list(COMPARISON_SEED_TARGETS)}"
        )
    duplicate_targets = sorted({target for target in targets if targets.count(target) > 1})
    if duplicate_targets:
        raise ValueError(f"comparison_seed_targets contains duplicates: {duplicate_targets}")
    target_set = set(targets)
    return tuple(target for target in COMPARISON_SEED_TARGETS if target in target_set)


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

    comparison_seeds = args.comparison_seeds
    has_comparison_seed_grid = comparison_seeds is not None
    if comparison_seeds is None:
        comparison_seeds = [None]
    elif not comparison_seeds:
        raise ValueError("--comparison_seeds must include at least one value when provided")
    comparison_seed_targets = get_comparison_seed_targets(args)

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
                    for comparison_seed in comparison_seeds:
                        if comparison_seed is None:
                            scenario_seed = int(base_ssl_config.seed)
                            run_seed = None
                            scenario_ssl_seed = int(base_ssl_config.seed)
                            data_split_seed = int(getattr(args, "data_split_seed", DEFAULT_DATA_SPLIT_SEED))
                            support_seed = int(base_ssl_config.support_seed)
                            hparam_seed = int(get_hparam_seed(args))
                            applied_seed_targets = ()
                        else:
                            scenario_seed = int(comparison_seed)
                            run_seed = (
                                scenario_seed
                                if COMPARISON_SEED_TARGET_RUNTIME in comparison_seed_targets
                                else None
                            )
                            scenario_ssl_seed = (
                                scenario_seed
                                if COMPARISON_SEED_TARGET_RUNTIME in comparison_seed_targets
                                else int(base_ssl_config.seed)
                            )
                            data_split_seed = (
                                scenario_seed
                                if COMPARISON_SEED_TARGET_DATA_SPLIT in comparison_seed_targets
                                else int(getattr(args, "data_split_seed", DEFAULT_DATA_SPLIT_SEED))
                            )
                            support_seed = (
                                scenario_seed
                                if COMPARISON_SEED_TARGET_SUPPORT in comparison_seed_targets
                                else int(base_ssl_config.support_seed)
                            )
                            hparam_seed = (
                                scenario_seed
                                if COMPARISON_SEED_TARGET_HPARAM in comparison_seed_targets
                                else int(get_hparam_seed(args))
                            )
                            applied_seed_targets = comparison_seed_targets
                        # dataclasses.replace creates a new immutable config,
                        # leaving the shared base config untouched.
                        scenario_ssl_config = replace(
                            base_ssl_config,
                            label_sampling_mode=label_sampling_mode,
                            labeled_fraction=float(labeled_fraction),
                            labeled_per_class=scenario_labeled_per_class,
                            seed=scenario_ssl_seed,
                            support_seed=support_seed,
                        )
                        semi_supervised.validate_ssl_config(scenario_ssl_config)
                        # Include only dimensions needed to distinguish the
                        # requested scenarios; this keeps paths readable.
                        scenario_name = make_scenario_name(
                            scenario_ssl_config,
                            loss=loss_name if include_loss_miner_in_name else None,
                            miner=miner_name if include_loss_miner_in_name else None,
                            comparison_seed=scenario_seed if has_comparison_seed_grid else None,
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
                                seed=scenario_seed,
                                label_sampling_mode=scenario_ssl_config.label_sampling_mode,
                                loss=loss_name,
                                miner=miner_name,
                                ssl_config_path=config_path,
                                run_seed=run_seed,
                                data_split_seed=data_split_seed,
                                support_seed=scenario_ssl_config.support_seed,
                                hparam_seed=hparam_seed,
                                comparison_seed_targets=applied_seed_targets,
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

def make_scenario_name(ssl_config, loss=None, miner=None, comparison_seed=None):
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
    if comparison_seed is not None:
        parts.extend(["comparison_seed", str(int(comparison_seed))])
        return "_".join(parts)
    if (
        ssl_config.support_seed is not None
        and ssl_config.seed is not None
        and int(ssl_config.support_seed) != int(ssl_config.seed)
    ):
        parts.extend(["support_seed", str(int(ssl_config.support_seed))])
    parts.extend(["seed", str(ssl_config.seed)])
    return "_".join(parts)

def make_frozen_hparam_group_name(scenario, reference_seed, support_seed=None):
    if support_seed is None:
        support_seed = scenario.support_seed
    tune_seed = reference_seed if support_seed is None else support_seed
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
        str(int(tune_seed)),
    ]
    if support_seed is not None and int(reference_seed) != int(support_seed):
        parts.extend(["run_seed", str(int(reference_seed))])
    for name, value in (
        ("data_split_seed", scenario.data_split_seed),
        ("hparam_seed", scenario.hparam_seed),
    ):
        if value is not None and int(value) != int(tune_seed):
            parts.extend([name, str(int(value))])
    return "_".join(parts)

def scenario_group_key(scenario):
    return (
        scenario.label_sampling_mode,
        scenario.labeled_fraction,
        scenario.labeled_per_class,
        scenario.loss,
        scenario.miner,
        scenario.run_seed,
        scenario.data_split_seed,
        scenario.support_seed,
        scenario.hparam_seed,
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

    support_seed = scenario.support_seed if scenario.support_seed is not None else base_ssl_config.support_seed
    run_seed = scenario.run_seed if scenario.run_seed is not None else args.seed
    group_name = make_frozen_hparam_group_name(
        scenario,
        run_seed,
        support_seed=support_seed,
    )
    reference_ssl_config = replace(
        base_ssl_config,
        label_sampling_mode=scenario.label_sampling_mode,
        labeled_fraction=float(scenario.labeled_fraction),
        labeled_per_class=scenario.labeled_per_class,
        seed=run_seed if scenario.run_seed is not None else base_ssl_config.seed,
        support_seed=support_seed,
    )
    semi_supervised.validate_ssl_config(reference_ssl_config)
    config_dir = Path("logs") / args.save_dir / grid_dir_name / "ssl_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{group_name}.json"
    write_json(config_path, reference_ssl_config.to_dict())
    return group_name, config_path, reference_ssl_config

def make_args_for_scenario(args, scenario):
    scenario_args = copy.deepcopy(args)
    if scenario.run_seed is not None:
        scenario_args.seed = int(scenario.run_seed)
    if scenario.data_split_seed is not None:
        scenario_args.data_split_seed = int(scenario.data_split_seed)
    if scenario.support_seed is not None:
        scenario_args.support_seed = int(scenario.support_seed)
    if scenario.hparam_seed is not None:
        scenario_args.hparam_seed = int(scenario.hparam_seed)
    scenario_args.loss = scenario.loss
    scenario_args.miner = scenario.miner
    scenario_args.ssl_config = scenario.ssl_config_path
    scenario_args.save_dir = Path(args.save_dir) / scenario.name
    return scenario_args

def make_reference_hpo_args(args, group_name, reference_ssl_config_path, scenario):
    reference_args = copy.deepcopy(args)
    if scenario.run_seed is not None:
        reference_args.seed = int(scenario.run_seed)
    if scenario.data_split_seed is not None:
        reference_args.data_split_seed = int(scenario.data_split_seed)
    if scenario.support_seed is not None:
        reference_args.support_seed = int(scenario.support_seed)
    if scenario.hparam_seed is not None:
        reference_args.hparam_seed = int(scenario.hparam_seed)
    reference_args.loss = scenario.loss
    reference_args.miner = scenario.miner
    reference_args.ssl_config = reference_ssl_config_path
    reference_args.save_dir = Path(args.save_dir) / group_name / "hpo"
    return reference_args

def format_float_token(value):
    return f"{value:g}".replace(".", "p")

def run_single_method_grid(args, hparam_config):
    """Run an outer experiment grid for only the selected training method."""

    base_ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
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
            final_result = None
            if should_run_final_hparam_evaluation(args):
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
        if should_run_final_hparam_evaluation(args):
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
    ssl_config = semi_supervised.load_ssl_config(
        args.ssl_config,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )
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

FINAL_STUDY_DIR_GRID_OVERRIDE_ARGS = {
    "label_budget_grid",
    "k_shot_grid",
    "ssl_label_sampling_modes",
    "comparison_seeds",
}

FINAL_STUDY_DIR_REPOSITORY_OVERRIDE_ARGS = FINAL_STUDY_DIR_GRID_OVERRIDE_ARGS | {
    "seed",
    "hparam_seed",
    "data_split_seed",
    "support_seed",
}

FINAL_STUDY_DIR_IGNORED_REQUEST_OVERRIDES = {
    "experiment_config",
    "final_test_after_hpo",
    "final_test_study_dir",
    "hparam_config",
    "loss",
    "loss_miner_grid",
    "miner",
}

FINAL_STUDY_REPOSITORY_FORBIDDEN_HPARAM_KEYS = (
    COMPARISON_FORBIDDEN_HPARAM_KEYS | SAMPLER_CAPACITY_HPARAM_KEYS
)

def run_final_from_study_dir_request(request_args):
    """Replay final-test evaluation from an existing study directory."""

    base_args, hparam_config, study_result = load_hparam_study_for_final_test(request_args.final_test_study_dir)
    base_args = copy_final_test_request_args(request_args, base_args)
    request_override_names = set(getattr(base_args, "final_study_request_overrides", []))
    if request_override_names & FINAL_STUDY_DIR_REPOSITORY_OVERRIDE_ARGS:
        validate_final_study_hparam_repository(hparam_config, study_result)
    if (
        get_study_dir_mode(base_args) == STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL
        and "comparison_seeds" not in request_override_names
    ):
        raise ValueError("study_dir_mode='cross_seed_train_val' requires comparison_seeds in the replay request")
    if has_final_study_dir_replay_grid(request_override_names):
        reset_unrequested_final_study_grid_args(base_args, request_override_names)
        return run_final_study_dir_grid(base_args, hparam_config, study_result)

    return run_study_dir_hparam_evaluation(
        base_args,
        hparam_config,
        study_result,
        role=getattr(base_args, "mode", None),
    )

def copy_final_test_request_args(source_args, target_args):
    """Apply current-request overrides to args loaded from a study directory."""

    override_names = get_final_study_request_override_names(source_args)
    if "loss_miner_grid" in override_names:
        logger.warning(
            "Ignoring loss_miner_grid for study-directory replay: one study directory represents one "
            "method's trials. Run the replay once for each method's own study directory."
        )
    for name in sorted(override_names - FINAL_STUDY_DIR_IGNORED_REQUEST_OVERRIDES):
        if hasattr(source_args, name):
            setattr(target_args, name, copy.deepcopy(getattr(source_args, name)))

    if "seed" in override_names and "hparam_seed" not in override_names and hasattr(source_args, "hparam_seed"):
        target_args.hparam_seed = source_args.hparam_seed
    for name in ("final_test_top_n", "final_test_trial_numbers", "final_test_visualization", "study_dir_mode"):
        if hasattr(source_args, name):
            setattr(target_args, name, getattr(source_args, name))
    target_args.final_test_after_hpo = True
    normalize_backbone_tuning_args(target_args)
    resolve_hparam_seed(target_args)
    resolve_data_split_seed(target_args)
    resolve_support_seed(target_args)
    target_args.final_study_request_overrides = sorted(override_names)
    return target_args

def get_final_study_request_override_names(args):
    """Return config/CLI fields explicitly supplied for a final-study replay."""

    config_values = getattr(args, "experiment_config_resolved", None) or {}
    explicit_cli_args = getattr(args, "explicit_cli_args", None) or []
    return set(config_values) | set(explicit_cli_args)

def has_final_study_dir_replay_grid(request_override_names):
    """Return whether the current replay request varies outer-grid settings."""

    return bool(set(request_override_names) & FINAL_STUDY_DIR_GRID_OVERRIDE_ARGS)

def reset_unrequested_final_study_grid_args(args, request_override_names):
    """Ignore grid metadata inherited from the original HPO run unless requested."""

    for name in FINAL_STUDY_DIR_GRID_OVERRIDE_ARGS:
        if name not in request_override_names:
            setattr(args, name, None)
    args.loss_miner_grid = None
    return args

def run_final_study_dir_grid(base_args, hparam_config, study_result):
    """Evaluate one existing HPO study's params across requested data settings."""

    base_ssl_config = semi_supervised.load_ssl_config(
        base_args.ssl_config,
        default_seed=base_args.seed,
        default_support_seed=get_support_seed(base_args),
    )
    scenarios = make_comparison_scenarios(base_args, base_ssl_config, grid_dir_name="study_replay_grid")
    if get_study_dir_mode(base_args) == STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL:
        return run_cross_seed_train_val_study_dir_grid(
            base_args,
            hparam_config,
            study_result,
            base_ssl_config,
            scenarios,
        )

    grid_results = []
    summary_prefix = get_study_dir_summary_stem(base_args)
    for scenario in scenarios:
        scenario_args = make_args_for_scenario(base_args, scenario)
        method = getattr(scenario_args, "mode", None) or "model"
        final_result = run_study_dir_hparam_evaluation(
            scenario_args,
            hparam_config,
            study_result,
            role=method,
            summary_stem=f"{summary_prefix}_{scenario.name}_{method}",
        )
        grid_results.append(
            {
                "method": method,
                "scenario": scenario,
                "study": study_result,
                "result": final_result,
            }
        )

    write_single_method_grid_summary(Path("logs") / base_args.save_dir / "study_replay_grid", grid_results)
    return grid_results


def run_cross_seed_train_val_study_dir_grid(
    base_args,
    hparam_config,
    study_result,
    base_ssl_config,
    scenarios,
):
    """Select HPO params by mean validation performance across seed scenarios."""

    scenario_groups = group_cross_seed_validation_scenarios(scenarios)
    selected_trials = select_hparam_trials_for_cross_seed_replay(
        base_args,
        hparam_config,
        study_result,
        role=getattr(base_args, "mode", None),
    )
    output_dir = Path("logs") / base_args.save_dir / "study_replay_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_results = []
    for scenario_group in scenario_groups:
        reference_scenario = scenario_group[0]
        role = getattr(base_args, "mode", None) or "model"
        group_name = make_cross_seed_validation_group_name(reference_scenario, role)
        selection_metric = getattr(base_args, "selection_metric", SELECTION_METRIC_MAP_AT_R)
        candidate_records = []
        best_candidate = None
        for trial in selected_trials:
            trial_study_result = make_study_result_for_completed_trial(study_result, trial)
            validation_results = []
            validation_metadata = []
            validation_runs = []
            for scenario in scenario_group:
                scenario_args = make_args_for_scenario(base_args, scenario)
                summary_stem = (
                    f"cross_seed_train_val_{group_name}_trial_{int(trial['trial_number']):04d}_"
                    f"comparison_seed_{int(scenario.seed)}"
                )
                validation_result = run_single_train_val_from_hparam(
                    scenario_args,
                    hparam_config,
                    trial_study_result,
                    role,
                    summary_stem=summary_stem,
                )
                validation_results.append(validation_result)
                metadata = {
                    "scenario": scenario.name,
                    "comparison_seed": int(scenario.seed),
                    "comparison_seed_targets": list(scenario.comparison_seed_targets),
                    "run_seed": scenario.run_seed,
                    "data_split_seed": scenario.data_split_seed,
                    "support_seed": scenario.support_seed,
                    "hparam_seed": scenario.hparam_seed,
                }
                validation_metadata.append(metadata)
                validation_runs.append(
                    {
                        **metadata,
                        "selection_value": float(
                            get_selection_metric_value(
                                selection_metric,
                                validation_result.best_valid_precision_at_1,
                                validation_result.best_valid_mean_average_precision_at_r,
                            )
                        ),
                        "result": result_to_dict(validation_result),
                    }
                )

            selected_study_result = make_validation_selected_study_result(
                study_result,
                trial,
                validation_results,
                selection_metric,
                validation_result_metadata=validation_metadata,
            )
            attrs = selected_study_result.best_user_attrs or {}
            candidate = {
                "trial": trial,
                "validation_runs": validation_runs,
                "mean_best_valid_precision_at_1": attrs["best_valid_precision_at_1"],
                "mean_best_valid_mean_average_precision_at_r": attrs[
                    "best_valid_mean_average_precision_at_r"
                ],
                "selection_metric": selection_metric,
                "mean_selection_value": attrs["mean_validation_selection_value"],
                "selected_for_final": False,
                "selected_study_result": selected_study_result,
            }
            candidate_records.append(candidate)
            if (
                best_candidate is None
                or candidate["mean_selection_value"] > best_candidate["mean_selection_value"]
                or (
                    candidate["mean_selection_value"] == best_candidate["mean_selection_value"]
                    and int(trial["trial_number"]) < int(best_candidate["trial"]["trial_number"])
                )
            ):
                best_candidate = candidate

        if best_candidate is None:
            raise ValueError("No cross-seed train/validation candidate is available for final retraining")

        best_candidate["selected_for_final"] = True
        selected_study_result = best_candidate.pop("selected_study_result")
        for candidate in candidate_records:
            candidate.pop("selected_study_result", None)
        final_args = make_cross_seed_final_args(
            base_args,
            base_ssl_config,
            reference_scenario,
            group_name,
            scenario_group,
        )
        final_summary_stem = f"cross_seed_train_val_{group_name}_final"
        epoch_plan = make_final_epoch_plan(selected_study_result)
        logger.info(
            "cross_seed_train_val selected trial "
            f"{best_candidate['trial']['trial_number']} for {group_name}: "
            f"mean {selection_metric}={best_candidate['mean_selection_value']:.6f} across "
            f"{len(scenario_group)} validation seeds; final epochs={epoch_plan['final_training_epochs']}"
        )
        final_result = run_single_final_from_hparam(
            final_args,
            hparam_config,
            selected_study_result,
            role,
            summary_stem=final_summary_stem,
        )
        summary_paths = write_cross_seed_train_val_evaluation_summary(
            output_dir=output_dir,
            summary_stem=f"cross_seed_train_val_{group_name}",
            role=role,
            study_result=study_result,
            scenarios=scenario_group,
            selection_metric=selection_metric,
            candidates=candidate_records,
            winner=best_candidate,
            epoch_plan=epoch_plan,
            final_result=final_result,
        )
        group_results.append(
            {
                "method": role,
                "group_name": group_name,
                "scenarios": scenario_group,
                "study": selected_study_result,
                "candidates": candidate_records,
                "winner": best_candidate,
                "result": final_result,
                "summary_paths": summary_paths,
            }
        )
    return group_results


def group_cross_seed_validation_scenarios(scenarios):
    groups = {}
    ordered_keys = []
    for scenario in scenarios:
        key = (
            scenario.label_sampling_mode,
            scenario.labeled_fraction,
            scenario.labeled_per_class,
            scenario.loss,
            scenario.miner,
        )
        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(scenario)
    grouped_scenarios = [groups[key] for key in ordered_keys]
    for group in grouped_scenarios:
        comparison_seeds = {int(scenario.seed) for scenario in group}
        if len(comparison_seeds) < 2:
            raise ValueError(
                "study_dir_mode='cross_seed_train_val' requires at least two distinct comparison_seeds "
                "for every non-seed scenario"
            )
        replay_targets = set().union(*(set(scenario.comparison_seed_targets) for scenario in group))
        if not replay_targets & {
            COMPARISON_SEED_TARGET_RUNTIME,
            COMPARISON_SEED_TARGET_DATA_SPLIT,
            COMPARISON_SEED_TARGET_SUPPORT,
        }:
            raise ValueError(
                "cross_seed_train_val comparison_seed_targets must include seed, data_split_seed, "
                "or support_seed; hparam_seed alone does not change fixed-parameter validation replays"
            )
    return grouped_scenarios


def select_hparam_trials_for_cross_seed_replay(base_args, hparam_config, study_result, role):
    completed_trials = get_completed_hparam_trials(study_result, role)
    trial_numbers = get_final_test_trial_numbers(base_args)
    if trial_numbers is not None:
        trials_by_number = {int(trial["trial_number"]): trial for trial in completed_trials}
        missing_trials = [number for number in trial_numbers if number not in trials_by_number]
        if missing_trials:
            raise ValueError(
                f"Selected cross-seed trial(s) are not completed: {missing_trials}. "
                f"Available completed trial numbers: {sorted(trials_by_number)}"
            )
        return [trials_by_number[number] for number in trial_numbers]

    top_n = get_final_test_top_n(base_args)
    maximize = getattr(hparam_config, "direction", "maximize") == "maximize"
    completed_trials = sorted(
        completed_trials,
        key=lambda trial: (
            -float(trial["value"]) if maximize else float(trial["value"]),
            int(trial["trial_number"]),
        ),
    )
    selected_trials = completed_trials[:top_n]
    if len(selected_trials) < top_n:
        logger.warning(
            f"Requested final_test_top_n={top_n}, but only {len(selected_trials)} completed HPO trial(s) exist."
        )
    return selected_trials


def make_cross_seed_validation_group_name(scenario, role):
    label_part = make_label_budget_name(
        scenario.label_sampling_mode,
        scenario.labeled_fraction,
        scenario.labeled_per_class,
    )
    return "_".join(
        [scenario.label_sampling_mode, label_part, scenario.loss, scenario.miner, role, "cross_seed"]
    )


def make_cross_seed_final_args(base_args, base_ssl_config, scenario, group_name, scenario_group):
    """Build a baseline-seeded full-development run for a cross-seed winner."""

    final_args = copy.deepcopy(base_args)
    final_args.loss = scenario.loss
    final_args.miner = scenario.miner
    final_args.comparison_seeds = None
    final_args.cross_seed_validation_seeds = [int(item.seed) for item in scenario_group]
    final_args.save_dir = Path(base_args.save_dir) / "cross_seed_train_val" / group_name
    final_ssl_config = replace(
        base_ssl_config,
        label_sampling_mode=scenario.label_sampling_mode,
        labeled_fraction=float(scenario.labeled_fraction),
        labeled_per_class=scenario.labeled_per_class,
    )
    semi_supervised.validate_ssl_config(final_ssl_config)
    config_dir = Path("logs") / base_args.save_dir / "study_replay_grid" / "ssl_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{group_name}_final.json"
    write_json(config_path, final_ssl_config.to_dict())
    final_args.ssl_config = config_path
    final_args.support_seed = final_ssl_config.support_seed
    return final_args


def run_study_dir_hparam_evaluation(base_args, hparam_config, study_result, role, summary_stem=None):
    """Replay selected HPO params from a study in the requested study-dir mode."""

    study_dir_mode = get_study_dir_mode(base_args)
    if summary_stem is None:
        summary_stem = get_study_dir_summary_stem(base_args)
    if study_dir_mode == STUDY_DIR_MODE_FINAL_TRAIN:
        return run_final_from_best_hparam(
            base_args,
            hparam_config,
            study_result,
            role=role,
            summary_stem=summary_stem,
        )
    if study_dir_mode == STUDY_DIR_MODE_TRAIN_VAL:
        return run_train_val_from_best_hparam(
            base_args,
            hparam_config,
            study_result,
            role=role,
            summary_stem=summary_stem,
        )
    if study_dir_mode == STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL:
        raise ValueError("study_dir_mode='cross_seed_train_val' requires a comparison_seeds replay grid")
    raise ValueError(f"study_dir_mode must be one of {STUDY_DIR_MODES}: {study_dir_mode}")

def get_study_dir_mode(args):
    return getattr(args, "study_dir_mode", STUDY_DIR_MODE_FINAL_TRAIN)

def get_study_dir_summary_stem(args):
    if get_study_dir_mode(args) == STUDY_DIR_MODE_TRAIN_VAL:
        return "train_val_evaluation"
    if get_study_dir_mode(args) == STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL:
        return "cross_seed_train_val_evaluation"
    return "final_evaluation"

def validate_final_study_hparam_repository(hparam_config, study_result):
    """Ensure replayed HPO params do not override requested data conditions."""

    configured_keys = set(getattr(hparam_config, "spaces", {}) or {})
    trial_param_keys = set(getattr(study_result, "best_params", None) or {})
    for trial in getattr(study_result, "completed_trials", None) or []:
        trial_param_keys.update((trial.get("params") or {}).keys())

    forbidden_keys = sorted(
        (configured_keys | trial_param_keys)
        & FINAL_STUDY_REPOSITORY_FORBIDDEN_HPARAM_KEYS
    )
    if forbidden_keys:
        raise ValueError(
            "Cannot replay this HPO study as a hyperparameter repository while varying data settings. "
            f"The study contains data/split/search-condition parameters: {forbidden_keys}."
        )

def load_hparam_study_for_final_test(study_dir):
    """Load saved base args, HPO config, and completed trials from a study directory."""

    study_dir = resolve_existing_hparam_study_dir(study_dir)
    study_config_path = study_dir / "study_config.json"
    if not study_config_path.exists():
        raise FileNotFoundError(f"Study config not found: {study_config_path}")

    with study_config_path.open(encoding="utf-8") as config_file:
        study_config = json.load(config_file)

    raw_base_args = study_config.get("base_args")
    if not isinstance(raw_base_args, dict):
        raise ValueError(f"Study config must contain a base_args object: {study_config_path}")
    raw_hparam_config = study_config.get("hparam_config")
    if not isinstance(raw_hparam_config, dict):
        raise ValueError(f"Study config must contain a hparam_config object: {study_config_path}")

    base_args = make_base_args_from_study_config(raw_base_args)
    hparam_config = HParamSearchConfig(**raw_hparam_config)
    study_result = load_hparam_study_result_from_artifacts(study_dir, study_config, hparam_config)
    return base_args, hparam_config, study_result

def resolve_existing_hparam_study_dir(study_dir):
    path = Path(study_dir)
    if path.exists():
        return path

    logs_path = Path("logs") / path
    if logs_path.exists():
        return logs_path

    raise FileNotFoundError(f"HPO study directory not found: {path} or {logs_path}")

def make_base_args_from_study_config(raw_base_args):
    args = parser.parse_args([])
    for name, value in raw_base_args.items():
        setattr(args, name, value)
    normalize_backbone_tuning_args(args)
    resolve_hparam_seed(args)
    resolve_data_split_seed(args)
    resolve_support_seed(args)
    return args

def load_hparam_study_result_from_artifacts(study_dir, study_config, hparam_config):
    study_name = (
        study_config.get("resolved_study_name")
        or hparam_config.study_name
        or Path(study_dir).name
    )
    trials_csv = Path(study_dir) / "trials.csv"
    trials_jsonl = Path(study_dir) / "trials.jsonl"

    if trials_jsonl.exists():
        completed_trials = load_completed_hparam_trials_jsonl(trials_jsonl)
        if not completed_trials:
            raise ValueError(f"No completed HPO trials found in {trials_jsonl}")
        best_trial = select_best_completed_hparam_trial(completed_trials, hparam_config.direction)
        return HParamStudyResult(
            study_name=study_name,
            study_dir=Path(study_dir),
            trials_csv=trials_csv,
            trials_jsonl=trials_jsonl,
            best_trial_number=best_trial["trial_number"],
            best_value=best_trial["value"],
            best_params=best_trial["params"],
            best_user_attrs=best_trial["user_attrs"],
            completed_trials=sort_completed_hparam_trials_by_value(completed_trials),
        )

    return load_hparam_study_result_from_optuna_storage(
        study_dir,
        study_name,
        study_config.get("resolved_storage") or resolve_optuna_storage(hparam_config.storage, study_dir),
        trials_csv,
        trials_jsonl,
    )

def load_completed_hparam_trials_jsonl(trials_jsonl):
    completed_trials = []
    with Path(trials_jsonl).open(encoding="utf-8") as trials_file:
        for line in trials_file:
            if not line.strip():
                continue
            trial = json.loads(line)
            if trial.get("state") != "COMPLETE" or trial.get("value") is None:
                continue
            completed_trials.append(
                {
                    "trial_number": int(trial["number"]),
                    "value": float(trial["value"]),
                    "params": trial.get("params") or {},
                    "user_attrs": trial.get("user_attrs") or {},
                }
            )
    return completed_trials

def sort_completed_hparam_trials_by_value(completed_trials):
    return sorted(
        completed_trials,
        key=lambda trial: (-float(trial["value"]), int(trial["trial_number"])),
    )

def select_best_completed_hparam_trial(completed_trials, direction):
    if direction == "minimize":
        return min(completed_trials, key=lambda trial: (float(trial["value"]), int(trial["trial_number"])))
    return min(completed_trials, key=lambda trial: (-float(trial["value"]), int(trial["trial_number"])))

def load_hparam_study_result_from_optuna_storage(study_dir, study_name, storage, trials_csv, trials_jsonl):
    try:
        import optuna
    except ImportError as exc:
        raise FileNotFoundError(
            f"Trials summary not found at {trials_jsonl}, and Optuna is not installed for storage fallback."
        ) from exc

    study = optuna.load_study(study_name=study_name, storage=storage)
    write_trials_summary(study, trials_csv, trials_jsonl)
    logger.info(f"Loaded HPO study {study_name!r} from {study_dir} for final-test evaluation.")
    return make_hparam_study_result(study, study_name, Path(study_dir), trials_csv, trials_jsonl)

def get_final_test_top_n(args):
    top_n = int(getattr(args, "final_test_top_n", 1))
    if top_n <= 0:
        raise ValueError("final_test_top_n must be positive")
    return top_n

def get_final_test_trial_numbers(args):
    trial_numbers = getattr(args, "final_test_trial_numbers", None)
    if trial_numbers is None:
        return None
    if not trial_numbers:
        raise ValueError("final_test_trial_numbers must include at least one trial number when provided")
    return [int(trial_number) for trial_number in trial_numbers]

def should_run_final_hparam_evaluation(args):
    return (
        bool(getattr(args, "final_test_after_hpo", False))
        or get_final_test_top_n(args) > 1
        or get_final_test_trial_numbers(args) is not None
    )

def run_final_from_best_hparam(base_args, hparam_config, study_result, role, summary_stem="final_evaluation"):
    """Train selected HPO configuration(s) on all development data and test them."""

    trial_numbers = get_final_test_trial_numbers(base_args)
    if trial_numbers is not None:
        return run_final_from_selected_hparams(
            base_args,
            hparam_config,
            study_result,
            role,
            trial_numbers=trial_numbers,
            summary_stem=summary_stem,
        )

    top_n = get_final_test_top_n(base_args)
    if top_n > 1:
        return run_final_from_top_hparams(
            base_args,
            hparam_config,
            study_result,
            role,
            top_n=top_n,
            summary_stem=summary_stem,
        )

    return run_single_final_from_hparam(
        base_args,
        hparam_config,
        study_result,
        role,
        summary_stem=summary_stem,
    )

def run_train_val_from_best_hparam(base_args, hparam_config, study_result, role, summary_stem="train_val_evaluation"):
    """Run selected HPO configuration(s) with validation selection and final testing."""

    trial_numbers = get_final_test_trial_numbers(base_args)
    if trial_numbers is not None:
        return run_train_val_from_selected_hparams(
            base_args,
            hparam_config,
            study_result,
            role,
            trial_numbers=trial_numbers,
            summary_stem=summary_stem,
        )

    top_n = get_final_test_top_n(base_args)
    if top_n > 1:
        return run_train_val_from_top_hparams(
            base_args,
            hparam_config,
            study_result,
            role,
            top_n=top_n,
            summary_stem=summary_stem,
        )

    best_trial = {
        "trial_number": study_result.best_trial_number,
        "value": study_result.best_value,
        "params": study_result.best_params,
        "user_attrs": study_result.best_user_attrs or {},
    }
    _, final_result = run_train_val_candidates(
        base_args,
        hparam_config,
        study_result,
        role,
        [(best_trial, summary_stem)],
    )
    return final_result

def get_completed_hparam_trials(study_result, role):
    completed_trials = list(getattr(study_result, "completed_trials", None) or [])
    if not completed_trials and study_result.best_params is not None:
        completed_trials = [
            {
                "trial_number": study_result.best_trial_number,
                "value": study_result.best_value,
                "params": study_result.best_params,
                "user_attrs": study_result.best_user_attrs or {},
            }
        ]
    if not completed_trials:
        raise ValueError(f"No completed {role or 'model'} HPO trial is available for final retraining")
    return completed_trials

def run_train_val_from_selected_hparams(
    base_args,
    hparam_config,
    study_result,
    role,
    trial_numbers,
    summary_stem="train_val_evaluation",
):
    """Rank selected trials on validation and fully retrain/test the winner."""

    completed_trials = get_completed_hparam_trials(study_result, role)
    trials_by_number = {int(trial["trial_number"]): trial for trial in completed_trials}
    missing_trials = [trial_number for trial_number in trial_numbers if trial_number not in trials_by_number]
    if missing_trials:
        available_trials = sorted(trials_by_number)
        raise ValueError(
            f"Selected train/validation trial(s) are not completed in study {study_result.study_name!r}: "
            f"{missing_trials}. Available completed trial numbers: {available_trials}"
        )

    selected_trials = []
    for trial_number in trial_numbers:
        trial = trials_by_number[int(trial_number)]
        trial_summary_stem = f"{summary_stem}_trial_{int(trial_number):04d}"
        selected_trials.append((trial, trial_summary_stem))

    evaluated, best_result = run_train_val_candidates(
        base_args,
        hparam_config,
        study_result,
        role,
        selected_trials,
    )

    write_hparam_selected_final_evaluation_summary(
        study_result=study_result,
        evaluated=evaluated,
        role=role,
        trial_numbers=trial_numbers,
        summary_stem=summary_stem,
    )
    return best_result

def run_train_val_from_top_hparams(base_args, hparam_config, study_result, role, top_n, summary_stem="train_val_evaluation"):
    """Rank the top-N trials on validation and fully retrain/test the winner."""

    completed_trials = get_completed_hparam_trials(study_result, role)
    maximize = getattr(hparam_config, "direction", "maximize") == "maximize"
    completed_trials = sorted(
        completed_trials,
        key=lambda trial: (
            -float(trial["value"]) if maximize else float(trial["value"]),
            int(trial["trial_number"]),
        ),
    )
    selected_trials = completed_trials[:top_n]
    if len(selected_trials) < top_n:
        logger.warning(
            f"Requested final_test_top_n={top_n}, but only {len(selected_trials)} completed HPO trial(s) exist."
        )

    trials_with_summary_stems = []
    for trial_index, trial in enumerate(selected_trials):
        trial_number = trial["trial_number"]
        trial_summary_stem = summary_stem if trial_index == 0 else f"{summary_stem}_trial_{trial_number:04d}"
        trials_with_summary_stems.append((trial, trial_summary_stem))

    evaluated, best_result = run_train_val_candidates(
        base_args,
        hparam_config,
        study_result,
        role,
        trials_with_summary_stems,
    )

    write_hparam_top_final_evaluation_summary(
        study_result=study_result,
        evaluated=evaluated,
        role=role,
        requested_top_n=top_n,
        summary_stem=summary_stem,
    )
    return best_result


def run_train_val_candidates(base_args, hparam_config, study_result, role, trials_with_summary_stems):
    """Rank candidates on validation, then fully retrain/test only the winner."""

    evaluated = []
    best_item = None
    best_value = None
    selection_metric = getattr(base_args, "selection_metric", SELECTION_METRIC_MAP_AT_R)
    for trial, trial_summary_stem in trials_with_summary_stems:
        trial_study_result = make_study_result_for_completed_trial(study_result, trial)
        train_result = run_single_train_val_from_hparam(
            base_args,
            hparam_config,
            trial_study_result,
            role,
            summary_stem=trial_summary_stem,
        )
        candidate_value = get_selection_metric_value(
            selection_metric,
            train_result.best_valid_precision_at_1,
            train_result.best_valid_mean_average_precision_at_r,
        )
        if candidate_value is None or not math.isfinite(float(candidate_value)):
            raise ValueError(
                f"Trial {trial['trial_number']} has no finite {selection_metric} validation result"
            )
        item = {
            "trial": trial,
            "summary_stem": trial_summary_stem,
            "final_result": train_result,
            "validation_result": train_result,
            "selected_for_test": False,
            "validation_selection_metric": selection_metric,
            "validation_selection_value": float(candidate_value),
        }
        evaluated.append(item)
        if best_value is None or candidate_value > best_value:
            best_value = candidate_value
            best_item = item

    if best_item is None:
        raise ValueError("No train_val candidate is available for final retraining")

    selected_study_result = make_validation_selected_study_result(
        study_result,
        best_item["trial"],
        [best_item["validation_result"]],
        selection_metric,
    )
    logger.info(
        "train_val selected trial "
        f"{best_item['trial']['trial_number']} for full retraining using "
        f"{selection_metric}={best_value:.6f}"
    )
    final_result = run_single_final_from_hparam(
        base_args,
        hparam_config,
        selected_study_result,
        role,
        summary_stem=f"{best_item['summary_stem']}_final",
    )
    best_item["final_result"] = final_result
    best_item["selected_for_test"] = True
    return evaluated, final_result


def make_validation_selected_study_result(
    study_result,
    trial,
    validation_results,
    selection_metric,
    validation_result_metadata=None,
):
    """Attach replay validation evidence to a trial for final epoch planning."""

    if not validation_results:
        raise ValueError("At least one validation result is required for final epoch planning")
    validation_result_metadata = validation_result_metadata or [{} for _ in validation_results]
    if len(validation_result_metadata) != len(validation_results):
        raise ValueError("validation_result_metadata must match validation_results")
    serialized_results = []
    for result, metadata in zip(validation_results, validation_result_metadata):
        serialized_result = result_to_dict(result)
        serialized_result.update(metadata)
        serialized_results.append(serialized_result)
    mean_precision = float(
        sum(result.best_valid_precision_at_1 for result in validation_results) / len(validation_results)
    )
    mean_map = float(
        sum(result.best_valid_mean_average_precision_at_r for result in validation_results)
        / len(validation_results)
    )
    selection_values = [
        get_selection_metric_value(
            selection_metric,
            result.best_valid_precision_at_1,
            result.best_valid_mean_average_precision_at_r,
        )
        for result in validation_results
    ]
    if any(value is None or not math.isfinite(float(value)) for value in selection_values):
        raise ValueError(f"Validation results must contain finite {selection_metric} values")
    mean_selection_value = float(sum(selection_values) / len(selection_values))
    best_user_attrs = dict(trial.get("user_attrs") or {})
    best_user_attrs.update(
        {
            "best_valid_precision_at_1": mean_precision,
            "best_valid_mean_average_precision_at_r": mean_map,
            "validation_selection_metric": selection_metric,
            "mean_validation_selection_value": mean_selection_value,
            "validation_replay_results": serialized_results,
            "validation_replay_source": (
                "cross_seed_validation_replays"
                if len(validation_results) > 1
                else "single_validation_replay"
            ),
        }
    )
    return replace(
        study_result,
        best_trial_number=trial["trial_number"],
        best_value=trial["value"],
        best_params=trial["params"],
        best_user_attrs=best_user_attrs,
    )

def run_final_from_selected_hparams(
    base_args,
    hparam_config,
    study_result,
    role,
    trial_numbers,
    summary_stem="final_evaluation",
):
    """Run final-test evaluation for explicitly selected completed HPO trial numbers."""

    completed_trials = get_completed_hparam_trials(study_result, role)
    trials_by_number = {int(trial["trial_number"]): trial for trial in completed_trials}
    missing_trials = [trial_number for trial_number in trial_numbers if trial_number not in trials_by_number]
    if missing_trials:
        available_trials = sorted(trials_by_number)
        raise ValueError(
            f"Selected final-test trial(s) are not completed in study {study_result.study_name!r}: "
            f"{missing_trials}. Available completed trial numbers: {available_trials}"
        )

    evaluated = []
    for trial_number in trial_numbers:
        trial = trials_by_number[int(trial_number)]
        trial_summary_stem = f"{summary_stem}_trial_{int(trial_number):04d}"
        final_result = run_single_final_from_hparam(
            base_args,
            hparam_config,
            make_study_result_for_completed_trial(study_result, trial),
            role,
            summary_stem=trial_summary_stem,
        )
        evaluated.append(
            {
                "trial": trial,
                "summary_stem": trial_summary_stem,
                "final_result": final_result,
            }
        )

    write_hparam_selected_final_evaluation_summary(
        study_result=study_result,
        evaluated=evaluated,
        role=role,
        trial_numbers=trial_numbers,
        summary_stem=summary_stem,
    )
    return evaluated[0]["final_result"]

def run_final_from_top_hparams(base_args, hparam_config, study_result, role, top_n, summary_stem="final_evaluation"):
    """Run final-test evaluation for the top-N completed HPO trials by objective value."""

    completed_trials = get_completed_hparam_trials(study_result, role)
    completed_trials = sorted(
        completed_trials,
        key=lambda trial: (-float(trial["value"]), int(trial["trial_number"])),
    )
    selected_trials = completed_trials[:top_n]
    if len(selected_trials) < top_n:
        logger.warning(
            f"Requested final_test_top_n={top_n}, but only {len(selected_trials)} completed HPO trial(s) exist."
        )

    evaluated = []
    for trial_index, trial in enumerate(selected_trials):
        trial_study_result = make_study_result_for_completed_trial(study_result, trial)
        trial_number = trial["trial_number"]
        trial_summary_stem = summary_stem if trial_index == 0 else f"{summary_stem}_trial_{trial_number:04d}"
        final_result = run_single_final_from_hparam(
            base_args,
            hparam_config,
            trial_study_result,
            role,
            summary_stem=trial_summary_stem,
        )
        evaluated.append(
            {
                "trial": trial,
                "summary_stem": trial_summary_stem,
                "final_result": final_result,
            }
        )

    write_hparam_top_final_evaluation_summary(
        study_result=study_result,
        evaluated=evaluated,
        role=role,
        requested_top_n=top_n,
        summary_stem=summary_stem,
    )
    return evaluated[0]["final_result"]

def make_study_result_for_completed_trial(study_result, trial):
    return replace(
        study_result,
        best_trial_number=trial["trial_number"],
        best_value=trial["value"],
        best_params=trial["params"],
        best_user_attrs=trial.get("user_attrs") or {},
    )

def run_single_final_from_hparam(base_args, hparam_config, study_result, role, summary_stem="final_evaluation"):
    """Train one HPO configuration on all development data and test it."""

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

def run_single_train_val_from_hparam(
    base_args,
    hparam_config,
    study_result,
    role,
    summary_stem="train_val_evaluation",
):
    """Train one HPO configuration with validation and without touching D_test."""

    role = role or "model"
    if study_result.best_params is None:
        raise ValueError(f"No completed {role} HPO trial is available for train/validation replay")

    train_args, train_ssl_config = make_args_and_ssl_config_from_params(base_args, study_result.best_params)
    train_args.hparam_config_resolved = hparam_config.to_dict()
    train_args.hparam_params = study_result.best_params
    train_args.hparam_replay_from_study = study_result.study_name
    train_args.hparam_replay_trial_number = study_result.best_trial_number
    train_args.final_full_train = False
    train_args.cv_k = 1
    train_args.evaluate_test = False
    train_args.skip_test_during_hpo = True
    train_args.save_dir = Path(base_args.save_dir) / "train_val" / role

    train_result = run_experiment(train_args, train_ssl_config)
    write_hparam_train_val_evaluation_summary(study_result, train_result, role, summary_stem=summary_stem)
    return train_result


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()
