import argparse
import copy
import csv
import json
import os
import multiprocessing as mp
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.miners as miners
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import semi_supervised
import utils
from retrieval_model import DinoWrapper

DATASETS = ["Cars196", "CUB", "INaturalist2018", "StanfordOnlineProducts", "CIFAR10"]

ALL_LOSSES = [
    "AngularLoss",
    "ArcFaceLoss",
    "BaseMetricLossFunction",
    "CircleLoss",
    "ContrastiveLoss",
    "CosFaceLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "GenericPairLoss",
    "HistogramLoss",
    "InstanceLoss",
    "IntraPairVarianceLoss",
    "LargeMarginSoftmaxLoss",
    "GeneralizedLiftedStructureLoss",
    "LiftedStructureLoss",
    "ManifoldLoss",
    "MarginLoss",
    "WeightRegularizerMixin",
    "MultiSimilarityLoss",
    "MultipleLosses",
    "NPairsLoss",
    "NCALoss",
    "NormalizedSoftmaxLoss",
    "NTXentLoss",
    "P2SGradLoss",
    "PNPLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "RankedListLoss",
    "SelfSupervisedLoss",
    "SignalToNoiseRatioContrastiveLoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletMarginLoss",
    "TupletMarginLoss",
    "VICRegLoss",
]

CLASSIFICATION_LOSSES = [
    "ArcFaceLoss",
    "CosFaceLoss",
    "LargeMarginSoftmaxLoss",
    "WeightRegularizerMixin",
    "NormalizedSoftmaxLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
]

ALL_MINERS = [
    "no_miner",
    "AngularMiner",
    "BatchEasyHardMiner",
    "BatchHardMiner",
    "DistanceWeightedMiner",
    "HDCMiner",
    "MultiSimilarityMiner",
    "PairMarginMiner",
    "TripletMarginMiner",
    "UniformHistogramMiner",
]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=1e-6, help="LR")
parser.add_argument("--classifier_lr", type=float, default=1.0, help="classifier LR (only for classification losses)")
parser.add_argument("--sampler_m", type=int, default=4, help="M value for MPerClassSampler")
parser.add_argument("--dataset", type=utils.normalize_dataset_name, default="Cars196", choices=DATASETS, help="dataset")
parser.add_argument("--dino_size", type=str, default="l", choices=["s", "b", "l", "g"], help="which Dino to use")
parser.add_argument("--loss", type=str, default="MultiSimilarityLoss", choices=ALL_LOSSES, help="loss")
parser.add_argument("--miner", type=str, default="MultiSimilarityMiner", choices=ALL_MINERS, help="miner")
parser.add_argument("--feat_dim", type=int, default=None, help="Output dimensionality. Set to None to use CLS")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="device")
parser.add_argument("--optim", type=str, default="adam", choices=["adam", "rmsprop"], help="optimizer")
parser.add_argument("--seed", type=int, default=7, help="random seed for dataset splits, sampling, and training")
parser.add_argument("--epochs", type=int, default=100, help="maximum number of training epochs")
parser.add_argument("--patience", type=int, default=3, help="early-stopping patience after SSL warmup")
parser.add_argument("--cv_k", type=int, default=4, help="number of cross-validation folds. Set to 1 to disable CV")
parser.add_argument(
    "--cv_mode",
    type=str,
    default="group_kfold",
    choices=utils.CV_MODES,
    help="sklearn cross-validation splitter to use when cv_k > 1",
)
parser.add_argument(
    "--val_mode",
    type=str,
    default=utils.VAL_MODE_ALL,
    choices=utils.VAL_MODES,
    help=(
        "validation data mode. 'all' keeps the current behavior and uses all validation samples; "
        "'match_train' downsamples validation to roughly the labeled/fractioned training size."
    ),
)
parser.add_argument("--num_workers", type=int, default=1, help="DataLoader worker count for training/evaluation")
parser.add_argument(
    "--dataloader_start_method",
    type=str,
    default="spawn",
    choices=utils.DATALOADER_START_METHODS,
    help="DataLoader multiprocessing start method for CPU runs or CUDA runs with zero workers.",
)
parser.add_argument(
    "--ssl_config",
    type=Path,
    default="configs/ssl_faiss_knn.json",
    help="path to a JSON semi-supervised config. Omit to disable SSL.",
)
parser.add_argument(
    "--hparam_config",
    type=Path,
    default=r"C:\Users\Sebastian\PycharmProjects\metric-learning\configs\k_shot_cars.json",
    help="path to a JSON Optuna hyperparameter search config. Omit to run a single training job.",
)
parser.add_argument(
    "--compare_supervised_ssl",
    action="store_true",
    help=(
        "run two separate Optuna searches with identical budget: a supervised baseline on the labeled "
        "subset only and an SSL run on the same labeled subset plus unlabeled data"
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    default="supervised",
    choices=["supervised", "ssl"],
    help="training mode. supervised uses only the labeled split; ssl uses the labeled split plus unlabeled data.",
)
parser.add_argument(
    "--skip_test_during_hpo",
    action="store_true",
    default=True,
    help="do not evaluate D_test inside Optuna trials; use a final retraining run for test evaluation",
)
parser.add_argument(
    "--label_budget_grid",
    type=float,
    nargs="*",
    default=[1],
    help="outer experiment grid over SSL labeled_fraction values, for example 0.01 0.05 0.10 0.25 0.50",
)
parser.add_argument(
    "--loss_miner_grid",
    type=str,
    nargs="*",
    default=["MultiSimilarityLoss:MultiSimilarityMiner", "TripletMarginLoss:TripletMarginMiner"],
    metavar="LOSS:MINER",
    help=(
        "outer experiment grid over paired loss/miner choices, for example "
        "MultiSimilarityLoss:MultiSimilarityMiner TripletMarginLoss:TripletMarginMiner"
    ),
)
parser.add_argument(
    "--comparison_seeds",
    type=int,
    nargs="*",
    default=None,
    help="outer experiment grid over split/training seeds, for example 0 1 2 3 4",
)
parser.add_argument(
    "--ssl_label_sampling_modes",
    type=str,
    nargs="*",
    default=["class_subset_k_shot"],
    choices=sorted(semi_supervised.LABEL_SAMPLING_MODES),
    help="outer experiment grid over labeled-sample selection modes",
)
parser.add_argument(
    "--save_dir",
    type=Path,
    default=Path("default"),
    help="name of directory in which to save the logs, under logs/save_dir",
)


@dataclass(frozen=True)
class TrainingResult:
    log_dir: Path
    metrics_csv: Path
    best_valid_precision_at_1: float
    best_valid_mean_average_precision_at_r: float
    test_precision_at_1: float | None
    test_mean_average_precision_at_r: float | None
    final_train_loss: float | None
    last_epoch: int
    global_step: int
    cv_k: int = 1
    cv_mode: str | None = None
    cv_fold: int | None = None
    fold_results: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class HParamSearchConfig:
    enabled: bool = True
    n_trials: int = 20
    timeout: int | None = None
    direction: str = "maximize"
    metric: str = "best_valid_precision_at_1"
    study_name: str | None = None
    study_dir: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    sampler: str = "tpe"
    sampler_params: dict[str, Any] = field(default_factory=dict)
    pruner: str = "none"
    pruner_params: dict[str, Any] = field(default_factory=dict)
    spaces: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class HParamStudyResult:
    study_name: str
    study_dir: Path
    trials_csv: Path
    trials_jsonl: Path
    best_trial_number: int | None
    best_value: float | None
    best_params: dict[str, Any] | None
    best_user_attrs: dict[str, Any] | None


@dataclass(frozen=True)
class ComparisonScenario:
    name: str
    labeled_fraction: float
    labeled_per_class: int | None
    seed: int
    label_sampling_mode: str
    loss: str
    miner: str
    ssl_config_path: Path


OBJECTIVE_METRICS = {
    "best_valid_precision_at_1",
    "best_valid_mean_average_precision_at_r",
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
    "final_train_loss",
}

COMPARISON_FORBIDDEN_HPARAM_KEYS = {
    "dataset",
    "mode",
    "seed",
    "cv_k",
    "cv_mode",
    "val_mode",
    "ssl.labeled_fraction",
    "ssl_config.labeled_fraction",
    "ssl.label_sampling_mode",
    "ssl_config.label_sampling_mode",
    "ssl.max_unlabeled_samples",
    "ssl_config.max_unlabeled_samples",
    "ssl.seed",
    "ssl_config.seed",
    "ssl.method",
    "ssl_config.method",
}

SUPERVISED_SPLIT_SSL_HPARAM_KEYS = {
    "ssl.labeled_per_class",
    "ssl_config.labeled_per_class",
}


def main():
    args = parser.parse_args()
    hparam_config = load_hparam_config(args.hparam_config)
    if args.compare_supervised_ssl:
        run_supervised_ssl_comparison(args, hparam_config)
        return

    if has_outer_comparison_grid(args):
        run_single_method_grid(args, hparam_config)
        return

    if hparam_config is not None and hparam_config.enabled:
        hparam_config = make_standalone_hparam_config(args, hparam_config)
        run_hparam_search(args, hparam_config)
        return

    if hparam_config is not None:
        args.hparam_config_resolved = hparam_config.to_dict()
    ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    ssl_config = resolve_mode_ssl_config(args, ssl_config)
    run_experiment(args, ssl_config)


def run_experiment(args, ssl_config, optuna_trial=None, optuna_metric=None):
    if args.cv_k > 1:
        return run_cross_validation(args, ssl_config, optuna_trial=optuna_trial, optuna_metric=optuna_metric)
    return run_training(args, ssl_config, optuna_trial=optuna_trial, optuna_metric=optuna_metric)


def run_supervised_ssl_comparison(args, hparam_config):
    base_ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    validate_comparison_setup(args, hparam_config, base_ssl_config)

    scenarios = make_comparison_scenarios(args, base_ssl_config)
    grid_results = []
    for scenario in scenarios:
        scenario_args = copy.deepcopy(args)
        scenario_args.seed = scenario.seed
        scenario_args.loss = scenario.loss
        scenario_args.miner = scenario.miner
        scenario_args.ssl_config = scenario.ssl_config_path
        if len(scenarios) > 1:
            scenario_args.save_dir = Path(args.save_dir) / scenario.name

        scenario_ssl_config = semi_supervised.load_ssl_config(scenario.ssl_config_path, default_seed=scenario.seed)
        grid_results.append(
            run_single_supervised_ssl_comparison(
                args=scenario_args,
                hparam_config=hparam_config,
                ssl_config=scenario_ssl_config,
                scenario=scenario,
            )
        )

    if len(scenarios) > 1 or has_outer_comparison_grid(args):
        write_comparison_grid_summary(Path("logs") / args.save_dir / "comparison_grid", grid_results)


def run_single_supervised_ssl_comparison(args, hparam_config, ssl_config, scenario):
    comparison_dir = Path("logs") / args.save_dir / "supervised_ssl_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    supervised_args = copy.deepcopy(args)
    supervised_args.mode = "supervised"
    supervised_args.skip_test_during_hpo = True

    ssl_args = copy.deepcopy(args)
    ssl_args.mode = "ssl"
    ssl_args.skip_test_during_hpo = True

    supervised_hparam_config = make_comparison_hparam_config(hparam_config, role="supervised")
    ssl_hparam_config = make_comparison_hparam_config(hparam_config, role="ssl")

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

    supervised_study = run_hparam_search(supervised_args, supervised_hparam_config)
    ssl_study = run_hparam_search(ssl_args, ssl_hparam_config)

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
    label_budgets = args.label_budget_grid
    if label_budgets is None:
        label_budgets = [base_ssl_config.labeled_fraction]
        use_labeled_per_class = base_ssl_config.labeled_per_class
    else:
        if not label_budgets:
            raise ValueError("--label_budget_grid must include at least one value when provided")
        use_labeled_per_class = None

    seeds = args.comparison_seeds
    if seeds is None:
        seeds = [base_ssl_config.seed]
    elif not seeds:
        raise ValueError("--comparison_seeds must include at least one value when provided")

    label_sampling_modes = args.ssl_label_sampling_modes
    if label_sampling_modes is None:
        label_sampling_modes = [base_ssl_config.label_sampling_mode]
    elif not label_sampling_modes:
        raise ValueError("--ssl_label_sampling_modes must include at least one value when provided")

    loss_miner_pairs = get_loss_miner_pairs(args)
    include_loss_miner_in_name = args.loss_miner_grid is not None

    config_dir = Path("logs") / args.save_dir / grid_dir_name / "ssl_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    scenario_names = set()
    for label_sampling_mode in label_sampling_modes:
        for labeled_fraction in label_budgets:
            if not (0 < labeled_fraction <= 1):
                raise ValueError(f"label budget must be in (0, 1], got {labeled_fraction}")
            for loss_name, miner_name in loss_miner_pairs:
                for seed in seeds:
                    scenario_labeled_per_class = use_labeled_per_class
                    if label_sampling_mode == "class_subset_k_shot" and scenario_labeled_per_class is None:
                        scenario_labeled_per_class = base_ssl_config.labeled_per_class or 1
                    scenario_ssl_config = replace(
                        base_ssl_config,
                        label_sampling_mode=label_sampling_mode,
                        labeled_fraction=float(labeled_fraction),
                        labeled_per_class=scenario_labeled_per_class,
                        seed=int(seed),
                    )
                    semi_supervised.validate_ssl_config(scenario_ssl_config)
                    scenario_name = make_scenario_name(
                        scenario_ssl_config,
                        loss=loss_name if include_loss_miner_in_name else None,
                        miner=miner_name if include_loss_miner_in_name else None,
                    )
                    if scenario_name in scenario_names:
                        raise ValueError(
                            f"Duplicate outer-grid scenario name {scenario_name!r}. "
                            "Check for duplicate label budgets, seeds, label sampling modes, "
                            "or loss/miner pairs."
                        )
                    scenario_names.add(scenario_name)
                    config_path = config_dir / f"{scenario_name}.json"
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


def has_outer_comparison_grid(args):
    return (
        args.label_budget_grid is not None
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


def make_scenario_name(ssl_config, loss=None, miner=None):
    if ssl_config.label_sampling_mode == "class_subset_k_shot":
        label_part = (
            f"label_{format_float_token(ssl_config.labeled_fraction)}"
            f"_k_{ssl_config.labeled_per_class}"
        )
    elif ssl_config.labeled_per_class is None:
        label_part = f"label_{format_float_token(ssl_config.labeled_fraction)}"
    else:
        label_part = f"per_class_{ssl_config.labeled_per_class}"
    parts = [ssl_config.label_sampling_mode, label_part]
    if loss is not None and miner is not None:
        parts.extend([loss, miner])
    parts.extend(["seed", str(ssl_config.seed)])
    return "_".join(parts)


def format_float_token(value):
    return f"{value:g}".replace(".", "p")


def is_supervised_mode(args):
    return getattr(args, "mode", "supervised") == "supervised"


def resolve_mode_ssl_config(args, ssl_config):
    if is_supervised_mode(args):
        return make_supervised_split_config(ssl_config)
    if not ssl_config.enabled:
        raise ValueError("--mode ssl requires an enabled --ssl_config")
    return ssl_config


def run_single_method_grid(args, hparam_config):
    base_ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    validate_single_method_grid_setup(args, hparam_config)

    scenarios = make_comparison_scenarios(args, base_ssl_config, grid_dir_name="experiment_grid")
    grid_results = []
    for scenario in scenarios:
        scenario_args = copy.deepcopy(args)
        scenario_args.seed = scenario.seed
        scenario_args.loss = scenario.loss
        scenario_args.miner = scenario.miner
        scenario_args.ssl_config = scenario.ssl_config_path
        scenario_args.save_dir = Path(args.save_dir) / scenario.name
        grid_results.append(run_single_method_scenario(scenario_args, hparam_config, scenario))

    write_single_method_grid_summary(Path("logs") / args.save_dir / "experiment_grid", grid_results)


def validate_single_method_grid_setup(args, hparam_config):
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
        return {
            "method": method,
            "scenario": scenario,
            "study": study_result,
            "result": None,
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


def run_final_from_best_hparam(base_args, hparam_config, study_result, role):
    if study_result.best_params is None:
        raise ValueError(f"No completed {role} HPO trial is available for final retraining")

    final_args, final_ssl_config = make_args_and_ssl_config_from_params(base_args, study_result.best_params)
    final_args.hparam_config_resolved = hparam_config.to_dict()
    final_args.hparam_params = study_result.best_params
    final_args.hparam_final_from_study = study_result.study_name
    final_args.hparam_final_trial_number = study_result.best_trial_number
    final_args.evaluate_test = False
    final_args.skip_test_during_hpo = False
    final_args.save_dir = Path(base_args.save_dir) / "final" / role

    return run_experiment(final_args, final_ssl_config)


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
    else:
        row.update(
            {
                "log_dir": str(result.log_dir),
                "metrics_csv": str(result.metrics_csv),
                "best_valid_precision_at_1": result.best_valid_precision_at_1,
                "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
                "test_precision_at_1": optional_number(result.test_precision_at_1),
                "test_mean_average_precision_at_r": optional_number(result.test_mean_average_precision_at_r),
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


def write_split_manifest(log_dir, dataset_bundle, ssl_config, ssl_split):
    split_dir = Path(log_dir) / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    if ssl_split is None:
        labeled_positions = np.arange(len(dataset_bundle.train_dataset), dtype=np.int64)
        unlabeled_positions = np.array([], dtype=np.int64)
    else:
        labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)

    train_indices = get_subset_indices(dataset_bundle.train_dataset)
    val_indices = get_subset_indices(dataset_bundle.valid_dataset)

    np.save(split_dir / "labeled_positions.npy", labeled_positions)
    np.save(split_dir / "unlabeled_positions.npy", unlabeled_positions)
    np.save(split_dir / "val_indices.npy", val_indices)
    np.save(split_dir / "train_indices.npy", train_indices)
    np.save(split_dir / "labeled_indices.npy", positions_to_indices(train_indices, labeled_positions))
    np.save(split_dir / "unlabeled_indices.npy", positions_to_indices(train_indices, unlabeled_positions))

    write_json(
        split_dir / "split_info.json",
        {
            "ssl_config": ssl_config.to_dict(),
            "dataset_split": dataset_bundle.split_info,
            "train_size": len(dataset_bundle.train_dataset),
            "valid_size": len(dataset_bundle.valid_dataset),
            "num_labeled": len(labeled_positions),
            "num_unlabeled": len(unlabeled_positions),
            "labeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, labeled_positions),
            "unlabeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, unlabeled_positions),
        },
    )
    write_json(split_dir / "test_info.json", make_test_info(dataset_bundle.test_dataset))


def get_subset_indices(dataset):
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return np.arange(len(dataset), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def positions_to_indices(indices, positions):
    if len(indices) == 0 or len(positions) == 0:
        return np.array([], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)[np.asarray(positions, dtype=np.int64)]


def label_counts(labels, positions=None):
    labels = np.asarray(labels, dtype=np.int64)
    if positions is not None:
        labels = labels[np.asarray(positions, dtype=np.int64)]
    if len(labels) == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def make_test_info(test_dataset):
    labels = getattr(test_dataset, "labels", None)
    info = {
        "size": len(test_dataset),
        "dataset_type": type(test_dataset).__name__,
    }
    if labels is not None:
        info["num_classes"] = int(len(set(int(label) for label in labels)))
        info["label_counts"] = label_counts(labels)
    return info


def run_training(args, ssl_config, optuna_trial=None, optuna_metric=None, cv_fold=None):
    args.ssl_config_resolved = ssl_config.to_dict()
    validate_run_args(args, ssl_config)
    utils.seed_everything(args.seed, device=args.device)

    torch.multiprocessing.set_sharing_strategy("file_system")  # Due to annoying "RuntimeError: Too many open files."
    utils.initialize_logger(args)
    write_run_config(args, ssl_config)

    model = DinoWrapper(dino_size=args.dino_size, feat_dim=args.feat_dim)
    model = model.to(args.device)

    dataset_bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_fold=cv_fold,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
    )
    supervised_mode = is_supervised_mode(args)
    if ssl_config.enabled:
        ssl_split = semi_supervised.prepare_ssl_split(dataset_bundle.train_dataset, ssl_config)
    elif supervised_mode:
        logger.info("Training supervised baseline on the labeled subset from the SSL split config")
        ssl_split = semi_supervised.prepare_label_split(dataset_bundle.train_dataset, ssl_config)
    else:
        ssl_split = None
    if ssl_split is None:
        target_train_size = len(dataset_bundle.train_dataset)
        target_train_num_classes = len(set(int(label) for label in dataset_bundle.train_dataset.labels))
    else:
        target_train_size = len(ssl_split.labeled_positions)
        train_labels = np.asarray(dataset_bundle.train_dataset.labels, dtype=np.int64)
        target_train_num_classes = int(len(np.unique(train_labels[np.asarray(ssl_split.labeled_positions)])))
    dataset_bundle = utils.apply_validation_mode(
        dataset_bundle=dataset_bundle,
        val_mode=args.val_mode,
        target_train_size=target_train_size,
        target_train_num_classes=target_train_num_classes,
        seed=args.seed,
    )
    write_split_manifest(args.log_dir, dataset_bundle, ssl_config, ssl_split)
    static_train_loader = None
    warmup_train_loader = None
    if supervised_mode and not ssl_config.enabled:
        train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
        )
        static_train_loader = utils.make_train_loader(
            train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
        )
    elif ssl_config.enabled and ssl_config.warmup_epochs > 0:
        warmup_train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
        )
        warmup_train_loader = utils.make_train_loader(
            warmup_train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
        )
    if static_train_loader is None and ssl_config.update_mode == "once" and ssl_config.warmup_epochs == 0:
        train_dataset = semi_supervised.build_ssl_training_dataset(
            model=model,
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            device=args.device,
            config=ssl_config,
            split=ssl_split,
            start_method=args.dataloader_start_method,
        )
        static_train_loader = utils.make_train_loader(
            train_dataset,
            args.batch_size,
            args.sampler_m,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
        )
    valid_loader = utils.make_eval_loader(
        dataset_bundle.valid_dataset,
        seed=args.seed,
        num_workers=args.num_workers,
        start_method=args.dataloader_start_method,
    )
    evaluate_test = bool(getattr(args, "evaluate_test", False))
    test_loader = None
    if evaluate_test:
        test_loader = utils.make_eval_loader(
            dataset_bundle.test_dataset,
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
        )
    train_labels_mapper = dataset_bundle.train_labels_mapper

    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    if args.loss in CLASSIFICATION_LOSSES:
        # The loss is a classification loss with a learnable matrix, like ArcFaceLoss
        criterion = getattr(losses, args.loss)(len(set(dataset_bundle.train_dataset.labels)), model.feat_dim).to(args.device) # moved explicitly to set device
        is_classification = True
        if args.optim == "adam":
            classifier_optim = torch.optim.Adam(criterion.parameters(), lr=args.classifier_lr) # Why not AdamW?
        elif args.optim == "rmsprop":
            classifier_optim = torch.optim.RMSprop(criterion.parameters(), lr=args.classifier_lr)
    else:
        # The loss is a standard contrastive loss with no learnable parameter, like Contrastive or Triplet
        criterion = getattr(losses, args.loss)()
        is_classification = False

    if not is_classification:
        if args.miner == "no_miner":
            miner = None
        else:
            miner = getattr(miners, args.miner)()

    metrics_logger = utils.MetricsLogger(args.log_dir, args)
    best_model_path = args.log_dir / "best_model.pth"
    final_train_loss = None
    test_precision = None
    test_map = None

    try:
        # Evaluate off-the-shelf model
        valid_precision, valid_map = utils.evaluate(model, valid_loader, "valid", device=args.device)
        metrics_logger.log_eval("valid", valid_precision, valid_map, step=0, epoch=-1)

        patience = args.patience
        best_precision = valid_precision
        best_map = valid_map
        epochs_no_improve = 0
        global_step = 0
        last_epoch = -1
        torch.save(model.state_dict(), best_model_path)

        for num_epoch in range(args.epochs):
            last_epoch = num_epoch
            if ssl_config.enabled and num_epoch < ssl_config.warmup_epochs:
                train_loader = warmup_train_loader
            elif ssl_config.update_mode == "every_epoch":
                train_dataset = semi_supervised.build_ssl_training_dataset(
                    model=model,
                    train_dataset=dataset_bundle.train_dataset,
                    train_labels_mapper=train_labels_mapper,
                    device=args.device,
                    config=ssl_config,
                    split=ssl_split,
                    epoch=num_epoch,
                    start_method=args.dataloader_start_method,
                )
                train_loader = utils.make_train_loader(
                    train_dataset,
                    args.batch_size,
                    args.sampler_m,
                    seed=args.seed + num_epoch,
                    num_workers=args.num_workers,
                    start_method=args.dataloader_start_method,
                )
            else:
                if static_train_loader is None:
                    train_dataset = semi_supervised.build_ssl_training_dataset(
                        model=model,
                        train_dataset=dataset_bundle.train_dataset,
                        train_labels_mapper=train_labels_mapper,
                        device=args.device,
                        config=ssl_config,
                        split=ssl_split,
                        epoch=num_epoch,
                        start_method=args.dataloader_start_method,
                    )
                    static_train_loader = utils.make_train_loader(
                        train_dataset,
                        args.batch_size,
                        args.sampler_m,
                        seed=args.seed + num_epoch,
                        num_workers=args.num_workers,
                        start_method=args.dataloader_start_method,
                    )
                train_loader = static_train_loader

            model.train()
            epoch_loss = 0.0
            num_batches = 0
            tqdm_bar = tqdm(train_loader)
            for images, labels in tqdm_bar:
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                    # Set map labels to start from 0 for classification losses like ArcFaceLoss
                    labels = torch.tensor([train_labels_mapper[int(label)] for label in labels]).to(args.device)
                    embeddings = model(images.to(args.device))

                    if not is_classification and miner is not None:
                        miner_outputs = miner(embeddings, labels)
                        loss = criterion(embeddings, labels, miner_outputs)
                    else:
                        loss = criterion(embeddings, labels)

                loss_value = loss.detach().item()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if is_classification:
                    classifier_optim.step()
                    classifier_optim.zero_grad()
                metrics_logger.log_train_batch(loss_value, num_epoch, global_step)
                epoch_loss += loss_value
                num_batches += 1
                global_step += 1
                tqdm_bar.desc = f"loss = {loss_value:.5f}"

            if num_batches > 0:
                final_train_loss = epoch_loss / num_batches
                metrics_logger.log_train_epoch(final_train_loss, num_epoch, global_step)

            cur_precision, cur_map = utils.evaluate(model, valid_loader, f"valid - epoch {num_epoch:>2}", device=args.device)
            metrics_logger.log_eval("valid", cur_precision, cur_map, step=global_step, epoch=num_epoch)
            best_precision_for_report = max(best_precision, cur_precision)
            best_map_for_report = max(best_map, cur_map)
            maybe_report_to_optuna(
                optuna_trial=optuna_trial,
                metric=optuna_metric,
                epoch=num_epoch,
                train_loss=final_train_loss,
                valid_precision=cur_precision,
                valid_map=cur_map,
                best_precision=best_precision_for_report,
                best_map=best_map_for_report,
            )

            is_after_warmup = num_epoch >= ssl_config.warmup_epochs

            if cur_map > best_map:
                best_map = cur_map
            if cur_precision > best_precision:
                best_precision = cur_precision
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            elif is_after_warmup:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    model.load_state_dict(torch.load(best_model_path, weights_only=True))
                    break

        if evaluate_test:
            test_precision, test_map = utils.evaluate(model, test_loader, "test", device=args.device)
            metrics_logger.log_eval("test", test_precision, test_map, step=global_step, epoch=last_epoch)
        else:
            logger.info("Skipping test evaluation for this run")
    finally:
        metrics_logger.close()
        if best_model_path.exists():
            os.remove(best_model_path)

    return TrainingResult(
        log_dir=args.log_dir,
        metrics_csv=args.log_dir / "metrics.csv",
        best_valid_precision_at_1=float(best_precision),
        best_valid_mean_average_precision_at_r=float(best_map),
        test_precision_at_1=None if test_precision is None else float(test_precision),
        test_mean_average_precision_at_r=None if test_map is None else float(test_map),
        final_train_loss=None if final_train_loss is None else float(final_train_loss),
        last_epoch=last_epoch,
        global_step=global_step,
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_mode=args.cv_mode if cv_fold is not None else None,
        cv_fold=cv_fold,
    )


def run_cross_validation(args, ssl_config, optuna_trial=None, optuna_metric=None):
    validate_run_args(args, ssl_config)
    cv_run_name = f"cv_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    cv_relative_dir = Path(args.save_dir) / cv_run_name
    cv_dir = Path("logs") / cv_relative_dir
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_index in range(args.cv_k):
        fold_args = copy.deepcopy(args)
        fold_args.cv_fold = fold_index
        fold_args.save_dir = cv_relative_dir / f"fold_{fold_index:02d}"
        result = run_training(
            fold_args,
            ssl_config,
            optuna_trial=None,
            optuna_metric=None,
            cv_fold=fold_index,
        )
        fold_results.append(result)
        write_cross_validation_summary(cv_dir, args, fold_results)
        maybe_report_cv_to_optuna(optuna_trial, optuna_metric, fold_results, fold_index)

    aggregate = aggregate_cross_validation_result(cv_dir, args, fold_results)
    write_cross_validation_summary(cv_dir, args, fold_results, aggregate)
    return aggregate


def aggregate_cross_validation_result(cv_dir, args, fold_results):
    fold_dicts = [result_to_dict(result) for result in fold_results]
    return TrainingResult(
        log_dir=cv_dir,
        metrics_csv=cv_dir / "cv_results.csv",
        best_valid_precision_at_1=mean_metric(fold_results, "best_valid_precision_at_1"),
        best_valid_mean_average_precision_at_r=mean_metric(
            fold_results,
            "best_valid_mean_average_precision_at_r",
        ),
        test_precision_at_1=mean_optional_metric(fold_results, "test_precision_at_1"),
        test_mean_average_precision_at_r=mean_optional_metric(fold_results, "test_mean_average_precision_at_r"),
        final_train_loss=mean_optional_metric(fold_results, "final_train_loss"),
        last_epoch=max(result.last_epoch for result in fold_results),
        global_step=sum(result.global_step for result in fold_results),
        cv_k=args.cv_k,
        cv_mode=args.cv_mode,
        fold_results=fold_dicts,
    )


def mean_metric(results, attr):
    return float(sum(getattr(result, attr) for result in results) / len(results))


def mean_optional_metric(results, attr):
    values = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def write_cross_validation_summary(cv_dir, args, fold_results, aggregate=None):
    rows = [make_cv_summary_row(result) for result in fold_results]
    if aggregate is not None:
        rows.append(make_cv_summary_row(aggregate, fold="mean"))

    csv_path = cv_dir / "cv_results.csv"
    fieldnames = [
        "fold",
        "cv_k",
        "cv_mode",
        "log_dir",
        "metrics_csv",
        "best_valid_precision_at_1",
        "best_valid_mean_average_precision_at_r",
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
        "final_train_loss",
        "last_epoch",
        "global_step",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        cv_dir / "cv_summary.json",
        {
            "args": namespace_to_dict(args),
            "completed_folds": len(fold_results),
            "cv_k": args.cv_k,
            "cv_mode": args.cv_mode,
            "folds": [result_to_dict(result) for result in fold_results],
            "aggregate": None if aggregate is None else result_to_dict(aggregate),
        },
    )


def make_cv_summary_row(result, fold=None):
    return {
        "fold": result.cv_fold if fold is None else fold,
        "cv_k": result.cv_k,
        "cv_mode": "" if result.cv_mode is None else result.cv_mode,
        "log_dir": str(result.log_dir),
        "metrics_csv": str(result.metrics_csv),
        "best_valid_precision_at_1": result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
        "test_precision_at_1": "" if result.test_precision_at_1 is None else result.test_precision_at_1,
        "test_mean_average_precision_at_r": ""
        if result.test_mean_average_precision_at_r is None
        else result.test_mean_average_precision_at_r,
        "final_train_loss": "" if result.final_train_loss is None else result.final_train_loss,
        "last_epoch": result.last_epoch,
        "global_step": result.global_step,
    }


def maybe_report_cv_to_optuna(optuna_trial, metric, fold_results, fold_index):
    if optuna_trial is None or metric is None:
        return
    partial_result = aggregate_cross_validation_result(Path("."), make_cv_args_stub(fold_results), fold_results)
    value = getattr(partial_result, metric)
    if value is None:
        return
    optuna_trial.report(float(value), step=fold_index)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()


def make_cv_args_stub(fold_results):
    stub = argparse.Namespace()
    stub.cv_k = fold_results[0].cv_k
    stub.cv_mode = fold_results[0].cv_mode
    return stub


def validate_run_args(args, ssl_config):
    if args.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}: {args.dataset}")
    if args.dino_size not in {"s", "b", "l", "g"}:
        raise ValueError(f"dino_size must be one of ['s', 'b', 'l', 'g']: {args.dino_size}")
    if args.loss not in ALL_LOSSES:
        raise ValueError(f"loss must be one of {ALL_LOSSES}: {args.loss}")
    if args.miner not in ALL_MINERS:
        raise ValueError(f"miner must be one of {ALL_MINERS}: {args.miner}")
    if args.device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be 'cuda' or 'cpu': {args.device}")
    if args.optim not in {"adam", "rmsprop"}:
        raise ValueError(f"optim must be 'adam' or 'rmsprop': {args.optim}")
    if args.mode not in {"supervised", "ssl"}:
        raise ValueError("mode must be 'supervised' or 'ssl'")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.classifier_lr <= 0:
        raise ValueError("classifier_lr must be positive")
    if args.sampler_m <= 0:
        raise ValueError("sampler_m must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if args.cv_k <= 0:
        raise ValueError("cv_k must be positive")
    if args.cv_mode not in utils.CV_MODES:
        raise ValueError(f"cv_mode must be one of {utils.CV_MODES}: {args.cv_mode}")
    if args.val_mode not in utils.VAL_MODES:
        raise ValueError(f"val_mode must be one of {utils.VAL_MODES}: {args.val_mode}")
    if args.feat_dim is not None and args.feat_dim <= 0:
        raise ValueError("feat_dim must be positive when set")
    utils.validate_dataloader_settings(
        device=args.device,
        num_workers=args.num_workers,
        ssl_embedding_num_workers=ssl_config.embedding_num_workers if ssl_config.enabled else 0,
        start_method=args.dataloader_start_method,
    )


def load_hparam_config(config_path):
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
    if config.direction not in {"maximize", "minimize"}:
        raise ValueError(f"direction must be 'maximize' or 'minimize'{source}")
    if config.metric not in OBJECTIVE_METRICS:
        raise ValueError(f"metric must be one of {sorted(OBJECTIVE_METRICS)}{source}")
    if config.sampler not in {"tpe", "random", "grid"}:
        raise ValueError(f"sampler must be one of ['tpe', 'random', 'grid']{source}")
    if config.pruner not in {"none", "median", "successive_halving"}:
        raise ValueError(f"pruner must be one of ['none', 'median', 'successive_halving']{source}")
    if not isinstance(config.sampler_params, dict):
        raise ValueError(f"sampler_params must be an object{source}")
    if not isinstance(config.pruner_params, dict):
        raise ValueError(f"pruner_params must be an object{source}")
    if not isinstance(config.spaces, dict) or not config.spaces:
        raise ValueError(f"spaces must be a non-empty object{source}")
    for name, spec in config.spaces.items():
        if name in {"loss", "miner"}:
            raise ValueError(
                f"{name!r} is not a valid HPO space key{source}. "
                f"Set it with --{name} or compare fixed pairs with --loss_miner_grid."
            )
        validate_space_spec(name, spec, source)


def validate_space_spec(name, spec, source=""):
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"Search space {name!r} choices must not be empty{source}")
        return
    if not isinstance(spec, dict):
        raise ValueError(f"Search space {name!r} must be an object or a list of categorical choices{source}")

    space_type = spec.get("type", "categorical" if "choices" in spec else None)
    if space_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical search space {name!r} requires a non-empty choices list{source}")
    elif space_type in {"float", "int"}:
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


def run_hparam_search(args, config):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna hyperparameter search requires the optuna package. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    if getattr(args, "skip_test_during_hpo", False) and config.metric.startswith("test_"):
        raise ValueError("Cannot use a test metric as Optuna objective when --skip_test_during_hpo is set")

    study_name = config.study_name or "optuna"
    study_dir, relative_study_dir = make_study_dir(args.save_dir, study_name, config.study_dir)
    storage = resolve_optuna_storage(config.storage, study_dir)
    write_json(
        study_dir / "study_config.json",
        {
            "base_args": namespace_to_dict(args),
            "hparam_config": config.to_dict(),
            "resolved_study_name": study_name,
            "resolved_storage": storage,
        },
    )

    sampler = make_optuna_sampler(optuna, config, args.seed)
    pruner = make_optuna_pruner(optuna, config)
    study = optuna.create_study(
        direction=config.direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=config.load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )
    validate_study_distributions_compatible(optuna, study, config, storage)
    trials_csv = study_dir / "trials.csv"
    trials_jsonl = study_dir / "trials.jsonl"

    def objective(trial):
        trial_args, ssl_config, suggested_params = make_trial_args_and_ssl_config(args, config, trial)
        trial_args.hparam_config_resolved = config.to_dict()
        trial_args.hparam_params = suggested_params
        trial_args.hparam_study_dir = study_dir
        trial_args.hparam_study_name = study.study_name
        trial_args.trial_number = trial.number
        trial_args.evaluate_test = not bool(getattr(args, "skip_test_during_hpo", False))
        trial_args.save_dir = relative_study_dir / f"trial_{trial.number:04d}"

        trial.set_user_attr("params", suggested_params)
        trial.set_user_attr("resolved_args", namespace_to_dict(trial_args))
        trial.set_user_attr("resolved_ssl_config", ssl_config.to_dict())

        result = run_experiment(
            trial_args,
            ssl_config,
            optuna_trial=trial,
            optuna_metric=config.metric,
        )
        result_dict = result_to_dict(result)
        for key, value in result_dict.items():
            trial.set_user_attr(key, value)
        return get_objective_value(result, config.metric)

    def record_trial(study, trial):
        write_trials_summary(study, trials_csv, trials_jsonl)

    finished_trials = count_finished_trials(optuna, study)
    remaining_trials = config.n_trials - finished_trials
    logger.info(
        f"Starting Optuna study outputs in {study_dir}. "
        f"Finished trials: {finished_trials}/{config.n_trials}. "
        f"Remaining this run: {max(remaining_trials, 0)}."
    )
    if remaining_trials <= 0:
        write_trials_summary(study, trials_csv, trials_jsonl)
        logger.info(f"Optuna study already has {finished_trials} finished trials; no new trials requested.")
        return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)

    study.optimize(
        objective,
        n_trials=remaining_trials,
        timeout=config.timeout,
        callbacks=[record_trial],
        gc_after_trial=True,
    )
    write_trials_summary(study, trials_csv, trials_jsonl)
    if any(trial.state.name == "COMPLETE" for trial in study.trials):
        logger.info(f"Best trial: {study.best_trial.number}, value={study.best_value}, params={study.best_trial.params}")
    return make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl)


def make_hparam_study_result(study, study_name, study_dir, trials_csv, trials_jsonl):
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
        )

    best_trial = study.best_trial
    return HParamStudyResult(
        study_name=study_name,
        study_dir=study_dir,
        trials_csv=trials_csv,
        trials_jsonl=trials_jsonl,
        best_trial_number=best_trial.number,
        best_value=float(best_trial.value),
        best_params=dict(best_trial.params),
        best_user_attrs=dict(best_trial.user_attrs),
    )


def make_study_dir(base_save_dir, study_name, configured_study_dir=None):
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


def make_optuna_sampler(optuna, config, seed):
    sampler_params = dict(config.sampler_params)
    if config.sampler in {"tpe", "random"}:
        sampler_params.setdefault("seed", seed)
    if config.sampler == "tpe":
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
    raise ValueError(f"Unsupported Optuna pruner: {config.pruner}")


def validate_study_distributions_compatible(optuna, study, config, storage):
    configured_distributions = make_optuna_distributions(optuna, config.spaces)
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
    suggested_params = {}
    for name, spec in config.spaces.items():
        suggested_params[name] = suggest_value(trial, name, spec)

    trial_args, ssl_config = make_args_and_ssl_config_from_params(base_args, suggested_params)
    return trial_args, ssl_config, suggested_params


def make_args_and_ssl_config_from_params(base_args, params):
    trial_args = copy.deepcopy(base_args)
    ssl_overrides = []

    for name, value in params.items():
        if is_ssl_override(name):
            ssl_overrides.append((name, value))
        else:
            set_arg_value(trial_args, name, value)

    ssl_config = semi_supervised.load_ssl_config(trial_args.ssl_config, default_seed=trial_args.seed)
    if ssl_overrides:
        ssl_dict = ssl_config.to_dict()
        for name, value in ssl_overrides:
            path_parts = name.split(".")[1:]
            set_nested_value(ssl_dict, path_parts, value)
        ssl_config = semi_supervised.SemiSupervisedConfig(**ssl_dict)
        semi_supervised.validate_ssl_config(ssl_config)

    ssl_config = resolve_mode_ssl_config(trial_args, ssl_config)

    validate_run_args(trial_args, ssl_config)
    return trial_args, ssl_config


def make_supervised_split_config(ssl_config):
    config_dict = ssl_config.to_dict()
    config_dict.update(
        {
            "method": "none",
            "update_mode": "once",
            "warmup_epochs": 0,
            "confidence_threshold": 0.0,
            "method_params": {},
        }
    )
    config = semi_supervised.SemiSupervisedConfig(**config_dict)
    semi_supervised.validate_ssl_config(config)
    return config


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


def set_arg_value(args, name, value):
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
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested SSL config path: {'.'.join(path_parts)}")
        if part not in current:
            current[part] = {}
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"Cannot set nested SSL config path: {'.'.join(path_parts)}")
    current[path_parts[-1]] = value


def maybe_report_to_optuna(
    optuna_trial,
    metric,
    epoch,
    train_loss,
    valid_precision,
    valid_map,
    best_precision,
    best_map,
):
    if optuna_trial is None or metric is None:
        return
    value_by_metric = {
        "best_valid_precision_at_1": best_precision,
        "best_valid_mean_average_precision_at_r": best_map,
        "final_train_loss": train_loss,
    }
    value = value_by_metric.get(metric)
    if value is None:
        return
    optuna_trial.report(float(value), step=epoch)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()


def get_objective_value(result, metric):
    value = getattr(result, metric)
    if value is None:
        raise ValueError(f"Objective metric {metric!r} is None; choose a metric available for this run")
    return float(value)


def result_to_dict(result):
    return {
        "log_dir": str(result.log_dir),
        "metrics_csv": str(result.metrics_csv),
        "best_valid_precision_at_1": result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
        "test_precision_at_1": result.test_precision_at_1,
        "test_mean_average_precision_at_r": result.test_mean_average_precision_at_r,
        "final_train_loss": result.final_train_loss,
        "last_epoch": result.last_epoch,
        "global_step": result.global_step,
        "cv_k": result.cv_k,
        "cv_mode": result.cv_mode,
        "cv_fold": result.cv_fold,
        "fold_results": result.fold_results,
    }


def write_run_config(args, ssl_config):
    write_json(
        args.log_dir / "run_config.json",
        {
            "args": namespace_to_dict(args),
            "ssl_config": ssl_config.to_dict(),
        },
    )


def write_trials_summary(study, csv_path, jsonl_path):
    trials = list(study.trials)
    param_names = sorted({name for trial in trials for name in trial.params})
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
        for trial in trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "datetime_start": "" if trial.datetime_start is None else trial.datetime_start.isoformat(),
                "datetime_complete": "" if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
                "duration_seconds": "" if trial.duration is None else trial.duration.total_seconds(),
            }
            for name in param_names:
                row[f"param:{name}"] = json.dumps(to_jsonable(trial.params.get(name)))
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
        "params": to_jsonable(trial.params),
        "user_attrs": to_jsonable(trial.user_attrs),
        "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
        "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
        "duration_seconds": None if trial.duration is None else trial.duration.total_seconds(),
    }


def namespace_to_dict(args):
    return {key: to_jsonable(value) for key, value in vars(args).items()}


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def is_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def write_json(path, data):
    with Path(path).open("w") as json_file:
        json.dump(to_jsonable(data), json_file, indent=2, sort_keys=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()
