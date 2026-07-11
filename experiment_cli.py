"""Command-line parser and top-level experiment configuration loading."""

import argparse
import json
import sys
from pathlib import Path

import semi_supervised
import utils
from dataset_constants import VAL_MODE_MATCH_TRAIN
from experiment_types import (
    ALL_LOSSES,
    ALL_MINERS,
    DATASETS,
    SELECTION_METRIC_MAP_AT_R,
    SELECTION_METRICS,
)
from retrieval_model import BACKBONE_TUNING_FULL, normalize_backbone_tuning

DEFAULT_DATA_SPLIT_SEED = 7
DEFAULT_SUPPORT_SEED = semi_supervised.DEFAULT_SUPPORT_SEED
FINAL_TEST_VISUALIZATION_NONE = "none"
FINAL_TEST_VISUALIZATION_PACMAP = "pacmap"
FINAL_TEST_VISUALIZATION_MODES = (FINAL_TEST_VISUALIZATION_NONE, FINAL_TEST_VISUALIZATION_PACMAP)
STUDY_DIR_MODE_FINAL_TRAIN = "final_train"
STUDY_DIR_MODE_TRAIN_VAL = "train_val"
STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL = "cross_seed_train_val"
STUDY_DIR_MODES = (
    STUDY_DIR_MODE_FINAL_TRAIN,
    STUDY_DIR_MODE_TRAIN_VAL,
    STUDY_DIR_MODE_CROSS_SEED_TRAIN_VAL,
)
COMPARISON_SEED_TARGET_RUNTIME = "seed"
COMPARISON_SEED_TARGET_DATA_SPLIT = "data_split_seed"
COMPARISON_SEED_TARGET_SUPPORT = "support_seed"
COMPARISON_SEED_TARGET_HPARAM = "hparam_seed"
COMPARISON_SEED_TARGETS = (
    COMPARISON_SEED_TARGET_RUNTIME,
    COMPARISON_SEED_TARGET_DATA_SPLIT,
    COMPARISON_SEED_TARGET_SUPPORT,
    COMPARISON_SEED_TARGET_HPARAM,
)


def parse_json_object(value):
    """Accept a JSON object from CLI while preserving dict values from config files."""

    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_non_negative_int(value):
    """Parse an integer value that may be zero."""

    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("value must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--experiment_config",
    "--experiment-config",
    type=Path,
    default=None,
    help=(
        "top-level JSON config containing CLI argument values. "
        "Explicit CLI arguments override values from this file."
    ),
)
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=1e-6, help="LR")
parser.add_argument("--classifier_lr", type=float, default=1.0, help="classifier LR (only for classification losses)")
parser.add_argument("--sampler_m", type=int, default=4, help="M value for MPerClassSampler")
parser.add_argument(
    "--length_before_new_iter",
    "--length-before-new-iter",
    type=int,
    default=None,
    help=(
        "number of examples sampled by MPerClassSampler per epoch. "
        "None derives the length from the active training dataset."
    ),
)
parser.add_argument("--dataset", type=utils.normalize_dataset_name, default="Cars196", choices=DATASETS, help="dataset")
parser.add_argument(
    "--cifar_imbalance_factor",
    type=float,
    default=None,
    help=(
        "CIFAR long-tail factor img_min/img_max. None keeps balanced data; "
        "for example, 0.01 produces approximately 100:1 head-to-tail imbalance."
    ),
)
parser.add_argument(
    "--cifar_train_fraction",
    "--cifar-train-fraction",
    type=float,
    default=0.5,
    help="per-class fraction assigned to development/training by dataset_protocol=cifar_balanced_fraction",
)
parser.add_argument(
    "--cifar_test_fraction",
    "--cifar-test-fraction",
    type=float,
    default=0.5,
    help="per-class fraction assigned to the final test set by dataset_protocol=cifar_balanced_fraction",
)
parser.add_argument(
    "--dataset_protocol",
    choices=utils.DATASET_PROTOCOLS,
    default="official",
    help=(
        "dataset split protocol. CIFAR custom protocols combine the official splits; "
        "cifar_balanced_fraction creates sample-disjoint balanced train/test subsets, while unseen-class "
        "protocols reserve fine classes or complete CIFAR-100 superclasses for final testing"
    ),
)
parser.add_argument("--dino_size", type=str, default="b", choices=["s", "b", "l", "g"], help="which Dino to use")
parser.add_argument(
    "--backbone-tuning",
    "--backbone_tuning",
    dest="backbone_tuning",
    type=normalize_backbone_tuning,
    default=BACKBONE_TUNING_FULL,
    metavar="POLICY",
    help="DINO tuning policy: full, frozen, or last_N_blocks (for example last_2_blocks)",
)

parser.add_argument(
    "--use-cache",
    "--use_cache",
    dest="use_cache",
    action="store_true",
    help=(
        "use deterministic frozen-backbone features; supervised frozen runs precompute one in-memory "
        "feature tensor per active dataset, while other supported modes use the persistent backbone cache"
    ),
)
parser.add_argument("--loss", type=str, default="MultiSimilarityLoss", choices=ALL_LOSSES, help="loss")
parser.add_argument("--miner", type=str, default="MultiSimilarityMiner", choices=ALL_MINERS, help="miner")
parser.add_argument("--loss_params", type=parse_json_object, default={}, help="JSON object passed to the selected loss")
parser.add_argument("--miner_params", type=parse_json_object, default={}, help="JSON object passed to the selected miner")
parser.add_argument(
    "--warmup_loss",
    type=str,
    default="MultiSimilarityLoss",
    choices=ALL_LOSSES,
    help="supervised loss used during labeled-only SSL warmup epochs",
)
parser.add_argument(
    "--warmup_miner",
    type=str,
    default="MultiSimilarityMiner",
    choices=ALL_MINERS,
    help="miner used with warmup_loss",
)
parser.add_argument(
    "--warmup_loss_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to warmup_loss",
)
parser.add_argument(
    "--warmup_miner_params",
    type=parse_json_object,
    default={},
    help="JSON object passed to warmup_miner",
)
parser.add_argument("--feat_dim", type=int, default=None, help="Output dimensionality. Set to None to use CLS")
parser.add_argument(
    "--stml_g_dim",
    type=int,
    default=None,
    help="STML background-head dimension. None uses the DINO backbone dimension.",
)
parser.add_argument(
    "--device",
    type=utils.normalize_device_name,
    default="cuda",
    metavar="DEVICE",
    help="device: cpu, cuda, or an indexed CUDA device such as cuda:2",
)
parser.add_argument("--optim", type=str, default="adam", choices=["adamw", "adam", "rmsprop"], help="optimizer")
parser.add_argument("--seed", type=int, default=7, help="random seed for training/runtime randomness")
parser.add_argument(
    "--hparam_seed",
    "--hparam-seed",
    "--hpo_seed",
    "--hpo-seed",
    dest="hparam_seed",
    type=int,
    default=None,
    help="seed for Optuna HPO samplers. Defaults to --seed.",
)
parser.add_argument(
    "--tpe_startup_trials",
    "--tpe-startup-trials",
    type=parse_non_negative_int,
    default=None,
    help=(
        "number of random startup trials for Optuna's TPE sampler. "
        "None uses the HPO config or Optuna default."
    ),
)
parser.add_argument(
    "--data_split_seed",
    type=int,
    default=None,
    help=(
        "seed for dataset protocol, train/validation, and validation downsampling splits. "
        f"Defaults to {DEFAULT_DATA_SPLIT_SEED} and is independent of --seed."
    ),
)
parser.add_argument(
    "--support_seed",
    "--support-seed",
    type=int,
    default=None,
    help=(
        "seed for labeled support/sample selection. "
        f"Defaults to {DEFAULT_SUPPORT_SEED} and is independent of --seed."
    ),
)
parser.add_argument("--epochs", type=int, default=1000, help="maximum number of training epochs")
parser.add_argument("--patience", type=int, default=3, help="early-stopping patience after SSL warmup")
parser.add_argument("--cv_k", type=int, default=4, help="number of cross-validation folds. Set to 1 to disable CV")
parser.add_argument(
    "--cv_mode",
    type=str,
    choices=utils.CV_MODES,
    default="kfold",
    help="cross-validation splitter to use when cv_k > 1",
)
parser.add_argument(
    "--val_mode",
    type=str,
    choices=utils.VAL_MODES,
    default=VAL_MODE_MATCH_TRAIN,
    help=(
        "validation data mode. 'all' keeps the current behavior and uses all validation samples; "
        "'match_train' downsamples validation to roughly the labeled/fractioned training size; "
        "'split_after_apportion' creates a standard train/validation split after the training data is apportioned."
    ),
)
parser.add_argument(
    "--selection_metric",
    default=SELECTION_METRIC_MAP_AT_R,
    choices=SELECTION_METRICS,
    help="validation metric used for checkpoint selection and early stopping",
)
parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count for training/evaluation")
parser.add_argument(
    "--dataloader_start_method",
    type=str,
    default="spawn",
    choices=utils.DATALOADER_START_METHODS,
    help="DataLoader multiprocessing start method for CPU runs or CUDA runs with zero workers.",
)
parser.add_argument(
    "--debug_batch_timing",
    "--debug-batch-timing",
    action="store_true",
    default=False,
    help="log detailed per-batch timing; this synchronizes CUDA and should stay off for benchmarks",
)
parser.add_argument(
    "--debug_batch_timing_interval",
    "--debug-batch-timing-interval",
    type=int,
    default=5,
    help="number of batches between debug timing log lines when --debug_batch_timing is enabled",
)
parser.add_argument(
    "--log_batch_diagnostics",
    "--log-batch-diagnostics",
    dest="log_batch_diagnostics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "log per-batch diagnostic metrics such as gradient norms and miner counts; "
        "this adds overhead and should stay off for throughput runs"
    ),
)
parser.add_argument(
    "--frozen_feature_batch_size",
    type=int,
    default=None,
    help="batch size for one-time frozen-backbone feature extraction; defaults to --batch_size",
)
parser.add_argument(
    "--frozen_feature_train_views",
    type=int,
    default=1,
    help=(
        "number of stochastic training views to precompute per sample when frozen feature precompute is active; "
        "1 preserves deterministic cached features"
    ),
)
parser.add_argument(
    "--ssl_config",
    type=Path,
    default="configs/ssl_hoffer_entropy.json",
    help="path to a JSON semi-supervised config. Omit to disable SSL.",
)
parser.add_argument(
    "--unlabeled_source",
    choices=["split", "external", "split_and_external"],
    default=None,
    help=(
        "SSL unlabeled pool: the current train split, an external recursive image directory, "
        "or both. None preserves legacy STML-specific settings."
    ),
)
parser.add_argument(
    "--external_unlabeled_dir",
    type=Path,
    default=None,
    help="recursive external image directory used as unlabeled SSL data, such as a local Fashion200K root",
)
parser.add_argument(
    "--external_unlabeled_filter",
    type=str,
    default=utils.EXTERNAL_UNLABELED_FILTER_NONE,
    choices=utils.EXTERNAL_UNLABELED_FILTERS,
    help=(
        "optional filtering applied to external unlabeled images. "
        "compcars_model_min_count keeps CompCars model-level categories with enough images; "
        "compcars_stml_paper reproduces the STML CompCars subset and checks for 16,537 images "
        "across 145 model classes."
    ),
)
parser.add_argument(
    "--compcars_min_model_images",
    type=int,
    default=100,
    help="minimum images per inferred CompCars model class for CompCars external-unlabeled filters",
)
parser.add_argument(
    "--compcars_strict_paper_counts",
    action="store_true",
    default=False,
    help=(
        "fail CompCars STML-paper filtering unless the filtered pool has exactly "
        "16,537 images across 145 model classes"
    ),
)
parser.add_argument(
    "--stml_unlabeled_source",
    choices=["split", "external", "split_and_external"],
    default="split",
    help=(
        "Deprecated alias for --unlabeled_source used by older STML configs."
    ),
)
parser.add_argument(
    "--stml_external_unlabeled_dir",
    type=Path,
    default=None,
    help="Deprecated alias for --external_unlabeled_dir used by older STML configs.",
)
parser.add_argument(
    "--hparam_config",
    type=Path,
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
    choices=["supervised", "ssl"],
    default="supervised",
    help="training mode. supervised uses only the labeled split; ssl uses the labeled split plus unlabeled data.",
)
parser.add_argument(
    "--skip_test_during_hpo",
    action="store_true",
    default=True,
    help="do not evaluate D_test inside Optuna trials; use a final retraining run for test evaluation",
)
parser.add_argument(
    "--retry_failed_hpo_trials",
    action="store_true",
    default=False,
    help="enqueue existing failed/pruned Optuna trials as new trials and run only those retries",
)
parser.add_argument(
    "--evaluate_test",
    "--evaluate-test",
    action="store_true",
    default=False,
    help="evaluate D_test after a direct non-HPO training run",
)
parser.add_argument(
    "--final_test_after_hpo",
    action="store_true",
    default=False,
    help=(
        "after each HPO study, train the best configuration once on the full development set "
        "for the selected mean fold epoch count and evaluate D_test"
    ),
)
parser.add_argument(
    "--final_test_top_n",
    type=int,
    default=1,
    help=(
        "number of highest-value completed HPO trials to replay; final_train tests every replay, "
        "while train_val and cross_seed_train_val fully retrain/test only the validation winner"
    ),
)
parser.add_argument(
    "--final_test_trial_numbers",
    type=int,
    nargs="*",
    default=None,
    help=(
        "specific completed HPO trial numbers to replay; final_train tests every replay, while "
        "train_val and cross_seed_train_val fully retrain/test only the validation winner"
    ),
)
parser.add_argument(
    "--final_test_study_dir",
    "--final-test-study-dir",
    type=Path,
    default=None,
    help=(
        "existing HPO study directory to load for final-test evaluation. "
        "Use with --final_test_trial_numbers or --final_test_top_n to avoid scheduling new HPO trials."
    ),
)
parser.add_argument(
    "--study_dir_mode",
    "--study-dir-mode",
    "--final_test_study_dir_mode",
    "--final-test-study-dir-mode",
    dest="study_dir_mode",
    choices=STUDY_DIR_MODES,
    default=STUDY_DIR_MODE_TRAIN_VAL,
    help=(
        "how to replay an existing HPO study: final_train trains once on the full development set "
        "and evaluates D_test; train_val ranks selected trials on one validation split and then fully "
        "retrains/tests the winner; cross_seed_train_val ranks trials by mean validation performance "
        "over comparison_seeds and then fully retrains/tests the winner once"
    ),
)
parser.add_argument(
    "--final_test_visualization",
    "--final-test-visualization",
    choices=FINAL_TEST_VISUALIZATION_MODES,
    default=FINAL_TEST_VISUALIZATION_PACMAP,
    help="optional final D_test embedding visualization artifact to create after test evaluation",
)
parser.add_argument(
    "--label_budget_grid",
    type=float,
    nargs="*",
    help="outer experiment grid over SSL labeled_fraction values, for example 0.01 0.05 0.10 0.25 0.50",
)
parser.add_argument(
    "--k_shot_grid",
    type=int,
    nargs="*",
    help="outer experiment grid over k-shot counts for label_sampling_mode='class_subset_k_shot', for example 1 2 5",
)
parser.add_argument(
    "--loss_miner_grid",
    type=str,
    nargs="*",
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
    help=(
        "outer experiment grid over runtime, dataset-split, labeled-support, "
        "and HPO sampler seeds, for example 0 1 2 3 4"
    ),
)
parser.add_argument(
    "--comparison_seed_targets",
    "--comparison-seed-targets",
    nargs="+",
    choices=COMPARISON_SEED_TARGETS,
    default=list(COMPARISON_SEED_TARGETS),
    help=(
        "seed channels replaced by each --comparison_seeds value. Configure as a JSON array in an "
        "experiment config; omitted defaults to seed, data_split_seed, support_seed, and hparam_seed"
    ),
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
    default="test",
    help="name of directory in which to save the logs, under logs/save_dir",
)

def parse_args_with_experiment_config(argv=None):
    """Parse CLI arguments after loading optional top-level JSON defaults."""

    explicit_cli_args = collect_explicit_cli_destinations(argv)
    config_path = get_experiment_config_path(argv)
    config_values = load_experiment_config(config_path)
    namespace = argparse.Namespace(**config_values)
    args = parser.parse_args(argv, namespace=namespace)
    args.explicit_cli_args = sorted(explicit_cli_args)
    normalize_backbone_tuning_args(args)
    resolve_hparam_seed(args)
    resolve_data_split_seed(args)
    resolve_support_seed(args)
    if config_path is not None:
        args.experiment_config_resolved = config_values
    return args

def collect_explicit_cli_destinations(argv=None):
    """Return argparse destination names set directly by the current CLI."""

    raw_argv = sys.argv[1:] if argv is None else list(argv)
    explicit_dests = set()
    for token in raw_argv:
        if not token.startswith("--") or token == "--":
            continue
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is None or action.dest in {"help", "experiment_config"}:
            continue
        explicit_dests.add(action.dest)
    return explicit_dests

def get_hparam_seed(args):
    """Return the seed used by Optuna's stochastic samplers."""

    hparam_seed = getattr(args, "hparam_seed", None)
    return int(getattr(args, "seed", 7)) if hparam_seed is None else int(hparam_seed)

def resolve_hparam_seed(args):
    """Default HPO sampling to the runtime seed unless explicitly separated."""

    args.hparam_seed = get_hparam_seed(args)
    return args

def resolve_data_split_seed(args):
    """Keep the validation/test split seed independent from the run seed."""

    if getattr(args, "data_split_seed", None) is None:
        args.data_split_seed = DEFAULT_DATA_SPLIT_SEED
    return args

def get_support_seed(args):
    """Return the fixed seed used for labeled support selection."""

    support_seed = getattr(args, "support_seed", None)
    return DEFAULT_SUPPORT_SEED if support_seed is None else int(support_seed)

def resolve_support_seed(args):
    """Keep labeled support selection independent from the run seed."""

    args.support_seed = get_support_seed(args)
    return args

def normalize_backbone_tuning_args(args):
    """Normalize the selected backbone fine-tuning policy."""

    args.backbone_tuning = normalize_backbone_tuning(
        getattr(args, "backbone_tuning", BACKBONE_TUNING_FULL)
    )
    return args

def get_experiment_config_path(argv=None):
    """Read only --experiment-config without validating the remaining CLI."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--experiment_config",
        "--experiment-config",
        type=Path,
        default=parser.get_default("experiment_config"),
    )
    config_args, _ = config_parser.parse_known_args(argv)
    return config_args.experiment_config

def load_experiment_config(config_path):
    """Load and validate JSON values that become argparse defaults."""

    if config_path is None:
        return {}

    path = Path(config_path)
    with path.open() as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {path}")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "experiment_config"}
    }
    unknown_keys = sorted(set(raw_config) - set(actions))
    if unknown_keys:
        raise ValueError(
            f"Unknown experiment config keys in {path}: {unknown_keys}. "
            "Use argparse destination names such as save_dir, ssl_label_sampling_modes, and loss_miner_grid."
        )

    return {
        name: normalize_experiment_config_value(actions[name], value, path)
        for name, value in raw_config.items()
    }

def normalize_experiment_config_value(action, value, path):
    """Apply an argparse action's type and choices to one JSON config value."""

    source = f" for {action.dest!r} in {path}"
    if value is None:
        return None
    if action.nargs == 0:
        if not isinstance(value, bool):
            raise ValueError(f"Experiment config value{source} must be true or false")
        return value

    expects_list = action.nargs in {"*", "+"} or isinstance(action.nargs, int)
    if expects_list:
        if not isinstance(value, list):
            raise ValueError(f"Experiment config value{source} must be a JSON array or null")
        if action.nargs == "+" and not value:
            raise ValueError(f"Experiment config value{source} must not be empty")
        values = [convert_experiment_config_scalar(action, item, source) for item in value]
        validate_experiment_config_choices(action, values, source)
        return values

    if isinstance(value, list):
        raise ValueError(f"Experiment config value{source} must be a scalar or null")
    if isinstance(value, dict) and action.type is None:
        raise ValueError(f"Experiment config value{source} must be a scalar or null")
    value = convert_experiment_config_scalar(action, value, source)
    validate_experiment_config_choices(action, [value], source)
    return value

def convert_experiment_config_scalar(action, value, source):
    if action.type is None:
        return value
    try:
        return action.type(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"Invalid experiment config value{source}: {value!r}") from exc

def validate_experiment_config_choices(action, values, source):
    if action.choices is None:
        return
    invalid_values = [value for value in values if value not in action.choices]
    if invalid_values:
        raise ValueError(
            f"Invalid experiment config value{source}: {invalid_values}. "
            f"Choose from {list(action.choices)}"
        )
