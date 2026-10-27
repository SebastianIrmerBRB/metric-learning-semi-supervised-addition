"""Emit a controlled training ablation that moves exactly one SSL dimension.

The offline benchmarks and the HPO trial analysis both stop one step short of the
end-to-end answer. The benchmarks show what a dimension does to the graph, the
pseudo-labels or the gradient; the trial analysis shows a confounded correlation.
Neither is a controlled experiment on retrieval quality, because in a study every
trial redraws lr, gamma, alpha, the loss parameters and the batch sampler
alongside the dimension of interest.

This pins a study's best trial and reopens exactly one of them. It writes, per
(method, dataset, loss):

* an SSL config carrying that trial's SSL parameters, minus the swept dimension
* an experiment config carrying that trial's lr, classifier_lr, batch sampler
  and loss parameters, inherited from the study's own experiment config
* an HPO config with ``sampler='grid'`` over the swept dimension alone, so the
  run visits every rung of the ladder exactly once and nothing else moves
* a run manifest that launches them

Seeds are the ablation's only other axis: ``--seeds`` repeats the whole grid so a
difference between two rungs can be read against within-rung spread instead of
being taken on faith from one run.

Dimensions (``--list-dimensions`` prints them with their ladders):

* graph methods: ``n_neighbors``, ``graph_batch_size``
* every SSL method: ``confidence_threshold``, ``warmup_epochs``
* every regularizer method: ``regularizer_target_ratio``
* SLADE: ``unlabeled_batch_size``, ``cluster_ratio``, ``basis_target_ratio``

Examples::

    python scripts/make_ssl_ablation.py --list
    python scripts/make_ssl_ablation.py --dimension graph_batch_size \\
        --study exporter/optuna-studies/ssl/in_shop/lrml_prx/.../hpo/cars_imb_search \\
        --experiment-config configs/phase_1/ssl_methods/lrml/in_shop/in_shop_nt.json
    python scripts/make_ssl_ablation.py --dimension confidence_threshold \\
        --values 0.0 0.2 0.4 0.6 0.8 0.99 --study ... --experiment-config ...
    python scripts/make_ssl_ablation.py --dimension unlabeled_batch_size \\
        --study logs/phase_1/slade_tr_capped/circle_loss_shop/.../hpo/slade_cars_initial \\
        --experiment-config configs/phase_1/in_shop/slade.json
"""

import argparse
import copy
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.hpo import (  # noqa: E402
    LABELED_BATCH_FRACTION_HPARAM_KEY as LABELED_BATCH_FRACTION_KEY,
    derive_labeled_batch_size_from_fraction,
    expand_joint_component_params,
)

BEST_TRIAL_QUERY = """
SELECT t.trial_id, t.state, tv.value, tp.param_name, tp.param_value, tp.distribution_json
FROM trials t
LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id
LEFT JOIN trial_params tp ON tp.trial_id = t.trial_id
"""

# Parameter names the HPO layer routes somewhere other than a plain argparse
# destination. Everything else in a trial is an argparse dest by construction.
BATCH_SAMPLER_KEY = "batch_sampler"
SSL_PREFIX = "ssl_config."
LOSS_PREFIX = "loss."
MINER_PREFIX = "miner."

# Where a dimension lives in the SSL config:
#   "graph"       - method_params for pseudo-label methods, one level deeper under
#                   regularizer_params for regularizer methods
#   "ssl"         - a top-level SSL setting shared by every method
#   "method"      - method_params of a regularizer method
#   "regularizer" - regularizer_params of a regularizer method
LOCATION_GRAPH = "graph"
LOCATION_SSL = "ssl"
LOCATION_METHOD = "method"
LOCATION_REGULARIZER = "regularizer"


class Dimension:
    def __init__(self, ladder, value_type, location, method=None, clears=(), keep_pinned=False):
        self.ladder = ladder
        self.value_type = value_type
        self.location = location
        # Leave the trial's value in the SSL config as a placeholder the grid
        # overrides, for a key whose absence makes the config itself invalid.
        self.keep_pinned = keep_pinned
        # The one SSL method the dimension exists for, or None for any.
        self.method = method
        # Pinned keys (same location) that would override the swept one and
        # silently turn the grid into repeats of one configuration.
        self.clears = tuple(clears)


# Default ladders per dimension. Each brackets the range the phase-1 spaces
# search so the ablation can show where the searched interval stops mattering,
# rather than only confirming the interval's interior.
DIMENSIONS = {
    "n_neighbors": Dimension([10, 25, 50, 100, 200], int, LOCATION_GRAPH),
    "graph_batch_size": Dimension([16, 64, 256, 1024, 4096], int, LOCATION_GRAPH),
    "confidence_threshold": Dimension([0.0, 0.3, 0.5, 0.7, 0.9, 0.99], float, LOCATION_SSL),
    # Searched 0-30 in steps of 5; 50 shows whether a longer supervised start
    # still changes what the SSL phase can add.
    "warmup_epochs": Dimension([0, 5, 10, 15, 30, 50], int, LOCATION_SSL),
    # Searched 0.5-1.5 on every regularizer method.
    # The calibration settings beside it (EMA alpha, statistic, cadence) are
    # rejected as orphaned without a target ratio, so it stays as a placeholder.
    "regularizer_target_ratio": Dimension(
        [0.1, 0.25, 0.5, 1.0, 1.5, 3.0], float, LOCATION_METHOD, keep_pinned=True
    ),
    # Searched 32-512. On In-Shop a batch of 64 holds ~0.5 same-class pairs, so
    # this is also the knob that decides how many batches mine no positive.
    "unlabeled_batch_size": Dimension(
        [32, 64, 128, 256, 512, 1024], int, LOCATION_REGULARIZER, method="slade",
        clears=("unlabeled_ratio",),
    ),
    # Searched {0.1, 0.5, 1, 2}; an absolute num_clusters would win over it.
    "cluster_ratio": Dimension(
        [0.1, 0.25, 0.5, 1.0, 2.0, 4.0], float, LOCATION_REGULARIZER, method="slade",
        clears=("num_clusters",),
    ),
    # Searched 0.5-1.5.
    "basis_target_ratio": Dimension(
        [0.1, 0.25, 0.5, 1.0, 1.5, 3.0], float, LOCATION_REGULARIZER, method="slade",
    ),
}
# Kept for callers that read the ladders directly.
DIMENSION_LADDERS = {
    name: (dimension.ladder, dimension.value_type) for name, dimension in DIMENSIONS.items()
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--study",
        help="Study directory containing optuna_study.db, whose best trial is pinned",
    )
    parser.add_argument(
        "--experiment-config",
        help="Experiment config the study was launched with; its dataset settings are inherited",
    )
    parser.add_argument(
        "--ssl-config",
        help="Base SSL config; defaults to the one named in --experiment-config",
    )
    parser.add_argument(
        "--dimension",
        choices=sorted(DIMENSIONS),
        default="n_neighbors",
        help="Which single SSL dimension to reopen",
    )
    parser.add_argument(
        "--no-save-heads",
        action="store_true",
        help=(
            "Do not write head_state.pt per fold. Heads are what "
            "evaluate/export_embedding_views.py needs for labeled/unlabeled and 3D views"
        ),
    )
    parser.add_argument(
        "--list-dimensions",
        action="store_true",
        help="Print every dimension with its location and default ladder, and exit",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help="Ladder the grid sampler visits; defaults to the dimension's own ladder",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 8, 9],
        help="Repeat the whole grid once per seed",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/ablations",
        help="Where the generated configs are written",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Ablation name; defaults to the study directory's run name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List studies whose best trial can be pinned, and exit",
    )
    parser.add_argument("--roots", nargs="+", default=["logs", "exporter"])
    return parser.parse_args()


def load_best_trial(study_dir):
    """Return ``(params, objective)`` of the best completed trial of a study."""

    database = Path(study_dir) / "optuna_study.db"
    if not database.is_file():
        raise SystemExit(f"No optuna_study.db under {study_dir}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    rows = connection.execute(BEST_TRIAL_QUERY).fetchall()
    connection.close()

    trials = defaultdict(lambda: {"params": {}})
    for trial_id, state, value, name, param_value, distribution in rows:
        trial = trials[trial_id]
        trial["state"] = state
        trial["value"] = value
        if name is not None:
            trial["params"][name] = decode_param(param_value, distribution)

    completed = [
        trial
        for trial in trials.values()
        if trial["state"] == "COMPLETE" and trial["value"] is not None
    ]
    if not completed:
        raise SystemExit(f"No completed trials in {study_dir}")
    best = max(completed, key=lambda trial: trial["value"])
    # Studies that constrain several dimensions against each other store them as
    # one serialized joint choice. Unpack it with the HPO layer's own helper so
    # the pinned values are exactly what the trial ran with.
    return expand_joint_component_params(best["params"]), float(best["value"])


def decode_param(param_value, distribution_json):
    """Turn Optuna's stored float back into the value the trial actually used."""

    try:
        distribution = json.loads(distribution_json)
    except (TypeError, ValueError):
        return param_value
    attributes = distribution.get("attributes", {})
    if "choices" in attributes:
        return attributes["choices"][int(param_value)]
    if distribution.get("name") == "IntDistribution":
        return int(param_value)
    return float(param_value)


def split_params(params):
    """Route a trial's parameters to the config file each one belongs in."""

    experiment = {}
    ssl_overrides = {}
    loss_params = {}
    miner_params = {}
    for name, value in params.items():
        if name == BATCH_SAMPLER_KEY:
            batch_size, sampler_m = str(value).split(":")
            experiment["batch_size"] = int(batch_size)
            experiment["sampler_m"] = int(sampler_m)
        elif name.startswith(SSL_PREFIX):
            ssl_overrides[name[len(SSL_PREFIX):]] = value
        elif name.startswith(LOSS_PREFIX):
            loss_params[name.split(".")[-1]] = value
        elif name.startswith(MINER_PREFIX):
            miner_params[name.split(".")[-1]] = value
        else:
            experiment[name] = value
    return experiment, ssl_overrides, loss_params, miner_params


def resolve_labeled_batch_fraction(experiment_params, ssl_overrides, base_experiment):
    """Turn a searched labeled-batch share into the absolute size it landed on.

    ``labeled_batch_fraction`` exists only inside the search space: the HPO layer
    snaps it onto the sampler grid and writes ``ssl_config.labeled_batch_size``.
    Replaying the fraction as a config key would be rejected by the CLI, so the
    same snapping is applied here with the HPO layer's own function.
    """

    fraction = experiment_params.pop(LABELED_BATCH_FRACTION_KEY, None)
    if fraction is None:
        return
    batch_size = experiment_params.get("batch_size", base_experiment.get("batch_size"))
    sampler_m = experiment_params.get("sampler_m", base_experiment.get("sampler_m"))
    if batch_size is None or sampler_m is None:
        raise SystemExit(
            f"Trial searched {LABELED_BATCH_FRACTION_KEY!r} but neither the trial nor "
            "the experiment config pins batch_size and sampler_m to resolve it against"
        )
    ssl_overrides["labeled_batch_size"] = derive_labeled_batch_size_from_fraction(
        batch_size, sampler_m, fraction
    )


def apply_nested(config, dotted_key, value):
    """Set ``a.b.c`` inside a nested SSL config dict."""

    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def remove_nested(config, dotted_key):
    """Drop ``a.b.c`` from a nested SSL config dict if it is present."""

    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def ssl_method_name(ssl_config):
    """The method a config runs: SLADE is ``method='slade'`` or ``regularizer='slade'``."""

    method_params = ssl_config.get("method_params", {})
    return method_params.get("regularizer") or ssl_config.get("method")


def find_dimension_key(dimension, ssl_config):
    """Return the SSL config path a swept dimension lives at for this method.

    The graph parameters sit under ``method_params`` for the pseudo-labelling
    methods and one level deeper under ``regularizer_params`` for the
    regularizer methods; ``confidence_threshold`` and ``warmup_epochs`` are
    top-level SSL settings for every method; the target ratio belongs to the
    regularizer method itself, SLADE's own settings to its regularizer_params.
    """

    spec = DIMENSIONS[dimension]
    method_params = ssl_config.get("method_params", {})
    is_regularizer = "regularizer_params" in method_params
    if spec.method is not None and ssl_method_name(ssl_config) != spec.method:
        raise SystemExit(
            f"{dimension} is a {spec.method} setting, but the SSL config runs "
            f"{ssl_method_name(ssl_config)!r}"
        )
    if spec.location == LOCATION_SSL:
        return f"ssl_config.{dimension}"
    if spec.location in (LOCATION_METHOD, LOCATION_REGULARIZER) and not is_regularizer:
        raise SystemExit(
            f"{dimension} needs a regularizer method (method_params.regularizer_params), "
            f"but the SSL config runs {ssl_method_name(ssl_config)!r}"
        )
    if spec.location == LOCATION_METHOD:
        return f"ssl_config.method_params.{dimension}"
    if spec.location == LOCATION_REGULARIZER or is_regularizer:
        return f"ssl_config.method_params.regularizer_params.{dimension}"
    return f"ssl_config.method_params.{dimension}"


def list_dimensions():
    for name, spec in DIMENSIONS.items():
        scope = spec.method or {
            LOCATION_GRAPH: "graph methods",
            LOCATION_SSL: "every SSL method",
            LOCATION_METHOD: "regularizer methods",
            LOCATION_REGULARIZER: "regularizer methods",
        }[spec.location]
        print(f"{name:<26} {scope:<20} {spec.ladder}")


def study_loss_miner(study_dir):
    """Read the ``<Loss>_<Miner>`` pair out of a study directory name.

    Directories look like
    ``class_subset_k_shot_label_1_k_2_ProxyAnchorLoss_no_miner_tune_seed_7``.
    """

    for part in Path(study_dir).parts:
        if "Loss" not in part:
            continue
        tokens = part.split("_")
        for index, token in enumerate(tokens):
            if not token.endswith("Loss"):
                continue
            remainder = tokens[index + 1:]
            if remainder[:2] == ["no", "miner"]:
                return f"{token}:no_miner"
            if remainder and remainder[0].endswith("Miner"):
                return f"{token}:{remainder[0]}"
            return f"{token}:no_miner"
    return None


def list_candidates(roots):
    for root in roots:
        for database in sorted(Path(root).rglob("optuna_study.db")):
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            searched = connection.execute(
                "SELECT COUNT(*) FROM trial_params WHERE param_name LIKE '%ssl_config%'"
            ).fetchone()[0]
            completed = connection.execute(
                "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'"
            ).fetchone()[0]
            connection.close()
            if searched:
                print(f"{completed:>4} complete  {database.parent}")


def main():
    args = parse_args()
    if args.list_dimensions:
        list_dimensions()
        return
    if args.list:
        list_candidates(args.roots)
        return
    if not args.study or not args.experiment_config:
        raise SystemExit("--study and --experiment-config are required (or use --list)")

    params, objective = load_best_trial(args.study)
    base_experiment = json.loads(Path(args.experiment_config).read_text())
    ssl_config_path = args.ssl_config or base_experiment.get("ssl_config")
    if ssl_config_path is None:
        raise SystemExit("No ssl_config in the experiment config; pass --ssl-config")
    base_ssl = json.loads(Path(ssl_config_path).read_text())

    experiment_params, ssl_overrides, loss_params, miner_params = split_params(params)
    resolve_labeled_batch_fraction(experiment_params, ssl_overrides, base_experiment)
    ssl_config = copy.deepcopy(base_ssl)
    for key, value in ssl_overrides.items():
        apply_nested(ssl_config, key, value)

    spec = DIMENSIONS[args.dimension]
    ladder = (
        list(spec.ladder)
        if args.values is None
        else [spec.value_type(value) for value in args.values]
    )
    dimension_key = find_dimension_key(args.dimension, ssl_config)
    # The ablation owns this dimension, so the pinned value is dropped rather
    # than left behind as a default the grid has to override -- unless the SSL
    # config is invalid without it; every trial overrides it either way.
    if not spec.keep_pinned:
        remove_nested(ssl_config, dimension_key[len(SSL_PREFIX):])
    parent_key = dimension_key[len(SSL_PREFIX):].rpartition(".")[0]
    for cleared in spec.clears:
        cleared_key = f"{parent_key}.{cleared}" if parent_key else cleared
        remove_nested(ssl_config, cleared_key)
    if args.dimension == "regularizer_target_ratio":
        cap = ssl_config.get("method_params", {}).get("regularizer_max_ratio")
        if cap is not None and max(ladder) > float(cap):
            print(
                f"Note: regularizer_max_ratio={cap} is below the ladder's top rung "
                f"{max(ladder)}; those rungs are clipped to the cap on every step"
            )

    # The study directory names the loss/miner pair it was tuned for. Carrying it
    # over stops the ablation from silently inheriting a different loss from the
    # experiment config and comparing k against parameters tuned for something else.
    loss_miner = study_loss_miner(args.study)
    if loss_miner is not None:
        base_experiment["loss_miner_grid"] = [loss_miner]

    name = args.name or Path(args.study).parts[-3]
    output_dir = Path(args.output_dir) / args.dimension / name
    output_dir.mkdir(parents=True, exist_ok=True)

    ssl_path = output_dir / "ssl_config.json"
    ssl_path.write_text(json.dumps(ssl_config, indent=2))

    hparam_config = {
        "enabled": True,
        # One trial per rung: a grid sampler visits each exactly once.
        "n_trials": len(ladder),
        "direction": "maximize",
        "metric": "best_valid_mean_average_precision_at_r",
        "study_name": f"{name}_{args.dimension}_ablation",
        "sampler": "grid",
        "sampler_params": {},
        "pruner": "none",
        "pruner_params": {},
        "spaces": {
            dimension_key: {"type": "categorical", "choices": list(ladder)}
        },
    }
    hparam_path = output_dir / "hparam_config.json"
    hparam_path.write_text(json.dumps(hparam_config, indent=2))

    runs = []
    for seed in args.seeds:
        experiment = copy.deepcopy(base_experiment)
        experiment.update(experiment_params)
        if loss_params:
            experiment["loss_params"] = loss_params
        if miner_params:
            experiment["miner_params"] = miner_params
        experiment["ssl_config"] = str(ssl_path)
        experiment["hparam_config"] = str(hparam_path)
        experiment["seed"] = seed
        # Each fold's selected head, so any rung can be re-embedded and viewed
        # (train split, validation and D_test) without training it again.
        experiment["save_head_state"] = not args.no_save_heads
        experiment["save_dir"] = f"ablations/{args.dimension}/{name}/seed_{seed}"
        # Holding the split and support seeds fixed keeps every k comparison on
        # the same folds and the same labeled subset; only training randomness
        # moves with --seeds.
        experiment.setdefault("data_split_seed", 7)
        experiment.setdefault("support_seed", 7)
        experiment_path = output_dir / f"experiment_seed_{seed}.json"
        experiment_path.write_text(json.dumps(experiment, indent=2))
        runs.append(
            {
                "name": f"{name}_seed_{seed}",
                "experiment_config": str(experiment_path),
                "hparam_config": str(hparam_path),
            }
        )

    manifest = {
        "version": 1,
        "gpus": [0],
        "max_concurrent_per_gpu": 1,
        "default_gpu_memory_mib": 2048,
        "reserve_gpu_memory_mib": 2048,
        "poll_interval_seconds": 2,
        "launch_delay_seconds": 2,
        "save_dir_root": f"ablations/{args.dimension}/{name}",
        "scheduler_log_dir": f"logs/run_scheduler/{args.dimension}_{name}",
        "runs": runs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Pinned best trial of {args.study} (objective {objective:.4f})")
    print(f"  swept dimension : {dimension_key} over {ladder}")
    print(f"  seeds           : {args.seeds}")
    print(f"  pinned SSL      : {json.dumps(ssl_overrides)}")
    print(f"  pinned training : {json.dumps({**experiment_params, 'loss_params': loss_params})}")
    print(f"  wrote           : {output_dir}")
    print(f"\nRun with:\n  python launch_runs.py --manifest {manifest_path}")
    print(
        f"Total runs: {len(ladder) * len(args.seeds)} "
        f"({len(ladder)} values x {len(args.seeds)} seeds)"
    )


if __name__ == "__main__":
    main()
