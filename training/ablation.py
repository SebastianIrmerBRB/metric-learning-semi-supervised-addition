"""Ablations of an existing run: named variants of sparse, last-word changes.

An ablation config is not a second experiment config. Each of its variants
carries only what that variant *changes*, and it is applied after everything
else has had its say -- the experiment config, the SSL config file, the HPO
study's winning parameters, and the CLI. That ordering is the point: an
ablation asks "this tuned run, but with one thing changed", so it has to win
against the tuned settings it is deliberately contradicting, and it must not
restate the settings it keeps.

A variant is written the way an experiment config is, or with the dotted
vocabulary of an HPO search space, so anything a run is configured or tuned
with can be ablated::

    "sampler_m": 8, "batch_size": 512, "unlabeled_source": "labeled"
                                                   -> CLI arguments
    "batch_sampler": "512:8"                       -> batch_size and sampler_m together
    "loss_params": {"temperature": 0.1}            -> the run's loss, by parameter
    "loss.MultiSimilarityLoss.beta": 2.0           -> a class-qualified loss param
    "ssl_config": {"method_params": {"regularizer_weight": 0.1}}
    "ssl_config.method_params.regularizer_weight": 0.1
                                                   -> nested SSL config fields
    "ssl_config": "configs/ssl_methods/hoffer_v2.json"
                                                   -> a different SSL config file

Nested objects are sparse -- ``loss_params`` and an object-valued ``ssl_config``
change only the keys they name. A path-valued ``ssl_config`` swaps the method
file but keeps the run's data condition (label split, seeds, unlabeled pool),
which every variant of an ablation has to share; the study's tuned SSL
parameters are then applied to the new file exactly as they were to the old.

One file holds a whole ablation. ``variants`` names each change set; ``sweep``
is the one-key-at-a-time shorthand and expands to one variant per value, named
``<last key segment>_<value>``::

    {
      "name": "hoffer_training",
      "sweep": {"sampler_m": [2, 8], "loss.NTXentLoss.temperature": [0.01, 0.1]},
      "variants": {"no_warmup": {"ssl_config": {"warmup_epochs": 0}}}
    }

    -> sampler_m_2, sampler_m_8, temperature_0p01, temperature_0p1, no_warmup

A run executes one variant (``--ablation_variant``); the run scheduler expands a
manifest run's ``ablation_config`` into one child per variant.

Because the changes are sparse, a typo would silently ablate nothing and turn
into a null result, so unknown CLI arguments, loss/miner keys the run's
components do not take, and unknown SSL fields are rejected rather than
ignored. So are keys the replay owns: seeds, outer-grid dimensions and the
label split vary per scenario, and a variant that pinned one would collapse
every scenario onto the same value.

Only the standard library is imported at module level, so the PyTorch-free run
scheduler can expand variants with this module.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

ABLATION_CONFIG_KEYS = {"name", "description", "sweep", "variants", "overrides"}

# Variant names become part of run names and directory names.
VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SSL_CONFIG_KEY = "ssl_config"
SSL_CONFIG_PREFIX = "ssl_config."
# The search-space alias for ``ssl_config.``.
SSL_ALIAS_PREFIX = "ssl."
COMPONENT_PARAMS_KEYS = {"loss_params": "loss", "miner_params": "miner"}
COMPONENTS = ("loss", "miner")
BATCH_SAMPLER_KEY = "batch_sampler"
BATCH_SAMPLER_PARTS = ("batch_size", "sampler_m")
LABELED_BATCH_SIZE_KEY = "labeled_batch_size"
LABELED_BATCH_FRACTION_KEYS = frozenset(
    {"labeled_batch_fraction", "ssl.labeled_batch_fraction", "ssl_config.labeled_batch_fraction"}
)

# Owned by the launcher and the replay request rather than by any one variant.
LAUNCH_ARGUMENTS = frozenset(
    {
        "ablation_config",
        "ablation_variant",
        "device",
        "experiment_config",
        "final_test_study_dir",
        "final_test_top_n",
        "final_test_trial_numbers",
        "hparam_config",
        "save_dir",
        "ssl_device",
        "study_dir_mode",
    }
)
# What every variant is compared under. A replay varies the seeds and grid
# dimensions per scenario after the variant is chosen, so a variant pinning one
# would collapse every scenario onto it; the dataset, protocol and loss/miner
# are the baseline the comparison is against, and a different loss would not
# even take the tuned loss parameters.
HELD_ARGUMENTS = frozenset(
    {
        "comparison_seed_targets",
        "comparison_seeds",
        "cv_k",
        "cv_mode",
        "data_split_seed",
        "dataset",
        "dataset_protocol",
        "hparam_seed",
        "k_shot_grid",
        "label_budget_grid",
        "loss",
        "loss_miner_grid",
        "miner",
        "mode",
        "seed",
        "ssl_label_sampling_modes",
        "support_seed",
        "val_mode",
    }
)
# SSL fields a variant may not change. The unlabeled pool is deliberately not
# among them -- a smaller or labeled-only pool is a legitimate ablation -- and
# the method changes by swapping the SSL config file, which brings its
# method_params along instead of leaving the old method's behind.
HELD_SSL_FIELDS = frozenset(
    {"label_sampling_mode", "labeled_fraction", "labeled_per_class", "method", "seed", "support_seed"}
)
# SSL fields a swapped SSL config file inherits from the config it replaces.
DATA_CONDITION_SSL_FIELDS = (
    "label_sampling_mode",
    "labeled_fraction",
    "labeled_per_class",
    "seed",
    "support_seed",
    "unlabeled_class_scope",
    "unlabeled_fraction",
    "max_unlabeled_samples",
)


def load_ablation_config(path):
    """Read one ablation config file and expand it into its named variants."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ablation config not found: {path}")
    with path.open(encoding="utf-8") as config_file:
        values = json.load(config_file)
    return parse_ablation_config(values, source=str(path), default_name=path.stem)


def parse_ablation_config(values, source, default_name):
    """Validate an ablation config object and expand its sweep into variants."""

    if not isinstance(values, dict):
        raise ValueError(f"Ablation config must be a JSON object: {source}")
    unknown = sorted(set(values) - ABLATION_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown keys in ablation config {source}: {unknown}. "
            f"Expected any of {sorted(ABLATION_CONFIG_KEYS)}"
        )
    name = str(values.get("name") or default_name)
    variants = {}
    if "overrides" in values:
        # The original one-ablation-per-file form, kept so existing files load.
        if "sweep" in values or "variants" in values:
            raise ValueError(
                f"Ablation config {source} mixes 'overrides' with 'sweep'/'variants'; "
                "move the overrides into a named variant"
            )
        add_variant(variants, name, values["overrides"], source)
    for key, choices in require_object(values.get("sweep", {}), f"{source}: sweep").items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{source}: sweep[{key!r}] must be a non-empty list of values")
        for value in choices:
            if isinstance(value, (dict, list)):
                raise ValueError(
                    f"{source}: sweep[{key!r}] holds {value!r}. A sweep moves one scalar at a "
                    "time; write a change to several keys as a named variant"
                )
            add_variant(variants, sweep_variant_name(key, value), {key: value}, source)
    for variant_name, changes in require_object(
        values.get("variants", {}), f"{source}: variants"
    ).items():
        add_variant(variants, variant_name, changes, source)
    if not variants:
        raise ValueError(
            f"Ablation config {source} defines no variants. Set 'sweep' and/or 'variants'; an "
            "ablation that changes nothing is the baseline, so run that without --ablation_config"
        )
    return {
        "name": name,
        "description": values.get("description"),
        "path": source,
        "variants": variants,
    }


def require_object(value, source):
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object")
    return value


def add_variant(variants, name, changes, source):
    if not isinstance(name, str) or not VARIANT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{source}: variant name {name!r} must match {VARIANT_NAME_PATTERN.pattern!r}, "
            "because it becomes part of a run name and a directory name"
        )
    if name in variants:
        raise ValueError(
            f"{source}: duplicate variant name {name!r}. Two sweep values, or a sweep value and a "
            "named variant, resolve to the same name; give one of them its own name under 'variants'"
        )
    variants[name] = normalize_changes(changes, f"{source}: variant {name!r}")


def sweep_variant_name(key, value):
    """Name a one-key variant after the key's last segment and its value."""

    if key == SSL_CONFIG_KEY and isinstance(value, str):
        token = Path(value).stem
    elif isinstance(value, bool) or value is None:
        token = str(value).lower()
    elif isinstance(value, (int, float)):
        token = f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")
    else:
        token = str(value)
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-")
    return f"{key.split('.')[-1]}_{token}"


def normalize_changes(changes, source):
    """Rewrite one variant into the flat dotted vocabulary the run applies.

    Experiment-config-shaped values are flattened -- ``loss_params`` into
    ``loss.<param>``, an object-valued ``ssl_config`` into ``ssl_config.<path>``
    -- so every key names exactly one value, and two spellings of the same value
    in one variant are caught here instead of one of them silently winning.
    """

    if not isinstance(changes, dict) or not changes:
        raise ValueError(f"{source} must be a non-empty object of changes")
    normalized = {}

    def put(key, value):
        if key in normalized:
            raise ValueError(f"{source} sets {key!r} twice")
        normalized[key] = value

    for key, value in changes.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{source} has a non-string key: {key!r}")
        if key == SSL_CONFIG_KEY:
            if isinstance(value, str) and value:
                put(key, value)
            elif isinstance(value, dict):
                for path, leaf in flatten_object(value, SSL_CONFIG_KEY, source):
                    put(path, leaf)
            else:
                raise ValueError(
                    f"{source}: ssl_config must be a path to an SSL config file or an object of "
                    f"SSL config fields, not {value!r}"
                )
        elif key in COMPONENT_PARAMS_KEYS:
            component = COMPONENT_PARAMS_KEYS[key]
            if not isinstance(value, dict) or not value:
                raise ValueError(
                    f"{source}: {key} must be a non-empty object of {component} parameters"
                )
            for parameter, leaf in value.items():
                put(f"{component}.{parameter}", leaf)
        elif key.startswith(SSL_ALIAS_PREFIX):
            put(SSL_CONFIG_PREFIX + key[len(SSL_ALIAS_PREFIX):], value)
        elif key == LABELED_BATCH_SIZE_KEY:
            put(SSL_CONFIG_PREFIX + key, value)
        else:
            put(key, value)
    validate_changes(normalized, source)
    return normalized


def flatten_object(value, prefix, source):
    if not value:
        raise ValueError(f"{source}: {prefix} is an empty object, which changes nothing")
    for key, leaf in value.items():
        path = f"{prefix}.{key}"
        if isinstance(leaf, dict):
            yield from flatten_object(leaf, path, source)
        else:
            yield path, leaf


def validate_changes(changes, source):
    """Reject what a variant must not change, and changes that contradict each other."""

    component_targets = {}
    for key in changes:
        if key in LAUNCH_ARGUMENTS:
            raise ValueError(
                f"{source} sets {key!r}, which the launcher and the replay request own"
            )
        if key in HELD_ARGUMENTS:
            raise ValueError(
                f"{source} sets {key!r}. An ablation holds the data condition, seeds, outer grid "
                "and loss/miner fixed so every variant is compared against the same baseline; "
                "vary those in the manifest instead"
            )
        if key in LABELED_BATCH_FRACTION_KEYS:
            raise ValueError(
                f"{source} sets {key!r}, which exists only inside a search space; set the "
                "absolute ssl_config.labeled_batch_size instead"
            )
        if key.startswith(SSL_CONFIG_PREFIX):
            path = key[len(SSL_CONFIG_PREFIX):].split(".")
            if not all(path):
                raise ValueError(f"{source} has a malformed SSL config key {key!r}")
            if path[0] == "method":
                raise ValueError(
                    f"{source} sets {key!r}; swap the SSL config file to change the method"
                )
            if path[0] in HELD_SSL_FIELDS:
                raise ValueError(
                    f"{source} sets {key!r}. The label split and the seeds are the replay's to "
                    "vary, not a variant's"
                )
        parts = key.split(".")
        if parts[0] in COMPONENTS and len(parts) > 1:
            if len(parts) > 3 or not all(parts):
                raise ValueError(
                    f"{source}: {key!r} must look like {parts[0]}.<param> or "
                    f"{parts[0]}.<Class>.<param>"
                )
            target = (parts[0], parts[-1])
            if target in component_targets:
                raise ValueError(
                    f"{source} sets {parts[0]} parameter {parts[-1]!r} through both "
                    f"{component_targets[target]!r} and {key!r}"
                )
            component_targets[target] = key
    if BATCH_SAMPLER_KEY in changes and any(part in changes for part in BATCH_SAMPLER_PARTS):
        raise ValueError(
            f"{source} sets batch_sampler together with batch_size or sampler_m; set one or the other"
        )


def split_changes(changes):
    """Route each change to the argument, component, SSL field or SSL file it names."""

    argument_changes = {}
    component_changes = {}
    ssl_changes = {}
    ssl_config_file = None
    for name, value in changes.items():
        if name == SSL_CONFIG_KEY:
            ssl_config_file = value
        elif name.startswith(SSL_CONFIG_PREFIX):
            ssl_changes[name] = value
        elif name.split(".")[0] in COMPONENTS and "." in name:
            component_changes[name] = value
        else:
            argument_changes[name] = value
    return argument_changes, component_changes, ssl_changes, ssl_config_file


def select_variant(config, requested):
    variants = config["variants"]
    if requested is None:
        if len(variants) != 1:
            raise ValueError(
                f"Ablation config {config['path']} defines {len(variants)} variants; choose one "
                f"with --ablation_variant: {list(variants)}"
            )
        return next(iter(variants))
    if requested not in variants:
        raise ValueError(
            f"Ablation config {config['path']} has no variant {requested!r}; it defines "
            f"{list(variants)}"
        )
    return requested


def resolve_ablation_config(args):
    """Return this run's selected ablation variant, loading it at most once per args."""

    path = getattr(args, "ablation_config", None)
    requested = getattr(args, "ablation_variant", None)
    if path is None:
        if requested is not None:
            raise ValueError("--ablation_variant requires --ablation_config")
        return None
    resolved = getattr(args, "ablation_config_resolved", None)
    if (
        resolved is not None
        and resolved.get("path") == str(Path(path))
        and requested in (None, resolved.get("variant"))
    ):
        return resolved
    config = load_ablation_config(path)
    variant = select_variant(config, requested)
    resolved = {
        "name": config["name"],
        "description": config["description"],
        "path": config["path"],
        "variant": variant,
        "changes": config["variants"][variant],
    }
    # Recorded on args so the run's saved metadata names the ablation and the
    # variant it is, rather than leaving a reader to diff two configs.
    args.ablation_config_resolved = resolved
    from loguru import logger

    logger.info(
        f"Ablation {resolved['name']!r}, variant {variant!r} from {resolved['path']}"
        + (f": {resolved['description']}" if resolved.get("description") else "")
        + f"; changes {json.dumps(resolved['changes'], sort_keys=True)}"
    )
    return resolved


def prepare_ablation(args):
    """Check this run's variant against its args and swap its SSL config file.

    Called once per request before any SSL config is read. A misspelled argument
    or SSL field then fails before data loading rather than partway into the
    first fold, and the swapped file is the one every outer-grid scenario,
    replayed trial and fold is built from. Everything else waits for
    :func:`apply_ablation_args` and :func:`apply_ablation_ssl_config`, which run
    last.
    """

    resolved = resolve_ablation_config(args)
    if resolved is None:
        return args
    argument_changes, _, ssl_changes, ssl_config_file = split_changes(resolved["changes"])
    unknown = sorted(
        name for name in argument_changes if name != BATCH_SAMPLER_KEY and not hasattr(args, name)
    )
    if unknown:
        raise ValueError(
            f"Ablation variant {resolved['variant']!r} sets unknown training arguments {unknown}"
        )
    if BATCH_SAMPLER_KEY in argument_changes:
        from .hpo import parse_batch_sampler_choice

        parse_batch_sampler_choice(
            argument_changes[BATCH_SAMPLER_KEY],
            source=f"ablation variant {resolved['variant']!r}",
        )
    if ssl_config_file is not None and "ssl_config_source" not in resolved:
        args.ssl_config = swap_ssl_config_file(args, resolved, ssl_config_file)
    if ssl_changes:
        apply_ssl_changes(load_run_ssl_config(args, args.ssl_config), ssl_changes, resolved)
    return args


def load_run_ssl_config(args, path):
    from . import semi_supervised
    from .cli import get_support_seed

    return semi_supervised.load_ssl_config(
        path,
        default_seed=args.seed,
        default_support_seed=get_support_seed(args),
    )


def swap_ssl_config_file(args, resolved, replacement_path):
    """Write the SSL config a file-swapping variant trains with and return its path.

    The replacement file supplies the method; the data condition stays the one
    the run was configured with, which for a study replay is the study's own
    scenario config. The merged config is written beside the run's outputs so
    the path the run records is a file that exists.
    """

    from loguru import logger

    from . import semi_supervised
    from .io import write_json

    current = load_run_ssl_config(args, args.ssl_config)
    replacement = load_run_ssl_config(args, replacement_path)
    swapped = replace(
        replacement,
        **{name: getattr(current, name) for name in DATA_CONDITION_SSL_FIELDS},
    )
    semi_supervised.validate_ssl_config(swapped)
    output_path = (
        Path("logs") / args.save_dir / "ablation" / f"{resolved['variant']}_ssl_config.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, swapped.to_dict())
    resolved["ssl_config_source"] = None if args.ssl_config is None else str(args.ssl_config)
    resolved["ssl_config_replacement"] = str(replacement_path)
    logger.info(
        f"Ablation variant {resolved['variant']!r} swaps SSL config {resolved['ssl_config_source']} "
        f"for {replacement_path}, keeping its {list(DATA_CONDITION_SSL_FIELDS)}; wrote {output_path}"
    )
    return output_path


def apply_ablation_args(args):
    """Apply the variant's argument and loss/miner changes to args in place.

    Runs on every trial's args right after its tuned parameters and again when
    the SSL config is resolved, so it has to be idempotent: arguments are
    assigned and component parameters replaced, never accumulated. Replacing is
    also where this differs from a search space, which rejects a parameter named
    twice -- here the tuned value is exactly what the variant exists to
    contradict.
    """

    resolved = resolve_ablation_config(args)
    if resolved is None:
        return args
    # Imported here rather than at module scope: hpo imports engine, engine
    # imports this module, and the scheduler must not import either.
    from .hpo import component_override_applies, set_arg_value

    argument_changes, component_changes, _, _ = split_changes(resolved["changes"])
    for name, value in argument_changes.items():
        set_arg_value(args, name, value)
    for name, value in component_changes.items():
        if not component_override_applies(args, name):
            raise ValueError(
                f"Ablation variant {resolved['variant']!r} sets {name!r}, which this run's "
                f"loss={getattr(args, 'loss', None)!r} and miner={getattr(args, 'miner', None)!r} "
                "do not take. Give runs with a different loss or miner their own ablation config"
            )
        component = name.split(".")[0]
        params_attr = f"{component}_params"
        params = dict(getattr(args, params_attr, None) or {})
        params[name.split(".")[-1]] = value
        setattr(args, params_attr, params)
    return args


def apply_ablation_ssl_config(args, ssl_config):
    """Apply the variant's SSL field changes last, so they outrank every other source."""

    resolved = resolve_ablation_config(args)
    if resolved is None:
        return ssl_config
    _, _, ssl_changes, _ = split_changes(resolved["changes"])
    if not ssl_changes:
        return ssl_config
    return apply_ssl_changes(ssl_config, ssl_changes, resolved)


def apply_ssl_changes(ssl_config, ssl_changes, resolved):
    from . import semi_supervised
    from .hpo import set_nested_value

    ssl_dict = ssl_config.to_dict()
    for name, value in ssl_changes.items():
        set_nested_value(ssl_dict, name[len(SSL_CONFIG_PREFIX):].split("."), value)
    unknown = sorted(set(ssl_dict) - set(semi_supervised.SemiSupervisedConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"Ablation variant {resolved['variant']!r} sets unknown SSL config fields {unknown}. "
            "Method-specific settings live under ssl_config.method_params"
        )
    resolved_config = semi_supervised.SemiSupervisedConfig(**ssl_dict)
    semi_supervised.validate_ssl_config(resolved_config)
    return resolved_config
