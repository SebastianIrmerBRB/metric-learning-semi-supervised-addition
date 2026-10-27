"""Warm JSON-lines bridge from the Trial Atlas config studio to the repo's validators.

The studio never reimplements a rule.  It asks this service, which imports the
same ``training.cli``/``training.hpo``/``training.run_scheduler`` code the real
runs use, so a config the UI calls valid is a config ``main.py`` accepts.

Protocol: one JSON request object per stdin line, one JSON response object per
stdout line.  Imports and handlers write to stderr only, so the protocol stream
stays clean even when a dependency chatters on import.

    {"id": 1, "op": "schema"}
    {"id": 1, "ok": true, "result": {...}}
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Everything the service imports is noisy on import, so keep the real stdout
# aside before the first import touches it.
PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

with contextlib.redirect_stdout(sys.stderr):
    import pytorch_metric_learning.losses as pml_losses
    import pytorch_metric_learning.miners as pml_miners

    from losses import metric_losses
    from training import cli, hpo, run_scheduler, types
    from training.ssl import config as ssl_config_module


SSL_METHOD_CONFIG_DIR = REPO_ROOT / "configs" / "ssl_methods"
HPARAM_CONFIG_DIRS = (REPO_ROOT / "configs" / "runs", REPO_ROOT / "configs" / "hpo")


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def jsonable(value: Any) -> Any:
    """Return the closest JSON representation of an arbitrary Python value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def type_name(action_type: Any) -> str | None:
    """Name the argparse ``type`` callable in a way the UI can switch on."""

    if action_type is None:
        return None
    if action_type is int:
        return "int"
    if action_type is float:
        return "float"
    if action_type is str:
        return "str"
    if action_type is Path:
        return "path"
    return getattr(action_type, "__name__", None) or "custom"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def describe_cli_arguments() -> list[dict[str, Any]]:
    """Describe every ``main.py`` option, which is also the experiment-config key set."""

    described = []
    for action in cli.parser._actions:
        if action.dest in {"help", "SUPPRESS"}:
            continue
        nargs = action.nargs
        described.append(
            {
                "dest": action.dest,
                "options": list(action.option_strings),
                "type": type_name(action.type),
                "nargs": nargs if isinstance(nargs, (int, str)) or nargs is None else str(nargs),
                "isFlag": nargs == 0,
                "isList": nargs in {"*", "+"} or isinstance(nargs, int),
                "choices": None if action.choices is None else jsonable(list(action.choices)),
                "default": jsonable(action.default),
                "help": (action.help or "").strip(),
                "required": bool(action.required),
            }
        )
    return described


def describe_component_params(component_class: Any) -> list[dict[str, Any]]:
    """Collect constructor keyword names along a class's MRO.

    A pytorch-metric-learning loss takes most of its arguments through the base
    classes it forwards ``**kwargs`` to, so a single ``__init__`` signature would
    miss almost everything the config files actually search.
    """

    collected: dict[str, dict[str, Any]] = {}
    for base in inspect.getmro(component_class):
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        try:
            signature = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, parameter in signature.parameters.items():
            if name == "self" or name in collected:
                continue
            if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                continue
            has_default = parameter.default is not inspect.Parameter.empty
            collected[name] = {
                "name": name,
                "default": jsonable(parameter.default) if has_default else None,
                "hasDefault": has_default,
                "declaredBy": base.__name__,
            }
    return sorted(collected.values(), key=lambda item: item["name"])


def describe_losses_and_miners() -> dict[str, Any]:
    """Map each configurable loss/miner class to its constructor parameters."""

    losses: dict[str, Any] = {}
    for name in types.ALL_LOSSES:
        component = metric_losses.LOSS_REGISTRY.get(name) or getattr(pml_losses, name, None)
        if component is None:
            continue
        try:
            losses[name] = describe_component_params(component)
        except Exception:  # A class the installed version does not expose cleanly.
            losses[name] = []

    miners: dict[str, Any] = {}
    for name in types.ALL_MINERS:
        if name == "no_miner":
            continue
        component = getattr(pml_miners, name, None)
        if component is None:
            continue
        try:
            miners[name] = describe_component_params(component)
        except Exception:
            miners[name] = []
    return {"loss": losses, "miner": miners}


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested objects into the dotted paths an HPO space key uses."""

    flat: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(flatten_json(item, f"{path}."))
        else:
            flat[path] = item
    return flat


def describe_ssl_methods() -> list[dict[str, Any]]:
    """Describe each shipped SSL method config as searchable ``ssl_config.`` keys."""

    described = []
    if not SSL_METHOD_CONFIG_DIR.is_dir():
        return described
    for path in sorted(SSL_METHOD_CONFIG_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        flat = flatten_json(raw)
        described.append(
            {
                "name": path.stem,
                "path": str(path.relative_to(REPO_ROOT)).replace(os.sep, "/"),
                "method": raw.get("method"),
                "keys": [
                    {
                        "key": f"ssl_config.{key}",
                        "value": jsonable(value),
                        "valueType": (
                            "bool" if isinstance(value, bool)
                            else "int" if isinstance(value, int)
                            else "float" if isinstance(value, float)
                            else "str" if isinstance(value, str)
                            else "null" if value is None
                            else "other"
                        ),
                    }
                    for key, value in sorted(flat.items())
                ],
            }
        )
    return described


def describe_ssl_config_fields() -> list[dict[str, Any]]:
    """Describe the ``SemiSupervisedConfig`` fields an ``ssl_config.`` key can target."""

    fields = []
    for name, field in ssl_config_module.SemiSupervisedConfig.__dataclass_fields__.items():
        default = field.default
        if default is not inspect.Parameter.empty and repr(default).startswith("<dataclasses._MISSING"):
            default = None
        fields.append(
            {
                "name": name,
                "key": f"ssl_config.{name}",
                "annotation": str(field.type),
                "default": jsonable(default),
            }
        )
    return fields


def describe_hpo_fields() -> list[dict[str, Any]]:
    """Describe the top-level keys of an Optuna search config."""

    described = []
    for name, field in types.HParamSearchConfig.__dataclass_fields__.items():
        default = field.default
        if repr(default).startswith("<dataclasses._MISSING"):
            default = {} if name in {"sampler_params", "pruner_params", "spaces", "per_loss"} else None
        described.append(
            {
                "name": name,
                "annotation": str(field.type),
                "default": jsonable(default),
            }
        )
    return described


def collect_used_space_keys() -> list[dict[str, Any]]:
    """Count how often each space key appears in the configs already on disk.

    An empirical list beats a curated one here: it stays correct as the search
    spaces evolve, and it puts the keys this project actually searches at the
    top of the picker.
    """

    counts: dict[str, int] = {}
    for root in HPARAM_CONFIG_DIRS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            blocks = [raw.get("spaces")]
            per_loss = raw.get("per_loss")
            if isinstance(per_loss, dict):
                blocks.extend(item.get("spaces") for item in per_loss.values() if isinstance(item, dict))
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                for key in block:
                    counts[key] = counts.get(key, 0) + 1
    return [
        {"key": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_schema() -> dict[str, Any]:
    return {
        "repoRoot": str(REPO_ROOT),
        "python": sys.executable,
        "cliArguments": describe_cli_arguments(),
        "componentParams": describe_losses_and_miners(),
        "sslMethods": describe_ssl_methods(),
        "sslConfigFields": describe_ssl_config_fields(),
        "hpoFields": describe_hpo_fields(),
        "usedSpaceKeys": collect_used_space_keys(),
        "constants": {
            "datasets": jsonable(types.DATASETS),
            "losses": jsonable(types.ALL_LOSSES),
            "classificationLosses": jsonable(types.CLASSIFICATION_LOSSES),
            "miners": jsonable(types.ALL_MINERS),
            "objectiveMetrics": sorted(types.OBJECTIVE_METRICS),
            "selectionMetrics": jsonable(types.SELECTION_METRICS),
            "samplers": ["tpe", "random", "grid"],
            "pruners": ["none", "median", "successive_halving", "hyperband"],
            "directions": ["maximize", "minimize"],
            "labelSamplingModes": sorted(ssl_config_module.LABEL_SAMPLING_MODES),
            "batchSamplerKey": types.BATCH_SAMPLER_HPARAM_KEY,
            "labeledBatchSizeKeys": sorted(types.LABELED_BATCH_SIZE_HPARAM_KEYS),
            "labeledBatchFractionKey": types.LABELED_BATCH_FRACTION_HPARAM_KEY,
            "labeledBatchFractionAliases": sorted(types.LABELED_BATCH_FRACTION_HPARAM_ALIASES),
            "maxLabeledBatchFraction": types.MAX_LABELED_BATCH_FRACTION,
            "comparisonForbiddenKeys": sorted(types.COMPARISON_FORBIDDEN_HPARAM_KEYS),
            "hpoModeKeys": sorted(types.HPO_MODE_KEYS),
            "twoStreamSamplerMethods": sorted(ssl_config_module.TWO_STREAM_SAMPLER_METHODS),
        },
        "manifest": {
            "version": run_scheduler.MANIFEST_VERSION,
            "topLevelKeys": sorted(run_scheduler._TOP_LEVEL_KEYS),
            "runKeys": sorted(run_scheduler._RUN_KEYS),
            "studyReplayKeys": sorted(run_scheduler._STUDY_REPLAY_KEYS),
            "studyDirModes": list(run_scheduler.STUDY_DIR_MODES),
            "defaultStudyDirMode": run_scheduler.DEFAULT_STUDY_DIR_MODE,
            "comparisonSeedTargets": list(run_scheduler.COMPARISON_SEED_TARGETS),
            "finalTestVisualizationModes": list(run_scheduler.FINAL_TEST_VISUALIZATION_MODES),
            "schedulerOwnedOptions": sorted(run_scheduler._SCHEDULER_OWNED_OPTIONS),
            "studyReplayOwnedOptions": sorted(run_scheduler._STUDY_REPLAY_OWNED_OPTIONS),
            "runNamePattern": run_scheduler._RUN_NAME_PATTERN.pattern,
            "defaults": {
                "version": run_scheduler.MANIFEST_VERSION,
                "device": "cuda",
                "python": sys.executable,
                "entrypoint": "main.py",
                "max_concurrent_per_gpu": 1,
                "max_total_runs": None,
                "default_gpu_memory_mib": run_scheduler.DEFAULT_GPU_MEMORY_MIB,
                "reserve_gpu_memory_mib": run_scheduler.DEFAULT_RESERVE_GPU_MEMORY_MIB,
                "poll_interval_seconds": run_scheduler.DEFAULT_POLL_INTERVAL_SECONDS,
                "launch_delay_seconds": 0,
                "fail_fast": False,
                "allow_parallel_trials": False,
                "scheduler_log_dir": "logs/run_scheduler",
            },
        },
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_hparam(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the real Optuna-config validation over an unsaved document.

    ``hpo.load_hparam_config`` reads from disk; this mirrors it exactly for a
    document still being edited, resolving ``extends`` against the directory the
    file would be saved into.
    """

    data = payload.get("data")
    if not isinstance(data, dict):
        return {"valid": False, "errors": ["The search config must be a JSON object."]}

    base_dir = REPO_ROOT / (payload.get("baseDir") or "configs/runs")
    pretend_path = base_dir / (payload.get("fileName") or "unsaved.json")

    raw = json.loads(json.dumps(data))
    parent_reference = raw.pop(hpo.HPARAM_CONFIG_EXTENDS_KEY, None)
    try:
        if parent_reference is not None:
            inherited: dict[str, Any] = {}
            for parent_path in hpo.resolve_parent_config_paths(parent_reference, pretend_path):
                parent = hpo.resolve_hparam_config_inheritance(parent_path)
                inherited = hpo.merge_hparam_config_dicts(inherited, parent)
            raw = hpo.merge_hparam_config_dicts(inherited, raw)

        allowed_keys = set(types.HParamSearchConfig.__dataclass_fields__)
        unknown_keys = sorted(set(raw) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"Unknown hyperparameter config keys: {unknown_keys}")

        config = types.HParamSearchConfig(**raw)
        hpo.validate_hparam_config(config)
        hpo.validate_per_loss_hparam_config(config)
    except Exception as error:
        return {"valid": False, "errors": [f"{type(error).__name__}: {error}"]}
    return {"valid": True, "errors": [], "resolved": jsonable(raw)}


def validate_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    """Type-check an experiment config the way ``cli.load_experiment_config`` does."""

    data = payload.get("data")
    if not isinstance(data, dict):
        return {"valid": False, "errors": ["The experiment config must be a JSON object."]}

    pretend_path = REPO_ROOT / (payload.get("path") or "configs/phase_1/unsaved.json")
    actions = {
        action.dest: action
        for action in cli.parser._actions
        if action.dest not in {"help", "experiment_config"}
    }
    errors: list[str] = []
    unknown_keys = sorted(set(data) - set(actions))
    if unknown_keys:
        errors.append(
            f"Unknown experiment config keys: {unknown_keys}. "
            "Use argparse destination names such as save_dir, ssl_label_sampling_modes, and loss_miner_grid."
        )
    normalized: dict[str, Any] = {}
    for name, value in data.items():
        if name not in actions:
            continue
        try:
            normalized[name] = jsonable(
                cli.normalize_experiment_config_value(actions[name], value, pretend_path)
            )
        except Exception as error:
            errors.append(str(error))
    return {"valid": not errors, "errors": errors, "resolved": normalized}


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Load a run manifest through the scheduler's own strict loader.

    The loader also checks that every referenced experiment/HPO config exists and
    that no run smuggles in trial-level parallelism, which is exactly the class of
    mistake this page exists to prevent.
    """

    data = payload.get("data")
    if not isinstance(data, dict):
        return {"valid": False, "errors": ["The run manifest must be a JSON object."]}

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="studio-manifest-", delete=False, encoding="utf-8"
    )
    try:
        json.dump(data, handle)
        handle.close()
        config = run_scheduler.load_scheduler_config(handle.name)
    except Exception as error:
        return {"valid": False, "errors": [f"{type(error).__name__}: {error}"]}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)

    commands = []
    for run in config.runs:
        if not run.enabled:
            continue
        try:
            command = run_scheduler.build_run_command(run, config)
        except Exception as error:
            return {"valid": False, "errors": [f"{type(error).__name__}: {error}"]}
        commands.append({"name": run.name, "command": [str(part) for part in command]})
    return {"valid": True, "errors": [], "commands": commands}


def inspect_target(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize one referenced config file so the UI can cross-check a manifest."""

    relative = payload.get("path")
    if not isinstance(relative, str) or not relative:
        return {"exists": False}
    path = REPO_ROOT / relative
    if not path.is_file():
        return {"exists": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return {"exists": True, "error": str(error)}
    return {"exists": True, "data": jsonable(raw)}


HANDLERS = {
    "ping": lambda payload: {"pong": True},
    "schema": lambda payload: build_schema(),
    "validate_hparam": validate_hparam,
    "validate_experiment": validate_experiment,
    "validate_manifest": validate_manifest,
    "inspect": inspect_target,
}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            PROTOCOL_OUT.write(json.dumps({"id": None, "ok": False, "error": str(error)}) + "\n")
            PROTOCOL_OUT.flush()
            continue

        request_id = request.get("id")
        handler = HANDLERS.get(request.get("op"))
        if handler is None:
            response = {"id": request_id, "ok": False, "error": f"Unknown op: {request.get('op')!r}"}
        else:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = {"id": request_id, "ok": True, "result": handler(request)}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"id": request_id, "ok": False, "error": f"{type(error).__name__}: {error}"}

        PROTOCOL_OUT.write(json.dumps(response, default=str) + "\n")
        PROTOCOL_OUT.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
