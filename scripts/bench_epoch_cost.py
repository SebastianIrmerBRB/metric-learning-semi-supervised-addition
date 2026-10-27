#!/usr/bin/env python3
"""Measure clean per-epoch training cost and estimate what the HPO studies cost.

The HPO trials ran concurrently on shared GPUs, so their recorded durations say
as much about the server's load as about the work. This script separates the
two.

``run`` (on the server) replays chosen trials of each study alone on one GPU,
through the same ``make_args_and_ssl_config_from_params`` path HPO used, with
early stopping disabled and a fixed epoch budget. Every epoch is timed in three
parts, each bracketed by ``torch.cuda.synchronize``:

* ``gap``: the work between the previous validation and this epoch's training.
  This is where SSL refreshes run (pseudo-labelling, graph construction,
  k-means, loader rebuilds), plus checkpointing of the previous epoch.
* ``train``: ``_EpochTrainer.train_epoch``.
* ``valid``: the per-epoch validation.

Cases run one after another, each in a fresh process, and the GPU is polled for
foreign compute processes throughout; a case that shared the GPU is flagged
``contaminated`` and ignored by the estimate unless asked otherwise.

The epoch budget covers the trial's own warm-up, then one first SSL epoch
(which carries the one-time switch: checkpoint restore, first refresh), then
enough SSL epochs to span whole refresh cycles, so that the mean SSL epoch
amortizes periodic refreshes (SLADE every n epochs) exactly.

``estimate`` (anywhere, standard library only) combines those rates with the
epochs every recorded trial actually ran (``fold_results`` in the Optuna
database) into a cost per trial, study, and method::

    fold  = fixed + min(E, W) * t_warm + [E > W] * t_ssl_first + max(0, E - W - 1) * t_ssl
    trial = sum(fold) + trial_overhead

with E = epochs the fold ran (``last_epoch + 1``) and W = the trial's warm-up.
Only the training part of each rate is scaled when a trial trained with a
different batch size than the benchmarked one (steps per epoch differ); that
approximation is recorded per trial as ``time_model``. ``run`` calls
``estimate`` at the end; call it again to re-estimate without re-measuring.

Examples (server, from the repository root)::

    venv/bin/python scripts/bench_epoch_cost.py run --gpu 0 --dry-run \\
        --manifest configs/final_manifests/cars_all.json
    venv/bin/python scripts/bench_epoch_cost.py run --gpu 0 --output-dir timings/phase1 \\
        --manifest configs/final_manifests/cars_all.json configs/final_manifests/cub_all.json \\
                   configs/final_manifests/in_shop_all.json configs/final_manifests/in_shop_all_10.json

Sweeps: ``run --set KEY=V1,V2`` replays the selected trials again with
parameters overridden, one ``--set`` per key and the grid over all of them,
plus the trial as HPO ran it as a baseline (``--no-baseline`` drops it). Keys
are the names the HPO spaces use and go through the same
``make_args_and_ssl_config_from_params``: a plain name is a training argument
(``batch_sampler=64:4``), ``ssl_config.…`` a nested SSL setting, ``loss.…`` a
loss parameter. Every point is resolved once before the run, so an unknown
key or an invalid combination is reported without training. Swept cases never
price a study; ``sweep_cases.csv`` lists them beside their baseline with the
per-epoch rates and the fold cost at the epochs the trial ran per fold (or at
``--fold-epochs``)::

    venv/bin/python scripts/bench_epoch_cost.py run --gpu 1 --output-dir timings/sweep_cars_lrml \\
        --study-dir /path/to/Cars196/LRML/ArcFace \\
        --set batch_sampler=32:4,64:4,128:4 \\
        --set ssl_config.update_interval_epochs=1,5 \\
        --set ssl_config.method_params.regularizer_target_ratio_update_interval_samples=1000,10000

``--sweep-file`` takes the same as JSON, either a list of override objects
(irregular points) or an object of value lists (a grid).

Examples (anywhere)::

    python3 scripts/bench_epoch_cost.py estimate timings/phase1
    python3 scripts/bench_epoch_cost.py estimate timings/phase1 --output-dir timings/phase0_estimate \\
        --study-dir /mnt/f/Studien/Phase_0

``graph`` times the out-of-batch graph work of Iscen and LRML on synthetic
embeddings, from dataset sizes to a million samples, which no real study
reaches. The embeddings are a Gaussian mixture on the unit sphere (one center
per class, isotropic noise), so the kNN graph is one connected component with
realistic label purity rather than one component per class; purity and
component count are recorded with every case because CHOLMOD's fill depends on
them. Each case calls the production functions in a fresh process:

* Iscen: ``iscen_label_spreading`` once per ``linear_solver`` (``cg``,
  ``cupy_cg``, ``torch_block_cg``, ``cholmod``), split into kNN search, graph
  assembly, solve (CHOLMOD: factorization and solve), row normalization and
  entropy, and the rest (components, degree scaling, system assembly).
* LRML: the global graph refresh (``_build_lrml_graph``, the edge list and the
  node scale) per edge weighting, split into kNN search and graph assembly.

Model forward passes are not part of it; ``run`` measures those on real data.
Sizes run smallest first. A case is killed at ``--case-timeout`` or when its
resident memory passes ``--max-rss-gb``, and larger sizes of the same
configuration are skipped once a smaller one failed or is projected (from the
measured growth) to exceed either limit::

    venv/bin/python scripts/bench_epoch_cost.py graph --dry-run
    setsid nohup venv/bin/python scripts/bench_epoch_cost.py graph \\
        --output-dir timings/graph_synthetic > graph.log 2>&1 < /dev/null &
    venv/bin/python scripts/bench_epoch_cost.py graph --methods iscen --solvers torch_block_cg cholmod \\
        --sizes 100000 1000000 --neighbors 50 --classes 100 1000
"""

from __future__ import annotations

import argparse
import csv
import functools
import glob
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG_FILENAME = "study_config.json"
STUDY_DB_FILENAME = "optuna_study.db"
TRIALS_JSONL_FILENAME = "trials.jsonl"
# The engine writes below logs/<save_dir>; benchmark runs get their own subtree.
BENCH_SAVE_ROOT = Path("_bench_epoch_cost")
SNAPSHOT_FILENAME = "timings.json"
SNAPSHOT_VERSION = 1
RUN_FILENAME = "run.json"
USER_ATTRIBUTE_KEYS = ("params", "resolved_args", "resolved_ssl_config", "fold_results", "last_epoch", "global_step")

METHOD_LABELS = {
    "supervised": "Ü",
    "hoffer_entropy": "Hoffer",
    "slade": "SLADE",
    "lrml": "LRML",
    "iscen_label_spreading": "LP",
}
LOSS_LABELS = {
    "NTXentLoss": "NT-Xent",
    "MultiSimilarityLoss": "MS",
    "CircleLoss": "Circle",
    "ProxyAnchorLoss": "Proxy-Anchor",
    "ArcFaceLoss": "ArcFace",
}
DATASET_LABELS = {
    "Cars196": "Cars196",
    "CUB": "CUB",
    "DeepFashionInShop": "In-Shop",
    "DeepFashionInShop_10pct": "In-Shop 10 %",
}


# --------------------------------------------------------------------------- io


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in columns})


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_input_path(path: str | Path) -> Path:
    """Absolute paths stay; relative ones resolve against cwd, then the repository."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), WORKSPACE_ROOT):
        if (base / candidate).exists():
            return (base / candidate).resolve()
    return (WORKSPACE_ROOT / candidate).resolve()


def study_key_for(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit] or "x"


# ---------------------------------------------------------------- inventory


@dataclass
class FoldEpochs:
    fold: int
    epochs: int
    steps: int | None


@dataclass
class TrialInfo:
    number: int
    state: str
    value: float | None
    params: dict
    batch_size: int | None
    warmup_epochs: int
    update_mode: str | None
    update_interval_epochs: int | None
    duration_seconds: float | None
    folds: list[FoldEpochs]
    has_config: bool


@dataclass
class StudyInfo:
    study_key: str
    study_dir: str
    name: str | None
    direction: str
    dataset: str
    method: str
    loss: str
    ssl_enabled: bool
    cv_k: int | None
    epochs_limit: int | None
    patience: int | None
    num_threads: int | None
    # Study name and first trial start: identical for a copied study, so bench
    # results measured on the server still match studies read from a local copy.
    fingerprint: str = ""
    trials: list[TrialInfo] = field(default_factory=list)

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.dataset, self.method, self.loss)


def decode_attribute(raw: str):
    value = json.loads(raw)
    # Optuna stores attributes as JSON; some writers nested a JSON string inside.
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def dataset_key(args: dict) -> str:
    dataset = str(args.get("dataset"))
    modes = args.get("ssl_label_sampling_modes") or []
    if any("k_shot" in str(mode) for mode in modes):
        return dataset
    budgets = args.get("label_budget_grid") or []
    try:
        budget = float(budgets[0]) if budgets else 1.0
    except (TypeError, ValueError):
        budget = 1.0
    return dataset if budget >= 1 else f"{dataset}_{round(budget * 100):g}pct"


def method_key(ssl_config: dict, args: dict) -> str:
    method = ssl_config.get("method")
    if args.get("mode") != "ssl" or method in (None, "none"):
        return "supervised"
    if method == "regularized":
        return str((ssl_config.get("method_params") or {}).get("regularizer") or method)
    return str(method)


def optional_int(value) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text))
    except ValueError:
        return None


def fold_epochs_from_attributes(attributes: dict) -> list[FoldEpochs]:
    """Epochs each fold ran; ``last_epoch`` is the 0-based index of the last one."""

    folds = []
    fold_results = attributes.get("fold_results")
    if isinstance(fold_results, list) and fold_results:
        for position, fold in enumerate(fold_results):
            last_epoch = optional_int(fold.get("last_epoch"))
            if last_epoch is None:
                continue
            index = optional_int(fold.get("cv_fold"))
            folds.append(
                FoldEpochs(
                    fold=position if index is None else index,
                    epochs=max(0, last_epoch + 1),
                    steps=optional_int(fold.get("global_step")),
                )
            )
        return folds
    last_epoch = optional_int(attributes.get("last_epoch"))
    if last_epoch is not None:
        # Holdout runs (cv_k == 1) record the single run at trial level.
        folds.append(FoldEpochs(0, max(0, last_epoch + 1), optional_int(attributes.get("global_step"))))
    return folds


def storage_path(study_dir: Path, study_config: dict) -> Path:
    # A study copy's own database first: resolved_storage still names the folder the search ran in,
    # and a later search there (In-Shop 10 % LRML, Hoffer/ArcFace) left a one-trial database behind.
    own = study_dir / STUDY_DB_FILENAME
    if own.is_file():
        return own
    storage = str(study_config.get("resolved_storage") or "")
    if storage.startswith("sqlite:///"):
        candidate = Path(storage[len("sqlite:///"):])
        if candidate.is_file():
            return candidate
    return own


def study_rows_from_database(database: Path, study_config: dict, study_dir: Path):
    """``(study_name, direction, rows)`` from the Optuna database, or None."""

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        studies = connection.execute("SELECT study_id, study_name FROM studies").fetchall()
        wanted = study_config.get("resolved_study_name")
        study_ids = [sid for sid, sname in studies if sname == wanted] or [sid for sid, _ in studies]
        if len(study_ids) != 1:
            print(f"skip {study_dir}: {len(study_ids)} studies in the database, none named {wanted!r}", file=sys.stderr)
            return None
        study_id = study_ids[0]
        study_name = dict(studies)[study_id]
        try:
            direction_row = connection.execute(
                "SELECT direction FROM study_directions WHERE study_id = ? ORDER BY objective LIMIT 1", (study_id,)
            ).fetchone()
            values = dict(
                connection.execute(
                    "SELECT tv.trial_id, tv.value FROM trial_values tv JOIN trials t ON t.trial_id = tv.trial_id "
                    "WHERE t.study_id = ? AND tv.objective = 0",
                    (study_id,),
                ).fetchall()
            )
        except sqlite3.OperationalError:
            # Pre-2.4 Optuna schema: one objective, stored on the trial row.
            direction_row = connection.execute("SELECT direction FROM studies WHERE study_id = ?", (study_id,)).fetchone()
            values = dict(connection.execute("SELECT trial_id, value FROM trials WHERE study_id = ?", (study_id,)).fetchall())
        direction = str(direction_row[0] if direction_row else "MAXIMIZE").upper()
        trial_rows = connection.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials WHERE study_id = ?",
            (study_id,),
        ).fetchall()
        placeholders = ",".join("?" for _ in USER_ATTRIBUTE_KEYS)
        attributes: dict[int, dict] = {}
        for trial_id, key, raw in connection.execute(
            "SELECT a.trial_id, a.key, a.value_json FROM trial_user_attributes a JOIN trials t "
            f"ON t.trial_id = a.trial_id WHERE t.study_id = ? AND a.key IN ({placeholders})",
            (study_id, *USER_ATTRIBUTE_KEYS),
        ):
            attributes.setdefault(trial_id, {})[key] = decode_attribute(raw)
    finally:
        connection.close()
    rows = [
        {
            "number": int(number),
            "state": str(state),
            "value": values.get(trial_id),
            "start": started,
            "complete": completed,
            "attrs": attributes.get(trial_id, {}),
        }
        for trial_id, number, state, started, completed in trial_rows
    ]
    return study_name, direction, rows


def sqlite_time(text: str | None) -> str | None:
    """Optuna's SQLite spelling of a timestamp, so a study hashes alike from either source."""

    moment = parse_time(text)
    return moment.isoformat(sep=" ") if moment else text


def study_rows_from_jsonl(path: Path, study_config: dict, study_dir: Path):
    """``(study_name, direction, rows)`` from ``trials.jsonl``, which study copies keep instead of the database."""

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trial = json.loads(line)
            attrs = trial.get("user_attrs") or {}
            rows.append(
                {
                    "number": int(trial["number"]),
                    "state": str(trial.get("state")),
                    "value": trial.get("value"),
                    "start": sqlite_time(trial.get("datetime_start")),
                    "complete": sqlite_time(trial.get("datetime_complete")),
                    "attrs": {
                        key: decode_attribute(json.dumps(attrs[key])) for key in USER_ATTRIBUTE_KEYS if key in attrs
                    },
                }
            )
    if not rows:
        print(f"skip {study_dir}: {path.name} holds no trials", file=sys.stderr)
        return None
    study_name = study_config.get("resolved_study_name") or study_dir.name
    direction = str((study_config.get("hparam_config") or {}).get("direction") or "maximize").upper()
    return study_name, direction, rows


def read_study(study_dir: Path, name: str | None) -> StudyInfo | None:
    config_path = study_dir / STUDY_CONFIG_FILENAME
    study_config = read_json(config_path) if config_path.is_file() else {}
    database = storage_path(study_dir, study_config)
    if database.is_file():
        loaded = study_rows_from_database(database, study_config, study_dir)
    elif (study_dir / TRIALS_JSONL_FILENAME).is_file():
        loaded = study_rows_from_jsonl(study_dir / TRIALS_JSONL_FILENAME, study_config, study_dir)
    else:
        print(f"skip {study_dir}: neither {STUDY_DB_FILENAME} nor {TRIALS_JSONL_FILENAME}", file=sys.stderr)
        return None
    if loaded is None:
        return None
    study_name, direction, rows = loaded

    info = None
    trials = []
    for row in sorted(rows, key=lambda row: row["number"]):
        attrs = row["attrs"]
        args = attrs.get("resolved_args") or {}
        ssl_config = attrs.get("resolved_ssl_config") or {}
        if info is None and args:
            method = method_key(ssl_config, args)
            info = StudyInfo(
                study_key=study_key_for(study_dir),
                study_dir=str(study_dir),
                name=name,
                direction=direction,
                dataset=dataset_key(args),
                method=method,
                loss=str(args.get("loss")),
                ssl_enabled=method != "supervised",
                cv_k=optional_int(args.get("cv_k")),
                epochs_limit=optional_int(args.get("epochs")),
                patience=optional_int(args.get("patience")),
                num_threads=optional_int(args.get("num_threads")),
            )
        start, end = parse_time(row["start"]), parse_time(row["complete"])
        ssl_enabled = bool(args) and method_key(ssl_config, args) != "supervised"
        trials.append(
            TrialInfo(
                number=row["number"],
                state=row["state"],
                value=row["value"],
                params=attrs.get("params") or {},
                batch_size=optional_int(args.get("batch_size")),
                warmup_epochs=int(ssl_config.get("warmup_epochs") or 0) if ssl_enabled else 0,
                update_mode=ssl_config.get("update_mode"),
                update_interval_epochs=optional_int(ssl_config.get("update_interval_epochs")),
                duration_seconds=(end - start).total_seconds() if start and end else None,
                folds=fold_epochs_from_attributes(attrs),
                has_config=bool(args) and "params" in attrs,
            )
        )
    if info is None:
        print(f"skip {study_dir}: no trial recorded resolved_args", file=sys.stderr)
        return None
    first_start = min(rows, key=lambda row: row["number"])["start"]
    info.fingerprint = hashlib.sha1(f"{study_name}|{first_start}".encode()).hexdigest()[:12]
    info.trials = trials
    return info


def walk_study_dirs(root: Path) -> list[Path]:
    if (root / STUDY_DB_FILENAME).is_file() or (root / STUDY_CONFIG_FILENAME).is_file():
        return [root]
    found = []
    # followlinks: the Studien folders are junctions into server_logs.
    for directory, _, files in os.walk(root, followlinks=True):
        if STUDY_DB_FILENAME in files or (STUDY_CONFIG_FILENAME in files and TRIALS_JSONL_FILENAME in files):
            found.append(Path(directory))
    return sorted(found)


def manifest_study_dir(value: str) -> Path:
    """Where the run scheduler finds a manifest's study_dir: as given, then below logs/."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (resolve_input_path(path), WORKSPACE_ROOT / "logs" / path):
        if candidate.is_dir():
            return candidate.resolve()
    return resolve_input_path(path)


def collect_study_refs(study_dirs: list[str], manifests: list[str]) -> list[tuple[Path, str | None]]:
    refs: dict[Path, str | None] = {}
    for manifest in manifests or []:
        runs = [run for run in read_json(resolve_input_path(manifest)).get("runs", []) if run.get("study_dir")]
        # An ablation replay adds no HPO cost beside its study's own run, so a
        # manifest mixing both counts the study once. A manifest of replays only
        # (the phase-2 geometry runs) names its studies through them; their
        # ablation args are not applied, the trial is timed as HPO ran it.
        replays_only = all(run.get("args") for run in runs)
        for run in runs:
            if run.get("enabled") is False or (run.get("args") and not replays_only):
                continue
            refs.setdefault(manifest_study_dir(run["study_dir"]), run.get("name"))
    for pattern in study_dirs or []:
        matches = sorted(glob.glob(str(resolve_input_path(pattern)))) or [str(resolve_input_path(pattern))]
        for match in matches:
            for study_dir in walk_study_dirs(Path(match)):
                refs.setdefault(study_dir, None)
    return list(refs.items())


def build_inventory(study_dirs: list[str], manifests: list[str]) -> list[StudyInfo]:
    studies = []
    for study_dir, name in collect_study_refs(study_dirs, manifests):
        info = read_study(study_dir, name)
        if info is not None:
            studies.append(info)
    if not studies:
        raise SystemExit("No readable studies; pass --manifest or --study-dir.")
    return studies


def inventory_rows(studies: list[StudyInfo]) -> list[dict]:
    rows = []
    for study in studies:
        for trial in study.trials:
            base = {
                "study_key": study.study_key,
                "fingerprint": study.fingerprint,
                "name": study.name,
                "dataset": study.dataset,
                "method": study.method,
                "loss": study.loss,
                "direction": study.direction,
                "cv_k": study.cv_k,
                "trial": trial.number,
                "state": trial.state,
                "value": trial.value,
                "has_config": int(trial.has_config),
                "batch_size": trial.batch_size,
                "warmup_epochs": trial.warmup_epochs,
                "update_mode": trial.update_mode,
                "update_interval_epochs": trial.update_interval_epochs,
                "duration_seconds": trial.duration_seconds,
            }
            if not trial.folds:
                rows.append({**base, "fold": None, "epochs": None, "steps": None})
            for fold in trial.folds:
                rows.append({**base, "fold": fold.fold, "epochs": fold.epochs, "steps": fold.steps})
    return rows


def studies_from_inventory(path: Path) -> list[StudyInfo]:
    """Rebuild the studies from a trials.csv written by ``inventory_rows``."""

    studies: dict[str, StudyInfo] = {}
    trials: dict[tuple[str, int], TrialInfo] = {}
    for row in read_csv(path):
        key = row["study_key"]
        study = studies.get(key)
        if study is None:
            study = studies[key] = StudyInfo(
                study_key=key,
                study_dir=key,
                name=row.get("name") or None,
                direction=row.get("direction") or "MAXIMIZE",
                dataset=row["dataset"],
                method=row["method"],
                loss=row["loss"],
                ssl_enabled=row["method"] != "supervised",
                cv_k=optional_int(row.get("cv_k")),
                epochs_limit=None,
                patience=None,
                num_threads=None,
                fingerprint=row.get("fingerprint") or "",
            )
        number = int(row["trial"])
        trial = trials.get((key, number))
        if trial is None:
            value = row.get("value")
            duration = row.get("duration_seconds")
            trial = trials[(key, number)] = TrialInfo(
                number=number,
                state=row["state"],
                value=float(value) if value not in (None, "") else None,
                params={},
                batch_size=optional_int(row.get("batch_size")),
                warmup_epochs=optional_int(row.get("warmup_epochs")) or 0,
                update_mode=row.get("update_mode") or None,
                update_interval_epochs=optional_int(row.get("update_interval_epochs")),
                duration_seconds=float(duration) if duration not in (None, "") else None,
                folds=[],
                has_config=row.get("has_config", "1") not in ("0", ""),
            )
            study.trials.append(trial)
        if row.get("fold") not in (None, ""):
            trial.folds.append(FoldEpochs(int(row["fold"]), int(row["epochs"]), optional_int(row.get("steps"))))
    return list(studies.values())


# --------------------------------------------------------------- case plan


@dataclass
class Case:
    case_id: str
    study_key: str
    fingerprint: str
    study_dir: str
    name: str | None
    dataset: str
    method: str
    loss: str
    trial: int
    repeat: int
    batch_size: int | None
    warmup_epochs: int
    update_mode: str | None
    update_interval_epochs: int | None
    ssl_enabled: bool
    epochs: int
    cv_k: int | None
    num_threads: int | None
    params: dict
    # Applied on top of the trial's params; empty for the trial as HPO ran it.
    overrides: dict = field(default_factory=dict)


def override_key(overrides: dict) -> str:
    return hashlib.sha1(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:8] if overrides else ""


def setting_label(overrides: dict) -> str:
    if not overrides:
        return "(as in the study)"
    return ", ".join(f"{key.rsplit('.', 1)[-1]}={value}" for key, value in overrides.items())


def parse_override_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_set_argument(text: str) -> tuple[str, list]:
    """``KEY=V1,V2`` or ``KEY=[json list]``; values are JSON where they parse (5, true, null)."""

    key, separator, raw = text.partition("=")
    key, raw = key.strip(), raw.strip()
    if not separator or not key or not raw:
        raise SystemExit(f"--set expects KEY=VALUE[,VALUE...], got {text!r}")
    values = json.loads(raw) if raw.startswith("[") else [parse_override_value(part.strip()) for part in raw.split(",")]
    if not isinstance(values, list) or not values:
        raise SystemExit(f"--set {key}: no values")
    return key, values


def sweep_points(set_arguments: list[str], sweep_file: str | None, baseline: bool) -> list[dict]:
    """Override dicts to measure: the --set grid crossed with the sweep file's points."""

    grid = [parse_set_argument(text) for text in set_arguments or []]
    keys = [key for key, _ in grid]
    if len(set(keys)) != len(keys):
        raise SystemExit("--set names a key twice; give all its values in one --set")
    points = [dict(zip(keys, values)) for values in itertools.product(*(values for _, values in grid))]
    if sweep_file:
        spec = read_json(resolve_input_path(sweep_file))
        if isinstance(spec, dict):
            spec = [dict(zip(spec, values)) for values in itertools.product(*(list(v) for v in spec.values()))]
        if not isinstance(spec, list) or not all(isinstance(point, dict) for point in spec):
            raise SystemExit("--sweep-file must hold a list of override objects or an object of value lists")
        points = [{**file_point, **point} for file_point in spec for point in points]
    points = [point for point in points if point]
    if not points:
        return [{}]
    if baseline:
        points.insert(0, {})
    unique = {override_key(point): point for point in points}
    return list(unique.values())


def planned_epochs(ssl_enabled: bool, warmup: int, update_mode: str | None, interval: int | None, window: int) -> int:
    """Warm-up, one first SSL epoch, then whole refresh cycles covering ``window``."""

    if not ssl_enabled:
        # Epoch 0 carries per-fold start-up work and is left out of the rate.
        return 1 + window
    cycle = max(1, int(interval or 1)) if update_mode == "every_n_epochs" else 1
    return warmup + 1 + cycle * math.ceil(window / cycle)


def select_trials(study: StudyInfo, mode: str) -> list[TrialInfo]:
    candidates = [trial for trial in study.trials if trial.state == "COMPLETE" and trial.has_config]
    if not candidates:
        return []
    maximize = study.direction != "MINIMIZE"

    def score(trial):
        value = trial.value if trial.value is not None else (-math.inf if maximize else math.inf)
        return value if maximize else -value

    if mode == "best":
        return [max(candidates, key=score)]
    if mode == "per-batch-size":
        best = {}
        for trial in candidates:
            if trial.batch_size not in best or score(trial) > score(best[trial.batch_size]):
                best[trial.batch_size] = trial
        return sorted(best.values(), key=lambda trial: (trial.batch_size or 0))
    if mode.startswith("trials:"):
        numbers = {int(part) for part in mode.split(":", 1)[1].split(",") if part.strip()}
        return [trial for trial in candidates if trial.number in numbers]
    raise SystemExit(f"unknown --select {mode!r}")


def plan_cases(
    studies: list[StudyInfo], select: str, window: int, repeats: int, points: list[dict] | None = None
) -> list[Case]:
    """Cases per selected trial, override point and repeat.

    Override points take their warm-up, refresh schedule and batch size from
    ``resolve_override_cases``; until then they carry the trial's own.
    """

    cases = []
    for study in studies:
        digest = hashlib.sha1(study.study_key.encode()).hexdigest()[:8]
        stem = f"{slug(study.dataset, 20)}_{slug(study.method, 16)}_{slug(study.loss, 20)}_{digest}"
        for trial, overrides, repeat in itertools.product(select_trials(study, select), points or [{}], range(repeats)):
            # A baseline keeps its old id, so earlier measurements are reused.
            suffix = f"_o{override_key(overrides)}" if overrides else ""
            cases.append(
                Case(
                    case_id=f"{stem}_t{trial.number:04d}{suffix}_r{repeat}",
                    study_key=study.study_key,
                    fingerprint=study.fingerprint,
                    study_dir=study.study_dir,
                    name=study.name,
                    dataset=study.dataset,
                    method=study.method,
                    loss=study.loss,
                    trial=trial.number,
                    repeat=repeat,
                    batch_size=trial.batch_size,
                    warmup_epochs=trial.warmup_epochs,
                    update_mode=trial.update_mode,
                    update_interval_epochs=trial.update_interval_epochs,
                    ssl_enabled=study.ssl_enabled,
                    epochs=planned_epochs(
                        study.ssl_enabled,
                        trial.warmup_epochs,
                        trial.update_mode,
                        trial.update_interval_epochs,
                        window,
                    ),
                    cv_k=study.cv_k,
                    num_threads=study.num_threads,
                    params=trial.params,
                    overrides=dict(overrides),
                )
            )
    return cases


def case_stem(case: Case) -> str:
    digest = hashlib.sha1(case.study_key.encode()).hexdigest()[:8]
    return f"{slug(case.dataset, 20)}_{slug(case.method, 16)}_{slug(case.loss, 20)}_{digest}"


def resolve_override_cases(cases: list[Case], window: int, gpu: dict) -> list[Case]:
    """Resolve every override point once through HPO's own path.

    One subprocess imports the training code, so an unknown key, an invalid
    combination or a batch no m can fill is reported here instead of after a
    GPU wait. A bare batch size in ``batch_sampler`` gets its m chosen there. A
    point that resolves to the trial's own configuration becomes the baseline,
    and points resolving to the same configuration are measured once.
    """

    pending = {}
    for case in cases:
        if case.overrides:
            for overrides in ({}, case.overrides):
                pending.setdefault(
                    f"{case.study_key}|{case.trial}|{override_key(overrides)}",
                    {
                        "study_dir": case.study_dir,
                        "params": case.params,
                        "overrides": overrides,
                        "label": f"{case.name or case.dataset + ' ' + case.method + ' ' + case.loss} "
                        f"trial {case.trial} {setting_label(overrides)}",
                    },
                )
    if not pending:
        return cases
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.get("uuid") or str(gpu.get("index"))
    print(f"resolving {len(pending)} trial settings (imports the training code, then one line per setting) ...", flush=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        request_path, response_path = Path(scratch) / "request.json", Path(scratch) / "response.json"
        stderr_path = Path(scratch) / "stderr.log"
        write_json(request_path, {"items": pending, "response": str(response_path)})
        other_output = []
        with stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "_resolve", str(request_path)],
                cwd=WORKSPACE_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
            )
            for line in process.stdout:
                if line.startswith(RESOLVE_PROGRESS_PREFIX):
                    print(f"  {line[len(RESOLVE_PROGRESS_PREFIX):].rstrip()}", flush=True)
                else:
                    other_output = (other_output + [line])[-50:]
            process.wait()
        if not response_path.is_file():
            errors = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise SystemExit(f"resolving the overrides failed:\n{''.join(other_output)}{errors}")
        answers = read_json(response_path)
    print(f"resolved in {time.monotonic() - started:.0f} s", flush=True)

    kept, seen, reported = [], set(), set()
    collapsed = 0
    for case in cases:
        base = answers.get(f"{case.study_key}|{case.trial}|")
        key = f"{case.study_key}|{case.trial}|{override_key(case.overrides)}"
        answer = answers.get(key) if case.overrides else base
        if answer is not None and not answer["ok"]:
            if key not in reported:
                print(
                    f"  invalid, not planned: {case.dataset} {case.method} {case.loss} trial {case.trial} "
                    f"{setting_label(case.overrides)}: {answer['error']}"
                )
                reported.add(key)
            continue
        if case.overrides and base is not None and base["ok"] and answer["fingerprint"] == base["fingerprint"]:
            # The point is the trial as HPO ran it (e.g. its own batch size): keep one baseline.
            case.overrides = {}
            collapsed += 1
        elif case.overrides:
            case.overrides = answer["overrides"]
            case.batch_size = answer["batch_size"]
            case.warmup_epochs = answer["warmup_epochs"]
            case.update_mode = answer["update_mode"]
            case.update_interval_epochs = answer["update_interval_epochs"]
            case.epochs = planned_epochs(
                case.ssl_enabled, case.warmup_epochs, case.update_mode, case.update_interval_epochs, window
            )
        suffix = f"_o{override_key(case.overrides)}" if case.overrides else ""
        case.case_id = f"{case_stem(case)}_t{case.trial:04d}{suffix}_r{case.repeat}"
        identity = (case.study_key, case.trial, answer["fingerprint"] if answer else "", case.repeat)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(case)
    if collapsed:
        print(f"  {collapsed} swept case(s) equal their trial's own setting and are measured as its baseline")
    return kept


def resolve_trial(main_module, study_dir: Path, params: dict, overrides: dict):
    """The trial's args and SSL config as HPO built them, with ``overrides`` applied last."""

    study_config = read_json(study_dir / STUDY_CONFIG_FILENAME)
    base_args = main_module.make_base_args_from_study_config(study_config["base_args"], study_dir=study_dir)
    params = main_module.expand_joint_component_params(dict(params))
    # After the joint expansion, so an override also wins over an expanded key.
    params.update(overrides or {})
    trial_args, ssl_config = main_module.make_args_and_ssl_config_from_params(base_args, params)
    ssl_enabled = getattr(trial_args, "mode", None) == "ssl" and ssl_config.method not in (None, "none")
    return study_config, base_args, params, trial_args, ssl_config, ssl_enabled


# Everything that decides a fold's labeled samples, and so the capacity check's
# cache key: the SSL fields prepare_label_split reads, and the arguments
# make_label_budget_training_label_sets builds the fold's dataset from.
LABEL_SPLIT_SSL_KEYS = (
    "label_sampling_mode",
    "labeled_fraction",
    "labeled_per_class",
    "max_unlabeled_samples",
    "support_seed",
    "unlabeled_class_scope",
    "unlabeled_fraction",
)
LABEL_SPLIT_ARGS = (
    "dataset",
    "dataset_protocol",
    "seed",
    "data_split_seed",
    "cv_k",
    "cv_mode",
    "val_mode",
    "final_full_train",
    "image_resize_mode",
)
RESOLVE_PROGRESS_PREFIX = "@@ "


def config_fingerprint(trial_args, ssl_config) -> str:
    arguments = {key: value for key, value in vars(trial_args).items() if key != "hparam_params"}
    return hashlib.sha1(
        json.dumps([arguments, ssl_config.to_dict()], sort_keys=True, default=str).encode()
    ).hexdigest()


def batch_sampler_candidates(value, own_m: int) -> list[str]:
    """``"128:4"`` as given; ``128`` only at the trial's own m; ``"128:auto"`` its own m, then doubled until one fits."""

    size_text, _, m_text = str(value).partition(":")
    size = int(size_text)
    if m_text and m_text != "auto":
        return [f"{size}:{int(m_text)}"]
    m = max(1, int(own_m))
    if not m_text:
        return [f"{size}:{m}"] if size % m == 0 else []
    candidates = []
    while m <= size:
        if size % m == 0:
            candidates.append(f"{size}:{m}")
        m *= 2
    return candidates


def sampler_fits(hpo, utils, trial_args, ssl_config, label_sets) -> str | None:
    """Why this batch cannot be served in every fold (as the HPO sampler filter judges it), or None."""

    m = int(trial_args.sampler_m)
    k = ssl_config.labeled_per_class if ssl_config.label_sampling_mode == "class_subset_k_shot" else None
    if k is not None and m > int(k):
        return f"sampler_m={m} exceeds the {k} labeled samples per class"
    two_stream = getattr(trial_args, "mode", None) == "ssl" and ssl_config.labeled_batch_size is not None
    labeled = int(ssl_config.labeled_batch_size) if two_stream else int(trial_args.batch_size)
    # TwoStreamMPerClassBatchSampler fills its pseudo-labeled share M-per-class
    # too, and pseudo-labels only come from seeded classes, so that share is
    # bounded by the fold's labeled classes as well. The HPO filter checks only
    # the labeled share; a replay that breaks this fails after the whole warm-up.
    streams = [("labeled", labeled)] + ([("pseudo-labeled", int(trial_args.batch_size) - labeled)] if two_stream else [])
    for stream, size in streams:
        try:
            for labels in label_sets:
                utils.validate_m_per_class_sampler_capacity(labels, size, m)
        except utils.MPerClassSamplerCapacityError:
            return f"{stream} batch {size} at m={m} needs more classes: {hpo.summarize_training_label_sets(label_sets)}"
    return None


def resolve_main(request_path: Path) -> int:
    request = read_json(request_path)
    os.chdir(WORKSPACE_ROOT)
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    import main
    import utils
    from training import hpo

    def progress(text: str) -> None:
        # The parent shows only these lines; the training code's own logging goes to stderr.
        print(f"{RESOLVE_PROGRESS_PREFIX}{text}", flush=True)

    label_sets_cache: dict[str, list] = {}

    def label_sets(trial_args, ssl_config):
        arguments = {name: getattr(trial_args, name, None) for name in LABEL_SPLIT_ARGS}
        config = {name: getattr(ssl_config, name, None) for name in LABEL_SPLIT_SSL_KEYS}
        key = json.dumps([arguments, config], sort_keys=True, default=str)
        if key not in label_sets_cache:
            started = time.perf_counter()
            progress(
                f"  building the labeled fold splits of {arguments['dataset']} "
                f"({config['label_sampling_mode']}, fraction {config['labeled_fraction']}, "
                f"k {config['labeled_per_class']}) ..."
            )
            label_sets_cache[key] = hpo.make_label_budget_training_label_sets(trial_args, ssl_config)
            progress(f"  ... done in {time.perf_counter() - started:.0f} s")
        return label_sets_cache[key]

    answers = {}
    for index, (key, item) in enumerate(request["items"].items(), start=1):
        study_dir = Path(item["study_dir"])
        overrides = dict(item["overrides"])
        try:
            if "batch_sampler" in overrides:
                _, _, _, own_args, _, _ = resolve_trial(main, study_dir, item["params"], {})
                reasons = []
                for candidate in batch_sampler_candidates(overrides["batch_sampler"], own_args.sampler_m):
                    _, _, _, trial_args, ssl_config, _ = resolve_trial(
                        main, study_dir, item["params"], {**overrides, "batch_sampler": candidate}
                    )
                    reason = sampler_fits(hpo, utils, trial_args, ssl_config, label_sets(trial_args, ssl_config))
                    if reason is None:
                        overrides["batch_sampler"] = candidate
                        break
                    reasons.append(f"{candidate}: {reason}")
                else:
                    raise ValueError(
                        "does not fit; " + "; ".join(reasons or [f"not divisible by sampler_m={own_args.sampler_m}"])
                    )
            _, _, _, trial_args, ssl_config, ssl_enabled = resolve_trial(main, study_dir, item["params"], overrides)
            answers[key] = {
                "ok": True,
                "overrides": overrides,
                "fingerprint": config_fingerprint(trial_args, ssl_config),
                "batch_size": getattr(trial_args, "batch_size", None),
                "warmup_epochs": int(ssl_config.warmup_epochs or 0) if ssl_enabled else 0,
                "update_mode": ssl_config.update_mode,
                "update_interval_epochs": ssl_config.update_interval_epochs,
            }
        except Exception as exc:  # noqa: BLE001 - reported per override point
            answers[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        outcome = "ok" if answers[key]["ok"] else answers[key]["error"].splitlines()[0][:120]
        progress(f"[{index}/{len(request['items'])}] {item.get('label', key)}: {outcome}")
    write_json(Path(request["response"]), answers)
    return 0


# -------------------------------------------------------------- GPU watching


def nvidia_smi(*arguments: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return [[cell.strip() for cell in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def gpu_identity(gpu: str) -> dict:
    try:
        for index, name, uuid, driver, memory in nvidia_smi(
            "--query-gpu=index,name,uuid,driver_version,memory.total"
        ):
            if index == gpu or uuid == gpu:
                return {"index": index, "name": name, "uuid": uuid, "driver": driver, "memory_total_mib": memory}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"index": gpu}


def process_tree(root_pid: int) -> set[int]:
    """``root_pid`` and all its descendants, from /proc."""

    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may contain spaces; ppid follows the closing paren.
        fields = stat[stat.rfind(")") + 2:].split()
        children.setdefault(int(fields[1]), []).append(int(entry.name))
    tree, stack = set(), [root_pid]
    while stack:
        pid = stack.pop()
        if pid not in tree:
            tree.add(pid)
            stack.extend(children.get(pid, ()))
    return tree


def gpu_compute_pids(uuid: str | None) -> set[int] | None:
    try:
        rows = nvidia_smi("--query-compute-apps=gpu_uuid,pid")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return {int(pid) for gpu_uuid, pid in rows if uuid is None or gpu_uuid == uuid}


def gpu_load(gpu: str) -> tuple[float | None, float | None]:
    """Utilization in percent and device memory in use (MiB, all processes)."""

    try:
        rows = nvidia_smi("--id", gpu, "--query-gpu=utilization.gpu,memory.used")
        return float(rows[0][0]), float(rows[0][1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


class GpuWatch:
    """Poll the GPU while a case runs and remember anyone else who used it."""

    def __init__(self, gpu: dict, interval: float):
        self.gpu = gpu
        self.interval = interval
        self.worker_pid: int | None = None
        self.foreign: dict[int, int] = {}
        self.own_pid_seen = False
        self.utilization: list[float] = []
        self.memory_used: list[float] = []
        self.load: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def sample(self):
        pids = gpu_compute_pids(self.gpu.get("uuid"))
        if pids and self.worker_pid is not None:
            own = process_tree(self.worker_pid)
            self.own_pid_seen |= bool(pids & own)
            for pid in pids - own - {os.getpid()}:
                self.foreign[pid] = self.foreign.get(pid, 0) + 1
        utilization, memory_used = gpu_load(self.gpu.get("index", "0"))
        if utilization is not None:
            self.utilization.append(utilization)
        if memory_used is not None:
            self.memory_used.append(memory_used)
        self.load.append(os.getloadavg()[0])

    def report(self) -> dict:
        return {
            "foreign_gpu_pids": sorted(self.foreign),
            # Inside some containers nvidia-smi lists no processes at all; then
            # foreign use cannot be seen and the case is marked unverified.
            "gpu_process_visibility": self.own_pid_seen,
            "contaminated": bool(self.foreign),
            "gpu_utilization_mean": statistics.mean(self.utilization) if self.utilization else None,
            "gpu_memory_used_max_mib": max(self.memory_used) if self.memory_used else None,
            "load_average_max": max(self.load) if self.load else None,
        }


def wait_until_idle(gpu: dict, policy: str, max_load: float) -> None:
    announced = False
    while True:
        pids = gpu_compute_pids(gpu.get("uuid")) or set()
        busy_pids = pids - {os.getpid()}
        load = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
        if not busy_pids:
            if load > max_load:
                print(f"  note: host load average is {load:.2f} per CPU; CPU-side work may be slower", flush=True)
            return
        if policy == "run":
            print(f"  warning: GPU busy (pids {sorted(busy_pids)}); running anyway", flush=True)
            return
        if policy == "abort":
            raise SystemExit(f"GPU {gpu.get('index')} is in use by pids {sorted(busy_pids)}; aborting (--if-busy abort)")
        if not announced:
            print(f"  GPU {gpu.get('index')} in use by pids {sorted(busy_pids)}; waiting ...", flush=True)
            announced = True
        time.sleep(30)


# ------------------------------------------------------------------ worker


class EpochRecorder:
    """Collect synchronized timestamps and named sections from the patched engine entry points.

    A section's time is its own: sections nested in it (an embedding pass inside
    a pseudo-label refresh) are subtracted and recorded under their own name. It
    lands in the part of the epoch it ran in: ``setup`` before the first epoch,
    ``gap`` between the previous validation and this epoch's training (SSL
    refreshes, loaders, checkpoints), ``train``, ``valid``, or ``teardown``
    after the last epoch; ``step`` holds the opt-in per-step timer's sections.
    """

    def __init__(self, torch_module):
        self._torch = torch_module
        self._cuda = torch_module.cuda.is_available()
        self.folds: list[dict] = []
        self.fold: dict | None = None
        self.in_train = False
        self.in_eval = False
        self.stack: list[dict] = []
        self.batch_timer = None

    def now(self) -> float:
        if self._cuda:
            self._torch.cuda.synchronize()
        return time.perf_counter()

    def begin_fold(self, index) -> None:
        start = self.now()
        self.fold = {
            "fold": len(self.folds) if index is None else int(index),
            "start": start,
            "end": None,
            "boundary": start,
            "baseline_valid": 0.0,
            "current": None,
            "epochs": {},
            "sections": {"setup": {}, "teardown": {}},
            "pending_gap": {},
        }
        self.folds.append(self.fold)

    def end_fold(self) -> None:
        if self.fold is not None:
            self.fold["end"] = self.now()
            # Work after the last epoch's validation belongs to no epoch.
            for name, (seconds, calls) in self.fold["pending_gap"].items():
                self._add(self.fold["sections"]["teardown"], name, seconds, calls)
            self.fold = None

    @staticmethod
    def _add(bucket: dict, name: str, seconds: float, calls: int = 1) -> None:
        total, count = bucket.get(name, (0.0, 0))
        bucket[name] = (total + seconds, count + calls)

    def section_bucket(self) -> dict | None:
        fold = self.fold
        if fold is None:
            return None
        if fold["current"] is None and not self.in_train:
            return fold["sections"]["setup"]
        if self.in_eval:
            return fold["epochs"][fold["current"]]["sections"]["valid"] if fold["current"] is not None else fold["sections"]["setup"]
        if self.in_train:
            return fold["epochs"][fold["current"]]["sections"]["train"]
        return fold["pending_gap"]

    def push(self, name: str) -> dict:
        frame = {"name": name, "start": self.now(), "children": 0.0}
        self.stack.append(frame)
        return frame

    def pop(self, frame: dict) -> None:
        inclusive = self.now() - frame["start"]
        self.stack.remove(frame)
        if self.stack:
            self.stack[-1]["children"] += inclusive
        bucket = self.section_bucket()
        if bucket is not None:
            self._add(bucket, frame["name"], max(0.0, inclusive - frame["children"]))

    def train_start(self, epoch: int) -> None:
        start = self.now()
        fold = self.fold
        if fold is None:
            return
        if fold["current"] is None:
            fold["setup_end"] = fold["boundary"]
        fold["epochs"][epoch] = {
            "epoch": epoch,
            "boundary": fold["boundary"],
            "train_start": start,
            "train_end": start,
            "steps": 0,
            "valid": 0.0,
            "other_eval": 0.0,
            "end": start,
            "sections": {"gap": fold["pending_gap"], "train": {}, "valid": {}, "step": {}},
        }
        fold["pending_gap"] = {}
        fold["current"] = epoch
        self.in_train = True
        self.batch_timer = None

    def train_end(self, epoch: int, steps: int) -> None:
        end = self.now()
        self.in_train = False
        fold = self.fold
        if fold is None or epoch not in fold["epochs"]:
            return
        record = fold["epochs"][epoch]
        record.update(train_end=end, steps=steps, end=end)
        if self.batch_timer is not None:
            for name, seconds in self.batch_timer.totals.items():
                self._add(record["sections"]["step"], name, seconds, 0)
        fold["boundary"] = end

    def evaluation(self, name: str, start: float, end: float) -> None:
        fold = self.fold
        if fold is None:
            return
        is_validation = name.startswith("valid")
        if fold["current"] is None:
            # The untrained-head baseline validation closes the fold's setup.
            if is_validation:
                fold["baseline_valid"] += end - start
            fold["boundary"] = end
            return
        record = fold["epochs"][fold["current"]]
        record["valid" if is_validation else "other_eval"] += end - start
        record["end"] = end
        fold["boundary"] = end


# Refresh-time work, timed by name. Containers open a refresh: inside a training
# epoch (a sample-scoped refresh) only they and what runs inside them are timed,
# so per-step uses of the same helpers (an in-batch kNN graph) stay unsynchronized.
SECTION_CONTAINERS = {
    "training.semi_supervised:build_ssl_training_dataset": "pseudo_label_refresh",
    "training.ssl.slade:SladeRegularizer._refresh_pseudo_labels": "slade_refresh",
    "regularizer.make_loader": "regularizer_refresh",
}
SECTION_LEAVES = {
    "training.ssl.embeddings:extract_embeddings": "extract_embeddings",
    "training.ssl.algorithms:faiss_flat_ip_search": "knn_search",
    "training.ssl.algorithms:faiss_flat_l2_search": "knn_search",
    "training.ssl.algorithms:make_mixed_label_affinity": "affinity_graph",
    "training.ssl.algorithms:build_lrml_knn_graph": "lrml_knn_graph",
    "training.ssl.algorithms:build_manifold_preserving_graph": "manifold_graph",
    "training.ssl.algorithms:iscen_label_spreading": "label_spreading",
    "training.ssl.algorithms:solve_sparse_label_system": "linear_solve",
    "training.ssl.algorithms:normalize_label_spreading_rows": "label_rows",
    "training.ssl.algorithms:entropy_confidence": "entropy_confidence",
    "training.ssl.slade:kmeans_cluster_labels": "kmeans",
    "training.ssl.data:graph_upper_triangle_edges": "graph_edges",
    "utils:make_train_loader": "make_train_loader",
    "utils:shutdown_dataloaders": "shutdown_dataloaders",
    "training.engine:_BestModelCheckpoint.save": "checkpoint_save",
    "training.engine:_BestModelCheckpoint.restore": "checkpoint_restore",
}


def install_section_timers(recorder: EpochRecorder) -> list[str]:
    """Wrap the refresh-time helpers wherever they are bound; returns what was not found."""

    import importlib

    def wrap(function, name: str, container: bool):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            inside_refresh = any(frame.get("container") for frame in recorder.stack)
            if recorder.fold is None or (recorder.in_train and not (container or inside_refresh)):
                return function(*args, **kwargs)
            frame = recorder.push(name)
            frame["container"] = container
            try:
                return function(*args, **kwargs)
            finally:
                recorder.pop(frame)

        timed.__wrapped_section__ = name
        return timed

    def rebind_everywhere(original, replacement) -> None:
        # Modules import helpers by name (from .ssl.embeddings import extract_embeddings),
        # so every module-level binding of the same object is replaced.
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.split(".")[0] in {"training", "utils", "main", "models", "losses"}:
                continue
            for attribute, value in list(vars(module).items()):
                if value is original:
                    setattr(module, attribute, replacement)

    missing = []
    targets = [(path, name, True) for path, name in SECTION_CONTAINERS.items()]
    targets += [(path, name, False) for path, name in SECTION_LEAVES.items()]
    for path, name, container in targets:
        if path == "regularizer.make_loader":
            interfaces = importlib.import_module("training.ssl.interfaces")
            pending, classes = [interfaces.BaseTrainingRegularizer], []
            while pending:
                cls = pending.pop()
                classes.append(cls)
                pending.extend(cls.__subclasses__())
            for cls in classes:
                if "make_loader" in vars(cls):
                    setattr(cls, "make_loader", wrap(vars(cls)["make_loader"], name, container))
            continue
        module_name, _, attribute = path.partition(":")
        try:
            owner = importlib.import_module(module_name)
            *class_path, function_name = attribute.split(".")
            for part in class_path:
                owner = getattr(owner, part)
            original = getattr(owner, function_name)
        except (ImportError, AttributeError):
            missing.append(path)
            continue
        replacement = wrap(original, name, container)
        if class_path:
            setattr(owner, function_name, replacement)
        else:
            rebind_everywhere(original, replacement)
    return missing


class FoldLimitReached(BaseException):
    """Ends a case after --max-folds folds. BaseException, so no ``except Exception`` in the engine swallows it."""


def install_timing_hooks(
    recorder: EpochRecorder, engine, utils, max_folds: int | None = None, step_sections: bool = False
) -> list[str]:
    run_training = engine.run_training

    @functools.wraps(run_training)
    def timed_run_training(*args, **kwargs):
        if max_folds is not None and len(recorder.folds) >= max_folds:
            # Raised before the next fold builds anything, so the finished folds stand as they are.
            raise FoldLimitReached
        recorder.begin_fold(kwargs.get("cv_fold"))
        try:
            return run_training(*args, **kwargs)
        finally:
            recorder.end_fold()

    train_epoch = engine._EpochTrainer.train_epoch

    @functools.wraps(train_epoch)
    def timed_train_epoch(self, train_loader, epoch, *args, **kwargs):
        steps_before = self.global_step
        recorder.train_start(epoch)
        try:
            return train_epoch(self, train_loader, epoch, *args, **kwargs)
        finally:
            recorder.train_end(epoch, self.global_step - steps_before)

    evaluate_split = utils.evaluate_split

    @functools.wraps(evaluate_split)
    def timed_evaluate_split(model, eval_loader, name="test set", *args, **kwargs):
        start = recorder.now()
        recorder.in_eval = True
        try:
            return evaluate_split(model, eval_loader, name, *args, **kwargs)
        finally:
            recorder.in_eval = False
            recorder.evaluation(str(name), start, recorder.now())

    # run_cross_validation and run_experiment look these names up at call time.
    engine.run_training = timed_run_training
    engine._EpochTrainer.train_epoch = timed_train_epoch
    utils.evaluate_split = timed_evaluate_split

    if step_sections:
        # --debug_batch_timing's own synchronized per-step timer; each epoch's totals are kept.
        class RecordedBatchTimer(engine._BatchTimer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                recorder.batch_timer = self

        engine._BatchTimer = RecordedBatchTimer
    return install_section_timers(recorder)


def recorder_rows(recorder: EpochRecorder, ssl_enabled: bool, warmup: int) -> tuple[list[dict], list[dict], list[dict]]:
    epoch_rows, fold_rows, section_rows = [], [], []
    for fold in recorder.folds:
        for part in ("setup", "teardown"):
            for name, (seconds, calls) in sorted(fold["sections"][part].items()):
                section_rows.append(
                    {"fold": fold["fold"], "epoch": -1, "phase": part, "part": part, "section": name, "seconds": seconds, "calls": calls}
                )
        epochs_seconds = 0.0
        for epoch in sorted(fold["epochs"]):
            record = fold["epochs"][epoch]
            total = record["end"] - record["boundary"]
            train = record["train_end"] - record["train_start"]
            gap = record["train_start"] - record["boundary"]
            phase = "sup" if not ssl_enabled else ("warmup" if epoch < warmup else "ssl")
            for part, bucket in record["sections"].items():
                for name, (seconds, calls) in sorted(bucket.items()):
                    section_rows.append(
                        {"fold": fold["fold"], "epoch": epoch, "phase": phase, "part": part, "section": name, "seconds": seconds, "calls": calls}
                    )
            epoch_rows.append(
                {
                    "fold": fold["fold"],
                    "epoch": epoch,
                    "phase": phase,
                    "steps": record["steps"],
                    "gap_s": gap,
                    "train_s": train,
                    "valid_s": record["valid"],
                    "other_s": total - gap - train - record["valid"],
                    "total_s": total,
                }
            )
            epochs_seconds += total
        end = fold["end"] if fold["end"] is not None else fold["boundary"]
        setup_end = fold.get("setup_end", fold["boundary"])
        last_epoch_end = max((record["end"] for record in fold["epochs"].values()), default=setup_end)
        fold_rows.append(
            {
                "fold": fold["fold"],
                "wall_s": end - fold["start"],
                "setup_s": setup_end - fold["start"],
                "baseline_valid_s": fold["baseline_valid"],
                "teardown_s": end - last_epoch_end,
                "epochs": len(fold["epochs"]),
                "epochs_s": epochs_seconds,
            }
        )
    return epoch_rows, fold_rows, section_rows


def worker_main(request_path: Path) -> int:
    request = read_json(request_path)
    case_dir = Path(request["case_dir"])
    result = {"status": "failed", "case_id": request["case_id"], "started": datetime.now().isoformat()}
    recorder = None
    try:
        os.chdir(WORKSPACE_ROOT)
        if str(WORKSPACE_ROOT) not in sys.path:
            sys.path.insert(0, str(WORKSPACE_ROOT))

        import torch

        import main
        import utils
        from training import engine

        study_dir = Path(request["study_dir"])
        overrides = request.get("overrides") or {}
        study_config, base_args, params, trial_args, ssl_config, ssl_enabled = resolve_trial(
            main, study_dir, request["params"], overrides
        )
        warmup = int(ssl_config.warmup_epochs or 0) if ssl_enabled else 0
        epochs = planned_epochs(
            ssl_enabled,
            warmup,
            ssl_config.update_mode,
            ssl_config.update_interval_epochs,
            int(request["window"]),
        )

        # Only what fixes the amount of work changes; everything else is the trial as HPO ran it.
        trial_args.epochs = epochs
        trial_args.patience = epochs + 1
        trial_args.cv_parallel_folds = 1
        trial_args.resume_interrupted_runs = False
        trial_args.save_dir = Path(request["save_dir"])
        trial_args.trial_number = request["trial"]
        trial_args.hparam_study_dir = study_dir
        trial_args.hparam_study_name = study_config.get("resolved_study_name")
        trial_args.hparam_params = params
        trial_args.hparam_config_resolved = study_config.get("hparam_config")
        trial_args.evaluate_test = not bool(getattr(base_args, "skip_test_during_hpo", False))
        trial_args.final_test_visualization = "none"
        trial_args.track_epoch_artifacts = "none"
        trial_args.track_epoch_interval = None
        main.configure_threads(trial_args)

        step_sections = bool(request.get("step_sections"))
        if step_sections:
            trial_args.debug_batch_timing = True
            # Totals are read per epoch; the periodic log lines are not needed.
            trial_args.debug_batch_timing_interval = 10**9
        recorder = EpochRecorder(torch)
        max_folds = request.get("max_folds")
        result["section_timers_missing"] = install_timing_hooks(recorder, engine, utils, max_folds, step_sections)
        result["step_sections"] = step_sections
        started = recorder.now()
        try:
            main.run_experiment(trial_args, ssl_config)
        except FoldLimitReached:
            # The trial's overhead then lacks the CV aggregation after the last fold.
            result["folds_limited_to"] = max_folds
        trial_wall = recorder.now() - started

        epoch_rows, fold_rows, section_rows = recorder_rows(recorder, ssl_enabled, warmup)
        if not epoch_rows:
            raise RuntimeError("no epoch was timed; the engine no longer calls the patched entry points")
        write_csv(case_dir / "epochs.csv", epoch_rows)
        write_csv(case_dir / "folds.csv", fold_rows)
        write_csv(
            case_dir / "sections.csv",
            section_rows,
            ["fold", "epoch", "phase", "part", "section", "seconds", "calls"],
        )
        result.update(
            status="ok",
            overrides=overrides,
            ssl_enabled=ssl_enabled,
            warmup_epochs=warmup,
            update_mode=ssl_config.update_mode,
            update_interval_epochs=ssl_config.update_interval_epochs,
            epochs=epochs,
            batch_size=getattr(trial_args, "batch_size", None),
            trial_wall_s=trial_wall,
            trial_overhead_s=trial_wall - sum(row["wall_s"] for row in fold_rows),
            num_threads=torch.get_num_threads(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        )
    except BaseException as exc:  # noqa: BLE001 - the parent needs every failure on record
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
        )
    finally:
        result["finished"] = datetime.now().isoformat()
        write_json(case_dir / "worker_result.json", result)
    return 0 if result["status"] == "ok" else 1


# --------------------------------------------------------------- summaries


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def quantile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def phase_rate(rows: list[dict]) -> dict | None:
    """Mean seconds per epoch (means amortize refreshes), split so train can be rescaled."""

    if not rows:
        return None
    total = mean(row["total_s"] for row in rows)
    train = mean(row["train_s"] for row in rows)
    return {
        "n": len(rows),
        "total_s": total,
        "train_s": train,
        "nontrain_s": total - train,
        "valid_s": mean(row["valid_s"] for row in rows),
        "gap_s": mean(row["gap_s"] for row in rows),
        "other_s": mean(row["other_s"] for row in rows),
        "steps": mean(row["steps"] for row in rows),
        "total_median_s": median(row["total_s"] for row in rows),
        "total_p90_s": quantile([row["total_s"] for row in rows], 0.9),
    }


def numeric_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key == "phase":
                item[key] = value
            elif value in ("", None):
                item[key] = None
            else:
                number = float(value)
                item[key] = int(number) if key in ("fold", "epoch", "steps", "epochs", "repeat") else number
        converted.append(item)
    return converted


def read_section_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        {
            **row,
            "fold": int(row["fold"]),
            "epoch": int(row["epoch"]),
            "seconds": float(row["seconds"]),
            "calls": int(float(row["calls"] or 0)),
        }
        for row in read_csv(path)
    ]


# Each part's time that no named section covers: the refresh gap's glue, the
# training steps themselves, and the validation's retrieval.
SECTION_REMAINDERS = (("gap", "gap_s", "other"), ("train", "train_s", "steps"), ("valid", "valid_s", "evaluation"))


def section_rates(epoch_rows: list[dict], section_rows: list[dict]) -> dict | None:
    """Mean seconds per epoch of every named section over these epochs, by part, plus each part's remainder."""

    if not epoch_rows:
        return None
    keys = {(row.get("repeat"), row["fold"], row["epoch"]) for row in epoch_rows}
    parts: dict[str, dict[str, float]] = {}
    for row in section_rows:
        if (row.get("repeat"), row["fold"], row["epoch"]) in keys:
            bucket = parts.setdefault(row["part"], {})
            bucket[row["section"]] = bucket.get(row["section"], 0.0) + row["seconds"] / len(keys)
    for part, column, remainder in SECTION_REMAINDERS:
        named = sum(parts.get(part, {}).values())
        parts.setdefault(part, {})[remainder] = max(0.0, mean(row[column] for row in epoch_rows) - named)
    return parts


def summarize_case(
    case: dict, results: list[dict], epoch_rows: list[dict], fold_rows: list[dict], section_rows: list[dict] | None = None
) -> dict:
    ssl_enabled = bool(results[0]["ssl_enabled"])
    warmup = int(results[0]["warmup_epochs"])
    warm_phase = "warmup" if ssl_enabled else "sup"
    warm_rows = [row for row in epoch_rows if row["phase"] == warm_phase]
    steady_warm = [row for row in warm_rows if row["epoch"] >= 1] or warm_rows
    first_ssl = [row for row in epoch_rows if ssl_enabled and row["epoch"] == warmup]
    later_ssl = [row for row in epoch_rows if ssl_enabled and row["epoch"] > warmup]

    for row in fold_rows:
        row["fixed_s"] = row["wall_s"] - row["epochs_s"]
    later_folds = [row for row in fold_rows if row["fold"] >= 1] or fold_rows
    fold_fixed = median(row["fixed_s"] for row in later_folds)
    first_fold_fixed = median(row["fixed_s"] for row in fold_rows if row["fold"] == 0)

    total_steps = sum(row["steps"] for row in epoch_rows)
    # Per-epoch curve, averaged over folds and repeats, for plotting.
    by_epoch: dict[int, list[dict]] = {}
    for row in epoch_rows:
        by_epoch.setdefault(row["epoch"], []).append(row)
    curve = [
        {
            "epoch": epoch,
            "phase": rows[0]["phase"],
            "gap_s": mean(row["gap_s"] for row in rows),
            "train_s": mean(row["train_s"] for row in rows),
            "valid_s": mean(row["valid_s"] for row in rows),
            "other_s": mean(row["other_s"] for row in rows),
        }
        for epoch, rows in sorted(by_epoch.items())
    ]
    return {
        **{key: case[key] for key in ("case_id", "study_key", "name", "dataset", "method", "loss", "trial", "batch_size")},
        "identity": case.get("fingerprint") or case["study_key"],
        "overrides": case.get("overrides") or {},
        "repeats": len(results),
        "ssl_enabled": ssl_enabled,
        "warmup_epochs": warmup,
        "update_mode": results[0].get("update_mode"),
        "update_interval_epochs": results[0].get("update_interval_epochs"),
        "epochs_per_fold": int(results[0]["epochs"]),
        "folds": len({(row.get("repeat"), row["fold"]) for row in fold_rows}),
        "steps_per_epoch": total_steps / len(epoch_rows) if epoch_rows else None,
        "warm": phase_rate(steady_warm),
        "ssl_first": phase_rate(first_ssl),
        "ssl": phase_rate(later_ssl),
        "fold_fixed_s": fold_fixed,
        "study_once_s": max(0.0, first_fold_fixed - fold_fixed) if None not in (first_fold_fixed, fold_fixed) else None,
        "trial_overhead_s": median(result.get("trial_overhead_s") for result in results),
        "fold_setup_s": median(row["setup_s"] for row in later_folds),
        "baseline_valid_s": median(row["baseline_valid_s"] for row in later_folds),
        "contaminated": any(result.get("contaminated") for result in results),
        "gpu_process_visibility": all(result.get("gpu_process_visibility") for result in results),
        "gpu_utilization_mean": mean(result.get("gpu_utilization_mean") for result in results),
        "load_average_max": max((result.get("load_average_max") or 0.0) for result in results),
        "measured_at": max(result.get("finished", "") for result in results),
        "curve": curve,
        "sections": (
            {
                "warm": section_rates(steady_warm, section_rows),
                "ssl_first": section_rates(first_ssl, section_rows),
                "ssl": section_rates(later_ssl, section_rows),
            }
            if section_rows
            else None
        ),
        "step_sections": any(result.get("step_sections") for result in results),
    }


def case_overrides(case: dict) -> dict:
    raw = case.get("overrides")
    return json.loads(raw) if isinstance(raw, str) and raw else dict(raw or {})


def load_bench_cases(bench_dir: Path) -> list[dict]:
    """One summary per (study, trial, override point), pooling that case's repeats."""

    cases_path = bench_dir / "cases.csv"
    if not cases_path.is_file():
        return []
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for case in read_csv(cases_path):
        case["overrides"] = case_overrides(case)
        identity = case.get("fingerprint") or case["study_key"]
        grouped.setdefault((identity, int(case["trial"]), override_key(case["overrides"])), []).append(case)
    summaries = []
    for _, cases in sorted(grouped.items()):
        results, epoch_rows, fold_rows, section_rows = [], [], [], []
        for case in cases:
            case_dir = bench_dir / "cases" / case["case_id"]
            result_path = case_dir / "result.json"
            if not result_path.is_file():
                continue
            result = read_json(result_path)
            if result.get("status") != "ok":
                continue
            repeat = int(case["repeat"])
            results.append(result)
            epoch_rows += [{**row, "repeat": repeat} for row in numeric_rows(read_csv(case_dir / "epochs.csv"))]
            fold_rows += [{**row, "repeat": repeat} for row in numeric_rows(read_csv(case_dir / "folds.csv"))]
            section_rows += [{**row, "repeat": repeat} for row in read_section_rows(case_dir / "sections.csv")]
        if results:
            first = dict(cases[0])
            first["trial"] = int(first["trial"])
            first["batch_size"] = optional_int(first.get("batch_size"))
            first["case_id"] = first["case_id"].rsplit("_r", 1)[0]
            summaries.append(summarize_case(first, results, epoch_rows, fold_rows, section_rows))
    return summaries


# ----------------------------------------------------------------- estimate


def rate_cost(rate: dict, ratio: float) -> float:
    return rate["nontrain_s"] + rate["train_s"] * ratio


def own_cases(study: StudyInfo, by_study: dict) -> list[dict]:
    # Runs from before fingerprints were recorded are matched by path.
    return by_study.get(study.fingerprint) or by_study.get(study.study_key) or []


def pick_case(study: StudyInfo, trial: TrialInfo, by_study: dict, by_group: dict) -> tuple[dict | None, str]:
    def nearest(cases):
        if trial.batch_size is None:
            return cases[0]
        return min(
            cases,
            key=lambda case: abs(math.log((case["batch_size"] or trial.batch_size) / trial.batch_size)),
        )

    own = own_cases(study, by_study)
    if own:
        case = nearest(own)
        return case, "measured" if case["batch_size"] == trial.batch_size else "scaled_batch_size"
    group = by_group.get(study.group, [])
    if group:
        case = nearest(group)
        return case, "same_combination" if case["batch_size"] == trial.batch_size else "same_combination_scaled"
    return None, "no_measurement"


def warm_rate_for(case: dict, study: StudyInfo, by_study: dict, by_group: dict) -> tuple[dict | None, str]:
    """Warm-up epochs train on labels only; borrow a rate when the case measured none."""

    if case.get("warm"):
        return case["warm"], "case"
    for other in own_cases(study, by_study):
        if other.get("warm"):
            return other["warm"], "study"
    for other in by_group.get((study.dataset, "supervised", study.loss), []):
        if other.get("warm"):
            return other["warm"], "supervised_same_loss"
    return case.get("ssl"), "ssl_rate"


def fold_cost(
    case: dict, warm: dict | None, ssl_enabled: bool, warmup_epochs: int, epochs: int, ratio: float = 1.0
) -> float | None:
    """``fixed + min(E, W) t_warm + [E > W] t_ssl_first + max(0, E - W - 1) t_ssl``; None without a needed rate."""

    cost = case.get("fold_fixed_s") or 0.0
    if ssl_enabled:
        warm_epochs, ssl_epochs = min(epochs, warmup_epochs), max(0, epochs - warmup_epochs)
    else:
        warm_epochs, ssl_epochs = epochs, 0
    if warm_epochs:
        if warm is None:
            return None
        cost += warm_epochs * rate_cost(warm, ratio)
    if ssl_epochs:
        first = case.get("ssl_first") or case.get("ssl")
        rest = case.get("ssl") or first
        if first is None:
            return None
        cost += rate_cost(first, ratio) + (ssl_epochs - 1) * rate_cost(rest, ratio)
    return cost


def estimate_trial(study: StudyInfo, trial: TrialInfo, case: dict, warm: dict | None) -> float | None:
    if not trial.folds:
        return None
    total = case.get("trial_overhead_s") or 0.0
    reference_steps = case.get("steps_per_epoch")
    for fold in trial.folds:
        ratio = 1.0
        if trial.batch_size != case["batch_size"] and reference_steps and fold.steps and fold.epochs:
            ratio = (fold.steps / fold.epochs) / reference_steps
        cost = fold_cost(case, warm, study.ssl_enabled, trial.warmup_epochs, fold.epochs, ratio)
        if cost is None:
            return None
        total += cost
    return total


def sweep_rows(cases: list[dict], studies: list[StudyInfo], fold_epochs: list[int]) -> list[dict]:
    """Every measured override point beside its trial's baseline, as rates and fold costs.

    Without ``fold_epochs`` the fold cost is taken at the mean epochs per fold
    the trial really ran in its study, so the comparison is at the trial's own length.
    """

    recorded = {}
    for study in studies:
        for trial in study.trials:
            if trial.folds:
                epochs = mean(fold.epochs for fold in trial.folds)
                recorded[(study.fingerprint or study.study_key, trial.number)] = epochs
                recorded.setdefault((study.study_key, trial.number), epochs)
    groups: dict[tuple[str, int], list[dict]] = {}
    for case in cases:
        groups.setdefault((case["identity"], case["trial"]), []).append(case)

    rows = []
    for (identity, trial), group in sorted(groups.items()):
        if not any(case["overrides"] for case in group):
            continue
        baseline = next((case for case in group if not case["overrides"]), None)
        own = recorded.get((identity, trial)) or recorded.get((group[0]["study_key"], trial))
        points = list(fold_epochs) or ([round(own)] if own else [])
        for case in sorted(group, key=lambda case: (bool(case["overrides"]), setting_label(case["overrides"]))):
            row = {
                "study_key": case["study_key"],
                "dataset": case["dataset"],
                "method": case["method"],
                "loss": case["loss"],
                "trial": trial,
                "setting": setting_label(case["overrides"]),
                "overrides": case["overrides"],
                "case_id": case["case_id"],
                "batch_size": case["batch_size"],
                "steps_per_epoch": case["steps_per_epoch"],
                "warmup_epochs": case["warmup_epochs"],
                "update_mode": case["update_mode"],
                "update_interval_epochs": case["update_interval_epochs"],
                "warm_epoch_s": (case.get("warm") or {}).get("total_s"),
                "ssl_first_epoch_s": (case.get("ssl_first") or {}).get("total_s"),
                "ssl_epoch_s": (case.get("ssl") or {}).get("total_s"),
                "fold_fixed_s": case.get("fold_fixed_s"),
                "recorded_epochs_per_fold": own,
                "contaminated": case["contaminated"],
            }
            for epochs in points:
                cost = fold_cost(case, case.get("warm"), case["ssl_enabled"], case["warmup_epochs"], epochs)
                base = (
                    fold_cost(baseline, baseline.get("warm"), baseline["ssl_enabled"], baseline["warmup_epochs"], epochs)
                    if baseline
                    else None
                )
                row[f"fold_s_at_{epochs}"] = cost
                row[f"vs_baseline_at_{epochs}"] = cost / base if cost is not None and base else None
            rows.append(row)
    return rows


def section_rate_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for phase, parts in (case.get("sections") or {}).items():
            rate = case.get(phase)
            if not parts or not rate:
                continue
            for part, bucket in parts.items():
                for section, seconds in sorted(bucket.items(), key=lambda item: -item[1]):
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "dataset": case["dataset"],
                            "method": case["method"],
                            "loss": case["loss"],
                            "trial": case["trial"],
                            "setting": setting_label(case["overrides"]),
                            "batch_size": case["batch_size"],
                            "phase": phase,
                            "part": part,
                            "section": section,
                            "seconds_per_epoch": seconds,
                            "share_of_epoch": seconds / rate["total_s"] if rate["total_s"] else None,
                            "step_sections": case.get("step_sections"),
                        }
                    )
    return rows


def print_section_breakdown(cases: list[dict]) -> None:
    """One line per case: where its steady epoch goes (SSL epochs, or warm-up/supervised ones)."""

    def parts_text(parts: dict) -> str:
        texts = []
        for part, label in (("gap", "refresh"), ("train", "train"), ("valid", "valid")):
            bucket = parts.get(part) or {}
            total = sum(bucket.values())
            top = sorted(((name, value) for name, value in bucket.items() if value >= 0.005), key=lambda item: -item[1])
            inner = ", ".join(f"{name} {value:.2f}" for name, value in top[:4])
            texts.append(f"{label} {total:.2f}" + (f" ({inner})" if inner and len(top) > 1 else ""))
        steps = parts.get("step") or {}
        if steps:
            top = sorted(steps.items(), key=lambda item: -item[1])[:5]
            texts.append("step: " + ", ".join(f"{name} {value:.2f}" for name, value in top))
        return "; ".join(texts)

    for case in sorted(cases, key=lambda case: (case["dataset"], case["method"], case["loss"], case["trial"], case["batch_size"] or 0)):
        sections = case.get("sections") or {}
        phase = "ssl" if sections.get("ssl") else "warm"
        if not sections.get(phase) or not case.get(phase):
            continue
        label = (
            f"{DATASET_LABELS.get(case['dataset'], case['dataset'])} {METHOD_LABELS.get(case['method'], case['method'])} "
            f"{LOSS_LABELS.get(case['loss'], case['loss'])} t{case['trial']} b{case['batch_size']}"
        )
        print(f"  {label:<40} {phase:<4} {case[phase]['total_s']:.2f} s/epoch = {parts_text(sections[phase])}")


def print_sweeps(rows: list[dict]) -> None:
    def fmt(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    current = None
    for row in rows:
        key = (row["study_key"], row["trial"])
        epoch_columns = [name for name in row if name.startswith("fold_s_at_")]
        if key != current:
            current = key
            label = (
                f"{DATASET_LABELS.get(row['dataset'], row['dataset'])} {METHOD_LABELS.get(row['method'], row['method'])} "
                f"{LOSS_LABELS.get(row['loss'], row['loss'])} trial {row['trial']}"
            )
            recorded = row["recorded_epochs_per_fold"]
            print(f"\n{label}" + ("" if recorded is None else f" (ran {recorded:.0f} epochs per fold in its study)"))
            heads = "".join(f" {'fold@' + name.rsplit('_', 1)[-1]:>10} {'x base':>6}" for name in epoch_columns)
            print(f"  {'setting':<52} {'batch':>5} {'steps':>6} {'warm s':>7} {'ssl s':>7} {'fixed s':>7}{heads}")
        cells = "".join(
            f" {fmt(row[name], 0):>10} {fmt(row['vs_baseline_at_' + name.rsplit('_', 1)[-1]]):>6}" for name in epoch_columns
        )
        print(
            f"  {row['setting'][:52]:<52} {row['batch_size'] or '-':>5} {fmt(row['steps_per_epoch'], 0):>6} "
            f"{fmt(row['warm_epoch_s']):>7} {fmt(row['ssl_epoch_s']):>7} {fmt(row['fold_fixed_s'], 1):>7}{cells}"
            + (" CONTAMINATED" if row["contaminated"] else "")
        )


def estimate(
    bench_dir: Path,
    output_dir: Path,
    studies: list[StudyInfo],
    allow_contaminated: bool,
    fold_epochs: list[int] | None = None,
) -> dict:
    all_cases = load_bench_cases(bench_dir)
    # A case with overrides is not the trial HPO ran, so it never prices a study.
    cases = [case for case in all_cases if not case["overrides"]]
    usable = [case for case in cases if allow_contaminated or not case["contaminated"]]
    by_study: dict[str, list[dict]] = {}
    by_group: dict[tuple, list[dict]] = {}
    for case in usable:
        by_study.setdefault(case["identity"], []).append(case)
        by_group.setdefault((case["dataset"], case["method"], case["loss"]), []).append(case)

    trial_rows, study_rows = [], []
    for study in studies:
        costs, observed, models = [], [], {}
        epochs_per_fold, ssl_epochs_per_fold, warmups = [], [], []
        once = None
        for trial in study.trials:
            case, model = pick_case(study, trial, by_study, by_group)
            warm, warm_source = warm_rate_for(case, study, by_study, by_group) if case else (None, None)
            cost = estimate_trial(study, trial, case, warm) if case else None
            if cost is None and trial.folds:
                model = "no_measurement" if case is None else "incomplete_rates"
            if not trial.folds:
                model = "no_epoch_record"
            if case and once is None:
                once = case.get("study_once_s")
            models[model] = models.get(model, 0) + 1
            if cost is not None:
                costs.append(cost)
            if trial.duration_seconds is not None:
                observed.append(trial.duration_seconds)
            for fold in trial.folds:
                epochs_per_fold.append(fold.epochs)
                ssl_epochs_per_fold.append(max(0, fold.epochs - trial.warmup_epochs) if study.ssl_enabled else None)
            if study.ssl_enabled and trial.folds:
                warmups.append(trial.warmup_epochs)
            trial_rows.append(
                {
                    "study_key": study.study_key,
                    "name": study.name,
                    "dataset": study.dataset,
                    "method": study.method,
                    "loss": study.loss,
                    "trial": trial.number,
                    "state": trial.state,
                    "batch_size": trial.batch_size,
                    "warmup_epochs": trial.warmup_epochs,
                    "folds": len(trial.folds),
                    "epochs_total": sum(fold.epochs for fold in trial.folds),
                    "estimated_s": cost,
                    "observed_s": trial.duration_seconds,
                    "case_id": case["case_id"] if case else None,
                    "time_model": model,
                    "warm_rate_source": warm_source,
                }
            )
        known = sum(costs)
        imputed = known + (len(study.trials) - len(costs)) * (median(costs) or 0.0)
        measured = sorted(own_cases(study, by_study), key=lambda case: case["trial"])
        selected = None
        best = select_trials(study, "best")
        if best:
            selected = next((case for case in measured if case["trial"] == best[0].number), None)
        study_rows.append(
            {
                "study_key": study.study_key,
                "name": study.name,
                "dataset": study.dataset,
                "dataset_label": DATASET_LABELS.get(study.dataset, study.dataset),
                "method": study.method,
                "method_label": METHOD_LABELS.get(study.method, study.method),
                "loss": study.loss,
                "loss_label": LOSS_LABELS.get(study.loss, study.loss),
                "trials": len(study.trials),
                "trials_estimated": len(costs),
                "epochs_per_fold_mean": mean(epochs_per_fold),
                "ssl_epochs_per_fold_mean": mean(ssl_epochs_per_fold),
                "warmup_epochs_mean": mean(warmups),
                "estimated_gpu_hours": (known + (once or 0.0)) / 3600 if costs else None,
                "estimated_gpu_hours_imputed": (imputed + (once or 0.0)) / 3600 if costs else None,
                "observed_hours": sum(observed) / 3600 if observed else None,
                "trial_minutes_median": (median(costs) or 0.0) / 60 if costs else None,
                "time_models": models,
                "selected_trial": selected["trial"] if selected else None,
                "selected_batch_size": selected["batch_size"] if selected else None,
                "selected_warmup_epochs": selected["warmup_epochs"] if selected else None,
                "selected_warm_epoch_s": (selected.get("warm") or {}).get("total_s") if selected else None,
                "selected_ssl_first_epoch_s": (selected.get("ssl_first") or {}).get("total_s") if selected else None,
                "selected_ssl_epoch_s": (selected.get("ssl") or {}).get("total_s") if selected else None,
                "selected_fold_fixed_s": selected.get("fold_fixed_s") if selected else None,
                "cases": [case["case_id"] for case in measured],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hpo_cost_trials.csv", trial_rows)
    write_csv(output_dir / "hpo_cost_studies.csv", study_rows)
    write_csv(
        output_dir / "bench_cases.csv",
        [{key: value for key, value in flatten_case(case).items()} for case in cases],
    )
    run_meta = read_json(bench_dir / RUN_FILENAME) if (bench_dir / RUN_FILENAME).is_file() else {}
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "bench_dir": str(bench_dir),
        "run": run_meta,
        "labels": {"datasets": DATASET_LABELS, "methods": METHOD_LABELS, "losses": LOSS_LABELS},
        "studies": study_rows,
        "cases": cases,
        "trials": trial_rows,
    }
    sweeps = sweep_rows(all_cases, studies, fold_epochs or [])
    if sweeps:
        write_csv(output_dir / "sweep_cases.csv", sweeps)
        snapshot["sweep_cases"] = [case for case in all_cases if case["overrides"]]
        snapshot["sweeps"] = sweeps
    write_json(output_dir / SNAPSHOT_FILENAME, snapshot)
    print_estimate(study_rows)
    written = f"{output_dir / 'hpo_cost_studies.csv'}, hpo_cost_trials.csv, bench_cases.csv, {SNAPSHOT_FILENAME}"
    section_rows = section_rate_rows(all_cases)
    if section_rows:
        write_csv(output_dir / "section_rates.csv", section_rows)
        print("\nWhere an epoch goes (seconds per epoch; each section without the ones nested in it)")
        print_section_breakdown(all_cases)
        written += ", section_rates.csv"
    if sweeps:
        print("\nSweeps: rates on this machine; fold@E = fixed + warm-up + SSL epochs for E epochs, x base = vs the trial as run")
        print_sweeps(sweeps)
        written += ", sweep_cases.csv"
    print(f"\nwrote {written}")
    return snapshot


def flatten_case(case: dict) -> dict:
    flat = {key: value for key, value in case.items() if key not in ("warm", "ssl_first", "ssl", "curve", "sections")}
    for phase in ("warm", "ssl_first", "ssl"):
        for key, value in (case.get(phase) or {}).items():
            flat[f"{phase}_{key}"] = value
    return flat


def print_estimate(study_rows: list[dict]) -> None:
    def fmt(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    print(
        f"\n{'dataset':<16} {'method':<7} {'loss':<13} {'warm s':>7} {'ssl s':>7} "
        f"{'ep/fold':>7} {'GPU-h':>7} {'obs h':>7}  trials"
    )
    for row in sorted(study_rows, key=lambda row: (row["dataset"], row["method"], row["loss"])):
        print(
            f"{row['dataset_label']:<16} {row['method_label']:<7} {row['loss_label']:<13} "
            f"{fmt(row['selected_warm_epoch_s']):>7} {fmt(row['selected_ssl_epoch_s']):>7} "
            f"{fmt(row['epochs_per_fold_mean'], 1):>7} {fmt(row['estimated_gpu_hours']):>7} "
            f"{fmt(row['observed_hours']):>7}  {row['trials_estimated']}/{row['trials']}"
        )


# ---------------------------------------------------------------------- run


def run_case(case: Case, args: argparse.Namespace, output_dir: Path, gpu: dict) -> dict:
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = BENCH_SAVE_ROOT / output_dir.name / case.case_id / attempt
    request = {
        "case_id": case.case_id,
        "case_dir": str(case_dir),
        "study_dir": case.study_dir,
        "trial": case.trial,
        "params": case.params,
        "window": args.window,
        "save_dir": str(save_dir),
        "overrides": case.overrides,
        "max_folds": args.max_folds,
        "step_sections": args.step_sections,
    }
    request_path = case_dir / "request.json"
    write_json(request_path, request)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.get("uuid") or str(gpu.get("index"))
    environment["PYTHONUNBUFFERED"] = "1"
    if case.num_threads is not None:
        # main.py sizes its thread pools from this before torch is imported.
        environment["METRIC_LEARNING_NUM_THREADS"] = str(case.num_threads)

    wait_until_idle(gpu, args.if_busy, args.max_load)
    started = time.monotonic()
    with (case_dir / "worker.log").open("w", encoding="utf-8") as log, GpuWatch(gpu, args.monitor_interval) as watch:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker", str(request_path)],
            cwd=WORKSPACE_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watch.worker_pid = process.pid
        try:
            process.wait(timeout=args.case_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    wall = time.monotonic() - started

    worker_path = case_dir / "worker_result.json"
    result = read_json(worker_path) if worker_path.is_file() else {"status": "failed", "error": "worker wrote no result"}
    result.update(watch.report())
    result.update(process_seconds=wall, returncode=process.returncode, gpu=gpu, case=asdict(case))
    write_json(case_dir / "result.json", result)

    if not args.keep_artifacts:
        shutil.rmtree(WORKSPACE_ROOT / "logs" / BENCH_SAVE_ROOT / output_dir.name / case.case_id, ignore_errors=True)
    return result


def case_done(case_dir: Path, rerun_contaminated: bool) -> bool:
    path = case_dir / "result.json"
    if not path.is_file():
        return False
    result = read_json(path)
    return result.get("status") == "ok" and not (rerun_contaminated and result.get("contaminated"))


def command_run(args: argparse.Namespace) -> None:
    output_dir = resolve_input_path(args.output_dir)
    studies = build_inventory(args.study_dir, args.manifest)
    points = sweep_points(args.set, args.sweep_file, baseline=not args.no_baseline)
    gpu = gpu_identity(args.gpu)
    cases = plan_cases(studies, args.select, args.window, args.repeats, points)
    cases = resolve_override_cases(cases, args.window, gpu)

    # Measured into this --output-dir before; the run skips them.
    done = {case.case_id for case in cases if case_done(output_dir / "cases" / case.case_id, args.rerun_contaminated)}
    print(
        f"{len(studies)} studies, {len(cases)} cases, select={args.select}, window={args.window}, "
        f"{len(points)} setting(s) per trial, gpu={gpu}"
    )
    for case in cases:
        print(
            f"  {case.dataset:<22} {case.method:<22} {case.loss:<20} trial {case.trial:>3} "
            f"r{case.repeat} batch {case.batch_size or '-':>4} warm-up {case.warmup_epochs:>3} "
            f"-> {case.epochs:>3} epochs x {min(case.cv_k or 1, args.max_folds or (case.cv_k or 1))} folds"
            + (f"  [{setting_label(case.overrides)}]" if case.overrides else "")
            + ("  done earlier" if case.case_id in done else "")
        )
    total = sum(
        case.epochs * min(case.cv_k or 1, args.max_folds or (case.cv_k or 1)) for case in cases if case.case_id not in done
    )
    print(
        f"total epochs to run: {total}"
        + (f" ({len(cases) - len(done)} cases; {len(done)} done earlier in {output_dir})" if done else "")
    )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / RUN_FILENAME
    settings = {"select": args.select, "window": args.window, "repeats": args.repeats}
    previous = read_json(run_path) if run_path.is_file() else {}
    if previous and previous.get("settings") != settings:
        raise SystemExit(f"{output_dir} was measured with {previous.get('settings')}; use a new --output-dir")
    previous_gpu = (previous.get("gpu") or {}).get("name")
    if previous_gpu and previous_gpu != gpu.get("name"):
        raise SystemExit(f"{output_dir} was measured on {previous_gpu}, not {gpu.get('name')}; timings would not be comparable")
    # A resumed run keeps what the earlier session recorded (start time, device, versions).
    run_meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        **previous,
        "settings": settings,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "argv": sys.argv,
    }
    run_meta.pop("finished", None)
    write_json(run_path, run_meta)
    write_csv(output_dir / "trials.csv", inventory_rows(studies))
    # Merged, not replaced: a later sweep into the same folder keeps the cases measured before it.
    listed = {row["case_id"]: row for row in read_csv(output_dir / "cases.csv")} if (output_dir / "cases.csv").is_file() else {}
    for case in cases:
        listed[case.case_id] = {key: value for key, value in asdict(case).items() if key != "params"}
    write_csv(output_dir / "cases.csv", list(listed.values()))

    for index, case in enumerate(cases, start=1):
        case_dir = output_dir / "cases" / case.case_id
        if case_done(case_dir, args.rerun_contaminated):
            print(f"[{index}/{len(cases)}] {case.case_id}: done earlier", flush=True)
            continue
        folds = min(case.cv_k or 1, args.max_folds or (case.cv_k or 1))
        setting = f" [{setting_label(case.overrides)}]" if case.overrides else ""
        print(f"[{index}/{len(cases)}] {case.case_id}{setting}: {case.epochs} epochs x {folds} fold(s)", flush=True)
        result = run_case(case, args, output_dir, gpu)
        status = result.get("status")
        note = " CONTAMINATED" if result.get("contaminated") else ""
        if not result.get("gpu_process_visibility"):
            note += " (GPU process list not visible; foreign use unverified)"
        print(f"  {status} in {result['process_seconds']:.0f}s{note}", flush=True)
        if status != "ok":
            print(f"  {result.get('error')}", flush=True)
        if result.get("device_name") and "device_name" not in run_meta:
            run_meta.update({key: result[key] for key in ("device_name", "torch", "cuda", "cudnn")})
            write_json(run_path, run_meta)

    run_meta["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_path, run_meta)
    estimate(output_dir, output_dir, studies, args.allow_contaminated, args.fold_epochs)


def command_estimate(args: argparse.Namespace) -> None:
    bench_dir = resolve_input_path(args.bench_dir)
    output_dir = resolve_input_path(args.output_dir) if args.output_dir else bench_dir
    if args.study_dir or args.manifest:
        studies = build_inventory(args.study_dir, args.manifest)
        write_csv(output_dir / "trials.csv", inventory_rows(studies))
    else:
        # The epoch counts the run recorded, so no study database is needed here.
        studies = studies_from_inventory(bench_dir / "trials.csv")
    estimate(bench_dir, output_dir, studies, args.allow_contaminated, args.fold_epochs)


# --------------------------------------------------------- synthetic graphs


GRAPH_RUN_FILENAME = "graph_run.json"
GRAPH_SNAPSHOT_FILENAME = "graph_timings.json"
GRAPH_REPORT_FILENAME = "graph_report.txt"
GRAPH_CASES_DIRNAME = "graph_cases"
# Mirrors training.ssl.algorithms.LINEAR_SOLVERS; the parent stays standard-library only.
ISCEN_SOLVERS = ("torch_block_cg", "cupy_cg", "cg", "cholmod")
SOLVER_IMPORTS = {"cupy_cg": ("cupy", "CuPy"), "cholmod": ("sksparse.cholmod", "scikit-sparse")}
LRML_GRAPHS = ("binary", "cosine", "manifold_preserving")
GRAPH_SETTING_KEYS = ("dim", "noise", "labeled_fraction", "gamma", "cg_rtol", "cg_max_iter", "seed")
GRAPH_FAILED_STATES = ("timeout", "memory", "failed")
GRAPH_WARMUP_SAMPLES = 4096
GRAPH_ROW_CHUNK = 65536
GRAPH_WATCHDOG_INTERVAL = 0.5
# Growth exponents projecting the next size from the last two measured ones.
GRAPH_TIME_EXPONENT_RANGE = (1.0, 3.0)
GRAPH_MEMORY_EXPONENT_RANGE = (1.0, 2.0)
GRAPH_SINGLE_POINT_TIME_EXPONENT = 1.2
# A timeout costs the whole --case-timeout, a memory kill only seconds (the
# watchdog sees it), and the resident growth also counts the CUDA context. So a
# projected overrun skips at once for time, but only well past the limit for memory.
GRAPH_MEMORY_SKIP_FACTOR = 1.5
# Iscen holds the seed matrix, the solution, its clamped copy and the
# probabilities as dense float64 N x C arrays at once.
ISCEN_DENSE_ARRAYS = 4
GiB = 1024 ** 3
# Per-refresh timing columns, medians over the repeats.
GRAPH_TIMING_KEYS = (
    "total_s",
    "search_s",
    "graph_s",
    "solve_s",
    "factor_s",
    "cholmod_solve_s",
    "post_s",
    "other_s",
    "iterations_max",
    "iterations_mean",
    "rhs_batch_size",
)


@dataclass
class GraphCase:
    case_id: str
    config_id: str
    method: str
    variant: str
    ssl_device: str
    samples: int
    n_neighbors: int
    # Clusters in the synthetic data; for Iscen also the width of the label system.
    classes: int
    alpha: float | None


def size_label(samples: int) -> str:
    for unit, suffix in ((1_000_000, "M"), (1000, "k")):
        if samples >= unit and samples % unit == 0:
            return f"{samples // unit}{suffix}"
    return str(samples)


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 100:
        return f"{seconds:.0f}"
    return f"{seconds:.1f}" if seconds >= 10 else f"{seconds:.2f}"


def plan_graph_cases(args: argparse.Namespace) -> list[GraphCase]:
    configs = []
    for device in args.ssl_devices:
        if "iscen" in args.methods:
            for solver in args.solvers:
                if solver == "cupy_cg" and device != "cuda":
                    # algorithms._cupy_device_id refuses a CPU SSL device.
                    continue
                for k in args.neighbors:
                    for classes in args.classes:
                        for alpha in args.alpha:
                            configs.append(("iscen", solver, device, k, classes, alpha))
        if "lrml" in args.methods:
            for graph in args.lrml_graphs:
                for k in args.neighbors:
                    # LRML never reads the labels, so only the first class count is timed.
                    configs.append(("lrml", graph, device, k, args.classes[0], None))
    cases = []
    # Smallest size first: the skip rules read a configuration's smaller sizes.
    for samples in sorted(set(args.sizes)):
        for method, variant, device, k, classes, alpha in configs:
            parts = [method, variant, device, f"k{k}", f"c{classes}"]
            if alpha is not None:
                parts.append(f"a{alpha:g}")
            config_id = "_".join(parts)
            cases.append(
                GraphCase(
                    case_id=f"{config_id}_n{samples}",
                    config_id=config_id,
                    method=method,
                    variant=variant,
                    ssl_device=device,
                    samples=samples,
                    n_neighbors=k,
                    classes=classes,
                    alpha=alpha,
                )
            )
    return cases


def mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_status_bytes(pid: int | str, key: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def tree_rss_bytes(root_pid: int) -> int:
    return sum(process_status_bytes(pid, "VmRSS") or 0 for pid in process_tree(root_pid))


def kill_tree(root_pid: int) -> None:
    for pid in process_tree(root_pid):
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def solver_unavailable(solver: str) -> str | None:
    """Why ``solver`` cannot run in the worker interpreter, or ``None``."""

    module, package = SOLVER_IMPORTS.get(solver, (None, None))
    if module is None:
        return None
    probe = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    return None if probe.returncode == 0 else f"{package} is not installed ({module} does not import)"


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def growth_exponent(points: list[tuple[int, float]], bounds: tuple[float, float], default: float) -> float:
    if len(points) < 2 or points[-2][1] <= 0 or points[-1][1] <= 0:
        return default
    (n1, v1), (n2, v2) = points[-2], points[-1]
    return clamp(math.log(v2 / v1) / math.log(n2 / n1), bounds)


def graph_skip_reason(case: GraphCase, history: list[dict], limits: dict) -> tuple[str | None, dict]:
    """Decide from the configuration's smaller sizes whether ``case`` can finish."""

    failed = [row for row in history if row["status"] in GRAPH_FAILED_STATES]
    if failed:
        return f"{failed[0]['status']} at {size_label(failed[0]['samples'])}", {}
    if case.method == "iscen":
        dense = ISCEN_DENSE_ARRAYS * case.samples * case.classes * 8
        if dense > limits["max_rss_bytes"]:
            return f"dense N x C arrays need ~{dense / GiB:.0f} GiB", {"projected_rss_gb": dense / GiB}
    measured = [row for row in history if row["status"] == "ok" and row.get("total_s")]
    if not measured:
        return None, {}
    last = measured[-1]
    ratio = case.samples / last["samples"]
    time_exponent = growth_exponent(
        [(row["samples"], row["total_s"]) for row in measured],
        GRAPH_TIME_EXPONENT_RANGE,
        GRAPH_SINGLE_POINT_TIME_EXPONENT,
    )
    projection = {"projected_s": last["total_s"] * ratio ** time_exponent}
    # Resident memory past the imports grows with the data; the imports do not.
    grown = [
        (row["samples"], row["peak_rss_gb"] - row["import_rss_gb"])
        for row in measured
        if row.get("peak_rss_gb") is not None and row.get("import_rss_gb") is not None
    ]
    if grown:
        memory_exponent = growth_exponent(grown, GRAPH_MEMORY_EXPONENT_RANGE, 1.0)
        projection["projected_rss_gb"] = last["import_rss_gb"] + max(0.0, grown[-1][1]) * ratio ** memory_exponent
    if projection["projected_s"] > limits["case_timeout"]:
        return f"projected {format_seconds(projection['projected_s'])} s > {limits['case_timeout']:.0f} s timeout", projection
    if projection.get("projected_rss_gb", 0.0) * GiB > GRAPH_MEMORY_SKIP_FACTOR * limits["max_rss_bytes"]:
        return (
            f"projected {projection['projected_rss_gb']:.1f} GiB > {limits['max_rss_bytes'] / GiB:.1f} GiB limit",
            projection,
        )
    return None, projection


def graph_row(case: GraphCase, result: dict | None, *, status=None, reason=None, projection=None) -> dict:
    row = {key: value for key, value in asdict(case).items()}
    if result is None:
        row.update(status=status, reason=reason, **(projection or {}))
        return row
    timing = result.get("timing") or {}
    graph = result.get("graph") or {}
    peak = max(result.get("peak_rss_bytes") or 0, result.get("watchdog_peak_rss_bytes") or 0)
    row.update(
        status=result.get("status"),
        reason=result.get("error"),
        repeats=len(result.get("repeats") or []),
        **{key: timing.get(key) for key in GRAPH_TIMING_KEYS},
        peak_rss_gb=peak / GiB if peak else None,
        import_rss_gb=(result.get("import_rss_bytes") or 0) / GiB or None,
        data_rss_gb=(result.get("data_rss_bytes") or 0) / GiB or None,
        torch_peak_gb=(result.get("torch_peak_bytes") or 0) / GiB or None,
        cupy_pool_gb=(result.get("cupy_pool_bytes") or 0) / GiB or None,
        gpu_memory_before_mib=result.get("gpu_memory_before_mib"),
        gpu_memory_used_max_mib=result.get("gpu_memory_used_max_mib"),
        knn_purity=graph.get("knn_purity"),
        components=graph.get("components"),
        undirected_edges=graph.get("undirected_edges"),
        mean_degree=graph.get("mean_degree"),
        generate_s=result.get("generate_s"),
        warmup_s=result.get("warmup_s"),
        process_s=result.get("process_seconds"),
        contaminated=result.get("contaminated"),
    )
    return row


def graph_report_lines(rows: list[dict], settings: dict, limits: dict, meta: dict) -> list[str]:
    sizes = sorted({row["samples"] for row in rows})
    configs: dict[str, list[dict]] = {}
    for row in rows:
        configs.setdefault(row["config_id"], []).append(row)

    def cell(row: dict | None) -> str:
        if row is None or row["status"] is None:
            return ""
        status = row["status"]
        if status == "ok":
            return format_seconds(row.get("total_s"))
        if status == "skipped":
            projected = row.get("projected_s")
            return f"~{format_seconds(projected)}" if projected and "timeout" in (row.get("reason") or "") else "skip"
        return {"timeout": "timeout", "memory": "RAM", "failed": "failed", "unavailable": "n/a"}.get(status, status)

    device = meta.get("device_name") or (meta.get("gpu") or {}).get("name") or "?"
    lines = [
        f"Synthetic graph refresh, seconds per refresh (median of repeats) on {device}",
        f"dim {settings['dim']}, noise {settings['noise']}, {settings['labeled_fraction']:.0%} labeled, "
        f"gamma {settings['gamma']}, CG rtol {settings['cg_rtol']:g} / max_iter {settings['cg_max_iter']}; "
        f"limits {limits['case_timeout']:.0f} s and {limits['max_rss_bytes'] / GiB:.1f} GiB resident per case",
        "",
    ]
    head = f"{'method':<6} {'variant':<19} {'dev':<4} {'k':>4} {'C':>5} {'alpha':>5} |"
    lines.append(head + "".join(f"{size_label(size):>9}" for size in sizes))
    lines.append("-" * (len(head) + 9 * len(sizes)))
    for config_rows in configs.values():
        first = config_rows[0]
        by_size = {row["samples"]: row for row in config_rows}
        alpha = "-" if first["alpha"] is None else f"{first['alpha']:g}"
        classes = first["classes"] if first["method"] == "iscen" else "-"
        lines.append(
            f"{first['method']:<6} {first['variant']:<19} {first['ssl_device']:<4} {first['n_neighbors']:>4} "
            f"{classes:>5} {alpha:>5} |" + "".join(f"{cell(by_size.get(size)):>9}" for size in sizes)
        )
    lines += [
        "",
        "~ = projected from the measured growth and not run; RAM = killed at the memory limit; n/a = solver not installed",
        "",
        "Breakdown at the largest size each configuration finished (seconds; GiB)",
        f"{'method':<6} {'variant':<19} {'dev':<4} {'k':>4} {'C':>5} {'N':>6} | {'total':>7} {'search':>7} "
        f"{'graph':>7} {'solve':>7} {'factor':>7} {'post':>6} {'other':>6} {'iters':>6} {'RSS':>5} {'VRAM':>5} "
        f"{'purity':>6} {'comp':>5}",
    ]
    for config_rows in configs.values():
        done = [row for row in config_rows if row["status"] == "ok"]
        if not done:
            continue
        row = done[-1]
        vram = None
        if row.get("gpu_memory_used_max_mib") is not None and row.get("gpu_memory_before_mib") is not None:
            vram = (row["gpu_memory_used_max_mib"] - row["gpu_memory_before_mib"]) / 1024
        classes = row["classes"] if row["method"] == "iscen" else "-"
        lines.append(
            f"{row['method']:<6} {row['variant']:<19} {row['ssl_device']:<4} {row['n_neighbors']:>4} {classes:>5} "
            f"{size_label(row['samples']):>6} | {format_seconds(row.get('total_s')):>7} "
            f"{format_seconds(row.get('search_s')):>7} {format_seconds(row.get('graph_s')):>7} "
            f"{format_seconds(row.get('solve_s')):>7} {format_seconds(row.get('factor_s')):>7} "
            f"{format_seconds(row.get('post_s')):>6} {format_seconds(row.get('other_s')):>6} "
            f"{'-' if row.get('iterations_max') is None else int(row['iterations_max']):>6} "
            f"{'-' if row.get('peak_rss_gb') is None else format(row['peak_rss_gb'], '.1f'):>5} "
            f"{'-' if vram is None else format(vram, '.1f'):>5} "
            f"{'-' if row.get('knn_purity') is None else format(row['knn_purity'], '.2f'):>6} "
            f"{'-' if row.get('components') is None else row['components']:>5}"
        )
    lines += [
        "",
        "search = FAISS kNN search; graph = graph assembly around it (LRML: the per-node neighbor loop and the",
        "sparse union); solve = linear solve (factor = CHOLMOD factorization inside it); post = row normalization",
        "and entropy (Iscen) or edge list and node scale (LRML); other = components, degree scaling and system",
        "assembly; iters = most CG iterations of any class (torch_block_cg only); RSS = peak resident memory of",
        "the case process; VRAM = peak device memory in use above the level before the case (all processes).",
    ]
    reasons = [row for row in rows if row["status"] not in ("ok", None) and row.get("reason")]
    if reasons:
        lines += ["", "Not measured:"]
        lines += [f"  {row['case_id']}: {row['status']} - {str(row['reason']).splitlines()[0][:160]}" for row in reasons]
    return lines


def write_graph_outputs(output_dir: Path, rows: list[dict], settings: dict, limits: dict, meta: dict) -> list[str]:
    write_csv(output_dir / "graph_cases.csv", rows)
    lines = graph_report_lines(rows, settings, limits, meta)
    (output_dir / GRAPH_REPORT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / GRAPH_SNAPSHOT_FILENAME,
        {
            "version": SNAPSHOT_VERSION,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "run": meta,
            "settings": settings,
            "limits": {"case_timeout_s": limits["case_timeout"], "max_rss_gb": limits["max_rss_bytes"] / GiB},
            "cases": rows,
        },
    )
    return lines


def run_graph_case(case: GraphCase, args: argparse.Namespace, output_dir: Path, gpu: dict, limits: dict) -> dict:
    case_dir = output_dir / GRAPH_CASES_DIRNAME / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "worker_result.json").unlink(missing_ok=True)
    request = {
        **asdict(case),
        **{key: getattr(args, key) for key in GRAPH_SETTING_KEYS},
        "case_dir": str(case_dir),
        "repeats": args.repeats,
        "repeat_budget": args.repeat_budget,
    }
    request_path = case_dir / "request.json"
    write_json(request_path, request)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.get("uuid") or str(gpu.get("index"))
    environment["PYTHONUNBUFFERED"] = "1"

    wait_until_idle(gpu, args.if_busy, args.max_load)
    _, memory_before = gpu_load(gpu.get("index", "0"))
    killed, peak_rss = None, 0
    started = time.monotonic()
    with (case_dir / "worker.log").open("w", encoding="utf-8") as log, GpuWatch(gpu, args.monitor_interval) as watch:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_graph_worker", str(request_path)],
            cwd=WORKSPACE_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watch.worker_pid = process.pid
        try:
            while process.poll() is None:
                rss = tree_rss_bytes(process.pid)
                peak_rss = max(peak_rss, rss)
                if rss > limits["max_rss_bytes"]:
                    killed = "memory"
                elif time.monotonic() - started > limits["case_timeout"]:
                    killed = "timeout"
                if killed:
                    kill_tree(process.pid)
                    break
                time.sleep(GRAPH_WATCHDOG_INTERVAL)
        finally:
            if process.poll() is None:
                kill_tree(process.pid)
            process.wait()
    wall = time.monotonic() - started

    worker_path = case_dir / "worker_result.json"
    result = read_json(worker_path) if worker_path.is_file() else {"status": "failed", "error": "worker wrote no result"}
    if killed:
        result["status"] = killed
        result["error"] = (
            f"killed after {wall:.0f} s at {peak_rss / GiB:.1f} GiB resident "
            f"(limits {limits['case_timeout']:.0f} s, {limits['max_rss_bytes'] / GiB:.1f} GiB)"
        )
    result.update(watch.report())
    result.update(
        process_seconds=wall,
        returncode=process.returncode,
        watchdog_peak_rss_bytes=peak_rss,
        gpu_memory_before_mib=memory_before,
        gpu=gpu,
        case=asdict(case),
    )
    write_json(case_dir / "result.json", result)
    return result


def command_graph(args: argparse.Namespace) -> None:
    output_dir = resolve_input_path(args.output_dir)
    cases = plan_graph_cases(args)
    available = mem_available_bytes()
    if args.max_rss_gb is not None:
        max_rss_bytes = int(args.max_rss_gb * GiB)
    elif available is not None:
        # Leave room for the rest of the machine; swapping would time the disk.
        max_rss_bytes = max(GiB, available - 4 * GiB)
    else:
        max_rss_bytes = 16 * GiB
    limits = {"case_timeout": args.case_timeout, "max_rss_bytes": max_rss_bytes}
    settings = {key: getattr(args, key) for key in GRAPH_SETTING_KEYS}
    unavailable = {
        solver: solver_unavailable(solver) for solver in {case.variant for case in cases if case.method == "iscen"}
    }

    configs = list(dict.fromkeys(case.config_id for case in cases))
    print(
        f"{len(configs)} configurations x {len(set(args.sizes))} sizes = {len(cases)} cases; "
        f"limits {args.case_timeout:.0f} s and {max_rss_bytes / GiB:.1f} GiB resident per case"
    )
    for config_id in configs:
        print(f"  {config_id}")
    for solver, reason in sorted(unavailable.items()):
        if reason:
            print(f"  note: {solver} cases are skipped: {reason}")
    if args.dry_run:
        return

    gpu = gpu_identity(args.gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / GRAPH_RUN_FILENAME
    previous = read_json(run_path) if run_path.is_file() else {}
    if previous and previous.get("settings") != settings:
        raise SystemExit(f"{output_dir} was measured with {previous.get('settings')}; use a new --output-dir")
    previous_gpu = (previous.get("gpu") or {}).get("name")
    if previous_gpu and previous_gpu != gpu.get("name"):
        raise SystemExit(f"{output_dir} was measured on {previous_gpu}, not {gpu.get('name')}; timings would not be comparable")
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        **previous,
        "settings": settings,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "mem_available_gb_at_start": None if available is None else available / GiB,
        "gpu": gpu,
        "argv": sys.argv,
    }
    meta.pop("finished", None)
    write_json(run_path, meta)

    rows, history = [], {}
    for index, case in enumerate(cases, start=1):
        prefix = f"[{index}/{len(cases)}] {case.case_id}:"
        result_path = output_dir / GRAPH_CASES_DIRNAME / case.case_id / "result.json"
        earlier = read_json(result_path) if result_path.is_file() else None
        keep = earlier is not None and (
            earlier.get("status") == "ok" or (earlier.get("status") in GRAPH_FAILED_STATES and not args.retry_failed)
        )
        reason = unavailable.get(case.variant) if case.method == "iscen" else None
        if reason:
            row = graph_row(case, None, status="unavailable", reason=reason)
        elif keep:
            row = graph_row(case, earlier)
            print(f"{prefix} {row['status']} earlier", flush=True)
        else:
            skip, projection = (None, {}) if args.no_skip else graph_skip_reason(
                case, history.get(case.config_id, []), limits
            )
            if skip:
                row = graph_row(case, None, status="skipped", reason=skip, projection=projection)
                print(f"{prefix} skipped, {skip}", flush=True)
            else:
                print(prefix, flush=True)
                result = run_graph_case(case, args, output_dir, gpu, limits)
                row = graph_row(case, result)
                if row["status"] == "ok":
                    print(
                        f"  ok: {format_seconds(row['total_s'])} s per refresh (search "
                        f"{format_seconds(row['search_s'])}, graph {format_seconds(row['graph_s'])}, solve "
                        f"{format_seconds(row['solve_s'])}), peak {row['peak_rss_gb']:.1f} GiB, "
                        f"purity {row['knn_purity']:.2f}, {row['components']} components",
                        flush=True,
                    )
                else:
                    print(f"  {row['status']}: {str(row['reason']).splitlines()[0][:300]}", flush=True)
                if result.get("device_name") and "device_name" not in meta:
                    meta.update({key: result.get(key) for key in ("device_name", "versions")})
                    write_json(run_path, meta)
        history.setdefault(case.config_id, []).append(row)
        rows.append(row)
        write_graph_outputs(output_dir, rows, settings, limits, meta)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_path, meta)
    lines = write_graph_outputs(output_dir, rows, settings, limits, meta)
    print("\n" + "\n".join(lines))
    print(f"\nwrote {output_dir / GRAPH_REPORT_FILENAME}, graph_cases.csv, {GRAPH_SNAPSHOT_FILENAME}")


# ------------------------------------------------------ synthetic graph worker


def synthetic_embeddings(np, samples: int, dim: int, classes: int, noise: float, seed: int):
    """Unit-norm Gaussian-mixture embeddings, labeled rows first.

    Every class gets one random unit center; a sample adds isotropic noise of
    expected norm ``noise`` and is renormalized, as the projection head's
    F.normalize does. Measured in 768 dimensions with 100 classes: noise 2
    leaves one graph component per class (purity 1.0), 3 joins them at ~99 %
    kNN purity, 4 gives 53-68 % in one component, 6 is nearly label-free.
    Labels are drawn i.i.d., so the first rows are as good a labeled subset as
    any, which matches the labeled-then-unlabeled order the SSL methods use.
    """

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, classes, size=samples, dtype=np.int64)
    features = np.empty((samples, dim), dtype=np.float32)
    scale = np.float32(noise / math.sqrt(dim))
    for start in range(0, samples, GRAPH_ROW_CHUNK):
        stop = min(samples, start + GRAPH_ROW_CHUNK)
        block = rng.standard_normal((stop - start, dim), dtype=np.float32)
        block *= scale
        block += centers[labels[start:stop]]
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        features[start:stop] = block
    return features, labels


def graph_statistics(np, adjacency, neighbor_indices, labels) -> dict:
    from scipy.sparse import csgraph

    same = 0
    for start in range(0, len(neighbor_indices), GRAPH_ROW_CHUNK):
        block = neighbor_indices[start:start + GRAPH_ROW_CHUNK]
        same += int(np.count_nonzero(labels[block] == labels[start:start + len(block), None]))
    return {
        "undirected_edges": int(adjacency.nnz // 2),
        "mean_degree": float(adjacency.nnz / adjacency.shape[0]),
        "knn_purity": same / max(1, neighbor_indices.size),
        "components": int(csgraph.connected_components(adjacency, directed=False, return_labels=False)),
    }


class GraphProbe:
    """Record search time and solver internals from inside the production helpers.

    The wrappers only observe: CHOLMOD's diagnostics dict holds two timings
    and nothing else, and every other wrapper passes its call through.
    """

    def __init__(self, algorithms):
        self.search_s = 0.0
        self.solver: dict = {}
        probe = self

        search = algorithms.faiss_flat_ip_search

        @functools.wraps(search)
        def timed_search(*args, **kwargs):
            started = time.perf_counter()
            try:
                return search(*args, **kwargs)
            finally:
                probe.search_s += time.perf_counter() - started

        torch_cg = algorithms._solve_sparse_label_system_torch_block_cg

        @functools.wraps(torch_cg)
        def recorded_torch_cg(*args, **kwargs):
            output = torch_cg(*args, **kwargs)
            probe.solver.update(kwargs.get("metadata") or {})
            probe.solver["iterations"] = output[4]
            return output

        cupy_cg = algorithms._solve_sparse_label_system_cupy_cg

        @functools.wraps(cupy_cg)
        def recorded_cupy_cg(*args, **kwargs):
            output = cupy_cg(*args, **kwargs)
            probe.solver.update(kwargs.get("metadata") or {})
            return output

        cholmod = algorithms.solve_sparse_label_system_cholmod

        @functools.wraps(cholmod)
        def recorded_cholmod(*args, diagnostics=None, **kwargs):
            diagnostics = {} if diagnostics is None else diagnostics
            output = cholmod(*args, diagnostics=diagnostics, **kwargs)
            probe.solver["factor_s"] = diagnostics.get("factorization_seconds")
            probe.solver["cholmod_solve_s"] = diagnostics.get("solve_seconds")
            return output

        # All four are looked up as module globals at call time.
        algorithms.faiss_flat_ip_search = timed_search
        algorithms._solve_sparse_label_system_torch_block_cg = recorded_torch_cg
        algorithms._solve_sparse_label_system_cupy_cg = recorded_cupy_cg
        algorithms.solve_sparse_label_system_cholmod = recorded_cholmod

    def reset(self) -> None:
        self.search_s = 0.0
        self.solver = {}


def time_iscen_refresh(np, algorithms, semi_supervised, probe, request, features, targets, labels, collect_stats):
    """One ``iscen_label_spreading`` call as the pseudo-labeler makes it, timed by phase."""

    phases = {"affinity_s": 0.0, "solve_s": 0.0, "normalize_s": 0.0, "entropy_s": 0.0}
    stats, excluded = {}, [0.0]

    def affinity(features, n_neighbors, gamma, return_diagnostics=False, **kwargs):
        started = time.perf_counter()
        # return_diagnostics only returns arrays the builder computes anyway.
        graph, construction = algorithms.make_mixed_label_affinity(
            features, n_neighbors=n_neighbors, gamma=gamma, return_diagnostics=True, **kwargs
        )
        phases["affinity_s"] += time.perf_counter() - started
        if collect_stats:
            started = time.perf_counter()
            stats.update(graph_statistics(np, graph, construction["neighbor_indices"], labels))
            excluded[0] += time.perf_counter() - started
        return (graph, construction) if return_diagnostics else graph

    def timed(name, function, **fixed):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **{**fixed, **kwargs})
            finally:
                phases[name] += time.perf_counter() - started

        return wrapper

    # Production passes the façade's globals; iscen_label_spreading skips the
    # entropy input validation only when it receives the bare helper, which the
    # façade's is. The wrapper would hide that identity, so pass it explicitly.
    entropy_fixed = (
        {"validate": False} if semi_supervised.entropy_confidence is algorithms.entropy_confidence else {}
    )
    dependencies = {
        "make_mixed_label_affinity": affinity,
        "solve_sparse_label_system": timed("solve_s", algorithms.solve_sparse_label_system),
        "normalize_label_spreading_rows": timed("normalize_s", algorithms.normalize_label_spreading_rows),
        "entropy_confidence": timed("entropy_s", algorithms.entropy_confidence, **entropy_fixed),
    }
    probe.reset()
    started = time.perf_counter()
    algorithms.iscen_label_spreading(
        features=features,
        targets=targets,
        num_classes=int(request["classes"]),
        n_neighbors=int(request["n_neighbors"]),
        gamma=float(request["gamma"]),
        alpha=float(request["alpha"]),
        cg_rtol=float(request["cg_rtol"]),
        cg_max_iter=int(request["cg_max_iter"]),
        linear_solver=request["variant"],
        _dependencies=dependencies,
    )
    total = time.perf_counter() - started - excluded[0]
    record = {
        "total_s": total,
        "search_s": probe.search_s,
        # Row normalization copy and neighbor/sparse assembly around the search.
        "graph_s": phases["affinity_s"] - probe.search_s,
        "solve_s": phases["solve_s"],
        "post_s": phases["normalize_s"] + phases["entropy_s"],
        "other_s": total - sum(phases.values()),
        "factor_s": probe.solver.get("factor_s"),
        "cholmod_solve_s": probe.solver.get("cholmod_solve_s"),
        "rhs_batch_size": probe.solver.get("right_hand_side_batch_size"),
    }
    iterations = probe.solver.get("iterations")
    if iterations is not None:
        live = iterations[iterations > 0]
        record["iterations_max"] = int(live.max()) if live.size else 0
        record["iterations_mean"] = float(live.mean()) if live.size else 0.0
    return record, stats


def time_lrml_refresh(np, torch, semi_supervised, probe, request, features, labels, collect_stats):
    """The global LRML graph refresh after embedding extraction, timed by phase."""

    from training.ssl.data import graph_upper_triangle_edges

    params = {"n_neighbors": int(request["n_neighbors"])}
    if request["variant"] == "manifold_preserving":
        params["mode"] = "manifold_preserving"
    else:
        params.update(edge_weighting=request["variant"], gamma=float(request["gamma"]))
    regularizer = semi_supervised.LRMLRegularizer(**params)
    # The node scale lives on the training device, which stays CUDA under a CPU SSL device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    probe.reset()
    started = time.perf_counter()
    neighbor_indices, adjacency, degrees = semi_supervised._build_lrml_graph(regularizer, features)
    built = time.perf_counter()
    edge_rows, _, edge_weights = graph_upper_triangle_edges(adjacency)
    listed = time.perf_counter()
    scale = semi_supervised._lrml_node_scale(regularizer, degrees)
    torch.as_tensor(scale, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    finished = time.perf_counter()
    record = {
        "total_s": finished - started,
        "search_s": probe.search_s,
        "graph_s": built - started - probe.search_s,
        "post_s": finished - built,
        "other_s": 0.0,
    }
    stats = graph_statistics(np, adjacency, neighbor_indices, labels) if collect_stats else {}
    return record, stats


def graph_worker_main(request_path: Path) -> int:
    request = read_json(request_path)
    case_dir = Path(request["case_dir"])
    result = {"status": "failed", "case_id": request["case_id"], "started": datetime.now().isoformat()}
    try:
        os.chdir(WORKSPACE_ROOT)
        if str(WORKSPACE_ROOT) not in sys.path:
            sys.path.insert(0, str(WORKSPACE_ROOT))

        import numpy as np
        import scipy
        import torch

        from training import semi_supervised
        from training.ssl import algorithms
        from training.ssl.config import UNLABELED_TARGET
        from training.ssl.embeddings import ssl_compute_device

        result["import_rss_bytes"] = process_status_bytes("self", "VmRSS")
        started = time.perf_counter()
        features, labels = synthetic_embeddings(
            np, int(request["samples"]), int(request["dim"]), int(request["classes"]), float(request["noise"]),
            int(request["seed"]),
        )
        targets = labels.copy()
        targets[int(round(float(request["labeled_fraction"]) * len(labels))):] = UNLABELED_TARGET
        result["generate_s"] = time.perf_counter() - started

        probe = GraphProbe(algorithms)

        def refresh(rows, collect_stats=False):
            if request["method"] == "iscen":
                return time_iscen_refresh(
                    np, algorithms, semi_supervised, probe, request,
                    features[rows], targets[rows], labels[rows], collect_stats,
                )
            return time_lrml_refresh(np, torch, semi_supervised, probe, request, features[rows], labels[rows], collect_stats)

        everything = slice(None)
        with ssl_compute_device(request["ssl_device"]):
            # CUDA context, cuBLAS/cuSPARSE handles, CuPy kernel compilation and
            # the first FAISS GPU resources are paid once per training process,
            # not per refresh, so a small refresh goes first and is not counted.
            warm_rows = np.sort(
                np.random.default_rng(int(request["seed"]) + 1).choice(
                    len(labels), size=min(len(labels), GRAPH_WARMUP_SAMPLES), replace=False
                )
            )
            started = time.perf_counter()
            refresh(warm_rows)
            result["warmup_s"] = time.perf_counter() - started
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result["data_rss_bytes"] = process_status_bytes("self", "VmRSS")
            try:
                # "5" resets VmHWM, so the peak below belongs to the timed refreshes.
                Path("/proc/self/clear_refs").write_text("5")
                result["peak_rss_reset"] = True
            except OSError:
                result["peak_rss_reset"] = False

            repeats, graph = [], {}
            for repeat in range(max(1, int(request["repeats"]))):
                record, stats = refresh(everything, collect_stats=repeat == 0)
                graph = graph or stats
                repeats.append(record)
                print(f"repeat {repeat}: {json.dumps(record, default=str)}", flush=True)
                if record["total_s"] > float(request["repeat_budget"]):
                    break

        versions = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}
        for module_name in ("faiss", "cupy", "sksparse"):
            module = sys.modules.get(module_name)
            if module is not None:
                versions[module_name] = getattr(module, "__version__", "?")
        result.update(
            status="ok",
            repeats=repeats,
            timing={
                key: median(record.get(key) for record in repeats)
                for key in GRAPH_TIMING_KEYS
                if any(record.get(key) is not None for record in repeats)
            },
            graph=graph,
            peak_rss_bytes=process_status_bytes("self", "VmHWM"),
            torch_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            cupy_pool_bytes=(
                sys.modules["cupy"].get_default_memory_pool().total_bytes() if "cupy" in sys.modules else None
            ),
            torch_threads=torch.get_num_threads(),
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            versions=versions,
        )
    except BaseException as exc:  # noqa: BLE001 - the parent needs every failure on record
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
        )
    finally:
        result["finished"] = datetime.now().isoformat()
        write_json(case_dir / "worker_result.json", result)
    return 0 if result["status"] == "ok" else 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_study_inputs(sub, required: bool):
        sub.add_argument("--manifest", nargs="+", default=[], help="execution manifests whose runs name the studies")
        sub.add_argument(
            "--study-dir",
            nargs="+",
            default=[],
            help="study directories, globs, or folders searched for optuna_study.db",
        )
        sub.set_defaults(require_studies=required)

    run = commands.add_parser("run", help="benchmark on this machine, then estimate")
    add_study_inputs(run, True)
    run.add_argument("--gpu", default="0", help="GPU index or UUID; the only GPU the workers see")
    run.add_argument(
        "--select",
        default="best",
        help="trials per study: 'best', 'per-batch-size' (best trial of every batch size), or 'trials:3,17'",
    )
    run.add_argument("--window", type=int, default=20, help="SSL (or supervised) epochs measured past warm-up")
    run.add_argument("--repeats", type=int, default=1, help="fresh processes per case")
    run.add_argument(
        "--output-dir",
        default=f"timings/epoch_cost_{datetime.now():%Y-%m-%d_%H-%M}",
        help="results; an existing directory is resumed",
    )
    run.add_argument("--if-busy", choices=("wait", "abort", "run"), default="wait", help="when another process uses the GPU")
    run.add_argument("--max-load", type=float, default=0.5, help="load average per CPU above which a note is printed")
    run.add_argument("--monitor-interval", type=float, default=2.0, help="seconds between GPU polls during a case")
    run.add_argument("--case-timeout", type=float, default=None, help="seconds before a case is killed")
    run.add_argument("--rerun-contaminated", action="store_true", help="measure contaminated cases again on resume")
    run.add_argument("--allow-contaminated", action="store_true", help="let contaminated cases into the estimate")
    run.add_argument("--keep-artifacts", action="store_true", help="keep the training logs below logs/_bench_epoch_cost")
    run.add_argument("--dry-run", action="store_true", help="print the plan only")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="override a trial parameter, in the HPO spaces' names (batch_sampler, ssl_config.update_interval_epochs, "
        "ssl_config.method_params.regularizer_params.reference_sets, ...); repeat for a grid over several keys",
    )
    run.add_argument(
        "--sweep-file",
        default=None,
        help="JSON: a list of override objects, or an object of value lists (a grid); crossed with --set",
    )
    run.add_argument("--no-baseline", action="store_true", help="with --set, skip the trial as HPO ran it")
    run.add_argument(
        "--step-sections",
        action="store_true",
        help="also split every training step (data wait, forward, loss, regularizer, calibration, backward, "
        "optimizer) with --debug_batch_timing; it synchronizes the GPU around each part, so the step "
        "itself gets slower, more so at small batches; the refresh sections are always recorded",
    )
    run.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="stop each case after this many CV folds; rates are per epoch, so one fold is usually enough",
    )

    est = commands.add_parser("estimate", help="combine measured rates with the studies' epoch counts")
    est.add_argument("bench_dir", help="output directory of a 'run'")
    add_study_inputs(est, False)
    est.add_argument("--output-dir", default=None, help="defaults to bench_dir")
    est.add_argument("--allow-contaminated", action="store_true")
    for sub in (run, est):
        sub.add_argument(
            "--fold-epochs",
            type=int,
            nargs="+",
            default=[],
            help="sweep table: fold cost at these epoch counts (default: the epochs the trial ran per fold)",
        )

    graph = commands.add_parser("graph", help="time Iscen/LRML graph construction and solves on synthetic embeddings")
    graph.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 30_000, 100_000, 300_000, 1_000_000], help="samples per graph"
    )
    graph.add_argument("--methods", nargs="+", choices=("iscen", "lrml"), default=["iscen", "lrml"])
    graph.add_argument("--solvers", nargs="+", choices=ISCEN_SOLVERS, default=list(ISCEN_SOLVERS), help="Iscen linear_solver")
    graph.add_argument("--lrml-graphs", nargs="+", choices=LRML_GRAPHS, default=["binary", "cosine"], help="LRML edge weighting")
    graph.add_argument(
        "--ssl-devices",
        nargs="+",
        choices=("cuda", "cpu"),
        default=["cuda"],
        help="--ssl_device of the run: where the kNN search and the Torch/CuPy solves go",
    )
    graph.add_argument("--neighbors", type=int, nargs="+", default=[10, 100], help="n_neighbors (the studies searched 2-100)")
    graph.add_argument("--classes", type=int, nargs="+", default=[100], help="classes; Iscen solves one system column each")
    graph.add_argument("--alpha", type=float, nargs="+", default=[0.99], help="Iscen alpha; CG iterations grow with it")
    graph.add_argument("--gamma", type=float, default=3.0, help="affinity exponent (Iscen, LRML cosine)")
    graph.add_argument("--cg-rtol", type=float, default=1e-5, help="as in configs/ssl_methods/iscen.json")
    graph.add_argument("--cg-max-iter", type=int, default=2000, help="as in configs/ssl_methods/iscen.json")
    graph.add_argument("--dim", type=int, default=768, help="embedding dimension (the studies' feat_dim)")
    graph.add_argument("--noise", type=float, default=4.0, help="noise norm per sample relative to its unit class center")
    graph.add_argument("--labeled-fraction", type=float, default=0.1)
    graph.add_argument("--seed", type=int, default=0)
    graph.add_argument("--repeats", type=int, default=3, help="refreshes timed per case (median reported)")
    graph.add_argument("--repeat-budget", type=float, default=20.0, help="no further repeat once one refresh took longer (s)")
    graph.add_argument("--case-timeout", type=float, default=1800.0, help="seconds before a case process is killed")
    graph.add_argument(
        "--max-rss-gb", type=float, default=None, help="resident memory at which a case is killed; default available - 4 GiB"
    )
    graph.add_argument("--gpu", default="0", help="GPU index or UUID; the only GPU the workers see")
    graph.add_argument("--if-busy", choices=("wait", "abort", "run"), default="wait")
    graph.add_argument("--max-load", type=float, default=0.5)
    graph.add_argument("--monitor-interval", type=float, default=2.0)
    graph.add_argument(
        "--output-dir",
        default=f"timings/graph_synthetic_{datetime.now():%Y-%m-%d_%H-%M}",
        help="results; an existing directory is resumed",
    )
    graph.add_argument("--retry-failed", action="store_true", help="run timed-out, killed and failed cases again")
    graph.add_argument("--no-skip", action="store_true", help="run every size even when a projection says it cannot finish")
    graph.add_argument("--dry-run", action="store_true", help="print the plan only")

    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("request")
    graph_worker = commands.add_parser("_graph_worker", help=argparse.SUPPRESS)
    graph_worker.add_argument("request")
    resolver = commands.add_parser("_resolve", help=argparse.SUPPRESS)
    resolver.add_argument("request")

    args = parser.parse_args(argv)
    if args.command == "run":
        if not (args.manifest or args.study_dir):
            parser.error("run needs --manifest or --study-dir")
        if args.window < 1 or args.repeats < 1:
            parser.error("--window and --repeats must be positive")
    if args.command == "graph":
        if min(args.sizes) <= max(args.neighbors) or min(args.neighbors) < 1:
            parser.error("--sizes must exceed --neighbors, and --neighbors must be positive")
        if args.repeats < 1 or min(args.classes) < 2 or not 0 < args.labeled_fraction <= 1:
            parser.error("--repeats >= 1, --classes >= 2 and 0 < --labeled-fraction <= 1 are required")
        if any(not 0 < alpha < 1 for alpha in args.alpha):
            parser.error("--alpha must lie in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "_worker":
        raise SystemExit(worker_main(Path(args.request)))
    if args.command == "_graph_worker":
        raise SystemExit(graph_worker_main(Path(args.request)))
    if args.command == "_resolve":
        raise SystemExit(resolve_main(Path(args.request)))
    if args.command == "run":
        command_run(args)
    elif args.command == "graph":
        command_graph(args)
    else:
        command_estimate(args)


if __name__ == "__main__":
    main()
