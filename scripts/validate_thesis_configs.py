"""Validate recovered thesis configs and manifests without starting training."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.config_inheritance import load_config


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="also use the experiment, SSL, HPO and scheduler loaders")
    parser.add_argument("--verify-sources", action="store_true", help="verify original recorded configs when their source files are available")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    catalog = ROOT / "configs/used/catalog"
    from training.hpo import resolve_hparam_config_inheritance

    failures = []
    counts = {}
    entries = list(rows(catalog / "configs.jsonl"))
    for entry in entries:
        try:
            resolver = resolve_hparam_config_inheritance if entry["kind"] == "hpo" else load_config
            actual = resolver(ROOT / entry["path"])
            if fingerprint(actual) != entry["resolved_sha256"]:
                raise ValueError("resolved content differs from the recovered source/projection")
            counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
        except Exception as exc:
            failures.append({"path": entry["path"], "error": str(exc)})
    run_entries = list(rows(catalog / "runs.jsonl"))
    study_entries = list(rows(catalog / "studies.jsonl"))
    summary = load_config(catalog / "summary.json")
    memberships = dict(Counter(group for run in run_entries for group in run["groups"]))
    if len(run_entries) != summary["unique_runs"] or memberships != summary["run_memberships"]:
        failures.append({"error": "run coverage differs from the recovered thesis inventory"})
    trial_count = 0
    for study in study_entries:
        try:
            trials = list(rows(ROOT / study["trials"]))
            compressed = (ROOT / study["trial_inputs"]).read_bytes()
            if hashlib.sha256(compressed).hexdigest() != study["trial_inputs_compressed_sha256"]:
                raise ValueError("compressed trial-input file hash changed")
            contents = gzip.decompress(compressed)
            if hashlib.sha256(contents).hexdigest() != study["trial_inputs_sha256"]:
                raise ValueError("recorded trial-input snapshot hash changed")
            inputs = [json.loads(line) for line in contents.decode().splitlines()]
            if len(trials) != study["trial_count"] or len(inputs) != len(trials):
                raise ValueError("archived trial count changed")
            for trial, snapshot in zip(trials, inputs):
                if any(snapshot.get(key) != value for key, value in trial.items()):
                    raise ValueError("trial parameters/state differ between archive views")
            trial_count += len(trials)
            if args.verify_sources:
                if hashlib.sha256(Path(study["source"]).read_bytes()).hexdigest() != study["source_sha256"]:
                    raise ValueError("original study source hash changed")
        except Exception as exc:
            failures.append({"path": study["trials"], "error": str(exc)})
    if len(study_entries) != summary["hpo_studies"] or trial_count != summary["hpo_trial_records"]:
        failures.append({"error": "study/trial coverage differs from the recovered thesis inventory"})
    for archived in rows(ROOT / "configs/archive_catalog.jsonl"):
        if hashlib.sha256((ROOT / archived["path"]).read_bytes()).hexdigest() != archived["archived_sha256"]:
            failures.append({"path": archived["path"], "error": "archived source-file content changed"})
    if len({r["cv_dir"] for r in run_entries}) != len(run_entries):
        failures.append({"error": "duplicate CV runs in catalog"})
    run_ids = {run["id"]: run for run in run_entries}
    links = list(rows(catalog / "reference_links.jsonl"))
    if len(links) != 25:
        failures.append({"error": "k=4 comparison must link its 25 reused supervised baselines"})
    for link in links:
        target = run_ids.get(link["target_run_id"])
        if target is None or any(target[key] != link[key] for key in ("cv_dir", "experiment", "recorded")):
            failures.append({"error": "invalid reused-baseline link", "link": link})
    for run in run_entries:
        try:
            recorded = load_config(ROOT / run["recorded"])
            experiment = load_config(ROOT / run["experiment"])
            ssl = load_config(ROOT / run["ssl_config"])
            if ssl != recorded["ssl_config"]:
                raise ValueError("SSL payload differs from training-time settings")
            if experiment["ssl_config"] != run["ssl_config"]:
                raise ValueError("experiment points to a different SSL config")
            original = recorded["args"]
            if len(run["fold_sources"]) != original["cv_k"]:
                raise ValueError("selected fold-source coverage is incomplete")
            expected_length = original.get("configured_length_before_new_iter", original.get("length_before_new_iter_override"))
            if experiment["length_before_new_iter"] != expected_length:
                raise ValueError("fold-resolved epoch length was frozen into the replay")
            for key in ("epochs", "patience", "seed", "support_seed", "data_split_seed", "lr", "classifier_lr", "batch_size", "loss", "loss_params", "miner", "miner_params", "cv_k", "cv_mode"):
                if key in original and experiment.get(key) != original[key]:
                    raise ValueError(f"training parameter changed: {key}")
            if args.verify_sources:
                source = Path(run["source"])
                if hashlib.sha256(source.read_bytes()).hexdigest() != run["source_sha256"]:
                    raise ValueError("original run source hash changed")
        except Exception as exc:
            failures.append({"path": run["experiment"], "error": str(exc)})
    manifest_entries = list(rows(catalog / "run_manifests.jsonl"))
    known_experiments = {r["experiment"] for r in run_entries} | {s["experiment"] for s in study_entries}
    manifest_references = set()
    for entry in manifest_entries:
        try:
            manifest = load_config(ROOT / entry["path"])
            names = [r["name"] for r in manifest["runs"]]
            if len(names) != len(set(names)):
                raise ValueError("duplicate enabled run names")
            for run in manifest["runs"]:
                ref = run["experiment_config"]
                if ref not in known_experiments or not (ROOT / ref).is_file():
                    raise ValueError(f"unknown/missing experiment: {ref}")
                manifest_references.add(ref)
        except Exception as exc:
            failures.append({"path": entry["path"], "error": str(exc)})
    if known_experiments != manifest_references:
        failures.append({"error": "run manifests do not cover every recovered run/study"})
    if args.native:
        from training.cli import load_experiment_config
        from training.hpo import load_hparam_config
        from training.semi_supervised import load_ssl_config
        from training.run_scheduler import load_scheduler_config, validate_run_files_and_hpo
        loaders = {"experiment": load_experiment_config, "ssl": load_ssl_config, "hpo": load_hparam_config}
        for entry in entries:
            if entry["kind"] not in loaders:
                continue
            try:
                loaders[entry["kind"]](ROOT / entry["path"])
            except Exception as exc:
                failures.append({"path": entry["path"], "error": f"native loader: {exc}"})
        for entry in manifest_entries:
            try:
                config = load_scheduler_config(ROOT / entry["path"])
                validate_run_files_and_hpo(config)
            except Exception as exc:
                failures.append({"path": entry["path"], "error": f"native scheduler: {exc}"})
    report = {"counts": counts, "runs": len(run_entries), "studies": len(study_entries), "trial_records": trial_count, "run_manifests": len(manifest_entries), "reference_links": len(links), "native_checked": args.native, "source_hashes_checked": args.verify_sources, "failures": failures}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "failures": failures[:20]}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
