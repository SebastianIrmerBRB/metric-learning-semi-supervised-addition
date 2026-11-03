"""Feed the Trial Atlas from the compact thesis data instead of local Optuna databases and run folders.

    python react_trial_visualizer/scripts/export_thesis_bundle.py --daten thesis_generator/daten

Reads daten/details.zip (studies.json, tables/hpo/studies.csv, tables/hpo/trials.jsonl) and daten/results.csv and writes

    react_trial_visualizer/public/optuna-studies.json   Hyperparameters workspace: the 235 HPO studies of the thesis
                                                        (110 random Vorexperimente, 125 Phase-1 TPE searches) with
                                                        fANOVA importances, computed like scripts/optuna_param_importance.py
                                                        (seed 7, 64 trees, depth 64, normalized)
    trials-files/scheduled/*.csv                        All-studies and study pages: one Optuna-style trials table per
                                                        study, with fold values, selected epochs and random/TPE marker
    trials-files/test-results.csv                       Compare runs and test results: every Phase-1 run, Mean and Concat

Without optuna installed, or with --skip-importance, importances are 0 and the parameters keep their search-space order.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
DATASET = {"cars": "Cars196", "cub": "CUB", "inshop": "In-Shop", "inshop10": "In-Shop (labeled classes)",
           "semiinat": "Semi-iNat-2021"}
METHOD = {"sup": "ÜBl. (supervised)", "iscen": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "MS", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
PHASE = {0: "Vorexperimente (Phase 0)", 1: "Phase 1"}
FILE_DATASET = {"cars": "Cars196", "cub": "CUB", "inshop": "InShop", "inshop10": "InShop", "semiinat": "SemiiNat2021"}
FILE_METHOD = {"sup": "ÜBl", "iscen": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
FILE_LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "MS", "nt": "NTXent", "prx": "ProxyAnchor"}
TARGET = "best_valid_mean_average_precision_at_r"


def label_setting(base):
    if base.get("k_shot_grid"):
        return f"{base['k_shot_grid'][0]}-shot"
    budget = (base.get("label_budget_grid") or [None])[0]
    return f"{budget * 100:g} % labeled classes" if budget is not None else "all labels"


def parameter(name, space, values):
    """Search-space descriptor in the Atlas schema, widened to every value the trials actually used."""
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    kind = (space or {}).get("type")
    if kind in ("float", "int") or (kind is None and numeric and len(numeric) == len(values) and len(set(values)) > 10):
        low = min([space["low"]] + numeric) if space else min(numeric)
        high = max([space["high"]] + numeric) if space else max(numeric)
        p = dict(name=name, kind="numeric", distribution=kind or "float", log=bool((space or {}).get("log")),
                 low=low, high=high)
        if space and space.get("step") is not None:
            p["step"] = space["step"]
        return p
    choices = list((space or {}).get("choices") or [])
    for v in values:
        if v not in choices:
            choices.append(v)
    return dict(name=name, kind="categorical", distribution="categorical", log=False, choices=choices)


def importances(params, trials, enabled):
    """fANOVA as in scripts/optuna_param_importance.py; {} when optuna is missing or the study cannot be evaluated."""
    if not enabled:
        return {}
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    except ImportError:
        return {}
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    dists = {}
    for p in params:
        if p["kind"] == "categorical":
            dists[p["name"]] = CategoricalDistribution(p["choices"])
        elif p["distribution"] == "int":
            dists[p["name"]] = IntDistribution(int(p["low"]), int(p["high"]), log=p["log"], step=1 if p["log"] else int(p.get("step") or 1))
        else:
            dists[p["name"]] = FloatDistribution(p["low"], p["high"], log=p["log"])
    study = optuna.create_study(direction="maximize")
    for t in trials:
        ps = {k: v for k, v in t["params"].items() if k in dists}
        study.add_trial(optuna.trial.create_trial(params=ps, distributions={k: dists[k] for k in ps}, value=t["objective"]))
    try:
        ev = optuna.importance.FanovaImportanceEvaluator(n_trees=64, max_depth=64, seed=7)
        return dict(optuna.importance.get_param_importances(study, evaluator=ev, normalize=True))
    except Exception:  # e.g. too few trials or only constant parameters
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--daten", type=Path, default=REPO / "thesis_generator" / "daten")
    ap.add_argument("--skip-importance", action="store_true")
    a = ap.parse_args()
    z = zipfile.ZipFile(a.daten / "details.zip")
    configs = {s["source_id"]: s for s in json.loads(z.read("studies.json"))}
    index = list(csv.DictReader(io.StringIO(z.read("tables/hpo/studies.csv").decode("utf-8"))))
    trials_by_study = {}
    for line in z.read("tables/hpo/trials.jsonl").decode("utf-8").splitlines():
        if line:
            t = json.loads(line)
            trials_by_study.setdefault(t["study_id"], []).append(t)

    out_csv = REPO / "trials-files" / "scheduled"
    out_csv.mkdir(parents=True, exist_ok=True)
    studies, n_trials = [], 0
    for row in sorted(index, key=lambda r: (int(r["phase"]), r["dataset"], r["method"], r["loss"])):
        cfg = configs[row["config_source_id"]]
        hp, base = cfg["hparam_config"], cfg["base_args"]
        phase = int(cfg["phase"])
        raw = sorted(trials_by_study.get(row["study_id"], []), key=lambda t: t["number"])
        done = [t for t in raw if t["state"] == "COMPLETE" and t["value"] is not None and math.isfinite(t["value"])]
        trials = [dict(number=t["number"], objective=t["value"], params=t["params"]) for t in done]
        names = list(hp.get("spaces", {}))
        names += sorted({k for t in raw for k in t["params"]} - set(names))
        params = [parameter(n, hp.get("spaces", {}).get(n), [t["params"][n] for t in raw if n in t["params"]])
                  for n in names if any(n in t["params"] for t in raw)]
        imp = importances(params, trials, not a.skip_importance)
        order = sorted(params, key=lambda p: -imp.get(p["name"], 0)) if imp else params
        for rank, p in enumerate(order, 1):
            p.update(rank=rank, importance=float(imp.get(p["name"], 0.0)))
        dataset, method, loss = DATASET[cfg["dataset_key"]], METHOD[cfg["method"]], LOSS[cfg["loss"]]
        setting = label_setting(base)
        best = max(trials, key=lambda t: t["objective"]) if trials else None
        sampler = hp.get("sampler", "")
        if sampler == "tpe" and (hp.get("sampler_params") or {}).get("multivariate"):
            sampler = f"tpe (multivariate, {hp.get('tpe_startup_trials')} random start trials)"
        slug = f"phase{phase}/{cfg['dataset_key']}/{cfg['method']}/{cfg['loss']}"
        studies.append(dict(
            id=f"thesis/{slug}", relativePath=f"thesis/{slug}", studyName=row["study_id"].rsplit("/", 1)[-1],
            displayName=f"{'P1' if phase else 'P0'} · {dataset} · {method} · {loss} · {setting}",
            dataset=dataset, method=method, loss=loss, labelSetting=setting, collection=PHASE[phase],
            direction="maximize", target=TARGET, sampler=sampler, completeTrials=len(trials),
            reportedCompleteTrials=int(row["complete"]),
            bestTrial=dict(number=best["number"], objective=best["objective"]) if best else None,
            parameters=order, trials=trials))
        n_trials += len(trials)

        folds = max((len(t.get("validation_fold_values") or []) for t in raw), default=0)
        # lib/trials.ts names a study after its file: the dataset before "__", the rest with "_" read as spaces
        short = setting.replace("-shot", "shot").replace(" % labeled classes", "pctLabeledClasses")
        name = f"{FILE_DATASET[cfg['dataset_key']]}__P{phase}_{FILE_METHOD[cfg['method']]}_{FILE_LOSS[cfg['loss']]}_{short}.csv"
        with (out_csv / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["number", "state", "value"] + [f"param:{n}" for n in names] + [f"attr:{TARGET}", "attr:selected_epoch"]
                       + [f"attr:fold_{i + 1}_valid_map_at_r" for i in range(folds)]
                       + ["attr:fold_selected_epochs", "attr:sampling"])
            for t in raw:
                fv, ep = t.get("validation_fold_values") or [], t.get("selected_epochs") or []
                w.writerow([t["number"], t["state"], "" if t["value"] is None else t["value"]]
                           + [json.dumps(t["params"][n]) if isinstance(t["params"].get(n), (list, dict)) else t["params"].get(n, "")
                              for n in names]
                           + ["" if t["value"] is None else t["value"], round(statistics.median(ep)) if ep else ""]
                           + [fv[i] if i < len(fv) else "" for i in range(folds)]
                           + ["|".join(str(e) for e in ep), "random" if t.get("is_random") else "tpe"])

    snapshot = dict(schemaVersion=1, generatedAt="", source="thesis_generator/daten/details.zip (compact thesis data)",
                    totalStudyDirectories=len(index), skippedStudyDirectories=0, studies=studies, failures=[])
    (APP / "public").mkdir(exist_ok=True)
    (APP / "public" / "optuna-studies.json").write_text(json.dumps(snapshot), encoding="utf-8")

    # Phase-1 test results, one row per run and scope, in the columns lib/test-results.ts reads
    rows = list(csv.DictReader((a.daten / "results.csv").open(encoding="utf-8")))
    with (REPO / "trials-files" / "test-results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["result_id", "dataset", "scenario", "mode", "role", "loss", "miner", "ssl_method", "label_sampling_mode",
                    "labeled_fraction", "labeled_per_class", "seed", "study_name", "trial_number",
                    "test_precision_at_1", "test_mean_average_precision_at_r"])
        for r in rows:
            fam, loss = r["method"].split("_", 1)
            k, pct = r["k"], r["budget_percent"]
            w.writerow([f"{r['run_id']}:{r['scope']}", DATASET[r["dataset_key"]], "Phase 1",
                        {"mean": "Mean (single model)", "concat": "Concat (ensemble)"}.get(r["scope"], r["scope"]),
                        "supervised" if fam == "sup" else "ssl", LOSS[loss], "no_miner", "none" if fam == "sup" else METHOD[fam],
                        "labeled classes" if pct else "k images per class", float(pct) / 100 if pct else "", k or "",
                        r["seed"], f"{METHOD[fam]} · {LOSS[loss]}", "-", r["precision_at_1"], r["mean_average_precision_at_r"]])
    print(f"{len(studies)} studies, {n_trials} complete trials, {len(rows)} test results")


if __name__ == "__main__":
    main()
