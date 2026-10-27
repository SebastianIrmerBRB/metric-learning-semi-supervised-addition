"""Validate the editable data bundle, optionally compare it to the raw exports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import datensammlung as data

DATASETS = ("cars", "cub", "inshop", "inshop10", "semiinat")


def validate(raw=None, verify_snapshot=False):
    errors, notes, counts = [], [], {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    m = data.manifest()
    check(m["schema_version"] == 1, "Unsupported schema version")
    with zipfile.ZipFile(data.DATA / "details.zip") as z:
        check(z.testzip() is None, "details.zip CRC failure")
        for name, view in m["views"].items():
            if "member" in view:
                check(view["member"] in z.namelist(), f"Missing detail: {name}")
    import tracking
    for epoch in tracking.epochs("qualitative"):
        name = f"qualitative/lp_{epoch}.csv"
        view = m["views"][name]
        frame = tracking.frame("qualitative", epoch)
        accepted = int(((frame["kind"] == "unlabeled") & (frame["lp_cert"] >= view["threshold"])).sum())
        check(accepted == view["accepted_count"], f"LP precision/threshold changed at epoch {epoch}")
    sources = pd.read_csv(data.DATA / "sources.csv")
    check(not sources.source_id.duplicated().any(), "Duplicate source IDs")
    sid = set(sources.source_id)
    hp = pd.read_csv(data.DATA / "hyperparameters.csv", keep_default_na=False)
    check(not hp.duplicated(["config_id", "parameter"]).any(), "Duplicate hyperparameter keys")
    check(set(hp.source_id) <= sid, "Unknown hyperparameter source_id")
    for value in hp.value_json:
        try:
            json.loads(value)
        except (ValueError, TypeError):
            errors.append(f"Invalid JSON parameter value: {value!r}")
            break
    config_ids = set(hp.config_id)
    results = pd.read_csv(data.DATA / "results.csv")
    folds = pd.read_csv(data.DATA / "folds.csv", low_memory=False)
    check(not ((folds.dataset_key != "cub") & (folds.patience == 40)).any(),
          "Recorded Patience 40 outside CUB in folds.csv")
    # Since 2026-10-23 every result is trained without delayed patience, except the Phase-2 series at k=8 (all with it) and
    # the supervised baseline with Proxy-Anchor in the In-Shop series at 10 to 50 %: with patience 20 alone a fold keeps the
    # untrained starting state (15 runs, author's decision 2026-10-23)
    delayed = set(hp.config_id[(hp.parameter == "patience_delay_epochs") & ~hp.value_json.isin(["0", "null"])])
    exception = ((folds.experiment == "phase1") & (folds.dataset_key == "inshop10") & (folds.method == "sup_prx")
                 & pd.to_numeric(folds.budget_percent, errors="coerce").isin([10, 25, 50]))
    check(not (folds.config_id.isin(delayed) & (folds.experiment != "phase2_k8") & ~exception).any(),
          "Delayed patience outside the Phase-2 series at k=8 in folds.csv")
    check(folds[exception].config_id.isin(delayed).all() and folds[exception].run_id.nunique() == 15,
          "Supervised Proxy-Anchor runs of the In-Shop series at 10 to 50 % not all with delayed patience")
    check(folds.config_id[folds.experiment == "phase2_k8"].isin(delayed).all(),
          "Phase-2 run at k=8 without delayed patience in folds.csv")
    check(not results.duplicated(["run_id", "scope"]).any(), "Duplicate result rows")
    check(set(results.config_id.dropna()) <= config_ids, "Unknown result config_id")
    check(set(folds.config_id.dropna()) <= config_ids, "Unknown fold config_id")
    for name in ("results.csv", "folds.csv", "ablations.csv", "external.csv"):
        r = pd.read_csv(data.DATA / name, low_memory=False)
        counts[name] = len(r)
        check(set(r.source_table_id.dropna()) <= sid, f"Unknown source_table_id in {name}")
    for ds in DATASETS:
        p = f"{ds}_ergebnisse/data/"
        r = data.read_csv(p + "phase1_runs.csv")
        f = data.read_csv(p + ("phase1_folds.csv" if ds == "cars" else "phase1_folds_enriched.csv"))
        b = "budget_percent" if ds == "inshop10" else "k"
        keys = ["method", b, "seed"]
        check(not f.duplicated(keys + ["fold"]).any(), f"Duplicate folds: {ds}")
        check(r.groupby(keys).scope.apply(lambda s: set(s) == {"mean", "concat"}).all(), f"Missing Mean/Concat pair: {ds}")
        check(set(r.seed) == {2, 8, 9, 10, 11}, f"Unexpected seeds: {ds}")
        check((r.groupby(["method", b, "scope"]).seed.nunique() == 5).all(), f"Missing seed: {ds}")
        for metric in ("mean_average_precision_at_r", "r_precision", "precision_at_1"):
            check(r[metric].notna().all() and r[metric].between(0, 1).all(), f"Invalid {metric}: {ds}")
            a = r[r.scope == "mean"].set_index(keys)[metric].sort_index()
            v = f.groupby(keys)[metric].mean().reindex(a.index)
            check(np.allclose(a, v, atol=1e-12, rtol=0, equal_nan=False), f"Mean differs from fold mean for {metric}: {ds}")
        expected = 5 if ds == "cub" else 4
        check((f.groupby(keys).fold.nunique() == expected).all(), f"Incomplete folds: {ds}")
        if ds != "cub":
            check((r.patience != 40).all(), f"Patience 40 outside CUB: {ds}")
    for experiment, g in folds[folds.run_id.notna()].groupby("experiment"):
        check(not g.duplicated(["run_id", "fold"]).any(), f"Duplicate run/fold: {experiment}")
        selected, last = pd.to_numeric(g.selected_epoch), pd.to_numeric(g.last_epoch)
        check((selected >= -1).all() and (last >= -1).all(), f"Invalid epoch index: {experiment}")
        check((selected <= last).all(), f"Selected epoch after last epoch: {experiment}")
        check(np.allclose(pd.to_numeric(g.epochs_trained), last + 1, equal_nan=False), f"Wrong epochs_trained: {experiment}")
        if experiment != "phase1":
            check(not (g.patience == 40).any(), f"Patience 40 outside CUB: {experiment}")
    # Phase 2: every run with all four folds, Mean of the variant table equal to the mean of its runs' fold means
    ab = pd.read_csv(data.DATA / "ablations.csv", low_memory=False)
    for experiment in ("phase2", "phase2_k8"):
        g = folds[folds.experiment == experiment].copy()
        check((g.groupby("run_id").fold.nunique() == 4).all(), f"Incomplete folds: {experiment}")
        g["variant"] = g.variant.fillna("").replace("", "defaults")
        g["k"] = pd.to_numeric(g.k)
        keys = ["dataset_key", "method", "loss", "k", "variant"]
        per_run = g.groupby(keys + ["run_id"]).mean_average_precision_at_r.mean()
        fold_mean = 100 * per_run.groupby(level=keys).mean()
        a = ab[ab.table_key == experiment].copy()
        a["variant"] = a.variant.fillna("").replace("", "defaults")
        a = a.rename(columns={"dataset": "dataset_key"}).set_index(keys).mean_value
        check(a.index.isin(fold_mean.index).all(), f"Variant without runs: {experiment}")
        # runs without a variant row: only SLADE sweeps (their own table, slade_sweeps)
        extra = fold_mean.index.difference(a.index).to_frame(index=False)
        regular = ["defaults", "pool_labeled", "sup_all_shared", "zero_weight", "no_warmup"]
        check(((extra.method == "slade") & ~extra.variant.isin(regular)).all(), f"Run without variant row: {experiment}")
        check(np.allclose(a.sort_index(), fold_mean.reindex(a.sort_index().index), atol=1e-9, rtol=0),
              f"Mean differs from fold mean: {experiment}")
    for name, expected in m["files"].items():
        actual = hashlib.sha256((data.DATA / name).read_bytes()).hexdigest()
        if expected["sha256"] != actual:
            (errors if verify_snapshot else notes).append(f"Changed since import: {name}")
    if raw:
        for name, view in m["views"].items():
            path = raw / name
            if path.suffix != ".csv" or not path.exists():
                continue
            a = data.read_csv(name)
            b = pd.read_csv(path, low_memory=False)
            # Selected geometry views intentionally retain only required rows/columns.
            if name.startswith("geometrie/epochen_"):
                keys = ["method", "loss", "k", "fold", "epoch"]
                b = b.set_index(keys).loc[pd.MultiIndex.from_frame(a[keys])].reset_index()[a.columns]
            elif name == "geometrie/epoch_metrics_cars.csv":
                b = b.loc[b.loss == "cir", a.columns].reset_index(drop=True)
            if view.get("derived") == "paired_differences":
                budget = "budget_percent" if "budget_percent" in a else "k"
                keys = ["method", budget, "scope", "metric"]
                a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                    check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            except AssertionError as e:
                errors.append(f"Raw comparison failed: {name}: {str(e)[:250]}")
        notes.append("Compared all retained CSV views with the historical exports (tolerance 1e-12).")
    counts.update(configurations=len(config_ids), hyperparameters=len(hp), sources=len(sources))
    return dict(passed=not errors, counts=counts, errors=errors, notes=notes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare-raw", type=Path)
    p.add_argument("--verify-snapshot", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = validate(args.compare_raw, args.verify_snapshot)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
