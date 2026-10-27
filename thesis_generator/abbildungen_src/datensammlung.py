"""Portable access to the compact thesis data. Never falls back to training logs.

The historical names accepted by read_csv/read_json identify documented views,
not folders that need to exist. See daten/README.md and manifest.json.
"""
import atexit
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import pandas as pd

# Konsolen unter Windows (cp1252) kennen Zeichen wie − oder σ nicht: lieber ersetzen als abbrechen
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("THESIS_DATA_DIR", ROOT / "daten")).resolve()
# Ausgaben: im Thesis-Ordner (main.tex) direkt in kapitel/ und bilder/, sonst (Paket) in ausgabe/ wie generate.py
OUTPUT = Path(os.environ.get("THESIS_OUTPUT_DIR", ROOT if (ROOT / "main.tex").exists() else ROOT / "ausgabe")).resolve()


def output_dir(relative):
    p = OUTPUT / relative
    p.mkdir(parents=True, exist_ok=True)
    return str(p) + "/"


@lru_cache(maxsize=1)
def manifest():
    with (DATA / "manifest.json").open(encoding="utf-8") as f:
        return json.load(f)


def key(path):
    s = str(path).replace("\\", "/")
    return s.split("/rohdaten/", 1)[-1].removeprefix("rohdaten/")


@lru_cache(maxsize=12)
def _csv_file(name):
    return pd.read_csv(DATA / name, dtype=str, keep_default_na=False)


def asset_bytes(name):
    with zipfile.ZipFile(DATA / "details.zip") as z:
        return z.read(name)


@lru_cache(maxsize=1)
def font_files():
    folder = Path(tempfile.mkdtemp(prefix="thesis-fonts-"))
    atexit.register(shutil.rmtree, folder, True)  # temporärer Ordner, am Ende des Skripts wieder gelöscht
    with zipfile.ZipFile(DATA / "details.zip") as z:
        for member in z.namelist():
            if member.startswith("fonts/") and member.endswith(".otf"):
                (folder / Path(member).name).write_bytes(z.read(member))
    return tuple(folder.glob("*.otf"))


def read_json(path):
    return json.loads(asset_bytes(manifest()["views"][key(path)]["member"]))


def _paired_differences(dataset, columns):
    """Recompute gains from the editable results, pairing exactly by seed."""
    r = read_csv(f"{dataset}_ergebnisse/data/phase1_runs.csv")
    budget = "budget_percent" if "budget_percent" in r else "k"
    metrics = ["r_precision", "mean_average_precision_at_r", "precision_at_1"]
    base = r[r.method.str.startswith("sup_")].copy()
    base["loss"] = base.method.str.split("_").str[-1]
    x = r[~r.method.str.startswith("sup_")].copy()
    x["loss"] = x.method.str.split("_").str[-1]
    x = x.merge(base[[budget, "scope", "seed", "loss"] + metrics],
                on=[budget, "scope", "seed", "loss"], suffixes=("", "_sup"), validate="many_to_one")
    rows = []
    for (method, b, scope), g in x.groupby(["method", budget, "scope"], sort=False):
        for metric in metrics:
            d = 100 * (g[metric] - g[metric + "_sup"])
            rows.append(dict(method=method, **{budget: b}, scope=scope, metric=metric,
                             mean=d.mean(), std=d.std(ddof=1), positive_seeds=int((d > 0).sum()),
                             min=d.min(), max=d.max(), n_seeds=d.count()))
    return pd.DataFrame(rows)[columns]


def _run_settings(dataset):
    """One row per Phase-1 run with its resolved settings; values decoded from value_json, missing ones None."""
    r = _csv_file("results.csv")
    r = r[(r.dataset_key == dataset) & (r.scope == "mean")]
    budget = "budget_percent" if dataset == "inshop10" else "k"
    runs = pd.read_csv(io.StringIO(r[["run_id", "config_id", "method", budget, "seed"]].to_csv(index=False)))
    h = _csv_file("hyperparameters.csv")
    h = h[h.config_id.isin(set(runs.config_id))]
    wide = h.pivot(index="config_id", columns="parameter", values="value_json")
    wide = wide.apply(lambda c: c.map(lambda v: json.loads(v) if isinstance(v, str) else None)).astype(object)
    return runs.merge(wide, left_on="config_id", right_index=True, how="left", validate="many_to_one")


def _refresh_references(name, frame):
    if name == "ausgangsleistung/alle_labels.csv":
        external = read_csv("extern/laeufe.csv")
        external = external[external.kind == "referenz"]
        for dataset, key in (("Cars196", "cars"), ("In-Shop", "inshop")):
            for r in external[external.dataset == dataset].itertuples():
                mask = (frame.key == key) & (frame.loss == r.loss) & (frame.seed == r.seed)
                for metric in ("mean_map", "concat_map", "concat_rp", "concat_p1"):
                    frame.loc[mask, metric] = getattr(r, metric)
    elif name == "phase2/slade_sweeps.csv":
        variants = read_csv("phase2/variants.csv")
        regular = variants[(variants.method == "slade") & (variants.variant == "defaults")].set_index(["dataset", "loss"])
        for scope in ("mean", "concat"):
            frame["regular_" + scope] = [regular.loc[(r.dataset, r.loss), scope + "_value"] for r in frame.itertuples()]
            difference = frame[scope + "_value"] - frame["regular_" + scope]
            previous = frame["diff_" + scope]
            # Preserve the source's sign for tiny differences that print as
            # signed zero; recompute when a measurement changes materially.
            frame["diff_" + scope] = difference.mask((difference - previous).abs() <= 1e-12, previous)
        frame["tuned_dev_mean"] = frame.diff_mean.where(frame.is_tuned)
    elif name == "phase2_k4_ensemble/runs.csv":
        # The rows of the supervised baseline (variant ue) repeat Phase-1 results on Cars196 at k=4: take them from results.
        runs = read_csv("cars_ergebnisse/data/phase1_runs.csv")
        runs = runs[(runs.k == 4) & runs.method.str.startswith("sup_")]
        for scope, column in (("mean", "mean"), ("concat", "concat")):
            sel = runs[runs.scope == scope]
            value = sel.set_index([sel.method.str.split("_").str[1], "seed"]).mean_average_precision_at_r
            ue = frame.variant == "ue"
            frame.loc[ue, column] = [100 * value[(r.loss, r.seed)] for r in frame[ue].itertuples()]
    return frame


def read_csv(path, **kwargs):
    name = key(path)
    view = manifest()["views"][name]
    if "derived" in view:
        if view["derived"] == "paired_differences":
            return _paired_differences(view["dataset"], view["columns"])
        if view["derived"] == "run_settings":
            return _run_settings(view["dataset"])
        # Data written by a preceding generator, e.g. geometry correlations. A file older than the data could describe
        # other values, so it is only used when it is newer; without it the copy stored in the bundle is used.
        generated = OUTPUT / "generiert" / name
        if generated.is_file():
            newest = max(p.stat().st_mtime for p in DATA.iterdir() if p.is_file())
            if generated.stat().st_mtime < newest:
                raise SystemExit(f"{generated} is older than the data in {DATA}. Run the script that writes it first "
                                 f"(fig_geometrie_weitere.py) or delete it.")
            return pd.read_csv(generated, **kwargs)
    if "file" in view:
        frame = _csv_file(view["file"])
        for column, value in view.get("where", {}).items():
            frame = frame[frame[column] == str(value)]
        # Re-infer per-view types: the union CSV contains other experiments too.
        contents = frame[view["columns"]].to_csv(index=False)
        frame = pd.read_csv(io.StringIO(contents), **kwargs)
    else:
        frame = pd.read_csv(io.BytesIO(asset_bytes(view["member"])), **kwargs)
    return _refresh_references(name, frame)


def write_csv(frame, name, **kwargs):
    """Derived outputs go beside figures, never into the source data bundle."""
    p = OUTPUT / "generiert" / key(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, **kwargs)


def clear_cache():
    _csv_file.cache_clear()
    manifest.cache_clear()
