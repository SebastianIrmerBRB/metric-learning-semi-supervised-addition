"""Export every documented thesis data view and a compact cross-experiment result table.

The original six CSVs and details.zip remain authoritative. This export makes
their logical views directly readable without opening ZIP members or running
individual figure scripts. It also exports the images and fonts used by them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

import datensammlung as data
from erstelle_uebersicht import overview


DATASETS = {"Cars196": "cars", "CUB": "cub", "In-Shop": "inshop",
            "DeepFashionInShop": "inshop", "iNat": "semiinat", "Semi-iNat-2021": "semiinat"}
SOURCE_FILES = ("results.csv", "folds.csv", "ablations.csv", "external.csv",
                "hyperparameters.csv", "sources.csv", "details.zip", "manifest.json", "README.md")


def category(name):
    if "_ergebnisse/" in name:
        return "Phase 1"
    if name.startswith("phase2_k4_ensemble/"):
        return "Fold ensemble"
    if name.startswith("phase2"):
        return "Ablations and diagnostics"
    return {"extern": "External unlabeled images", "geometrie": "Geometry and training histories",
            "qualitative": "Qualitative examples", "ausgangsleistung": "Reference models",
            "timings": "Computing time", "patience_delay": "Early stopping comparison",
            "cvsum_rerun": "Run selection and provenance", "datensaetze": "Dataset statistics",
            "fonts": "Plotting fonts", "hpo": "HPO trial measurements and configurations",
            "hpo_sensitivitaet": "HPO sensitivity and plateau statistics",
            "phase0_vergleich": "Preliminary experiments and search-space selection"}.get(name.split("/")[0], "HPO studies" if name == "studies.json" else "Source data")


def result_overview(phase1):
    """Keep different experiment families explicit; never pool their seeds or units."""
    tables = []

    def add(frame, experiment, source, **columns):
        result = pd.DataFrame(index=frame.index)
        result["experiment"] = experiment
        result["source_view"] = source
        result["source_data_row"] = range(1, len(frame) + 1)
        for name, value in columns.items():
            result[name] = frame[value] if isinstance(value, str) and value in frame else value
        tables.append(result.reset_index(drop=True))

    add(phase1, "phase1", "../uebersicht.csv", dataset="dataset", method="method", loss="loss",
        budget_type="budget_type", budget="budget", n_seeds="seeds", aggregation="mean_over_seeds",
        mean_map_percent="mean_map_percent", concat_map_percent="concat_map_percent",
        mean_seed_sd_pp="mean_map_seed_sd_pp", concat_seed_sd_pp="concat_map_seed_sd_pp",
        gain_mean_vs_supervised_pp="paired_mean_gain_pp", gain_concat_vs_supervised_pp="paired_concat_gain_pp",
        thesis_selection=True)
    for family in ("phase2", "phase2_k8", "phase2_klassen"):
        name = family + "/variants.csv"
        frame = data.read_csv(name)
        add(frame, family, name, dataset=frame.dataset.replace(DATASETS), method="method", loss="loss",
            budget_type="percent_classes" if family == "phase2_klassen" else "images_per_class", budget="k",
            variant=frame.variant.fillna("defaults").replace("", "defaults"), n_seeds="mean_seeds",
            aggregation="source_variant_summary", uses_hidden_labels="uses_hidden_labels",
            mean_map_percent="mean_value", concat_map_percent="concat_value",
            mean_seed_sd_pp="mean_sd", concat_seed_sd_pp="concat_sd",
            gain_mean_vs_supervised_pp="mean_diff_supervised", gain_concat_vs_supervised_pp="concat_diff_supervised",
            thesis_selection=True)
    name = "phase2/slade_sweeps.csv"
    frame = data.read_csv(name)
    add(frame, "slade_sweeps", name, dataset=frame.dataset.replace(DATASETS), method="slade",
        loss="loss", parameter="param", parameter_value="value", aggregation="individual_run",
        budget_type="images_per_class", budget=frame.dataset.map({"Cars196": 16, "In-Shop": 2, "DeepFashionInShop": 2}),
        seed=2, n_seeds=1, mean_map_percent="mean_value", concat_map_percent="concat_value",
        gain_mean_vs_regular_pp="diff_mean", gain_concat_vs_regular_pp="diff_concat", thesis_selection=True)
    name = "phase2_k4_ensemble/runs.csv"
    frame = data.read_csv(name)
    add(frame, "k4_ensemble", name, dataset="cars", method=frame.variant.map(lambda v: "sup" if v == "ue" else "lrml"),
        loss="loss", budget_type="images_per_class", budget=4, variant="variant", seed="seed", n_seeds=1,
        aggregation="individual_run", mean_map_percent="mean", concat_map_percent="concat", thesis_selection=True)
    name = "extern/laeufe.csv"
    frame = data.read_csv(name)
    add(frame, "external", name, dataset=frame.dataset.replace(DATASETS), method="method", loss="loss",
        budget_type="percent_external_images", budget="pct", variant="kind", parameter_value="sweep_value",
        seed="seed", n_seeds=1, aggregation="individual_run", mean_map_percent="mean_map", concat_map_percent="concat_map",
        thesis_selection=frame.seed.eq(2), note="Additional seeds are retained; thesis tables use seed 2.")
    name = "ausgangsleistung/alle_labels.csv"
    frame = data.read_csv(name)
    add(frame, "all_labels_reference", name, dataset="key", method="sup", loss="loss", budget_type="all_labels",
        seed="seed", n_seeds=1, aggregation="individual_run", mean_map_percent="mean_map", concat_map_percent="concat_map",
        thesis_selection=True, note="Cars196/In-Shop reference values also occur in the external experiment view.")
    return pd.concat(tables, ignore_index=True)


def export(output, expand=False):
    output = Path(output).resolve()
    for name in SOURCE_FILES:
        if not (data.DATA / name).is_file():
            raise FileNotFoundError(data.DATA / name)
    output.mkdir(parents=True, exist_ok=True)
    inventory = []

    def record(content, name, kind, rows=None, columns=None, view=None):
        view = view or {}
        container = "../daten/" + view.get("file", "details.zip")
        if view.get("derived") in ("paired_differences", "run_settings"):
            container = "../daten/results.csv"
            if view["derived"] == "run_settings":
                container += "; ../daten/hyperparameters.csv"
        relative = ("ansichten/" if kind == "view" else "assets/") + name
        if expand:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        inventory.append(dict(category=category(name), kind=kind, logical_view=name,
            file=container, member=view.get("member", ""),
            expanded_file=relative if expand else "", rows=rows, columns=columns,
            export_bytes=len(content), export_sha256=hashlib.sha256(content).hexdigest(),
            source_id=view.get("source_id", ""), source_file=view.get("file", view.get("member", "")),
            selection=view.get("selection", ""), derived=view.get("derived", "")))

    manifest = data.manifest()
    for name, view in sorted(manifest["views"].items()):
        rows = columns = None
        if Path(name).suffix == ".csv":
            frame = data.read_csv(name, float_precision="round_trip", low_memory=False)
            content = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
            rows, columns = frame.shape
        else:
            content = data.asset_bytes(view["member"])
            if Path(name).suffix == ".jsonl":
                rows = len(content.splitlines())
            elif Path(name).suffix == ".json":
                obj = json.loads(content)
                rows = len(obj) if isinstance(obj, (dict, list)) else 1
        record(content, name, "view", rows, columns, view)

    # These members are inputs too: retrieval thumbnails, plotting fonts/license.
    indexed = {v.get("member") for v in manifest["views"].values()}
    with zipfile.ZipFile(data.DATA / "details.zip") as archive:
        for name in sorted(archive.namelist()):
            if name in indexed or name.endswith("/"):
                continue
            record(archive.read(name), name, "asset", view={"member": name})

    phase1 = overview()
    results = result_overview(phase1)
    results.to_csv(output / "ergebnisuebersicht.csv", index=False, float_format="%.12g")
    pd.DataFrame(inventory).to_csv(output / "dateninventar.csv", index=False)
    dictionary = pd.DataFrame([
        ("experiment", "Experiment family; different families are never pooled."),
        ("source_view / source_data_row", "Logical view from dateninventar.csv and 1-based data-row number, excluding the CSV header. ../uebersicht.csv is the Phase-1 summary. Use export_data.py --expand for individual view files."),
        ("dataset", "cars=Cars196; cub=CUB; inshop=In-Shop; inshop10=In-Shop with fully labeled classes; semiinat=Semi-iNat-2021."),
        ("method", "sup=supervised; iscen/lp=Label Propagation; lrml, hoffer, slade as in the thesis."),
        ("budget_type / budget", "Images per class, percent of labeled classes, percent of external images, or all labels; these are different protocols."),
        ("aggregation / n_seeds / seed", "Phase 1 averages five seeds per setting; ablation summaries and individual runs keep their recorded seed coverage."),
        ("mean_map_percent / concat_map_percent", "Test MAP@R in percent. Mean averages folds; Concat is a separately measured fold ensemble."),
        ("mean_seed_sd_pp / concat_seed_sd_pp", "Sample standard deviation across seeds in percentage points; blank for a single seed."),
        ("gain_*_pp", "Paired differences in percentage points; the reference is named in the column."),
        ("uses_hidden_labels", "Oracle diagnosis using hidden labels. Blank means the source view has no such column."),
        ("thesis_selection", "For external data, only seed 2 enters the thesis tables. True elsewhere identifies a retained thesis experiment family, not a claim that each row is individually printed."),
        ("missing values", "Empty cells mean absent or inapplicable; never replace them with zero."),
        ("overlap", "Families contain controls and repeated references; do not count rows as independent experiments or pool their statistics."),
    ], columns=["column", "meaning"])
    dictionary.to_csv(output / "spalten.csv", index=False)
    counts = dict(views=sum(r["kind"] == "view" for r in inventory),
                  assets=sum(r["kind"] == "asset" for r in inventory),
                  expanded=expand, overview_rows=len(results), overview_rows_by_family=results.groupby("experiment").size().to_dict())
    if counts["views"] != len(manifest["views"]):
        raise RuntimeError("Not every documented data view was indexed")
    if "submission_extension" in manifest:
        pd.DataFrame([
            ("ergebnisse", "sec:e:phase0", "Preliminary searches and learning-rate boundary discussion", "phase0_vergleich/trials.csv; hpo/trials.jsonl"),
            ("ergebnisse", "sec:e:sensitivitaet", "Plateau counts, nearly optimal trials and parameter correlations", "hpo_sensitivitaet/; hpo/; hpo_plateau.csv"),
            ("ergebnisse", "fig:e:retrieval", "All-query top-six accuracy and 32-versus-4 example-selection counts", "qualitative/retrieval_all_queries_top6.csv; qualitative/retrieval_all_queries_provenance.json"),
        ], columns=["chapter", "thesis_label", "empirical_prose", "data_files"]).to_csv(output / "textbelege.csv", index=False)
        stats = data.read_csv("hpo_sensitivitaet/study_stats.csv")
        stats = stats[stats.phase == 1].copy()
        stats["series"] = stats.dataset
        stats.loc[stats.path.str.contains("class_subset_label"), "series"] = "In-Shop (classes)"
        plateau = stats.groupby("series")[["n_within_1pp", "n_within_1se", "second_gap_pp"]].median()
        plateau.to_csv(output / "hpo_plateau.csv", float_format="%.12g")
        params = data.read_csv("hpo_sensitivitaet/param_stats.csv")
        lp = params[(params.phase == 1) & (params.method == "lp") &
                    params.short.isin(["alpha", "gamma", "confidence_threshold", "n_neighbors"])]
        retrieval = data.read_csv("qualitative/retrieval_all_queries_top6.csv")
        pivot = retrieval.pivot(index="query", columns="method", values="correct_top6")
        counts["prose_checks"] = dict(hpo_studies=len(data.read_csv("hpo/studies.csv")),
            hpo_phase1_studies=len(stats), lp_significant=int((lp.spearman_random_p < 0.05).sum()), lp_tests=len(lp),
            retrieval_queries=len(pivot), sup_at_most2_lp_at_least5=int(((pivot.sup <= 2) & (pivot.lp >= 5)).sum()),
            lp_at_most2_sup_at_least5=int(((pivot.lp <= 2) & (pivot.sup >= 5)).sum()),
            correct_top6_percent=(retrieval.groupby("method").correct_top6.mean() / 6 * 100).to_dict())
    (output / "export.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(README.format(**counts), encoding="utf-8")
    return counts


README = """# Datenansicht für die Abgabe

`ergebnisuebersicht.csv` bündelt {overview_rows} Ergebniszeilen über Phase 1,
Ablationen (beide Budgets und gelabelte Klassen), SLADE-Sweeps, Fold-Ensemble,
externe Bilder und die Referenz mit allen Labels. Sie ist ein Leseeinstieg;
Geometrie, Laufzeiten, HPO und Diagnosen bleiben kompakt in `../daten/details.zip`.
`spalten.csv` erklärt die Übersichtsspalten und Einheiten.

`dateninventar.csv` erschließt **alle {views} dokumentierten Datensichten** und
{assets} zusätzliche Bild-/Schriftdateien. Jede Zeile nennt Datei, Umfang,
Auswahl und vorhandene Quellreferenz. `file` nennt die Quelldatei, `member` den
Eintrag in `details.zip`, sofern vorhanden. `export_sha256` beschreibt den bei
Bedarf erzeugten Ansichtsinhalt, nicht den ZIP-Container. Abgeleitete Sichten
werden mit dem mitgelieferten Code neu berechnet.

Die Abgabe speichert jeden Detaildatensatz einmal komprimiert. Optional erzeugt
`python export_data.py --expand` alle Ansichten unter `ansichten/` und die
Bild-/Schriftdateien unter `assets/`. Der Index nennt dann auch `expanded_file`.

| Bereich | Logische Datensicht (siehe Inventar) |
| --- | --- |
| Phase 1, fünf Datenserien, fünf Seeds, alle Metriken/Folds/Settings | `*_ergebnisse/data/` |
| Ablationen, 2×2-Zerlegung, Sweeps, LP-Diagnostik | `phase2/`, `phase2_k8/`, `phase2_klassen/`, `phase2_lp_diagnostik.csv` |
| Ensemble und geteilte Retrieval-Fehler | `phase2_k4_ensemble/` |
| Externe ungelabelte Bilder, Kontrollen und Sweeps | `extern/` |
| Geometrie, Epochen, Validierung, Pseudo-Label-Qualität, Korrelationen | `geometrie/` |
| Unveränderte Merkmale, untrainierter Head, alle Labels | `ausgangsleistung/` |
| Laufzeitmessungen | `timings/faelle.csv` |
| Verzögerte Patience und Auswahlentscheidungen | `patience_delay/`, `cvsum_rerun/` |
| HPO-Suchräume und Protokolle (235 Studien) | `studies.json` |
| HPO-Versuche, Parameter, Validierungs-Folds, komplette Studienkonfigurationen | `hpo/` |
| Sensitivität und flache Optima; Textzahlen | `hpo_sensitivitaet/`, zusätzlich `hpo_plateau.csv` im Ansichtsordner |
| Vorexperimente und Verteilung guter Versuche | `phase0_vergleich/` |
| Klassen- und Bildzahlen | `datensaetze/statistik.json` |
| Retrieval-Nachbarn, t-SNE und übernommene Pseudo-Labels | `qualitative/` |

Die sechs CSVs und `details.zip` unter `../daten/` sind die unveränderten
Quellen dieses Exports. Sie erhalten auch die ursprüngliche Genauigkeit und
Provenienz. Die Ansicht schreibt Fließkommazahlen mit 17 signifikanten Stellen,
die kondensierte Ergebnisübersicht mit zwölf. Einheiten unterscheiden sich je
Quellsicht; die Quellenbeschreibung `../daten/README.md` ist maßgeblich.
CSV: UTF-8, Komma als Trennzeichen, Punkt als Dezimalzeichen. In Excel über
„Daten → Aus Text/CSV“ mit diesen Einstellungen öffnen.

Die Abgabe enthält die Messungen, die für die aktuelle Arbeit benötigt werden,
mit den im Manifest dokumentierten Ausschnitten. Bei Cars196 sind alle
Geometrie-Epochen enthalten; bei den anderen Geometrieserien die ausgewerteten
Anfangs-/Endstände und Validierungsverläufe. Dies ist kein vollständiges Archiv
aller Trainingslogs. Die ergänzte Abgabefassung enthält die kompakten Trial-Daten
der 235 ausgewerteten HPO-Studien und die Retrieval-Rangfolgen aller 8131 Querys.
Vollständige Bilddatensätze,
Modellgewichte und fremde Literaturdatensätze gehören nicht zu diesem Paket.
Die kleinen Abbildungs-Thumbnails sind enthalten. Synthetische Lehrabbildungen
werden aus dem mitgelieferten Code erzeugt.

Nach Datenänderungen: `python generate.py`, dann `python export_data.py` im
Paketstamm. Das regeneriert die Ausgaben und diese Ansichten ohne Trainingslogs.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data.ROOT / "datenansicht")
    parser.add_argument("--expand", action="store_true", help="also write every individual view and asset (duplicates compact sources)")
    args = parser.parse_args()
    print(json.dumps(export(args.output, expand=args.expand), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
