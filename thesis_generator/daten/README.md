## Pseudo-label views — 2026-11-05

Four LP members were replaced or added in place, all other members are byte-identical:
- `phase2/lp_hersteller.csv` now comes from the geometry runs at all four Cars196 budgets (new column `k`).
- `phase2/lp_schwellen.csv` now reads coverage and accuracy of the first update from `pseudo_label_diagnostics.jsonl` of the 100 sweep folds named by `cv_summary.json`, instead of the rounded line in `info.log`. Counts and coverage are identical, accuracy differs by at most 0.005 percentage points.
- New view `geometrie/lp_verfolgung.csv`: per fold, first versus last LP update of the whole pool (class switches, kept and corrected wrong pseudo-labels), 200 folds.
- New view `geometrie/lp_pseudolabels_inshop.csv`: coverage and accuracy at every epoch of the In-Shop geometry runs (k=2 and k=4, seed 2).

Sources by measurement: MAP@R of each threshold-sweep run from `cv_summary.json`, coverage and accuracy with the selected configuration from the geometry runs, coverage and accuracy at other thresholds from the diagnostics of the separate sweep runs. A condensed reading view of these measurements is in [`../datenansicht/diagnostics`](../datenansicht/diagnostics/README.md).

## Added series — 2026-10-28

`phase2_klassen`: the 2×2 decomposition for LRML in the In-Shop series with fully labelled classes at 10 %, seed 2, all five
losses (15 variant rows in `ablations.csv`, 60 folds in `folds.csv`, view `phase2_klassen/variants.csv`). Collected by
`abbildungen_src/sammle_phase2_klassen.py` from `Studien/all_logs/final/phase_2/in_shop_10_lrml_mult`; the regular runs
reproduce Phase 1 with seed 2 exactly. The supervised baseline and the noise thresholds come from the Phase-1 rows at 10 %.
Table `kapitel/ergebnisse_tabellen/klassen_2x2.tex` from `tab_klassen_2x2.py`. `pruefe_daten.py` checks the series like
Phase 2. Existing rows of all four CSVs are byte-identical to before (backup `_backup_vor_klassen_2x2_2026-10-28`).

## Run-source update — 2026-10-27

The newest eligible In-Shop supervised runs at k=4 are selected for Circle seeds 10/11 and NT-Xent seed 8 (9 October repeats, 1000-epoch cap). All three repeat the same trial, hyperparameters, splits and selected epochs as their earlier sources. Maximum Mean change: 0.000026 percentage points; maximum Concat change: 0.001221 percentage points. Results, folds, configurations and provenance were updated together. The replacement records in details.zip now contain 551 decisions. The local freshness audit checked all 2250 seeded runs, 2123 ablation/ensemble/external runs and 350 geometry runs. The report is review/run_freshness_2026-10-27/README.md in the thesis workspace.

# Compact thesis data

This directory is the data source for the current thesis figure and table generators. Training logs, datasets and model files are not needed to generate those outputs.

Start with **`uebersicht.csv`** in the shareable archive: one row per dataset, method, loss and budget (450 settings). It shows Mean/Concat MAP@R, seed standard deviations, paired gains, and selected/trained epochs. It is a derived overview; edit the six source CSVs below and regenerate it.

| File | Contents | Identifiers / units |
| --- | --- | --- |
| `results.csv` | 4,500 Phase-1 result rows; Mean and Concat per seed, including MAP@R, RP, P@1 and available additional metrics | `run_id` + `scope`; metrics are fractions in [0,1]. `epochs` is the training cap, not the epochs actually trained. |
| `folds.csv` | 17,400 fold records: Phase 1, Phase-2 runs with summaries (including the labeled-class series), and external-data experiments | `experiment`, `run_id`, `fold` (external rows use `dataset`, `method`, `loss`, `pct`, `seed`, `kind`, `sweep_value`, `fold`, `source`). `selected_epoch` and `last_epoch` are zero-based; `epochs_trained = last_epoch + 1`. Selected epoch -1 denotes the untrained head. `mean_average_precision_at_r`, `precision_at_1`, `r_precision` are fractions; the external experiment's `test_map`, `test_p1`, `test_rp` columns are percent. |
| `ablations.csv` | Phase-2 variants at both budgets and in the labeled-class series, SLADE sweeps, and the k=4 ensemble experiment | `table_key` selects `phase2`, `phase2_k8`, `phase2_klassen`, `slade_sweeps`, or `k4_ensemble`. Performance values are in percent; differences are percentage points. Shared CSV columns are blank where a particular experiment does not have that field. |
| `external.csv` | 525 run summaries for experiments with external unlabeled images, with controls, selected epochs and available settings | `dataset`, `method`, `loss`, `pct`, `seed`, `kind`, `sweep_value`; performance columns are percent. |
| `hyperparameters.csv` | Resolved training settings, deduplicated into 1,793 configurations | `config_id` + `parameter`; `value_json` preserves numbers, booleans, nulls, lists and strings. Includes learning rates, batches, loss/miner settings, SSL parameters, warm-up, epoch caps, patience and HPO trial identifiers when recorded. |
| `sources.csv` | Source paths, availability at import, file sizes and SHA-256 hashes | `source_id`. Paths are provenance references; they are never opened during generation. Hashes describe the import snapshot. |
| `details.zip` | Geometry measurements and validation curves, LP diagnostics, baseline results, timing measurements, the configurations of all 235 evaluated HPO studies (`studies.json`), dataset class and image counts (`datensaetze/statistik.json`), run-selection records, two LP t-SNE frames, the published retrieval neighbors/thumbnails, and the licensed plotting fonts | Detailed inputs, stored once in compressed form. Its contents and selection rules are listed in `manifest.json`. |
| `manifest.json` | Schema, CSV views, source references, retained columns/rows and import checksums | Historical names identify logical views, not directories that must exist. |

`dataset_key`: `cars` = Cars196, `cub` = CUB, `inshop` = In-Shop, `inshop10` = In-Shop with fully labeled classes, `semiinat` = Semi-iNat-2021. In `results.csv`, `k` is labeled images per class; `budget_percent` is the percentage of fully labeled classes. Implementation name `iscen` denotes LP. CUB has five folds, the other Phase-1 series have four. Seeds are 2, 8, 9, 10 and 11. Mean and Concat remain separate evaluation scopes.
