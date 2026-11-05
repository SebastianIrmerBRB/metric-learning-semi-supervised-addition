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

## Run

See `README.md` in the folder above: `python3 generate.py` writes all figures and tables to `ausgabe/`. Set `THESIS_DATA_DIR` to use another copy of the data.

## Update correctly

1. Edit the authoritative result rows in `results.csv`, `ablations.csv`, or `external.csv`; keep source references with every new or corrected observation. For Phase 1, update the corresponding folds so their arithmetic mean agrees with the Mean result. Concat is a separately measured ensemble result; it cannot be inferred from the fold average.
2. Keep `run_id` and `config_id` links consistent. Hyperparameters describe the configuration that produced a measured result; changing the metadata does not create a new measurement. To record another configuration, give it a new ID and preserve the old configuration when still referenced.
3. Run `abbildungen_src/pruefe_daten.py`, regenerate into a separate output directory, and review the figures/tables. Paired Phase-1 gains are calculated afresh from `results.csv`, pairing by loss, budget, scope and seed. Derived files are written outside the data directory.
`uebersicht.csv` is regenerated from the source tables, so it will include the edits.

The validator checks unique keys, configuration/source links, five-seed coverage, fold counts, metric ranges, selected/final epoch consistency, agreement of Mean values with the folds in Phase 1, Phase 2 and Phase 2 at k=8, four folds for every Phase-2 run, the patience-40 exclusion outside CUB, and that only the Phase-2 series at k=8 and the 15 supervised Proxy-Anchor runs of the In-Shop series at 10 % to 50 % train with delayed patience. Changed import checksums are reported as notes after intentional edits. `--verify-snapshot` requires the original import checksums to match.

## Provenance and scope

- Phase-1 replacement-run decisions and the original Mean/Concat values are preserved. Hyperparameters come from the selected runs' resolved `run_config.json`, with file references and hashes. Per-fold epochs come from the existing fold exports or the selected Phase-2 `cv_summary.json`; missing information is left blank.
- Every test result (Mean, Concat and fold values) that a figure or table uses comes from the `cv_summary.json` of the run that produced it (since 2026-10-22). Runs whose summary lacked the Concat evaluation, whose values had been read from `info.log`, or that had no summary were rerun unchanged, and three In-Shop Phase-1 runs of the supervised baseline that stopped at their 600-epoch cap were rerun with the 1000-epoch cap of the other methods. `cvsum_rerun/ersatz.csv` in `details.zip` lists each original with its rerun and the reproduction check. Since 2026-10-23 this includes 74 external-image runs with seed 2 whose `cv_summary.json` lacked the Concat evaluation (source `cvsum_extern` in `ersatz.csv`). 72 of them reproduce their originals. Two Cars196 Hoffer/Multi-Similarity runs with 25 % and 50 % CompCars match their originals up to the end of the warm-up (epoch 25) and then differ, because the originals of 9 October ran before the CompCars copy on the server was rebuilt on 16 October (224x224 PNG instead of JPEG, new file names; the share is drawn by hashing the file path, so it has the same number of images and classes but different images); the reruns reproduce the patience-delay reruns of 20 October exactly. Diagnostics that runs record only in their logs (for example LP pseudo-label counts, the LP threshold diagnostics in `phase2/lp_schwellen.csv` and the size of the external pool) come from the `info.log` of the same run. `external.csv` also keeps 55 Hoffer runs on In-Shop with seeds 8 to 11 whose values come from `info.log` (column `werte`, fold values rounded to 0.1 percentage points). No figure or table uses them.
- Since 2026-10-23 (audit of all runs) every result is trained without delayed patience, so the runs of a comparison are trained alike. Until then, a run with a fold that patience stopped too early (best epoch within the first 20 epochs after the warm-up while the validation curve was still rising) entered with its rerun under delayed patience when that rerun validated higher, and the patience-40 runs entered with patience 20 plus the delay. The patience-40 runs now use patience 20 without the delay. Exceptions: the Phase-2 series at k=8, which runs entirely with delayed patience, and the supervised baseline with Proxy-Anchor in the In-Shop series at 10 % to 50 % (15 runs and three geometry runs), where patience 20 alone keeps the untrained starting state in at least one fold. The validator checks both. The values with delayed patience are reported as a second value in the appendix (`patience_delay/paare.csv`). The same audit reran the LRML/Proxy-Anchor runs with the supervised hyperparameters on Cars196 with the supervised batch sampler (32:4 instead of a batch of 64) and every Cars196 run with 10 % to 50 % CompCars from before the CompCars rebuild of 16 October (sources `audit_*` in `cvsum_rerun/ersatz.csv`).
- The packed geometry keeps all Cars196 epochs and the columns used by the generators. For the other datasets it keeps the initial head and the last tracked epoch at or before the selected epoch, plus validation histories. These are exactly the rows needed by the current geometry tables. Selection rules are explicit in the manifest; this is not a full training-log archive.
- The qualitative retrieval data contains only the published queries 758 and 5348, six neighbors per method, and small thumbnails. No full embeddings, checkpoints or full datasets are shipped. LP has the two plotted t-SNE frames (LP recomputed on the tracked images) and, since 25.10.2026, the frames of Figure 5.6 with the pseudo-labels the training adopted for Cars196 and Semi-iNat-2021 (`qualitative/lp_adopted_*.csv`, step `pseudolabels`; frame e with the pseudo-labels LP computed from it for epoch e+1). `geometrie/lp_pseudolabels_semiinat.csv` keeps the LP pseudo-label quality of every epoch on Semi-iNat-2021 (Figure 5.5e, f). Adding new queries or frames requires extracting those extra inputs from the original files.
- `studies.json` holds, for the 125 Phase-1 and 110 Phase-0 studies evaluated in the thesis, the search spaces and protocol settings (`hparam_config`) and the relevant `base_args` of each `study_config.json`. The appendix search-space, protocol and fixed-settings tables and the Phase-1 HPO tables are generated from it and from the run settings.
- The views `<dataset>_ergebnisse/data/phase1_settings.csv` are derived: one row per Phase-1 run with its resolved settings, joined from `results.csv` and `hyperparameters.csv` (values decoded from `value_json`).
- `datensaetze/statistik.json` holds the class and image counts of the four datasets as the training project's dataset code loads them.
- `hyperparameters.csv` is a compact selection of resolved training settings, not a complete reproduction environment. Full training and alternative exploratory analyses still require the original project and data.

The TeX Gyre Pagella font is included in `details.zip` together with `fonts/GUST-FONT-LICENSE.txt`.

## Submission extension — 2026-11-02: evidence for empirical prose

The submission additionally retains the Phase-0 trial comparison (`phase0_vergleich/*`),
the current HPO sensitivity tables (`hpo_sensitivitaet/*`, snapshot 18 October),
and compact trial records for all 235 evaluated studies (`hpo/trials.jsonl`).
Those records include every recorded trial's number, state, objective, sampled
parameters, validation-fold values and selected epochs. Full study configurations
are in `hpo/study_configs.json`; `hpo/studies.csv` joins them to original source
paths and SHA-256 references in `sources.csv`. Source paths are provenance only.
The six-neighbor rankings of all 8,131 Cars196 queries for DINOv2, the supervised
head and LP are in `qualitative/retrieval_all_queries_top6.csv` (Circle, k=16,
seed 2, Concat). They reproduce the two published examples, the 32 versus 4 query
comparison, and the reported 64.0/83.8/85.0 percent correct neighbors. Source hashes
are in `qualitative/retrieval_all_queries_provenance.json`.

The empirical result, fold, ablation, external-result and hyperparameter tables
are unchanged by this extension. Existing detail members are unchanged. Source
references and new logical views were added, with separate snapshot hashes in
`manifest.json`. Historical import checksums are retained.
