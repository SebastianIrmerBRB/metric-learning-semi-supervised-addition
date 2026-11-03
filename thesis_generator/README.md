# Thesis figures and tables: generator code and data

## Submission data: start here

[`datenansicht/ergebnisuebersicht.csv`](datenansicht/ergebnisuebersicht.csv) is the
condensed result view across Phase 1, all ablation series, sweeps, the fold-ensemble
experiment, external images and all-label references. Other measurements
(geometry, timing, diagnostics, HPO, qualitative examples) are indexed in
[`datenansicht/dateninventar.csv`](datenansicht/dateninventar.csv).
[`datenansicht/README.md`](datenansicht/README.md) explains the complete data view;
[`datenansicht/spalten.csv`](datenansicht/spalten.csv) describes the overview columns.
The core hand-in consists of the six source CSVs in `daten/`, `details.zip`,
`manifest.json` and the data README. The inventory lists all logical views,
including the detail ZIP members, without storing duplicate uncompressed data.
The figure thumbnails and licensed fonts are also in `details.zip`.

`abdeckung.csv` maps every figure and included table referenced by the current
thesis chapters to its generator, thesis label, number and page. Packaging fails
if an empirical output has no bundled generator. The three hand-drawn TikZ
schematics are marked as belonging to the LaTeX thesis source. Literature values
are attributed in the thesis, rather than presented as measurements of this work.

```bash
python3 export_data.py       # regenerate the condensed result view and complete inventory
python3 export_data.py --expand  # optional: write each individual view and asset
sha256sum -c SHA256SUMS      # check the original submission snapshot, before regenerating
```

The overview contains summaries and individual runs at different granularities;
these rows must not be pooled as independent observations. Units and retained
data selections are documented. This package contains the empirical inputs used
by the thesis generators; original training logs, complete image datasets,
model weights and full per-epoch HPO logs are outside this compact submission.
The submission extension includes compact trial records for the 235 evaluated
HPO studies, sensitivity statistics, the Phase-0 trial comparison, and the six
nearest-neighbor rankings of every Cars196 query used by the retrieval prose.
`datenansicht/export.json` records recomputed checks of those prose results.

This folder contains the Python code that produces every data figure and every generated table of the thesis, together
with the compact experiment data it reads. Training logs, models and image datasets are not needed. The schematic
diagrams drawn in TikZ (loss functions, pipeline, protocol) are LaTeX code in the thesis source and are not part of
this folder.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 generate.py
```

On Windows, use `python` if `python3` is not available. All text files are read and written as UTF-8, also on
Windows. Everything is written to `ausgabe/`: figures as PDF in
`ausgabe/bilder/neu/`, LaTeX tables in `ausgabe/kapitel/`, PNG previews in `ausgabe/vorschau/`, one log per script in
`ausgabe/protokoll/` and intermediate tables in `ausgabe/generiert/`. A full run takes about a minute.

```bash
python3 generate.py --output some/folder            # another output folder
python3 generate.py --only fig_phase2 tab_extern    # selected scripts only
python3 abbildungen_src/fig_phase2.py               # one script on its own, also writes to ausgabe/
```

Requires Python 3.10 or newer. Built and tested with Python 3.12.13, numpy 2.4.6, pandas 3.0.3, matplotlib 3.11.1, scipy 1.18.0, Pillow 12.3.0; `requirements-getestet.txt` pins exactly these
versions (they need Python 3.12 or newer) and reproduces the thesis files exactly. Also tested with Python 3.10 and
3.12 and the minimum versions in `requirements.txt` (numpy 2.0.2, pandas 2.2.2, matplotlib 3.10.0, scipy 1.14.1,
Pillow 10.4.0): every table comes out byte-identical and the figures show the same values, but matplotlib 3.10 places
axes and text slightly differently (shifts of about one point), so the figure files are not byte-identical. The figures use the TeX Gyre Pagella font shipped in
`daten/details.zip` (GUST Font License, included there).

## Outputs and their place in the thesis

Numbers and pages refer to the thesis PDF built on 2026-11-03 09:23.

| Output | Script | In the thesis | Page | Chapter file | Label |
| --- | --- | --- | --- | --- | --- |
| `bilder/neu/geometrie_budget.pdf` | `fig_geometrie.py` | Fig. 5.9 | 70 | ergebnisse | `fig:e:geo:budget` |
| `bilder/neu/geometrie_konzept.pdf` | `fig_konzepte.py` | Fig. 2.2 | 14 | grundlagen | `fig:geometrie:konzept` |
| `bilder/neu/geometrie_korrelation.pdf` | `fig_geometrie.py` | Fig. D.4 | 138 | anhang_ergebnisse | `fig:e:geo:korrelation` |
| `bilder/neu/hoffer_skala.pdf` | `fig_konzepte.py` | Fig. 4.3 | 40 | methodik | `fig:hoffer:skala` |
| `bilder/neu/inshop10_gewinn.pdf` | `fig_inshop10.py` | Fig. 5.3 | 56 | ergebnisse | `fig:e:klassen` |
| `bilder/neu/lp_cars.pdf` | `fig_lp_cars.py` | Fig. 5.6 | 63 | ergebnisse | `fig:e:lpcars` |
| `bilder/neu/lp_toy.pdf` | `fig_konzepte.py` | Fig. 3.1 | 23 | ssdml | `fig:lp:toy` |
| `bilder/neu/phase1_gewinne.pdf` | `fig_phase1_gewinne.py` | Fig. 5.1 | 51 | ergebnisse | `fig:e:gewinne` |
| `bilder/neu/phase1_uebersicht.pdf` | `fig_phase1_uebersicht.py` | Fig. B.1 | 109 | anhang_ergebnisse | `fig:e:uebersicht` |
| `bilder/neu/phase1_verluste.pdf` | `fig_phase1_verluste.py` | Fig. B.2 | 110 | anhang_ergebnisse | `fig:e:verluste` |
| `bilder/neu/phase2_2x2_verluste.pdf` | `fig_phase2.py` | Fig. 5.4 | 60 | ergebnisse | `fig:e:2x2:verluste` |
| `bilder/neu/phase2_lp_qualitaet.pdf` | `fig_phase2.py` | Fig. 5.5 | 62 | ergebnisse | `fig:e:lpqualitaet` |
| `bilder/neu/phase2_sweeps_verluste.pdf` | `fig_phase2.py` | Fig. 5.7 | 65 | ergebnisse | `fig:e:sweeps` |
| `bilder/neu/phase2_sweeps_verluste_inshop.pdf` | `fig_phase2.py` | Fig. 5.8 | 66 | ergebnisse | `fig:e:sweeps:inshop` |
| `bilder/neu/retrieval_ergebnisse.pdf` | `fig_retrieval.py` | Fig. 5.2 | 55 | ergebnisse | `fig:e:retrieval` |
| `bilder/neu/verlauf_k32.pdf` | `fig_verlauf.py` | Fig. D.1 | 136 | anhang_ergebnisse | `fig:a:verlauf32` |
| `bilder/neu/verlauf_rho.pdf` | `fig_verlauf.py` | Fig. D.3 | 137 | anhang_ergebnisse | `fig:a:rho` |
| `bilder/neu/verlauf_verluste.pdf` | `fig_verlauf.py` | Fig. D.2 | 137 | anhang_ergebnisse | `fig:a:verluste` |
| `bilder/neu/verzoegerung.pdf` | `fig_verzoegerung.py` | Fig. E.1 | 141 | anhang_ergebnisse | `fig:a:verzoegerung` |
| `kapitel/anhang_tabellen/extern_cars.tex` | `tab_extern.py` | Table C.9 | 131 | anhang_ergebnisse | `tab:anhang:extern:cars` |
| `kapitel/anhang_tabellen/extern_inshop.tex` | `tab_extern.py` | Table C.10 | 132 | anhang_ergebnisse | `tab:anhang:extern:inshop` |
| `kapitel/anhang_tabellen/extern_sweeps.tex` | `tab_extern.py` | Table C.11 | 133 | anhang_ergebnisse | `tab:anhang:extern:sweeps` |
| `kapitel/anhang_tabellen/geometrie_reinheit_lrml.tex` | `fig_geometrie.py` | Table D.2 | 139 | anhang_ergebnisse | `tab:a:reinheit:lrml` |
| `kapitel/anhang_tabellen/geometrie_verluste.tex` | `fig_geometrie.py` | Table D.1 | 138 | anhang_ergebnisse | `tab:a:geometrie` |
| `kapitel/anhang_tabellen/geometrie_weitere_budgets.tex` | `fig_geometrie_weitere.py` | Table D.3 | 134 | anhang_ergebnisse | `tab:a:geometrie:weitere` |
| `kapitel/anhang_tabellen/gewinne.tex` | `tab_hauptteil.py` | Table B.6 | 108 | anhang_ergebnisse | `tab:anhang:gewinne` |
| `kapitel/anhang_tabellen/hpo_protokoll.tex` | `tab_suchraeume.py` | Table A.2 | 101 | anhang | `tab:phase1:hpo-protokoll` |
| `kapitel/anhang_tabellen/klassen.tex` | `tab_hauptteil.py` | Table B.7 | 108 | anhang_ergebnisse | `tab:anhang:klassen` |
| `kapitel/anhang_tabellen/map_cars.tex` | `tab_anhang.py` | Table B.1 | 103 | anhang_ergebnisse | `tab:anhang:cars` |
| `kapitel/anhang_tabellen/map_cub.tex` | `tab_anhang.py` | Table B.2 | 104 | anhang_ergebnisse | `tab:anhang:cub` |
| `kapitel/anhang_tabellen/map_inshop.tex` | `tab_anhang.py` | Table B.3 | 105 | anhang_ergebnisse | `tab:anhang:inshop` |
| `kapitel/anhang_tabellen/map_inshop10.tex` | `tab_anhang.py` | Table B.4 | 106 | anhang_ergebnisse | `tab:anhang:inshop10` |
| `kapitel/anhang_tabellen/map_semiinat.tex` | `tab_anhang.py` | Table B.5 | 107 | anhang_ergebnisse | `tab:anhang:semiinat` |
| `kapitel/anhang_tabellen/negativ.tex` | `tab_hauptteil.py` | Table B.8 | 111 | anhang_ergebnisse | `tab:anhang:negativ` |
| `kapitel/anhang_tabellen/phase2_slade_sweeps.tex` | `tab_slade_sweeps.py` | Table C.7 | 130 | anhang_ergebnisse | `tab:anhang:phase2:slade` |
| `kapitel/anhang_tabellen/phase2_varianten.tex` | `tab_phase02.py` | Table C.3 | 119 | anhang_ergebnisse | `tab:anhang:phase2` |
| `kapitel/anhang_tabellen/phase2_varianten_concat.tex` | `tab_phase2_concat.py` | Table C.5 | 123 | anhang_ergebnisse | `tab:anhang:phase2:concat` |
| `kapitel/anhang_tabellen/phase2_varianten_inshop.tex` | `tab_phase02.py` | Table C.4 | 121 | anhang_ergebnisse | `tab:anhang:phase2:inshop` |
| `kapitel/anhang_tabellen/phase2_varianten_inshop_concat.tex` | `tab_phase2_concat.py` | Table C.6 | 125 | anhang_ergebnisse | `tab:anhang:phase2:inshop:concat` |
| `kapitel/anhang_tabellen/phase2_varianten_k8.tex` | `tab_phase2_k8.py` | Table C.8 | 127 | anhang_ergebnisse | `tab:anhang:phase2:k8` |
| `kapitel/anhang_tabellen/suchraum_phase0.tex` | `tab_suchraeume.py` | Table A.1 | 96 | anhang | `tab:first-run-hyperparameter-spaces` |
| `kapitel/anhang_tabellen/suchraum_phase1.tex` | `tab_suchraeume.py` | Table A.3 | 98 | anhang | `tab:hyperparameter-spaces` |
| `kapitel/anhang_tabellen/verzoegerung.tex` | `fig_verzoegerung.py` | Table E.1 | 141 | anhang_ergebnisse | `tab:anhang:verzoegerung` |
| `kapitel/cars_tabellen/phase1_hpo.tex` | `tab_hpo.py` | Table B.9 | 112 | anhang_ergebnisse | `tab:cars:hpo` |
| `kapitel/cars_tabellen/zahlen.tex` | `zahlen_ergebnisse.py` | (numbers in the text) |  | ergebnisse | `(Zahlen im Text)` |
| `kapitel/cub_tabellen/phase1_hpo.tex` | `tab_hpo.py` | Table B.10 | 113 | anhang_ergebnisse | `tab:cub:hpo` |
| `kapitel/ergebnisse_tabellen/ablationen_verluste.tex` | `tab_phase02.py` | Table 5.5 | 59 | ergebnisse | `tab:e:ablationen:verluste` |
| `kapitel/ergebnisse_tabellen/beste.tex` | `tab_beste.py` | Table 5.2 | 53 | ergebnisse | `tab:e:beste` |
| `kapitel/ergebnisse_tabellen/extern.tex` | `tab_extern.py` | Table 5.4 | 57 | ergebnisse | `tab:e:extern` |
| `kapitel/ergebnisse_tabellen/geometrie.tex` | `fig_geometrie.py` | Table 5.7 | 69 | ergebnisse | `tab:e:geometrie` |
| `kapitel/ergebnisse_tabellen/k4_ensemble.tex` | `k4_ensemble.py` | Table 5.6 | 68 | ergebnisse | `tab:e:k4ensemble` |
| `kapitel/ergebnisse_tabellen/klassen_2x2.tex` | `tab_klassen_2x2.py` | Table 5.3 | 56 | ergebnisse | `tab:e:klassen:2x2` |
| `kapitel/ergebnisse_tabellen/korrelation_budgets.tex` | `tab_korrelation_budgets.py` | Table 5.8 | 72 | ergebnisse | `tab:e:korrelation` |
| `kapitel/ergebnisse_tabellen/referenz.tex` | `tab_hauptteil.py` | Table 5.1 | 50 | ergebnisse | `tab:e:referenz` |
| `kapitel/inshop10_tabellen/phase1_hpo.tex` | `tab_hpo.py` | Table B.12 | 115 | anhang_ergebnisse | `tab:inshop10:hpo` |
| `kapitel/inshop_tabellen/phase1_hpo.tex` | `tab_hpo.py` | Table B.11 | 114 | anhang_ergebnisse | `tab:inshop:hpo` |
| `kapitel/methodik_tabellen/datensaetze.tex` | `tab_datensaetze.py` | Table 4.1 | 31 | methodik | `tab:datensaetze` |
| `kapitel/semiinat_tabellen/phase1_hpo.tex` | `tab_hpo.py` | Table B.13 | 116 | anhang_ergebnisse | `tab:semiinat:hpo` |

Generated as well, but not used in the thesis: `bilder/neu/graphreinheit.pdf` (`fig_graphreinheit.py`), `bilder/neu/phase2_2x2.pdf` (`fig_phase2.py`), `bilder/neu/phase2_graph.pdf` (`fig_phase2.py`), `bilder/neu/phase2_sweeps.pdf` (`fig_phase2.py`), `bilder/neu/verlauf_k4.pdf` (`fig_verlauf.py`), `kapitel/anhang_tabellen/feste_einstellungen.tex` (`tab_suchraeume.py`), `kapitel/ergebnisse_tabellen/aufwand.tex` (`tab_aufwand.py`), `kapitel/ergebnisse_tabellen/korrelation.tex` (`fig_geometrie.py`).

The script logs in `ausgabe/protokoll/` contain the further numbers quoted in the text (for example the computing
time in the section on computational cost, from `tab_aufwand.py`).

## Data

`daten/` holds six CSV tables, `details.zip` with the detailed inputs (geometry measurements, validation curves,
LP diagnostics, search spaces, the two published retrieval queries) and `manifest.json` with the provenance of every
view. `daten/README.md` describes columns, units and identifiers. `uebersicht.csv` summarizes all 450 Phase-1
settings (dataset, method, loss and budget) in one row each. `pruefung.json` is the result of the data check.

Every test result (Mean, Concat and fold values) that a figure or table uses comes from the `cv_summary.json` of the
run that produced it. Runs whose summary lacked a value (mostly the evaluation of the fold ensemble, "Concat") or whose
values had been read from the training log were run again unchanged, and three runs of the supervised baseline on
In-Shop that stopped at an epoch cap of 600 were run again with the cap of 1000 used by the other methods. Diagnostics
that runs record only in their training log (for example LP pseudo-label counts) come from the log of the same run.
`cvsum_rerun/ersatz.csv` in `daten/details.zip` lists every original with its rerun and whether the rerun reproduced it.
Of the 307 unchanged reruns of this kind, 289 reproduce their original, 16 differ in Concat by at most 0.0015 points
with identical fold values, and two (Hoffer with Multi-Similarity on Cars196 with 25 % and 50 % CompCars) differ by at most 0.1
points in Mean: their originals of 9 October ran before the CompCars copy on the server was rebuilt on 16 October
(224x224 PNG instead of JPEG, new file names). The share is drawn by hashing the file path, so it has the same number
of images and classes but different images; runs with 100 % reproduce exactly, and the reruns reproduce the reruns of
20 October exactly. The thesis uses the rerun in every
case. A final audit of all runs (23 October) added reruns for three further reasons: the LRML/Proxy-Anchor runs with the
supervised hyperparameters on Cars196 had used a batch of 64 instead of the supervised batch sampler (32:4), every Cars196
run with 10 % to 50 % CompCars from before the rebuild of 16 October was run again on the current copy, and all results
are now trained without delayed patience (runs that had patience 40 use patience 20), so that the runs of a comparison are
trained alike. Two exceptions remain: the Phase-2 series at k=8 runs entirely with delayed patience, and the supervised
baseline with Proxy-Anchor in the In-Shop series at 10 % to 50 % uses it because patience 20 alone keeps the untrained
starting state in at least one fold. The values with delayed patience are a second value in the appendix
(`patience_delay/paare.csv`, figure `verzoegerung`). `daten/external.csv` also contains 55 runs with seeds 8 to 11 whose
values come from the log; no figure or table uses them. `daten/sources.csv` lists the source file of every value with
its SHA-256 checksum at import.

## What the code does, checked on 23 October 2026

- It reads only `daten/` and writes only to the output folder. It needs no network access and no absolute paths.
  Only if the bundled font were missing would `stil.py` look for it in the usual TeX font folders.
- Values are not typed into the code. Reference values (DINOv2 features and the untrained head), the LP confidence
  threshold, search spaces and epoch caps are read from `daten/`. The scripts contain only the selection of what to
  show (datasets, budgets, methods) and the layout.
- In the figures that show results, every plotted value lies inside its axes. Fixed axis limits (the same range in
  several panels) widen automatically when a value would fall outside (`stil.grenzen`), so a corrected value is never
  cut off. Two cuts are deliberate: in the In-Shop sweeps (`phase2_sweeps_verluste_inshop.pdf`) values below -20 points
  appear as arrows at the lower edge, with a legend entry, and in the supplementary `phase2_graph.pdf` one value
  (-22.6) appears as a triangle at the left edge. The three concept figures (`geometrie_konzept`, `lp_toy`,
  `hoffer_skala`) show synthetic points. In panel (b) of `geometrie_konzept` three of them lie outside the axes. They
  still count in the printed value of that panel.
- Exclusions are explicit: the geometry tables leave out folds without a test value or with incomplete tracking
  (list in `ausgabe/generiert/geometrie/ausgeschlossen_weitere.csv`). The tables state how many loss functions enter
  each mean.
- Paired gains are recomputed from `results.csv` on every run, pairing by loss, budget, seed and scope. Three derived
  views are likewise recomputed from the source tables when loaded, so a corrected value propagates: the reference
  with all labels, the differences of the SLADE sweeps and the supervised-baseline rows of the k=4 ensemble table. With
  the shipped data they are unchanged.
- Since 26 October 2026: `fig_geometrie_weitere.py` computes three files, `geometrie/korrelation_weitere.csv`
  (correlations), `mittel_weitere.csv` (means) and `ausgeschlossen_weitere.csv` (excluded combinations).
  `tab_korrelation_budgets.py` reads all three, `fig_geometrie.py` the correlations. `generate.py` runs
  `fig_geometrie_weitere.py` before both, also with `--only`. Such a file left in `ausgabe/generiert/` from an earlier
  run is used only if it is newer than the data, `fig_geometrie_weitere.py` and the modules it imports. Otherwise, and
  when no such file exists, a script run on its own first runs `fig_geometrie_weitere.py` in a temporary folder and keeps
  only these three files. Copies of them are also stored in the data. The correlations and means there predate the
  geometry runs of the In-Shop series with labelled classes and differ from the computed files in those rows, so only
  `pruefe_daten.py --compare-raw` reads the stored copies.
- Two parts use LRML configurations from a different search with the same loss function: the computing-time
  measurements and the geometry runs. The thesis says so in the methods chapter.

Checks you can run:

```bash
python3 abbildungen_src/pruefe_daten.py        # keys, five seeds per setting, fold counts, value ranges,
                                               # Mean equal to the mean of the folds, patience rule, checksums
python3 abbildungen_src/test_datensammlung.py  # an edited result reaches the gains; inconsistent folds are rejected
```
