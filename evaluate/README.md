# evaluate/

Readers for the output of `--track_epoch_artifacts` (see
[docs/epoch_tracking.md](../docs/epoch_tracking.md)). Training writes one
directory per fold; these turn any number of those into one set of tables and
pictures.

Nothing here trains, and nothing it reads is written back into a run.

**To watch a run while it trains, you do not need anything below.** The Trial Atlas
dev server reads `epoch_tracking/` directly (Embeddings → Live runs) through
`live_tracking.py`, one epoch at a time. The exports here are for the static site,
for the thesis tables and figures, and for finished runs you want as files.

## One command

```bash
python evaluate/report.py --root logs/phase_1 --out evaluate_out
xdg-open evaluate_out/index.html
```

`--root` is anything above the runs — a single fold, one manifest's output tree,
or all of `logs/`. Discovery walks down looking for `epoch_tracking/`.

## What you get

```
evaluate_out/
  index.html                 <- start here: everything below, in reading order
  data/
    epochs.csv               tidy: one row per run, fold and epoch
    tracked_test.csv         the full D_test table (all measurements + Recall@K)
    runs.csv                 one row per fold: dataset, method, loss, budget, seed, path
    summary.csv              baseline / final / delta per run and metric
  tables/
    summary.md               the same, readable in a terminal
    summary.csv
    summary.tex              booktabs, ready to \input into the thesis
    per_run/<run_id>.md      the full epoch-by-epoch curve for one run
  figures/
    delta_baseline.png       how far each run moved each metric (green = improved)
    trajectory.png           structure vs. score, one arrow per run
    overview_<group>.png     a metric family as small multiples
    curve_<metric>.png       one metric, all runs, fold band
  filmstrips/
    <run_id>_graph.png       the per-epoch graph plots, left to right
    <run_id>_embeddings.png  the per-epoch projections, left to right
```

## The scripts

Each runs standalone with the same `--root` / `--out`, so you can rebuild one
part without the rest.

| Script | Does |
| --- | --- |
| `collect.py` | Walks the tree, writes `data/`. Everything else builds on this. |
| `tables.py` | `data/` → Markdown, CSV and LaTeX tables. |
| `figures.py` | `data/` → the PNG curves, overviews, trajectory and delta chart. |
| `filmstrip.py` | Per-epoch PNGs → one strip per run. `--all-folds` for every fold. |
| `report.py` | All of the above, plus `index.html`. |
| `common.py` | Shared discovery, run identity, metric grouping. Not a script. |

## Two things worth knowing

**Runs are identified from `run_config.json`, not from directory names.**
Directory names are built from a subset of the settings and collide across
datasets — an In-Shop and a Cars study can produce the same leaf name — while
the recorded `args` and `ssl_config` say exactly what ran.

**Folds are averaged, pictures are not.** A metric line is the mean over a
cross-validation run's folds with a min/max band; a filmstrip shows the first
fold only, because pictures of different folds cannot be averaged into anything
meaningful. Use `--all-folds` if you want each fold's strip.

## Reading the trajectory figure

`trajectory.png` puts a structure metric on x and test P@1 on y, with one arrow
per run from its epoch -1 baseline to its last tracked epoch. It exists because a
line chart cannot distinguish two runs that reach the same score by opposite
routes: one spreading its representation out, one collapsing it. Up-and-right is
a method that earned its score. Up-and-left is one that got the score while
losing representational spread — worth a second look before it goes in a table.
