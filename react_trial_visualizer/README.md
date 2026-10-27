# Trial Atlas

A React + Tailwind dashboard for Optuna/HPO CSV files. It recreates and extends the diagrams from `build_trial_dashboard.py` with live CSV import, automatic schema detection, trial inspection, cross-study comparison, TensorBoard training curves, and SVG/PNG export.

The **Hyperparameters** workspace reads every `optuna_study.db` below the configured study roots: `../exporter/optuna-studies` (the mirror `exporter/export_optuna_studies.sh` downloads from the server) and `../logs` (studies run on this machine). It includes a searchable study catalog, ranked fANOVA importances, numeric slice plots with binned-median trends, categorical choice distributions, rank associations, and grouped response summaries.

## Recommended start

From PowerShell:

```powershell
cd react_trial_visualizer
.\Start-Trial-Atlas.ps1 -OpenBrowser
```

This builds the static React site and starts one local service at `http://127.0.0.1:8765`. The service hosts the website and reads TensorBoard scalars from the repository's `tensorboard-files` directory. On later starts, use `-SkipBuild` when the frontend has not changed.

## Run locally

```powershell
cd react_trial_visualizer
npm install
npm run dev
```

Open `http://localhost:3000`. The repository's `../trials-files_ol/cars` CSVs are available through the repository-study picker, but none are added by default. The app also accepts one or many CSVs by file picker or drag and drop. Run `tensorboard_api.py` separately on port 8765 when using the development server and training curves are needed.

Use **Log folder** to import a complete experiment collection in one step. The picker searches the selected folder recursively for files named `trials.csv`, preserves their relative paths for study names and identity, and ignores per-trial files such as `metrics.csv`, `diagnostics.csv`, and `cv_results.csv`. For example, selecting `logs/supervised_losses` imports its ten nested HPO studies and opens the portfolio overview.

The Optuna browser snapshot is refreshed automatically before `dev` and `build`. To refresh it without restarting the frontend, run `npm run sync-optuna`, then use **Reload snapshot** in the Hyperparameters workspace.

`sync-optuna` walks each study root for `optuna_study.db` files. Parameter importances come from the root's `parameter_importance_summary.csv` when that file exists; every remaining study is ranked by running `../scripts/optuna_param_importance.py` (fANOVA) once and caching its report as `parameter_importance.json` next to the database. Later runs reuse the cache and finish in under a second, and a study is recomputed only when its database is newer than its cache. Studies with fewer than two completed trials are skipped. Because both roots are active, study identifiers and paths are prefixed with the root folder name (`logs/…`, `optuna-studies/…`), which is also what the catalog search box matches.

| Variable | Effect |
| --- | --- |
| `OPTUNA_STUDIES_DIR` | Replaces the default roots. Accepts several paths separated by `:` (`;` on Windows) or `,`. |
| `OPTUNA_PYTHON` | Interpreter used for importances. Defaults to the repository's `venv`/`.venv`, then `python3`/`python`. |
| `OPTUNA_SKIP_IMPORTANCE=1` | Export only what is already cached or listed in a summary CSV; never start Python. |
| `OPTUNA_FORCE_IMPORTANCE=1` | Recompute every importance report, ignoring the caches. |

## Epoch tracking

The **Epoch tracking** workspace reads every tracked run bundled into
`public/epoch-tracking.json` and provides a run picker above the charts. Refresh the snapshot with:

```bash
npm run sync-epoch-tracking
```

By default the exporter searches all of `../logs`, so older tracked runs remain selectable alongside
the latest one. Set `EPOCH_TRACKING_ROOT` only when you intentionally want to restrict the snapshot;
pointing it at one run (for example, `../logs/final_tracking`) produces a picker with just that run.
The discovery and aggregation code lives in `../evaluate/`.

## Live runs (dev server)

With `npm run dev` (or `npx next dev` to skip the snapshot sync), the **Embeddings** page lists
every `epoch_tracking/` below `../logs` under **Live runs**, running or not, and reads it straight
from disk: no export step, no waiting for the fold to finish. A tracked epoch appears within a
couple of seconds of its 2D projection being written; its 3D projection, graph, class evidence
and propagation follow as the tracker writes them. If you are on the newest epoch, the view moves
on to each new one; pan, zoom, selection and colour mode are kept. **Follow newest run** switches
to a run as soon as it starts writing, so the page can stay open while a manifest works through
its folds. `/embeddings/?live=newest` opens the most recently written run directly.

How it works: `app/api/live-tracking/` serves a manifest built from file stats (polled every 2 s),
the tracked points once per run, and one frame per epoch, built by `../evaluate/live_tracking.py`
only when that epoch's files changed and cached in the dev server's memory. The page aligns the
frames itself (`lib/procrustes.ts`, the same Procrustes chain as the exporter) and hands the result
to the same viewer that draws exported views. The tracker (`training/epoch_tracking.py`) renames
its CSVs and npz archives into place once complete and keeps `epoch_tracking/status.json` current
(epoch, stage, running/finished), which the status strip above the canvas shows.

| Variable | Effect |
| --- | --- |
| `LIVE_TRACKING_ROOTS` | Directories searched for `epoch_tracking/`, relative to the repository; default `logs`. Several separated by `:`, `;` or `,`. |
| `LIVE_TRACKING_PYTHON` | Interpreter for `evaluate/live_tracking.py`; defaults to the repository's `venv`. |

The static site has no route handlers, so it keeps using exported views; `evaluate/export_embedding_views.py`
is only needed for that, and for saved-head and test-projection views.

## Phase 2 todo

`/todo/` is a ranked list of what to do next for Phase 2, with links to the page each task is done
on and copyable commands. It reads `data/todo.json`; on the dev server, checking an item off,
writing a note or adding a task saves back to that file (`app/api/todo/`). On the static site the
checks are kept in the browser. `/?view=parameters`, `/?view=archive` and `/?view=compare` open
those workspaces directly.

## Run configs

`/configs/` shows what every study and run under `/mnt/f/Studien` actually ran with: Phase 0 and
Phase 1 HPO studies (and, behind a toggle, `_weitere`), the seeded series in `Seeded_Runs`, and
every Phase-2 folder, split into ablations and geometry. Build its snapshot with

```bash
npm run sync-configs
```

The exporter (`scripts/export-study-configs.mjs`) reads each study's `study_config.json` and
`optuna_study.db`, and each replay run's `cv_summary.json` (or, for a run that never finished,
its first fold's `run_config.json`). It only reads; nothing under the studies root is written. The
first export takes about two and a half minutes because F: is slow; every parsed file is cached in
`.study-configs-cache.json` by size and mtime, so later exports only read what changed. The page
fetches `public/study-configs.json` in the browser.

| Tab | What it answers |
| --- | --- |
| Checks | Everything that ran differently from its peers or is missing: protocol settings that differ within one phase and dataset, seeded runs of one folder started with different settings, replays that did not take the linked Phase-1 study's best trial (and which study's best they took instead), folds that hit the epoch cap, unfinished studies and series, folders without a config, and Phase-2 folders a manifest plans but F: does not have. |
| Protocol | Datasets × methods for one phase: epochs, patience, folds, sampler and trials, label budget, regularizer cap and more, plus wall time and epochs trained. Values that differ from the dataset's majority are shaded; the first column is each dataset's majority, so it compares datasets. |
| Compare | Every setting side by side for any selection, grouped into protocol, data, optimisation, loss, SSL, HPO and seeds, with runtime rows. Cells are compared with their row's majority or with a column you pick as reference; only differing rows are shown by default. |
| Phase trace | One dataset · method · loss through Phase 0, Phase 1, the seeded series and Phase 2, compared by default with the Phase-1 study the replays took their trial from, plus the search spaces side by side. |
| Search spaces | HPO ranges per dataset on a shared (log) scale, the best trial's value, and the Phase-0 range under each Phase-1 bar. |
| Coverage | Dataset × method × loss grids of what exists and how far it got, and for Phase 2 the folders the manifests in `configs/run_manifests/phase2/` plan (expanded to `<run>_<variant>` like `run_scheduler.py` does, one run per budget and seed) against what is on F:. |

Tuned settings of an HPO study show the best trial's value. Runtime is measured, not configured:
HPO wall time comes from the trial timestamps; a replay run lasts from its `cv_<timestamp>`
folder to the time its `cv_summary.json` was written, so a summary rewritten later stretches it.
A fold hit the epoch cap when its last epoch reached `epochs - 1`, the same rule as the HPO archive.

| Variable | Effect |
| --- | --- |
| `STUDIES_DIR` | Replaces the default `/mnt/f/Studien`. |
| `PHASE2_MANIFESTS_DIR` | Replaces the default `../configs/run_manifests/phase2`. |

## Embeddings

The **Embeddings** page draws one run's points on a canvas in 2D or 3D. Drag to pan (2D) or
rotate (3D; right-drag pans), scroll to zoom around the cursor, click a point to select it,
shift-drag to box-select a region, double-click or press R to reset. Labeled, unlabeled,
validation and test points toggle independently. Colour follows the subset (labeled blue
diamonds, unlabeled orange circles, test aqua squares/triangles for gallery/queries,
validation gray), or up to three pinned classes, with every other point grayed. Only three
colours are used because any two scatter points can sit side by side, and the reference
palette clears the colour-blind checks for all pairs only with its first three slots. Marker
shape carries the subset in both modes. A selected point, a pinned class or a region gets a
per-subset count table beside the canvas.

Views are files under `public/embedding-views/`, written by
`../evaluate/export_embedding_views.py`:

```bash
# A run saved with --save_head_state: train split (labeled/unlabeled), validation and D_test, 2D + 3D
python evaluate/export_embedding_views.py --root logs/ablations/unlabeled_batch_size
# Epoch tracking: one layout and one graph per tracked epoch
python evaluate/export_embedding_views.py --tracking-root logs/final_tracking/lrml_prx_replay
# test_tsne.csv files, and a CV run's test_tsne_concatenated.csv (all folds' embeddings joined)
python evaluate/export_embedding_views.py --tsne-csv-root logs/final_tracking
```

Final-test replays can keep only the reported models with
`--save_final_head_state`; run-scheduler manifests expose the same behavior as
`"save_head_state": true` inside `study_replay`. After export, every fold is
selectable under **Saved heads**. The page's run finder also accepts pasted
Linux or `\\wsl.localhost\...` artifact paths; a `test_embeddings.npz` path
opens the adjacent 2-D/3-D test projection even before the static index is
refreshed.

**Images** come straight from `../data/`: views carry no image paths. The image route
(`/api/embedding-image/?run=&set=&sample=`, in `next dev` and in `tensorboard_api.py`) rebuilds
the run's datasets once with `evaluate/embedding_image_index.py`, so the first image of a run
takes a few seconds and later ones are immediate. The **Images** column beside the canvas
(on top of the side panel on narrow screens) shows the selected point next to its neighbors, or
a box-selected region's images; click an image to select that point.

**Neighbors**: for epoch-tracking views, selecting a point draws every edge of the run's own
graph at that epoch, read from `epoch_tracking/epoch_*/graph_edges.npz`. Runs from before that
file existed only have each node's "strongest neighbor", which is an arbitrary tie on a binary
kNN graph such as LRML's. For those views, and for test-only views, the gallery falls back to
the points nearest on screen. The **Neighbors** slider picks how many of a point's own ranked
neighbors (by cosine, from `graph_edges.npz`'s `knn`) are drawn and shown, up to the graph's k;
its last notch shows every edge of the graph row.

**Mechanism diagnostics** (epoch tracking with `graph`, from 2026-09-17 on). Every tracked epoch
writes `class_evidence.npz` for any method: Iscen's label propagation from the labeled points
(with its weight ω), Hoffer's reference softmax, and the network's own prediction (a softmax over
labeled class prototypes, τ = 0.05). `epoch_metrics.csv` adds their accuracy on unlabeled
points, accuracy of the 10/25/50% most confident, certainty AUC, and the conditional entropy
H(y|x) on labeled, unlabeled and test points. Pseudo-label methods also save each generation to
`pseudo_labels/`. The viewer shows them as color modes, per-point class-probability bars, and a
**Mechanism diagnostics** panel: accuracy over training (diffusion vs network vs Hoffer vs the
run's pseudo labels), conditional entropy, top-confidence accuracy, confidence histograms of right
vs wrong predictions, and confident mistakes by true → predicted class (click to see the images).
Hidden labels only score these; nothing reaches training.

**Label propagation, step by step** (`epoch_tracking/epoch_*/label_propagation.npz`). The
mechanism panel above shows where the diffusion *ends*; this shows how it got there. The same
propagation is saved as its series — F₀ holds only the labeled seeds, and each step lets the mass
travel one more edge — so a **Propagation step** slider sits beside the epoch slider, with its own
play button. Three color modes read it: the class each point holds at that step, whether that class
is its true one, and how many edges the mass travelled from the labeled point it started at.
Switch the Images column to **Label propagation** to draw the arrival edges themselves: with nothing
selected, every edge the mass has travelled so far, the current step's front at full strength over
the faded earlier ones (orange crosses a class boundary); with a point selected, the chain back to
the labeled node its label came from, the points reached through it, and the neighbors holding the
most of its class now. The panel adds the per-step accuracy curve and the epoch's numbers — how much
of the pool was reached, how often an arrival edge stayed inside a class, how often the mass came
from the true class, and how much of it the strongest single neighbor carries. Runs tracked before
this existed can be filled in from their saved graphs with
`python scripts/backfill_label_propagation.py logs/final_tracking`, which propagates over the run's
own tracked graph (the viewer says which graph a view used).

**SLADE**: refreshes also save the cluster centers and each member's cosine to its center, and
`slade_pairs/epoch_*.npz` keeps a sample of the pairs Eq 7 mined. SLADE never reads the centers;
the cluster IDs only fit its positive/negative similarity distributions, and mined pairs may cross
clusters. Switch the Images column to **SLADE mined pairs** to draw a point's mined positives
(blue) and negatives (orange). The **Outliers** mode marks the 5% of points farthest from their own
nearest neighbors.

A `test_tsne.csv` can also be dropped straight onto the page. It carries neither a
labeled/unlabeled split (the test set has none) nor a third axis (t-SNE was fitted in 2D), so
those views are 2D test-only. `scripts/make_ssl_ablation.py` writes `save_head_state: true`
into every ablation it generates, so ablation folds can always be exported in full. On
class-disjoint datasets (Cars196, CUB, In-Shop) a class appears in only one of train,
validation and test, so a class's count table fills a single group.

## Sorted HPO archive

The **HPO archive** workspace reads `sorted_logs/` — the 378 Optuna studies sorted into
`phase_0/` (random search), `phase_1/` (TPE) and `unfinished/`. Its snapshot is refreshed before
`dev` and `build`; run `npm run sync-sorted` to rebuild it on its own, then use **Reload** in the
workspace.

`sync-sorted` walks the archive for `optuna_study.db` files and reads the facets straight out of
the directory layout (`<bucket>/<mode>/<dataset>/[<ssl_method>/]<dml_loss>`), so phase, training
mode, dataset, SSL method, DML loss and the `__suffix` campaign tag are all filterable without any
guessing from run names. `sorting_map.csv` and each study's `study_config.json` fill in what a path
cannot express — miner, label budget, search space, original campaign path — and every trial's
parameters, objective, per-fold scores, selected epoch and runtime come out of the database.
Studies with a stored fANOVA report under `parameter_importance/` carry their ranking too.

| Variable | Effect |
| --- | --- |
| `SORTED_LOGS_DIR` | Replaces the default `../sorted_logs` archive root. |

**Validation only.** Nothing in this archive is tuned against the test split, so the exporter
carries no test metric at all — not even the null ones the HPO trials recorded. `sync-sorted`
prints every objective metric it saw and warns loudly about any that is not a `best_valid_*`
one, so a test-set objective cannot slip in unnoticed.

Seven tabs share one faceted filter panel. **Compare** ranks any grouping of the selection and
crosses two facets into a clickable matrix; scores can be read absolutely or as a delta against the
best supervised sweep with the same dataset, loss, label budget and phase. **Studies** and
**Trials** are the sortable tables behind it, with a CSV export that carries every hyperparameter.
**Hyperparameters** puts one response panel per group on shared axes. **Folds** and **Gradients**
are described below. Clicking a point in any chart highlights that trial in all of them.

## Per-fold views

Every trial's objective is the mean over its cross-validation folds, and `sync-sorted` exports
the folds themselves — score, P@1, selected epoch, last epoch, train loss — not just the
aggregate. The **Folds** tab reads them two ways at once:

- **Aggregate**: one row per fold index with its median score, its median epochs and how often it
  ran out of budget, plus an "all folds" row above it.
- **Fold offset**: how much harder or easier each fold is than the trial it belongs to. This is
  the fold main effect the noise model removes; it is a property of the split, not of the
  hyperparameters, and reading it as trial-to-trial variation is what makes a sweep look noisier
  than it is. Fold 2 of this archive sits 0.013 below its trials' mean across 17,000 runs.
- **Trial by fold**: for the selected study, every trial against every fold, each cell coloured
  by its distance from that trial's own mean. Vertical stripes are a hard fold; scattered colour
  is the trial-by-fold interaction the noise floor is actually built from.

Offsets only cancel over studies that ran the same number of folds. This archive mixes 4-fold and
5-fold runs, so the tab says so and each row names how many studies stand behind it.

## Gradient contribution

SSL runs write `diagnostics.csv` per fold with the gradient-contribution and GradNorm series —
about 10 GB across 42,550 files. `sync-sorted` summarises each fold's series (median, inner
deciles, within-fold spread in decades, steady-state value over the last quarter of the run) and
caches the result next to the study as `gradient_summary.json`, so a repeat export is seconds
rather than minutes. The summaries ship in their own `public/sorted-gradients.json`, fetched only
when the **Gradients** tab is opened.

The tab covers, aggregate and per fold:

- Any recorded series — regularizer/supervised norm ratio, regularizer share, cosine between the
  two gradients, weights, relative training rate, SLADE basis ratio — read either over the whole
  run or at steady state.
- **Spread**: how far the series wandered *within* a fold (in decades) beside how much folds
  differ *from each other* (p10–p90).
- **Deviation from target**: where a controller declared a target ratio, the achieved ratio is
  joined to it step by step and reported as log10(achieved/target), so 0 is on target and +1 is
  ten times too strong. Across the 132 studies that run one, the achieved ratio sits 0.94 decades
  — about 8.6x — above target, consistently on every fold.
- **Does it move the objective?** One point per fold against that trial's shortfall from its own
  study's best, in small multiples by any facet.

A negative cosine means the regularizer gradient points against the supervised one rather than
with it, which is common here and worth knowing before reading any weight as "how much help".

## Startup trials and epoch budget

Both sit at the top of the page, above every tab, because they are budget decisions for Phase 1
rather than properties of one view.

**Startup trials.** TPE only benefits from its better group once that group contains a point that
is genuinely good rather than a lucky mediocre one. The hit rate is measured, not assumed: p is
the share of a sweep's trials inside the same tolerance band the Range finder uses, counting
failed trials in the denominator because a startup trial that diverges still costs one of the n.
Two estimates are shown — the analytic `n >= ln(1-c)/ln(1-p)` for independent draws, and the exact
probability that a subset of n of the trials actually run contains a hit,
`1 - C(N-K,n)/C(N,n)`. The second is the closed form of the bootstrap-over-subsets experiment, so
there is nothing to resample, and both automatically carry whatever dimensionality cost the search
space imposed: a good region that shrank in higher dimensions shows up as a lower p and a curve
that crosses later. The answer is raised to a floor of 20, because with ten startup trials TPE's
better group is the top ceil(0.1 N) = 1 trial and the "good" density is a single point plus the
prior. Per-study numbers differ by method and are all kept.

**Epoch budget.** Early stopping fires `patience` epochs after the best one, so a fold that
reaches the final epoch never stopped — it ran out of room while still within patience of an
improvement, and its score would move if the budget went up. The header reports what share of
folds that is, over how many studies. On this archive it is 393 of 71,390 folds (0.55%) across 74
of 428 studies, with the median fold stopping at epoch 45 of a 500 budget: early stopping is doing
the work and the budget is not the binding constraint.

## Phase 0 range finder

The **Range finder** tab turns a Phase 0 sweep into Phase 1 bounds, following the three signals in
`lib/phase0.ts`: hard failures, a tolerance band around the best, and a boundary hit. Every
threshold is editable on the page and everything else follows from it.

- **Tolerance ε.** By default it is measured, not assumed. Because every trial of a study is scored
  on the *same* cross-validation folds, a hard fold shifts all trials together and cancels in a
  trial-to-trial comparison, so the raw fold spread is not the noise floor — on this archive it
  overstates it by about 4x, and by 12x on some studies. A two-way decomposition removes the shared
  fold effect and reports the standard error of the difference between two trials, `sigma *
  sqrt(2/F)`; ε is a multiple of that (2 by default). An absolute or percentage ε is available for
  when you want to fix it by hand.
- **Hard failures.** Optuna FAIL, PRUNED and unfinished trials are counted as failed, trials whose
  best epoch was their first as stalled, and trials in the study's failure tail (Tukey fence, or an
  absolute floor when the collapse level is known) as collapsed.
- **The verdict.** The good region is the parameter interval the in-band trials span. It extends
  when it reaches an edge, when the best trial sits at one, or when the binned response is still
  flat there; it narrows when it covers a small share of the searched span and the rest is degraded
  or dead; otherwise it stays. Margins are then added — half a decade on a log axis by default,
  half the good span on a linear one — and the result is offered as a `spaces` block you can copy
  straight into a `configs/runs/*.json`. Bounds are snapped to readable numbers by default
  (`3e-5`, not `0.0000357`) and always snapped **outward**, so a rounded range is a superset of
  the exact one and never drops a trial the band kept. The Exact toggle turns it off.
- **Resolvability.** A parameter whose whole binned response fits inside ε is reported as
  unresolved rather than as flat, and no choice is dropped from a categorical space on that
  evidence. The same check runs per study: the diagnostics strip shows the effect size in units of
  ε, so "this parameter does nothing" and "my noise floor is larger than the effect" stay distinct.
  The share of in-band trials whose best epoch fell in the last fifth of the run is shown next to
  it, because a truncated run flatters high learning rates.

The strip carrying ε, the noise floor, the effect size, the failure counts and the truncation risk
sits above the tab bar, so those numbers are in view on every tab.

Response plots use *shortfall* — each trial minus its own study's best — rather than the raw
objective. That gives the axis a meaningful zero, lets studies scoring at different levels share a
panel, and stops one collapsed run from owning the axis and squeezing the real candidates into a
few pixels; the failure tail is folded onto the bottom edge as triangles instead. The ε band is
drawn behind the points as a yardstick, and each caption states the axis span.

Run `npm run verify-phase0` to check the rules. It pins them against three synthetic sweeps whose
answer is known by construction and re-runs the statistics over the exported archive.

## Production build

```powershell
npm run build
```

The static site is written to `out/`. Every chart can be exported independently as SVG or PNG, and the filtered trial table can be exported as CSV.

The trial table includes a searchable **Columns** selector for showing any attribute in the source CSV. Column selections are saved per study in browser local storage, alongside quick actions for restoring the inferred defaults, clearing the table, or selecting every attribute.

Every chart can also be collapsed, reordered with its drag handle, or opened in a near-fullscreen inspection view. The enlarged view preserves exports and chart interactions, closes from its minimize control, backdrop, or the Escape key, and returns to the original page position. Layout and collapsed state are stored per study in browser local storage, so the workspace returns to the same arrangement after a reload.

## Hosted Phase-2 site

```bash
npm run build:share                  # geometry + embeddings
SHARE_TABS=all npm run build:share   # also the ablation tabs on the geometry page
SHARE_VIEWS=all npm run build:share  # every exported view (SHARE_VIEWS=<regex> filters on the view label)
```

Writes a static site to `share/` for Cloudflare Pages (served at the domain root). The geometry page is at `/` with its snapshot inlined, and the embedding viewer is at `/embeddings/` over the views in `public/embedding-views/`. By default the scratch runs (`demo_epoch_tracking`, `headrun`, `trackrun`) are left out. The build uses only `app/**/*.share.tsx`. There is no dev-server polling, no live runs, no path lookup and no link back to the atlas. Local paths are removed: log roots become folder names, and views lose `runDir`, so the hosted viewer shows no sample images. Cloudflare Pages refuses files over 25 MiB, so views above 24 MiB are stored as `<file>.0`, `<file>.1`, … and the viewer joins them. Refresh `public/geometry.json` and the embedding views first, because the build bundles whatever is there.

Upload `share/` in the Cloudflare dashboard (Workers & Pages → your project → new deployment) or with `npx wrangler pages deploy share --project-name <name>`.

## Attribute correlations

Each study includes an interactive numeric-attribute correlation explorer. Use Spearman correlation for ranked or monotonic hyperparameter relationships, or Pearson correlation for linear relationships on the original values. The selectable heatmap uses pairwise-complete observations; clicking a cell updates a scatter plot with the exact coefficient, paired-trial count, and least-squares trend. Attribute choices, method, and selected pair are persisted per study.

## TensorBoard integration

Selecting a chart point or trial row opens a compact summary fixed at the top-left of the workspace. Toggle chart points, use the summary action, or use the **Curves** checkboxes in the trial table to compare multiple trials. The dashboard loads each selection through `attr:log_dir` and overlays validation loss (or validation mAP@R), batch loss, and epoch loss for a shared fold. Dense batch curves are shape-preserving downsampled by the local Python service before they reach the browser.

Trial Atlas does not preload studies or preselect a default trial. The exact workspace study list—including imported CSV studies—active view, selected comparison trials, active trial, and summary visibility are stored in browser local storage. Removed studies remain removed, and an intentionally empty workspace or comparison remains empty after a reload.

Hover a learning curve to inspect the nearest logged point. The inspector shows the trial/fold, TensorBoard tag, value, step, wall time, relative run time, and both raw and smoothed values when batch smoothing is enabled.

## Automatic columns

The detector follows the same priorities as `build_trial_dashboard.py` for trial number, validation metric, state, learning rate, last epoch, and selected epoch. Use **Configure columns** in the UI to override any inferred field.
