# Epoch Tracking

A run records what its model *scores* every epoch. Epoch tracking records what
its representation and its neighbourhood graph *look like* while it gets there:
one bundle at **epoch -1**, before any optimizer step, and then every
`--track_epoch_interval` epochs.

The baseline is the point. Without it a curve cannot say whether the SSL phase
built structure or merely failed to destroy the structure the pretrained
backbone already had.

Everything here is off the selection path. Nothing it computes is read back by
training, early stopping, or Optuna, and it is forced off for HPO trials.

## Asking for it

```bash
python main.py --final_test_study_dir <study> --study_dir_mode final_train \
  --evaluate_test \
  --track_epoch_artifacts graph embeddings test \
  --track_epoch_interval 4
```

| Option | Meaning |
| --- | --- |
| `--track_epoch_artifacts` | `graph`, `embeddings`, `test`, several of them, or `none` alone. |
| `--track_epoch_interval N` | Epochs between bundles. Epoch -1 and the last epoch are always included. `null`/`0` tracks nothing. |
| `--track_epoch_layout` | `tsne` (default), `pacmap`, or `pca`. The graph and the embedding space share it, so the two can be read together. |
| `--track_epoch_max_samples` | Cap on the pool samples each tracked epoch embeds, projects and graphs. Default 4000; `0` keeps the whole pool. |

Requesting artifacts without an interval, an interval without artifacts, or
`test` without `--evaluate_test` is an error rather than a null result.

For a whole manifest, the same four settings are `study_replay` fields:

```json
"study_replay": {
  "study_dir_mode": "final_train",
  "track_epoch_artifacts": ["graph", "embeddings", "test"],
  "track_epoch_interval": 4,
  "track_epoch_layout": "tsne"
}
```

They are owned by that block, so a run may not also pass them in its `args`.

## The three artifact kinds

### `graph`

The kNN graph over the SSL pool — labeled positions first, then unlabeled —
rebuilt from the current embeddings with the method's own `n_neighbors`,
`edge_weighting` and `gamma`, then handed to `training/ssl/graph_diagnostics.py`,
which already draws a graph on its own 2-D layout and measures purity, degree,
components and spectral gap.

The graph is **rebuilt here rather than captured from the regularizer**. At
epoch -1 no method has built one yet; a method's own rebuild schedule is not the
interval being asked about; and a supervised baseline builds none at all — yet
"what would the graph have looked like" is exactly what a supervised row in the
comparison is for. The construction is read-only and never touches the
regularizer's training state.

### `embeddings`

The same pool, projected with the chosen visualizer, plus the geometry that says
whether a representation is collapsing:

| Metric | Reads |
| --- | --- |
| `embedding_effective_rank` | Roy and Vetterli's: `exp` of the entropy of the normalized singular spectrum. A space whose variance has fallen into one direction reports ~1 however many dimensions it nominally has. |
| `embedding_top1_variance_share` | The same collapse from the other end. |
| `embedding_alignment`, `embedding_uniformity` | Wang and Isola's, on the unit sphere. Alignment is a distance, so tighter classes score *lower*. |
| `embedding_intra_class_cosine`, `embedding_inter_class_cosine`, `embedding_class_separation` | The plain cosine gap a metric-learning table already gives intuition for. |
| `embedding_norm_mean`, `embedding_norm_std` | Before normalization, so a shrinking-norm failure is still visible. |

Same-class pairs are drawn per sample rather than from uniform random pairs: the
same-class fraction of a uniform sample is ~1/C, which on In-Shop's thousands of
classes leaves far too few pairs to estimate alignment from.

### `test`

D_test evaluated off the selection path at the tracked epochs, logged to
`metrics.csv` under the split name `tracked_test` so it can never be mistaken
for the single end-of-run test evaluation the run is reported on. It runs for
the final full-development fit **and** for CV folds, which otherwise compute no
pre-training test score at all. The final fit's existing epoch-0 evaluation is
reused as the baseline rather than paid for twice.

## Output

Inside each fold's own log directory:

```
epoch_tracking/
  epoch_metrics.csv      one row per tracked epoch: test, embedding and graph columns
  epoch_m001/            the pre-training baseline
    graph.png            the graph drawn on its own layout
    graph_summary.json   the full diagnostics summary
    graph_nodes.csv
    graph_edges.npz      every edge, plus each node's ranked neighbor search
    class_evidence.npz   how label propagation, Hoffer and the prototypes read this epoch
    label_propagation.npz  the same propagation, step by step (below)
    embeddings_tsne.png
    embedding_metrics.json
  epoch_0000/
  epoch_0004/
  ...
```

`epoch_metrics.csv` is the table a plot is built from; the per-epoch directories
hold the pictures and the complete JSON.

Because the graph series keeps a stable `series_slug`, each epoch's
`graph_summary.json` also carries a `temporal` block comparing it with the
previous tracked epoch — how much of the edge set survived, and how the degree
distribution moved.

## How the labels were propagated

`class_evidence.npz` keeps the *answer* label propagation arrives at:
`(I - alpha S) F = Y` solved, one class distribution per node.
`label_propagation.npz` keeps the *propagation*. The same answer is the limit of

```
F_0 = Y                      only the labeled seeds hold anything
F_t = alpha S F_{t-1} + Y    one more edge of travel per step
```

so walking that series says what the solve collapses: at step `t` a node holds
only the label mass that reached it over at most `t` edges.

| Array | Per | Holds |
| --- | --- | --- |
| `step_top_classes`, `step_top_probabilities`, `step_share` | node × step | The class it holds then (-1 while it holds nothing), that class's probability, and its mass as a share of its own final mass |
| `arrival_step`, `parent`, `parent_share` | node | The step its first mass arrived, the neighbour that delivered it, and that neighbour's share of the mass arriving that step |
| `source`, `hop` | node | The labeled node at the root of that chain, and its distance in edges |
| `influence`, `influence_share` | node × 3 | Once the mass has settled: the neighbours contributing most to the node's own class, strongest first |
| `step_accuracy`, `step_reached` | step | Accuracy on the unlabeled nodes reached so far, and the share of them reached |

`parent`, `source` and `influence` index `positions`, exactly as the edges of
`graph_edges.npz` do.

The arrival tree and the settled flow answer different questions, and on a graph
with a large `k` the first is nearly trivial: almost every node touches a labeled
node directly, so it arrives at step 1 and the tree is one hop deep. The settled
contributors still spread over a dozen neighbours — `lp_influence_share` says how
much the strongest one carries.

`epoch_metrics.csv` gains `lp_reached_share`, `lp_mean_hops`,
`lp_tree_edge_purity` (how often the edge a label arrived over stayed inside a
class), `lp_source_accuracy` (how often the labeled node it came from was the
node's own class), `lp_influence_share`, `lp_influence_purity` and
`lp_trace_agreement` — the share of nodes where the truncated series and the
run's own solve end on the same class. The same numbers ride along inside the
`.npz`, so a reader has them without the CSV.

Hidden labels only score all of this; nothing here reaches training.

### Epochs tracked before this existed

An earlier tracked epoch still has its graph (`graph_edges.npz`) and which nodes
carried a label (`graph_nodes.csv`), but not its embeddings, so the graph cannot
be rebuilt with label propagation's own settings. `scripts/backfill_label_propagation.py`
runs the same series over the graph the epoch saved instead, with the run's own
`alpha`, and records `graph_source="tracked_graph"` rather than
`"label_propagation"`:

```bash
python scripts/backfill_label_propagation.py logs/final_tracking
```

For an Iscen run the two graphs are the same matrix, and the backfilled trace
lands on the saved `class_evidence.npz` answer exactly. For a method whose graph
is built differently (LRML's binary kNN union), it is "how labels would spread
over this run's graph", not a replay of Iscen's numbers.

The viewer (`react_trial_visualizer` → **Embeddings**) draws the result: a step
slider beside the epoch slider, three color modes (the class a point holds at
this step, whether it is the true one, how many edges the mass travelled), and
the arrival edges themselves, with the step's own front over the faded earlier
ones. Selecting a point draws the chain back to the labeled node its label came
from.

## Cost

Two different things dominate depending on what you track.

**Structure only** (`graph`, `embeddings`): one extra embedding pass over the
capped pool, a graph build, and a 2-D projection — and the projection dominates.
The propagation trace adds one more kNN search and ~0.4 s per tracked epoch at
4000 nodes, and ~200 KiB on disk.
Prefer raising `--track_epoch_interval` over raising
`--track_epoch_max_samples`.

**With `test`, the test evaluation dominates, and it drives peak memory.** A
tracked test epoch computes the *full* `REPORT_TEST_MEASUREMENTS` table — NMI and
AMI (which share a k-means over the query embeddings) plus full-depth MAP and
MRR — exactly the set validation deliberately avoids because of its cost. An
ordinary run pays that once, at the end; tracking makes it recur every interval.

Measured on Cars196 (8131 test queries, frozen DINOv2-B, `cv_k=4`) at
`--track_epoch_interval 5`: 15 full test evaluations per fold against 65 cheap
two-metric validation ones, with `VmRSS` ~3.0 GB and **`VmHWM` 7.2 GB** — against
a ~2.2 GB steady-state floor for a training process with no tracking. Size
concurrency on the peak, not the steady state, and read `VmHWM` from
`/proc/<pid>/status` rather than a sampled `VmRSS`.

If that is too expensive, the lever is the measurement set rather than the
interval: `test` tracking currently inherits `--test_measurements`, so narrowing
that narrows every tracked epoch too.

The sample cap is drawn once per fold, at the pool's own labeled/unlabeled
ratio, so every tracked epoch describes the same samples and the rows form a
series rather than a sequence of unrelated snapshots.

## With an ablation

The tracking options are ordinary training arguments, so `--ablation_config`
composes with them: a manifest run can name an ablation config *and* track
structure, and every variant the launcher expands writes its own
`epoch_tracking/` directory under that variant's save directory. See
[ablation_config.md](ablation_config.md).
