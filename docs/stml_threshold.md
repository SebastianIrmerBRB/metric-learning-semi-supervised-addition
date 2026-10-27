# STML Threshold Pseudo-Labeling

A variation of STML that trades STML's own loss function for any metric loss in
this repo.

## What changes

STML ([Kim et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.pdf))
estimates the relation between two samples from a teacher embedding as

```
w_ij = (s_ij + c_ij) / 2
```

where `s_ij = exp(-d_ij^2 / sigma)` is a Gaussian kernel on teacher distances and
`c_ij` is the reciprocal-neighbor contextual similarity of its Sec 3.2. STML's
relaxed contrastive loss then uses `w` as a *soft pairwise weight*: it pulls with
`w` and pushes with `1 - w`. That weighting is only expressible inside a loss
written for it, which is why STML ships its own objective.

`w` is already a confidence — the teacher's estimate that two samples share a
class. This method keeps `w` unchanged and cuts it at a threshold instead, the
way [Iscen et al. (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.pdf)
accept only pseudo labels above a certainty cutoff. The result is an ordinary
pseudo-label method: it emits a class id and a confidence per unlabeled sample,
which the shared filter and relabeled-dataset path hand to whatever loss the run
configures.

## Two modes

`graph_batch_mode` picks where the relations are computed.

| | STML | `in_batch` | `global` (default) |
| --- | --- | --- | --- |
| Relation `w` | same | same, dense over the batch | same, on a kNN graph |
| Consumed as | soft pull/push weight | threshold, then hard class id | threshold, then hard class id |
| Teacher | EMA per optimizer step | EMA per optimizer step | EMA per pseudo-label refresh |
| Batches | nearest-neighbor sampler, rebuilt each epoch | same sampler | not applicable |
| Scope | one batch | one batch | whole labeled + unlabeled pool |
| Losses | STMLLoss only | any configured loss | any configured loss |

`in_batch` is the paper's own loop with only the objective swapped: the same
nearest-neighbor batches, the same per-step momentum teacher, and `w` computed
densely across the batch by the same function `STMLLoss` calls — no kNN
approximation, because a batch is small enough to compare exhaustively. Labels
propagate among the batch's own rows, so the labeled stream in each step
supplies the seeds.

`global` trades that fidelity for reach: one graph over the entire pool,
refreshed on a schedule, which propagates much further at low label budgets and
costs one embedding pass per refresh instead of a teacher forward per step.

Both modes share the relation code, the threshold, the propagation, and the
confidence semantics; only the scope and the teacher cadence differ. Because the
cadences differ, so do their momenta: `in_batch` defaults to the paper's `0.999`
per step, `global` to `0.9` per refresh.

Everything below describes the `global` mode unless stated otherwise;
[configs/runs/ssl/stml/in_batch.json](../configs/runs/ssl/stml/in_batch.json)
carries the paper's values (`sigma=1`, `num_neighbors=5`, `topk=10`,
`momentum=0.999`) and requires `graph_labeled_batch_size` /
`graph_unlabeled_batch_size` like every other in-batch graph method here.
`supervised_weight` and `regularizer_weight` are accepted only in that mode.

### Fidelity to the paper

Checked against the official implementation
([sung-yeon-kim/STML-CVPR22](https://github.com/sung-yeon-kim/STML-CVPR22), `code/loss.py`
and `code/main.py`).

Kept: `W_P = exp(-T_dist^2 / sigma)`, the reciprocal nearest-neighbor graph `V`,
the shared-neighbor contextual term, half-`topk` query expansion, and
`W = (W_P + W_C)/2` — a test asserts the sparse graph reproduces the dense
implementation of that formula to 1e-7. The teacher is an EMA of the student
with gradients disabled, as in `Momentum_Update`.

The teacher is held as an EMA of the trainable weights and swapped into the
student for the relation forward, rather than as a second model. A cached
backbone carries a thread lock that cannot be deep-copied, and a frozen backbone
would duplicate weights that never diverge; tracking only trainable parameters
avoids both.

Changed, and why:

* **Teacher cadence** — `global` only. A pseudo-label method has no per-step
  hook, so that teacher advances once per refresh; `0.999` per step over a
  ~40-step epoch is ~0.96 per epoch, which is the range its default sits in.
  `in_batch` keeps the paper's per-step update.
* **No `g` head or KL self-distillation.** Both exist to serve the relaxed
  contrastive objective this method replaces, and nothing would train `g` once
  that objective is gone, so the teacher trails the retrieval embedding directly.
* **No nearest-neighbor batch sampler** — `global` only, where `w` covers the
  whole pool and the sampler has nothing left to arrange. `in_batch` uses it.
* **One view per sample, not two.** The joint labeled/unlabeled forward
  concatenates image tensors, so a two-view batch cannot travel that path; in
  `global` mode the deterministic eval transform is also what keeps pseudo labels
  independent of random augmentation. The nearest-neighbor sampler still puts
  related samples in every `in_batch` batch.
* **Labels come from your labeled split.** This is the substantive departure.
  STML is label-free: `w` itself is the training signal and no sample is ever
  assigned a class. Turning `w` into class ids requires seeds, so the thresholded
  graph propagates the labeled split outward, and unlabeled samples in components
  with no seed are dropped rather than trained on. The literal "STML but
  thresholded" alternative keeps relations pairwise — positives at `w >= t`,
  negatives at `w <= 1 - t`, handed to a loss as a mined `indices_tuple` — which
  needs the per-step training path and excludes proxy and classification losses.

## How a relation becomes a label

1. **Embed** the labeled and unlabeled pools with the EMA teacher, at the
   `update_mode` / `update_interval_epochs` cadence every pseudo-labeler here
   uses. The teacher is seeded from the student at the first refresh — after
   `warmup_epochs`, so it starts from a model that already fits the labeled
   split — and interpolated toward the student at every later refresh. Only
   trainable parameters are tracked, so a frozen backbone costs nothing beyond
   the projection head. `teacher_momentum: 0` disables it and reads relations
   off the student itself.
2. **Build the relation graph.** `stml_relation_graph` computes `w` on a FAISS
   kNN graph: `s` from the Gaussian kernel, `c` from reciprocal neighbors, shared
   neighbors, and query expansion. It reproduces
   `losses.metric_losses.stml_teacher_pair_weights` — STML's own dense math,
   which the test suite checks directly — restricted to the kNN support so it
   scales past a batch.
3. **Threshold.** Relations at or above `positive_threshold` become edges;
   everything else is discarded as a relation the teacher is unsure about.
4. **Propagate.** Edges are consumed strongest first and merged by union-find, so
   labels spread from the labeled split outward. Two properties matter:
   * a sample's confidence is the **bottleneck** weight of its strongest path
     back to a labeled sample, not the weight of the last edge, so samples
     reached through long weak chains score low;
   * two labeled components of **different classes never merge**. Transitive
     merging is the failure mode a pairwise view does not have, and one confident
     but wrong edge would otherwise weld two classes together. Refused merges are
     logged and counted.
5. **Filter and train.** Predictions pass through the shared
   `confidence_threshold` filter, then `make_relabeled_training_dataset` builds
   the training view of labeled plus accepted pseudo-labeled samples. Unlabeled
   samples that never reach a labeled component are dropped with a warning.

## Configuration

```json
{
  "method": "stml_threshold",
  "update_mode": "every_n_epochs",
  "update_interval_epochs": 5,
  "warmup_epochs": 5,
  "confidence_threshold": 0.0,
  "method_params": {
    "n_neighbors": 10,
    "sigma": 3.0,
    "context_topk": 10,
    "positive_threshold": 0.7,
    "sample_weight_mode": "confidence"
  }
}
```

| Parameter | Meaning |
| --- | --- |
| `n_neighbors` | kNN degree of the relation graph. |
| `sigma` | STML's Gaussian bandwidth for `s`. Smaller values make `w` fall off faster with distance, so fewer relations clear the threshold. |
| `context_topk` | STML's neighborhood size for `c` (its `num_views * num_neighbors`), counting the sample itself. |
| `positive_threshold` | The cut on `w`. This is the *relation* threshold that builds the graph. |
| `sample_weight_mode` | `confidence` emits the bottleneck weight, `uniform` emits 1 for every accepted sample. |
| `teacher_momentum` | Per-*refresh* EMA decay of the teacher that reads the relations. `0` uses the student directly. |
| `confidence_threshold` | The standard pipeline cut on the *emitted* confidence, applied after propagation. |

The two thresholds do different jobs: `positive_threshold` decides which
relations are trusted enough to carry a label at all, and `confidence_threshold`
decides which of the resulting pseudo labels are kept. Raising the first prunes
the graph, raising the second prunes long propagation chains.

### Two-stream batches

`global` relabels the whole pool offline, so a training batch can be composed
from the two streams explicitly. Setting `labeled_batch_size` swaps the plain
`MPerClassSampler` for `TwoStreamMPerClassBatchSampler`, exactly as for
[Iscen label spreading](iscen_label_spreading.md#optional-two-stream-m-per-class-batches):

```json
{
  "method": "stml_threshold",
  "labeled_batch_size": 32
}
```

At `batch_size=128` that is 96 accepted pseudo-labeled and 32 true-labeled
samples per batch. Class groups are assigned to the streams without overlap,
making the combined batch globally M-per-class; both stream sizes must therefore
divide by `sampler_m`, and their class union must be large enough for the full
batch. The pseudo-labeled stream sets the epoch length, and confidence filtering
becomes capacity-aware so it cannot starve that stream of classes — the Iscen
page describes the rescue rules in full.

Under HPO, search the labeled *share* with the top-level
`labeled_batch_fraction` key rather than the absolute size: it means the same
thing at every `batch_sampler` choice, so the two dimensions stay independent.
The quota is a whole-pool concept and is rejected under
`graph_batch_mode: "in_batch"`, where `graph_labeled_batch_size` and
`graph_unlabeled_batch_size` size the per-step graph instead.

### Searching the in-batch sizes

`in_batch` ties four knobs together: the graph's labeled rows are drawn from the
supervised batch, so `graph_labeled_batch_size` cannot exceed `batch_size`, and
its unlabeled stream is filled with whole neighbor groups, so
`graph_unlabeled_batch_size` must divide by `n_neighbors`. Searching those
independently would hand Optuna combinations that die in `validate_run_args` and
burn the trial.

`make_stml_in_batch_spaces_constraint_aware` therefore expands the searched
dimensions into their cartesian product, keeps only the tuples the regularizer
itself accepts — it calls that same `validate_run_args`, so the space and the
run-time check cannot drift apart — and collapses them into one joint
categorical space. A single searched dimension is pruned in place instead. If
nothing valid remains, the study fails immediately with the reason rather than
after a trial.

The same study config works for both modes: `graph_labeled_batch_size` and
`graph_unlabeled_batch_size` are dropped under `global`, which has no per-step
graph to size. [configs/runs/stml/cars.json](../configs/runs/stml/cars.json)
carries them and is safe either way;
[cars_in_batch.json](../configs/runs/stml/cars_in_batch.json) is the in-batch
study, differing in the `teacher_momentum` range — per step it belongs near the
paper's `0.999`, per refresh nearer `0.9`.

Ready-made configs live in [configs/runs/ssl/stml/](../configs/runs/ssl/stml):
`hard.json` (threshold only, uniform weights), `weighted.json` (confidence as a
sample weight), and `soft.json` (a permissive threshold that lets confidence do
the discriminating). HPO spaces are in [configs/runs/stml/](../configs/runs/stml)
and a runnable experiment in
[configs/runs/experiments/stml.json](../configs/runs/experiments/stml.json).

## On applying the weight instead of thresholding

`sample_weight_mode="confidence"` weights **samples**, not pairs, and only losses
that declare `supports_sample_weights` read it — `MixedLabelPropagationProxyLoss`
today. Every stock pytorch-metric-learning loss ignores the weight and trains on
the accepted pseudo labels uniformly, which is what `hard.json` states outright.

Restoring STML's actual *per-pair* weighting is not a configuration change.
pytorch-metric-learning has no hook for per-pair weights, so it would mean
forking every loss that should support them, and multiplying a loss by a scalar
weight afterwards only rescales the step size — it cannot express "this pair is
0.8 positive". The faithful per-pair objective already exists in this repo as
`STMLLoss`; see [stml.md](stml.md). This method is the alternative for the case
where the loss must stay free.
