# Iscen / LP-DeepSSL label propagation: implementation and audit

Audit date: 2026-08-11

## Bottom line

The whole-pool `iscen_label_spreading` function implements the graph and
diffusion part of Iscen et al., *Label Propagation for Deep Semi-supervised
Learning* (CVPR 2019), reasonably closely. Its defaults match the authors'
public implementation, and its handling of disconnected graph components is
safer than the reference code.

The complete training method is **not an exact reproduction of LP-DeepSSL**.
This project adapts the propagated labels to metric-learning losses and
M-per-class sampling. Most configured losses do not consume Iscen's certainty
weights, the paper's inverse pseudo-class-population loss weight is replaced by
class-balanced sampling, and several configs named `iscen` use very different
graph/solver parameters. The optional in-batch path is a project-specific
variant. The two in-batch correctness problems found during this audit were
fixed and now have regression coverage.

## Code paths

The normal, whole-pool path is:

```text
training engine
  -> IscenLabelSpreadingPseudoLabeler.generate_pseudo_labels
  -> deterministic embedding extraction over [labeled, unlabeled]
  -> iscen_label_spreading
  -> PseudoLabelResult(position, predicted class, certainty)
  -> confidence filtering / optional sampler-capacity rescue
  -> RelabeledSubset
  -> M-per-class or two-stream M-per-class training
```

The main implementation points are:

- `training/ssl/algorithms.py`
  - `make_mixed_label_affinity`: shared cosine-kNN graph builder.
  - `iscen_label_spreading`: propagation, normalization, and certainty.
  - `solve_sparse_label_system`: SciPy, CuPy, Torch, or CHOLMOD solver
    dispatch.
- `training/semi_supervised.py`
  - `IscenLabelSpreadingPseudoLabeler`: whole-pool adapter.
  - `InBatchGraphPseudoLabelRegularizer`: optional per-step local-graph
    adaptation.
- `training/ssl/pseudo_labels.py`: filtering, capacity rescue, and relabeling.
- `training/ssl/data.py`: relabeled datasets and the two-stream sampler input
  contract.
- `training/engine.py`: warm-up, refresh timing, and loss dispatch.

## Whole-pool algorithm

Let `N` be the number of labeled plus unlabeled training examples and `C` the
number of mapped training classes. The adapter always places labeled examples
first, so the unlabeled result is a suffix of every algorithm output.

### 1. Embeddings and graph construction

The current model is put in evaluation mode and evaluated with the dataset's
deterministic feature transform. Embeddings are converted to contiguous
`float32`, L2-normalized, and searched with an exact FAISS inner-product index.

For every example the builder retrieves up to `k` non-self neighbors and
constructs the directed affinity from paper equation (10):

```text
A_ij = max(v_i^T v_j, 0)^gamma
```

It then constructs the symmetric nonnegative graph and removes its diagonal:

```text
W = A + A^T
```

The implementation explicitly clips negative cosine similarities as required
by the paper. The authors' repository instead raises the retrieved similarity
directly to the third power; those paths differ when a retained neighbor has
negative cosine similarity.

### 2. Label seeds

The paper displays an ordinary one-hot seed matrix `Y`. The authors' repository
adds a class-balancing detail: every class receives total seed mass one, so a
labeled example from class `c` receives `1 / n_labeled(c)`. This project follows
the repository here.

Classes with no labeled examples receive an all-zero right-hand side. See the
closed-set warning in the audit: propagation cannot discover such a class.

### 3. Degree normalization and disconnected components

The graph is symmetrically degree-normalized:

```text
D_ii = sum_j W_ij
W_normalized = D^(-1/2) W D^(-1/2)
```

Before solving, the code finds connected components. Any component with no
labeled node has a zero source term for every class, so its exact propagated
score is zero. Those components are excluded from the solve and their rows are
left as an explicit all-zero sentinel. The adapter omits those unlabeled rows
from training instead of assigning the arbitrary class produced by `argmax` on
an all-zero vector.

This is a sensible robustness improvement over the public reference code.

### 4. Sparse linear solve

The method solves paper equation (11), once for all class right-hand sides:

```text
(I - alpha W_normalized) Z = Y
```

The registry defaults match the paper/reference recipe:

| Setting | Default |
| --- | ---: |
| `n_neighbors` | `50` |
| `gamma` | `3.0` |
| `alpha` | `0.99` |
| `cg_rtol` | `1e-6` |
| `cg_max_iter` | `20` |
| `linear_solver` | `cg` |

`linear_solver="cg"` runs independent SciPy CG solves and deliberately keeps
the final iterate when the 20-iteration reference limit is reached. The
non-converged result is therefore expected behavior for the reference recipe,
not necessarily a numerical failure.

The project also supports `cupy_cg`, `torch_block_cg`, and `cholmod`.
The GPU solvers are batched implementations of the same class-wise systems.
CHOLMOD is an exact-solve opt-in and ignores the CG tolerance and iteration
limit. Because `0 < alpha < 1` and the graph is symmetric nonnegative, the
system is symmetric positive definite. An exact solve can still yield different
post-normalization scores from the deliberately truncated reference CG solve.

### 5. Probabilities and certainty

Small negative values from a finite CG iterate are clamped to zero. Every row
with positive class mass is then L1-normalized:

```text
p_ij = max(Z_ij, 0) / sum_c max(Z_ic, 0)
```

The prediction is `argmax_j p_ij`. Certainty follows paper equation (12):

```text
omega_i = 1 - H(p_i) / log(C)
```

As in the reference implementation, all certainties are divided by the global
maximum and true-labeled examples are finally assigned certainty one. If every
propagated row is uniform, this project returns zero certainty rather than the
reference code's `0 / 0` result.

Rows inside a seed-reachable component that nevertheless have zero positive
mass after truncated CG are also marked unpropagatable and omitted.

## Training integration

### Refresh schedule

`update_mode="every_epoch"` rebuilds embeddings, the graph, pseudo-labels, and
certainties before each post-warm-up training epoch. `warmup_epochs` trains only
on the labeled support first. This represents the two phases in the paper, but
the configured warm-up duration is project-controlled; most current Iscen
configs set it to zero or tune only a short warm-up rather than reproducing the
paper's long supervised first stage.

`update_mode="once"` pseudo-labels once and reuses the result. In in-batch mode,
`update_mode` is retained only for config compatibility: a new local graph is
built every optimization step.

`update_mode="every_n_samples"` with `update_interval_samples` schedules the
whole refresh -- embeddings, graph, pseudo-labels, certainties, and the loader
those labels imply -- on consumed training samples instead of epoch boundaries,
so the cadence means the same amount of training on a small dataset and on
semi-iNat. It is off unless the config selects it; see
[the LRML notes](lrml.md#sample-scoped-refreshes) for how the interval is
converted and which combinations are rejected.

### Filtering and sampling

The paper trains on every pseudo-labeled example and attenuates its loss by
`omega_i`. This project can instead reject examples below
`confidence_threshold`. A two-stream M-per-class batch may also rescue the best
otherwise-rejected classes down to
`pseudo_label_rescue_confidence_floor` so that the sampler has sufficient
class capacity.

With `labeled_batch_size` set, a batch has a fixed true-labeled portion and a
fixed originally-unlabeled portion. Both portions must be divisible by
`sampler_m`, and each stream always contributes exactly `sampler_m` samples per
class it draws. Without an explicit `length_before_new_iter`, the pseudo-labeled
stream defines the epoch length; the labeled stream repeats as needed.

`class_overlap` decides how the two streams' class selections relate.

| | `independent` (default) | `disjoint` |
| --- | --- | --- |
| class selection | each stream draws from its own labels, independently | coordinated; a class belongs to at most one stream per batch |
| samples per class | `sampler_m` per stream, so `2 * sampler_m` for a shared class | exactly `sampler_m` across the whole batch |
| distinct classes per batch | between `unlabeled_batch_size / sampler_m` and `batch_size / sampler_m` | exactly `batch_size / sampler_m` |
| cross-stream positive pairs | whenever the streams share a class | none, ever |
| class-capacity requirement | each stream's own quota only | each stream's own quota, plus `batch_size / sampler_m` distinct classes across the union |

`independent` is the default and matches LP-DeepSSL, whose two-stream sampler
(inherited from mean-teacher) draws the streams with no class coordination at
all and handles class balance through the loss weights `zeta_c` instead. Shared
classes produce the cross-stream positive pairs that anchor pseudo-labels to
true labels. None of the configured metric losses require exactly `sampler_m`
samples per class, so the `2 * sampler_m` groups are harmless.

It is also the behavior of every run made before `class_overlap` existed, and it
is byte-for-byte that implementation: the label array keeps the grouping's
first-appearance order, classes come from `permutation` rather than `choice`,
and each stream draws its indices before the next stream picks its classes. A
seed therefore replays historical batches exactly.

`disjoint` makes the combined batch globally M-per-class, but nothing
downstream consumes that invariant: the training loop hands the loss one flat
mixed batch. Its measurable cost is that a pseudo-labeled sample never shares a
batch with a true-labeled sample of the same class, so the metric loss can only
push the streams apart and never pulls a pseudo-labeled sample onto its class's
labeled exemplars. It also anti-correlates supervision with pseudo-label
coverage: once thresholding leaves fewer pseudo classes than labeled classes,
the labeled stream must prefer classes that have no pseudo-labels. Measured over
2000 batches at `batch_size=128`, `sampler_m=4`, `labeled_batch_size=32`, 100
classes with 40 surviving pseudo classes, a class with pseudo-labels appeared in
the true-labeled stream 0.39x as often as a labeled-only class, against 0.99x
under `independent`.

The union class requirement exists only for `disjoint`, which is also why
`independent` puts less pressure on `pseudo_label_rescue_confidence_floor` to
admit low-confidence pseudo-labels purely to satisfy sampler capacity.

### Loss weighting differs from the paper

The paper's equation (13) is a certainty- and class-weighted cross-entropy:

```text
labeled:        zeta_y * CE(x, y)
pseudo-labeled: omega_i * zeta_yhat * CE(x, yhat)
zeta_c proportional to 1 / pseudo_class_population(c)
```

Here, `omega_i` is stored in `RelabeledSubset`, but a loss receives it only if
the loss declares `supports_sample_weights`. At the time of this audit, the
project-local `MixedLabelPropagationProxyLoss` is the only such metric loss.
The commonly configured `ArcFaceLoss`, `CircleLoss`, `MultiSimilarityLoss`,
`NTXentLoss`, and `ProxyAnchorLoss` therefore use certainty for filtering and
diagnostics only; accepted pseudo-labels have full loss strength.

M-per-class sampling gives classes a more balanced sampling frequency and is
the project's analogue of `zeta_c`, but it is not mathematically equivalent to
multiplying every example loss by the paper's inverse class-population weight.
Pair/triplet interactions also make a metric loss structurally different from
the paper's per-example cross-entropy.

## Optional in-batch variant

`graph_batch_mode="in_batch"` is not part of LP-DeepSSL. It samples labeled and
unlabeled nodes, builds a CPU FAISS graph from detached current-batch
embeddings, propagates labels locally, and adds a second metric-learning loss
over the labeled graph nodes plus accepted pseudo-labeled nodes. The main
supervised loss is still evaluated separately, so labeled examples contribute
to both terms.

This path is best understood as an experimental local-graph regularizer, not a
memory-reduced implementation of the paper's whole-dataset algorithm. The main
supervised objective and this auxiliary objective still give the selected
labeled graph nodes two loss contributions; that is the current regularizer
design rather than an exact LP-DeepSSL objective.

## Audit findings

### Resolved: in-batch classification losses crashed after accepting a pseudo-label

The original `InBatchGraphPseudoLabelRegularizer.compute_loss` implementation
left the `supervised_is_classification` branch empty. The following finite
check therefore read an unbound `loss` after accepting a pseudo-label.

```text
UnboundLocalError: cannot access local variable 'loss' where it is not associated with a value
```

The branch now calls the same float32 classification-loss helper as the main
trainer and the STML in-batch path, including sample weights when the objective
supports them. A regression test exercises accepted pseudo-labels and a full
backward pass through this branch.

### Resolved: saving in-batch graph diagnostics changed training

Previously, `_run_graph_pseudo_label_batch` compacted the solve to classes
represented by the local labeled nodes only when `graph_diagnostics is None`.
The first saved batch of each epoch instead solved over the dataset-wide class
count.

The extra zero-RHS columns normally leave `argmax` unchanged, but they change
the entropy denominator from `log(C_local)` to `log(C_global)`, and therefore
change certainty, threshold acceptance, and possibly the loss.

Local class compaction is now unconditional. Diagnostic propagation receives a
column-to-global-label lookup, so artifacts report global mapped class IDs while
the actual solve, entropy, confidence threshold, and loss remain identical with
diagnostics enabled or disabled. Both invariance and diagnostic label mapping
have regression tests.

### Intentional difference: hard confidence filtering instead of certainty weighting

Pseudo-label confidence is used as a binary acceptance threshold. Once
accepted, a pseudo-labeled sample receives the same weight as other accepted
samples unless the selected objective explicitly supports sample weights. This
is coherent for the intended hard-filtering protocol, but differs from
LP-DeepSSL equation (13), where every sample's loss is scaled continuously by
its certainty.

Results should therefore be described as an Iscen label-propagation adaptation
with hard confidence filtering rather than an exact reproduction of the
LP-DeepSSL training objective. With `confidence_threshold=0`, a propagated
zero-certainty prediction is accepted; use a positive threshold if that is not
desired.

### High experiment risk: configs named `iscen` are not reference configs

The Python registry defaults are correct, but the commonly referenced JSON
files override them:

| Config | Important effective behavior |
| --- | --- |
| `configs/ssl_iscen_label_spreading.json` | Reference graph values except `cg_max_iter=100`; no supervised warm-up; threshold `0.0`. |
| `configs/ssl_methods/iscen.json` | `k=10`, `gamma=1`, `alpha=0.2`, `rtol=1e-5`, `max_iter=2000`, `torch_block_cg`, threshold `0.8`, no warm-up. |
| `configs/ssl_label_spreading.json` | Still selects `iscen_label_spreading`, but uses `k=5`, `alpha~=0.595`, `max_iter=2000`, `torch_block_cg`, threshold `0.8`, no warm-up. |
| `configs/ssl_iscen_label_spreading_in_batch.json` | Experimental in-batch mode with `k=10`, `gamma=1`, `alpha=0.2`, and threshold `0.8`. |

These may be legitimate tuned variants, but comparisons must not describe them
as the paper/reference recipe. In particular, `alpha=0.2` produces a much
better-conditioned and qualitatively less diffusive system than `alpha=0.99`.

There is also stale generated experiment state: the Phase-1 Cars Iscen files
for `nt`, `ms`, `cir`, and `prx` currently contain the ArcFace loss, ArcFace HPO
path, and ArcFace save directory; the In-Shop files similarly all contain the
ProxyAnchor setup. Run manifests may override the loss argument, but using
those experiment files directly does not do what their filenames say. The
generator's read-only `--check` currently fails before comparison because its
expected Phase-0 template files are missing, so it cannot catch this drift.

### Medium: closed-set class coverage is assumed but not validated

The method accepts a split where some dataset classes have no labeled seed.
Those class columns remain exactly zero, so examples from those classes must be
assigned to a seeded class. Because certainty still divides entropy by the
dataset-wide `log(C)`, using more global classes than seeded classes can also
make confidence look artificially high.

This is incompatible with `class_subset` or `class_subset_k_shot` when
`labeled_fraction < 1`. The current main k-shot configs use
`labeled_fraction=1`, so every class is seeded there. Validation should reject
unseeded-class protocols for Iscen or explicitly define them as a closed-set
evaluation over the seeded classes only.

### Medium: global propagation materializes dense `N x C` arrays

The graph and linear system are sparse, but seed, solution, probability, and
entropy inputs are dense over samples by classes. This is expected for this
form of multi-class propagation, but it becomes expensive on metric-learning
datasets with thousands of classes. Compacting globally to actually seeded
classes would at least avoid zero right-hand sides in partial-class protocols.

### Low: `embedding_num_workers` is ignored by the whole-pool Iscen adapter

`IscenLabelSpreadingPseudoLabeler.generate_pseudo_labels` passes
`num_workers=0` to `extract_embeddings` instead of
`config.embedding_num_workers`. For example,
`configs/ssl_iscen_label_spreading.json` requests eight workers but always runs
with zero. This is an operational/performance bug and makes the config field
misleading; it should be forwarded like the other graph pseudo-labelers do.

### Low: shared-helper names leak into logs and errors

Iscen reuses `make_mixed_label_affinity` and
`normalize_label_spreading_rows`. As a result, FAISS timing/error text can say
`mixed label propagation` or `faiss_label_spreading` during an Iscen run. This
does not change results, but it makes incident diagnosis unnecessarily
confusing.

### Intentional differences that are not bugs

- Positive-part clipping follows paper equation (10), even though the public
  repository does not explicitly clip before cubing.
- Per-class unit seed mass follows the public repository, even though the
  paper's displayed seed matrix is ordinary one-hot.
- Keeping the final 20-step CG iterate follows the public repository.
- Omitting seedless components and zero-mass rows is a safe extension.
- The global maximum normalization of certainty follows the reference recipe,
  although it makes a fixed threshold relative to the current graph rather
  than calibrated across datasets/epochs. In-batch mode makes it relative to
  each local batch, which is another reason not to interpret that variant as
  the paper method.

## Reference-style configuration

The graph/refresh portion of a reference-style run should contain at least:

```json
{
  "method": "iscen_label_spreading",
  "update_mode": "every_epoch",
  "warmup_epochs": 5,
  "confidence_threshold": 0.0,
  "graph_batch_mode": "global",
  "labeled_batch_size": null,
  "class_overlap": "independent",
  "method_params": {
    "n_neighbors": 50,
    "gamma": 3.0,
    "alpha": 0.99,
    "cg_rtol": 1e-6,
    "cg_max_iter": 20,
    "linear_solver": "cg"
  }
}
```

The value `5` above is only an example project warm-up; the paper used a full
supervised first stage. Reproducing the complete paper additionally requires a
certainty-weighted per-example classification objective, inverse
pseudo-class-population weights, the paper's labeled/unlabeled batch quotas,
and one pass through all originally unlabeled examples per epoch. The current
metric-learning integration does not provide that exact combination.

## Verification and remaining regression coverage

The focused implementation and graph-diagnostic suites pass under `unittest`
(19 tests total):

```text
tests.test_iscen_label_spreading
tests.test_graph_diagnostics
```

They cover the normalized system, class-balanced seeds, truncated CG, CHOLMOD
dispatch, negative-score clamping, seedless components, result order, the
in-batch classification backward path, diagnostics invariance, and compact
diagnostic class mapping. Remaining gaps include:

- forwarding `embedding_num_workers` through the Iscen adapter;
- rejection or explicit handling of classes with no labeled seed;
- whether every intended training objective actually consumes certainty.

## Primary references

- [Iscen et al., CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html)
- [LP-DeepSSL reference repository](https://github.com/ahmetius/LP-DeepSSL)
- [Reference graph, propagation, certainty, and class weights](https://github.com/ahmetius/LP-DeepSSL/blob/master/lp/db_semisuper.py)
- [Reference second-stage epoch loop](https://github.com/ahmetius/LP-DeepSSL/blob/master/train_stage2.py)
