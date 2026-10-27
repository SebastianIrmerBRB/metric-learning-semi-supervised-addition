# SLADE in the paper vs. SLADE in this repository

This document compares the method described in Duan et al., *SLADE: A
Self-Training Framework for Distance Metric Learning* (CVPR 2021) with the
implementation in this repository.

## Bottom line

The repository is a faithful implementation of SLADE's **student-side core**:
k-means pseudo-labels, feature-basis learning, Gaussian similarity statistics,
confidence-based pair mining, and the three-term objective from Equation 9 are
all present.

It is **not an out-of-the-box reproduction of the paper's experiments**. The
largest differences are outside Equations 2--9:

- the paper fine-tunes a SwAV-initialized ResNet-50 end to end, while this
  project normally trains a projection head over a frozen DINOv2 ViT;
- the paper uses fully labeled CUB/Cars/In-shop training data plus a separate
  related unlabeled dataset, while the shipped SLADE configs normally use a
  16-shot split and may draw unlabeled images from the same dataset;
- the default repository schedule continues one model within a fold and
  periodically refreshes its clusters. An opt-in teacher/student mode now
  creates a supervised-teacher and joint-student stage independently in every
  fold; a second, separately named option reproduces the paper's unusual
  selected-student handoff between folds;
- the paper evaluates four 128-dimensional folds as one concatenated
  512-dimensional embedding. The repository can now do this, but the shipped
  SLADE experiments are not configured that way.

Consequently, `configs/runs/ssl/slade/paper.json` should be read as **paper
method defaults** (`beta`, `lambda1`, and `lambda2`), not as a complete paper
reproduction config.

## Sources and scope

| Role | Source |
| --- | --- |
| Paper | Jiali Duan, Yen-Liang Lin, Son Tran, Larry S. Davis, and C.-C. Jay Kuo. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_SLADE_A_Self-Training_Framework_for_Distance_Metric_Learning_CVPR_2021_paper.pdf), [arXiv:2011.10269](https://arxiv.org/abs/2011.10269) |
| Core implementation | [`training/ssl/slade.py`](../training/ssl/slade.py) |
| Method configs | [`configs/runs/ssl/slade/`](../configs/runs/ssl/slade/) and [`configs/ssl_methods/slade.json`](../configs/ssl_methods/slade.json) |
| Dataset composition | [`training/engine.py`](../training/engine.py), [`utils/dataset_composition.py`](../utils/dataset_composition.py), and [`utils/local_datasets.py`](../utils/local_datasets.py) |
| Cross-validation evaluation | [`training/engine.py`](../training/engine.py) and [`docs/cross_validation.md`](cross_validation.md) |

The paper is the normative reference here. Several implementation details are
not stated in it, including the margin and variance coefficient in Equation 5,
the initialization of the running Gaussian statistics, exact k-means settings,
and the exact number of student/teacher iterations. Those are identified below
as repository choices rather than treated as paper mismatches.

## At-a-glance comparison

| Aspect | Paper SLADE | This repository | Assessment |
| --- | --- | --- | --- |
| Pseudo-labels | Teacher embeddings clustered with k-means; cluster IDs are pseudo-labels (Section 3.2) | `_refresh_pseudo_labels` plus `kmeans_cluster_labels` | Core match |
| Feature basis | Learn `W_a`; represent an embedding as `r = W_a f` (Section 3.3.1) | `SladeBasisLayer` | Match |
| Basis loss | `L_Basis = L_CE + L_SD` (Equations 2--6) | Cross-entropy plus `similarity_distribution_loss` | Match, with explicit numerical safeguards |
| Pair mining | `s = cos(W_a f_i, W_a f_j)`; positive if `s >= mu+`, negative if `s <= mu-` (Equation 7) | `pairwise_similarity`, `mining_thresholds`, and `_mined_pairs` | Match, with an added separation guard |
| Final objective | `L_rank(D_l) + lambda1 L_rank(D_u) + lambda2 L_Basis` (Equation 9) | Engine supervised loss plus `SladeRegularizer.compute_loss` and `combine_losses` | Match |
| Published weights | `beta = 0.99`, `lambda1 = 1`, `lambda2 = 0.25` | Same in `paper.json`, but `paper.json` also sets `grad_norm_alpha`, which makes both lambdas *learned* and leaves the published values as starting points only | Starting values match; the fixed weights do not |
| Ranking loss | Same pair-based loss on labeled and mined-unlabeled pairs; paper reports contrastive and multi-similarity | `unlabeled_ranking_loss: "supervised"` (default) reuses the configured pair loss; `"contrastive"` gives the unlabeled term its own Eq 1 loss | Default matches; `"contrastive"` decouples the two streams |
| Teacher/student lifecycle | Fine-tune teacher, train a distinct student, promote student, repeat a few times | Default: warm up and periodically re-cluster one continuing optimization. `within_fold_teacher_student`: cluster once, retain teacher weights, reset optimization, and select only student epochs | Default differs; opt-in has an explicit stage boundary |
| Cross-fold lifecycle | The trained student becomes the teacher for the next fold (Section 4.3) | Independent by default; `cross_fold_teacher_handoff` promotes only the selected projection head and resets the basis/optimizer | Opt-in match for a frozen backbone |
| Initialization/backbone | SwAV-initialized ResNet-50, including an unlabeled union fine-tuning stage; end-to-end tuning | DINOv2 ViT-B/14 by default; backbone frozen by default; no dataset-specific self-supervised stage | Different |
| Labeled/unlabeled data | Fully labeled benchmark training split plus external NABirds, CompCars, or Fashion200k | Internal split by default; external pools are opt-in; shipped configs use at most 16 labels per class | Different by default |
| Number of clusters | NABirds ablation selects `k = 400`; other exact choices are not fully documented | Defaults to the number of labeled classes unless `num_clusters` (absolute) or `cluster_ratio` (a multiple of the class count) is set | Repository choice; unsafe default for an external pool |
| Batch sizes | 32 labeled and 32 unlabeled | Equal sizes when `unlabeled_ratio = 1`, but the absolute main batch size comes from the experiment; `unlabeled_batch_size` pins the unlabeled stream independently | Matches the unlabeled side at `unlabeled_batch_size: 32`, the labeled side only when main `batch_size = 32` |
| Fold embeddings | Four 128-d embeddings concatenated to 512 dimensions | Supported when four fold test embeddings are saved; shipped SLADE configs use other dimensions/settings | Capability matches; configs do not |
| Metrics | CUB/Cars: MAP@R, R-precision, P@1; In-shop: Recall@K | MAP@R and P@1; optional Recall@K; no reported R-precision | Partially matches |

## 1. Core method mapping

### 1.1 Teacher clustering and pseudo-labels

The paper's teacher embeds the unlabeled pool, k-means assigns every image to a
cluster, and the cluster ID becomes its pseudo-label. The repository follows
that flow:

1. `_refresh_pseudo_labels` extracts the current model's retrieval embeddings
   for every unlabeled position.
2. `kmeans_cluster_labels` runs FAISS k-means when available and otherwise uses
   scikit-learn.
3. Cluster IDs are stored by training-dataset position and reused until the next
   refresh.

The code exposes k-means iterations, restarts, backend, and seed. The paper does
not publish those settings, so their defaults (`25` iterations and `10`
scikit-learn initializations, with a backend-specific FAISS mapping) are local
implementation choices.

### 1.2 Feature-basis learning, Equations 2--6

The following equation-to-code mapping is direct:

| Paper | Repository |
| --- | --- |
| `r = W_a f` | `SladeBasisLayer.forward` |
| Equation 3: cross-entropy over basis responses for labeled samples | `F.cross_entropy(basis.logits(...), supervised_labels)` |
| Equation 4: `cos(W_a f_i, W_a f_j)` | `SladeBasisLayer.pairwise_similarity` |
| Equation 5: `max(mu- - mu+ + m, 0) + lambda (v+ + v-)` | `similarity_distribution_loss` |
| Equation 6: `(1 - beta) * batch + beta * running` | `SladeBasisLayer.update_distributions` |

There is one basis row per labeled training class. During
`basis_warmup_steps`, the retrieval embeddings are detached, so Equation 2
updates only `W_a`. Afterward, both the student embedding and the basis are
trained end to end, matching the paper's stated warm-up sequence.

The implementation deliberately computes the basis projection and similarity
statistics in float32. That changes numerical precision, not the mathematical
objective.

### 1.3 Mining and the ranking loss, Equations 7 and 9

Once both running distributions exist and `mu+ > mu-`, the repository uses
exactly the paper's thresholds:

```text
positive pair: s(i, j) >= mu+
negative pair: s(i, j) <= mu-
```

The selected pair indices are detached from the basis similarity and the
ranking loss is evaluated on the retrieval embeddings `f`, not on `W_a f`.
This matches the role of the basis in the paper: it selects trustworthy pairs
but is not the test-time embedding.

For `unlabeled_ranking_loss: "supervised"`, the mined pairs are passed directly
to the configured pair-based loss. The code emits each unordered pair in both
anchor directions because pytorch-metric-learning reduces many losses per
anchor. This is what Equation 9 literally writes: one `L_rank` on both streams.

For `unlabeled_ranking_loss: "contrastive"`, the same mined pairs go to SLADE's
own Equation 1 loss instead:

```text
L_rank = sum_{(i,j) in P} max(d(i, j) - m_pos, 0)
       + sum_{(i,j) in N} max(m_neg - d(i, j), 0)
```

That is exactly `ContrastiveLoss(pos_margin=m_pos, neg_margin=m_neg)` over the
library's default `LpDistance` -- Euclidean distance on L2-normalized
embeddings, the metric Equation 1's `d` refers to. Using the library loss keeps
the unlabeled term's reduction identical to the labeled term's, which is what
makes the paper's `lambda1 = 1` mean what Section 3.3.3 says it means
("empirically chosen to make the magnitudes of the losses similar in scale").
A literal sum over mined pairs weighed against a mean-reduced labeled term would
not be on a similar scale.

Equation 1 is the paper's own worked example of `L_rank` ("for example, a
constrastive loss [12]") and Table 1 reports SLADE with it, so this mode is a
re-parameterization of Equation 9 rather than a new objective. What it changes
is that the two streams no longer share one loss. That is the point: because the
supervised criterion is no longer applied to Equation 7's mined pairs, it is
free to be a proxy or classification loss. See section 3 for the constraint this
lifts.

The final composition is also direct:

```text
engine supervised term                       = L_rank(D_l)
regularizer_weight * regularizer loss        = lambda1 * L_rank(D_u)
lambda_basis * basis loss                    = lambda2 * L_Basis
```

Keeping the two SLADE additions separate also lets the repository report or
optionally calibrate their gradient contributions independently. With no
calibration configured, `paper.json` retains the paper's fixed weights.

## 2. Material differences

### 2.1 Initialization, architecture, and trainable parameters

The paper uses a ResNet-50. Its preferred teacher initialization is SwAV, then
the self-supervised model is further fine-tuned on the union of labeled and
unlabeled images without labels. The teacher is subsequently fine-tuned on the
labeled data with the ranking loss. During student training, the network and
basis vectors are optimized end to end.

This project defaults to a self-supervised DINOv2 ViT-B/14 whose backbone is
frozen. It has no self-supervised fine-tuning stage on the current labeled and
unlabeled pool. Normally the trainable model components are the projection head
and `W_a`, plus any parameters owned by the configured supervised loss. Full or
last-block DINO tuning can be requested, but that still does not reproduce the
paper's ResNet-50/SwAV initialization or its union fine-tuning stage.

There is therefore **no directly comparable row in the paper's Table 1**:

- the ImageNet rows use a supervised ImageNet ResNet-50 that is fine-tuned end
  to end;
- the SwAV rows add dataset-specific self-supervised fine-tuning;
- this repository normally uses a different, stronger self-supervised model,
  training corpus, architecture, and tuning policy.

Any comparison with the paper's reported MAP@R or P@1 is an experimental
comparison between systems, not an implementation-reproduction check.

### 2.2 Teacher/student and iterative self-training schedule

The paper describes stage-wise self-training:

```text
fine-tuned teacher -> fixed pseudo-labels -> train student -> promote student
                                                     -> repeat a few times
```

It additionally says that, during four-fold training, the student trained in
one fold becomes the teacher for the next fold.

The default repository schedule uses a single continuing model inside each fold:

```text
labeled-only warm-up -> cluster current model -> train same model
                                  -> re-cluster every N epochs -> continue
```

No separate teacher module is needed between refreshes because the teacher is
not queried after its cluster IDs have been produced. Nevertheless, the overall
optimization schedule is different from the paper:

- `warmup_epochs` is a finite labeled-only warm-up, not the paper's complete
  teacher pre-training and fine-tuning pipeline;
- `update_interval_epochs` approximates the duration of one self-training
  iteration, but does not create or reinitialize a distinct student;
- the paper does not publish an epoch-equivalent refresh interval;
- repository cross-validation folds are initialized and trained independently,
  so there is no cross-fold student-to-teacher hand-off.

The paper gives the models separate symbols (`theta_t` and `theta_s`) but does
not state clearly whether a new student is initialized from its teacher or from
a common pretrained checkpoint. The repository's explicit mode makes a
conservative choice: the student copies the teacher embedding weights, but gets
fresh optimizer and learning-rate-scheduler state.

#### Opt-in independent teacher/student lifecycle

[`paper_teacher_student.json`](../configs/runs/ssl/slade/paper_teacher_student.json)
creates the non-leaky version independently in every fold:

1. The fold starts with a fresh projection head.
2. Labeled-only warm-up fine-tunes that head as the fold's teacher.
3. The teacher clusters the unlabeled pool exactly once.
4. The same projection weights initialize the student, which trains against
   fixed pseudo-labels for the remainder of the fold.
5. The model optimizer and learning-rate scheduler restart at the boundary, and
   validation checkpoint selection/HPO reporting restart so a warm-up teacher
   cannot be returned or optimized as the nominal student.
6. `W_a` and its Gaussian buffers are student-local. They are fresh at model
   construction and receive no gradients during labeled-only teacher training.

A second live DINO model would add memory but no information: the backbone is
immutable and the teacher is not queried after clustering. The saved projection
head plus the shared frozen backbone is the complete teacher snapshot; the
fixed cluster-ID array is all student training subsequently consumes.

#### Optional cross-fold handoff

[`paper_cross_fold_teacher_student.json`](../configs/runs/ssl/slade/paper_cross_fold_teacher_student.json)
adds `cross_fold_teacher_handoff: true` to the lifecycle above. The selected
student projection from fold `i` initializes fold `i + 1`; the new fold still
gets fresh basis/Gaussian state and optimizer state.

Only this second mode makes the folds sequential. A later fold's validation
classes may already have appeared in an earlier fold's labeled training set, so
the fold metrics are order-dependent and their mean is not an independent
cross-validation estimate. Use this to reproduce the paper's training protocol
or build its concatenated ensemble, not for unbiased HPO/model selection. The
independent `paper_teacher_student.json` mode does not have that problem.

### 2.3 Data protocol and label budget

The paper uses all available labels in the benchmark's training classes and
adds a related external unlabeled dataset:

| Labeled benchmark | External unlabeled data in the paper |
| --- | --- |
| CUB-200-2011 | NABirds, approximately 48k images |
| Cars-196 | Filtered CompCars, 16,537 images in 145 model classes |
| In-shop | Fashion200k |

The repository supports `unlabeled_source` values `split`, `external`, and
`split_and_external`. It also includes a metadata-aware NABirds loader and a
reconstructed `compcars_slade_paper` filter; see
[`slade_compcars.md`](slade_compcars.md). A generic recursive external pool can
be used for Fashion200k. A run pointing at `data/Fashion200k` without images
downloads it from the Kaggle mirror `mayukh18/fashion200k-dataset` (6.1 GB),
since the authors' Google Drive no longer holds the images.

Those external pools are not selected by `paper.json` itself. Without
experiment-level overrides, `unlabeled_source` defaults to `split`, and the
config's `class_subset_k_shot` plus `labeled_per_class: 16` exposes only 16
ground-truth examples per training class. The remaining in-dataset examples
become the unlabeled pool. That is a useful semi-supervised benchmark, but it is
not the paper's external-data protocol.

The provided
[`configs/experiments/slade_cars196_compcars.json`](../configs/experiments/slade_cars196_compcars.json)
does select the paper's Cars/CompCars pairing, but it still caps Cars labels at
16 per class, uses one fold, and uses batch size 256. It reproduces the dataset
pairing and filtered CompCars counts, not the complete experiment.

### 2.4 Cluster count

Section 4.5.4 reports a NABirds sweep over `k = 100, 200, 300, 400, 500`, with
the best MAP@R at 400. The authors note that NABirds has 400 species and that
the result is not very sensitive in that range. The paper does not give a full
per-dataset k-means configuration table.

This repository defaults `num_clusters` to the number of labeled classes. That
is reasonable when the unlabeled pool is the held-out portion of the same
dataset, but it is not a safe paper-protocol default for external data. Set it
explicitly for external pools; `400` follows the paper's NABirds ablation, while
`145` for filtered CompCars follows the paper's documented number of retained
model classes.

`cluster_ratio` expresses the same k as a multiple of the labeled class count C
(`0.5` is `k = C/2`), which is the form that survives a change of dataset. It is
the one to search: the paper's finding is that k near C works and the result is
insensitive nearby, and that is a statement about the ratio, not about the
absolute count 400. It does not, however, replace an explicit `num_clusters` for
an external pool, where the right k is the *pool's* class count and has no fixed
relationship to C.

### 2.5 Additional objective choices and safeguards

The code must resolve several details that the paper leaves open:

| Detail | Paper | Repository choice |
| --- | --- | --- |
| Equation 5 margin `m` | Not reported | `sd_margin = 0.5` by default |
| Equation 5 variance coefficient `lambda` | Not reported | `sd_variance_weight = 1.0` by default |
| Basis warm-up duration | "a few iterations" | `basis_warmup_steps = 100` by default |
| Gaussian initialization | Not reported | First observed batch seeds each distribution independently |
| Pair sampling for Equation 5 | Says a set of pseudo-positive and pseudo-negative pairs is sampled; exact sampler is not given | All distinct within-batch pairs are used |
| Threshold validity | Directly sets thresholds from `mu+` and `mu-` | Mining waits until both distributions exist and `mu+ > mu-`. Kept because pytorch-metric-learning rejects a pair that is both positive and negative, which is what an overlapping Equation 7 produces |
| Equation 5's gradient into `W_a` | Not discussed; Equation 6 is given as a value update | `sd_gradient_mode` — `"full"` (default) keeps beta out of the gradient scale, `"damped"` is the literal composition |
| State carried across a self-training iteration | Not discussed | `reset_basis_on_refresh` and `reset_basis_warmup_on_refresh`, both off by default |
| Basis logit scale | Not reported | Configurable, default `1.0` |
| Equation 1 margins `m_pos`/`m_neg` | Not reported | `contrastive_pos_margin = 0.0` and `contrastive_neg_margin = 1.0`, the library's standard contrastive setting. Only read when `unlabeled_ranking_loss: "contrastive"`; the regularizer requires `m_neg > m_pos` so Equation 1 has a satisfiable band |
| Final ragged unlabeled batch | Not discussed | Dropped to avoid unstable batch statistics |

The last three behaviors are stabilizers around underspecified cases. They are
not equations claimed by the paper.

### 2.6 Batch, folds, embedding dimensions, and metrics

The paper reports:

- batch size 32 for labeled data and 32 for unlabeled data;
- four-fold cross-validation;
- a 128-dimensional student embedding per fold;
- concatenation of the four normalized fold embeddings into one
  512-dimensional test embedding;
- MAP@R, R-precision, and P@1 for CUB/Cars, and Recall@K for In-shop.

In this repository, `unlabeled_ratio: 1.0` makes the two streams equal in size,
but does not set the absolute size. The command or experiment config controls
`batch_size`: the CLI default is 16, `slade_cars.json` uses 32, and the external
Cars/CompCars example uses 256. `unlabeled_batch_size` sets the unlabeled stream
absolutely instead, so it stays fixed while `batch_size` varies -- which is what
lets an HPO study search the two stream sizes independently rather than tying the
unlabeled one to whichever `batch_sampler` a trial drew.

The evaluation engine supports the paper's concatenated-fold construction. A
four-fold run with `feat_dim: 128`, test evaluation, and saved fold embeddings
produces both per-fold metrics and the L2-normalized 512-dimensional
concatenated result. `final_fit_mode: "cross_validation_folds"` enables the
required fold test embeddings automatically. The shipped SLADE experiment
configs do not combine all of those settings: for example,
`configs/runs/experiments/slade_cars.json` uses four 768-dimensional folds and
does not enable final test evaluation.

The evaluator reports P@1 and MAP@R and can report requested Recall@K values.
Although the underlying metric library contains R-precision machinery, this
project does not currently expose R-precision in its result objects or reports,
so the paper's RP column cannot be reproduced directly.

## 3. Repository extensions beyond the paper

These options are useful, but they should not be attributed to Duan et al.:

- FAISS and scikit-learn k-means backends, configurable iterations and restarts,
  and a GPU FAISS index selected by problem size (`kmeans_gpu`). The paper does
  not discuss k-means cost; at In-Shop scale it is 61 s on a single CPU thread
  against 0.72 s on a GPU, for the same objective;
- fixed paper weights, per-term target-ratio calibration, or three-term
  GradNorm balancing;
- frozen-feature precomputation and caching;
- separate SSL compute-device selection;
- pseudo-label, graph, per-batch loss, and gradient-contribution diagnostics;
- internal, external, or combined unlabeled pools;
- explicit validation of loss compatibility and pair construction.

Proxy and classification losses such as ArcFace and ProxyAnchor cannot consume
the mined pair tuple. Under the default `unlabeled_ranking_loss: "supervised"`,
which applies the configured loss to `D_u` as well, `validate_run_args`
therefore rejects every classification loss and every loss that cannot take a
mined `indices_tuple` before training starts.

`unlabeled_ranking_loss: "contrastive"` lifts that restriction. Equation 9's
unlabeled term then uses Equation 1 rather than the configured loss, so nothing
hands the supervised criterion a pair tuple and any supervised loss is
admissible. Two knock-on effects:

- the `sampler_m >= 2` requirement is also dropped for a classification loss.
  It exists so the pair-based labeled term sees in-batch positives, which a
  proxy loss scoring against learned class centers does not need;
- `contrastive_pos_margin` and `contrastive_neg_margin` are removed from an HPO
  search space whenever the resolved mode is not `"contrastive"`
  (`filter_slade_contrastive_spaces` in `training/hpo.py`). Every space is
  suggested on every trial, so leaving them in would give the sampler two
  dimensions that cannot move the objective.

Using one loss on both streams is what the paper does, so `"contrastive"` should
be read as a repository extension for pairing SLADE with a proxy objective, not
as the paper's configuration.

## 4. Configuration guide

| File | What it actually represents |
| --- | --- |
| [`configs/runs/ssl/slade/paper.json`](../configs/runs/ssl/slade/paper.json) | Core published values `beta = 0.99`, `lambda1 = 1`, and `lambda2 = 0.25`, with the configured pair loss on both streams. Its 16-shot label budget, five warm-up epochs, 25-epoch refresh interval, Equation 5 coefficients, and 100-step basis warm-up are repository settings. |
| [`configs/runs/ssl/slade/paper_contrastive.json`](../configs/runs/ssl/slade/paper_contrastive.json) | The same core values with `unlabeled_ranking_loss: "contrastive"`, so Equation 9's unlabeled term uses Equation 1 and the supervised loss may be a proxy/classification loss. A repository extension, not the paper's configuration. |
| [`configs/runs/ssl/slade/calibrated.json`](../configs/runs/ssl/slade/calibrated.json) | Repository extension that calibrates the ranking and basis contributions from measured gradients. |
| [`configs/runs/ssl/slade/paper_teacher_student.json`](../configs/runs/ssl/slade/paper_teacher_student.json) | Independent teacher/student lifecycle: supervised teacher, one fixed clustering, fresh student optimization and student-only model selection in every fold. |
| [`configs/runs/ssl/slade/paper_cross_fold_teacher_student.json`](../configs/runs/ssl/slade/paper_cross_fold_teacher_student.json) | The same lifecycle plus the paper's sequential selected-student projection handoff between folds. |
| [`configs/runs/slade/cars_teacher_student.json`](../configs/runs/slade/cars_teacher_student.json) | Cars HPO space for either explicit lifecycle; excludes zero teacher warm-up and does not search the inert refresh interval. |
| [`configs/ssl_methods/slade.json`](../configs/ssl_methods/slade.json) | Minimal validated example, currently equivalent to the method-level `paper.json` settings. |
| [`configs/experiments/slade_cars196_compcars.json`](../configs/experiments/slade_cars196_compcars.json) | Cars/CompCars external-pool example with strict checking of the paper's CompCars counts; not a complete paper reproduction. |
| [`configs/runs/experiments/slade_cars.json`](../configs/runs/experiments/slade_cars.json) | Primary frozen-DINOv2, 16-shot Cars experiment over an internal unlabeled split. |
| [`configs/runs/experiments/slade_cars_teacher_student.json`](../configs/runs/experiments/slade_cars_teacher_student.json) | Same frozen-DINOv2 Cars setup with independent teacher/student stages in each fold. |
| [`configs/runs/experiments/slade_cars_cross_fold_teacher_student.json`](../configs/runs/experiments/slade_cars_cross_fold_teacher_student.json) | Fixed-setting setup with the paper's additional sequential fold handoff; HPO is disabled because the fold scores are order-dependent. |

## 5. What a closer reproduction would require

A paper-level reproduction would need more than selecting `paper.json`:

1. Use all labeled images from the benchmark's training classes. With an
   external pool, use `per_class_min`, `labeled_fraction: 1.0`, and
   `labeled_per_class: null` rather than the shipped 16-shot setting.
2. Select the matching external pool: CUB/NABirds, Cars/filtered CompCars, or
   In-shop/Fashion200k. Do not mix the benchmark's leftover training images into
   that pool when reproducing the paper.
3. Set the external-pool cluster count explicitly (`400` for the reported
   NABirds setting; `145` is the natural CompCars reconstruction).
4. Use 32 labeled and 32 unlabeled images per step.
5. Use four 128-dimensional folds and enable concatenated fold test evaluation.
6. Add R-precision to reporting if reproducing every CUB/Cars table column.
7. Use `within_fold_teacher_student: true` with `update_mode: "once"` for the
   explicit teacher/student stage. Add `cross_fold_teacher_handoff: true` only
   for the paper's sequential fold handoff. Reproducing ResNet-50/SwAV initialization,
   union self-supervised fine-tuning, and end-to-end backbone optimization still
   requires implementation work; no current config can enable those pieces.
8. Resolve and record the paper's unspecified settings--especially Equation 5's
   margin and variance coefficient, k-means details, and the precise number and
   duration of self-training iterations--because the published PDF alone does
   not make them reproducible.

## Verdict

Use this implementation to study **the SLADE objective inside this project's
DINOv2 semi-supervised metric-learning framework**. Its equations and pair
selection mechanism closely track the paper. Do not use the paper's published
numbers as a direct baseline for the shipped configs: the representation model,
data protocol, label budget, self-training lifecycle, fold coupling, and default
evaluation setup are materially different.
