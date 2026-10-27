# STML: Implemented Method, Worked Example, and Differences to the Official Release

This document describes the Self-Taught Metric Learning (STML) implementation in this
repository, walks through one optimizer step with real numbers, and enumerates every
difference against the official code and the paper.

Companion documents: [stml.md](stml.md) (how to configure and run it),
[stml_threshold.md](stml_threshold.md) (the derived hard-pseudo-label variant).

---

## 0. Sources compared

| Role | Reference |
| --- | --- |
| Paper | Sungyeon Kim, Dongwon Kim, Minsu Cho, Suha Kwak. *Self-Taught Metric Learning without Labels*. CVPR 2022. [arXiv:2205.01903](https://arxiv.org/abs/2205.01903) |
| Official code | `github.com/sung-yeon-kim/STML-CVPR22`, commit `c39ff1e` (2023-04-22, latest on `main`). Relevant files: `code/loss.py`, `code/main.py`, `code/dataset/sampler.py`, `code/dataset/utils.py`, `code/net/resnet.py`, `code/utils.py` |
| This project | `losses/metric_losses.py`, `models/retrieval_model.py`, `utils/__init__.py`, `training/ssl/data.py`, `training/ssl/embeddings.py`, `training/ssl/config.py`, `training/semi_supervised.py`, `training/engine.py` |

Throughout, **"upstream"** means the official code at that commit, and **"the paper"**
means the equations and experimental protocol as written in the publication. These two
are *not* the same thing — Section 6 lists the places where upstream itself departs from
its own paper, and this project inherits those departures.

---

## 1. The method as published

STML learns an embedding without class labels by alternating between (a) *predicting*
class-equivalence relations with a momentum teacher and (b) *learning* the student from
those predicted relations as soft supervision.

### 1.1 Networks

* Student: a shared backbone `φ_s` with **two heads**
  * `f_s` — the **retrieval head**, low-dimensional, the model that is actually evaluated.
  * `g_s` — the **background head**, high-dimensional (paper: "set as the output size of the
    last pooling"), used for batch construction and as the self-distillation target.
* Teacher: backbone `φ_t` plus **only** a `g_t` head, L2-normalized, no gradients, updated
  by EMA of the student.

### 1.2 Step 1 — nearest-neighbor batch construction

Sample `q` queries at random, retrieve their `k−1` nearest neighbors in the *student* `g`
space, and take the union: the batch is `q·k` images. Neighbors are recomputed **every
epoch**. Each image is then augmented into `V` views (paper: `V = 2`), so the loss sees
`n = q·k·V` rows.

### 1.3 Step 2 — contextualized semantic similarity (the pseudo label)

All computed **inside the mini-batch** on the teacher's L2-normalized `g` embeddings
`z^t`.

**Pairwise similarity** (Eq. 5), a Gaussian kernel on teacher distance with bandwidth `σ`:

```
w^P_ij = exp( − ‖z^t_i − z^t_j‖² / σ )
```

**Contextual similarity** (Eqs. 6–9). With `N_k(i)` the `k` nearest neighbors of `i` in the
batch *including itself*, the `k`-reciprocal neighborhood is

```
R_k(i) = { j : j ∈ N_k(i) ∧ i ∈ N_k(j) }
```

An asymmetric Jaccard overlap of neighborhoods (Eq. 7):

```
w̃^C_ij = |R_k(i) ∩ R_k(j)| / |R_k(i)|   if j ∈ R_k(i),   else 0
```

Query expansion over the closest `k/2` neighbors of `i` (Eq. 8), then symmetrization
(Eq. 9):

```
ŵ^C_ij = (1/|N_{k/2}(i)|) · Σ_{h ∈ N_{k/2}(i)} w̃^C_hj
w^C_ij = ( ŵ^C_ij + ŵ^C_ji ) / 2
```

**Contextualized semantic similarity** (Eq. 10) — both terms live in `[0,1]`, so they are
averaged directly:

```
w_ij = ( w^P_ij + w^C_ij ) / 2
```

`w_ij` is read as *"the probability that `i` and `j` belong to the same class"*.

### 1.4 Step 3 — student objective

Relative distance (row-mean normalized) on a student head `h ∈ {f_s, g_s}`:

```
d^h_ij = ‖z^h_i − z^h_j‖₂ / ( (1/n) Σ_k ‖z^h_i − z^h_k‖₂ )
```

Relaxed contrastive loss (Eq. 1), which consumes soft relations directly:

```
L_RC(Z^h) = (1/n) Σ_i Σ_{j≠i}  w_ij (d^h_ij)²                 ← pull
          + (1/n) Σ_i Σ_{j≠i}  (1 − w_ij) [ δ − d^h_ij ]₊²    ← push
```

Self-distillation from the high-dimensional `g` to the retrieval head `f` (Eq. 2), with
`ψ` = softmax and the `g` branch detached:

```
L_KL = (1/n) Σ_i Σ_{j≠i} ψ(−d^g_ij) · log[ ψ(−d^g_ij) / ψ(−d^f_ij) ]
```

Total (Eq. 3):

```
L_STML = ½ [ L_RC(Z^f) + L_RC(Z^g) ] + L_KL
```

### 1.5 Step 4 — teacher EMA

```
θ_t ← m·θ_t + (1 − m)·θ_s
```

applied to backbone and `g` head; the `f` head is excluded because the teacher does not
have one.

### 1.6 Published hyperparameters

| Setting | σ | δ | m | k | Batch | Views |
| --- | --- | --- | --- | --- | --- | --- |
| CUB / Cars (unsupervised) | 3 | 1.0 | 0.999 | 10 | 120 imgs = 24 queries × 5 | 2 |
| SOP (unsupervised) | 3 | 0.9 | 0.9 | 4 | 120 imgs = 60 queries × 2 | 2 |
| Semi-supervised (CUB+NABirds, Cars+CompCars) | 2 | 1.0 | 0.9999 | 10 | as above | 2 |

Note the paper's own remark: *"k in the nearest neighbor batch construction and that in
Eq. (6) are identical."* With 2 views, a group of 5 images occupies 10 rows in the loss
matrix, so upstream sets `topk = num_neighbors × views = 5 × 2 = 10`. This project does
the same (`losses/metric_losses.py:164`).

### 1.7 The paper's semi-supervised protocol

Paper §4.2.1, verbatim: *"STML adopts an embedding model trained on a labeled dataset as
the initial teacher and student models and trains them using both labeled and unlabeled
data. Note that it computes and utilizes contextualized semantic similarities as
supervision for labeled data as well as unlabeled data."* SLADE's protocol is adopted
directly (CUB+NABirds, Cars+CompCars; MAP@R, RP, R@1), the backbone is ResNet50/512-d,
pre-trained on ImageNet or SwAV and fine-tuned with MS loss before STML starts.

> **This protocol has no released reference implementation.** The official repository
> only contains unsupervised metric learning (`code/main.py`) plus MoCo/BYOL/MeanShift
> baselines (`code/main_SSL.py`). There is no supervised warm-up, no labeled/unlabeled
> split, and no external-dataset loader anywhere in it. Every semi-supervised detail in
> this project is therefore reconstructed from the paper text, not ported from code.

---

## 2. The implementation in this repository

### 2.1 Two operating modes

| | Legacy / faithful | Composed regularizer |
| --- | --- | --- |
| Selector | `ssl_config.method = "stml"`, `loss = "STMLLoss"` | `method = "regularized"`, `method_params.regularizer = "stml"` |
| Registry | `LOSS_DRIVEN_METHODS` (`training/ssl/config.py:37`) | `REGULARIZER_REGISTRY` (`training/semi_supervised.py:3263`) |
| Objective after warm-up | `L_STML` **only** | `w_sup · L_supervised + w_reg · L_STML` |
| STML pool | labeled **+** unlabeled positions, all labels hidden (`training/semi_supervised.py:3728`) | unlabeled positions **only** (`training/semi_supervised.py:1390`) |
| Batches per step | one STML batch | one labeled batch **and** one STML batch |
| Corresponds to | the paper's semi-supervised recipe | a project-only variant, not in the paper |

The rest of this section describes the legacy mode; §2.8 covers where the regularizer
differs.

### 2.2 Phase schedule

```
epoch <  warmup_epochs :  supervised training on labeled split only
                          loss = warmup_loss (+ warmup_miner), e.g. MultiSimilarityLoss
epoch >= warmup_epochs :  STML phase
```

At the first STML epoch and only then, the teacher is created
(`initialize_stml_teacher_for_phase`, `training/engine.py:3829`, called from
`train_epoch`, `training/engine.py:1979`). During warm-up no teacher exists and no EMA
runs. The `embedding_g` head is not on the supervised forward path
(`project_features` uses only `fc`), so it receives no gradient during warm-up and stays
at its orthogonal initialization until STML begins.

### 2.3 Model — the two heads

`models/retrieval_model.py:305-338, 419-443`:

```python
self.fc          = nn.Linear(backbone_dim, feat_dim)      # f — retrieval head
self.embedding_g = nn.Linear(backbone_dim, stml_g_dim)    # g — background head
# both orthogonal-initialized with zero bias when stml=True

def project_stml_features(self, features):          # student: (g, f)
    retrieval = self.fc(features)
    if self.stml_normalize_student:
        retrieval = F.normalize(retrieval, p=2, dim=1)
    return self.embedding_g(features), retrieval

def project_stml_teacher_features(self, features):  # teacher: g only
    return self.embedding_g(features)
```

`stml_g_dim = None` (default) means `g` has the backbone's own width — DINOv2 gives
384/768/1024/1536 for `s/b/l/g` (`models/retrieval_model.py:24-29`). That matches the
paper's *"dimension of `g_t` and `g_s` set as the output size of the last pooling"*.

### 2.4 Data — hidden labels and multi-view augmentation

`training/ssl/data.py:93-110`:

```python
class UnlabeledSubset(Dataset):
    def __getitem__(self, index):
        position = int(self.positions[index])
        views = [self.dataset[position][0] for _ in range(self.num_views)]
        return (views[0] if self.num_views == 1 else views), UNLABELED_TARGET, position
```

Indexing the source dataset `num_views` times re-runs its stochastic train transform, so
the views are independent augmentations of the same image — the equivalent of upstream's
`MultiTransforms`. The third return value is the source-dataset position; it becomes the
**instance id** that marks views of the same image. The label is always `-1`, never the
ground truth.

### 2.5 Batch construction — the NN sampler

`utils/__init__.py:1365-1440`:

```python
class STMLNearestNeighborBatchSampler(Sampler):
    def _make_neighbor_indices(self, embeddings, chunk_size=512):
        # chunked torch.cdist, topk(largest=False) → each row is [self, 4 nearest]
    def __iter__(self):
        for _ in range(len(self)):
            q = torch.randperm(self.num_samples, generator=self.generator)[:self.queries_per_batch]
            yield self.neighbor_indices[q].reshape(-1).tolist()
    def __len__(self):
        return self.num_samples // self.batch_size
```

`batch_size` counts **images**; the loss then sees `batch_size × num_views` rows.
`batch_size % num_neighbors == 0` is enforced at config validation
(`training/engine.py:3399`). Neighbors are recomputed by a full deterministic-transform
embedding pass over the STML pool in the current student `g` space
(`extract_embeddings(..., embedding_kind="stml_g")`, `training/ssl/embeddings.py:96-100`),
scheduled by `should_rebuild_on_epoch` (`training/ssl/config.py:123`) — `every_epoch`
reproduces upstream.

### 2.6 The loss

`losses/metric_losses.py:82-228`. The contextual computation is fully vectorized:

```python
def stml_contextual_similarity(pair_similarity, topk, instance_ids=None):
    topk = min(int(topk), num_samples)
    ranking_similarity = pair_similarity.clone()
    ranking_similarity[instance_ids.unsqueeze(1) == instance_ids.unsqueeze(0)] = 1
    topk_indices = ranking_similarity.topk(topk, dim=1).indices
    neighbor_mask = torch.zeros_like(pair_similarity).scatter_(1, topk_indices, 1)

    reciprocal_neighbors = ((neighbor_mask + neighbor_mask.t()) == 2).to(dtype)      # V
    reciprocal_counts    = reciprocal_neighbors.sum(1, keepdim=True).clamp_min(1)    # |R(i)|
    shared_neighbors     = reciprocal_neighbors @ reciprocal_neighbors.t()           # |R(i) ∩ R(j)|
    contextual = (shared_neighbors / reciprocal_counts) * reciprocal_neighbors       # w̃^C

    half_k = max(1, int(round(topk / 2)))
    contextual = contextual[topk_indices[:, :half_k]].mean(dim=1)                    # ŵ^C
    return (contextual + contextual.t()) / 2                                         # w^C
```

Upstream computes `w̃^C` with a Python `for i in range(N)` loop and
`V[:, indNonzero].sum(1) / len(indNonzero)`; the matrix product `V Vᵀ` is the same
quantity. `clamp_min(1)` is a guard that never fires in practice: `w^P_ii = 1` is
maximal, so `i ∈ N_k(i)` and hence `i ∈ R_k(i)`, giving `|R_k(i)| ≥ 1`.

`stml_teacher_pair_weights` L2-normalizes the teacher, forms `w^P`, adds `w^C`, halves.
`STMLLoss._relaxed_contrastive` builds `d^h`, splits `W` into `pos = W ⊙ (1−I)` and
`neg = (1−W) ⊙ (1−I)`, and divides the summed pull/push by `n(n−1)`.
`_kl_self_distillation` is `F.kl_div(log_softmax(−d^f), softmax(−d^g.detach()),
reduction="sum") / n`.

### 2.7 Teacher construction and EMA

`training/engine.py:3818-3826`:

```python
def make_stml_teacher(student_model):
    teacher_model = copy.deepcopy(student_model)      # backbone + fc + g, cloned
    torch.nn.init.orthogonal_(teacher_model.embedding_g.weight)   # fresh, independent g
    torch.nn.init.zeros_(teacher_model.embedding_g.bias)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    return teacher_model
```

and the update, after the optimizer step (`training/engine.py:2152-2171`):

```python
self.model_optimizer.step()
self.model_optimizer.zero_grad(set_to_none=True)
...
update_ema_teacher(self.teacher_model, self.model,
                   momentum=criterion.teacher_momentum,
                   excluded_parameter_prefixes=("fc.",),      # never EMA the f head
                   only_trainable_parameters=True)
```

`update_ema_teacher` (`training/engine.py:3839`) lerps parameters and **floating-point
buffers**, and copies integer buffers verbatim (matching upstream's
`num_batches_tracked` special case). It uses `torch._foreach_lerp_` for a single fused
kernel over all parameters.

### 2.8 The regularizer mode, in one paragraph

`STMLRegularizer` (`training/semi_supervised.py:1346-1507`) reuses `STMLLoss` verbatim
and reuses the same NN sampler and multi-view dataset, but: it builds its pool from
**unlabeled positions only**; it keeps its teacher in `regularizer_state` rather than
`trainer.teacher_model`; its EMA (`after_optimizer_step`) excludes `fc.` but has **no**
`only_trainable_parameters` filter; and each step combines it with a supervised loss via
`supervised_weight`/`regularizer_weight`. This is a project construct; the paper trains
STML alone after the supervised phase.

---

## 3. Worked example — one complete step, with numbers

Everything below is produced by the actual code in this repository. It is small enough
to check by hand. Configuration: `σ = 3`, `δ = 1`, `num_views = 2`,
`num_neighbors = 2`, therefore `topk = 2 × 2 = 4` and `k/2 = 2`.

### 3.1 Input

Three source images A, B, C, each augmented twice. The trainer builds the batch as
`torch.cat(views, dim=0)`, so rows are ordered `[A₁, B₁, C₁, A₂, B₂, C₂]` and
`instance_ids = [0, 1, 2, 0, 1, 2]`.

Teacher `g` embeddings (2-D so the geometry is visible; already unit norm, angles
0°, 36.9°, 90°, 16.3°, 53.1°, 106.3°):

| row | image | `z^t` |
| --- | --- | --- |
| 0 | A₁ | `[ 1.00, 0.00]` |
| 1 | B₁ | `[ 0.80, 0.60]` |
| 2 | C₁ | `[ 0.00, 1.00]` |
| 3 | A₂ | `[ 0.96, 0.28]` |
| 4 | B₂ | `[ 0.60, 0.80]` |
| 5 | C₂ | `[−0.28, 0.96]` |

Student `f` (the head that gets evaluated) and student `g`:

```
f = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [-0.8, 0.4]]
g = [[1.0, 0.5], [0.5, 1.0], [-1.0, 0.5], [0.9, 0.4], [0.4, 0.9], [-0.9, 0.6]]
```

### 3.2 Step A — teacher pairwise similarity `w^P = exp(−d²/σ)`

Teacher distances `‖z^t_i − z^t_j‖`:

```
        A₁      B₁      C₁      A₂      B₂      C₂
A₁   0.0000  0.6325  1.4142  0.2828  0.8944  1.6000
B₁   0.6325  0.0000  0.8944  0.3578  0.2828  1.1384
C₁   1.4142  0.8944  0.0000  1.2000  0.6325  0.2828
A₂   0.2828  0.3578  1.2000  0.0000  0.6325  1.4142
B₂   0.8944  0.2828  0.6325  0.6325  0.0000  0.8944
C₂   1.6000  1.1384  0.2828  1.4142  0.8944  0.0000
```

`w^P`:

```
        A₁      B₁      C₁      A₂      B₂      C₂
A₁   1.0000  0.8752  0.5134  0.9737  0.7659  0.4260
B₁   0.8752  1.0000  0.7659  0.9582  0.9737  0.6492
C₁   0.5134  0.7659  1.0000  0.6188  0.8752  0.9737
A₂   0.9737  0.9582  0.6188  1.0000  0.8752  0.5134
B₂   0.7659  0.9737  0.8752  0.8752  1.0000  0.7659
C₂   0.4260  0.6492  0.9737  0.5134  0.7659  1.0000
```

Hand check: `w^P_{A₁B₁} = exp(−0.6325²/3) = exp(−0.4/3) = exp(−0.13333) = 0.87517`. ✓

Notice the problem STML is designed around: `w^P_{B₂C₁} = 0.8752` is high even though B
and C are different objects. The teacher space alone is not trustworthy.

### 3.3 Step B — neighbor ranking with same-instance pairs pinned

The ranking copy sets every same-instance entry to exactly 1, so the two views of one
image always rank as each other's neighbor:

```
        A₁      B₁      C₁      A₂      B₂      C₂
A₁   1.0000  0.8752  0.5134  1.0000  0.7659  0.4260   ← (A₁,A₂) forced to 1
B₁   0.8752  1.0000  0.7659  0.9582  1.0000  0.6492
C₁   0.5134  0.7659  1.0000  0.6188  0.8752  1.0000
A₂   1.0000  0.9582  0.6188  1.0000  0.8752  0.5134
B₂   0.7659  1.0000  0.8752  0.8752  1.0000  0.7659
C₂   0.4260  0.6492  1.0000  0.5134  0.7659  1.0000
```

This copy is used **only** for ranking; `w^P` itself stays untouched.

`N_k(i)` = top-4 indices per row, and `N_{k/2}(i)` = the first 2 of those:

| row | `N_k(i)` | `N_{k/2}(i)` |
| --- | --- | --- |
| 0 (A₁) | 0, 3, 1, 4 | 0, 3 |
| 1 (B₁) | 1, 4, 3, 0 | 1, 4 |
| 2 (C₁) | 5, 2, 4, 1 | 5, 2 |
| 3 (A₂) | 0, 3, 1, 4 | 0, 3 |
| 4 (B₂) | 1, 4, 3, 2 | 1, 4 |
| 5 (C₂) | 5, 2, 4, 1 | 5, 2 |

### 3.4 Step C — reciprocal neighborhoods `R_k`

Directed mask `N` (1 where `j ∈ N_k(i)`), then `V = R_k` keeps only mutual pairs:

```
      N (directed)                    V = R_k (reciprocal)          |R_k(i)|
  1 1 0 1 1 0                       1 1 0 1 0 0                        3
  1 1 0 1 1 0                       1 1 0 1 1 0                        4
  0 1 1 0 1 1                       0 0 1 0 1 1                        3
  1 1 0 1 1 0                       1 1 0 1 1 0                        4
  0 1 1 1 1 0                       0 1 1 1 1 0                        4
  0 1 1 0 1 1                       0 0 1 0 0 1                        2
```

`N[0,4] = 1` (A₁ lists B₂) but `N[4,0] = 0` (B₂ does not list A₁), so the pair is dropped.
This asymmetry filter is the whole point of using reciprocal rather than plain neighbors.

### 3.5 Step D — asymmetric Jaccard `w̃^C`

`|R(i) ∩ R(j)| = V Vᵀ`:

```
  3 3 0 3 2 0
  3 4 1 4 3 0
  0 1 3 1 2 2
  3 4 1 4 3 0
  2 3 2 3 4 1
  0 0 2 0 1 2
```

Divide row `i` by `|R(i)|` and mask by `V`:

```
w̃^C
  1.0000  1.0000  0.0000  1.0000  0.0000  0.0000
  0.7500  1.0000  0.0000  1.0000  0.7500  0.0000
  0.0000  0.0000  1.0000  0.0000  0.6667  0.6667
  0.7500  1.0000  0.0000  1.0000  0.7500  0.0000
  0.0000  0.7500  0.5000  0.7500  1.0000  0.0000
  0.0000  0.0000  1.0000  0.0000  0.0000  1.0000
```

Hand check: `w̃^C_{A₁B₁} = |R(0) ∩ R(1)| / |R(0)| = |{0,1,3}| / 3 = 3/3 = 1.0`. ✓
It is asymmetric: `w̃^C_{B₁A₁} = 3/4 = 0.75`.

### 3.6 Step E — query expansion, then symmetrization

Row `i` of `ŵ^C` is the mean of the `w̃^C` rows listed in `N_{k/2}(i)`:

```
ŵ^C
  0.8750  1.0000  0.0000  1.0000  0.3750  0.0000
  0.3750  0.8750  0.2500  0.8750  0.8750  0.0000
  0.0000  0.0000  1.0000  0.0000  0.3333  0.8333
  0.8750  1.0000  0.0000  1.0000  0.3750  0.0000
  0.3750  0.8750  0.2500  0.8750  0.8750  0.0000
  0.0000  0.0000  1.0000  0.0000  0.3333  0.8333
```

`w^C = (ŵ^C + ŵ^Cᵀ)/2`:

```
w^C
  0.8750  0.6875  0.0000  0.9375  0.3750  0.0000
  0.6875  0.8750  0.1250  0.9375  0.8750  0.0000
  0.0000  0.1250  1.0000  0.0000  0.2917  0.9167
  0.9375  0.9375  0.0000  1.0000  0.6250  0.0000
  0.3750  0.8750  0.2917  0.6250  0.8750  0.1667
  0.0000  0.0000  0.9167  0.0000  0.1667  0.8333
```

Hand check for `(A₁,B₁)`: `ŵ^C_{01} = (w̃^C_{01} + w̃^C_{31})/2 = (1.0 + 1.0)/2 = 1.0`;
`ŵ^C_{10} = (w̃^C_{10} + w̃^C_{40})/2 = (0.75 + 0.0)/2 = 0.375`; so
`w^C_{01} = (1.0 + 0.375)/2 = 0.6875`. ✓

### 3.7 Step F — the pseudo label `w = (w^P + w^C)/2`

```
w
        A₁      B₁      C₁      A₂      B₂      C₂
A₁   0.9375  0.7813  0.2567  0.9556  0.5705  0.2130
B₁   0.7813  0.9375  0.4455  0.9479  0.9243  0.3246
C₁   0.2567  0.4455  1.0000  0.3094  0.5834  0.9452
A₂   0.9556  0.9479  0.3094  1.0000  0.7501  0.2567
B₂   0.5705  0.9243  0.5834  0.7501  0.9375  0.4663
C₂   0.2130  0.3246  0.9452  0.2567  0.4663  0.9167
```

What context bought:

| pair | `w^P` | `w^C` | `w` | effect |
| --- | --- | --- | --- | --- |
| A₁–A₂ (same image) | 0.9737 | 0.9375 | 0.9556 | confirmed |
| A₁–B₁ (related) | 0.8752 | 0.6875 | 0.7813 | mildly damped |
| A₁–C₂ (unrelated) | 0.4260 | 0.0000 | 0.2130 | **halved** — no shared context |
| B₂–C₁ (false friend) | 0.8752 | 0.2917 | 0.5834 | strongly corrected |

The `w^P`-only signal would have pulled B₂ and C₁ together with weight 0.88; the
contextual term cuts that to 0.58.

### 3.8 Step G — relaxed contrastive loss on `f`

Row-mean-normalized student distances `d^f` (note the matrix is **not** symmetric — each
row is divided by its own mean):

```
d^f
  0.0000  1.2717  1.7985  0.2543  1.0174  1.6581
  1.6185  0.0000  1.6185  1.2948  0.3237  1.1445
  1.6866  1.1926  0.0000  1.5273  1.2163  0.3771
  0.2984  1.1938  1.9110  0.0000  0.8953  1.7014
  1.4195  0.3549  1.8096  1.0647  0.0000  1.3514
  1.8499  1.0032  0.4487  1.6177  1.0805  0.0000
```

Pull `w_ij (d^f_ij)²` (diagonal zeroed):

```
  0.0000  1.2636  0.8303  0.0618  0.5905  0.5856
  2.0468  0.0000  1.1669  1.5891  0.0969  0.4252
  0.7303  0.6336  0.0000  0.7217  0.8630  0.1344
  0.0851  1.3508  1.1299  0.0000  0.6013  0.7431
  1.1495  0.1164  1.9104  0.8502  0.0000  0.8515
  0.7289  0.3267  0.1903  0.6718  0.5444  0.0000
```

Push `(1 − w_ij)[δ − d^f_ij]₊²` — almost everything is zero because most relative
distances already exceed `δ = 1`:

```
  0.0000  0.0000  0.0000  0.0247  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0346  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0213
  0.0219  0.0000  0.0000  0.0000  0.0027  0.0000
  0.0000  0.0315  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0167  0.0000  0.0000  0.0000
```

```
pull.sum() = 22.99020
push.sum() =  0.15331
L_RC(f) = (22.99020 + 0.15331) / (6·5) = 23.14351 / 30 = 0.771450
```

The only surviving push terms are exactly the same-image and same-class pairs that the
student currently places *too close* — which is the intended behavior of a margin loss
under high `w`: they contribute a large pull and a small residual push.

### 3.9 Step H — the full objective

```
L_RC(f) = 0.771450
L_RC(g) = 0.753109
L_RC    = (0.771450 + 0.753109)/2 = 0.762279
L_KL    = 0.025045
L_STML  = 0.762279 + 0.025045 = 0.787324
```

Backpropagating gives `∂L/∂z^f`:

```
  [ 0.0108, -0.0258]      A₁
  [-0.0804,  0.0014]      B₁
  [ 0.0060, -0.0112]      C₁
  [ 0.0287, -0.0446]      A₂
  [-0.0395,  0.0380]      B₂
  [ 0.0745,  0.0423]      C₂
```

### 3.10 Step I — teacher EMA

With `m = 0.999`, after `optimizer.step()`:

```
θ_t ← 0.999·θ_t + 0.001·θ_s     for backbone and embedding_g (trainable params only)
θ_t(fc)  unchanged              the retrieval head is excluded
integer buffers copied verbatim
```

### 3.11 Reproducing this example

The script that produced every number above, including a cross-check against a literal
transcription of upstream `code/loss.py`, is reproducible from
`losses/metric_losses.py` directly:

```python
from losses import metric_losses
criterion = metric_losses.STMLLoss(sigma=3.0, delta=1.0, num_views=2,
                                   num_neighbors=2, normalize_student=False)
criterion(student_f, student_g, teacher_g, instance_ids)   # → 0.787324
```

---

## 4. Verified parity of the loss

Running upstream's `RC_STML` and `KL_STML` **verbatim** (Python `for`-loop `w̃^C`
included) on the inputs of §3 and comparing against `STMLLoss`:

| Term | Upstream `code/loss.py` | `losses/metric_losses.py` | Difference |
| --- | --- | --- | --- |
| `L_RC(f)` | 0.7714502811431885 | 0.7714502811431885 | 0.0 |
| `L_RC(g)` | 0.7531085610389710 | 0.7531085610389710 | 0.0 |
| `L_KL` | 0.0250448063015938 | 0.0250448063015938 | 0.0 |
| `L_STML` | 0.7873241901397705 | 0.7873241901397705 | 0.0 |

Bit-identical in float32. This is not a one-off: `tests/test_stml.py::
test_loss_matches_direct_upstream_equations` (line 73) permanently asserts the same
equivalence against an inlined copy of upstream's code, including
`((W_NN + W_NN.t())/2 == 1)`, `int(np.around(topk/2))`, and the `for i in range(N)`
loop.

Term-by-term correspondence:

| Paper | Upstream | This project |
| --- | --- | --- |
| Eq. 5 `w^P` | `torch.exp(-T_dist.pow(2) / sigma)` | `torch.exp(-teacher_distances.square() / sigma)` |
| same-instance pin | `W_P_copy[idx.unsqueeze(1)==idx.unsqueeze(1).t()] = 1` | `ranking_similarity[same_instance] = 1` |
| Eq. 6 `R_k` | `((W_NN + W_NN.t())/2 == 1).float()` | `((neighbor_mask + neighbor_mask.t()) == 2).to(dtype)` |
| Eq. 7 `w̃^C` | `for i: V[:,idx].sum(1)/len(idx)` | `(V @ V.t() / V.sum(1,keepdim=True)) * V` |
| Eq. 8 `ŵ^C` | `W_C_tilda[topk_half_index].mean(1)` | `contextual[topk_indices[:, :half_k]].mean(1)` |
| Eq. 9 `w^C` | `(W_C_hat + W_C_hat.t())/2` | `(contextual + contextual.t())/2` |
| Eq. 10 `w` | `(W_P + W_C)/2` | `(pair_similarity + contextual_similarity)/2` |
| Eq. 1 `L_RC` | `(pull.sum()+push.sum())/(n*(n-1))` | `(pull.sum()+push.sum())/off_diagonal.sum()` |
| Eq. 2 `L_KL` | `kl_div(log_softmax(-d_f), softmax(-d_g.detach()), 'sum')*1/n` | identical |
| Eq. 3 `L_STML` | `(RC_f+RC_g)/2 + KL` | identical |
| Eq. 4 EMA | `v_t*m + (1-m)*v_s` over `state_dict()` | `torch._foreach_lerp_(teacher, student, 1-m)` |

Two implementation choices in this project are safety guards that cannot change a result
in any valid configuration: `topk = min(topk, num_samples)` (upstream would raise if the
batch were smaller than `k`) and `.clamp_min(...)` on `|R(i)|` and on the row-mean
divisor (both quantities are provably positive).

---

## 5. Differences to the official implementation

Grouped by whether they change the numbers, the training dynamics, or only the surrounding
infrastructure.

### 5.1 Loss mathematics — no differences

Established in §4. `STMLLoss` is a faithful, bit-identical port of `STML_loss`. The
constructor additionally validates `sigma > 0`, `delta > 0`, `num_views ≥ 2`,
`num_neighbors > 0`, `0 ≤ teacher_momentum < 1`, and `batch_size ≥ 2` — upstream validates
nothing.

### 5.2 Training-loop differences

| # | Aspect | Upstream | This project | Consequence |
| --- | --- | --- | --- | --- |
| 1 | **Supervised warm-up** | none — STML runs from epoch 0 on an ImageNet-pretrained backbone | `warmup_epochs` of supervised training on the labeled split with `warmup_loss` (`MultiSimilarityLoss` + `MultiSimilarityMiner` in the shipped configs) | This is the paper's §4.2 semi-supervised recipe, absent from the released code. The teacher is created only at the STML boundary. |
| 2 | **EMA timing** | `loss.backward()` → `momentum_update()` → `opt.step()` (`code/main.py:320-324`) — the teacher is averaged toward the student's weights **before** the current update is applied | `opt.step()` → `zero_grad()` → `update_ema_teacher()` (`training/engine.py:2152-2171`) — averaged toward the **post-update** weights | **This project follows the paper; upstream does not.** Algorithm 1 orders Step #3 ("compute `L_STML` and optimize the student") strictly before Step #4 (`θ_t ← mθ_t + (1−m)θ_s`), which is `opt.step()` then EMA. Upstream's code contradicts its own algorithm listing. Do **not** "fix" this by matching upstream — see the note below. |
| 3 | **EMA scope** | iterates `zip(student.state_dict(), teacher.state_dict())`; the teacher has no `embedding_f`, so `zip` truncates and the `f` head is skipped implicitly | `excluded_parameter_prefixes=("fc.",)` skips the `f` head explicitly, and `only_trainable_parameters=True` also skips frozen backbone parameters | Functionally identical: frozen parameters do not change on the student either, so EMAing them is a no-op. The teacher here still carries a dead `fc` module (deepcopied, never updated, never read) — wasted memory only. |
| 4 | **EMA scope, regularizer mode** | n/a | `STMLRegularizer.after_optimizer_step` excludes `fc.` but has **no** `only_trainable` filter, and lerps parameters one at a time rather than via `_foreach_lerp_` | An internal inconsistency between the two modes of this project. Numerically equivalent, slower. |
| 5 | **Teacher init** | a second network built from scratch: same ImageNet weights, **independently** random-orthogonal `g` head | `deepcopy(student)` after warm-up, then `g` re-initialized orthogonally | Same *relationship* (shared backbone, independent `g`), but anchored at a fine-tuned rather than a pretrained backbone. The paper's semi-supervised text says the labeled-trained model initializes "the teacher and student models" and is silent about the heads; re-initializing `g` follows the released code. |
| 6 | **Neighbor rebuild cadence** | hard-coded `if epoch % 1 == 0` — every epoch | `update_mode` ∈ `every_epoch` / `every_n_epochs` / `once` (`training/ssl/config.py:123`) | `every_epoch` reproduces upstream; the shipped configs use it. |
| 7 | **Sampling RNG** | `np.random.choice(n, q, replace=False)` on the global NumPy RNG | `torch.randperm(n, generator=seeded)[:q]` | Same distribution, seed-controlled here. |
| 8 | **NN search** | manual `‖x‖² − 2XXᵀ + ‖x‖²` expansion in 5000-row chunks | `torch.cdist` in 512-row chunks, `topk(largest=False)` | Same neighbor sets (`topk` of `d` = `topk` of `d²`); `cdist` is numerically better behaved than the expansion. Runs on `--ssl_device`. |
| 9 | **Sampling embeddings** | whole train set, student `g` head, model forced to `eval()`, deterministic resize-256 transform | STML pool positions only, student `g` head (`embedding_kind="stml_g"`), `model.eval()` restored afterwards, dataset's deterministic `feature_transform` | Equivalent mechanism, different transform because the backbones differ. |
| 10 | **Drop-last** | `__len__ = n // batch_size`, fixed-size batches | identical | — |

> **On difference 2 — leave it alone.** It is tempting to move `update_ema_teacher`
> before `optimizer.step()` "to match upstream", but that would trade fidelity to the
> published algorithm for fidelity to a released script that disagrees with it. The
> `opt.step()` → EMA order used here is also the standard momentum-encoder convention
> (MoCo, BYOL, Mean Teacher). The magnitude either way is one student update scaled by
> `1 − m`, i.e. 0.1 % at `m = 0.999` and 0.01 % at the semi-supervised `m = 0.9999`. If
> you want an exact-parity ablation against upstream's script, swap the two lines behind
> a flag and report it as such — but the default should stay as it is.

### 5.3 Backbone and head differences

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 11 | Backbone | GoogLeNet / BN-Inception / ResNet18 / ResNet50, ImageNet or SwAV pretrained, optional `bn_freeze` | DINOv2 ViT-S/B/L/g via Torch Hub, `backbone_tuning ∈ {full, frozen, last_N_blocks}` |
| 12 | Pooling | `GAP + GMP` summed, then the heads | DINOv2 CLS token, then the heads |
| 13 | BatchNorm | present; `bn_freeze` puts BN in `eval()` and freezes affine params; running stats are EMA'd into the teacher | ViT uses LayerNorm — no running buffers exist, so the buffer branch of `update_ema_teacher` is inert |
| 14 | `f` dimension | `--embedding_size`, default 512 | `feat_dim`; `None` means the backbone width |
| 15 | `g` dimension | `--bg_embedding_size`, default 1024 | `stml_g_dim`; `None` means the backbone width (384/768/1024/1536) — this is what the paper actually specifies |
| 16 | Head init | orthogonal weight, zero bias, both heads | identical (`models/retrieval_model.py:325-336`) |
| 17 | **Frozen backbone** | not possible | `backbone_tuning: "frozen"` and `last_N_blocks` are *available*, but **no STML config uses them** — every `configs/experiments/stml_*.json` leaves `backbone_tuning` unset, so it defaults to `full` (`training/cli.py:223`) | With the default `full`, the teacher's backbone is EMA-updated exactly as in the paper. |

> **The teacher–student design is fully intact, and is the default.** Under
> `backbone_tuning: "full"` — what every STML experiment config uses — the teacher is a
> genuinely separate network: its own deepcopied DINOv2 backbone plus its own `g` head,
> `requires_grad_(False)`, permanently in `eval()`, never in the optimizer, forwarded
> under `torch.no_grad()`, and drifting behind the student by EMA over **all** trainable
> parameters (backbone included), exactly as in Eq. (4). Nothing about the mechanism is
> reduced.
>
> The caveat is narrow and opt-in: *if* someone sets `backbone_tuning: "frozen"`, student
> and teacher share a backbone that can never change, so the EMA only ever moves
> `embedding_g` and the teacher degenerates to "the student's `g` head, lagged" — a
> slowly-moving *linear projection* instead of a slowly-moving *representation*. The one
> frozen config in the tree, `configs/runs/experiments/stml.json`, belongs to the
> **`stml_threshold`** method (`configs/runs/ssl/stml/*.json` all set
> `method: "stml_threshold"` and train with `NTXentLoss`), not to the `STMLLoss` method
> this document covers. If a frozen-backbone STML run is ever added, state the reduction
> explicitly when reporting it.

Frozen and partially-frozen runs are supported; the table below is what each setting
actually does to the teacher and to the multi-view signal. `fc` and `embedding_g` are
always trainable — `backbone_tuning` only ever touches `self.dinov2`
(`models/retrieval_model.py:363-388`) — so the loss always has parameters to optimize.

| `backbone_tuning` | `use_cache` | Teacher lags… | Two views differ? | Notes |
| --- | --- | --- | --- | --- |
| `full` (default, all STML configs) | any | backbone **and** `g` | yes | Eq. (4) as published |
| `last_N_blocks` | any | last *N* blocks + final norm **and** `g` | yes | genuine EMA-lagged *representation* at a fraction of the cost — the closest frozen-budget setting to the paper |
| `frozen` | `false` | `g` only | yes | teacher = lagged linear projection of a shared frozen space; augmentation invariance preserved |
| `frozen` | `true` | `g` only | **no** | also triggers feature precompute (§5.7 #33): both views collapse to one deterministic vector |

Under `frozen`, the teacher is still a correct EMA of *every trainable parameter the
teacher uses* — `embedding_g` is independently initialized, then averaged toward the
student's at rate `1 − m`, and `w` is still computed in a space the student is not
directly optimizing. What is lost is the paper's premise that the teacher carries a
slowly-moving **representation**: with `φ_t ≡ φ_s ≡ φ` frozen, both `g` maps are linear
functions of the same features. `last_N_blocks` is the recommended way to restore a
lagging representation without full fine-tuning.

### 5.4 Optimization differences

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 18 | Optimizer | AdamP with Nesterov (also sgd/adam/rmsprop/adamw) | `--optim ∈ {adam, adamw, rmsprop}`, default `adam`. No AdamP. |
| 19 | Parameter groups | **two**: backbone at `--lr`, `embedding_f + embedding_g` at `--emb-lr` with explicit weight decay | **one** group, single `--lr`, over `[p for p in model.parameters() if p.requires_grad]` (`training/engine.py:2612`) |
| 20 | LR schedule | `CosineAnnealingLR(T_max=90)` by default | `--lr_scheduler` default `none` (constant); cosine/step/warm-restarts available |
| 21 | Weight decay | `1e-2` default | `0.0` default |
| 22 | Epochs / LR | 90 epochs, `lr = 1e-4` | shipped STML configs: 25 epochs, `lr = 1e-5` (CIFAR) or `1e-6` (Cars+CompCars), with early stopping (`patience`) |
| 23 | Gradient scale | n/a | `regularizer_weight`/`supervised_weight` and the optional `grad_norm` weighting apply only in the composed mode |

### 5.5 Evaluation differences

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 24 | **Evaluated embedding** | `_, J = model(x)` → raw `f`, **not** L2-normalized (`student_norm` defaults to 0), retrieval by squared Euclidean distance | `project_features()` → `F.normalize(fc(features))`, i.e. **L2-normalized** `f` |
| 25 | Metric | Recall@k only | Recall/precision@k, MAP@R, per-class metrics, validation-driven model selection |
| 26 | Checkpointing | saves every epoch that improves R@1 | validation-driven with early stopping |

Difference 24 is a real train/eval mismatch worth naming in a thesis: with
`normalize_student = false` (the paper's and this project's default) the loss shapes an
**unnormalized** `f` space, and this project then retrieves on the **normalized** version
of it. Direction is preserved but relative magnitudes are discarded, so ranking can
differ from upstream's. Setting `normalize_student: true` makes training and evaluation
agree — at the cost of departing from the paper's default.

### 5.6 Numerical precision

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 27 | Forward precision | float32 throughout | the whole step runs under `torch.autocast(dtype=torch.bfloat16)` (`training/engine.py:1618`), so `student_f`, `student_g`, `teacher_g` are produced in bf16 |
| 28 | Loss precision | float32 | `STMLLoss` casts every input to `.float()` before `cdist` (`losses/metric_losses.py:196, 215-216`) and normalizes the teacher in float32 (`:125`) — all pairwise geometry, the exponential kernel, the top-k ranking, and the softmaxes are float32 |

So the *loss arithmetic* is float32 in both, but the *embeddings entering it* have bf16
precision here (~3 decimal digits). Two consequences to be aware of: near-tied `w^P`
values can flip the `topk` ranking, and identical-view pairs can round to exactly equal
vectors.

### 5.7 Data and protocol differences

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 29 | Task | unsupervised only: the whole training split, no labels anywhere | semi-supervised: a labeled/unlabeled split (`label_sampling_mode` ∈ `per_class_min`, `per_class_imbalanced`, `global_budget`, `class_subset`, `class_subset_k_shot`), warm-up on the labeled part, then STML over labeled **+** unlabeled with all labels hidden |
| 30 | Datasets | CUB, Cars196, SOP | CIFAR-10/100, CUB, Cars196, Aircraft, Food101, iNaturalist, Semi-Aves, In-Shop, Stanford Dogs, … |
| 31 | External unlabeled pools | none | `unlabeled_source ∈ {split, external, split_and_external}` with verified CompCars (`compcars_stml_paper`, 16 537 images / 145 classes) and NABirds filters — this is what reconstructs the paper's §4.2 dataset pairing |
| 32 | Augmentation | `RandomResizedCrop(227, scale=(0.16,1))` + `RandomHorizontalFlip`, applied `view` times | the project's own train transform, applied `num_views` times by `UnlabeledSubset` |
| 33 | Frozen-feature precompute | n/a | when `ssl_frozen_feature_precompute` is active the dataset yields deterministic 2-D backbone features and `forward_stml_cached` short-circuits on `images.ndim == 2` |

Difference 33 deserves emphasis, and its trigger is precise:
`should_precompute_frozen_features` (`training/engine.py:282-298`) requires
**`use_cache: true` *and* `backbone_tuning: "frozen"`** together. When both hold, the
dataset yields deterministic 2-D backbone features, so **both views of an image are the
identical feature vector**. Same-instance distances become exactly 0, `w^P` becomes
exactly 1 for those pairs, and the augmentation-invariance signal that STML's multi-view
construction supplies is gone — the model still trains from the relaxed contrastive and
KL terms, but one of the paper's ingredients has been removed. (Gradients remain finite;
PyTorch's `cdist` backward guards zero distances, verified.) `frozen_feature_train_views`
is forced to 1 in this mode with a warning (`training/engine.py:2343, 2357`).

`backbone_tuning: "frozen"` with `use_cache: false` does **not** trigger this: every view
is re-augmented and pushed through the frozen backbone, so the two views genuinely
differ. That combination costs backbone compute but keeps multi-view augmentation intact.

### 5.8 Configuration surface

| # | Aspect | Upstream | This project |
| --- | --- | --- | --- |
| 34 | `view` | any `≥ 1` | `num_views ≥ 2` enforced |
| 35 | Naming | `--view`, `--num_neighbors`, `--momentum`, `--student_norm` (a flag named `disable_mu` inside the loss, whose value 1 actually *enables* normalization) | `num_views`, `num_neighbors`, `teacher_momentum`, `normalize_student` (unambiguous) |
| 36 | `topk` | `--num_neighbors * --view`, computed in `main.py` | `num_views * num_neighbors`, computed in the loss (`losses/metric_losses.py:164`) — same value, no way to desynchronize it |
| 37 | Miner | n/a | explicitly rejected: `miner` must be `no_miner`, since STML consumes no labels (`training/engine.py:3390`) |

### 5.9 Project-only extensions with no upstream counterpart

* **Composed regularizer mode** (§2.8) — STML as an auxiliary term alongside any
  supervised loss, on unlabeled data only. Not in the paper, not in upstream.
* **`stml_threshold`** — keeps STML's `w` but cuts it at a confidence threshold to
  produce hard pseudo-labels usable by any metric loss. See
  [stml_threshold.md](stml_threshold.md).
* Backbone feature caching, GPU-resident loaders, CUDA-graph capture, HPO integration,
  cross-validation, gradient-contribution diagnostics, step visualization.

---

## 6. Shared deviations — where *upstream* differs from the paper

These are inherited by this project verbatim, so they are differences to the **paper**
but not to the **code**. They matter for any thesis that quotes the equations.

### 6.1 `L_RC` normalization: `n(n−1)` instead of `n`

Paper Eq. (1) is `(1/n) Σ_i Σ_{j≠i}`. Both implementations divide by `n(n−1)`:

```python
loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb)-1))
```

So the implemented `L_RC` equals the paper's `L_RC / (n−1)`. Since `L_KL` *does* use
`1/n` in both the equation and the code, the implemented balance between the relaxed
contrastive term and the self-distillation term is `(n−1)×` more KL-weighted than Eq. (3)
as written. At the paper's `n = 240` that is a factor of 239. Anyone re-deriving the
objective from the paper and implementing it literally will not reproduce either
codebase.

### 6.2 `L_KL` includes the diagonal

Paper Eq. (2) sums over `j ≠ i`. Both implementations softmax over the **full** row,
including `j = i` where `−d_ii = 0`, and `reduction="sum"` then includes the `(i,i)`
term. The diagonal both shifts every softmax denominator and contributes a nonzero KL
term of its own (`p_ii ≠ q_ii` because the row sums of `f` and `g` differ).

### 6.3 Teacher L2-normalization lives in the loss, not the model

The paper describes *"an `l₂` normalization layer on top of the teacher"*. Upstream's
teacher is built with `is_norm=1`, but that flag only normalizes `x_f`, which the teacher
never returns — so `x_g` leaves the model unnormalized and `RC_STML.forward` normalizes
it instead (`t_emb = F.normalize(t_emb)`). This project does the same thing directly in
`stml_teacher_pair_weights` (`losses/metric_losses.py:125`). Same result; the model-level
flag in upstream is vestigial.

### 6.4 Duplicate rows in a batch

The NN sampler concatenates `q` neighbor groups without de-duplication, so an image that
is a neighbor of two different queries appears twice in the batch. Those duplicate rows
share an instance id and are therefore pinned to similarity 1 in the ranking, which
distorts `R_k` slightly. Both implementations behave identically; neither guards against
it.

### 6.5 `w^P` is redundantly recomputed

`L_RC` is evaluated once for `f` and once for `g`, and each call recomputes the full
teacher pipeline (`w^P`, top-k, `R_k`, `w^C`) from the same `teacher_g`. Both
implementations pay this cost twice per step. The second computation is deterministic and
identical, so it is pure waste, not a numerical issue.

### 6.6 The `disable_mu` naming inversion

Upstream passes `disable_mu = args.student_norm` (default 0). Inside the loss,
`if self.disable_mu: s_emb = F.normalize(s_emb)` — the flag *enables* normalization
despite its name. This project renames it to `normalize_student` and keeps the same
default (`False`). No behavioral difference, but the upstream name is a trap when reading
their code.

---

## 7. Hyperparameter mapping

| Paper | Upstream flag | This project | Default here | Paper value |
| --- | --- | --- | --- | --- |
| `σ` | `--sigma` | `loss_params.sigma` | `3.0` | 3 (unsup.), 2 (semi-sup.) |
| `δ` | `--delta` | `loss_params.delta` | `1.0` | 1.0 (CUB/Cars), 0.9 (SOP) |
| `m` | `--momentum` | `loss_params.teacher_momentum` | `0.999` | 0.999 (unsup.), 0.9999 (semi-sup.) |
| `V` | `--view` | `loss_params.num_views` | `2` | 2 |
| `k / V` | `--num_neighbors` | `loss_params.num_neighbors` | `5` | 5 (CUB/Cars), 2 (SOP) |
| `k` | derived | derived: `num_views × num_neighbors` | `10` | 10 (CUB/Cars), 4 (SOP) |
| — | `--student_norm` | `loss_params.normalize_student` | `false` | 0 |
| `dim(f)` | `--embedding_size` | `feat_dim` | `None` → backbone width | 512 (128/64 in ablations) |
| `dim(g)` | `--bg_embedding_size` | `stml_g_dim` | `None` → backbone width | last-pooling width |
| `q·k` | `--batch-size` | `batch_size` (images; rows = `× num_views`) | 16 (CLI), 40/256 in shipped configs | 120 |
| epochs | `--epochs` | `epochs` | 25 in shipped configs | 90 |
| — | — | `ssl_config.warmup_epochs` | 5 | n/a (unsup.) / implicit (semi-sup.) |

The shipped [stml_cars196_compcars.json](../configs/experiments/stml_cars196_compcars.json)
adopts the paper's semi-supervised values `sigma = 2.0`, `teacher_momentum = 0.9999`,
`batch_size = 256`, `num_neighbors = 8`.

---

## 8. Summary for the thesis

1. **The objective is an exact port.** `STMLLoss` reproduces `STML_loss` bit-for-bit in
   float32, verified by a permanent regression test against a literal transcription of
   upstream's code. Any performance difference is *not* attributable to the loss.
2. **The scientific setting differs by design.** Upstream implements only the paper's
   unsupervised experiment. This project implements the paper's §4.2 semi-supervised
   experiment — supervised warm-up, then STML over labeled + unlabeled data with labels
   hidden — for which **no reference implementation exists**. That reconstruction (warm-up
   loss, teacher creation at the phase boundary, external unlabeled pools) is this
   project's contribution and cannot be validated against upstream code.
3. **The teacher–student architecture is unchanged.** Two networks, EMA-coupled over the
   full backbone plus `g`, teacher frozen and in `eval()`, `f` excluded from the average —
   Eq. (4) as written. Every STML config runs `backbone_tuning: full`, so nothing about
   the momentum-encoder mechanism is weakened (§5.3 #17).
4. **Three changes can plausibly move numbers.** In rough order of expected impact:
   (a) DINOv2 ViT replacing GoogLeNet/BN-Inception/ResNet50 — different pretraining,
   different pooling, no BatchNorm; (b) evaluating an L2-normalized `f` while training an
   unnormalized one (§5.5 #24); (c) single-parameter-group Adam at a constant LR instead
   of two-group AdamP with cosine decay (§5.4). Behind those, bf16 embeddings feeding a
   float32 loss (§5.6). The EMA-timing offset (§5.2 #2) is *not* on this list — there the
   implementation follows the paper and upstream's script is the outlier.
5. **Two of the paper's own equations are not what either codebase computes** — the
   `L_RC` normalizer and the KL diagonal (§6.1, §6.2). Quote the code, not Eq. (1), when
   describing the implemented objective.
