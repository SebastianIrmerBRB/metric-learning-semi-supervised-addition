# SLADE: How It Works

SLADE is the self-training framework of *"SLADE: A Self-Training Framework For
Distance Metric Learning"* (Duan, Lin, Tran, Davis, Kuo — CVPR 2021,
[arXiv:2011.10269](https://arxiv.org/abs/2011.10269)). It is implemented in
`training/ssl/slade.py` as a `BaseTrainingRegularizer`, so it never replaces the
configured metric-learning loss — it adds two unlabeled terms on top of it.

This document explains the method and the implementation together. For the
line-by-line ledger of where this implementation departs from the paper, see
[slade_implementation_vs_paper.md](slade_implementation_vs_paper.md); for using
an external unlabeled pool, see [slade_compcars.md](slade_compcars.md).

---

## 1. The idea in one paragraph

Cluster the unlabeled pool with a teacher network and call the cluster IDs pseudo
labels. Those labels are noisy, so do **not** train a ranking loss on them
directly. Instead learn a set of *basis vectors* `W_a` — one per labeled class,
trained by cross entropy on labeled data — and use the basis to decide which
unlabeled pairs are trustworthy. Only pairs the basis is confident about enter
the ranking loss. The paper's own ablation (Table 4, CUB-200) is the argument
for the machinery: pseudo labels alone move MAP@R 29.29 → 30.81, adding the basis
gives 32.45, and adding mining on top gives 33.59.

---

## 2. The objective

Equation 9 of the paper is the whole training objective:

```
L = L_rank(D_l)  +  λ1 · L_rank(D_u)  +  λ2 · L_Basis
     ^ labeled       ^ mined unlabeled     ^ basis vectors
     ranking loss    pairs, same loss      (Eq 2 = CE + SD)
```

with `λ1 = 1`, `λ2 = 0.25`, chosen by the authors "to make the magnitude of each
loss in a similar scale".

The three terms map onto the codebase like this:

| Paper | Where it lives |
| --- | --- |
| `L_rank(D_l)` | the engine's ordinary supervised loss — SLADE never touches it |
| `λ1 · L_rank(D_u)` | `SladeRegularizer.compute_loss` returns raw `L_rank(D_u)`; the framework's `regularizer_weight` supplies `λ1` |
| `λ2 · L_Basis` | added by `SladeRegularizer.combine_losses` |

Keeping the two unlabeled terms apart matters: `L_Basis` trains `W_a`, a
parameter the supervised loss never touches, and during basis warm-up it is the
*only* gradient the regularizer produces. Summing them would report one
meaningless `regularizer_norm` in the gradient diagnostics and let the basis
gradient drive a weight meant for the ranking gradient.

```python
# training/ssl/slade.py
def combine_losses(self, supervised_loss, regularization_loss):
    total = super().combine_losses(supervised_loss, regularization_loss)
    if self._basis_loss is None:
        return total
    return total + self.lambda_basis * self._basis_loss
```

---

## 3. Step by step

### 3.1 Pseudo labels (Sec 3.2)

The teacher embeds the unlabeled pool and k-means turns the features into
cluster IDs. This happens once per *self-training iteration*, not once per step:

```python
# training/ssl/slade.py -- make_loader
if self._cluster_by_position is None or should_rebuild_on_epoch(
    config.update_mode, config.update_interval_epochs, epoch, self._last_cluster_rebuild_epoch
):
    self._refresh_pseudo_labels(...)     # promote student -> teacher, re-cluster
```

`update_interval_epochs` **is** the length of one self-training iteration. The
paper trains a student to convergence, promotes it to teacher, and repeats "a few
times". An interval of `1` is the setting that departs from that: the student
then re-clusters every epoch with a head the previous clustering just shaped, so
cluster errors reinforce themselves.

For an explicit teacher/student lifecycle without coupling the CV folds, use
`paper_teacher_student.json`. It sets `update_mode: "once"`: labeled-only
warm-up trains the teacher, that teacher clusters once, and the student starts
from the teacher's embedding weights with fresh optimizer/scheduler state. Only
student epochs participate in final checkpoint selection, and every fold starts
independently.

To additionally reproduce the paper's unusual sequential fold handoff, use
`paper_cross_fold_teacher_student.json`. Its `cross_fold_teacher_handoff`
promotes the validation-selected student projection into the next fold. The new
fold still gets a fresh `W_a`, fresh Gaussian statistics, and fresh optimization
state, but its score is no longer an independent CV observation.

Cluster IDs are stored by *training-subset position*, not by row in the unlabeled
subset, because that is how batches identify their samples:

```python
lookup = np.full(int(self.dataset.positions.max()) + 1, -1, dtype=np.int64)
lookup[self.dataset.positions] = clusters
self._cluster_by_position = lookup
```

A batch that contains a sample without a pseudo label raises rather than
silently training on a `-1`.

#### `num_clusters` vs `cluster_ratio`

k has two forms, the same way the basis warm-up and the unlabeled batch do.
`num_clusters` is the absolute count. `cluster_ratio` is relative to the C
labeled classes the basis layer is sized to, so `0.5` is `k = C/2` and `2.0` is
`k = 2C`; the historical default of one cluster per class is the ratio `1.0`.
Both null keeps that default.

The absolute form means a different thing on every dataset. `k = 98` is a fifth
of Cars-196 and a thirtieth of In-Shop's training classes, so a study that
searches an absolute count is pinned to the dataset it was tuned on: its winner
cannot be replayed elsewhere, and a range picked for one dataset is mostly out
of bounds on another. Searching `cluster_ratio` instead makes the space
transferable, and a range like `[0.25, 2.0]` states the thing the paper's
NABirds ablation actually found — that the useful k sits near the class count
and is not very sensitive around it.

The absolute count wins when both are set, and a warning says so. The ratio is
resolved once per fold in `build_dataset`, rounded to the nearest integer, and
floored at 2 — with a warning, because every ratio below `2/C` is otherwise the
same run under a different name. The existing clamp to the unlabeled pool size
applies afterwards, to whichever form produced k.

### 3.2 The basis vectors (Sec 3.3.1, Eq 2–4)

`W_a` is a `k × d` matrix, one row per labeled class, and `r = W_a f` is the
"class-wise similarity representation". Because it is trained with softmax cross
entropy, its rows approximate class centers.

```python
class SladeBasisLayer(nn.Module):
    def forward(self, embeddings):
        """Return the feature representation r = W_a f (Sec 3.3.1)."""
        return embeddings @ self.weight.t().to(dtype=embeddings.dtype)
```

Two losses train it — Eq 2, `L_Basis = L_CE + L_SD`:

**Eq 3, cross entropy on labeled data.** Autocast is disabled *around* the
projection rather than casting after it, because matmul is cast down to bfloat16
even from float32 operands:

```python
with torch.autocast(device_type=labeled_features.device.type, enabled=False):
    cross_entropy_loss = F.cross_entropy(
        basis.logits(labeled_features.float()), supervised_labels,
    )
```

**Eq 4, the basis similarity** — cosine in the *basis* space, not the retrieval
space. Same float32 reasoning, and here it is load-bearing: Eq 5 reads the
**variance** of these cosines, and a spread around a common mean is exactly what
eight mantissa bits cannot resolve.

```python
def pairwise_similarity(self, embeddings, eps=1e-8):
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        representations = self(embeddings.float())
        normalized = representations / representations.norm(dim=1, keepdim=True).clamp_min(eps)
        return normalized @ normalized.t()
```

**Eq 5, the similarity distribution loss.** Pairs are split by pseudo label into
"same cluster" and "different cluster", and their similarities are modelled as
two Gaussians `G+` and `G−`. The loss pushes the means apart and the variances
down — deliberately *not* a per-pair loss, which is what makes it robust to
individual pseudo-label noise:

```python
def similarity_distribution_loss(positive_mean, positive_variance,
                                 negative_mean, negative_variance,
                                 margin, variance_weight):
    """Eq 5: separate the two Gaussians and penalize their spread."""
    margin_term = F.relu(negative_mean - positive_mean + float(margin))
    return margin_term + float(variance_weight) * (positive_variance + negative_variance)
```

**Eq 6, the momentum update.** `β` weights the *running* estimate, which is what
makes the two Gaussians global rather than per-batch — at the paper's `β = 0.99`
a single batch moves `µ+` by 1% of the gap:

```
µ+ = (1 − β) · µ+_batch  +  β · µ+
```

Each Gaussian updates on its own batches. A batch that formed no same-cluster
pair has no `µ+_batch`, so `G+` keeps its global value while `G−` still moves —
pass `None` for a statistic the batch could not form:

```python
statistics = basis.update_distributions(
    positive_mean=positive_similarity.mean() if len(positive_similarity) else None,
    ...
    beta=self.beta,
)
```

This is not a corner case. A batch of `b` samples over `k` clusters holds only
about `C(b, 2) / k` same-cluster pairs — for the Cars run below, 28 out of 2016;
for In-shop, where `k` is in the thousands, well under one.

### 3.3 Mining (Sec 3.3.2, Eq 7)

The two Gaussian means become confidence thresholds: `T1 = µ+`, `T2 = µ−`.

```python
positive_mask = pair_similarity >= positive_threshold   # P = {s >= T1}
negative_mask = pair_similarity <= negative_threshold   # N = {s <= T2}
```

Mining is disabled until the Gaussians actually separate, because while
`µ+ <= µ−` the two bands overlap and every pair in between would qualify as both:

```python
def mining_thresholds(self):
    if not self.is_calibrated:
        return None
    ...
    if positive_mean <= negative_mean:
        return None
    return positive_mean, negative_mean
```

Two gates, and only the first is the paper's. `is_calibrated` is Eq 6, not Eq 7:
until a batch has seeded each distribution the buffers hold placeholder zeros, so
both thresholds would be `0.0` and would split the batch at an arbitrary cut.

`µ+ > µ−` is **an addition to Eq 7**, which sets `T1 = µ+` and `T2 = µ−` with no
condition on their order. It is kept because the overlapping case has no usable
behaviour, not because the paper asks for it: pytorch-metric-learning rejects a
duplicated pair outright (`GenericPairLoss._assert_either_pos_or_neg` raises, so
`unlabeled_ranking_loss: "supervised"` would abort the run), and
`unlabeled_ranking_loss: "contrastive"` would apply both of Eq 1's hinges to it,
pulling the pair together and apart on the same step. The
paper's own defence is the Sec 3.3.3 basis warm-up — `basis_warmup_steps` here —
and this guard covers the case where that warm-up was too short.

Its cost is that it suppresses *all* mining while the bands overlap, including
the confidently separated pairs outside the intersection. Watch
`train/slade/mining_active`: if it stays at zero well past the warm-up, the basis
is not separating, and the fix is `basis_warmup_steps` or `sd_margin`, not this
threshold.

Note what mining selects: **the pairs the model is already most confident
about.** That is the point of the method — and it is also the source of its main
failure mode (§5).

Eq 7 as written never looks at the pseudo labels, so a mined positive can
cross clusters. How often that happens depends on how the batch is built. With the
default uniform shuffle, a pseudo-positive pair is ~1/k of all pairs, and the
upper tail of the ~k times more numerous unrelated pairs clears `T1` too. With two
Gaussians d′ = 2 standard deviations apart and k ≈ 73, roughly 23% of mined
positives share a cluster, which matches the ~25% measured. Sampling 4 members of
each of 8 clusters per batch of 32 raises the pseudo-positive share to ~10% and
the same estimate to ~70%. The paper says "we use the pseudo labels to sample a
set of pseudo positive pairs and pseudo negative pairs", and its framework [21]
samples m per class. Neither is explicit about the unlabeled stream, and no
reference code is public. Two switches therefore test the other reading instead of
changing the default: `unlabeled_sampler: "pseudo_label_m_per_class"` and
`mining_pseudo_label_filter: true`. Both are in
`configs/ablations/slade_pseudo_label_sampling.json`.

### 3.4 The unlabeled ranking loss (Eq 9's middle term)

The mined pairs are fed to a ranking loss. The paper's framework "is generic and
applicable to different pair-based ranking losses"; they report contrastive and
multi-similarity. Two modes are available here:

* `unlabeled_ranking_loss: "supervised"` (default) — the configured metric loss
  is applied to the mined pairs, via a pytorch-metric-learning `indices_tuple`.
  This is the paper's genericity claim, and it keeps the labeled and unlabeled
  terms on the same loss (which the paper also does: "We use the same loss for
  both teacher and student networks").
* `unlabeled_ranking_loss: "contrastive"` — Eq 9's unlabeled term uses the
  paper's own Eq 1 instead of borrowing the supervised loss:

  ```text
  L_rank = Σ_{(i,j) ∈ P} max(d(i, j) − m_pos, 0)
         + Σ_{(i,j) ∈ N} max(m_neg − d(i, j), 0)
  ```

  This is exactly `ContrastiveLoss(pos_margin=m_pos, neg_margin=m_neg)`, whose
  `pos_calc` is `relu(d − pos_margin)` and `neg_calc` is `relu(neg_margin − d)`,
  over the default `LpDistance` — Euclidean distance on L2-normalized
  embeddings, which is the `d` Eq 1 is written on. Reusing the library loss also
  keeps this term's reduction identical to the labeled term's, so the paper's
  `λ1 = 1` still means "similar in scale" as Sec 3.3.3 intends; a literal sum
  over mined pairs against a mean-reduced labeled term would not be.

  **The margins are gates, not scales.** While a hinge is active its gradient
  w.r.t. the embedding is ±1 regardless of the margin, so a margin's *only*
  effect is which pairs it switches on (plus the `AvgNonZeroReducer`
  denominator). A margin sitting entirely inside the always-active or
  never-active regime is therefore a no-op: `m_pos = 0.0` and `m_pos = 0.4`
  produce bit-identical gradients when every mined positive is farther apart
  than both.

  That makes the geometry decisive. Embeddings are L2-normalized, so Eq 1's `d`
  relates to cosine as `d = sqrt(2 − 2·cos)`:

  | cos | 1.0 | 0.9 | 0.8 | 0.6 | 0.4 | 0.2 | 0.0 | −0.3 | −1.0 |
  |---|---|---|---|---|---|---|---|---|---|
  | `d` | 0.00 | 0.45 | 0.63 | 0.89 | 1.10 | 1.27 | 1.41 | 1.61 | 2.00 |

  Unrelated pairs pile up near `sqrt(2) ≈ 1.414`. Measured on the cached
  DINOv2-B Cars196 features (768-d, L2-normed) the pairwise median is `1.348`
  (cos `0.09`), with only 1.4% of pairs below `0.8` and 98.5% below `1.5`. So
  the negative hinge is dead below roughly `1.2` and saturated above roughly
  `1.55`; `make_equation_one_contrastive_loss` warns below
  `CONTRASTIVE_NEG_MARGIN_LIVE_FLOOR = 1.2`, because an unfired repulsion turns
  Eq 1 into attraction-only on pseudo-labeled pairs — the collapse mode.

  This is also why the library's own defaults (`pos_margin=0`, `neg_margin=1`)
  are not the right shipped values here despite being the standard contrastive
  setting: `neg_margin=1` fires on about 6% of pairs, and Eq 7 mines negatives
  from the *far* tail, so in practice it fires on far less than that.
  `paper_contrastive.json` uses `0.3 / 1.4` instead.

  **Margin orientation.** `m_neg > m_pos` is the satisfiable ordering only for a
  non-inverted distance. pytorch-metric-learning's guidance is that the default
  margins suit a non-inverted measure like `LpDistance`, and that an inverted
  one such as `CosineSimilarity` wants them swapped (`pos_margin = 1`,
  `neg_margin = 0`). Eq 1 is written on a distance, so the non-inverted default
  is the faithful choice; `make_equation_one_contrastive_loss` raises if the
  constructed loss's distance is inverted rather than letting the validation
  silently invert its meaning.

**Which losses each mode admits.** Proxy/classification losses (`ArcFaceLoss`,
`ProxyAnchorLoss`, …) score against learned class centers rather than pairs, so
they cannot consume a mined `indices_tuple`. Under `"supervised"`, which hands
`D_u` the configured loss, `validate_run_args` therefore rejects every
classification loss and every loss that cannot take a mined `indices_tuple`
before training starts.

`"contrastive"` lifts that: nothing hands the supervised criterion a pair tuple,
so any supervised loss is admissible. The `sampler_m >= 2` requirement is
dropped too for a classification loss — it exists so the *pair-based* labeled
term sees in-batch positives, which a proxy loss does not need. Note that using
one loss on both streams is what the paper does, so `"contrastive"` is best read
as a repository extension for pairing SLADE with a proxy objective.

**Pair direction matters.** Eq 7 defines `P` and `N` as sets of *unordered*
pairs, and `_mined_pairs` returns each one once from the upper triangle. Eq 1
read them that way — a symmetric per-pair hinge. A pytorch-metric-learning loss
does not: it scatters the tuple into an `n × n` mask and reduces **per anchor
row**, so an upper-triangular mask gives row `i` only its partners `j > i`, and
the last row none at all. Losses that need both a positive and a negative on the
same row (`CircleLoss`, `NTXentLoss`) drop the row outright when mining put the
pair's two ends on different rows. So the tuple is symmetrized, matching
`get_all_pairs_indices`, which is what these losses see when a caller passes
labels instead of a tuple:

```python
indices_tuple = (
    *_both_directions(positive_pairs),
    *_both_directions(negative_pairs),
)
return supervised_criterion(embeddings, cluster_labels, indices_tuple)
```

Measured on a 64-sample Cars batch over 73 clusters: 13 of 64 anchors reached the
loss before, 26 of 64 after — the mined signal was being halved.

### 3.5 Basis warm-up

The paper: "We first train the basis vectors for a few iterations to get a good
initialization, then train the student network and basis vectors end-to-end."
Implemented by detaching `f`, so only `W_a` moves:

```python
basis_warmup_active = self._optimizer_steps < self.basis_warmup_steps
labeled_features = (
    supervised_embeddings.detach() if basis_warmup_active else supervised_embeddings
)
```

Mining is also off during warm-up, so `L_rank(D_u)` is exactly zero. **Both
unlabeled terms therefore connect to the embedding on the same step**, which is
the transition to watch (§5.2). Keep `basis_warmup_steps` at "a few iterations" —
the default is 100.

#### `basis_warmup_steps` vs `basis_warmup_epochs`

Two alternative ways to length the same warm-up, the same pairing
`unlabeled_batch_size` / `unlabeled_ratio` uses. Null for both keeps the
historical default of 100 steps; setting both warns and the absolute step count
wins.

Steps are what the warm-up actually runs on: `W_a` moves once per optimizer step,
and Eq 6's statistics track the live basis with a lag of roughly `1/(1 - beta)`
steps. Neither depends on the batch size.

The number of steps in an *epoch* does. One epoch is the fold training pool
divided by `batch_size` (`override_length_before_new_iter_from_fold`), so on Cars
with `cv_k = 4` an epoch is 201 steps at batch 32 and 25 at batch 256. An
absolute step count therefore means something different in every trial of a study
that searches `batch_sampler`:

| batch | steps/epoch | `basis_warmup_steps = 100` | `basis_warmup_epochs = 0.5` |
| --- | --- | --- | --- |
| 32 | 201 | 0.5 epochs | 100 steps |
| 64 | 100 | 1.0 epochs | 50 steps |
| 128 | 50 | 2.0 epochs | 25 steps |
| 256 | 25 | 4.0 epochs | 12 steps |

That spread is not merely untidy. Warm-up epochs are ordinary supervised training
that still consumes early-stopping patience — `is_after_warmup` exempts only
`ssl_config.warmup_epochs`, not this one — so a long warm-up at a large batch can
end the trial before either unlabeled term has reached the model, and the trial
reports a score under SLADE's name without SLADE having run.

`basis_warmup_epochs` is the batch-size-independent form, and it is an ordinary
searchable key: put it in `spaces` like any other. `set_steps_per_epoch` resolves
it to a step count per fold, once the sampler epoch length is known, and logs the
arithmetic. This is the same move `labeled_batch_fraction` makes for
`labeled_batch_size` — a value that means the same thing at every `batch_sampler`
choice never has to be filtered against one.

To search it, the ssl config must leave `basis_warmup_steps` null, or the
absolute count wins and the search is inert. `configs/runs/ssl/slade/initial.json`
is `paper.json` with exactly that change.

The same batch-size reasoning bounds `beta` from above. At `beta = 0.999` the lag
is ~1000 steps, so mining would open against a basis a thousand steps stale no
matter what warm-up is set. The `_initial` spaces cap it at the paper's `0.99`.

### 3.6 `sd_gradient_mode`: keeping `beta` out of Eq 5's gradient

**The paper specifies neither mode.** Eq 6 is written as a *value* update; the
paper never discusses gradients, backpropagation, or where the graph is cut, and
neither the CVPR text nor the supplementary (which is only t-SNE figures) says
anything about it. No official implementation appears to have been released. So
this is an underspecified detail the code has to resolve, not a place where one
setting reproduces the paper and the other departs from it.

What forces the choice is that the running estimate *must* be detached — keeping
it in the graph would grow the graph without bound across the whole run. Once it
is detached a factor appears, and the only question is whether to compensate.

Eq 6 blends a batch statistic with a stored one, and the stored half is history
that carries no gradient. Composed literally, every gradient path from Eq 5 into
`W_a` therefore picks up a factor of `(1 - beta)`:

```text
dL_CE/dW_a  ->  full scale
dL_SD/dW_a  ->  (1 - beta) * d(batch statistic)/dW_a
```

Eq 2 sums those two terms, so **`beta` sets the `L_CE : L_SD` ratio inside
`L_Basis`** as well as the updating rate. Nothing downstream can undo it:
`extra_loss_components` reports `L_CE + L_SD` as a single number, so `lambda2`
(and GradNorm, when it learns `lambda2`) scales both together. At the paper's
`beta = 0.99` the similarity-distribution term pulls on the basis with 1% of the
cross-entropy's scale — which sits badly with the paper's own Table 5, where the
SD form beats local-CE by 0.9 and global-CE by 1.4 MAP@R. A term worth a point of
MAP@R is not a term contributing 1% of the gradient.

That argument is suggestive rather than conclusive: `W_a` is trained *only* by
`L_Basis`, so a small but persistent gradient still accumulates over thousands of
steps, and Eq 5's hinge is a directional pressure rather than a magnitude-matched
term.

`sd_gradient_mode` picks the reading. `"damped"` is what falls out of
transcribing Eq 5 and Eq 6 naively — most reimplementations would produce it
without noticing. `"full"` is what Eq 5 arguably *means*: it is defined over the
global Gaussians, and the `(1 - beta)` is an artifact of where the graph was cut
rather than anything the paper asks for. Differentiating the estimator at its
fixed point also gives full scale, since at steady state the global mean tracks
the batch mean in expectation.

| | Forward value of `µ⁺` | Gradient scale | What `beta` means |
| --- | --- | --- | --- |
| `"damped"` (default) | interpolated global | `(1 - beta)` | updating rate **and** an L_SD weight |
| `"full"` | interpolated global | 1 | updating rate only |

`"full"` adds `beta * (x - x.detach())` to the blend. That term is exactly zero in
the forward pass and has derivative `beta` in the backward one, so `(1 - beta) +
beta = 1` restores full scale while the value Eq 5 reads — and the buffers Eq 7
mines against — stay bit-identical. The Gaussians remain global, which is the
paper's stated reason for using them.

`"damped"` is the default so that new runs stay comparable with the studies in
`exporter/optuna-studies/`, all of which predate this option. Be aware of what
that costs: those studies searched `beta` over `[0.3, 0.99]` (Cars) and
`[0.3, 0.999]` (CUB), which under `"damped"` is also a sweep of L_SD's weight
across 70x and 700x respectively, confounded with `sd_variance_weight` and
`lambda_basis`. Switch to `"full"` for a run meant to isolate `beta` as the
updating rate, and do not mix the two modes inside one study.

### 3.7 Per-iteration resets

Sec 3.3.3 ends a self-training iteration by promoting the student to teacher and
starting the next one. Two pieces of SLADE's own state would otherwise cross that
boundary unchanged, and each has its own switch because the paper does not say
whether the next student inherits them:

- **`reset_basis_on_refresh`** re-initializes `W_a` and clears both Gaussians. The
  stored `µ⁺`/`µ⁻` summarize similarities under the *previous* iteration's cluster
  IDs, and Eq 7 turns them straight into mining thresholds — so carrying them past
  a re-clustering mines the new pseudo-labels against a stale notion of a
  confident pair. The re-initialized parameter is queued and the trainer drops its
  optimizer state (`_apply_pending_regularizer_optimizer_resets`), because Adam's
  moments were fitted to the basis that was just discarded.
- **`reset_basis_warmup_on_refresh`** resets the optimizer-step counter so
  `basis_warmup_steps` runs again. Without it the `W_a`-only warm-up fires during
  the first student phase and never again, so every later iteration starts
  end-to-end.

Both apply only to refreshes *after* the first: the first clustering feeds a
student that has not trained yet, so there is nothing to reset. Both require a
schedule that actually re-clusters; combined with `update_mode: "once"` they are
rejected at validation rather than silently ignored.

They are independent of `within_fold_teacher_student`, which pins
`update_mode: "once"` and so cannot currently be combined with them — that mode
makes one explicit teacher/student boundary, these make a boundary at every
iteration of a refreshing schedule.

---

## 4. This project's setup: frozen backbone, cached features

Every run here uses a **frozen DINOv2 ViT** and trains only the projection head,
with backbone features precomputed and cached. SLADE opts into that path:

```python
class SladeRegularizer(BaseTrainingRegularizer):
    name = "slade"
    # One deterministic view per unlabeled image is enough, so the frozen
    # backbone feature cache stays usable.
    supports_frozen_feature_precompute = True
    uses_joint_forward = True
    requires_supervised_objective = True
```

The example experiment config is `configs/runs/experiments/slade_cars.json`:

```json
{
  "dataset": "Cars196",
  "backbone_tuning": "frozen",
  "use_cache": true,
  "feat_dim": 768,
  "mode": "ssl",
  "batch_size": 32,
  "sampler_m": 4,
  "ssl_config": "configs/runs/ssl/slade/paper.json"
}
```

Three consequences follow, and none of them is a choice about SLADE:

**No dataset-specific pre-training stage.** Sec 3.1 initializes the teacher with
SwAV and fine-tunes it on labeled + unlabeled images. Here every run starts from
the same frozen DINOv2 features. Neither the paper's ImageNet rows nor its SwAV
rows are directly comparable: both use an end-to-end tuned ResNet-50, while the
SwAV rows additionally include the union fine-tuning stage.

**The teacher is a projection-head snapshot, not a second DINO model.** SLADE
consults the teacher only when it embeds the pool for k-means, unlike STML's EMA
teacher on every step. With a frozen backbone, a saved projection head plus the
shared DINO features is the complete teacher. The default schedule represents
promotion by re-clustering with the current student. The opt-in
`paper_teacher_student.json` schedule makes the within-fold stage boundary
explicit and resets optimization before student training. Only
`paper_cross_fold_teacher_student.json` saves the selected student head and
restores it as the next fold's teacher.

**Cluster assignments drift less.** k-means runs on frozen-backbone features, so
the teacher can only move the pool's embedding as far as the head allows.

---

## 5. Failure modes worth knowing

### 5.1 The mined pairs are, by construction, the ones the loss has already solved

Eq 7 selects high-confidence pairs. A ranking loss is a hinge. A hinge on a pair
that already satisfies its margin is exactly zero — including the paper's own
Eq 1, `max(d − m_pos, 0)`. **A zero gradient from the unlabeled ranking term is
expected behaviour, not a bug.** The term fires on mining's mistakes and on pairs
near the boundary; when the student has fully separated the mined set, it
correctly contributes nothing until the next re-clustering moves the thresholds.

What turns that into a problem is loss scale. `CircleLoss` computes
`softplus(LSE_pos + LSE_neg)` — the two exponents **add**, so both sides being
satisfied drives the loss down twice as fast as either alone. On a batch where
every mined pair is already satisfied (`s_pos ≈ 0.83`, `s_neg ≈ 0.06`, `m = 0.47`):

| `gamma` | loss | `|grad|` |
| --- | --- | --- |
| 32 | 2.2e-05 | 2.1e-04 |
| 64 | 7.7e-12 | 1.1e-10 |
| 128 | 4.1e-22 | 2.7e-20 |
| 265 | 1.4e-44 | 0 |
| 512 | 0 | 0 |

Read this as the *converged* regime, not the starting one — on a realistic
mid-training batch the same `gamma = 265` still yields a loss of 4.3e-03 and a
gradient of 1.14. The table says what happens once the student has satisfied the
mined set: above `gamma ≈ 64` there is no gradual decay left, only a cliff into
float32 underflow. `pytorch_metric_learning`'s own docstring recommends
`m = 0.4, gamma = 80` for fine-grained image retrieval; `gamma = 256` is its
face-recognition setting. `configs/runs/slade/cars_paper.json` searches 32–128
for this reason.

The same cliff exists for `NTXentLoss` at low temperature.

### 5.2 The end of basis warm-up is a step change

At `_optimizer_steps == basis_warmup_steps`, `f` stops being detached and mining
turns on, so both unlabeled terms start reaching the model **on the same step**.
Measured on a Cars trial (`lr = 4.9`, `gamma = 265`, `basis_warmup_steps = 500`):
the basis gradient arrived at 4245× the supervised one and the ranking gradient
at 20×, batch loss went 14 → 7725, and validation MAP@R fell 57.1 → 29.9 within
one epoch. Keep `basis_warmup_steps` short, and keep the learning rate in a range
where a single out-of-scale step is survivable.

### 5.3 Gradient-ratio calibration and small gradients

`regularizer_target_ratio` sets `regularizer_weight` from the running **median**
of 20 sample-scoped `supervised / regularizer` gradient-norm probes in every
10,000-sample window by default. The shared target-ratio update and window
sample parameters can change both cadences. SLADE's ranking gradient can be
large on batches where mining catches a violation and tiny or zero when the
mined pairs are already separated.

Calibration accepts every finite positive component norm, with no minimum
component/supervised ratio and no cap on the resulting finite weight. The
former 1e-6 ratio floor and exception after 50 unusable probes have been removed
at the user's request. This applies to the ranking and basis terms alike.
Exactly zero or non-finite norms defer calibration and leave the previous
weight in place; they do not end a trial after a fixed number of batches.

Fixed weights remain available by omitting the target ratios. `grad_norm_alpha`
selects GradNorm instead and supersedes target-ratio calibration. Existing HPO
trials marked PRUNED retain their status until explicitly retried; removing the
guard does not rewrite study history.

---

## 6. Configuration reference

### Files

| File | Purpose |
| --- | --- |
| `configs/runs/ssl/slade/paper.json` | The published method: `λ1 = 1`, `λ2 = 0.25`, `β = 0.99`, no calibration |
| `configs/runs/ssl/slade/paper_teacher_student.json` | Independent teacher/student stages: train teacher, cluster once, reset optimization, train/select student |
| `configs/runs/ssl/slade/paper_cross_fold_teacher_student.json` | The same stages plus the paper's sequential student-to-next-fold-teacher handoff |
| `configs/runs/ssl/slade/calibrated.json` | Same, plus this project's gradient-ratio weight calibration |
| `configs/runs/ssl/slade/initial.json` | `paper.json` with `basis_warmup_steps` null so `basis_warmup_epochs` can be searched |
| `configs/runs/slade/cars_paper.json` | HPO space that leaves `λ1`/`λ2` at the paper's values |
| `configs/runs/slade/cars_teacher_student.json` | Teacher/student HPO space: warm-up is always positive and the inert refresh interval is not searched |
| `configs/runs/slade/{cars,cub,in_shop,semi_inat_2021}_initial.json` | Current HPO spaces. One shared SLADE search space per dataset base: GradNorm instead of target ratios, `beta` capped at 0.99, the §5.1 loss ranges narrowed, and the warm-up searched in epochs |
| `configs/runs/slade/{cars,cub,in_shop}.json` | Older HPO spaces |
| `configs/runs/experiments/slade_cars.json` | Runnable Cars-196 experiment |
| `configs/runs/experiments/slade_cars_teacher_student.json` | The Cars experiment with independent teacher/student stages per fold |
| `configs/runs/experiments/slade_cars_cross_fold_teacher_student.json` | Fixed-setting Cars run with the additional sequential fold handoff; HPO is disabled because its fold scores are order-dependent |

### Parameters

| Key | Default | What it does |
| --- | --- | --- |
| `method_params.regularizer_weight` | `1.0` | λ1 — fixed weight of `L_rank(D_u)`, or its starting weight when GradNorm is enabled |
| `lambda_basis` | `0.25` | λ2 — weight of `L_Basis` |
| `beta` | `0.99` | Eq 6 updating rate; weights the **running** estimate, so higher = more global |
| `sd_gradient_mode` | `"damped"` | How much of Eq 5's gradient reaches `W_a`. `"damped"` is the literal Eq 6 composition, in which the gradient is scaled by `(1 - beta)`; `"full"` removes that factor so `beta` is the updating rate only. See §3.6 |
| `sd_margin` | `0.5` | Eq 5 margin `m`. Not specified by the paper |
| `sd_variance_weight` | `1.0` | Eq 5 `λ` on the variance penalty. Not specified by the paper |
| `basis_warmup_steps` | `100` | "A few iterations" of `W_a`-only training, as an absolute optimizer-step count. See §3.5 and §5.2 |
| `basis_warmup_epochs` | `null` | The same warm-up as a fraction of one sampler epoch, resolved to steps per fold. Batch-size independent, so it is the form to search. Ignored when `basis_warmup_steps` is also set |
| `num_clusters` | `null` | k for k-means, as an absolute count. Defaults to the labeled class count — **set explicitly for an external pool** (145 for filtered CompCars, 400 for NABirds) |
| `cluster_ratio` | `null` | The same k relative to the C labeled classes: `0.5` is `k = C/2`, `2.0` is `k = 2C`, and the default is the ratio `1.0`. Dataset-independent, so it is the form to search. Rounded to the nearest integer, floored at 2 with a warning, and ignored when `num_clusters` is also set |
| `unlabeled_ranking_loss` | `"supervised"` | `"supervised"` applies the configured loss to the mined pairs; `"contrastive"` uses the paper's Eq 1 loss instead, which frees the supervised half to be a proxy/classification loss. See §3.4 |
| `contrastive_pos_margin` | `0.0` | Eq 1 `m_pos`, a Euclidean distance on L2-normalized embeddings (`d = sqrt(2 − 2·cos)`, range `[0, 2]`). Only read when `unlabeled_ranking_loss: "contrastive"`; not specified by the paper. `paper_contrastive.json` sets `0.3`; see §3.4 for why the constructor default is not a good run value |
| `contrastive_neg_margin` | `1.0` | Eq 1 `m_neg`. Must exceed `contrastive_pos_margin`. **Below ~1.2 the repulsion never fires** on L2-normalized embeddings and the term becomes attraction-only; a warning is logged. `paper_contrastive.json` sets `1.4`. Not specified by the paper |
| `unlabeled_batch_size` | `null` | Absolute size of the unlabeled batch drawn per step, independent of `batch_sampler`. Clamped to the pool size with a warning; wins over `unlabeled_ratio` when both are set |
| `unlabeled_ratio` | `1.0` | Relative alternative: unlabeled batch size as a multiple of the labeled one. The paper uses 32 for both. `null` for both keys falls back to a ratio of 1.0 |
| `unlabeled_sampler` | `"random"` | `"random"` shuffles the pool uniformly, so only ~1/k of batch pairs share a cluster. `"pseudo_label_m_per_class"` draws `unlabeled_batch_size / unlabeled_sampler_m` clusters with `unlabeled_sampler_m` members each, which is the m-per-class sampling of the framework the paper builds on. Clusters smaller than m are never drawn, so no image is duplicated. With fewer clusters than slots, a cluster fills several. The step count per epoch is unchanged. Ablation: `configs/ablations/slade_pseudo_label_sampling.json` |
| `unlabeled_sampler_m` | `4` | Members per cluster under `"pseudo_label_m_per_class"`. The unlabeled batch size must be divisible by it |
| `mining_pseudo_label_filter` | `false` | `false` is Eq 7 read literally: basis similarity alone, so mined pairs may disagree with the clusters (~25% of mined positives shared one). `true` keeps a mined positive only within a cluster and a mined negative only across clusters, with the same thresholds. The mining diagnostics count the filtered pairs |
| `pair_source` | `"basis_mining"` | Which unlabeled pairs the ranking term trains on. `"basis_mining"` is Eq 7. `"pseudo_label"` trains on every same-cluster pair as a positive and every cross-cluster pair as a negative (Table 4's pseudo-label rows; add `lambda_basis: 0` for the row without the basis). `"pseudo_label_matched"` draws pseudo-label pairs at random, as many as Eq 7 mines on the same step. All three wait for the basis warm-up, and the mining diagnostics count the pairs actually used. See §7 |
| `basis_unlabeled_loss` | `"sd"` | The unlabeled half of Eq 2 (Table 5). `"sd"` is Eq 5, `"local_ce"` a per-pair cross entropy against the pseudo labels, `"global_ce"` the same cross entropy on the Eq 6 means, `"none"` leaves Eq 3 alone. The cross entropies read a cosine as the probability `(1 + s) / 2` and average the two pair kinds separately; the paper specifies neither. `train/slade/similarity_distribution_loss` reports whichever term is active. See §7 |
| `reset_basis_on_refresh` | `false` | Re-initialize `W_a` and clear the Eq 6 Gaussians at every re-clustering after the first, so each self-training iteration gets a fresh basis. Requires a refreshing `update_mode`. See §3.7 |
| `reset_basis_warmup_on_refresh` | `false` | Restart the `basis_warmup_steps` counter at every re-clustering after the first, so each iteration re-runs the `W_a`-only warm-up. Requires a refreshing `update_mode`. See §3.7 |
| `within_fold_teacher_student` | `false` | With `update_mode: "once"` and labeled warm-up, make a real teacher/student stage boundary: retain teacher weights, reset optimizer/scheduler, fix pseudo-labels, and select only student epochs |
| `cross_fold_teacher_handoff` | `false` | Also initialize the next fold's teacher from this fold's selected student projection; requires `within_fold_teacher_student`, multiple folds, and a frozen backbone |
| `logit_scale` | `1.0` | Scale on Eq 3's logits. Not in the paper; `1.0` is the paper |
| `kmeans_backend` | `"auto"` | `faiss` when available, else `sklearn` |
| `kmeans_gpu` | `true` | FAISS GPU k-means. `true`/`false` force the choice; `null` defers to the size rule below. Any GPU failure falls back to the CPU index |
| `update_interval_epochs` | — | Length of one self-training iteration (§3.1) |
| `warmup_epochs` | — | Labeled-only epochs before SLADE starts; the Sec 3.1 fine-tuning role |

### k-means on the GPU

`kmeans_cluster_labels` attaches a FAISS GPU flat index. `kmeans_gpu` defaults to
`true`, so the GPU is used whenever one is visible; setting it to `null` defers to
a size rule instead — `len(features) * num_clusters`, the size of the assignment
problem each Lloyd iteration solves, against `DEFAULT_KMEANS_GPU_MIN_WORK` (10^6).

Under `low_memory_mode` (the run default) each call reserves 256 MiB of device
scratch for the duration of one refresh. Refreshes are short and infrequent, so
concurrent trials rarely overlap in them; without `low_memory_mode` FAISS would
reserve its own ~1.5 GiB per caller instead, which at `max_concurrent_per_gpu: 10`
is worth avoiding.

Measured on an RTX 5070 Ti against a **single-threaded** FAISS CPU build, which
is what a trial actually gets: `num_threads=1` is the run default and several
trials share a GPU.

| Setting | n | k | n·k | CPU (1 thread) | GPU | Speed-up |
| --- | --- | --- | --- | --- | --- | --- |
| CUB / Cars, 16-shot | 3,392 | 80 | 0.27M | **0.09 s** | 0.17 s | 0.5x |
| — | 5,000 | 200 | 1.0M | 0.38 s | 0.19 s | 2x |
| — | 8,000 | 500 | 4.0M | 1.23 s | 0.18 s | 7x |
| — | 10,000 | 1,000 | 10M | 3.52 s | 0.27 s | 13x |
| In-Shop, 2-shot, per fold | 13,500 | 3,000 | 40M | 18.35 s | 0.58 s | **32x** |
| In-Shop, all labels | 25,882 | 3,997 | 103M | 61.34 s | 0.72 s | **85x** |

The GPU has a flat ~0.17 s floor while the CPU grows linearly in `n·k`, so the
crossover sits near `n·k = 5·10^5`. CUB and Cars are two orders of magnitude
below it — k-means there is 0.09 s and not worth a device transfer. In-Shop is
two orders above it, and at `update_interval_epochs: 1` the CPU cost compounds:
~40 refreshes per fold x 4 folds x 18 s is roughly 48 minutes of pure clustering
per trial, against under two minutes on the GPU.

Objectives agree to five significant figures and the assignments are identical at
CUB scale (99.7% agreement at In-Shop scale, from float ordering inside Lloyd
iterations).

The index is built through `faiss_gpu_flat_index`, not `faiss.Kmeans(gpu=True)`.
The latter routes through `index_cpu_to_all_gpus`, which builds its own
`StandardGpuResources` that the SSL low-memory policy cannot size — measured at
~1.25 GiB of extra device memory per caller, which at `max_concurrent_per_gpu: 10`
would exhaust the card on its own.

`ssl_device: cpu` suppresses the GPU path entirely, through the same
`_configured_ssl_device` check the other FAISS helpers use.

### Diagnostics

With `log_batch_diagnostics` on, each batch reports `train/slade/…`:
`cross_entropy_loss`, `similarity_distribution_loss`, `unlabeled_ranking_loss`,
`positive_mean` / `negative_mean` / `positive_variance` / `negative_variance`
(the two Gaussians), `mined_positive_pairs` / `mined_negative_pairs`,
`pseudo_positive_pairs`, `basis_warmup_active`, `mining_active`.

`mined_positive_pairs` dropping to zero while `mining_active` is 1 means the
thresholds have moved past what the batch contains — check it before concluding
the term is broken.

`ssl_gradient_contribution_log_interval` additionally reports
`regularizer_norm` (the ranking term weighted by `regularizer_weight`) and
`slade_basis_norm` (the basis term weighted by `lambda_basis`) separately, which
is why the two are not summed inside `compute_loss`.

---

## 7. The paper's ablations and the Phase-2 controls

`python main_eval/make_phase2_ablations.py paper --dataset <ds> --combo slade_<loss>`
rebuilds Sec. 4.5 of the paper on a tuned study, one change per variant:

| Paper | Variant | Change |
| --- | --- | --- |
| Table 4, Teacher | `zero_weight` (controls) | no unlabeled term, no basis |
| Table 4, Student (pseudo label) | `t4_pseudo_label` | `pair_source: "pseudo_label"`, `lambda_basis: 0` |
| Table 4, + Basis | `t4_pseudo_label_basis` | `pair_source: "pseudo_label"` |
| Table 4, + Basis + Mining | the replay itself | — |
| — (Phase-2 plan B5) | `t4_matched_count` | `pair_source: "pseudo_label_matched"` |
| Table 5, local-CE / global-CE | `t5_local_ce`, `t5_global_ce` | `basis_unlabeled_loss` |
| — | `t5_ce_only` | `basis_unlabeled_loss: "none"` |
| Table 6, number of clusters | `t6_cluster_ratio_<r>` | `cluster_ratio` over `--cluster-ratios` |
| Sec. 4.3 settings | `paper_beta`, `paper_unlabeled_batch`, `paper_weights` | `beta: 0.99`; batch 32; fixed `λ1 = 1`, `λ2 = 0.25` |
| Sec. 3.3.3 | `paper_no_basis_warmup`, `paper_teacher_student`, `paper_new_student_per_iteration` | no `W_a`-only warm-up; one teacher clustering and a fresh student; a fresh basis at every re-clustering |

Table 3 (teacher initialization) has no counterpart: the backbone is a frozen
DINOv2 for every run (§4).

`t4_pseudo_label_basis` differs from the method in both the number of pairs and
which pairs, since Eq 7 keeps fewer and easier pairs. `t4_matched_count` keeps
Eq 7's count per step and draws that many pseudo-label pairs at random, so
comparing it with each neighbour separates the two.

`controls` adds, for SLADE, `clusters_once` (`update_mode: "once"`: the first
teacher's clusters are kept for the whole run) and `oracle_noise_<p>`
(`pseudo_label_source: "oracle"`). The oracle replaces k-means with the hidden
classes of the pool, so k becomes the pool's class count, and moves a seeded,
nested p % of the images into a wrong class. Like LP's oracle it is diagnostic
only.
