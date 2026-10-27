# Hoffer & Ailon: the paper, the released code, and this repository

This document compares three things — the paper *Semi-supervised deep learning by
metric embedding*, Elad Hoffer's own released implementation, and
`HofferEntropyRegularizer` in this repository — in plain language, and then
answers a specific question: **do we actually need `distance_scale`?**

| Role | Source |
| --- | --- |
| Paper | Elad Hoffer and Nir Ailon, [arXiv:1611.01449v2](https://arxiv.org/abs/1611.01449). Equation numbers below are from the arXiv LaTeX source (`semisup.tex`) |
| Released code | [github.com/eladhoffer/SemiSupContrast](https://github.com/eladhoffer/SemiSupContrast) — `ContrastingMinEntropyCriterion.lua`, `main.lua`, `models/MNIST_contrast.lua` |
| Our code | [`training/semi_supervised.py`](../training/semi_supervised.py) (`HofferEntropyRegularizer`), [`training/ssl/data.py`](../training/ssl/data.py) (`HofferReferenceDataset`, `HofferReferenceBatchSampler`) |
| Our configs | [`configs/ssl_methods/hoffer.json`](../configs/ssl_methods/hoffer.json), [`configs/runs/hoffer/`](../configs/runs/hoffer/) |
| Benchmark | [`scripts/bench_hoffer_distance_scale.py`](../scripts/bench_hoffer_distance_scale.py) |

## Bottom line

- **The released code does not match its own paper** in several places — squared
  vs. unsquared distance, which images the entropy is applied to, and the batch
  composition. Where the two disagree, **this repository follows the paper.**
- Our genuine deviations from the paper are three, and one causes another: the
  supervised term `L_L` is replaced by a modern DML loss (the methodological core
  of this project); embeddings are L2-normalized (forced by our pipeline); and
  `distance_scale` exists **only** to restore the sharpness that normalization
  removes.
- **Yes, we need `distance_scale`.** Pinned at the paper's implicit `1.0` the
  entropy term measured −1.3 points of R@1 against a supervised-only control on
  Cars196; the released code's `-d` at `s = 1` was inert to within 0.1 points.
  Tuned to `s = 8` the same term gains +2.0 points. All figures are means over 3
  seeds with a per-seed spread under 1 point.
- It is **not** a duplicate of `regularizer_weight`, and GradNorm cannot replace
  it: arms holding `weight · scale²` constant span 10 points of R@1 and change
  sign, while gradient magnitude is the only thing GradNorm controls.
- The searched range is nonetheless wrong. The useful region is roughly
  `[4, 16]`; below that the term is not weak, it is **actively harmful**
  (−4.9 points at `s = 2`).
- `d` vs `d²` is only a reparametrization of that scale — measured, not assumed.

## The method in one paragraph

Pick one labeled image per class — the **references** `z_1..z_c` — and turn the
distances from an image `x` to those references into a softmax (Eq. 1). That
distribution says which class `x` looks like, under a 1-NN rule. The paper then
writes two losses over the *same* distribution: for a labeled image, the
cross-entropy against its true class (Eq. 3, the "distance ratio criterion"); for
an unlabeled image, the Shannon entropy (Eq. 5, the "minimum entropy criterion").
The total objective is their weighted sum (Eq. 6):

```
L(x_l, x_u, {z_1..z_c}) = λ_L · L(x_l, z_1..z_c)_L  +  λ_U · L(x_u, z_1..z_c)_U
```

We port Eq. 5 as a regularizer and fill Eq. 3's role with a configured
metric-learning loss.

## Three-way comparison

| Aspect | Paper | Released code | This repository |
| --- | --- | --- | --- |
| Softmax logit | `-‖F(x) − F(z_i)‖²` — **squared**, Eq. 1 | `-d`, **unsquared** (`nn.PairwiseDistance(2)` then `MulConstant(-1)`) | `-distance_scale · d²` — matches the paper, plus a temperature |
| Entropy applied to | `x_u`, the unlabeled sample (Eq. 6); §Experiments: "the remaining training images are used without their labels to form `X_U`" | the **whole** training set — `main.lua` captures only `extractEachTarget`'s first return value and iterates `trainData` | `split.unlabeled_positions` only — matches the paper |
| Objective structure | two separately weighted terms, `λ_L L_L + λ_U L_U` (Eq. 6) | both terms computed from one shared log-softmax matrix in one criterion | `supervised_weight · L_sup + regularizer_weight · H` — matches the paper |
| Batch composition | `2b + |C|`: a labeled batch of `b`, an unlabeled batch of `b`, and one reference per class | `b + |C|·(numCompared+1)`, so the supervised term sees only `|C|` labeled queries per step, not `b` | a supervised batch, an unlabeled batch, and `reference_sets × C` references — matches the paper |
| References `z` | one per class, "sampled in uniform over all available samples for each class", redrawn every iteration | same | same |
| Supervised term `L_L` | Eq. 3: cross-entropy over the **same** reference softmax | same | **replaced** by any configured PML loss (MultiSimilarity, ArcFace, NTXent, Circle, ProxyAnchor) |
| Embeddings | unnormalized conv-net output | unnormalized | **L2-normalized** (`DinoWrapper.project_features`) |
| Temperature | none — Eq. 1 has no scale | none | **`distance_scale`** |
| Weights | `λ_L, λ_U ∈ [0,1]`; experiments use `λ_1 = λ_2 = 1` | both 1, implicit | searched, or learned by GradNorm |
| Warmup | none | none | `warmup_epochs` supervised-only epochs |
| Backbone | 5-layer conv net trained from scratch | same | frozen DINOv2 ViT + trainable linear head |
| Class count | 10 (MNIST, SVHN) | 10 | 98 (Cars196), 100 (CUB), 3997 (In-Shop) |
| Labeled budget | 100 images total on MNIST = 0.17% of the training set | 10/class | k-shot: 19% of Cars196, 27% of CUB |
| Evaluation | k-NN classification against `X_L` | 1-NN classification | retrieval (MAP@R, R@k) |
| Precision | Torch float | Torch float | entropy forced to `float32` even under autocast |

### The loss, side by side

Paper, Eq. 1 and Eq. 5, verbatim from `semisup.tex`:

```latex
P(x;z_1,...,z_c)_i = \frac{e^{-\|F(x)-F(z_i)\|^2}}{\sum_{j=1}^c e^{-\|F(x)-F(z_j)\|^2}}

L(x, z_1,...,z_c)_U = -\sum_{i=1}^c  \frac{e^{-\|F(x)-F(z_i)\|^2}}{\sum_{j=1}^c e^{-\|F(x)-F(z_j)\|^2}}
                      \cdot \log{\frac{e^{-\|F(x)-F(z_i)\|^2}}{\sum_{j=1}^c e^{-\|F(x)-F(z_j)\|^2}}}
```

The notation is deliberate: the author writes `\|·\|_2` (subscript — the plain L2
norm) in objectives (i) and (ii), and `\|·\|^2` (superscript — squared) inside the
softmax of Eq. 1, 3 and 5. Rendered PDF text loses that distinction; the source
does not.

Ours (`HofferEntropyRegularizer.compute_loss`):

```python
squared_distances = (unlabeled[None, :, None, :] - reference_sets[:, None, :, :]).pow(2).sum(dim=-1)
log_probabilities = F.log_softmax(-self.distance_scale * squared_distances, dim=-1)
entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
return entropy.mean()
```

Released code (`ContrastingMinEntropyCriterion.lua`, `__rebuildModule`):

```lua
self.module:add(nn.PairwiseDistance(self.dist))   -- d = ||F(x) - F(z)||, dist = 2
self.module:add(nn.MulConstant(-1, false))        -- logit = -d, NOT -d^2
```

### Where the released code departs from its own paper

Worth knowing, because "the reference implementation does X" is not by itself a
reason to do X.

1. **`-d` instead of `-d²`.** `nn.PairwiseDistance(2)` is the L2 norm, not its
   square. Eq. 1 squares it. The code's form is a monotone reparametrization of
   the paper's, so it is not *wrong*, but it is not Eq. 1.
2. **Entropy on labeled images.** `main.lua` calls
   `local labeledData = extractEachTarget{dataset=trainData, num=opt.numLabeled}`,
   discarding the `remainder` return value, then iterates `trainData` — the full
   training set. The paper's §Experiments says `X_U` is the remainder.
3. **Batch composition.** The paper's iteration is `2b + |C|` samples: a labeled
   batch of `b`, an unlabeled batch of `b`, and `|C|` references. The code uses
   `b + |C|·(numCompared+1)`, so its supervised term sees only `|C|` = 10 labeled
   queries per step instead of `b` = 32.
4. **`numCompared > 1` does not run.** `main.lua:286` copies
   `torch.range(1, numClasses)` — `C` values — into a slice of `C · numCompared`
   elements, which only matches at the default `numCompared = 1`. So the released
   code is effectively fixed at exactly one reference per class, as in the paper.

Note also that `__rebuildModule` offers `dist = 'dot'` and `dist = 'cosine'` as
alternatives, and those branches deliberately skip the `MulConstant(-1)`. On unit
vectors both return `cos`, which — see deviation 3 below — is exactly our
objective at `distance_scale = 0.5`.

## Our deviations from the paper

Three, and the third exists only because of the second.

### 1. The supervised term `L_L` is replaced by a modern DML loss

Eq. 6 has two separately weighted terms and so do we — that structure is not a
deviation. What we change is the *content* of `L_L`. The paper's Eq. 3 is a
cross-entropy over the **same** `P(x_l; z_1..z_c)`, i.e. an NCA-style distance
ratio loss over the same references. We fill that role with whatever
`pytorch-metric-learning` loss the run is configured with, on its own m-per-class
batch.

This is the methodological core of the project, not an approximation: the
classical SSDML formulation is decomposed into *supervised term + unlabeled
mechanism + degeneracy constraint*, and the supervised term is the component
being replaced by a modern DML loss. Eq. 5 is the unlabeled mechanism being
ported; Eq. 3 is the component being substituted **on purpose**. Reinstating
Eq. 3 would mean hard-wiring a second supervised loss inside the regularizer,
competing with the configured one, and would defeat the comparison across losses
the study is built on.

One consequence, stated precisely because it is easy to overstate:

- In the paper, `L_L` reads the same softmax as `L_U`, so the references `z` are
  **directly** supervised.
- With us, within a regularizer batch the reference embeddings receive gradient
  only from the entropy term. Across the run they are still shaped by the labels
  — references, unlabeled images and the supervised batch all pass through the
  *same* projection head, which the supervised loss trains on every step — but
  they are not corrected inside the same term, in the same step.

The cost of that indirection is measurable, and it is measured below as
`argmax_purity`: the fraction of unlabeled images whose nearest reference really
is their own class. It reaches ~0.72 on Cars196 after 1000 supervised steps and
~0.22 at initialization. That number is what bounds how confident the entropy
term should be allowed to become — which is exactly what `distance_scale` and
`warmup_epochs` control.

### 2. Embeddings are L2-normalized

The paper's `F(x)` is the raw conv-net output; `DinoWrapper.project_features`
returns `F.normalize(self.fc(features), p=2, dim=1)`. This is a property of our
pipeline — every loss and every retrieval metric in this project operates on the
unit sphere — not a choice made about Hoffer's method.

It has one consequence that matters here. With a free embedding norm the network
controls its own softmax sharpness: scaling `F` by `α` scales every logit in
Eq. 1 by `α²`, so Eq. 1 needs no temperature. Normalization removes that degree
of freedom and pins `d² = 2 − 2cos` into `[0, 4]`.

### 3. `distance_scale`

The temperature that replaces the degree of freedom deviation 2 removed. On unit
vectors it is not an arbitrary reparametrization:

```
-s · d²  =  -s · (2 − 2cos)  =  -2s + 2s·cos
```

and the constant cancels inside the softmax, so `-s d²` is exactly a
scaled-cosine softmax at inverse temperature `2s` — the same object NT-Xent and
the normalized-softmax family use, and the same thing the released code's
`dist = 'dot'` branch computes. `distance_scale = s` on unit vectors is precisely
what an embedding norm of `sqrt(s)` would have been in the paper's setting.

`-s d` has no comparable interpretation on the sphere and is non-differentiable
at `d = 0`. It is, however, only a monotone reparametrization of `-s d²`, so with
a tuned scale the two behave alike — measured in *4. `-d²` vs `-d` at matched
sharpness* below. We keep `-s d²` because it is Eq. 1, not because it measured
better.

The rest of this document is about whether that knob earns its place.

### 4. `reference_sets`

The paper has exactly `c` references, one per class, and the released code cannot
run with more. `reference_sets = K` draws K complete reference sets and averages
K separate `C`-way entropies (rather than flattening them into one `K·C`-way
softmax, which would stop being class-shaped). This is ours; treat it as an
extension, not as paper-supported.

## Do we need `distance_scale`?

### The suspicion: does it just duplicate `regularizer_weight`?

Write the logits as `l_j = -s d_j²`. Near a uniform softmax the entropy gradient
linearizes to

```
dH / d(d_j²)  ~  -(s² / C) · (d_j² − mean_k d_k²)
```

which is quadratic in `s` with a **fixed direction** — "push every distance away
from the row's mean distance". So for a single step, while the softmax stays
flat, `distance_scale` and `regularizer_weight` look like the same knob: weight
`w` at scale `s` should behave like weight `w · s² / s'²` at scale `s'`.

That was the hypothesis going in. The benchmark confirms the first half of it
(the per-step gradient really does scale as `s²`) and **refutes** the conclusion
(the arms do not end up in the same place). Both halves matter, so both are
reported.

### What was measured

`scripts/bench_hoffer_distance_scale.py` on **Cars196**: real cached frozen
DINOv2 ViT-B/14 features, a trainable `Linear(768, 768) + L2 normalize` head
standing in for `DinoWrapper.fc` / `project_features`, the shipped
`class_subset_k_shot` split at 16 labeled per class (1568 labeled, 6486
unlabeled, 98 classes), `MultiSimilarityLoss + MultiSimilarityMiner` as the
supervised loss, a `128:4` batch, and 128 unlabeled + 98 references per
regularizer step. The entropy itself is the repository's own — the benchmark
asserts its closed form equals `HofferEntropyRegularizer.compute_loss` at
startup. Gradient norms come from `engine.measure_gradient_component_norms`, the
same function the trainer logs `train/gradient_contribution/*` from.

#### 1. Sharpness and gradient share, per `distance_scale`

Head after 1000 supervised steps, `regularizer_weight = 0.5`, uniform
probability `1/98 = 0.0102`:

| `distance_scale` | logit span | `p_max` | × uniform | `H / log C` | grad ratio | grad slope |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.08 | 0.0111 | 1.09 | 0.9999 | 4.1e-03 | — |
| 0.5 | 0.42 | 0.0156 | 1.53 | 0.9979 | 1.1e-01 | 2.02 |
| **1.0** (Eq. 1's implicit value, our default) | 0.84 | 0.0236 | 2.32 | 0.9913 | 4.4e-01 | 2.06 |
| 2.0 | 1.67 | 0.0524 | 5.14 | 0.9602 | 1.9e+00 | 2.11 |
| 4.0 | 3.35 | 0.1839 | 18.0 | 0.8073 | 7.4e+00 | 1.96 |
| 8.0 | 6.69 | 0.5114 | 50.1 | 0.4292 | 1.1e+01 | 0.60 |
| 16.0 | 13.39 | 0.7782 | 76.3 | 0.1570 | 9.1e+00 | −0.31 |
| 64.0 | 53.55 | 0.9476 | 92.9 | 0.0282 | 1.2e+01 | 0.21 |

"grad slope" is the local `d log(grad norm) / d log(s)`. The predicted slope of 2
holds all the way to `distance_scale = 4` and only breaks between 4 and 8.

So **Eq. 1 as written — no temperature, i.e. `s = 1` — barely does anything on
normalized embeddings**: the most likely reference gets 2.3× the uniform
probability and the normalized entropy is 0.991. The released code's `-d` form is
flatter still: at the same head its logit span at `s = 1` is only **0.33**,
against 0.84 for `-d²`, because the spread of `d` is smaller than the spread of
`d²` on unit vectors.

#### 2. The redundancy test

Four training arms holding `regularizer_weight · distance_scale² = 2` constant.
By the linearization they should be the same run. Cars196, 20 epochs × 50 steps,
R@1 on 4096 held-out test rows:

Mean over 3 seeds, with the per-seed min/max, and `Δ` against each seed's own
supervised-only control:

| arm | grad ratio¹ | R@1 (mean) | R@1 min–max | Δ vs `sup_only` | final `H / log C` | final purity |
| --- | --- | --- | --- | --- | --- | --- |
| `s = 1`, `w = 2` | 1.77 | 0.7677 | 0.7632–0.7729 | **−0.0840** | 0.874 | 0.32 |
| `s = 2`, `w = 0.5` | 1.92 | 0.8031 | 0.8008–0.8062 | **−0.0485** | 0.747 | 0.49 |
| `s = 4`, `w = 0.125` | 1.86 | 0.8717 | 0.8696–0.8752 | **+0.0200** | 0.505 | 0.67 |
| `s = 8`, `w = 0.031` | 0.70 | 0.8637 | 0.8591–0.8672 | **+0.0120** | 0.249 | 0.72 |

¹ the probe's measured `grad_ratio` for that `s`, rescaled from its
`regularizer_weight = 0.5` to the arm's weight.

The gradient shares match across the first three arms, as predicted (`s = 8` is
already outside the flat regime, which is exactly where the prediction stops
applying). The outcomes nonetheless span **10 points of R@1**, monotonically in
`s`, with a per-seed spread under 1 point — and the effect changes *sign*: at
matched gradient magnitude, `s = 1` costs 8.4 points against the control while
`s = 4` gains 2.0.

So the redundancy is exact for one step and false for a run: the entropy term
reshapes the distance distribution as it trains, and how far it pushes the
softmax out of the flat regime depends on `s`, not on `w`. `distance_scale`
controls not "how strong" but **"which update"**.

#### 2b. What this means when GradNorm is on

Our newer studies set `grad_norm_alpha`, which makes both loss weights learnable.
That raises the obvious question: if the weight is learned anyway, can
`distance_scale` be dropped?

No — and the redundancy test above is exactly the experiment that settles it.

GradNorm (`training/ssl/grad_norm.py`, after Chen et al. 2018) trains the weights
against

```
L_grad = sum_i | G_i − Gbar · r_i^alpha |,     G_i = || grad_W(w_i · L_i) ||
```

Every quantity GradNorm reads and every quantity it controls is a **gradient
norm**. It has exactly one degree of freedom per term, and that degree of freedom
is the term's gradient magnitude.

The redundancy arms held that magnitude fixed by construction and still landed 10
points apart. Everything separating them lies in a direction GradNorm cannot see.
The reason is mechanical: in the flat regime the gradient direction is "move
every distance away from this row's mean distance", which contains no class
information at all; once the softmax sharpens, the gradient concentrates on the
nearest reference and becomes "commit to this neighbour". `distance_scale`
chooses between those two shapes; the weight only chooses how hard to push
whichever shape is already selected.

So with GradNorm on, `distance_scale` is not redundant — it is the *only*
remaining control over what the term does. Removing it means pinning `s = 1`,
which pins the objective in the flat regime, and GradNorm will then faithfully
scale up an update that measured worse than switching the regularizer off.

The defensible simplification is a different one: **drop `distance_scale` from
the search space, not from the method.** Fix it in `[4, 16]` and let GradNorm own
the magnitude — `distance_scale` sets the shape, GradNorm sets the size.

#### 3. The full scale ladder at a fixed weight

`regularizer_weight = 0.5` (the shipped default), same setup, mean over 3 seeds,
`Δ` against each seed's own `sup_only` control:

| arm | R@1 (mean) | R@1 min–max | Δ vs `sup_only` | final `H / log C` | final purity |
| --- | --- | --- | --- | --- | --- |
| `sup_only` | 0.8516 | 0.8472–0.8569 | — | — | — |
| `s = 0.1` | 0.8519 | 0.8477–0.8577 | +0.0002 | 0.9999 | 0.70 |
| `s = 0.5` | 0.8495 | 0.8464–0.8538 | −0.0021 | 0.9928 | 0.69 |
| `s = 1` (Eq. 1, our default) | 0.8391 | 0.8340–0.8440 | −0.0125 | 0.9200 | 0.60 |
| `s = 2` | 0.8031 | 0.8008–0.8062 | **−0.0485** | 0.7468 | 0.49 |
| `s = 4` | 0.8547 | 0.8496–0.8584 | +0.0031 | 0.4409 | 0.59 |
| `s = 8` | 0.8719 | 0.8667–0.8752 | **+0.0203** | 0.1752 | 0.60 |
| `s = 16` | 0.8608 | 0.8569–0.8643 | +0.0091 | 0.0862 | 0.60 |
| `s = 64` | 0.8248 | 0.8167–0.8330 | −0.0269 | 0.0308 | 0.53 |
| released code's `-d`, `s = 1` | 0.8521 | 0.8472–0.8579 | +0.0004 | 0.9971 | 0.70 |

Two results stand out.

**Without a temperature, the objective does nothing.** The `-d`-at-`s = 1` row —
the released code's exact form — is within 0.1 points of the supervised-only
control on all three seeds, and its normalized entropy never leaves 0.997.
Eq. 1's own `-d²` at `s = 1` is not inert but *negative*. On L2-normalized
embeddings the published objective, as published, is not a regularizer. That
answers the question in the heading.

**The low end of the range is worse than useless.** `s = 2` loses 4.9 points on
average. This is not a weak version of the entropy term; it is a *different*
update. In the flat regime the gradient direction knows nothing about class
structure, and it shows in `purity`: from ~0.70 at `s = 0.1` (term inert) down to
~0.50 at `s = 2`, then partly recovering at `s ≥ 4`, where the term starts
committing to a nearest neighbour instead.

#### 4. `-d²` vs `-d` at matched sharpness

Deviation 3 argued for `-s d²` on the grounds that it is Eq. 1, that it is the
scaled-cosine softmax on the unit sphere, and that `-s d` is non-differentiable
at `d = 0`. None of that says the two behave differently once the scale is tuned.
This measures it. `regularizer_weight = 0.5`, mean over 2 seeds:

| arm | R@1 (mean) | Δ vs `sup_only` | final `H / log C` | final `p_max` |
| --- | --- | --- | --- | --- |
| `sup_only` | 0.8539 | — | — | — |
| `-d²`, `s = 4` | 0.8573 | +0.0034 | 0.438 | 0.427 |
| `-d²`, `s = 8` | 0.8745 | **+0.0206** | 0.173 | 0.742 |
| `-d`, `s = 1` (released code) | 0.8545 | +0.0006 | 0.997 | 0.016 |
| `-d`, `s = 8` | 0.8138 | −0.0400 | 0.363 | 0.519 |
| `-d`, `s = 16` | 0.8669 | **+0.0131** | 0.156 | 0.766 |

The two forms trace the **same curve, shifted in `s`**. `-d` at `s = 16` reaches
almost exactly the sharpness `-d²` reaches at `s = 8` (`p_max` 0.766 vs 0.742,
`H/logC` 0.156 vs 0.173) and gains +0.013 against `-d²`'s +0.021. Both forms also
have the same harmful middle band — `-d²` at `s = 2` (−0.049) and `-d` at `s = 8`
(−0.040) — displaced by the same factor of roughly two that the probe's sharpness
columns predicted.

So the choice of `d` vs `d²` is a **reparametrization of the temperature**, not a
different objective. Only two `-d` scales were run, so the +0.021 / +0.013 gap is
not a claim that `-d²` is better; a finer `-d` grid could well close it. What the
table does establish is that neither form escapes the need for a tuned scale, and
that the `-d²` choice costs nothing. We keep `-d²` because it is Eq. 1, not
because it measured better.


#### The cost of turning the scale up

`argmax_purity` never gets close to 1. On a head 1000 supervised steps in it is
**0.72**, and it does not depend on `distance_scale` at all (the argmin does not
move). So at `distance_scale = 16`, where `p_max = 0.78`, the term is confidently
sharpening toward the **wrong** class for roughly a quarter of the unlabeled
batch. That is why `s = 64` falls back below the control — total confidence in a
72%-accurate assignment.

Purity is 0.53 after 200 supervised steps and 0.22 at initialization, so a large
`distance_scale` combined with `warmup_epochs = 0` is the worst corner of the
space.

#### Cross-check against the real pipeline

These are proxy runs — a linear head, 1000 steps, one loss. The finished In-Shop
HPO studies in `exporter/optuna-studies/ssl/in_shop/hoffer/` are the real
pipeline on a different dataset (3997 classes, `k_shot = 2`, `reference_sets = 1`,
`distance_scale` searched log-uniformly over `[0.1, 100]`, GradNorm learning the
weights), and they land in the same place:

| study | `distance_scale` of the top 4 trials | median `distance_scale`, top vs bottom half |
| --- | --- | --- |
| `hoffer_ms` (30 trials) | 11.2, 51, 19.5, 9.3 | 4.25 vs 1.53 |
| `hoffer_arc` (31 trials) | 11.4, 13.4, 4.9, 76.6 | 4.9 vs 4.23 |

Every top-4 trial in both studies sits at `distance_scale ≥ 4.9`. The best
`hoffer_arc` trial below 4 (`s = 1.15`) scores 0.5707 MAP@R against the study's
best 0.6478. An independent tuner, on another dataset, with the weights learned
rather than fixed, chose the same region the Cars benchmark points at.

## Recommendation

**Keep `distance_scale`.** It is not redundant with `regularizer_weight`, GradNorm
cannot substitute for it, and without it the published objective is inert on
normalized embeddings.

**But narrow the search range to roughly `[4, 32]`, or fix it.**

- `configs/runs/hoffer/cars.json`, `cifar.json` and `in_shop.json` search
  `distance_scale` over `[0.1, 16]` **jointly with** `regularizer_weight`. Those
  two are collinear over the bottom two thirds of that log-range, which wastes
  trials and makes the fanova importances of both parameters unreadable.
- `configs/runs/hoffer/cub_initial.json` searches `[0.1, 100]` with
  `grad_norm_alpha` on. GradNorm cancels the `s²` magnitude the low end applies —
  but not its *direction*, which is the harmful part. The low end is worth even
  less there.
- Change the default in `configs/ssl_methods/hoffer.json` from `1.0`. It sits at
  2.3× uniform and measured −1.2 points on both seeds. Something in `[4, 8]` is
  where the term starts behaving like a regularizer.
- Keep `warmup_epochs > 0` whenever `distance_scale ≥ 4`. Reference assignment is
  only ~22% correct at initialization, and a sharp softmax over a wrong
  assignment is the one combination that clearly destroys the embedding.

**Scope of these numbers.** One dataset, one supervised loss, a frozen backbone
with a linear head, 1000 training steps per arm, and R@1 rather than the MAP@R
the studies select on. Per-seed and per-epoch rows are in
`scripts/reports/hoffer_distance_scale/`. Treat the ordering as the result and
the absolute R@1 values as indicative; the In-Shop cross-check is the evidence
that the ordering survives the real pipeline.

## Reproducing

```bash
# the sharpness / gradient-share tables
python scripts/bench_hoffer_distance_scale.py --stages probe \
    --csv scripts/reports/hoffer_distance_scale/cars_probe.csv

# the training arms, one seed per file
for seed in 0 1 2; do
  python scripts/bench_hoffer_distance_scale.py --stages train degeneracy \
      --seed $seed --degeneracy-product 2.0 \
      --csv scripts/reports/hoffer_distance_scale/cars_train_seed$seed.csv
done

# -d vs -d^2 at matched softmax sharpness
python scripts/bench_hoffer_distance_scale.py --stages train \
    --scales 4 8 --reference-scales 1 8 16 \
    --csv scripts/reports/hoffer_distance_scale/cars_form_seed0.csv
```

Every stage needs a `--use_cache` training run on the dataset first, so the
frozen backbone features exist under `data/<dataset>/backbone_cache/`.

> **Cache note.** Cars196 has a stale `dinov2_vitg14` cache with the right row
> count but rows that do not line up with the dataset labels (1-NN label
> agreement 0.010, i.e. chance). `find_cached_feature_matrix` in
> `scripts/bench_seraph_gradient_contribution.py` picks caches by row count and
> would take that one; this benchmark picks the candidate with the best label
> agreement instead and prints it.
