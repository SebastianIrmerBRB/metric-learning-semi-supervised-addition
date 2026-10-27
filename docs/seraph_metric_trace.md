# What a SERAPH run optimizes

The objective one optimizer step descends, written out, and what `metric`
changes inside it. Implementation: `training/ssl/seraph.py`,
`SeraphRegularizer.compute_loss` (line 2040), `SeraphRegularizer._trace_penalty`
(line 1712) and `SeraphPosteriorHead` (line 836). Operational knob reference:
[`docs/seraph.md`](seraph.md).

---

## 1. The whole thing

Each step draws two batches: a class-balanced **labeled** batch
$\mathcal{B}_L$ ($n_L$ samples) and an independent **unlabeled** batch
$\mathcal{B}_U$ ($n_U$ = `unlabeled_batch_size`). One joint forward pass
produces L2-normalized retrieval embeddings $z$ for both. Then

$$
\boxed{\;
\mathcal{J} \;=\;
\underbrace{w_{\text{sup}}\; \mathcal{L}_{\text{sup}}(\mathcal{B}_L)}_{\text{the configured DML loss}}
\;+\;
\underbrace{w_{\text{reg}} \sum_{b \in \{uu,\,lu\}} \pi_b \;
\frac{1}{|P_b|} \sum_{(i,j) \in P_b} \ell\big(\text{logit}_{ij}\big)}_{\text{SERAPH}}
\;+\;
w_{\text{fair}}\,\mathcal{L}_{\text{fair}}
\;+\;
\underbrace{\lambda\,\mathcal{T}(A)}_{\text{Eq 7's trace term}}
\;}
$$

with $w_{\text{sup}} =$ `supervised_weight`, $w_{\text{reg}} =$ `regularizer_weight`,
$w_{\text{fair}} =$ `fairness_weight` (0 by default — that term disappears
unless `prior_mode="freematch"` is in play) and $\lambda =$ `trace_weight`
(0 by default, so that term disappears too). There is no labeled-pair likelihood
term; deviation 1 in the module docstring says why. **This is the entire
objective.**

$\mathcal{T}(A)$ is $\mathrm{tr}(A)$ under `trace_mode="nuclear"` and
$\mathrm{tr}(A)^2 / \lVert A \rVert_F^2$ under `"soft_rank"`, taken over the
posterior's own $A$ or over the model projection head's induced
$A = M^\top M$, $M = [W \mid b]$, as `trace_target` selects. The appendix
covers when each is a regularizer and when it is a constant.

Two things that move $w_{\text{reg}}$ at runtime: `warmup_epochs` holds it at 0
for the first $N$ epochs, and if `regularizer_target_ratio` is set (the Cars
search space sets it) the engine recalibrates $w_{\text{reg}}$ from 20
sample-scoped probes in every 10,000-sample window by default so the two
gradient norms track that ratio. The shared target-ratio update and window
sample parameters can change both cadences, including an explicit every-step
mode.

## 2. The SERAPH term, expanded

**Pair blocks.** $P_{uu}$ is every unordered pair inside $\mathcal{B}_U$
($\binom{n_U}{2}$ of them); $P_{lu}$ is every labeled × unlabeled pair
($n_L n_U$), present only if `include_labeled_unlabeled=true`.
Labeled-labeled pairs are never scored — their weak label is known.

**Block weights** come from the *pool*, not the batch, so batch composition
can't reweight the objective ($N_L$, $N_U$ = pool sizes):

$$
\pi_{uu} = \frac{\binom{N_U}{2}}{\binom{N_U}{2} + N_L N_U},
\qquad
\pi_{lu} = \frac{N_L N_U}{\binom{N_U}{2} + N_L N_U},
\qquad
\pi_{uu} + \pi_{lu} = 1 .
$$

**The logit** is Eq 5, one $\eta$ and $\kappa$ per block:

$$
\text{logit}_{ij} \;=\; \kappa_b \,\big(\eta_b - d_A^2(i,j)\big),
\qquad
p_{ij} = \sigma(\text{logit}_{ij}) = p(y_{ij} = +1) .
$$

$$
d_A^2(i,j) = \lVert B (z_i - z_j) \rVert^2 = (z_i - z_j)^\top A\, (z_i - z_j),
\qquad A = B^\top B .
$$

Both $\eta_b$ and $\kappa_b$ are **detached** — they calibrate the posterior,
they are not optimized:

| mode | $\eta_b$ | $\kappa_b$ |
|---|---|---|
| `quantile` (default) | EMA of the `positive_pair_prior`-quantile of this block's $d_A^2$ | — |
| `fixed` | the `threshold` constant | — |
| `self_tuning` / `umap` / `tsne` | per-anchor, from $k$-th neighbor distances, symmetrized | — |
| `labeled_logistic` | $b/\kappa$ from a logistic fit on labeled pairs | $-a$ from the same fit |
| `batch_std` (default $\kappa$) | — | $\kappa / \mathrm{std}(d_A^2)$ over the block |
| `fixed` $\kappa$ | — | the `kappa` constant |

**The per-pair term** $\ell$ is where `objective` branches:

$$
\ell_{\text{entropy}}(\text{logit}) = H(p_{ij})
= -p_{ij}\ln p_{ij} - (1 - p_{ij})\ln(1 - p_{ij})
$$

$$
\ell_{\text{em}}(\text{logit}) = \mathrm{BCE}\big(\text{logit},\, q_{ij}\big),
\qquad
q_{ij} = \sigma\big(\beta \cdot \text{logit}_{ij}^{\;\text{detach}}\big),
\quad \beta > 1
$$

where $\beta$ is `em_sharpening`.

`entropy` descends Eq 7's term directly. `em` is Eq 9: the E-step's analytic
solution $q \propto p^{\beta}$ is a rescaling of the logit, and the M-step is a
cross entropy against that fixed target. Since $\beta > 1$ makes $q$ more
confident than $p$ in the *same* direction, the gradient is $\propto (p - q)$ —
self-training on pair labels. $\beta = 2$ is the paper's $\mu = 1$.

Both terms are minimized by confident posteriors. What stops the trivial
"everything is dissimilar" solution is that $\eta$ is a quantile: it re-centers
each step to keep a `positive_pair_prior` fraction of pairs on the similar side.

## 3. What `metric` actually does

**`metric` changes exactly one thing in §2: $d_A^2$.** Nothing else in the
objective sees it. Two consequences worth being explicit about:

- **$A$ never touches retrieval or evaluation.** The head is a submodule of the
  model but is not in its `forward`; recall/mAP are always computed on the plain
  embedding $z$. $A$ only decides how SERAPH *scores* pairs during training,
  and so only affects $z$ indirectly, through the gradient.
- All four start at $A = I_D$ ($D =$ `feat_dim` = 768), i.e. at the plain
  squared Euclidean distance of the normalized embedding.

| `metric` | learnable | $d_A^2(i,j)$ with $\delta = z_i - z_j$ | what it can express |
|---|---|---|---|
| `identity` | nothing | $\lVert \delta \rVert^2$ | nothing — the objective can only be lowered by moving embeddings |
| `scale` | $u \in \mathbb{R}$ | $e^{u}\lVert \delta \rVert^2$ | a global stretch. Cannot reorder any pair; it only rescales the logit, i.e. learns *how confident to be*, not *what is near what* |
| `diag` | $r \in \mathbb{R}^{D}$ | $\sum_k \mathrm{softplus}(r_k)\,\delta_k^2$ | per-dimension reweighting — soft feature selection over the 768 DINO dims. Can reorder pairs |
| `lowrank` | $L \in \mathbb{R}^{m \times D}$ | $\lVert L \delta \rVert^2$ | any linear reweighting + rotation into a rank-$m$ subspace. The paper's own setting; $m =$ `metric_dim` |

The ordering caveat matters. `scale` is monotone in $\lVert\delta\rVert^2$, so
the *ranking* of pairs by similarity is identical to `identity` — every pair the
posterior calls similar under one, it calls similar under the other, at a
different threshold. Only `diag` and `lowrank` can decide that pair $(i,j)$ is
closer than $(k,l)$ when the raw embedding says otherwise.

**The measured catch** ([`docs/seraph.md` §2.2](seraph.md)): with `scale` and
full-rank `lowrank`, 99.4% and 97.6% of the regularizer's gradient lands on $B$
rather than on $z$. SERAPH sharpens its own posterior by growing $A$ instead of
separating embeddings, because nothing in §1 bounds $A$'s scale. Restricting to
`metric_dim=64` flips it — $B$ can no longer absorb the sharpening and 81% of
the gradient reaches the embedding. `identity` remains the module default.

### `lowrank` is the paper's `A`, factored

The paper's optimization variable is $A$ itself: "a symmetric and positive
semi-definite matrix to be learned" (Sec 2.1), solved by the **gradient
projection method** (Sec 2.4), which enforces $A \succeq 0$ by projecting onto
the PSD cone. It never introduces a factor. What is low-rank in the paper is the
*projection induced from* $A$ — the map $x \mapsto Lx$ that $A = L^\top L$
implies — and it becomes low-rank because of the $-\lambda\,\mathrm{tr}(A)$
term in Eq 6, the trace being a convex relaxation of the rank.

So `lowrank` here is not a different model. It is the same $A$, reached by
optimizing the factor instead: $A = L^\top L$ is PSD by construction, so the
projection step is unnecessary (deviation 7). The name describes the intent, not
a constraint.

The divergence is where the low rank comes from:

| | paper | this repo |
|---|---|---|
| variable | $A$, PSD-projected each step | $L$, PSD by construction |
| rank pressure | $\lambda\,\mathrm{tr}(A)$, emergent and soft | `metric_dim` $= m$, a hard cap on $\mathrm{rank}(A) \le m$, **and** `trace_weight` $= \lambda$ |
| $\lambda$ | a tuned hyperparameter | `trace_weight`, default $0$ |

**Therefore `"metric": "lowrank"` with `"metric_dim": null` and
`"trace_weight": 0` sets $m = D = 768$ and has no low-rank pressure of any
kind** — no trace term, no cap. It is the paper's full metric $A$ with the one
mechanism that made it low-rank removed, and it is the exact setting the 97.6%
measurement above was taken at. The initialization ($L$ orthogonal, scaled
$\sqrt{D/m}$) makes a square $L$ a pure rotation, so it starts at $A = I$ and
drifts from there unconstrained.

Two ways to restore the paper's behaviour, and they are not equivalent:

- `metric_dim` is the *hard* substitute — the measured table puts $m = 64$ at
  81% of the regularizer gradient reaching the embedding, versus 2.4% at
  $m = 768$.
- `trace_weight` is the paper's own *soft* one. It only earns that description
  where $A$'s scale is identifiable, which for this run's `quantile` +
  `batch_std` calibration it is not — see the appendix.

## 4. The run in `configs/test_bed/ssl_seraph.json`

`objective="em"`, $\beta = 2$, `metric="lowrank"` with `metric_dim=null`
($m = D = 768$), $\kappa = 1$ with `batch_std`, `threshold_mode="quantile"` at
prior $0.1$ with momentum $0.1$, `include_labeled_unlabeled=true`, $n_U = 128$,
$w_{\text{sup}} = 1$, $w_{\text{reg}} = 15$, `fairness_weight` unset,
`warmup_epochs=20`; the loss is NT-Xent
(`configs/phase_1/ssl_methods/seraph/cars/cars_nt.json`). Substituting into §1:

$$
\mathcal{J} = \mathcal{L}_{\text{NT-Xent}}(\mathcal{B}_L)
\;+\; 15 \sum_{b \in \{uu,\,lu\}} \pi_b \,
\frac{1}{|P_b|} \sum_{(i,j) \in P_b}
\mathrm{BCE}\!\left(\ell_{ij},\, \sigma(2\,\ell_{ij}^{\;\text{detach}})\right)
$$

$$
\ell_{ij} = \frac{\eta_b - \lVert L(z_i - z_j) \rVert^2}
{\mathrm{std}_b\!\left(\lVert L(z - z') \rVert^2\right)},
\qquad
\eta_b \leftarrow 0.9\,\eta_b
+ 0.1\,Q_{0.1}\big(\lVert L(z - z')\rVert^2\big)
$$

after epoch 20; before it, the second term is absent. For an illustrative
Cars196 split with $N_L \approx 800$, $N_U \approx 7200$ that gives
$\pi_{uu} \approx 0.82$, $\pi_{lu} \approx 0.18$.

$L$ is $768 \times 768$ and trainable, so this run optimizes 590k metric
parameters alongside the embedding, starting from a rotation ($A = I$). Per §3,
that is the configuration in which the regularizer spends almost all of its
gradient on $L$ rather than on $z$, and per the appendix the overall scale of
$A$ is unidentifiable under `quantile` + `batch_std` — only $L$'s anisotropy
changes the objective.

---

## Appendix: when $\mathrm{tr}(A)$ is a regularizer

Eq 7's third term is `trace_weight`, and it defaults to $0$. This is why: it is
a rank relaxation only where $A$'s scale is identifiable, and two of this
module's deviations remove exactly that.

**1. The calibration.** Under `quantile` + `batch_std` the objective is
homogeneous of degree 0 in $A$: rescaling $A \mapsto cA$ sends
$\eta_b \mapsto c\,\eta_b$ and $\kappa_b \mapsto \kappa_b / c$, which cancel in
the logit. A trace penalty there is unopposed and drives $A \to 0$ rather than
leashing it. Eq 7's term presupposes the paper's fixed $\eta$ and $\kappa$, and
fixing *either* one is enough:

| $\eta$ mode | $\kappa$ mode | loss as $A$ is scaled by $0.5, 1, 2, 8$ | |
|---|---|---|---|
| `fixed` | `fixed` | 0.662, 0.579, 0.017, 0.006 | responds |
| `fixed` | `batch_std` | 0.017, 0.198, 0.040, 0.026 | responds |
| `quantile` | `fixed` | 0.692, 0.670, 0.482, 0.033 | responds |
| `quantile` | `batch_std` | 0.541, 0.541, 0.541, 0.541 | **degenerate** |

**2. The normalization.** `trace_target="projection"` is scale-free for a
second, independent reason, and no calibration setting rescues it:
`project_features` L2-normalizes the head's output, so the objective is
invariant to $(W, b) \mapsto (cW, cb)$. Euler's theorem then forces
$\langle \nabla \mathcal{L}, (W,b) \rangle = 0$ — measured cosine
$-1.8 \times 10^{-8}$ — while $\mathrm{tr}(A)$'s gradient is exactly $2M$.
The penalty pushes along the one direction the loss cannot see. It rescales the
head, an effective learning rate in the sense of arXiv:1706.05350, and moves no
part of the spectrum.

`trace_mode="soft_rank"` is the way out. $\mathrm{tr}(A)^2 / \lVert A \rVert_F^2$
is the participation ratio of $A$'s eigenvalues — $1$ for a rank-one metric,
$D$ for the identity — and it relaxes rank the same way the trace does while
being homogeneous of degree 0 like the objective around it. Its gradient is
100% tangential to the scale ray, and descending it alone takes a 16-dimensional
head from a participation ratio of $8.08$ to $1.00$.

### What each parameterization contributes

| `metric` | $\mathrm{tr}(A)$ | $\mathrm{tr}(A)^2/\lVert A\rVert_F^2$ |
|---|---|---|
| `identity` | $D$ (constant — rejected) | $D$ (constant) |
| `scale` | $D e^{u}$ | $D$ (constant — rejected) |
| `diag` | $\sum_k \mathrm{softplus}(r_k)$ | a real rank surrogate over the $D$ dims |
| `lowrank` | $\lVert L \rVert_F^2$ | a real rank surrogate, capped at $m$ |
| projection head | $\lVert W \rVert_F^2 + \lVert b \rVert^2$ | a real rank surrogate over the head |

The bias belongs in the last row. The head is affine, and augmenting its input
with a constant $1$ makes $M = [W \mid b]$ a linear map whose induced metric is
genuinely $A = M^\top M$. Penalizing $W$ alone instead leaves 8.4% of the
gradient off the rescaling ray, and that component shrinks $W$ relative to $b$,
pulling the normalized embedding towards the constant direction $b / \lVert b
\rVert$ — a collapse, not a low rank.

### Why not `weight_decay`

An earlier reading of deviation 2 held that `weight_decay` already applies
$\lambda\,\mathrm{tr}(A)$ to the head. It does not, on three counts:

| `metric` | is $\lambda\,\mathrm{tr}(A)$ the same as `weight_decay`? |
|---|---|
| `identity` | no-op either way |
| `scale` | no — decay penalizes $u^2 = (\ln s)^2$ and pulls $s \to 1$; the trace pulls $s \to 0$ with a force $D$ times larger |
| `diag` | no — decay pulls $r_k \to 0$, i.e. $a_k \to \ln 2 \approx 0.693$; the trace pulls $a_k \to 0$ |
| `lowrank` | yes, with $\nu = 2\lambda$ — and only under `optim=adam`/`rmsprop`, which couple decay into the gradient. `adamw` decouples, so the loss-side term gets normalized by $\sqrt{\hat v}$ and the decay does not |

And `--weight_decay` defaults to `0.0` with no SSL config setting it, so
$\lambda = 0$ in every SERAPH run predating `trace_weight` — the term was
absent, not delegated.

### The weight is its own

$\lambda$ sits beside $w_{\text{reg}}$ in §1, not inside it, because the trace
reads parameters rather than pairs: folding it into `regularizer_weight` would
put a batch-independent term inside the weight that `regularizer_target_ratio`
calibrates from measured per-batch gradients.

It has its own calibration instead. `trace_target_ratio` makes the configured
$\lambda$ a probe and sets the weight that hits a measured gradient ratio
against $\mathcal{L}_{\text{sup}}$, and it is on by default at $0.1$ for
`trace_target="projection"` — the penalty's units (a squared Frobenius norm, or
a dimensionless ratio in $[1, D]$) say nothing about how hard it should push.
`grad_norm_alpha` can learn it instead, and clears the ratio when it does —
unless the trace is named in `grad_norm_exclude_components`, which keeps the term
out of the balancing and leaves its ratio standing. That is the combination worth
reaching for: GradNorm's signal is the relative inverse training rate
$L(t)/L(0)$, and for `soft_rank` that ratio *is* how much rank has already been
surrendered, so a falling value reads as "training fast, downweight" and the
prior is relaxed exactly when it starts to bite. Excluding it balances the
supervised and entropy terms while the rank prior stays where one measurement
put it.

```json
"method_params": {
  "grad_norm_alpha": 1.5,
  "grad_norm_exclude_components": ["seraph_trace"],
  "regularizer_params": {"trace_weight": 1.0}
}
```

Both are confined to the projection target, for one reason measured in §2 of
this appendix: the posterior's metric contributes a gradient of exactly $0$ to
the shared trunk, which is what both mechanisms measure over. On the `metric`
target `trace_target_ratio` raises at construction rather than exhausting the
trainer's dead-term budget mid-run, and GradNorm is not registered at all —
it skips its whole update on any non-positive component norm, so registering a
structurally-zero term would freeze the supervised and regularizer weights along
with it.
