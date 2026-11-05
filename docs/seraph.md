# SERAPH Parameter Reference

SERAPH is the unlabeled entropy term of "Information-theoretic Semi-supervised
Metric Learning via Entropy Regularization" (Niu et al., ICML 2012), implemented
in `training/ssl/seraph.py` as a `BaseTrainingRegularizer`. Every optimizer step
computes

```
supervised_weight * L_supervised(labeled batch)
  + regularizer_weight * L_seraph(unlabeled batch)
```

SERAPH adds an unlabeled entropy term to the configured metric-learning loss.
This guide explains its parameters and their effects.

## Config layout

```json
{
  "method": "seraph",
  "label_sampling_mode": "class_subset_k_shot",
  "labeled_fraction": 1.0,
  "labeled_per_class": 16,
  "warmup_epochs": 0,
  "method_params": {
    "supervised_weight": 1.0,
    "regularizer_weight": 5.0,
    "regularizer_params": {
      "objective": "entropy",
      "metric": "identity",
      "kappa": 1.5,
      "kappa_mode": "batch_std",
      "threshold_mode": "quantile",
      "threshold": 0.1,
      "threshold_momentum": 0.9,
      "positive_pair_prior": null,
      "include_labeled_unlabeled": true,
      "unlabeled_batch_size": 32
    }
  }
}
```

Registered as `"seraph"` in `semi_supervised.REGULARIZER_REGISTRY`.

---

## 1. The posterior — how the knobs compose

Everything SERAPH does flows through one quantity, the **pair logit** (Eq 5):

```
logit = kappa_eff * (eta - d_A^2)        p(similar) = sigmoid(logit)
```

- `d_A^2` is the squared distance between two embeddings in the metric's
  projected space. With `metric="identity"` and L2-normalized embeddings this is
  confined to `[0, 4]` and in practice concentrates near 2.0 (mutually orthogonal).
- `eta` is the decision threshold, set by `threshold_mode` / `positive_pair_prior`.
- `kappa_eff` is the inverse temperature, set by `kappa` / `kappa_mode`.

The objective is the mean binary entropy of that posterior, whose gradient w.r.t.
the logit is `-l * sigmoid'(l)`. **That vanishes at both ends** — at `l = 0` the
posterior is undecided and there is nothing to sharpen, at large `|l|` it is
saturated and the sigmoid is flat. Intermediate logits give the strongest
sharpening signal.

> **The single number to read when tuning: mean `|logit|`.** Aim for ~2.4.
> `kappa`, `positive_pair_prior`, `threshold_mode` and `metric` are four
> different ways of moving it, and they trade off against each other rather than
> stacking.

---

## 2. `regularizer_params` — the SERAPH knobs

### 2.1 Objective

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `objective` | `"entropy"`, `"em"` | `"em"` | `"entropy"` descends Eq 7's entropy term directly. `"em"` follows Sec 2.4: the analytical E-step of Eq 9 (`q(y) ~ p(y)^em_sharpening`, detached) then a cross-entropy M-step against those soft targets. |
| `em_sharpening` | float ≥ 1 | `2.0` | The E-step exponent `1 + mu`; `2.0` is the paper's `mu = 1`. Ignored when `objective="entropy"`. |

**`em_sharpening = 1.0` produces an identically zero gradient.** Validation
permits it (the check is `>= 1`), but `q = p` exactly, so the M-step's cross
entropy is taken against the posterior's own detached value.
Never put `1.0` in a search space.

### 2.2 The metric `A = B^T B`

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `metric` | `"identity"`, `"scale"`, `"diag"`, `"lowrank"` | `"identity"` | Parameterization of `B`. All are PSD by construction and all start at the plain squared Euclidean distance of the L2-normalized embedding. `identity` = no metric parameters; `scale` = one learnable scalar; `diag` = per-dimension weights via softplus; `lowrank` = a `metric_dim x feat_dim` matrix, the paper's own setting. |
| `metric_dim` | int in `(0, feat_dim]` or `null` | `null` (= `feat_dim`) | Rows of `B`. Only valid with `metric="lowrank"`; raises otherwise. |

`metric` changes the pair distance used by SERAPH. Retrieval and evaluation
always use the model's ordinary embedding. A learnable metric can sharpen its
own posterior without moving that embedding; reducing `metric_dim` limits the
directions available to it. Inspect the gradient's `head_share` alongside its
overall magnitude.

### 2.2b The trace term `- lambda tr(A)` (Eq 7)

Off by default. Eq 7's hyper-sparsity term, which is what makes the paper's
projection low-rank. `tr(A)` is the nuclear norm of a PSD `A`, i.e. the convex
relaxation of `rank(A)`.

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `trace_weight` | float ≥ 0 | `0.0` | The paper's `lambda`. `0` leaves the term out of the objective entirely. Carries its own weight beside `regularizer_weight` rather than inside it — see §3. |
| `trace_target` | `"metric"`, `"projection"` | `"metric"` | Which `A` is penalized. `metric` is the posterior head's own `A = B^T B`, the paper's literal variable; rejected for `metric="identity"`, where `tr(I) = feat_dim` is a constant. `projection` is the `A` the **model's** projection head induces: with frozen cached features it maps `x -> Wx + b`, so `A = M^T M` for `M = [W \| b]`. That is the projection being learned when the posterior's metric has no parameters of its own. |
| `trace_mode` | `"nuclear"`, `"soft_rank"` | `"nuclear"` | `nuclear` is `tr(A)`, the term as Eq 7 writes it. `soft_rank` is `tr(A)^2 / ||A||_F^2`, the participation ratio of `A`'s eigenvalues — the same rank relaxation with the scale divided out. Runs from `1` (rank one) to `feat_dim` (isotropic). |
| `trace_target_ratio` | float ≥ 0, or `null` | `0.1` for `projection`, off for `metric` | Calibrates `trace_weight` from measured gradients instead of taking it literally, exactly as `regularizer_target_ratio` does for `regularizer_weight`: the configured `trace_weight` becomes a probe, and the trainer sets the weight from 20 gradient probes distributed over each 10,000-sample window by default. The shared target-ratio sample-cadence parameters can change the probe interval and window length. `0` opts out. Requires `trace_weight > 0` — the ratio calibrates that weight, it does not switch the term on. |

`projection` calibrates by default because the penalty's units are arbitrary: a
squared Frobenius norm and a dimensionless ratio in `[1, feat_dim]` are not
commensurable with a metric-learning loss, so a hand-picked `lambda` is a guess
at a scale rather than a choice about strength. `0.1` puts the rank prior an
order of magnitude below the supervised gradient — visible, not competitive. It
is a starting point, not a tuned constant.

`trace_target_ratio` on the **`metric`** target raises. The posterior head is in
`private_model_module_names()`, which every gradient comparison strips out, so
the measured norm would be exactly `0`, the calibration would exhaust its
dead-term budget and raise `FloatingPointError` several epochs into the run.
Set `trace_weight` directly there.

**Scale identifiability matters for `tr(A)`.** The constructor warns when
calibration or output normalization makes the objective insensitive to scale:

| setting | is `A`'s scale visible to the objective? |
|---|---|
| `threshold_mode` and `kappa_mode` both adaptive | no — `A -> cA` sends `eta -> c eta` and `kappa -> kappa / c`, which cancel in the logit. `tr(A)` is unopposed and drives `A -> 0`. |
| either one `"fixed"` | yes — this is the paper's setting, and the one where `nuclear` behaves as published |
| `trace_target="projection"`, any calibration | no — `project_features` L2-normalizes the head, so the objective is invariant to `(W, b) -> (cW, cb)` |

On an L2-normalized projection head, `nuclear` rescales the parameters without
changing the normalized embeddings. `soft_rank` measures the spectrum after
dividing out scale. Include the bias in `M = [W | b]`: penalizing `W` alone can
shrink it relative to the bias and pull embeddings toward a constant direction.

Two combinations raise instead of warning, because they are constants rather
than weak: `trace_target="metric"` with `metric="identity"` (`tr(I) = feat_dim`),
and `trace_mode="soft_rank"` with `metric="scale"` (`A = sI` is isotropic, so its
participation ratio is `feat_dim` for every `s`). `soft_rank` with
`trace_target="projection"` also requires `projection_layers=1`: across a ReLU
there is no single `A` whose spectrum the ratio would be measuring.

Diagnostics report both numbers whichever mode is being descended —
`train/seraph/trace/nuclear` and `train/seraph/trace/soft_rank` — because the
trace alone cannot distinguish a metric that shrank from one that lost rank.

### 2.3 Inverse temperature

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `kappa` | float > 0 | `1.0` | Inverse temperature of the pair posterior. Never trainable — that would re-add the redundant scale Theorem 2 removes, and entropy minimization would inflate it without moving any embedding (deviation 3). |
| `kappa_mode` | `"fixed"`, `"batch_std"` | `"batch_std"` | `fixed` uses `kappa` as-is. `batch_std` uses `kappa / std(d^2)` over the batch's own distances, making the posterior independent of how concentrated the frozen embedding's distances happen to be. Falls back to raw `kappa` if the spread is zero or non-finite. |

With `metric="identity"`, `kappa` controls posterior sharpness. Increasing it
can strengthen gradients until the sigmoid saturates. `batch_std` divides by
the current distance spread, so its effective scale changes during training;
a search range for `fixed` should not be reused without checking the logits.

### 2.4 Decision threshold `eta`

| Parameter | Live in | Values | Default | What it does |
|---|---|---|---|---|
| `threshold_mode` | both | `"quantile"`, `"fixed"` | `"quantile"` | `quantile` places `eta` at the `positive_pair_prior` quantile of the batch's own distances, pinning the fraction of pairs on the similar side. `fixed` uses `threshold` as the paper's constant `eta`. |
| `threshold` | both | float > 0 | `1.0` | The `fixed`-mode `eta`. In `quantile` mode it is only the seed value of the running buffers, which the first batch overwrites. |
| `threshold_momentum` | `quantile` only | float in `(0, 1)` exclusive | `0.1` | EMA rate of the running quantile estimate: `eta <- (1 - m) * eta + m * batch_quantile`. Seeded by the first batch rather than interpolated from `threshold`. |
| `positive_pair_prior` | `quantile` only | float in `(0, 1)` exclusive, or `null` | `null` | The quantile `eta` sits at. `null` derives it as `1 / num_classes` — the probability that two samples from balanced classes are similar. |

**`threshold_mode: "fixed"` silently deactivates `threshold_momentum` and
`positive_pair_prior`.** `resolve_threshold` returns `self.threshold` before
reaching the quantile branch, so neither value is ever read — but both are still
validated in the constructor (an out-of-range value still raises), the
`threshold_uu` / `threshold_lu` buffers stay pinned at their seed and never
initialize, and `positive_pair_prior` is still reported in the
`train/seraph/positive_pair_prior` diagnostic and the head's startup log line.
Logs and diagnostics will show a prior that has no effect on the loss.

`quantile` mode exists because deviation 1 drops the paper's labeled likelihood
term, and it was what anchored `eta`. Without it, minimizing entropy over a free
`eta` would simply declare every pair dissimilar; pinning a quantile makes that
collapse unreachable. The estimate is detached — it calibrates the posterior, it
is not something the entropy term may optimize.

`positive_pair_prior` moves the threshold through the distance distribution.
Its effect depends on that distribution. `threshold_momentum` controls how
quickly the running quantile follows a change and how much batch noise it retains.

### 2.5 Which pairs are scored

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `include_labeled_unlabeled` | bool | `false` | Adds the labeled-unlabeled ("lu") block to the objective alongside unlabeled-unlabeled ("uu"). Both carry an unknown weak label, which is what puts them in the paper's `U`. Labeled-labeled pairs are never scored — their weak label is known, so they belong to the likelihood term delegated to the supervised loss. |
| `unlabeled_batch_size` | int ≥ 2, or `null` | `null` | Absolute size of the unlabeled batch drawn per step. Clamped to the pool size with a warning. |
| `unlabeled_ratio` | float > 0, or `null` | `1.0` | Relative alternative: `round(labeled batch_size * ratio)`. If both are set, the absolute size wins and a warning is logged. If both are `null`, the ratio defaults to 1.0. |

The two blocks keep **separate running `eta` buffers** — labeled embeddings are
already organized by the supervised loss, so the blocks' distance distributions
differ systematically and a shared quantile would sit wrong for both. They are
combined by their share of the *pool's* pair counts (`n_u^2/2` vs `n_l * n_u`),
not the batch's, so batch composition cannot silently reweight the objective.

Including labeled-unlabeled pairs changes which relationships are regularized.
The loss averages over pairs, while batch size changes the noise and cancellation
in the resulting gradient. Use the mixing weight to control overall strength.

---

## 3. `method_params` — the mixing weights

| Parameter | Values | Default | What it does |
|---|---|---|---|
| `regularizer_weight` | float ≥ 0 | `1.0` | Multiplies `L_seraph`. Scales the gradient contribution **linearly** — the cleanest strength knob. `0` disables SERAPH entirely: no posterior head is attached and no unlabeled dataset is built. |
| `supervised_weight` | float ≥ 0 | `1.0` | Multiplies the configured metric-learning loss. Takes the role of the paper's supervised/unsupervised tradeoff `mu` together with `regularizer_weight`. |

Both must be finite and non-negative, and they cannot both be zero.

`fairness_weight` and `trace_weight` are added *beside* these two rather than
inside them:

```
J = supervised_weight * L_sup + regularizer_weight * L_seraph
                              + fairness_weight * L_fair
                              + trace_weight * tr(A)
```

so `regularizer_weight` does not multiply either, and neither does the
`regularizer_target_ratio` rescaling of it. For the trace that is deliberate:
the term reads parameters, not pairs, so folding it into `regularizer_weight`
would put a batch-independent term inside the weight that
`regularizer_target_ratio` calibrates from measured per-batch gradients.

Two mechanisms can set `trace_weight` for you, and both are restricted to
`trace_target="projection"` for the same reason: under `trace_target="metric"`
the term's gradient norm over shared parameters is exactly `0`, because the
metric lives in `seraph_posterior`, which `private_model_module_names()` excludes
from every gradient comparison.

| | `metric` target | `projection` target |
|---|---|---|
| `trace_target_ratio` | **raises** — the calibration would hit its dead-term budget and fail mid-run | **on by default** at `0.1`; uses the configurable sample-scoped target-ratio schedule (10,000-sample windows by default) |
| `grad_norm_alpha` | not registered — GradNorm skips its entire update on any non-positive component norm, so registering it would freeze the supervised and regularizer weights too | registered; GradNorm learns the weight alongside the other two |

Between the two, prefer the ratio. GradNorm's signal is the inverse training rate
`L(t)/L(0)`, and for `soft_rank` that ratio measures how much rank has already
been surrendered — a falling value reads as "learning fast, downweight", which is
backwards for a structural prior. Setting `grad_norm_alpha` clears
`trace_target_ratio` (the base class hands every weight to GradNorm from the
first regularized batch), so they never fight over the same number.

To keep both — GradNorm on the entropy term, the ratio on the trace — name the
trace in `method_params.grad_norm_exclude_components`:

```json
"grad_norm_alpha": 1.5,
"grad_norm_exclude_components": ["seraph_trace"]
```

Precedence is then per weight rather than per run: GradNorm takes the supervised
and regularizer weights, the excluded term keeps its configured weight and its
`trace_target_ratio` calibration. An unknown component name raises; a known one
this configuration does not produce (`trace_weight: 0`) warns and does nothing.

---

## 4. SSL config fields SERAPH reads

| Field | What it does for SERAPH |
|---|---|
| `method` | Must be `"seraph"`. |
| `label_sampling_mode`, `labeled_fraction`, `labeled_per_class` | Build the labeled/unlabeled split. SERAPH's unlabeled pool is `split.unlabeled_positions`; it needs ≥ 2 samples, and ≥ 1 labeled sample if `include_labeled_unlabeled` is on. |
| `max_unlabeled_samples` | Caps the unlabeled pool, so it caps what SERAPH samples from. |
| `warmup_epochs` | Supervised-only epochs before the regularizer switches on. |
| `seed`, `support_seed` | Split and support-set determinism. |
| `graph_batch_mode` | Must stay `"global"`. SERAPH does not override `configure_graph_batching`, so `"in_batch"` raises. |

### Fields that are inert for SERAPH

These are pseudo-label / graph-method machinery. They are accepted and logged but
never read on the SERAPH path — do not tune them and do not read meaning into
them:

| Field | Belongs to |
|---|---|
| `update_mode`, `update_interval_epochs` | Epoch-scoped artifact rebuilds. SERAPH forms its pairs inside the current batch and keeps no such artifact, so the cadence has nothing to rebuild. |
| `confidence_threshold`, `pseudo_label_rescue_confidence_floor`, `pseudo_label_rescue_top_k`, `pseudo_label_diagnostics_mode` | Pseudo-labelers and SimMatch. |
| `graph_diagnostics_*`, `graph_labeled_batch_size`, `graph_unlabeled_batch_size` | Graph SSL methods. |
| `embedding_batch_size`, `embedding_num_workers` | Whole-pool embedding extraction. SERAPH rides along in the supervised forward pass (`uses_joint_forward = True`). |
| `labeled_batch_size` | Two-stream HPO batch sizing. |

---

## 5. Diagnostics

`ssl_gradient_contribution_log_interval > 0` writes to `diagnostics.csv` under
`train_batch`:

| Series | Meaning |
|---|---|
| `train/gradient_contribution/supervised_norm` | `\|\|grad(supervised_weight * L_sup)\|\|` over model parameters |
| `train/gradient_contribution/regularizer_norm` | `\|\|grad(regularizer_weight * L_seraph)\|\|` |
| `train/gradient_contribution/regularizer_to_supervised_ratio` | The headline number: their quotient |
| `train/gradient_contribution/regularizer_fraction` | `reg / (sup + reg)` |
| `train/gradient_contribution/cosine_similarity` | Angle between the two gradients; ~0 means the regularizer is not opposing or reinforcing the supervised direction, just adding noise |

`log_batch_diagnostics: true` additionally emits `train/seraph/*` — per block
(`uu`, `lu`) `pair_entropy`, `posterior_mean`, `positive_pair_rate`, `threshold`,
`kappa`, `mean_squared_distance`, `scored_pairs`, `block_weight`, plus
`unlabeled_batch_size` and `positive_pair_prior`. The `uu` series are mirrored
under unprefixed names. **`pair_entropy` is the tuning signal** — 0.32 nats
corresponds to the `|logit| ≈ 2.4` optimum, 0.69 means the term is inert.

`scripts/bench_seraph_gradient_contribution.py` sweeps these axes offline against
real cached features. Its `head_share` column is the "gradient landing on `B`"
number from §2.2. Note its hardcoded `BASELINE_REGULARIZER_PARAMS` describe a
different operating point than the `seraph_prx` study its docstring cites —
override them for the trial under study.

---

## 6. Tuning summary

Direction that **increases** the gradient contribution:

| Knob | Direction | Shape | Strength |
|---|---|---|---|
| `regularizer_weight` | higher | linear | strongest, monotonic |
| `kappa` | higher, until mean `\|logit\|` ≈ 2.4 | inverted U | ~quadratic below the peak |
| `unlabeled_batch_size` | lower | ~`1/sqrt(n)` | 2.5x from 256 to 32 |
| `em_sharpening` (em only) | higher | monotonic here | zero at 1.0 |
| `positive_pair_prior` | away from the quantile where `eta ≈ E[d^2]` | U-shaped | setup-dependent sign |
| `metric` | `scale` / `lowrank` | large, but mostly self-gradient | read `head_share`, not the ratio |
| `threshold_momentum` | — | flat | negligible in `quantile` mode, inert in `fixed` |

When writing a search space:

- Set `kappa`'s range from `kappa_mode` and inspect posterior saturation.
- Do not put `em_sharpening: 1.0` in a range.
- Prefer raising `regularizer_weight`'s floor over widening `kappa` — it is
  linear and has no saturation to overshoot.
- `threshold_momentum` is nearly free to fix at a constant.
- If you search `metric`, log `head_share`; otherwise the study will select
  whatever inflates `B` fastest.
