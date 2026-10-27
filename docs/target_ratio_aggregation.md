# How target-ratio probes are aggregated

The trainer calibrates `regularizer_weight` so that the regularizer gradient
reaches a configured fraction of the supervised one
(`training/engine.py:_calibrate_regularizer_weight`). Because the gradient norm
is linear in the weight, one probe at unit weight already gives the answer for
that batch:

```
q_b = ||grad(w_sup L_sup)|| / ||grad(L_reg)||        w = target_ratio * A({q_b})
```

`A` is the aggregation, and this note records which one to use and why. All
numbers below come from

```
venv/bin/python scripts/analyze_target_ratio_aggregation.py --section all \
    --out scripts/reports/target_ratio_aggregation
```

## The four rules

| config | `A` | memory |
| --- | --- | --- |
| default | median over the window | resets every `recalibration_interval_samples` (10 000) |
| `regularizer_target_ratio_aggregation="mean"` | arithmetic mean | same window |
| `regularizer_target_ratio_probe_memory=k` | configured statistic over the last `k` probes | never resets |
| `regularizer_target_ratio_ema_alpha=a` | `a q + (1-a) est` | never resets, never forgets |

`ema_alpha` and `aggregation` are mutually exclusive: the EMA *is* an
aggregation rule. `probe_memory` and `ema_alpha` both replace the window and
require an explicit `update_interval_samples`.

## What the probe stream actually looks like

Ten complete Cars196 runs (five DML losses x batch size 64/256, seed 7, LRML,
target ratio 0.5, one probe per 1 000 samples) give 313 probes each. Every
training step also logs the weighted contribution, so `q` can be reconstructed
per step and cross-checks against the probe rows to 1.4e-7 relative.

Around a local level, the probe noise is small and only mildly heavy-tailed:

| | batch 256 | batch 64 |
| --- | --- | --- |
| sd of `log q` residual | 0.043-0.051 | 0.084-0.115 |
| excess kurtosis | 0.08-1.40 | 0.29-16.2 |
| skew | -0.26-+0.04 | -2.81--0.15 |

Bootstrapping ten of each run's own residuals, `Var(median)/Var(mean)` averages
**1.11** (Gaussian reference 1.37) - the mean is slightly more efficient in
eight of ten runs, the median in the two with the heaviest tails (ArcFace and
ProxyAnchor at batch 64).

Outliers, in the sense of a probe more than a factor of two from the median of
its ten predecessors, are almost absent here: **1 of 3 030** probes, and
**29 of 31 150** steps. All of them are *downward* - the regularizer gradient
spiking, not vanishing. The single probe event is `q = 955` against a local
median of 2 086, with the regularizer norm at 1.94x and the supervised norm at
0.90x their local medians.

**This is a property of that one fixed configuration, not of the method.** The
tuned Cars196 LRML study, whose LRML parameters and target ratios are searched,
shows the other regime. There the achieved ratio divided by its target is
exactly `local median of q / this batch's q`, so the same statistic is readable
from `gradient_contribution` alone:

Snapshot at 158 folds / 24 582 logged steps (the study is still running, so the
percentages move; the qualitative split does not):

| loss | steps beyond 2x above its local median | beyond 10x | worst |
| --- | --- | --- | --- |
| ArcFace | 3.01 % | 0.91 % | **656x** |
| NTXent | 1.22 % | 0 | 3.7x |
| ProxyAnchor | 0.25 % | 0 | 2.5x |
| MultiSimilarity | 0.20 % | 0 | 2.6x |
| Circle | 0.13 % | 0 | 2.3x |

The events are concentrated, not spread: **9 of 158 folds** - all ArcFace, from
two trials - contain an event beyond 10x, and inside those folds it is 0.65-7.3 %
of steps. In one of them the LRML gradient falls to ~1 % of its local norm while
the ArcFace gradient spikes 3.4x, pushing `q` ~320x above its local median.

That is the case a robust default is for: it protects the few configurations
that need it and costs the rest nothing. Note the 1e-6 floor is an *absolute*
criterion - it only rejects a probe whose regularizer gradient is below a
millionth of the supervised one, so the 2x-10x events (1.02 % and 0.24 % of
steps) pass through it untouched, and a 10x probe already multiplies a
ten-sample mean by ~1.9.

## Clean streams: mean and median are interchangeable

Replaying every rule over the recorded stream (weights recomputed offline, error
scored against the per-step `q`):

| estimator | mean abs. target error | vs rolling median |
| --- | --- | --- |
| ema a=0.3 | 6.16 % | -0.24 pp, p = 0.005 |
| ema a=0.2 | 6.28 % | -0.12 pp, p = 0.005 |
| **rolling mean k=10** | **6.38 %** | -0.02 pp, CI [-0.08, +0.03], p = 0.37 |
| **rolling median k=10** | **6.40 %** | - |
| windowed mean | 6.44 % | +0.03 pp, p = 0.72 |
| windowed median | 6.51 % | +0.11 pp, p = 0.35 |
| ema a=0.1 | 7.07 % | +0.67 pp, p = 0.003 |
| ema a=0.05 | 9.25 % | +2.84 pp |

The mean's eight-of-ten win over the median is not significant. Both are limited
by something else: the estimator's own error is 5.4 % and the drift of `q`
between probes is 7.0 %, so the aggregation rule is the small term.

Training all five rules end to end (five losses x two batch sizes, seed 7,
50 epochs, ten runs each) says the same:

| rule | mean abs. target error | mean p95 | mean best val mAP@R |
| --- | --- | --- | --- |
| windowed median | 6.57 % | 16.86 % | 0.47048 |
| windowed mean | 6.54 % | 16.57 % | 0.47055 |
| rolling median | 6.40 % | 16.33 % | 0.46999 |
| rolling mean | 6.37 % | 16.06 % | 0.46984 |
| ema a=0.1 | 7.02 % | 18.67 % | 0.46960 |

Mean vs median is a wash in both window types (+0.00006 mAP@R windowed, 7/10,
p = 0.43; -0.00015 rolling, 3/10, p = 0.16). The EMA is worst on target tracking
(+0.61 pp, 9/10, p = 0.004) and loses 8 of 10 on mAP@R. Note that target fidelity
and validation quality do not line up here - the reset-window variants track the
target worse but score marginally higher - and every gap is ~0.0005 mAP@R on a
single seed.

## Contaminated streams: the median is the only one that survives

Injecting probes whose regularizer gradient is 100x smaller than usual (still
far above the rejection floor):

| contamination | windowed median | rolling median | windowed mean | rolling mean | ema a=0.1 |
| --- | --- | --- | --- | --- | --- |
| 0 % | 6.5 % | 6.4 % | 6.4 % | 6.4 % | 7.1 % |
| 0.5 % | 16.3 % | **6.9 %** | 54 % | 55 % | 57 % |
| 2 % | 47.6 % | **7.3 %** | 199 % | 201 % | 197 % |
| 5 % | 122 % | **9.6 %** | 493 % | 494 % | 489 % |
| 10 % | 274 % | **23.6 %** | 971 % | 978 % | 984 % |

A *single* injected outlier:

| estimator | peak weight error | probes until back under 10 % |
| --- | --- | --- |
| windowed median | 2.3 % | 0 |
| rolling median | 2.6 % | 0 |
| windowed mean | 1 409 % | 4 |
| rolling mean | 1 006 % | 10 |
| ema a=0.1 | 989 % | 44 |
| ema a=0.3 | 2 963 % | 17 |

The EMA is the worst of both: it takes `alpha` times the full excursion and then
decays it over tens of probes, i.e. tens of thousands of samples.

The reset window is far more fragile than the sliding one, and the reason is its
opening probes: with one or two values in the buffer a median has no majority to
protect it. At 2 % contamination the windowed median averages 46.7 %, but only
9.2 % once its buffer holds three or more probes; the sliding window, which
never returns to that state, stays at 7.7 %.

## How long the sliding buffer should be

`k` trades two things off. A longer buffer averages away more per-batch noise
and absorbs more contaminated probes (the median needs a clean majority), but it
lags a drifting `q` by roughly `k/2` probes - and `q` does drift, because it
tracks the projection head. Scored strictly causally at the 1 000-sample cadence
the runs used:

| k | lookback | clean | 2 % contaminated |
| --- | --- | --- | --- |
| 1 | 1 000 | 8.05 % | 252 % |
| 2 | 2 000 | 6.99 % | 251 % |
| 3 | 3 000 | 6.94 % | 19.4 % |
| 5 | 5 000 | 6.57 % | 6.64 % |
| **8** | 8 000 | **6.38 %** | 6.43 % |
| 10 | 10 000 | 6.39 % | 6.43 % |
| 15 | 15 000 | 6.51 % | 6.55 % |
| 20 | 20 000 | 6.72 % | 6.76 % |
| 30 | 30 000 | 7.26 % | 7.29 % |
| 50 | 50 000 | 8.43 % | 8.44 % |

A flat bowl with its minimum at `k=8`; everything from 5 to 15 is within 0.2 pp.
The lower end does not degrade gracefully - at `k<=2` the median has no majority
and the same 2 % contamination takes it from ~7 % to over 250 %. Above `k~20`
the lag dominates. Both batch sizes have their minimum at the same `k`, which is
what a sample-scoped cadence should give.

Re-drawing the probes at other cadences moves the optimum so that the *sample*
lookback stays roughly constant: `k=20` at a 256-sample cadence (~5 000 samples),
`k=15` at 512 (~7 700), `k=8` at 1 000 and at 2 000 (8 000 and 16 000).

Five buffer lengths were also trained end to end (ten runs each: five losses x
two batch sizes, seed 7, 50 epochs):

| k | mean abs. target error | mean p95 | mean best val mAP@R | vs k=10 |
| --- | --- | --- | --- | --- |
| 3 | 6.57 % | 16.80 % | 0.47014 | +0.17 pp, p = 0.28 |
| 5 | 6.40 % | 16.32 % | 0.47004 | +0.00 pp, p = 1.00 |
| 10 | 6.40 % | 16.33 % | 0.46999 | - |
| 20 | 6.94 % | 18.16 % | 0.46978 | +0.53 pp, **p = 0.020** |
| 30 | 7.62 % | 20.46 % | 0.46965 | +1.22 pp, **p = 0.014** |

`k=5` and `k=10` are indistinguishable; `k=20` and `k=30` miss the target
significantly more. mAP@R separates nothing (all within 0.0005, no paired
comparison significant).

Split by batch size, the reason the optimum exists at all: at batch 256
(sd log q = 0.047) the error grows monotonically with k (4.22, 4.24, 4.46, 5.19,
6.06 % for k = 3, 5, 10, 20, 30) - the probes are quiet enough that a short
buffer's lower lag wins. At batch 64 (sd 0.099) there is a bowl with its minimum
at k=10 (8.92, 8.57, 8.35, 8.69, 9.18 %). `k=10` is the compromise that suits
both; `k=30` is worse in both.

So the rule of thumb is

```
k ~ 10000 / update_interval_samples,   never below 5
```

and the default `k=10` at a 1 000-sample cadence is 0.006 pp off its optimum.

## Which to use

- **`median` (default).** Correct whenever a contaminated probe is possible,
  which the tuned study shows it is. Costs about 11 % more estimator variance
  than the mean on clean data - roughly 0.02 pp of a 6.4 % target error.
- **`mean`.** For an ablation, or when the probe stream is known to be clean and
  near-symmetric and the last fraction of a percent matters. Note that it makes
  the 1e-6 rejection floor load-bearing: below the floor a probe is dropped,
  above it a single probe moves the weight by `1/k` of its distance.
- **`probe_memory` (sliding) over the reset window.** Constant lookback, no
  frozen stretch, and no repeated pass through the unprotected one- and
  two-probe buffer. Set it to about `10000 / update_interval_samples`, and never
  below 5.
- **`ema_alpha`.** Implemented for ablations. `alpha=0.1` tracks the target
  measurably worse than a ten-probe window (+0.67 pp, p = 0.003); `alpha=0.2-0.3`
  tracks it better on clean data but keeps every outlier for tens of probes.

## Samples or steps? Measured, per knob

The schedule has two knobs and they belong in different units. Splitting them
apart - cadence as a step stride, lookback as `cadence x k` - and scoring each
batch size separately (both saw the same 6 400 samples per epoch, so batch 64 took
four times as many optimizer steps):

| cadence [samples] | lookback 1 000 | 2 000 | 4 000 | 8 000 | 16 000 |
| --- | --- | --- | --- | --- | --- |
| **batch 64** | | | | | |
| 64 (1 step) | 8.01 | 7.87 | **7.78** | 7.84 | 8.14 |
| 256 (4 steps) | 8.72 | 8.27 | 8.04 | 7.97 | 8.16 |
| 1 024 (16 steps) | 10.68 | 9.27 | 8.66 | 8.36 | 8.31 |
| **batch 256** | | | | | |
| 256 (1 step) | 4.36 | 4.14 | **4.11** | 4.19 | 4.58 |
| 512 (2 steps) | 4.67 | 4.37 | 4.23 | 4.23 | 4.59 |
| 1 024 (4 steps) | 5.42 | 4.70 | 4.47 | 4.39 | 4.73 |

**Lookback: samples.** Both batch sizes bottom out at the same 4 000 samples,
flat from 2 000 to 8 000. In steps that is 62 vs 16 - a factor of four - so a
step-scoped lookback would mistune on any batch-size change. Searching cadence
and buffer jointly and freely gives 3 200 samples (batch 64) vs 3 840 (batch 256),
a factor 1.2; in steps, 50 vs 15, a factor 3.3.

**Cadence: a cost knob, and it matters more at small batches.** Denser is
monotonically better at a fixed lookback, because more probes fit the same window.
At a 4 000-sample lookback, going from a 1 024-sample cadence to one probe per
step is worth 0.88 pp at batch 64 but only 0.36 pp at batch 256 - a 1 000-sample
cadence fills a 4 000-sample window with just four probes. Calibrating at small
batch sizes therefore wants a finer cadence, which in step terms means probing
about as often per step.

The default (cadence 1 000, k=10, so a 10 000-sample lookback) sits 0.58 pp
(batch 64) and 0.28 pp (batch 256) above its optimum. Cadence 250 with k=16
(4 000-sample lookback) would be the better default at four times the probe cost.

`update_interval_samples` rounds up to the next batch boundary, so at 1 000
samples the probe lands after about 1 024 samples - 16 steps at batch 64, 4 at
batch 256.

Probing every step is available (`update_interval_samples=0`) and is what Esser
et al. do for the VQGAN adaptive weight, but it buys little here. Scored strictly
causally - the weight formed from probes *before* the step it is applied to,
which is the only fair comparison because a per-step probe otherwise fits the
weight to the very batch it trains -

| schedule | causal target error |
| --- | --- |
| 1 000 samples, k=10 | 6.53 % |
| every step, k=10 | 6.16 % |
| every step, matched 10 000-sample lookback | 6.09 % |
| every step, no smoothing (k=1, Esser-style) | 8.08 % |

Probing every step measures 16x (batch 64) or 4x (batch 256) as often and removes 0.4 pp, and dropping the smoothing
altogether is 1.6 pp *worse* than the 1 000-sample schedule. The measured cost of
per-step probing is +25.5 % (batch 64) and +21.5 % (batch 256) of recorded step
time.


## Two knobs the calibration does not cover

The target ratio sets the regularizer's **magnitude** and nothing else. Two
options address the parts it leaves open; both are off by default.

### `gradient_surgery` — direction

On the measured runs LRML conflicts with the supervised loss on **74-99 % of
steps** (mean cosine -0.10 to -0.47), so a norm-only controller routinely raises
the weight of a term that is partly undoing supervised progress. PCGrad (Yu et
al., NeurIPS 2020) removes the conflicting component by projection. Two modes:

| mode | what it does |
| --- | --- |
| `supervised_priority` | asymmetric: `g_sup` untouched, `g_reg <- g_reg - (g_reg.g_sup)/\|g_sup\|^2 g_sup` |
| `pcgrad` | Algorithm 1 unchanged: both gradients projected off each other |

`supervised_priority` is the right default here and is *not* vanilla PCGrad —
worth stating precisely when reporting, since PCGrad treats its tasks as
co-equal, and letting an auxiliary regularizer bend the supervised update is the
interference it is supposed to avoid causing.

The projection shortens the regularizer gradient by `sqrt(1 - cos^2)` (1-13 % on
the measured runs), so the calibration probe measures the **projected** norm when
surgery is on. Otherwise the realized ratio would sit below the configured one by
exactly that factor.

Refused for regularizers with extra objective terms (SLADE, SERAPH) and for those
that step the optimizer between their own losses — there is no single gradient
pair to project.

Diagnostics: `train/gradient_surgery/{cosine_before,fired,regularizer_length_kept,
supervised_length_kept}`.

### `regularizer_target_ratio_statistic` — which statistic

`l2` (default) compares Euclidean norms. `max_mean` is the statistic Wang et al.
use for learning-rate annealing in physics-informed networks
(doi:10.1137/20M1318043, Algorithm 1): `max|grad L_sup| / mean|grad L_reg|`, with
auxiliary terms also reduced by `mean`. Their scheme is otherwise the same
construction as this one — per-term gradient statistics, ratio as the candidate
weight, temporal smoothing — differing in the statistic, in using an EMA with
alpha = 0.9, and in probing every step (they also suggest "every 10 gradient
descent steps").

The two statistics disagree exactly when a gradient is concentrated on few
coordinates rather than spread over them: `l2` follows the largest coordinates,
`mean` the typical one. That is the regime where an L2 ratio is unstable while a
mean one is not.

## Diagnosing a collapsing regularizer

`train/gradient_contribution/regularizer_norm` is the **weighted** norm
`||grad(w_reg L_reg)||`. Under a calibrated weight it is pinned near
`target_ratio * supervised_norm` by construction, so it cannot show the
regularizer's own gradient collapsing — a shrinking `||grad(L_reg)||` and a
growing `w_reg` cancel in it. Measured on the fixed diagnostic runs, first vs
last tenth of training:

| loss (batch 64) | logged weighted norm | unweighted `||grad L_reg||` | weight |
| --- | --- | --- | --- |
| ArcFace | 9.85 -> 0.95 (/10) | 0.0229 -> 0.000676 (**/34**) | 444 -> 1405 |
| ProxyAnchor | 3.15 -> 1.17 (/2.7) | 0.0219 -> 0.00108 (**/20**) | 148 -> 1096 |
| NTXent | 1.11 -> 0.35 | 0.0209 -> 0.0054 | 53 -> 65 |
| Circle | 11.5 -> 7.9 | 0.0219 -> 0.0161 | 527 -> 491 |
| MultiSimilarity | 0.099 -> 0.070 | 0.0240 -> 0.0186 | 4.1 -> 3.8 |

So the per-step contribution log now also carries
`regularizer_norm_at_unit_weight`, `unit_regularizer_to_supervised_ratio` and
`regularizer_weight`. They cost nothing: `||grad(w L)|| == w ||grad(L)||` is
exact, so the unweighted norm is a division, not a second backward pass.
