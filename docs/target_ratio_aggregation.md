# Target-ratio gradient calibration

The trainer calibrates `regularizer_weight` so the regularizer gradient reaches
a configured fraction of the supervised gradient. For a probe at unit weight:

```text
q_b = ||grad(w_sup * L_sup)|| / ||grad(L_reg)||
w_reg = target_ratio * aggregate(q_b)
```

The implementation lives in
`training/engine.py:_calibrate_regularizer_weight`. The ratio controls gradient
magnitude; the direction of each gradient remains a separate consideration.

## Aggregation and memory

| Configuration | Aggregation | Memory |
| --- | --- | --- |
| Default | Median of probes in the window | Resets every `recalibration_interval_samples` (10,000 by default) |
| `regularizer_target_ratio_aggregation="mean"` | Arithmetic mean | Same reset window |
| `regularizer_target_ratio_probe_memory=k` | Configured statistic over the last `k` probes | Sliding buffer across window boundaries |
| `regularizer_target_ratio_ema_alpha=a` | `a * q + (1 - a) * estimate` | Exponentially decaying history |

The median limits the influence of unusually large or small probes. The mean
uses every probe's magnitude directly. A sliding buffer gives a fixed lookback
in probe counts, while an EMA retains older observations with decaying weight.

`ema_alpha` and `aggregation` are mutually exclusive. `probe_memory` and
`ema_alpha` replace the reset-window behavior and require an explicit
`update_interval_samples`.

## Probe schedule

`update_interval_samples` controls the distance between probes in consumed
training samples. It rounds up to a batch boundary; `0` probes every optimizer
step. `recalibration_interval_samples` controls the reset-window length.

With a sliding buffer, its approximate lookback in samples is
`update_interval_samples * probe_memory`. When changing the cadence, adjust
the buffer if the same sample lookback is desired. More frequent probes add
gradient-computation cost.

## Gradient statistic

`regularizer_target_ratio_statistic="l2"` compares Euclidean gradient norms.
`"max_mean"` uses `max(abs(grad L_sup)) / mean(abs(grad L_reg))`, reducing
auxiliary terms by their mean absolute gradient too. The two statistics respond
differently when gradients concentrate on a small number of coordinates.

## Gradient surgery

`gradient_surgery` can remove conflicting gradient components when the
supervised and regularizer gradients have a negative dot product:

| Mode | Behavior |
| --- | --- |
| `supervised_priority` | Keeps the supervised gradient and projects the regularizer away from its conflicting component |
| `pcgrad` | Projects both gradients away from their conflicting components |

When surgery is active, calibration measures the projected regularizer norm.
Regularizers with extra objective terms, such as SLADE and SERAPH, and methods
that step the optimizer between their losses do not support this option.

Diagnostics appear under
`train/gradient_surgery/{cosine_before,fired,regularizer_length_kept,supervised_length_kept}`.

## Reading diagnostics

`train/gradient_contribution/regularizer_norm` is the weighted norm
`||grad(w_reg * L_reg)||`. Calibration can keep it near the target even while
the unweighted regularizer gradient shrinks and its weight grows.

Read `regularizer_norm_at_unit_weight`,
`unit_regularizer_to_supervised_ratio`, and `regularizer_weight` alongside the
weighted contribution to distinguish a changing regularizer from compensation
by the controller.
