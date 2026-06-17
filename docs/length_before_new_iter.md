# Sampler Epoch Length

`pytorch_metric_learning.samplers.MPerClassSampler` does not infer one epoch from the dataset length by default.
Its constructor default is:

```python
length_before_new_iter=100000
```

With `batch_size=16`, this creates:

```text
100000 / 16 = 6250 batches per epoch
```

That is why a Cars196 run with all labeled training data and a run with only the SSL labeled subset both showed `6250` training batches.
The subset was applied correctly, but the sampler was repeatedly sampling from the small labeled set until it reached the fixed default sampler length.

## Observed Problem

Example full-label run:

```text
Semi-supervised split: mode=balanced, 6446 labeled, 0 unlabeled candidates
loss = ...: 1%| | 38/6250
```

Example 1% label-budget run:

```text
Semi-supervised split: mode=balanced, 78 labeled, 6368 unlabeled candidates
loss = ...: 1%| | 54/6250
```

Those two runs should not have the same epoch length.
The 1% run was effectively doing heavy oversampling inside each epoch.

For Cars196 this produced 78 labeled samples because the split keeps at least one labeled sample per training class.

## Current Behavior

`utils.make_train_loader()` derives `length_before_new_iter` from the actual training dataset size when
`--length_before_new_iter` is omitted:

```python
length_before_new_iter = max(
    batch_size,
    ceil(len(train_dataset) / batch_size) * batch_size,
)
```

This preserves full batches while making epoch length proportional to the selected training set.

For `batch_size=16`:

```text
78 samples   -> 80 sampled examples/epoch   -> 5 batches/epoch
6446 samples -> 6448 sampled examples/epoch -> 403 batches/epoch
```

Set `--length_before_new_iter` or the experiment-config key `length_before_new_iter` to use a fixed
sampling budget instead. Experiment configs set this explicitly when a fixed sampling budget is desired.
`MPerClassSampler` rounds an explicit value down to a complete batch, so the effective length can be
slightly smaller when `length_before_new_iter` is not divisible by `batch_size`.

## Log Check

Training logs now include a train-loader summary:

```text
Train loader: 78 samples, 78 labels, 80 sampled examples/epoch, 5 batches/epoch
```

Use this line to verify that the sampler length matches the intended labeled or pseudo-labeled training set size.

## Constraints

`MPerClassSampler` still requires:

```text
batch_size % sampler_m == 0
sampler_m * number_of_unique_labels >= batch_size
```

When a class has fewer than `sampler_m` examples, `MPerClassSampler` follows its normal behavior and samples that class with replacement.
