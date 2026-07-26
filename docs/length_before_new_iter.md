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

Each fold now overrides `length_before_new_iter` with the size of its complete
training pool:

```python
length_before_new_iter = num_labeled + num_unlabeled
```

This happens after the fold and label-budget splits are built. Therefore a 20%
labeled run uses the same epoch sampling budget as a 100% labeled run on the
same fold; the smaller labeled support is sampled repeatedly as needed. The
configured JSON/CLI value is retained in `configured_length_before_new_iter`
for auditing, but it does not control training.

For grouped cross-validation, the exact value depends on `cv_k`, the
individual fold, and the sizes of the complete class groups assigned to that
fold:

```text
cv_k=4 group_kfold: approximately 75% is training
cv_k=5 group_kfold: approximately 80% is training
```

Whole class groups are never divided merely to make the sample counts equal,
so folds can have different exact lengths. Changing the labeled fraction only
changes how each training fold is divided between labeled and unlabeled
samples. Changing `cv_k` changes the total training-fold length and therefore
changes `length_before_new_iter`.

`MPerClassSampler` emits only complete batches, so the effective sampled length
can be slightly smaller when the resolved fold size is not divisible by
`batch_size`.

## Log Check

Training logs now include a train-loader summary:

```text
Resolved length_before_new_iter from the complete fold training pool:
6040 = 1208 labeled + 4832 unlabeled (configured value 6041 was overridden)
```

Use this line to verify the labeled/unlabeled counts and resolved fold length.
The following train-loader summary reports the batch-aligned sampled length.

## Constraints

`MPerClassSampler` still requires:

```text
batch_size % sampler_m == 0
sampler_m * number_of_unique_labels >= batch_size
```

When a class has fewer than `sampler_m` examples, `MPerClassSampler` follows its normal behavior and samples that class with replacement.
